// --- decode-into-fragment: decode two adjacent mul1 codes directly into a
// half2 mma B-fragment register. Eliminates the w[8] intermediate and the
// separate halves2half2 packing; the affine half (h*k_inv+k_bias) runs as
// ONE hfma2 over the pair. Bit-identical to decode_3inst on both codes.
__device__ __forceinline__ half2 decode_mul1_pair_fused(uint32_t c0, uint32_t c1)
{
    const uint32_t x0 = c0 * 0x83DCD12Du;
    const uint32_t x1 = c1 * 0x83DCD12Du;
    const uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 h = __halves2half2(__ushort_as_half((uint16_t)(s0 & 0xffffu)),
                             __ushort_as_half((uint16_t)(s1 & 0xffffu)));
    const __half2 k_inv2 = __half2half2(__ushort_as_half(0x1eeeu));
    const __half2 k_bias2 = __half2half2(__ushort_as_half(0xc931u));
    return __hfma2(h, k_inv2, k_bias2);
}

// --- direct kernel, fused-decode variant (layout A): the k-loop decodes
// pairs straight into FragB registers.
template <int BITS, bool HALF_K = false>
__global__ void exl3_grouped_gemm_direct_f(
    const half* __restrict__ g_x,
    const uint16_t* const* __restrict__ g_packed_arr,
    const int* __restrict__ g_counts,
    const long* __restrict__ g_offsets,
    int splits, int k, int n, int n16, int cb,
    float* __restrict__ g_scratch)
{
    constexpr int words16 = BITS * 16 + (HALF_K ? 8 : 0);
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int col_block = blockIdx.x;
    const int e = blockIdx.y;
    const int ks = blockIdx.z;

    const int cnt = g_counts[e];
    if (cnt <= 0) return;
    if (e == gridDim.y - 1) return;   // routing sentinel: NULL panel
    int m = cnt;
    const int groups = (m + ROWS_PER_PASS - 1) / ROWS_PER_PASS;
    const int k16 = k >> 4;
    const int k_beg = (int)((long)k16 * ks / splits);
    const int k_end = (int)((long)k16 * (ks + 1) / splits);
    if (k_beg >= k_end) return;
    const int n_base = col_block * 128;

    const uint16_t* g_packed = g_packed_arr[e];
    const long pair_base = g_offsets[e];
    const int m0 = lane >> 2;
    const int m8 = m0 + 8;
    const int kp = (lane & 3) << 1;

    constexpr int n_words = BITS * 256 / 32;
    int wp0[8], wp1[8], ws0[8];
    if constexpr (!HALF_K)
    {
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const int e32 = lane * 8 + j;
            const int b0 = e32 * BITS + BITS - 16 + 256 * BITS;
            const int b1 = b0 + 16;
            const int i0 = b0 / 32;
            const int i1 = (b1 - 1) / 32;
            wp0[j] = i0 % n_words;
            wp1[j] = i1 % n_words;
            ws0[j] = (i1 + 1) * 32 - b1;
        }
    }

    for (int g = 0; g < groups; g++)
    {
        const int r_base = g * ROWS_PER_PASS;
        FragC frag_c[2];
        frag_c[0] = {};
        frag_c[1] = {};

        for (int kk = k_beg; kk < k_end; kk++)
        {
            const uint32_t* tile32 =
                (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);

            FragB fb0, fb1;
            if constexpr (HALF_K)
            {
                half w[8];
                dq8_half_rt<BITS>(tile32, lane << 3, cb, w);
                fb0.elems[0] = __halves2half2(w[0], w[1]);
                fb0.elems[1] = __halves2half2(w[2], w[3]);
                fb1.elems[0] = __halves2half2(w[4], w[5]);
                fb1.elems[1] = __halves2half2(w[6], w[7]);
            }
            else
            {
                uint32_t wb[16];
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    wb[2 * j] = tile32[wp0[j]];
                    wb[2 * j + 1] = tile32[wp1[j]];
                }
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    const uint32_t w0 = __funnelshift_r(wb[4 * j + 1], wb[4 * j], ws0[2 * j]) & 0xffffu;
                    const uint32_t w1 = __funnelshift_r(wb[4 * j + 3], wb[4 * j + 2], ws0[2 * j + 1]) & 0xffffu;
                    const half2 hpair = decode_mul1_pair_fused(w0, w1);
                    if (j == 0) fb0.elems[0] = hpair;
                    else if (j == 1) fb0.elems[1] = hpair;
                    else if (j == 2) fb1.elems[0] = hpair;
                    else fb1.elems[1] = hpair;
                }
            }

            const half z = __float2half(0.0f);
            const bool gv0 = (m0 + r_base) < m;
            const bool gv8 = (m8 + r_base) < m;
            const half* xa = g_x + (long)(pair_base + r_base + m0) * k + kk * 16;
            const half* xb = g_x + (long)(pair_base + r_base + m8) * k + kk * 16;
            FragA fa;
            fa.elems[0] = __halves2half2(gv0 ? xa[kp] : z, gv0 ? xa[kp + 1] : z);
            fa.elems[1] = __halves2half2(gv8 ? xb[kp] : z, gv8 ? xb[kp + 1] : z);
            fa.elems[2] = __halves2half2(gv0 ? xa[kp + 8] : z, gv0 ? xa[kp + 9] : z);
            fa.elems[3] = __halves2half2(gv8 ? xb[kp + 8] : z, gv8 ? xb[kp + 9] : z);
            ptx_mma_m16n8k16(fa, fb0, frag_c[0]);
            ptx_mma_m16n8k16(fa, fb1, frag_c[1]);
        }

        const int c0 = (lane & 3) << 1;
        const int col0 = n_base + warp * 16 + c0;
        const int col1 = col0 + 8;
        const int rm0 = r_base + m0;
        const int rm8 = r_base + m8;
        if (rm0 < m)
        {
            atomicAdd(g_scratch + (long)(pair_base + rm0) * n + col0,     frag_c[0].elems[0]);
            atomicAdd(g_scratch + (long)(pair_base + rm0) * n + col0 + 1, frag_c[0].elems[1]);
            atomicAdd(g_scratch + (long)(pair_base + rm0) * n + col1,     frag_c[1].elems[0]);
            atomicAdd(g_scratch + (long)(pair_base + rm0) * n + col1 + 1, frag_c[1].elems[1]);
        }
        if (rm8 < m)
        {
            atomicAdd(g_scratch + (long)(pair_base + rm8) * n + col0,     frag_c[0].elems[2]);
            atomicAdd(g_scratch + (long)(pair_base + rm8) * n + col0 + 1, frag_c[0].elems[3]);
            atomicAdd(g_scratch + (long)(pair_base + rm8) * n + col1,     frag_c[1].elems[2]);
            atomicAdd(g_scratch + (long)(pair_base + rm8) * n + col1 + 1, frag_c[1].elems[3]);
        }
    }
}

