// EXL3 GROUPED trellis GEMM for the packed MoE decode path.
//
// One launch covers all active experts of a layer: pairs (token, expert-slot)
// are pre-sorted by expert id; each block processes ONE expert's rows for one
// (column-block, k-split). Grid dims are FIXED at capture (graph-safe); all
// data-dependent behavior lives in device memory read at replay time:
//   counts[e]  — pairs routed to expert e this step (torch.histc, no sync)
//   offsets[e] — prefix sum, first pair row of expert e
//   packed/svh/bias pointer arrays — one entry per expert
//
// Contract (graph-capture-safe): mutates only `out` and `g_cnt` (the split
// counters reset to zero in-kernel so graph replays see the initial state).
// The caller guarantees max(counts) <= ROWS_PER_PASS * chunks_cap; rows beyond
// that are the caller's fallback problem (python loop path).
//
// Attribution: kernel math is the v2 decode path of exl3_linear.cu (same
// file lineage/attributions as that file: ExLlamaV3 decode math MIT 2025,
// cuda-exl3 M-tiling MIT 2026). Grouping, workspace indexing and epilogue
// row-mapping are new for the SGLang packed MoE runner.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "utils.h"

#include "exl3/exl3_decode.cuh"
#include "exl3/exl3_had.cuh"

namespace exl3 {

constexpr float kRScaleG = 0.088388347648f;  // 1 / sqrt(128)
constexpr int ROWS_PER_PASS = 16;

// --- local copies of the shared fragment/MMA helpers (exl3_linear.cu keeps
// its own definitions; duplicated here so the standalone patch module builds
// without coupling to the AOT translation unit). ---
template <typename T, int n>
struct Vec { T elems[n]; __device__ T& operator[](int i) { return elems[i]; } };
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;

__device__ inline void ptx_mma_m16n8k16(const FragA& a, const FragB& b, FragC& c)
{
    const uint32_t* ar = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* br = reinterpret_cast<const uint32_t*>(&b);
    float* cr = reinterpret_cast<float*>(&c);
    const float* d = reinterpret_cast<const float*>(&c);
    asm (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(cr[0]), "=f"(cr[1]), "=f"(cr[2]), "=f"(cr[3])
        : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
          "r"(br[0]), "r"(br[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
}

template <int BITS, bool HALF_K = false>
__global__ void exl3_grouped_gemm_kernel_v2(
    const half* __restrict__ g_x,                    // (P, k) pair rows
    const uint16_t* const* __restrict__ g_packed_arr,  // [E]
    const half* const* __restrict__ g_svh_arr,         // [E]
    const half* const* __restrict__ g_bias_arr,        // [E] (nullable entries)
    float* __restrict__ g_ws,        // (col_blocks, splits, E*CHUNKS, 16, 128)
    int* __restrict__ g_cnt,         // (E, col_blocks, CHUNKS) split arrivals
    const int* __restrict__ g_counts,   // [E]
    const long* __restrict__ g_offsets, // [E]
    int chunks_cap,
    int k, int n, int n16, int cb, int splits,
    half* __restrict__ g_out,        // (P, n)
    int bf16_out)
{
    constexpr int words16 = BITS * 16 + (HALF_K ? 8 : 0);
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int col_block = blockIdx.x;
    const int ks = blockIdx.y;
    const int flat_z = blockIdx.z;
    const int e = flat_z / chunks_cap;
    const int chunk = flat_z % chunks_cap;

    const int cnt = g_counts[e];
    const int r0 = chunk * ROWS_PER_PASS;
    int m = cnt - r0;
    if (m <= 0) return;
    if (m > ROWS_PER_PASS) m = ROWS_PER_PASS;

    const int k16 = k >> 4;
    const int k_beg = (int)((long)k16 * ks / splits);
    const int k_end = (int)((long)k16 * (ks + 1) / splits);
    if (k_beg >= k_end) return;
    const int n_base = col_block * 128;

    const uint16_t* g_packed = g_packed_arr[e];
    const half* g_svh = g_svh_arr[e];
    const half* g_bias = g_bias_arr[e];

    const long pair_base = g_offsets[e] + r0;
    const int m0 = lane >> 2;
    const int m8 = m0 + 8;
    const bool v0 = m0 < m;
    const bool v8 = m8 < m;
    const int kp = (lane & 3) << 1;
    const half* x0 = g_x + (pair_base + m0) * k;
    const half* x8 = g_x + (pair_base + m8) * k;

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

    FragC frag_c[2];
    frag_c[0] = {};
    frag_c[1] = {};

    for (int kk = k_beg; kk < k_end; kk++)
    {
        const uint32_t* tile32 =
            (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);

        half w[8];
        if constexpr (HALF_K)
        {
            dq8_half_rt<BITS>(tile32, lane << 3, cb, w);
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
            for (int j = 0; j < 8; j++)
            {
                const uint32_t w0 = __funnelshift_r(wb[2 * j + 1], wb[2 * j], ws0[j]) & 0xffffu;
                w[j] = decode_3inst(w0, cb);
            }
        }

        const half* x0k = x0 + kk * 16;
        const half* x8k = x8 + kk * 16;
        const half z = __float2half(0.0f);
        FragA fa;
        fa.elems[0] = __halves2half2(v0 ? x0k[kp] : z, v0 ? x0k[kp + 1] : z);
        fa.elems[1] = __halves2half2(v8 ? x8k[kp] : z, v8 ? x8k[kp + 1] : z);
        fa.elems[2] = __halves2half2(v0 ? x0k[kp + 8] : z, v0 ? x0k[kp + 9] : z);
        fa.elems[3] = __halves2half2(v8 ? x8k[kp + 8] : z, v8 ? x8k[kp + 9] : z);

        FragB fb0, fb1;
        fb0.elems[0] = __halves2half2(w[0], w[1]);
        fb0.elems[1] = __halves2half2(w[2], w[3]);
        fb1.elems[0] = __halves2half2(w[4], w[5]);
        fb1.elems[1] = __halves2half2(w[6], w[7]);

        ptx_mma_m16n8k16(fa, fb0, frag_c[0]);
        ptx_mma_m16n8k16(fa, fb1, frag_c[1]);
    }

    // Workspace rows are PER-(EXPERT, CHUNK): (col_blocks, splits, E*chunks, 16, 128).
    const int c0 = (lane & 3) << 1;
    const long echunk = (long)e * chunks_cap + chunk;
    const long ws_plane = ((long)col_block * splits + ks) * (long)gridDim.z + echunk;
    float* ws_row0 = g_ws + (ws_plane * ROWS_PER_PASS + m0) * 128;
    float* ws_row8 = g_ws + (ws_plane * ROWS_PER_PASS + m8) * 128;
    if (v0)
    {
        ws_row0[warp * 16 + c0]         = frag_c[0].elems[0];
        ws_row0[warp * 16 + c0 + 1]     = frag_c[0].elems[1];
        ws_row0[warp * 16 + 8 + c0]     = frag_c[1].elems[0];
        ws_row0[warp * 16 + 8 + c0 + 1] = frag_c[1].elems[1];
    }
    if (v8)
    {
        ws_row8[warp * 16 + c0]         = frag_c[0].elems[2];
        ws_row8[warp * 16 + c0 + 1]     = frag_c[0].elems[3];
        ws_row8[warp * 16 + 8 + c0]     = frag_c[1].elems[2];
        ws_row8[warp * 16 + 8 + c0 + 1] = frag_c[1].elems[3];
    }

    // Split-arrival epilogue (same pattern as v2): the last split to arrive
    // for (col_block, chunk) reduces all splits deterministically. The chunk
    // dim makes the counter collision-free across row chunks; the counter
    // resets to zero so graph replays see the initial state.
    __shared__ bool s_is_last;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        const int done = atomicAdd(
            g_cnt + ((long)e * gridDim.x + col_block) * chunks_cap + chunk, 1);
        s_is_last = (done == splits - 1);
    }
    __syncthreads();
    if (!s_is_last) return;
    __threadfence();

    for (int rr2 = 0; rr2 < 2; rr2++)
    {
        const int r = warp * 2 + rr2;
        if (r >= m) continue;
        float v[4];
        for (int j = 0; j < 4; j++)
        {
            const int cl = lane + 32 * j;
            float acc = 0.0f;
            for (int s2 = 0; s2 < splits; s2++)
            {
                const long plane2 = ((long)col_block * splits + s2) * (long)gridDim.z + echunk;
                acc += g_ws[(plane2 * ROWS_PER_PASS + r) * 128 + cl];
            }
            v[j] = acc;
        }
        had_warp_butterfly(v, lane);
        const long out_row = pair_base + r;
        for (int j = 0; j < 4; j++)
        {
            const int col = n_base + lane + 32 * j;
            float fv = v[j] * kRScaleG;
            if (bf16_out)
            {
                float pp = fv * __half2float(g_svh[col]);
                if (g_bias) pp += __half2float(g_bias[col]);
                reinterpret_cast<__nv_bfloat16 *>(g_out)[(long)out_row * n + col] =
                    __float2bfloat16_rn(pp);
                continue;
            }
            if (fv > 65504.f) fv = 65504.f; else if (fv < -65504.f) fv = -65504.f;
            half h = __float2half_rn(fv);
            float pp = __half2float(h) * __half2float(g_svh[col]);
            if (pp > 65504.f) pp = 65504.f; else if (pp < -65504.f) pp = -65504.f;
            h = __float2half_rn(pp);
            if (g_bias)
            {
                float b2 = __half2float(h) + __half2float(g_bias[col]);
                if (b2 > 65504.f) b2 = 65504.f; else if (b2 < -65504.f) b2 = -65504.f;
                h = __float2half_rn(b2);
            }
            g_out[(long)out_row * n + col] = h;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        g_cnt[((long)e * gridDim.x + col_block) * chunks_cap + chunk] = 0;  // replay reset
}

}  // namespace exl3

namespace exl3 {
void sgl_exl3_grouped_linear(
    at::Tensor x,                     // (P, k) fp16 pair rows, pre-had_in
    at::Tensor packed_ptrs,           // (E,) int64 device pointers
    at::Tensor svh_ptrs,              // (E,) int64
    at::Tensor bias_ptrs,             // (E,) int64 (0 = none)
    at::Tensor counts,                // (E,) int32
    at::Tensor offsets,               // (E,) int64
    at::Tensor ws,                    // (col_blocks, splits, chunks, 16, 128) fp32
    at::Tensor cnt_buf,               // (col_blocks, chunks) int32 zeros
    int64_t cb,
    int64_t splits_in,
    int64_t chunks_cap_in,
    int64_t bits_in,
    int64_t half_k_in,
    at::Tensor out)                   // (P, n) fp16/bf16
{
    TORCH_CHECK(x.is_cuda() && counts.is_cuda() && offsets.is_cuda() && out.is_cuda(),
                "exl3 grouped: CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kHalf && counts.scalar_type() == at::kInt &&
                offsets.scalar_type() == at::kLong &&
                (out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16),
                "exl3 grouped: fp16 x, int32 counts, int64 offsets, fp16/bf16 out required");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "exl3 grouped: contiguous required");
    const int E = (int)packed_ptrs.size(0);
    const int P = x.size(0);
    const int k = x.size(1);
    const int chunks_cap = (int)chunks_cap_in;
    const int bits = (int)bits_in;
    const bool half_k = half_k_in != 0;
    int splits = (int)splits_in;
    const int n = out.size(1);
    const int n16 = n / 16;
    const int col_blocks = n / 128;
    TORCH_CHECK(ws.size(0) == col_blocks && ws.size(1) == splits &&
                ws.size(2) == (long)E * chunks_cap && ws.size(3) == 16 && ws.size(4) == 128,
                "exl3 grouped: workspace shape mismatch");
    TORCH_CHECK(cnt_buf.size(0) == (long)E * col_blocks && cnt_buf.size(1) == chunks_cap,
                "exl3 grouped: counter buffer shape mismatch");
    TORCH_CHECK(bits >= 1 && bits <= 8, "exl3 grouped: bitrate out of range");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const uint16_t** packed_arr =
        reinterpret_cast<const uint16_t**>(packed_ptrs.data_ptr<int64_t>());
    const half** svh_arr = reinterpret_cast<const half**>(svh_ptrs.data_ptr<int64_t>());
    const half** bias_arr = reinterpret_cast<const half**>(bias_ptrs.data_ptr<int64_t>());
    const half* xp = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    float* wsp = ws.data_ptr<float>();
    int* cp = cnt_buf.data_ptr<int>();
    const int* counts_p = counts.data_ptr<int>();
    const long* offsets_p = offsets.data_ptr<long>();
    half* op = reinterpret_cast<half *>(out.data_ptr());
    const int bf16_out = out.scalar_type() == at::kBFloat16 ? 1 : 0;
    int icb = (int)cb;

    dim3 grid(col_blocks, splits, E * chunks_cap);
    switch (bits)
    {
        case 1: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<1, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<1, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 2: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<2, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<2, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 3: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<3, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<3, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 4: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<4, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<4, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 5: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<5, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<5, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 6: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<6, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<6, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 7: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<7, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<7, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        case 8: if (half_k) exl3::exl3_grouped_gemm_kernel_v2<8, true><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); else exl3::exl3_grouped_gemm_kernel_v2<8, false><<<grid, 256, 0, stream>>>(xp, packed_arr, svh_arr, bias_arr, wsp, cp, counts_p, offsets_p, chunks_cap, k, n, n16, icb, splits, op, bf16_out); break;
        default: TORCH_CHECK(false, "exl3 grouped: bad bits");
    }
}
}  // namespace exl3
