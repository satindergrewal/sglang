// EXL3 M-tiled batched trellis GEMM for SGLang (dense path, continuous
// batching, Blackwell sm_120).
//
// ATTRIBUTION:
// - Decode math: turboderp ExLlamaV3 (MIT, (c) 2025) — see exl3_decode.cuh
//   for the full per-file license and modification notes. Decoding must match
//   the on-disk EXL3 format that ExLlamaV3 defines, bit-exactly.
// - M-tiling / single-read batched GEMM design follows cuda-exl3 (MIT,
//   Copyright (c) 2026 cuda-exl3 contributors); the Hadamard butterfly
//   structure follows ExLlamaV3's natural-order Sylvester butterfly
//   (hadamard_inner.cuh, MIT (c) 2025 turboderp).
// - Independent: kernel body, fused epilogue, host wrappers (own work for
//   SGLang; prior art cited for context).
//
// Kernel contract (functional, graph-capture-safe: the op mutates only
// `out`; all temporaries are allocated by the caller):
//   sgl_exl3_had_in(x, suh, out):  Hadamard-transform x along 128-chunks of
//                                  the input dim, after elementwise suh, with
//                                  the PyTorch reference's rounding points.
//   sgl_exl3_linear(x_had, packed, svh, bias?, cb, out):
//       out = round16( had_r(acc) ) * svh + bias
//     acc[r][n] = sum_k x_had[r][k] * dequant_trellis(packed)[k][n], fp32
//     accumulated; each block spans whole Hadamard blocks on the output side
//     (128 cols), so the epilogue applies the output-side Hadamard fused
//     (cuda-exl3 fused-epilogue shape).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "utils.h"

#include "exl3/exl3_decode.cuh"
#include "exl3/exl3_had.cuh"

namespace exl3 {

constexpr float kRScale = 0.088388347648f;  // 1 / sqrt(128)

// One warp per (row, 128-input-chunk): elementwise suh -> fp16 round ->
// natural-order butterfly (fp32) -> fp16 round with 1/sqrt(128) folded.
__global__ void exl3_had_in_kernel(
    const half* __restrict__ g_x,    // (m, k) fp16 or bf16
    const half* __restrict__ g_suh,  // (k,) fp16
    half* __restrict__ g_out,        // (m, k) fp16
    int k, int x_bf16)
{
    const int row = blockIdx.x;
    const int chunk = blockIdx.y * 8 + (threadIdx.x >> 5);  // 8 warps = 8 chunks
    const int lane = threadIdx.x & 31;
    if (chunk * 128 >= k) return;
    const int base = chunk * 128;

    float v[4];
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        float xf;
        if (x_bf16)
            xf = __bfloat162float(
                reinterpret_cast<const __nv_bfloat16 *>(g_x)[row * k + base + i]);
        else
            xf = __half2float(g_x[row * k + base + i]);
        const half sv = g_suh[base + i];
        // fp16 path: fp16 product round then fp32 (bit-identical to v1);
        // bf16 path: exact fp32 product (bf16->fp32 is lossless).
        v[j] = x_bf16 ? xf * __half2float(sv) : __half2float(__hmul(__float2half_rn(xf), sv));
    }
    had_warp_butterfly(v, lane);
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        g_out[row * k + base + i] = __float2half_rn(v[j] * kRScale);
    }
}

// Local tensor-core fragment types + mma.m16n8k16 helper. The fragment asm
// contract follows the PTX ISA m16n8k16 fragment layout; the helper shape
// follows turboderp's ExLlamaV3 ptx.cuh (MIT, (c) 2025 turboderp) —
// independent implementation for SGLang.
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

// Fused fragment-MMA trellis GEMM + output Hadamard/svh epilogue.
// grid = (n16 / 8, ceil(m / 16)); block = 256 threads (8 warps).
// Per warp: 16 output cols (two m16n8k16 B fragments) x full k; the warp
// decodes its own fragment elements directly to registers (e = lane*8 + j)
// each k-tile — no smem for the weights, no syncthreads in the k loop.
// x rows are hand-assembled into the A fragment directly from g_x
// (m = {lane/4, lane/4+8} x k = {2*lane%4 pairs}).
template <int BITS, int BM>
__global__ void exl3_gemm_kernel(
    const half* __restrict__ g_x,          // (m, k) fp16, Hadamard-transformed
    const uint16_t* __restrict__ g_packed, // (k16, n16, 16*BITS) le int16 words
    const half* __restrict__ g_svh,        // (n,) fp16
    const half* __restrict__ g_bias,       // (n,) fp16 or nullptr
    half* __restrict__ g_out,              // (m, n) fp16 or bf16
    int m, int k, int n, int n16, int cb, int bf16_out)
{
    constexpr int words16 = BITS * 16;   // uint16 words per 16x16 tile
    const int t = threadIdx.x;           // 256 = 8 warps
    const int lane = t & 31;
    const int warp = t >> 5;             // warp's n16 tile within the 128-col block
    const int col_block = blockIdx.x;
    const int row0 = blockIdx.y * 16;
    if (row0 >= m) return;
    const int n_base = col_block * 128;
    const int k16 = k >> 4;
    const int m0 = row0 + (lane >> 2);   // A-fragment lower m-row
    const int m8 = m0 + 8;
    const bool v0 = m0 < m;
    const bool v8 = m8 < m;
    const int kp = (lane & 3) << 1;      // lane's k-pair start within the tile
    const half* x0 = g_x + m0 * k;
    const half* x8 = g_x + m8 * k;

    __shared__ float s_acc[16][128];  // fp32 accum staged once for the epilogue

    FragC frag_c[2];
    frag_c[0] = {};
    frag_c[1] = {};

    // Per-lane decode window params, hoisted out of the k loop (they depend
    // only on e = lane*8 + j): word indices into the circular tile and the
    // funnel-shift amount. Recomputing 2 integer modulo per element per
    // k-tile was ~70% of the whole kernel at decode shapes.
    constexpr int n_words = BITS * 256 / 32;
    int wp0[8], wp1[8], ws0[8];
    for (int j = 0; j < 8; j++)
    {
        const int e = lane * 8 + j;
        const int b0 = e * BITS + BITS - 16 + 256 * BITS;
        const int b1 = b0 + 16;
        const int i0 = b0 / 32;
        const int i1 = (b1 - 1) / 32;
        wp0[j] = i0 % n_words;
        wp1[j] = i1 % n_words;
        ws0[j] = (i1 + 1) * 32 - b1;
    }

    for (int kk = 0; kk < k16; kk++)
    {
        const uint32_t* tile32 =
            (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);

        // decode this lane's own B-fragment elements: e = lane*8 + j.
        // Per lane: fragment 1 (n = lane/4): k rows {r0, r0+1, r0+8, r0+9} =
        // e = lane*8 + {0,1,2,3}; fragment 2 (n = lane/4 + 8): e + {4,5,6,7}.
        half w[8];
        for (int j = 0; j < 8; j++)
            w[j] = dq_decode(tile32, lane * 8 + j, BITS, cb);

        // A fragment (m16 x k16): m = {m0, m8}, k = {kp, kp+1, kp+8, kp+9}.
        const half* x0k = x0 + kk * 16;
        const half* x8k = x8 + kk * 16;
        half z = __float2half(0.0f);
        FragA fa;
        // A packing: (same m, k-pair {2p, 2p+1}); k = {2p, 2p+1, 2p+8, 2p+9}
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

    // stage the D fragment values per the pinned PTX D layout (verified vs
    // the bit-exact reference kernel): per lane, r0 = lane/4, c = 2*(lane%4):
    //   elems[0] = (m r0,     n c),     elems[1] = (m r0,     n c+1),
    //   elems[2] = (m r0 + 8, n c),     elems[3] = (m r0 + 8, n c + 1)
    // frag_c[0] covers n {0..7} of the warp's 16, frag_c[1] covers n {8..15}.
    {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
        if (row0 + r0 < m)
        {
            s_acc[r0][warp * 16 + c0]         = frag_c[0].elems[0];
            s_acc[r0][warp * 16 + c0 + 1]     = frag_c[0].elems[1];
            s_acc[r0 + 8][warp * 16 + c0]     = frag_c[0].elems[2];
            s_acc[r0 + 8][warp * 16 + c0 + 1] = frag_c[0].elems[3];
            s_acc[r0][warp * 16 + 8 + c0]         = frag_c[1].elems[0];
            s_acc[r0][warp * 16 + 8 + c0 + 1] = frag_c[1].elems[1];
            s_acc[r0 + 8][warp * 16 + 8 + c0]     = frag_c[1].elems[2];
            s_acc[r0 + 8][warp * 16 + 8 + c0 + 1] = frag_c[1].elems[3];
        }
    }
    __syncthreads();

    // fused epilogue: output Hadamard (per 128-col block) + svh + bias.
    // Replicates the PyTorch rounding points: fp32 Hadamard (butterfly),
    // fp16 round, then fp16 elementwise svh / bias.
    constexpr int rr = BM >> 3;  // rows per warp in the epilogue
    for (int r = 0; r < rr; r++)
    {
        const int row_local_ep = warp * rr + r;
        const int row_ep = row0 + row_local_ep;
        if (row_ep >= m) continue;
        float v[4];
        for (int j = 0; j < 4; j++)
            v[j] = s_acc[row_local_ep][lane + 32 * j];
        had_warp_butterfly(v, lane);
        for (int j = 0; j < 4; j++)
        {
            const int col = n_base + lane + 32 * j;
            float fv = v[j] * kRScale;
            // Numerics-only epilogue guard: saturate past fp16 range instead of
            // letting the fp16 stores overflow to inf. Tight-calibration
            // checkpoints (the DFlash2 drafter's down_proj moves a base outlier
            // into a ~10x svh pivot, pushing worst-case output dots past 65504
            // AFTER the svh multiply) poison the decode with inf -> NaN
            // otherwise. Each fp16 stage is replaced by its exact fp32
            // product/sum + one RTNE round + saturate: fp16->fp32 is lossless,
            // an fp16*fp16 product and (up to 24-bit mantissa) an fp16+fp16 sum
            // are exact in fp32, so in-range values round bit-identically to
            // __float2half_rn/__hmul/__hadd and only past-range values change
            // (inf -> +-65504). The on-disk decode format is untouched.
            // bf16 output (out dtype bf16): no fp16 ceiling — the drafter's
            // legitimately-large outputs (the vendor runs the drafter in bf16)
            // store exactly, so the saturating guard is skipped entirely.
            if (bf16_out)
            {
                float p = fv * __half2float(g_svh[col]);
                if (g_bias) p += __half2float(g_bias[col]);
                reinterpret_cast<__nv_bfloat16 *>(g_out)[row_ep * n + col] =
                    __float2bfloat16_rn(p);
                continue;
            }
            if (fv > 65504.f) fv = 65504.f; else if (fv < -65504.f) fv = -65504.f;
            half h = __float2half_rn(fv);
            float p = __half2float(h) * __half2float(g_svh[col]);
            if (p > 65504.f) p = 65504.f; else if (p < -65504.f) p = -65504.f;
            h = __float2half_rn(p);
            if (g_bias)
            {
                float b = __half2float(h) + __half2float(g_bias[col]);
                if (b > 65504.f) b = 65504.f; else if (b < -65504.f) b = -65504.f;
                h = __float2half_rn(b);
            }
            g_out[row_ep * n + col] = h;
        }
    }
}

}  // namespace exl3

// ---------------------------------------------------------------------------

namespace exl3 {

// V2 decode-path GEMM (m <= 16). V1's grid gives n16/8 blocks (~136 on a
// 17k-col matrix), each running a long serial k-loop: at one block per SM the
// per-k-tile decode+MMA latency is fully exposed (measured 210us for a
// 44.6MB matrix = ~212GB/s, vs ~500+GB/s for the exllamav3 bar). V2 splits K
// across grid.y (2D grid = col_blocks x splits), shortening each block's
// serial chain so sibling blocks hide the latency, batches each lane's
// window-word loads before the decode chain, and stages fp32 partials in a
// workspace. A separate reduce kernel sums the k-splits in fixed order
// (deterministic) and applies the identical fused epilogue.
template <int BITS, int BM>
__global__ void exl3_gemm_kernel_v2(
    const half* __restrict__ g_x,
    const uint16_t* __restrict__ g_packed,
    float* __restrict__ g_ws,          // (col_blocks, splits, m, n) fp32
    int m, int k, int n, int n16, int cb, int splits)
{
    constexpr int words16 = BITS * 16;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int col_block = blockIdx.x;
    const int ks = blockIdx.y;
    const int k16 = k >> 4;
    const int k_beg = (int)((long)k16 * ks / splits);
    const int k_end = (int)((long)k16 * (ks + 1) / splits);
    if (k_beg >= k_end) return;
    const int n_base = col_block * 128;

    const int m0 = lane >> 2;
    const int m8 = m0 + 8;
    const bool v0 = m0 < m;
    const bool v8 = m8 < m;
    const int kp = (lane & 3) << 1;
    const half* x0 = g_x + m0 * k;
    const half* x8 = g_x + m8 * k;

    constexpr int n_words = BITS * 256 / 32;
    int wp0[8], wp1[8], ws0[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        const int e = lane * 8 + j;
        const int b0 = e * BITS + BITS - 16 + 256 * BITS;
        const int b1 = b0 + 16;
        const int i0 = b0 / 32;
        const int i1 = (b1 - 1) / 32;
        wp0[j] = i0 % n_words;
        wp1[j] = i1 % n_words;
        ws0[j] = (i1 + 1) * 32 - b1;
    }

    FragC frag_c[2];
    frag_c[0] = {};
    frag_c[1] = {};

    for (int kk = k_beg; kk < k_end; kk++)
    {
        const uint32_t* tile32 =
            (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);

        // Batch the lane's window-word loads (independent), then decode from
        // registers — the v1 form re-enters memory inside each dependent
        // decode chain.
        uint32_t wb[16];
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            wb[2 * j] = tile32[wp0[j]];
            wb[2 * j + 1] = tile32[wp1[j]];
        }
        half w[8];
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const uint32_t w0 = __funnelshift_r(wb[2 * j + 1], wb[2 * j], ws0[j]) & 0xffffu;
            w[j] = decode_3inst(w0, cb);
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

    // Stage fp32 partials in (row, col) layout. Fragment element (m, n)
    // mapping per the pinned PTX D layout (see v1's s_acc staging): per lane,
    // frag f covers warp columns {warp*16 + 8f + c0, +1} for rows {m0, m8}.
    const int c0 = (lane & 3) << 1;
    float* ws_row0 = g_ws + (((long)col_block * splits + ks) * m + m0) * n;
    float* ws_row8 = g_ws + (((long)col_block * splits + ks) * m + m8) * n;
    if (v0)
    {
        ws_row0[n_base + warp * 16 + c0]         = frag_c[0].elems[0];
        ws_row0[n_base + warp * 16 + c0 + 1]     = frag_c[0].elems[1];
        ws_row0[n_base + warp * 16 + 8 + c0]     = frag_c[1].elems[0];
        ws_row0[n_base + warp * 16 + 8 + c0 + 1] = frag_c[1].elems[1];
    }
    if (v8)
    {
        ws_row8[n_base + warp * 16 + c0]         = frag_c[0].elems[2];
        ws_row8[n_base + warp * 16 + c0 + 1]     = frag_c[0].elems[3];
        ws_row8[n_base + warp * 16 + 8 + c0]     = frag_c[1].elems[2];
        ws_row8[n_base + warp * 16 + 8 + c0 + 1] = frag_c[1].elems[3];
    }
}

// V2 reduce + fused epilogue: one block per 128-col output block; each warp
// covers 2 rows; per lane, cols {lane, lane+32, lane+64, lane+96} are summed
// over the k-splits in fixed order (deterministic), then the identical
// Hadamard/svh/bias epilogue as v1.
__global__ void exl3_reduce_epilogue_kernel(
    const float* __restrict__ g_ws,   // (col_blocks, splits, m, n) fp32
    const half* __restrict__ g_svh,
    const half* __restrict__ g_bias,  // or nullptr
    half* __restrict__ g_out,         // (m, n) fp16 or bf16
    int m, int n, int splits, int bf16_out)
{
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int col_block = blockIdx.x;
    const int n_base = col_block * 128;
    constexpr int rr = 2;  // rows per warp (m <= 16)
#pragma unroll
    for (int r = 0; r < rr; r++)
    {
        const int row = warp * rr + r;
        if (row >= m) continue;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int col = n_base + lane + 32 * j;
            float acc = 0.0f;
            for (int ks = 0; ks < splits; ks++)
                acc += g_ws[(((long)col_block * splits + ks) * m + row) * n + col];
            v[j] = acc;
        }
        had_warp_butterfly(v, lane);
        for (int j = 0; j < 4; j++)
        {
            const int col = n_base + lane + 32 * j;
            float fv = v[j] * kRScale;
            if (bf16_out)
            {
                float p = fv * __half2float(g_svh[col]);
                if (g_bias) p += __half2float(g_bias[col]);
                reinterpret_cast<__nv_bfloat16 *>(g_out)[(long)row * n + col] =
                    __float2bfloat16_rn(p);
                continue;
            }
            if (fv > 65504.f) fv = 65504.f; else if (fv < -65504.f) fv = -65504.f;
            half h = __float2half_rn(fv);
            float p = __half2float(h) * __half2float(g_svh[col]);
            if (p > 65504.f) p = 65504.f; else if (p < -65504.f) p = -65504.f;
            h = __float2half_rn(p);
            if (g_bias)
            {
                float b = __half2float(h) + __half2float(g_bias[col]);
                if (b > 65504.f) b = 65504.f; else if (b < -65504.f) b = -65504.f;
                h = __float2half_rn(b);
            }
            g_out[(long)row * n + col] = h;
        }
    }
}

}  // namespace exl3

// ---------------------------------------------------------------------------

void sgl_exl3_had_in(at::Tensor x, at::Tensor suh, at::Tensor out)
{
    TORCH_CHECK(x.is_cuda() && suh.is_cuda() && out.is_cuda(), "exl3 had_in: CUDA tensors required");
    TORCH_CHECK((x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                suh.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kHalf, "exl3 had_in: fp16/bf16 x, fp16 suh/out required");
    TORCH_CHECK(x.is_contiguous() && suh.is_contiguous() && out.is_contiguous(),
                "exl3 had_in: contiguous required");
    const int m = x.size(0);
    const int k = x.size(1);
    TORCH_CHECK(k % 128 == 0, "exl3 had_in: k must be a multiple of 128");
    TORCH_CHECK(suh.size(0) == k && out.size(0) == m && out.size(1) == k,
                "exl3 had_in: shape mismatch");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(m, (k + 1023) / 1024);
    exl3::exl3_had_in_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const half *>(x.data_ptr()),
        reinterpret_cast<const half *>(suh.data_ptr<at::Half>()),
        reinterpret_cast<half *>(out.data_ptr<at::Half>()), k,
        x.scalar_type() == at::kBFloat16 ? 1 : 0);
}

void sgl_exl3_linear(
    at::Tensor x,                    // (m, k) fp16, Hadamard-transformed
    at::Tensor packed,               // (k16, n16, 16*bits) int16
    at::Tensor svh,                  // (n,) fp16
    c10::optional<at::Tensor> bias,  // (n,) fp16 or none
    int64_t cb,
    at::Tensor out)                  // (m, n) fp16
{
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && svh.is_cuda() && out.is_cuda(),
                "exl3 linear: CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kHalf && packed.scalar_type() == at::kShort &&
                svh.scalar_type() == at::kHalf &&
                (out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16),
                "exl3 linear: fp16 x/svh, int16 packed, fp16/bf16 out required");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && svh.is_contiguous() &&
                out.is_contiguous(), "exl3 linear: contiguous required");
    const int m = x.size(0);
    const int k = x.size(1);
    const int n = svh.size(0);
    const int k16 = packed.size(0);
    const int n16 = packed.size(1);
    const int bits = packed.size(2) / 16;
    TORCH_CHECK(k == k16 * 16 && n == n16 * 16, "exl3 linear: packed shape mismatch");
    TORCH_CHECK(k % 128 == 0 && n % 128 == 0, "exl3 linear: k/n must be multiples of 128");
    TORCH_CHECK(bits >= 1 && bits <= 8, "exl3 linear: bitrate out of range 1..8");
    TORCH_CHECK(cb >= 0 && cb <= 2, "exl3 linear: cb out of range");
    TORCH_CHECK(out.size(0) == m && out.size(1) == n, "exl3 linear: out shape mismatch");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bias_ptr =
        bias.has_value() && bias->numel() > 0
            ? reinterpret_cast<const half *>(bias->data_ptr<at::Half>())
            : nullptr;
    const uint16_t* packed_ptr =
        reinterpret_cast<const uint16_t *>(packed.data_ptr<int16_t>());
    const half* xp = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    const half* svp = reinterpret_cast<const half *>(svh.data_ptr<at::Half>());
    half* op = reinterpret_cast<half *>(out.data_ptr());  // fp16 or bf16 out
    int icb = (int)cb;
    int mi = m, ki = k, ni = n, n16i = n16;
    const int bf16_out = out.scalar_type() == at::kBFloat16 ? 1 : 0;

    if (m <= 16)
    {
        // V2 decode path: split K so sibling blocks hide the serial decode
        // latency (see kernel comment). Workspace partials are fully written
        // by the gemm (every (col_block, split, row<m, col) cell exactly
        // once), so no zero-init is needed.
        const int col_blocks = n16 / 8;
        int splits = (680 + col_blocks - 1) / col_blocks;
        if (splits < 1) splits = 1;
        const int max_splits = ki / 16 / 8;  // keep >= 8 k-tiles per block
        if (splits > max_splits) splits = max_splits;
        if (splits > 32) splits = 32;
        if (splits < 1) splits = 1;
        auto ws = at::empty(
            {(long)col_blocks * splits * mi * ni}, x.options().dtype(at::kFloat));
        float* wsp = ws.data_ptr<float>();
        dim3 grid(col_blocks, splits);
        switch (bits)
        {
            case 1: exl3::exl3_gemm_kernel_v2<1, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 2: exl3::exl3_gemm_kernel_v2<2, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 3: exl3::exl3_gemm_kernel_v2<3, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 4: exl3::exl3_gemm_kernel_v2<4, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 5: exl3::exl3_gemm_kernel_v2<5, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 6: exl3::exl3_gemm_kernel_v2<6, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 7: exl3::exl3_gemm_kernel_v2<7, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            case 8: exl3::exl3_gemm_kernel_v2<8, 16><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits); break;
            default: TORCH_CHECK(false, "exl3 linear: unsupported bitrate");
        }
        exl3::exl3_reduce_epilogue_kernel<<<col_blocks, 256, 0, stream>>>(
            wsp, svp, bias_ptr, op, mi, ni, splits, bf16_out);
        return;
    }

#define EXL3_DISPATCH(B)                                                                   \
    exl3::exl3_gemm_kernel<B, 16><<<grid, 256, 0, stream>>>(                               \
        xp, packed_ptr, svp, bias_ptr, op, mi, ki, ni, n16i, icb, bf16_out);               \
    break;
    dim3 grid(n16 / 8, (m + 15) / 16);
    switch (bits)
    {
        case 1: EXL3_DISPATCH(1);
        case 2: EXL3_DISPATCH(2);
        case 3: EXL3_DISPATCH(3);
        case 4: EXL3_DISPATCH(4);
        case 5: EXL3_DISPATCH(5);
        case 6: EXL3_DISPATCH(6);
        case 7: EXL3_DISPATCH(7);
        case 8: EXL3_DISPATCH(8);
        default: TORCH_CHECK(false, "exl3 linear: unsupported bitrate");
    }
#undef EXL3_DISPATCH
}
