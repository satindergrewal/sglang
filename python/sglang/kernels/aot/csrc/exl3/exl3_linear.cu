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
    const half* __restrict__ g_x,    // (m, k) fp16
    const half* __restrict__ g_suh,  // (k,) fp16
    half* __restrict__ g_out,        // (m, k) fp16
    int k)
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
        const half xv = g_x[row * k + base + i];
        const half sv = g_suh[base + i];
        v[j] = __half2float(__hmul(xv, sv));  // fp16 product round, then fp32
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
    half* __restrict__ g_out,              // (m, n) fp16
    int m, int k, int n, int n16, int cb)
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

void sgl_exl3_had_in(at::Tensor x, at::Tensor suh, at::Tensor out)
{
    TORCH_CHECK(x.is_cuda() && suh.is_cuda() && out.is_cuda(), "exl3 had_in: CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == at::kHalf && suh.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kHalf, "exl3 had_in: fp16 required");
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
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(suh.data_ptr<at::Half>()),
        reinterpret_cast<half *>(out.data_ptr<at::Half>()), k);
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
                svh.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "exl3 linear: fp16 x/svh/out, int16 packed required");
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
    dim3 grid(n16 / 8, (m + 15) / 16);
    const half* xp = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    const half* svp = reinterpret_cast<const half *>(svh.data_ptr<at::Half>());
    half* op = reinterpret_cast<half *>(out.data_ptr<at::Half>());
    int icb = (int)cb;
    int mi = m, ki = k, ni = n, n16i = n16;

#define EXL3_DISPATCH(B)                                                                   \
    exl3::exl3_gemm_kernel<B, 16><<<grid, 256, 0, stream>>>(                               \
        xp, packed_ptr, svp, bias_ptr, op, mi, ki, ni, n16i, icb);                         \
    break;
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
