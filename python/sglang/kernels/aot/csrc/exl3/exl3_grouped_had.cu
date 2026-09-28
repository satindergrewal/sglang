// GROUPED per-expert input Hadamard for the packed MoE decode path.
//
// One launch transforms ALL pair rows: each block handles one (expert,
// row-chunk) slot from device-resident counts/offsets, so the launch grid is
// fixed at capture (CUDA-graph-safe) and all routing data-dependence is read
// on device at replay. Rows belonging to experts with count 0 or beyond the
// per-expert capacity are zeroed in g_out (the downstream grouped GEMM never
// reads them, and the combine must not see garbage).
//
// grid = (ceil(max_rows_per_expert), E); block = 256 threads (8 warps = 8
// 128-wide chunks per row). suh pointers are per-expert (each projection owns
// its suh vector).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "utils.h"

#include "exl3/exl3_had.cuh"

namespace exl3 {

constexpr float kRScaleG2 = 0.088388347648f;  // 1 / sqrt(128)

__global__ void exl3_grouped_had_in_kernel(
    const half* __restrict__ g_x,               // (P, k) fp16 pair rows
    const half* const* __restrict__ g_suh_arr,  // [E] per-expert suh
    const int* __restrict__ g_counts,           // [E]
    const long* __restrict__ g_offsets,         // [E]
    half* __restrict__ g_out,                   // (P, k) fp16
    int rows_cap,                               // per-expert row capacity
    int k)
{
    const int e = blockIdx.y;
    const int row_in = blockIdx.x;  // row index within this expert's segment
    const int cnt = g_counts[e];
    const int chunk = blockIdx.z * 8 + (threadIdx.x >> 5);  // 8 warps/block
    const int lane = threadIdx.x & 31;

    const bool row_valid = (row_in < cnt) && (row_in < rows_cap);
    if (chunk * 128 >= k) return;
    const int base = chunk * 128;
    const long src_row = (long)g_offsets[e] + row_in;

    const half* g_suh = g_suh_arr[e];

    float v[4];
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        float xf = 0.0f;
        half sv = __float2half(0.0f);
        if (row_valid)
        {
            xf = __half2float(g_x[src_row * k + base + i]);
            sv = g_suh[base + i];
        }
        v[j] = row_valid ? __half2float(__hmul(__float2half_rn(xf), sv)) : 0.0f;
    }
    had_warp_butterfly(v, lane);
    // NOTE: rows beyond this expert's count belong to the NEXT expert's
    // segment (offsets are a prefix sum) — they must be left untouched; the
    // grouped GEMM never reads past count rows of each segment.
    if (!row_valid) return;
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        g_out[src_row * k + base + i] = __float2half_rn(v[j] * kRScaleG2);
    }
}

}  // namespace exl3

namespace exl3 {
void sgl_exl3_grouped_had_in(
    at::Tensor x,               // (P, k) fp16
    at::Tensor suh_ptrs,        // (E,) int64 device pointers
    at::Tensor counts,          // (E,) int32
    at::Tensor offsets,         // (E,) int64
    int64_t rows_cap,
    at::Tensor out)             // (P, k) fp16 (may contain stale rows; kernel
                                //  zeroes every row it does not transform)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kHalf,
                "exl3 grouped had_in: fp16 required");
    const int E = (int)suh_ptrs.size(0);
    const int P = x.size(0);
    const int k = x.size(1);
    const int rows_cap_i = (int)rows_cap;
    TORCH_CHECK(k % 1024 == 0, "exl3 grouped had_in: k must be a multiple of 1024");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const half** suh_arr = reinterpret_cast<const half**>(suh_ptrs.data_ptr<int64_t>());
    const half* xp = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    half* op = reinterpret_cast<half *>(out.data_ptr());
    const int* counts_p = counts.data_ptr<int>();
    const long* offsets_p = offsets.data_ptr<long>();
    TORCH_CHECK(k % 1024 == 0, "exl3 grouped had_in: k must be a multiple of 1024");
    dim3 grid3(rows_cap_i, E, k / 1024);
    exl3::exl3_grouped_had_in_kernel<<<grid3, 256, 0, stream>>>(
        xp, suh_arr, counts_p, offsets_p, op, rows_cap_i, k);
}
}  // namespace exl3

// ---------------------------------------------------------------------------
// Pair-indexed variant: grid = (P, k/1024) — one block per PAIR ROW (x_pairs
// is already expert-sorted, so the row id IS the pair id and the expert comes
// from sids[row]). No (rows_cap x E) wasted blocks: at decode (P ~ 32,
// E = 256) the old grid launched ~66k mostly-empty blocks; this one launches
// P * k/1024. CUDA-graph-safe: the grid depends only on P (fixed at capture),
// and the expert lookup is a device read at replay.
namespace exl3 {

__global__ void exl3_grouped_had_in_pairs_kernel(
    const half* __restrict__ g_x,               // (P, k) fp16 pair rows
    const int* __restrict__ g_sids,             // (P,) expert id per pair
    const half* const* __restrict__ g_suh_arr,  // [E] per-expert suh
    int num_experts,
    half* __restrict__ g_out,                   // (P, k) fp16
    int k)
{
    const int row = blockIdx.x;
    const int e = g_sids[row];
    const int chunk = blockIdx.y * 8 + (threadIdx.x >> 5);  // 8 warps/block
    const int lane = threadIdx.x & 31;
    if (chunk * 128 >= k) return;
    if (e >= num_experts) return;  // sentinel rows: never read downstream

    const int base = chunk * 128;
    const half* g_suh = g_suh_arr[e];

    float v[4];
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        v[j] = __half2float(__hmul(g_x[(long)row * k + base + i], g_suh[base + i]));
    }
    had_warp_butterfly(v, lane);
    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        g_out[(long)row * k + base + i] = __float2half_rn(v[j] * kRScaleG2);
    }
}

}  // namespace exl3

namespace exl3 {
void sgl_exl3_grouped_had_in_pairs(
    at::Tensor x,               // (P, k) fp16
    at::Tensor sids,            // (P,) int32 expert id per pair (E sentinel ok)
    at::Tensor suh_ptrs,        // (E,) int64 device pointers
    at::Tensor out)             // (P, k) fp16
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kHalf &&
                sids.scalar_type() == at::kInt,
                "exl3 grouped had_in_pairs: fp16 x/out, int32 sids required");
    const int E = (int)suh_ptrs.size(0) - 1;  // arrays carry a null sentinel
    const int P = x.size(0);
    const int k = x.size(1);
    TORCH_CHECK(k % 1024 == 0, "exl3 grouped had_in_pairs: k must be a multiple of 1024");
    TORCH_CHECK(sids.size(0) == P, "exl3 grouped had_in_pairs: sids length mismatch");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const half** suh_arr = reinterpret_cast<const half**>(suh_ptrs.data_ptr<int64_t>());
    const half* xp = reinterpret_cast<const half *>(x.data_ptr<at::Half>());
    half* op = reinterpret_cast<half *>(out.data_ptr());
    const int* sids_p = sids.data_ptr<int>();
    dim3 grid3(P, k / 1024, 1);
    exl3::exl3_grouped_had_in_pairs_kernel<<<grid3, 256, 0, stream>>>(
        xp, sids_p, suh_arr, E, op, k);
}
}  // namespace exl3
