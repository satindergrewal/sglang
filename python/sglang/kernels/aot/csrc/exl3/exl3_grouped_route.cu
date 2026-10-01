// Fused routing for the packed MoE decode path (P = bs*topk pair rows, small).
//
// The torch chain (where/argsort/histc/cat/cumsum/gather/casts) is ~10 tiny
// launches (~80us) that are pure overhead at decode sizes. Two launches:
//
//   1. exl3_route_sort_kernel  — ONE block: bitonic-sort (key, idx) pairs in
//      shared memory, emit sids (sorted expert ids), order (original flat
//      positions), tok (pair -> token), counts (E+1, sentinel zeroed) and
//      offsets (E+2). Capacity P <= ROUTE_MAX_PAIRS, padded with a top
//      sentinel so the sort network is a fixed shape (graph-safe).
//
//   2. exl3_route_gather_kernel — grid (P, k/1024): materialize x_pairs from
//      the sorted token ids and the scaled routing weights (zeroed for
//      sentinel rows) in the same pass.
//
// Bitonic sort is not stable, which is fine: pair rows within an expert
// segment may permute freely as long as tok/pw follow the same permutation
// (the combine is a per-pair weighted index_add).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace exl3 {

constexpr int ROUTE_MAX_PAIRS = 512;
constexpr int ROUTE_PAD = 512;  // fixed sort width (power of two)
constexpr int ROUTE_TOP_SENTINEL = 0x7fffffff;

__global__ void exl3_route_sort_kernel(
    const int* __restrict__ g_keys,      // (P,) expert id per pair (>= E allowed)
    int P,
    int num_experts,
    int topk,
    int* __restrict__ g_sids,            // (P,) sorted expert ids
    int* __restrict__ g_order,           // (P,) original flat pair positions
    int* __restrict__ g_tok,             // (P,) order[i] / topk
    int* __restrict__ g_counts,          // (E+1,) counts, sentinel slot zero
    long* __restrict__ g_offsets)        // (E+2,) exclusive prefix of counts
{
    __shared__ int s_key[ROUTE_PAD];
    __shared__ int s_idx[ROUTE_PAD];
    __shared__ int s_cnt[1024];  // supports E <= 1023 (+1 sentinel slot)
    __shared__ long s_off[1025];

    const int tid = threadIdx.x;
    // Load + pad. Expert ids >= E (remote/sentinel pairs) keep their value
    // for counts bookkeeping but must sort BELOW the pad sentinel.
    if (tid < ROUTE_PAD)
    {
        if (tid < P)
        {
            // Capture dummies can carry negative "invalid" expert ids; clamp
            // them (and any out-of-range value) to the sentinel slot E.
            const int k = g_keys[tid];
            s_key[tid] = (k < 0 || k > num_experts) ? num_experts : k;
            s_idx[tid] = tid;
        }
        else
        {
            s_key[tid] = ROUTE_TOP_SENTINEL;
            s_idx[tid] = -1;
        }
    }
    __syncthreads();

    // Bitonic sort ascending by key.
    for (int k = 2; k <= ROUTE_PAD; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            const int ixj = tid ^ j;
            if (ixj > tid)
            {
                const bool desc = ((tid & k) == 0);
                const int ka = s_key[tid], kb = s_key[ixj];
                if ((ka > kb) == desc)
                {
                    s_key[tid] = kb; s_key[ixj] = ka;
                    const int ia = s_idx[tid], ib = s_idx[ixj];
                    s_idx[tid] = ib; s_idx[ixj] = ia;
                }
            }
            __syncthreads();
        }
    }

    if (tid < P)
    {
        g_sids[tid] = s_key[tid];
        g_order[tid] = s_idx[tid];
        g_tok[tid] = s_idx[tid] / topk;
    }

    // Counts over the first num_experts+1 slots (sentinel slot E stays 0
    // unless real pairs carry id E; the runner zeroes its weight either way).
    if (tid < 1024) s_cnt[tid] = 0;
    __syncthreads();
    if (tid < P && s_key[tid] <= num_experts)
        atomicAdd(&s_cnt[s_key[tid]], 1);
    __syncthreads();
    if (tid == 0)
    {
        long acc = 0;
        for (int e = 0; e <= num_experts; e++)
        {
            g_counts[e] = s_cnt[e];
            s_off[e] = acc;
            acc += s_cnt[e];
        }
        g_counts[num_experts] = 0;  // sentinel slot: zero (remote pairs)
        s_off[num_experts] = acc - g_counts[num_experts];
        // Recompute offsets with the sentinel forced to zero: remote pairs
        // sort to the tail; their rows are excluded from every segment.
        acc = 0;
        for (int e = 0; e <= num_experts; e++)
        {
            g_offsets[e] = acc;
            acc += (e < num_experts ? s_cnt[e] : 0);
        }
        g_offsets[num_experts + 1] = acc;
    }
}

__global__ void exl3_route_gather_kernel(
    const half* __restrict__ g_x,        // (bs, k)
    const int* __restrict__ g_order,     // (P,)
    const int* __restrict__ g_tok,       // (P,)
    const int* __restrict__ g_sids,      // (P,)
    const float* __restrict__ g_wts,     // (P,) routing weights (flat)
    int num_experts,
    int topk,
    float wscale,                        // comp * routed_scaling_factor
    half* __restrict__ g_xp,             // (P, k)
    float* __restrict__ g_pw)            // (P,)
{
    const int pair = blockIdx.x;
    const int chunk = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (chunk * 128 >= gridDim.y * 1024) return;
    const int k = gridDim.y * 1024;
    const int base = chunk * 128;
    const int src_tok = g_tok[pair];
    const int src_flat = g_order[pair];

    for (int j = 0; j < 4; j++)
    {
        const int i = lane + 32 * j;
        g_xp[(long)pair * k + base + i] = g_x[(long)src_tok * k + base + i];
    }
    if (chunk == 0 && lane == 0)
    {
        const int e = g_sids[pair];
        g_pw[pair] = (e < num_experts) ? g_wts[src_flat] * wscale : 0.0f;
    }
}

}  // namespace exl3

namespace exl3 {
void sgl_exl3_route_sort(
    at::Tensor keys,        // (P,) int32
    int64_t num_experts,
    int64_t topk,
    at::Tensor sids,        // (P,) int32 out
    at::Tensor order,       // (P,) int32 out
    at::Tensor tok,         // (P,) int32 out
    at::Tensor counts,      // (E+1,) int32 out
    at::Tensor offsets)     // (E+2,) int64 out
{
    TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kInt,
                "exl3 route_sort: int32 keys required");
    const int P = (int)keys.size(0);
    TORCH_CHECK(P <= ROUTE_MAX_PAIRS, "exl3 route_sort: P exceeds kernel capacity");
    TORCH_CHECK(num_experts < 1024, "exl3 route_sort: E exceeds kernel capacity");
    const at::cuda::OptionalCUDAGuard guard(keys.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    exl3_route_sort_kernel<<<1, ROUTE_PAD, 0, stream>>>(
        keys.data_ptr<int>(), P, (int)num_experts, (int)topk,
        sids.data_ptr<int>(), order.data_ptr<int>(), tok.data_ptr<int>(),
        counts.data_ptr<int>(), offsets.data_ptr<long>());
}

void sgl_exl3_route_gather(
    at::Tensor x,           // (bs, k) fp16
    at::Tensor order,       // (P,) int32
    at::Tensor tok,         // (P,) int32
    at::Tensor sids,        // (P,) int32
    at::Tensor wts,         // (P,) fp32
    int64_t num_experts,
    int64_t topk,
    double wscale,
    at::Tensor xp,          // (P, k) fp16 out
    at::Tensor pw)          // (P,) fp32 out
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                xp.scalar_type() == at::kHalf,
                "exl3 route_gather: fp16 x/xp required");
    const int P = (int)xp.size(0);
    const int k = (int)x.size(1);
    TORCH_CHECK(k % 1024 == 0, "exl3 route_gather: k must be a multiple of 1024");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid3(P, k / 1024, 1);
    exl3_route_gather_kernel<<<grid3, 256, 0, stream>>>(
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        order.data_ptr<int>(), tok.data_ptr<int>(), sids.data_ptr<int>(),
        wts.data_ptr<float>(), (int)num_experts, (int)topk, (float)wscale,
        reinterpret_cast<half *>(xp.data_ptr<at::Half>()),
        pw.data_ptr<float>());
}
}  // namespace exl3
