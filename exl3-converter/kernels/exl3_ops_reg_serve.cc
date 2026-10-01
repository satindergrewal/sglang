// Serve variant: registers ONLY the grouped op (own namespace). The dense
// ops (sgl_exl3_had_in / sgl_exl3_linear) come from exl3_patch.so, which
// loads first; re-registering those schemas here would abort the process.
#include <torch/extension.h>

namespace exl3 {
void sgl_exl3_grouped_had_in(at::Tensor x, at::Tensor suh_ptrs,
                             at::Tensor counts, at::Tensor offsets,
                             int64_t rows_cap, at::Tensor out);
void sgl_exl3_grouped_had_in_pairs(at::Tensor x, at::Tensor sids,
                                   at::Tensor suh_ptrs, at::Tensor out);
void sgl_exl3_route_sort(at::Tensor keys, int64_t num_experts, int64_t topk,
                         at::Tensor sids, at::Tensor order, at::Tensor tok,
                         at::Tensor counts, at::Tensor offsets);
void sgl_exl3_route_gather(at::Tensor x, at::Tensor order, at::Tensor tok,
                           at::Tensor sids, at::Tensor wts, int64_t num_experts,
                           int64_t topk, double wscale, at::Tensor xp,
                           at::Tensor pw);
void sgl_exl3_grouped_linear_direct_b(at::Tensor x, at::Tensor packed_ptrs,
                                      at::Tensor svh_ptrs, at::Tensor bias_ptrs,
                                      at::Tensor counts, at::Tensor offsets,
                                      int64_t cb, int64_t bits_in, int64_t half_k_in,
                                      int64_t splits_in, at::Tensor scratch,
                                      at::Tensor out);
void sgl_exl3_grouped_linear_direct(at::Tensor x, at::Tensor packed_ptrs,
                                    at::Tensor svh_ptrs, at::Tensor bias_ptrs,
                                    at::Tensor counts, at::Tensor offsets,
                                    int64_t cb, int64_t bits_in, int64_t half_k_in,
                                    int64_t splits_in, at::Tensor scratch,
                                    at::Tensor out);
void sgl_exl3_grouped_linear(at::Tensor x, at::Tensor packed_ptrs,
                             at::Tensor svh_ptrs, at::Tensor bias_ptrs,
                             at::Tensor counts, at::Tensor offsets, at::Tensor ws,
                             at::Tensor cnt_buf, int64_t cb, int64_t splits_in,
                             int64_t chunks_cap_in, int64_t bits_in,
                             int64_t half_k_in, at::Tensor out);
}  // namespace exl3

TORCH_LIBRARY(sgl_exl3_grouped, m)
{
    m.def("grouped_had_in(Tensor x, Tensor suh_ptrs, Tensor counts, Tensor offsets, int rows_cap, Tensor(a!) out) -> ()");
    m.def("grouped_had_in_pairs(Tensor x, Tensor sids, Tensor suh_ptrs, Tensor(a!) out) -> ()");
    m.def("route_sort(Tensor keys, int num_experts, int topk, Tensor(a!) sids, Tensor(a!) order, Tensor(a!) tok, Tensor(a!) counts, Tensor(a!) offsets) -> ()");
    m.def("route_gather(Tensor x, Tensor order, Tensor tok, Tensor sids, Tensor wts, int num_experts, int topk, float wscale, Tensor(a!) xp, Tensor(a!) pw) -> ()");
    m.def("grouped_linear(Tensor x, Tensor packed_ptrs, Tensor svh_ptrs, Tensor bias_ptrs, "
          "Tensor counts, Tensor offsets, Tensor ws, Tensor cnt_buf, int cb, int splits, "
          "int chunks_cap, int bits, int half_k, Tensor(a!) out) -> ()");
    m.def("grouped_linear_direct(Tensor x, Tensor packed_ptrs, Tensor svh_ptrs, Tensor bias_ptrs, "
          "Tensor counts, Tensor offsets, int cb, int bits, int half_k, int splits, "
          "Tensor scratch, Tensor(a!) out) -> ()");
    m.def("grouped_linear_direct_b(Tensor x, Tensor packed_ptrs, Tensor svh_ptrs, Tensor bias_ptrs, "
          "Tensor counts, Tensor offsets, int cb, int bits, int half_k, int splits, "
          "Tensor scratch, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_exl3_grouped, CUDA, m)
{
    m.impl("grouped_had_in", torch::kCUDA, &exl3::sgl_exl3_grouped_had_in);
    m.impl("grouped_had_in_pairs", torch::kCUDA, &exl3::sgl_exl3_grouped_had_in_pairs);
    m.impl("route_sort", torch::kCUDA, &exl3::sgl_exl3_route_sort);
    m.impl("route_gather", torch::kCUDA, &exl3::sgl_exl3_route_gather);
    m.impl("grouped_linear", torch::kCUDA, &exl3::sgl_exl3_grouped_linear);
    m.impl("grouped_linear_direct", torch::kCUDA, &exl3::sgl_exl3_grouped_linear_direct);
    m.impl("grouped_linear_direct_b", torch::kCUDA, &exl3::sgl_exl3_grouped_linear_direct_b);
}
