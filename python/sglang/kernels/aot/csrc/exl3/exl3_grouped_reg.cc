// Standalone patch module: registers sgl_exl3_grouped_linear into the
// sgl_kernel op namespace WITHOUT touching the AOT build (same mechanism as
// exl3_patch.so). Built with torch.utils.cpp_extension.load at dev time.
#include <torch/extension.h>

namespace exl3 {
void sgl_exl3_grouped_linear(
    at::Tensor x, at::Tensor packed_ptrs, at::Tensor svh_ptrs,
    at::Tensor bias_ptrs, at::Tensor counts, at::Tensor offsets,
    at::Tensor ws, at::Tensor cnt_buf, int64_t cb, int64_t splits_in,
    int64_t chunks_cap_in, int64_t bits_in, int64_t half_k_in, at::Tensor out);
}  // namespace exl3

TORCH_LIBRARY(sgl_exl3_grouped, m)
{
    m.def("grouped_linear(Tensor x, Tensor packed_ptrs, Tensor svh_ptrs, Tensor bias_ptrs, "
          "Tensor counts, Tensor offsets, Tensor ws, Tensor cnt_buf, int cb, int splits, "
          "int chunks_cap, int bits, int half_k, Tensor(a!) out) -> ()");
    m.impl("grouped_linear", torch::kCUDA, &exl3::sgl_exl3_grouped_linear);
}
