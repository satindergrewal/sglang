// Standalone EXL3 ops module: registers sgl_exl3_had_in + sgl_exl3_linear
// (from exl3_linear.cu) and the grouped variant into the sgl_kernel namespace,
// letting the 5090 dev box run EXL3 weights without the full AOT build.
// Same mechanism as the proven exl3_patch.so override.
#include <torch/extension.h>

// sgl_exl3_had_in / sgl_exl3_linear live at global scope in exl3_linear.cu;
// the grouped variant is inside namespace exl3 (exl3_grouped.cu).
void sgl_exl3_had_in(at::Tensor x, at::Tensor suh, at::Tensor out);
void sgl_exl3_linear(at::Tensor x_had, at::Tensor packed, at::Tensor svh,
                     c10::optional<at::Tensor> bias, int64_t cb, at::Tensor out);
namespace exl3 {
void sgl_exl3_grouped_linear(at::Tensor x, at::Tensor packed_ptrs,
                             at::Tensor svh_ptrs, at::Tensor bias_ptrs,
                             at::Tensor counts, at::Tensor offsets, at::Tensor ws,
                             at::Tensor cnt_buf, int64_t cb, int64_t splits_in,
                             int64_t chunks_cap_in, int64_t bits_in,
                             int64_t half_k_in, at::Tensor out);
}  // namespace exl3

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)
{
    m.def("sgl_exl3_had_in(Tensor x, Tensor suh, Tensor(a!) out) -> ()");
    m.def("sgl_exl3_linear(Tensor x_had, Tensor packed, Tensor svh, Tensor? bias, int cb, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY(sgl_exl3_grouped, m)
{
    m.def("grouped_linear(Tensor x, Tensor packed_ptrs, Tensor svh_ptrs, Tensor bias_ptrs, "
          "Tensor counts, Tensor offsets, Tensor ws, Tensor cnt_buf, int cb, int splits, "
          "int chunks_cap, int bits, int half_k, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m)
{
    m.impl("sgl_exl3_had_in", torch::kCUDA, &sgl_exl3_had_in);
    m.impl("sgl_exl3_linear", torch::kCUDA, &sgl_exl3_linear);
}

TORCH_LIBRARY_IMPL(sgl_exl3_grouped, CUDA, m)
{
    m.impl("grouped_linear", torch::kCUDA, &exl3::sgl_exl3_grouped_linear);
}
