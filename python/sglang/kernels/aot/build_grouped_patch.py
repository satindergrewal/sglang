import os, sys, torch
from torch.utils.cpp_extension import load

ROOT = '/home/satinder/Documents/Github/sglang/python/sglang/kernels/aot'
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.0a')
mod = load(
    name='exl3_ops_patch_serve',
    sources=[
        f'{ROOT}/csrc/exl3/exl3_linear.cu',
        f'{ROOT}/csrc/exl3/exl3_grouped.cu',
        f'{ROOT}/csrc/exl3/exl3_grouped_had.cu',
        f'{ROOT}/csrc/exl3/exl3_ops_reg_serve.cc',
    ],
    extra_include_paths=[f'{ROOT}/csrc', f'{ROOT}/include'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_120a,code=sm_120a'],
    extra_ldflags=[f'-Wl,-rpath,{torch.__path__[0]}/lib'],
    verbose=True,
    is_python_module=False,
)
print('grouped patch built:', mod)
