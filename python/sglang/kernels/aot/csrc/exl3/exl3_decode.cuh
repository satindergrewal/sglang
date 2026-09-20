// EXL3 trellis decode primitives for SGLang.
//
// ATTRIBUTION (per-file provenance):
// Portions of this file are derived from turboderp's ExLlamaV3 (MIT License,
// Copyright (c) 2025 turboderp) — github.com/turboderp-org/exllamav3 — from
// exllamav3/exllamav3_ext/quant/codebook.cuh and exllamav3/exllamav3_ext/
// quant/exl3_dq.cuh. Those files carry the license below, reproduced in full,
// and the decoded bitstream MUST match the EXL3 on-disk format that ExLlamaV3
// defines exactly:
//
//   MIT License
//
//   Copyright (c) 2025 turboderp
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
//
// LOCAL MODIFICATIONS (vs the ExLlamaV3 1.4.2 sources):
//   - codebook.cuh and exl3_dq.cuh are merged into this header, constrained
//     to the general 16-bit funnel-shift window decode (dq_decode) used by
//     the SGLang M-tiled GEMM; the aligned 1/2/4-bit fast paths are perf-only
//     and omitted.
//   - decode_3inst takes the codebook as a runtime arg instead of a template
//     parameter (the dispatch table per (K, cb) instance set is not needed
//     with the per-element window decode).
//   - The 16x16 tile element order is the tensor-core fragment order carried
//     through from exllamav3's serializer (quantize.py tensor_core_perm); the
//     inverse map TENSOR_CORE_PERM_I below is defined to match it exactly
//     (verified bit-exact against exllamav3 1.4.2 by this project's pure
//     PyTorch regression suite).
//   - No change to the decode math (windows, codebooks, fragment order):
//     decoding must match the on-disk format bit-exactly.
//
// Design context: the M-tiling / single-read batched GEMM shape follows
// cuda-exl3 (MIT, Copyright (c) 2026 cuda-exl3 contributors). QTIP needs no
// notice (the trellis/numeric format credit goes to turboderp ExLlamaV3).

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace exl3 {

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t)(merged >> shift);
}

// Procedural codebooks (verbatim decode math from ExLlamaV3 codebook.cuh):
//   cb 0 ("3inst", no marker): x = w*89226354 + 64248484, lop3 majority select
//   cb 1 (mcg, marker 0xCBAC1FED): x = w * 0xCBAC1FED, lop3 majority select
//   cb 2 (mul1, marker 0x83DCD12D): x = w * 0x83DCD12D, dp4a byte-sum decode
__device__ inline half decode_3inst(uint32_t x, int cb)
{
    if (cb == 0)
    {
        x *= 89226354u;
        x += 64248484u;
        asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
        half2 d = __halves2half2(__ushort_as_half((uint16_t)(x & 0xffffu)),
                                 __ushort_as_half((uint16_t)(x >> 16)));
        return __hadd(__low2half(d), __high2half(d));
    }
    if (cb == 1)
    {
        x *= 0xCBAC1FEDu;
        asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
        half2 d = __halves2half2(__ushort_as_half((uint16_t)(x & 0xffffu)),
                                 __ushort_as_half((uint16_t)(x >> 16)));
        return __hadd(__low2half(d), __high2half(d));
    }
    // cb 2 (mul1)
    x *= 0x83DCD12Du;
    const uint32_t acc = 0x6400u;  // fp16(bs) embeds 1024.0 .. 2047.0 exactly
    uint32_t sum = __dp4a(x, 0x01010101u, acc);  // byte_sum(x) + 1024, bit-identical to vabsdiff4
    const __half k_inv_h = __ushort_as_half(0x1eee);  //  0.00677 = 1/147.7
    const __half k_bias_h = __ushort_as_half(0xc931); // -10.39 = (-1024.0 - 510.0) * k_inv
    __half h = __ushort_as_half((uint16_t)(sum & 0xffffu));
    return __hfma(h, k_inv_h, k_bias_h);
}

// General 16-bit codebook window for element t_offset of one 16x16 tile.
// Circular indexing and window alignment are verbatim ExLlamaV3 exl3_dq.cuh:
// element t's window spans bits [(t+1)*b + 256*b - 16, (t+1)*b + 256*b).
__device__ inline half dq_decode(const uint32_t* ptr, int t_offset, int bits, int cb)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // start of word0
    int b1 = b0 + 16;                                   // end of word0
    int i0 = b0 / 32;
    int i1 = (b1 - 1) / 32;
    int s0 = (i1 + 1) * 32 - b1;
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];
    uint32_t w0 = __funnelshift_r(b, a, s0) & 0xffff;
    return decode_3inst(w0, cb);
}

namespace {
struct TensorCorePermI {
    int v[256];
    constexpr TensorCorePermI() : v{}
    {
        for (int t = 0; t < 32; t++)
        {
            int r0 = (t % 4) * 2;
            int c0 = t / 4;
            const int entries[8] = {
                (r0) * 16 + c0,      (r0 + 1) * 16 + c0,
                (r0 + 8) * 16 + c0,  (r0 + 9) * 16 + c0,
                (r0) * 16 + c0 + 8,  (r0 + 1) * 16 + c0 + 8,
                (r0 + 8) * 16 + c0 + 8, (r0 + 9) * 16 + c0 + 8,
            };
            for (int j = 0; j < 8; j++) v[t * 8 + j] = entries[j];
        }
    }
};
constexpr TensorCorePermI tensor_core_perm_i{};
}  // namespace

// Matrix (row, col) within one 16x16 tile for tile element e, in the
// tensor-core fragment order carried through from exllamav3's serializer
// (inverse of quantize.py tensor_core_perm; e indexes the tile in the packed
// bitstream order the encode produced). Computed inline: nvcc does not inline
// host-side constexpr-initialized globals into device code that references
// them, so the mapping folds into the call.
__device__ __forceinline__ int tile_matrix_pos(int e)
{
    const int t = e >> 3;           // lane of the m16n8k16 fragment
    const int j = e & 7;            // element within the lane's 8
    const int r0 = ((t & 3) << 1);
    const int c0 = t >> 2;
    const int rdelta[8] = {0, 1, 8, 9, 0, 1, 8, 9};
    const int cdelta[8] = {0, 0, 0, 0, 8, 8, 8, 8};
    return (r0 + rdelta[j]) * 16 + c0 + cdelta[j];
}

}  // namespace exl3
