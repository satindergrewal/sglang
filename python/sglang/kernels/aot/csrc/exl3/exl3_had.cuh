// EXL3 Hadamard butterfly helpers for SGLang.
//
// ATTRIBUTION:
// - The butterfly structure follows the natural-order Sylvester butterfly of
//   turboderp's ExLlamaV3 hadamard_inner.cuh (MIT, (c) 2025 turboderp);
//   normalization is 1/sqrt(128) after the last stage, matching the PyTorch
//   reference (fp32 Hadamard, fp16 round, then fp16 elementwise svh / bias).
// - Independent: the warp-level implementation and host wrappers were
//   written for SGLang.
//
// Hadamard convention (pinned by this project's regression suite against
// exllamav3 1.4.2): H128 is the natural-order Sylvester matrix,
// H128[i][j] = (-1)^popcount(i & j), applied as fp32 matmuls
// preapply_had_l (H @ x along the input dim) and preapply_had_r
// (x @ H along the output dim), per 128-block.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace exl3 {

// One warp: 128 fp32 values (one Hadamard block, one row) held as v[4] with
// i = lane + 32 * j. Natural-order Sylvester butterfly: stages are xor
// pairs, sign by the SELF element's bit (element i gets -x_i iff bit d of i
// is set, always +x_{i^d} on the partner). Verified against the H128 matmul
// reference by brute-force sign probe (fp32-exact match); note the n=2
// identity: H2 row 1 = x0 - x1 = partner - self.
// Stages commute; d = 64 and 32 are register pairs, 16..1 are shfl_xor.
__device__ __forceinline__ void had_warp_butterfly(float* v, int lane)
{
    // d = 64: partner register j^2, sign bit = (j >> 1) & 1
    {
        float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a2;   // bit = 0: self + partner
        v[2] = a0 - a2;   // bit = 1: -self + partner
        v[1] = a1 + a3;
        v[3] = a1 - a3;
    }
    // d = 32: partner register j^1, sign bit = j & 1
    {
        float a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
        v[0] = a0 + a1;   // bit = 0: self + partner
        v[1] = a0 - a1;   // bit = 1: -self + partner
        v[2] = a2 + a3;
        v[3] = a2 - a3;
    }
    // cross-lane stages: d = 16, 8, 4, 2, 1 (bit d of i is the lane bit)
    for (int s = 4; s >= 0; s--)
    {
        int d = 1 << s;
        for (int j = 0; j < 4; j++)
        {
            float w = __shfl_xor_sync(0xffffffffu, v[j], d);
            v[j] = (lane & d) ? (w - v[j]) : (v[j] + w);
        }
    }
}

}  // namespace exl3
