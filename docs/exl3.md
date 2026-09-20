# Native EXL3 quantization support in SGLang

EXL3 is turboderp's QTIP-derived trellis quantization format (16×16-block
bitstreams, per-matrix integer bitrates b 1..8, three procedural codebooks).
This fork adds **native** EXL3 support to SGLang: models quantized with
exllamav3 load and run here with **zero third-party runtime dependencies** —
no exllamav3 import anywhere under `sglang/` or `sgl-kernel/`; the bitstream
decode is implemented from scratch (with attribution) in the fork's
`sgl-kernel` package.

## Usage

```bash
python -m sglang.launch_server \
  --model-path /path/to/model-EXL3-bpw \
  --quantization exl3
```

`--quantization exl3` is picked up automatically from the checkpoint's
`quantization_config.json` as well. Models quantized at any bitrate
(b=1..8) and codebook (3inst / mcg / mul1) are supported on
NVIDIA Blackwell (sm_120) GPUs; `min_capability` is 120.

Feature gates (env):

| Env var | Effect |
|---|---|
| `EXL3_NO_KERNEL=1` | Force the pure-PyTorch dequant fallback (correct, slow; no `sgl_kernel` fast path). Useful for debugging. |

With the fast path, weights stay packed in the EXL3 trellis format at rest
(`.trellis` int16 + `.suh`/`.svh` fp16 + scalar sentinels) and are decoded
on the fly by the trellis GEMM kernel; nothing is materialized.

## Dataflow

Per matrix: `W_orig = diag(suh) @ H128 @ W_hat @ H128 @ diag(svh)`.
Forward: `x·suh` → Hadamard (128-block) → trellis GEMM (fp32 accum) →
Hadamard → `·svh` + bias. `lm_head` dequantizes at load (chunked);
`in_proj_ba` / visual towers run through the unquantized path as a fallback.

## Provenance and licenses

- Trellis decode math (windows, funnel-shift, procedural codebooks) is
  derived from turboderp's **ExLlamaV3** (MIT, (c) 2025 turboderp) —
  attribution and the full MIT text in
  `python/sglang/kernels/aot/csrc/exl3/exl3_decode.cuh`. The on-disk format
  must match ExLlamaV3's EXL3 bit-exactly.
- The Hadamard butterfly structure follows ExLlamaV3's natural-order
  Sylvester butterfly (`hadamard_inner.cuh`, MIT) — see
  `csrc/exl3/exl3_had.cuh`.
- The M-tiling / fragment-decode batched GEMM shape follows
  **cuda-exl3** (MIT, (c) 2026 cuda-exl3 contributors) design lessons — see
  `python/sglang/kernels/aot/csrc/exl3/exl3_linear.cu`.
- QTIP needs no notice (the trellis/numeric format credit goes to
  turboderp's ExLlamaV3).
- Independent for this fork: the warp/butterfly helpers, host wrappers,
  quantization method wiring (`python/sglang/srt/layers/quantization/exl3.py`),
  and the fused epilogue.

## Performance (RTX 5090, Qwen3.8-27B-EXL3-3.5bpw, dense)

| Metric | Value |
|---|---|
| Single-stream greedy decode | 3.3 tok/s (eager; 7.6× over the initial kernel) |
| 8-way shared-prefix concurrency | 25.5 tok/s aggregate (radix cache reuse) |
| 2000-token prefill TTFT | ~310 ms (eager) |
| Baseline: exllamav3 1.4.2 single-stream | 40.3 tok/s (same checkpoint) |

Correctness gates: kernel unit tests match the bit-exact PyTorch reference
(derived from exllamav3 1.4.2 outputs) with **zero ULP violations at every
shape and bitrate b=3/4/5/6**; greedy outputs match the exllamav3 oracle on
16/20 prompts at the exact-32-token-prefix bar (all divergences are coherent
phrase-level near-ties inherent to 3.5-bit greedy sampling).

## Building the kernel

The fork's aot package (scikit-build-core + CMake) builds the trellis GEMM
into `sgl-kernel`'s per-architecture `.so`s:

```bash
cd python/sglang/kernels/aot
uv pip install --no-build-isolation --no-deps -e . \
  -Cbuild-dir=<persistent build dir> \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1
# then copy build/skbuild/{sm100,sm90}/common_ops.abi3.so into the editable
# package's python/sgl_kernel/{sm100,sm90}/ (editable installs don't stage
# compiled artifacts).
```

`SGL_KERNEL_COMPILE_THREADS=1` is mandatory on low-memory hosts (the default
spawns 32 nvcc front-ends per file and can exhaust system RAM). With a
persistent `-Cbuild-dir`, rebuilds are ninja-incremental (~2-3 min).

## Known follow-ups

- Single-stream decode is still decode-heavy: attention, lm_head and the
  per-token scheduler overhead dominate the remaining gap to exllamav3.
- CUDA-graph capture of the default breakable prefill backend is slow
  (~141 s/shape × 50 shapes) because the kernel re-decodes the trellis per
  16-row fragment block; larger-row tiling is the follow-up.
- MoE routed-expert fast path: not implemented (the primary checkpoint is
  dense); revisit if a MoE EXL3 checkpoint is scoped.
