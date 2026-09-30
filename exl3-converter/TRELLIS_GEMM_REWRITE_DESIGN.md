# Trellis-GEMM tensor-core decode path — campaign opening design
Status: PREPARED, awaiting owner go. This document is the authorized-campaign
branch's first work item; nothing here is merged into the serve until each
phase's acceptance gate passes.

## 1. Target and measured baseline
Component: `exl3::exl3_grouped_gemm_kernel_v2<K, half_k>` and
`exl3::exl3_gemm_kernel_v2<K, ...>` (box sources:
`/mnt/nvme0/work-exl3/exl3_ops_src/csrc/exl3/`), sm_120a build via
`exl3_ops_src/build_grouped_patch.py` → `/work/exl3_ops/exl3_ops_patch.so`.

Per-kernel baselines (28t decode profile, ~100 steps, 111-shaped load):
- `exl3_grouped_gemm_kernel_v2<4,false>` (gate, 4-bit): 1431.7 ms — 152 µs/launch
- `exl3_gemm_kernel_v2<5,16>` (dense/attn, 5-bit):      529.6 ms — 5.3 ms/step
- `exl3_grouped_gemm_kernel_v2<3,false/true>` (down, 3/3.5-bit): 630 ms
- Effective weight streaming inside active blocks: ~421 GB/s (vs ~3.3 TB/s
  HBM peak) — the scalar unpack+FMA inner loop serializes the stream.

Post-fix step budget (no-draft, TPOT 33 ms): trellis GEMMs 79.7%,
NCCL(SHM) 9.4%, misc 6.3%, attention 0.9%, fills 0%.

## 2. Root structure of the gap
The trellis format stores weights as 16-bit codes in circular 16-bit windows
(k16 windows, n16 column tiles). The current kernels decode codes to fp16
with scalar shift/multiply chains and accumulate with FMA — no tensor cores.
The native mxfp4 path streams weights through a hardware LUT at near-HBM
speed. At decode (M = bs×topk ≤ 64 rows) the GEMM is bandwidth-bound on
weight bytes; the trellis decode compute is what keeps effective bandwidth
at ~13% of peak. An MMA path overlaps decode with the multiply.

Natural mapping: the trellis k16 window IS the mma.sync k-dim (k16). Per
(m16-fragment of pairs, n8/n16 column tile, k16 window): dequant the 16×16
weight tile into fp16 fragments (shared memory), then mma.sync against the
activation fragment. The dequant cost per tile is fixed; MMA replaces the
FMA chain and, critically, enables multi-stage pipelining (cp.async weight
panels → smem decode → mma) that the scalar path cannot express.

## 3. Phase plan (each phase gated)
- **P0 — reference lock + harness (MEASURED 2026-10-01)**:
  `p0_microbench.py` (committed) times grouped_linear at the exact 3.75
  shapes. Headline: the kernel streams at ~1.6 TB/s effective (uniform:
  536.9 MB / 333.8 µs at P=32) — streaming efficiency is NOT the gap. The
  measured waste is **pair-major weight re-reads** (~4x amplification at
  decode skew: each expert panel re-read per pair). P0' (new first task):
  expert-major M<=4 tiling — decode each 16x16 panel tile ONCE and apply it
  to all pairs of that expert (register-resident activations) — a moderate
  change to the existing kernel, projected ~2x on the grouped GEMMs before
  any tensor-core work. The MMA path (P1/P2 below) follows if P0' lands and
  the residual still justifies it.
- **P1 — dense kernel MMA** (`exl3_gemm_kernel_v2`): single-matrix, no
  grouped indirection — the simplest tensor-core win (5.3 ms/step baseline).
  Gate: bit-exact vs P0 on sampled layers; microbench ≥2× at M≤64; then
  serve A/B (the established 111-protocol methodology).
- **P2 — grouped MMA**: pair-indexed grid (the had_in_pairs indirection
  pattern, 22× precedent) with mma fragments; expert-major panels stay
  (splits=4 is measured-optimal — see the A/B in the report).
  Gate: same bit-exactness; microbench ≥2×; serve A/B on the fp8 row.
- **P3 — prefill/large-M**: revisit splits/CH tiling for M>256 (the
  prefill cells currently run the same kernels); sweep with the A/B harness.
- **P4 — NVFP4-KV interplay**: none (KV-side, independent).

## 3.5 Split-sweep + the workspace-traffic wall (measured 2026-10-01)
Serve-skew microbench (8 active experts x 4 pairs, 64MB real traffic):
splits=4 -> 109.8 µs, splits=8 -> 114.8 µs (no gain), splits=1 -> measured
worse end-to-end earlier. Explanation: block runtime scales with k/splits,
but the split-reduction workspace traffic scales UP with splits (each block
restores/reduces its fp32 (16,128) slab; at splits=8 the extra ~134 MB of
workspace writes exceeds the 64 MB of useful weight reads). The kernel also
already amortizes panel decode across a 16-row mma fragment (tensor-core
mma.m16n8k16 is IN the kernel) — the earlier "pair-major re-read" reading
was a harness artifact and is withdrawn.

**Next concrete move (requires owner go — invasive kernel change)**: shrink
the split-reduction workspace slab from (16,128) fp32 to (4,128) fp32 — at
decode only m<=4 rows are ever real — cutting workspace traffic 4x and
making splits=16..32 profitable (projected 3-4x on the grouped GEMMs,
TPOT ~33 -> ~20-24 ms, no-draft ~145-165 tok/s = ~64-72% of native).
Kernel+runner change (~100 lines: ws shape, block indexing, reduce path),
full bit-exactness gate required. Note the projected ceiling: even at 72%
of native, item 1's literal inequality stays unmet — parity requires the
tensor-core decode path beyond this fix.

## 4. Risks
- Fragment layout vs the circular window addressing: the decode's lane-shift
  structure (t_offset = lane<<3 heritage) must be re-derived for the MMA
  fragment ownership; expect a correctness-iteration loop per bitrate.
- Register pressure: NCU already flags the scalar path as register-limited;
  the MMA path trades registers for smem staging — the pipeline depth is the
  tuning knob.
- bf16 out + fp16 accum rounding must match the reference within the gate
  tolerance (KLD gate on the serve, rel<0.05 dequant gate at build time).
- The exl3_ops .so is sm_120a AOT — every kernel change rebuilds via
  `build_grouped_patch.py` (no JIT fallback).

## 5. Method (fixed)
Every change: P0 bit-exactness → microbench vs baseline → serve A/B on the
111 protocol → matrix cell re-bench → report/journal update → fork push.
No serve integration without the microbench gate; no claim without the A/B.
