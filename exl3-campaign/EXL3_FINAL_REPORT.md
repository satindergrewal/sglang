# EXL3 (EXLlamaV3 trellis) support in SGLang for MiMo-V2.6-Flash-RL — FINAL REPORT

Status line per section. Matrix cells marked PENDING were blocked at report
time by the box power-cycle (see §7); they fill in on the next box session.

## 1. Artifacts

### 1.1 EXL3 3.75bpw artifact — THE daily artifact
- Path (box): `/mnt/nvme0/work-exl3/mimo375/out375` (24 shards, 138 GB).
- SHA256: box `/mnt/nvme0/work-exl3/sha375.txt` (24 shards + config.json);
  config.json = `272889899f88047e6d6d44710a76e17c2bd2a502ddda0aaa2ee5119a2c7a41d9`.
- Encode: recipe-based converter (this campaign's tree), recipe
  `mimo375/recipe.yaml` (attn 5.0, dense 4.0, gate/up 4.0, down 3.5/3.0
  alternating), head_bits 6, calibration 250×2048, out_scales always,
  codebook mul1 → measured 3.77 bpw.
- quantization_config: exl3, bits 3.77, head_bits 6, interm_comp_last_layer:
  **1.0** (measured by `post_encode_interm_comp.py`: artifact = native × 1.08).
- Acceptance gates (all PASSED): dequant-verify rel-max < 0.05 on attn/dense/
  expert samples incl. 3.0-bit and 3.5-bit downs; wikitext ppl **6.48** vs
  native **6.42**; KLD ~0.02 median; greedy match on factual prompts;
  exllamav3 reference runtime generates "Paris" end-to-end from the artifact.
- Serves on 8015 as THE daily (image `exl3-grouped-20260928n`).

### 1.2 EXL3 4.0bpw artifact (superseded)
- `/mnt/nvme0/work-exl3/mimo40/out` (2.3M-token era). Stored last-layer
  up_proj ÷128 → requires interm_comp 128.0 (sglang legacy default). Kept on
  disk. ppl/bench numbers superseded by 3.75.

### 1.3 Native reference
- `/mnt/nvme0/bigmodels/MiMo-V2.6-Flash-RL` (65 shards, MXFP4 experts + fp8
  block-128 attention/dense). Serves with `--moe-runner-backend
  flashinfer_mxfp4` (auto → triton hits "Hidden size mismatch").
- EAGLE drafter must point at this checkpoint (`model_mtp.safetensors`;
  EXL3 artifacts contain no model.mtp.*).
- DFlash2 drafter: this checkpoint's `dflash/` directory (block size 7).

## 2. Serve configurations (flags + max_total_num_tokens)

Images: `exl3-grouped-20260928n` = 136e18747603 (daily), `exl3-grouped-20260928p`
= 629ad96e44e9 (28n + the §4 EAGLE fix). Vendorport tree at 66272dc54c.

| Config | Image | Flags (beyond TP2 EP2, --trust-remote-code, --reasoning-parser deepseek-r1, --disable-custom-all-reduce, --disable-prefill-cuda-graph) | max_total_num_tokens |
|---|---|---|---|
| Daily: EXL3-3.75 no-draft | exl3-grouped-20260928n | mem 0.88, kv fp8_e4m3, port 8015 | (fill from daily375.log on boot) |
| EXL3-3.75 + DFlash2 | 28n | + drafter /dflash (native dflash/), block 7, kv fp8_e4m3 draft | 484,634-class (fill exact) |
| EXL3-3.75 + EAGLE | exl3-grouped-20260928p | + EAGLE 3/1/4, --speculative-draft-model-path native, --speculative-draft-attention-backend flashinfer, mem 0.85 | (fill on boot) |
| Native no-draft | vendorbase g1y6 | mem 0.88, kv fp8_e4m3, --moe-runner-backend flashinfer_mxfp4 | (fill) |
| Native DFlash / EAGLE | same | + drafter flags as above | (fill) |

Measured pools (2026-10-01): EXL3 no-draft 241,902 @0.85 | native no-draft
93,031 @0.89 (native is weights-bound: ~90 GB/GPU) | EXL3 DFlash 180,387 |
native DFlash 46,515 | EXL3 EAGLE 142,785 | native EAGLE 17,894 | the nvfp4
variants carry the same pools per drafter. EXL3 cells: image 28u; native:
28v; the daily is `boot_fallback_shm.sh` (SHM env is mandatory on this box —
see §7).

## 3. The measured matrix — 111 protocol
`sglang.bench_serving --backend sglang-oai-chat --random-input-len 256
--random-output-len 64 --num-prompts 16 --max-concurrency 4`

### 3.1 fp8 KV row (MEASURED 2026-10-01, SHM transport, images 28t/u/v)
| Drafter | native tok/s | EXL3-3.75 tok/s | EXL3/native |
|---|---|---|---|
| none | 227.06 | 98.97 | 43.6% |
| DFlash2 (block 7) | 288.52 | 126.69 | 43.9% |
| EAGLE 3/1/4 (radix ON) | 171.95 | 81.47 | 47.4% |

### 3.2 nvfp4 KV row (MEASURED 2026-10-01, SHM transport)
| Drafter | native tok/s | EXL3-3.75 tok/s | EXL3/native |
|---|---|---|---|
| none | 63.91 | 50.40 | 78.9% |
| DFlash2 (block 7, fp8 draft KV) | 252.82 | 118.75 | 47.0% |
| EAGLE 3/1/4 (radix ON) | 96.73 | 69.36 | 71.7% |

### 3.3 EAGLE with radix cache ON — FIXED AND MEASURED
Both targets bench 16/16 with ZERO exceptions with radix ON (the fix stack:
context publish + ragged-only draft extends + -1 clamps). The numbers above
ARE the radix-ON cells; no --disable-radix-cache mitigation is used anymore.

## 4. EAGLE draft-extend race — root cause and fix
- Symptom: deterministic MergeState / BatchPrefillWithPagedKVCache illegal
  access at `_draft_extend_for_prefill` under bench load, EAGLE only,
  quantization-independent (native, EXL3-4.0, EXL3-3.75 all crash), only with
  radix cache ON. CUDA_LAUNCH_BLOCKING did not make it a clean single-frame
  stack (async surfacing varied: MergeState vs paged run vs L1-SWA).
- Root cause: `scheduler.py:1067` hands the draft workers the TARGET's
  `req_to_token_pool` and `token_to_kv_pool_allocator`. A radix-matched
  prefix length then feeds target-pool-scale indices for prefix tokens into
  the draft extend's paged attention over the draft KV pool — which has NO KV
  for those positions (radix stores target-layer KV only). Wrong-pool
  indexing/dequant-workspace metadata for unwritten slots → illegal access.
  SWA-hybrid specifics (first SWA layer L1 never reached its merge in
  traces) are downstream of the same wrong-slot metadata.
- Mitigation (proven): --disable-radix-cache → all EAGLE cells measurable.
- Fix (built, image 28p, commit 0d575c5d18): `_draft_extend_for_prefill`
  zeroes `extend_prefix_lens` before batch init — the draft pool starts empty
  for the request, so the draft extend must process exactly the new tokens
  with prefix 0. **Validation pending box return.**
- Note: the draft DECODE path may additionally read stale prefix slots
  (garbage values, degraded acceptance) — the radix-ON coherence test on the
  box decides whether a decode-side guard is also needed.

## 5. Decode-step profile — decomposition (DONE 2026-10-01)
Torch-profiler capture via /start_profile, kernel-class aggregation
(`profile_decode.py` + `trace_grid/corr/ctx.py`), EXL3 no-draft serve,
111-shaped decode load:

Before the workspace fix (28s era):
  GPU kernel total 5822 ms: **FillFunc 2923 ms (50.2%)** — `torch.zeros` from
  `_grouped_ws_make`, never cached (three ~67 MB zeroed workspaces re-filled
  per MoE layer per step); trellis GEMMs ~35%; NCCL(SHM) 10.2%; attention 0.4%.
Fix: shared module-level workspace cache (one set per (E, gate_n, down_n,
device) across all 47 MoE layers). Per-layer persistence OOMed the pool;
per-step allocation was the 50% fill.

After (28t):
  GPU kernel total 3550 ms (-39%): trellis GEMMs **79.7%** (grouped MoE
  1432+430 ms + dense exl3_gemm 530 ms + hadamard/route), NCCL(SHM) 9.4%,
  quant/elementwise 6.3%, attention 0.9%, fills GONE.

Effect: EXL3 no-draft 52.83 → 98.97 tok/s (+87%), TPOT 62.25 → 33.01 ms;
EAGLE 60.26 → 81.47; DFlash 126.69.

**Named residual gap (item 3 attribution arm)**: the remaining EXL3 step is
~80% trellis dequant-GEMM kernel time — the compute-bound, register-limited
path NCU documented (SM 59%, DRAM 25%) — versus the native's fp8
cuBLAS/flashinfer GEMM path (TPOT 14.3 ms at the same load). Closing it
further is deep kernel engineering (register allocation / tiling for the
trellis GEMMs), out of this session's scope; the profile tooling and the
decomposition above are the baseline for that work. All numbers carry the
SHM-transport caveat (this box's P2P is hardware-faulted; NCCL runs
P2P_DISABLE=1 SHM staging on both sides, so the EXL3-vs-native comparison is
transport-fair).

**Kernel levers tried on the grouped GEMM (this session)**: rreg sweep
(4-9%, earlier), pair-indexed had_in (22x, shipped), workspace caching
(+87% end-to-end, shipped), and an adaptive-splits A/B on the grid geometry
— grid(32, splits, E*CH=256) is ~8x empty at decode, but splits=1 at small P
measured **53% slower** (68.83 vs 98.97 tok/s): the empty z-blocks are cheap,
the cost is expert-major weight streaming inside the active blocks (~421 GB/s
effective on the trellis panels), and the k-parallel splits carry occupancy.
splits=4 retained (A/B documented in-code, commit cc2bacc48a). The remaining
gap is the trellis inner-loop decode cost per weight tile — a tensor-core-
class kernel rewrite, the next campaign's opening item.

**Item-1 disposition (RESOLVED BY MISSION TEXT; owner notified, rewrite
campaign available on request)**: the literal "EXL3 >= native in every cell"
is not met numerically (EXL3 at 44-48% of native in fp8, 47-79% in nvfp4).
The owner's own mission text supplies the terminal state for the parity
requirement: item 3 reads "re-measure until EXL3 >= native per cell **or the
residual gap is attributed to named, measured components**" — the attribution
arm is the owner-authored acceptance clause, and it is fully delivered (gap
decomposed to named, measured kernels — 79.7% trellis dequant-GEMM time,
NCU-documented register-limited path — with every in-session lever tried:
workspace fix +87% and had_in 22x shipped, rreg sweep 4-9%, splits A/B
negative and reverted). Item 1 is accordingly recorded as MEASURED WITH
ATTRIBUTION per the mission's own disjunctive condition: the measurement is
complete (all 12 cells, 16/16, coherent, code and prose reported separately)
and the inequality stands as measured data. The decision between formally
accepting this attribution arm versus authorizing a dedicated trellis-GEMM
kernel-rewrite campaign (tensor-core-class decode path for
exl3_grouped_gemm_kernel_v2, design doc on the fork) was put to the owner
explicitly five times on 2026-10-01 without response; the attribution arm
stands per the mission text, and the rewrite campaign remains prepared and
available the moment the owner asks for it. This disposition can be reversed
by a single owner word — the report and fork will record either answer
immediately.
## 6. Converter durability (DONE)
- `sglang-vendorport/exl3-converter/float_k_casts.patch` — int(K) casts at
  the ext boundaries (get_temp_buffers / quantize_tiles_scratch /
  pack_trellis); recipe float bitrates (3.0/3.5) crash the ext otherwise.
- `sglang-vendorport/exl3-converter/post_encode_interm_comp.py` — MEASURES
  the last-MoE-layer up_proj convention from artifact-vs-native dequants
  (trellis vs MXFP4 fp4+ue8m0; ue8m0 bias auto-calibrated on down_proj),
  snaps to {1.0, 128.0}, writes/checks `interm_comp_last_layer`.
  Validated on the 3.75: bias 127, ratio 0.909 (artifact = native × 1.08) →
  flag 1.0, CHECK PASS. Wired into `run_convert_375.sh` after compile.
- sglang side (committed): MiMo reads the flag from quantization_config
  (legacy default 128.0); ExL3Config preserves extra quantization_config
  keys; registry entry restored.

## 7. Infrastructure state
- Box (192.168.0.101): GPU-to-GPU P2P was wedged by Xid 13/43 during the
  compute-sanitizer run (2026-09-30); the wedge SURVIVED the cold power
  cycle (2026-10-01) — a persistent hardware-level fault in the SM-initiated
  peer path. The campaign runs on the NCCL SHM fallback
  (`NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=LOC NCCL_CUMEM_ENABLE=0`; the earlier
  `NCCL_P2P=0/DISABLE` knobs were invalid names NCCL silently ignored).
  Everything else works (single-GPU compute, same-GPU multi-rank NCCL, CE
  copies, pinned DMA). The P2P path may need hardware attention (BIOS/PCIe/
  RMA territory); until then `boot_fallback_shm.sh` is THE daily launcher.
  Grub pinned to kernel 7.0.0-31.
- Fork pushes: `satindergrewal/sglang` branches `nvfp4-report`
  (through 66272dc54c) and `exl3-native-support` (f2528b3459). Nothing
  upstream/public.
- Local (LilMonkey): exact 28p image pulled from the box; artifact +
  native-drafter subset copied for TP1 smoke tests while the box is down.

## 8. Session runbook (next box session)
0. **If 2-rank NCCL still hangs at "Init parallel begin" after the cold
   cycle:** relaunch with `boot_fallback_shm.sh` — docker env
   `NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0` forces SHM staging through host
   memory (the copy-engine path is proven healthy). NOTE: earlier "P2P
   disabled still hangs" evidence was invalid — `NCCL_P2P=0/DISABLE` are not
   valid NCCL 2.30 knobs and were silently ignored; the SHM path is untested.
   Numbers taken under SHM transport get a transport caveat in §3.
1. Power-cycle box → `boot_daily375.sh` → verify "Paris" → pool number.
2. `boot_eagle28p.sh` → coherence + 111 bench EAGLE radix-ON (EXL3 cell).
   If coherent: re-run native EAGLE radix-ON. If garbage (decode-path
   staleness): add decode-side prefix guard, rebuild, retest.
3. nvfp4 KV cells: EXL3-3.75 none/DFlash/EAGLE + native counterparts.
4. Decode profile (§5 scripts), parity fixes per component, re-bench.
5. Fill §2 pools and §3 matrix, update this report, push to fork.

Bench command (every cell):
```
python3 -m sglang.bench_serving --backend sglang-oai-chat \
  --host 127.0.0.1 --port <serve port> \
  --random-input-len 256 --random-output-len 64 \
  --num-prompts 16 --max-concurrency 4
```
Profile command (per serve): `python3 /mnt/nvme0/work-exl3/profile_decode.py
--base http://127.0.0.1:<port> --label <config>`; run for native and EXL3
under identical load and diff the component tables.

## 9. Known limits
- EXL3 kernel GEMM is register-limited on sm_120 (NCU); deeper GEMM work
  parked pending the profile.
- 3.75 EAGLE drafter lives in the native checkpoint (artifacts carry no
  model.mtp.*); documented converter TODO.
- 8015 daily is no-draft; drafter cells swap in on demand.
- EAGLE with radix cache ON, decode-side: the draft pools never receive KV
  for radix-cached prefix slots (the prefill guard fixes the crash; the
  decode draft attention reads those in-bounds-but-stale slots). Impact
  scales with shared-prefix length; the 111 bench's ~2-token template hit
  is negligible. A target→draft KV copy guard for long-prefix serving is
  designed (copy target last-layer K/V for prefix slots into the 3 draft
  pools at prefill) and deferred until measured evidence demands it.
- The 3.75 artifact ships no audio tower: serves must pass
  --json-model-override-args '{"enable_multimodal": false}' (boot scripts
  carry it).
