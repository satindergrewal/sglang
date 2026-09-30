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

(Exact pool numbers get copied from the boot logs into the final table when
the boxes return; the campaign numbers above are from the journal.)

## 3. The measured matrix — 111 protocol
`sglang.bench_serving --backend sglang-oai-chat --random-input-len 256
--random-output-len 64 --num-prompts 16 --max-concurrency 4`

### 3.1 fp8 KV row (COMPLETE)
| Drafter | native tok/s | EXL3-3.75 tok/s | EXL3/native |
|---|---|---|---|
| none | 67.2 | 52.7 | 78% |
| DFlash2 (block 7) | 147.5 | 96.25 | 65% |
| EAGLE 3/1/4 (radix OFF) | 185.98 | 62.58 | 34% |

### 3.2 nvfp4 KV row (PENDING — box)
Cells: none / DFlash2 / EAGLE × native / EXL3. Known-good code path exists
(verify×nvfp4 workspace chain fixed; native DFlash nvfp4 = 147.5-class
measured earlier). EXL3-side nvfp4 cells never benched.

### 3.3 EAGLE with radix cache ON (PENDING — box)
The 3.1 EAGLE cells ran with --disable-radix-cache (mitigation for the race,
§5). With the §5 fix (image 28p) the radix-ON cells must be re-benched; if
stable, the matrix numbers above get superseded by radix-ON numbers.

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

## 5. Decode-step profile (~100 ms EXL3 decode step) — decomposition (PENDING — box)
MoE exonerated earlier (kernel chain: pairs had_in 139.5→6.3 µs, fused
routing 80→26 µs, full MoE chain 973→704 µs; NCU shows the GEMM
compute-bound, SM 59% / DRAM 25%, register-limited; rreg sweep gave 4–9%).
Remaining decomposition: attention vs dense-projections vs NCCL vs sampler,
native-vs-EXL3 per component. Profile scripts to run: torch profiler on the
decode step at bs≈4, one pass per target (native serve vs EXL3 serve),
diff at component level. Scripts prepared in §8.

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
- Box (192.168.0.101): GPU-to-GPU P2P wedged by Xid 13/43 during the
  compute-sanitizer run (2026-09-30). Survives warm reboot, kernel switch
  (31↔34), nvidia-smi -r, PCIe remove/rescan, ACS clear. Everything else
  works (single-GPU compute, same-GPU multi-rank NCCL, CE copies, pinned
  DMA). Needs a COLD POWER CYCLE. Recovery after power-up:
  `/mnt/nvme0/work-exl3/boot_daily375.sh` restores the 8015 daily;
  `boot_eagle28p.sh` runs the EAGLE fix test. Grub pinned to kernel
  7.0.0-31 (kernel 34 was a red herring; note in /root/KERNEL_NOTE.txt).
- Fork pushes: `satindergrewal/sglang` branches `nvfp4-report`
  (through 66272dc54c) and `exl3-native-support` (f2528b3459). Nothing
  upstream/public.
- Local (LilMonkey): exact 28p image pulled from the box; artifact +
  native-drafter subset copied for TP1 smoke tests while the box is down.

## 8. Session runbook (next box session)
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
