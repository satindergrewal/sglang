#!/usr/bin/env python3
"""Reorder the MoE expert trellis panels of an EXL3 artifact from the
(kk, col_block, tile)-major converter layout to (col_block, kk, tile)-major
(fully contiguous per-block k-scans). Pure byte permutation — the trellis
codes, suh, svh, and all non-panel tensors are copied verbatim; the dequant
math is unchanged. Enables the layout-B grouped kernel (1.55-1.62x measured
on the microbench).

Usage: reorder_panels.py <src_artifact> <dst_artifact>
Run inside a container with torch + safetensors.
"""
import json, os, sys, shutil
import torch
from safetensors.torch import load_file, save_file

SRC = sys.argv[1]; DST = sys.argv[2]
os.makedirs(DST, exist_ok=True)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
shutil.copy2(f"{SRC}/model.safetensors.index.json", f"{DST}/model.safetensors.index.json")
wm = idx["weight_map"]
shards = sorted(set(wm.values()))
print(f"{len(shards)} shards")

for shard in shards:
    tensors = load_file(f"{SRC}/{shard}", device="cpu")
    out = {}
    n_panels = 0
    for name, t in tensors.items():
        if name.endswith(".trellis"):
            # ALL trellis panels: MoE experts AND attention/dense projections.
            # The B-layout addressing in the kernels matches this order.
            k16, n16, wb = t.shape
            if n16 % 8 != 0:
                out[name] = t
                continue
            # (kk, col_block, tile, wb) -> (col_block, kk, tile, wb)
            t2 = t.view(k16, n16 // 8, 8, wb).permute(1, 0, 2, 3).contiguous()
            out[name] = t2.view(k16, n16, wb)
            n_panels += 1
        else:
            out[name] = t
    save_file(out, f"{DST}/{shard}")
    print(f"{shard}: {n_panels} panels reordered", flush=True)

for f in os.listdir(SRC):
    src_f = os.path.join(SRC, f)
    if os.path.isfile(src_f) and not f.endswith(".safetensors"):
        shutil.copy2(src_f, os.path.join(DST, f))
print("REORDER DONE")
