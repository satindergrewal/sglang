#!/usr/bin/env python3
"""Post-encode fixup: measure and write quantization_config.interm_comp_last_layer.

Why this exists: sglang.srt.models.mimo_v2 reads interm_comp_last_layer from the
artifact quantization_config (legacy default 128.0 when absent). The value must
state the divisor F the encode applied to the LAST MoE layer up_proj relative to
the native checkpoint tensor:
    F = 128.0  if the encode stored up_proj / 128 (upstream exllamav3 interm_div
               convention; sglang compensates x128 at load)
    F = 1.0    if the encode stored up_proj at native scale (this campaign's
               recipe converter never divides)
A future encode that drops this flag falls back to the legacy 128.0 default and
the last MoE layer output explodes x128 -> garbage generations. Wire this script
into the encode pipeline (run_convert*.sh) so the flag is always written from a
MEASURED ratio, never assumed.

Method: dequantize the last MoE expert up_proj from both the artifact (trellis,
via sglang dequant_matrix_orig) and the native checkpoint (MXFP4 fp4+ue8m0 for
experts), take the median absolute value ratio, snap to {1.0, 128.0}, and write
the flag. The MXFP4 ue8m0 bias convention is auto-calibrated against the
artifact itself (trellis dequant is validated by gate_dequant375.py) on a
reference expert tensor the interm convention never touches (down_proj), so a
wrong scale convention cannot silently flip 1.0<->128.
"""
import argparse, json, sys
import torch
from safetensors import safe_open


def load_wm(root):
    try:
        return json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    except FileNotFoundError:
        return {}


class Reader:
    def __init__(self, root):
        self.root, self.wm = root, load_wm(root)

    def get(self, name):
        f = safe_open(f"{self.root}/{self.wm[name]}", framework="pt", device="cuda")
        return f.get_tensor(name)


def fp4_lut():
    lut = []
    for nib in range(16):
        s = -1.0 if nib & 8 else 1.0
        e, m = (nib >> 1) & 3, nib & 1
        v = m * 0.5 if e == 0 else (1 + m * 0.5) * (2.0 ** (e - 1))
        lut.append(s * v)
    return torch.tensor(lut, dtype=torch.float32, device="cuda")


BIAS = [127]  # ue8m0 bias; auto-calibrated


def dequant_mxfp4(w_u8, sc_u8):
    lut = fp4_lut()
    R, C = w_u8.shape
    lo, hi = (w_u8 & 0xF).long(), (w_u8 >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(R, C * 2)
    scale = torch.exp2(sc_u8.float() - BIAS[0])
    scale = scale.repeat_interleave(32, dim=1)  # one ue8m0 scale per 32 fp4 elems
    return vals * scale


def dequant_exl3(reader, prefix):
    from sglang.srt.layers.quantization.exl3 import (
        dequant_matrix_orig,
        _sentinel_to_codebook,
    )

    trellis = reader.get(prefix + "trellis")
    suh = reader.get(prefix + "suh")
    svh = reader.get(prefix + "svh")
    mul1 = reader.get(prefix + "mul1")
    cb = _sentinel_to_codebook(mul1)
    return dequant_matrix_orig(trellis, suh, svh, cb).float()


def dequant_native_mxfp4(reader, name):
    w = reader.get(name + ".weight")
    s = reader.get(name + ".weight_scale")
    return dequant_mxfp4(w, s)


def med_abs(t):
    return t.abs().median().item()


def align(n, q):
    # expert trellis packs dequant to (k, n); native fp4 unpacks to (out, in)
    if n.shape == q.shape:
        return n
    if n.T.shape == q.shape:
        return n.T.contiguous()
    raise RuntimeError(f"unmatchable shapes {tuple(n.shape)} vs {tuple(q.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--num-layers", type=int, default=48)
    ap.add_argument("--write", action="store_true", help="write flag into artifact config.json")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    art, nat = Reader(args.artifact), Reader(args.native)
    last = args.num_layers - 1
    up = f"model.layers.{last}.mlp.experts.0.up_proj"
    ref = f"model.layers.{last}.mlp.experts.0.down_proj"  # convention never touches down_proj

    # Calibrate the ue8m0 bias on the reference tensor: down_proj is never
    # divided, so its artifact/native ratio must be ~1. Scan plausible biases,
    # keep the one minimizing rel-error vs the validated trellis dequant.
    q_ref = dequant_exl3(art, ref + ".")
    best = (1e9, None)
    for b in range(118, 137):
        BIAS[0] = b
        try:
            n = dequant_native_mxfp4(nat, ref)
        except Exception:
            continue
        try:
            n = align(n, q_ref)
        except RuntimeError:
            continue
        rel = ((n - q_ref).abs().max() / q_ref.abs().max().clamp(min=1e-9)).item()
        if rel < best[0]:
            best = (rel, b)
    if best[1] is None:
        print("FAIL: could not calibrate MXFP4 scale convention")
        sys.exit(1)
    BIAS[0] = best[1]
    print(f"ue8m0 bias calibrated: {best[1]} (ref rel-max {best[0]:.4f})")

    q = dequant_exl3(art, up + ".")

    n_up = align(dequant_native_mxfp4(nat, up), q)
    ratio = med_abs(n_up) / max(med_abs(q), 1e-30)  # native / artifact
    print(f"last-layer up_proj median|native|/median|artifact| = {ratio:.4f}")

    if abs(ratio - 1.0) < 0.35:
        flag = 1.0  # artifact at native scale
    elif abs(ratio - 128.0) < 45.0:
        flag = 128.0  # artifact stored /128 vs native
    else:
        print(f"FAIL: ratio {ratio:.4f} matches neither convention")
        sys.exit(1)

    cfgp = f"{args.artifact}/config.json"
    cfg = json.load(open(cfgp))
    cur = cfg.get("quantization_config", {}).get("interm_comp_last_layer")
    print(f"measured flag = {flag}; artifact currently declares {cur}")
    if args.check_only:
        ok = cur is not None and abs(float(cur) - flag) < 1e-6
        print("CHECK", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if args.write:
        cfg.setdefault("quantization_config", {})["interm_comp_last_layer"] = flag
        json.dump(cfg, open(cfgp, "w"), indent=4)
        print(f"WROTE interm_comp_last_layer: {flag}")


if __name__ == "__main__":
    main()
