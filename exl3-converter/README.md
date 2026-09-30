# Converter durability patches (MiMo-V2.6 EXL3 encode pipeline)

Two regressions that silently produce broken/garbage artifacts, both now
captured durably. Apply these to any fresh exllamav3 converter checkout before
encoding MiMo-V2.6-Flash targets, and always run the post-encode fixup.

## 1. float-K casts (`float_k_casts.patch`)

Recipe bitrates arrive as floats (3.0/3.5/3.77); the C++ ext entry points and
bit math want the integer KA. Upstream `quantize.py` passes `K` through
un-cast, so half-rate recipes crash inside `get_temp_buffers` /
`ext.quantize_tiles_scratch` / `pack_trellis`.

    cd <converter>/exllamav3 && git apply /path/to/float_k_casts.patch

(Paths in the patch are relative to the two trees used during the campaign:
`exl3-upstream-dev/exllamav3` pristine vs `exl3-mimo-conv/exllamav3` patched;
apply with `patch -p1` style matching or by hand — it is a 5-line diff.)

## 2. interm_comp_last_layer measurement (`post_encode_interm_comp.py`)

sglang's MiMo loader reads `quantization_config.interm_comp_last_layer`
(legacy default **128.0** when absent = "last MoE layer up_proj stored ÷128,
exllamav3 interm_div convention"). The recipe converter does NOT divide, so an
artifact that omits the flag silently explodes the last MoE layer ×128 →
garbage generations (this exact failure cost a full debug cycle on the 3.75
encode).

The script MEASURES the convention instead of assuming it: dequantizes the last
MoE expert up_proj from the artifact (trellis) and the native checkpoint
(MXFP4 fp4 + ue8m0), auto-calibrates the ue8m0 bias against a tensor the
convention never touches (down_proj), takes the median-abs ratio, snaps to
{1.0, 128.0}, and writes/checks the flag.

    # end of every encode, after compile:
    python3 post_encode_interm_comp.py --artifact <out_dir> --native <native_ckpt> --write
    # CI / gate mode (exit 1 on mismatch):
    python3 post_encode_interm_comp.py --artifact <out_dir> --native <native_ckpt> --check-only

Validated on the 3.75 artifact: bias calibrates to 127, measured ratio 0.909
(artifact = native × 1.08), flag 1.0 == declared 1.0 → CHECK PASS.

## Pipeline wiring

`run_convert.sh` / `run_convert_375.sh` run the fixup right after convert.py
(see the `post_encode_interm_comp.py --write` line). If you re-encode with a
different converter version, re-run the fixup — never hand-edit the flag.
