# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM's quantization config/method pattern (Apache-2.0).
#
# EXL3 quantization method — support for turboderp's EXL3 (ExLlamaV3) trellis
# packs: per-matrix integer bitrates 1..8, 16x16-block trellis bitstreams,
# per-matrix Hadamard scale vectors, and the procedural 3inst/mcg/mul1
# codebooks.
#
# THIRD-PARTY ATTRIBUTION:
# The EXL3 on-disk format, packed layout, procedural codebooks and quantization
# math are defined by turboderp's ExLlamaV3 (MIT):
#   Copyright (c) 2025 turboderp — MIT License
#   github.com/turboderp-org/exllamav3
# The bitstream decoder below was written in PyTorch from the exllamav3 MIT
# sources (exllamav3_ext/quant/exl3_dq.cuh, codebook.cuh, modules/quant/exl3.py,
# modules/quant/exl3_lib/quantize.py) and validated bit-exact against them
# (dense Bit-exact vs exllamav3 1.4.2 ext.reconstruct / get_weight_tensor on
# sm_120; see exl3-workspace regression suite). It carries the ExLlamaV3 MIT
# notice accordingly. This file has no exllamav3 imports and no new
# dependencies: the decoder is pure PyTorch.
#
# v1 scope (TP=1 / EP=1):
#   - dense: dequant-on-apply with the native CUDA trellis GEMM (fast path) or
#     the pure-torch fallback (EXL3_NO_KERNEL=1); Phase 3 tunes the kernel.
#   - MoE: per-expert trellis packs are stashed at load and dequantized once in
#     process_weights_after_loading into fused fp16 w13/w2 expert weights, run
#     by the standard Triton fused-MoE runner. Phase 3.5 replaces this with the
#     packed-trellis grouped GEMM (fp16 materialization is O(4.6x) the pack).
#   - mixed checkpoints (GLM r7 "non_routed_dtype_policy": "official_source_native"
#     — only routed experts are trellis-packed): modules whose tensors arrive as
#     plain native `.weight` fall back to an unquantized linear transparently.
#   - quantized lm_head dequantizes once at load so logits stay a plain GEMM;
#     native (unquantized) lm_heads load through the same plain path.
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.moe_runner.base import (
    DispatchMoeRunnerCore,
    MoeRunnerConfig,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

# ---------------------------------------------------------------------------
# EXL3 bitstream decoder (pure PyTorch; bit-exact vs exllamav3 1.4.2)
#
# Per logical matrix (input K, output N):
#   trellis int16 [K//16, N//16, 16*b]  b = per-matrix integer bitrate 1..8
#   suh fp16 [K], svh fp16 [N]          diagonal Hadamard-vector scales
#   mul1/mcg I32 scalar sentinel        PRESENCE selects the codebook
#   bias fp16 [N] optional
# W_hat = trellis_dequant(trellis, codebook) lives in the 128-order Hadamard
# basis; W_orig = diag(suh) @ Hs @ W_hat @ Hs @ diag(svh) per 128-block.
# ---------------------------------------------------------------------------

_MCG_SENTINEL = 0xCBAC1FED
_MUL1_SENTINEL = 0x83DCD12D
_MCG_MULT = 0xCBAC1FED
_MUL1_MULT = 0x83DCD12D
_DEF_MULT = 89226354
_DEF_ADD = 64248484
_HAD_N = 128
# lop3 majority constants per 16-bit half: (A|B)=0xB4BF, (A&B)=0x0B40 for A=0x8FFF, B=0x3B60
_MAJ_OR = 0xB4BF
_MAJ_AND = 0x0B40
_EXL3_FP16_CACHE: Dict[str, torch.Tensor] = {}


def _fp16_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """Integer bit patterns 0..0xFFFF -> fp16 bit-identical (pure torch).

    torch raises on int16 construction overflow, so wrap values >= 0x8000 into
    the signed range manually before the bit-identical view(reinterpret).
    """
    bits = bits.to(torch.int64)
    signed = torch.where(bits >= 0x8000, bits - 0x10000, bits)
    return signed.to(torch.int16).view(torch.float16)


def _fp64(f):
    return float(f)


def _sylvester_hadamard(n: int = _HAD_N) -> torch.Tensor:
    """Raw ±1 natural-order Sylvester Hadamard of order n (n a power of two)."""
    h = torch.ones((1, 1), dtype=torch.float32)
    for _ in range(n.bit_length() - 1):
        d = h.shape[0]
        s = torch.empty((d * 2, d * 2), dtype=torch.float32)
        s[:d, :d] = h
        s[:d, d:] = h
        s[d:, :d] = h
        s[d:, d:] = -h
        h = s
    return h


def _hadamard128(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """H128 * (1/sqrt(128)) — orthogonal, self-inverse, natural Sylvester order."""
    key = f"{str(device)}:fp32"
    h = _EXL3_FP16_CACHE.get(key)
    if h is None:
        h = _sylvester_hadamard(_HAD_N).to(device=device)
        h *= 1.0 / math.sqrt(_HAD_N)
        _EXL3_FP16_CACHE[key] = h
    return h


def _decode_codebook(w0: torch.Tensor, codebook: str) -> torch.Tensor:
    """uint32 window words (int64, values < 2**32) -> fp16 codebook values.

    Single-rounding semantics of the CUDA decoders ($exllamav3_ext
    quant/codebook.cuh): mul1 is __hfma (exact a*b+c, one RTNE round), emulated
    with fp64 a*b+c then a single RTNE round to fp16; default/mcg are __hadd
    (one RTNE round) of the two transformed fp16 halves.
    """
    if codebook not in ("mul1", "mcg", "default"):
        raise ValueError(f"unknown exl3 codebook: {codebook}")
    w = w0.to(torch.int64)
    if codebook == "mul1":
        m0 = _MUL1_MULT & 0xFFFF
        m1 = _MUL1_MULT >> 16
        x = ((w * m0) + (((((w & 0xFFFF) * m1) & 0xFFFF) << 16))) & 0xFFFFFFFF
        bsum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        k_inv = float(
            _fp16_from_bits(torch.tensor([0x1EEE], dtype=torch.int64)).item()
        )
        k_bias = float(
            _fp16_from_bits(torch.tensor([0xC931], dtype=torch.int64)).item()
        )
        v64 = bsum.to(torch.float64) + 1024.0
        return (v64 * k_inv + k_bias).to(torch.float16)
    if codebook == "mcg":
        m0 = _MCG_MULT & 0xFFFF
        m1 = _MCG_MULT >> 16
        x = ((w * m0) + (((((w & 0xFFFF) * m1) & 0xFFFF) << 16))) & 0xFFFFFFFF
    else:  # default ("3inst")
        x = (w * _DEF_MULT + _DEF_ADD) & 0xFFFFFFFF
    lo = (x & _MAJ_OR) | _MAJ_AND
    hi = ((x >> 16) & _MAJ_OR) | _MAJ_AND
    lo16 = _fp16_from_bits(lo)
    hi16 = _fp16_from_bits(hi)
    return (lo16.to(torch.float64) + hi16.to(torch.float64)).to(torch.float16)


def _sentinel_to_codebook(sentinel: torch.Tensor | None) -> str:
    """Scalar I32 marker tensor -> codebook name (presence semantics)."""
    if sentinel is None:
        return "default"
    u = int(sentinel.item()) & 0xFFFFFFFF
    if u == _MUL1_SENTINEL:
        return "mul1"
    if u == _MCG_SENTINEL:
        return "mcg"
    raise ValueError(f"unknown exl3 codebook sentinel: {u:#x}")


def _tile_windows(w32: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tile uint32 words ((..., 8*b)) -> 256 window words (uint32, int64).

    Exact port of exl3_dq.cuh dq(): element t's 16-bit codebook window spans
    bits [(t+1)*b + 256*b - 16, (t+1)*b + 256*b) of the circular tile.
    """
    t = torch.arange(256, dtype=torch.int64, device=w32.device)
    b0 = t * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    i0 = b0 // 32
    i1 = (b1 - 1) // 32
    s0 = (i1 + 1) * 32 - b1
    n_words = 8 * bits
    a_idx = (i0 % n_words).clamp(0, n_words - 1)
    b_idx = (i1 % n_words).clamp(0, n_words - 1)
    A = w32[..., a_idx]
    B = w32[..., b_idx]
    lo_mask = s0 <= 16  # window entirely inside one word
    shift = (32 - s0).clamp(0, 63)
    hi_branch = ((B >> s0) | (A << shift)) & 0xFFFF
    lo_branch = (B >> s0) & 0xFFFF
    return torch.where(lo_mask, lo_branch, hi_branch)


def _pairs_to_u32(w: torch.Tensor) -> torch.Tensor:
    """(..., 2) int64 low/high halves (LE) -> (...,) uint32-as-int64 words."""
    return (w[..., 0] & 0xFFFF) | ((w[..., 1] & 0xFFFF) << 16)


@torch.no_grad()
def trellis_dequant_f16(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """Packed trellis (K//16, N//16, 16*b) int16 -> (K, N) fp16 W_hat.

    W_hat lives in the 128-order Hadamard basis. Tile element order is the
    tensor-core fragment order (carry-through of exllamav3's quantize_tensor_
    core_perm); decoded lane values are re-ordered row-major through the
    inverse perm. Bit-exact versus exllamav3 1.4.2.
    """
    assert trellis.dtype == torch.int16, "exl3 trellis must be int16"
    assert trellis.ndim == 3, "exl3 trellis must have dim = 3"
    k16, n16, wb = trellis.shape
    assert wb % 16 == 0, "exl3 trellis third dim must be a multiple of 16"
    bits = wb // 16
    assert 1 <= bits <= 8, "exl3 bitrate out of range 1..8"
    pairs = trellis.view(k16, n16, 8 * bits, 2).to(torch.int64) & 0xFFFF
    w32 = (pairs[..., 0] & 0xFFFF) | ((pairs[..., 1] & 0xFFFF) << 16)
    windows = _tile_windows(w32, bits)  # (k16, n16, 256) uint32
    lanes = _decode_codebook(windows, codebook)  # lane order fp16
    perm = _fp16_cache_perm()
    rm = lanes[..., perm]
    rm = rm.view(k16, n16, 16, 16).permute(0, 2, 1, 3).reshape(k16 * 16, n16 * 16)
    return rm.contiguous()


def _fp16_cache_perm() -> torch.Tensor:
    perm = _EXL3_FP16_CACHE.get("perm_i")
    if perm is None:
        perm_a = [0] * 256
        for t in range(32):
            r0 = (t % 4) * 2
            c0 = t // 4
            entries = (
                (r0, c0),
                (r0 + 1, c0),
                (r0 + 8, c0),
                (r0 + 9, c0),
                (r0, c0 + 8),
                (r0 + 1, c0 + 8),
                (r0 + 8, c0 + 8),
                (r0 + 9, c0 + 8),
            )
            for j, (r, c) in enumerate(entries):
                perm_a[t * 8 + j] = r * 16 + c
        perm = torch.argsort(torch.tensor(perm_a, dtype=torch.int64))
        _EXL3_FP16_CACHE["perm_i"] = perm
    return perm


@torch.no_grad()
def dequant_matrix_orig(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                        codebook: str) -> torch.Tensor:
    """Full-matrix dequant: W_orig = diag(suh) @ Hs @ W_hat @ Hs @ diag(svh).

    Replicates exllamav3's get_weight_tensor chain op-for-op
    (preapply_had_l -> *suh -> preapply_had_r -> *svh): Hadamard matmuls in
    fp32 with the fp32-scaled H128, rounded back to fp16 between stages, fp16
    elementwise Hadamard-vector scales.
    """
    w = trellis_dequant_f16(trellis, codebook)
    k, n = w.shape
    assert k % _HAD_N == 0 and n % _HAD_N == 0, "dims must be multiples of 128"
    device = w.device
    hs_key = f"{str(device)}:had"
    hs = _EXL3_FP16_CACHE.get(hs_key)
    if hs is None:
        hs = _sylvester_hadamard(_HAD_N).to(device=device)
        hs *= 1.0 / math.sqrt(_HAD_N)
        _EXL3_FP16_CACHE[hs_key] = hs
    # preapply_had_l: fp32 batched matmul, round back to fp16
    w1 = torch.matmul(hs, w.to(torch.float32).view(-1, _HAD_N, n)).view(k, n).to(torch.float16)
    w2 = w1 * suh[:, None]  # fp16 elementwise (row scale)
    # preapply_had_r: fp32 batched matmul, round back to fp16
    w3 = torch.matmul(w2.to(torch.float32).view(k, -1, _HAD_N), hs).view(k, n).to(torch.float16)
    return (w3 * svh).contiguous()  # fp16 elementwise (col scale)


# ---------------------------------------------------------------------------
# SGLang quantization config / methods
# ---------------------------------------------------------------------------

_EXL3_PARAMS = ("trellis", "suh", "svh", "mul1", "mcg", "bias")
_SHARD_NAME_TO_INDEX = {"q": 0, "k": 1, "v": 2}
_PARAM_DTYPES = {
    "trellis": torch.int16,
    "suh": torch.float16,
    "svh": torch.float16,
    "mul1": torch.int32,
    "mcg": torch.int32,
    "bias": torch.float16,
}


def _shard_first_index(shard_id) -> int:
    """First merged-shard index covered by a weight-loader shard_id.

    Integer shard ids route one merged shard (gate_up 0/1, in_proj_z 3), named
    shard ids the qkv components ("q"/"k"/"v"), tuple shard ids one checkpoint
    matrix covering several merged shards (in_proj_qkv (0,1,2)), and None the
    single non-merged matrix.
    """
    if shard_id is None:
        return 0
    if isinstance(shard_id, str):
        return _SHARD_NAME_TO_INDEX.get(shard_id, 0)
    if isinstance(shard_id, int):
        return shard_id
    return min(int(i) for i in shard_id)


class ExL3Config(QuantizationConfig):
    """Config for EXL3 packs (quantization_config.json / config.json summary)."""

    def __init__(
        self,
        version: Optional[str] = None,
        bpw: Optional[float] = None,
        head_bits: int = 6,
        codebook: str = "mul1",
        out_scales: str = "always",
        mtp_bits: Optional[int] = None,
        calibration: Optional[Dict[str, Any]] = None,
        tensor_storage: Optional[Dict[str, Any]] = None,
        **extra,
    ):
        super().__init__()
        self.version = version
        self.bpw = bpw
        self.head_bits = head_bits
        self.codebook = codebook
        self.out_scales = out_scales
        self.mtp_bits = mtp_bits
        self.calibration = calibration
        self.tensor_storage = tensor_storage or {}

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantization_config.json", "config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExL3Config":
        return cls(
            version=config.get("version"),
            bpw=config.get("bpw"),
            head_bits=config.get("head_bits", 6),
            codebook=config.get("codebook", "mul1"),
            out_scales=config.get("out_scales", "always"),
            mtp_bits=config.get("mtp_bits"),
            calibration=config.get("calibration"),
            tensor_storage=config.get("tensor_storage"),
        )

    def get_scaled_act_names(self) -> List[str]:
        return ["gate", "up"]

    def get_quant_method(self, layer, prefix: str) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        # qwen3_5-class plain modules: b/a gates (plain F16), embeddings and
        # the vision tower stay unquantized. SGLang Linear requires a quant
        # method when quant_config is given — plain prefixes get the
        # UnquantizedLinearMethod (embeddings fall back internally on None).
        if "in_proj_ba" in prefix or "visual" in prefix:
            return UnquantizedLinearMethod()
        if "embed_tokens" in prefix:
            return None
        # FusedMoE under EXL3 (lazy import: this module is imported by the
        # quantization registry long before the MoE layer tree). isinstance
        # first: the prefix is empty for bare FusedMoE constructions.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            return ExL3MoEMethod(self)
        if isinstance(layer, ParallelLMHead):
            return ExL3HeadMethod(self)
        if isinstance(layer, RadixAttention):
            return ExL3KVCacheMethod(self)
        return ExL3LinearMethod(self)


class ExL3KVCacheMethod(QuantizeMethodBase):
    """No-op KV-cache method for RadixAttention under EXL3 (fp16/bf16 KV —
    v1 has no KV-cache quantization). RadixAttention calls
    create_weights(layer) with no weight args, hence the flexible signature."""

    def __init__(self, config: ExL3Config):
        self.config = config

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        return

    @torch.no_grad()
    def process_weights_after_loading(self, layer) -> None:
        return

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError("ExL3KVCacheMethod is not applied directly")


class ExL3LinearMethod(LinearMethodBase):
    """Dense EXL3 linear. Fast path (default): the native sgl_kernel M-tiled
    trellis GEMM (torch.ops.sgl_kernel.sgl_exl3_had_in + sgl_exl3_linear,
    cuda-exl3-lineage M-tiling with fused output Hadamard + svh epilogue) —
    weights stay packed on GPU and nothing is materialized. Fallback (env
    EXL3_NO_KERNEL=1 or the op unavailable): dequant-on-apply in pure torch.
    Groups follow the merged module's shard order, so fused modules (qkv_proj,
    gate_up_proj, in_proj_qkvz) cake per-checkpoint-matrix outputs along the
    output dim."""

    _kernel_state: Optional[bool] = None  # None = unchecked

    @classmethod
    def _kernel_available(cls) -> bool:
        if cls._kernel_state is None:
            try:
                cls._kernel_state = (
                    os.environ.get("EXL3_NO_KERNEL") != "1"
                    and hasattr(torch.ops.sgl_kernel, "sgl_exl3_had_in")
                )
            except Exception:
                cls._kernel_state = False
        return cls._kernel_state

    def __init__(self, config: ExL3Config):
        self.config = config

    # -- parameters ---------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer._exl3_output_sizes = list(output_partition_sizes)
        layer._exl3_records = {}
        layer._exl3_plain = {}
        for suffix in _EXL3_PARAMS:
            p = Parameter(torch.empty(0, dtype=_PARAM_DTYPES[suffix]), requires_grad=False)
            p.weight_loader = self._make_loader(layer, suffix)
            layer.register_parameter(suffix, p)
        # Mixed-checkpoint support (GLM r7 non_routed_dtype_policy =
        # official_source_native: only routed experts are trellis-packed).
        # Plain `.weight` tensors materialize INTO this param immediately on
        # first arrival — post-load hooks (the Deepseek MLA w_kc/w_vc split in
        # DeepseekV2WeightLoaderMixin.post_load_weights) read layer.weight
        # directly during load_weights, before process_weights_after_loading.
        # Quantized modules never receive a `.weight` tensor, so the param
        # stays 0-size and costs nothing.
        w = Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
        w.weight_loader = self._make_plain_loader(
            layer, sum(output_partition_sizes), input_size_per_partition,
            list(output_partition_sizes),
        )
        layer.register_parameter("weight", w)

    @staticmethod
    def _make_loader(layer: torch.nn.Module, suffix: str):
        def loader(param, loaded_weight, shard_id=None):
            rec = layer._exl3_records.setdefault(suffix, {})
            rec.setdefault(shard_id, []).append(loaded_weight)

        return loader

    @staticmethod
    def _make_plain_loader(layer: torch.nn.Module, out_total: int, in_size: int,
                           output_partition_sizes: List[int]):
        def loader(param, loaded_weight, shard_id=None):
            layer._exl3_plain.setdefault(shard_id, []).append(loaded_weight)
            w = layer.weight
            if w.numel() == 0:
                w = Parameter(
                    torch.empty(out_total, in_size, dtype=loaded_weight.dtype,
                                device="cuda"),
                    requires_grad=False,
                )
                layer.weight = w
            idx = _shard_first_index(shard_id)
            off = sum(output_partition_sizes[:idx])
            w.data[off:off + loaded_weight.shape[0]].copy_(loaded_weight)

        return loader

    # -- post-processing -----------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        recs = layer._exl3_records
        trellis_by = recs.get("trellis", {})
        suh_by = recs.get("suh", {})
        svh_by = recs.get("svh", {})
        if not trellis_by:
            # Mixed checkpoint: this module's tensors arrived as plain native
            # weights (already materialized into layer.weight by the loader).
            layer._exl3_groups = []
            layer._exl3_plain = {}
            layer._exl3_records = {}
            if layer.weight.numel() == 0:
                raise RuntimeError(
                    "exl3: no trellis tensors and no plain weights arrived for "
                    f"{type(layer).__name__}; legacy .su/.sv sign-bitfield packs "
                    "are not supported"
                )
            return
        sizes = layer._exl3_output_sizes
        groups = []
        for shard_id in trellis_by:
            order = _shard_first_index(shard_id)
            suh = suh_by.get(shard_id)[0]
            svh = svh_by.get(shard_id)[0]
            mul1 = recs.get("mul1", {}).get(shard_id, [None])[0]
            mcg = recs.get("mcg", {}).get(shard_id, [None])[0]
            codebook = _sentinel_to_codebook(mul1 if mul1 is not None else mcg)
            if codebook == "default":
                raise ValueError(
                    "exl3: this checkpoint has no codebook marker; only mul1/mcg packs are supported in v1"
                )
            trellis = trellis_by[shard_id][0]
            if trellis.device.type != "cuda":
                trellis = trellis.cuda()
            if suh.device.type != "cuda":
                suh = suh.cuda()
            if svh.device.type != "cuda":
                svh = svh.cuda()
            bias = recs.get("bias", {}).get(shard_id, [None])[0]
            bias = recs.get("bias", {}).get(shard_id, [None])[0]
            groups.append({
                "shard_id": shard_id,
                "order": order,
                "trellis": trellis,
                "suh": suh,
                "svh": svh,
                "codebook": codebook,
                "cb": {"default": 0, "mcg": 1, "mul1": 2}[codebook],
                "bias": bias.cuda() if bias is not None else None,
            })
        groups.sort(key=lambda g: g["order"])
        layer._exl3_groups = groups
        gbias = [g["bias"] for g in groups if g["bias"] is not None]
        layer._exl3_gbias = torch.cat(gbias, dim=-1) if gbias else None

    # -- forward -------------------------------------------------------------

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_shape = x.shape
        if not getattr(layer, "_exl3_groups", None):
            # Mixed-checkpoint plain module: native weight, native GEMM.
            return F.linear(x, layer.weight, bias)
        # bf16 x goes straight into had_in (converted in-kernel) — the fp16
        # round-trip copy showed up as 0.5 ms/token of pure direct_copy in
        # the decode profile. fp16 x is unchanged. The GEMM stays
        # fp16-in/fp32-acc; output dtype follows x (see below).
        if x.dtype == torch.bfloat16:
            x2 = x.reshape(-1, x_shape[-1]).contiguous()
        else:
            x2 = x.reshape(-1, x_shape[-1]).to(torch.float16).contiguous()
        # bf16 models get a bf16 GEMM output: the fp16 epilogue's 65504
        # ceiling would saturate tight-calibration checkpoints whose outputs
        # legitimately exceed fp16 range (the DFlash2 drafter; the vendor
        # runs it in bf16). The GEMM itself stays fp16-in/fp32-acc.
        out_dt = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float16
        used_kernel = False
        if self._kernel_available() and layer._exl3_groups:
            used_kernel = True
            outs = []
            for g in layer._exl3_groups:
                xh = torch.empty(x2.shape, dtype=torch.float16, device=x2.device)
                torch.ops.sgl_kernel.sgl_exl3_had_in(x2, g["suh"], xh)
                out_g = torch.empty((x2.shape[0], g["svh"].shape[0]),
                                    dtype=out_dt, device=x2.device)
                torch.ops.sgl_kernel.sgl_exl3_linear(xh, g["trellis"], g["svh"],
                                                     g["bias"], g["cb"], out_g)
                outs.append(out_g)
            y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        else:
            if x2.dtype != torch.float16:
                x2 = x2.to(torch.float16)
            outs = []
            for g in layer._exl3_groups:
                W = dequant_matrix_orig(g["trellis"], g["suh"], g["svh"], g["codebook"])
                outs.append(F.linear(x2, W.t()))
            y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
            # Same numerics-only guard as the kernel epilogue: tight-calibration
            # checkpoints can produce final outputs past fp16 range, and the
            # fp16 GEMM output store overflows to inf -> NaN identically.
            y = torch.clamp(y.float(), -65504.0, 65504.0).to(torch.float16)
        gb = None if used_kernel else layer._exl3_gbias
        if gb is not None:
            y = y + gb.to(torch.float16)
        return y.view(*x_shape[:-1], y.shape[-1]).to(x.dtype)


class ExL3HeadMethod(ExL3LinearMethod):
    """Quantized lm_head: EXL3 params at load, dequanted ONCE at
    process_weights_after_loading into a plain fp16 weight, so logits stay a
    plain GEMM (the 27B-class head materialization is ~2.5 GB fp16). Native
    (unquantized) lm_heads — GLM r7 stores lm_head.weight plain — load through
    the base plain-stash path unchanged."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not layer._exl3_records.get("trellis", {}):
            # Plain native head: base materializes layer.weight from the stash.
            super().process_weights_after_loading(layer)
            return
        super().process_weights_after_loading(layer)
        g = layer._exl3_groups
        if len(g) != 1:
            raise RuntimeError("exl3: quantized lm_head must be a single matrix")
        trellis, suh = g[0]["trellis"], g[0]["suh"]
        svh, cb = g[0]["svh"], g[0]["codebook"]
        # Chunk over the vocab dim: the output-side Hadamard is blockwise per
        # 128 columns, so 128-aligned column chunks are bit-identical while
        # bounding the fp32 intermediates of the preapply chain.
        vocab = svh.shape[0]
        in_dim = suh.shape[0]
        layer.weight = Parameter(
            torch.empty(vocab, in_dim, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        chunk = 8192
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            piece = dequant_matrix_orig(trellis[:, start // 16:end // 16, :],
                                        suh, svh[start:end], cb)  # (in, end-start) fp16
            layer.weight.data[start:end].copy_(piece.t())
        layer._exl3_groups = []
        layer._exl3_records = {}
        g.clear()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        x_dtype = x.dtype
        logits = F.linear(x.to(torch.float16), layer.weight.to(torch.float16))
        return logits.to(x_dtype)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class ExL3MoeRunnerCore(DispatchMoeRunnerCore):
    """Packed-trellis MoE runner core.

    Consumes the standard dispatch output (hidden states + localized topk)
    directly and runs the EXL3 trellis kernels per active expert — weights
    never leave their packed quantized form. EP-localized topk ids arrive
    with -1 for remote experts (StandardDispatcher handles the mapping);
    remote pairs contribute nothing on this rank and the per-layer all-reduce
    combines the partial sums.
    """

    def __init__(self, config: MoeRunnerConfig, method: "ExL3MoEMethod"):
        super().__init__(config)
        self._method = method

    @property
    def runner_backend(self) -> Any:
        return "exl3"

    def run_from_dispatch(
        self,
        dispatch_output: Any,
        quant_info: Any,
        runner_config: MoeRunnerConfig,
        hooks: Any = None,
    ) -> Any:
        return self._method.run_packed_moe(dispatch_output, runner_config)


class ExL3MoEMethod(FusedMoEMethodBase):
    """EXL3 routed experts on FusedMoE.

    Per-expert trellis packs arrive as fused-module params w13_{trellis,suh,...}
    / w2_{...} (the model's expert_params_mapping replaces
    ``experts.<e>.<proj>.<suffix>`` with ``experts.w13_<suffix>`` /
    ``experts.w2_<suffix>``), are stashed per (expert_id, prefix, shard_id,
    suffix) by the loader, and are dequantized ONCE in
    process_weights_after_loading into fused fp16 w13/w2 expert weights run by
    the standard Triton fused-MoE runner — the same numerics path as
    unquantized MoE. Mixed-expert checkpoints (routed experts quantized, shared
    expert native — GLM r7) materialize plain expert tensors through the same
    path: gate/up stacked to (2I, H), down kept (H, I).

    Memory note: this materializes every local expert at params_dtype — for a
    288-expert GLM-5.3 layer that is ~14.5 GB fp16 versus ~3.2 GB packed.
    Validation-scale only; Phase 3.5 replaces it with the packed-trellis
    grouped GEMM (weights stay packed on GPU, R2 design).

    TP/EP: experts load whole; TP > 1 or EP > 1 is rejected for now.
    """

    def __init__(self, config: ExL3Config):
        super().__init__()
        self.config = config
        self.with_bias = False
        self.moe_runner_config = None
        self.hidden_size = None
        self.intermediate_size = None
        self.params_dtype = None
        # Packed mode keeps per-expert trellis/suh/svh on GPU and runs the
        # EXL3 kernels per active expert instead of materializing fp16
        # experts. Opt in with SGLANG_EXL3_MOE_PACKED=1; required for
        # tp/ep > 1 (sharding applies to packed experts, not fp16 copies).
        self.packed = (
            os.environ.get("SGLANG_EXL3_MOE_PACKED", "0") == "1"
            and os.environ.get("EXL3_NO_KERNEL") != "1"
            and hasattr(torch.ops.sgl_kernel, "sgl_exl3_had_in")
        )
        self._packed_state = None

    # -- parameters ---------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition
        self.params_dtype = params_dtype
        layer._exl3_moe_records = {}
        for prefix in ("w13", "w2"):
            for suffix in _EXL3_PARAMS:
                p = Parameter(
                    torch.empty(0, dtype=_PARAM_DTYPES[suffix]), requires_grad=False
                )
                p.weight_loader = self._make_expert_loader(layer, prefix, suffix)
                layer.register_parameter(f"{prefix}_{suffix}", p)
        # Plain-expert fallback targets: unquantized expert tensors
        # (`experts.<e>.<proj>.weight`, e.g. the fused shared expert) map here
        # through the same expert_params_mapping replace, and are materialized
        # in process_weights_after_loading alongside trellis experts.
        for prefix in ("w13", "w2"):
            p = Parameter(torch.empty(0, dtype=params_dtype), requires_grad=False)
            p.weight_loader = self._make_expert_loader(layer, prefix, "weight")
            layer.register_parameter(f"{prefix}_weight", p)

    @staticmethod
    def _make_expert_loader(layer: torch.nn.Module, prefix: str, suffix: str):
        def loader(param, loaded_weight, weight_name=None, shard_id=None,
                   expert_id=None):
            if expert_id is None:
                raise RuntimeError(
                    "exl3 MoE: expert weights must arrive with an expert_id"
                )
            key = (expert_id, prefix, shard_id, suffix)
            layer._exl3_moe_records.setdefault(key, []).append(loaded_weight)

        return loader

    # -- post-processing -----------------------------------------------------

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.packed:
            self._process_packed(layer)
            return
        if layer.moe_ep_size != 1 or layer.moe_tp_size != 1:
            raise NotImplementedError(
                "exl3 MoE: TP/EP sharding of packed trellis experts is not "
                "supported yet; run with tp=1 ep=1 (or SGLANG_EXL3_MOE_PACKED=1)"
            )
        recs = layer._exl3_moe_records
        E = layer.num_local_experts
        I = self.intermediate_size
        H = self.hidden_size
        dt = self.params_dtype
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gated = self.moe_runner_config.is_gated if self.moe_runner_config else True
        w13_n = 2 * I if gated else I
        w13 = torch.empty((E, w13_n, H), dtype=dt, device=dev)
        w2 = torch.empty((E, H, I), dtype=dt, device=dev)

        def get(e, prefix, shard, suffix):
            lst = recs.get((e, prefix, shard, suffix))
            return lst[0] if lst else None

        def cb_of(e, prefix, shard):
            mul1 = get(e, prefix, shard, "mul1")
            mcg = get(e, prefix, shard, "mcg")
            codebook = _sentinel_to_codebook(mul1 if mul1 is not None else mcg)
            if codebook == "default":
                raise ValueError(
                    "exl3 MoE: expert matrix without codebook marker; only "
                    "mul1/mcg packs are supported in v1"
                )
            return codebook

        def deq(trellis, suh, svh, codebook):
            return dequant_matrix_orig(
                trellis.to(dev, non_blocking=True), suh.to(dev, non_blocking=True),
                svh.to(dev, non_blocking=True), codebook,
            )  # (K, N) fp16

        for e in range(E):
            gt = get(e, "w13", "w1", "trellis")
            dtt = get(e, "w2", "w2", "trellis")
            if gt is not None:
                if dtt is None or get(e, "w13", "w3", "trellis") is None:
                    raise RuntimeError(f"exl3 MoE: expert {e} has partial trellis set")
                gw = deq(gt, get(e, "w13", "w1", "suh"), get(e, "w13", "w1", "svh"),
                         cb_of(e, "w13", "w1"))  # (H, I)
                uw = deq(get(e, "w13", "w3", "trellis"), get(e, "w13", "w3", "suh"),
                         get(e, "w13", "w3", "svh"), cb_of(e, "w13", "w3"))
                dw = deq(dtt, get(e, "w2", "w2", "suh"), get(e, "w2", "w2", "svh"),
                         cb_of(e, "w2", "w2"))  # (I, H)
                w13[e] = torch.cat(
                    [gw.t(), *( [uw.t()] if gated else [] )], dim=0
                ).to(dt)
                w2[e] = dw.t().to(dt)
            else:
                # Plain native expert (fused shared expert / mixed checkpoint).
                gp = get(e, "w13", "w1", "weight")
                up = get(e, "w13", "w3", "weight")
                dp = get(e, "w2", "w2", "weight")
                if dp is None or (gp is None and (not gated or up is None)):
                    raise RuntimeError(
                        f"exl3 MoE: expert {e} has neither trellis nor plain tensors"
                    )
                w13[e] = torch.cat(
                    [gp, *( [up] if gated else [] )], dim=0
                ).to(dt) if gated else gp.to(dt)
                w2[e] = dp.to(dt)
            # Free this expert's stashed pieces (packed CPU/GPU refs).
            for k in [k for k in recs if k[0] == e]:
                del recs[k]
        layer._exl3_moe_records = {}
        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        # EXL3 per-matrix biases fold into per-expert fp32 bias vectors.
        gb = [get(e, "w13", "w1", "bias") for e in range(E)]
        ub = [get(e, "w13", "w3", "bias") for e in range(E)]
        db = [get(e, "w2", "w2", "bias") for e in range(E)]
        if any(b is not None for b in gb + ub + db):
            b13 = torch.zeros((E, w13_n), dtype=torch.float32, device=dev)
            b2 = torch.zeros((E, H), dtype=torch.float32, device=dev)
            for e in range(E):
                if gb[e] is not None:
                    b13[e, :I] = gb[e].to(dev).float()
                if gated and ub[e] is not None:
                    b13[e, I:] = ub[e].to(dev).float()
                if db[e] is not None:
                    b2[e] = db[e].to(dev).float()
            layer.w13_weight_bias = Parameter(b13, requires_grad=False)
            layer.w2_weight_bias = Parameter(b2, requires_grad=False)
        torch.cuda.empty_cache()

    # -- packed mode ---------------------------------------------------------

    _EXL3_CB_IDS = {"default": 0, "mcg": 1, "mul1": 2}

    @torch.no_grad()
    def _process_packed(self, layer: torch.nn.Module) -> None:
        """Keep per-expert trellis packs on GPU; no fp16 materialization."""
        recs = layer._exl3_moe_records
        E = layer.num_local_experts
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        gated = self.moe_runner_config.is_gated if self.moe_runner_config else True

        def get(e, prefix, shard, suffix):
            lst = recs.get((e, prefix, shard, suffix))
            return lst[0] if lst else None

        def pack_matrix(e, prefix, shard):
            trellis = get(e, prefix, shard, "trellis")
            if trellis is None:
                raise RuntimeError(
                    "exl3 MoE packed mode requires every local expert to be "
                    f"trellis-quantized (expert {e} {prefix}/{shard} is not)"
                )
            suh = get(e, prefix, shard, "suh")
            svh = get(e, prefix, shard, "svh")
            mul1 = get(e, prefix, shard, "mul1")
            mcg = get(e, prefix, shard, "mcg")
            codebook = _sentinel_to_codebook(mul1 if mul1 is not None else mcg)
            if codebook == "default":
                raise ValueError(
                    "exl3 MoE packed: expert matrix without codebook marker"
                )
            bias = get(e, prefix, shard, "bias")
            return (
                trellis.to(dev),
                suh.to(dev),
                svh.to(dev),
                bias.to(dev) if bias is not None else None,
                self._EXL3_CB_IDS[codebook],
            )

        packed = {"gate": [], "up": [], "down": []}
        for e in range(E):
            packed["gate"].append(pack_matrix(e, "w13", "w1"))
            if gated:
                packed["up"].append(pack_matrix(e, "w13", "w3"))
            packed["down"].append(pack_matrix(e, "w2", "w2"))
            for k in [k for k in recs if k[0] == e]:
                del recs[k]
        layer._exl3_moe_records = {}
        layer._exl3_moe_packed = packed
        self._packed_state = packed
        self._packed_gated = gated
        torch.cuda.empty_cache()

    def _exl3_gemm(self, x2, matrix, out_dt):
        """One packed trellis GEMM, same numerics contract as the dense path."""
        trellis, suh, svh, bias, cb = matrix
        xh = torch.empty(x2.shape, dtype=torch.float16, device=x2.device)
        torch.ops.sgl_kernel.sgl_exl3_had_in(x2, suh, xh)
        out = torch.empty((x2.shape[0], svh.shape[0]), dtype=out_dt, device=x2.device)
        torch.ops.sgl_kernel.sgl_exl3_linear(xh, trellis, svh, bias, cb, out)
        return out

    def run_packed_moe(self, dispatch_output, runner_config):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        packed = self._packed_state
        hs = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids = topk.topk_ids
        wts = topk.topk_weights
        if runner_config.apply_router_weight_on_input:
            raise NotImplementedError(
                "exl3 MoE packed: apply_router_weight_on_input is not supported"
            )
        x = hs.reshape(-1, hs.shape[-1])
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.float16)
        out_dt = x.dtype
        E = len(packed["down"])
        K = ids.shape[-1]
        flat = ids.reshape(-1).to(torch.long)
        # Sort (token, expert) pairs by expert; remote pairs (-1) trail at the
        # sentinel E and are never touched.
        key = torch.where(flat >= 0, flat, torch.full_like(flat, E))
        order = torch.argsort(key, stable=True)
        sids = key[order]
        tok = order // K
        pw = wts.reshape(-1)[order].float()
        rsf = runner_config.routed_scaling_factor
        if rsf is not None:
            pw = pw * rsf
        counts = torch.bincount(sids, minlength=E + 1)[:E]
        counts_l = counts.cpu().tolist()
        starts_l = [0]
        for c in counts_l:
            starts_l.append(starts_l[-1] + c)

        out = torch.zeros(
            (x.shape[0], packed["down"][0][2].shape[0]),
            dtype=torch.float32,
            device=x.device,
        )
        gated = self._packed_gated
        act = runner_config.activation
        for e in range(E):
            s0, s1 = starts_l[e], starts_l[e + 1]
            if s1 == s0:
                continue
            rows = tok[s0:s1]
            x_e = x[rows]
            o_g = self._exl3_gemm(x_e, packed["gate"][e], out_dt)
            if gated:
                o_u = self._exl3_gemm(x_e, packed["up"][e], out_dt)
                if act == "silu":
                    mid = F.silu(o_g.float()) * o_u.float()
                elif act == "gelu":
                    mid = F.gelu(o_g.float()) * o_u.float()
                else:
                    raise NotImplementedError(
                        f"exl3 MoE packed: activation {act} not supported"
                    )
            else:
                mid = F.silu(o_g.float()) if act == "silu" else F.gelu(o_g.float())
            mid = mid.to(out_dt)
            o_d = self._exl3_gemm(mid, packed["down"][e], out_dt)
            out.index_add_(0, rows, o_d.float() * pw[s0:s1, None])
        return StandardCombineInput(hidden_states=out.to(hs.dtype))

    # -- runner --------------------------------------------------------------

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
        if self.packed:
            self.runner.fused_func = None
            self.runner.runner_core = ExL3MoeRunnerCore(moe_runner_config, self)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        if self.packed:
            # Packed mode ignores quant info; the runner core reads the
            # per-expert packs straight off the method.
            return TritonMoeQuantInfo(w13_weight=None, w2_weight=None)
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def apply(self, layer: torch.nn.Module, dispatch_output):
        return self.runner.run(dispatch_output, self.get_triton_quant_info(layer))
