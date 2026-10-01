import torch, json
from safetensors import safe_open
import sys
sys.path.insert(0, "/sgl-workspace/sglang/python")
torch.ops.load_library("/mnt/nvme0/work-exl3/exl3_ops_out2/exl3_ops_patch_serve/exl3_ops_patch_serve.so")
LIN = torch.ops.sgl_kernel.sgl_exl3_linear
HAD = torch.ops.sgl_kernel.sgl_exl3_had_in
E, K, N = 64, 4096, 2048
dev = "cuda"
wm = json.load(open("/art/model.safetensors.index.json"))["weight_map"]

# real panels: layer 47 expert 0 up_proj (4-bit mul1) + layer 0 attention qkv (5-bit)
name_up = "model.layers.47.mlp.experts.0.up_proj.trellis"
panel = safe_open("/art/" + wm[name_up], framework="pt", device=dev).get_tensor(name_up).contiguous()
svh = safe_open("/art/" + wm["model.layers.47.mlp.experts.0.up_proj.svh"], framework="pt", device=dev).get_tensor("model.layers.47.mlp.experts.0.up_proj.svh").contiguous().to(torch.half)

from sglang.srt.layers.quantization.exl3 import dequant_matrix_orig, _sentinel_to_codebook
# dequant reference for the panel
cb = 2  # mul1
W_ref = dequant_matrix_orig(panel, torch.zeros(K, dtype=torch.half, device=dev), svh, {0: "default", 1: "mcg", 2: "mul1"}[cb]).float()
print("ref weight:", W_ref.shape, W_ref.abs().max().item())

# gate: the dense kernel's output vs the reference matmul
P = 4
x = torch.randn(P, K, dtype=torch.half, device=dev)
xh = torch.empty_like(x)
HAD(x, torch.ones(K, dtype=torch.half, device=dev), xh)  # suh=1: identity hadamard
ref = (xh.float() @ W_ref.t().float())

out = torch.empty(P, N, dtype=torch.half, device=dev)
import os
LIN(xh, panel, svh, torch.zeros(N, dtype=torch.half, device=dev), 2, out)
torch.cuda.synchronize()
d = (out.float() - ref.float()).abs()
rel = (d / ref.float().abs().clamp(min=1.0)).max().item()
print(f"dense A-layout (cb=2): maxdiff={d.max().item():.4f} maxrel={rel:.4f}")

# layout B: reorder the panel and set the env
panelB = panel.view(K // 16, N // 128, 8, panel.shape[2]).permute(1, 0, 2, 3).contiguous().view(-1, panel.shape[2]).contiguous()
os.environ["EXL3_PANEL_LAYOUT"] = "B"
out2 = torch.empty(P, N, dtype=torch.half, device=dev)
LIN(xh, panelB, svh, torch.zeros(N, dtype=torch.half, device=dev), 2, out2)
torch.cuda.synchronize()
d2 = (out2.float() - ref.float()).abs()
rel2 = (d2 / ref.float().abs().clamp(min=1.0)).max().item()
print(f"dense B-layout (cb=2): maxdiff={d2.max().item():.4f} maxrel={rel2:.4f}")

# timing
import time
for tag, pn in (("A", panel), ("B", panelB)):
    for _ in range(5):
        LIN(xh, pn, svh, torch.zeros(N, dtype=torch.half, device=dev), 2, out)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200):
        LIN(xh, pn, svh, torch.zeros(N, dtype=torch.half, device=dev), 2, out)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 200 * 1e6
    wb_total = K * N * 4 / 8
    print(f"{tag}: {t:.1f}us ({wb_total / (t * 1e-6) / 1e9:.0f} GB/s)")
print("DENSE GATE DONE")
