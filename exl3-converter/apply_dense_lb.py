import sys

p = "/mnt/nvme0/work-exl3/exl3_ops_src/csrc/exl3/exl3_linear.cu"
s = open(p).read()
i = s.find("template <int BITS, int BM, bool HALF_K = false>")
i = s.find("__global__ void exl3_gemm_kernel_v2(", i)
# the tile32 line is the FIRST one after this kernel start
tile_anchor = "(const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16)"
i_tile = s.find(tile_anchor, i)
assert i_tile > 0, "tile32 anchor"
# widen the segment: from the kernel start to 400 lines past the tile line
j = s.find("\n", i_tile + len(tile_anchor) + 400)
seg = s[i:j]

old_sig = """    const half* __restrict__ g_svh, const half* __restrict__ g_bias,
    half* __restrict__ g_out, int bf16_out)"""
assert old_sig in seg, "sig anchor"
seg = seg.replace(old_sig, """    const half* __restrict__ g_svh, const half* __restrict__ g_bias,
    half* __restrict__ g_out, int bf16_out, int layout_b)""", 1)

old_loop = """    for (int kk = k_beg; kk < k_end; kk++)
    {
        const uint32_t* tile32 =
            (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);"""
new_loop = """    // layout B: the panel is pre-ordered (col_block, kk, tile, words) so the
    // block's k-scan is contiguous (reorder_panels.py, EXL3_PANEL_LAYOUT=B).
    const long stride = layout_b ? (long)8 * words16 : (long)n16 * words16;
    const uint32_t* tile_base = (const uint32_t*)(g_packed +
        (layout_b ? ((long)col_block * k16 + k_beg) * 8 * words16
                  : ((long)k_beg * n16 + col_block * 8) * words16));
    for (int kk = k_beg; kk < k_end; kk++)
    {
        const uint32_t* tile32 = (const uint32_t*)(tile_base + (kk - k_beg) * stride);"""
assert old_loop in seg, "loop anchor"
seg = seg.replace(old_loop, new_loop, 1)
s = s[:i] + seg + s[j:]

# host: read the env once and pass layout_b to every v2 dispatch
old_disp = 'exl3::exl3_gemm_kernel_v2<1, 16, true><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, cp, mi, ki, ni, n16i, icb, splits, svp, bias_ptr, op, bf16_out)'
assert old_disp in s, "dispatch anchor"
import re
s = re.sub(r"exl3::exl3_gemm_kernel_v2<(\d), 16, (true|false)><<<grid, 256, 0, stream>>>\((xp, packed_ptr, wsp, cp, mi, ki, ni, n16i, icb, splits, svp, bias_ptr, op, bf16_out)\)",
           r"exl3::exl3_gemm_kernel_v2<\1, 16, \2><<<grid, 256, 0, stream>>>(xp, packed_ptr, wsp, cp, mi, ki, ni, n16i, icb, splits, svp, bias_ptr, op, bf16_out, layout_b)", s)
# declare layout_b in the host before the dispatch
old_host = "    dim3 grid"
k = s.find(old_host, i)
assert k > 0, "host grid anchor"
host_line = "    const int layout_b = (getenv(\"EXL3_PANEL_LAYOUT\") != nullptr && getenv(\"EXL3_PANEL_LAYOUT\")[0] == 'B');\n    (void)layout_b;\n"
s = s[:k] + host_line + s[k:]
open(p, 'w').write(s)
print("dense kernel + host layout_b applied; dispatch sites:", len(re.findall(r"layout_b\)", s)))
