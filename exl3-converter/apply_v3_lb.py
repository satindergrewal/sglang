import re

p = "/mnt/nvme0/work-exl3/exl3_ops_src/csrc/exl3/exl3_linear.cu"
s = open(p).read()

# 1) re-apply the char-literal fix (overwritten by the earlier scp)
s = s.replace('getenv("EXL3_PANEL_LAYOUT")[0] == B)', 'getenv("EXL3_PANEL_LAYOUT")[0] == (char)66)')

# 2) V3 kernel: add the layout_b arg (idempotent)
old_sig = """    float* __restrict__ g_ws,          // (col_blocks, splits, m, n) fp32
    int m, int k, int n, int n16, int cb, int splits)"""
if "int m, int k, int n, int n16, int cb, int splits, int layout_b)" not in s:
    assert old_sig in s, "v3 sig"
    s = s.replace(old_sig, """    float* __restrict__ g_ws,          // (col_blocks, splits, m, n) fp32
    int m, int k, int n, int n16, int cb, int splits, int layout_b)""", 1)

# 3) V3 kernel: replace ALL remaining A-addressing tile32 lines (there is one
#    per kernel: v2 done, v3 pending — handle both by replacing every
#    occurrence of the A tile32 line with the stride-based form)
a_line = """        const uint32_t* tile32 =
            (const uint32_t*)(g_packed + ((kk * n16) + col_block * 8 + warp) * words16);"""
b_line = """        const uint32_t* tile32 = (const uint32_t*)(tile_base + (kk - k_beg) * stride);"""
prelude = """        const long stride = layout_b ? (long)8 * words16 : (long)n16 * words16;
        const uint32_t* tile_base = (const uint32_t*)(g_packed +
            (layout_b ? ((long)col_block * k16 + k_beg) * 8 * words16
                      : ((long)k_beg * n16 + col_block * 8) * words16));
"""
count = s.count(a_line)
s = s.replace(a_line, prelude + b_line)
print(f"A->B tile32 replacements: {count}")

# 4) the V3 dispatches: append layout_b
n3 = len(re.findall(r"exl3::exl3_gemm_kernel_v3<\d, MP, (?:true|false)><<<grid, 256, 0, stream>>>\(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits\)", s))
s = re.sub(r"(exl3::exl3_gemm_kernel_v3<\d, MP, (?:true|false)><<<grid, 256, 0, stream>>>\(xp, packed_ptr, wsp, mi, ki, ni, n16i, icb, splits)\)",
           r"\1, layout_b)", s)
print(f"v3 dispatches wired: {n3}")

open(p, 'w').write(s)
