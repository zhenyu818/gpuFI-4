#!/usr/bin/env python3
import re
from collections import defaultdict

LOGFILE = "inst_exec.log"   # 你可以改成 exec.log 或别的

# {(kernel, instr_id, instr): {"Masked":x, "SDC":y, "DUE":z}}
stats = defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0})

kernel = ""
instr = ""
instr_id = ""

with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # 提取 kernel 名称
        m_kernel = re.search(r"inst=\s*([a-zA-Z0-9_]+)\s+PC", line)
        if m_kernel:
            kernel = m_kernel.group(1)

        # 提取目标指令编号 A(dst)_icount
        m_id = re.search(r"A\(dst\)_icount=(\d+)", line)
        if m_id:
            instr_id = m_id.group(1)

        # 提取指令内容（分号结尾）
        m_instr = re.search(r"inst=\s*[a-zA-Z0-9_]+\s+PC=.*?\)\s*([^;]+;)", line)
        if m_instr:
            instr = m_instr.group(1).strip()

        # 判断结果
        if "tmp.out" in line and ":" in line:
            if "Masked" in line:
                result = "Masked"
            elif "SDC" in line:
                result = "SDC"
            elif "DUE" in line:
                result = "DUE"
            else:
                result = None

            if result:
                key = (kernel or "N/A", instr_id or "N/A", instr or "N/A")
                stats[key][result] += 1

# ---------------- 输出结果 ----------------
print(f"{'Kernel':20} | {'Instr_ID':8} | {'Instruction':60} | {'Masked':6} | {'SDC':6} | {'DUE':6}")
print("-" * 120)
for (k, idx, i), c in sorted(stats.items(), key=lambda x: (x[0][0], int(x[0][1]) if x[0][1].isdigit() else 99999)):
    print(f"{k:20} | {idx:8} | {i:60} | {c['Masked']:6d} | {c['SDC']:6d} | {c['DUE']:6d}")

# ---------------- 每个 kernel 汇总 ----------------
kernel_sum = defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0})
for (k, _, _), c in stats.items():
    for t in ("Masked", "SDC", "DUE"):
        kernel_sum[k][t] += c[t]

print("\nSummary (per kernel):")
print(f"{'Kernel':20} | {'Masked':6} | {'SDC':6} | {'DUE':6}")
print("-" * 45)
for k, c in kernel_sum.items():
    print(f"{k:20} | {c['Masked']:6d} | {c['SDC']:6d} | {c['DUE']:6d}")
