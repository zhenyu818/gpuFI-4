#!/usr/bin/env python3
import re
from collections import defaultdict

LOGFILE = "inst_exec.log"   # 你可以改成 exec.log 或别的

# {(kernel, instr_id, instr): {"Masked":x, "SDC":y, "DUE":z}}
stats = defaultdict(lambda: {"Masked": 0, "SDC": 0, "DUE": 0})

# 存储每个故障注入对应的指令信息
injection_to_instruction = {}

with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
    
    # 第一遍：建立故障注入编号到指令信息的映射
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 查找 Effects from 行
        m_injection = re.search(r"Effects from \./logs1/tmp\.out(\d+):", line)
        if m_injection:
            injection_num = m_injection.group(1)
            
            # 在下一行查找指令信息
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                # 提取指令信息
                m_kernel = re.search(r"inst=\s*([a-zA-Z0-9_]+)\s+PC", next_line)
                m_id = re.search(r"A\(dst\)_icount=(\d+)", next_line)
                m_instr = re.search(r"inst=\s*[a-zA-Z0-9_]+\s+PC=.*?\)\s*([^;]+;)", next_line)
                
                if m_kernel and m_id and m_instr:
                    kernel = m_kernel.group(1)
                    instr_id = m_id.group(1)
                    instr = m_instr.group(1).strip()
                    injection_to_instruction[injection_num] = (kernel, instr_id, instr)
    
    # 第二遍：统计故障注入结果
    for line in lines:
        line = line.strip()
        
        # 判断结果
        if "tmp.out" in line and ":" in line:
            m_result = re.search(r"tmp\.out(\d+):\s*(\w+)(?:\s*\([^)]*\))?", line)
            if m_result:
                injection_num = m_result.group(1)
                result = m_result.group(2)
                
                # 跳过 Unclassified 类型，不记录到统计中
                if result == "Unclassified":
                    continue
                
                if injection_num in injection_to_instruction:
                    # 有指令信息的情况
                    stats[injection_to_instruction[injection_num]][result] += 1
                else:
                    # 没有指令信息的情况，统计到"未知指令"
                    unknown_key = ("Invalid Injection", "-", "-")
                    stats[unknown_key][result] += 1

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

# ---------------- 总计 ----------------
total = {"Masked": 0, "SDC": 0, "DUE": 0}
for c in stats.values():
    for t in ("Masked", "SDC", "DUE"):
        total[t] += c[t]

print(f"\nTotal: Masked: {total['Masked']}, SDC: {total['SDC']}, DUE: {total['DUE']}")