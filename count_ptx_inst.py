#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_ptx_instructions.py
用法:
  python count_ptx_instructions.py app
说明:
  读取 ./app.ptx，统计并打印其中 PTX 指令的条数（仅输出数字）。
"""

import sys
import re
from pathlib import Path

PRED_RE = re.compile(r'^(?:@!?%p\d+\s+)+')


def is_ptx_instruction(line: str) -> bool:
    """判断一行是否是真正的 PTX 指令"""
    # 去掉行内注释与空白
    line = re.sub(r'//.*', '', line).strip()
    if not line:
        return False

    # 忽略标签与花括号
    if line.endswith(':'):
        return False
    if line[0] in '{}':
        return False

    # 忽略以 . 开头的编译器/汇编指示与声明（.version/.target/.reg/.loc/.file/.entry/.param 等）
    if line.startswith('.'):
        return False

    # 忽略参数列表收尾的括号
    if line[0] in '()':
        return False

    # 去掉前缀谓词（例如: @%p1 / @!%p3），可能有多个
    line = PRED_RE.sub('', line).strip()
    if not line:
        return False

    # 规范的 PTX 指令以分号结束
    return line.endswith(';')


def main():
    if len(sys.argv) != 2:
        print("用法: python count_ptx_instructions.py <basename>\n例如: python count_ptx_instructions.py app", file=sys.stderr)
        sys.exit(1)

    base = sys.argv[1]
    path = Path(base if base.endswith(".ptx") else base + ".ptx")
    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        sys.exit(2)

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if is_ptx_instruction(line):
                count += 1

    # 只输出数量
    print()
    print("ptx_inst_count:", count)


if __name__ == "__main__":
    main()
