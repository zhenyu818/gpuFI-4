#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import os

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <basename>")
        sys.exit(1)

    base = sys.argv[1]
    ptx_file = f"{base}.ptx"
    txt_file = "register_used.txt"

    if not os.path.exists(ptx_file):
        print(f"Error: {ptx_file} not found.")
        sys.exit(1)

    # 读取 PTX 文件
    with open(ptx_file, "r") as f:
        text = f.read()

    # 匹配寄存器：形如 %r32, %rd5, %f250, %p1
    registers = re.findall(r"%[a-zA-Z]+\d+", text)

    # 去重并排序
    registers = sorted(
        set(registers),
        key=lambda x: (re.sub(r"\d+", "", x), int(re.sub(r"\D", "", x) or 0))
    )

    # 写入日志（覆盖原有内容），每行一个寄存器
    with open(txt_file, "w") as f:
        f.write("\n".join(registers))

    print(f"Extracted {len(registers)} registers -> {txt_file}")

if __name__ == "__main__":
    main()
