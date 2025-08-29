#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "用法: $0 <m> <n>"
    echo "  m: 最大值（范围0到m）"
    echo "  n: 要生成的数字数量"
    exit 1
fi

m=$1
n=$2

# 检查参数是否为正整数
if ! [[ "$m" =~ ^[0-9]+$ ]] || [ "$m" -lt 0 ]; then
    echo "错误: m 必须是非负整数"
    exit 1
fi

if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -le 0 ]; then
    echo "错误: n 必须是正整数"
    exit 1
fi

# 检查cycles.txt是否存在
if [ -f "cycles.txt" ]; then
    echo "检测到 cycles.txt 文件，正在清空..."
    > cycles.txt
else
    echo "cycles.txt 文件不存在，正在创建..."
    touch cycles.txt
fi

# 生成从0到m均匀分布的n个数
echo "正在生成 $n 个从0到 $m 的均匀分布数字..."

# 使用awk生成均匀分布的数字
awk -v m="$m" -v n="$n" 'BEGIN {
    if (n == 1) {
        print int(m/2)
    } else {
        for (i = 0; i < n; i++) {
            value = int((i * m) / (n - 1))
            print value
        }
    }
}' > cycles.txt

echo "完成！已在 cycles.txt 中生成 $n 个数字"
echo "文件内容预览："
head -10 cycles.txt
if [ $(wc -l < cycles.txt) -gt 10 ]; then
    echo "..."
    tail -5 cycles.txt
fi
