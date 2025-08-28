#!/bin/bash

TEST_APP_NAME="pathfinder"

# # set cuda installation path
# export CUDA_INSTALL_PATH=/usr/local/cuda

# # load environment variables
# source setup_environment

# # compile project
# make -j$(nproc)

# 删除test_apps/${TEST_APP_NAME}目录下所有.txt文件，但保留size_list.txt
find test_apps/${TEST_APP_NAME}/ -maxdepth 1 -type f -name "*.txt" ! -name "size_list.txt" -exec rm -f {} +

# 遍历test_apps/${TEST_APP_NAME}/size_list.txt的每一行，并编号打印
idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
    echo "$idx: $line"
    idx=$((idx+1))
if [[ "$idx" == "0" || "$idx" == "2" || "$idx" == "4" ]]; then
    cu_file="test_apps/${TEST_APP_NAME}/${TEST_APP_NAME}_0.cu"
    if [[ -f "$cu_file" ]]; then
        filename=$(basename "$cu_file")
        x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
        if [[ -n "$x_val" ]]; then
            cp "$cu_file" "${cu_file}.bak"
            sed -i "s/^#define[[:space:]]\+EXP_NAME.*/#define EXP_NAME \"$idx-$x_val\"/" "$cu_file"
            /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/gen
            # 将输出重定向到目录下的txt文件
            ./test_apps/${TEST_APP_NAME}/gen $line > "test_apps/${TEST_APP_NAME}/${idx}-${x_val}.txt"
            rm -rf test_apps/${TEST_APP_NAME}/gen
            mv "${cu_file}.bak" "$cu_file"
        fi
    fi
else
    for cu_file in test_apps/${TEST_APP_NAME}/${TEST_APP_NAME}_*.cu; do
        filename=$(basename "$cu_file")
        x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
        if [[ -z "$x_val" ]]; then
            continue
        fi

        cp "$cu_file" "${cu_file}.bak"
        sed -i "s/^#define[[:space:]]\+EXP_NAME.*/#define EXP_NAME \"$idx-$x_val\"/" "$cu_file"
        /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/gen
        # 将输出重定向到目录下的txt文件
        ./test_apps/${TEST_APP_NAME}/gen $line > "test_apps/${TEST_APP_NAME}/${idx}-${x_val}.txt"
        rm -rf test_apps/${TEST_APP_NAME}/gen
        mv "${cu_file}.bak" "$cu_file"
    done
fi

done < test_apps/${TEST_APP_NAME}/size_list.txt
