#!/bin/bash

TEST_APP_NAME="pathfinder"

# set cuda installation path
export CUDA_INSTALL_PATH=/usr/local/cuda

# load environment variables
source setup_environment

# compile project
make -j$(nproc)

# # 删除test_apps/${TEST_APP_NAME}/result下的所有文件
# rm -rf test_apps/${TEST_APP_NAME}/result/*


# # 生成result
# idx=0
# while IFS= read -r line || [[ -n "$line" ]]; do
#     echo "$idx: $line"
# if [[ "$idx" == "1" || "$idx" == "3" || "$idx" == "5" ]]; then
#     cu_file="test_apps/${TEST_APP_NAME}/result_gen/${TEST_APP_NAME}_0.cu"
#     if [[ -f "$cu_file" ]]; then
#         filename=$(basename "$cu_file")
#         x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
#         if [[ -n "$x_val" ]]; then
#             cp "$cu_file" "${cu_file}.bak"
#             /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/result_gen/gen
#             # 将输出重定向到目录下的txt文件
#             ./test_apps/${TEST_APP_NAME}/result_gen/gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
#             rm -rf test_apps/${TEST_APP_NAME}/result_gen/gen
#             mv "${cu_file}.bak" "$cu_file"
#         fi
#     fi
# else
#     for cu_file in test_apps/${TEST_APP_NAME}/result_gen/${TEST_APP_NAME}_*.cu; do
#         filename=$(basename "$cu_file")
#         x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
#         if [[ -z "$x_val" ]]; then
#             continue
#         fi

#         cp "$cu_file" "${cu_file}.bak"
#         /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/result_gen/gen
#         # 将输出重定向到目录下的txt文件
#         ./test_apps/${TEST_APP_NAME}/result_gen/gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
#         rm -rf test_apps/${TEST_APP_NAME}/result_gen/gen
#         mv "${cu_file}.bak" "$cu_file"
#     done
# fi
#     idx=$((idx+1))
# done < test_apps/${TEST_APP_NAME}/size_list.txt

for result_file in test_apps/${TEST_APP_NAME}/result/*; do
    filename=$(basename "$result_file")
    # 提取a和b
    a=$(echo "$filename" | cut -d'-' -f1)
    b_with_ext=$(echo "$filename" | cut -d'-' -f2)
    b=$(echo "$b_with_ext" | cut -d'.' -f1)

    # 复制result文件到根目录并重命名为result.txt
    cp "$result_file" ./result.txt

    # 查找inject_app下b对应的cu文件并移动到根目录，重命名为${TEST_APP_NAME}
    cu_file="test_apps/${TEST_APP_NAME}/inject_app/${TEST_APP_NAME}_${b}.cu"
    if [[ -f "$cu_file" ]]; then
        cp "$cu_file" "./${TEST_APP_NAME}.cu"
    fi

    echo "正在进行输入规模${a}、输入内容${b}的故障注入"
    nvcc ${TEST_APP_NAME}.cu -o ${TEST_APP_NAME} -g -lcudart -arch=sm_75
    ./${TEST_APP_NAME} > "exec_log.txt"
done
