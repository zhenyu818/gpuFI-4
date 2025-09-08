#!/bin/bash

TEST_APP_NAME="pathfinder"
COMPONENTS_TO_FLIP=0
# 0:RF, 1:local_mem, 2:shared_mem, 3:L1D_cache, 4:L1C_cache, 5:L1T_cache, 6:L2_cache (e.g. components_to_flip=0:1 for both RF and local_mem)

# set cuda installation path
export CUDA_INSTALL_PATH=/usr/local/cuda

# -------- Global metrics storage (script-wide) --------
GLOBAL_CYCLES=""
GLOBAL_MAX_REGISTERS_USED=""
GLOBAL_SHADER_USED=""
GLOBAL_DATATYPE_SIZE=""
GLOBAL_LMEM_SIZE_BITS=""
GLOBAL_SMEM_SIZE_BITS=""

# -------- CYCLES --------
get_cycles() {
    local v
    v=$(grep -E "^gpu_tot_sim_cycle\s*=\s*[0-9]+" "$FILE_PATH" | tail -n1 | sed -E 's/.*=\s*([0-9]+).*/\1/')
    if [ -z "$v" ]; then
        v=$(grep -E "^gpu_sim_cycle\s*=\s*[0-9]+" "$FILE_PATH" | tail -n1 | sed -E 's/.*=\s*([0-9]+).*/\1/')
    fi
    echo "${v:-0}"
}

# -------- regs/lmem/smem (bytes) --------
get_kernel_triplet() {
    local line regs lmem smem
    line=$(grep -m1 -E "regs=[0-9]+,\s*lmem=[0-9]+,\s*smem=[0-9]+" "$FILE_PATH")
    if [ -z "$line" ]; then
        echo "0 0 0"
        return
    fi
    regs=$(echo "$line" | sed -E 's/.*regs=([0-9]+).*/\1/')
    lmem=$(echo "$line" | sed -E 's/.*lmem=([0-9]+).*/\1/')
    smem=$(echo "$line" | sed -E 's/.*smem=([0-9]+).*/\1/')
    echo "$regs $lmem $smem"
}

# -------- MULTI-KERNEL: collect per-kernel info and global maxima --------
collect_kernels_info() {
    # Output two sections via stdout separated by a blank line:
    # 1) Per-kernel lines: KERNEL\t<idx>\t<name>\t<regs>\t<lmem_bytes>\t<smem_bytes>\t<shader_ids_space_separated>
    # 2) One summary line: SUMMARY\t<max_regs>\t<max_lmem_bytes>\t<max_smem_bytes>\t<all_shader_ids_union>
    declare -A REG_MAP LMEM_MAP SMEM_MAP KSHADERS SEEN_SHADER_K SEEN_SHADER_G
    declare -A KID_NAME KID_CYCLES KID_SHADERS SEEN_SHADER_KID
    kernels_order=()
    kids_order=()

    # kernel resource lines
    while IFS= read -r line; do
        kname=$(echo "$line" | sed -E "s/.*Kernel '([^']+)'.*/\1/")
        regs=$(echo "$line" | sed -E 's/.*regs=([0-9]+).*/\1/')
        lmem=$(echo "$line" | sed -E 's/.*lmem=([0-9]+).*/\1/')
        smem=$(echo "$line" | sed -E 's/.*smem=([0-9]+).*/\1/')
        if [[ -n "$kname" ]]; then
            if [[ -z ${REG_MAP[$kname]} || ${REG_MAP[$kname]} -lt $regs ]]; then REG_MAP[$kname]=$regs; fi
            if [[ -z ${LMEM_MAP[$kname]} || ${LMEM_MAP[$kname]} -lt $lmem ]]; then LMEM_MAP[$kname]=$lmem; fi
            if [[ -z ${SMEM_MAP[$kname]} || ${SMEM_MAP[$kname]} -lt $smem ]]; then SMEM_MAP[$kname]=$smem; fi
            # record order once
            if [[ ! " ${kernels_order[*]} " =~ " $kname " ]]; then kernels_order+=("$kname"); fi
        fi
    done < <(grep -E "GPGPU-Sim PTX: Kernel '.*' : regs=[0-9]+, lmem=[0-9]+, smem=[0-9]+" "$FILE_PATH")

    # shader binding lines, also map kernel id -> name
    while IFS= read -r line; do
        parsed=$(echo "$line" | sed -E "s/.*Shader ([0-9]+) bind to kernel ([0-9]+) '([^']+)'.*/\1\t\2\t\3/")
        sid=$(echo "$parsed" | cut -f1)
        kid=$(echo "$parsed" | cut -f2)
        kname=$(echo "$parsed" | cut -f3-)
        if [[ -n "$kname" ]]; then
            key="$kname|$sid"
            if [[ -z ${SEEN_SHADER_K[$key]} ]]; then
                if [[ -z ${KSHADERS[$kname]} ]]; then KSHADERS[$kname]="$sid"; else KSHADERS[$kname]="${KSHADERS[$kname]} $sid"; fi
                SEEN_SHADER_K[$key]=1
            fi
            # per-invocation shader list
            key_kid="$kid|$sid"
            if [[ -z ${SEEN_SHADER_KID[$key_kid]} ]]; then
                if [[ -z ${KID_SHADERS[$kid]} ]]; then KID_SHADERS[$kid]="$sid"; else KID_SHADERS[$kid]="${KID_SHADERS[$kid]} $sid"; fi
                SEEN_SHADER_KID[$key_kid]=1
            fi
            if [[ -z ${SEEN_SHADER_G[$sid]} ]]; then
                SHADER_UNION+=("$sid")
                SEEN_SHADER_G[$sid]=1
            fi
            if [[ -z ${KID_NAME[$kid]} ]]; then KID_NAME[$kid]="$kname"; kids_order+=("$kid"); fi
            if [[ ! " ${kernels_order[*]} " =~ " $kname " ]]; then kernels_order+=("$kname"); fi
        fi
    done < <(grep -E "GPGPU-Sim uArch: Shader [0-9]+ bind to kernel [0-9]+ '.*'" "$FILE_PATH")

    # per-invocation cycles: scan sequentially to associate the first gpu_sim_cycle after each kernel_launch_uid
    pending_kid=""
    while IFS= read -r line; do
        if [[ $line =~ ^kernel_launch_uid[[:space:]]*=[[:space:]]*([0-9]+) ]]; then
            pending_kid="${BASH_REMATCH[1]}"
            continue
        fi
        if [[ -n "$pending_kid" && $line =~ ^gpu_sim_cycle[[:space:]]*=[[:space:]]*([0-9]+) ]]; then
            # only set once per pending kid
            if [[ -z ${KID_CYCLES[$pending_kid]} ]]; then
                KID_CYCLES[$pending_kid]="${BASH_REMATCH[1]}"
            fi
            pending_kid=""
            continue
        fi
    done < "$FILE_PATH"

    # print per-kernel lines
    idx=0
    for k in "${kernels_order[@]}"; do
        [[ -z "$k" ]] && continue
        ((idx++))
        # cycles per kernel (sum of its kids)
        sumc=0
        for kid in "${kids_order[@]}"; do
            [[ -z "$kid" ]] && continue
            [[ "${KID_NAME[$kid]}" != "$k" ]] && continue
            (( sumc += ${KID_CYCLES[$kid]:-0} ))
        done
        printf "KERNEL\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$idx" "$k" "${REG_MAP[$k]:-0}" "${LMEM_MAP[$k]:-0}" "${SMEM_MAP[$k]:-0}" "${KSHADERS[$k]}" "$sumc"
    done

    # summary
    max_regs=0; max_lmem=0; max_smem=0
    for k in "${kernels_order[@]}"; do
        (( ${REG_MAP[$k]:-0} > max_regs )) && max_regs=${REG_MAP[$k]:-0}
        (( ${LMEM_MAP[$k]:-0} > max_lmem )) && max_lmem=${LMEM_MAP[$k]:-0}
        (( ${SMEM_MAP[$k]:-0} > max_smem )) && max_smem=${SMEM_MAP[$k]:-0}
    done
    # sort union shader ids numerically
    all_shaders=$(printf "%s\n" "${SHADER_UNION[@]}" | sort -n | uniq | paste -sd' ' -)
    printf "\nSUMMARY\t%s\t%s\t%s\t%s\n" "$max_regs" "$max_lmem" "$max_smem" "$all_shaders"

    # print per-invocation lines
    if ((${#kids_order[@]} > 0)); then
        printf "\nKERNEL_INVOCATIONS:\n"
        # sort kid order numerically
        sorted_kids=$(printf "%s\n" "${kids_order[@]}" | sort -n | uniq)
        while IFS= read -r kid; do
            [[ -z "$kid" ]] && continue
            kname="${KID_NAME[$kid]}"
            printf -- "- id=%s name=%s regs=%s lmem=%s smem=%s shader_used=%s cycles=%s\n" \
                "$kid" "${kname}" "${REG_MAP[$kname]:-0}" "${LMEM_MAP[$kname]:-0}" "${SMEM_MAP[$kname]:-0}" \
                "${KID_SHADERS[$kid]}" "${KID_CYCLES[$kid]:-0}"
        done <<< "$sorted_kids"
    fi
}

# -------- SHADER_USED (space-separated list of shader IDs) --------
get_shader_used_list() {
    grep -oP 'Shader\s+\K\d+' "$FILE_PATH" | sort -n | uniq | paste -sd' ' -
}

# -------- DATATYPE_SIZE (bits) --------
get_datatype_bits() {
    # 优先取 32 位（常见 int/float），否则按 64/16/8 的存在性依次回退
    if grep -qE '\.(u|s|f|b)32\b' "$FILE_PATH"; then
        echo 32
        return
    fi
    if grep -qE '\.(u|s|f|b)64\b' "$FILE_PATH"; then
        echo 64
        return
    fi
    if grep -qE '\.(u|s|f|b)16\b' "$FILE_PATH"; then
        echo 16
        return
    fi
    if grep -qE '\.(u|s|f|b)8\b' "$FILE_PATH"; then
        echo 8
        return
    fi
    echo 32
}

get_metrics() {
    local cycles regs lmem smem lmem_bits smem_bits shader_used_list dtype_bits
    local max_regs max_lmem max_smem all_shaders

    cycles=$(get_cycles)
    dtype_bits=$(get_datatype_bits)

    # Multi-kernel aware collection
    mapfile -t kernel_lines < <(collect_kernels_info)

    # Defaults in case nothing is found
    max_regs=0; max_lmem=0; max_smem=0; all_shaders=""

    # Parse summary and print per-kernel details
    echo "KERNELS:"
    for line in "${kernel_lines[@]}"; do
        [[ -z "$line" ]] && continue
        if [[ "$line" == SUMMARY* ]]; then
            # SUMMARY\t<max_regs>\t<max_lmem>\t<max_smem>\t<all_shaders>
            IFS=$'\t' read -r _ max_regs max_lmem max_smem all_shaders <<< "$line"
        elif [[ "$line" == KERNEL* ]]; then
            # KERNEL\t<idx>\t<name>\t<regs>\t<lmem>\t<smem>\t<shader_list>\t<cycles_sum>
            IFS=$'\t' read -r _ kidx kname kregs klmem ksmem kshaders kcycles <<< "$line"
            [[ -z "$kname" ]] && continue
            echo "- name=${kname} regs=${kregs} lmem=${klmem} smem=${ksmem} shader_used=${kshaders} cycles=${kcycles}"
        fi
    done

    # Echo kernel invocation details if present
    for line in "${kernel_lines[@]}"; do
        if [[ "$line" == KERNEL_INVOCATIONS:* ]] || [[ "$line" == -\ id=* ]]; then
            echo "$line"
        fi
    done

    # Fallback to single-triplet if no kernels parsed
    if [[ $max_regs -eq 0 && $max_lmem -eq 0 && $max_smem -eq 0 ]]; then
        read -r regs lmem smem < <(get_kernel_triplet)
        max_regs=$regs; max_lmem=$lmem; max_smem=$smem
        all_shaders=$(get_shader_used_list)
    fi

    lmem_bits=$((max_lmem * 8))
    smem_bits=$((max_smem * 8))

    echo
    # Store into global variables (keep existing prints unchanged)
    GLOBAL_CYCLES="${cycles}"
    GLOBAL_MAX_REGISTERS_USED="${max_regs}"
    GLOBAL_SHADER_USED="${all_shaders}"
    GLOBAL_DATATYPE_SIZE="${dtype_bits}"
    
    if [[ "${lmem_bits}" -eq 0 ]]; then
        GLOBAL_LMEM_SIZE_BITS="1"
    else
        GLOBAL_LMEM_SIZE_BITS="${lmem_bits}"
    fi

    if [[ "${smem_bits}" -eq 0 ]]; then
        GLOBAL_SMEM_SIZE_BITS="1"
    else
        GLOBAL_SMEM_SIZE_BITS="${smem_bits}"
    fi
    echo "CYCLES: ${cycles}"
    echo "MAX_REGISTERS_USED: ${max_regs}"
    echo "SHADER_USED: ${all_shaders}"
    echo "DATATYPE_SIZE: ${dtype_bits}"
    echo "LMEM_SIZE_BITS: ${lmem_bits}"
    echo "SMEM_SIZE_BITS: ${smem_bits}"

}

main() {

    # load environment variables
    source setup_environment

    make clean

    # compile project
    make -j$(nproc)

    # 删除test_apps/${TEST_APP_NAME}/result下的所有文件
    rm -rf test_apps/${TEST_APP_NAME}/result/*


    # 生成result
    idx=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        echo "$idx: $line"
    if [[ "$idx" == "1" || "$idx" == "3" || "$idx" == "5" ]]; then
        cu_file="test_apps/${TEST_APP_NAME}/result_gen/${TEST_APP_NAME}_0.cu"
        if [[ -f "$cu_file" ]]; then
            filename=$(basename "$cu_file")
            x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
            if [[ -n "$x_val" ]]; then
                cp "$cu_file" "${cu_file}.bak"
                /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/result_gen/gen
                # 将输出重定向到目录下的txt文件
                ./test_apps/${TEST_APP_NAME}/result_gen/gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                rm -rf test_apps/${TEST_APP_NAME}/result_gen/gen
                mv "${cu_file}.bak" "$cu_file"
            fi
        fi
    else
        for cu_file in test_apps/${TEST_APP_NAME}/result_gen/${TEST_APP_NAME}_*.cu; do
            filename=$(basename "$cu_file")
            x_val=$(echo "$filename" | sed -n "s/^${TEST_APP_NAME}_\([0-9]\+\)\.cu$/\1/p")
            if [[ -z "$x_val" ]]; then
                continue
            fi

            cp "$cu_file" "${cu_file}.bak"
            /usr/local/cuda/bin/nvcc "$cu_file" -o test_apps/${TEST_APP_NAME}/result_gen/gen
            # 将输出重定向到目录下的txt文件
            ./test_apps/${TEST_APP_NAME}/result_gen/gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
            rm -rf test_apps/${TEST_APP_NAME}/result_gen/gen
            mv "${cu_file}.bak" "$cu_file"
        done
    fi
        idx=$((idx+1))
    done < test_apps/${TEST_APP_NAME}/size_list.txt



    FILE_PATH="${1:-./logs1/tmp.out1}"

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

        bash campaign_profile.sh
        if [ ! -f "$FILE_PATH" ]; then
            echo "Error: file not found: $FILE_PATH" >&2
        exit 1
        fi
        get_metrics
        # 读取size_list.txt的第a行（a从0开始）
        size_list_file="test_apps/${TEST_APP_NAME}/size_list.txt"
        if [[ ! -f "$size_list_file" ]]; then
            echo "Error: 未找到size_list.txt: $size_list_file" >&2
            exit 1
        fi

        # a变量已由上文提取
        size_line=$(awk "NR==$((a+1))" "$size_list_file")
        echo "第${a}行内容: $size_line"
        # 读取campaign.sh内容到变量
        campaign_file="campaign_exec.sh"
        if [[ ! -f "$campaign_file" ]]; then
            echo "Error: 未找到campaign_exec.sh: $campaign_file" >&2
            exit 1
        fi
        bash generate_cycles.sh $GLOBAL_CYCLES $GLOBAL_CYCLES

        # 生成新的内容
        awk -v test_app_name="$TEST_APP_NAME" -v size_line="$size_line" \
            -v global_cycles="$GLOBAL_CYCLES" \
            -v global_max_registers="$GLOBAL_MAX_REGISTERS_USED" \
            -v global_shader="$GLOBAL_SHADER_USED" \
            -v global_datatype_size="$GLOBAL_DATATYPE_SIZE" \
            -v global_lmem_size_bits="$GLOBAL_LMEM_SIZE_BITS" \
            -v global_smem_size_bits="$GLOBAL_SMEM_SIZE_BITS" \
            -v components_to_flip="$COMPONENTS_TO_FLIP" '
        {
            # 替换CUDA_UUT
            if ($0 ~ /^CUDA_UUT=/) {
                print "CUDA_UUT=\"./" test_app_name " " size_line "\""
                next
            }
            # 替换CYCLES
            if ($0 ~ /^CYCLES=/) {
                print "CYCLES=" global_cycles
                next
            }
            # 替换MAX_REGISTERS_USED
            if ($0 ~ /^MAX_REGISTERS_USED=/) {
                print "MAX_REGISTERS_USED=" global_max_registers
                next
            }
            # 替换SHADER_USED
            if ($0 ~ /^SHADER_USED=/) {
                print "SHADER_USED=" global_shader
                next
            }
            # 替换DATATYPE_SIZE
            if ($0 ~ /^DATATYPE_SIZE=/) {
                print "DATATYPE_SIZE=" global_datatype_size
                next
            }
            # 替换LMEM_SIZE_BITS
            if ($0 ~ /^LMEM_SIZE_BITS=/) {
                print "LMEM_SIZE_BITS=" global_lmem_size_bits
                next
            }
            # 替换SMEM_SIZE_BITS
            if ($0 ~ /^SMEM_SIZE_BITS=/) {
                print "SMEM_SIZE_BITS=" global_smem_size_bits
                next
            }
            # 替换components_to_flip
            if ($0 ~ /^components_to_flip=/) {
                print "components_to_flip=" components_to_flip
                next
            }
            print $0
        }' "$campaign_file" > "${campaign_file}.tmp" && mv "${campaign_file}.tmp" "$campaign_file"


        bash campaign_exec.sh > inst_exec.log
        # python3 parse_exec.py > parse_exec.log
        # # 如果$filename以.txt结尾，先去掉再传入
        filename_no_ext="${filename%.txt}"
        python3 analysis_fault.py -a $TEST_APP_NAME -t $filename_no_ext -c $COMPONENTS_TO_FLIP

        # 删除 inst_exec.log 文件
        # rm -f inst_exec.log

        
    done
}

# Run main 7 times with COMPONENTS_TO_FLIP = 0..6
for COMPONENTS_TO_FLIP in 0 1 2 3 4 5 6; do
    echo "=== Running main with COMPONENTS_TO_FLIP=${COMPONENTS_TO_FLIP} ==="
    main "$@"
done
