#!/bin/bash

TEST_APP_NAME="RayTracing"
COMPONENT_SET="0"
INJECT_BIT_FLIP_COUNT=1 # number of bits to flip per injection (e.g. 2 means flip 2 bits per injection)
# 0:RF, 1:local_mem, 2:shared_mem, 3:L1D_cache, 4:L1C_cache, 5:L1T_cache, 6:L2_cache
RUN_PER_EPOCH=1000
EPOCH=200


DO_BUILD=1 # 1: build before run, 0: skip build
DO_RESULT_GEN=0 # 1: generate result files, 0: skip result generation



# set cuda installation path
export CUDA_INSTALL_PATH=/usr/local/cuda

# -------- Global metrics storage (script-wide) --------
GLOBAL_CYCLES=""
GLOBAL_MAX_REGISTERS_USED=""
GLOBAL_SHADER_USED=""
GLOBAL_DATATYPE_SIZE=""
GLOBAL_LMEM_SIZE_BITS=""
GLOBAL_SMEM_SIZE_BITS=""
GLOBAL_EXEC_TIME=""

cleanup() {
    echo -e "\nInterrupted. Killing campaign_exec.sh (PID=$CMD_PID)..."
    kill $CMD_PID 2>/dev/null
    exit 1
}

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
    echo "=== Collecting kernel information ==="
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
    echo "=== Extracting metrics from logs ==="
    local cycles regs lmem smem lmem_bits smem_bits shader_used_list dtype_bits exec_time
    local max_regs max_lmem max_smem all_shaders

    # Maps for dynamic shared-memory inference (per kernel)
    declare -A K_STATIC_SMEM_BYTES  # from resource line smem= (static only)
    declare -A K_BLOCK_ELEMS        # blockDim.x*blockDim.y*blockDim.z
    declare -A K_SHARED_ELEM_BYTES  # inferred element width from ld/st.shared.* (bytes)
    declare -A K_EFFECTIVE_SMEM     # effective smem bytes per kernel (max(static, dynamic_est))

    cycles=$(get_cycles)
    dtype_bits=$(get_datatype_bits)

    # Multi-kernel aware collection
    mapfile -t kernel_lines < <(collect_kernels_info)

    # Build per-kernel static smem map from resource lines
    while IFS= read -r line; do
        kname=$(echo "$line" | sed -E "s/.*Kernel '([^']+)'.*/\1/")
        ksmem=$(echo "$line" | sed -E 's/.*smem=([0-9]+).*/\1/')
        [[ -n "$kname" ]] && K_STATIC_SMEM_BYTES["$kname"]="$ksmem"
    done < <(grep -E "GPGPU-Sim PTX: Kernel '.*' : regs=[0-9]+, lmem=[0-9]+, smem=[0-9]+" "$FILE_PATH")

    # Parse blockDim per kernel from push lines
    while IFS= read -r line; do
        kname=$(echo "$line" | sed -E "s/.*pushing kernel '([^']+)'.*/\1/")
        bxyz=$(echo "$line" | sed -E "s/.*blockDim = \(([0-9]+),([0-9]+),([0-9]+)\).*/\1 \2 \3/")
        bx=$(echo "$bxyz" | awk '{print $1}')
        by=$(echo "$bxyz" | awk '{print $2}')
        bz=$(echo "$bxyz" | awk '{print $3}')
        if [[ -n "$kname" && -n "$bx" && -n "$by" && -n "$bz" ]]; then
            K_BLOCK_ELEMS["$kname"]=$(( bx * by * bz ))
        fi
    done < <(grep -E "pushing kernel '.*'.*blockDim\s*=\s*\([0-9]+,[0-9]+,[0-9]+\)" "$FILE_PATH")

    # Infer shared element byte width from PTX_INST_SUM lines (ld/st.shared.<type>) per kernel
    while IFS= read -r kname; do
        [[ -z "$kname" ]] && continue
        maxb=0
        while IFS= read -r tline; do
            suf=$(echo "$tline" | sed -nE 's/.*(ld|st)\.shared\.([a-z0-9]+).*/\2/p')
            case "$suf" in
                *64) bytes=8 ;;
                *32) bytes=4 ;;
                *16) bytes=2 ;;
                *8)  bytes=1 ;;
                *)   bytes=0 ;;
            esac
            (( bytes > maxb )) && maxb=$bytes
        done < <(grep -F "kernel=\"$kname\"" "$FILE_PATH" | grep -E "\[PTX_INST_SUM\].*(ld\.shared|st\.shared)\.")
        if (( maxb > 0 )); then
            K_SHARED_ELEM_BYTES["$kname"]=$maxb
        fi
    done < <(grep -oE "kernel='[^']+'|kernel=\"[^\"]+\"" "$FILE_PATH" | sed -E "s/kernel=['\"]([^'\"]+)['\"]/\1/" | sort -u)

    exec_time=$(grep -E "^gpgpu_simulation_time" "$FILE_PATH" | tail -n1)

    if [[ -n "$exec_time" ]]; then
        days=$(echo "$exec_time" | sed -E 's/.*=\s*([0-9]+)\s+days.*/\1/')
        hrs=$(echo "$exec_time"  | sed -E 's/.*days,\s*([0-9]+)\s+hrs.*/\1/')
        mins=$(echo "$exec_time" | sed -E 's/.*hrs,\s*([0-9]+)\s+min.*/\1/')
        secs=$(echo "$exec_time" | sed -E 's/.*min,\s*([0-9]+)\s+sec.*/\1/')

        days=${days:-0}
        hrs=${hrs:-0}
        mins=${mins:-0}
        secs=${secs:-0}

        exec_time=$(( days*86400 + hrs*3600 + mins*60 + secs ))
    else
        exec_time=0
    fi


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

    # Compute effective shared memory (consider dynamic + static) across kernels
    # Start with the static max as baseline
    eff_smem_bytes=${max_smem}
    # Build kernels_order copy from previously collected function by re-parsing names
    # and compute per-kernel effective smem
    while IFS= read -r kname; do
        [[ -z "$kname" ]] && continue
        bs_elems=${K_BLOCK_ELEMS[$kname]:-0}
        elem_bytes=${K_SHARED_ELEM_BYTES[$kname]:-0}
        dyn_bytes=0
        if (( bs_elems > 0 && elem_bytes > 0 )); then
            dyn_bytes=$(( bs_elems * elem_bytes ))
        fi
        static_bytes=${K_STATIC_SMEM_BYTES[$kname]:-0}
        # choose max to avoid double-counting
        (( dyn_bytes > static_bytes )) && chosen=$dyn_bytes || chosen=$static_bytes
        K_EFFECTIVE_SMEM["$kname"]=$chosen
        (( chosen > eff_smem_bytes )) && eff_smem_bytes=$chosen
    done < <(grep -E "GPGPU-Sim PTX: Kernel '.*' : regs=[0-9]+, lmem=[0-9]+, smem=[0-9]+" "$FILE_PATH" | sed -E "s/.*Kernel '([^']+)'.*/\1/" | sort -u)

    smem_bits=$((eff_smem_bytes * 8))

    echo
    # Store into global variables (keep existing prints unchanged)
    GLOBAL_CYCLES="${cycles}"
    GLOBAL_MAX_REGISTERS_USED="${max_regs}"
    GLOBAL_SHADER_USED="${all_shaders}"
    GLOBAL_DATATYPE_SIZE="${dtype_bits}"
    GLOBAL_EXEC_TIME="${exec_time}"
    
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
    echo "EXEC_TIME: ${exec_time}s"

}

main() {

    # pip3 install pathlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    # pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    # pip3 install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

    pip3 install pathlib
    pip3 install numpy
    pip3 install pandas
    pip3 install scipy

    # load environment variables
    source setup_environment

    # Ensure destination directory for classification artifacts exists
    error_classification_dir="error_classification"
    if [[ -d "$error_classification_dir" ]]; then
        rm -rf "$error_classification_dir"
    fi
    mkdir -p error_classification

    inconsistent_dir="inconsistent_with_gpu"
    if [[ -d "$inconsistent_dir" ]]; then
        rm -rf "$inconsistent_dir"
    fi
    mkdir -p "$inconsistent_dir"

    if [[ $DO_BUILD -eq 1 ]]; then
        echo "=== Start compiling ==="

        # 执行 make clean，静默模式
        make clean >/dev/null 2>&1

        # 执行 make，输出保存到 log，不直接打印
        if make -j"$(nproc)" >build.log 2>&1; then
            echo "=== Make success ==="
        else
            echo "=== Build failed, showing errors ==="
            # 只显示包含 error 的行
            grep -i "error" build.log
            exit 1
        fi
    else
        echo "=== Build skipped ==="
    fi


    if [[ $DO_RESULT_GEN -eq 1 ]]; then
        echo "=== Start result generation ==="

        # 删除旧的 result 文件
        rm -rf test_apps/${TEST_APP_NAME}/result/*

        # 生成 result
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
                        nvcc "$cu_file" -o "./gen" -g -lcudart -arch=sm_75 
                        /usr/local/cuda/bin/nvcc "$cu_file" -o "./gpu_gen" -arch=sm_75
                        ./gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                        ./gpu_gen $line >> "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}_gpu.txt"
                        # 仅保留最后一个GPGPU-Sim所在行到倒数第二个GPGPU-Sim所在行之间的内容（不包括GPGPU-Sim所在行）
                        tmpfile="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt.tmp"
                        gpgpu_lines=($(grep -n "GPGPU-Sim" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt" | cut -d: -f1))
                        if (( ${#gpgpu_lines[@]} >= 2 )); then
                            start_line=$(( ${gpgpu_lines[-2]} + 1 ))
                            end_line=$(( ${gpgpu_lines[-1]} - 1 ))
                            if (( start_line <= end_line )); then
                                sed -n "${start_line},${end_line}p" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt" > "$tmpfile"
                                mv "$tmpfile" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                            else
                                # 区间无内容，清空文件
                                > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                            fi
                        fi
                        # 比较CPU和GPU结果文件是否一致
                        cpu_file="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                        gpu_file="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}_gpu.txt"
                        # 比较CPU和GPU结果文件是否一致，并输出不同项的索引和值
                        if ! diff -q "$cpu_file" "$gpu_file" >/dev/null 2>&1; then
                            out_file="${inconsistent_dir}/${TEST_APP_NAME}_${idx}-${x_val}.txt"
                            echo "GPGPU-Sim and GPU results are inconsistent:" > "$out_file"
                            # 将两个文件的内容按空白分割为数组
                            cpu_arr=($(tr -s ' \n' '\n' < "$cpu_file"))
                            gpu_arr=($(tr -s ' \n' '\n' < "$gpu_file"))
                            len_cpu=${#cpu_arr[@]}
                            len_gpu=${#gpu_arr[@]}
                            max_len=$(( len_cpu > len_gpu ? len_cpu : len_gpu ))
                            for ((i=0; i<max_len; i++)); do
                                cpu_val="${cpu_arr[i]}"
                                gpu_val="${gpu_arr[i]}"
                                if [[ "$cpu_val" != "$gpu_val" ]]; then
                                    echo "Index $i: CPU='$cpu_val' GPU='$gpu_val'" >> "$out_file"
                                fi
                            done
                        fi
                        rm -rf "./gen"
                        rm -rf "./gpu_gen"
                        rm -f "$gpu_file"
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
                        nvcc "$cu_file" -o "./gen" -g -lcudart -arch=sm_75 
                        /usr/local/cuda/bin/nvcc "$cu_file" -o "./gpu_gen" -arch=sm_75
                        ./gen $line > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                        ./gpu_gen $line >> "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}_gpu.txt"
                        # 仅保留最后一个GPGPU-Sim所在行到倒数第二个GPGPU-Sim所在行之间的内容（不包括GPGPU-Sim所在行）
                        tmpfile="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt.tmp"
                        gpgpu_lines=($(grep -n "GPGPU-Sim" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt" | cut -d: -f1))
                        if (( ${#gpgpu_lines[@]} >= 2 )); then
                            start_line=$(( ${gpgpu_lines[-2]} + 1 ))
                            end_line=$(( ${gpgpu_lines[-1]} - 1 ))
                            if (( start_line <= end_line )); then
                                sed -n "${start_line},${end_line}p" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt" > "$tmpfile"
                                mv "$tmpfile" "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                            else
                                # 区间无内容，清空文件
                                > "test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                            fi
                        fi
                        # 比较CPU和GPU结果文件是否一致
                        cpu_file="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}.txt"
                        gpu_file="test_apps/${TEST_APP_NAME}/result/${idx}-${x_val}_gpu.txt"
                        # 比较CPU和GPU结果文件是否一致，并输出不同项的索引和值
                        if ! diff -q "$cpu_file" "$gpu_file" >/dev/null 2>&1; then
                            out_file="${inconsistent_dir}/${TEST_APP_NAME}_${idx}-${x_val}.txt"
                            echo "GPGPU-Sim and GPU results are inconsistent:" > "$out_file"
                            # 将两个文件的内容按空白分割为数组
                            cpu_arr=($(tr -s ' \n' '\n' < "$cpu_file"))
                            gpu_arr=($(tr -s ' \n' '\n' < "$gpu_file"))
                            len_cpu=${#cpu_arr[@]}
                            len_gpu=${#gpu_arr[@]}
                            max_len=$(( len_cpu > len_gpu ? len_cpu : len_gpu ))
                            for ((i=0; i<max_len; i++)); do
                                cpu_val="${cpu_arr[i]}"
                                gpu_val="${gpu_arr[i]}"
                                if [[ "$cpu_val" != "$gpu_val" ]]; then
                                    echo "Index $i: CPU='$cpu_val' GPU='$gpu_val'" >> "$out_file"
                                fi
                            done
                        fi
                        rm -rf "./gen"
                        rm -rf "./gpu_gen"
                        rm -f "$gpu_file"
                        mv "${cu_file}.bak" "$cu_file"
                done
            fi
            idx=$((idx+1))
        done < "test_apps/${TEST_APP_NAME}/size_list.txt"

        echo "=== Result generation finished ==="
    else
        echo "=== Result generation skipped ==="
    fi

    # register_used.txt will be consumed by campaign_exec.sh per-injection

    FILE_PATH="${1:-./logs1/tmp.out1}"

    for result_file in test_apps/${TEST_APP_NAME}/result/*; do
        rm -f invalid_param_combos.txt
        echo "=== Preparing injection for file: $result_file ==="
        filename=$(basename "$result_file")
        # 提取a和b
        a=$(echo "$filename" | cut -d'-' -f1)
        b_with_ext=$(echo "$filename" | cut -d'-' -f2)
        b=$(echo "$b_with_ext" | cut -d'.' -f1)

        echo "=== Copying result and source files ==="
        # 复制result文件到根目录并重命名为result.txt
        cp "$result_file" ./result.txt

        # 查找inject_app下b对应的cu文件并移动到根目录，重命名为${TEST_APP_NAME}
        cu_file="test_apps/${TEST_APP_NAME}/inject_app/${TEST_APP_NAME}_${b}.cu"
        if [[ -f "$cu_file" ]]; then
            cp "$cu_file" "./${TEST_APP_NAME}.cu"
        fi

        echo "=== Compiling CUDA application for injection ==="
        nvcc ${TEST_APP_NAME}.cu -o ${TEST_APP_NAME} -g -lcudart -arch=sm_75

        # 读取size_list.txt的第a行（a从0开始）
        size_list_file="test_apps/${TEST_APP_NAME}/size_list.txt"
        if [[ ! -f "$size_list_file" ]]; then
            echo "=== Error: size_list.txt not found: $size_list_file ===" >&2
            exit 1
        fi

        # a变量已由上文提取
        size_line=$(awk "NR==$((a+1))" "$size_list_file")

        echo "=== Updating campaign_profile.sh ==="
        FILE="campaign_profile.sh"
        # 使用 sed 替换 CUDA_UUT 开头的行
        sed -i "s|^CUDA_UUT.*|CUDA_UUT=\"./${TEST_APP_NAME} ${size_line}\"|" "$FILE"

        echo "=== Running campaign_profile.sh ==="
        bash campaign_profile.sh

        if [ ! -f "$FILE_PATH" ]; then
            echo "Error: file not found: $FILE_PATH" >&2
        exit 1
        fi
        echo "=== Collecting metrics ==="
        app_info_file="test_apps/${TEST_APP_NAME}/app_info.txt"
        if [[ -f "$app_info_file" ]]; then
            rm -f "$app_info_file"
        fi
        touch "$app_info_file"
        { get_metrics; } > >(tee "test_apps/${TEST_APP_NAME}/app_info.txt")


        # 读取campaign.sh内容到变量
        campaign_file="campaign_exec.sh"
        if [[ ! -f "$campaign_file" ]]; then
            echo "Error: 未找到campaign_exec.sh: $campaign_file" >&2
            exit 1
        fi
        bash generate_cycles.sh $GLOBAL_CYCLES $GLOBAL_CYCLES
        if [ $? -ne 0 ]; then
            echo "Error: generate_cycles.sh failed." >&2
            exit 1
        fi

        echo "=== Updating campaign_exec.sh with metrics ==="
        # 生成新的内容
        awk -v test_app_name="$TEST_APP_NAME" -v size_line="$size_line" \
            -v global_cycles="$GLOBAL_CYCLES" \
            -v global_max_registers="$GLOBAL_MAX_REGISTERS_USED" \
            -v global_shader="$GLOBAL_SHADER_USED" \
            -v global_datatype_size="$GLOBAL_DATATYPE_SIZE" \
            -v global_lmem_size_bits="$GLOBAL_LMEM_SIZE_BITS" \
            -v global_smem_size_bits="$GLOBAL_SMEM_SIZE_BITS" \
            -v run_times="$RUN_PER_EPOCH" \
            -v exec_time="$GLOBAL_EXEC_TIME" \
            -v component_set="$COMPONENT_SET" \
            -v inject_bit_flip_count="$INJECT_BIT_FLIP_COUNT" '
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
            # 替换SHADER_USED（用双引号括住）
            if ($0 ~ /^SHADER_USED=/) {
                print "SHADER_USED=\"" global_shader "\""
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
            # 替换run_times
            if ($0 ~ /^RUNS=/) {
                print "RUNS=" run_times
                next
            }
            # 替换TIMEOUT_VAL
            if ($0 ~ /^TIMEOUT_VAL=/) {
                et = (exec_time * 3) + 10
                print "TIMEOUT_VAL=" et "s"
                next
            }
            # 替换COMPONENT_SET
            if ($0 ~ /^COMPONENT_SET=/) {
                print "COMPONENT_SET=\"" component_set "\""
                next
            }
            # 替换INJECT_BIT_FLIP_COUNT
            if ($0 ~ /^INJECT_BIT_FLIP_COUNT=/) {
                print "INJECT_BIT_FLIP_COUNT=" inject_bit_flip_count
                next
            }
            # 其他行保持不变
            print $0
        }' "$campaign_file" > "${campaign_file}.tmp" && mv "${campaign_file}.tmp" "$campaign_file"

        echo "=== Extracting register information ==="

        nvcc -arch=sm_75 -ptx -g -lineinfo $TEST_APP_NAME.cu -o $TEST_APP_NAME.ptx

        python3 extract_registers.py $TEST_APP_NAME
        python3 count_ptx_inst.py $TEST_APP_NAME >> "test_apps/${TEST_APP_NAME}/app_info.txt"

        echo "=== Starting fault injection experiment: ${TEST_APP_NAME}, file ${filename} ==="
        filename_no_ext="${filename%.txt}"

        for i in $(seq 1 $EPOCH); do
            echo "  --- Injection run $i / $EPOCH ---"
            PARALLEL_TASKS=$(( $(grep -c ^processor /proc/cpuinfo) - 1 ))
            # 实际任务总数
            TOTAL_TASKS="$RUN_PER_EPOCH"   # 总任务数由脚本前面定义的 RUN_PER_EPOCH 决定

            # 后台执行，不在控制台输出日志
            bash campaign_exec.sh > inst_exec.log 2>&1 &
            CMD_PID=$!

            trap cleanup INT

            last_run=0

            # 监控日志，实时更新进度条
            tail -n0 -F inst_exec.log 2>/dev/null | while read -r line; do
                if [[ "$line" =~ ^\[Run[[:space:]]+([0-9]+)\] ]]; then
                    current_run=${BASH_REMATCH[1]}
                    if (( current_run != last_run )); then
                        last_run=$current_run
                        done_tasks=$(( last_run * PARALLEL_TASKS ))
                        (( done_tasks > TOTAL_TASKS )) && done_tasks=$TOTAL_TASKS

                        left=$(( TOTAL_TASKS - done_tasks ))
                        percent=$(( done_tasks * 100 / TOTAL_TASKS ))

                        # 绘制进度条
                        bar_len=50
                        filled=$(( percent * bar_len / 100 ))
                        bar=$(printf "%${filled}s" | tr " " "#")
                        empty=$(printf "%$(( bar_len - filled ))s")

                        printf "\rProgress: [%-50s] %3d%%  runs left %d" "$bar$empty" "$percent" "$left"

                        if (( done_tasks >= TOTAL_TASKS )); then
                            echo
                            break
                        fi
                    fi
                fi
            done

            # 等待主进程结束
            wait $CMD_PID
            echo "=== Fault injection for ${filename} finished ==="
            python3 analysis_fault.py -a $TEST_APP_NAME -t $filename_no_ext  -c $COMPONENT_SET -b $INJECT_BIT_FLIP_COUNT --coverage 0.5
            ret=$?
            if [ $ret -eq 99 ]; then
                echo "=== Early stopping triggered. Exiting loop ==="
                break
            fi
        done
    done
    rm -f register_used.txt
    rm -f $TEST_APP_NAME.ptx
    rm -f $TEST_APP_NAME.1.sm_75.ptx
    rm -f $TEST_APP_NAME.1.sm_75.ptxas
    rm -f $TEST_APP_NAME.cu
    rm -f $TEST_APP_NAME
    rm -f result.txt

}
echo "=== Running main with COMPONENT_SET=${COMPONENT_SET} ==="
echo "=== Component mapping: 0=RF, 1=local_mem, 2=shared_mem, 3=L1D_cache, 4=L1C_cache, 5=L1T_cache, 6=L2_cache ==="
echo "=== Test application: ${TEST_APP_NAME} ==="
echo "=== Injection bit flip count: ${INJECT_BIT_FLIP_COUNT} ==="
main "$@"
# 遍历 test_apps 目录下所有文件夹，依次作为 TEST_APP_NAME 调用 main
# if [[ -d "test_apps" ]]; then
#     for app_dir in test_apps/*; do
#         if [[ -d "$app_dir" ]]; then
#             TEST_APP_NAME=$(basename "$app_dir")
#             echo "=== Running main for TEST_APP_NAME=${TEST_APP_NAME} ==="
#             main "$@"
#         fi
#     done
# else
#     echo "Error: test_apps directory not found."
#     exit 1
# fi
