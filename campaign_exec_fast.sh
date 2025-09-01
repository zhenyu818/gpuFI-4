#!/bin/bash

# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
CONFIG_FILE=./gpgpusim.config
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
RUNS=64
BATCH=$(( $(grep -c ^processor /proc/cpuinfo) - 1 ))
DELETE_LOGS=1
# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------
L1D_SIZE_BITS=524345
L1C_SIZE_BITS=524345
L1T_SIZE_BITS=1048633
L2_SIZE_BITS=24576057
# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS ----------------------------------------------
CUDA_UUT="./pathfinder 640 24 23"
CYCLES=4040
CYCLES_FILE=./cycles.txt
MAX_REGISTERS_USED=15
SHADER_USED=0
SUCCESS_MSG='Success'
FAILED_MSG='Failed'
TIMEOUT_VAL=10s
DATATYPE_SIZE=32
LMEM_SIZE_BITS=1
SMEM_SIZE_BITS=16384
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS ------------------------------------------------

FAULT_INJECTION_OCCURRED="Fault injection"
CYCLES_MSG="gpu_tot_sim_cycle ="

masked=0
performance=0
SDC=0
crashes=0

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS ----------------------------------------------
profile=0
components_to_flip=0
per_warp=0
kernel_n=0
blocks=1
# ---------------------------------------------- END PER INJECTION CAMPAIGN PARAMETERS ----------------------------------------------

# 全局变量用于跟踪进程
declare -A running_processes
declare -A process_configs
declare -A process_outputs

# 清理函数
cleanup() {
    echo "Cleaning up processes..."
    for pid in "${!running_processes[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null || true
        fi
    done
    exit 1
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 创建临时配置文件
create_temp_config() {
    local run_id=$1
    local batch_id=$2
    local temp_config="${TMP_DIR}${run_id}/config_${batch_id}.config"
    
    # 复制原始配置文件
    cp ${CONFIG_FILE} ${temp_config}
    
    # 生成随机参数
    local thread_rand=$(shuf -i 0-6000 -n 1)
    local warp_rand=$(shuf -i 0-6000 -n 1)
    local total_cycle_rand
    if [[ "$profile" -eq 3 ]]; then
        total_cycle_rand=-1
    else
        total_cycle_rand="$(shuf ${CYCLES_FILE} -n 1)"
    fi
    
    local register_rand_n="$(shuf -i 1-${MAX_REGISTERS_USED} -n 1)"
    local reg_bitflip_rand_n="$(shuf -i 1-${DATATYPE_SIZE} -n 1)"
    local local_mem_bitflip_rand_n="$(shuf -i 1-${LMEM_SIZE_BITS} -n 3)"
    local block_rand=$(shuf -i 0-6000 -n 1)
    local shared_mem_bitflip_rand_n="$(shuf -i 1-${SMEM_SIZE_BITS} -n 1)"
    local l1d_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"
    local l1d_cache_bitflip_rand_n="$(shuf -i 1-${L1D_SIZE_BITS} -n 1)"
    local l1c_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"
    local l1c_cache_bitflip_rand_n="$(shuf -i 1-${L1C_SIZE_BITS} -n 1)"
    local l1t_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"
    local l1t_cache_bitflip_rand_n="$(shuf -i 1-${L1T_SIZE_BITS} -n 1)"
    local l2_cache_bitflip_rand_n="$(shuf -i 1-${L2_SIZE_BITS} -n 1)"
    
    # 修改临时配置文件
    sed -i -e "s/^-components_to_flip.*$/-components_to_flip ${components_to_flip}/" ${temp_config}
    sed -i -e "s/^-profile.*$/-profile ${profile}/" ${temp_config}
    sed -i -e "s/^-last_cycle.*$/-last_cycle ${CYCLES}/" ${temp_config}
    sed -i -e "s/^-thread_rand.*$/-thread_rand ${thread_rand}/" ${temp_config}
    sed -i -e "s/^-warp_rand.*$/-warp_rand ${warp_rand}/" ${temp_config}
    sed -i -e "s/^-total_cycle_rand.*$/-total_cycle_rand ${total_cycle_rand}/" ${temp_config}
    sed -i -e "s/^-register_rand_n.*$/-register_rand_n ${register_rand_n}/" ${temp_config}
    sed -i -e "s/^-reg_bitflip_rand_n.*$/-reg_bitflip_rand_n ${reg_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-per_warp.*$/-per_warp ${per_warp}/" ${temp_config}
    sed -i -e "s/^-kernel_n.*$/-kernel_n ${kernel_n}/" ${temp_config}
    sed -i -e "s/^-local_mem_bitflip_rand_n.*$/-local_mem_bitflip_rand_n ${local_mem_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-block_rand.*$/-block_rand ${block_rand}/" ${temp_config}
    sed -i -e "s/^-block_n.*$/-block_n ${blocks}/" ${temp_config}
    sed -i -e "s/^-shared_mem_bitflip_rand_n.*$/-shared_mem_bitflip_rand_n ${shared_mem_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1d_shader_rand_n.*$/-l1d_shader_rand_n ${l1d_shader_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1d_cache_bitflip_rand_n.*$/-l1d_cache_bitflip_rand_n ${l1d_cache_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1c_shader_rand_n.*$/-l1c_shader_rand_n ${l1c_shader_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1c_cache_bitflip_rand_n.*$/-l1c_cache_bitflip_rand_n ${l1c_cache_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1t_shader_rand_n.*$/-l1t_shader_rand_n ${l1t_shader_rand_n}/" ${temp_config}
    sed -i -e "s/^-l1t_cache_bitflip_rand_n.*$/-l1t_cache_bitflip_rand_n ${l1t_cache_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-l2_cache_bitflip_rand_n.*$/-l2_cache_bitflip_rand_n ${l2_cache_bitflip_rand_n}/" ${temp_config}
    sed -i -e "s/^-run_uid.*$/-run_uid r${run_id}b${batch_id}/" ${temp_config}
    
    echo ${temp_config}
}

# 检查进程状态
check_processes() {
    local completed=0
    for pid in "${!running_processes[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            # 进程已完成
            local config_file="${process_configs[$pid]}"
            local output_file="${process_outputs[$pid]}"
            
            # 处理结果
            process_result "$output_file"
            
            # 清理
            unset running_processes[$pid]
            unset process_configs[$pid]
            unset process_outputs[$pid]
            
            # 删除临时配置文件
            rm -f "$config_file" 2>/dev/null || true
            
            let completed++
        fi
    done
    echo $completed
}

# 处理单个结果
process_result() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        return
    fi
    
    grep -iq "${SUCCESS_MSG}" $file; success_msg_grep=$(echo $?)
    grep -i "${CYCLES_MSG}" $file | tail -1 | grep -q "${CYCLES}"; cycles_grep=$(echo $?)
    grep -iq "${FAILED_MSG}" $file; failed_msg_grep=$(echo $?)
    
    if grep -q "FI_VALIDATION:" "$file"; then
        echo "Effects from ${file}:"
        grep -h "FI_VALIDATION:" "$file"
    fi
    
    result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
    filename=$(basename "$file")
    
    case $result in
    "001")
        let RUNS--
        let masked++
        echo "${filename}: Masked (no performance impact)" ;;
    "011")
        let RUNS--
        let masked++ 
        let performance++
        echo "${filename}: Masked (with performance impact)" ;;
    "100" | "110")
        let RUNS--
        let SDC++
        echo "${filename}: SDC" ;;
    *)
        grep -iq "${FAULT_INJECTION_OCCURRED}" $file
        if [ $? -eq 0 ]; then
            let RUNS--
            let crashes++
            echo "${filename}: DUE (Crash)"
        else
            echo "${filename}: Unclassified (${result})"
        fi ;;
    esac
}

# 启动新进程
start_process() {
    local run_id=$1
    local batch_id=$2
    
    # 创建临时配置文件
    local temp_config=$(create_temp_config $run_id $batch_id)
    local output_file="${TMP_DIR}${run_id}/${TMP_FILE}${batch_id}"
    
    # 启动进程
    timeout ${TIMEOUT_VAL} $CUDA_UUT -config ${temp_config} > "$output_file" 2>&1 &
    local pid=$!
    
    # 记录进程信息
    running_processes[$pid]=1
    process_configs[$pid]=$temp_config
    process_outputs[$pid]=$output_file
    
    echo "Started process ${batch_id} with PID ${pid}"
}

# 主执行循环
main_execution() {
    local current_run=1
    local current_batch=1
    local total_started=0
    
    echo "Starting execution with RUNS=${RUNS}, BATCH=${BATCH}"
    
    # 创建必要的目录
    mkdir -p ${TMP_DIR}${current_run}
    
    while [[ $RUNS -gt 0 ]]; do
        # 检查已完成的进程
        local completed=$(check_processes)
        
        # 启动新进程直到达到批处理大小
        while [[ ${#running_processes[@]} -lt $BATCH ]] && [[ $RUNS -gt 0 ]]; do
            start_process $current_run $current_batch
            let total_started++
            let current_batch++
            
            # 如果当前批次满了，创建新的批次
            if [[ $current_batch -gt $BATCH ]]; then
                let current_run++
                current_batch=1
                mkdir -p ${TMP_DIR}${current_run}
            fi
        done
        
        # 显示进度
        echo "Progress: ${total_started} started, ${#running_processes[@]} running, ${RUNS} remaining"
        
        # 如果没有运行的进程且还有任务要完成，说明有问题
        if [[ ${#running_processes[@]} -eq 0 ]] && [[ $RUNS -gt 0 ]]; then
            echo "Warning: No processes running but RUNS > 0. This might indicate an issue."
            break
        fi
        
        # 等待一段时间再检查
        sleep 2
    done
    
    # 等待所有剩余进程完成
    echo "Waiting for remaining processes to complete..."
    while [[ ${#running_processes[@]} -gt 0 ]]; do
        local completed=$(check_processes)
        if [[ $completed -eq 0 ]]; then
            sleep 5
        fi
    done
    
    echo "All processes completed!"
}

# 主函数
main() {
    # 清理旧的日志目录
    find . -type d -name "logs*" -exec rm -rf {} + 2>/dev/null || true
    
    if [[ "$profile" -eq 1 ]] || [[ "$profile" -eq 2 ]] || [[ "$profile" -eq 3 ]]; then
        RUNS=1
    fi
    
    mkdir -p ${CACHE_LOGS_DIR}
    
    # 开始执行
    main_execution
    
    # 显示最终结果
    echo "Execution completed successfully!"
    echo "Masked: ${masked} (performance = ${performance})"
    echo "SDCs: ${SDC}"
    echo "DUEs: ${crashes}"
    
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -rf ${CACHE_LOGS_DIR} > /dev/null 2>&1
    fi
}

main "$@"
exit 0
