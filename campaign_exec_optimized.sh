#!/bin/bash

# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
# needed by gpgpu-sim for real register usage on PTXPlus mode
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
CONFIG_FILE=./gpgpusim.config
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
RUNS=64
BATCH=$(( $(grep -c ^processor /proc/cpuinfo) - 1 )) # -1 core for computer not to hang
DELETE_LOGS=1 # if 1 then all logs will be deleted at the end of the script
# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------
# L1 cache size per SIMT core (30 SIMT cores on RTX 2060, 30 clusters with 1 core each)
L1D_SIZE_BITS=524345  # nsets=1, line_size=128 bytes + 57 bits, assoc=512 (64 KB per core)
L1C_SIZE_BITS=524345 # nsets=128, line_size=64 bytes + 57 bits, assoc=8 (64 KB per core)
L1T_SIZE_BITS=1048633 # nsets=4, line_size=128 bytes + 57 bits, assoc=256 (128 KB per core)
# L2 cache total size from all sub partitions (RTX 2060: 12 memory controllers × 2 sub partitions = 24 total)
L2_SIZE_BITS=24576057 # (nsets=64, line_size=128 bytes + 57 bits, assoc=16) x 24 sub partitions = 3 MB total
# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS (+profile=1) ----------------------------------------------
CUDA_UUT="./pathfinder 640 24 23"
# total cycles for all kernels
CYCLES=4040
# Get the exact cycles, max registers and SIMT cores used for each kernel with profile=1 
# fix cycles.txt with kernel execution cycles
# (e.g. seq 1 10 >> cycles.txt, or multiple seq commands if a kernel has multiple executions)
# use the following command from profiling execution for easier creation of cycles.txt file
# e.g. grep "_Z12lud_diagonalPfii" cycles.in | awk  '{ system("seq " $12 " " $18 ">> cycles.txt")}'
CYCLES_FILE=./cycles.txt
MAX_REGISTERS_USED=15
SHADER_USED=0
SUCCESS_MSG='Success'
FAILED_MSG='Failed'
TIMEOUT_VAL=10s
DATATYPE_SIZE=32
# lmem and smem values are taken from gpgpu-sim ptx output per kernel
# e.g. GPGPU-Sim PTX: Kernel '_Z9vectorAddPKdS0_Pdi' : regs=8, lmem=0, smem=0, cmem=380
# if 0 put a random value > 0
LMEM_SIZE_BITS=1
SMEM_SIZE_BITS=16384
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS (+profile=1) ------------------------------------------------

FAULT_INJECTION_OCCURRED="Fault injection"
CYCLES_MSG="gpu_tot_sim_cycle ="

masked=0
performance=0
SDC=0
crashes=0

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS (profile=0) ----------------------------------------------
# 0: perform injection campaign, 1: get cycles of each kernel, 2: get mean value of active threads, during all cycles in CYCLES_FILE, per SM,
# 3: single fault-free execution
profile=0
# 0:RF, 1:local_mem, 2:shared_mem, 3:L1D_cache, 4:L1C_cache, 5:L1T_cache, 6:L2_cache (e.g. components_to_flip=0:1 for both RF and local_mem)
components_to_flip=0
# 1: per warp bit flip, 0: per thread bit flip
per_warp=0
# in which kernels to inject the fault. e.g. 0: for all running kernels, 1: for kernel 1, 1:2 for kernel 1 & 2 
kernel_n=0
# in how many blocks (smems) to inject the bit flip
blocks=1

# 新增：创建临时配置文件的函数
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

gather_results() {
    for file in ${TMP_DIR}${1}/${TMP_FILE}*; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        grep -iq "${SUCCESS_MSG}" $file; success_msg_grep=$(echo $?)
        grep -i "${CYCLES_MSG}" $file | tail -1 | grep -q "${CYCLES}"; cycles_grep=$(echo $?)
        grep -iq "${FAILED_MSG}" $file; failed_msg_grep=$(echo $?)
        
        if grep -q "FI_VALIDATION:" "$file"; then
            echo "[Run ${1}] Effects from ${file}:"
            grep -h "FI_VALIDATION:" "$file"
        fi
        
        result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
        filename=$(basename "$file")
        
        case $result in
        "001")
            let RUNS--
            let masked++
            echo "[Run ${1}] ${filename}: Masked (no performance impact)" ;;
        "011")
            let RUNS--
            let masked++ 
            let performance++
            echo "[Run ${1}] ${filename}: Masked (with performance impact)" ;;
        "100" | "110")
            let RUNS--
            let SDC++
            echo "[Run ${1}] ${filename}: SDC" ;;
        *)
            grep -iq "${FAULT_INJECTION_OCCURRED}" $file
            if [ $? -eq 0 ]; then
                let RUNS--
                let crashes++
                echo "[Run ${1}] ${filename}: DUE (Crash)"
            else
                echo "[Run ${1}] ${filename}: Unclassified (${result})"
            fi ;;
        esac
    done
}

# 优化后的并行执行函数
parallel_execution() {
    local batch=$1
    local run_id=$2
    local pids=()
    
    mkdir -p ${TMP_DIR}${run_id}
    
    echo "Starting batch ${run_id} with ${batch} parallel executions..."
    
    for i in $(seq 1 $batch); do
        # 为每个执行创建独立的配置文件
        local temp_config=$(create_temp_config $run_id $i)
        
        # 启动进程并记录PID
        timeout ${TIMEOUT_VAL} $CUDA_UUT -config ${temp_config} > ${TMP_DIR}${run_id}/${TMP_FILE}${i} 2>&1 &
        local pid=$!
        pids+=($pid)
        
        echo "Started process ${i} with PID ${pid}"
    done
    
    # 等待所有进程完成，使用超时机制
    local timeout_counter=0
    local max_timeout=300  # 5分钟超时
    
    while [[ ${#pids[@]} -gt 0 ]] && [[ $timeout_counter -lt $max_timeout ]]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 ${pids[$i]} 2>/dev/null; then
                echo "Process ${pids[$i]} completed"
                unset pids[$i]
            fi
        done
        
        # 重新索引数组
        pids=("${pids[@]}")
        
        if [[ ${#pids[@]} -gt 0 ]]; then
            sleep 1
            let timeout_counter++
            if [[ $((timeout_counter % 30)) -eq 0 ]]; then
                echo "Waiting for ${#pids[@]} processes to complete... (${timeout_counter}s elapsed)"
            fi
        fi
    done
    
    # 如果还有进程未完成，强制终止
    if [[ ${#pids[@]} -gt 0 ]]; then
        echo "Force killing remaining processes: ${pids[@]}"
        for pid in "${pids[@]}"; do
            kill -9 $pid 2>/dev/null || true
        done
    fi
    
    echo "Batch ${run_id} completed"
    gather_results $run_id
    
    # 清理
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -f _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
        rm -rf ${TMP_DIR}${run_id} > /dev/null 2>&1
    fi
    
    if [[ "$profile" -ne 1 ]]; then
        rm -f _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
    fi
}

main() {
    # 清理旧的日志目录
    find . -type d -name "logs*" -exec rm -rf {} + 2>/dev/null || true
    
    if [[ "$profile" -eq 1 ]] || [[ "$profile" -eq 2 ]] || [[ "$profile" -eq 3 ]]; then
        RUNS=1
    fi
    
    # 增加重试机制
    MAX_RETRIES=3
    LOOP=1
    mkdir -p ${CACHE_LOGS_DIR}
    
    echo "Starting execution with RUNS=${RUNS}, BATCH=${BATCH}"
    
    while [[ $RUNS -gt 0 ]] && [[ $MAX_RETRIES -gt 0 ]]; do
        echo "Runs left: ${RUNS}, Retries left: ${MAX_RETRIES}"
        let MAX_RETRIES--
        
        LOOP_START=${LOOP}
        unset LAST_BATCH
        
        if [ "$BATCH" -gt "$RUNS" ]; then
            BATCH=${RUNS}
            LOOP_END=$(($LOOP_START))
        else
            BATCH_RUNS=$(($RUNS/$BATCH))
            if (( $RUNS % $BATCH )); then
                LAST_BATCH=$(($RUNS-$BATCH_RUNS*$BATCH))
            fi
            LOOP_END=$(($LOOP_START+$BATCH_RUNS-1))
        fi

        for i in $(seq $LOOP_START $LOOP_END); do
            echo "Starting loop ${i}"
            parallel_execution $BATCH $i
            let LOOP++
        done

        if [[ ! -z ${LAST_BATCH+x} ]]; then
            echo "Executing last batch with ${LAST_BATCH} runs"
            parallel_execution $LAST_BATCH $LOOP
            let LOOP++
        fi
    done

    if [[ $MAX_RETRIES -eq 0 ]]; then
        echo "Maximum retries reached. Please check your configuration!"
    else
        echo "Execution completed successfully!"
        echo "Masked: ${masked} (performance = ${performance})"
        echo "SDCs: ${SDC}"
        echo "DUEs: ${crashes}"
    fi
    
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -rf ${CACHE_LOGS_DIR} > /dev/null 2>&1
    fi
}

main "$@"
exit 0
