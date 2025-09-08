// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"
#include <sys/time.h>

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"

#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "GPUWattch XML file",
                         "gpuwattch.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 8)", "8");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_UINT32,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_INT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_INT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_INT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_INT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-run_uid", OPT_CSTR, &run_uid,
                         "TODO", "0");
  option_parser_register(opp, "-profile", OPT_INT32, &profile,
                         "TODO", "0");
  option_parser_register(opp, "-last_cycle", OPT_INT32, &last_cycle,
                        "TODO", "0");
  option_parser_register(opp, "-component_to_flip", OPT_INT32, &component_to_flip,
                         "TODO", "0");
  option_parser_register(opp, "-thread_rand", OPT_INT32, &thread_rand,
                         "TODO", "0");
  option_parser_register(opp, "-warp_rand", OPT_INT32, &warp_rand,
                         "TODO", "0");
  option_parser_register(opp, "-total_cycle_rand", OPT_INT32, &total_cycle_rand,
                         "TODO", "0");
  option_parser_register(opp, "-register_rand_n", OPT_CSTR, &register_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-reg_bitflip_rand_n", OPT_CSTR, &reg_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-per_warp", OPT_BOOL, &per_warp,
                         "TODO", "0");
  option_parser_register(opp, "-kernel_n", OPT_CSTR, &kernel_n,
                         "TODO", "0");
  option_parser_register(opp, "-local_mem_bitflip_rand_n", OPT_CSTR, &local_mem_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-components_to_flip", OPT_CSTR, &components_to_flip,
                         "TODO", "0");
  option_parser_register(opp, "-block_n", OPT_INT32, &block_n,
                         "TODO", "0");
  option_parser_register(opp, "-shared_mem_bitflip_rand_n", OPT_CSTR, &shared_mem_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-block_rand", OPT_INT32, &block_rand,
                         "TODO", "0");
  option_parser_register(opp, "-l1d_shader_rand_n", OPT_CSTR, &l1d_shader_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l1d_cache_bitflip_rand_n", OPT_CSTR, &l1d_cache_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l1c_shader_rand_n", OPT_CSTR, &l1c_shader_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l1c_cache_bitflip_rand_n", OPT_CSTR, &l1c_cache_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l1t_shader_rand_n", OPT_CSTR, &l1t_shader_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l1t_cache_bitflip_rand_n", OPT_CSTR, &l1t_cache_bitflip_rand_n,
                         "TODO", "0");
  option_parser_register(opp, "-l2_cache_bitflip_rand_n", OPT_CSTR, &l2_cache_bitflip_rand_n,
                         "TODO", "0");

  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU", "8");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,
                                             config.g_power_config_name);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;

  l2_enabled = false;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

void gpgpu_sim::print_stats() {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat();

  // Print aggregated active-thread counts per static PTX instruction
  if (!gpgpu_ctx->ptx_pc_active_agg.empty()) {
    printf("\n[PTX_INST_SUM] Aggregated active threads per PTX instruction:\n");
    for (std::map<unsigned, unsigned long long>::const_iterator it =
             gpgpu_ctx->ptx_pc_active_agg.begin();
         it != gpgpu_ctx->ptx_pc_active_agg.end(); ++it) {
      unsigned pc = it->first;
      unsigned long long total_active = it->second;
      const ptx_instruction *pi = gpgpu_ctx->pc_to_instruction(pc);
      unsigned uid = pi ? pi->uid() : 0u;
      // kernel/function name
      std::string kname = "<unknown>";
      {
        std::map<unsigned, function_info *>::iterator f =
            gpgpu_ctx->func_sim->g_pc_to_finfo.find(pc);
        if (f != gpgpu_ctx->func_sim->g_pc_to_finfo.end() && f->second) {
          kname = f->second->get_name();
        }
      }
      std::string insn_str = gpgpu_ctx->func_sim->ptx_get_insn_str(pc);
      printf(
          "[PTX_INST_SUM] id=%u pc=%u kernel=\"%s\" text=\"%s\" total_active=%llu\n",
          uid, pc, kname.c_str(), insn_str.c_str(), total_active);
    }
  }

  // Print aggregated PTX register usage counts (sorted desc by uses)
  if (!gpgpu_ctx->ptx_reg_use_counts.empty()) {
    printf("\n[PTX_REG_SUM] Aggregated register usage across program (reads+writes):\n");
    std::vector<std::pair<std::string, unsigned long long> > regs;
    regs.reserve(gpgpu_ctx->ptx_reg_use_counts.size());
    for (std::map<std::string, unsigned long long>::const_iterator it =
             gpgpu_ctx->ptx_reg_use_counts.begin();
         it != gpgpu_ctx->ptx_reg_use_counts.end(); ++it) {
      regs.push_back(*it);
    }
    std::sort(regs.begin(), regs.end(),
              [](const std::pair<std::string, unsigned long long> &a,
                 const std::pair<std::string, unsigned long long> &b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
              });
    for (size_t i = 0; i < regs.size(); ++i) {
      printf("[PTX_REG_SUM] reg=\"%s\" uses=%llu\n", regs[i].first.c_str(),
             regs[i].second);
    }
  }

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld)\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle);
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

void read_colon_option(std::vector<unsigned> &result_vector, char *option) {
  char *tmp = strtok(option, ":");
  while (tmp != NULL)
  {
    result_vector.push_back(strtoul(tmp, NULL, 0));
    tmp = strtok (NULL, ":");
  }
}

struct cmp_str
{
   bool operator()(char const *a, char const *b) const
   {
      return std::strcmp(a, b) < 0;
   }
};


tr1_hash_map<unsigned, char*> kernel_name;
tr1_hash_map<unsigned, std::vector<unsigned long long>> kernel_start_end_cycle;
std::map<char *, unsigned, cmp_str> max_active_regs;
std::map<char *, std::set<unsigned>, cmp_str> shaders_used;
unsigned active_threads_sum;
unsigned cycles_txt_lines;
std::vector<unsigned> cycles_txt;


void find_active_kernels_warps(tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> &active_threads_map,
    const shader_core_config *m_shader_config,
    class simt_core_cluster **m_cluster) {
  for (unsigned cluster_idx = 0; cluster_idx < m_shader_config->n_simt_clusters; cluster_idx++) {
    simt_core_cluster *simt_core_cluster = m_cluster[cluster_idx];
    for (unsigned shd_core_idx = 0; shd_core_idx < simt_core_cluster->get_config()->n_simt_cores_per_cluster; shd_core_idx++) {
      shader_core_ctx *shader_core_ctx = (simt_core_cluster->get_core())[shd_core_idx];
      if (shader_core_ctx->get_not_completed()) {
//        printf("shader idx=%u on cluster=%u with warp size=%u\n", shd_core_idx, cluster_idx, shader_core_ctx->get_warp_size());
        kernel_info_t *k = shader_core_ctx->get_kernel();
        if (kernel_name.find(k->get_uid()) == kernel_name.end()) {
          char *kernel_name_cpy = new char [strlen(k->name().c_str()) + 1];
          strcpy(kernel_name_cpy, k->name().c_str());
          kernel_name[k->get_uid()] = kernel_name_cpy;
        }
//        printf("HMMMMM Shader %u bind to kernel %u \'%s\'\n", shader_core_ctx->get_sid(), k->get_uid(), k->name().c_str());
//        std::vector<std::vector<ptx_thread_info*>> warp_threads_vector;
        for (unsigned warp_idx = 0; warp_idx < simt_core_cluster->get_config()->max_warps_per_shader; warp_idx++) {
          shd_warp_t *shd_warp = (shader_core_ctx->get_warp())[warp_idx];
          if (! shd_warp->done_exit()) {
//              printf("warp id =%u\n", shd_warp->get_warp_id());
            unsigned m_warp_id = shd_warp->get_warp_id();
            unsigned m_warp_size = shd_warp->get_warp_size();
//            printf("shader idx=%u on cluster=%u with warp size=%u\n", shd_core_idx, cluster_idx, shd_warp->get_warp_size());
            std::vector<ptx_thread_info*> threads_vector;
            for (unsigned thread_shd_idx = m_warp_id * m_warp_size; thread_shd_idx < (m_warp_id + 1) * m_warp_size; thread_shd_idx++) {
              ptx_thread_info *ptx_thread_info = (shader_core_ctx->get_thread_info())[thread_shd_idx];
              if (ptx_thread_info!=NULL && !ptx_thread_info->is_done()) {
//                printf("m_warp_id=%u\n",m_warp_id);
                threads_vector.push_back(ptx_thread_info);
              }
            }
//            warp_threads_vector.push_back(threads_vector);
            if (threads_vector.size()>0) {
              std::vector<std::vector<ptx_thread_info*>> &temp = active_threads_map[k->get_uid()];
              temp.push_back(threads_vector);
            }
          }
        }
//        active_threads_map[k->get_uid()] = warp_threads_vector;
      }
    }
  }
}

void find_active_threads(std::vector<ptx_thread_info*> &active_threads, tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> &active_kernels_warps, std::vector<unsigned> &kernel_vector) {
  for(tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>>::iterator itK = active_kernels_warps.begin(); itK != active_kernels_warps.end(); ++itK) {
//  printf("Kernel: %u\n", itK->first);
    if (kernel_vector.size()>0 && kernel_vector[0]!=0) {
      if (std::find(kernel_vector.begin(), kernel_vector.end(), itK->first) == kernel_vector.end()) { continue; }
    }
    for(std::vector<std::vector<ptx_thread_info*>>::iterator itW = itK->second.begin(); itW != itK->second.end(); ++itW) {
      for(std::vector<ptx_thread_info*>::iterator itT = itW->begin(); itT != itW->end(); ++itT) {
        active_threads.push_back(*itT);
      }
    }
  }
}

void find_active_warps(std::vector<std::vector<ptx_thread_info*>> &active_warps, tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> &active_kernels_warps, std::vector<unsigned> &kernel_vector) {
  for(tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>>::iterator itK = active_kernels_warps.begin(); itK != active_kernels_warps.end(); ++itK) {
//  printf("Kernel: %u\n", itK->first);
    if (kernel_vector.size()>0 && kernel_vector[0]!=0) {
      if (std::find(kernel_vector.begin(), kernel_vector.end(), itK->first) == kernel_vector.end()) { continue; }
    }
    for(std::vector<std::vector<ptx_thread_info*>>::iterator itW = itK->second.begin(); itW != itK->second.end(); ++itW) {
//    printf("Warp size: %u\n", itW->size());
      active_warps.push_back(*itW);
//    printf("warp size=%u\n",(*itW).size());
    }
  }
}

void find_active_shared_memories(std::vector<memory_space*> &shared_memories, tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> &active_kernels_warps, std::vector<unsigned> &kernel_vector) {
  for(tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>>::iterator itK = active_kernels_warps.begin(); itK != active_kernels_warps.end(); ++itK) {
    if (kernel_vector.size()>0 && kernel_vector[0]!=0) {
      if (std::find(kernel_vector.begin(), kernel_vector.end(), itK->first) == kernel_vector.end()) { continue; }
    }
    tr1_hash_map<unsigned, memory_space*> shared_memory_map;
    for(std::vector<std::vector<ptx_thread_info*>>::iterator itW = itK->second.begin(); itW != itK->second.end(); ++itW) {
      for(std::vector<ptx_thread_info*>::iterator itT = itW->begin(); itT != itW->end(); ++itT) {
//      printf("Thread uid=%u Thread hw tid=%u, cta hwid=%u, cta uid=%u\n", (*itT)->get_uid(), (*itT)->get_hw_tid(), (*itT)->get_hw_ctaid(), (*itT)->get_cta_uid());
        shared_memory_map[(*itT)->get_cta_uid()] = (*itT)->m_shared_mem;
      }
    }
    for(tr1_hash_map<unsigned, memory_space*>::iterator itSm = shared_memory_map.begin(); itSm != shared_memory_map.end(); ++itSm) {
      shared_memories.push_back(itSm->second);
    }
  }
}

void bitflip_n_nregs(std::vector<ptx_thread_info*> &threads_vector, char *register_rand_n, char *reg_bitflip_rand_n) {
  std::vector<unsigned> reg_bitflip_vector, register_rand_n_vector;
  read_colon_option(reg_bitflip_vector, reg_bitflip_rand_n);
  read_colon_option(register_rand_n_vector, register_rand_n);
  for(std::vector<ptx_thread_info*>::iterator threads_it = threads_vector.begin(); threads_it != threads_vector.end(); ++threads_it) {
    std::vector<const symbol*> reg_symbols;
    std::vector<ptx_reg_t*> regs;
    std::list<tr1_hash_map<const symbol *, ptx_reg_t>> &regs_list = (*threads_it)->get_regs();
    for(std::list<tr1_hash_map<const symbol *, ptx_reg_t>>::iterator regs_map_it = regs_list.begin(); regs_map_it != regs_list.end(); ++regs_map_it) {
      tr1_hash_map<const symbol *, ptx_reg_t> &regs_map = *regs_map_it;
      for(tr1_hash_map<const symbol *, ptx_reg_t>::iterator regs_it = regs_map.begin(); regs_it != regs_map.end(); ++regs_it) {
        if (std::find(reg_symbols.begin(), reg_symbols.end(), regs_it->first) != reg_symbols.end()) { continue; }
        reg_symbols.push_back(regs_it->first);
        regs.push_back(&(regs_it->second));
      }
    }
    for(std::vector<unsigned>::iterator reg_num_it = register_rand_n_vector.begin(); reg_num_it != register_rand_n_vector.end(); ++reg_num_it) {
      int reg_idx = (*reg_num_it) - 1;
      if (reg_idx < regs.size()) {
        ptx_reg_t *reg_to_bitflip = regs[reg_idx];
        unsigned long* reg_64b = (unsigned long*) reg_to_bitflip; // 8 bytes
        for(std::vector<unsigned>::iterator bf_it = reg_bitflip_vector.begin(); bf_it != reg_bitflip_vector.end(); ++bf_it) {
//          printf("Before bit flip of reg=%s, value = %lu\n", reg_symbols[reg_idx]->name().c_str(), *reg_to_bitflip);
          *reg_64b ^= 1UL << (*bf_it-1);
//          printf("After bit %u flip of reg=%s, value = %lu\n", *bf_it, reg_symbols[reg_idx]->name().c_str(), *reg_to_bitflip);
        }
        // Register the injection for effectiveness tracking and print last-writer info
        std::vector<unsigned> flipped_bits = reg_bitflip_vector;
        gpgpu_sim *gpu = (gpgpu_sim *)((*threads_it)->get_gpu());
        unsigned long long cyc = gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
        unsigned pc = (*threads_it)->get_pc();
        ptx_thread_info::reg_write_info lastw = (*threads_it)->get_last_writer(reg_symbols[reg_idx]);
        (*threads_it)->register_reg_injection(reg_symbols[reg_idx], flipped_bits, cyc, pc, lastw);
      }
    }
//    (*threads_it)->dump_regs(stdout);
  }
}

void bitflip_n_local_mem(std::vector<ptx_thread_info*> &threads_vector, char *local_mem_bitflip_rand_n) {
  std::vector<unsigned> local_mem_bitflip_vector;
  read_colon_option(local_mem_bitflip_vector, local_mem_bitflip_rand_n);

  const unsigned bsize = 32U; // BSIZE=32 for local memory
  for(std::vector<ptx_thread_info*>::iterator threads_it = threads_vector.begin(); threads_it != threads_vector.end(); ++threads_it) {
//    unsigned block_size = 1U << (*threads_it)->m_local_mem->get_log2_block_size();
    memory_space_impl<bsize> *local_mem = (memory_space_impl<bsize>*) (*threads_it)->m_local_mem;
    mem_map<mem_addr_t, mem_storage<bsize> > &memory_data = local_mem->get_m_data();
    for(std::vector<unsigned>::iterator bf_it = local_mem_bitflip_vector.begin(); bf_it != local_mem_bitflip_vector.end(); ++bf_it) {
      unsigned bf = *bf_it;
      unsigned block_idx = bf/(bsize*8); // this in fact is the mem_addr_t of memory_data
      unsigned bit_in_block = bf - block_idx*bsize*8;
      unsigned idx_64b = bit_in_block/64;
      unsigned bit_in_64b = bit_in_block - idx_64b*64;

      if (memory_data.find(block_idx) != memory_data.end()) {
        unsigned long long *i_data = (unsigned long long *)memory_data[block_idx].get_m_data();
//        printf("BEFORE BIT FLIP: address=%p\n", i_data);
//        (*threads_it)->m_local_mem->print("%d", stdout);
        i_data[idx_64b] ^= 1UL << (bit_in_64b-1);
//        printf("AFTER BIT FLIP: address=%p\n", i_data);
//        g_print_memory_space((*threads_it)->m_local_mem, "%d");
      }
      // Register for effectiveness tracking in the thread's local FI state
      gpgpu_sim *gpu = (gpgpu_sim *)((*threads_it)->get_gpu());
      unsigned long long cyc = gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
      unsigned pc = (*threads_it)->get_pc();
      mem_addr_t byte_addr = block_idx * bsize + (bit_in_block - 1) / 8;
      unsigned bit_in_byte_1based = ((bit_in_block - 1) % 8) + 1;
      ptx_thread_info::reg_write_info lastw = (*threads_it)->get_last_local_mem_writer(byte_addr);
      (*threads_it)->register_local_mem_injection(byte_addr, bit_in_byte_1based, cyc, pc, lastw);
      printf("bf=%u, block_idx=%u, bit_in_block=%u, idx_64b=%u, bit_in_64b=%u\n", bf, block_idx, bit_in_block, idx_64b, bit_in_64b);
    }
  }
}

void bitflip_n_shared_mem_nblocks(std::vector<memory_space*> shared_memories,
                                  unsigned block_rand, unsigned block_n,
                                  char *shared_mem_bitflip_rand_n,
                                  std::vector<ptx_thread_info*> &active_threads) {
  std::vector<unsigned> shared_mem_bitflip_vector;
  read_colon_option(shared_mem_bitflip_vector, shared_mem_bitflip_rand_n);

  const unsigned bsize = 16*1024; // BSIZE=16*1024 for shared memory
  unsigned blk_n=block_n;
  while (blk_n>0 && !shared_memories.empty()) {
    int block_idx = block_rand % shared_memories.size();
    memory_space_impl<bsize> *shared_mem_to_bitflip = (memory_space_impl<bsize>*) shared_memories[block_idx];
    mem_map<mem_addr_t, mem_storage<bsize> > &memory_data = shared_mem_to_bitflip->get_m_data();

    for(std::vector<unsigned>::iterator bf_it = shared_mem_bitflip_vector.begin(); bf_it != shared_mem_bitflip_vector.end(); ++bf_it) {
      unsigned bf = *bf_it;
      unsigned page_idx = bf/(bsize*8); // mem_addr_t (index) of memory_data page
      unsigned bit_in_block = bf - page_idx*bsize*8;
      unsigned idx_64b = bit_in_block/64;
      unsigned bit_in_64b = bit_in_block - idx_64b*64;

      if (memory_data.find(page_idx) != memory_data.end()) {
        unsigned long long *i_data = (unsigned long long *)memory_data[page_idx].get_m_data();
//        printf("BEFORE BIT FLIP: address=%p\n", i_data);
//        shared_mem_to_bitflip->print("%d", stdout);
        i_data[idx_64b] ^= 1UL << (bit_in_64b-1);
//        printf("AFTER BIT FLIP: address=%p\n", i_data);
//        shared_mem_to_bitflip->print("%08x", stdout);
      }
      // Register for shared-memory FI effectiveness tracking (CTA-scoped)
      // Find a thread belonging to this shared memory to access its CTA info
      ptx_thread_info *t0 = NULL;
      for (size_t ti = 0; ti < active_threads.size(); ++ti) {
        if (active_threads[ti] && active_threads[ti]->m_shared_mem == shared_mem_to_bitflip) {
          t0 = active_threads[ti];
          break;
        }
      }
      if (t0 && t0->m_cta_info) {
        gpgpu_sim *gpu = (gpgpu_sim *)(t0->get_gpu());
        unsigned long long cyc = gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
        unsigned pc = t0->get_pc();
        mem_addr_t byte_addr = page_idx * bsize + (bit_in_block - 1) / 8;
        unsigned bit_in_byte_1based = ((bit_in_block - 1) % 8) + 1;
        ptx_cta_info::smem_write_info lastw = t0->m_cta_info->get_last_shared_mem_writer(byte_addr);
        t0->m_cta_info->register_shared_mem_injection(byte_addr, bit_in_byte_1based, cyc, pc, lastw, t0);
      }
      printf("bf=%u, page_idx=%u, bit_in_block=%u, idx_64b=%u, bit_in_64b=%u\n", bf, page_idx, bit_in_block, idx_64b, bit_in_64b);
    }

    shared_memories.erase(shared_memories.begin() + block_idx);
    blk_n--;
  }
}

// cache_type <= 0: L1D, 1: L1C, 2: L1T
void gpgpu_sim::bitflip_l1_cache(unsigned cache_type) {
  std::vector<unsigned> l1_bitflip_vector, l1_shader_vector;
  char *l1_cache_bitflip_rand_n = cache_type==0 ? m_config.l1d_cache_bitflip_rand_n
                                                : cache_type==1 ? m_config.l1c_cache_bitflip_rand_n
                                                                : m_config.l1t_cache_bitflip_rand_n;
  char *l1_shader_rand_n = cache_type==0 ? m_config.l1d_shader_rand_n
                                   : cache_type==1 ? m_config.l1c_shader_rand_n
                                                   : m_config.l1t_shader_rand_n;
  read_colon_option(l1_bitflip_vector, l1_cache_bitflip_rand_n);
  read_colon_option(l1_shader_vector, l1_shader_rand_n);

  std::vector<cache_t*> l1_to_bitflip;
  std::vector<unsigned> l1_cluster_to_bitflip;
  std::vector<unsigned> l1_shader_to_bitflip;

  for (unsigned cluster_idx = 0; cluster_idx < m_shader_config->n_simt_clusters; cluster_idx++) {
    simt_core_cluster *simt_core_cluster = m_cluster[cluster_idx];
    for (unsigned shd_core_idx = 0; shd_core_idx < simt_core_cluster->get_config()->n_simt_cores_per_cluster; shd_core_idx++) {
      shader_core_ctx *shader_core_ctx = (simt_core_cluster->get_core())[shd_core_idx];
      if (std::find(l1_shader_vector.begin(), l1_shader_vector.end(), shader_core_ctx->get_sid()) != l1_shader_vector.end()) {
        cache_t *cache_temp = cache_type==0 ? shader_core_ctx->m_ldst_unit->m_L1D
                                              : cache_type==1 ? (cache_t*) shader_core_ctx->m_ldst_unit->m_L1C
                                                              : (cache_t*) shader_core_ctx->m_ldst_unit->m_L1T;
        l1_to_bitflip.push_back(cache_temp);
        l1_cluster_to_bitflip.push_back(cluster_idx);
        l1_shader_to_bitflip.push_back(shd_core_idx);
      }
    }
  }

  for(int i=0; i<l1_to_bitflip.size(); i++) {
    cache_t *l1 = l1_to_bitflip[i];
    std::string m_name = (cache_type==0 || cache_type==1) ? ((baseline_cache*)l1)->m_name.c_str()
                                                           : ((tex_cache*)l1)->m_name.c_str();
    std::vector<bool> l1_bf_enabled;
    std::vector<unsigned> l1_line_bitflip_bits_idx;
    std::vector<new_addr_type> l1_tag;
    std::vector<unsigned> l1_index;
    const cache_config &m_config = (cache_type==0 || cache_type==1) ? ((baseline_cache*)l1)->m_config
                                                                    : ((tex_cache*)l1)->m_config;
    tag_array *m_tag_array = (cache_type==0 || cache_type==1) ? ((baseline_cache*)l1)->m_tag_array
                                                              : &(((tex_cache*)l1)->m_tags);

    std::ofstream outfile;
    std::string file = "cache_logs/" + m_name.substr(0, m_name.size()-4) + std::string("_") + std::string(this->m_config.run_uid);
    outfile.open(file, std::ios::app); // append instead of overwrite

    unsigned tag_array_size_bits = 57;
  for(int j=0; j<l1_bitflip_vector.size(); j++) {
      unsigned bf_l1 = l1_bitflip_vector[j] - 1;
      unsigned l1_line_sz_extra_bits = m_config.get_line_sz()*8 + tag_array_size_bits;
      unsigned bf_line_idx = bf_l1 / l1_line_sz_extra_bits;
      unsigned bf_line_sz_bits_extra_idx = bf_l1 - bf_line_idx * l1_line_sz_extra_bits;

      // L1D: sector cache block, L1C & L1T: line cache block
      cache_block_t *line = m_tag_array->m_lines[bf_line_idx];
      outfile << m_name.c_str() << ", line " << bf_line_idx << ", bf " << bf_l1+1 << std::endl;

      // bf on tag array
      if (bf_line_sz_bits_extra_idx <= (tag_array_size_bits-1)) {
        unsigned bf_tag = 63 - bf_line_sz_bits_extra_idx;
        printf("Tag before = %llu, bf_tag=%u\n", line->m_tag, bf_tag+1);
        line->m_tag ^= 1UL << bf_tag;
        printf("Tag after = %llu, bf_tag=%u\n", line->m_tag, bf_tag+1);
        continue;
      }

      bool is_valid_line = false;
      if (cache_type==0) {
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
          if (((sector_cache_block*)line)->m_status[i] != INVALID) {
            is_valid_line = true;
            break;
          }
        }
      } else {
        is_valid_line = line->is_valid_line();
      }

      unsigned l1_line_sz_data_bits_idx = bf_line_sz_bits_extra_idx - tag_array_size_bits;
      if (is_valid_line) {
        l1_bf_enabled.push_back(true);
        l1_line_bitflip_bits_idx.push_back(l1_line_sz_data_bits_idx);
        l1_tag.push_back(line->m_tag);
        l1_index.push_back(bf_line_idx);

        // Track injection info and print injection logs
        cache_injection_info inj;
        inj.pending = true;
        inj.inject_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
        inj.inject_pc = 0; // cache injection not tied to a thread PC

        // Try to fetch last writer for this line (L1D only; L1C/L1T are read-only)
        if (cache_type == 0) {
          cache_t *cache_obj = l1;
          std::pair<unsigned, new_addr_type> key = std::make_pair(bf_line_idx, line->m_tag);
          if (l1d_last_writers[cache_obj].find(key) != l1d_last_writers[cache_obj].end()) {
            inj.last_writer_at_inject = l1d_last_writers[cache_obj][key];
          }
        }

        const char *label = cache_type==0 ? "L1D_cache" : (cache_type==1 ? "L1C_cache" : "L1T_cache");
        printf("[%s_FI_INJECT] cache=%s line_idx=%u tag=%u bit_idx=%u at cycle=%llu\n",
               label, m_name.c_str(), bf_line_idx, (unsigned)line->m_tag, l1_line_sz_data_bits_idx+1, inj.inject_cycle);
        if (inj.last_writer_at_inject.inst) {
          unsigned writer_uid = inj.last_writer_at_inject.inst->uid();
          printf("[%s_FI_WRITER] last_writer uid=%u at cycle=%llu PC=%u -> ",
                 label, writer_uid, inj.last_writer_at_inject.cycle, inj.last_writer_at_inject.pc);
          // print by PC using global context
          this->gpgpu_ctx->func_sim->ptx_print_insn(inj.last_writer_at_inject.pc, stdout);
          printf("\n");
        }

        printf("L1 %s ENABLED: bf_l1d = %u, l1d_line_sz_bits = %u, bf_line_idx = %u, bf_1024bits_idx = %u and tag = %x\n", m_name.c_str(), bf_l1, l1_line_sz_extra_bits, bf_line_idx, l1_line_sz_data_bits_idx+1, line->m_tag);

        // store injection info parallel to metadata vectors
        if (cache_type==0) {
          // ensure container size
          if (l1d_inject_info.size() <= i) l1d_inject_info.resize(i+1);
          l1d_inject_info[i].push_back(inj);
        } else if (cache_type==1) {
          if (l1c_inject_info.size() <= i) l1c_inject_info.resize(i+1);
          l1c_inject_info[i].push_back(inj);
        } else {
          if (l1t_inject_info.size() <= i) l1t_inject_info.resize(i+1);
          l1t_inject_info[i].push_back(inj);
        }
      }
    }
    // different variables because we might run bit flips on more than one cache type
    if (cache_type==0) {
      l1d_enabled.push_back(l1_bf_enabled.size() > 0);
      l1d_bf_enabled.push_back(l1_bf_enabled);
      l1d_cluster_idx.push_back(l1_cluster_to_bitflip[i]);
      l1d_shader_core_ctx.push_back(l1_shader_to_bitflip[i]);
      l1d_line_bitflip_bits_idx.push_back(l1_line_bitflip_bits_idx);
      l1d_tag.push_back(l1_tag);
      l1d_index.push_back(l1_index);
    } else if (cache_type==1) {
      l1c_enabled.push_back(l1_bf_enabled.size() > 0);
      l1c_bf_enabled.push_back(l1_bf_enabled);
      l1c_cluster_idx.push_back(l1_cluster_to_bitflip[i]);
      l1c_shader_core_ctx.push_back(l1_shader_to_bitflip[i]);
      l1c_line_bitflip_bits_idx.push_back(l1_line_bitflip_bits_idx);
      l1c_tag.push_back(l1_tag);
      l1c_index.push_back(l1_index);
    } else {
      l1t_enabled.push_back(l1_bf_enabled.size() > 0);
      l1t_bf_enabled.push_back(l1_bf_enabled);
      l1t_cluster_idx.push_back(l1_cluster_to_bitflip[i]);
      l1t_shader_core_ctx.push_back(l1_shader_to_bitflip[i]);
      l1t_line_bitflip_bits_idx.push_back(l1_line_bitflip_bits_idx);
      l1t_tag.push_back(l1_tag);
      l1t_index.push_back(l1_index);
    }
  }
}

void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      m_memory_sub_partition[i]->accumulate_L2cache_stats(
          m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for GPUWattch
      m_cluster[i]->get_icnt_stats(
          m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
      m_cluster[i]->get_cache_stats(
          m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      m_cluster[i]->get_current_occupancy(
          gpu_occupancy.aggregate_warp_slot_filled,
          gpu_occupancy.aggregate_theoretical_warp_slots);
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }

    unsigned long long current_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
    if (current_cycle == 1) {
      read_colon_option(kernel_vector, m_config.kernel_n);

      if (m_config.profile == 2) {
        active_threads_sum = 0;
        FILE *file = fopen("./cycles.txt", "r");
        cycles_txt_lines = 0;
        int num;
        while (fscanf(file, "%d", &num) > 0) {
          cycles_txt_lines++;
          cycles_txt.push_back(num);
        }
        fclose(file);
      }
    }

    if (m_config.profile == 2) {
      // key: kernel_id (index starts from 1)
      tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> active_kernels_warps;
      find_active_kernels_warps(active_kernels_warps, m_shader_config, m_cluster);
      std::vector<ptx_thread_info*> active_threads;

      find_active_threads(active_threads, active_kernels_warps, kernel_vector);

      if (std::find(cycles_txt.begin(), cycles_txt.end(), current_cycle) != cycles_txt.end()) {
        active_threads_sum += active_threads.size();
      }

    } else if (m_config.profile == 1) {
      // key: kernel_id (index starts from 1)
      tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> active_kernels_warps;
      find_active_kernels_warps(active_kernels_warps, m_shader_config, m_cluster);

      for(tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>>::iterator itK = active_kernels_warps.begin(); itK != active_kernels_warps.end(); ++itK) {
        std::vector<ptx_thread_info*> active_threads;
        std::vector<unsigned> kernel_uid = {itK->first};
        char *kernel_cstr = kernel_name[itK->first];
        unsigned max_regs_size = 0;

        // find start and last cycle of a kernel
        if (kernel_start_end_cycle.find(itK->first) != kernel_start_end_cycle.end()) {
          kernel_start_end_cycle[itK->first][1] = current_cycle;
        } else {
          kernel_start_end_cycle[itK->first] = {current_cycle, current_cycle};
        }

        // find maximum number of registers used by each kernel
        find_active_threads(active_threads, active_kernels_warps, kernel_uid);
        for(std::vector<ptx_thread_info*>::iterator itT = active_threads.begin(); itT != active_threads.end(); ++itT) {
          if ((*itT)->get_regs().back().size() > max_regs_size) {
            max_regs_size = (*itT)->get_regs().back().size();
          }
        }
        if (max_active_regs.find(kernel_cstr) != max_active_regs.end()) {
          if (max_regs_size > max_active_regs[kernel_cstr]) {
            max_active_regs[kernel_cstr] = max_regs_size;
          }
        } else {
          max_active_regs[kernel_cstr] = max_regs_size;
        }

        // find SIMT cores used by each kernel
        for (unsigned cluster_idx = 0; cluster_idx < m_shader_config->n_simt_clusters; cluster_idx++) {
          simt_core_cluster *simt_core_cluster = m_cluster[cluster_idx];
          for (unsigned shd_core_idx = 0; shd_core_idx < simt_core_cluster->get_config()->n_simt_cores_per_cluster; shd_core_idx++) {
            shader_core_ctx *shader_core_ctx = (simt_core_cluster->get_core())[shd_core_idx];
            if (shader_core_ctx->isactive()) {
              unsigned kernel_uid = shader_core_ctx->get_kernel()->get_uid();
              char *kernel_sh = kernel_name[kernel_uid];
              if ( shaders_used.find(kernel_sh) == shaders_used.end() ) {
                shaders_used[kernel_sh] = {shader_core_ctx->get_sid()};
              } else {
                shaders_used[kernel_sh].insert(shader_core_ctx->get_sid());
              }
            }
          }
        }
      }
    } else {
      if (current_cycle == m_config.total_cycle_rand) {
        printf("#### gpu_sim_cycle=%llu and gpu_tot_sim_cycle=%llu\n", gpu_sim_cycle, gpu_tot_sim_cycle);
        // Start measuring time
        struct timeval begin, end;
        gettimeofday(&begin, 0);

        bool register_file=false, local_memory=false, shared_memory=false, l1d_cache=false, l1c_cache=false, l1t_cache=false, l2_cache_comp=false;

        std::vector<unsigned> components_to_flip_vector;
        read_colon_option(components_to_flip_vector, m_config.components_to_flip);

        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 0) != components_to_flip_vector.end()) {
          register_file=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 1) != components_to_flip_vector.end()) {
          local_memory=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 2) != components_to_flip_vector.end()) {
          shared_memory=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 3) != components_to_flip_vector.end()) {
          l1d_cache=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 4) != components_to_flip_vector.end()) {
          l1c_cache=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 5) != components_to_flip_vector.end()) {
          l1t_cache=true;
        }
        if (std::find(components_to_flip_vector.begin(), components_to_flip_vector.end(), 6) != components_to_flip_vector.end()) {
          l2_cache_comp=true;
        }

        // key: kernel_id (index starts from 1)
        tr1_hash_map<unsigned, std::vector<std::vector<ptx_thread_info*>>> active_kernels_warps;
        find_active_kernels_warps(active_kernels_warps, m_shader_config, m_cluster);

        std::vector<ptx_thread_info*> active_threads;
        std::vector<std::vector<ptx_thread_info*>> active_warps;
        std::vector<memory_space*> shared_memories;
        std::vector<memory_space*> l1d_caches;

        std::vector<ptx_thread_info*> threads_bitflip;

        if (register_file || local_memory) {
          if (m_config.per_warp) {
            find_active_warps(active_warps, active_kernels_warps, kernel_vector);
            if (active_warps.size()>0) {
              threads_bitflip = active_warps[m_config.warp_rand % active_warps.size()];
            }
          } else {
            find_active_threads(active_threads, active_kernels_warps, kernel_vector);
            if (active_threads.size()>0) {
//              printf("ACTIVE THREADS %u\n", active_threads.size());
//              g_print_memory_space(this->get_global_memory());
              threads_bitflip.push_back(active_threads[m_config.thread_rand % active_threads.size()]);
            }
          }
        }

        if (register_file) {
          bitflip_n_nregs(threads_bitflip, m_config.register_rand_n, m_config.reg_bitflip_rand_n);
        }
        if (local_memory) {
          bitflip_n_local_mem(threads_bitflip, m_config.local_mem_bitflip_rand_n);
        }
        if (shared_memory) {
          find_active_shared_memories(shared_memories, active_kernels_warps, kernel_vector);
          // Ensure we have a list of active threads to map shared-mem owners
          if (active_threads.empty()) {
            find_active_threads(active_threads, active_kernels_warps, kernel_vector);
          }
          bitflip_n_shared_mem_nblocks(shared_memories, m_config.block_rand,
                                       m_config.block_n,
                                       m_config.shared_mem_bitflip_rand_n,
                                       active_threads);
        }
        if (l1d_cache) {
          bitflip_l1_cache(0);
        }
        if (l1c_cache) {
          bitflip_l1_cache(1);
        }
        if (l1t_cache) {
          bitflip_l1_cache(2);
        }
        if (l2_cache_comp) {
          std::ofstream outfile;
          std::string file = "cache_logs/L2_" + std::string(this->m_config.run_uid);
          outfile.open(file, std::ios::app); // append instead of overwrite
          std::vector<unsigned> l2_bitflip_vector;
          read_colon_option(l2_bitflip_vector, m_config.l2_cache_bitflip_rand_n);
          for(int j=0; j<l2_bitflip_vector.size(); j++) {
            // get_total_size_inKB() -> total size of L2 cache per bank
            unsigned bf_l2 = l2_bitflip_vector[j] - 1;
            unsigned l2_size_per_bank = this->m_memory_config->m_L2_config.get_total_size_inKB()*1024*8 + this->m_memory_config->m_L2_config.get_num_lines()*57;
            unsigned bank_id = bf_l2 / l2_size_per_bank;
            unsigned bf_l2_cache_bank = bf_l2 % l2_size_per_bank;
            l2_cache *l2_cache_bank = this->m_memory_sub_partition[bank_id]->m_L2cache;
            unsigned l2_line_sz_extra_bits = l2_cache_bank->m_config.get_line_sz()*8 + 57;
            unsigned bf_line_idx = bf_l2_cache_bank / l2_line_sz_extra_bits;
            unsigned bf_line_sz_bits_extra_idx = bf_l2_cache_bank - bf_line_idx * l2_line_sz_extra_bits;

            // line cache block
            cache_block_t *line = l2_cache_bank->m_tag_array->m_lines[bf_line_idx];
            outfile << l2_cache_bank->m_name.c_str() << ", line " << bf_line_idx << ", total l2 bf " << bf_l2+1 << std::endl;

            // bf on tag array
            if (bf_line_sz_bits_extra_idx <= 56) {
              unsigned bf_tag = 63 - bf_line_sz_bits_extra_idx;
              printf("Tag before = %x, bf_tag=%u\n", line->m_tag, bf_tag);
              line->m_tag ^= 1UL << bf_tag;
              printf("Tag after = %x, bf_tag=%u\n", line->m_tag, bf_tag);
              continue;
            }

            unsigned l2_line_sz_data_bits_idx = bf_line_sz_bits_extra_idx - 57;
            if (line->is_valid_line()) {
              this->l2_enabled = true;
              this->l2_bf_enabled.push_back(true);
              this->l2_bank_id.push_back(bank_id);
              this->l2_line_bitflip_bits_idx.push_back(l2_line_sz_data_bits_idx);
              this->l2_tag.push_back(line->m_tag);
              this->l2_index.push_back(bf_line_idx);

              // Track injection info and print injection logs
              cache_injection_info inj;
              inj.pending = true;
              inj.inject_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
              inj.inject_pc = 0;
              std::pair<unsigned, new_addr_type> key = std::make_pair(bf_line_idx, line->m_tag);
              if (this->l2_last_writers[bank_id].find(key) != this->l2_last_writers[bank_id].end()) {
                inj.last_writer_at_inject = this->l2_last_writers[bank_id][key];
              }
              this->l2_inject_info.push_back(inj);

              printf("[L2_cache_FI_INJECT] bank=%u line_idx=%u tag=%u bit_idx=%u at cycle=%llu\n",
                     bank_id, bf_line_idx, (unsigned)line->m_tag, l2_line_sz_data_bits_idx+1,
                     (unsigned long long)inj.inject_cycle);
              if (inj.last_writer_at_inject.inst) {
                unsigned writer_uid = inj.last_writer_at_inject.inst->uid();
                printf("[L2_cache_FI_WRITER] last_writer uid=%u at cycle=%llu PC=%u -> ",
                       writer_uid, inj.last_writer_at_inject.cycle, inj.last_writer_at_inject.pc);
                this->gpgpu_ctx->func_sim->ptx_print_insn(inj.last_writer_at_inject.pc, stdout);
                printf("\n");
              }

              printf("L2 %s ENABLED: bf_l2_cache_bank = %u, l2_line_sz_bits = %u, bf_line_idx = %u, bf_line_sz_bits_idx = %u and tag = llu%\n", l2_cache_bank->m_name.c_str(), bf_l2_cache_bank, l2_line_sz_extra_bits, bf_line_idx, l2_line_sz_data_bits_idx+1, line->m_tag);
            }
          }
        }

        // Stop measuring time and calculate the elapsed time
        gettimeofday(&end, 0);
        long seconds = end.tv_sec - begin.tv_sec;
        long microseconds = end.tv_usec - begin.tv_usec;
        double elapsed = seconds + microseconds*1e-6;
        printf("Fault injection time taken=  %.6f seconds\n", elapsed);
        printf("Fault injection on total_cycle = %llu\n", current_cycle);
      }
    }

    // print profiling information on last cycle
    if (current_cycle == m_config.last_cycle-1) {
      if (m_config.profile == 2) {
        if (cycles_txt_lines > 0) {
          printf("Mean active threads = %d\n", active_threads_sum/cycles_txt_lines);
        }
      } else if (m_config.profile == 1) {
        for(std::map<char *, unsigned>::iterator itREG = max_active_regs.begin(); itREG != max_active_regs.end(); ++itREG) {
           printf("Kernel = %s, max active regs = %u\n", itREG->first, itREG->second);
        }

        for(std::map<char *, std::set<unsigned>>::iterator itK = shaders_used.begin(); itK != shaders_used.end(); ++itK) {
           printf("Kernel = %s used shaders: ", itK->first);
           for(std::set<unsigned>::iterator itSH = itK->second.begin(); itSH != itK->second.end(); ++itSH) {
             printf("%u ", *itSH);
           }
           printf("\n");
        }
        for(tr1_hash_map<unsigned, std::vector<unsigned long long>>::iterator itK = kernel_start_end_cycle.begin(); itK != kernel_start_end_cycle.end(); ++itK) {
           printf("Kernel = %u with name = %s, started on cycle = %llu and finished on cycle = %llu\n", itK->first, kernel_name[itK->first], (itK->second)[0], (itK->second)[1]);
           delete kernel_name[itK->first];
        }
      }
    }

    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                  m_power_stats, m_config.gpu_stat_sample_freq,
                  gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                  gpu_sim_insn);
    }
#endif

    issue_block2core();
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f\% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }
