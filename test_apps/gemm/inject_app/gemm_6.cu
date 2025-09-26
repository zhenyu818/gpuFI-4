// 用法：
//   nvcc -arch=sm_70 -O3 -o gemm gemm.cu
//   ./gemm s         # M=N=K=16*s
//   ./gemm mt nt kt  # M=16*mt, N=16*nt, K=16*kt
//
// 说明：把原先编译期固定的 M_TILES/N_TILES/K_TILES 与 M_GLOBAL/N_GLOBAL/K_GLOBAL
//       改为运行时通过命令行参数决定；其余逻辑保持一致（含对抗性初始化/NaN/Inf 注入）。

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <limits>   // infinity
#include <cmath>    // NAN

// ==== 辅助宏（保留）====
#ifndef MAX
#define MAX(a,b) (( (a) > (b) ) ? (a) : (b))
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU 配置
#define WARP_SIZE 32

// WMMA tile 尺寸（固定 16x16x16）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 基础 tile 尺寸（与 WMMA 一致）
#define M 16
#define N 16
#define K 16

#define M_SEED 6432

// Implementation constants（保留以兼容你的设置）
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

// 运行时版本移除 GLOBAL_MEM_STRIDE（避免依赖编译期 N_GLOBAL）
// #define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// shared memory bank 冲突移位
#define SKEW_HALF 16

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;

// ================= 辅助函数 ==================
#define checkCudaErrors(val)  check( (val), #val, __FILE__, __LINE__ )

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (int)result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

/**
 * 对抗性输入生成（运行时尺寸）：
 * - A：棋盘格 ±1；每 8 列重复；每 5 行置 0；对角附近放大到 FP16 上限；稀疏注入 NaN/Inf/TINY。
 * - B：列周期（每 7 列重复）；偶列极小、奇列较大；每 6 行 -1；对角附近放大；稀疏 NaN/±Inf/0。
 * - C：小随机+行偏置；稀疏 NaN/±Inf。
 */
__host__ void init_host_matrices(half *a, half *b, float *c,
                                 int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
  srand(M_SEED);

  const float FP16_MAX = 65504.0f;
  const float TINY     = 1.0e-8f;   // 转 half 可能为 0 或次法线
  const float BIG      = FP16_MAX;  // 接近 FP16 上限
  const float INF_F    = std::numeric_limits<float>::infinity();
  const float NAN_F    = NAN;

  // ====== A: M_GLOBAL x K_GLOBAL（row_major）======
  for (int i = 0; i < M_GLOBAL; ++i) {
    for (int j = 0; j < K_GLOBAL; ++j) {
      float v = ((i ^ j) & 1) ? 1.0f : -1.0f;          // 棋盘格
      int j_rep = j % 8;                               // 8 列周期
      v *= (j_rep < 4 ? 1.0f : -1.0f);
      if ((i % 5) == 0) v = 0.0f;                      // 每 5 行清零
      if (abs(i - (j % M_GLOBAL)) <= 1) v = BIG;       // 近对角放大
      if ((i + 3*j) % 17 == 0) v = TINY;               // 极小值
      if ((i == 1 && j == 1) || ((i * 131 + j) % 997 == 0)) v = NAN_F; // NaN
      if ((i == 2 && j == 3) || ((i * 211 + j) % 1231 == 0)) v =  INF_F; // +Inf
      a[i * K_GLOBAL + j] = __float2half(v);
    }
  }

  // ====== B: N_GLOBAL x K_GLOBAL（kernel 以 col_major 读）======
  for (int i = 0; i < N_GLOBAL; ++i) {
    for (int j = 0; j < K_GLOBAL; ++j) {
      int jp = j % 7;
      float base  = (jp < 3) ? 1.0f : -1.0f;
      float scale = (jp % 2 == 0) ? TINY : 16.0f;      // 偶列极小、奇列较大
      if ((i % 6) == 0) base = -1.0f;                  // 每 6 行 -1
      float v = base * scale;
      if (abs(i - (j % N_GLOBAL)) <= 1) v = BIG;       // 近对角放大
      if (((i * 53 + j) % 101) == 0) v = 0.0f;         // 少量 0
      if ((i == 0 && j == 0) || ((i * 97 + j) % 1493 == 0)) v = NAN_F;   // NaN
      if ((i == 3 && j == 2) || ((i * 199 + j) % 1877 == 0)) v = -INF_F; // -Inf
      b[i * K_GLOBAL + j] = __float2half(v);
    }
  }

  // ====== C: M_GLOBAL x N_GLOBAL（float 累加器初值）======
  for (int i = 0; i < M_GLOBAL; ++i) {
    for (int j = 0; j < N_GLOBAL; ++j) {
      float r = (static_cast<float>(rand()) / RAND_MAX) - 0.5f; // 小随机
      if ((i % 4) == 0) r += 1.0f;                              // 行偏置
      if ((i == 0 && j == 1) || ((i * 383 + j) % 2003 == 0)) r =  INF_F;
      if ((i == 1 && j == 0) || ((i * 577 + j) % 2221 == 0)) r = -INF_F;
      if ((i == 2 && j == 2) || ((i * 811 + j) % 2551 == 0)) r =  NAN;
      c[i * N_GLOBAL + j] = r;
    }
  }
}

// 简易 WMMA kernel（保持你的写法，使用运行时维度）
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // warp 级二维 tiling
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // WMMA 片段
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // 沿 K 维累加
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N; // = 16（用 WMMA_N 更直观）
    int bRow = i;

    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // C = alpha*acc + beta*C
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
  }
}

int main(int argc, char **argv) {
  // ---- 解析命令行参数：支持 ./gemm s 或 ./gemm mt nt kt ----
  int mt = 2, nt = 2, kt = 2;  // 默认各维 2 个 16x16 tile
  if (argc == 2) {
    int s = atoi(argv[1]);
    if (s > 0) mt = nt = kt = s;
  } else if (argc >= 4) {
    int t1 = atoi(argv[1]);
    int t2 = atoi(argv[2]);
    int t3 = atoi(argv[3]);
    if (t1 > 0) mt = t1;
    if (t2 > 0) nt = t2;
    if (t3 > 0) kt = t3;
  }

  // 运行时全局尺寸（16 的倍数，满足 WMMA 要求）
  const int M_GLOBAL = M * mt;
  const int N_GLOBAL = N * nt;
  const int K_GLOBAL = K * kt;

  // ---- 主机内存 ----
  half  *A_h = (half *)malloc(sizeof(half)  * M_GLOBAL * K_GLOBAL);
  half  *B_h = (half *)malloc(sizeof(half)  * N_GLOBAL * K_GLOBAL);
  float *C_h = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  float *result_hD = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

  // ---- 设备内存 ----
  half  *A = NULL;
  half  *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half)  * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half)  * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  // ---- 初始化并拷贝 ----
  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half)  * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half)  * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

  // 共享内存需求估计（demo 内核未直接使用；保留计算以兼容）
  enum {
    SHMEM_SZ = MAX(
        sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
        M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };
  (void)SHMEM_SZ;

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  // ---- 计时 ----
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // ---- 网格/线程块 ----
  dim3 gridDim, blockDim;
  // blockDim.x 必须是 warpSize 的倍数；128x4 => 16 warps
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (WMMA_M * (blockDim.x / 32) - 1)) / (WMMA_M * (blockDim.x / 32));
  gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  // ---- kernel ----
  simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // ---- D2H ----
  checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

  FILE *file = fopen("result.txt", "r");
  if (file == NULL) {
    printf("Fault Injection Test Failed!\n");
    free(A_h); free(B_h); free(C_h); free(result_hD);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
    return 0;
  }

  float *expected = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  int count = 0;
  while (fscanf(file, "%f", &expected[count]) == 1 && count < M_GLOBAL * N_GLOBAL) {
    count++;
  }
  fclose(file);

  if (count != M_GLOBAL * N_GLOBAL) {
    printf("Fault Injection Test Failed!\n");
    free(expected);
    free(A_h); free(B_h); free(C_h); free(result_hD);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
    return 0;
  }

  // ===== 显式比较 NaN 和 Inf =====
  bool match = true;
  const float eps = 1e-5f;
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    float actual = result_hD[i];
    float expected_val = expected[i];

    if (isnan(actual) && isnan(expected_val)) continue;
    if (isnan(actual) || isnan(expected_val)) { match = false; break; }

    if (isinf(actual) && isinf(expected_val)) {
      if (signbit(actual) != signbit(expected_val)) { match = false; break; }
      else continue;
    }
    if (isinf(actual) || isinf(expected_val)) { match = false; break; }

    if (fabs(actual - expected_val) > eps) { match = false; break; }
  }

  if (match) {
    printf("Fault Injection Test Success!\n");
  } else {
    printf("Fault Injection Test Failed!\n");
  }

  free(expected);
  free(A_h);
  free(B_h);
  free(C_h);
  free(result_hD);

  checkCudaErrors(cudaFree(A));
  checkCudaErrors(cudaFree(B));
  checkCudaErrors(cudaFree(C));
  checkCudaErrors(cudaFree(D));

  return 0;
}
