#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ==== 辅助宏 ====
#ifndef MAX
#define MAX(a,b) (( (a) > (b) ) ? (a) : (b))
#endif

// ===== 可选：限制共享内存为 64KB（沿用你的原设定）=====
#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// ===== GPU 配置 =====
#define WARP_SIZE 32

// ===== WMMA tile 尺寸（固定 16x16x16）=====
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 为了和你原代码一致，保留 M/N/K 基础 tile 常量
#define M 16
#define N 16
#define K 16

#define M_SEED 6432

// ===== 实现相关常量（原样保留）=====
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

// 移除依赖编译期 M_GLOBAL 的宏 GLOBAL_MEM_STRIDE，运行时不需要这个宏。
// #define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

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

// 初始化主机端矩阵：改成带维度参数（运行时）
__host__ void init_host_matrices(half *a, half *b, float *c,
                                 int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
  srand(M_SEED);
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = __float2half(static_cast<float>(rand() % 3));
    }
  }
  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = __float2half(static_cast<float>(rand() % 3));
    }
  }
  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<float>(rand() % 3);
  }
}

// ===== 核函数（保持你的简单 WMMA 版本，按运行时维度工作）=====
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // 将线程块/warp 映射到 tile
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // WMMA 片段
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // 沿 K 维聚合
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;   // N = 16，是 WMMA 列 tile 步长
    int bRow = i;

    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // 累加到 C，并写回 D
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
  // -------- 解析命令行参数：支持 ./gemm s 或 ./gemm mt nt kt --------
  int mt = 2, nt = 2, kt = 2;  // 默认各维 tile 倍数
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

  // 运行时全局尺寸（保证是 16 的整数倍，满足 WMMA 要求）
  const int M_GLOBAL = M * mt;  // rows of A/C, rows-tiles
  const int N_GLOBAL = N * nt;  // cols of B/C, cols-tiles
  const int K_GLOBAL = K * kt;  // shared inner dim tiles

  // -------- 分配主机内存 --------
  half  *A_h = (half *) malloc(sizeof(half)  * M_GLOBAL * K_GLOBAL);
  half  *B_h = (half *) malloc(sizeof(half)  * N_GLOBAL * K_GLOBAL);
  float *C_h = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  float *result_hD = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

  // -------- 分配设备内存 --------
  half  *A = NULL;
  half  *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half)  * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half)  * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  // 可选：设备指针对齐断言（一般 cudaMalloc 已对齐）
  // assert(((unsigned long long)A) % 128 == 0);
  // assert(((unsigned long long)B) % 128 == 0);
  // assert(((unsigned long long)C) % 128 == 0);
  // assert(((unsigned long long)D) % 128 == 0);

  // -------- 初始化主机数据并拷贝到设备 --------
  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half)  * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half)  * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

  // -------- 共享内存需求（沿用你原来的写法；此 demo 内核并未使用到该值）--------
  enum {
    SHMEM_SZ = MAX(
      sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  // -------- 计时 --------
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // -------- 设置网格/线程块：保持你的配置 --------
  dim3 gridDim, blockDim;
  // blockDim.x 必须是 warpSize 的倍数；128x4 => 16 个 warps
  blockDim.x = 128;
  blockDim.y = 4;

  // 每个 warp 负责一个 16x16 输出 tile：
  // x 方向步长：WMMA_M * (blockDim.x/32)
  // y 方向步长：WMMA_N * blockDim.y
  gridDim.x = (M_GLOBAL + (WMMA_M * (blockDim.x / 32) - 1)) / (WMMA_M * (blockDim.x / 32));
  gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  // -------- Launch kernel --------
  simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // -------- 结果拷回 --------
  checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

  // ====== 从 result.txt 读取期望结果 ======
  FILE *file = fopen("result.txt", "r");
  if (file == NULL) {
    printf("Failed\n");
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
    printf("Failed\n");
    free(expected);
    free(A_h); free(B_h); free(C_h); free(result_hD);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
    return 0;
  }

  // ====== 逐项比较结果，支持 NaN ======
  bool match = true;
  const float eps = 1e-5f;
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    float actual = result_hD[i];
    float expected_val = expected[i];

    if (isnan(actual) && isnan(expected_val)) continue;
    if (isnan(actual) || isnan(expected_val)) { match = false; break; }
    if (fabs(actual - expected_val) > eps) { match = false; break; }
  }

  if (match) {
    printf("Success\n");
  } else {
    printf("Failed\n");
  }

  // ====== 资源回收 ======
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
