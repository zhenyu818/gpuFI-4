#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>

#include <cfloat> // for FLT_MIN
#include <cmath>  // for NAN
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h> // for ldexpf
#include <mma.h>

// ==== 辅助宏 ====
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
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

#define M_SEED 1413

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

#define checkKernelErrors(expr)                                                                                        \
    do {                                                                                                               \
        expr;                                                                                                          \
        cudaError_t __err = cudaGetLastError();                                                                        \
        if (__err != cudaSuccess) {                                                                                    \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err));                          \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

using namespace nvcuda;

// ================= 辅助函数 ==================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", file, line, (int)result, cudaGetErrorString(result),
                func);
        exit(EXIT_FAILURE);
    }
}

// ---- 生成子正规数：half 与 float ----
// half 子正规范围：(2^-24, 2^-14) ≈ (5.96e-8, 6.10e-5)
// 生成一个严格小于 2^-14 的正数，避免转换为 half 时变为 0 或变为 normal
static inline float gen_half_subnormal_pos() {
    const float min_sub = ldexpf(1.0f, -24);          // 2^-24
    const float max_sub = ldexpf(1.0f, -14) * 0.999f; // < 2^-14
    float u = rand() / (float)RAND_MAX;               // [0,1]
    return min_sub + u * (max_sub - min_sub);         // (2^-24, 2^-14)
}

// float 子正规：小于 FLT_MIN 的正数
static inline float gen_float_subnormal_pos() {
    // 缩放 FLT_MIN 以确保进入 subnormal 区间
    return (float)(rand() % 100 + 1) * FLT_MIN / 1e5f; // 典型 ~1e-43
}

// 改：带维度参数（运行时）
__host__ void init_host_matrices(half *a, half *b, float *c, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
    srand(M_SEED);

    // A: half 子正规
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = __float2half(gen_half_subnormal_pos());
        }
    }

    // B: half 子正规
    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = __float2half(gen_half_subnormal_pos());
        }
    }

    // C: float 子正规
    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = gen_float_subnormal_pos();
    }
}

// ===== 确定性 FP32 GEMM 核函数（避免 WMMA 引入的差异）=====
// A: row-major [M×K], B: row-major [K×N], C/D: row-major [M×N]
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha,
                                 float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m_ld || col >= n_ld)
        return;

    float acc = 0.0f;
    for (int k = 0; k < k_ld; ++k) {
        float av = __half2float(a[row * k_ld + k]);
        float bv = __half2float(b[k * n_ld + col]);
        float prod = __fmul_rn(av, bv);
        acc = __fadd_rn(acc, prod);
    }

    float cv = c[row * n_ld + col];
    float out = __fadd_rn(__fmul_rn(alpha, acc), __fmul_rn(beta, cv));
    d[row * n_ld + col] = out;
}

int main(int argc, char **argv) {
    // -------- 解析命令行参数：支持 ./gemm s 或 ./gemm mt nt kt --------
    int mt = 2, nt = 2, kt = 2; // 默认各维 tile 倍数
    if (argc == 2) {
        int s = atoi(argv[1]);
        if (s > 0)
            mt = nt = kt = s;
    } else if (argc >= 4) {
        int t1 = atoi(argv[1]);
        int t2 = atoi(argv[2]);
        int t3 = atoi(argv[3]);
        if (t1 > 0)
            mt = t1;
        if (t2 > 0)
            nt = t2;
        if (t3 > 0)
            kt = t3;
    }

    // 运行时全局尺寸（保证是 16 的整数倍，满足 WMMA 要求）
    const int M_GLOBAL = M * mt; // rows of A/C
    const int N_GLOBAL = N * nt; // cols of B/C
    const int K_GLOBAL = K * kt; // shared inner dim

    // -------- 分配主机内存 --------
    half *A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);          // [M×K]
    half *B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);          // [K×N] ← 修复点：注释与形状
    float *C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);       // [M×N]
    float *result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL); // [M×N]

    // -------- 分配设备内存 --------
    half *A = NULL;  // [M×K]
    half *B = NULL;  // [K×N]
    float *C = NULL; // [M×N]
    float *D = NULL; // [M×N]

    checkCudaErrors(cudaMalloc((void **)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void **)&B, sizeof(half) * K_GLOBAL * N_GLOBAL)); // ← 修复点：形状说明
    checkCudaErrors(cudaMalloc((void **)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc((void **)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // -------- 初始化主机数据并拷贝到设备 --------
    init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * K_GLOBAL * N_GLOBAL,
                               cudaMemcpyHostToDevice)); // ← 修复点：大小一致，但语义正确
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // -------- 共享内存需求（demo 内核未使用到该值）--------
    enum {
        SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    const float alpha = 1.1f;
    const float beta = 1.2f;

    // -------- 计时 --------
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // -------- 设置网格/线程块：采用 16x16 每线程计算 1 元素，保证确定性 --------
    dim3 blockDim(16, 16);
    dim3 gridDim((N_GLOBAL + blockDim.x - 1) / blockDim.x, (M_GLOBAL + blockDim.y - 1) / blockDim.y);

    // -------- Launch kernel --------
    simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // -------- 结果拷回 --------
    checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Fault Injection Test Failed!\n");
        free(A_h);
        free(B_h);
        free(C_h);
        free(result_hD);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(D);
        return 0;
    }

    float *expected = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    int count = 0;
    while (fscanf(file, "%f", &expected[count]) == 1 && count < M_GLOBAL * N_GLOBAL) {
        count++;
    }
    fclose(file);

    if (count != M_GLOBAL * N_GLOBAL) {
        printf("Fault Injection Test Failed!\n");
        free(expected);
        free(A_h);
        free(B_h);
        free(C_h);
        free(result_hD);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(D);
        return 0;
    }

    // ===== 显式比较 NaN 和 Inf =====
    bool match = true;
    const float eps = 1e-5f;
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float actual = result_hD[i];
        float expected_val = expected[i];

        if (isnan(actual) && isnan(expected_val))
            continue;
        if (isnan(actual) || isnan(expected_val)) {
            match = false;
            break;
        }

        if (isinf(actual) && isinf(expected_val)) {
            if (signbit(actual) != signbit(expected_val)) {
                match = false;
                break;
            } else
                continue;
        }
        if (isinf(actual) || isinf(expected_val)) {
            match = false;
            break;
        }

        if (fabs(actual - expected_val) > eps) {
            match = false;
            break;
        }
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
