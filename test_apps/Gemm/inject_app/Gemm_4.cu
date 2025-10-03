// 用法：
//   nvcc -arch=sm_70 -O3 -o gemm gemm.cu
//   ./gemm s         # M=N=K=16*s
//   ./gemm mt nt kt  # M=16*mt, N=16*nt, K=16*kt

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1
#endif

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

// Implementation constants.
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

// 运行时版本不再需要 GLOBAL_MEM_STRIDE 宏
// #define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// 见注释说明
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

// ---- 2:4 稀疏输入生成（half 版，逐组 4 个元素里保留 2 个非零）----
static void generate_2to4_sparse_half(half *data, int size) {
    for (int i = 0; i < size; i += 4) {
        bool selected[4] = {false, false, false, false};
        int cnt = 0;
        while (cnt < 2) {
            int idx = rand() % 4;
            if (!selected[idx]) {
                selected[idx] = true;
                cnt++;
            }
        }
        for (int k = 0; k < 4 && i + k < size; ++k) {
            if (selected[k]) {
                float v = static_cast<float>(rand() % 3);
                data[i + k] = __float2half(v);
            } else {
                data[i + k] = __float2half(0.0f);
            }
        }
    }
}
// ---- 2:4 稀疏输入生成（float 版，逐组 4 个元素里保留 2 个非零）----
static void generate_2to4_sparse_float(float *data, int size) {
    for (int i = 0; i < size; i += 4) {
        bool selected[4] = {false, false, false, false};
        int cnt = 0;
        while (cnt < 2) {
            int idx = rand() % 4;
            if (!selected[idx]) {
                selected[idx] = true;
                cnt++;
            }
        }
        for (int k = 0; k < 4 && i + k < size; ++k) {
            if (selected[k]) {
                float v = static_cast<float>(rand() % 3); // 0,1,2
                data[i + k] = v;
            } else {
                data[i + k] = 0.0f;
            }
        }
    }
}

// 改：带维度参数（运行时）
__host__ void init_host_matrices(half *a, half *b, float *c, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
    srand(M_SEED);

    // A: M_GLOBAL x K_GLOBAL，按连续内存每4元素一组做 2:4 稀疏
    generate_2to4_sparse_half(a, M_GLOBAL * K_GLOBAL);

    // B: N_GLOBAL x K_GLOBAL，同样做 2:4 稀疏
    generate_2to4_sparse_half(b, N_GLOBAL * K_GLOBAL);

    // C : M_GLOBAL x N_GLOBAL，同样做 2:4 稀疏
    generate_2to4_sparse_float(c, M_GLOBAL * N_GLOBAL);
}

// 简化 wmma 演示 kernel（按运行时维度工作）
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha,
                                 float beta) {
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid (warp 为单位)
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N; // = 16
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
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

    // 运行时全局尺寸（保证 16 的整数倍，满足 WMMA）
    const int M_GLOBAL = M * mt;
    const int N_GLOBAL = N * nt;
    const int K_GLOBAL = K * kt;

    // 主机端分配
    half *A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    half *B_h = (half *)malloc(sizeof(half) * N_GLOBAL * K_GLOBAL);
    float *C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    float *result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    // 设备端分配
    half *A = NULL;
    half *B = NULL;
    float *C = NULL;
    float *D = NULL;

    checkCudaErrors(cudaMalloc((void **)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void **)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void **)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc((void **)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // init & H2D
    init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // 共享内存需求（此 demo kernel 未直接使用；沿用计算）
    enum {
        SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };
    (void)SHMEM_SZ;

    const float alpha = 1.1f;
    const float beta = 1.2f;

    // 计时
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // 网格/线程块
    dim3 gridDim, blockDim;
    blockDim.x = 128; // 必须是 warpSize 的倍数
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * (blockDim.x / 32) - 1)) / (WMMA_M * (blockDim.x / 32));
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    // kernel
    simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // D2H
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