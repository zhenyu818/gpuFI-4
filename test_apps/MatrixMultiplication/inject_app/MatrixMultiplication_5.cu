#include <assert.h>
#include <climits>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_SEED 1745
#define M_BLOCK_SIZE 16

// ================= 辅助函数 ==================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, (unsigned int)result,
                cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

bool checkCmdLineFlag(int argc, const char **argv, const char *flag) {
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], flag))
            return true;
    }
    return false;
}

int getCmdLineArgumentInt(int argc, const char **argv, const char *argName) {
    for (int i = 0; i < argc - 1; i++) {
        if (!strcmp(argv[i], argName)) {
            return atoi(argv[i + 1]);
        }
    }
    return 0;
}

// ================= CUDA Kernel ==================
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(double *C, const double *A, const double *B, int wA, int wB) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    double Csub = 0.0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// ================= 输入生成函数（包含 NaN） ==================
void RandomInitSpecial(double *data, int size) {
    srand(M_SEED);
    for (int i = 0; i < size; i++) {
        int rand_val = rand() % 100;
        if (rand_val < 50) {
            data[i] = NAN; // 50% 概率 NaN
        } else {
            data[i] = (double)(rand() % 10); // 50% 概率 0–9
        }
    }
}

// ================= 主计算函数 ==================
int MatrixMultiply(const dim3 &dimsA, const dim3 &dimsB) {
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(double) * size_A;
    double *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(double) * size_B;
    double *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    RandomInitSpecial(h_A, size_A);
    RandomInitSpecial(h_B, size_B);

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = sizeof(double) * dimsC.x * dimsC.y;
    double *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    double *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    dim3 threads(M_BLOCK_SIZE, M_BLOCK_SIZE);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    MatrixMulCUDA<M_BLOCK_SIZE><<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    // ====== 从 result.txt 读取期望结果 ======
    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Fault Injection Test Failed!\n");
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));
        return 0;
    }

    double *expected = (double *)malloc(mem_size_C);
    int count = 0;
    while (fscanf(file, "%lf", &expected[count]) == 1 && count < (int)(dimsC.x * dimsC.y)) {
        count++;
    }
    fclose(file);

    if (count != (int)(dimsC.x * dimsC.y)) {
        printf("Fault Injection Test Failed!\n");
        free(expected);
        checkCudaErrors(cudaFreeHost(h_A));
        checkCudaErrors(cudaFreeHost(h_B));
        checkCudaErrors(cudaFreeHost(h_C));
        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_B));
        checkCudaErrors(cudaFree(d_C));
        return 0;
    }

    // ====== 逐项比较结果 ======
    bool match = true;
    const double eps = 1e-12;
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
        double actual = h_C[i];
        double expected_val = expected[i];

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

    if (match)
        printf("Fault Injection Test Success!\n");
    else
        printf("Fault Injection Test Failed!\n");

    free(expected);
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}

// ================= 主函数 ==================
int main(int argc, char **argv) {
    int block_size = M_BLOCK_SIZE;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        fprintf(stderr, "Error: Matrix dimensions do not match for multiplication!\n");
        exit(EXIT_FAILURE);
    }

    MatrixMultiply(dimsA, dimsB);
    return 0;
}