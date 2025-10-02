#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#define SEED 42
#define TILE_ROW 4

#define checkCudaError(a)                                                                                              \
    do {                                                                                                               \
        if (cudaSuccess != (a)) {                                                                                      \
            fprintf(stderr, "CUDA error in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));          \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

template <typename T>
__global__ void spmm_test0(int A_nrows, int B_ncols, int *A_csrRowPtr, int *A_csrColInd, T *A_csrVal, T *B_dnVal,
                           T *C_dnVal) {
    int rid = blockDim.y * blockIdx.x + threadIdx.y;
    if (rid < A_nrows) {
        int cid = (blockIdx.y << 5) + threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid + 1)];
        int offset = 0;
        T acc = 0;
        if (blockIdx.y != gridDim.y - 1) {
            for (int ptr = lb; ptr < hb; ptr++) {
                offset = A_csrColInd[ptr] * B_ncols + cid;
                acc += A_csrVal[ptr] * B_dnVal[offset];
            }
            C_dnVal[(rid * B_ncols + cid)] = acc;
        } else {
            for (int ptr = lb; ptr < hb; ptr++) {
                if (cid < B_ncols) {
                    offset = A_csrColInd[ptr] * B_ncols + cid;
                }
                acc += A_csrVal[ptr] * B_dnVal[offset];
            }
            if (cid < B_ncols) {
                C_dnVal[(rid * B_ncols + cid)] = acc;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <A_nrows> <A_ncols> <B_ncols> <nnz>\n", argv[0]);
        return 1;
    }

    int A_nrows = atoi(argv[1]);
    int A_ncols = atoi(argv[2]);
    int B_ncols = atoi(argv[3]);
    int nnz = atoi(argv[4]);

    if (nnz > A_nrows * A_ncols || nnz < 0 || A_nrows <= 0 || A_ncols <= 0 || B_ncols <= 0) {
        fprintf(stderr, "Invalid dimensions or nnz.\n");
        return 1;
    }

    srand(SEED);

    // 生成 COO 格式的 A
    std::vector<int> row_indices(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<float> values(nnz);
    for (int i = 0; i < nnz; ++i) {
        row_indices[i] = rand() % A_nrows;
        col_indices[i] = rand() % A_ncols;
        values[i] = (float)(rand() % 1000 - 500) / 500.0f; // -1.0 到 1.0
    }

    // 主机分配 CSR 格式
    int *A_indptr = (int *)malloc((A_nrows + 1) * sizeof(int));
    int *A_indices = (int *)malloc(nnz * sizeof(int));
    float *A_data = (float *)malloc(nnz * sizeof(float));
    if (!A_indptr || !A_indices || !A_data) {
        fprintf(stderr, "Host malloc failed for A.\n");
        return 1;
    }

    // COO to CSR 转换
    for (int i = 0; i < A_nrows + 1; ++i) {
        A_indptr[i] = 0;
    }
    for (int n = 0; n < nnz; ++n) {
        int row = row_indices[n];
        A_indptr[row + 1]++;
    }
    for (int n = 1; n < A_nrows + 1; ++n) {
        A_indptr[n] += A_indptr[n - 1];
    }
    for (int n = 0; n < nnz; ++n) {
        int ptr = A_indptr[row_indices[n]];
        A_indices[ptr] = col_indices[n];
        A_data[ptr] = values[n];
        A_indptr[row_indices[n]] = ++ptr;
    }
    for (int n = A_nrows - 1; n > 0; --n) {
        A_indptr[n] = A_indptr[n - 1];
    }
    A_indptr[0] = 0;

    // 生成 B
    float *B = (float *)malloc(A_ncols * B_ncols * sizeof(float));
    if (!B) {
        fprintf(stderr, "Host malloc failed for B.\n");
        free(A_indptr);
        free(A_indices);
        free(A_data);
        return 1;
    }
    for (int i = 0; i < A_ncols * B_ncols; ++i) {
        B[i] = (float)(rand() % 100 - 50) / 50.0f; // -1.0 到 1.0
    }

    // 主机分配 C
    float *C = (float *)malloc(A_nrows * B_ncols * sizeof(float));
    if (!C) {
        fprintf(stderr, "Host malloc failed for C.\n");
        free(A_indptr);
        free(A_indices);
        free(A_data);
        free(B);
        return 1;
    }

    // 设备内存分配
    int *A_indptr_dev;
    int *A_indices_dev;
    float *A_data_dev;
    float *B_dev;
    float *C_dev;

    checkCudaError(cudaMalloc(&A_indptr_dev, (A_nrows + 1) * sizeof(int)));
    checkCudaError(cudaMalloc(&A_indices_dev, nnz * sizeof(int)));
    checkCudaError(cudaMalloc(&A_data_dev, nnz * sizeof(float)));
    checkCudaError(cudaMalloc(&B_dev, A_ncols * B_ncols * sizeof(float)));
    checkCudaError(cudaMalloc(&C_dev, A_nrows * B_ncols * sizeof(float)));

    // H2D 复制
    checkCudaError(cudaMemcpy(A_indptr_dev, A_indptr, (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_indices_dev, A_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_data_dev, A_data, nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_dev, B, A_ncols * B_ncols * sizeof(float), cudaMemcpyHostToDevice));

    // 内核启动配置
    dim3 blockDim;
    dim3 gridDim;
    gridDim.x = (A_nrows + TILE_ROW - 1) / TILE_ROW;
    if (B_ncols > 32) {
        gridDim.y = (B_ncols + 31) / 32;
        blockDim.x = 32;
    } else {
        gridDim.y = 1;
        blockDim.x = B_ncols;
    }
    blockDim.y = TILE_ROW;

    // 启动内核一次
    spmm_test0<float><<<gridDim, blockDim>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaGetLastError());

    // D2H 复制 C
    checkCudaError(cudaMemcpy(C, C_dev, A_nrows * B_ncols * sizeof(float), cudaMemcpyDeviceToHost));

    // 读取 result.txt 并解析参考结果
    std::vector<float> ref(A_nrows * B_ncols);
    std::ifstream in("result.txt");
    if (!in.is_open()) {
        fprintf(stderr, "Failed to open result.txt\n");
        free(A_indptr);
        free(A_indices);
        free(A_data);
        free(B);
        free(C);
        checkCudaError(cudaFree(A_indptr_dev));
        checkCudaError(cudaFree(A_indices_dev));
        checkCudaError(cudaFree(A_data_dev));
        checkCudaError(cudaFree(B_dev));
        checkCudaError(cudaFree(C_dev));
        return 1;
    }

    std::string token;
    size_t idx = 0;
    while (in >> token && idx < ref.size()) {
        float val;
        if (token == "inf" || token == "Inf" || token == "+Inf") {
            val = std::numeric_limits<float>::infinity();
        } else if (token == "-inf" || token == "-Inf") {
            val = -std::numeric_limits<float>::infinity();
        } else if (token == "nan" || token == "NaN" || token == "NAN") {
            val = std::numeric_limits<float>::quiet_NaN();
        } else {
            try {
                val = std::stof(token);
            } catch (...) {
                in.close();
                fprintf(stderr, "Failed to parse value in result.txt\n");
                return 1;
            }
        }
        ref[idx++] = val;
    }
    in.close();

    if (idx != ref.size()) {
        fprintf(stderr, "Mismatch in number of elements from result.txt\n");
        return 1;
    }

    // 比对 C 和 ref
    bool match = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        float a = C[i];
        float b = ref[i];
        bool eq = false;

        if (std::isnan(a) && std::isnan(b)) {
            eq = true;
        } else if (std::isinf(a) && std::isinf(b)) {
            // 检查符号相同的 inf
            eq = (a > 0.0f) == (b > 0.0f);
        } else if (!std::isnan(a) && !std::isinf(a) && !std::isnan(b) && !std::isinf(b)) {
            eq = std::fabs(a - b) < 1e-5f;
        } else {
            eq = false;
        }

        if (!eq) {
            match = false;
            break;
        }
    }

    if (match) {
        printf("Fault Injection Test Success!\n");
    } else {
        printf("Fault Injection Test Failed!\n");
    }

    // 清理
    free(A_indptr);
    free(A_indices);
    free(A_data);
    free(B);
    free(C);
    checkCudaError(cudaFree(A_indptr_dev));
    checkCudaError(cudaFree(A_indices_dev));
    checkCudaError(cudaFree(A_data_dev));
    checkCudaError(cudaFree(B_dev));
    checkCudaError(cudaFree(C_dev));

    return 0;
}