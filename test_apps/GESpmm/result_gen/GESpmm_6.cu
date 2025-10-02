#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define SEED 6666
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

static inline float gen_subnormal() {
    float tiny = (float)(rand() % 100 + 1) * FLT_MIN / 1e5f;
    if (rand() % 2 == 0)
        tiny = -tiny;
    return tiny;
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
    std::vector<int> row_indices(nnz), col_indices(nnz);
    for (int i = 0; i < nnz; ++i) {
        row_indices[i] = rand() % A_nrows;
        col_indices[i] = rand() % A_ncols;
    }
    int *A_indptr = (int *)malloc((A_nrows + 1) * sizeof(int));
    int *A_indices = (int *)malloc(nnz * sizeof(int));
    float *A_data = (float *)malloc(nnz * sizeof(float));
    if (!A_indptr || !A_indices || !A_data) {
        fprintf(stderr, "Host malloc failed for A.\n");
        return 1;
    }
    for (int i = 0; i < A_nrows + 1; ++i)
        A_indptr[i] = 0;
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
        A_indptr[row_indices[n]] = ++ptr;
    }
    for (int n = A_nrows - 1; n > 0; --n) {
        A_indptr[n] = A_indptr[n - 1];
    }
    A_indptr[0] = 0;
    for (int i = 0; i < nnz; ++i) {
        A_data[i] = gen_subnormal();
    }
    float *B = (float *)malloc(A_ncols * B_ncols * sizeof(float));
    if (!B) {
        fprintf(stderr, "Host malloc failed for B.\n");
        free(A_indptr);
        free(A_indices);
        free(A_data);
        return 1;
    }
    for (int i = 0; i < A_ncols * B_ncols; ++i) {
        B[i] = gen_subnormal();
    }
    float *C = (float *)malloc(A_nrows * B_ncols * sizeof(float));
    if (!C) {
        fprintf(stderr, "Host malloc failed for C.\n");
        free(A_indptr);
        free(A_indices);
        free(A_data);
        free(B);
        return 1;
    }
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
    checkCudaError(cudaMemcpy(A_indptr_dev, A_indptr, (A_nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_indices_dev, A_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_data_dev, A_data, nnz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_dev, B, A_ncols * B_ncols * sizeof(float), cudaMemcpyHostToDevice));
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
    spmm_test0<float><<<gridDim, blockDim>>>(A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaMemcpy(C, C_dev, A_nrows * B_ncols * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < A_nrows * B_ncols; ++i) {
        printf("%.6f ", C[i]);
    }
    printf("\n");
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
