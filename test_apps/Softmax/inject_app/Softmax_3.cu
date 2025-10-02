#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define BLOCK_SIZE 256
#define M_SEED 3608

__global__ void softMax(const int numSlice, const int sliceSize, const float *src, float *dest) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSlice)
        return;

    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
        max_ = max(max_, src[i * sliceSize + j]);
    }

    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
        sum += expf(src[i * sliceSize + j] - max_);
    }

    for (int j = 0; j < sliceSize; j++) {
        dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number of slices> <slice size>\n", argv[0]);
        return 1;
    }

    int numSlice = atoi(argv[1]);
    int sliceSize = atoi(argv[2]);
    int repeat = 1;
    int numElem = numSlice * sliceSize;

    float *input = (float *)aligned_alloc(1024, sizeof(float) * numElem);
    float *output_gpu = (float *)aligned_alloc(1024, sizeof(float) * numElem);

    srand(M_SEED);
    for (int i = 0; i < numSlice; i++)
        for (int j = 0; j < sliceSize; j++)
            input[i * sliceSize + j] = 1;

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, sizeof(float) * numElem);
    cudaMalloc((void **)&d_output, sizeof(float) * numElem);
    cudaMemcpy(d_input, input, sizeof(float) * numElem, cudaMemcpyHostToDevice);

    dim3 global_work_size((numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
    dim3 local_work_size(BLOCK_SIZE);

    cudaDeviceSynchronize();

    for (int n = 0; n < repeat; n++) {
        softMax<<<global_work_size, local_work_size>>>(numSlice, sliceSize, d_input, d_output);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output_gpu, d_output, sizeof(float) * numElem, cudaMemcpyDeviceToHost);

    // ===== 从 result.txt 读取期望结果 =====
    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Fault Injection Test Failed!\n");
        free(input);
        free(output_gpu);
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    float *expected = (float *)malloc(sizeof(float) * numElem);
    int count = 0;
    while (fscanf(file, "%f", &expected[count]) == 1 && count < numElem) {
        count++;
    }
    fclose(file);

    if (count != numElem) {
        printf("Fault Injection Test Failed!\n");
        free(input);
        free(output_gpu);
        free(expected);
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    // ===== 逐项比对结果，支持 NaN / Inf =====
    bool match = true;
    const float eps = 1e-5f;
    for (int i = 0; i < numElem; i++) {
        float actual = output_gpu[i];
        float expected_val = expected[i];

        if (isnan(actual) && isnan(expected_val)) {
            continue; // 两个 NaN 算相等
        }
        if (isnan(actual) || isnan(expected_val)) {
            match = false;
            break;
        }

        if (isinf(actual) && isinf(expected_val)) {
            if (signbit(actual) != signbit(expected_val)) {
                match = false; // +Inf vs -Inf 不相等
                break;
            } else {
                continue; // 两个同号 Inf 相等
            }
        }
        if (isinf(actual) || isinf(expected_val)) {
            match = false; // 一个是 Inf，一个不是
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

    free(input);
    free(output_gpu);
    free(expected);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
