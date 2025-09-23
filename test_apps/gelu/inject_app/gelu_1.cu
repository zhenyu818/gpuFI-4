#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

#define M_SEED 9182
#define M_BLOCK_SIZE 1024

/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// width is hidden_dim and height is seq_len
__global__ void gelu_bias_loop(float* src, const float* bias, int width, int height)
{
    int batch = blockIdx.x;
    int x     = blockIdx.y;  // seq length
    int y     = threadIdx.x;

    if (x < height) {
        int index = batch * width * height + x * width;

        for (; y < width; y += blockDim.x) {
            float v_src  = src[index + y];
            float v_bias = bias[y];
            float v      = v_src + v_bias;

            // GELU近似公式
            float t = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));

            src[index + y] = t;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Usage: %s <batch> <sequence length> <hidden dimension> <repeat>\n", argv[0]);
        return 1;
    }

    const int batch_size = atoi(argv[1]);
    const int seq_len = atoi(argv[2]);
    const int hidden_dim = atoi(argv[3]);
    const int repeat = 1;

    const size_t src_size = (size_t)batch_size * seq_len * hidden_dim;

    const size_t src_size_bytes =  src_size * sizeof(float);
    const int bias_size_bytes = hidden_dim * sizeof(float);

    srand(M_SEED);
    float* output = (float*) malloc (src_size_bytes);
    for (size_t i = 0; i < src_size; i++) {
        output[i] = rand() / (float)RAND_MAX;
    }

    float* bias = (float*) malloc (bias_size_bytes);
    for (int i = 0; i < hidden_dim; i++) {
        bias[i] = -6.0f + (rand() % 12);
    }

    float* d_output;
    cudaMalloc((void**)&d_output, src_size_bytes);
    cudaMemcpy(d_output, output, src_size_bytes, cudaMemcpyHostToDevice);

    float* d_bias;
    cudaMalloc((void**)&d_bias, bias_size_bytes);
    cudaMemcpy(d_bias, bias, bias_size_bytes, cudaMemcpyHostToDevice);
  
    dim3 block(1024, 1);
    dim3 grid(batch_size, seq_len);

    cudaDeviceSynchronize();
    for (int i = 0; i < repeat; i++) {
        gelu_bias_loop <<<grid, block>>> (d_output, d_bias, hidden_dim, seq_len);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, src_size_bytes, cudaMemcpyDeviceToHost);

    // ==== 从 result.txt 读取参考值 ====
    FILE* file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Failed\n");
        cudaFree(d_output);
        cudaFree(d_bias);
        free(output);
        free(bias);
        return 1;
    }

    float* expected = (float*)malloc(sizeof(float) * src_size);
    int count = 0;
    while (fscanf(file, "%f", &expected[count]) == 1 && count < (int)src_size) {
        count++;
    }
    fclose(file);

    if (count != (int)src_size) {
        printf("Failed\n");
        free(expected);
        cudaFree(d_output);
        cudaFree(d_bias);
        free(output);
        free(bias);
        return 1;
    }

    // ==== 逐项比较 ====
    bool match = true;
    const float eps = 1e-5f;  // 容许误差
    for (size_t i = 0; i < src_size; i++) {
        if (fabs(output[i] - expected[i]) > eps) {
            match = false;
            break;
        }
    }

    if (match) {
        printf("Success\n");
    } else {
        printf("Failed\n");
    }

    free(expected);
  
    cudaFree(d_output);
    cudaFree(d_bias);
    free(output);
    free(bias);

    return 0;
}
