#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M_SEED 5311
#define BLOCK_SIZE 256

// kernel1: dot product
__global__ void kernel1(const __half *__restrict__ key, const __half *__restrict__ query,
                        __half *__restrict__ dot_product,
                        float *__restrict__ exp_sum, // float 累加
                        const int n, const int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += __half2float(key[i * d + j]) * __half2float(query[j]);
        }
        dot_product[i] = __float2half(sum);
        atomicAdd(exp_sum, expf(sum));
    }
}

// kernel2: softmax
__global__ void kernel2(const float *__restrict__ exp_sum, const __half *__restrict__ dot_product,
                        __half *__restrict__ score, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float dp = __half2float(dot_product[i]);
        float s = expf(dp) / exp_sum[0];
        score[i] = __float2half(s);
    }
}

// kernel3: weighted sum
__global__ void kernel3(const __half *__restrict__ score, const __half *__restrict__ value, __half *__restrict__ output,
                        const int n, const int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += __half2float(score[i]) * __half2float(value[i * d + j]);
        }
        output[j] = __float2half(sum);
    }
}

// device function
__half *attention_device(const __half *key, const __half *value, const __half *query, const int n, const int d,
                         const int repeat) {
    __half *d_key;
    cudaMalloc((void **)&d_key, n * d * sizeof(__half));
    cudaMemcpy(d_key, key, n * d * sizeof(__half), cudaMemcpyHostToDevice);

    __half *d_value;
    cudaMalloc((void **)&d_value, n * d * sizeof(__half));
    cudaMemcpy(d_value, value, n * d * sizeof(__half), cudaMemcpyHostToDevice);

    __half *d_query;
    cudaMalloc((void **)&d_query, d * sizeof(__half));
    cudaMemcpy(d_query, query, d * sizeof(__half), cudaMemcpyHostToDevice);

    __half *d_dot_product;
    cudaMalloc((void **)&d_dot_product, n * sizeof(__half));

    __half *d_score;
    cudaMalloc((void **)&d_score, n * sizeof(__half));

    float *d_exp_sum;
    cudaMalloc((void **)&d_exp_sum, sizeof(float));

    __half *output = (__half *)malloc(d * sizeof(__half));
    __half *d_output;
    cudaMalloc((void **)&d_output, d * sizeof(__half));

    dim3 n_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 n_block(BLOCK_SIZE);
    dim3 d_grid((d + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 d_block(BLOCK_SIZE);

    cudaDeviceSynchronize();

    for (int k = 0; k < repeat; k++) {
        cudaMemset(d_exp_sum, 0, sizeof(float));

        kernel1<<<n_grid, n_block>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
        kernel2<<<n_grid, n_block>>>(d_exp_sum, d_dot_product, d_score, n);
        kernel3<<<d_grid, d_block>>>(d_score, d_value, d_output, n, d);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, d * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_score);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_key);
    cudaFree(d_dot_product);
    cudaFree(d_exp_sum);
    return output;
}

// ---- 修改后的输入生成函数：包含 NaN ----
static void generate_input_with_nan(__half *buf, int total_len) {
    srand(M_SEED);
    for (int i = 0; i < total_len; i++) {
        if (rand() % 2 == 0) {
            buf[i] = __float2half(NAN); // 50% 概率 NaN
        } else {
            // 生成 -1.0f 到 1.0f 范围的随机浮点数
            float r = (float)rand() / (float)RAND_MAX; // [0,1]
            float val = r * 2.0f - 1.0f;               // 映射到 [-1,1]
            buf[i] = __float2half(val);
        }
    }
}
static void generate_input_with_nan_2_2(__half *buf, int total_len) {
    srand(M_SEED);
    for (int i = 0; i < total_len; i++) {
        if (rand() % 2 == 0) {
            buf[i] = __float2half(NAN); // 50% 概率 NaN
        } else {
            // 生成 -2.0f 到 2.0f 范围的随机浮点数
            float r = (float)rand() / (float)RAND_MAX; // [0,1]
            float val = r * 4.0f - 2.0f;               // 映射到 [-2,2]
            buf[i] = __float2half(val);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <rows> <columns>\n", argv[0]);
        return 1;
    }
    const int n = atoi(argv[1]);
    const int d = atoi(argv[2]);
    const int r = 1;

    __half *key = (__half *)malloc(n * d * sizeof(__half));
    __half *value = (__half *)malloc(n * d * sizeof(__half));
    __half *query = (__half *)malloc(d * sizeof(__half));

    generate_input_with_nan(key, n * d);
    generate_input_with_nan(value, n * d);
    generate_input_with_nan_2_2(query, d);

    __half *dout = attention_device(key, value, query, n, d, r);

    for (int i = 0; i < d; i++)
        printf("%.3f ", __half2float(dout[i]));
    printf("\n");
    free(key);
    free(value);
    free(query);
    free(dout);
    return 0;
}
