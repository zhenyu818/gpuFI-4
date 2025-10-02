#define SEED 6666

#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
static inline float gen_subnormal() {
    float tiny = (float)(rand() % 100 + 1) * FLT_MIN / 1e5f;
    if (rand() % 2 == 0)
        tiny = -tiny;
    return tiny;
}
__global__ void mean_kernel(float *mean, const float *inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}
__global__ void rstd_kernel(float *rstd, const float *inp, const float *mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;
    float m = mean[idx];
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}
__global__ void normalization_kernel(float *out, const float *inp, float *mean, float *rstd, const float *weight,
                                     const float *bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total)
        return;
    int bt = idx / C;
    int c = idx % C;
    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];
    out[idx] = o;
}
void layernorm_forward2(float *out, float *mean, float *rstd, const float *inp, const float *weight, const float *bias,
                        int B, int T, int C, const int block_size) {
    int N = B * T;
    mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaDeviceSynchronize();
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaDeviceSynchronize();
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaDeviceSynchronize();
}
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s B T C\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int C = atoi(argv[3]);
    int N = B * T;
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *mean = (float *)malloc(N * sizeof(float));
    float *rstd = (float *)malloc(N * sizeof(float));
    float *inp = (float *)malloc(B * T * C * sizeof(float));
    float *weight = (float *)malloc(C * sizeof(float));
    float *bias = (float *)malloc(C * sizeof(float));
    srand(SEED);
    for (int i = 0; i < B * T * C; ++i)
        inp[i] = gen_subnormal();
    for (int i = 0; i < C; ++i) {
        weight[i] = gen_subnormal();
        bias[i] = gen_subnormal();
    }
    float *d_out, *d_mean, *d_rstd, *d_inp, *d_weight, *d_bias;
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_mean, N * sizeof(float));
    cudaMalloc(&d_rstd, N * sizeof(float));
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    cudaMalloc(&d_weight, C * sizeof(float));
    cudaMalloc(&d_bias, C * sizeof(float));
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 128;
    layernorm_forward2(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rstd, d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < B * T * C; ++i)
        printf("%.6f ", out[i]);
    for (int i = 0; i < N; ++i)
        printf("%.6f ", mean[i]);
    for (int i = 0; i < N; ++i)
        printf("%.6f ", rstd[i]);
    printf("\n");
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_bias);
    return 0;
}
