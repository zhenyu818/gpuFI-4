#define SEED 4364

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32  // Assuming standard warp size

// Utility to generate random floats in [-1, 1)
float* make_random_float(int size) {
    float* data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
    return data;
}

// ceil_div utility
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Kernels for layernorm forward pass (version 2 only)

__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Guard against extra threads in the last block to avoid OOB accesses
    int total = B * T * C;
    if (idx >= total) return;
    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

// Launcher for version 2
void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaDeviceSynchronize();
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaDeviceSynchronize();
    // in the normalization, everything just gets flattened out
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

    srand(SEED);

    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int C = atoi(argv[3]);

    int deviceIdx = 0;
    cudaSetDevice(deviceIdx);

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_mean, B * T * sizeof(float));
    cudaMalloc(&d_rstd, B * T * sizeof(float));
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    cudaMalloc(&d_weight, C * sizeof(float));
    cudaMalloc(&d_bias, C * sizeof(float));
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice);

    // Fixed block_size for version 2
    int block_size = 128;

    // Run the kernel once
    layernorm_forward2(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

    // Copy back results
    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, B * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rstd, d_rstd, B * T * sizeof(float), cudaMemcpyDeviceToHost);

    // Print all results: out (flattened), then mean, then rstd, separated by spaces
    for (int i = 0; i < B * T * C; ++i) {
        printf("%.6f ", out[i]);
    }
    for (int i = 0; i < B * T; ++i) {
        printf("%.6f ", mean[i]);
    }
    for (int i = 0; i < B * T; ++i) {
        printf("%.6f ", rstd[i]);  // No trailing space after last
    }
    printf("\n");

    // free memory
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