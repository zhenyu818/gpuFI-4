#define SEED 4444

#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define GROUP 4
#define KEEP 2

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
bool approx_equal(float a, float b) {
    if (isnan(a) || isnan(b))
        return isnan(a) && isnan(b);
    if (isinf(a) || isinf(b))
        return isinf(a) && isinf(b) && ((a > 0.0f) == (b > 0.0f));
    return fabsf(a - b) <= 1e-5f;
}
static void gen_2to4(float *a, size_t n) {
    srand(SEED);
    for (size_t i = 0; i < n; i += GROUP) {
        bool keep[GROUP] = {0};
        int cnt = 0;
        while (cnt < KEEP) {
            int idx = rand() % GROUP;
            if (!keep[idx]) {
                keep[idx] = 1;
                cnt++;
            }
        }
        for (int k = 0; k < GROUP && (i + k) < n; k++) {
            a[i + k] = keep[k] ? (((float)rand() / RAND_MAX) * 2.0f - 1.0f) : 0.0f;
        }
    }
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
    gen_2to4(inp, (size_t)B * T * C);
    gen_2to4(weight, (size_t)C);
    gen_2to4(bias, (size_t)C);
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
    FILE *fp = fopen("result.txt", "r");
    if (!fp) {
        printf("Fault Injection Test Failed!\n");
        return 1;
    }
    float *ref_out = (float *)malloc(B * T * C * sizeof(float));
    float *ref_mean = (float *)malloc(N * sizeof(float));
    float *ref_rstd = (float *)malloc(N * sizeof(float));
    bool ok = true;
    for (int i = 0; i < B * T * C; ++i) {
        if (fscanf(fp, "%f", &ref_out[i]) != 1) {
            ok = false;
            break;
        }
    }
    if (ok) {
        for (int i = 0; i < N; ++i) {
            if (fscanf(fp, "%f", &ref_mean[i]) != 1) {
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        for (int i = 0; i < N; ++i) {
            if (fscanf(fp, "%f", &ref_rstd[i]) != 1) {
                ok = false;
                break;
            }
        }
    }
    fclose(fp);
    if (!ok) {
        printf("Fault Injection Test Failed!\n");
        free(ref_out);
        free(ref_mean);
        free(ref_rstd);
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
        return 1;
    }
    bool success = true;
    for (int i = 0; i < B * T * C && success; ++i) {
        if (!approx_equal(out[i], ref_out[i]))
            success = false;
    }
    for (int i = 0; i < N && success; ++i) {
        if (!approx_equal(mean[i], ref_mean[i]))
            success = false;
    }
    for (int i = 0; i < N && success; ++i) {
        if (!approx_equal(rstd[i], ref_rstd[i]))
            success = false;
    }
    printf(success ? "Fault Injection Test Success!\n" : "Fault Injection Test Failed!\n");
    free(ref_out);
    free(ref_mean);
    free(ref_rstd);
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
