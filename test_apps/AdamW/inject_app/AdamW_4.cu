#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARSE_GROUP 4
#define SPARSE_KEEP 2
#define SEED 13579

void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}
__global__ void adamw_kernel2(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory,
                              long num_parameters, float learning_rate, float beta1, float beta2,
                              float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return;
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];
    m = lerp(grad, m, beta1);
    m_memory[i] = m;
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;
    m /= beta1_correction;
    v /= beta2_correction;
    params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}
static void gen_sparse_2to4(float *a, long n) {
    srand(SEED);
    for (long i = 0; i < n; i += SPARSE_GROUP) {
        bool keep[SPARSE_GROUP] = {0};
        int cnt = 0;
        while (cnt < SPARSE_KEEP) {
            int idx = rand() % SPARSE_GROUP;
            if (!keep[idx]) {
                keep[idx] = true;
                cnt++;
            }
        }
        for (int k = 0; k < SPARSE_GROUP && (i + k) < n; k++) {
            a[i + k] = keep[k] ? (((float)rand() / RAND_MAX) * 2.0f - 1.0f) : 0.0f;
        }
    }
}
static void gen_sparse_2to4_0_1(float *a, long n) {
    srand(SEED);
    for (long i = 0; i < n; i += SPARSE_GROUP) {
        bool keep[SPARSE_GROUP] = {0};
        int cnt = 0;
        while (cnt < SPARSE_KEEP) {
            int idx = rand() % SPARSE_GROUP;
            if (!keep[idx]) {
                keep[idx] = true;
                cnt++;
            }
        }
        for (int k = 0; k < SPARSE_GROUP && (i + k) < n; k++) {
            a[i + k] = keep[k] ? ((float)rand() / RAND_MAX) : 0.0f;
        }
    }
}
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <num_parameters> <t>\n", argv[0]);
        return 1;
    }
    long num_parameters = atol(argv[1]);
    int t = atoi(argv[2]);
    if (num_parameters <= 0 || t < 1)
        return 1;
    const float learning_rate = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, weight_decay = 0.0f;
    float beta1_correction = 1.0f - powf(beta1, t), beta2_correction = 1.0f - powf(beta2, t);
    float *params_memory = (float *)malloc(num_parameters * sizeof(float));
    float *grads_memory = (float *)malloc(num_parameters * sizeof(float));
    float *m_memory = (float *)malloc(num_parameters * sizeof(float));
    float *v_memory = (float *)malloc(num_parameters * sizeof(float));
    gen_sparse_2to4(params_memory, num_parameters);
    gen_sparse_2to4(grads_memory, num_parameters);
    gen_sparse_2to4(m_memory, num_parameters);
    gen_sparse_2to4_0_1(v_memory, num_parameters);

    float *d_params, *d_grads, *d_m, *d_v;
    cudaCheck(cudaMalloc(&d_params, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grads, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v, num_parameters * sizeof(float)));
    cudaCheck(cudaMemcpy(d_params, params_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grads, grads_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m, m_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v, v_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(512);
    dim3 grid((num_parameters + block.x - 1) / block.x);
    adamw_kernel2<<<grid, block>>>(d_params, d_grads, d_m, d_v, num_parameters, learning_rate, beta1, beta2,
                                   beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(params_memory, d_params, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m_memory, d_m, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(v_memory, d_v, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));
    long total = 3 * num_parameters;
    float *new_results = (float *)malloc(total * sizeof(float));
    memcpy(new_results, params_memory, num_parameters * sizeof(float));
    memcpy(new_results + num_parameters, m_memory, num_parameters * sizeof(float));
    memcpy(new_results + 2 * num_parameters, v_memory, num_parameters * sizeof(float));
    FILE *fp = fopen("result.txt", "r");
    if (!fp) {
        printf("Fault Injection Test Failed!\n");
        return 1;
    }
    float *ref = (float *)malloc(total * sizeof(float));
    long idx = 0;
    while (idx < total && fscanf(fp, "%f", &ref[idx]) == 1)
        idx++;
    fclose(fp);
    if (idx != total) {
        printf("Fault Injection Test Failed!\n");
        free(ref);
        free(new_results);
        return 1;
    }
    bool match = true;
    const float tol = 1e-5f;
    for (long i = 0; i < total; i++) {
        float a = new_results[i], b = ref[i];
        if (isnan(a) && isnan(b))
            continue;
        else if (isinf(a) && isinf(b) && signbit(a) == signbit(b))
            continue;
        else if (!isnan(a) && !isnan(b) && !isinf(a) && !isinf(b)) {
            if (fabs(a - b) > tol) {
                match = false;
                break;
            }
        } else {
            match = false;
            break;
        }
    }
    printf(match ? "Fault Injection Test Success!\n" : "Fault Injection Test Failed!\n");
    free(ref);
    free(new_results);
    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);
    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_grads));
    cudaCheck(cudaFree(d_m));
    cudaCheck(cudaFree(d_v));
    return 0;
}
