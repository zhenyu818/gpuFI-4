#include <chrono>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_SEED 9182
#define M_BLOCK_SIZE 1024

// ---------- 工具：生成 (0,1] 内随机浮点 ----------
static inline float rand_open0_closed1_nonzero() {
    return ((float)(rand() + 1)) / ((float)RAND_MAX + 1.0f);
}

// ---------- 2:4 稀疏下标选择 ----------
static void pick_2_of_4(bool keep[4]) {
    keep[0] = keep[1] = keep[2] = keep[3] = false;
    int cnt = 0;
    while (cnt < 2) {
        int idx = rand() % 4;
        if (!keep[idx]) {
            keep[idx] = true;
            cnt++;
        }
    }
}

// ---------- output: 2:4 稀疏，非零取 (0,1] ----------
static void gen_sparse_2to4_output(float *a, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        bool keep[4];
        pick_2_of_4(keep);
        for (int k = 0; k < 4 && (i + k) < n; ++k) {
            a[i + k] = keep[k] ? rand_open0_closed1_nonzero() : 0.0f;
        }
    }
}

// ---------- bias: 2:4 稀疏，非零取 {-6..-1,1..5} ----------
static inline float sample_bias_nonzero() {
    int r = rand() % 11;                 // 0..10
    int v = (r < 6) ? (r - 6) : (r - 5); // 映射 [-6..-1] U [1..5]
    return (float)v;
}

static void gen_sparse_2to4_bias(float *a, int n) {
    for (int i = 0; i < n; i += 4) {
        bool keep[4];
        pick_2_of_4(keep);
        for (int k = 0; k < 4 && (i + k) < n; ++k) {
            a[i + k] = keep[k] ? sample_bias_nonzero() : 0.0f;
        }
    }
}

/*
 * CUDA Kernel: GELU + bias
 */
__global__ void gelu_bias_loop(float *src, const float *bias, int width, int height) {
    int batch = blockIdx.x;
    int x = blockIdx.y; // seq length
    int y = threadIdx.x;

    if (x < height) {
        int index = batch * width * height + x * width;

        for (; y < width; y += blockDim.x) {
            float v_src = src[index + y];
            float v_bias = bias[y];
            float v = v_src + v_bias;

            // GELU近似公式
            float t = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));

            src[index + y] = t;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <batch> <sequence length> <hidden dimension>\n", argv[0]);
        return 1;
    }

    const int batch_size = atoi(argv[1]);
    const int seq_len = atoi(argv[2]);
    const int hidden_dim = atoi(argv[3]);
    const int repeat = 1;

    const size_t src_size = (size_t)batch_size * seq_len * hidden_dim;
    const size_t src_size_bytes = src_size * sizeof(float);
    const int bias_size_bytes = hidden_dim * sizeof(float);

    srand(M_SEED);

    // ---- 修改: 2:4 稀疏输入 ----
    float *output = (float *)malloc(src_size_bytes);
    gen_sparse_2to4_output(output, src_size);

    float *bias = (float *)malloc(bias_size_bytes);
    gen_sparse_2to4_bias(bias, hidden_dim);

    float *d_output;
    cudaMalloc((void **)&d_output, src_size_bytes);
    cudaMemcpy(d_output, output, src_size_bytes, cudaMemcpyHostToDevice);

    float *d_bias;
    cudaMalloc((void **)&d_bias, bias_size_bytes);
    cudaMemcpy(d_bias, bias, bias_size_bytes, cudaMemcpyHostToDevice);

    dim3 block(1024, 1);
    dim3 grid(batch_size, seq_len);

    cudaDeviceSynchronize();
    for (int i = 0; i < repeat; i++) {
        gelu_bias_loop<<<grid, block>>>(d_output, d_bias, hidden_dim, seq_len);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, src_size_bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < src_size; i++) {
        printf("%.6f ", output[i]); // 打印每一项
    }

    cudaFree(d_output);
    cudaFree(d_bias);
    free(output);
    free(bias);

    return 0;
}
