#include <chrono>
#include <cuda.h>
#include <math.h> // <- fabsf/fabs
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_SEED 9182
#define M_BLOCK_SIZE 1024

// ---------- 工具：在 [0,1] 内生成非零浮点 ----------
static inline float rand_open0_closed1_nonzero() {
    // 取 (0,1]，保证非零： (rand()+1)/(RAND_MAX+1.0f)
    // 注意：当 rand()==RAND_MAX 时恰为 1.0
    return ((float)(rand() + 1)) / ((float)RAND_MAX + 1.0f);
}

// ---------- 每组 4 中选 2 个下标 ----------
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

// ---------- 2:4 稀疏生成：output 非零取 (0,1] ----------
static void gen_sparse_2to4_output(float *a, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        bool keep[4];
        pick_2_of_4(keep);
        for (int k = 0; k < 4 && (i + k) < n; ++k) {
            a[i + k] = keep[k] ? rand_open0_closed1_nonzero() : 0.0f;
        }
    }
}

// ---------- 2:4 稀疏生成：bias 非零取 {-6..-1,1..5}（排除 0） ----------
static inline float sample_bias_nonzero() {
    // 从 11 个值中采样：{-6,-5,-4,-3,-2,-1, 1,2,3,4,5}
    int r = rand() % 11;                 // 0..10
    int v = (r < 6) ? (r - 6) : (r - 5); // 映射到 [-6..-1] U [1..5]
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

            // GELU 近似
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
    const size_t bias_size = (size_t)hidden_dim;
    const size_t bias_size_bytes = bias_size * sizeof(float);

    // 统一设种子，保证可复现
    srand(M_SEED);

    // ---- 2:4 稀疏输入（非零∈(0,1]）----
    float *output = (float *)malloc(src_size_bytes);
    gen_sparse_2to4_output(output, src_size);

    // ---- 2:4 稀疏 bias（非零∈{-6..-1,1..5}）----
    float *bias = (float *)malloc(bias_size_bytes);
    gen_sparse_2to4_bias(bias, hidden_dim);

    // ---- CUDA 内存与 kernel 调用 ----
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

    // ==== 读取并对比 ====
    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Fault Injection Test Failed!\n");
        cudaFree(d_output);
        cudaFree(d_bias);
        free(output);
        free(bias);
        return 1;
    }

    float *expected = (float *)malloc(sizeof(float) * src_size);
    size_t count = 0;
    while (count < src_size && fscanf(file, "%f", &expected[count]) == 1) {
        count++;
    }
    fclose(file);

    if (count != src_size) {
        printf("Fault Injection Test Failed!\n");
        free(expected);
        cudaFree(d_output);
        cudaFree(d_bias);
        free(output);
        free(bias);
        return 1;
    }

    bool match = true;
    const float eps = 1e-5f;
    for (size_t i = 0; i < src_size; i++) {
        if (fabsf(output[i] - expected[i]) > eps) { // 用 fabsf
            match = false;
            break;
        }
    }

    if (match) {
        printf("Fault Injection Test Success!\n");
    } else {
        printf("Fault Injection Test Failed!\n");
    }

    free(expected);
    cudaFree(d_output);
    cudaFree(d_bias);
    free(output);
    free(bias);

    return 0;
}
