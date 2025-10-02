#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

#define M_SEED 9182
#define M_BLOCK_SIZE 1024

// ---- 生成 2:4 稀疏分组 ----
static void generate_2to4_sparse_group(float *group, int group_len) {
  for (int k = 0; k < group_len; k++) group[k] = 0.0f;

  bool selected[4] = {false};
  int selected_count = 0;
  while (selected_count < 2) {
    int idx = rand() % 4;
    if (!selected[idx]) {
      selected[idx] = true;
      selected_count++;
    }
  }

  for (int k = 0; k < group_len; k++) {
    if (selected[k]) {
      float val = (rand() % 200 - 100) / 100.0f; // [-1,1] 范围
      group[k] = val;
    }
  }
}

// ---- 使用 2:4 稀疏初始化 ----
static void RandomInitSparse2to4(float *data, size_t size) {
  srand(M_SEED);
  for (size_t i = 0; i < size; i += 4) {
    float group[4];
    generate_2to4_sparse_group(group, 4);
    for (int k = 0; k < 4 && i + k < size; k++) {
      data[i + k] = group[k];
    }
  }
}

/*
 * CUDA Kernel: GELU + bias
 */
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
    printf("Usage: %s <batch> <sequence length> <hidden dimension>\n", argv[0]);
    return 1;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int hidden_dim = atoi(argv[3]);
  const int repeat = 1;

  const size_t src_size = (size_t)batch_size * seq_len * hidden_dim;
  const size_t src_size_bytes =  src_size * sizeof(float);
  const int bias_size_bytes = hidden_dim * sizeof(float);

  // ---- 修改: 2:4 稀疏输入 ----
  float* output = (float*) malloc (src_size_bytes);
  RandomInitSparse2to4(output, src_size);

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

  for (size_t i = 0; i < src_size; i++) {
      printf("%.6f ", output[i]);  // 打印每一项
  }

  cudaFree(d_output);
  cudaFree(d_bias);
  free(output);
  free(bias);

  return 0;
}
