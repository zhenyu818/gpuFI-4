#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <climits>   // for INT_MAX, INT_MIN
#include <cfloat>    // for INFINITY, NAN

#define BLOCK_SIZE 256
#define M_SEED 3608

__global__
void softMax(const int numSlice, const int sliceSize,
             const float* src, float* dest)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;

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

// ---- 输入生成函数：对抗性模式 ----
static void generate_input_with_special(float* input, int numSlice, int sliceSize) {
  srand(M_SEED);
  for (int i = 0; i < numSlice; i++) {
    int pattern = rand() % 5;  // 随机选择一种对抗模式
    for (int j = 0; j < sliceSize; j++) {
      switch (pattern) {
        case 0:
          // 模式0: 极大值 vs 极小值
          input[i * sliceSize + j] = (j % 2 == 0) ? 1e9f : -1e9f;
          break;

        case 1:
          // 模式1: 单点极大值 (one-hot-like)
          input[i * sliceSize + j] = (j == 0) ? 1e6f : -1e6f;
          break;

        case 2:
          // 模式2: 全相等值
          input[i * sliceSize + j] = 123.456f;  // 任意常数
          break;

        case 3:
          // 模式3: 交替正负
          input[i * sliceSize + j] = (j % 2 == 0) ? (float)(rand() % 100) : -(float)(rand() % 100);
          break;

        case 4:
          // 模式4: 特殊浮点值
          if (j % 3 == 0)
            input[i * sliceSize + j] = INFINITY;   // +Inf
          else if (j % 3 == 1)
            input[i * sliceSize + j] = -INFINITY;  // -Inf
          else
            input[i * sliceSize + j] = NAN;        // NaN
          break;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of slices> <slice size>\n", argv[0]);
    return 1;
  }
   
  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int repeat = 1;
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  // ---- 使用对抗性输入生成 ----
  generate_input_with_special(input, numSlice, sliceSize);

  float *d_input, *d_output;
  cudaMalloc((void**)&d_input, sizeof(float) * numElem);
  cudaMalloc((void**)&d_output, sizeof(float) * numElem);
  cudaMemcpy(d_input, input, sizeof(float) * numElem, cudaMemcpyHostToDevice);

  dim3 global_work_size ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  dim3 local_work_size (BLOCK_SIZE);

  cudaDeviceSynchronize();

  for (int n = 0; n < repeat; n++) {
    softMax<<<global_work_size, local_work_size>>>(numSlice, sliceSize, d_input, d_output);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(output_gpu, d_output, sizeof(float) * numElem, cudaMemcpyDeviceToHost);

  // ===== 从 result.txt 读取期望结果 =====
  FILE *file = fopen("result.txt", "r");
  if (file == NULL) {
    printf("Failed\n");
    free(input);
    free(output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  float *expected = (float*) malloc(sizeof(float) * numElem);
  int count = 0;
  while (fscanf(file, "%f", &expected[count]) == 1 && count < numElem) {
    count++;
  }
  fclose(file);

  if (count != numElem) {
    printf("Failed\n");
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
    printf("Success\n");
  } else {
    printf("Failed\n");
  }

  free(input);
  free(output_gpu);
  free(expected);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
