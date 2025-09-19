#include <chrono> 
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <climits>
#include <cfloat>    // for FLT_MIN

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

// ---- 输入生成函数：只生成非正规数 ----
static void generate_input_subnormal(float* input, int numSlice, int sliceSize) {
  srand(M_SEED);
  for (int i = 0; i < numSlice; i++) {
    for (int j = 0; j < sliceSize; j++) {
      // 生成比 FLT_MIN 更小的正数，属于 subnormal
      float tiny = (float)(rand() % 100 + 1) * FLT_MIN / 1e5f;
      input[i * sliceSize + j] = tiny;
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

  // ---- 使用非正规数输入生成 ----
  generate_input_subnormal(input, numSlice, sliceSize);

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
    printf("Failed (cannot open result.txt)\n");
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
    printf("Failed (result.txt does not match expected size)\n");
    free(input);
    free(output_gpu);
    free(expected);
    cudaFree(d_input);
    cudaFree(d_output);
    return 1;
  }

  // ===== 逐项比对结果 =====
  bool match = true;
  const float eps = 1e-6;
  for (int i = 0; i < numElem; i++) {
    if (fabs(output_gpu[i] - expected[i]) > eps) {
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
