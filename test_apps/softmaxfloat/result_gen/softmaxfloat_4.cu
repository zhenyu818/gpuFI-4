#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <stdbool.h>

#define BLOCK_SIZE 256
#define M_SEED 3608
#define SPARSE_N 2    // 每组保留的非零数
#define SPARSE_M 4    // 每组的元素个数（2:4 稀疏）

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

// ---- 生成 2:4 稀疏分组 ----
static void generate_2to4_sparse_group(float *group, int group_len) {
  for (int k = 0; k < group_len; k++) group[k] = 0.0f;

  bool selected[SPARSE_M] = {false};
  int selected_count = 0;
  while (selected_count < SPARSE_N) {
    int idx = rand() % SPARSE_M;
    if (!selected[idx]) {
      selected[idx] = true;
      selected_count++;
    }
  }

  for (int k = 0; k < group_len; k++) {
    if (selected[k]) {
      group[k] = static_cast<float>(rand() % 10);
      while (group[k] == 0.0f) {
        group[k] = static_cast<float>(rand() % 10);
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

  // ---- 确保 sliceSize 为 4 的倍数 ----
  if (sliceSize % SPARSE_M != 0) {
    int new_size = (sliceSize / SPARSE_M + 1) * SPARSE_M;
    printf("Warning: sliceSize (%d) not multiple of %d, auto-adjust to %d\n",
           sliceSize, SPARSE_M, new_size);
    sliceSize = new_size;
    numElem = numSlice * sliceSize;
  }

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(M_SEED);
  for (int i = 0; i < numSlice; i++) {
    for (int j = 0; j < sliceSize; j += SPARSE_M) {
      float group[SPARSE_M];
      generate_2to4_sparse_group(group, SPARSE_M);
      for (int k = 0; k < SPARSE_M; k++) {
        input[i * sliceSize + j + k] = group[k];
      }
    }
  }

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

  // ---------------- 直接输出结果 ----------------
  for (int i = 0; i < numElem; i++) {
      printf("%.6f%c", output_gpu[i], (i == numElem - 1) ? '\n' : ' ');
  }

  free(input);
  free(output_gpu);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
