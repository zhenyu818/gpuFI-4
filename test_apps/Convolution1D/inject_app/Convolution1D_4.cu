/*
  Reference
  Chapter 7 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <iostream>

#define M_SEED 6498
#define BLOCK_SIZE 256
#define MAX_MASK_WIDTH 10
#define TILE_SIZE BLOCK_SIZE
#define SPARSE_N 2
#define SPARSE_M 4

template<typename T>
__constant__ T mask [MAX_MASK_WIDTH];

template<typename T>
__global__
void conv1d_tiled_caching(const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  __shared__ T tile[TILE_SIZE];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  tile[threadIdx.x] = in[i];
  __syncthreads();

  int this_tile_start = blockIdx.x * blockDim.x;
  int next_tile_start = (blockIdx.x + 1) * blockDim.x;
  int start = i - (mask_width / 2);
  T s = 0;
  for (int j = 0; j < mask_width; j++) {
    int in_index = start + j;
    if (in_index >= 0 && in_index < input_width) {
      if (in_index >= this_tile_start && in_index < next_tile_start) {
        s += tile[threadIdx.x + j - (mask_width / 2)] * mask<T>[j];
      } else {
        s += in[in_index] * mask<T>[j];
      }
    }
  }
  out[i] = s;
}

/**
 * ---- 新增: 2:4 稀疏输入生成 ----
 */
template<typename T>
static void generate_2to4_sparse(T *data, int size) {
  srand(M_SEED);
  for (int i = 0; i < size; i += SPARSE_M) {
    bool selected[SPARSE_M] = {false};
    int selected_count = 0;
    // 随机挑 2 个位置保留非零
    while (selected_count < SPARSE_N) {
      int idx = rand() % SPARSE_M;
      if (!selected[idx]) {
        selected[idx] = true;
        selected_count++;
      }
    }
    for (int k = 0; k < SPARSE_M && i + k < size; k++) {
      if (selected[k]) {
        data[i + k] = (T)((rand() % 10) + 1);  // 非零范围 [1,10]
      } else {
        data[i + k] = (T)0;
      }
    }
  }
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); // input
  b = (T *)malloc(size_bytes); // output

  T h_mask[MAX_MASK_WIDTH];
  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  // ---- 修改: 使用 2:4 稀疏初始化 ----
  generate_2to4_sparse<T>(a, input_width);

  T *d_a, *d_b;
  cudaMalloc((void **)&d_a, size_bytes);
  cudaMalloc((void **)&d_b, size_bytes);

  cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask<T>, h_mask, mask_width * sizeof(T));

  dim3 grids (input_width / BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();

  // conv1D tiling and caching
  for (int i = 0; i < repeat; i++) {
    conv1d_tiled_caching <<< grids, blocks >>> (d_a, d_b, input_width, mask_width);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(b, d_b, size_bytes, cudaMemcpyDeviceToHost);
  // ===== 从 result.txt 读取参考结果 =====
  FILE *file = fopen("result.txt", "r");
  if (file == NULL) {
    printf("Fault Injection Test Failed!\n");
    free(a); free(b);
    cudaFree(d_a); cudaFree(d_b);
    return;
  }

  T *expected = (T*) malloc(size_bytes);
  int count = 0;
  while (fscanf(file, "%hd", &expected[count]) == 1 && count < input_width) {
    count++;
  }
  fclose(file);

  if (count != input_width) {
    printf("Fault Injection Test Failed!\n");
    free(expected);
    free(a); free(b);
    cudaFree(d_a); cudaFree(d_b);
    return;
  }

  // ===== 逐项比较结果 =====
  bool match = true;
  for (int i = 0; i < input_width; i++) {
    if (b[i] != expected[i]) {
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

  free(a);
  free(b);
  cudaFree(d_a);
  cudaFree(d_b);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <mask_width>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  // a multiple of BLOCK_SIZE
  input_width = (input_width + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  
  const int mask_width = atoi(argv[2]);

  const int repeat = 1;

  conv1D<int16_t>(input_width, mask_width, repeat);

  return 0;
}
