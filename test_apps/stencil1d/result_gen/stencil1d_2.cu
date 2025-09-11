#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <stdbool.h>

#define RADIUS 7
#define BLOCK_SIZE 256
#define M_SEED 9409

__global__
void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // 只处理在范围内的元素
  if (gindex < length) {
    temp[lindex] = in[gindex];

    // 边界加载
    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
      if (gindex + BLOCK_SIZE < length + RADIUS)
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    __syncthreads();

    // stencil 计算
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
      result += temp[lindex + offset];

    out[gindex] = result;
  }
}


int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <length>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = 1;

  int size = length * sizeof(int);
  int pad_size = (length + RADIUS) * sizeof(int);

  int *a, *b;
  a = (int *)malloc(pad_size); 
  b = (int *)malloc(size);

  srand(M_SEED);
  for (int i = 0; i < length + RADIUS; i++) {
      a[i] = 0;
  }

  int *d_a, *d_b;
  cudaMalloc((void **)&d_a, pad_size);
  cudaMalloc((void **)&d_b, size);

  cudaMemcpy(d_a, a, pad_size, cudaMemcpyHostToDevice);

  dim3 grids (length/BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();

  for (int i = 0; i < repeat; i++)
    stencil_1d <<< grids, blocks >>> (d_a, d_b, length);

  cudaDeviceSynchronize();

  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  for(int i=0;i<length;i++){
    printf("%d ", b[i]);
  }
  free(a);
  free(b); 
  cudaFree(d_a); 
  cudaFree(d_b); 
  return 0;
}
