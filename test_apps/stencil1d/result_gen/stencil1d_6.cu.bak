#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <stdbool.h>

#define RADIUS 7
#define BLOCK_SIZE 256
#define M_SEED 3031

// ===================== 输入生成函数 =====================
static void generate_input_1(int length, int *a) {
    srand(M_SEED);

    for (int i = 0; i < length + RADIUS; i++) {
        // 基础值设为0
        a[i] = 0;

        // Pattern 1: 棋盘模式 - 交替极值 (0和9)
        if (i % 2 == 0) {
            a[i] = 0;
        } else {
            a[i] = 9;
        }

        // Pattern 2: 边界特殊情况 - 首尾位置
        if (i == 0 || i == length - 1 || i == length + RADIUS - 1) {
            a[i] = 7;
        }

        // Pattern 3: 中心区域陷阱 - 高成本区域
        if (i >= length / 4 && i < 3 * length / 4) {
            a[i] = 8;
        }

        // Pattern 4: 垂直条纹效果 - 用模数制造周期障碍
        if (i % 3 == 0) {
            a[i] = 6;
        }

        // Pattern 5: 水平条纹效果 - 每隔4个制造障碍
        if (i % 4 == 0) {
            a[i] = 5;
        }

        // Pattern 6: 随机噪声 - 增加不可预测性
        if (rand() % 15 == 0) {  // 约 6.7% 概率
            a[i] = rand() % 10;
        }

        // Pattern 7: 角落特殊值
        if (i == 0 || i == length - 1) {
            a[i] = 3;
        }
    }
}

// ===================== 核函数 =====================
__global__
void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  if (gindex < length) {
    temp[lindex] = in[gindex];

    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
      if (gindex + BLOCK_SIZE < length + RADIUS)
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    __syncthreads();

    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
      result += temp[lindex + offset];

    out[gindex] = result;
  }
}

// ===================== 主函数 =====================
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

  // 使用对抗输入生成方式
  generate_input_1(length, a);

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
