#include <cuda.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RADIUS 7
#define BLOCK_SIZE 256
#define M_SEED 3031

#define SPARSE_N 2 // 每组非零个数
#define SPARSE_M 4 // 每组大小

__global__ void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int length) {
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
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
            result += temp[lindex + offset];

        out[gindex] = result;
    }
}

// ==================== 生成 2:4 稀疏输入 ====================
static void generate_2to4_sparse_group(int *group, int group_len) {
    for (int k = 0; k < group_len; k++)
        group[k] = 0;

    bool selected[SPARSE_M] = {false};
    int selected_count = 0;
    while (selected_count < SPARSE_N) {
        int rand_idx = rand() % SPARSE_M;
        if (!selected[rand_idx]) {
            selected[rand_idx] = true;
            selected_count++;
        }
    }

    for (int k = 0; k < SPARSE_M; k++) {
        if (selected[k]) {
            group[k] = rand() % 10;
            while (group[k] == 0) {
                group[k] = rand() % 10;
            }
        }
    }
}

static void generate_sparse_input(int *a, int length) {
    srand(M_SEED);
    int aligned_len = (length % SPARSE_M == 0) ? length : (length / SPARSE_M + 1) * SPARSE_M;

    for (int i = 0; i < aligned_len; i += SPARSE_M) {
        int group[SPARSE_M];
        generate_2to4_sparse_group(group, SPARSE_M);
        for (int k = 0; k < SPARSE_M; k++) {
            if (i + k < length) {
                a[i + k] = group[k];
            }
        }
    }

    // pad 区域保持随机 (避免访问越界)
    for (int i = length; i < length + RADIUS; i++) {
        a[i] = rand() % 10;
    }
}
// =========================================================

int main(int argc, char *argv[]) {
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

    // 使用 2:4 稀疏生成方法
    generate_sparse_input(a, length);

    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, pad_size);
    cudaMalloc((void **)&d_b, size);

    cudaMemcpy(d_a, a, pad_size, cudaMemcpyHostToDevice);

    dim3 grids(length / BLOCK_SIZE);
    dim3 blocks(BLOCK_SIZE);

    cudaDeviceSynchronize();

    for (int i = 0; i < repeat; i++)
        stencil_1d<<<grids, blocks>>>(d_a, d_b, length);

    cudaDeviceSynchronize();

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    // ===== 比对结果 =====
    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Fault Injection Test Failed!\n");
        free(a);
        free(b);
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    int *expected = (int *)malloc(size);
    int i = 0;
    while (fscanf(file, "%d", &expected[i]) == 1 && i < length) {
        i++;
    }
    fclose(file);

    if (i != length) {
        printf("Fault Injection Test Failed!\n");
    } else {
        bool match = true;
        for (i = 0; i < length; i++) {
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
    }

    free(expected);
    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
