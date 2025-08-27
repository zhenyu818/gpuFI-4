#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define M_SEED 3415          // 保留原随机种子
#define EXP_NAME "1-1"       // 保留原实验名称
#define SPARSE_N 2           // 2:4稀疏中的N（每M个元素保留的非零值数量）
#define SPARSE_M 4           // 2:4稀疏中的M（连续元素分组大小）

int rows, cols;
int *data;
int **wall;

// 生成2:4结构化稀疏的随机值（每4个连续元素中随机选2个置非零，其余置0）
static void generate_2to4_sparse_value(int *group, int group_len) {
    // 1. 初始化分组为全0（满足稀疏约束的基础）
    for (int k = 0; k < group_len; k++) {
        group[k] = 0;
    }
    
    // 2. 随机选择2个不同的位置作为非零值索引（确保每4个元素仅保留2个非零）
    bool selected[SPARSE_M] = {false};
    int selected_count = 0;
    while (selected_count < SPARSE_N) {
        int rand_idx = rand() % SPARSE_M;  // 0~3范围内随机选索引
        if (!selected[rand_idx]) {
            selected[rand_idx] = true;
            selected_count++;
        }
    }
    
    // 3. 为选中的位置生成0~9的随机非零值（匹配原代码的随机值范围）
    for (int k = 0; k < SPARSE_M; k++) {
        if (selected[k]) {
            group[k] = rand() % 10;
            // 确保非零（若随机到0则重新生成，避免与稀疏置0混淆）
            while (group[k] == 0) {
                group[k] = rand() % 10;
            }
        }
    }
}

// 生成2:4结构化稀疏的输入矩阵
static void generate_input(int argc, char **argv) {
    if (argc == 3) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        // 检查列数是否为4的整数倍（确保2:4稀疏分组完整，若不满足则自动补齐）
        if (cols % SPARSE_M != 0) {
            int new_cols = (cols / SPARSE_M + 1) * SPARSE_M;
            printf("Warning: Column length (%d) is not a multiple of %d. Auto-adjust to %d\n", 
                   cols, SPARSE_M, new_cols);
            cols = new_cols;
        }
    } else {
        printf("Usage: gen_input col_len row_len\n");
        exit(0);
    }

    // 内存分配（与原代码逻辑一致）
    data = new int[rows * cols];
    wall = new int*[rows];
    for (int n = 0; n < rows; n++) {
        wall[n] = data + cols * n;
    }

    srand(M_SEED);  // 保留原随机种子，确保可复现性
    // 按行生成2:4稀疏数据
    for (int i = 0; i < rows; i++) {
        // 按4个元素为一组处理，确保每组满足2:4稀疏约束
        for (int j = 0; j < cols; j += SPARSE_M) {
            int group[SPARSE_M];
            generate_2to4_sparse_value(group, SPARSE_M);
            // 将稀疏分组赋值到矩阵对应位置
            for (int k = 0; k < SPARSE_M; k++) {
                wall[i][j + k] = group[k];
            }
        }
    }
}

int main(int argc, char **argv) {
    generate_input(argc, argv);

    // 写入到 input/EXP_NAME.txt（与原代码逻辑完全一致）
    FILE *f = fopen(EXP_NAME ".txt", "w");
    if (!f) {
        perror("fopen EXP_NAME.txt");
        return 1;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(f, "%d%c", wall[i][j], (j == cols - 1) ? '\n' : ' ');
        }
    }
    fclose(f);

    // 内存释放（与原代码逻辑一致）
    delete[] data;
    delete[] wall;
    return 0;
}
