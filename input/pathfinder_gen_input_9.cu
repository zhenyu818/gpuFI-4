// 仅保留生成 wall 输入并写入到 EXP_NAME.txt 的代码
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define M_SEED 3415
#define EXP_NAME "1-1"

int rows, cols;
int *data;
int **wall;

static inline int gen_denormal_int_bits()
{
	// 构造 IEEE-754 单精度次正规数的位模式：exp=0, frac!=0, sign 随机
	uint32_t sign = (rand() & 1u) ? 0x80000000u : 0u;
	uint32_t frac = (uint32_t)(rand() % 0x7FFFFF) + 1u; // [1, 0x7FFFFF]
	uint32_t bits = sign | frac;
	return (int)bits;
}

static void generate_input(int argc, char **argv)
{
	if (argc == 3) {
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	} else {
		printf("Usage: gen_input col_len row_len\n");
		exit(0);
	}

	data = new int[rows * cols];
	wall = new int*[rows];
	for (int n = 0; n < rows; n++)
		wall[n] = data + cols * n;

	srand(M_SEED);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			// 20% 概率写入 denormal 位模式，其余为 0-9 的普通整数
			if ((rand() % 100) < 20) {
				wall[i][j] = gen_denormal_int_bits();
			} else {
				wall[i][j] = rand() % 10;
			}
		}
	}
}

int main(int argc, char **argv)
{
	generate_input(argc, argv);

	// 写入到 EXP_NAME.txt
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

	delete [] data;
	delete [] wall;
	return 0;
}
