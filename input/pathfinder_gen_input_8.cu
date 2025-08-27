// 仅保留生成 wall 输入并写入到 input/EXP_NAME.txt 的代码
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_SEED 3415
#define EXP_NAME "1-1"

int rows, cols;
float *data;
float **wall;

static void generate_input(int argc, char **argv)
{
	if (argc == 3) {
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	} else {
		printf("Usage: gen_input col_len row_len\n");
		exit(0);
	}

	data = new float[rows * cols];
	wall = new float*[rows];
	for (int n = 0; n < rows; n++)
		wall[n] = data + cols * n;

	srand(M_SEED);
	int has_nan = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if ((rand() % 10) == 0) {
				wall[i][j] = NAN;
				has_nan = 1;
			} else {
				wall[i][j] = (float)(rand() % 10);
			}
		}
	}
	if (!has_nan && rows > 0 && cols > 0) {
		wall[0][0] = NAN;
	}
}

int main(int argc, char **argv)
{
	generate_input(argc, argv);

	// 写入到 input/EXP_NAME.txt
	FILE *f = fopen(EXP_NAME ".txt", "w");
	if (!f) {
		perror("fopen EXP_NAME.txt");
		return 1;
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			fprintf(f, "%g%c", wall[i][j], (j == cols - 1) ? '\n' : ' ');
		}
	}
	fclose(f);

	delete [] data;
	delete [] wall;
	return 0;
}
