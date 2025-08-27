// 仅保留生成 wall 输入并写入到 input/EXP_NAME.txt 的代码
#include <stdio.h>
#include <stdlib.h>

#define M_SEED 5325
#define EXP_NAME "1-1"

int rows, cols;
int *data;
int **wall;

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
			wall[i][j] = rand() % 10;
		}
	}
}

int main(int argc, char **argv)
{
	generate_input(argc, argv);

	// 写入到 input/EXP_NAME.txt
	FILE *f = fopen("EXP_NAME ".txt", "w");
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
