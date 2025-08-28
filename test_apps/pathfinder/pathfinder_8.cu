#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define M_SEED 3415
#define EXP_NAME "1-1"

//#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
int pyramid_height;
// 对抗性输入扰动函数前置声明
static void apply_adversarial_patterns(int **grid, int rows, int cols);

// 从pathfinder_gen_input_1.cu集成的输入生成函数
static void generate_input_1(int argc, char **argv)
{
	if (argc == 4) {
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		pyramid_height = atoi(argv[3]);
	} else {
		printf("Usage: dynproc row_len col_len pyramid_height\n");
		exit(0);
	}

	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];

	// 直接生成输入数据，而不是从文件读取
	srand(M_SEED);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			wall[i][j] = rand() % 10;
		}
	}
	// 在随机基础上施加对抗性模式（整数扰动，保持0..9范围）
	apply_adversarial_patterns(wall, rows, cols);
    printf("input generated\n");
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%d ", wall[i][j]);
        }
        printf("\n");
    }
}

void
init(int argc, char** argv)
{
	// 调用集成的输入生成函数
	generate_input_1(argc, argv);
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// 生成对抗性整数扰动：在看似平滑/随机的数据上，构造“诱饵走廊”与“陷阱带”
// 目标：在不显著改变整体分布（只做±1/±2小幅调整，且保持0..9区间）的情况下，
// 让局部最优路径在中段被提高成本，迫使全局最优发生偏移，从而具有对抗性特征。
static void apply_adversarial_patterns(int **grid, int rows, int cols)
{
	int corridor1_col = cols/2;
	int corridor2_col = (corridor1_col + 5 < cols) ? corridor1_col + 5 : (cols - 1);
	int width_c1 = (cols > 16) ? 2 : 1;
	int width_c2 = (cols > 16) ? 2 : 1;

	int r1_end = (int)(rows * 0.4f);
	int trap_start = (int)(rows * 0.45f);
	int trap_end = (int)(rows * 0.7f);
	int r2_start = (int)(rows * 0.55f);

	if (r1_end < 1) r1_end = 1;
	if (trap_start < r1_end) trap_start = r1_end;
	if (trap_end < trap_start + 1) trap_end = trap_start + 1;
	if (r2_start < trap_start) r2_start = trap_start;

	// 稀疏±1随机微扰（~1% 概率）
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int r = rand() % 100;
			if (r == 0) {
				int delta = (rand() & 1) ? 1 : -1;
				int v = grid[i][j] + delta;
				CLAMP_RANGE(v, 0, 9);
				grid[i][j] = v;
			}
		}
	}

	// 段1：c1 附近小幅降低
	for (int i = 0; i < r1_end; ++i) {
		for (int w = -width_c1; w <= width_c1; ++w) {
			int j = corridor1_col + w;
			if (j >= 0 && j < cols) {
				int v = grid[i][j] - 2;
				CLAMP_RANGE(v, 0, 9);
				grid[i][j] = v;
			}
		}
	}

	// 段2（陷阱）：c1 附近抬高
	for (int i = trap_start; i < trap_end; ++i) {
		for (int w = -width_c1; w <= width_c1; ++w) {
			int j = corridor1_col + w;
			if (j >= 0 && j < cols) {
				int v = grid[i][j] + 3;
				CLAMP_RANGE(v, 0, 9);
				grid[i][j] = v;
			}
		}
	}

	// 段3：c2 附近小幅降低
	for (int i = r2_start; i < rows; ++i) {
		for (int w = -width_c2; w <= width_c2; ++w) {
			int j = corridor2_col + w;
			if (j >= 0 && j < cols) {
				int v = grid[i][j] - 2;
				CLAMP_RANGE(v, 0, 9);
				grid[i][j] = v;
			}
		}
	}

	// 细微“扰动栅栏”
	for (int i = r1_end; i < rows; ++i) {
		int fence_left = corridor1_col - (width_c1 + 3);
		int fence_right = corridor2_col + (width_c2 + 3);
		if (fence_left >= 0 && fence_left < cols) {
			int v = grid[i][fence_left] + 1;
			CLAMP_RANGE(v, 0, 9);
			grid[i][fence_left] = v;
		}
		if (fence_right >= 0 && fence_right < cols) {
			int v = grid[i][fence_right] + 1;
			CLAMP_RANGE(v, 0, 9);
			grid[i][fence_right] = v;
		}
	}
}

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols, 
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  
	
        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);

            // for the measurement fairness
            cudaDeviceSynchronize();
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);
	
    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);

#ifdef  TIMING
    gettimeofday(&tv_kernel_start, NULL);
#endif

    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

#ifdef  TIMING
    gettimeofday(&tv_kernel_end, NULL);
    tvsub(&tv_kernel_end, &tv_kernel_start, &tv);
    kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
    printf("result generated\n");
    // output result array to console instead of txt file
    for (int i = 0; i < cols; ++i) {
        printf("%d%c", result[i], (i == cols - 1) ? '\n' : ' ');
    }


    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;
}
