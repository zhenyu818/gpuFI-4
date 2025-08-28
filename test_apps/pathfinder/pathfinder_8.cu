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

	// 生成包含对抗性模式的输入数据
	srand(M_SEED);
	
	// 对抗性模式1: 极值模式 - 在关键位置放置极大值和极小值
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			// 在边界和关键路径上放置极值
			if (i == 0 || i == rows-1 || j == 0 || j == cols-1) {
				// 边界放置极大值，增加边界处理复杂度
				wall[i][j] = (i + j) % 2 == 0 ? 999 : 1;
			} else if (i % 3 == 0 && j % 3 == 0) {
				// 在3的倍数位置放置极小值，创建"陷阱"
				wall[i][j] = 0;
			} else if (i == rows/2 && j == cols/2) {
				// 中心位置放置极大值，增加中心路径计算复杂度
				wall[i][j] = 999;
			} else if ((i + j) % 7 == 0) {
				// 在特定模式位置放置交替的极值
				wall[i][j] = (i + j) % 14 == 0 ? 999 : 1;
			} else {
				// 其他位置生成中等范围的随机值，但偏向于创建复杂路径
				int base_val = rand() % 20;
				// 增加一些"噪声"，使路径规划更复杂
				if (base_val < 5) {
					wall[i][j] = rand() % 100 + 50;  // 50-149的高值
				} else if (base_val < 10) {
					wall[i][j] = rand() % 10;        // 0-9的低值
				} else {
					wall[i][j] = rand() % 30 + 10;   // 10-39的中等值
				}
			}
		}
	}
	
	// 对抗性模式2: 创建"迷宫"模式 - 在特定行创建高成本路径
	for (int i = 1; i < rows-1; i += 4) {
		for (int j = 0; j < cols; j++) {
			if (j % 2 == 0) {
				wall[i][j] = 888;  // 创建高成本行
			}
		}
	}
	
	// 对抗性模式3: 在关键计算路径上放置交替的高低值
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if ((i + j) % 5 == 0 && i > 0 && i < rows-1 && j > 0 && j < cols-1) {
				// 在内部位置创建交替模式，增加动态规划的计算复杂度
				wall[i][j] = (i + j) % 10 == 0 ? 777 : 3;
			}
		}
	}
	
	// 对抗性模式4: 在算法可能优化的路径上放置挑战性值
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == j || i == cols-1-j) {
				// 对角线位置放置特殊值，挑战算法的优化策略
				wall[i][j] = (i + j) % 3 == 0 ? 666 : 5;
			}
		}
	}
	
    printf("Adversarial input generated with seed %d\n", M_SEED);
    printf("Input matrix (%dx%d):\n", rows, cols);
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%3d ", wall[i][j]);
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
