#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

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

#define M_SEED 3415          // 保留原随机种子
#define SPARSE_N 2           // 2:4稀疏中的N（每M个元素保留的非零值数量）
#define SPARSE_M 4           // 2:4稀疏中的M（连续元素分组大小）

//#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
int pyramid_height;

// 从pathfinder_gen_input_6.cu集成的2:4稀疏输入生成函数
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
static void generate_input_6(int argc, char **argv) {
    if (argc == 4) {
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
        pyramid_height = atoi(argv[3]);
        // 检查列数是否为4的整数倍（确保2:4稀疏分组完整，若不满足则自动补齐）
        if (cols % SPARSE_M != 0) {
            int new_cols = (cols / SPARSE_M + 1) * SPARSE_M;
            cols = new_cols;
        }
    } else {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(0);
    }

    // 内存分配（与原代码逻辑一致）
    data = new int[rows * cols];
    wall = new int*[rows];
    for (int n = 0; n < rows; n++) {
        wall[n] = data + cols * n;
    }
    result = new int[cols];

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

void
init(int argc, char** argv)
{
	// 调用集成的输入生成函数
	generate_input_6(argc, argv);
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

    // 读取result.txt文件进行比对
    FILE *file = fopen("result.txt", "r");
    if (file == NULL) {
        printf("Failed\n");
        cudaFree(gpuWall);
        cudaFree(gpuResult[0]);
        cudaFree(gpuResult[1]);
        delete [] data;
        delete [] wall;
        delete [] result;
        return;
    }
    
    int expected_result[cols];
    int i = 0;
    while (fscanf(file, "%d", &expected_result[i]) == 1 && i < cols) {
        i++;
    }
    fclose(file);
    
    // 检查是否读取了足够的元素
    if (i != cols) {
        printf("Failed\n");
        cudaFree(gpuWall);
        cudaFree(gpuResult[0]);
        cudaFree(gpuResult[1]);
        delete [] data;
        delete [] wall;
        delete [] result;
        return;
    }
    
    // 比对结果
    bool match = true;
    for (i = 0; i < cols; i++) {
        if (result[i] != expected_result[i]) {
            match = false;
            break;
        }
    }
    
    
    if (match) {
        printf("Success\n");
    } else {
        printf("Failed\n");
    }

    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;
}
