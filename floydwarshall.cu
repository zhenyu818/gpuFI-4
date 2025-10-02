#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define MAXDISTANCE (200)
#define SEED (2)
// Total threads in the single block used by the single kernel launch
#define BLOCK_THREADS (256)

// Single-kernel implementation of Floyd–Warshall using one block and __syncthreads()
// Each thread updates a strided subset of elements for every k, then synchronizes.
__global__ void floydWarshallSingleKernel(unsigned int* __restrict__ dist,
                                          const unsigned int n)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int T   = blockDim.x; // total threads in this (single) block
  const unsigned int total = n * n;

  for (unsigned int k = 0; k < n; ++k) {
    for (unsigned int idx = tid; idx < total; idx += T) {
      const unsigned int i = idx / n;
      const unsigned int j = idx % n;

      const unsigned int via = dist[i * n + k] + dist[k * n + j];
      const unsigned int cur = dist[idx];
      if (via < cur) {
        dist[idx] = via;
      }
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }

  unsigned int numNodes = atoi(argv[1]);

  unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
  unsigned int* pathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);

  srand(SEED);
  for(unsigned int i = 0; i < numNodes; i++) {
    for(unsigned int j = 0; j < numNodes; j++) {
      int index = i * numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
    pathDistanceMatrix[i * numNodes + i] = 0;
  }

  unsigned int *pathDistanceBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);

  cudaMemcpy(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);

  // Launch exactly one kernel (single block). Threads stride over the matrix.
  unsigned int threads = BLOCK_THREADS;
  if (threads > numNodes * numNodes) threads = numNodes * numNodes; // cap by work
  if (threads == 0) threads = 1; // handle n=0 (though not expected)
  floydWarshallSingleKernel<<<1, threads>>>(pathDistanceBuffer, numNodes);
  cudaDeviceSynchronize();

  cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
  cudaFree(pathDistanceBuffer);

  // Read expected results from result.txt
  FILE* fp = fopen("result.txt", "r");
  if (!fp) {
    printf("Fault Injection Test Failed!\n"); // File not found or error
    free(pathDistanceMatrix);
    return 1;
  }

  unsigned int* expected = (unsigned int *) malloc(matrixSizeBytes);
  unsigned int readCount = 0;
  unsigned int val;
  while (fscanf(fp, "%u", &val) == 1) {
    if (readCount < numNodes * numNodes) {
      expected[readCount] = val;
      readCount++;
    }
  }
  fclose(fp);

  // Check if we read exactly the expected number of values
  int match = (readCount == numNodes * numNodes) && (memcmp(pathDistanceMatrix, expected, matrixSizeBytes) == 0);
  free(expected);
  free(pathDistanceMatrix);

  if (match) {
    printf("Fault Injection Test Success!\n");
  } else {
    printf("Fault Injection Test Failed!\n");
  }
  return 0;
}