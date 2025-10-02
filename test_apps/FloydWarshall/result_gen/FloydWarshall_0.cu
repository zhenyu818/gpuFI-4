/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
   Modified to execute kernel only once with deterministic random inputs.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define MAXDISTANCE (200)
#define RANDOM_SEED 2
#define BLOCK_SIZE 16

__global__ void floydWarshallPass(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int pass)
{
  int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  int yValue = threadIdx.y + blockIdx.y * blockDim.y;

  int k = pass;
  int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
  int tempWeight = pathDistanceBuffer[yValue * numNodes + k] + 
                   pathDistanceBuffer[k * numNodes + xValue];

  if (tempWeight < oldWeight)
  {
    pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
    pathBuffer[yValue * numNodes + xValue] = k;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <number of nodes>\n", argv[0]);
    return 1;
  }

  unsigned int numNodes = atoi(argv[1]);

  unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);

  unsigned int* pathDistanceMatrix = (unsigned int*) malloc(matrixSizeBytes);
  assert(pathDistanceMatrix != NULL);

  unsigned int* pathMatrix = (unsigned int*) malloc(matrixSizeBytes);
  assert(pathMatrix != NULL);

  // Initialize input with fixed random seed for deterministic results
  srand(RANDOM_SEED);
  for(unsigned int i = 0; i < numNodes; i++) {
    for(unsigned int j = 0; j < numNodes; j++) {
      int index = i * numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  }

  // Set diagonal to 0
  for(unsigned int i = 0; i < numNodes; ++i) {
    pathDistanceMatrix[i * numNodes + i] = 0;
  }

  // Initialize pathMatrix
  for(unsigned int i = 0; i < numNodes; ++i) {
    for(unsigned int j = 0; j < i; ++j) {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  // Setup kernel launch parameters
  dim3 grids(numNodes / BLOCK_SIZE, numNodes / BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  // Allocate device memory
  unsigned int *pathDistanceBuffer, *pathBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);
  cudaMalloc((void**)&pathBuffer, matrixSizeBytes);

  // Copy input to device
  cudaMemcpy(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pathBuffer, pathMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);

  // Execute kernel only once with pass=0
  floydWarshallPass<<<grids, threads>>>(pathDistanceBuffer, pathBuffer, numNodes, 0);

  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(pathMatrix, pathBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);

  // Print all results from pathDistanceMatrix
  for(unsigned int i = 0; i < numNodes * numNodes; i++) {
    printf("%u", pathDistanceMatrix[i]);
    if(i < numNodes * numNodes - 1) {
      printf(" ");
    }
  }
  printf("\n");

  // Print all results from pathMatrix
  for(unsigned int i = 0; i < numNodes * numNodes; i++) {
    printf("%u", pathMatrix[i]);
    if(i < numNodes * numNodes - 1) {
      printf(" ");
    }
  }
  printf("\n");

  // Cleanup
  cudaFree(pathDistanceBuffer);
  cudaFree(pathBuffer);
  free(pathDistanceMatrix);
  free(pathMatrix);

  return 0;
}