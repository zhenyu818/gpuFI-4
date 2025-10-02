/* Inputs all zeros */
#include <assert.h>
#include <cuda.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void floydWarshallPass(unsigned int *__restrict__ pathDistanceBuffer, unsigned int *__restrict__ pathBuffer,
                                  const unsigned int numNodes, const unsigned int pass) {
    int xValue = threadIdx.x + blockIdx.x * blockDim.x;
    int yValue = threadIdx.y + blockIdx.y * blockDim.y;
    int k = pass;
    int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
    int tempWeight = pathDistanceBuffer[yValue * numNodes + k] + pathDistanceBuffer[k * numNodes + xValue];
    if (tempWeight < oldWeight) {
        pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
        pathBuffer[yValue * numNodes + xValue] = k;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <number of nodes>\n", argv[0]);
        return 1;
    }
    unsigned int numNodes = atoi(argv[1]);
    unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *pathDistanceMatrix = (unsigned int *)malloc(matrixSizeBytes);
    assert(pathDistanceMatrix);
    unsigned int *pathMatrix = (unsigned int *)malloc(matrixSizeBytes);
    assert(pathMatrix);
    for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++) {
            int index = i * numNodes + j;
            pathDistanceMatrix[index] = 0;
        }
    }
    for (unsigned int i = 0; i < numNodes; ++i) {
        pathDistanceMatrix[i * numNodes + i] = 0;
    }
    for (unsigned int i = 0; i < numNodes; ++i) {
        for (unsigned int j = 0; j < i; ++j) {
            pathMatrix[i * numNodes + j] = i;
            pathMatrix[j * numNodes + i] = j;
        }
        pathMatrix[i * numNodes + i] = i;
    }
    dim3 grids(numNodes / BLOCK_SIZE, numNodes / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    unsigned int *pathDistanceBuffer, *pathBuffer;
    cudaMalloc((void **)&pathDistanceBuffer, matrixSizeBytes);
    cudaMalloc((void **)&pathBuffer, matrixSizeBytes);
    cudaMemcpy(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pathBuffer, pathMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);
    floydWarshallPass<<<grids, threads>>>(pathDistanceBuffer, pathBuffer, numNodes, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(pathMatrix, pathBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
    FILE *fp = fopen("result.txt", "r");
    if (!fp) {
        printf("Failed to open result.txt\n");
        return 1;
    }
    unsigned int *expected_distance = (unsigned int *)malloc(matrixSizeBytes);
    unsigned int *expected_path = (unsigned int *)malloc(matrixSizeBytes);
    for (unsigned int i = 0; i < numNodes * numNodes; i++) {
        if (fscanf(fp, "%u", &expected_distance[i]) != 1) {
            printf("Failed to read expected distance data\n");
            fclose(fp);
            free(expected_distance);
            free(expected_path);
            return 1;
        }
    }
    for (unsigned int i = 0; i < numNodes * numNodes; i++) {
        if (fscanf(fp, "%u", &expected_path[i]) != 1) {
            printf("Failed to read expected path data\n");
            fclose(fp);
            free(expected_distance);
            free(expected_path);
            return 1;
        }
    }
    fclose(fp);
    bool match_distance = true;
    for (unsigned int i = 0; i < numNodes * numNodes; i++) {
        if (pathDistanceMatrix[i] != expected_distance[i]) {
            match_distance = false;
            break;
        }
    }
    bool match_path = true;
    for (unsigned int i = 0; i < numNodes * numNodes; i++) {
        if (pathMatrix[i] != expected_path[i]) {
            match_path = false;
            break;
        }
    }
    printf((match_distance && match_path) ? "Fault Injection Test Success!\n" : "Fault Injection Test Failed!\n");
    free(expected_distance);
    free(expected_path);
    cudaFree(pathDistanceBuffer);
    cudaFree(pathBuffer);
    free(pathDistanceMatrix);
    free(pathMatrix);
    return 0;
}
