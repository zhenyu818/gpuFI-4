// Inject app for Dijkstra Kernel1 (all-zero weights)
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D_SEED 1111
#define BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err__ = (call);                                                                                    \
        if (err__ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, (int)err__,                       \
                    cudaGetErrorString(err__));                                                                        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

__device__ inline float atomicMinFloat(float *addr, float value) {
    float old = __int_as_float(atomicCAS((int *)addr, __float_as_int(*addr), __float_as_int(*addr)));
    while (value < old) {
        float assumed = old;
        int old_int = atomicCAS((int *)addr, __float_as_int(assumed), __float_as_int(value));
        old = __int_as_float(old_int);
        if (old == assumed)
            break;
    }
    return old;
}
__device__ inline float atomicMin(float *addr, float value) { return atomicMinFloat(addr, value); }

__global__ void Kernel1(const int *__restrict__ vertexArray, const int *__restrict__ edgeArray,
                        const float *__restrict__ weightArray, bool *__restrict__ finalizedVertices,
                        float *__restrict__ shortestDistances, float *__restrict__ updatingShortestDistances,
                        const int numVertices, const int numEdges) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (finalizedVertices[tid] == true) {

            finalizedVertices[tid] = false;

            int edgeStart = vertexArray[tid], edgeEnd;

            if (tid + 1 < (numVertices))
                edgeEnd = vertexArray[tid + 1];
            else
                edgeEnd = numEdges;

            for (int edge = edgeStart; edge < edgeEnd; edge++) {
                int nid = edgeArray[edge];
                atomicMin(&updatingShortestDistances[nid], shortestDistances[tid] + weightArray[edge]);
            }
        }
    }
}

static void generate_graph(int V, int deg, int *vertexArray, int *edgeArray) {
    for (int i = 0; i < V; ++i) vertexArray[i] = i * deg;
    for (int u = 0; u < V; ++u) {
        for (int j = 0; j < deg; ++j) {
            int v;
            int tries = 0;
            for (;;) {
                v = rand() % V;
                if (v == u) {
                    if (++tries < 1000) continue;
                }
                bool dup = false;
                for (int k = 0; k < j; ++k) {
                    if (edgeArray[u * deg + k] == v) {
                        dup = true;
                        break;
                    }
                }
                if (!dup && v != u) break;
            }
            edgeArray[u * deg + j] = v;
        }
    }
}

int main(int argc, char **argv) {
    int V = 64;
    int DEG = 4;
    int src = 0;

    if (argc >= 3) {
        V = atoi(argv[1]);
        DEG = atoi(argv[2]);
        if (V <= 0 || DEG <= 0) return 0;
        if (DEG > V - 1) DEG = V - 1;
    }

    srand(D_SEED);

    const int E = V * DEG;

    int *h_vertexArray = (int *)malloc(sizeof(int) * V);
    int *h_edgeArray = (int *)malloc(sizeof(int) * E);
    float *h_weightArray = (float *)malloc(sizeof(float) * E);
    bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * V);
    float *h_shortestDistances = (float *)malloc(sizeof(float) * V);
    float *h_updatingShortestDistances = (float *)malloc(sizeof(float) * V);

    generate_graph(V, DEG, h_vertexArray, h_edgeArray);
    for (int i = 0; i < E; ++i) {
        h_weightArray[i] = 0.0f;
    }

    for (int i = 0; i < V; ++i) {
        h_finalizedVertices[i] = (i == src);
        h_shortestDistances[i] = (i == src) ? 0.0f : FLT_MAX;
        h_updatingShortestDistances[i] = h_shortestDistances[i];
    }

    int *d_vertexArray = nullptr, *d_edgeArray = nullptr;
    float *d_weightArray = nullptr, *d_shortestDistances = nullptr, *d_updatingShortestDistances = nullptr;
    bool *d_finalizedVertices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_vertexArray, sizeof(int) * V));
    CUDA_CHECK(cudaMalloc(&d_edgeArray, sizeof(int) * E));
    CUDA_CHECK(cudaMalloc(&d_weightArray, sizeof(float) * E));
    CUDA_CHECK(cudaMalloc(&d_finalizedVertices, sizeof(bool) * V));
    CUDA_CHECK(cudaMalloc(&d_shortestDistances, sizeof(float) * V));
    CUDA_CHECK(cudaMalloc(&d_updatingShortestDistances, sizeof(float) * V));

    CUDA_CHECK(cudaMemcpy(d_vertexArray, h_vertexArray, sizeof(int) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edgeArray, h_edgeArray, sizeof(int) * E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weightArray, h_weightArray, sizeof(float) * E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_finalizedVertices, h_finalizedVertices, sizeof(bool) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shortestDistances, h_shortestDistances, sizeof(float) * V, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_updatingShortestDistances, h_updatingShortestDistances, sizeof(float) * V, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(iDivUp(V, BLOCK_SIZE));
    Kernel1<<<grid, block>>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances,
                             d_updatingShortestDistances, V, E);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(h_updatingShortestDistances, d_updatingShortestDistances, sizeof(float) * V, cudaMemcpyDeviceToHost));

    FILE *f = fopen("result.txt", "r");
    if (!f) {
        printf("Fault Injection Test Failed!\n");
        return 0;
    }
    float *expected = (float *)malloc(sizeof(float) * V);
    int count = 0;
    while (count < V && fscanf(f, "%f", &expected[count]) == 1) count++;
    fclose(f);
    if (count != V) {
        printf("Fault Injection Test Failed!\n");
        free(expected);
        return 0;
    }
    const float eps = 1e-5f;
    bool ok = true;
    for (int i = 0; i < V; ++i) {
        float a = h_updatingShortestDistances[i];
        float b = expected[i];
        if (isnan(a) && isnan(b)) continue;
        if (isnan(a) || isnan(b)) { ok = false; break; }
        if (isinf(a) && isinf(b)) continue;
        if (isinf(a) || isinf(b)) { ok = false; break; }
        if (fabsf(a - b) > eps) { ok = false; break; }
    }
    if (ok)
        printf("Fault Injection Test Success!\n");
    else
        printf("Fault Injection Test Failed!\n");
    free(expected);

    free(h_vertexArray);
    free(h_edgeArray);
    free(h_weightArray);
    free(h_finalizedVertices);
    free(h_shortestDistances);
    free(h_updatingShortestDistances);
    cudaFree(d_vertexArray);
    cudaFree(d_edgeArray);
    cudaFree(d_weightArray);
    cudaFree(d_finalizedVertices);
    cudaFree(d_shortestDistances);
    cudaFree(d_updatingShortestDistances);
    return 0;
}

