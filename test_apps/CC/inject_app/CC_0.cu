#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef SEED
#define SEED 12345
#endif

#define THREADS_PER_BLOCK 256

__device__ int representative(const int idx, int *const __restrict__ nstat) {
    int curr = nstat[idx];
    if (curr != idx) {
        int next, prev = idx;
        while (curr > (next = nstat[curr])) {
            nstat[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

__global__ void compute1(const int nodes, const int *const __restrict__ nidx, const int *const __restrict__ nlist,
                         int *const __restrict__ nstat) {
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        int vstat = nstat[v];
        if (v != vstat) {
            const int beg = nidx[v];
            const int end = nidx[v + 1];
            for (int i = beg; i < end; i++) {
                const int nli = nlist[i];
                if (v > nli) {
                    int ostat = representative(nli, nstat);
                    bool repeat;
                    do {
                        repeat = false;
                        if (vstat != ostat) {
                            int ret;
                            if (vstat < ostat) {
                                if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                                    ostat = ret;
                                    repeat = true;
                                }
                            } else {
                                if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                                    vstat = ret;
                                    repeat = true;
                                }
                            }
                        }
                    } while (repeat);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <numNodes> <numEdges>\n", argv[0]);
        return 1;
    }

    const int nodes = atoi(argv[1]);
    const int edges = atoi(argv[2]);

    srand(SEED);

    // Host-side arrays
    int *nidx = (int *)malloc((nodes + 1) * sizeof(int));
    int *nlist = (int *)malloc(edges * sizeof(int));
    int *nstat = (int *)malloc(nodes * sizeof(int));
    assert(nidx && nlist && nstat);

    // Generate CSR graph
    nidx[0] = 0;
    for (int v = 0; v < nodes; v++) {
        int deg = edges / nodes;
        if (v < edges % nodes)
            deg++;
        nidx[v + 1] = nidx[v] + deg;
        for (int i = nidx[v]; i < nidx[v + 1]; i++) {
            nlist[i] = rand() % nodes;
        }
    }

    // Random initialize nstat
    for (int v = 0; v < nodes; v++) {
        nstat[v] = rand() % nodes;
    }

    // Device memory
    int *nidx_d, *nlist_d, *nstat_d;
    cudaMalloc((void **)&nidx_d, (nodes + 1) * sizeof(int));
    cudaMalloc((void **)&nlist_d, edges * sizeof(int));
    cudaMalloc((void **)&nstat_d, nodes * sizeof(int));

    cudaMemcpy(nidx_d, nidx, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nlist_d, nlist, edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nstat_d, nstat, nodes * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel once
    int blocks = (nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute1<<<blocks, THREADS_PER_BLOCK>>>(nodes, nidx_d, nlist_d, nstat_d);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(nstat, nstat_d, nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Compare with result.txt
    FILE *f = fopen("result.txt", "r");
    if (!f) {
        fprintf(stderr, "ERROR: could not open result.txt\n");
        return 1;
    }
    int *ref = (int *)malloc(nodes * sizeof(int));
    for (int v = 0; v < nodes; v++) {
        if (fscanf(f, "%d", &ref[v]) != 1) {
            fprintf(stderr, "ERROR: result.txt format mismatch\n");
            fclose(f);
            free(ref);
            return 1;
        }
    }
    fclose(f);

    // Check equality
    int ok = 1;
    for (int v = 0; v < nodes; v++) {
        if (nstat[v] != ref[v]) {
            ok = 0;
            break;
        }
    }

    if (ok) {
        printf("Fault Injection Test Success!\n");
    } else {
        printf("Fault Injection Test Failed!\n");
    }

    // Cleanup
    cudaFree(nidx_d);
    cudaFree(nlist_d);
    cudaFree(nstat_d);
    free(nidx);
    free(nlist);
    free(nstat);
    free(ref);

    return 0;
}
