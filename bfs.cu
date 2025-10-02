#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <stdbool.h>

#define MAX_THREADS_PER_BLOCK 256
#define SEED 12345

// Structure to hold a node information
struct Node {
    int starting;
    int no_of_edges;
};

__global__ void
Kernel(const Node* __restrict__ d_graph_nodes, 
       const int* __restrict__ d_graph_edges,
       char* __restrict__ d_graph_mask,
       char* __restrict__ d_updating_graph_mask,
       const char *__restrict__ d_graph_visited,
       int* __restrict__ d_cost,
       const int no_of_nodes) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < no_of_nodes && d_graph_mask[tid]) {
        d_graph_mask[tid] = 0;
        const int num_edges = d_graph_nodes[tid].no_of_edges;
        const int starting = d_graph_nodes[tid].starting;

        for (int i = starting; i < (num_edges + starting); i++) {
            int id = d_graph_edges[i];
            if (!d_graph_visited[id]) {
                d_cost[id] = d_cost[tid] + 1;
                d_updating_graph_mask[id] = 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <no_of_nodes> <out_degree>\n", argv[0]);
        return 1;
    }

    int no_of_nodes = atoi(argv[1]);
    int out_degree = atoi(argv[2]);

    if (no_of_nodes <= 0 || out_degree < 0) {
        fprintf(stderr, "Invalid arguments: no_of_nodes must be positive, out_degree non-negative\n");
        return 1;
    }

    srand(SEED);

    int edge_list_size = no_of_nodes * out_degree;
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node) * no_of_nodes);
    int* h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
    char *h_graph_mask = (char*) malloc(sizeof(char) * no_of_nodes);
    char *h_updating_graph_mask = (char*) malloc(sizeof(char) * no_of_nodes);
    char *h_graph_visited = (char*) malloc(sizeof(char) * no_of_nodes);
    int *h_cost = (int*) malloc(sizeof(int) * no_of_nodes);

    // Generate random graph
    int current_edge = 0;
    for (int i = 0; i < no_of_nodes; i++) {
        h_graph_nodes[i].starting = current_edge;
        h_graph_nodes[i].no_of_edges = out_degree;
        for (int j = 0; j < out_degree; j++) {
            h_graph_edges[current_edge + j] = rand() % no_of_nodes;
        }
        current_edge += out_degree;
    }

    // Initialize
    for (int i = 0; i < no_of_nodes; i++) {
        h_graph_mask[i] = 0;
        h_updating_graph_mask[i] = 0;
        h_graph_visited[i] = 0;
        h_cost[i] = -1;
    }
    int source = 0;
    h_graph_mask[source] = 1;
    h_graph_visited[source] = 1;
    h_cost[source] = 0;

    // Allocate device memory and copy inputs
    Node* d_graph_nodes;
    cudaMalloc((void**) &d_graph_nodes, sizeof(Node) * no_of_nodes);
    cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, cudaMemcpyHostToDevice);

    int* d_graph_edges;
    cudaMalloc((void**) &d_graph_edges, sizeof(int) * edge_list_size);
    cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, cudaMemcpyHostToDevice);

    char* d_graph_mask;
    cudaMalloc((void**) &d_graph_mask, sizeof(char) * no_of_nodes);
    cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);

    char* d_updating_graph_mask;
    cudaMalloc((void**) &d_updating_graph_mask, sizeof(char) * no_of_nodes);
    cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);

    char* d_graph_visited;
    cudaMalloc((void**) &d_graph_visited, sizeof(char) * no_of_nodes);
    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(char) * no_of_nodes, cudaMemcpyHostToDevice);

    int* d_cost;
    cudaMalloc((void**) &d_cost, sizeof(int) * no_of_nodes);
    cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes, cudaMemcpyHostToDevice);

    // Setup execution parameters
    dim3 grid((no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
    dim3 threads(MAX_THREADS_PER_BLOCK);

    // Execute kernel only once
    Kernel<<<grid, threads>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, 
                              d_graph_visited, d_cost, no_of_nodes);

    cudaDeviceSynchronize();

    // Copy back the output arrays
    cudaMemcpy(h_graph_mask, d_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_updating_graph_mask, d_updating_graph_mask, sizeof(char) * no_of_nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, sizeof(int) * no_of_nodes, cudaMemcpyDeviceToHost);

    // Compare with result.txt
    FILE* fp = fopen("result.txt", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open result.txt\n");
        // Cleanup code below
    } else {
        bool match = true;
        int total_elements = 3 * no_of_nodes;
        int read_count = 0;
        int ref_val;

        // Read graph_mask
        for (int i = 0; i < no_of_nodes && match; i++) {
            if (fscanf(fp, "%d", &ref_val) != 1) {
                match = false;
                break;
            }
            read_count++;
            if (ref_val != (int)h_graph_mask[i]) {
                match = false;
            }
        }

        // Read updating_graph_mask
        for (int i = 0; i < no_of_nodes && match; i++) {
            if (fscanf(fp, "%d", &ref_val) != 1) {
                match = false;
                break;
            }
            read_count++;
            if (ref_val != (int)h_updating_graph_mask[i]) {
                match = false;
            }
        }

        // Read cost
        for (int i = 0; i < no_of_nodes && match; i++) {
            if (fscanf(fp, "%d", &ref_val) != 1) {
                match = false;
                break;
            }
            read_count++;
            if (ref_val != h_cost[i]) {
                match = false;
            }
        }

        // Check if exactly the right number of elements were read (no extra or missing)
        if (read_count != total_elements) {
            match = false;
        }

        // Consume any remaining input to check for extra data
        int extra;
        if (fscanf(fp, "%d", &extra) == 1) {
            match = false;
        }

        fclose(fp);

        if (match) {
            printf("Fault Injection Test Success!\n");
        } else {
            printf("Fault Injection Test Failed!\n");
        }
    }

    // Cleanup
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);

    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);

    return 0;
}