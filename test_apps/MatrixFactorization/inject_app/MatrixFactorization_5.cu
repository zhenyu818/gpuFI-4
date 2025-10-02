#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define SEED 24680

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err__ = (call);                                                                                    \
        if (err__ != cudaSuccess) {                                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err__ << " ("                  \
                      << cudaGetErrorString(err__) << ")\n";                                                           \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

namespace config {
__constant__ int n_factors;
__constant__ float learning_rate;
__constant__ float P_reg;
__constant__ float Q_reg;
__constant__ float user_bias_reg;
__constant__ float item_bias_reg;
__constant__ bool is_train;
} // namespace config

static float parse_float(const std::string &s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "inf")
        return std::numeric_limits<float>::infinity();
    if (lower == "-inf")
        return -std::numeric_limits<float>::infinity();
    if (lower == "nan")
        return std::numeric_limits<float>::quiet_NaN();
    try {
        return std::stof(s);
    } catch (...) {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

static bool compare_floats(float a, float b, float tol) {
    if (std::isnan(a) && std::isnan(b))
        return true;
    if (std::isinf(a) && std::isinf(b))
        return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b))
        return false;
    return std::fabs(a - b) <= tol;
}

__device__ float get_prediction(int factors, const float *p, const float *q, float user_bias, float item_bias,
                                float global_bias) {
    float pred = global_bias + user_bias + item_bias;
    for (int f = 0; f < factors; ++f)
        pred += q[f] * p[f];
    return pred;
}

__global__ void select_item_owner(const int *indptr, const int *indices, const int *random_offsets, int n_rows,
                                  int *item_owner) {
    int user = blockDim.x * blockIdx.x + threadIdx.x;
    if (user >= n_rows)
        return;
    int low = indptr[user];
    int high = indptr[user + 1];
    if (low == high)
        return;
    int width = high - low;
    int choice = random_offsets[user] % width;
    int y_i = low + choice;
    int item = indices[y_i];
    atomicMin(&item_owner[item], user);
}

__global__ void sgd_update_deterministic(const int *indptr, const int *indices, const float *data, float *P,
                                         const float *Q, float *Q_target, int n_rows, float *user_bias,
                                         const float *item_bias, float *item_bias_target, const int *random_offsets,
                                         float global_bias, const int *item_owner, unsigned char *item_is_updated) {
    int user = blockDim.x * blockIdx.x + threadIdx.x;
    if (user >= n_rows)
        return;
    int low = indptr[user];
    int high = indptr[user + 1];
    if (low == high)
        return;
    int width = high - low;
    int choice = random_offsets[user] % width;
    int y_i = low + choice;
    int item = indices[y_i];
    float ub = user_bias[user];
    float ib = item_bias[item];
    float error = data[y_i] - get_prediction(config::n_factors, &P[user * config::n_factors],
                                             &Q[item * config::n_factors], ub, ib, global_bias);
    for (int f = 0; f < config::n_factors; ++f) {
        float P_old = P[user * config::n_factors + f];
        float Q_old = Q[item * config::n_factors + f];
        P[user * config::n_factors + f] = P_old + config::learning_rate * (error * Q_old - config::P_reg * P_old);
    }
    user_bias[user] += config::learning_rate * (error - config::user_bias_reg * ub);
    if (config::is_train && item_owner[item] == user) {
        for (int f = 0; f < config::n_factors; ++f) {
            float P_old = P[user * config::n_factors + f];
            float Q_old = Q[item * config::n_factors + f];
            Q_target[item * config::n_factors + f] =
                Q_old + config::learning_rate * (error * P_old - config::Q_reg * Q_old);
        }
        item_bias_target[item] = ib + config::learning_rate * (error - config::item_bias_reg * ib);
        item_is_updated[item] = 1u;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_users> <num_items> <items_per_user>\n";
        return EXIT_FAILURE;
    }
    int num_users = std::stoi(argv[1]);
    int num_items = std::stoi(argv[2]);
    int items_per_user = std::stoi(argv[3]);
    if (num_users <= 0 || num_items <= 0 || items_per_user <= 0 || items_per_user > num_items) {
        std::cerr << "Arguments must satisfy num_users>0, num_items>0, 0<items_per_user<=num_items.\n";
        return EXIT_FAILURE;
    }

    srand(SEED);

    const int latent_factors = 8;
    const float learning_rate = 0.01f;
    const float lambda_p = 0.02f;
    const float lambda_q = 0.02f;
    const float lambda_user_bias = 0.02f;
    const float lambda_item_bias = 0.02f;
    const bool is_train = true;
    CUDA_CHECK(cudaMemcpyToSymbol(config::n_factors, &latent_factors, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::learning_rate, &learning_rate, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::P_reg, &lambda_p, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::Q_reg, &lambda_q, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::user_bias_reg, &lambda_user_bias, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::item_bias_reg, &lambda_item_bias, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(config::is_train, &is_train, sizeof(bool)));

    int nnz = num_users * items_per_user;
    std::vector<int> h_indptr(num_users + 1);
    for (int u = 0; u <= num_users; ++u)
        h_indptr[u] = u * items_per_user;
    std::vector<int> h_indices(nnz);
    std::vector<float> h_data(nnz);
    for (int i = 0; i < nnz; ++i) {
        h_indices[i] = rand() % num_items;
        h_data[i] = (rand() % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : (float)(rand() % 10);
    }
    std::vector<float> h_P(num_users * latent_factors);
    std::vector<float> h_Q(num_items * latent_factors);
    std::vector<float> h_Q_target(num_items * latent_factors);
    for (size_t i = 0; i < h_P.size(); ++i)
        h_P[i] = (rand() % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : (float)(rand() % 10);
    for (size_t i = 0; i < h_Q.size(); ++i)
        h_Q[i] = (rand() % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : (float)(rand() % 10);
    h_Q_target = h_Q;
    std::vector<float> h_user_bias(num_users);
    std::vector<float> h_item_bias(num_items);
    std::vector<float> h_item_bias_target(num_items);
    for (int i = 0; i < num_users; ++i)
        h_user_bias[i] = (rand() % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : (float)(rand() % 10);
    for (int i = 0; i < num_items; ++i)
        h_item_bias[i] = (rand() % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : (float)(rand() % 10);
    h_item_bias_target = h_item_bias;
    std::vector<unsigned char> h_item_updated(num_items, 0u);
    std::vector<int> h_random_choice(num_users);
    for (int u = 0; u < num_users; ++u)
        h_random_choice[u] = rand() % items_per_user;
    float global_bias = 0.0f;

    int *d_indptr = nullptr, *d_indices = nullptr;
    float *d_data = nullptr, *d_P = nullptr, *d_Q = nullptr, *d_Q_target = nullptr, *d_user_bias = nullptr,
          *d_item_bias = nullptr, *d_item_bias_target = nullptr;
    unsigned char *d_item_is_updated = nullptr;
    int *d_random_choice = nullptr, *d_item_owner = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indptr, h_indptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, h_P.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, h_Q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_target, h_Q_target.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_user_bias, h_user_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_item_bias, h_item_bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_item_bias_target, h_item_bias_target.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_item_is_updated, h_item_updated.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_random_choice, h_random_choice.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_item_owner, num_items * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr.data(), h_indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, h_P.data(), h_P.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q_target, h_Q_target.data(), h_Q_target.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_user_bias, h_user_bias.data(), h_user_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_bias, h_item_bias.data(), h_item_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_bias_target, h_item_bias_target.data(), h_item_bias_target.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_is_updated, h_item_updated.data(), h_item_updated.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_random_choice, h_random_choice.data(), h_random_choice.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    {
        std::vector<int> tmp(num_items, std::numeric_limits<int>::max());
        CUDA_CHECK(cudaMemcpy(d_item_owner, tmp.data(), tmp.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    dim3 block_dim(std::min(128, num_users));
    if (block_dim.x == 0)
        block_dim.x = 1;
    dim3 grid_dim((num_users + block_dim.x - 1) / block_dim.x);
    if (grid_dim.x == 0)
        grid_dim.x = 1;
    select_item_owner<<<grid_dim, block_dim>>>(d_indptr, d_indices, d_random_choice, num_users, d_item_owner);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    sgd_update_deterministic<<<grid_dim, block_dim>>>(d_indptr, d_indices, d_data, d_P, d_Q, d_Q_target, num_users,
                                                      d_user_bias, d_item_bias, d_item_bias_target, d_random_choice,
                                                      0.0f, d_item_owner, d_item_is_updated);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_P.data(), d_P, h_P.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Q_target.data(), d_Q_target, h_Q_target.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_user_bias.data(), d_user_bias, h_user_bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_item_bias_target.data(), d_item_bias_target, h_item_bias_target.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_item_updated.data(), d_item_is_updated, h_item_updated.size() * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    size_t total_elements = (size_t)num_users * latent_factors + (size_t)num_items * latent_factors +
                            (size_t)num_users + (size_t)num_items + (size_t)num_items;
    std::ifstream file("result.txt");
    if (!file.is_open()) {
        std::cout << "Fault Injection Test Failed!\n";
        cudaFree(d_item_owner);
        cudaFree(d_random_choice);
        cudaFree(d_item_is_updated);
        cudaFree(d_item_bias_target);
        cudaFree(d_item_bias);
        cudaFree(d_user_bias);
        cudaFree(d_Q_target);
        cudaFree(d_Q);
        cudaFree(d_P);
        cudaFree(d_data);
        cudaFree(d_indices);
        cudaFree(d_indptr);
        return EXIT_FAILURE;
    }
    std::vector<std::string> ref_tokens;
    std::string line, token;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        while (iss >> token)
            ref_tokens.push_back(token);
    }
    file.close();
    if (ref_tokens.size() != total_elements) {
        std::cout << "Fault Injection Test Failed!\n";
        cudaFree(d_item_owner);
        cudaFree(d_random_choice);
        cudaFree(d_item_is_updated);
        cudaFree(d_item_bias_target);
        cudaFree(d_item_bias);
        cudaFree(d_user_bias);
        cudaFree(d_Q_target);
        cudaFree(d_Q);
        cudaFree(d_P);
        cudaFree(d_data);
        cudaFree(d_indices);
        cudaFree(d_indptr);
        return EXIT_FAILURE;
    }
    size_t p_start = 0, p_end = p_start + h_P.size();
    size_t q_start = p_end, q_end = q_start + h_Q_target.size();
    size_t ub_start = q_end, ub_end = ub_start + h_user_bias.size();
    size_t ib_start = ub_end, ib_end = ib_start + h_item_bias_target.size();
    size_t updated_start = ib_end, updated_end = updated_start + h_item_updated.size();
    const float tol = 1e-5f;
    bool match = true;
    for (size_t i = 0; i < h_P.size() && match; ++i)
        match &= compare_floats(h_P[i], parse_float(ref_tokens[p_start + i]), tol);
    for (size_t i = 0; i < h_Q_target.size() && match; ++i)
        match &= compare_floats(h_Q_target[i], parse_float(ref_tokens[q_start + i]), tol);
    for (size_t i = 0; i < h_user_bias.size() && match; ++i)
        match &= compare_floats(h_user_bias[i], parse_float(ref_tokens[ub_start + i]), tol);
    for (size_t i = 0; i < h_item_bias_target.size() && match; ++i)
        match &= compare_floats(h_item_bias_target[i], parse_float(ref_tokens[ib_start + i]), tol);
    for (size_t i = 0; i < h_item_updated.size() && match; ++i) {
        int ref_int;
        try {
            ref_int = std::stoi(ref_tokens[updated_start + i]);
        } catch (...) {
            match = false;
            break;
        }
        if (ref_int != (int)h_item_updated[i])
            match = false;
    }
    std::cout << (match ? "Fault Injection Test Success!\n" : "Fault Injection Test Failed!\n");
    cudaFree(d_item_owner);
    cudaFree(d_random_choice);
    cudaFree(d_item_is_updated);
    cudaFree(d_item_bias_target);
    cudaFree(d_item_bias);
    cudaFree(d_user_bias);
    cudaFree(d_Q_target);
    cudaFree(d_Q);
    cudaFree(d_P);
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_indptr);
    return EXIT_SUCCESS;
}
