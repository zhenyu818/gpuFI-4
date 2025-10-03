#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Vec3 {
    float e[3];
    __host__ __device__ Vec3() : e{0, 0, 0} {
    }
    __host__ __device__ Vec3(float x, float y, float z) : e{x, y, z} {
    }
    __host__ __device__ float x() const {
        return e[0];
    }
    __host__ __device__ float y() const {
        return e[1];
    }
    __host__ __device__ float z() const {
        return e[2];
    }
    __host__ __device__ Vec3 operator+(const Vec3 &o) const {
        return Vec3(e[0] + o.e[0], e[1] + o.e[1], e[2] + o.e[2]);
    }
    __host__ __device__ Vec3 operator-(const Vec3 &o) const {
        return Vec3(e[0] - o.e[0], e[1] - o.e[1], e[2] - o.e[2]);
    }
    __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(e[0] * t, e[1] * t, e[2] * t);
    }
    __host__ __device__ Vec3 operator/(float t) const {
        return Vec3(e[0] / t, e[1] / t, e[2] / t);
    }
    __host__ __device__ Vec3 &operator+=(const Vec3 &o) {
        e[0] += o.e[0];
        e[1] += o.e[1];
        e[2] += o.e[2];
        return *this;
    }
};
__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return v * t;
}
__host__ __device__ inline float dot(const Vec3 &a, const Vec3 &b) {
    return a.e[0] * b.e[0] + a.e[1] * b.e[1] + a.e[2] * b.e[2];
}
__host__ __device__ inline float length(const Vec3 &v) {
    return sqrtf(dot(v, v));
}
__host__ __device__ inline Vec3 unit_vector(const Vec3 &v) {
    return v / length(v);
}
__host__ __device__ inline Vec3 clip01(const Vec3 &v) {
    return Vec3(fminf(fmaxf(v.e[0], 0.0f), 0.999f), fminf(fmaxf(v.e[1], 0.0f), 0.999f),
                fminf(fmaxf(v.e[2], 0.0f), 0.999f));
}
struct Ray {
    Vec3 A;
    Vec3 B;
    __host__ __device__ Ray() {
    }
    __host__ __device__ Ray(const Vec3 &a, const Vec3 &b) : A(a), B(b) {
    }
    __host__ __device__ Vec3 origin() const {
        return A;
    }
    __host__ __device__ Vec3 direction() const {
        return B;
    }
    __host__ __device__ Vec3 point_at_parameter(float t) const {
        return A + t * B;
    }
};
__device__ bool hit_sphere(const Vec3 &center, float radius, const Ray &r, float &tHit, Vec3 &n) {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0f)
        return false;
    float sdisc = sqrtf(disc);
    float t0 = (-b - sdisc) / (2.0f * a);
    float t1 = (-b + sdisc) / (2.0f * a);
    float t = t0;
    if (t < 0.001f)
        t = t1;
    if (t < 0.001f)
        return false;
    tHit = t;
    n = (r.point_at_parameter(t) - center) / radius;
    return true;
}
__global__ void render(Vec3 *colorBuffer, const float *randU, const float *randV, int nx, int ny, int samples) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny)
        return;
    int pixel_index = y * nx + x;
    Vec3 lower_left_corner(-2.0f, -1.0f, -1.0f);
    Vec3 horizontal(4.0f, 0.0f, 0.0f);
    Vec3 vertical(0.0f, 2.0f, 0.0f);
    Vec3 origin(0.0f, 0.0f, 0.0f);
    Vec3 col(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < samples; ++s) {
        int idx = pixel_index * samples + s;
        float u = (x + randU[idx]) / float(nx);
        float v = (y + randV[idx]) / float(ny);
        Ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
        float tHit;
        Vec3 n;
        Vec3 sample;
        if (hit_sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, r, tHit, n)) {
            sample = 0.5f * Vec3(n.x() + 1.0f, n.y() + 1.0f, n.z() + 1.0f);
        } else {
            Vec3 unit_dir = unit_vector(r.direction());
            float t = 0.5f * (unit_dir.y() + 1.0f);
            sample = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
        }
        col += sample;
    }
    col = col / float(samples);
    col = Vec3(sqrtf(col.x()), sqrtf(col.y()), sqrtf(col.z()));
    colorBuffer[pixel_index] = clip01(col);
}
int main(int argc, char **argv) {
    int nx = 8, ny = 4, samples = 8;
    if (argc > 1) {
        nx = std::stoi(std::string(argv[1]));
        if (argc > 2) {
            ny = std::stoi(std::string(argv[2]));
            if (argc > 3) {
                samples = std::stoi(std::string(argv[3]));
            }
        }
    }
    size_t num_pixels = (size_t)nx * (size_t)ny;
    size_t jitter_count = num_pixels * (size_t)samples;
    std::vector<float> h_randU(jitter_count, 0.0f), h_randV(jitter_count, 0.0f);
    std::vector<Vec3> h_colorBuffer(num_pixels);
    for (size_t i = 0; i < jitter_count; ++i) {
        if (rand() % 2 == 0) {
            h_randU[i] = NAN;
            h_randV[i] = NAN;
        } else {
            h_randU[i] = (float)rand() / RAND_MAX;
            h_randV[i] = (float)rand() / RAND_MAX;
        }
    }
    Vec3 *d_colorBuffer = nullptr;
    float *d_randU = nullptr, *d_randV = nullptr;
    cudaMalloc(&d_colorBuffer, num_pixels * sizeof(Vec3));
    cudaMalloc(&d_randU, jitter_count * sizeof(float));
    cudaMalloc(&d_randV, jitter_count * sizeof(float));
    cudaMemcpy(d_randU, h_randU.data(), jitter_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_randV, h_randV.data(), jitter_count * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    render<<<grid, block>>>(d_colorBuffer, d_randU, d_randV, nx, ny, samples);
    cudaDeviceSynchronize();
    cudaMemcpy(h_colorBuffer.data(), d_colorBuffer, num_pixels * sizeof(Vec3), cudaMemcpyDeviceToHost);
    std::cout << std::setprecision(6) << std::fixed;
    for (size_t i = 0; i < num_pixels; ++i) {
        std::cout << h_colorBuffer[i].x() << ' ' << h_colorBuffer[i].y() << ' ' << h_colorBuffer[i].z();
        if (i + 1 < num_pixels)
            std::cout << ' ';
    }
    std::cout << '\n';
    cudaFree(d_randV);
    cudaFree(d_randU);
    cudaFree(d_colorBuffer);
    return 0;
}
