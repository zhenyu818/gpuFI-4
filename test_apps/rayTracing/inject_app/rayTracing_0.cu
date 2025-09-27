// Single-file minimal CUDA ray tracer
// Contains only code used by main render path

#include <ctime>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <float.h>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>

#ifndef MAXFLOAT
#define MAXFLOAT FLT_MAX
#endif

// ---------------- core/vec3 ----------------
class vec3  {
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0){ e[0]=e0; e[1]=e0; e[2]=e0; }
    __host__ __device__ vec3(float e0, float e1, float e2){ e[0]=e0; e[1]=e1; e[2]=e2; }
    __host__ __device__ inline float x() const {return e[0];}
    __host__ __device__ inline float y() const {return e[1];}
    __host__ __device__ inline float z() const {return e[2];}
    __host__ __device__ inline float r() const {return e[0];}
    __host__ __device__ inline float g() const {return e[1];}
    __host__ __device__ inline float b() const {return e[2];}
    __host__ __device__ inline float operator[](int i) const {return e[i];}
    __host__ __device__ inline float& operator[](int i) {return e[i];}
    __host__ __device__ inline vec3& operator+=(const vec3 &v){ e[0]+=v.e[0]; e[1]+=v.e[1]; e[2]+=v.e[2]; return *this; }
    __host__ __device__ inline vec3& operator-=(const vec3 &v){ e[0]-=v.e[0]; e[1]-=v.e[1]; e[2]-=v.e[2]; return *this; }
    // removed unused element-wise and scalar +/- operators
    __host__ __device__ inline vec3& operator*=(const float t){ e[0]*=t; e[1]*=t; e[2]*=t; return *this; }
    __host__ __device__ inline vec3& operator/=(const float t){ float k=1.0f/t; e[0]*=k; e[1]*=k; e[2]*=k; return *this; }
    __host__ __device__ inline float length() const{ return sqrtf(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]); }
    // removed squared_length and make_unit_vector (unused)
    float e[3];
};
inline std::ostream& operator<<(std::ostream &os, const vec3 &t){ os << "(" << t[0] << ", " << t[1] << ", " << t[2] << ")"; return os; }
__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2){ return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]); }
__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2){ return vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]); }
// removed vec3 +/- float (unused)
__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2){ return vec3(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]); }
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) { return vec3(t*v.e[0], t*v.e[1], t*v.e[2]); }
// removed vec3 * float (unused)
__host__ __device__ inline vec3 operator/(const vec3& v, float t){ return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t); }
__host__ __device__ inline float dot(const vec3& v1, const vec3& v2){ return v1.e[0]*v2.e[0]+v1.e[1]*v2.e[1]+v1.e[2]*v2.e[2]; }
__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2){ return vec3((v1.e[1]*v2.e[2]-v1.e[2]*v2.e[1]), (-(v1.e[0]*v2.e[2]-v1.e[2]*v2.e[0])), (v1.e[0]*v2.e[1]-v1.e[1]*v2.e[0])); }
__host__ __device__ float clip_single(float f, int minv, int maxv){ if(f>maxv) return maxv; else if(f<minv) return minv; return f; }
__host__ __device__ inline vec3 clip(const vec3& v, int minv=0.0f, int maxv=1.0f){ vec3 vr(0,0,0); vr[0]=clip_single(v[0],minv,maxv); vr[1]=clip_single(v[1],minv,maxv); vr[2]=clip_single(v[2],minv,maxv); return vr; }
__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.length(); }

// ---------------- core/ray ----------------
class Ray {
public:
    __device__ Ray() : _time(0.f), _origin(vec3(0.f)), _direction(vec3(0.f)) {}
    __device__ Ray(const vec3& o, const vec3& d, float t=0.f) : _time(t), _origin(o), _direction(d) {}
    __device__ float time() const { return _time; }
    __device__ vec3 origin() const { return _origin; }
    __device__ vec3 direction() const { return _direction; }
    __device__ vec3 point_at_t(float t) const { return _origin + t * _direction; }
    float _time; vec3 _origin; vec3 _direction;
};

// ---------------- core/aabb ----------------
__device__ inline float ffmin(float a, float b) {return a < b ? a : b;}
__device__ inline float ffmax(float a, float b) {return a > b ? a : b;}
class AABB {
public:
    __device__ AABB(){ float minNum=FLT_MIN, maxNum=FLT_MAX; _min=vec3(maxNum,maxNum,maxNum); _max=vec3(minNum,minNum,minNum); }
    __device__ AABB(const vec3& p1, const vec3& p2) : _min(p1), _max(p2) {}
    __device__ bool hit(const Ray& r, float t_min, float t_max) const {
        for(int a=0;a<3;a++){
            float t0 = ffmin((_min[a]-r.origin()[a])/r.direction()[a], (_max[a]-r.origin()[a])/r.direction()[a]);
            float t1 = ffmax((_min[a]-r.origin()[a])/r.direction()[a], (_max[a]-r.origin()[a])/r.direction()[a]);
            t_min = ffmax(t0, t_min);
            t_max = ffmin(t1, t_max);
            if(t_max <= t_min) return false;
        }
        return true;
    }
    __device__ vec3 min() const { return _min; }
    __device__ vec3 max() const { return _max; }
    vec3 _min, _max;
};
__device__ AABB surrounding_box(AABB box0, AABB box1){ vec3 small(fminf(box0.min().x(), box1.min().x()), fminf(box0.min().y(), box1.min().y()), fminf(box0.min().z(), box1.min().z())); vec3 big(fmaxf(box0.max().x(), box1.max().x()), fmaxf(box0.max().y(), box1.max().y()), fmaxf(box0.max().z(), box1.max().z())); return AABB(small, big); }

// ---------------- hitable ----------------
class Material; // fwd
struct HitRecord{ float t; float u; float v; vec3 p; vec3 normal; Material* mat_ptr; };
class Hitable { public: __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0; __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const = 0; };

// ---------------- texture/material ----------------
class Texture { public: __device__ virtual vec3 value(float, float, const vec3&) const = 0; };
class ConstantTexture: public Texture { public: __device__ ConstantTexture() {} __device__ ConstantTexture(vec3 c): color(c) {} __device__ virtual vec3 value(float, float, const vec3&) const { return color; } vec3 color; };
class Material { public: __device__ virtual bool scatter(const Ray&, const HitRecord&, vec3&, Ray&, curandState*) const = 0; __device__ virtual vec3 emitted(float, float, const vec3&) const { return vec3(0,0,0); } };
class DiffuseLight: public Material{ public: __device__ DiffuseLight(Texture* texture): emit(texture) {} __device__ virtual bool scatter(const Ray&, const HitRecord&, vec3&, Ray&, curandState*) const{ return false; } __device__ virtual vec3 emitted(float u, float v, const vec3& p) const{ return emit->value(u,v,p); } Texture* emit; };

// ---------------- triangle ----------------
class Triangle: public Hitable {
public:
    __device__ Triangle() : EPSILON(0.000001f) {}
    __device__ Triangle(vec3 vs[3], Material* mat, bool cull=false) : EPSILON(0.000001f) { vertices[0]=vs[0]; vertices[1]=vs[1]; vertices[2]=vs[2]; material=mat; backCulling=cull; }
    __device__ virtual bool hit(const Ray& r, float, float, HitRecord& rec) const{
        vec3 v0=vertices[0], v1=vertices[1], v2=vertices[2]; vec3 e1=v1-v0, e2=v2-v0; vec3 h=cross(r.direction(), e2), s, q; float a=dot(e1,h); if(a<EPSILON && backCulling) return false; if(a>-EPSILON && a<EPSILON) return false; float f=1.0f/a; s=r.origin()-v0; float u=f*dot(s,h); if(u<0.0f || u>1.0f) return false; q=cross(s,e1); float v=f*dot(r.direction(),q); if(v<0.0f || u+v>1.0f) return false; float t=f*dot(e2,q); rec.t=t; rec.p=r.point_at_t(rec.t); rec.normal=unit_vector(cross(e1,e2)); rec.mat_ptr=material; return true;
    }
    __device__ virtual bool bounding_box(float, float, AABB& bbox) const{
        float minX=fminf(vertices[0][0], fminf(vertices[1][0], vertices[2][0])); float minY=fminf(vertices[0][1], fminf(vertices[1][1], vertices[2][1])); float minZ=fminf(vertices[0][2], fminf(vertices[1][2], vertices[2][2]));
        float maxX=fmaxf(vertices[0][0], fmaxf(vertices[1][0], vertices[2][0])); float maxY=fmaxf(vertices[0][1], fmaxf(vertices[1][1], vertices[2][1])); float maxZ=fmaxf(vertices[0][2], fmaxf(vertices[1][2], vertices[2][2]));
        bbox=AABB(vec3(minX,minY,minZ), vec3(maxX,maxY,maxZ)); return true;
    }
    const float EPSILON; vec3 vertices[3]; bool backCulling; Material* material;
};

// ---------------- transform/camera ----------------
__device__ vec3 random_in_unit_disk(curandState* state){ vec3 p; do{ p = 2.0f * vec3(curand_uniform(state), curand_uniform(state), 0) - vec3(1,1,0); } while(dot(p,p) >= 1.0f); return p; }
class Camera {
public:
    __device__ Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist){ lens_radius=aperture/2.0f; float theta=vfov * M_PI / 180.0f; float half_height=tanf(theta/2.0f); float half_width=half_height*aspect; origin=lookfrom; z=unit_vector(lookfrom - lookat); x=unit_vector(cross(vup, z)); y=cross(z, x); lower_left_corner=origin - half_width*focus_dist*x - half_height*focus_dist*y - focus_dist*z; horizontal=2.0f * half_width * focus_dist * x; vertical=2.0f * half_height * focus_dist * y; }
    __device__ Ray get_ray(float s, float t, curandState* state){ vec3 rd = lens_radius * random_in_unit_disk(state); vec3 offset = x * rd.x() + y * rd.y(); return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset, 0.0f); }
    vec3 lower_left_corner, horizontal, vertical, origin; vec3 x,y,z; float lens_radius;
};
class MotionCamera: public Camera{
public:
    __device__ MotionCamera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) : Camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist){ time0=t0; time1=t1; }
    __device__ Ray get_ray(float s, float t, curandState* state){ vec3 rd = lens_radius * random_in_unit_disk(state); vec3 offset = x * rd.x() + y * rd.y(); float time = time0 + curand_uniform(state) * (time1 - time0); return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset, time); }
    vec3 lower_left_corner, horizontal, vertical, origin; vec3 x,y,z; float lens_radius; float time0, time1;
};

// ---------------- BVH ----------------
struct BoxCompare { __device__ BoxCompare(int m): mode(m) {} __device__ bool operator()(Hitable* a, Hitable* b) const { AABB box_left, box_right; if(!a->bounding_box(0,0,box_left) || !b->bounding_box(0,0,box_right)) return false; float val1=0, val2=0; if(mode==1){ val1=box_left.min().x(); val2=box_right.min().x(); } else if(mode==2){ val1=box_left.min().y(); val2=box_right.min().y(); } else { val1=box_left.min().z(); val2=box_right.min().z(); } return !(val1 - val2 < 0.0f); } int mode; };
class BVHNode: public Hitable{
public:
    __device__ BVHNode() {}
    __device__ BVHNode(Hitable **l, int n, float time0, float time1, curandState *state){ int axis=int(3*curand_uniform(state)); if(axis==0){ thrust::sort(l,l+n,BoxCompare(1)); } else if(axis==1){ thrust::sort(l,l+n,BoxCompare(2)); } else { thrust::sort(l,l+n,BoxCompare(3)); } if(n==1){ left=right=l[0]; } else if(n==2){ left=l[0]; right=l[1]; } else { left=new BVHNode(      l,   n/2, time0,time1,state); right=new BVHNode(l+n/2, n-n/2, time0,time1,state); } AABB box_left, box_right; if(!left->bounding_box(time0,time1,box_left) || !right->bounding_box(time0,time1,box_right)) return; box=surrounding_box(box_left, box_right); }
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const{ if(box.hit(r,t_min,t_max)){ HitRecord left_rec, right_rec; bool hit_left=left->hit(r,t_min,t_max,left_rec); bool hit_right=right->hit(r,t_min,t_max,right_rec); if(hit_left && hit_right){ rec = (left_rec.t < right_rec.t) ? left_rec : right_rec; return true; } else if(hit_left){ rec=left_rec; return true; } else if(hit_right){ rec=right_rec; return true; } else { return false; } } return false; }
    __device__ virtual bool bounding_box(float, float, AABB& b) const{ b=box; return true; }
    Hitable* left; Hitable* right; AABB box;
};

// ---------------- obj loader ----------------
vec3 computeMean(vec3* points, int np){ vec3 mean(0,0,0); for(int i=0;i<np;i++){ mean += points[i]; } return mean / float(np); }
void centering(vec3* points, vec3& mean, int np){ for(int i=0;i<np;i++){ points[i] -= mean; } }
void parseObjByName(std::string fn, vec3* points, vec3* idxVertex, int& nPoints, int& nTriangles){ std::ifstream objFile(fn.c_str()); int np=0, nt=0; if(objFile.is_open()){ std::string line; while(std::getline(objFile,line)){ std::stringstream ss; ss<<line; std::string label; ss>>label; if(label=="v"){ vec3 point; ss>>point[0]>>point[1]>>point[2]; points[np++]=point; } else if(label=="f"){ vec3 idx; ss>>idx[0]>>idx[1]>>idx[2]; idxVertex[nt++]=idx; } } } else { std::cerr << "Can't open the file " << fn << std::endl; } nPoints=np; nTriangles=nt; vec3 mean=computeMean(points,np); centering(points,mean,np); }

// ---------------- device rand ----------------
__device__ float rand(curandState *state){ return float(curand_uniform(state)); }

// ---------------- scene: mesh builder ----------------
__device__ void draw_one_mesh(Hitable** mesh, Hitable** triangles, vec3* points, vec3* idxVertex, int /*np*/, int nt, curandState *state){ Material* mat=new DiffuseLight(new ConstantTexture(vec3(0.4,0.7,0.5))); int l=0; for(int i=0;i<nt;i++){ vec3 idx=idxVertex[i]; vec3 v[3]={points[int(idx[2])], points[int(idx[1])], points[int(idx[0])]}; triangles[l++]=new Triangle(v,mat,true); } *mesh=new BVHNode(triangles,l,0,1,state); }

// ---------------- rendering kernels ----------------
__device__ vec3 shade(const Ray& r, Hitable **world, int depth, curandState *state){ HitRecord rec; if((*world)->hit(r, 0.001f, MAXFLOAT, rec)) { Ray scattered; vec3 attenuation; vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p); if(depth < 15 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)){ return emitted + attenuation * shade(scattered, world, depth + 1, state); } else { return emitted; } } else { return vec3(0,0,0); } }

__global__ void random_init(int nx, int ny, curandState *state){ int x=threadIdx.x + blockIdx.x * blockDim.x; int y=threadIdx.y + blockIdx.y * blockDim.y; if((x>=nx)||(y>=ny)) return; int pixel_index=y*nx+x; curand_init(0, pixel_index, 0, &state[pixel_index]); }

__global__ void build_mesh(Hitable** mesh, Camera** camera, Hitable** triangles, vec3* points, vec3* idxVertex, int np, int nt, curandState *state, int nx, int ny, int /*cnt*/){ if(threadIdx.x==0 && blockIdx.x==0){ draw_one_mesh(mesh, triangles, points, idxVertex, np, nt, state); vec3 lookfrom(0,0,10); vec3 lookat(0,0,0); float dist_to_focus=10.0f; float aperture=0.0f; float vfov=60.0f; *camera = new MotionCamera(lookfrom, lookat, vec3(0,1,0), vfov, float(nx)/float(ny), aperture, dist_to_focus, 0.0f, 1.0f); } }

__global__ void destroy(Hitable** obj_list, Hitable** world, Camera** camera, int obj_cnt){ for(int i=0;i<obj_cnt;i++){ delete *(obj_list + i); } delete *world; delete *camera; }

__global__ void render(vec3* colorBuffer, Hitable** world, Camera** camera, curandState* state, int nx, int ny, int samples){ int x=threadIdx.x + blockIdx.x * blockDim.x; int y=threadIdx.y + blockIdx.y * blockDim.y; if((x>=nx)||(y>=ny)) return; int pixel_index=y*nx+x; vec3 col(0,0,0); for(int i=0;i<samples;i++){ float u=float(x + rand(&(state[pixel_index]))) / float(nx); float v=float(y + rand(&(state[pixel_index]))) / float(ny); Ray r = (*camera)->get_ray(u, v, state); col += shade(r, world, 0, &(state[pixel_index])); } col/=float(samples); col[0]=sqrtf(col[0]); col[1]=sqrtf(col[1]); col[2]=sqrtf(col[2]); colorBuffer[pixel_index]=clip(col); }

// ---------------- main ----------------
#define RESOLUTION 1
#define SAMPLES 2

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){ if(result){ std::cerr << "CUDA error = "<< static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "'\n"; cudaDeviceReset(); exit(99); } }

int main(){
    std::time_t tic = std::time(NULL); std::cout << "Start running at: " << std::asctime(std::localtime(&tic)) << std::endl;
    std::ofstream imgWrite("images/image.ppm");
    int nx=8*RESOLUTION, ny=8*RESOLUTION; int tx=16, ty=16; int num_pixel=nx*ny;
    vec3 *colorBuffer; checkCudaErrors(cudaMallocManaged((void**)&colorBuffer, num_pixel*sizeof(vec3)));
    curandState* curand_state; checkCudaErrors(cudaMallocManaged((void**)&curand_state, num_pixel*sizeof(curandState)));
    int obj_cnt=488; Hitable** obj_list; Hitable** world; Camera** camera; checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt*sizeof(Hitable*))); checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(Hitable*))); checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    dim3 blocks(nx/tx + 1, ny/ty + 1), threads(tx, ty); random_init<<<blocks, threads>>>(nx, ny, curand_state); checkCudaErrors(cudaGetLastError()); checkCudaErrors(cudaDeviceSynchronize());
    vec3* points; vec3* idxVertex; checkCudaErrors(cudaMallocManaged((void**)&points,    2600 * sizeof(vec3))); checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));
    int nPoints, nTriangles; parseObjByName("./shapes/random_small.obj", points, idxVertex, nPoints, nTriangles);
    std::cout << "# of points: " << nPoints << std::endl; std::cout << "# of triangles: " << nTriangles << std::endl;
    for(int i=0;i<nPoints;i++){ points[i] *= 30.0f; }
    for(int i=0;i<nPoints;i++){ std::cout << points[i] << std::endl; }
    Hitable** triangles; checkCudaErrors(cudaMallocManaged((void**)&triangles, nTriangles * sizeof(Hitable*)));
    build_mesh<<<1,1>>>(world, camera, triangles, points, idxVertex, nPoints, nTriangles, curand_state, nx, ny, obj_cnt); checkCudaErrors(cudaGetLastError()); checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(colorBuffer, world, camera, curand_state, nx, ny, SAMPLES); checkCudaErrors(cudaGetLastError()); checkCudaErrors(cudaDeviceSynchronize());
    imgWrite << "P3\n" << nx << " " << ny << "\n255\n"; for(int i=ny-1;i>=0;i--){ for(int j=0;j<nx;j++){ size_t idx=i*nx+j; int ir=int(255.99f*colorBuffer[idx].r()); int ig=int(255.99f*colorBuffer[idx].g()); int ib=int(255.99f*colorBuffer[idx].b()); imgWrite << ir << " " << ig << " " << ib << "\n"; } }
    checkCudaErrors(cudaDeviceSynchronize()); destroy<<<1,1>>>(obj_list, world, camera, obj_cnt); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world)); checkCudaErrors(cudaFree(obj_list)); checkCudaErrors(cudaFree(camera)); checkCudaErrors(cudaFree(curand_state)); checkCudaErrors(cudaFree(colorBuffer));
    cudaDeviceReset(); std::time_t toc = std::time(NULL); std::cout << "Finish running at: " << std::asctime(std::localtime(&toc)) << std::endl; std::cout << "Time consuming: " << toc - tic << "s" << std::endl; return 0;
}
