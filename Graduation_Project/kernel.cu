/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 // Utilities and system includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <curand_kernel.h>
#include <math_functions.h>
#include <hittable_list.h>
#include <ray.h>
#include <sphere.h>
#include <hittable_list.h>
#include <material.h>
#include <vec3.h>
#include <camera.h>
hittable_list** world;
hittable** objects;
camera** cam;
curandState* random_state;
// convert floating point rgb color to 8-bit integer
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}
__global__ void CalculatePerPixel(hittable_list** world,camera** camera,curandState* global_rand_state,int spp,unsigned int* g_odata, int imgh) {
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    int index = x + y * imgh;
    //ray r = cam.get_ray(x, y);
    curandState local_rand_state = global_rand_state[index];
    vec3 color(0, 0, 0);
    for (int i = 0; i < spp; i++) {
        ray r = (*camera)->get_ray(x + curand_uniform(&local_rand_state), y + curand_uniform(&local_rand_state),&local_rand_state);
        color += (*camera)->ray_color(r, (*camera)->max_depth, world,&local_rand_state);
    }
    color /= float(spp);
    printf("%f %f %f\n", color.x(), color.y(), color.z());
    //g_odata[index] = rgbToInt(color.x(), color.y(), color.z());
    g_odata[index] = rgbToInt((float)x / 800 * 255, (float)y / 800 * 255, 0);

    global_rand_state[index] = local_rand_state;
}
__global__ void initCamera(camera** ca,float ar,int iw,int spp,int md,int vfov, vec3 lf, vec3 la,vec3 vu) {

    *ca = new camera(ar,iw,spp,md,vfov,lf,la,vu);

}

__global__ void initWorld(hittable_list** world,hittable** objects) {
    lambertian* material_ground = new lambertian(vec3(0.8, 0.8, 0.0));
    lambertian* material_center = new lambertian(vec3(0.7, 0.3, 0.3));

    sphere* ground = new sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground);
    sphere* center = new sphere(vec3(0.0, 0.0, -1.0), 0.5, material_center);


    (*world)->object_counts = 10;//오브젝트 개수 정의 아래 cudamalloc과 맞춰줄 필요 있음
    (*world) = new hittable_list(objects);
    (*world)->add(ground);
    (*world)->add(center);
}
extern "C" void initTracing() {

    cudaMalloc(&cam, sizeof(camera*));
    initCamera << <1, 1 >> > (cam, 16.0 / 9.0, 400, 10, 50, 90, vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0));
    cudaMalloc(&objects, 10*sizeof(hittable*));//오브젝트 개수만큼 할당 필요
    cudaMalloc(&world, sizeof(hittable_list*));
    initWorld << <1, 1 >> > (world,objects);
}

__global__ void Random_Init(curandState* global_state,int ih) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    unsigned int pixel_index = x+y*ih;
    curandState s;
    curand_init(pixel_index, 0, 0, &global_state[pixel_index]);
}
extern "C" void initCuda(dim3 grid,dim3 block,int image_height,int image_width,int pixels) {
    cudaMalloc(&random_state, pixels * sizeof(curandState));
    Random_Init << <grid,block,0 >> > (random_state,image_height);
}


extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
    unsigned int* g_odata, int imgh) {
    CalculatePerPixel << <grid, block, sbytes >> > (world,cam,random_state,10,g_odata, imgh);
}
