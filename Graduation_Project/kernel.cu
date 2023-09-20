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
#include <math_functions.h>
#include <hittable_list.h>
#include <ray.h>
#include <color.h>
#include <sphere.h>
#include <hittable_list.h>
#include <material.h>
#include <vec3.h>
#include <camera.h>
hittable_list world;
camera cam;
// convert floating point rgb color to 8-bit integer
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}
__global__ void CalculatePerPixel(unsigned int* g_odata, int imgh) {
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;
    //ray r = cam.get_ray(x, y);
    
    
    
    g_odata[x  + y * imgh] = rgbToInt((float)x / 800 * 255, (float)y / 800 * 255, 0);
}
extern "C" void initWorld() {
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 10;
    cam.max_depth = 50;
    cam.vfov = 90;
    cam.lookfrom = vec3(0, 0, 0);
    cam.lookat = vec3(0, 0, -1);
    cam.vup = vec3(0, 1, 0);
    cam.initialize();




    auto ground_material = new lambertian(vec3(0.5, 0.5, 0.5));
    auto ground = new sphere(vec3(0, -1000, 0), 1000, ground_material);
    world.add(ground);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            vec3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = vec3::random() * vec3::random();
                    sphere_material = new lambertian(albedo);
                    auto center2 = center + vec3(0, random_double(0, .5), 0);
                    auto sp = new sphere(center, center2, 0.2, sphere_material);
                    world.add(sp);
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = vec3::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    auto sp = new sphere(center, 0.2, sphere_material);
                    world.add(sp);
                }
                else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    auto sp = new sphere(center, 0.2, sphere_material);
                    world.add(sp);
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    auto sp1 = new sphere(vec3(0, 1, 0), 1.0, material1);
    world.add(sp1);

    auto material2 = new lambertian(vec3(0.4, 0.2, 0.1));
    auto sp2 = new sphere(vec3(-4, 1, 0), 1.0, material2);
    world.add(sp2);

    auto material3 = new metal(vec3(0.7, 0.6, 0.5), 0.0);
    auto sp3 = new sphere(vec3(4, 1, 0), 1.0, material3);
    world.add(sp3);
}
extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
    unsigned int* g_odata, int imgh) {
    CalculatePerPixel << <grid, block, sbytes >> > (g_odata, imgh);
}
