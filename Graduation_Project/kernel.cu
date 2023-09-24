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
__global__ void CalculatePerPixel(hittable_list** world, camera** camera, curandState* global_rand_state, int spp, unsigned int* g_odata, int imgh, int imgw) {

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
		ray r = (*camera)->get_ray(x + curand_uniform(&local_rand_state), y + curand_uniform(&local_rand_state), &local_rand_state);
		//color += (*camera)->ray_color(r, (*camera)->max_depth, world, &local_rand_state);//이부분 문제
		vec3 test= (*camera)->ray_color(r, (*camera)->max_depth, world, &local_rand_state);
		//printf("%f %f %f\n", test.x(), test.y(), test.z());
	}
	color /= float(spp);
	//printf("%f %f %f\n", color.x(), color.y(), color.z());
	//g_odata[index] = rgbToInt(color.x(), color.y(), color.z());
	//g_odata[index] = rgbToInt((float)x / 800 * 255, (float)y / 800 * 255, 0);

	global_rand_state[index] = local_rand_state;
	g_odata[x + y * imgw] = rgbToInt((float)x / 1600 * 255, (float)y / 900 * 255, 0);
}
__global__ void initCamera(camera** ca, float ar, int iw, int spp, int md, int vfov, vec3 lf, vec3 la, vec3 vu) {

	*ca = new camera(ar, iw, spp, md, vfov, lf, la, vu);

}

__global__ void initWorld(hittable_list** world, hittable** objects) {
	lambertian* material_ground = new lambertian(vec3(0.8, 0.8, 0.0));
	lambertian* material_center = new lambertian(vec3(0.7, 0.3, 0.3));

	sphere* ground = new sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground);
	sphere* center = new sphere(vec3(0.0, 0.0, -1.0), 0.5, material_center);

	//objects[0] = new sphere(vec3(0.0, -100.5, -1.0), 100.0, material_ground);
	//objects[1] = new sphere(vec3(0.0, 0.0, -1.0), 0.5, material_center);
	//ground->hit(ray(),interval(),hit_record());
	(*world) = new hittable_list(objects,1);//오브젝트 개수 정의 아래 cudamalloc과 맞춰줄 필요 있음
	(*world)->add(ground);
	//(*world)->add(center);
}
extern "C" void initTracing() {

	cudaMalloc(&cam, sizeof(camera*));
	initCamera << <1, 1 >> > (cam, 16.0 / 9.0, 400, 10, 50, 90, vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0));
	cudaMalloc(&objects, 1 * sizeof(hittable*));//오브젝트 개수만큼 할당 필요
	cudaMalloc(&world, sizeof(hittable_list*));
	initWorld << <1, 1 >> > (world, objects);
}

__global__ void Random_Init(curandState* global_state, int ih) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	unsigned int pixel_index = x + y * ih;
	curandState s;
	curand_init(pixel_index, 0, 0, &global_state[pixel_index]);
}
extern "C" void initCuda(dim3 grid, dim3 block, int image_height, int image_width, int pixels) {
	cudaMalloc(&random_state, pixels * sizeof(curandState));
	Random_Init << <grid, block, 0 >> > (random_state, image_height);
}


extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
	unsigned int* g_odata, int imgh, int imgw) {
	CalculatePerPixel << <grid, block, sbytes >> > (world, cam, random_state, 10, g_odata, imgh, imgw);
}