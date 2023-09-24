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
__device__ float clamp(double x, double a, double b) { return max(a, min(b, x)); }
__device__ int rgbToInt(double r, double g, double b) {
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}
__device__ int vectorgb(vec3 color) {
	return (int(clamp(color.z()*255,0.0f,255.0f))<<16)|(int(clamp(color.y()*255,0.0f,255.0f))<<8)|int(clamp(color.x() * 255, 0.0f, 255.0f));
}
__device__ vec3 ray_color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}
__global__ void CalculatePerPixel(hittable_list** world, camera** camera, curandState* global_rand_state, int spp, unsigned int* g_odata, int imgh, int imgw) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int i = blockIdx.x * bw + tx;
	int j = blockIdx.y * bh + ty;
	int index = i + j * imgh;







	auto aspect_ratio = 16.0 / 9.0;
	int image_width = 1600;

	// Calculate the image height, and ensure that it's at least 1.
	int image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;

	// Camera

	auto focal_length = 1.0;
	auto viewport_height = 2.0;
	auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
	auto camera_center = vec3(0, 0, 0);

	// Calculate the vectors across the horizontal and down the vertical viewport edges.
	auto viewport_u = vec3(viewport_width, 0, 0);
	auto viewport_v = vec3(0, -viewport_height, 0);

	// Calculate the horizontal and vertical delta vectors from pixel to pixel.
	auto pixel_delta_u = viewport_u / image_width;
	auto pixel_delta_v = viewport_v / image_height;

	// Calculate the location of the upper left pixel.
	auto viewport_upper_left = camera_center
		- vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
	auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
	auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
	auto ray_direction = pixel_center - camera_center;

	ray r(camera_center, ray_direction);
	vec3 pc = ray_color(r);


	//printf("%f %f %f\n", pc.x(), pc.y(), pc.z());
	curandState local_rand_state = global_rand_state[index];
	vec3 color(0, 0, 0);
	color /= float(spp);
	//g_odata[index] = rgbToInt((float)x / 800 * 255, (float)y / 800 * 255, 0);
	global_rand_state[index] = local_rand_state;
	g_odata[i + j * imgw] = vectorgb(pc);
}
__global__ void initCamera(camera** ca) {

	*ca = new camera(16.0 / 9.0, 1600, 10, 50, 90, vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0));

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
	initCamera << <1, 1 >> > (cam);
	//*ca = new camera(ar, iw, spp, md, vfov, lf, la, vu);
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