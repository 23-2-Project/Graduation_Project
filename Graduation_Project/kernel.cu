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
#include <bvh.h>
#include <triangle.h>
#include <obj.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

hittable_list** world;
//hittable** objects;
camera** cam;
int object_counts = 29000;

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
	return rgbToInt(color.x()*255,color.y()*255,color.z()*255);
}
__global__ void movCam(camera** ca, int direction, int weight) {
	(*ca)->moveorigin(direction, weight);
}
__global__ void RotateCam(camera** ca, vec3 direction) {
	(*ca)->rotate(direction);
}
__global__ void ManipulateVFOV(camera** ca, int x) {
	(*ca)->changevfov(x);
}
extern "C" void moveCamera(int direction, int weight) {
	movCam << <1, 1 >> > (cam, direction, weight);
}
extern "C" void RotateCamera(int x, int y) {
	RotateCam << <1, 1 >> > (cam, vec3(x, y, 0));
}
extern "C" void manivfov(int x) {
	ManipulateVFOV << <1, 1 >> > (cam, x);
}

__global__ void CalculatePerPixel(hittable_list** world, camera** camera, curandState* global_rand_state, unsigned int* g_odata, int imgh, int imgw) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int i = blockIdx.x * bw + tx;
	int j = blockIdx.y * bh + ty;
	int index = i + j * imgh;

	curandState local_rand_state = global_rand_state[index];
	vec3 color(0, 0, 0);

	int depth = (*camera)->max_depth;
	int spp = (*camera)->samples_per_pixel;
	ray r = (*camera)->get_ray(&local_rand_state, i, j);
	for (int i = 0; i < spp; i++) {
		color += (*camera)->ray_color(&local_rand_state, r, depth, world);
	}
	color /= float(spp);
	global_rand_state[index] = local_rand_state;
	g_odata[i + j * imgw] = vectorgb(color);
}
__global__ void initCamera(camera** ca) {

	*ca = new camera(16.0 / 9.0, //종횡비
		1600, //이미지 가로길이
		1,  //픽셀당 샘플수
		50,  //반사 횟수
		90,  //시야각
		vec3(-50, 0, 0), //카메라 위치 
		vec3(0, 0, -1), //바라보는곳
		vec3(0, 1, 0)); //업벡터

}
__global__ void initWorld(hittable_list** world, int object_counts) {
	(*world) = new hittable_list(object_counts);
}
#define RND (curand_uniform(&local_rand_state))
__global__ void addObjects(curandState* global_state, hittable_list** world, int object_counts) {
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	(*world)->add(new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5))));
	for (int a = -2; a < 2; a++) {
		for (int b = -2; b < 2; b++) {
			float choose_mat = RND;
			vec3 center(a + RND, 0.2, b + RND);
			if (choose_mat < 0.8f) {
				(*world)->add(new sphere(center, 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND))));
			}
			else if (choose_mat < 0.95f) {
				(*world)->add(new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.0f/*0.5f * RND*/)));
			}
			else {
				(*world)->add(new sphere(center, 0.2, new dielectric(1.5)));
			}
		}
	}
}
__global__ void makeBVH(curandState* global_state, hittable_list** world, int object_counts) {
	printf("%d개\n", (*world)->now_size);
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	(*world) = new hittable_list((hittable*)new bvh_node(world, &local_rand_state), object_counts);
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

	//랜덤 초기화

	cudaMalloc((void**)&world, sizeof(hittable*));
	initWorld << <1, 1 >> > (world, object_counts); cudaDeviceSynchronize();
	//월드 초기화 OBJ 읽기 및 카메라 등
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile("FileName.obj", aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
	
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
	{
		printf("Read File Exception\n");
	}
	//int cnt = 0;
	//for (int i = 0; i < scene->mNumMeshes; i++) {
	//	auto mesh = scene->mMeshes[i];
	//	for (int j = 0; j < mesh->mNumFaces; j++) {
	//		auto Face = mesh->mFaces[j];
	//		/*newTriangle(mesh->mVertices[Face.mIndices[0]].x,
	//			mesh->mVertices[Face.mIndices[0]].y,
	//			mesh->mVertices[Face.mIndices[0]].z,

	//			mesh->mVertices[Face.mIndices[1]].x,
	//			mesh->mVertices[Face.mIndices[1]].y,
	//			mesh->mVertices[Face.mIndices[1]].z,

	//			mesh->mVertices[Face.mIndices[2]].x,
	//			mesh->mVertices[Face.mIndices[2]].y,
	//			mesh->mVertices[Face.mIndices[2]].z);*/
	//	}
	//}
	





	//여기까지 OBJ 읽기
	curandState* objectinit;
	cudaMalloc(&objectinit, sizeof(curandState));
	addObjects<< <1, 1 >> > (objectinit, world, object_counts);
	cudaMalloc(&cam, sizeof(camera*));
	initCamera << <1, 1 >> > (cam);

	cudaDeviceSynchronize();//쿠다커널이 종료될때까지 기다리는 함수. 위의 world에 오브젝트 다 담길때까지 기다리는 거임. 
	//그래야 BVH할수있으니까. BVH하다가 오브젝트 담기면 안됌.

	curandState* bvh_state;
	cudaMalloc(&bvh_state, sizeof(curandState));
	makeBVH << <1, 1 >> > (bvh_state, world, object_counts);
}
extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
	unsigned int* g_odata, int imgh, int imgw) {

	CalculatePerPixel << <grid, block, sbytes >> > (world, cam, random_state, g_odata, imgh, imgw);
}