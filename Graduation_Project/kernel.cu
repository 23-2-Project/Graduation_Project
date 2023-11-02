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
bvh_node** bvh_tree;
camera** cam;
int object_counts = 130000;
curandState* random_state;
int** test;
// convert floating point rgb color to 8-bit integer
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }
__device__ int rgbToInt(float r, float g, float b) {
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
extern "C" void constructBVH() {
	// object 개수 구하기
	int object_count;
	int* kernel_value;
	cudaMalloc((void**)&kernel_value, sizeof(int));
	get_object_count << <1, 1 >> > (world, kernel_value);
	cudaMemcpy(&object_count, kernel_value, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(kernel_value);
	printf("%d개\n", object_count);
	
	//정렬
	/*srand(time(NULL));
	int axis = rand() % 3;
	dim3 sortBlock(512, 1, 1);
	dim3 sortGrid(object_count / sortBlock.x + 1, 1, 1);
	for (int i = 0; i < object_count; ++i) {
		int odd_even = i % 2;
		object_swap << <sortGrid, sortBlock>> > (world, object_count, odd_even, axis);
		cudaDeviceSynchronize();
	}
	printf("정렬 완료\n");*/

	//bvh 생성
	curandState* bvh_state;
	cudaMalloc(&bvh_state, sizeof(curandState));
	cudaMalloc((void**)&bvh_tree, sizeof(bvh_node*));
	make_bvh_tree << <1, 1 >> > (bvh_state, world, bvh_tree, object_count);
	cudaDeviceSynchronize();
	printf("bvh 생성 완료\n");
}

__global__ void CalculatePerPixel(bvh_node** bvh_tree, camera** camera, curandState* global_rand_state, unsigned int* g_odata, int imgh, int imgw) {
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
	float rate = 1 / float(spp);
	ray r = (*camera)->get_ray(&local_rand_state, i, j);
	for (int i = 0; i < spp; i++) {
		color += (*camera)->ray_color(&local_rand_state, r, depth, bvh_tree);
	}
	color *= rate;
	global_rand_state[index] = local_rand_state;
	g_odata[i + j * imgw] = vectorgb(color);
}
__global__ void initCamera(camera** ca) {
	*ca = new camera(16.0 / 9.0, //종횡비
		1600,                    //이미지 가로길이
		1,                       //픽셀당 샘플수
		5,                      //반사 횟수
		90,                      //시야각
		vec3(-20, 0, 0),         //카메라 위치 
		vec3(0, 0, -1),          //바라보는곳
		vec3(0, 1, 0),           //업벡터
		vec3(0.5f,0.7f,1));      //배경색
}
__global__ void initWorld(hittable_list** world, int object_counts) {
	(*world) = new hittable_list(object_counts);
}
#define RND (curand_uniform(&local_rand_state))
__global__ void addObjects(curandState* global_state, hittable_list** world, int object_counts) {
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	(*world)->add(new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5))));

	//(*world)->add(new triangle(vec3(50, 50, 50), vec3(-50, 50, 50), vec3(50, -50, 50), new metal(vec3(0.5, 0.7, 0.8),0)));

	(*world)->add(new sphere(vec3(0, 200, 0), 100, new light(vec3(1, 1, 1))));
	int sphere_count = 10;
	for (int a = -sphere_count; a < sphere_count; a++) {
		for (int b = -sphere_count; b < sphere_count; b++) {
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

__global__ void addTriangle(hittable_list** world,vec3 a,vec3 b,vec3 c,vec3 color) {
	(*world)->add(new triangle(a, b, c, new lambertian(color)));
}
__global__ void Random_Init(curandState* global_state, int ih) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x * bw + tx;
	int y = blockIdx.y * bh + ty;
	unsigned int pixel_index = x + y * ih;
	curand_init(pixel_index, 0, 0, &global_state[pixel_index]);
}

void ReadOBJ(const char* objlist[], int obj_counts,const vec3 translist[],const vec3 scalelist[]) {
	Assimp::Importer importer;
	for (int c = 0; c < obj_counts; c++) {
		char str[100] = "resource/";
		strcat(str, objlist[c]);
		printf("%s\n", str);
		const aiScene* scene = importer.ReadFile(str, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
		{
			printf("Read File Exception\n");
		}
		vec3 translate = translist[c];
		vec3 scale = scalelist[c];
		int cnt = 0;
		for (int i = 0; i < scene->mNumMeshes; i++) {
			auto mesh = scene->mMeshes[i];
			for (int j = 0; j < mesh->mNumFaces; j++) {
				auto Face = mesh->mFaces[j];
				vec3 a(mesh->mVertices[Face.mIndices[0]].x, mesh->mVertices[Face.mIndices[0]].y, mesh->mVertices[Face.mIndices[0]].z);
				vec3 b(mesh->mVertices[Face.mIndices[1]].x, mesh->mVertices[Face.mIndices[1]].y, mesh->mVertices[Face.mIndices[1]].z);
				vec3 c(mesh->mVertices[Face.mIndices[2]].x, mesh->mVertices[Face.mIndices[2]].y, mesh->mVertices[Face.mIndices[2]].z);
				a *= scale;				b *= scale;				c *= scale;
				a += translate;		b += translate;		c += translate;
				aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
				aiColor4D diffuse,specular,ambient;
				aiGetMaterialColor(material, AI_MATKEY_COLOR_AMBIENT, &ambient);
				aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &specular);
				aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse);
				aiColor4D sum = diffuse + specular + ambient;
				vec3 color(sum.r, sum.g, sum.b);
				//color = vec3(1.0f, 0.0f, 0.0f);
				addTriangle << <1, 1 >> > (world, a, b, c,color);
				cudaError_t err = cudaGetLastError();
				if (cudaSuccess != err) {
					printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
				}
				else {
					printf("%d\n", cnt++);
				}
			}
		}
	}
}

extern "C" void initCuda(dim3 grid, dim3 block, int image_height, int image_width, int pixels) {
	//cudaDeviceSetLimit(cudaLimitStackSize, 256 * 1024 * 1024);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
	cudaMalloc(&random_state, pixels * sizeof(curandState));
	Random_Init << <grid, block, 0 >> > (random_state, image_height);

	/*printf("%d\n", sizeof(int));
	cudaMalloc(&test, 2600000000*sizeof(int));
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA:ERROR:cuda failure \"%s\"\n", cudaGetErrorString(err));
		exit(1);
	}
	else {
		printf("CUDA Success\n");
	}*/

	//랜덤 초기화
	cudaMalloc((void**)&world, sizeof(hittable*));
	initWorld << <1, 1 >> > (world, object_counts); cudaDeviceSynchronize();

	//월드 초기화 OBJ 읽기 및 카메라 등
	const char* objlist[] = { "chair.obj"};      //읽을 OBJ 리스트, 및의 배열들과 순서 맞춰야함
	const vec3 translist[] = { 
										vec3(10.0f,10.0f,0.0f)};  //위에서 읽을 OBJ를 옮겨주는 벡터
	const vec3 scalelist[] = { 
										vec3(0.5f,0.5f,0.5f) };   //위에서 읽을 OBJ의 크기를 바꿔주는 벡터
	ReadOBJ(objlist, 1,translist,scalelist);
	
	//여기까지 OBJ 읽기
	curandState* objectinit;
	cudaMalloc(&objectinit, sizeof(curandState));
	addObjects<< <1, 1 >> > (objectinit, world, object_counts);
	cudaMalloc(&cam, sizeof(camera*));
	initCamera << <1, 1 >> > (cam);

	cudaDeviceSynchronize();        //쿠다커널이 종료될때까지 기다리는 함수. 위의 world에 오브젝트 다 담길때까지 기다림.
	                                //BVH 생성 중 오브젝트 담기는 것 방지용

	constructBVH();
}
extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
	unsigned int* g_odata, int imgh, int imgw) {
	CalculatePerPixel << <grid, block, sbytes >> > (bvh_tree, cam, random_state, g_odata, imgh, imgw);
}