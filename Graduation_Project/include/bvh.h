#ifndef BVH_H
#define BVH_H

#include <rtweekend.h>

#include <hittable.h>
#include <hittable_list.h>

#include <algorithm>

class bvh_node : public hittable {
public:
	hittable** objects;
	bool* isObject;
	aabb* bbox;
	int startIdx;

	__device__ bvh_node() {};

	__device__ bvh_node(hittable_list** world, curandState* state) {
		objects = (*world)->list;
		int object_num = (*world)->now_size;

		startIdx = 1 << 30;
		while (true) {
			if ((startIdx >> 1) > object_num) { startIdx >>= 1; }
			else { break; }
		}

		isObject = new bool[startIdx * 2];
		bbox = new aabb[startIdx * 2];

		// bvh 트리 생성
		for (int i = 0; i < object_num; ++i) {
			int idx = i + startIdx;
			isObject[idx] = true;
			bbox[idx] = objects[i]->bounding_box();

			if (i & 1) {
				isObject[idx / 2] = false;
				bbox[idx / 2] = aabb(bbox[idx - 1], bbox[idx]);
			}
		}

		for (int i = object_num + startIdx; i < startIdx * 2; ++i) {
			isObject[i] = false;
			bbox[i] = aabb();

			if (i & 1) {
				isObject[i / 2] = false;
				bbox[i / 2] = aabb(bbox[i - 1], bbox[i]);
			}
		}

		for (int i = startIdx / 2 - 1; i >= 1; --i) {
			isObject[i] = false;
			bbox[i] = aabb(bbox[i * 2], bbox[i * 2 + 1]);
		}
	}

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {}

	__device__ bool bvh_hit(const ray& r, interval ray_t, hit_record& rec) {
		float tmax = ray_t.maxv;
		if (!bbox[1].hit(r, ray_t)) {
			return false;
		}
		bool isHit = false;
		int stk[32];
		int idx = 0;
		stk[idx++] = 3;
		stk[idx++] = 2;
		hit_record temp_rec;
		while (idx > 0) {
			int now = stk[--idx];
			if (isObject[now]) {
				if (objects[now - startIdx]->hit(r, interval(ray_t.minv, tmax), temp_rec)) {
					tmax = temp_rec.t;
					rec = temp_rec;
					isHit = true;
				}
			}
			else {
				if (bbox[now].hit(r, ray_t)) {
					stk[idx++] = now * 2 + 1;
					stk[idx++] = now * 2;
				}
			}
		}
		return isHit;
	}

	__device__ aabb bounding_box() const override { return aabb(); }

	__device__ static bool box_compare(const hittable* a, const hittable* b, int axis_index) {
		return a->bounding_box().axis(axis_index).minv < b->bounding_box().axis(axis_index).minv;
	}

	__device__ static bool box_x_compare(const hittable* a, const hittable* b) {
		return box_compare(a, b, 0);
	}

	__device__ static bool box_y_compare(const hittable* a, const hittable* b) {
		return box_compare(a, b, 1);
	}

	__device__ static bool box_z_compare(const hittable* a, const hittable* b) {
		return box_compare(a, b, 2);
	}
};

__global__ void object_swap(hittable_list** world, int object_counts, int odd_even, int axis) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + odd_even;
	if (idx >= object_counts - 1) { return; }

	hittable** objects = (*world)->list;
	auto comparator = (axis == 0) ? bvh_node::box_x_compare
		: (axis == 1) ? bvh_node::box_y_compare
		: bvh_node::box_z_compare;

	if (!comparator(objects[idx], objects[idx + 1])) {
		hittable* tmp = objects[idx];
		objects[idx] = objects[idx + 1];
		objects[idx + 1] = tmp;
	}
	return;
}

__global__ void make_bvh_tree(curandState* global_state, hittable_list** world, bvh_node** bvh_tree, int object_count) {
	curand_init(0, 0, 0, &global_state[0]);
	curandState local_rand_state = *global_state;
	*bvh_tree = new bvh_node(world, &local_rand_state);
}

#endif