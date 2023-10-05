#ifndef BVH_H
#define BVH_H

#include <rtweekend.h>

#include <hittable.h>
#include <hittable_list.h>

#include <algorithm>

class bvh_node : public hittable {
public:
	__device__ bvh_node(hittable_list** h_list, curandState* state) : bvh_node((*h_list)->list, 0, (*h_list)->now_size, state) {}

	__device__ bvh_node(hittable** src_objects, size_t start, size_t end, curandState* state) {
		int axis = random_int(state, 0, 2);
		auto comparator = (axis == 0) ? box_x_compare
			: (axis == 1) ? box_y_compare
			: box_z_compare;

		size_t object_span = end - start;

		if (object_span == 1) {
			left = right = src_objects[start];
		}
		else if (object_span == 2) {
			if (comparator(src_objects[start], src_objects[start + 1])) {
				left = src_objects[start];
				right = src_objects[start + 1];
			}
			else {
				left = src_objects[start + 1];
				right = src_objects[start];
			}
		}
		else {
			hittable* tmp;
			for (int i = start; i < end - 1; ++i) {
				for (int j = i; j < end - 1; ++j) {
					if (!comparator(src_objects[j], src_objects[j + 1])) {
						tmp = src_objects[j];
						src_objects[j] = src_objects[j + 1];
						src_objects[j + 1] = tmp;
					}
				}
			}

			auto mid = start + object_span / 2;
			left = new bvh_node(src_objects, start, mid, state);
			right = new bvh_node(src_objects, mid, end, state);
		}

		bbox = aabb(left->bounding_box(), right->bounding_box());
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		interval iv = interval(t_min, t_max);
		if (!bbox.hit(r, iv))
			return false;

		bool hit_left = left->hit(r, t_min, t_max, rec);
		bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

		return hit_left || hit_right;
	}

	__device__ aabb bounding_box() const override { return bbox; }

private:
	hittable* left;
	hittable* right;
	aabb bbox;

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

#endif