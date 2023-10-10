#ifndef BVH_H
#define BVH_H

#include <rtweekend.h>

#include <hittable.h>
#include <hittable_list.h>

#include <algorithm>

class bvh_node : public hittable {
public:
	hittable* left;
	hittable* right;
	aabb bbox;

	__device__ bvh_node() {};

	__device__ bvh_node(hittable_list** world, bvh_node** bvh_list, curandState* state ) {
		hittable** objects = (*world)->list;
		int object_num = (*world)->now_size;

		int startIdx = 1 << 30;
		while (true) {
			if ((startIdx >> 1) > object_num) { startIdx >>= 1; }
			else { break; }
		}

		bvh_list = (bvh_node**)malloc((startIdx << 1) * sizeof(bvh_node*));
		for (int i = 0; i < (startIdx << 1); ++i) {
			bvh_list[i] = new bvh_node();
		}

		int axis = random_int(state, 0, 2);
		auto comparator = (axis == 0) ? box_x_compare
			: (axis == 1) ? box_y_compare
			: box_z_compare;

		if (object_num == 1) {
			left = right = objects[0];
		}
		else if (object_num == 2) {
			if (comparator(objects[0], objects[1])) {
				left = objects[0];
				right = objects[1];
			}
			else {
				left = objects[1];
				right = objects[0];
			}
		}
		else {
			// 정렬하는 부분
			hittable* tmp;
			for (int i = 0; i < object_num - 1; ++i) {
				for (int j = i; j < object_num - 1; ++j) {
					if (!comparator(objects[j], objects[j + 1])) {
						tmp = objects[j];
						objects[j] = objects[j + 1];
						objects[j + 1] = tmp;
					}
				}
			}

			// bvh 트리 생성
			for (int i = 0; i < object_num; ++i) {
				int parent = (i + startIdx) / 2;
				if (i % 2 == 0) { bvh_list[parent]->left = objects[i]; }
				else { 
					bvh_list[parent]->right = objects[i]; 
					bvh_list[parent]->bbox = aabb(bvh_list[parent]->left->bounding_box(), bvh_list[parent]->right->bounding_box());
				}
			}

			for (int i = object_num + startIdx; i < startIdx * 2; ++i) {
				if (i % 2 == 0) { bvh_list[i / 2]->left = bvh_list[i]; }
				else {
					bvh_list[i / 2]->right = bvh_list[i];
					if (i == object_num + startIdx) {
						bvh_list[i / 2]->bbox = bvh_list[i / 2]->left->bounding_box();
					}
					else {
						bvh_list[i / 2]->bbox = aabb();
					}				
				}
			}

			for (int i = startIdx / 2 - 1; i >= 2; --i) {
				bvh_list[i]->left = bvh_list[i * 2];
				bvh_list[i]->right = bvh_list[i * 2 + 1];
				bvh_list[i]->bbox = aabb(bvh_list[i]->left->bounding_box(), bvh_list[i]->right->bounding_box());
			}

			left = bvh_list[2];
			right = bvh_list[3];
		}

		bbox = aabb(left->bounding_box(), right->bounding_box());
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		float tmax = t_max;
		interval iv = interval(t_min, tmax);
		if (!bbox.hit(r, iv)) {
			return false;
		}

		bool isHit = false;
		hittable* stk[30];
		int idx = 0;
		stk[idx++] = right;
		stk[idx++] = left;

		while (idx > 0) {
			hittable* now = stk[--idx];
			if (now->isLeaf()) {
				if (now->hit(r, t_min, tmax, rec)) {
					tmax = rec.t;
					iv = interval(t_min, tmax);
					isHit = true;
				}
			}
			else {
				if (now->bounding_box().hit(r, iv)) {
					stk[idx++] = ((bvh_node*)now)->right;
					stk[idx++] = ((bvh_node*)now)->left;
				}
			}
		}

		return isHit;
	}

	/*__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		interval iv = interval(t_min, t_max);
		if (!bbox.hit(r, iv))
			return false;

		bool hit_left = left->hit(r, t_min, t_max, rec);
		bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

		return hit_left || hit_right;
	}*/

	__device__ aabb bounding_box() const override { return bbox; }

	__device__ bool isLeaf() const override { return false; }

private:
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