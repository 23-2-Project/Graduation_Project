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
	hittable** tmp; // 머지소트용 배열
	aabb bbox;

	__device__ bvh_node(): left(nullptr), right(nullptr), tmp(nullptr) {};

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
			// Iterative Merge Sort
			tmp = (hittable**)malloc(object_num * sizeof(hittable*));

			for (int len = 1; len < object_num; len <<= 1) {
				int l = 0;
				for (int cnt = (len / (object_num << 1)); cnt > 0; cnt--) {
					int r = l + len;
					int end = r + len;
					merge(objects, l, r, end, object_num);
					l = end;
				}

				if ((object_num & ((len << 1) - 1)) > len) {
					merge(objects, l, l + len, object_num - 1, object_num);
				}
			}

			free(tmp);

			// bvh 트리 생성
			for (int i = 0; i < object_num; ++i) {
				int parent = (i + startIdx) / 2;
				if (~i & 1) { bvh_list[parent]->left = objects[i]; }
				else { 
					bvh_list[parent]->right = objects[i]; 
					bvh_list[parent]->bbox = aabb(bvh_list[parent]->left->bounding_box(), bvh_list[parent]->right->bounding_box());
				}
			}

			for (int i = object_num + startIdx; i < startIdx * 2; ++i) {
				if (~i & 1) { bvh_list[i / 2]->left = bvh_list[i]; }
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

	__device__ bool hit(const ray& r, float maxt, hit_record& rec) const override {
		float tmax = maxt;
		if (!bbox.hit(r, maxt)) {
			return false;
		}

		bool isHit = false;
		hittable* stk[32];
		int idx = 0;
		stk[idx++] = right;
		stk[idx++] = left;
		hit_record temp_rec;
		while (idx > 0) {
			hittable* now = stk[--idx];
			if (now->isLeaf()) {
				if (now->hit(r, tmax, temp_rec)) {
					tmax = temp_rec.t;
					rec = temp_rec;
					isHit = true;
				}
			}
			else {
				if (now->bounding_box().hit(r, tmax)) {
					stk[idx++] = ((bvh_node*)now)->right;
					stk[idx++] = ((bvh_node*)now)->left;
				}
			}
		}
		return isHit;
	}

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

	__device__ void merge(hittable** arr, int left, int mid, int right, int cnt) {
		int l = left, r = mid + 1, idx = 0;
		while (l <= mid && r <= right) {
			if (box_compare(arr[l], arr[r], 0)) {
				tmp[idx++] = arr[l++];
			}
			else {
				tmp[idx++] = arr[r++];
			}
		}
		while(l <= mid) { tmp[idx++] = arr[l++]; }
		while(r <= right) { tmp[idx++] = arr[r++]; }
		for (int i = left; i <= right; ++i) {
			arr[i] = tmp[i - left];
		}
	}
};

#endif