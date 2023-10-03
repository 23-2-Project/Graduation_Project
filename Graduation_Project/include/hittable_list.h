#ifndef HITTABLELISTH
#define HITTABLELISTH

#include <hittable.h>

class hittable_list : public hittable {
public:
	hittable** list;
	int max_size;
	int now_size;

	__device__ hittable_list(int n) {
		max_size = n;
		list = (hittable**)malloc(max_size * sizeof(hittable*));
		now_size = 0;
	}

	__device__ void add(hittable* object) {
		//objects.push_back(object);
		list[now_size++] = object;
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
		hit_record temp_rec;
		bool hit_anything = false;
		float closest_so_far = t_max;
		for (int i = 0; i < now_size; i++) {
			if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}
};

#endif