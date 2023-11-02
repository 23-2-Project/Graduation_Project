#ifndef HITTABLELISTH
#define HITTABLELISTH

#include <hittable.h>

class hittable_list : public hittable {
public:
	hittable** list;
	int max_size;
	int now_size;
	aabb bbox;

	__device__ hittable_list(int n) {
		max_size = n;
		list = (hittable**)malloc(max_size * sizeof(hittable*));
		now_size = 0;
	}

	__device__ hittable_list(hittable* object,int n) : hittable_list(n) {
		add(object);
	}

	__device__ void add(hittable* object) {
		//objects.push_back(object);
		list[now_size++] = object;
		bbox = aabb(bbox, object->bounding_box());
		

	}

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
		hit_record temp_rec;
		bool hit_anything = false;
		float closest_so_far = ray_t.maxv;
		for (int i = 0; i < now_size; i++) {
			if (list[i]->hit(r, interval(ray_t.minv, closest_so_far), temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

	__device__ aabb bounding_box() const override { return bbox; }
};

__global__ void get_object_count(hittable_list** world, int* count) {
	*count = (*world)->now_size;
}

#endif