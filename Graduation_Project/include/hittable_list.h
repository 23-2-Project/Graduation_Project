#ifndef HITTABLELISTH
#define HITTABLELISTH

#include <hittable.h>
#include <box.h>

class hittable_list : public hittable {
public:
	hittable** list;
	int max_size;
	int now_size;
	aabb bbox;

	__device__ hittable_list(int n) {
		max_size = n;
		list = new hittable* [n];
		//list = (hittable**)malloc(max_size * sizeof(hittable*));
		now_size = 0;
	}

	__device__ hittable_list(hittable* object,int n) : hittable_list(n) {
		add(object);
	}

	__device__ void add(hittable* object) {
		//objects.push_back(object);
		list[now_size++] = object;
	}

	__device__ void add(hittable* object, int idx) {
		list[idx] = object;
	}

	__device__ void add_box(const vec3& v1, const vec3& v2, material* m) {
		auto min_point = vec3(fmin(v1.x(), v2.x()), fmin(v1.y(), v2.y()), fmin(v1.z(), v2.z()));
		auto max_point = vec3(fmax(v1.x(), v2.x()), fmax(v1.y(), v2.y()), fmax(v1.z(), v2.z()));

		auto dx = vec3(max_point.x() - min_point.x(), 0, 0);
		auto dy = vec3(0, max_point.y() - min_point.y(), 0);
		auto dz = vec3(0, 0, max_point.z() - min_point.z());

		add(new quad(vec3(min_point.x(), min_point.y(), max_point.z()), dx, dy, m));  // front 
		add(new quad(vec3(max_point.x(), min_point.y(), max_point.z()), -dz, dy, m)); // right
		add(new quad(vec3(max_point.x(), min_point.y(), min_point.z()), -dx, dy, m)); // back
		add(new quad(vec3(min_point.x(), min_point.y(), min_point.z()), dz, dy, m));  // left
		add(new quad(vec3(min_point.x(), max_point.y(), max_point.z()), dx, -dz, m)); // top
		add(new quad(vec3(min_point.x(), min_point.y(), min_point.z()), dx, dz, m));  // bottom
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

__global__ void add_object_count(hittable_list** world, int cnt) {
	(*world)->now_size += cnt;
}

#endif