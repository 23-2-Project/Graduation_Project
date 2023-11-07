#ifndef SPHEREH
#define SPHEREH

#include <hittable.h>

class sphere : public hittable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {
		vec3 rvec = vec3(radius, radius, radius);
		bbox = aabb(cen - rvec, cen + rvec);
	};

	__device__ bool hit(const ray& r, float maxt, hit_record& rec) const {
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - a * c;
		if (discriminant > 0) {
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < maxt && temp > 0.001f) {
				rec.t = temp;
				rec.p = r.at(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat = mat_ptr;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < maxt && temp > 0.001f) {
				rec.t = temp;
				rec.p = r.at(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat = mat_ptr;
				return true;
			}
		}
		return false;
	}
	__device__ aabb bounding_box() const override { return bbox; }

	__device__ bool isLeaf() const override { return true; }

private:
	vec3 center;
	float radius;
	material* mat_ptr;
	aabb bbox;
};

#endif