#ifndef TRIANGLEH
#define TRIANGLEH

#include <hittable.h>

class triangle : public hittable {
public:
	__device__ triangle() {}
	__device__ triangle(const vec3& _a, const vec3& _b, const vec3& _c, material* m) : mat_ptr(m) {
		u = _b - _a;
		v = _c - _a;
		Q = _a;
		normal = unit_vector(cross(u, v));
		D = dot(normal, Q);

		// aabb box 생성
		vec3 min_point, max_point;
		min_point = vec3(min(min(_a.x(), _b.x()), _c.x()),
			min(min(_a.y(), _b.y()), _c.y()),
			min(min(_a.z(), _b.z()), _c.z()));
		max_point = vec3(max(max(_a.x(), _b.x()), _c.x()),
			max(max(_a.y(), _b.y()), _c.y()),
			max(max(_a.z(), _b.z()), _c.z()));

		bbox = aabb(min_point, max_point);
	};

	__device__ bool hit(const ray& r, float maxt, hit_record& rec) const {


		auto denom = dot(normal, r.direction());

		if (denom < 1e-8f && denom>-(1e-8f)) {
			return false;//면과 평행하면 hit안함
		}

		auto t = (D - dot(normal, r.origin())) / denom;
		if (t > maxt || t < 0.001f) {
			return false;//범위내에 없다
		}
		auto h = cross(r.direction(), v);
		auto a = dot(u, h);
		if (a > -(1e-8f) && a < 1e-8f) {
			return false;
		}
		float f = 1.0 / a;
		auto s = r.origin() - Q;
		auto ue = f * dot(s, h);
		if (ue < 0.0f || ue>1.0f) {
			return false;
		}
		auto q = cross(s, u);
		auto ve = f * dot(r.direction(), q);
		if (ve < 0.0f || ue + ve>1.0f) {
			return false;
		}
		rec.t = t;
		rec.p = r.at(t);
		rec.set_face_normal(r, normal);
		rec.mat = mat_ptr;
		return true;
	}

	__device__ aabb bounding_box() const override { return bbox; }

	__device__ bool isLeaf() const override { return true; }

private:
	vec3 u, v;
	vec3 normal;
	float D;
	vec3 Q;
	vec3 center;
	float radius;
	material* mat_ptr;
	aabb bbox;
};

#endif