#ifndef AABB_H
#define AABB_H

#include <rtweekend.h>
#include <interval.h>

class aabb {
public:
	interval x, y, z;

	__device__ aabb() {} // The default AABB is empty, since intervals are empty by default.

	__device__ aabb(const interval& ix, const interval& iy, const interval& iz)
		: x(ix), y(iy), z(iz) { }

	__device__ aabb(const vec3& a, const vec3& b) {
		x = interval(fmin(a.x(), b.x()), fmax(a.x(), b.x()));
		y = interval(fmin(a.y(), b.y()), fmax(a.y(), b.y()));
		z = interval(fmin(a.z(), b.z()), fmax(a.z(), b.z()));
	}

	__device__ aabb(const aabb& box0, const aabb& box1) {
		x = interval(box0.x, box1.x);
		y = interval(box0.y, box1.y);
		z = interval(box0.z, box1.z);
	}

	__device__ aabb pad() {
		// Return an AABB that has no side narrower than some delta, padding if necessary.
		float delta = 0.0001;
		interval new_x = (x.size() >= delta) ? x : x.expand(delta);
		interval new_y = (y.size() >= delta) ? y : y.expand(delta);
		interval new_z = (z.size() >= delta) ? z : z.expand(delta);

		return aabb(new_x, new_y, new_z);
	}

	__device__ const interval& axis(int n) const {
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__device__ bool hit(const ray& r, float& maxt) const {
		auto left = 0.001f;
		auto right = maxt;

		for (int a = 0; a < 3; a++) {
			auto invD = 1 / r.direction()[a];
			auto orig = r.origin()[a];
			


			auto t0 = (axis(a).minv - orig) * invD;
			auto t1 = (axis(a).maxv - orig) * invD;

			if (invD < 0) {
				//std::swap(t0, t1);
				auto tmp = t0;
				t0 = t1;
				t1 = tmp;
			}

			if (t0 > left) left = t0;
			if (t1 < right) right = t1;

			if (right <= left)
				return false;
		}
		//maxt = FLT_MAX;
		return true;
	}
};

__device__ aabb operator+(const aabb& bbox, const vec3& offset) {
	return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__device__ aabb operator+(const vec3& offset, const aabb& bbox) {
	return bbox + offset;
}

#endif