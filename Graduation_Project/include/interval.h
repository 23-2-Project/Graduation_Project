#ifndef INTERVAL_H
#define INTERVAL_H

constexpr float inf = FLT_MAX;

class interval {
public:
	float minv, maxv;

	__device__ interval() : minv(+inf), maxv(-inf) {} // Default interval is empty

	__device__ interval(float _min, float _max) : minv(_min), maxv(_max) {}

	__device__ interval(const interval& a, const interval& b) : minv(fmin(a.minv, b.minv)), maxv(fmax(a.maxv, b.maxv)) {}

	__device__ float size() const {
		return maxv - minv;
	}

	__device__ interval expand(float delta) const {
		auto padding = delta / 2;
		return interval(minv - padding, maxv + padding);
	}

	__device__ inline bool contains(float x) const {
		return minv <= x && x <= maxv;
	}

	__device__ inline bool surrounds(float x) const {
		return minv < x && x < maxv;
	}

	__device__ float clamp(float x) const {
		if (x < minv) return minv;
		if (x > maxv) return maxv;
		return x;
	}
	static const interval empty, universe;
};

const static interval empty(+inf, -inf);
const static interval universe(-inf, +inf);

__device__ interval operator+(const interval& ival, float displacement) {
	return interval(ival.minv + displacement, ival.maxv + displacement);
}

__device__ interval operator+(float displacement, const interval& ival) {
	return ival + displacement;
}

#endif