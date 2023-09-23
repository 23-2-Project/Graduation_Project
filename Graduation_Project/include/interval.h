#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
  public:
    double minv, maxv;

    __device__ interval() : minv(+infinity), maxv(-infinity) {} // Default interval is empty

    __device__ interval(double _min, double _max) : minv(_min), maxv(_max) {}

    __device__ bool contains(double x) const {
        return minv <= x && x <= maxv;
    }

    __device__ bool surrounds(double x) const {
        return minv < x && x < maxv;
    }

    __device__ double clamp(double x) const {
        if (x < minv) return minv;
        if (x > maxv) return maxv;
        return x;
    }
    static const interval empty, universe;
};

const static interval empty   (+infinity, -infinity);
const static interval universe(-infinity, +infinity);


#endif