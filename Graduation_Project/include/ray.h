#ifndef RAY_H
#define RAY_H

#include <vec3.h>

class ray {
public:
    __device__ ray() {}

    __device__ ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction), time(0.0) {}
    __device__ ray(const vec3& origin, const vec3& direction,const float t) : orig(origin), dir(direction), time(t) {}

    __device__ vec3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }
    __device__ float t() const { return time; }

    __device__ vec3 at(float t) const {
        return orig + t * dir;
    }

private:
    vec3 orig;
    vec3 dir;
    float time;
};

#endif