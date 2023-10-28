#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <random>
#include <limits>
#include <memory>
#include <curand_kernel.h>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}
__device__ inline float random_float(curandState* state) {
    //static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    //static std::mt19937 generator;
    return curand_uniform(state);
}
__device__ inline float random_float(curandState* state, float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float(state);
}

__device__ inline int random_int(curandState* state, int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(state, min, max + 1));
}

// Common Headers

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif