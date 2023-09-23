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

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}
__device__ inline double random_double(curandState* state) {
    //static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    //static std::mt19937 generator;
    return curand_uniform(state);
}
__device__ inline double random_double(curandState* state,double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}
// Common Headers

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif