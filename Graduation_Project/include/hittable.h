#ifndef HITTABLEH
#define HITTABLEH

#include <aabb.h>
#include <ray.h>

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat;
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual aabb bounding_box() const = 0;
    __device__ virtual bool isLeaf() const = 0;
};

#endif