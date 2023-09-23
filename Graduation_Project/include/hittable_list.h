#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
  public:
    hittable** objects;

    __device__ hittable_list() {}

    __device__ hittable_list(hittable* object) { add(object); }
    __device__ hittable_list(hittable** list) { objects=list; }
    __device__ hittable_list(int count) { 
        object_counts=count; 
        //(*objects) = new hittable*
    }

    __device__ void clear() { delete* objects; }

    __device__ void add(hittable* object) {
        objects[current_count++] = object;
        //objects.push_back(object);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.maxv;
        //for (const auto& object : objects) {
        for (int i = 0; i < current_count; i++) {
            if (objects[i]->hit(r, interval(ray_t.minv, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
    int object_counts = 10;
    int current_count=0;
};

#endif