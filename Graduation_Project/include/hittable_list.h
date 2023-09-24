#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
  public:
    hittable** objects;

    __device__ hittable_list() {}

    __device__ hittable_list(hittable** list, int n) { objects = list; object_counts= n; }

    __device__ void clear() { delete* objects; }

    __device__ void add(hittable* object) {
        printf("받고나서 %d\n", object->id);
        object->hit(ray(), interval(), hit_record());
        objects[0] = object;
        current_count += 1;
        printf("넣고나서 %d\n", objects[0]);
        objects[0]->hit(ray(), interval(), hit_record());
        //objects.push_back(object);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.maxv;
        //for (const auto& object : objects) {
        for (int i = 0; i < current_count; i++) {
            //이부분이 문제
            //objects[0]->hit(ray(), interval(), hit_record());
            //printf("%d\n", objects[0]);
            //objects[i]->id;
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