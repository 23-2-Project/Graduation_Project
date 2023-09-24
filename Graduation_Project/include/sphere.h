#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
  public:
    // Stationary Sphere
    __device__ sphere(vec3 _center, double _radius, material* _material)
        : center1(_center), radius(_radius), mat(_material), is_moving(false) {
        id = 1;
    }

    // Moving Sphere
    __device__ sphere(vec3 _center1, vec3 _center2, double _radius, material* _material)
        : center1(_center1), radius(_radius), mat(_material), is_moving(true)
    {
        id = 1;
        center_vec = _center2 - _center1;
    }
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const{
        printf("��Ʈ Ȯ��\n");
        //vec3 center = is_moving ? sphere_center(r.time()) : center1;
        //vec3 oc = r.origin() - center;
        //auto a = r.direction().length_squared();
        //auto half_b = dot(oc, r.direction());
        //auto c = oc.length_squared() - radius*radius;

        //auto discriminant = half_b*half_b - a*c;
        //if (discriminant < 0) return false;
        //auto sqrtd = sqrt(discriminant);

        //// Find the nearest root that lies in the acceptable range.
        //auto root = (-half_b - sqrtd) / a;
        //if (!ray_t.surrounds(root)) {
        //    root = (-half_b + sqrtd) / a;
        //    if (!ray_t.surrounds(root))
        //        return false;
        //}

        //rec.t = root;
        //rec.p = r.at(rec.t);
        //vec3 outward_normal = (rec.p - center) / radius;
        //rec.set_face_normal(r, outward_normal);
        //rec.mat = mat;

        return true;
    }

  private:
      vec3 center1;
    double radius;
    material* mat;
    bool is_moving;
    vec3 center_vec;

    __device__ vec3 sphere_center(double time) const {
        // Linearly interpolate from center1 to center2 according to time, where t=0 yields
        // center1, and t=1 yields center2.
        return center1 + time*center_vec;
    }
};

#endif