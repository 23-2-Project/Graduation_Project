#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"

class hit_record;

class material {
  public:
    virtual ~material() = default;

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* state) const = 0;
};
class lambertian : public material {
  public:
    __device__ lambertian(const vec3& a) : albedo(a) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* state)
    const override {
        auto scatter_direction = rec.normal;//랜덤 추가 필요
        // Catch degenerate scatter direction
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

  private:
     vec3 albedo;
};

class metal : public material {
  public:
    metal(const vec3& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* state)
    const override {
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
      vec3 albedo;
    double fuzz;
};
class dielectric : public material {
  public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,curandState* state)
    const override {
        attenuation = vec3(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;


        scattered = ray(rec.p, direction);
        return true;
    }

  private:
    double ir; // Index of Refraction
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};
#endif