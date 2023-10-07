#ifndef OBJH
#define OBJH

#include <hittable.h>

__device__ void LoadOBJ(hittable **list,const int v_counts, const int f_counts, const vec3* vertices, double scale, vec3 translate,material* mat) {
	//int i;
}
#endif
//}
//class obj : public hittable {
//public:
//    __device__ obj() {}
//    __device__ obj(const vec3& _a, const vec3& _b, const vec3& _c, material* m) : mat_ptr(m) {
//        u = _b - _a;
//        v = _c - _a;
//        Q = _a;
//        normal = unit_vector(cross(u, v));
//        D = dot(normal, Q);
//    };
//    __device__ void LoadOBJ(const int v_counts,const int f_counts,const vec3* vertices,double scale, vec3 translate) {
//        for (int i = 0; i < v_counts; i++) {
//
//        }
//    }
//    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
//    vec3 u, v;
//    vec3 normal;
//    double D;
//
//    vec3 Q;
//    vec3 center;
//    float radius;
//    material* mat_ptr;
//};
//
//__device__ bool obj::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//    auto denom = dot(normal, r.direction());
//
//    if (denom < 1e-8 && denom>-(1e-8)) {
//        return false;//면과 평행하면 hit안함
//    }
//
//    auto t = (D - dot(normal, r.origin())) / denom;
//    if (t_min > t || t_max < t) {
//        return false;//범위내에 없다
//    }
//
//    auto intersection = r.at(t);
//
//    auto edge1 = u;
//    auto edge2 = v;
//
//    auto h = cross(r.direction(), edge2);
//    auto a = dot(edge1, h);
//    if (a > -1e-8 && a < 1e-8) {
//        return false;
//    }
//    auto f = 1.0 / a;
//    auto s = r.origin() - Q;
//    auto ue = f * dot(s, h);
//    if (ue < 0.0 || ue>1.0) {
//        return false;
//    }
//    auto q = cross(s, edge1);
//    auto ve = f * dot(r.direction(), q);
//    if (ve < 0.0 || ue + ve>1.0) {
//        return false;
//    }
//    rec.t = t;
//    rec.p = r.at(t);
//    rec.mat = mat_ptr;
//    return true;
//
//
//
//}
//
//
//#endif