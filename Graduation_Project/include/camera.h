#ifndef CAMERA_H
#define CAMERA_H
#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include <iostream>

class camera {
  public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 100;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene

    double vfov = 90;  // Vertical view angle (field of view)
    vec3 lookfrom = vec3(0,0,-1);  // Point camera is looking from
    vec3 lookat   = vec3(0,0,0);   // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction
    int    image_height;   // Rendered image height
    vec3 center;         // Camera center
    vec3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;        // Camera frame basis vectors
    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius
    double defocus_angle = 0;  // Variation angle of rays through each pixel
    double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus
    __device__ camera(float ar,int iw,int spp,int md,int vf,vec3 lf,vec3 la,vec3 vu) {
        aspect_ratio = ar;
        image_width = iw;
        samples_per_pixel = spp;
        max_depth = md;
        vfov = vf;
        lookfrom = lf;
        lookat = la;
        vup = vu;
        //cam->aspect_ratio = 16.0 / 9.0;
//cam->image_width = 400;
//cam->samples_per_pixel = 10;
//cam->max_depth = 50;
//cam->vfov = 90;
// cam->lookfrom = vec3(0, 0, 0);
 //cam->lookat = vec3(0, 0, -1);
// cam->vup = vec3(0, 1, 0);
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }
    __device__ void change_origin(vec3 camera_location) {
        lookfrom = camera_location;
    }
    __device__ ray get_ray(int i,int j,curandState *state) {
        vec3 pixel_location = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        pixel_location += (curand_uniform(state) - 0.5) * pixel_delta_u + (curand_uniform(state) - 0.5) * pixel_delta_v;
        vec3 ray_origin = (defocus_angle <= 0) ? center : center;//렌즈 넣고싶으면 우항 바꿔야함
        vec3 ray_direction = pixel_location - ray_origin;
        return ray(ray_origin, ray_direction, curand_uniform(state));
    }
    __device__ vec3 ray_color(const ray& r, int depth, hittable_list** world,curandState *state) {
        ray cur_ray = r;
        vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
        for (int i = 0; i < depth; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, interval(0.001f,infinity), rec)) {
                ray scattered;
                vec3 attenuation;
                if (rec.mat->scatter(cur_ray, rec, attenuation, scattered, state)) {
                    cur_attenuation *= attenuation;
                    cur_ray = scattered;
                }
                else {
                    return vec3(0, 0, 0);
                }
            }
            else {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f * (unit_direction.y() + 1.0f);
                vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
            }
        }
        return vec3(0, 0, 0);
    }
};

#endif