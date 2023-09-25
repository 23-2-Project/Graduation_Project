#ifndef CAMERA_H
#define CAMERA_H
#include <rtweekend.h>

#include <hittable.h>
#include <material.h>

class camera {
  public:

      int    max_depth = 10;
    __device__ camera(float ar,int iw,int spp,int md,int vf,vec3 lf,vec3 la,vec3 vu) {
        aspect_ratio = ar;
        image_width = iw;
        samples_per_pixel = spp;
        max_depth = md;
        vfov = vf;
        lookfrom = lf;
        lookat = la;
        vup = vu;

        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        movdir[0] = w;//전
        movdir[1] = -w;//후
        movdir[2] = u;//우
        movdir[3] = -u;//좌

        printf("전방 %f %f %f\n", movdir[0].x(), movdir[0].y(), movdir[0].z());
        printf("후방 %f %f %f\n", movdir[1].x(), movdir[1].y(), movdir[1].z());
        printf("우방 %f %f %f\n", movdir[2].x(), movdir[2].y(), movdir[2].z());
        printf("좌방 %f %f %f\n", movdir[3].x(), movdir[3].y(), movdir[3].z());
    }

    __device__ ray get_ray(curandState* state,int i, int j) {
        auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        pixel_center += pixel_delta_u * (random_double(state) - 0.5) + pixel_delta_v * (random_double(state) - 0.5);//픽셀 내 임의의 점

        return ray(center, pixel_center- center);//렌즈 처리하려면 center 바꿔야함
    }
    __device__ void moveorigin(int direction) {
        printf("카메라 위치 %f %f %f\n", lookfrom.x(), lookfrom.y(), lookfrom.z());
        lookfrom += movdir[direction];
        printf("  이후 카메라 위치 %f %f %f\n", lookfrom.x(), lookfrom.y(), lookfrom.z());
        lookat += movdir[direction];
        initialize();
    }
    __device__ void initialize() {

        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        movdir[0] = w;//전
        movdir[1] = -w;//후
        movdir[2] = u;//우
        movdir[3] = -u;//좌
    }
private:
    vec3* movdir = new vec3[4];
    double aspect_ratio = 1.0;  
    int    image_width = 100;  
    int    samples_per_pixel = 10;   

    double vfov = 90;              
    vec3 lookfrom = vec3(0, 0, -1);
    vec3 lookat = vec3(0, 0, 0);
    vec3   vup = vec3(0, 1, 0);     

    int    image_height;  
    vec3 center;         
    vec3 pixel00_loc;    
    vec3 pixel_delta_u;  
    vec3 pixel_delta_v;  
    vec3 u, v, w;        

};

#endif