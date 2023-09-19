
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>
#include <stdio.h>
#include <windows.h>
int windowWidth = 800, windowHeight = 800;

GLuint pbo_dest;
struct cudaGraphicsResource* cuda_pbo_dest_resource;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;
unsigned int image_width = 800;
unsigned int image_height = 800;
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource) {

    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void* data = malloc(size_tex_data);

    //예제에서는 GL_ARRAY_BUFFER인데 GL_PIXEL_UNPACK_BUFFER가 텍스쳐를 담는다고 함. 알아봐야될듯
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone);


}
void deletePBO(GLuint* pbo) {
    glDeleteBuffers(1, pbo);
    *pbo = 0;
}











//__global__ void update_frame(cudaSurfaceObject_t surface, int max_x, int max_y, int ns, curandState* rand_state, float t) {

//}

void renderScene() {
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

void initGL() {

    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("CType Ray Tracing");
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, windowWidth, windowHeight, 0, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glutDisplayFunc(renderScene);
    glutMainLoop();
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    initGL();
    return 0;
}
