
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
int windowWidth = 800, windowHeight = 800;


//__global__ void update_frame(cudaSurfaceObject_t surface, int max_x, int max_y, int ns, curandState* rand_state, float t) {

//}


static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_A) {
        //    camera_location = camera_location - vec3(0.1f, 0.0f, 0.0f);
    }
    else if (key == GLFW_KEY_W) {
        printf("A눌림");
        //      camera_location = camera_location - vec3(0.0f, 0.0f, 0.1f);
    }
    else if (key == GLFW_KEY_S) {
        printf("A눌림");
        //    camera_location = camera_location + vec3(0.0f, 0.0f, 0.1f);
    }
    else if (key == GLFW_KEY_D) {
        printf("A눌림");
        //     camera_location = camera_location + vec3(0.1f, 0.0f, 0.0f);
    }
}
static void resize_callback(GLFWwindow* window, int new_width, int new_height) {
    glViewport(0, 0, windowWidth = new_width, windowHeight = new_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, windowWidth, windowHeight, 0.0, 0.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}


int main()
{
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);


    if (!glfwInit()) {
        return -1;
    }

    window = glfwCreateWindow(windowWidth, windowHeight, "CType Ray Tracing", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, resize_callback);

    glfwMakeContextCurrent(window);

    if (!gladLoadGL()) {
        printf("GLAD 초기화 실패");
        return -1;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    while (!glfwWindowShouldClose(window)) {

        //cudaDeviceSynchronize(); //쿠다 연산이 완료될 때 까지 기다리는 함수.
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
