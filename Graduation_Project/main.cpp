#include <stdio.h>
#include <sstream>
#include <windows.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include "timer.h"
int windowWidth = 1600, windowHeight = 900;
unsigned int image_width = windowWidth;
unsigned int image_height = windowHeight;
int mouse_prev_x = 0, mouse_prev_y = 0;
int mouse_dx = 0, mouse_dy = 0;

int keyboard_prev = 0;
int keyboard_d = 0;
int pixels = windowWidth * windowHeight;
dim3 block(32, 16, 1);
dim3 grid(image_width / block.x, image_height / block.y, 1);

GLuint pbo_dest;
GLuint texture;
GLuint shader;
struct cudaGraphicsResource* cuda_pbo_dest_resource;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;
static const char* glsl_draw_fragmentShader =
"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";
extern "C" void generatePixel(dim3 grid, dim3 block, int sbytes,
    unsigned int* g_odata, int imgh,int imgw);
extern "C" void initCuda(dim3 grid, dim3 block,int image_height, int image_width,int pixels);
extern "C" void moveCamera(int direction,int weight);
extern "C" void RotateCamera(int x, int y);
extern "C" void manivfov(int x);
void createPBO(GLuint* pbo, struct cudaGraphicsResource** pbo_resource) {
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void* data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo,cudaGraphicsMapFlagsNone);

}

void deletePBO(GLuint* pbo) {
    glDeleteBuffers(1, pbo);
    *pbo = 0;
}

void generateImage() {
    unsigned int* out_data;

    cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&out_data, &num_bytes, cuda_pbo_dest_resource);

    //쿠다함수 추가 필요
    generatePixel(grid, block, 0, out_data, image_height,image_width);

    cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void displayImage(GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, windowWidth, windowHeight);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

void renderScene() {
    std::ostringstream Title;
    Timer t;
    generateImage();
    displayImage(texture);

    cudaDeviceSynchronize();
    glutSwapBuffers();
    Title << "CType Ray Tracing ";
    Title << 1/t.now();
    Title << "f";
    glutSetWindowTitle(Title.str().c_str());
}

void createTexture(GLuint* texture, unsigned int size_x, unsigned int size_y) {
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA,GL_UNSIGNED_BYTE, NULL);
}
void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);

    *tex = 0;
}
void FreeResource() {
    deletePBO(&pbo_dest);
    deleteTexture(&texture);
}
GLuint compileGLSLprogram(const char* vs, const char* fs) {
    GLuint v, f, p = 0;
    
    p = glCreateProgram();

    if (vs) {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vs, NULL);
        glCompileShader(v);
        
        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled) {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(v);
            return 0;
        }
        else {
            glAttachShader(p, v);
        }
    }

    if (fs) {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fs, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled) {
            //#ifdef NV_REPORT_COMPILE_ERRORS
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            //#endif
            glDeleteShader(f);
            return 0;
        }
        else {
            glAttachShader(p, f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint*)&infologLength);

    if (infologLength > 0) {
        char* infoLog = (char*)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei*)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
}
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    keyboard_d += 1;
    if (key == 'w') {
        moveCamera(0,keyboard_d);
    }
    else if (key == 'a') {
        moveCamera(3,keyboard_d);
    }
    else if (key == 's') {
        moveCamera(1, keyboard_d);
    }
    else if (key == 'd') {
        moveCamera(2, keyboard_d);
    }
    //default:
        //printf("%c 눌림", key);

}
void keyboardup(unsigned char key, int /*x*/, int /*y*/) {
    if (key == 'w' || key == 'a' || key == 's' || key == 'd') {
        keyboard_d = 0;
    }
    //default:
        //printf("%c 눌림", key);

}
void initGLBuffer() {
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
    createTexture(&texture, image_width, image_height);
    shader = compileGLSLprogram(NULL, glsl_draw_fragmentShader);
}

void initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE );
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("CType Ray Tracing");
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "glewInit() failed!" << std::endl;
    }
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, windowWidth, windowHeight);
    //이밑으로 없어도될거같긴한데 일단넣음
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (GLfloat)windowWidth / (GLfloat)windowHeight, 0.1f,
        10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);
}
void myMouseMove(int x, int y) {
    int i = mouse_dx;
    mouse_dx = x - mouse_prev_x;
    mouse_dy = y - mouse_prev_y;
    mouse_prev_x = x;
    mouse_prev_y = y;
    if (i != -1) {
        RotateCamera(mouse_dx, -mouse_dy);
    }
}
void myMouseWheel(int button, int dir, int x, int y) {
    if (button == 3) {//아래
        manivfov(-1);
    }
    else if (button == 4) {//위
        manivfov(1);
    }
    else if (button == GLUT_LEFT_BUTTON) {
        if (dir == GLUT_UP) {
            mouse_dx = -1;
            mouse_dy = -1;
        }
    }
}
int main(int argc,char **argv) {
    initGL(&argc, argv);
    initCuda(grid,block,image_height,image_width,pixels);
    findCudaDevice(argc, (const char **)argv);

    cudaDeviceSynchronize();
    glutDisplayFunc(renderScene);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardup);
    glutMouseFunc(myMouseWheel);
    glutMotionFunc(myMouseMove);
    glutIdleFunc(renderScene);
    initGLBuffer();
    glutMainLoop();
    FreeResource();


    return 1;
}