/**
 * @file      main.cpp
 * @brief     Example Boids flocking simulation for CIS 565
 * @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
 * @date      2013-2017
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
// #define VISUALIZE 1
#define UNIFORM_GRID 1
#define COHERENT_GRID 1


// LOOK-1.2 - change this to adjust particle count in the simulation

int N_FOR_VIS = 100000;
bool save = false;
bool visualize = false;
const float DT = 0.2f;


#if UNIFORM_GRID && COHERENT_GRID
    const std::string filename = "coherent_grid.txt";
#elif UNIFORM_GRID
    const std::string filename = "uniform_grid.txt";
#else
    const std::string filename = "naive.txt";
#endif
/**
 * C main function.
 */
int main(int argc, char *argv[]) {
    projectName = "565 CUDA Intro: Boids";
	if (init(argc, argv)) {
		mainLoop();
		Boids::endSimulation();
	} else {
		return 1;
	}	
	return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;
bool stop = false;
/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--save" || std::string(argv[i]) == "-s") {
            save = true;
        }
        if (std::string(argv[i]) == "--vis" || std::string(argv[i]) == "-v") {
            visualize = true;
        }
        if (std::string(argv[i]) == "--num" || std::string(argv[i]) == "-n") {
            N_FOR_VIS = std::stoi(argv[i + 1]);
            i++;
        }
    }

    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
            << "Error: GPU device number is greater than the number of devices!"
            << " Perhaps a CUDA-capable GPU is not installed?"
            << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    initVAO(N_FOR_VIS);

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(boidVBO_positions);
    cudaGLRegisterBufferObject(boidVBO_velocities);

    // Initialize N-body simulation
    Boids::initSimulation(N_FOR_VIS);

    updateCamera();

    initShaders(program);

    glEnable(GL_DEPTH_TEST);

    return true;
}

void initVAO(int N) {

    std::unique_ptr<GLfloat[]> bodies{new GLfloat[4 * (N)]};
    std::unique_ptr<GLuint[]> bindices{new GLuint[N]};

    glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
    glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

    for (int i = 0; i < N; i++) {
        bodies[4 * i + 0] = 0.0f;
        bodies[4 * i + 1] = 0.0f;
        bodies[4 * i + 2] = 0.0f;
        bodies[4 * i + 3] = 1.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
    glGenBuffers(1, &boidVBO_positions);
    glGenBuffers(1, &boidVBO_velocities);
    glGenBuffers(1, &boidIBO);

    glBindVertexArray(boidVAO);

    // Bind the positions array to the boidVAO by way of the boidVBO_positions
    glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions);                                        // bind the buffer
    glBufferData(GL_ARRAY_BUFFER, 4 * (N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data from the CPU to the GPU
    glEnableVertexAttribArray(positionLocation);                                             // create a pointer to the boidVBO_positions data
    glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
    glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
    glBufferData(GL_ARRAY_BUFFER, 4 * (N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // the same value
    glEnableVertexAttribArray(velocitiesLocation);                                           // different location
    glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GLuint *program) {
    GLint location;

    program[PROG_BOID] = glslUtility::createProgram(
        "shaders/boid.vert.glsl",
        "shaders/boid.geom.glsl",
        "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
        glUniform3fv(location, 1, &cameraPosition[0]);
    }
}

//====================================
// Main loop
//====================================
void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void **)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void **)&dptrVertVelocities, boidVBO_velocities);

// execute the kernel
#if UNIFORM_GRID && COHERENT_GRID
    Boids::stepSimulationCoherentGrid(DT);
#elif UNIFORM_GRID
    Boids::stepSimulationScatteredGrid(DT);
#else
    Boids::stepSimulationNaive(DT);
#endif

    if(visualize){
        Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
    }
    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
}

void mainLoop() {
    double fps = 0;
    double timebase = glfwGetTime();
    int frame = 0;
    int fps_count = 0;
    double fps_sum = 0;
    int fps_samples = 10;  // Number of FPS samples to average
    int start_sample = 5;
    if(!save){
        start_sample = 0;
        fps_samples = 100000;
    }

    while (!glfwWindowShouldClose(window) && fps_count < fps_samples) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            if(fps_count >= start_sample)
                fps_sum += fps;
            fps_count++;
            timebase = time;
            frame = 0;
        }

        if (stop)
            continue;

        runCUDA();

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << deviceName;
        glfwSetWindowTitle(window, ss.str().c_str());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(visualize){
            glUseProgram(program[PROG_BOID]);
            glBindVertexArray(boidVAO);
            glPointSize((GLfloat)pointSize);
            glDrawElements(GL_POINTS, N_FOR_VIS, GL_UNSIGNED_INT, 0);
            glPointSize(1.0f);

            glUseProgram(0);
            glBindVertexArray(0);

            glfwSwapBuffers(window);
        }
    }
    double average_fps = fps_sum /(fps_count - start_sample);
    if(save){
        std::ofstream fpsFile;
        fpsFile.open(filename, std::ios::app);
        if (fps_count > 0) {
            fpsFile << N_FOR_VIS << "," << average_fps << "\n";  // 写入粒子数和平均FPS值
        }

        fpsFile.close();  // 关闭文件
    }
    std::cout << "N = " << N_FOR_VIS << ", FPS = " << average_fps << std::endl;
    glfwDestroyWindow(window);
    glfwTerminate();
}

void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    } else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        stop = !stop;
    }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow *window, double xpos, double ypos) {
    if (leftMousePressed) {
        // compute new camera parameters
        phi += (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
        updateCamera();
    } else if (rightMousePressed) {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
        updateCamera();
    }

    lastX = xpos;
    lastY = ypos;
}

void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
}