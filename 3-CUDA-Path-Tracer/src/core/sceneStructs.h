#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include "glm/glm.hpp"
#include "ray.h"
#include "material.h"
#include "camera.h"
#include "shape.h"
#include "bvh.h"


#define BACKGROUND_COLOR (glm::vec3(0.1f))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

struct GeomGPU {
    enum Primitive type;
    Triangle* dev_triangles;
    BVHNode* dev_bvh_nodes;
    //Material* dev_material;
	size_t materialId;
    Transform transform;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
	glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
	//Material* material;
	size_t materialId;
    bool outside;
};

struct Sample {
    float pdf;
    glm::vec3 BSDF;
    Ray ray;
};