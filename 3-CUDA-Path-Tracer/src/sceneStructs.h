#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum Primitive {
    SPHERE,
    CUBE,
    TRIANGLE,
};

enum MaterialType {
    LIGHT,
    DIFFUSE,
    SPECULAR,
    DIELECTRIC,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum Primitive type;
    size_t num = 1;
    size_t materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material {
    enum MaterialType type;
    glm::vec3 color;
    float indexOfRefraction;
    float emittance;  
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    // Depth of field
    float aperture=0.0f;
    float focalLength=1.0f;
    
    // near and far plane
    float farClip = 1000.f;
    float nearClip = 0.001f;
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
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    float u, v;
    size_t materialId;
};

struct Sample {
    float pdf;
    glm::vec3 BSDF;
    Ray ray;
};

namespace Reflectance {
    enum Type {
        RGB,
        BITMAP,
    };
    struct Texture {
        Type type = RGB;
        glm::vec3 color = glm::vec3(0.0f);
        std::vector<glm::vec3> bitmap; 
    };
}