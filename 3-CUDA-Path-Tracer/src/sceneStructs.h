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

#define BACKGROUND_COLOR (glm::vec3(0.1f))

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


struct Triangle{
	glm::vec3 v0, v1, v2;
	glm::vec3 n0, n1, n2;
	glm::vec2 uv0, uv1, uv2;
};

struct TriangleMesh {
	Triangle* triangles;
	size_t num;
};

struct Geom {
    enum Primitive type;
	size_t trimeshId;
    size_t materialId;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum TextureType {
    RGB,
    BITMAP,
};
struct Texture {
    TextureType type = RGB;
    glm::vec3 color = glm::vec3(0.0f);
    size_t bitmapId = 0; 
};

struct Bitmap {
	int width;
	int height;
	glm::u8vec4* pixels;
};

__host__ __device__ inline glm::vec3 getPixel(const Bitmap& bmp, glm::vec2 uv) {
    int i = uv.x * bmp.width;
	int j = (1 - uv.y) * bmp.height;// flip y

    if (i >= bmp.width) i = bmp.width - 1;
    if (j >= bmp.height) j = bmp.height - 1;

    int index = (i + j * bmp.width);
	float r = bmp.pixels[index].r / 255.f;
	float g = bmp.pixels[index].g / 255.f;
	float b = bmp.pixels[index].b / 255.f;
	float a = bmp.pixels[index].a / 255.f;
    return glm::vec3(r, g, b) * a;
}

struct Material {
    enum MaterialType type;
    Texture texture;
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
struct Intersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    size_t materialId;
};

struct Sample {
    float pdf;
    glm::vec3 BSDF;
    Ray ray;
};