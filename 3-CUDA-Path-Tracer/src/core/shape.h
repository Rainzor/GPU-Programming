#pragma once

#include "glm/glm.hpp"

enum Primitive: unsigned char {
    SPHERE,
    CUBE,
    TRIANGLE,
};

struct Transform {
	glm::vec3 translation = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);
	glm::mat4 transform;
	glm::mat4 inverseTransform;
	glm::mat4 invTranspose;
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
    enum Primitive type = TRIANGLE;
    int trimeshId = -1;
    size_t materialId = 0;
    Transform transform;
};