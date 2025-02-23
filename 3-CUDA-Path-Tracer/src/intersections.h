#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"
#include "bvh.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - 0.00001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}


/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param Intersection       Output the record of the intersection.
 * @return                   Whether the intersection test was successful.
 */
__host__ __device__ bool boxIntersectionTest(Ray r, int tmax,
        Intersection & intersection, bool &outside) {
	intersection.t = -1;
	glm::vec3 intersectionPoint;
    Ray &q = r;


    float tmin = -1e38f;
    //float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }

		intersection.t = tmin;
		intersection.surfaceNormal = tmin_n;
		return true;
    }
    return false;
}

/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 1 and is centered at the origin.
 *
 * @param  r				 The ray to test.
 * @param  tmax			     The maximum distance along the ray to test.
 * @param  intersection      Output the record of the intersection.
 * @param  outside           Whether the ray came from outside the sphere.
 * @return                   Whether the intersection test was successful.
 */
__host__ __device__ bool sphereIntersectionTest(Ray r, float tmax,
        Intersection& intersection, bool &outside) {
	glm::vec3 intersectionPoint;
    glm::vec3 normal;
	intersection.t = -1;
    float radius = 1.f;


	Ray &rt = r;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return false;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return false;
    } else if (t1 > 0 && t2 > 0) {
        t = MIN(t1, t2);
        outside = true;
    } else {
        t = MAX(t1, t2);
        outside = false;
    }

	if (t > tmax) {
		return false;
	}

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
	glm::vec3 outward_normal = glm::normalize(objspaceIntersection);
	glm::vec2 uv = glm::vec2(0.f);

	float theta = glm::acos(outward_normal.y);
	float phi = glm::atan(outward_normal.z, outward_normal.x) + PI;
	uv.x = phi / (2 * PI);
	uv.y = theta / PI;

	normal = outward_normal;
    if (!outside) {
        normal = -normal;
    }

	intersection.surfaceNormal = normal;
	intersection.uv = uv;
	intersection.t = t;
	return true;
}

/**
* Test intersection between a ray and a transformed triangle mesh.
* 
* @param  r				    The ray to test.
* @param  tmax			    The maximum distance along the ray to test.
* @param  intersection      Output the record of the intersection.
* @param  outside           Whether the ray came from outside the triangle mesh.
* @return                   Whether the intersection test was successful.
*/

__host__ __device__ bool trimeshIntersectionTest(Ray r, float tmax,
	Intersection& intersection, bool& outside, Triangle* triangles, BVHNode* bvh_nodes) {
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	intersection.t = -1;

    Ray &q = r;

	float tmin = 0;
	float t = tmax;
	glm::vec3 weight;

	BVHNode* stack[STACK_SIZE];
	BVHNode** stackPtr = stack;

	int stack_size = 0;
	float t_root_max = tmax;
	float t_root_min = tmin;
    if (!bvh_nodes[0].bbox.intersect(q, t_root_min, t_root_max)) {
		return false;
    }
	stack_size ++;
	*(++stackPtr) = &bvh_nodes[0];
	while(stack_size > 0 && stack_size < STACK_SIZE) {
		BVHNode* node = *(stackPtr--); // pop
		stack_size--;
		if(node == NULL) break;
        // Bounding box intersection check
        else {
            if (node->isLeaf()) {
                Triangle& tri = triangles[node->primId];
                glm::vec3 baryPos;
                // Triangle-ray intersection check
                if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0, tri.v1, tri.v2, baryPos)) {
                    float t_temp = baryPos.z;
                    if (t_temp < t && t_temp > 0) {
                        t = t_temp;
                        weight = glm::vec3(1 - baryPos.x - baryPos.y, baryPos.x, baryPos.y);
                    }
                }
            }
            else {
				float tl_min = tmin;
				float tl_max = t;
				float tr_min = tmin;
				float tr_max = t;

				bool hit_left =false, hit_right =  false;
				if (node->leftId != -1)
                    hit_left = bvh_nodes[node->leftId].bbox.intersect(q, tl_min, tl_max);
				if (node->rightId != -1)
                    hit_right = bvh_nodes[node->rightId].bbox.intersect(q, tr_min, tr_max);

				if (hit_left && hit_right) {
					if (tl_min < tr_min) {
						stack_size += 2;
						*(++stackPtr) = &bvh_nodes[node->rightId];
						*(++stackPtr) = &bvh_nodes[node->leftId];
					}
					else {
						stack_size += 2;
						*(++stackPtr) = &bvh_nodes[node->leftId];
						*(++stackPtr) = &bvh_nodes[node->rightId];
					}
				}
				else if (hit_left) {
					stack_size++;
					*(++stackPtr) = &bvh_nodes[node->leftId];
				}
				else if (hit_right) {
					stack_size++;
					*(++stackPtr) = &bvh_nodes[node->rightId];
				}
            }
        }
	}

	if (t < tmax) {
		normal = weight.x * triangles[0].n0 + 
                 weight.y * triangles[0].n1 + 
                 weight.z * triangles[0].n2;
		intersection.surfaceNormal = normal;
		intersection.t = t;

		intersection.uv = weight.x * triangles[0].uv0 +
			              weight.y * triangles[0].uv1 +
			              weight.z * triangles[0].uv2;
		outside = glm::dot(q.direction, normal) < 0;
		return true;
	}
	return false;
}