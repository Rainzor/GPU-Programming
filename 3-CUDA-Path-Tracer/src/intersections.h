#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "sceneStructs.h"
#include "utilities.h"

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
    return r.origin + (t - 0.001f) * glm::normalize(r.direction);
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
__host__ __device__ bool boxIntersectionTest(Geom box, Ray r, int tmax,
        Intersection & intersection, bool &outside) {
	intersection.t = -1;
	glm::vec3 intersectionPoint;
    Ray q;
    q.origin    =                multiplyMV(box.transform.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

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
        intersectionPoint = multiplyMV(box.transform.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        intersection.surfaceNormal = glm::normalize(multiplyMV(box.transform.invTranspose, glm::vec4(tmin_n, 0.0f)));
		intersection.t = glm::length(r.origin - intersectionPoint);
		intersection.materialId = box.materialId;
		return true;
    }
    return false;
}

/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param  r				 The ray to test.
 * @param  tmax			     The maximum distance along the ray to test.
 * @param  intersection      Output the record of the intersection.
 * @param  outside           Whether the ray came from outside the sphere.
 * @return                   Whether the intersection test was successful.
 */
__host__ __device__ bool sphereIntersectionTest(Geom sphere, Ray r, float tmax,
        Intersection& intersection, bool &outside) {
	glm::vec3 intersectionPoint;
    glm::vec3 normal;
	intersection.t = -1;
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

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

    intersectionPoint = multiplyMV(sphere.transform.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.transform.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

	intersection.surfaceNormal = normal;
	intersection.uv = uv;
	intersection.t = glm::length(r.origin - intersectionPoint);
	intersection.materialId = sphere.materialId;
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

__host__ __device__ bool trimeshIntersectionTest(Geom mesh, Ray r, float tmax,
	Intersection& intersection, bool& outside, const TriangleMesh& trimesh) {
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	intersection.t = -1;

	Ray q;
	q.origin = multiplyMV(mesh.transform.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(mesh.transform.inverseTransform, glm::vec4(r.direction, 0.0f)));
    
	float t = tmax;
	glm::vec3 weight;
	for (size_t i = 0; i < trimesh.num; i++) {
		Triangle &tri = trimesh.triangles[i];
		glm::vec3 v0 = tri.v0;
		glm::vec3 v1 = tri.v1;
		glm::vec3 v2 = tri.v2;
        glm::vec3 baryPos;
		if (glm::intersectRayTriangle(q.origin, q.direction, v0, v1, v2, baryPos)) {
			float t_temp = baryPos.z;
			if (t_temp < t && t_temp > 0) {
				t = t_temp;
                weight = glm::vec3(1 - baryPos.x - baryPos.y, baryPos.x, baryPos.y);
			}
		}
	}
	if (t < tmax) {
		intersectionPoint = multiplyMV(mesh.transform.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        
		normal = weight.x * trimesh.triangles[0].n0 + 
                 weight.y * trimesh.triangles[0].n1 + 
                 weight.z * trimesh.triangles[0].n2;
		normal = glm::normalize(multiplyMV(mesh.transform.invTranspose, glm::vec4(normal, 0.0f)));
		intersection.surfaceNormal = normal;
		intersection.t = glm::length(r.origin - intersectionPoint);

		intersection.uv = weight.x * trimesh.triangles[0].uv0 +
			              weight.y * trimesh.triangles[0].uv1 +
			              weight.z * trimesh.triangles[0].uv2;

		intersection.materialId = mesh.materialId;
		return true;
	}
	return false;
}