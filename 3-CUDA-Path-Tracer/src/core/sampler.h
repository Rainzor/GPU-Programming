#pragma once

#include "intersections.h"



// MIS balance heuristic
__host__ __device__ float powerHeuristic(float f, float g, int nf=1, int ng=1) {
    f = nf * f;
    g = ng * g;
    return (f * f) / (g * g + f * f);
}

/**
 * Computes a cosine-weighted random direction on a hemisphere surface.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionOnHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or 
    // whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec2 sampleUnitDiskConcentric(const glm::vec2& u){
    glm::vec2 uOffset = 2.f * u - glm::vec2(1.f);
    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::vec2(0.f);
    }

    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI / 4.f * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PI / 2.f - PI / 4.f * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(std::cos(theta), std::sin(theta));
}


/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method returns a Sample struct with the scattered ray direction, the bsdf and the pdf.
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
Sample scatterRay(
        const PathSegment & pathSegment,
        const Intersection & intersection,
        const Material *material,
	    const glm::vec3& abedo,
        thrust::default_random_engine &rng) {
    // ! Scatter the ray according to the type of material
    thrust::uniform_real_distribution<float> u01(0, 1);
    Sample sample;
    glm::vec3 direction = glm::vec3(0.f);
    if (material->type == MaterialType::SPECULAR){ // Perfect Reflection
        direction = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
        float cosTheta = glm::dot(direction, intersection.surfaceNormal);
        sample.BSDF = abedo / cosTheta;
        sample.pdf = 1.f;
    } else if (material->type == MaterialType::DIFFUSE){ // Lambertian
        direction = calculateRandomDirectionOnHemisphere(intersection.surfaceNormal, rng);
        sample.BSDF = abedo / PI;
        sample.pdf = glm::dot(direction, intersection.surfaceNormal) / PI;
    }

    sample.ray.origin = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
    sample.ray.direction = glm::normalize(direction);
    return sample;
}
