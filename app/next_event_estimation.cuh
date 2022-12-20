#if !defined(_NEXT_EVENT_ESTIMATION_CUH_) && !defined(NO_NEXT_EVENT_ESTIMATION)
#define _NEXT_EVENT_ESTIMATION_CUH_

#include <curand_kernel.h>
#include "utils.cuh"
#include "path_tracing.cuh"

#define ORTHOGONALIZE(a, b) normalize(b - dot(a, b)*a)

/**
 * Samples the given triangle from the position `o` with a distribution uniform w.r.t. the triangle's subtended solid angle.
 * This function implements [Arvo, 1996 - https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf]
 * 
 * @param randState pseudo-RNG state
 * @param o the position in world space from which to sample the triangle
 * @param tri the triangle to sample
 * @param solidAngle (out) the solid angle subtended by the triangle `tri` from the position `o`
 * @returns a normalized direction towards the triangle
 */
inline __device__ float3 sampleTriangleUniformSolidAngle(curandState* randState, float3 o, Triangle& tri, float* solidAngle) {
    // Get the positions of the vertices of the triangle's projection on the unit sphere with center o
    auto a = normalize(tri.v1 - o);
    auto b = normalize(tri.v2 - o);
    auto c = normalize(tri.v3 - o);

    // Calculate internal angles of spherical triangle
    auto ba = ORTHOGONALIZE(a, b - a);
    auto ca = ORTHOGONALIZE(a, c - a);
    auto ab = ORTHOGONALIZE(b, a - b);
    auto cb = ORTHOGONALIZE(b, c - b);
    auto bc = ORTHOGONALIZE(c, b - c);
    auto ac = ORTHOGONALIZE(c, a - c);
    auto alpha = acosf(clamp(dot(ba, ca), -1.0, 1.0));
    auto beta = acosf(clamp(dot(ab, cb), -1.0, 1.0));
    auto gamma = acosf(clamp(dot(bc, ac), -1.0, 1.0));

    // Calculate lengths of spherical triangle's edges
    auto aLen = acosf(clamp(dot(b, c), -1.0, 1.0));
    auto bLen = acosf(clamp(dot(c, a), -1.0, 1.0));
    auto cLen = acosf(clamp(dot(a, b), -1.0, 1.0));

    *solidAngle = alpha + beta + gamma - PI;

    // Select sub-triangle area
    float areaS = curand_uniform(randState) * *solidAngle;

    auto s = sinf(areaS - alpha);
    auto t = cosf(areaS - alpha);
    auto u = t - cosf(alpha);
    auto v = s + sinf(alpha) * cosf(cLen);
    auto q = ((v*t - u*s)*cosf(alpha) - v) / ((v*s + u*t)*sinf(alpha));

    // Compute third vertex of sub-triangle
    auto cS = q*a + sqrtf(1.0 - q*q)*ORTHOGONALIZE(a, c);
    auto z = 1.0 - curand_uniform(randState)*(1.0 - dot(cS, b));

    return z*b + sqrtf(1.0 - z*z)*ORTHOGONALIZE(b, cS);
}

inline __device__ void findClosestIntersection(Ray& r, Triangle* tris, uint32_t triNum, Triangle** intersectionTri, float* t);

/**
 * Checks if a given ray hits a specific triangle first
 * 
 * @param r the ray
 * @param to the destination triangle
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @returns true if there is visibility between the two points, false otherwise
 */
inline __device__ bool visibility(Ray& r, Triangle* to, Triangle* tris, uint32_t triNum) {
    Triangle* intersectionTri;
    float t;
    findClosestIntersection(r, tris, triNum, &intersectionTri, &t);

    return intersectionTri == to;
}

/**
 * Samples the specified triangle light
 * 
 * @param randState pseudo-RNG state
 * @param tris list of all the triangles in the scene
 * @param mats list of all the materials in the scene
 * @param triNum number of triangles pointed to by `tris`
 * @param lightIndex index of the light to sample in `tris`
 * @param currentHit the triangle from which to sample direct lighting
 * @param samplePosition position in world space from which to sample direct lighting
 * @param n the normal at samplePosition
 * @returns the direct lighting contribution
 */
inline __device__ float3 sampleLight(
    curandState* randState,
    Triangle* tris,
    Material* mats,
    uint32_t triNum,
    uint32_t lightIndex,
    Triangle* currentHit,
    float3 samplePosition,
    float3 n
) {
    // Get the sample direction
    float solidAngle;
    auto sampleDirection = sampleTriangleUniformSolidAngle(randState, samplePosition, tris[lightIndex], &solidAngle);

    // Check for NaN (TODO: this is not great)
    if (sampleDirection.x != sampleDirection.x) return make_float3(0.0, 0.0, 0.0);
    
    Ray r = {
        samplePosition,
        sampleDirection
    };

    // Compute the cosine of the angle between the hit normal and the sample direction
    auto dotnw = dot(sampleDirection, n);

    // Check if the sample is behind the current hit
    if (dotnw < 0.0) return make_float3(0.0, 0.0, 0.0);
    
    return mats[tris[lightIndex].materialIndex].emissivity              * // emissivity
           (visibility(r, tris + lightIndex, tris, triNum) ? 1.0 : 0.0) * // visibility
           dotnw                                                        * // cosine term
           solidAngle;                                                    // 1/pdf of sample
}

/**
 * Samples emissive triangles directly
 * 
 * @param randState pseudo-RNG state
 * @param tris list of all the triangles in the scene
 * @param mats list of all the materials in the scene
 * @param triNum number of triangles pointed to by `tris`
 * @param lightsIndices indices of emissive triangles in `tris`
 * @param lightsNum number of indices pointed to by `tris`
 * @param currentHit the triangle from which to sample direct lighting
 * @param samplePosition position in world space from which to sample direct lighting
 * @param n the normal at samplePosition
 * @returns the direct lighting contribution
 */
inline __device__ float3 sampleLights(
    curandState* randState,
    Triangle* tris,
    Material* mats,
    uint32_t triNum,
    uint32_t* lightsIndices,
    uint32_t lightsNum,
    Triangle* currentHit,
    float3 samplePosition,
    float3 n
) {
    uint32_t currentHitIndex = currentHit - tris;
    
    // If we can't choose a suitable light from `lightsIndices`, return immediately
    if (lightsNum == 0 || (lightsNum == 1 && *lightsIndices == currentHitIndex)) {
        return make_float3(0.0, 0.0, 0.0);
    }

    // Choose a light to sample randomly
    uint32_t lightIndex;
    do {
        lightIndex = lightsIndices[curand(randState) % lightsNum];
    } while (lightIndex == currentHitIndex);

    // Note that we multiply by lightsNum. That is because we are sampling only one random light, and not
    // all lightsNum of them
    return (float)lightsNum * sampleLight(randState, tris, mats, triNum, lightIndex, currentHit, samplePosition, n);
}

#endif
