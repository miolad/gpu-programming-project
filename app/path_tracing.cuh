#ifndef _PATH_TRACING_CUH_
#define _PATH_TRACING_CUH_

#include <curand_kernel.h>

#include "utils.cuh"
#include "camera.cuh"

#ifndef NO_NEXT_EVENT_ESTIMATION
#include "next_event_estimation.cuh"
#endif

/**
 * Randomly samples the hemisphere defined by `w` with a cosine weighted distribution
 * 
 * @param randState pseudo-RNG state
 * @param w normalized direction defining the hemisphere to sample
 */
inline __device__ float3 sampleHemisphereCosineWeighted(curandState* randState, float3 w) {
    // Get two uniformly random values in [0, 1]
    float u1 = curand_uniform(randState), u2 = curand_uniform(randState);

    // Generate sample direction in hemisphere over (0, 0, 1)
    float cos_theta = sqrtf(1.0 - u1);
    float sin_theta = sqrtf(u1);
    float phi       = 2.0 * PI * u2;
    float3 sample   = {
        cosf(phi) * sin_theta,
        sinf(phi) * sin_theta,
        cos_theta
    };

    // Rotate the sample to match the provided normal
    float3 u = normalize(cross(abs(w.x) > 0.1 ? float3{0.0, 1.0, 0.0} : float3{1.0, 0.0, 1.0}, w));
    float3 v = cross(u, w);

    return sample.x * u + sample.y * v + sample.z * w;
}

/**
 * Computes the intersection between the provided ray and triangle.
 * This function implements the Moller-Trumbore algorithm (https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)
 * 
 * @param r the ray
 * @param tri triangle to intersect
 * @returns the "time" of the ray's intersection with the triangle, i.e. the distance along the ray's direction
 *          to the point of the intersection, in world units.
 *          If no intersection is found, -1.0 is returned.
 */
inline __device__ float rayTriangleIntersection(Ray& r, Triangle& tri) {
    auto edge1 = tri.v2 - tri.v1;
    auto edge2 = tri.v3 - tri.v1;
    auto h = cross(r.direction, edge2);
    auto a = dot(edge1, h);

    if (a > -EPS && a < EPS)
        return -1.0;

    auto f = 1.0 / a;
    auto s = r.origin - tri.v1;
    auto u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return -1.0;

    auto q = cross(s, edge1);
    auto v = f * dot(r.direction, q);

    if (v < 0.0 || u + v > 1.0)
        return -1.0;

    auto t = f * dot(edge2, q);
    return t > EPS ? t : -1.0;
}

/**
 * Find the closest intersection between a ray and a list of triangles by iteratively testing the
 * intersection between the ray and each triangle
 * 
 * @param r the ray
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @param intersectionTri (out) pointer to the closest triangle intersected, or NULL if no intersection was found
 * @param t (out) "time" of the ray to the closest intersection. This is invalid if intersectionTri is NULL
 */
inline __device__ void findClosestIntersection(Ray& r, Triangle* tris, uint32_t triNum, Triangle** intersectionTri, float* t) {
    *t = RAY_MAX_T;
    *intersectionTri = NULL;
    
    for (uint32_t i = 0; i < triNum; ++i) {
        float ti = rayTriangleIntersection(r, tris[i]);

        if (ti > 0.0 && ti < *t) {
            *t = ti;
            *intersectionTri = tris + i;
        }
    }
}

/**
 * Render the scene through the provided virtual camera into the framebuffer
 * 
 * @param tris list of all the triangles in the scene
 * @param mats list of all the materials in the scene
 * @param triNum number of triangles pointed to by `tris`
 * @param lightsIndices indices of emissive triangles in `tris`
 * @param lightsNum number of indices pointed to by `tris`
 * @param cam virtual camera
 * @param batch batch number of this invocation
 * @param fb framebuffer of RES_X by RES_Y pixels to render to
 */
__global__ void __launch_bounds__(16*16) pathTrace(
    Triangle* tris,
    Material* mats,
    uint32_t triNum,
#ifndef NO_NEXT_EVENT_ESTIMATION
    uint32_t* lightsIndices,
    uint32_t lightsNum,
#endif
    Camera cam,
    uint32_t batch,
    float3* fb
) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t pixelIndex = x + y * RES_X;

    // Terminate pixels outside the framebuffer
    if (x >= RES_X || y >= RES_Y) return;

    // Initialize cuRAND state
    curandState randState;
    curand_init(batch, pixelIndex, 0, &randState);

    // Get this pixel's camera ray (flipping the y axis)
    auto cameraRay = cam.getRayThroughPixel(make_int2(x, RES_Y - y - 1));

    // Cache first bounce for all samples in this batch
    Triangle* cameraBounceIntersectionTri;
    float ti;
    findClosestIntersection(cameraRay, tris, triNum, &cameraBounceIntersectionTri, &ti);

    if (cameraBounceIntersectionTri == NULL) {
        fb[pixelIndex] = make_float3(0.0, 0.0, 0.0);
        return;
    }

    float3 initialThroughput = mats[cameraBounceIntersectionTri->materialIndex].albedo;

    // Will accumulate the contribution of SAMPLES_PER_BATCH samples
    float3 color = mats[cameraBounceIntersectionTri->materialIndex].emissivity * (float)SAMPLES_PER_BATCH;

    for (uint32_t sample = 0; sample < SAMPLES_PER_BATCH; ++sample) {
        auto r = cameraRay;
        auto throughput = initialThroughput;
        auto intersectionTri = cameraBounceIntersectionTri;
        auto t = ti;

        // Note that `bounce` starts at 1 because the first camera ray is cached for all samples in the batch
        for (uint32_t bounce = 1; bounce < MAX_BOUNCES; ++bounce) {
            // Get new ray
            auto n      = (dot(r.direction, intersectionTri->normal) < 0.0 ? 1.0 : -1.0) * intersectionTri->normal;
            r.origin    = r.origin + r.direction * t + n * EPS; // Shift ray origin by a small amount to avoid self intersections due to floating point precision
            r.direction = sampleHemisphereCosineWeighted(&randState, n);

#ifndef NO_NEXT_EVENT_ESTIMATION
            // Sample direct lighting
            auto directLighting = sampleLights(
                &randState,
                tris,
                mats,
                triNum,
                lightsIndices,
                lightsNum,
                intersectionTri,
                r.origin,
                n
            );
            color = color + throughput * directLighting * ONE_OVER_PI;
#endif
            
            // Intersect ray with geometry
            findClosestIntersection(r, tris, triNum, &intersectionTri, &t);

            // if no intersection, break the loop, this sample is done
            if (intersectionTri == NULL) break;

            // Get the intersection material
            Material* mat = mats + intersectionTri->materialIndex;

#ifdef NO_NEXT_EVENT_ESTIMATION
            // Add emission to output color
            color = color + throughput * mat->emissivity;
#endif

            // Add surface contribution to path throughput
            // Note that a lot of stuff cancels out here, due to the simple diffuse surface constraint.
            // In particular, the PI at the denominator of the Lambertian BRDF cancels out with the numerator
            // of the cosine weighted PDF, whose denominator in turn cancels out with the rendering equation's
            // cosine term.
            throughput = throughput * mat->albedo;

            // Russian roulette
            if (bounce >= MIN_BOUNCES) {
                auto p = max(throughput.x, max(throughput.y, throughput.z));
                if (p < curand_uniform(&randState)) break;
                throughput = throughput * (1.0/p);
            }
        }
    }

    // Write normalized color to framebuffer
    fb[pixelIndex] = (
        fb[pixelIndex] * float(batch) +
        color * (1.0 / float(SAMPLES_PER_BATCH))
    ) * (1.0 / float(batch + 1));
}

#endif
