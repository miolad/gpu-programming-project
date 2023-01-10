#ifndef _PATH_TRACING_CUH_
#define _PATH_TRACING_CUH_

#include <curand_kernel.h>

#include "utils.cuh"
#include "camera.cuh"
#include "ray_tracing.cuh"

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
    float cos_theta = sqrtf(1.0f - u1);
    float sin_theta = sqrtf(u1);
    float phi       = PI2 * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);
    float3 sample   = {
        cosPhi * sin_theta,
        sinPhi * sin_theta,
        cos_theta
    };

    // Rotate the sample to match the provided normal
    float3 u = normalize(cross(abs(w.x) > 0.1f ? float3{0.0f, 1.0f, 0.0f} : float3{1.0f, 0.0f, 1.0f}, w));
    float3 v = cross(u, w);

    return sample.x * u + sample.y * v + sample.z * w;
}

/**
 * Render the scene through the provided virtual camera into the framebuffer
 * 
 * @param tris list of all the triangles in the scene
 * @param mats list of all the materials in the scene
 * @param triNum number of triangles pointed to by `tris`
 * @param bvhRoot root node of the BVH
 * @param bvhCachedNodesNum number of nodes of the BVH to cache in shared memory
 * @param lightsIndices indices of emissive triangles in `tris`
 * @param lightsNum number of indices pointed to by `tris`
 * @param cam virtual camera
 * @param batch batch number of this invocation
 * @param fb framebuffer of RES_X by RES_Y pixels to render to
 */
__global__ void __launch_bounds__(16*16) pathTrace(
    const Triangle* __restrict__ tris,
    const Material* __restrict__ mats,
    uint32_t triNum,
#ifndef NO_BVH
    const Node* __restrict__ bvhRoot,
    uint32_t bvhCachedNodesNum,
#endif
#ifndef NO_NEXT_EVENT_ESTIMATION
    const uint32_t* __restrict__ lightsIndices,
    uint32_t lightsNum,
#endif
    Camera cam,
    uint32_t batch,
    float3* __restrict__ fb
) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t pixelIndex = x + y * RES_X;

#ifndef NO_BVH
    // Cache part of the BVH into shared memory
    extern __shared__ Node s_bvhCache[];

    for (uint32_t loadIndex = threadIdx.x + threadIdx.y*blockDim.x; loadIndex < bvhCachedNodesNum; loadIndex += blockDim.x*blockDim.y) {
        s_bvhCache[loadIndex] = bvhRoot[loadIndex];
    }
    __syncthreads();
#endif

    // Terminate pixels outside the framebuffer
    if (x >= RES_X || y >= RES_Y) return;

    // Initialize cuRAND state
    curandState randState;
    curand_init(batch, pixelIndex, 0, &randState);

    // Get this pixel's camera ray (flipping the y axis)
    auto cameraRay = cam.getRayThroughPixel(make_int2(x, RES_Y - y - 1), batch % 16);

    // Cache first bounce for all samples in this batch
    Triangle* cameraBounceIntersectionTri;
    float ti;
    findClosestIntersection(cameraRay, tris, triNum,
#ifndef NO_BVH
        bvhRoot,
        s_bvhCache,
        bvhCachedNodesNum,
#endif
        &cameraBounceIntersectionTri, &ti
    );

    if (cameraBounceIntersectionTri == NULL) {
        fb[pixelIndex] = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    float3 initialThroughput = mats[cameraBounceIntersectionTri->materialIndex].albedo;

    // Will accumulate the contribution of SAMPLES_PER_BATCH samples
    float3 color = mats[cameraBounceIntersectionTri->materialIndex].emissivity * (float)SAMPLES_PER_BATCH;

    #pragma unroll
    for (uint32_t sample = 0; sample < SAMPLES_PER_BATCH; ++sample) {
        auto r = cameraRay;
        auto throughput = initialThroughput;
        auto intersectionTri = cameraBounceIntersectionTri;
        auto t = ti;

        // #pragma unroll // This unroll hurts performance for some reason
        for (uint32_t bounce = 0; bounce <= MAX_BOUNCES; ++bounce) {
            // Get new ray
            auto n      = (dot(r.direction, intersectionTri->normal) < 0.0f ? 1.0f : -1.0f) * intersectionTri->normal;
            r.origin    = r.origin + r.direction * t + n * EPS; // Shift ray origin by a small amount to avoid self intersections due to floating point precision
            r.direction = sampleHemisphereCosineWeighted(&randState, n);

#ifndef NO_NEXT_EVENT_ESTIMATION
            // Sample direct lighting
            auto directLighting = sampleLights(
                &randState,
                tris,
                mats,
                triNum,
#ifndef NO_BVH
                bvhRoot,
                s_bvhCache,
                bvhCachedNodesNum,
#endif
                lightsIndices,
                lightsNum,
                intersectionTri,
                r.origin,
                n
            );
            color = color + throughput * directLighting * ONE_OVER_PI;

            // Don't trace useless rays
            if (bounce == MAX_BOUNCES) break;
#endif
            
            // Intersect ray with geometry
            findClosestIntersection(r, tris, triNum,
#ifndef NO_BVH
                bvhRoot,
                s_bvhCache,
                bvhCachedNodesNum,
#endif
                &intersectionTri, &t
            );

            // if no intersection, break the loop, this sample is done
            if (intersectionTri == NULL) break;

            // Get the intersection material
            const Material* mat = mats + intersectionTri->materialIndex;

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
                throughput = throughput * (1.0f/p);
            }
        }

        //  This helps improve divergence, and thus performance in general
        __syncthreads();
    }

    // Write normalized color to framebuffer
    fb[pixelIndex] = (
        fb[pixelIndex] * (float)batch +
        color * (1.0f / (float)SAMPLES_PER_BATCH)
    ) * (1.0f / (float)(batch + 1));
}

#endif
