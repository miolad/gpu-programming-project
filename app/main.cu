#include <iostream>
#include <inttypes.h>

#include "helper.cuh"
#include "framebuffer.cuh"
#include "scene.cuh"
#include "utils.cuh"
#include "camera.cuh"
#include "cudaviewer.h"

#define RES_X 800
#define RES_Y 600
#define SCENE "scenes/cornell_box.obj"

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
__device__ void findClosestIntersection(Ray& r, Triangle* tris, uint32_t triNum, Triangle** intersectionTri, float* t) {
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
 * @param cam virtual camera
 * @param fb framebuffer of RES_X by RES_Y pixels to render to
 */
__global__ void rayTrace(Triangle* tris, Material* mats, uint32_t triNum, Camera cam, float3* fb) {
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Terminate pixels outside the framebuffer
    if (x >= RES_X || y >= RES_Y) return;

    // Get this pixel's ray (flipping the y axis)
    auto r = cam.getRayThroughPixel(make_int2(x, RES_Y - y - 1));

    // Trace the ray through the geometry
    Triangle* intersectionTri;
    float t;
    findClosestIntersection(r, tris, triNum, &intersectionTri, &t);

    // Get the triangle's material and output on the framebuffer
    float3 outColor;
    if (intersectionTri != NULL) {
        auto matAlbedo     = mats[intersectionTri->materialIndex].albedo;
        auto matEmissivity = mats[intersectionTri->materialIndex].emissivity;
        outColor = matAlbedo + matEmissivity;
    } else {
        outColor = {
            0.0, 0.0, 0.0
        };
    }

    uint32_t pixelIndex = x + y * RES_X;
    fb[pixelIndex] = outColor;
}

int main() {
    // Create a Vulkan shared framebuffer
    Framebuffer fb(make_int2(RES_X, RES_Y));
    
    // Load scene from obj
    Scene scene;
    if (!scene.load(SCENE, make_int2(RES_X, RES_Y))) {
        exit(EXIT_FAILURE);
    }

    std::cout << "Loaded scene (" << SCENE << "): " << scene.m_numTriangles << " triangles" << std::endl;

    // Initialize viewer
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    void* viewerCtx = viewer::init(fb.m_shareableHandle, fb.m_size, RES_X, RES_Y, (uint8_t*)&deviceProps.uuid.bytes);
    
    // Render main loop
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    
    while (true) {
        checkCudaErrors(cudaEventRecord(event));
        rayTrace<<<dim3(ceil(float(RES_X)/32.0), ceil(float(RES_Y)/32.0)), dim3(32, 32)>>>(
            scene.m_devTriangles,
            scene.m_devMaterials,
            scene.m_numTriangles,
            *scene.m_camera,
            fb.m_devPtr
        );

        // Run the viewer's event loop while we wait to resubmit the CUDA kernel
        bool shouldClose = false;
        while (!(shouldClose = viewer::run_event_loop(viewerCtx)) && cudaEventQuery(event) != cudaSuccess);

        if (shouldClose) break;
    }

    checkCudaErrors(cudaEventDestroy(event));

    // Deinit viewer
    viewer::deinit(viewerCtx);

    return EXIT_SUCCESS;
}
