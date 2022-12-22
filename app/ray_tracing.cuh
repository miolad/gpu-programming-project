#ifndef _RAY_TRACING_CUH_
#define _RAY_TRACING_CUH_

#include "utils.cuh"

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
inline __device__ float rayTriangleIntersection(Ray& r, const Triangle& tri) {
    auto edge1 = tri.v2 - tri.v1;
    auto edge2 = tri.v3 - tri.v1;
    auto h = cross(r.direction, edge2);
    auto a = dot(edge1, h);

    if (a > -EPS && a < EPS)
        return -1.0f;

    auto f = 1.0f / a;
    auto s = r.origin - tri.v1;
    auto u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    auto q = cross(s, edge1);
    auto v = f * dot(r.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    auto t = f * dot(edge2, q);
    return t > EPS ? t : -1.0f;
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
inline __device__ void findClosestIntersection(Ray& r, const Triangle* tris, uint32_t triNum, Triangle** intersectionTri, float* t) {
    *t = RAY_MAX_T;
    *intersectionTri = NULL;
    
    // This unroll increases register pressure (thus hurting occupancy), but doesn't really show any performance variations,
    // neither positive nor negative
    // #pragma unroll 4
    for (uint32_t i = 0; i < triNum; ++i) {
        auto ti = rayTriangleIntersection(r, tris[i]);

        if (ti > 0.0f && ti < *t) {
            *t = ti;
            *intersectionTri = const_cast<Triangle*>(tris + i);
        }
    }
}

#ifndef NO_NEXT_EVENT_ESTIMATION
/**
 * Checks if a given ray hits a specific triangle first
 * 
 * @param r the ray
 * @param to the destination triangle
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @returns true if there is visibility between the two points, false otherwise
 */
inline __device__ bool visibility(Ray& r, const Triangle* to, const Triangle* tris, uint32_t triNum) {
    // Get the intersection point of the ray with the destination triangle
    auto maxT = rayTriangleIntersection(r, *to) - EPS;

    for (uint32_t i = 0; i < triNum; ++i) {
        auto ti = rayTriangleIntersection(r, tris[i]);

        if (ti > 0.0f && ti < maxT) return false;
    }

    return true;
}
#endif

#endif