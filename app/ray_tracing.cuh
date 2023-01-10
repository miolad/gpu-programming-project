#ifndef _RAY_TRACING_CUH_
#define _RAY_TRACING_CUH_

#include "utils.cuh"

#ifndef NO_BVH
#include "bvh.cuh"

#define GET_BVH_NODE(sRoot, gRoot, numCached, index) ((index < numCached ? sRoot : gRoot) + index) 
#endif

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

    auto f = 1.0f / a;
    auto s = r.origin - tri.v1;
    auto u = f * dot(s, h);

    auto q = cross(s, edge1);
    auto v = f * dot(r.direction, q);

    auto t = f * dot(edge2, q);
    return (a < -EPS || a > EPS) && (u >= 0.0f && u <= 1.0f) && (v >= 0.0f && u + v <= 1.0f) && (t > EPS) ? t : -1.0f;
}

#ifndef NO_BVH
/**
 * Computes the intersection between the provided ray and AABB.
 * This function implements the "slab" method (https://tavianator.com/2011/ray_box.html)
 * 
 * @param r the ray
 * @param aabb AABB to intersect
 * @returns the "time" of the ray's intersection with the AABB,
 *          or -1.0 if no intersection is found.
 */
inline __device__ float rayAABBIntersection(Ray& r, const AABB& aabb) {
    float3 rayInvDirection = {
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    };
    float tmin = 0.0f, tmax = RAY_MAX_T;

    auto t1 = (aabb.lo.x - r.origin.x) * rayInvDirection.x;
    auto t2 = (aabb.hi.x - r.origin.x) * rayInvDirection.x;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    t1 = (aabb.lo.y - r.origin.y) * rayInvDirection.y;
    t2 = (aabb.hi.y - r.origin.y) * rayInvDirection.y;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    t1 = (aabb.lo.z - r.origin.z) * rayInvDirection.z;
    t2 = (aabb.hi.z - r.origin.z) * rayInvDirection.z;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    return tmin <= tmax ? tmin : -1.0f;
}

/**
 * Find the closest intersection between a ray and a list of triangles by traversing the BVH acceleration structure.
 * This is based on the code in https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
 * 
 * @param r the ray
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @param from triangle the ray is originating from, or NULL
 * @param bvhRoot root node of the BVH
 * @param bvhCache cache of the BVH in shared memory
 * @param bvhCachedNodesNum number of nodes of the BVH in the shared memory cache
 * @param intersectionTri (out) pointer to the closest triangle intersected, or NULL if no intersection was found
 * @param t (out) "time" of the ray to the closest intersection. This is invalid if intersectionTri is NULL
 */
inline __device__ void findClosestIntersection(
    Ray& r,
    const Triangle* tris,
    uint32_t triNum,
    const Triangle* from,
    const Node* bvhRoot,
    const Node* bvhCache,
    uint32_t bvhCachedNodesNum,
    Triangle** intersectionTri,
    float* t
) {
    *t = RAY_MAX_T;
    *intersectionTri = NULL;
    
    const Node* stack[64];
    const Node** stackPtr = stack;
    *stackPtr++ = NULL;

    const Node* node = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, 0);
    do {
        // Note that this breaks if the root is a leaf node, i.e. if the scene has only one triangle
        auto childL = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, node->node.internal.left);
        auto childR = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, node->node.internal.left + 1);

        auto aabbIntersectionTL = rayAABBIntersection(r, childL->aabb);
        auto aabbIntersectionTR = rayAABBIntersection(r, childR->aabb);

        auto aabbIntersectionL = aabbIntersectionTL > -0.5f && aabbIntersectionTL < *t;
        if (aabbIntersectionL && childL->type == LEAF_NODE) {
            auto tri = tris + childL->node.leaf.triangleIndex;
            auto ti = from != tri ? rayTriangleIntersection(r, *tri) : -1.0f;
            if (ti > 0.0f && ti < *t) {
                *t = ti;
                *intersectionTri = const_cast<Triangle*>(tris + childL->node.leaf.triangleIndex);
            }
        }
        auto aabbIntersectionR = aabbIntersectionTR > -0.5f && aabbIntersectionTR < *t;
        if (aabbIntersectionR && childR->type == LEAF_NODE) {
            auto tri = tris + childR->node.leaf.triangleIndex;
            auto ti = from != tri ? rayTriangleIntersection(r, *tri) : -1.0f;
            if (ti > 0.0f && ti < *t) {
                *t = ti;
                *intersectionTri = const_cast<Triangle*>(tris + childR->node.leaf.triangleIndex);
            }
        }

        auto traverseL = aabbIntersectionL && childL->type == INTERNAL_NODE;
        auto traverseR = aabbIntersectionR && childR->type == INTERNAL_NODE;

        if (!traverseL && !traverseR) {
            node = *--stackPtr;
        } else {
            node = traverseL ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR;
        }
    } while (node != NULL);
}
#else
/**
 * Find the closest intersection between a ray and a list of triangles by iteratively testing the
 * intersection between the ray and each triangle
 * 
 * @param r the ray
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @param from triangle the ray is originating from, or NULL
 * @param intersectionTri (out) pointer to the closest triangle intersected, or NULL if no intersection was found
 * @param t (out) "time" of the ray to the closest intersection. This is invalid if intersectionTri is NULL
 */
inline __device__ void findClosestIntersection(
    Ray& r,
    const Triangle* tris,
    uint32_t triNum,
    const Triangle* from,
    Triangle** intersectionTri,
    float* t
) {
    *t = RAY_MAX_T;
    *intersectionTri = NULL;
    
    // This unroll increases register pressure (thus hurting occupancy), but doesn't really show any performance variations,
    // neither positive nor negative
    // #pragma unroll 4
    for (uint32_t i = 0; i < triNum; ++i) {
        auto ti = from != tris + i ? rayTriangleIntersection(r, tris[i]) : -1.0f;

        if (ti > 0.0f && ti < *t) {
            *t = ti;
            *intersectionTri = const_cast<Triangle*>(tris + i);
        }
    }
}
#endif

#ifndef NO_NEXT_EVENT_ESTIMATION
#ifndef NO_BVH
/**
 * Checks if a given ray hits a specific triangle first
 * 
 * @param r the ray
 * @param from triangle the ray is originating from, or NULL
 * @param to the destination triangle
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @param bvhRoot root node of the BVH
 * @param bvhCache cache of the BVH in shared memory
 * @param bvhCachedNodesNum number of nodes of the BVH in the shared memory cache
 * @returns true if there is visibility between the two points, false otherwise
 */
inline __device__ bool visibility(
    Ray& r,
    const Triangle* from,
    const Triangle* to,
    const Triangle* tris,
    uint32_t triNum,
    const Node* bvhRoot,
    const Node* bvhCache,
    uint32_t bvhCachedNodesNum
) {
    // Get the intersection point of the ray with the destination triangle
    auto maxT = rayTriangleIntersection(r, *to);
    
    const Node* stack[64];
    const Node** stackPtr = stack;
    *stackPtr++ = NULL;

    const Node* node = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, 0);
    do {
        // Note that this breaks if the root is a leaf node, i.e. if the scene has only one triangle
        auto childL = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, node->node.internal.left);
        auto childR = GET_BVH_NODE(bvhCache, bvhRoot, bvhCachedNodesNum, node->node.internal.left + 1);

        auto aabbIntersectionTL = rayAABBIntersection(r, childL->aabb);
        auto aabbIntersectionTR = rayAABBIntersection(r, childR->aabb);

        auto aabbIntersectionL = aabbIntersectionTL > -0.5f && aabbIntersectionTL < maxT;
        if (aabbIntersectionL && childL->type == LEAF_NODE) {
            auto tri = tris + childL->node.leaf.triangleIndex;
            auto ti = from != tri && to != tri ?
                rayTriangleIntersection(r, tris[childL->node.leaf.triangleIndex]) : -1.0f;
            if (ti > 0.0f && ti < maxT) return false;
        }
        auto aabbIntersectionR = aabbIntersectionTR > -0.5f && aabbIntersectionTR < maxT;
        if (aabbIntersectionR && childR->type == LEAF_NODE) {
            auto tri = tris + childR->node.leaf.triangleIndex;
            auto ti = from != tri && to != tri ?
                rayTriangleIntersection(r, tris[childR->node.leaf.triangleIndex]) : -1.0f;
            if (ti > 0.0f && ti < maxT) return false;
        }

        auto traverseL = aabbIntersectionL && childL->type == INTERNAL_NODE;
        auto traverseR = aabbIntersectionR && childR->type == INTERNAL_NODE;

        if (!traverseL && !traverseR) {
            node = *--stackPtr;
        } else {
            node = traverseL ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR;
        }
    } while (node != NULL);

    return true;
}
#else
/**
 * Checks if a given ray hits a specific triangle first
 * 
 * @param r the ray
 * @param from triangle the ray is originating from, or NULL
 * @param to the destination triangle
 * @param tris list of all the triangles
 * @param triNum number of triangles pointed to by `tris`
 * @returns true if there is visibility between the two points, false otherwise
 */
inline __device__ bool visibility(Ray& r, const Triangle* from, const Triangle* to, const Triangle* tris, uint32_t triNum) {
    // Get the intersection point of the ray with the destination triangle
    auto maxT = rayTriangleIntersection(r, *to);

    for (uint32_t i = 0; i < triNum; ++i) {
        auto ti = from != tris + i && to != tris + i ?
            rayTriangleIntersection(r, tris[i]) : -1.0f;

        if (ti > 0.0f && ti < maxT) return false;
    }

    return true;
}
#endif
#endif

#endif
