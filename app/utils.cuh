#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <inttypes.h>

#define PI                          3.14159265359
/// @brief Horizontal resolution of the framebuffer
#define RES_X                       800
/// @brief Vertical resolution of the framebuffer
#define RES_Y                       600
/// @brief Scene to load
#define SCENE                       "scenes/cornell_box.obj"
/// @brief Approximation of zero, needed to deal with possible rounding errors in floating point arithmetic
#define EPS                         0.000001
/// @brief Maximum ray time. This clamps the maximum size of a scene in world units
#define RAY_MAX_T                   99999999.0
/// @brief Number of samples to compute for each invocation of the main kernel
#define SAMPLES_PER_BATCH           32
/// @brief Maximum number of indirect light bounces
#define MAX_BOUNCES                 8

/**
 * Triangle representation in device memory
 */
struct Triangle {
    /// @brief The vertices making up the triangle, in no particular order
    float3 v1, v2, v3;
    /// @brief Normalized triangle normal vector. Note that this is not guaranteed to point to the "outside" of the mesh
    float3 normal;
    /// @brief An index into the associated scene's materials array
    uint32_t materialIndex;
};

/**
 * Material data
 */
struct Material {
    float3 albedo, emissivity;
};

/**
 * A ray with an origin and a direction
 */
struct Ray {
    float3 origin, direction;
};

inline __host__ __device__ float clamp(float d, float min, float max) {
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return {
        a.x + b.x,
        a.y + b.y
    };
}

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return {
        a.x - b.x,
        a.y - b.y
    };
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

inline __host__ __device__ float2 operator*(float2 a, float b) {
    return {
        a.x * b,
        a.y * b
    };
}

inline __host__ __device__ float2 operator*(float a, float2 b) {
    return b * a;
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return {
        a.x * b,
        a.y * b,
        a.z * b
    };
}

inline __host__ __device__ float3 operator*(float a, float3 b) {
    return b * a;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b) {
    return {
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline __host__ __device__ float3 normalize(float3 a) {
    float invLen = rsqrtf(dot(a, a));
    return a * invLen;
}

#endif
