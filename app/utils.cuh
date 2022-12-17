#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <inttypes.h>

#define PI  3.14159265359

/// @brief Approximation of zero, needed to deal with possible rounding errors in floating point arithmetic
#define EPS 0.000001

/// @brief Maximum ray time. This clamps the maximum size of a scene in world units
#define RAY_MAX_T 99999999.0

/**
 * RGB Color with generic component types and 4 byte alignment
 */
template<typename T>
struct __align__(4) RGBColor {
    T r, g, b;
};

/**
 * 3 vertices plus material index
 */
struct Triangle {
    /// @brief The vertices making up the triangle, in no particular order
    float3 v1, v2, v3;
    /// @brief An index into the associated scene's materials array
    uint32_t materialIndex;
};

/**
 * Material data
 */
struct Material {
    RGBColor<float> albedo, emissivity;
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
