#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <inttypes.h>

/**
 * RGB Color with generic component types and 4 byte alignment
 */
template<typename T>
struct __align__(4) RGBColor {
    T r, g, b;
};

/**
 * Position in the 3D space with 32 bit float components
 */
struct Vertex {
    float x, y, z;
};

/**
 * 3 vertices plus material index
 */
struct Triangle {
    /// @brief The vertices of making up the triangle, in no particular order
    Vertex v1, v2, v3;
    /// @brief An index into the associated scene's materials array
    uint32_t materialIndex;
};

/**
 * Material data
 */
struct Material {
    RGBColor<float> albedo, emissivity;
};

#endif
