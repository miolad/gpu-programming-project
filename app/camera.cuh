#ifndef _CAMERA_CUH_
#define _CAMERA_CUH_

#include "utils.cuh"

/**
 * Simple fixed virtual camera
 */
class Camera {
private:
    /// @brief Camera's Field Of View in radians
    const float m_fov = 75.0 * PI / 180.0;
    /// @brief Camera's position in the 3D world
    const float3 m_position = make_float3(0.0, 0.0, 0.0);
    /// @brief Camera's normalized up direction
    const float3 m_up = make_float3(0.0, 1.0, 0.0);
    /// @brief Camera's normalized view direction
    const float3 m_viewDir = make_float3(0.0, 0.0, -1.0);
    /// @brief Camera's normalized right direction
    const float3 m_right = normalize(cross(m_viewDir, m_up));
    
    /// @brief Framebuffer's half resolution
    float2 m_halfResolution;
    /// @brief Size of a pixel in world space
    float m_pixelSize;
    
public:
    Camera(int2 resolution) {
        m_halfResolution = make_float2((float)resolution.x, (float)resolution.y) * 0.5;
        m_pixelSize = tanf(m_fov/2.0) / m_halfResolution.x;
    }

    /**
     * Generate a ray from the camera position to the specified screen's pixel
     * 
     * @param pixel the pixel through to generate the ray for
     * @returns a ray from the camera position to the middle of the specified pixel
     */
    __device__ inline Ray getRayThroughPixel(int2 pixel) {
        // Pixel's offset relative to the center of the screen
        float2 pixelOffset = make_float2((float)pixel.x, (float)pixel.y) + make_float2(0.5, 0.5) - m_halfResolution;
        float3 rayDir = normalize(m_viewDir                             +
                                  m_right * m_pixelSize * pixelOffset.x +
                                  m_up    * m_pixelSize * pixelOffset.y  );
        return {
            m_position,
            rayDir
        };
    }
};

#endif
