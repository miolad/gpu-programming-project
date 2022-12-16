#ifndef _CAMERA_CUH_
#define _CAMERA_CUH_

#include "utils.cuh"

/**
 * Simple fixed virtual camera
 */
class Camera {
private:
    /// @brief Camera's position in the 3D world
    float3 m_position;
    /// @brief Camera's normalized view direction
    float3 m_viewDir;
    /// @brief Camera's normalized right direction
    float3 m_right;
    /// @brief Camera's normalized up direction
    float3 m_up;
    /// @brief Framebuffer's half resolution
    float2 m_halfResolution;
    /// @brief Size of a pixel in world space
    float m_pixelSize;
    
public:
    /**
     * @param resolution the full framebuffer resolution
     * @param position the camera's world space position
     * @param upDirection up direction
     * @param viewDirection direction the camera is pointed at
     * @param hfov horizontal field of view in degrees. The vertical one is inferred from the resolution, considering the pixels square
     */
    Camera(int2 resolution, float3 position, float3 upDirection, float3 viewDirection, float hfov) {
        auto normalizedUp = normalize(upDirection);
        
        m_position = position;
        m_viewDir = normalize(viewDirection);
        m_right = normalize(cross(m_viewDir, normalizedUp));
        m_up = normalize(cross(m_right, m_viewDir));
        m_halfResolution = make_float2((float)resolution.x, (float)resolution.y) * 0.5;
        m_pixelSize = tanf(hfov * PI / 360.0) / m_halfResolution.x;
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
