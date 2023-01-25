#ifndef _CAMERA_CUH_
#define _CAMERA_CUH_

#include "utils.cuh"

/**
 * Simple virtual camera with supersampling antialiasing
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
        m_halfResolution = make_float2((float)resolution.x, (float)resolution.y) * 0.5f;
        m_pixelSize = tanf(hfov * PI / 360.0f) / m_halfResolution.x;
    }

    /**
     * Generate a ray from the camera position to the specified screen's pixel with 4xSSAA
     * 
     * @param pixel the pixel through to generate the ray for
     * @param superSample sample index in [0, 16)
     * @returns a ray from the camera position to the middle of the specified pixel
     */
    inline __device__ Ray getRayThroughPixel(int2 pixel, uint32_t superSample) {
        float2 superSampleOffset = (make_float2((float)(superSample % 4), (float)(superSample / 4)) - make_float2(1.5f, 1.5f)) * 0.25f;
        
        // Pixel's offset relative to the center of the screen
        float2 pixelOffset = make_float2((float)pixel.x, (float)pixel.y) + make_float2(0.5f, 0.5f) - m_halfResolution;
        float3 rayDir = normalize(m_viewDir                              +
                                  m_right * m_pixelSize * (pixelOffset.x + superSampleOffset.x) +
                                  m_up    * m_pixelSize * (pixelOffset.y + superSampleOffset.y)  );
        return {
            m_position,
            rayDir
        };
    }
};

#endif
