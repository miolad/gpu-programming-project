#version 460

#extension GL_EXT_buffer_reference2 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Pixels {
    uint pixel;
};
layout(push_constant, std430) uniform PushConstants {
    Pixels pixels;
    uint framebuffer_size_x;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    // Get the pixel's index
    uint pixel_index = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * pc.framebuffer_size_x;

    // Get the color data
    uint pixel_color = pc.pixels[pixel_index].pixel;

    // Separate channels
    out_color = vec4(
        vec3(
            (pixel_color >>  0) & 0xFF,
            (pixel_color >>  8) & 0xFF,
            (pixel_color >> 16) & 0xFF
        ) / 255.0,
        1.0
    );
}
