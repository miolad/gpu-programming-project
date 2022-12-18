#version 460

#extension GL_EXT_buffer_reference2 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Pixels {
    float r;
    float g;
    float b;
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
    Pixels pixel = pc.pixels[pixel_index];

    // Separate channels
    out_color = vec4(
        pixel.r,
        pixel.g,
        pixel.b,
        1.0
    );
}
