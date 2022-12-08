#version 460

const vec2 FS_QUAD_VERTICES[] = {
    vec2(-1.0, -1.0), // Top left
    vec2( 1.0, -1.0), // Top right
    vec2( 1.0,  1.0), // Bottom right
    vec2(-1.0,  1.0)  // Bottom left
};

layout(location = 0) out float out_u;

void main() {
    vec2 vertex;

    switch (gl_VertexIndex) {
    case 0:
    case 3:
        vertex = FS_QUAD_VERTICES[0];
        out_u = 0.0;
        break;

    case 1:
    case 5:
        vertex = FS_QUAD_VERTICES[2];
        out_u = 1.0;
        break;

    case 2:
        vertex = FS_QUAD_VERTICES[1];
        out_u = 1.0;
        break;

    case 4:
        vertex = FS_QUAD_VERTICES[3];
        out_u = 0.0;
    }
    
    gl_Position = vec4(vertex, 0.0, 1.0);
}
