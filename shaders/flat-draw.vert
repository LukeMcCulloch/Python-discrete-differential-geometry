#version 150

uniform mat4 u_viewMatrix, u_projMatrix;
uniform float u_depthOffset;
uniform vec3 u_dataCenter;

in vec3 a_position;
in vec3 a_color;

out vec3 color;
out vec3 position;

void main()
{
    gl_PointSize = 6.0;
    gl_Position = u_projMatrix * u_viewMatrix * vec4(a_position - u_dataCenter,1.0) - vec4(0.0, 0.0, u_depthOffset, 0.0);

    color = a_color;
    position = a_position;
}
