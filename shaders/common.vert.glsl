#version 450

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec4 a_object_params0;
layout(location = 3) in vec4 a_object_params1;

layout(set = 1, binding = 0, std140) uniform TransformUBO {
    mat4 u_mvp;
};

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_world_position;
layout(location = 2) flat out vec4 v_object_params0;
layout(location = 3) flat out vec4 v_object_params1;
void main()
{
    gl_Position = u_mvp * vec4(a_position, 0.0, 1.0);
    v_color = a_color;
    v_world_position = a_position;
    v_object_params0 = a_object_params0;
    v_object_params1 = a_object_params1;
}
