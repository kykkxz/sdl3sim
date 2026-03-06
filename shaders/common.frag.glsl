#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_world_position;
layout(location = 2) flat in vec4 v_object_params0;
layout(location = 3) flat in vec4 v_object_params1;

layout(set = 3, binding = 0, std140) uniform GlobalParamsUBO {
    // x=density, y=quality, z/w reserved
    vec4 u_globalParams0;
    // x=viewportWidth, y=viewportHeight, z=timeSeconds, w=frameIndex
    vec4 u_globalParams1;
};

layout(location = 0) out vec4 Color;

void main()
{
    float density = max(u_globalParams0.x * v_object_params0.x, 0.0);
    float quality = max(u_globalParams0.y * v_object_params0.y, 0.001);
    vec3 shaded = clamp(v_color * density, vec3(0.0), vec3(1.0));
    shaded = pow(shaded, vec3(1.0 / quality));

    float particleEnabled = v_object_params0.z;
    if (particleEnabled > 0.5) {
        float smoothingRadius = max(v_object_params0.w, 0.001);
        vec2 center = v_object_params1.xy;
        float particleRadius = max(v_object_params1.z, 0.001);
        float dist = length(v_world_position - center);

        if (dist > smoothingRadius) {
            discard;
        }

        float falloff = 1.0 - smoothstep(particleRadius, smoothingRadius, dist);
        float core = 1.0 - smoothstep(0.0, particleRadius, dist);
        float blend = max(falloff, core);
        Color = vec4(shaded * blend, blend);
        return;
    }

    Color = vec4(shaded, 1.0);
}
