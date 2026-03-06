#pragma once

#include <cstdint>
#include <glm/vec2.hpp>

namespace sim::ecs {

using EntityId = std::uint64_t;

struct IdComponent {
    explicit IdComponent(EntityId inValue = 0) : value(inValue) {}

    EntityId value = 0;
};

struct TransformComponent {
    TransformComponent() = default;
    TransformComponent(
        glm::vec2 inPosition,
        float inRotationRadians = 0.0f,
        glm::vec2 inScale = glm::vec2(1.0f, 1.0f)
    )
        : position(inPosition), rotationRadians(inRotationRadians), scale(inScale) {}

    glm::vec2 position{0.0f, 0.0f};
    float rotationRadians = 0.0f;
    glm::vec2 scale{1.0f, 1.0f};
};

}  // namespace sim::ecs
