#pragma once

#include <glm/vec2.hpp>

#include "DrawTypes2D.hpp"

namespace sim::runtime {

struct BallRenderComponent {
    float radius = 4.0f;
    float smoothingRadius = 18.0f;
    int segments = 24;
    sim::renderer::Color color{0.3f, 0.65f, 1.0f};
    bool enabled = true;
};

struct BoundingBoxRenderComponent {
    glm::vec2 size{220.0f, 120.0f};
    float lineThickness = 3.0f;
    sim::renderer::Color color{1.0f, 0.8f, 0.25f};
    bool enabled = true;
};

struct BallPhysicsComponent {
    glm::vec2 velocity{0.0f, 0.0f};
    glm::vec2 lastAnchorCenter{0.0f, 0.0f};
    glm::vec2 lastUiOffset{0.0f, 0.0f};
    float density = 0.0f;
    float pressure = 0.0f;
    float normalizedDensity = 0.0f;
    bool anchorInitialized = false;
};

}  // namespace sim::runtime
