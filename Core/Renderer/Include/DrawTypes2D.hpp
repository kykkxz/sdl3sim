#pragma once

#include <glm/vec2.hpp>
#include <vector>

namespace sim::renderer {

struct Color {
    float r;
    float g;
    float b;

    constexpr Color()
        : r(1.0f), g(1.0f), b(1.0f) {
    }

    constexpr Color(float inR, float inG, float inB)
        : r(inR), g(inG), b(inB) {
    }
};

struct Triangle2D {
    glm::vec2 p0{0.0f, 0.5f};
    glm::vec2 p1{0.5f, -0.5f};
    glm::vec2 p2{-0.5f, -0.5f};
    Color color = Color();
    bool wireframe = false;
    float lineThickness = 0.0f;
};

struct Rect2D {
    glm::vec2 center{0.0f, 0.0f};
    glm::vec2 size{1.0f, 1.0f};
    float rotationRadians = 0.0f;
    Color color = Color();
    bool wireframe = false;
    float lineThickness = 0.0f;
};

struct Circle2D {
    glm::vec2 center{0.0f, 0.0f};
    float radius = 0.5f;
    int segments = 32;
    Color color = Color();
    bool wireframe = false;
    float lineThickness = 0.0f;
};

struct Polygon2D {
    std::vector<glm::vec2> points;
    Color color = Color();
    bool wireframe = false;
    float lineThickness = 0.0f;
};

struct Arc2D {
    glm::vec2 center{0.0f, 0.0f};
    float radius = 0.5f;
    float startRadians = 0.0f;
    float endRadians = 3.1415926535f;
    int segments = 32;
    Color color = Color();
    bool wireframe = true;
    float lineThickness = 0.0f;
};

}  // namespace sim::renderer
