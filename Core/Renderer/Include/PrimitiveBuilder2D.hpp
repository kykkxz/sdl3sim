#pragma once

#include <glm/vec2.hpp>
#include <vector>

namespace sim::renderer {

struct PrimitiveColor2D {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

struct PrimitiveShaderParams2D {
    float density = 1.0f;
    float quality = 1.0f;
    float particleEnabled = 0.0f;
    float smoothingRadius = 0.0f;
    float particleCenter[2]{0.0f, 0.0f};
    float particleRadius = 0.0f;
    float reserved0 = 0.0f;
};

struct PrimitiveVertex2D {
    float position[2];
    float color[3];
    float shaderParams0[4];
    float shaderParams1[4];
};

class PrimitiveBuilder2D {
public:
    static bool AppendLine(
        const glm::vec2& start,
        const glm::vec2& end,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        float thickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );

    static bool AppendTriangle(
        const glm::vec2& p0,
        const glm::vec2& p1,
        const glm::vec2& p2,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        bool wireframe,
        float lineThickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );

    static bool AppendRect(
        const glm::vec2& center,
        const glm::vec2& size,
        float rotationRadians,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        bool wireframe,
        float lineThickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );

    static bool AppendCircle(
        const glm::vec2& center,
        float radius,
        int segments,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        bool wireframe,
        float lineThickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );

    static bool AppendPolygon(
        const std::vector<glm::vec2>& points,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        bool wireframe,
        float lineThickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );

    static bool AppendArc(
        const glm::vec2& center,
        float radius,
        float startRadians,
        float endRadians,
        int segments,
        PrimitiveColor2D color,
        PrimitiveShaderParams2D shaderParams,
        bool wireframe,
        float lineThickness,
        std::vector<PrimitiveVertex2D>& ioFillVertices,
        std::vector<PrimitiveVertex2D>& ioLineVertices
    );
};

}  // namespace sim::renderer
