#include "PrimitiveBuilder2D.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>

namespace sim::renderer {

namespace {

constexpr int kMinCircleSegments = 3;
constexpr float kMinAngleDelta = 1.0e-6f;
constexpr float kMinLineLengthSquared = 1.0e-8f;

PrimitiveColor2D ClampColor(PrimitiveColor2D color) {
    color.r = std::clamp(color.r, 0.0f, 1.0f);
    color.g = std::clamp(color.g, 0.0f, 1.0f);
    color.b = std::clamp(color.b, 0.0f, 1.0f);
    return color;
}

PrimitiveShaderParams2D ClampShaderParams(PrimitiveShaderParams2D shaderParams) {
    shaderParams.density = std::max(0.0f, shaderParams.density);
    shaderParams.quality = std::max(0.001f, shaderParams.quality);
    shaderParams.particleEnabled = shaderParams.particleEnabled > 0.5f ? 1.0f : 0.0f;
    shaderParams.smoothingRadius = std::max(0.0f, shaderParams.smoothingRadius);
    shaderParams.particleRadius = std::max(0.0f, shaderParams.particleRadius);
    return shaderParams;
}

PrimitiveVertex2D MakeVertex(const glm::vec2& p, PrimitiveColor2D color, PrimitiveShaderParams2D shaderParams) {
    shaderParams = ClampShaderParams(shaderParams);
    PrimitiveVertex2D v{};
    v.position[0] = p.x;
    v.position[1] = p.y;
    v.color[0] = color.r;
    v.color[1] = color.g;
    v.color[2] = color.b;
    v.shaderParams0[0] = shaderParams.density;
    v.shaderParams0[1] = shaderParams.quality;
    v.shaderParams0[2] = shaderParams.particleEnabled;
    v.shaderParams0[3] = shaderParams.smoothingRadius;
    v.shaderParams1[0] = shaderParams.particleCenter[0];
    v.shaderParams1[1] = shaderParams.particleCenter[1];
    v.shaderParams1[2] = shaderParams.particleRadius;
    v.shaderParams1[3] = shaderParams.reserved0;
    return v;
}

glm::vec2 RotatePoint(const glm::vec2& point, float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return glm::vec2(point.x * c - point.y * s, point.x * s + point.y * c);
}

}  // namespace

bool PrimitiveBuilder2D::AppendLine(
    const glm::vec2& start,
    const glm::vec2& end,
    PrimitiveColor2D color,
    PrimitiveShaderParams2D shaderParams,
    float thickness,
    std::vector<PrimitiveVertex2D>& ioFillVertices,
    std::vector<PrimitiveVertex2D>& ioLineVertices
) {
    color = ClampColor(color);
    shaderParams = ClampShaderParams(shaderParams);

    if (thickness <= 0.0f) {
        ioLineVertices.push_back(MakeVertex(start, color, shaderParams));
        ioLineVertices.push_back(MakeVertex(end, color, shaderParams));
        return true;
    }

    const glm::vec2 delta = end - start;
    const float lengthSquared = delta.x * delta.x + delta.y * delta.y;
    if (lengthSquared <= kMinLineLengthSquared) {
        return false;
    }

    const float invLength = 1.0f / std::sqrt(lengthSquared);
    const glm::vec2 normal = glm::vec2(-delta.y * invLength, delta.x * invLength);
    const glm::vec2 offset = normal * (0.5f * thickness);

    const glm::vec2 p0 = start + offset;
    const glm::vec2 p1 = end + offset;
    const glm::vec2 p2 = end - offset;
    const glm::vec2 p3 = start - offset;

    ioFillVertices.push_back(MakeVertex(p0, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p1, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p2, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p0, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p2, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p3, color, shaderParams));
    return true;
}

bool PrimitiveBuilder2D::AppendTriangle(
    const glm::vec2& p0,
    const glm::vec2& p1,
    const glm::vec2& p2,
    PrimitiveColor2D color,
    PrimitiveShaderParams2D shaderParams,
    bool wireframe,
    float lineThickness,
    std::vector<PrimitiveVertex2D>& ioFillVertices,
    std::vector<PrimitiveVertex2D>& ioLineVertices
) {
    color = ClampColor(color);
    shaderParams = ClampShaderParams(shaderParams);

    if (wireframe) {
        return AppendLine(p0, p1, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)
            && AppendLine(p1, p2, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)
            && AppendLine(p2, p0, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices);
    }

    ioFillVertices.push_back(MakeVertex(p0, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p1, color, shaderParams));
    ioFillVertices.push_back(MakeVertex(p2, color, shaderParams));
    return true;
}

bool PrimitiveBuilder2D::AppendRect(
    const glm::vec2& center,
    const glm::vec2& size,
    float rotationRadians,
    PrimitiveColor2D color,
    PrimitiveShaderParams2D shaderParams,
    bool wireframe,
    float lineThickness,
    std::vector<PrimitiveVertex2D>& ioFillVertices,
    std::vector<PrimitiveVertex2D>& ioLineVertices
) {
    const glm::vec2 halfSize = size * 0.5f;
    glm::vec2 corners[4] = {
        {-halfSize.x, -halfSize.y},
        {halfSize.x, -halfSize.y},
        {halfSize.x, halfSize.y},
        {-halfSize.x, halfSize.y},
    };

    for (glm::vec2& corner : corners) {
        corner = RotatePoint(corner, rotationRadians) + center;
    }

    if (wireframe) {
        return AppendLine(corners[0], corners[1], color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)
            && AppendLine(corners[1], corners[2], color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)
            && AppendLine(corners[2], corners[3], color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)
            && AppendLine(corners[3], corners[0], color, shaderParams, lineThickness, ioFillVertices, ioLineVertices);
    }

    return AppendTriangle(corners[0], corners[1], corners[2], color, shaderParams, false, 0.0f, ioFillVertices, ioLineVertices)
        && AppendTriangle(corners[0], corners[2], corners[3], color, shaderParams, false, 0.0f, ioFillVertices, ioLineVertices);
}

bool PrimitiveBuilder2D::AppendCircle(
    const glm::vec2& center,
    float radius,
    int segments,
    PrimitiveColor2D color,
    PrimitiveShaderParams2D shaderParams,
    bool wireframe,
    float lineThickness,
    std::vector<PrimitiveVertex2D>& ioFillVertices,
    std::vector<PrimitiveVertex2D>& ioLineVertices
) {
    if (radius <= 0.0f) {
        return false;
    }

    color = ClampColor(color);
    shaderParams = ClampShaderParams(shaderParams);
    const int segmentCount = std::max(segments, kMinCircleSegments);
    const float angleStep = glm::two_pi<float>() / static_cast<float>(segmentCount);
    const float stepCos = std::cos(angleStep);
    const float stepSin = std::sin(angleStep);

    glm::vec2 unitCurrent{1.0f, 0.0f};
    if (wireframe) {
        bool submitted = true;
        for (int i = 0; i < segmentCount; ++i) {
            const glm::vec2 unitNext{
                unitCurrent.x * stepCos - unitCurrent.y * stepSin,
                unitCurrent.x * stepSin + unitCurrent.y * stepCos
            };
            const glm::vec2 p0 = center + unitCurrent * radius;
            const glm::vec2 p1 = center + unitNext * radius;
            if (!AppendLine(p0, p1, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)) {
                submitted = false;
                break;
            }
            unitCurrent = unitNext;
        }
        return submitted;
    }

    const PrimitiveVertex2D centerVertex = MakeVertex(center, color, shaderParams);
    for (int i = 0; i < segmentCount; ++i) {
        const glm::vec2 unitNext{
            unitCurrent.x * stepCos - unitCurrent.y * stepSin,
            unitCurrent.x * stepSin + unitCurrent.y * stepCos
        };
        const glm::vec2 p0 = center + unitCurrent * radius;
        const glm::vec2 p1 = center + unitNext * radius;
        ioFillVertices.push_back(centerVertex);
        ioFillVertices.push_back(MakeVertex(p0, color, shaderParams));
        ioFillVertices.push_back(MakeVertex(p1, color, shaderParams));
        unitCurrent = unitNext;
    }
    return true;
}

bool PrimitiveBuilder2D::AppendPolygon(
    const std::vector<glm::vec2>& points,
    PrimitiveColor2D color,
    PrimitiveShaderParams2D shaderParams,
    bool wireframe,
    float lineThickness,
    std::vector<PrimitiveVertex2D>& ioFillVertices,
    std::vector<PrimitiveVertex2D>& ioLineVertices
) {
    if (points.size() < 2) {
        return false;
    }

    if (wireframe) {
        if (points.size() == 2) {
            return AppendLine(points[0], points[1], color, shaderParams, lineThickness, ioFillVertices, ioLineVertices);
        }

        bool submitted = true;
        const size_t pointCount = points.size();
        for (size_t i = 0; i < pointCount; ++i) {
            const glm::vec2& p0 = points[i];
            const glm::vec2& p1 = points[(i + 1) % pointCount];
            if (!AppendLine(p0, p1, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)) {
                submitted = false;
                break;
            }
        }
        return submitted;
    }

    if (points.size() < 3) {
        return false;
    }

    bool submitted = true;
    const glm::vec2& anchor = points[0];
    for (size_t i = 1; i + 1 < points.size(); ++i) {
        if (!AppendTriangle(anchor, points[i], points[i + 1], color, shaderParams, false, 0.0f, ioFillVertices, ioLineVertices)) {
            submitted = false;
            break;
        }
    }
    return submitted;
}

bool PrimitiveBuilder2D::AppendArc(
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
) {
    if (radius <= 0.0f) {
        return false;
    }

    const float angleDelta = endRadians - startRadians;
    if (std::abs(angleDelta) <= kMinAngleDelta) {
        return false;
    }

    const int segmentCount = wireframe ? std::max(segments, 1) : std::max(segments, 2);
    auto pointOnArc = [&](float angleRadians) {
        return center + glm::vec2(std::cos(angleRadians), std::sin(angleRadians)) * radius;
    };

    bool submitted = true;
    glm::vec2 previous = pointOnArc(startRadians);
    for (int i = 1; i <= segmentCount; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segmentCount);
        const float angle = startRadians + angleDelta * t;
        const glm::vec2 current = pointOnArc(angle);
        if (wireframe) {
            if (!AppendLine(previous, current, color, shaderParams, lineThickness, ioFillVertices, ioLineVertices)) {
                submitted = false;
                break;
            }
        } else if (!AppendTriangle(center, previous, current, color, shaderParams, false, 0.0f, ioFillVertices, ioLineVertices)) {
            submitted = false;
            break;
        }
        previous = current;
    }
    return submitted;
}

}  // namespace sim::renderer
