#include "DensityFieldOverlay2D.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>

#include "GpuRenderer.hpp"

namespace sim::renderer {

namespace {

float ClampUnitFloat(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

Color LerpColor(const Color& a, const Color& b, float t) {
    const float mixT = ClampUnitFloat(t);
    return Color{
        a.r + (b.r - a.r) * mixT,
        a.g + (b.g - a.g) * mixT,
        a.b + (b.b - a.b) * mixT
    };
}

PrimitiveVertex2D BuildOverlayVertex(
    const glm::vec2& position,
    const Color& color,
    const GpuRenderer::ObjectShaderParams& shaderParams
) {
    PrimitiveVertex2D vertex{};
    vertex.position[0] = position.x;
    vertex.position[1] = position.y;
    vertex.color[0] = ClampUnitFloat(color.r);
    vertex.color[1] = ClampUnitFloat(color.g);
    vertex.color[2] = ClampUnitFloat(color.b);

    vertex.shaderParams0[0] = std::max(0.0f, shaderParams.density);
    vertex.shaderParams0[1] = std::max(0.001f, shaderParams.quality);
    vertex.shaderParams0[2] = shaderParams.particleEnabled > 0.5f ? 1.0f : 0.0f;
    vertex.shaderParams0[3] = std::max(0.0f, shaderParams.smoothingRadius);
    vertex.shaderParams1[0] = shaderParams.particleCenter.x;
    vertex.shaderParams1[1] = shaderParams.particleCenter.y;
    vertex.shaderParams1[2] = std::max(0.0f, shaderParams.particleRadius);
    vertex.shaderParams1[3] = 0.0f;
    return vertex;
}

std::size_t NodeIndex(std::size_t row, std::size_t col, std::size_t nodeCols) {
    return row * nodeCols + col;
}

}  // namespace

Color EncodeDensityDeltaColor(float density, float targetDensity) {
    const float safeTarget = std::max(0.01f, targetDensity);
    const float delta = density - safeTarget;
    const float diffScale = safeTarget * 0.75f;
    const float normalizedDelta = ClampUnitFloat(std::abs(delta) / std::max(0.000001f, diffScale));
    const float intensity = std::pow(normalizedDelta, 0.75f);
    const Color lowBlue{0.05f, 0.62f, 1.0f};
    const Color white{1.0f, 1.0f, 1.0f};
    const Color highRed{1.0f, 0.1f, 0.1f};

    if (delta <= 0.0f) {
        return LerpColor(white, lowBlue, intensity);
    }

    return LerpColor(white, highRed, intensity);
}

void GpuRenderer::DrawDensityFieldOverlay2D(
    const DensityFieldGrid2D& grid,
    const DensityFieldOverlayStyle2D& style
) {
    if (grid.cols == 0 || grid.rows == 0) {
        return;
    }

    const std::size_t nodeCols = grid.cols + 1;
    const std::size_t nodeRows = grid.rows + 1;
    if (grid.nodeDensities.size() != nodeCols * nodeRows) {
        return;
    }

    const float sampleStep = std::max(1.0f, grid.sampleStep);
    const float width = std::max(0.0f, grid.width);
    const float height = std::max(0.0f, grid.height);
    if (width <= 0.0f || height <= 0.0f) {
        return;
    }

    DisableObjectParticle();
    SetObjectDensity(1.0f);
    SetObjectQuality(1.0f);
    const ObjectShaderParams overlayShaderParams = m_objectShaderParams;

    auto sampleX = [&](std::size_t col) {
        return grid.origin.x + std::min(width, static_cast<float>(col) * sampleStep);
    };
    auto sampleY = [&](std::size_t row) {
        return grid.origin.y + std::min(height, static_cast<float>(row) * sampleStep);
    };

    for (std::size_t row = 0; row < grid.rows; ++row) {
        const float y0 = sampleY(row);
        const float y1 = sampleY(row + 1);
        for (std::size_t col = 0; col < grid.cols; ++col) {
            const std::size_t fillStart = m_fillBatch.size();
            const std::size_t lineStart = m_lineBatch.size();

            const float x0 = sampleX(col);
            const float x1 = sampleX(col + 1);
            const glm::vec2 p00{x0, y0};
            const glm::vec2 p10{x1, y0};
            const glm::vec2 p01{x0, y1};
            const glm::vec2 p11{x1, y1};

            const float d00 = grid.nodeDensities[NodeIndex(row, col, nodeCols)];
            const float d10 = grid.nodeDensities[NodeIndex(row, col + 1, nodeCols)];
            const float d01 = grid.nodeDensities[NodeIndex(row + 1, col, nodeCols)];
            const float d11 = grid.nodeDensities[NodeIndex(row + 1, col + 1, nodeCols)];

            const Color c00 = EncodeDensityDeltaColor(d00, style.targetDensity);
            const Color c10 = EncodeDensityDeltaColor(d10, style.targetDensity);
            const Color c01 = EncodeDensityDeltaColor(d01, style.targetDensity);
            const Color c11 = EncodeDensityDeltaColor(d11, style.targetDensity);

            m_fillBatch.push_back(BuildOverlayVertex(p00, c00, overlayShaderParams));
            m_fillBatch.push_back(BuildOverlayVertex(p10, c10, overlayShaderParams));
            m_fillBatch.push_back(BuildOverlayVertex(p11, c11, overlayShaderParams));
            m_fillBatch.push_back(BuildOverlayVertex(p00, c00, overlayShaderParams));
            m_fillBatch.push_back(BuildOverlayVertex(p11, c11, overlayShaderParams));
            m_fillBatch.push_back(BuildOverlayVertex(p01, c01, overlayShaderParams));

            if (!ValidateBatchCapacity(fillStart, lineStart)) {
                return;
            }

            ++m_queuedPrimitiveCount;
        }
    }
}

}  // namespace sim::renderer
