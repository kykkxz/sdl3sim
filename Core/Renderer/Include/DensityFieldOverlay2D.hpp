#pragma once

#include <cstddef>
#include <vector>

#include <glm/vec2.hpp>

#include "DrawTypes2D.hpp"

namespace sim::renderer {

struct DensityFieldGrid2D {
    // Node density values laid out in row-major order.
    // Node dimensions are (cols + 1) x (rows + 1).
    std::vector<float> nodeDensities;
    std::size_t cols = 0;  // Cell columns.
    std::size_t rows = 0;  // Cell rows.
    float sampleStep = 10.0f;
    float width = 0.0f;
    float height = 0.0f;
    glm::vec2 origin{0.0f, 0.0f};
};

struct DensityFieldOverlayStyle2D {
    float targetDensity = 1.000f;
};

class GpuRenderer;

Color EncodeDensityDeltaColor(float density, float targetDensity);

}  // namespace sim::renderer
