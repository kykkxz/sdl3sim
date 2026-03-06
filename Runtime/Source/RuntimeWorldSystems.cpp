#include "RuntimeWorldSystems.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include "RuntimeCuda.hpp"
#include "RuntimeProfiling.hpp"

namespace sim::runtime {

namespace {

constexpr float kDensityFloorRatio = 0.1f;
constexpr float kMaximumNegativePressureRatio = 0.25f;
constexpr float kMaximumPressureToStiffnessRatio = 400.0f;
constexpr float kArtificialViscosityEpsilon = 0.01f;
constexpr float kHydrostaticCalibrationPixelsPerMeter = 100.0f;
constexpr float kMaximumAutoStiffnessMultiplier = 512.0f;
constexpr std::size_t kParticleReserveHint = 500;
constexpr std::size_t kRendererFillVertexBudget = 120000;
constexpr float kVelocityColorReferenceSpeed = 700.0f;
constexpr float kVelocityColorStillThreshold = 14.0f;
constexpr float kVelocityColorSensitivityPower = 0.58f;
constexpr float kParticleBillboardEdgeFeatherPx = 0.75f;

struct BoxBounds2D {
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    bool valid = false;
};

struct UniformGrid2D {
    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
    std::size_t cols = 0;
    std::size_t rows = 0;
    std::vector<std::size_t> cellStarts;
    std::vector<std::size_t> indices;
};

sim::renderer::Color BuildColorFromArray(const float color[3]) {
    return sim::renderer::Color{
        std::clamp(color[0], 0.0f, 1.0f),
        std::clamp(color[1], 0.0f, 1.0f),
        std::clamp(color[2], 0.0f, 1.0f)
    };
}

float ClampUnitFloat(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

glm::vec2 ComputeMouseForceAcceleration(
    const glm::vec2& particlePosition,
    const glm::vec2& particleVelocity,
    const MouseForceInput& mouseInput,
    float forceRadius,
    float forceStrength
) {
    const int forceDirection = static_cast<int>(mouseInput.rightPressed) - static_cast<int>(mouseInput.leftPressed);
    if (forceDirection == 0 || forceRadius <= 0.0f || forceStrength <= 0.0f) {
        return {0.0f, 0.0f};
    }

    const glm::vec2 toMouse = mouseInput.position - particlePosition;
    const float distanceSquared = glm::dot(toMouse, toMouse);
    const float distance = std::sqrt(std::max(0.0f, distanceSquared));

    const float springCoefficient = forceStrength / std::max(1.0f, forceRadius);
    const float dampingCoefficient = 2.0f * std::sqrt(std::max(0.0f, springCoefficient));
    glm::vec2 safeMouseVelocity = mouseInput.velocity;
    if (!std::isfinite(safeMouseVelocity.x) || !std::isfinite(safeMouseVelocity.y)) {
        safeMouseVelocity = {0.0f, 0.0f};
    }

    if (forceDirection > 0) {
        // Right click: wider capture with distance-aware gain to keep dragging at high cursor speed.
        const float captureRadius = forceRadius * 2.2f;
        const float captureRadiusSquared = captureRadius * captureRadius;
        if (distanceSquared >= captureRadiusSquared) {
            return {0.0f, 0.0f};
        }

        const float nearInfluence = std::clamp(1.0f - (distance / forceRadius), 0.0f, 1.0f);
        const float captureInfluence = std::clamp(1.0f - (distance / captureRadius), 0.0f, 1.0f);
        const float influence = std::max(nearInfluence, captureInfluence * captureInfluence);
        if (influence <= 0.0f) {
            return {0.0f, 0.0f};
        }

        const float normalizedDistance = std::clamp(distance / captureRadius, 0.0f, 1.0f);
        const float distanceGain = 1.0f + 1.6f * normalizedDistance;
        const float tunedSpring = springCoefficient * distanceGain;
        const float tunedDamping = dampingCoefficient * std::sqrt(distanceGain);
        const glm::vec2 springAcceleration = toMouse * tunedSpring;
        const glm::vec2 dampingAcceleration = (safeMouseVelocity - particleVelocity) * tunedDamping;
        return (springAcceleration + dampingAcceleration) * influence;
    }

    const float radiusSquared = forceRadius * forceRadius;
    if (distanceSquared >= radiusSquared) {
        return {0.0f, 0.0f};
    }
    const float influence = std::clamp(1.0f - (distance / forceRadius), 0.0f, 1.0f);
    if (influence <= 0.0f) {
        return {0.0f, 0.0f};
    }

    if (distanceSquared <= 1e-8f) {
        return {0.0f, 0.0f};
    }

    // Left click: drive particles toward a bounded outward target speed to avoid ballistic decoupling.
    const glm::vec2 directionToMouse = toMouse / distance;
    const glm::vec2 outwardDirection = -directionToMouse;
    const float nominalPushSpeed = std::sqrt(std::max(0.0f, forceStrength * forceRadius));
    const float desiredPushSpeed = nominalPushSpeed * std::sqrt(influence);
    const glm::vec2 targetVelocity = outwardDirection * desiredPushSpeed;
    const float velocityTrackingGain = dampingCoefficient * 0.85f;
    const glm::vec2 velocityTrackingAcceleration = (targetVelocity - particleVelocity) * velocityTrackingGain;
    const glm::vec2 radialAssist = outwardDirection * (forceStrength * 0.25f * influence * influence);
    return velocityTrackingAcceleration + radialAssist;
}

sim::renderer::Color LerpColor(
    const sim::renderer::Color& a,
    const sim::renderer::Color& b,
    float t
) {
    const float mixT = ClampUnitFloat(t);
    return sim::renderer::Color{
        a.r + (b.r - a.r) * mixT,
        a.g + (b.g - a.g) * mixT,
        a.b + (b.b - a.b) * mixT
    };
}

float ComputeVelocityColorMix(float speed, float referenceSpeed) {
    const float safeReference = std::max(kVelocityColorStillThreshold + 1.0f, referenceSpeed);
    const float safeSpeed = std::max(0.0f, speed);
    if (safeSpeed <= kVelocityColorStillThreshold) {
        return 0.0f;
    }

    const float linearMix = ClampUnitFloat(
        (safeSpeed - kVelocityColorStillThreshold) / (safeReference - kVelocityColorStillThreshold)
    );
    // Boost low-to-mid speeds so color changes are easier to observe.
    return std::pow(linearMix, kVelocityColorSensitivityPower);
}

sim::renderer::Color BuildVelocityGradientColor(
    const sim::renderer::Color& baseColor,
    float speedMix
) {
    const float t = ClampUnitFloat(speedMix);
    const sim::renderer::Color cyan{0.10f, 0.84f, 1.0f};
    const sim::renderer::Color yellow{1.0f, 0.90f, 0.20f};
    const sim::renderer::Color red{1.0f, 0.22f, 0.10f};
    const sim::renderer::Color coolBase = LerpColor(baseColor, cyan, 0.45f);

    if (t < 0.40f) {
        return LerpColor(coolBase, cyan, t / 0.40f);
    }
    if (t < 0.78f) {
        return LerpColor(cyan, yellow, (t - 0.40f) / 0.38f);
    }
    return LerpColor(yellow, red, (t - 0.78f) / 0.22f);
}

float DistanceSquared(const glm::vec2& a, const glm::vec2& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

std::size_t GridCellIndex(std::size_t row, std::size_t col, std::size_t cols) {
    return row * cols + col;
}

template <typename PositionAccessor>
void BuildUniformGrid2DInPlace(
    UniformGrid2D& grid,
    std::vector<std::size_t>& writeOffsetsScratch,
    float minX,
    float maxX,
    float minY,
    float maxY,
    float cellSize,
    std::size_t elementCount,
    const PositionAccessor& positionOf
) {
    grid.cellSize = std::max(0.001f, cellSize);
    grid.invCellSize = 1.0f / grid.cellSize;
    grid.minX = std::min(minX, maxX);
    grid.maxX = std::max(minX, maxX);
    grid.minY = std::min(minY, maxY);
    grid.maxY = std::max(minY, maxY);

    const float width = std::max(0.001f, grid.maxX - grid.minX);
    const float height = std::max(0.001f, grid.maxY - grid.minY);
    grid.cols = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(width / grid.cellSize)));
    grid.rows = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(height / grid.cellSize)));
    const std::size_t cellCount = grid.cols * grid.rows;
    if (grid.cellStarts.size() != cellCount + 1) {
        grid.cellStarts.resize(cellCount + 1, 0);
    }
    std::fill(grid.cellStarts.begin(), grid.cellStarts.end(), 0);

    auto clampCol = [&](float x) {
        const int value = static_cast<int>(std::floor((x - grid.minX) * grid.invCellSize));
        return std::clamp(value, 0, static_cast<int>(grid.cols) - 1);
    };
    auto clampRow = [&](float y) {
        const int value = static_cast<int>(std::floor((y - grid.minY) * grid.invCellSize));
        return std::clamp(value, 0, static_cast<int>(grid.rows) - 1);
    };

    for (std::size_t i = 0; i < elementCount; ++i) {
        const glm::vec2 p = positionOf(i);
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        const int col = clampCol(p.x);
        const int row = clampRow(p.y);
        const std::size_t cellIndex =
            GridCellIndex(static_cast<std::size_t>(row), static_cast<std::size_t>(col), grid.cols);
        ++grid.cellStarts[cellIndex + 1];
    }

    for (std::size_t i = 1; i < grid.cellStarts.size(); ++i) {
        grid.cellStarts[i] += grid.cellStarts[i - 1];
    }

    const std::size_t indexCount = grid.cellStarts.back();
    if (grid.indices.size() != indexCount) {
        grid.indices.resize(indexCount);
    }

    if (writeOffsetsScratch.size() != cellCount) {
        writeOffsetsScratch.resize(cellCount);
    }
    if (cellCount > 0) {
        std::copy_n(grid.cellStarts.begin(), cellCount, writeOffsetsScratch.begin());
    }

    for (std::size_t i = 0; i < elementCount; ++i) {
        const glm::vec2 p = positionOf(i);
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        const int col = clampCol(p.x);
        const int row = clampRow(p.y);
        const std::size_t cellIndex =
            GridCellIndex(static_cast<std::size_t>(row), static_cast<std::size_t>(col), grid.cols);
        const std::size_t writeIndex = writeOffsetsScratch[cellIndex]++;
        grid.indices[writeIndex] = i;
    }
}

template <typename PositionAccessor>
UniformGrid2D BuildUniformGrid2D(
    float minX,
    float maxX,
    float minY,
    float maxY,
    float cellSize,
    std::size_t elementCount,
    const PositionAccessor& positionOf
) {
    UniformGrid2D grid{};
    std::vector<std::size_t> writeOffsetsScratch;
    BuildUniformGrid2DInPlace(
        grid,
        writeOffsetsScratch,
        minX,
        maxX,
        minY,
        maxY,
        cellSize,
        elementCount,
        positionOf
    );
    return grid;
}

template <typename CandidateVisitor>
void VisitGridCandidates(
    const UniformGrid2D& grid,
    const glm::vec2& position,
    float queryRadius,
    const CandidateVisitor& visitor
) {
    if (grid.cols == 0 || grid.rows == 0 || grid.cellStarts.empty() || grid.indices.empty()) {
        return;
    }

    const float safeQueryRadius = std::max(0.0f, queryRadius);
    const int baseCol = std::clamp(
        static_cast<int>(std::floor((position.x - grid.minX) * grid.invCellSize)),
        0,
        static_cast<int>(grid.cols) - 1
    );
    const int baseRow = std::clamp(
        static_cast<int>(std::floor((position.y - grid.minY) * grid.invCellSize)),
        0,
        static_cast<int>(grid.rows) - 1
    );
    const int cellRange = std::max(1, static_cast<int>(std::ceil(safeQueryRadius * grid.invCellSize)));

    const int minRow = std::max(0, baseRow - cellRange);
    const int maxRow = std::min(static_cast<int>(grid.rows) - 1, baseRow + cellRange);
    const int minCol = std::max(0, baseCol - cellRange);
    const int maxCol = std::min(static_cast<int>(grid.cols) - 1, baseCol + cellRange);
    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            const std::size_t cellIndex =
                GridCellIndex(static_cast<std::size_t>(row), static_cast<std::size_t>(col), grid.cols);
            const std::size_t start = grid.cellStarts[cellIndex];
            const std::size_t end = grid.cellStarts[cellIndex + 1];
            for (std::size_t i = start; i < end; ++i) {
                visitor(grid.indices[i]);
            }
        }
    }
}

float ComputeDensityKernelRadius(const BallRenderComponent& render) {
    // SPH density support is controlled by smoothing radius h.
    return std::max(0.001f, std::max(render.smoothingRadius, render.radius));
}

float ComputeNormalizedCompactKernel2D(float distanceSquared, float smoothingLength) {
    if (smoothingLength <= 0.0f) {
        return 0.0f;
    }

    // SPH kernels must integrate to 1 over support (Muller et al., SCA 2003).
    // For W(r,h)=C*(1-r/h)^2 with r<h in 2D, C = 6/(pi*h^2).
    const float h2 = smoothingLength * smoothingLength;
    if (distanceSquared >= h2) {
        return 0.0f;
    }

    const float distance = std::sqrt(std::max(0.0f, distanceSquared));
    const float q = 1.0f - distance / smoothingLength;
    const float normalization = 6.0f / (glm::pi<float>() * h2);
    return normalization * q * q;
}

glm::vec2 ComputeNormalizedCompactKernelGradient2D(const glm::vec2& delta, float smoothingLength) {
    if (smoothingLength <= 0.0f) {
        return {0.0f, 0.0f};
    }

    const float distanceSquared = delta.x * delta.x + delta.y * delta.y;
    const float h2 = smoothingLength * smoothingLength;
    if (distanceSquared <= 1e-10f || distanceSquared >= h2) {
        return {0.0f, 0.0f};
    }

    const float distance = std::sqrt(distanceSquared);
    const float q = 1.0f - distance / smoothingLength;
    const float normalization = 6.0f / (glm::pi<float>() * h2);
    const float dWdr = -2.0f * normalization * q / smoothingLength;
    const float invDistance = 1.0f / distance;
    return glm::vec2{
        dWdr * delta.x * invDistance,
        dWdr * delta.y * invDistance
    };
}

float ComputeDensityParticleMass(float restDensity, float representativeArea) {
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeArea = std::max(0.0001f, representativeArea);
    return safeRestDensity * safeArea;
}

float ComputeRepresentativeParticleArea(
    const BoxBounds2D& bounds,
    std::size_t particleCount,
    float fallbackParticleRadius
) {
    const float safeFallbackRadius = std::max(0.001f, fallbackParticleRadius);
    const float fallbackArea = glm::pi<float>() * safeFallbackRadius * safeFallbackRadius;
    if (particleCount == 0) {
        return fallbackArea;
    }

    if (!bounds.valid) {
        return fallbackArea;
    }

    const float boundsWidth = std::max(1.0f, bounds.maxX - bounds.minX);
    const float boundsHeight = std::max(1.0f, bounds.maxY - bounds.minY);
    const float areaPerParticle =
        (boundsWidth * boundsHeight) / static_cast<float>(particleCount);
    return std::max(0.0001f, areaPerParticle);
}

template <typename ParticleRefType>
float EstimateFluidColumnHeight(
    const std::vector<ParticleRefType>& particles,
    const BoxBounds2D& bounds
) {
    const float boundsHeight = bounds.valid
        ? std::max(1.0f, bounds.maxY - bounds.minY)
        : 1.0f;
    if (particles.empty()) {
        return boundsHeight;
    }

    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    bool hasValidParticle = false;
    for (const auto& particle : particles) {
        if (particle.transform == nullptr || particle.render == nullptr) {
            continue;
        }
        const glm::vec2 position = particle.transform->position;
        if (!std::isfinite(position.x) || !std::isfinite(position.y)) {
            continue;
        }
        const float radius = std::max(0.0f, particle.render->radius);
        minY = std::min(minY, position.y - radius);
        maxY = std::max(maxY, position.y + radius);
        hasValidParticle = true;
    }
    if (!hasValidParticle) {
        return boundsHeight;
    }

    const float measuredHeight = std::max(1.0f, maxY - minY);
    if (!bounds.valid) {
        return measuredHeight;
    }
    return std::clamp(measuredHeight, 1.0f, boundsHeight);
}

float ComputeWcSphRecommendedMaxGravity(
    float restDensity,
    float stiffness,
    float gamma,
    float fluidHeight,
    float targetDensityErrorRatio
) {
    // WCSPH hydrostatic estimate:
    // deltaRho/rho0 ~= (rho0 * g * H) / (gamma * B), B ~= stiffness
    // References: Monaghan (2005), Becker & Teschner (2007).
    // The runtime integrates in pixel units; convert this diagnostic estimate to a
    // meter-like scale so auto-calibration stays in a practical numeric range.
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeStiffness = std::max(0.0f, stiffness);
    const float safeGamma = std::max(1.0f, gamma);
    const float safeHeightPixels = std::max(1.0f, fluidHeight);
    const float safeHeightMeters = safeHeightPixels / kHydrostaticCalibrationPixelsPerMeter;
    const float safeTargetError = std::clamp(targetDensityErrorRatio, 0.001f, 0.2f);
    const float maxGravityMeters =
        safeTargetError * safeStiffness * safeGamma / (safeRestDensity * safeHeightMeters);
    const float maxGravityPixels = maxGravityMeters * kHydrostaticCalibrationPixelsPerMeter;
    return std::isfinite(maxGravityPixels) ? std::max(0.0f, maxGravityPixels) : 0.0f;
}

float ComputeWcSphRequiredStiffnessForGravity(
    float restDensity,
    float gravity,
    float gamma,
    float fluidHeight,
    float targetDensityErrorRatio
) {
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeGravityPixels = std::max(0.0f, gravity);
    const float safeGamma = std::max(1.0f, gamma);
    const float safeHeightPixels = std::max(1.0f, fluidHeight);
    const float safeGravityMeters = safeGravityPixels / kHydrostaticCalibrationPixelsPerMeter;
    const float safeHeightMeters = safeHeightPixels / kHydrostaticCalibrationPixelsPerMeter;
    const float safeTargetError = std::clamp(targetDensityErrorRatio, 0.001f, 0.2f);
    const float requiredStiffness =
        (safeRestDensity * safeGravityMeters * safeHeightMeters) / (safeGamma * safeTargetError);
    return std::isfinite(requiredStiffness) ? std::max(0.0f, requiredStiffness) : 0.0f;
}

float ComputeEffectiveStiffness(float baseStiffness, float requiredStiffness) {
    const float safeBaseStiffness = std::max(0.0f, baseStiffness);
    const float safeRequiredStiffness = std::max(safeBaseStiffness, requiredStiffness);
    const float autoStiffnessCap = safeBaseStiffness > 0.0f
        ? safeBaseStiffness * kMaximumAutoStiffnessMultiplier
        : safeRequiredStiffness;
    return std::min(safeRequiredStiffness, autoStiffnessCap);
}

float ComputeEquationOfStatePressure(
    float density,
    float restDensity,
    float stiffness,
    float gamma,
    float maxPressureLimit
) {
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeStiffness = std::max(0.0f, stiffness);
    const float safeGamma = std::max(1.0f, gamma);
    float ratio = density / safeRestDensity;
    if (!std::isfinite(ratio)) {
        ratio = 1.0f;
    }
    ratio = std::max(0.001f, ratio);
    float pressure = safeStiffness * (std::pow(ratio, safeGamma) - 1.0f);
    const float minPressure = -kMaximumNegativePressureRatio * safeStiffness;
    const float baselineMaxPressure = std::max(4000.0f, safeStiffness * kMaximumPressureToStiffnessRatio);
    const float safeMaxPressureLimit = std::max(0.0f, maxPressureLimit);
    const bool pressureLimitEnabled = safeMaxPressureLimit > 0.0f;
    const float maxPressure = pressureLimitEnabled ? safeMaxPressureLimit : baselineMaxPressure;
    if (!std::isfinite(pressure)) {
        pressure = maxPressure;
    }
    if (pressureLimitEnabled) {
        return std::clamp(pressure, minPressure, maxPressure);
    }
    return std::max(minPressure, pressure);
}

bool IsFiniteVec2(const glm::vec2& v) {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

glm::vec2 ClampVectorMagnitude(const glm::vec2& v, float maxMagnitude) {
    const float safeMaxMagnitude = std::max(0.0001f, maxMagnitude);
    const float len2 = glm::dot(v, v);
    if (!std::isfinite(len2) || len2 <= safeMaxMagnitude * safeMaxMagnitude) {
        return v;
    }

    const float len = std::sqrt(std::max(0.0f, len2));
    if (len <= 1e-8f || !std::isfinite(len)) {
        return {0.0f, 0.0f};
    }
    return v * (safeMaxMagnitude / len);
}

float ComputeBoundaryDensityGhostContribution(
    const glm::vec2& position,
    const BoxBounds2D& bounds,
    float smoothingLength,
    float mass
) {
    if (!bounds.valid || smoothingLength <= 0.0f) {
        return 0.0f;
    }

    float contribution = 0.0f;
    auto addMirror = [&](float distanceToWall) {
        if (distanceToWall <= 0.0f) {
            return;
        }
        const float mirrorDistance = 2.0f * distanceToWall;
        if (mirrorDistance >= smoothingLength) {
            return;
        }
        contribution += mass * ComputeNormalizedCompactKernel2D(
            mirrorDistance * mirrorDistance,
            smoothingLength
        );
    };

    addMirror(position.x - bounds.minX);
    addMirror(bounds.maxX - position.x);
    addMirror(position.y - bounds.minY);
    addMirror(bounds.maxY - position.y);
    return contribution;
}

void AccumulateBoundaryPressureGhostAcceleration(
    glm::vec2& pressureAcceleration,
    const glm::vec2& position,
    const BoxBounds2D& bounds,
    float smoothingLength,
    float mass,
    float pressure,
    float density
) {
    if (!bounds.valid || smoothingLength <= 0.0f || pressure <= 0.0f || density <= 0.0f) {
        return;
    }

    const float safeDensity = std::max(0.001f, density);
    const float pressureTerm = 2.0f * pressure / (safeDensity * safeDensity);
    const float h2 = smoothingLength * smoothingLength;
    auto addMirror = [&](const glm::vec2& mirrorPoint) {
        const glm::vec2 delta = position - mirrorPoint;
        if (std::abs(delta.x) >= smoothingLength || std::abs(delta.y) >= smoothingLength) {
            return;
        }
        const float distSquared = delta.x * delta.x + delta.y * delta.y;
        if (distSquared <= 1e-10f || distSquared >= h2) {
            return;
        }
        const glm::vec2 gradW = ComputeNormalizedCompactKernelGradient2D(delta, smoothingLength);
        pressureAcceleration += -mass * pressureTerm * gradW;
    };

    addMirror(glm::vec2{2.0f * bounds.minX - position.x, position.y});
    addMirror(glm::vec2{2.0f * bounds.maxX - position.x, position.y});
    addMirror(glm::vec2{position.x, 2.0f * bounds.minY - position.y});
    addMirror(glm::vec2{position.x, 2.0f * bounds.maxY - position.y});
}

float ComputeStableSubstep(
    const std::vector<float>& densityRadii,
    const std::vector<glm::vec2>& velocities,
    const UpdateSettings& settings,
    float remainingTime,
    float restDensity,
    float stiffness,
    float gamma,
    float gravity,
    float mouseForceStrength
) {
    if (densityRadii.empty() || velocities.empty()) {
        return remainingTime;
    }

    const std::size_t sampleCount = std::min(densityRadii.size(), velocities.size());
    float minSmoothing = std::numeric_limits<float>::max();
    float maxSpeedSquared = 0.0f;
    for (std::size_t i = 0; i < sampleCount; ++i) {
        minSmoothing = std::min(minSmoothing, std::max(0.001f, densityRadii[i]));
        const glm::vec2 v = velocities[i];
        const float speedSquared = v.x * v.x + v.y * v.y;
        if (std::isfinite(speedSquared)) {
            maxSpeedSquared = std::max(maxSpeedSquared, std::max(0.0f, speedSquared));
        }
    }
    minSmoothing = std::max(0.001f, minSmoothing);
    const float maxSpeed = std::sqrt(maxSpeedSquared);

    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeStiffness = std::max(0.0f, stiffness);
    const float safeGamma = std::max(1.0f, gamma);
    const float safeViscosity = std::max(0.0f, settings.viscosity);
    const float safeCfl = std::clamp(settings.cflFactor, 0.05f, 1.0f);
    const float speedOfSound = std::sqrt(std::max(0.0001f, safeStiffness * safeGamma / safeRestDensity));
    const float dtCfl = safeCfl * minSmoothing / std::max(0.001f, speedOfSound + maxSpeed);
    const float dtVisc = safeViscosity > 0.0f
        ? 0.125f * minSmoothing * minSmoothing / std::max(0.0001f, safeViscosity)
        : std::numeric_limits<float>::max();
    const float safeMouseForceStrength = std::max(0.0f, mouseForceStrength);
    const float accelerationScale = std::max(
        0.0001f,
        std::abs(gravity) + safeMouseForceStrength + safeStiffness / safeRestDensity
    );
    const float dtForce = 0.25f * std::sqrt(minSmoothing / accelerationScale);

    const float safeMinStep = std::max(0.0001f, settings.minTimeStep);
    const float safeMaxStep = std::max(safeMinStep, settings.maxTimeStep);
    float dt = std::min({remainingTime, dtCfl, dtVisc, dtForce, safeMaxStep});
    dt = std::max(dt, safeMinStep);
    return std::min(dt, remainingTime);
}

void ResolveParticleBoundsCollision(
    glm::vec2& position,
    float radius,
    glm::vec2& velocity,
    const BoxBounds2D& bounds,
    float restitution
) {
    if (!bounds.valid) {
        return;
    }

    const float safeRadius = std::max(1.0f, radius);
    if (position.x - safeRadius < bounds.minX) {
        position.x = bounds.minX + safeRadius;
        velocity.x = std::abs(velocity.x) * restitution;
    }
    if (position.x + safeRadius > bounds.maxX) {
        position.x = bounds.maxX - safeRadius;
        velocity.x = -std::abs(velocity.x) * restitution;
    }
    if (position.y - safeRadius < bounds.minY) {
        position.y = bounds.minY + safeRadius;
        velocity.y = std::abs(velocity.y) * restitution;
    }
    if (position.y + safeRadius > bounds.maxY) {
        position.y = bounds.maxY - safeRadius;
        velocity.y = -std::abs(velocity.y) * restitution;
        if (std::abs(velocity.y) < 2.0f) {
            velocity.y = 0.0f;
        }
    }
}

std::vector<float> BlurDensityGrid(
    const std::vector<float>& source,
    std::size_t cols,
    std::size_t rows
) {
    if (source.empty() || cols == 0 || rows == 0) {
        return source;
    }

    auto clampIndex = [](int value, int low, int high) {
        return std::clamp(value, low, high);
    };
    auto indexOf = [cols](std::size_t row, std::size_t col) {
        return row * cols + col;
    };

    std::vector<float> temp(source.size(), 0.0f);
    std::vector<float> output(source.size(), 0.0f);

    // Separable 1-2-1 blur pass for smoother density-view boundaries.
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const int c0 = clampIndex(static_cast<int>(col) - 1, 0, static_cast<int>(cols) - 1);
            const int c1 = static_cast<int>(col);
            const int c2 = clampIndex(static_cast<int>(col) + 1, 0, static_cast<int>(cols) - 1);
            temp[indexOf(row, col)] =
                (source[indexOf(row, static_cast<std::size_t>(c0))] +
                 2.0f * source[indexOf(row, static_cast<std::size_t>(c1))] +
                 source[indexOf(row, static_cast<std::size_t>(c2))]) *
                0.25f;
        }
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const int r0 = clampIndex(static_cast<int>(row) - 1, 0, static_cast<int>(rows) - 1);
        const int r1 = static_cast<int>(row);
        const int r2 = clampIndex(static_cast<int>(row) + 1, 0, static_cast<int>(rows) - 1);
        for (std::size_t col = 0; col < cols; ++col) {
            output[indexOf(row, col)] =
                (temp[indexOf(static_cast<std::size_t>(r0), col)] +
                 2.0f * temp[indexOf(static_cast<std::size_t>(r1), col)] +
                 temp[indexOf(static_cast<std::size_t>(r2), col)]) *
                0.25f;
        }
    }

    return output;
}

std::vector<glm::vec2> GenerateRandomSpawnPositions(
    std::size_t count,
    float spawnHalfWidth,
    float spawnHalfHeight,
    float minDistance,
    std::mt19937& rng
) {
    std::vector<glm::vec2> positions;
    positions.reserve(count);
    if (count == 0) {
        return positions;
    }

    const float safeHalfWidth = std::max(1.0f, spawnHalfWidth);
    const float safeHalfHeight = std::max(1.0f, spawnHalfHeight);
    const float width = safeHalfWidth * 2.0f;
    const float height = safeHalfHeight * 2.0f;
    const float area = width * height;
    const float safeMinDistance = std::max(0.5f, minDistance);
    const float areaBasedSpacing = std::sqrt(std::max(1e-6f, area / static_cast<float>(count)));
    const float targetSpacing = std::min(safeMinDistance, std::max(0.5f, areaBasedSpacing));

    std::size_t cols = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(width / targetSpacing)));
    std::size_t rows = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(height / targetSpacing)));
    const float aspect = width / std::max(1.0f, height);
    while (cols * rows < count) {
        const float gridAspect = static_cast<float>(cols) / static_cast<float>(rows);
        if (gridAspect < aspect) {
            ++cols;
        } else {
            ++rows;
        }
    }

    const float cellWidth = width / static_cast<float>(cols);
    const float cellHeight = height / static_cast<float>(rows);
    const float jitterX = std::max(0.0f, 0.5f * (cellWidth - safeMinDistance));
    const float jitterY = std::max(0.0f, 0.5f * (cellHeight - safeMinDistance));
    std::uniform_real_distribution<float> jitterDistX(-jitterX, jitterX);
    std::uniform_real_distribution<float> jitterDistY(-jitterY, jitterY);

    const std::size_t cellCount = cols * rows;
    std::vector<std::size_t> shuffledCellIndices(cellCount, 0);
    std::iota(shuffledCellIndices.begin(), shuffledCellIndices.end(), 0);
    std::shuffle(shuffledCellIndices.begin(), shuffledCellIndices.end(), rng);

    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t shuffledIndex = shuffledCellIndices[i];
        const std::size_t row = shuffledIndex / cols;
        const std::size_t col = shuffledIndex % cols;
        const float baseX = -safeHalfWidth + (static_cast<float>(col) + 0.5f) * cellWidth;
        const float baseY = -safeHalfHeight + (static_cast<float>(row) + 0.5f) * cellHeight;
        const float candidateX = baseX + (jitterX > 0.0f ? jitterDistX(rng) : 0.0f);
        const float candidateY = baseY + (jitterY > 0.0f ? jitterDistY(rng) : 0.0f);
        positions.push_back(glm::vec2{
            std::clamp(candidateX, -safeHalfWidth, safeHalfWidth),
            std::clamp(candidateY, -safeHalfHeight, safeHalfHeight)
        });
    }

    return positions;
}

void RunTransformNormalizationSystem(sim::ecs::World& world) {
    auto view = world.Registry().view<sim::ecs::TransformComponent>();
    for (const auto entity : view) {
        auto& transform = view.get<sim::ecs::TransformComponent>(entity);
        transform.rotationRadians = std::remainder(transform.rotationRadians, glm::two_pi<float>());
        transform.scale.x = std::max(transform.scale.x, 0.001f);
        transform.scale.y = std::max(transform.scale.y, 0.001f);
    }
}

void RunBallRenderValidationSystem(sim::ecs::World& world) {
    auto view = world.Registry().view<BallRenderComponent>();
    for (const auto entity : view) {
        auto& render = view.get<BallRenderComponent>(entity);
        render.radius = std::max(render.radius, 1.0f);
        render.smoothingRadius = std::max(render.smoothingRadius, render.radius);
        render.segments = std::clamp(render.segments, 6, 96);
        render.color.r = ClampUnitFloat(render.color.r);
        render.color.g = ClampUnitFloat(render.color.g);
        render.color.b = ClampUnitFloat(render.color.b);
    }
}

void RunBoundingBoxValidationSystem(sim::ecs::World& world) {
    auto view = world.Registry().view<BoundingBoxRenderComponent>();
    for (const auto entity : view) {
        auto& render = view.get<BoundingBoxRenderComponent>(entity);
        render.size.x = std::max(render.size.x, 1.0f);
        render.size.y = std::max(render.size.y, 1.0f);
        render.lineThickness = std::max(render.lineThickness, 0.0f);
        render.color.r = ClampUnitFloat(render.color.r);
        render.color.g = ClampUnitFloat(render.color.g);
        render.color.b = ClampUnitFloat(render.color.b);
    }
}

void RunSceneStateSyncSystem(sim::ecs::World& world, const UiState& state, Uint32 viewportWidth, Uint32 viewportHeight) {
    const float w = static_cast<float>(viewportWidth);
    const float h = static_cast<float>(viewportHeight);
    const glm::vec2 screenCenter(w * 0.5f, h * 0.5f);
    const float boxInset = std::max(0.0f, state.world.boxInset);

    auto boxView = world.Registry().view<sim::ecs::TransformComponent, BoundingBoxRenderComponent>();
    for (const auto entity : boxView) {
        auto& transform = boxView.get<sim::ecs::TransformComponent>(entity);
        auto& render = boxView.get<BoundingBoxRenderComponent>(entity);
        transform.position = screenCenter;
        transform.rotationRadians = 0.0f;
        transform.scale = {1.0f, 1.0f};
        render.size = {
            std::max(1.0f, w - 2.0f * boxInset),
            std::max(1.0f, h - 2.0f * boxInset)
        };
        render.lineThickness = std::max(0.0f, state.world.boxLineThickness);
        render.color = BuildColorFromArray(state.world.boxColor);
        render.enabled = state.render.boundingBox;
    }

    auto ballView = world.Registry().view<sim::ecs::TransformComponent, BallRenderComponent, BallPhysicsComponent>();
    for (const auto entity : ballView) {
        auto& transform = ballView.get<sim::ecs::TransformComponent>(entity);
        auto& render = ballView.get<BallRenderComponent>(entity);
        auto& physics = ballView.get<BallPhysicsComponent>(entity);
        const glm::vec2 desiredOffset(state.world.ballOffsetX, state.world.ballOffsetY);

        if (!physics.anchorInitialized) {
            transform.position = screenCenter + desiredOffset;
            physics.lastAnchorCenter = screenCenter;
            physics.lastUiOffset = desiredOffset;
            physics.anchorInitialized = true;
        }

        const glm::vec2 anchorDelta =
            (screenCenter - physics.lastAnchorCenter) + (desiredOffset - physics.lastUiOffset);
        transform.position += anchorDelta;
        physics.lastAnchorCenter = screenCenter;
        physics.lastUiOffset = desiredOffset;

        render.radius = std::max(1.0f, state.world.ballRadius);
        render.smoothingRadius = std::max(state.update.smoothingRadius, render.radius);
        render.segments = std::clamp(state.world.ballSegments, 6, 96);
        render.color = BuildColorFromArray(state.world.ballColor);
        // Keep per-entity availability independent from UI draw switches.
        render.enabled = true;

    }
}

BoxBounds2D BuildPrimaryBounds(const sim::ecs::World& world) {
    BoxBounds2D bounds;
    auto view = world.Registry().view<const sim::ecs::TransformComponent, const BoundingBoxRenderComponent>();
    for (const auto entity : view) {
        const auto& transform = view.get<const sim::ecs::TransformComponent>(entity);
        const auto& render = view.get<const BoundingBoxRenderComponent>(entity);
        const float halfW = std::max(0.5f, render.size.x * 0.5f);
        const float halfH = std::max(0.5f, render.size.y * 0.5f);
        bounds.minX = transform.position.x - halfW;
        bounds.maxX = transform.position.x + halfW;
        bounds.minY = transform.position.y - halfH;
        bounds.maxY = transform.position.y + halfH;
        bounds.valid = true;
        break;
    }
    return bounds;
}

struct DensityFieldParticle {
    glm::vec2 position{0.0f, 0.0f};
    float renderRadius = 0.0f;
    float densityRadius = 0.0f;
    float densityMass = 0.0f;
};

float EvaluateDensityAtPoint(
    const glm::vec2& samplePoint,
    const std::vector<DensityFieldParticle>& particles,
    const BoxBounds2D& bounds,
    const UniformGrid2D& grid,
    float maxSupportRadius
) {
    float density = 0.0f;
    VisitGridCandidates(grid, samplePoint, maxSupportRadius, [&](std::size_t particleIndex) {
        const auto& particle = particles[particleIndex];
        const glm::vec2 delta = samplePoint - particle.position;
        const float h = particle.densityRadius;
        if (std::abs(delta.x) >= h || std::abs(delta.y) >= h) {
            return;
        }
        const float distSquared = delta.x * delta.x + delta.y * delta.y;
        if (distSquared >= h * h) {
            return;
        }
        density += particle.densityMass * ComputeNormalizedCompactKernel2D(distSquared, h);

        if (bounds.valid) {
            auto addMirrorContribution = [&](const glm::vec2& mirrorPosition) {
                const glm::vec2 mirrorDelta = samplePoint - mirrorPosition;
                if (std::abs(mirrorDelta.x) >= h || std::abs(mirrorDelta.y) >= h) {
                    return;
                }
                const float mirrorDistSquared =
                    mirrorDelta.x * mirrorDelta.x + mirrorDelta.y * mirrorDelta.y;
                if (mirrorDistSquared >= h * h) {
                    return;
                }
                density += particle.densityMass * ComputeNormalizedCompactKernel2D(mirrorDistSquared, h);
            };
            addMirrorContribution(glm::vec2{2.0f * bounds.minX - particle.position.x, particle.position.y});
            addMirrorContribution(glm::vec2{2.0f * bounds.maxX - particle.position.x, particle.position.y});
            addMirrorContribution(glm::vec2{particle.position.x, 2.0f * bounds.minY - particle.position.y});
            addMirrorContribution(glm::vec2{particle.position.x, 2.0f * bounds.maxY - particle.position.y});
        }
    });
    return density;
}

void RunParticleDensitySystem(sim::ecs::World& world, float restDensity, float massDensityReference) {
    struct ParticleRefs {
        sim::ecs::TransformComponent* transform = nullptr;
        BallPhysicsComponent* physics = nullptr;
        float renderRadius = 0.0f;
        float densityRadius = 0.0f;
        float densityMass = 0.0f;
    };

    auto view = world.Registry().view<sim::ecs::TransformComponent, BallRenderComponent, BallPhysicsComponent>();
    thread_local std::vector<ParticleRefs> particles;
    particles.clear();
    particles.reserve(std::max<std::size_t>(kParticleReserveHint, view.size_hint()));
    const BoxBounds2D bounds = BuildPrimaryBounds(world);
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeMassDensityReference = std::max(0.01f, massDensityReference);
    const float safeDensityFloor = std::max(0.001f, safeRestDensity * kDensityFloorRatio);
    for (const auto entity : view) {
        const auto& render = view.get<BallRenderComponent>(entity);
        if (!render.enabled) {
            continue;
        }
        const float densityRadius = ComputeDensityKernelRadius(render);
        particles.push_back(ParticleRefs{
            &view.get<sim::ecs::TransformComponent>(entity),
            &view.get<BallPhysicsComponent>(entity),
            render.radius,
            densityRadius,
            0.0f
        });
    }

    if (particles.empty()) {
        return;
    }

    const float representativeArea = ComputeRepresentativeParticleArea(
        bounds,
        particles.size(),
        particles.front().renderRadius
    );
    float maxSupportRadius = 0.0f;
    for (auto& particle : particles) {
        particle.densityMass = ComputeDensityParticleMass(safeMassDensityReference, representativeArea);
        maxSupportRadius = std::max(maxSupportRadius, particle.densityRadius);
    }
    maxSupportRadius = std::max(0.001f, maxSupportRadius);
    const UniformGrid2D particleGrid = BuildUniformGrid2D(
        bounds.minX,
        bounds.maxX,
        bounds.minY,
        bounds.maxY,
        maxSupportRadius,
        particles.size(),
        [&](std::size_t index) {
            return particles[index].transform->position;
        }
    );

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const glm::vec2 pi = particles[i].transform->position;
        float density = 0.0f;
        VisitGridCandidates(particleGrid, pi, maxSupportRadius, [&](std::size_t j) {
            const glm::vec2 pj = particles[j].transform->position;
            const glm::vec2 delta = pi - pj;
            const float h = 0.5f * (particles[i].densityRadius + particles[j].densityRadius);
            if (std::abs(delta.x) >= h || std::abs(delta.y) >= h) {
                return;
            }
            const float distSquared = delta.x * delta.x + delta.y * delta.y;
            if (distSquared >= h * h) {
                return;
            }

            density += particles[j].densityMass * ComputeNormalizedCompactKernel2D(distSquared, h);
        });
        density += ComputeBoundaryDensityGhostContribution(
            pi,
            bounds,
            particles[i].densityRadius,
            particles[i].densityMass
        );
        density = std::max(density, safeDensityFloor);

        auto& physics = *particles[i].physics;
        physics.density = density;
        physics.pressure = 0.0f;
        physics.normalizedDensity = ClampUnitFloat(density / safeRestDensity);
    }
}

sim::renderer::DensityFieldGrid2D BuildDensityFieldGrid(
    sim::ecs::World& world,
    Uint32 viewportWidth,
    Uint32 viewportHeight,
    float sampleStep,
    float restDensity,
    float massDensityReference,
    float& outEffectiveStep
) {
    sim::renderer::DensityFieldGrid2D densityGrid{};
    densityGrid.sampleStep = std::max(1.0f, sampleStep);
    densityGrid.width = static_cast<float>(viewportWidth);
    densityGrid.height = static_cast<float>(viewportHeight);

    const BoxBounds2D bounds = BuildPrimaryBounds(world);
    auto ballView = world.Registry().view<const sim::ecs::TransformComponent, const BallRenderComponent>();
    std::vector<DensityFieldParticle> particles;
    particles.reserve(std::max<std::size_t>(kParticleReserveHint, ballView.size_hint()));
    const float safeRestDensity = std::max(0.01f, restDensity);
    const float safeMassDensityReference = std::max(0.01f, massDensityReference);
    for (const auto entity : ballView) {
        const auto& transform = ballView.get<const sim::ecs::TransformComponent>(entity);
        const auto& render = ballView.get<const BallRenderComponent>(entity);
        if (!render.enabled) {
            continue;
        }
        const float densityRadius = ComputeDensityKernelRadius(render);
        particles.push_back(DensityFieldParticle{
            transform.position,
            render.radius,
            densityRadius,
            0.0f
        });
    }

    if (particles.empty() || viewportWidth == 0 || viewportHeight == 0) {
        outEffectiveStep = densityGrid.sampleStep;
        return densityGrid;
    }

    const float width = static_cast<float>(viewportWidth);
    const float height = static_cast<float>(viewportHeight);
    const float representativeArea = ComputeRepresentativeParticleArea(
        bounds,
        particles.size(),
        particles.front().renderRadius
    );
    float maxSupportRadius = 0.0f;
    for (auto& particle : particles) {
        particle.densityMass = ComputeDensityParticleMass(safeMassDensityReference, representativeArea);
        maxSupportRadius = std::max(maxSupportRadius, particle.densityRadius);
    }
    maxSupportRadius = std::max(0.001f, maxSupportRadius);

    const UniformGrid2D particleGrid = BuildUniformGrid2D(
        0.0f,
        width,
        0.0f,
        height,
        maxSupportRadius,
        particles.size(),
        [&](std::size_t index) {
            return particles[index].position;
        }
    );
    float effectiveStep = densityGrid.sampleStep;

    // DrawDensityFieldOverlay2D emits two triangles per cell (6 fill vertices).
    // Keep a safety headroom below renderer fill-batch capacity (131072) to
    // avoid skipping tail cells when box lines and other fill draws are present.
    constexpr std::size_t kOverlayVerticesPerCell = 6;
    constexpr std::size_t kFillVertexBudget = 120000;
    constexpr std::size_t kSafeMaxCells = kFillVertexBudget / kOverlayVerticesPerCell;
    while (true) {
        const std::size_t cols = static_cast<std::size_t>(std::ceil(width / effectiveStep));
        const std::size_t rows = static_cast<std::size_t>(std::ceil(height / effectiveStep));
        if (cols * rows <= kSafeMaxCells || effectiveStep >= 64.0f) {
            break;
        }
        effectiveStep += 1.0f;
    }

    const std::size_t cols = static_cast<std::size_t>(std::ceil(width / effectiveStep));
    const std::size_t rows = static_cast<std::size_t>(std::ceil(height / effectiveStep));
    const std::size_t nodeCols = cols + 1;
    const std::size_t nodeRows = rows + 1;
    std::vector<float> rawNodeDensities(nodeCols * nodeRows, 0.0f);

    auto nodeIndex = [nodeCols](std::size_t row, std::size_t col) {
        return row * nodeCols + col;
    };
    for (std::size_t row = 0; row < nodeRows; ++row) {
        const float y = std::min(height, static_cast<float>(row) * effectiveStep);
        for (std::size_t col = 0; col < nodeCols; ++col) {
            const float x = std::min(width, static_cast<float>(col) * effectiveStep);
            rawNodeDensities[nodeIndex(row, col)] = EvaluateDensityAtPoint(
                glm::vec2{x, y},
                particles,
                bounds,
                particleGrid,
                maxSupportRadius
            );
        }
    }

    const auto blurredNodeDensities = BlurDensityGrid(rawNodeDensities, nodeCols, nodeRows);
    // Keep density-view physically consistent with particle density by default.
    constexpr float kDensityFieldBlurMix = 0.0f;

    densityGrid.cols = cols;
    densityGrid.rows = rows;
    densityGrid.sampleStep = effectiveStep;
    densityGrid.width = width;
    densityGrid.height = height;
    densityGrid.nodeDensities.resize(rawNodeDensities.size(), 0.0f);
    for (std::size_t i = 0; i < rawNodeDensities.size(); ++i) {
        densityGrid.nodeDensities[i] = rawNodeDensities[i]
            + (blurredNodeDensities[i] - rawNodeDensities[i]) * kDensityFieldBlurMix;
    }

    outEffectiveStep = effectiveStep;
    return densityGrid;
}

bool TryRunBallPhysicsSystemCuda(
    sim::ecs::World& world,
    const UpdateSettings& settings,
    const MouseForceInput& mouseInput,
    float deltaSeconds,
    const BoxBounds2D& bounds
) {
    static int s_cudaPathState = -1;
    if (s_cudaPathState == -1) {
        s_cudaPathState = (IsCudaBuildEnabled() && IsCudaRuntimeAvailable()) ? 1 : 0;
    }
    if (s_cudaPathState == 0) {
        return false;
    }

    const float safeRestDensity = std::max(0.01f, settings.restDensity);
    const float safeGamma = std::max(1.0f, settings.gamma);
    const float baseStiffness = std::max(0.0f, settings.stiffness);
    const float gravity = settings.enableGravity ? std::max(0.0f, settings.gravity) : 0.0f;
    const float targetDensityErrorRatio = std::clamp(settings.targetDensityErrorRatio, 0.001f, 0.2f);

    struct ParticleRefs {
        sim::ecs::TransformComponent* transform = nullptr;
        BallRenderComponent* render = nullptr;
        BallPhysicsComponent* physics = nullptr;
        float densityRadius = 0.0f;
        float densityMass = 0.0f;
    };

    auto view = world.Registry().view<sim::ecs::TransformComponent, BallRenderComponent, BallPhysicsComponent>();
    thread_local std::vector<ParticleRefs> particles;
    particles.clear();
    particles.reserve(std::max<std::size_t>(kParticleReserveHint, view.size_hint()));
    for (const auto entity : view) {
        auto& render = view.get<BallRenderComponent>(entity);
        if (!render.enabled) {
            continue;
        }
        particles.push_back(ParticleRefs{
            &view.get<sim::ecs::TransformComponent>(entity),
            &render,
            &view.get<BallPhysicsComponent>(entity),
            ComputeDensityKernelRadius(render),
            0.0f
        });
    }

    const float boundsHeight = std::max(1.0f, bounds.maxY - bounds.minY);
    const float fluidHeight = EstimateFluidColumnHeight(particles, bounds);
    const float referenceFluidHeight = particles.empty() ? boundsHeight : fluidHeight;
    const float recommendedGravityMax = ComputeWcSphRecommendedMaxGravity(
        safeRestDensity,
        baseStiffness,
        safeGamma,
        referenceFluidHeight,
        targetDensityErrorRatio
    );
    const float requiredStiffness = ComputeWcSphRequiredStiffnessForGravity(
        safeRestDensity,
        gravity,
        safeGamma,
        referenceFluidHeight,
        targetDensityErrorRatio
    );
    const float effectiveStiffness = ComputeEffectiveStiffness(baseStiffness, requiredStiffness);

    if (particles.empty()) {
        SetRuntimePhysicsStepDiagnostics(
            0,
            0.0f,
            0.0f,
            recommendedGravityMax,
            effectiveStiffness
        );
        return true;
    }

    const float safeMassDensityReference = std::max(0.01f, settings.massDensityReference);
    const float representativeArea = ComputeRepresentativeParticleArea(
        bounds,
        particles.size(),
        particles.front().render->radius
    );
    for (auto& particle : particles) {
        particle.densityMass = ComputeDensityParticleMass(safeMassDensityReference, representativeArea);
    }

    thread_local std::vector<CudaParticleState> cudaParticles;
    cudaParticles.clear();
    cudaParticles.reserve(particles.size());
    for (const auto& particle : particles) {
        const auto& transform = *particle.transform;
        const auto& physics = *particle.physics;
        cudaParticles.push_back(CudaParticleState{
            transform.position.x,
            transform.position.y,
            physics.velocity.x,
            physics.velocity.y,
            particle.render->radius,
            particle.densityRadius,
            particle.densityMass,
            physics.density,
            physics.pressure,
            physics.normalizedDensity
        });
    }

    CudaPhysicsParams cudaParams{};
    cudaParams.deltaSeconds = deltaSeconds;
    cudaParams.boundsMinX = bounds.minX;
    cudaParams.boundsMaxX = bounds.maxX;
    cudaParams.boundsMinY = bounds.minY;
    cudaParams.boundsMaxY = bounds.maxY;
    cudaParams.restDensity = safeRestDensity;
    cudaParams.stiffness = effectiveStiffness;
    cudaParams.gamma = safeGamma;
    cudaParams.viscosity = std::max(0.0f, settings.viscosity);
    cudaParams.gravity = gravity;
    cudaParams.enableGravity = settings.enableGravity ? 1 : 0;
    cudaParams.restitution = std::clamp(settings.restitution, 0.0f, 1.2f);
    cudaParams.linearDamping = std::max(0.0f, settings.linearDamping);
    cudaParams.maxVelocityLimit = std::max(0.0f, settings.maxVelocityLimit);
    cudaParams.maxPressureLimit = std::max(0.0f, settings.maxPressureLimit);
    cudaParams.mousePosX = std::isfinite(mouseInput.position.x) ? mouseInput.position.x : 0.0f;
    cudaParams.mousePosY = std::isfinite(mouseInput.position.y) ? mouseInput.position.y : 0.0f;
    cudaParams.mouseVelX = std::isfinite(mouseInput.velocity.x) ? mouseInput.velocity.x : 0.0f;
    cudaParams.mouseVelY = std::isfinite(mouseInput.velocity.y) ? mouseInput.velocity.y : 0.0f;
    cudaParams.mouseLeftPressed = mouseInput.leftPressed ? 1 : 0;
    cudaParams.mouseRightPressed = mouseInput.rightPressed ? 1 : 0;
    cudaParams.mouseForceRadius = std::max(1.0f, settings.mouseForceRadius);
    cudaParams.mouseForceStrength = std::max(0.0f, settings.mouseForceStrength);
    cudaParams.mousePushMachCap = std::clamp(settings.mousePushMachCap, 1.0f, 80.0f);
    cudaParams.xsphVelocityBlend = std::clamp(settings.xsphVelocityBlend, 0.0f, 0.5f);
    cudaParams.maxSubsteps = std::max(1, settings.maxSubsteps);
    cudaParams.cflFactor = std::clamp(settings.cflFactor, 0.05f, 1.0f);
    cudaParams.minTimeStep = std::max(0.0001f, settings.minTimeStep);
    cudaParams.maxTimeStep = std::max(cudaParams.minTimeStep, settings.maxTimeStep);

    CudaPhysicsDiagnostics cudaDiagnostics{};
    if (!RunCudaBallPhysics(cudaParticles, cudaParams, cudaDiagnostics)) {
        s_cudaPathState = 0;
        return false;
    }

    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto& transform = *particles[i].transform;
        auto& physics = *particles[i].physics;
        const auto& cudaState = cudaParticles[i];

        transform.position = {cudaState.posX, cudaState.posY};
        physics.velocity = {cudaState.velX, cudaState.velY};
        physics.density = cudaState.density;
        physics.pressure = cudaState.pressure;
        physics.normalizedDensity = cudaState.normalizedDensity;
    }

    SetRuntimePhysicsStepDiagnostics(
        cudaDiagnostics.substeps,
        cudaDiagnostics.lastSubstepMs,
        cudaDiagnostics.maxDensityErrorRatio,
        recommendedGravityMax,
        effectiveStiffness
    );
    return true;
}

void RunBallPhysicsSystem(
    sim::ecs::World& world,
    const UpdateSettings& settings,
    const MouseForceInput& mouseInput,
    float deltaSeconds
) {
    if (deltaSeconds <= 0.0f) {
        SetRuntimePhysicsStepDiagnostics(0, 0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const BoxBounds2D bounds = BuildPrimaryBounds(world);
    if (!bounds.valid) {
        SetRuntimePhysicsStepDiagnostics(0, 0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const float safeRestDensity = std::max(0.01f, settings.restDensity);
    const float safeGamma = std::max(1.0f, settings.gamma);
    const float gravity = settings.enableGravity ? std::max(0.0f, settings.gravity) : 0.0f;
    const float baseStiffness = std::max(0.0f, settings.stiffness);
    const float targetDensityErrorRatio = std::clamp(settings.targetDensityErrorRatio, 0.001f, 0.2f);

    const float safeMouseForceRadius = std::max(1.0f, settings.mouseForceRadius);
    const float safeMouseForceStrength = std::max(0.0f, settings.mouseForceStrength);
    const bool mouseForceActive =
        safeMouseForceStrength > 0.0f && (mouseInput.leftPressed != mouseInput.rightPressed);
    if (TryRunBallPhysicsSystemCuda(world, settings, mouseInput, deltaSeconds, bounds)) {
        return;
    }

    struct ParticleRefs {
        sim::ecs::TransformComponent* transform = nullptr;
        BallRenderComponent* render = nullptr;
        BallPhysicsComponent* physics = nullptr;
        float densityRadius = 0.0f;
        float densityMass = 0.0f;
    };

    const float safeMassDensityReference = std::max(0.01f, settings.massDensityReference);
    auto view = world.Registry().view<sim::ecs::TransformComponent, BallRenderComponent, BallPhysicsComponent>();
    thread_local std::vector<ParticleRefs> particles;
    particles.clear();
    particles.reserve(std::max<std::size_t>(kParticleReserveHint, view.size_hint()));
    for (const auto entity : view) {
        auto& render = view.get<BallRenderComponent>(entity);
        if (!render.enabled) {
            continue;
        }
        particles.push_back(ParticleRefs{
            &view.get<sim::ecs::TransformComponent>(entity),
            &render,
            &view.get<BallPhysicsComponent>(entity),
            ComputeDensityKernelRadius(render),
            0.0f
        });
    }

    const float boundsHeight = std::max(1.0f, bounds.maxY - bounds.minY);
    const float fluidHeight = EstimateFluidColumnHeight(particles, bounds);
    const float referenceFluidHeight = particles.empty() ? boundsHeight : fluidHeight;
    const float recommendedGravityMax = ComputeWcSphRecommendedMaxGravity(
        safeRestDensity,
        baseStiffness,
        safeGamma,
        referenceFluidHeight,
        targetDensityErrorRatio
    );
    const float requiredStiffness = ComputeWcSphRequiredStiffnessForGravity(
        safeRestDensity,
        gravity,
        safeGamma,
        referenceFluidHeight,
        targetDensityErrorRatio
    );
    const float effectiveStiffness = ComputeEffectiveStiffness(baseStiffness, requiredStiffness);

    if (particles.empty()) {
        SetRuntimePhysicsStepDiagnostics(
            0,
            0.0f,
            0.0f,
            recommendedGravityMax,
            effectiveStiffness
        );
        return;
    }

    const float representativeArea = ComputeRepresentativeParticleArea(
        bounds,
        particles.size(),
        particles.front().render->radius
    );
    float maxSupportRadius = 0.0f;
    for (auto& particle : particles) {
        particle.densityMass = ComputeDensityParticleMass(safeMassDensityReference, representativeArea);
        maxSupportRadius = std::max(maxSupportRadius, particle.densityRadius);
    }
    maxSupportRadius = std::max(0.001f, maxSupportRadius);

    const float safeStiffness = effectiveStiffness;
    const float safeViscosity = std::max(0.0f, settings.viscosity);
    const float safeMousePushMachCap = std::clamp(settings.mousePushMachCap, 1.0f, 80.0f);
    const float safeXsphVelocityBlend = std::clamp(settings.xsphVelocityBlend, 0.0f, 0.5f);
    const float restitution = std::clamp(settings.restitution, 0.0f, 1.2f);
    const float damping = std::max(0.0f, settings.linearDamping);
    const float densityFloor = std::max(0.001f, safeRestDensity * kDensityFloorRatio);
    const float speedOfSound = std::sqrt(std::max(0.0001f, safeStiffness * safeGamma / safeRestDensity));
    const float boundsWidth = std::max(1.0f, bounds.maxX - bounds.minX);
    const float boundsScale = std::max(boundsWidth, boundsHeight);
    const glm::vec2 boundsCenter{
        0.5f * (bounds.minX + bounds.maxX),
        0.5f * (bounds.minY + bounds.maxY)
    };
    const float maxAcceleration = std::max(
        20000.0f,
        std::max(
            safeStiffness * boundsScale,
            (safeStiffness / std::max(0.01f, safeRestDensity)) * boundsScale * 35.0f
        )
    );
    const float maxVelocityLimit = std::max(0.0f, settings.maxVelocityLimit);
    const bool velocityLimitEnabled = maxVelocityLimit > 0.0f;

    const int safeMaxSubsteps = std::max(1, settings.maxSubsteps);
    const float safeMinStep = std::max(0.0001f, settings.minTimeStep);
    const float safeMaxStep = std::max(safeMinStep, settings.maxTimeStep);
    const float maxSimulatedDelta = safeMaxStep * static_cast<float>(safeMaxSubsteps);
    const float simulatedDeltaSeconds = std::min(
        std::max(0.0f, deltaSeconds),
        maxSimulatedDelta
    );
    const int substepBudget = safeMaxSubsteps;
    float remainingTime = simulatedDeltaSeconds;
    int substepCount = 0;
    float lastStepSeconds = 0.0f;
    float maxDensityErrorRatio = 0.0f;

    const std::size_t particleCount = particles.size();
    thread_local std::vector<glm::vec2> positions;
    thread_local std::vector<glm::vec2> velocities;
    thread_local std::vector<float> densities;
    thread_local std::vector<float> pressures;
    thread_local std::vector<float> normalizedDensities;
    thread_local std::vector<float> safeDensities;
    thread_local std::vector<float> pressureOverDensitySquared;
    thread_local std::vector<float> densityRadii;
    thread_local std::vector<float> densityMasses;

    positions.resize(particleCount);
    velocities.resize(particleCount);
    densities.resize(particleCount);
    pressures.resize(particleCount);
    normalizedDensities.resize(particleCount);
    safeDensities.resize(particleCount);
    pressureOverDensitySquared.resize(particleCount);
    densityRadii.resize(particleCount);
    densityMasses.resize(particleCount);

    for (std::size_t i = 0; i < particleCount; ++i) {
        glm::vec2 currentPosition = particles[i].transform->position;
        if (!IsFiniteVec2(currentPosition)) {
            currentPosition = boundsCenter;
        }
        glm::vec2 currentVelocity = particles[i].physics->velocity;
        if (!IsFiniteVec2(currentVelocity)) {
            currentVelocity = {0.0f, 0.0f};
        }

        float density = particles[i].physics->density;
        if (!std::isfinite(density)) {
            density = safeRestDensity;
        }
        density = std::max(density, densityFloor);
        float pressure = particles[i].physics->pressure;
        if (!std::isfinite(pressure)) {
            pressure = 0.0f;
        }

        positions[i] = currentPosition;
        velocities[i] = currentVelocity;
        densities[i] = density;
        pressures[i] = pressure;
        normalizedDensities[i] = ClampUnitFloat(density / safeRestDensity);
        safeDensities[i] = std::max(densityFloor, density);
        const float invDensitySquared = 1.0f / (safeDensities[i] * safeDensities[i]);
        float pressureCoeff = pressure * invDensitySquared;
        if (!std::isfinite(pressureCoeff)) {
            pressureCoeff = 0.0f;
        }
        pressureOverDensitySquared[i] = pressureCoeff;
        densityRadii[i] = particles[i].densityRadius;
        densityMasses[i] = particles[i].densityMass;
    }

    thread_local std::vector<glm::vec2> stagedPositions;
    thread_local std::vector<glm::vec2> stagedVelocities;
    stagedPositions.resize(particleCount);
    stagedVelocities.resize(particleCount);

    thread_local UniformGrid2D particleGrid;
    thread_local std::vector<std::size_t> particleGridWriteOffsets;

    for (int step = 0; step < substepBudget && remainingTime > 1e-7f; ++step) {
        float stepSeconds = ComputeStableSubstep(
            densityRadii,
            velocities,
            settings,
            remainingTime,
            safeRestDensity,
            safeStiffness,
            safeGamma,
            gravity,
            mouseForceActive ? safeMouseForceStrength : 0.0f
        );
        const int stepsLeft = std::max(1, substepBudget - step);
        const float minStepForCoverage = remainingTime / static_cast<float>(stepsLeft);
        stepSeconds = std::max(stepSeconds, minStepForCoverage);
        stepSeconds = std::max(safeMinStep, stepSeconds);
        stepSeconds = std::min(safeMaxStep, stepSeconds);
        stepSeconds = std::min(stepSeconds, remainingTime);
        remainingTime -= stepSeconds;
        lastStepSeconds = stepSeconds;
        ++substepCount;

        BuildUniformGrid2DInPlace(
            particleGrid,
            particleGridWriteOffsets,
            bounds.minX,
            bounds.maxX,
            bounds.minY,
            bounds.maxY,
            maxSupportRadius,
            particleCount,
            [&](std::size_t index) {
                return positions[index];
            }
        );

        for (std::size_t i = 0; i < particleCount; ++i) {
            glm::vec2 pi = positions[i];
            if (!IsFiniteVec2(pi)) {
                pi = boundsCenter;
                positions[i] = pi;
            }
            if (!IsFiniteVec2(velocities[i])) {
                velocities[i] = {0.0f, 0.0f};
            }

            float density = 0.0f;
            const float densityRadiusI = densityRadii[i];
            VisitGridCandidates(particleGrid, pi, maxSupportRadius, [&](std::size_t j) {
                const glm::vec2 delta = pi - positions[j];
                const float h = 0.5f * (densityRadiusI + densityRadii[j]);
                if (std::abs(delta.x) >= h || std::abs(delta.y) >= h) {
                    return;
                }
                const float distSquared = delta.x * delta.x + delta.y * delta.y;
                if (distSquared >= h * h) {
                    return;
                }
                density += densityMasses[j] * ComputeNormalizedCompactKernel2D(distSquared, h);
            });
            density += ComputeBoundaryDensityGhostContribution(
                pi,
                bounds,
                densityRadiusI,
                densityMasses[i]
            );
            if (!std::isfinite(density)) {
                density = safeRestDensity;
            }
            density = std::max(density, densityFloor);
            const float pressure = ComputeEquationOfStatePressure(
                density,
                safeRestDensity,
                safeStiffness,
                safeGamma,
                settings.maxPressureLimit
            );

            densities[i] = density;
            pressures[i] = pressure;
            normalizedDensities[i] = ClampUnitFloat(density / safeRestDensity);
            safeDensities[i] = std::max(densityFloor, density);
            const float invDensitySquared = 1.0f / (safeDensities[i] * safeDensities[i]);
            float pressureCoeff = pressure * invDensitySquared;
            if (!std::isfinite(pressureCoeff)) {
                pressureCoeff = 0.0f;
            }
            pressureOverDensitySquared[i] = pressureCoeff;
            maxDensityErrorRatio = std::max(
                maxDensityErrorRatio,
                std::abs(density - safeRestDensity) / safeRestDensity
            );
        }

        const float decay = std::max(0.0f, 1.0f - damping * stepSeconds);
        for (std::size_t i = 0; i < particleCount; ++i) {
            const glm::vec2 currentPosition = positions[i];
            const glm::vec2 currentVelocity = velocities[i];
            const float safeDensityI = safeDensities[i];
            const float pressureI = pressures[i];
            const float pressureCoeffI = pressureOverDensitySquared[i];

            glm::vec2 pressureAcceleration{0.0f, 0.0f};
            glm::vec2 viscosityAcceleration{0.0f, 0.0f};
            glm::vec2 xsphVelocityDelta{0.0f, 0.0f};

            VisitGridCandidates(particleGrid, currentPosition, maxSupportRadius, [&](std::size_t j) {
                if (i == j) {
                    return;
                }

                const glm::vec2 delta = currentPosition - positions[j];
                const float h = 0.5f * (densityRadii[i] + densityRadii[j]);
                if (std::abs(delta.x) >= h || std::abs(delta.y) >= h) {
                    return;
                }
                const float r2 = glm::dot(delta, delta);
                if (r2 <= 1e-10f || r2 >= h * h) {
                    return;
                }

                const glm::vec2 gradW = ComputeNormalizedCompactKernelGradient2D(delta, h);
                const float safeDensityJ = safeDensities[j];
                const float pressureTerm = pressureCoeffI + pressureOverDensitySquared[j];
                pressureAcceleration += -densityMasses[j] * pressureTerm * gradW;
                if (safeXsphVelocityBlend > 0.0f) {
                    const glm::vec2 velocityDelta = velocities[j] - currentVelocity;
                    const float rhoBar = 0.5f * (safeDensityI + safeDensityJ);
                    const float kernelWeight = ComputeNormalizedCompactKernel2D(r2, h);
                    if (rhoBar > 0.0f && kernelWeight > 0.0f) {
                        xsphVelocityDelta += (densityMasses[j] / rhoBar) * velocityDelta * kernelWeight;
                    }
                }

                if (safeViscosity > 0.0f) {
                    const glm::vec2 vij = currentVelocity - velocities[j];
                    const float vijDotRij = glm::dot(vij, delta);
                    if (vijDotRij < 0.0f) {
                        const float mu = h * vijDotRij / (r2 + kArtificialViscosityEpsilon * h * h);
                        const float rhoBar = 0.5f * (safeDensityI + safeDensityJ);
                        const float piViscosity = (-safeViscosity * speedOfSound * mu) / std::max(0.001f, rhoBar);
                        viscosityAcceleration += -densityMasses[j] * piViscosity * gradW;
                    }
                }
            });

            AccumulateBoundaryPressureGhostAcceleration(
                pressureAcceleration,
                currentPosition,
                bounds,
                densityRadii[i],
                densityMasses[i],
                pressureI,
                safeDensityI
            );

            if (!IsFiniteVec2(pressureAcceleration)) {
                pressureAcceleration = {0.0f, 0.0f};
            }
            if (!IsFiniteVec2(viscosityAcceleration)) {
                viscosityAcceleration = {0.0f, 0.0f};
            }
            if (!IsFiniteVec2(xsphVelocityDelta)) {
                xsphVelocityDelta = {0.0f, 0.0f};
            }
            glm::vec2 mouseAcceleration = mouseForceActive
                ? ComputeMouseForceAcceleration(
                    currentPosition,
                    currentVelocity,
                    mouseInput,
                    safeMouseForceRadius,
                    safeMouseForceStrength
                )
                : glm::vec2{0.0f, 0.0f};
            if (!IsFiniteVec2(mouseAcceleration)) {
                mouseAcceleration = {0.0f, 0.0f};
            }
            pressureAcceleration = ClampVectorMagnitude(pressureAcceleration, maxAcceleration);
            viscosityAcceleration = ClampVectorMagnitude(viscosityAcceleration, maxAcceleration);
            mouseAcceleration = ClampVectorMagnitude(mouseAcceleration, maxAcceleration);
            glm::vec2 nextVelocity = currentVelocity * decay;
            nextVelocity += (pressureAcceleration + viscosityAcceleration + mouseAcceleration) * stepSeconds;
            nextVelocity += safeXsphVelocityBlend * xsphVelocityDelta;
            nextVelocity.y += gravity * stepSeconds;
            if (!IsFiniteVec2(nextVelocity)) {
                nextVelocity = {0.0f, 0.0f};
            }
            const bool leftPushOnlyActive = mouseForceActive && mouseInput.leftPressed && !mouseInput.rightPressed;
            if (leftPushOnlyActive) {
                const float leftPushRadiusSquared = safeMouseForceRadius * safeMouseForceRadius;
                const float distanceToMouseSquared = glm::dot(currentPosition - mouseInput.position, currentPosition - mouseInput.position);
                if (distanceToMouseSquared < leftPushRadiusSquared) {
                    const float nominalPushSpeedCap = std::max(
                        120.0f,
                        std::sqrt(std::max(0.0f, safeMouseForceStrength * safeMouseForceRadius)) * 1.35f
                    );
                    const float acousticPushSpeedCap = std::max(80.0f, safeMousePushMachCap * speedOfSound);
                    const float leftPushSpeedCap = std::min(nominalPushSpeedCap, acousticPushSpeedCap);
                    nextVelocity = ClampVectorMagnitude(nextVelocity, leftPushSpeedCap);
                }
            }
            if (velocityLimitEnabled) {
                nextVelocity = ClampVectorMagnitude(nextVelocity, maxVelocityLimit);
            }
            glm::vec2 nextPosition = currentPosition + nextVelocity * stepSeconds;
            if (!IsFiniteVec2(nextPosition)) {
                nextPosition = boundsCenter;
                nextVelocity = {0.0f, 0.0f};
            }

            ResolveParticleBoundsCollision(
                nextPosition,
                particles[i].render->radius,
                nextVelocity,
                bounds,
                restitution
            );
            stagedPositions[i] = nextPosition;
            stagedVelocities[i] = nextVelocity;
        }

        for (std::size_t i = 0; i < particleCount; ++i) {
            positions[i] = stagedPositions[i];
            velocities[i] = stagedVelocities[i];
        }
    }

    for (std::size_t i = 0; i < particleCount; ++i) {
        particles[i].transform->position = positions[i];
        auto& physics = *particles[i].physics;
        physics.velocity = velocities[i];
        physics.density = densities[i];
        physics.pressure = pressures[i];
        physics.normalizedDensity = normalizedDensities[i];

        if (!IsFiniteVec2(particles[i].transform->position)) {
            particles[i].transform->position = boundsCenter;
        }
        if (!IsFiniteVec2(physics.velocity)) {
            physics.velocity = {0.0f, 0.0f};
        }
    }

    SetRuntimePhysicsStepDiagnostics(
        substepCount,
        lastStepSeconds * 1000.0f,
        maxDensityErrorRatio,
        recommendedGravityMax,
        effectiveStiffness
    );
}

}  // namespace

void ResetWorldToInitialTransforms(
    sim::ecs::World& world,
    std::span<const sim::ecs::Entity> ballEntities,
    sim::ecs::Entity boundingBoxEntity
) {
    constexpr float kSpawnHalfWidth = 450.0f;
    constexpr float kSpawnHalfHeight = 270.0f;
    std::random_device randomDevice;
    std::mt19937 rng(randomDevice());
    const float minSpawnDistance = std::max(2.0f, BallRenderComponent{}.radius * 2.25f);
    const std::vector<glm::vec2> spawnPositions = GenerateRandomSpawnPositions(
        ballEntities.size(),
        kSpawnHalfWidth,
        kSpawnHalfHeight,
        minSpawnDistance,
        rng
    );

    for (std::size_t i = 0; i < ballEntities.size(); ++i) {
        const auto ballEntity = ballEntities[i];
        auto& ballTransform = world.GetComponent<sim::ecs::TransformComponent>(ballEntity);
        ballTransform.position = spawnPositions[i];
        ballTransform.scale = {1.0f, 1.0f};
        ballTransform.rotationRadians = 0.0f;

        world.AddComponent<BallRenderComponent>(ballEntity);
        auto& physics = world.AddComponent<BallPhysicsComponent>(ballEntity);
        physics.velocity = {0.0f, 0.0f};
        physics.lastAnchorCenter = {0.0f, 0.0f};
        physics.lastUiOffset = {0.0f, 0.0f};
        physics.density = 0.0f;
        physics.pressure = 0.0f;
        physics.normalizedDensity = 0.0f;
        physics.anchorInitialized = true;
    }

    auto& boxTransform = world.GetComponent<sim::ecs::TransformComponent>(boundingBoxEntity);
    boxTransform.position = {0.0f, 0.0f};
    boxTransform.scale = {1.0f, 1.0f};
    boxTransform.rotationRadians = 0.0f;

    world.AddComponent<BoundingBoxRenderComponent>(boundingBoxEntity);
}

void ResetToInitialState(
    UiState& state,
    sim::ecs::World& world,
    std::span<const sim::ecs::Entity> ballEntities,
    sim::ecs::Entity boundingBoxEntity
) {
    state = UiState{};
    ResetWorldToInitialTransforms(world, ballEntities, boundingBoxEntity);
}

void UpdateWorld(
    sim::ecs::World& world,
    const UiState& state,
    Uint32 viewportWidth,
    Uint32 viewportHeight,
    float deltaSeconds,
    const MouseForceInput& mouseInput
) {
    BeginRuntimeUpdateProfiling();
    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateSceneSync);
        RunSceneStateSyncSystem(world, state, viewportWidth, viewportHeight);
    }
    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateBallValidation);
        RunBallRenderValidationSystem(world);
    }
    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateBoxValidation);
        RunBoundingBoxValidationSystem(world);
    }
    const float restDensity = std::max(0.01f, state.update.restDensity);
    const float massDensityReference = std::max(0.01f, state.update.massDensityReference);
    const bool running = state.runtime.state == RuntimeState::Running;
    SetRuntimePhysicsStepDiagnostics(0, 0.0f, 0.0f, 0.0f, 0.0f);

    if (running) {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateBallPhysics);
        RunBallPhysicsSystem(world, state.update, mouseInput, deltaSeconds);
    } else {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateParticleDensity);
        RunParticleDensitySystem(world, restDensity, massDensityReference);
    }

    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::UpdateTransformNormalize);
        RunTransformNormalizationSystem(world);
    }
    EndRuntimeUpdateProfiling();
}

void DrawWorld2D(sim::renderer::GpuRenderer& renderer, sim::ecs::World& world, const UiState& state) {
    BeginRuntimeDrawProfiling();
    Uint32 viewportWidth = 0;
    Uint32 viewportHeight = 0;
    (void)renderer.GetWindowResolution(viewportWidth, viewportHeight);

    const bool densityViewEnabled = state.render.densityView;
    if (densityViewEnabled) {
        sim::renderer::DensityFieldOverlayStyle2D overlayStyle{};
        const float restDensity = std::max(0.01f, state.update.restDensity);
        overlayStyle.targetDensity = restDensity;
        float effectiveSampleStep = std::max(1.0f, state.update.densityViewSampleStep);
        sim::renderer::DensityFieldGrid2D overlayGrid{};
        {
            RuntimeScopedMetricTimer timer(RuntimeProfileMetric::DrawDensityViewGrid);
            overlayGrid = BuildDensityFieldGrid(
                world,
                viewportWidth,
                viewportHeight,
                effectiveSampleStep,
                restDensity,
                std::max(0.01f, state.update.massDensityReference),
                effectiveSampleStep
            );
        }
        {
            RuntimeScopedMetricTimer timer(RuntimeProfileMetric::DrawDensityViewOverlay);
            renderer.DrawDensityFieldOverlay2D(overlayGrid, overlayStyle);
        }
    }

    auto ballView = world.Registry().view<const sim::ecs::TransformComponent, const BallRenderComponent, const BallPhysicsComponent>();
    const bool densityFieldEnabled = state.render.densityField;
    const bool drawBallEnabled = state.render.ball;
    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::DrawBall);
        if (!densityViewEnabled && drawBallEnabled) {
            const float configuredSpeedRange = std::max(0.0f, state.update.maxVelocityLimit);
            const float speedReference = configuredSpeedRange > 0.0f
                ? configuredSpeedRange
                : kVelocityColorReferenceSpeed;
            const int safeSegments = std::clamp(state.world.ballSegments, 6, 96);
            const std::size_t particleCountHint =
                static_cast<std::size_t>(std::max(0, state.world.particleCount));
            const std::size_t estimatedCircleFillVertices =
                particleCountHint * static_cast<std::size_t>(safeSegments) * 3u;
            // Above budget, switch to quad billboards so fill vertices scale as 6 * N.
            const bool useParticleBillboards = estimatedCircleFillVertices > kRendererFillVertexBudget;

            renderer.SetObjectQuality(1.0f);
            if (!densityFieldEnabled && !useParticleBillboards) {
                renderer.DisableObjectParticle();
                renderer.SetObjectDensity(1.0f);
            }

            for (const auto entity : ballView) {
                const auto& transform = ballView.get<const sim::ecs::TransformComponent>(entity);
                const auto& render = ballView.get<const BallRenderComponent>(entity);
                const auto& physics = ballView.get<const BallPhysicsComponent>(entity);
                if (!render.enabled) {
                    continue;
                }
                const float objectDensity = densityFieldEnabled
                    ? std::clamp(0.35f + physics.normalizedDensity * 1.65f, 0.05f, 3.0f)
                    : 1.0f;
                const float speed = glm::length(physics.velocity);
                const float speedMix = ComputeVelocityColorMix(speed, speedReference);
                const sim::renderer::Color drawColor = BuildVelocityGradientColor(render.color, speedMix);

                if (useParticleBillboards) {
                    const float particleRadius = std::max(1.0f, render.radius);
                    const float smoothingRadius = densityFieldEnabled
                        ? std::max(particleRadius, render.smoothingRadius)
                        : (particleRadius + kParticleBillboardEdgeFeatherPx);
                    const glm::vec2 quadSize{
                        smoothingRadius * 2.0f,
                        smoothingRadius * 2.0f
                    };
                    renderer.SetObjectDensity(objectDensity);
                    renderer.SetObjectParticle(transform.position, particleRadius, smoothingRadius);
                    renderer.DrawRect2D(
                        transform.position,
                        quadSize,
                        drawColor,
                        false,
                        0.0f,
                        0.0f
                    );
                } else if (densityFieldEnabled) {
                    renderer.SetObjectDensity(objectDensity);
                    renderer.SetObjectParticle(transform.position, render.radius, render.smoothingRadius);
                    renderer.DrawCircle2D(
                        transform.position,
                        render.smoothingRadius,
                        drawColor,
                        false,
                        render.segments
                    );
                } else {
                    renderer.DrawCircle2D(
                        transform.position,
                        render.radius,
                        drawColor,
                        false,
                        render.segments
                    );
                }
            }
        }
    }

    renderer.ResetObjectShaderParams();

    {
        RuntimeScopedMetricTimer timer(RuntimeProfileMetric::DrawBox);
        auto boxView = world.Registry().view<const sim::ecs::TransformComponent, const BoundingBoxRenderComponent>();
        for (const auto entity : boxView) {
            const auto& transform = boxView.get<const sim::ecs::TransformComponent>(entity);
            const auto& render = boxView.get<const BoundingBoxRenderComponent>(entity);
            if (!render.enabled) {
                continue;
            }
            renderer.DrawRect2D(
                transform.position,
                render.size,
                render.color,
                true,
                0.0f,
                render.lineThickness
            );
        }
    }
    EndRuntimeDrawProfiling();
}

}  // namespace sim::runtime
