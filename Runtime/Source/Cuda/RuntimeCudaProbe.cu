#include "RuntimeCuda.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kDensityFloorRatio = 0.1f;
constexpr float kMaximumNegativePressureRatio = 0.25f;
constexpr float kMaximumPressureToStiffnessRatio = 400.0f;
constexpr float kArtificialViscosityEpsilon = 0.01f;
constexpr float kPi = 3.14159265358979323846f;

struct DevicePhysicsParams {
    float boundsMinX = 0.0f;
    float boundsMaxX = 0.0f;
    float boundsMinY = 0.0f;
    float boundsMaxY = 0.0f;
    float boundsCenterX = 0.0f;
    float boundsCenterY = 0.0f;
    int boundsValid = 0;
    float restDensity = 60.0f;
    float stiffness = 140.0f;
    float gamma = 7.0f;
    float viscosity = 0.04f;
    float gravity = 0.0f;
    int enableGravity = 0;
    float restitution = 0.35f;
    float linearDamping = 0.1f;
    float maxVelocityLimit = 0.0f;
    int velocityLimitEnabled = 0;
    float maxPressureLimit = 0.0f;
    float mousePosX = 0.0f;
    float mousePosY = 0.0f;
    float mouseVelX = 0.0f;
    float mouseVelY = 0.0f;
    int mouseLeftPressed = 0;
    int mouseRightPressed = 0;
    float mouseForceRadius = 120.0f;
    float mouseForceStrength = 0.0f;
    float mousePushMachCap = 20.0f;
    float xsphVelocityBlend = 0.0f;
    float densityFloor = 0.001f;
    float speedOfSound = 1.0f;
    float maxAcceleration = 20000.0f;
};

struct DeviceUniformGrid2D {
    float minX = 0.0f;
    float minY = 0.0f;
    float cellSize = 1.0f;
    float invCellSize = 1.0f;
    int cols = 1;
    int rows = 1;
};

struct DeviceParticleBuffers {
    sim::runtime::CudaParticleState* a = nullptr;
    sim::runtime::CudaParticleState* b = nullptr;
    int capacity = 0;
};

DeviceParticleBuffers g_deviceParticleBuffers{};

struct DeviceNeighborGridBuffers {
    int* cellCounts = nullptr;
    int* cellStarts = nullptr;
    int* cellWriteOffsets = nullptr;
    int* particleIndices = nullptr;
    void* scanTempStorage = nullptr;
    size_t scanTempStorageBytes = 0;
    int cellCapacity = 0;
    int particleCapacity = 0;
};

DeviceNeighborGridBuffers g_deviceNeighborGridBuffers{};

bool IsFiniteHostFloat(float value) {
    return std::isfinite(static_cast<double>(value));
}

void ReleaseDeviceParticleBuffers() {
    if (g_deviceParticleBuffers.a != nullptr) {
        cudaFree(g_deviceParticleBuffers.a);
        g_deviceParticleBuffers.a = nullptr;
    }
    if (g_deviceParticleBuffers.b != nullptr) {
        cudaFree(g_deviceParticleBuffers.b);
        g_deviceParticleBuffers.b = nullptr;
    }
    g_deviceParticleBuffers.capacity = 0;

    if (g_deviceNeighborGridBuffers.cellCounts != nullptr) {
        cudaFree(g_deviceNeighborGridBuffers.cellCounts);
        g_deviceNeighborGridBuffers.cellCounts = nullptr;
    }
    if (g_deviceNeighborGridBuffers.cellStarts != nullptr) {
        cudaFree(g_deviceNeighborGridBuffers.cellStarts);
        g_deviceNeighborGridBuffers.cellStarts = nullptr;
    }
    if (g_deviceNeighborGridBuffers.cellWriteOffsets != nullptr) {
        cudaFree(g_deviceNeighborGridBuffers.cellWriteOffsets);
        g_deviceNeighborGridBuffers.cellWriteOffsets = nullptr;
    }
    if (g_deviceNeighborGridBuffers.particleIndices != nullptr) {
        cudaFree(g_deviceNeighborGridBuffers.particleIndices);
        g_deviceNeighborGridBuffers.particleIndices = nullptr;
    }
    if (g_deviceNeighborGridBuffers.scanTempStorage != nullptr) {
        cudaFree(g_deviceNeighborGridBuffers.scanTempStorage);
        g_deviceNeighborGridBuffers.scanTempStorage = nullptr;
    }
    g_deviceNeighborGridBuffers.scanTempStorageBytes = 0;
    g_deviceNeighborGridBuffers.cellCapacity = 0;
    g_deviceNeighborGridBuffers.particleCapacity = 0;
}

bool EnsureDeviceParticleBuffers(int count) {
    if (count <= 0) {
        return false;
    }

    static bool s_releaseRegistered = false;
    if (!s_releaseRegistered) {
        std::atexit(ReleaseDeviceParticleBuffers);
        s_releaseRegistered = true;
    }

    if (g_deviceParticleBuffers.capacity >= count
        && g_deviceParticleBuffers.a != nullptr
        && g_deviceParticleBuffers.b != nullptr) {
        return true;
    }

    ReleaseDeviceParticleBuffers();
    const size_t bytes = static_cast<size_t>(count) * sizeof(sim::runtime::CudaParticleState);
    if (cudaMalloc(&g_deviceParticleBuffers.a, bytes) != cudaSuccess) {
        ReleaseDeviceParticleBuffers();
        return false;
    }
    if (cudaMalloc(&g_deviceParticleBuffers.b, bytes) != cudaSuccess) {
        ReleaseDeviceParticleBuffers();
        return false;
    }

    g_deviceParticleBuffers.capacity = count;
    return true;
}

bool EnsureDeviceNeighborGridBuffers(int particleCount, int cellCount) {
    if (particleCount <= 0 || cellCount <= 0) {
        return false;
    }

    const bool particleCapacityOk =
        g_deviceNeighborGridBuffers.particleCapacity >= particleCount
        && g_deviceNeighborGridBuffers.particleIndices != nullptr;
    const bool cellCapacityOk =
        g_deviceNeighborGridBuffers.cellCapacity >= cellCount
        && g_deviceNeighborGridBuffers.cellCounts != nullptr
        && g_deviceNeighborGridBuffers.cellStarts != nullptr
        && g_deviceNeighborGridBuffers.cellWriteOffsets != nullptr;

    if (!particleCapacityOk) {
        if (g_deviceNeighborGridBuffers.particleIndices != nullptr) {
            cudaFree(g_deviceNeighborGridBuffers.particleIndices);
            g_deviceNeighborGridBuffers.particleIndices = nullptr;
        }
        if (cudaMalloc(
                &g_deviceNeighborGridBuffers.particleIndices,
                static_cast<size_t>(particleCount) * sizeof(int)
            ) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return false;
        }
        g_deviceNeighborGridBuffers.particleCapacity = particleCount;
    }

    if (!cellCapacityOk) {
        if (g_deviceNeighborGridBuffers.cellCounts != nullptr) {
            cudaFree(g_deviceNeighborGridBuffers.cellCounts);
            g_deviceNeighborGridBuffers.cellCounts = nullptr;
        }
        if (g_deviceNeighborGridBuffers.cellStarts != nullptr) {
            cudaFree(g_deviceNeighborGridBuffers.cellStarts);
            g_deviceNeighborGridBuffers.cellStarts = nullptr;
        }
        if (g_deviceNeighborGridBuffers.cellWriteOffsets != nullptr) {
            cudaFree(g_deviceNeighborGridBuffers.cellWriteOffsets);
            g_deviceNeighborGridBuffers.cellWriteOffsets = nullptr;
        }

        const size_t cellBytes = static_cast<size_t>(cellCount) * sizeof(int);
        if (cudaMalloc(&g_deviceNeighborGridBuffers.cellCounts, cellBytes) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return false;
        }
        if (cudaMalloc(&g_deviceNeighborGridBuffers.cellStarts, cellBytes) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return false;
        }
        if (cudaMalloc(&g_deviceNeighborGridBuffers.cellWriteOffsets, cellBytes) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return false;
        }
        g_deviceNeighborGridBuffers.cellCapacity = cellCount;
    }

    size_t requiredScanTempStorageBytes = 0;
    const cudaError_t scanQueryResult = cub::DeviceScan::ExclusiveSum(
        nullptr,
        requiredScanTempStorageBytes,
        g_deviceNeighborGridBuffers.cellCounts,
        g_deviceNeighborGridBuffers.cellStarts,
        cellCount
    );
    if (scanQueryResult != cudaSuccess) {
        ReleaseDeviceParticleBuffers();
        return false;
    }

    const bool scanTempStorageOk =
        g_deviceNeighborGridBuffers.scanTempStorage != nullptr
        && g_deviceNeighborGridBuffers.scanTempStorageBytes >= requiredScanTempStorageBytes;
    if (!scanTempStorageOk) {
        if (g_deviceNeighborGridBuffers.scanTempStorage != nullptr) {
            cudaFree(g_deviceNeighborGridBuffers.scanTempStorage);
            g_deviceNeighborGridBuffers.scanTempStorage = nullptr;
            g_deviceNeighborGridBuffers.scanTempStorageBytes = 0;
        }
        if (requiredScanTempStorageBytes > 0
            && cudaMalloc(
                &g_deviceNeighborGridBuffers.scanTempStorage,
                requiredScanTempStorageBytes
            ) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return false;
        }
        g_deviceNeighborGridBuffers.scanTempStorageBytes = requiredScanTempStorageBytes;
    }

    return true;
}

float ComputeStableSubstepSeconds(
    const sim::runtime::CudaParticleState* particles,
    int count,
    const sim::runtime::CudaPhysicsParams& params,
    const DevicePhysicsParams& deviceParams,
    float remainingTime
) {
    if (particles == nullptr || count <= 0 || remainingTime <= 0.0f) {
        return 0.0f;
    }

    float minSmoothing = 0.001f;
    float maxSpeed = 0.0f;
    bool hasValidSmoothing = false;
    for (int i = 0; i < count; ++i) {
        const sim::runtime::CudaParticleState& particle = particles[i];
        float smoothing = particle.densityRadius;
        if (!IsFiniteHostFloat(smoothing)) {
            smoothing = 0.001f;
        }
        smoothing = fmaxf(0.001f, smoothing);
        if (!hasValidSmoothing) {
            minSmoothing = smoothing;
            hasValidSmoothing = true;
        } else {
            minSmoothing = fminf(minSmoothing, smoothing);
        }

        float speedSquared = particle.velX * particle.velX + particle.velY * particle.velY;
        if (!IsFiniteHostFloat(speedSquared) || speedSquared < 0.0f) {
            speedSquared = 0.0f;
        }
        const float speed = sqrtf(speedSquared);
        if (IsFiniteHostFloat(speed)) {
            maxSpeed = fmaxf(maxSpeed, speed);
        }
    }
    minSmoothing = fmaxf(0.001f, minSmoothing);

    const float safeCfl = fminf(1.0f, fmaxf(0.05f, params.cflFactor));
    const float safeViscosity = fmaxf(0.0f, params.viscosity);
    const float safeRestDensity = fmaxf(0.01f, params.restDensity);
    const float safeStiffness = fmaxf(0.0f, params.stiffness);
    const float safeGravity = IsFiniteHostFloat(params.gravity) ? fabsf(params.gravity) : 0.0f;
    const bool mouseForceActive =
        (params.mouseLeftPressed != params.mouseRightPressed) && (params.mouseForceStrength > 0.0f);
    const float safeMouseForceStrength = mouseForceActive ? fmaxf(0.0f, params.mouseForceStrength) : 0.0f;
    const float dtCfl = safeCfl * minSmoothing / fmaxf(0.001f, deviceParams.speedOfSound + maxSpeed);
    const float dtViscosity = safeViscosity > 0.0f
        ? 0.125f * minSmoothing * minSmoothing / fmaxf(0.0001f, safeViscosity)
        : 3.402823466e+38f;
    const float accelerationScale = fmaxf(
        0.0001f,
        safeGravity + safeMouseForceStrength + safeStiffness / safeRestDensity
    );
    const float dtForce = 0.25f * sqrtf(minSmoothing / accelerationScale);
    const float safeMinStep = fmaxf(0.0001f, params.minTimeStep);
    const float safeMaxStep = fmaxf(safeMinStep, params.maxTimeStep);
    float stepSeconds = fminf(
        remainingTime,
        fminf(dtCfl, fminf(dtViscosity, fminf(dtForce, safeMaxStep)))
    );
    stepSeconds = fmaxf(stepSeconds, safeMinStep);
    return fminf(stepSeconds, remainingTime);
}

__device__ inline float ClampUnitFloat(float value) {
    return fminf(1.0f, fmaxf(0.0f, value));
}

__device__ inline bool IsFiniteFloat(float value) {
    return isfinite(value) != 0;
}

__device__ inline bool IsFiniteVec2(float x, float y) {
    return IsFiniteFloat(x) && IsFiniteFloat(y);
}

__device__ inline void ClampVectorMagnitude(float& x, float& y, float maxMagnitude) {
    const float safeMaxMagnitude = fmaxf(0.0001f, maxMagnitude);
    const float len2 = x * x + y * y;
    if (!IsFiniteFloat(len2) || len2 <= safeMaxMagnitude * safeMaxMagnitude) {
        return;
    }
    const float len = sqrtf(fmaxf(0.0f, len2));
    if (len <= 1e-8f || !IsFiniteFloat(len)) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    const float scale = safeMaxMagnitude / len;
    x *= scale;
    y *= scale;
}

__device__ inline void ComputeMouseForceAcceleration(
    float particlePosX,
    float particlePosY,
    float particleVelX,
    float particleVelY,
    const DevicePhysicsParams& params,
    float& outAx,
    float& outAy
) {
    outAx = 0.0f;
    outAy = 0.0f;

    const int forceDirection = params.mouseRightPressed - params.mouseLeftPressed;
    const float forceRadius = params.mouseForceRadius;
    const float forceStrength = params.mouseForceStrength;
    if (forceDirection == 0 || forceRadius <= 0.0f || forceStrength <= 0.0f) {
        return;
    }

    const float toMouseX = params.mousePosX - particlePosX;
    const float toMouseY = params.mousePosY - particlePosY;
    const float distanceSquared = toMouseX * toMouseX + toMouseY * toMouseY;
    const float distance = sqrtf(fmaxf(0.0f, distanceSquared));

    const float springCoefficient = forceStrength / fmaxf(1.0f, forceRadius);
    const float dampingCoefficient = 2.0f * sqrtf(fmaxf(0.0f, springCoefficient));
    float safeMouseVelX = params.mouseVelX;
    float safeMouseVelY = params.mouseVelY;
    if (!IsFiniteVec2(safeMouseVelX, safeMouseVelY)) {
        safeMouseVelX = 0.0f;
        safeMouseVelY = 0.0f;
    }

    if (forceDirection > 0) {
        const float captureRadius = forceRadius * 2.2f;
        const float captureRadiusSquared = captureRadius * captureRadius;
        if (distanceSquared >= captureRadiusSquared) {
            return;
        }

        const float nearInfluence = fminf(1.0f, fmaxf(0.0f, 1.0f - (distance / forceRadius)));
        const float captureInfluence = fminf(1.0f, fmaxf(0.0f, 1.0f - (distance / captureRadius)));
        const float influence = fmaxf(nearInfluence, captureInfluence * captureInfluence);
        if (influence <= 0.0f) {
            return;
        }

        const float normalizedDistance = fminf(1.0f, fmaxf(0.0f, distance / captureRadius));
        const float distanceGain = 1.0f + 1.6f * normalizedDistance;
        const float tunedSpring = springCoefficient * distanceGain;
        const float tunedDamping = dampingCoefficient * sqrtf(distanceGain);
        const float springAx = toMouseX * tunedSpring;
        const float springAy = toMouseY * tunedSpring;
        const float dampingAx = (safeMouseVelX - particleVelX) * tunedDamping;
        const float dampingAy = (safeMouseVelY - particleVelY) * tunedDamping;
        outAx = (springAx + dampingAx) * influence;
        outAy = (springAy + dampingAy) * influence;
        return;
    }

    const float radiusSquared = forceRadius * forceRadius;
    if (distanceSquared >= radiusSquared) {
        return;
    }
    const float influence = fminf(1.0f, fmaxf(0.0f, 1.0f - (distance / forceRadius)));
    if (influence <= 0.0f || distanceSquared <= 1e-8f) {
        return;
    }

    const float invDistance = 1.0f / distance;
    const float directionToMouseX = toMouseX * invDistance;
    const float directionToMouseY = toMouseY * invDistance;
    const float outwardDirectionX = -directionToMouseX;
    const float outwardDirectionY = -directionToMouseY;
    const float nominalPushSpeed = sqrtf(fmaxf(0.0f, forceStrength * forceRadius));
    const float desiredPushSpeed = nominalPushSpeed * sqrtf(influence);
    const float targetVelX = outwardDirectionX * desiredPushSpeed;
    const float targetVelY = outwardDirectionY * desiredPushSpeed;
    const float velocityTrackingGain = dampingCoefficient * 0.85f;
    const float velocityTrackingAx = (targetVelX - particleVelX) * velocityTrackingGain;
    const float velocityTrackingAy = (targetVelY - particleVelY) * velocityTrackingGain;
    const float radialAssistScale = forceStrength * 0.25f * influence * influence;
    const float radialAssistAx = outwardDirectionX * radialAssistScale;
    const float radialAssistAy = outwardDirectionY * radialAssistScale;
    outAx = velocityTrackingAx + radialAssistAx;
    outAy = velocityTrackingAy + radialAssistAy;
}

__device__ inline float ComputeNormalizedCompactKernel2D(float distanceSquared, float smoothingLength) {
    if (smoothingLength <= 0.0f) {
        return 0.0f;
    }

    const float h2 = smoothingLength * smoothingLength;
    if (distanceSquared >= h2) {
        return 0.0f;
    }

    const float distance = sqrtf(fmaxf(0.0f, distanceSquared));
    const float q = 1.0f - distance / smoothingLength;
    const float normalization = 6.0f / (kPi * h2);
    return normalization * q * q;
}

__device__ inline void ComputeNormalizedCompactKernelGradient2D(
    float deltaX,
    float deltaY,
    float smoothingLength,
    float& outX,
    float& outY
) {
    outX = 0.0f;
    outY = 0.0f;
    if (smoothingLength <= 0.0f) {
        return;
    }

    const float distanceSquared = deltaX * deltaX + deltaY * deltaY;
    const float h2 = smoothingLength * smoothingLength;
    if (distanceSquared <= 1e-10f || distanceSquared >= h2) {
        return;
    }

    const float distance = sqrtf(distanceSquared);
    const float q = 1.0f - distance / smoothingLength;
    const float normalization = 6.0f / (kPi * h2);
    const float dWdr = -2.0f * normalization * q / smoothingLength;
    const float invDistance = 1.0f / distance;
    outX = dWdr * deltaX * invDistance;
    outY = dWdr * deltaY * invDistance;
}

__device__ inline float ComputeEquationOfStatePressure(
    float density,
    float restDensity,
    float stiffness,
    float gamma,
    float maxPressureLimit
) {
    const float safeRestDensity = fmaxf(0.01f, restDensity);
    const float safeStiffness = fmaxf(0.0f, stiffness);
    const float safeGamma = fmaxf(1.0f, gamma);
    float ratio = density / safeRestDensity;
    if (!IsFiniteFloat(ratio)) {
        ratio = 1.0f;
    }
    ratio = fmaxf(0.001f, ratio);
    float pressure = safeStiffness * (powf(ratio, safeGamma) - 1.0f);
    const float minPressure = -kMaximumNegativePressureRatio * safeStiffness;
    const float baselineMaxPressure = fmaxf(4000.0f, safeStiffness * kMaximumPressureToStiffnessRatio);
    const float safeMaxPressureLimit = fmaxf(0.0f, maxPressureLimit);
    const bool pressureLimitEnabled = safeMaxPressureLimit > 0.0f;
    const float maxPressure = pressureLimitEnabled ? safeMaxPressureLimit : baselineMaxPressure;
    if (!IsFiniteFloat(pressure)) {
        pressure = maxPressure;
    }
    if (pressureLimitEnabled) {
        return fminf(maxPressure, fmaxf(minPressure, pressure));
    }
    return fmaxf(minPressure, pressure);
}

__device__ inline float ComputeBoundaryDensityGhostContribution(
    float posX,
    float posY,
    const DevicePhysicsParams& params,
    float smoothingLength,
    float mass
) {
    if (!params.boundsValid || smoothingLength <= 0.0f) {
        return 0.0f;
    }

    float contribution = 0.0f;
    const float h2 = smoothingLength * smoothingLength;
    auto addMirror = [&](float distanceToWall) {
        if (distanceToWall <= 0.0f) {
            return;
        }
        const float mirrorDistance = 2.0f * distanceToWall;
        if (mirrorDistance >= smoothingLength) {
            return;
        }
        const float mirrorDistanceSquared = mirrorDistance * mirrorDistance;
        if (mirrorDistanceSquared >= h2) {
            return;
        }
        contribution += mass * ComputeNormalizedCompactKernel2D(
            mirrorDistanceSquared,
            smoothingLength
        );
    };

    addMirror(posX - params.boundsMinX);
    addMirror(params.boundsMaxX - posX);
    addMirror(posY - params.boundsMinY);
    addMirror(params.boundsMaxY - posY);
    return contribution;
}

__device__ inline void AccumulateBoundaryPressureGhostAcceleration(
    float& pressureAx,
    float& pressureAy,
    float posX,
    float posY,
    const DevicePhysicsParams& params,
    float smoothingLength,
    float mass,
    float pressure,
    float density
) {
    if (!params.boundsValid || smoothingLength <= 0.0f || pressure <= 0.0f || density <= 0.0f) {
        return;
    }

    const float safeDensity = fmaxf(0.001f, density);
    const float pressureTerm = 2.0f * pressure / (safeDensity * safeDensity);
    const float h2 = smoothingLength * smoothingLength;
    auto addMirror = [&](float mirrorX, float mirrorY) {
        const float deltaX = posX - mirrorX;
        const float deltaY = posY - mirrorY;
        if (fabsf(deltaX) >= smoothingLength || fabsf(deltaY) >= smoothingLength) {
            return;
        }
        const float distSquared = deltaX * deltaX + deltaY * deltaY;
        if (distSquared <= 1e-10f || distSquared >= h2) {
            return;
        }
        float gradX = 0.0f;
        float gradY = 0.0f;
        ComputeNormalizedCompactKernelGradient2D(deltaX, deltaY, smoothingLength, gradX, gradY);
        pressureAx += -mass * pressureTerm * gradX;
        pressureAy += -mass * pressureTerm * gradY;
    };

    addMirror(2.0f * params.boundsMinX - posX, posY);
    addMirror(2.0f * params.boundsMaxX - posX, posY);
    addMirror(posX, 2.0f * params.boundsMinY - posY);
    addMirror(posX, 2.0f * params.boundsMaxY - posY);
}

__device__ inline void ResolveParticleBoundsCollision(
    float& posX,
    float& posY,
    float renderRadius,
    float& velX,
    float& velY,
    const DevicePhysicsParams& params
) {
    if (!params.boundsValid) {
        return;
    }

    const float radius = fmaxf(1.0f, renderRadius);
    if (posX - radius < params.boundsMinX) {
        posX = params.boundsMinX + radius;
        velX = fabsf(velX) * params.restitution;
    }
    if (posX + radius > params.boundsMaxX) {
        posX = params.boundsMaxX - radius;
        velX = -fabsf(velX) * params.restitution;
    }
    if (posY - radius < params.boundsMinY) {
        posY = params.boundsMinY + radius;
        velY = fabsf(velY) * params.restitution;
    }
    if (posY + radius > params.boundsMaxY) {
        posY = params.boundsMaxY - radius;
        velY = -fabsf(velY) * params.restitution;
        if (fabsf(velY) < 2.0f) {
            velY = 0.0f;
        }
    }
}

__device__ inline int GridClampCol(const DeviceUniformGrid2D& grid, float x) {
    const int col = static_cast<int>(floorf((x - grid.minX) * grid.invCellSize));
    return max(0, min(grid.cols - 1, col));
}

__device__ inline int GridClampRow(const DeviceUniformGrid2D& grid, float y) {
    const int row = static_cast<int>(floorf((y - grid.minY) * grid.invCellSize));
    return max(0, min(grid.rows - 1, row));
}

__device__ inline int GridCellIndex(int row, int col, int cols) {
    return row * cols + col;
}

__global__ void CountParticlesPerCellKernel(
    const sim::runtime::CudaParticleState* particles,
    int particleCount,
    DeviceUniformGrid2D grid,
    int* cellCounts
) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= particleCount) {
        return;
    }

    const sim::runtime::CudaParticleState p = particles[i];
    if (!IsFiniteVec2(p.posX, p.posY)) {
        return;
    }

    const int col = GridClampCol(grid, p.posX);
    const int row = GridClampRow(grid, p.posY);
    const int cell = GridCellIndex(row, col, grid.cols);
    atomicAdd(&cellCounts[cell], 1);
}

__global__ void FillParticlesPerCellKernel(
    const sim::runtime::CudaParticleState* particles,
    int particleCount,
    DeviceUniformGrid2D grid,
    int* cellWriteOffsets,
    int* particleIndices
) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= particleCount) {
        return;
    }

    const sim::runtime::CudaParticleState p = particles[i];
    if (!IsFiniteVec2(p.posX, p.posY)) {
        return;
    }

    const int col = GridClampCol(grid, p.posX);
    const int row = GridClampRow(grid, p.posY);
    const int cell = GridCellIndex(row, col, grid.cols);
    const int writeIndex = atomicAdd(&cellWriteOffsets[cell], 1);
    particleIndices[writeIndex] = i;
}

__global__ void DensityPressureKernel(
    const sim::runtime::CudaParticleState* inParticles,
    sim::runtime::CudaParticleState* outParticles,
    int particleCount,
    DevicePhysicsParams params,
    DeviceUniformGrid2D grid,
    const int* cellStarts,
    const int* cellCounts,
    const int* particleIndices,
    float queryRadius
) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= particleCount) {
        return;
    }

    sim::runtime::CudaParticleState pi = inParticles[i];
    if (!IsFiniteVec2(pi.posX, pi.posY)) {
        pi.posX = params.boundsCenterX;
        pi.posY = params.boundsCenterY;
    }
    float density = 0.0f;

    const int baseCol = GridClampCol(grid, pi.posX);
    const int baseRow = GridClampRow(grid, pi.posY);
    const int cellRange = max(1, static_cast<int>(ceilf(fmaxf(0.0f, queryRadius) * grid.invCellSize)));
    const int minRow = max(0, baseRow - cellRange);
    const int maxRow = min(grid.rows - 1, baseRow + cellRange);
    const int minCol = max(0, baseCol - cellRange);
    const int maxCol = min(grid.cols - 1, baseCol + cellRange);

    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            const int cell = GridCellIndex(row, col, grid.cols);
            const int start = cellStarts[cell];
            const int end = start + cellCounts[cell];
            for (int k = start; k < end; ++k) {
                const int j = particleIndices[k];
                const sim::runtime::CudaParticleState pj = inParticles[j];
                const float deltaX = pi.posX - pj.posX;
                const float deltaY = pi.posY - pj.posY;
                const float h = 0.5f * (pi.densityRadius + pj.densityRadius);
                if (fabsf(deltaX) >= h || fabsf(deltaY) >= h) {
                    continue;
                }
                const float distSquared = deltaX * deltaX + deltaY * deltaY;
                if (distSquared >= h * h) {
                    continue;
                }
                density += pj.densityMass * ComputeNormalizedCompactKernel2D(distSquared, h);
            }
        }
    }

    density += ComputeBoundaryDensityGhostContribution(
        pi.posX,
        pi.posY,
        params,
        pi.densityRadius,
        pi.densityMass
    );
    if (!IsFiniteFloat(density)) {
        density = params.restDensity;
    }
    density = fmaxf(density, params.densityFloor);

    pi.density = density;
    pi.pressure = ComputeEquationOfStatePressure(
        density,
        params.restDensity,
        params.stiffness,
        params.gamma,
        params.maxPressureLimit
    );
    pi.normalizedDensity = ClampUnitFloat(density / params.restDensity);
    outParticles[i] = pi;
}

__global__ void IntegrateKernel(
    const sim::runtime::CudaParticleState* inParticles,
    sim::runtime::CudaParticleState* outParticles,
    int particleCount,
    DevicePhysicsParams params,
    DeviceUniformGrid2D grid,
    const int* cellStarts,
    const int* cellCounts,
    const int* particleIndices,
    float queryRadius,
    float stepSeconds
) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= particleCount) {
        return;
    }

    sim::runtime::CudaParticleState pi = inParticles[i];
    if (!IsFiniteVec2(pi.posX, pi.posY)) {
        pi.posX = params.boundsCenterX;
        pi.posY = params.boundsCenterY;
    }
    if (!IsFiniteVec2(pi.velX, pi.velY)) {
        pi.velX = 0.0f;
        pi.velY = 0.0f;
    }
    const float safeDensityI = fmaxf(params.densityFloor, pi.density);
    const float pressureI = pi.pressure;

    float pressureAx = 0.0f;
    float pressureAy = 0.0f;
    float viscosityAx = 0.0f;
    float viscosityAy = 0.0f;
    float xsphDeltaX = 0.0f;
    float xsphDeltaY = 0.0f;
    const float safeXsphVelocityBlend = fminf(0.5f, fmaxf(0.0f, params.xsphVelocityBlend));

    const int baseCol = GridClampCol(grid, pi.posX);
    const int baseRow = GridClampRow(grid, pi.posY);
    const int cellRange = max(1, static_cast<int>(ceilf(fmaxf(0.0f, queryRadius) * grid.invCellSize)));
    const int minRow = max(0, baseRow - cellRange);
    const int maxRow = min(grid.rows - 1, baseRow + cellRange);
    const int minCol = max(0, baseCol - cellRange);
    const int maxCol = min(grid.cols - 1, baseCol + cellRange);
    for (int row = minRow; row <= maxRow; ++row) {
        for (int col = minCol; col <= maxCol; ++col) {
            const int cell = GridCellIndex(row, col, grid.cols);
            const int start = cellStarts[cell];
            const int end = start + cellCounts[cell];
            for (int k = start; k < end; ++k) {
                const int j = particleIndices[k];
                if (i == j) {
                    continue;
                }

                const sim::runtime::CudaParticleState pj = inParticles[j];
                const float deltaX = pi.posX - pj.posX;
                const float deltaY = pi.posY - pj.posY;
                const float h = 0.5f * (pi.densityRadius + pj.densityRadius);
                if (fabsf(deltaX) >= h || fabsf(deltaY) >= h) {
                    continue;
                }
                const float r2 = deltaX * deltaX + deltaY * deltaY;
                if (r2 <= 1e-10f || r2 >= h * h) {
                    continue;
                }

                float gradX = 0.0f;
                float gradY = 0.0f;
                ComputeNormalizedCompactKernelGradient2D(deltaX, deltaY, h, gradX, gradY);
                const float safeDensityJ = fmaxf(params.densityFloor, pj.density);
                const float pressureJ = pj.pressure;
                const float pressureTerm =
                    (pressureI / (safeDensityI * safeDensityI))
                    + (pressureJ / (safeDensityJ * safeDensityJ));
                pressureAx += -pj.densityMass * pressureTerm * gradX;
                pressureAy += -pj.densityMass * pressureTerm * gradY;
                if (safeXsphVelocityBlend > 0.0f) {
                    const float velocityDeltaX = pj.velX - pi.velX;
                    const float velocityDeltaY = pj.velY - pi.velY;
                    const float rhoBar = 0.5f * (safeDensityI + safeDensityJ);
                    const float kernelWeight = ComputeNormalizedCompactKernel2D(r2, h);
                    if (rhoBar > 0.0f && kernelWeight > 0.0f) {
                        xsphDeltaX += (pj.densityMass / rhoBar) * velocityDeltaX * kernelWeight;
                        xsphDeltaY += (pj.densityMass / rhoBar) * velocityDeltaY * kernelWeight;
                    }
                }

                if (params.viscosity > 0.0f) {
                    const float vijX = pi.velX - pj.velX;
                    const float vijY = pi.velY - pj.velY;
                    const float vijDotRij = vijX * deltaX + vijY * deltaY;
                    if (vijDotRij < 0.0f) {
                        const float mu = h * vijDotRij / (r2 + kArtificialViscosityEpsilon * h * h);
                        const float rhoBar = 0.5f * (safeDensityI + safeDensityJ);
                        const float piViscosity =
                            (-params.viscosity * params.speedOfSound * mu) / fmaxf(0.001f, rhoBar);
                        viscosityAx += -pj.densityMass * piViscosity * gradX;
                        viscosityAy += -pj.densityMass * piViscosity * gradY;
                    }
                }
            }
        }
    }

    AccumulateBoundaryPressureGhostAcceleration(
        pressureAx,
        pressureAy,
        pi.posX,
        pi.posY,
        params,
        pi.densityRadius,
        pi.densityMass,
        pressureI,
        safeDensityI
    );

    if (!IsFiniteVec2(pressureAx, pressureAy)) {
        pressureAx = 0.0f;
        pressureAy = 0.0f;
    }
    if (!IsFiniteVec2(viscosityAx, viscosityAy)) {
        viscosityAx = 0.0f;
        viscosityAy = 0.0f;
    }
    if (!IsFiniteVec2(xsphDeltaX, xsphDeltaY)) {
        xsphDeltaX = 0.0f;
        xsphDeltaY = 0.0f;
    }

    float mouseAx = 0.0f;
    float mouseAy = 0.0f;
    ComputeMouseForceAcceleration(
        pi.posX,
        pi.posY,
        pi.velX,
        pi.velY,
        params,
        mouseAx,
        mouseAy
    );
    if (!IsFiniteVec2(mouseAx, mouseAy)) {
        mouseAx = 0.0f;
        mouseAy = 0.0f;
    }

    ClampVectorMagnitude(pressureAx, pressureAy, params.maxAcceleration);
    ClampVectorMagnitude(viscosityAx, viscosityAy, params.maxAcceleration);
    ClampVectorMagnitude(mouseAx, mouseAy, params.maxAcceleration);

    const float decay = fmaxf(0.0f, 1.0f - params.linearDamping * stepSeconds);
    float outVelX = pi.velX * decay;
    float outVelY = pi.velY * decay;
    outVelX += (pressureAx + viscosityAx + mouseAx) * stepSeconds;
    outVelY += (pressureAy + viscosityAy + mouseAy) * stepSeconds;
    outVelX += safeXsphVelocityBlend * xsphDeltaX;
    outVelY += safeXsphVelocityBlend * xsphDeltaY;
    if (params.enableGravity) {
        outVelY += params.gravity * stepSeconds;
    }

    if (!IsFiniteVec2(outVelX, outVelY)) {
        outVelX = 0.0f;
        outVelY = 0.0f;
    }
    const bool leftPushOnlyActive =
        params.mouseLeftPressed != 0 && params.mouseRightPressed == 0 && params.mouseForceStrength > 0.0f;
    if (leftPushOnlyActive) {
        const float leftPushRadiusSquared = params.mouseForceRadius * params.mouseForceRadius;
        const float dxMouse = pi.posX - params.mousePosX;
        const float dyMouse = pi.posY - params.mousePosY;
        const float distanceToMouseSquared = dxMouse * dxMouse + dyMouse * dyMouse;
        if (distanceToMouseSquared < leftPushRadiusSquared) {
            const float nominalPushSpeedCap = fmaxf(
                120.0f,
                sqrtf(fmaxf(0.0f, params.mouseForceStrength * params.mouseForceRadius)) * 1.35f
            );
            const float acousticPushSpeedCap = fmaxf(80.0f, params.mousePushMachCap * params.speedOfSound);
            const float leftPushSpeedCap = fminf(nominalPushSpeedCap, acousticPushSpeedCap);
            ClampVectorMagnitude(outVelX, outVelY, leftPushSpeedCap);
        }
    }
    if (params.velocityLimitEnabled) {
        ClampVectorMagnitude(outVelX, outVelY, params.maxVelocityLimit);
    }

    float outPosX = pi.posX + outVelX * stepSeconds;
    float outPosY = pi.posY + outVelY * stepSeconds;
    if (!IsFiniteVec2(outPosX, outPosY)) {
        outPosX = params.boundsCenterX;
        outPosY = params.boundsCenterY;
        outVelX = 0.0f;
        outVelY = 0.0f;
    }

    ResolveParticleBoundsCollision(
        outPosX,
        outPosY,
        pi.renderRadius,
        outVelX,
        outVelY,
        params
    );

    sim::runtime::CudaParticleState out = pi;
    out.posX = outPosX;
    out.posY = outPosY;
    out.velX = outVelX;
    out.velY = outVelY;
    outParticles[i] = out;
}

}  // namespace

extern "C" int SimCudaProbeKernel() {
    int deviceCount = 0;
    const cudaError_t queryResult = cudaGetDeviceCount(&deviceCount);
    if (queryResult != cudaSuccess) {
        return 0;
    }
    return deviceCount > 0 ? 1 : 0;
}

extern "C" int SimCudaRunBallPhysics(
    sim::runtime::CudaParticleState* particles,
    int count,
    const sim::runtime::CudaPhysicsParams* params,
    sim::runtime::CudaPhysicsDiagnostics* diagnostics
) {
    if (particles == nullptr || count <= 0 || params == nullptr || diagnostics == nullptr) {
        return 0;
    }

    const size_t bytes = static_cast<size_t>(count) * sizeof(sim::runtime::CudaParticleState);

    if (!EnsureDeviceParticleBuffers(count)) {
        return 0;
    }
    if (cudaMemcpy(g_deviceParticleBuffers.a, particles, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        ReleaseDeviceParticleBuffers();
        return 0;
    }

    DevicePhysicsParams deviceParams{};
    deviceParams.boundsMinX = fminf(params->boundsMinX, params->boundsMaxX);
    deviceParams.boundsMaxX = fmaxf(params->boundsMinX, params->boundsMaxX);
    deviceParams.boundsMinY = fminf(params->boundsMinY, params->boundsMaxY);
    deviceParams.boundsMaxY = fmaxf(params->boundsMinY, params->boundsMaxY);
    deviceParams.boundsCenterX = 0.5f * (deviceParams.boundsMinX + deviceParams.boundsMaxX);
    deviceParams.boundsCenterY = 0.5f * (deviceParams.boundsMinY + deviceParams.boundsMaxY);
    deviceParams.boundsValid = 1;
    deviceParams.restDensity = fmaxf(0.01f, params->restDensity);
    deviceParams.stiffness = fmaxf(0.0f, params->stiffness);
    deviceParams.gamma = fmaxf(1.0f, params->gamma);
    deviceParams.viscosity = fmaxf(0.0f, params->viscosity);
    deviceParams.enableGravity = params->enableGravity != 0 ? 1 : 0;
    deviceParams.gravity = deviceParams.enableGravity ? fmaxf(0.0f, params->gravity) : 0.0f;
    deviceParams.restitution = fminf(1.2f, fmaxf(0.0f, params->restitution));
    deviceParams.linearDamping = fmaxf(0.0f, params->linearDamping);
    deviceParams.maxVelocityLimit = fmaxf(0.0f, params->maxVelocityLimit);
    deviceParams.velocityLimitEnabled = deviceParams.maxVelocityLimit > 0.0f ? 1 : 0;
    deviceParams.maxPressureLimit = fmaxf(0.0f, params->maxPressureLimit);
    deviceParams.mousePosX = IsFiniteHostFloat(params->mousePosX) ? params->mousePosX : 0.0f;
    deviceParams.mousePosY = IsFiniteHostFloat(params->mousePosY) ? params->mousePosY : 0.0f;
    deviceParams.mouseVelX = IsFiniteHostFloat(params->mouseVelX) ? params->mouseVelX : 0.0f;
    deviceParams.mouseVelY = IsFiniteHostFloat(params->mouseVelY) ? params->mouseVelY : 0.0f;
    deviceParams.mouseLeftPressed = params->mouseLeftPressed != 0 ? 1 : 0;
    deviceParams.mouseRightPressed = params->mouseRightPressed != 0 ? 1 : 0;
    deviceParams.mouseForceRadius = fmaxf(1.0f, params->mouseForceRadius);
    deviceParams.mouseForceStrength = fmaxf(0.0f, params->mouseForceStrength);
    deviceParams.mousePushMachCap = fminf(80.0f, fmaxf(1.0f, params->mousePushMachCap));
    deviceParams.xsphVelocityBlend = fminf(0.5f, fmaxf(0.0f, params->xsphVelocityBlend));
    deviceParams.densityFloor = fmaxf(0.001f, deviceParams.restDensity * kDensityFloorRatio);
    deviceParams.speedOfSound = sqrtf(
        fmaxf(
            0.0001f,
            deviceParams.stiffness * deviceParams.gamma / deviceParams.restDensity
        )
    );
    const float boundsWidth = fmaxf(1.0f, deviceParams.boundsMaxX - deviceParams.boundsMinX);
    const float boundsHeight = fmaxf(1.0f, deviceParams.boundsMaxY - deviceParams.boundsMinY);
    const float boundsScale = fmaxf(boundsWidth, boundsHeight);
    deviceParams.maxAcceleration = fmaxf(
        20000.0f,
        fmaxf(
            deviceParams.stiffness * boundsScale,
            (deviceParams.stiffness / fmaxf(0.01f, deviceParams.restDensity)) * boundsScale * 35.0f
        )
    );

    const float totalDeltaRaw = fmaxf(0.0f, params->deltaSeconds);
    const int safeMaxSubsteps = params->maxSubsteps > 0 ? params->maxSubsteps : 1;
    const float safeMinStep = fmaxf(0.0001f, params->minTimeStep);
    const float safeMaxStep = fmaxf(safeMinStep, params->maxTimeStep);
    const float maxSimulatedDelta = safeMaxStep * static_cast<float>(safeMaxSubsteps);
    const float totalDelta = fminf(totalDeltaRaw, maxSimulatedDelta);
    const int substepBudget = safeMaxSubsteps;
    const int threadsPerBlock = 128;
    const int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    float maxSupportRadius = 0.001f;
    for (int i = 0; i < count; ++i) {
        float supportRadius = particles[i].densityRadius;
        if (!IsFiniteHostFloat(supportRadius)) {
            supportRadius = 0.001f;
        }
        maxSupportRadius = fmaxf(maxSupportRadius, fmaxf(0.001f, supportRadius));
    }

    DeviceUniformGrid2D grid{};
    grid.minX = deviceParams.boundsMinX;
    grid.minY = deviceParams.boundsMinY;
    grid.cellSize = fmaxf(0.001f, maxSupportRadius);
    grid.invCellSize = 1.0f / grid.cellSize;
    const float gridWidth = fmaxf(0.001f, deviceParams.boundsMaxX - deviceParams.boundsMinX);
    const float gridHeight = fmaxf(0.001f, deviceParams.boundsMaxY - deviceParams.boundsMinY);
    grid.cols = std::max(1, static_cast<int>(ceilf(gridWidth / grid.cellSize)));
    grid.rows = std::max(1, static_cast<int>(ceilf(gridHeight / grid.cellSize)));
    const int cellCount = grid.cols * grid.rows;
    if (!EnsureDeviceNeighborGridBuffers(count, cellCount)) {
        return 0;
    }

    sim::runtime::CudaParticleState* current = g_deviceParticleBuffers.a;
    sim::runtime::CudaParticleState* next = g_deviceParticleBuffers.b;
    float remaining = totalDelta;
    float lastStep = 0.0f;
    int executedSubsteps = 0;
    const float stableSubstepSeconds = ComputeStableSubstepSeconds(
        particles,
        count,
        *params,
        deviceParams,
        totalDelta
    );
    for (int step = 0; step < substepBudget && remaining > 1e-7f; ++step) {
        float stepSeconds = fminf(stableSubstepSeconds, remaining);
        const int stepsLeft = (substepBudget - step) > 0 ? (substepBudget - step) : 1;
        const float minStepForCoverage = remaining / static_cast<float>(stepsLeft);
        stepSeconds = fmaxf(stepSeconds, minStepForCoverage);
        stepSeconds = fmaxf(safeMinStep, stepSeconds);
        stepSeconds = fminf(stepSeconds, safeMaxStep);
        stepSeconds = fminf(stepSeconds, remaining);
        remaining -= stepSeconds;
        lastStep = stepSeconds;
        ++executedSubsteps;

        const size_t gridCellBytes = static_cast<size_t>(cellCount) * sizeof(int);
        if (cudaMemset(g_deviceNeighborGridBuffers.cellCounts, 0, gridCellBytes) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }

        CountParticlesPerCellKernel<<<blocks, threadsPerBlock>>>(
            current,
            count,
            grid,
            g_deviceNeighborGridBuffers.cellCounts
        );
        if (cudaPeekAtLastError() != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }

        const cudaError_t scanResult = cub::DeviceScan::ExclusiveSum(
            g_deviceNeighborGridBuffers.scanTempStorage,
            g_deviceNeighborGridBuffers.scanTempStorageBytes,
            g_deviceNeighborGridBuffers.cellCounts,
            g_deviceNeighborGridBuffers.cellStarts,
            cellCount
        );
        if (scanResult != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }

        if (cudaMemcpy(
                g_deviceNeighborGridBuffers.cellWriteOffsets,
                g_deviceNeighborGridBuffers.cellStarts,
                gridCellBytes,
                cudaMemcpyDeviceToDevice
            ) != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }

        FillParticlesPerCellKernel<<<blocks, threadsPerBlock>>>(
            current,
            count,
            grid,
            g_deviceNeighborGridBuffers.cellWriteOffsets,
            g_deviceNeighborGridBuffers.particleIndices
        );
        if (cudaPeekAtLastError() != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }

        DensityPressureKernel<<<blocks, threadsPerBlock>>>(
            current,
            next,
            count,
            deviceParams,
            grid,
            g_deviceNeighborGridBuffers.cellStarts,
            g_deviceNeighborGridBuffers.cellCounts,
            g_deviceNeighborGridBuffers.particleIndices,
            maxSupportRadius
        );
        if (cudaPeekAtLastError() != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }
        sim::runtime::CudaParticleState* temp = current;
        current = next;
        next = temp;

        IntegrateKernel<<<blocks, threadsPerBlock>>>(
            current,
            next,
            count,
            deviceParams,
            grid,
            g_deviceNeighborGridBuffers.cellStarts,
            g_deviceNeighborGridBuffers.cellCounts,
            g_deviceNeighborGridBuffers.particleIndices,
            maxSupportRadius,
            stepSeconds
        );
        if (cudaPeekAtLastError() != cudaSuccess) {
            ReleaseDeviceParticleBuffers();
            return 0;
        }
        temp = current;
        current = next;
        next = temp;
    }

    if (cudaMemcpy(particles, current, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        ReleaseDeviceParticleBuffers();
        return 0;
    }

    float maxDensityErrorRatio = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float density = particles[i].density;
        const float errorRatio = fabsf(density - deviceParams.restDensity) / deviceParams.restDensity;
        maxDensityErrorRatio = fmaxf(maxDensityErrorRatio, errorRatio);
    }
    diagnostics->substeps = executedSubsteps;
    diagnostics->lastSubstepMs = lastStep * 1000.0f;
    diagnostics->maxDensityErrorRatio = maxDensityErrorRatio;

    return 1;
}
