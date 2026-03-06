#pragma once

#include <vector>

namespace sim::runtime {

struct CudaParticleState {
    float posX = 0.0f;
    float posY = 0.0f;
    float velX = 0.0f;
    float velY = 0.0f;
    float renderRadius = 0.0f;
    float densityRadius = 0.0f;
    float densityMass = 0.0f;
    float density = 0.0f;
    float pressure = 0.0f;
    float normalizedDensity = 0.0f;
};

struct CudaPhysicsParams {
    float deltaSeconds = 0.0f;
    float boundsMinX = 0.0f;
    float boundsMaxX = 0.0f;
    float boundsMinY = 0.0f;
    float boundsMaxY = 0.0f;
    float restDensity = 60.0f;
    float stiffness = 140.0f;
    float gamma = 7.0f;
    float viscosity = 0.04f;
    float gravity = 0.0f;
    int enableGravity = 0;
    float restitution = 0.35f;
    float linearDamping = 0.1f;
    float maxVelocityLimit = 0.0f;
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
    int maxSubsteps = 1;
    float cflFactor = 0.4f;
    float minTimeStep = 1.0f / 1000.0f;
    float maxTimeStep = 1.0f / 120.0f;
};

struct CudaPhysicsDiagnostics {
    int substeps = 0;
    float lastSubstepMs = 0.0f;
    float maxDensityErrorRatio = 0.0f;
};

bool IsCudaBuildEnabled();
bool IsCudaRuntimeAvailable();
bool RunCudaBallPhysics(
    std::vector<CudaParticleState>& particles,
    const CudaPhysicsParams& params,
    CudaPhysicsDiagnostics& outDiagnostics
);

}  // namespace sim::runtime
