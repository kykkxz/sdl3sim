#pragma once

#include <glm/vec2.hpp>

namespace sim::runtime {

constexpr int kMinParticleCount = 50;
constexpr int kDefaultParticleCount = 1200;
constexpr int kMaxParticleCount = 20000;

struct RenderSwitches {
    bool ball = true;
    bool boundingBox = true;
    bool densityField = true;
    bool densityView = false;
};

struct UpdateSettings {
    bool enableGravity = true;
    float gravity = 90.0f;
    float restitution = 0.35f;
    float linearDamping = 0.1f;
    float maxVelocityLimit = 0.0f;
    float mouseForceRadius = 120.0f;
    float mouseForceStrength = 9000.0f;
    float mousePushMachCap = 20.0f;
    float xsphVelocityBlend = 0.10f;
    float maxPressureLimit = 0.0f;
    float smoothingRadius = 40.0f;
    float massDensityReference = 60.0f;
    float restDensity = 60.0f;
    float stiffness = 1800.0f;
    float gamma = 7.0f;
    float targetDensityErrorRatio = 0.02f;
    float viscosity = 0.04f;
    float cflFactor = 0.4f;
    int maxSubsteps = 12;
    float minTimeStep = 1.0f / 1000.0f;
    float maxTimeStep = 1.0f / 120.0f;
    float densityColorMix = 0.65f;
    float densityViewSampleStep = 9.0f;
};

struct WorldSettings {
    int particleCount = kDefaultParticleCount;
    float ballOffsetX = 0.0f;
    float ballOffsetY = 0.0f;
    float ballRadius = 4.0f;
    int ballSegments = 24;
    float ballColor[3]{0.3f, 0.65f, 1.0f};

    float boxInset = 12.0f;
    float boxLineThickness = 3.0f;
    float boxColor[3]{1.0f, 0.8f, 0.25f};
};

enum class RuntimeState {
    Running,
    Paused
};

struct RuntimeControls {
    RuntimeState state = RuntimeState::Paused;
};

struct ShaderGlobalSettings {
    float density = 1.0f;
    float quality = 1.0f;
};

struct MouseForceInput {
    glm::vec2 position{0.0f, 0.0f};
    glm::vec2 velocity{0.0f, 0.0f};
    bool leftPressed = false;
    bool rightPressed = false;
};

struct UiState {
    RenderSwitches render{};
    UpdateSettings update{};
    WorldSettings world{};
    RuntimeControls runtime{};
    ShaderGlobalSettings shaderGlobals{};
    float clearColor[3]{0.08f, 0.11f, 0.16f};
};

struct EditorActions {
    bool resetToInitialState = false;
    bool rebuildParticleEntities = false;
};

}  // namespace sim::runtime
