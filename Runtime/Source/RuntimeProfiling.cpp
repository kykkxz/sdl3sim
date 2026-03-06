#include "RuntimeProfiling.hpp"

#include <algorithm>

namespace sim::runtime {

namespace {

using ProfileClock = std::chrono::steady_clock;

struct RuntimeProfilingState {
    RuntimeFrameProfiling frame{};
    ProfileClock::time_point updateStart{};
    ProfileClock::time_point drawStart{};
};

RuntimeProfilingState g_runtimeProfilingState{};

float ToMilliseconds(ProfileClock::duration duration) {
    return std::chrono::duration<float, std::milli>(duration).count();
}

void SetMetricValue(RuntimeProfileMetric metric, float valueMs) {
    RuntimeFrameProfiling& frame = g_runtimeProfilingState.frame;
    switch (metric) {
    case RuntimeProfileMetric::UpdateSceneSync:
        frame.updateSceneSyncMs = valueMs;
        break;
    case RuntimeProfileMetric::UpdateBallValidation:
        frame.updateBallValidationMs = valueMs;
        break;
    case RuntimeProfileMetric::UpdateBoxValidation:
        frame.updateBoxValidationMs = valueMs;
        break;
    case RuntimeProfileMetric::UpdateBallPhysics:
        frame.updateBallPhysicsMs = valueMs;
        break;
    case RuntimeProfileMetric::UpdateParticleDensity:
        frame.updateParticleDensityMs = valueMs;
        break;
    case RuntimeProfileMetric::UpdateTransformNormalize:
        frame.updateTransformNormalizeMs = valueMs;
        break;
    case RuntimeProfileMetric::DrawDensityViewGrid:
        frame.drawDensityViewGridMs = valueMs;
        break;
    case RuntimeProfileMetric::DrawDensityViewOverlay:
        frame.drawDensityViewOverlayMs = valueMs;
        break;
    case RuntimeProfileMetric::DrawBall:
        frame.drawBallMs = valueMs;
        break;
    case RuntimeProfileMetric::DrawBox:
        frame.drawBoxMs = valueMs;
        break;
    }
}

}  // namespace

void BeginRuntimeUpdateProfiling() {
    RuntimeFrameProfiling& frame = g_runtimeProfilingState.frame;
    frame.updateTotalMs = 0.0f;
    frame.updateSceneSyncMs = 0.0f;
    frame.updateBallValidationMs = 0.0f;
    frame.updateBoxValidationMs = 0.0f;
    frame.updateBallPhysicsMs = 0.0f;
    frame.updateParticleDensityMs = 0.0f;
    frame.updateTransformNormalizeMs = 0.0f;
    frame.physicsSubsteps = 0;
    frame.physicsSubstepDtMs = 0.0f;
    frame.physicsMaxDensityErrorRatio = 0.0f;
    frame.physicsRecommendedGravityMax = 0.0f;
    frame.physicsEffectiveStiffness = 0.0f;
    g_runtimeProfilingState.updateStart = ProfileClock::now();
}

void EndRuntimeUpdateProfiling() {
    g_runtimeProfilingState.frame.updateTotalMs =
        ToMilliseconds(ProfileClock::now() - g_runtimeProfilingState.updateStart);
}

void BeginRuntimeDrawProfiling() {
    RuntimeFrameProfiling& frame = g_runtimeProfilingState.frame;
    frame.drawTotalMs = 0.0f;
    frame.drawDensityViewGridMs = 0.0f;
    frame.drawDensityViewOverlayMs = 0.0f;
    frame.drawBallMs = 0.0f;
    frame.drawBoxMs = 0.0f;
    g_runtimeProfilingState.drawStart = ProfileClock::now();
}

void EndRuntimeDrawProfiling() {
    g_runtimeProfilingState.frame.drawTotalMs =
        ToMilliseconds(ProfileClock::now() - g_runtimeProfilingState.drawStart);
}

const RuntimeFrameProfiling& GetRuntimeFrameProfiling() {
    return g_runtimeProfilingState.frame;
}

void SetRuntimePhysicsStepDiagnostics(
    int substeps,
    float substepDtMs,
    float maxDensityErrorRatio,
    float recommendedGravityMax,
    float effectiveStiffness
) {
    RuntimeFrameProfiling& frame = g_runtimeProfilingState.frame;
    frame.physicsSubsteps = std::max(0, substeps);
    frame.physicsSubstepDtMs = std::max(0.0f, substepDtMs);
    frame.physicsMaxDensityErrorRatio = std::max(0.0f, maxDensityErrorRatio);
    frame.physicsRecommendedGravityMax = std::max(0.0f, recommendedGravityMax);
    frame.physicsEffectiveStiffness = std::max(0.0f, effectiveStiffness);
}

RuntimeScopedMetricTimer::RuntimeScopedMetricTimer(RuntimeProfileMetric metric)
    : m_metric(metric), m_start(ProfileClock::now()) {
}

RuntimeScopedMetricTimer::~RuntimeScopedMetricTimer() {
    SetMetricValue(m_metric, ToMilliseconds(ProfileClock::now() - m_start));
}

}  // namespace sim::runtime
