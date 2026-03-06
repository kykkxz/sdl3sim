#pragma once

#include <chrono>

namespace sim::runtime {

struct RuntimeFrameProfiling {
    float updateTotalMs = 0.0f;
    float updateSceneSyncMs = 0.0f;
    float updateBallValidationMs = 0.0f;
    float updateBoxValidationMs = 0.0f;
    float updateBallPhysicsMs = 0.0f;
    float updateParticleDensityMs = 0.0f;
    float updateTransformNormalizeMs = 0.0f;
    int physicsSubsteps = 0;
    float physicsSubstepDtMs = 0.0f;
    float physicsMaxDensityErrorRatio = 0.0f;
    float physicsRecommendedGravityMax = 0.0f;
    float physicsEffectiveStiffness = 0.0f;

    float drawTotalMs = 0.0f;
    float drawDensityViewGridMs = 0.0f;
    float drawDensityViewOverlayMs = 0.0f;
    float drawBallMs = 0.0f;
    float drawBoxMs = 0.0f;
};

enum class RuntimeProfileMetric {
    UpdateSceneSync,
    UpdateBallValidation,
    UpdateBoxValidation,
    UpdateBallPhysics,
    UpdateParticleDensity,
    UpdateTransformNormalize,
    DrawDensityViewGrid,
    DrawDensityViewOverlay,
    DrawBall,
    DrawBox
};

void BeginRuntimeUpdateProfiling();
void EndRuntimeUpdateProfiling();
void BeginRuntimeDrawProfiling();
void EndRuntimeDrawProfiling();

const RuntimeFrameProfiling& GetRuntimeFrameProfiling();
void SetRuntimePhysicsStepDiagnostics(
    int substeps,
    float substepDtMs,
    float maxDensityErrorRatio,
    float recommendedGravityMax,
    float effectiveStiffness
);

class RuntimeScopedMetricTimer {
public:
    explicit RuntimeScopedMetricTimer(RuntimeProfileMetric metric);
    ~RuntimeScopedMetricTimer();

    RuntimeScopedMetricTimer(const RuntimeScopedMetricTimer&) = delete;
    RuntimeScopedMetricTimer& operator=(const RuntimeScopedMetricTimer&) = delete;

private:
    RuntimeProfileMetric m_metric;
    std::chrono::steady_clock::time_point m_start;
};

}  // namespace sim::runtime
