#include "RuntimeApp.hpp"

#include <SDL3/SDL_log.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_timer.h>
#include <algorithm>
#include <cstddef>
#include <cmath>

#include "RuntimeCuda.hpp"
#include "RuntimeWorldSystems.hpp"

namespace sim::runtime {
namespace {

int ClampParticleCount(int value) {
    return std::clamp(value, kMinParticleCount, kMaxParticleCount);
}

}  // namespace

int RuntimeApp::Run() {
    const int initCode = Initialize();
    if (initCode != 0) {
        Shutdown();
        return initCode;
    }

    bool running = true;
    while (running) {
        running = Tick();
    }

    Shutdown();
    return 0;
}

int RuntimeApp::Initialize() {
    if (!m_windowSystem.Initialize(sim::window::WindowSystem::Config{})) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize window system");
        return 1;
    }

    if (!m_renderer.Initialize(m_windowSystem.GetNativeHandle(), sim::renderer::GpuRenderer::Config{})) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize GPU renderer");
        return 2;
    }
    if (!m_renderer.UseWindowResolutionProjection2D(true)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "Failed to enable window-resolution projection, fallback to default projection");
    }

    if (!m_gui.Initialize(m_windowSystem.GetNativeHandle(), m_renderer)) {
        return 3;
    }

    m_uiState = UiState{};
    m_boundingBoxEntity = m_world.CreateEntity();
    RebuildBallEntities(m_uiState.world.particleCount);

    const auto& boundingBoxId = m_world.GetComponent<sim::ecs::IdComponent>(m_boundingBoxEntity);
    if (!m_ballEntities.empty()) {
        const auto& firstBallId = m_world.GetComponent<sim::ecs::IdComponent>(m_ballEntities.front());
        SDL_Log(
            "ECS entities initialized: particles=%u firstParticle=%llu boundingBox=%llu",
            static_cast<unsigned>(m_ballEntities.size()),
            static_cast<unsigned long long>(firstBallId.value),
            static_cast<unsigned long long>(boundingBoxId.value)
        );
    } else {
        SDL_Log(
            "ECS entities initialized: particles=0 boundingBox=%llu",
            static_cast<unsigned long long>(boundingBoxId.value)
        );
    }

    if (IsCudaBuildEnabled()) {
        const bool runtimeAvailable = IsCudaRuntimeAvailable();
        SDL_Log("CUDA backend: build=ON, runtime=%s", runtimeAvailable ? "available" : "unavailable");
    } else {
        SDL_Log("CUDA backend: build=OFF (CPU path)");
    }
    m_lastTicksNs = SDL_GetTicksNS();
    return 0;
}

void RuntimeApp::Shutdown() {
    m_gui.Shutdown();
    m_renderer.Shutdown();
    m_windowSystem.Shutdown();
}

bool RuntimeApp::Tick() {
    const bool running = m_windowSystem.PumpEvents([this](const SDL_Event& event) {
        m_gui.ProcessEvent(event);
    });
    if (!running) {
        return false;
    }

    const Uint64 nowTicksNs = SDL_GetTicksNS();
    const float deltaSeconds = ComputeDeltaSeconds(nowTicksNs);
    const float fps = 1.0f / deltaSeconds;

    m_gui.BeginFrame();
    const EditorActions actions = m_gui.BuildRuntimeEditor(m_uiState, deltaSeconds, fps);
    if (actions.resetToInitialState) {
        m_uiState = UiState{};
        RebuildBallEntities(m_uiState.world.particleCount);
    } else if (actions.rebuildParticleEntities) {
        RebuildBallEntities(m_uiState.world.particleCount);
    }
    UpdateMouseForceInput(deltaSeconds);

    Uint32 viewportWidth = 1600;
    Uint32 viewportHeight = 960;
    (void)m_renderer.GetWindowResolution(viewportWidth, viewportHeight);
    UpdateWorld(
        m_world,
        m_uiState,
        viewportWidth,
        viewportHeight,
        deltaSeconds,
        m_mouseForceInput
    );
    m_gui.EndFrame();

    m_renderer.SetClearColor(BuildClearColor());
    m_renderer.SetGlobalDensity(m_uiState.shaderGlobals.density);
    m_renderer.SetGlobalQuality(m_uiState.shaderGlobals.quality);
    m_renderer.SetGlobalTimeSeconds(static_cast<float>(nowTicksNs) / 1'000'000'000.0f);
    m_renderer.BeginScene2D();
    DrawWorld2D(m_renderer, m_world, m_uiState);

    sim::renderer::GpuRenderer::FrameHooks hooks;
    hooks.preRenderPass = [this](SDL_GPUCommandBuffer* commandBuffer) {
        m_gui.PrepareDrawData(commandBuffer);
    };
    hooks.onRenderPass = [this](SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass) {
        m_gui.RenderDrawData(commandBuffer, renderPass);
    };

    if (!m_renderer.RenderFrame(hooks)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "RenderFrame failed");
        return false;
    }

    return true;
}

void RuntimeApp::RebuildBallEntities(int desiredParticleCount) {
    const int safeParticleCount = ClampParticleCount(desiredParticleCount);
    m_uiState.world.particleCount = safeParticleCount;

    for (const sim::ecs::Entity entity : m_ballEntities) {
        m_world.DestroyEntity(entity);
    }
    m_ballEntities.clear();
    m_ballEntities.reserve(static_cast<std::size_t>(safeParticleCount));
    for (int i = 0; i < safeParticleCount; ++i) {
        m_ballEntities.push_back(m_world.CreateEntity());
    }

    ResetWorldToInitialTransforms(m_world, m_ballEntities, m_boundingBoxEntity);
}

void RuntimeApp::UpdateMouseForceInput(float deltaSeconds) {
    const glm::vec2 previousMousePosition = m_mouseForceInput.position;
    float mouseX = 0.0f;
    float mouseY = 0.0f;
    const SDL_MouseButtonFlags mouseButtons = SDL_GetMouseState(&mouseX, &mouseY);
    const glm::vec2 currentMousePosition{mouseX, mouseY};
    m_mouseForceInput.position = currentMousePosition;

    glm::vec2 mouseVelocity{0.0f, 0.0f};
    if (m_mousePositionInitialized && deltaSeconds > 1e-6f) {
        mouseVelocity = (currentMousePosition - previousMousePosition) / deltaSeconds;
    }
    if (!std::isfinite(mouseVelocity.x) || !std::isfinite(mouseVelocity.y)) {
        mouseVelocity = {0.0f, 0.0f};
    }
    constexpr float kMaxMouseSpeed = 16000.0f;
    const float speedSquared = mouseVelocity.x * mouseVelocity.x + mouseVelocity.y * mouseVelocity.y;
    if (speedSquared > kMaxMouseSpeed * kMaxMouseSpeed) {
        const float speed = std::sqrt(speedSquared);
        if (speed > 1e-6f) {
            mouseVelocity *= (kMaxMouseSpeed / speed);
        }
    }
    m_mouseForceInput.velocity = mouseVelocity;
    m_mousePositionInitialized = true;

    if (m_gui.WantsMouseCapture()) {
        m_mouseForceInput.leftPressed = false;
        m_mouseForceInput.rightPressed = false;
        return;
    }
    m_mouseForceInput.leftPressed = (mouseButtons & SDL_BUTTON_LMASK) != 0;
    m_mouseForceInput.rightPressed = (mouseButtons & SDL_BUTTON_RMASK) != 0;
}

float RuntimeApp::ComputeDeltaSeconds(Uint64 nowTicksNs) {
    float deltaSeconds = static_cast<float>(nowTicksNs - m_lastTicksNs) / 1'000'000'000.0f;
    m_lastTicksNs = nowTicksNs;
    if (deltaSeconds <= 0.0f || deltaSeconds > 0.25f) {
        deltaSeconds = 1.0f / 60.0f;
    }
    return deltaSeconds;
}

SDL_FColor RuntimeApp::BuildClearColor() const {
    SDL_FColor color{};
    color.r = std::clamp(m_uiState.clearColor[0], 0.0f, 1.0f);
    color.g = std::clamp(m_uiState.clearColor[1], 0.0f, 1.0f);
    color.b = std::clamp(m_uiState.clearColor[2], 0.0f, 1.0f);
    color.a = 1.0f;
    return color;
}

}  // namespace sim::runtime
