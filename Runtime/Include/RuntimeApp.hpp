#pragma once

#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_stdinc.h>
#include <vector>

#include "GpuRenderer.hpp"
#include "RuntimeGui.hpp"
#include "RuntimeTypes.hpp"
#include "WindowSystem.hpp"
#include "World.hpp"

namespace sim::runtime {

class RuntimeApp {
public:
    int Run();

private:
    int Initialize();
    void Shutdown();
    bool Tick();
    void RebuildBallEntities(int desiredParticleCount);
    void UpdateMouseForceInput(float deltaSeconds);

    float ComputeDeltaSeconds(Uint64 nowTicksNs);
    SDL_FColor BuildClearColor() const;

    sim::window::WindowSystem m_windowSystem;
    sim::renderer::GpuRenderer m_renderer;
    RuntimeGui m_gui;
    sim::ecs::World m_world;
    std::vector<sim::ecs::Entity> m_ballEntities;
    sim::ecs::Entity m_boundingBoxEntity{};
    UiState m_uiState{};
    MouseForceInput m_mouseForceInput{};
    bool m_mousePositionInitialized = false;
    Uint64 m_lastTicksNs = 0;
};

}  // namespace sim::runtime
