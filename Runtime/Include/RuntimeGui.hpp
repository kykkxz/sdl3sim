#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_video.h>

#include "GpuRenderer.hpp"
#include "RuntimeTypes.hpp"

namespace sim::runtime {

class RuntimeGui {
public:
    RuntimeGui() = default;
    ~RuntimeGui();

    RuntimeGui(const RuntimeGui&) = delete;
    RuntimeGui& operator=(const RuntimeGui&) = delete;

    bool Initialize(SDL_Window* window, sim::renderer::GpuRenderer& renderer);
    void Shutdown();

    void ProcessEvent(const SDL_Event& event) const;
    void BeginFrame() const;
    bool WantsMouseCapture() const;
    EditorActions BuildRuntimeEditor(
        UiState& state,
        float deltaSeconds,
        float fps
    ) const;
    void EndFrame() const;

    void PrepareDrawData(SDL_GPUCommandBuffer* commandBuffer) const;
    void RenderDrawData(SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass) const;

private:
    bool m_initialized = false;
};

}  // namespace sim::runtime
