#pragma once

#include <span>

#include "GpuRenderer.hpp"
#include "RuntimeComponents.hpp"
#include "RuntimeTypes.hpp"
#include "World.hpp"

namespace sim::runtime {

void ResetWorldToInitialTransforms(
    sim::ecs::World& world,
    std::span<const sim::ecs::Entity> ballEntities,
    sim::ecs::Entity boundingBoxEntity
);

void ResetToInitialState(
    UiState& state,
    sim::ecs::World& world,
    std::span<const sim::ecs::Entity> ballEntities,
    sim::ecs::Entity boundingBoxEntity
);

void UpdateWorld(
    sim::ecs::World& world,
    const UiState& state,
    Uint32 viewportWidth,
    Uint32 viewportHeight,
    float deltaSeconds,
    const MouseForceInput& mouseInput
);

void DrawWorld2D(
    sim::renderer::GpuRenderer& renderer,
    sim::ecs::World& world,
    const UiState& state
);

}  // namespace sim::runtime
