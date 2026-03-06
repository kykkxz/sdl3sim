#pragma once

#include <cassert>
#include <entt/entt.hpp>
#include <utility>

#include "EcsComponents.hpp"

namespace sim::ecs {

using Entity = entt::entity;

class World {
public:
    Entity CreateEntity();
    void DestroyEntity(Entity entity);

    [[nodiscard]] bool IsAlive(Entity entity) const;

    template <typename Component, typename... Args>
    Component& AddComponent(Entity entity, Args&&... args) {
        assert(m_registry.valid(entity));
        return m_registry.emplace_or_replace<Component>(entity, std::forward<Args>(args)...);
    }

    template <typename Component>
    [[nodiscard]] bool HasComponent(Entity entity) const {
        return m_registry.valid(entity) && m_registry.all_of<Component>(entity);
    }

    template <typename Component>
    Component& GetComponent(Entity entity) {
        assert(m_registry.valid(entity));
        return m_registry.get<Component>(entity);
    }

    template <typename Component>
    const Component& GetComponent(Entity entity) const {
        assert(m_registry.valid(entity));
        return m_registry.get<Component>(entity);
    }

    template <typename Component>
    Component* TryGetComponent(Entity entity) {
        if (!m_registry.valid(entity)) {
            return nullptr;
        }
        return m_registry.try_get<Component>(entity);
    }

    template <typename Component>
    const Component* TryGetComponent(Entity entity) const {
        if (!m_registry.valid(entity)) {
            return nullptr;
        }
        return m_registry.try_get<Component>(entity);
    }

    [[nodiscard]] entt::registry& Registry();
    [[nodiscard]] const entt::registry& Registry() const;

private:
    entt::registry m_registry;
    EntityId m_nextId = 1;
};

}  // namespace sim::ecs
