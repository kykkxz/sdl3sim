#include "World.hpp"

namespace sim::ecs {

Entity World::CreateEntity() {
    const Entity entity = m_registry.create();
    m_registry.emplace<IdComponent>(entity, m_nextId++);
    m_registry.emplace<TransformComponent>(entity);
    return entity;
}

void World::DestroyEntity(Entity entity) {
    if (!m_registry.valid(entity)) {
        return;
    }
    m_registry.destroy(entity);
}

bool World::IsAlive(Entity entity) const {
    return m_registry.valid(entity);
}

entt::registry& World::Registry() {
    return m_registry;
}

const entt::registry& World::Registry() const {
    return m_registry;
}

}  // namespace sim::ecs
