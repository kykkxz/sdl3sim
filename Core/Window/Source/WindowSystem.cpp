#include "WindowSystem.hpp"

#include <SDL3/SDL_init.h>
#include <SDL3/SDL_log.h>

namespace sim::window {

WindowSystem::~WindowSystem() {
    Shutdown();
}

bool WindowSystem::Initialize(const Config& config) {
    if (m_window != nullptr) {
        return true;
    }

    if (!SDL_SetAppMetadata("SDL3Simulator", "0.1.0", "SDL3sim")) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "SDL_SetAppMetadata failed: %s", SDL_GetError());
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init failed: %s", SDL_GetError());
        return false;
    }
    m_sdlInitialized = true;

    SDL_WindowFlags flags = 0;
    if (config.resizable) {
        flags |= SDL_WINDOW_RESIZABLE;
    }
    if (config.highPixelDensity) {
        flags |= SDL_WINDOW_HIGH_PIXEL_DENSITY;
    }

    m_window = SDL_CreateWindow(config.title.c_str(), config.width, config.height, flags);
    if (m_window == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_CreateWindow failed: %s", SDL_GetError());
        Shutdown();
        return false;
    }

    return true;
}

void WindowSystem::Shutdown() {
    if (m_window != nullptr) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    if (m_sdlInitialized) {
        SDL_Quit();
        m_sdlInitialized = false;
    }
}

bool WindowSystem::PumpEvents(const std::function<void(const SDL_Event&)>& eventCallback) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (eventCallback) {
            eventCallback(event);
        }
        if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
            return false;
        }
    }
    return true;
}

}  // namespace sim::window
