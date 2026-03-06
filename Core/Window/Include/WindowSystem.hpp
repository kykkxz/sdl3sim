#pragma once

#include <functional>
#include <string>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

namespace sim::window {

class WindowSystem {
public:
    struct Config {
        int width = 1600;
        int height = 960;
        std::string title = "SDL3 GPU Learning Sandbox";
        bool resizable = true;
        bool highPixelDensity = true;
    };

    WindowSystem() = default;
    ~WindowSystem();

    WindowSystem(const WindowSystem&) = delete;
    WindowSystem& operator=(const WindowSystem&) = delete;

    bool Initialize(const Config& config);
    void Shutdown();

    bool PumpEvents(const std::function<void(const SDL_Event&)>& eventCallback = {});
    SDL_Window* GetNativeHandle() const { return m_window; }

private:
    SDL_Window* m_window = nullptr;
    bool m_sdlInitialized = false;
};

}  // namespace sim::window
