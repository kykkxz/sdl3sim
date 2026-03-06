#pragma once

#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_video.h>
#include <cstddef>
#include <functional>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <vector>

#include "DensityFieldOverlay2D.hpp"
#include "DrawTypes2D.hpp"
#include "PrimitiveBuilder2D.hpp"

namespace sim::renderer {

class GpuRenderer {
public:
    struct FrameHooks {
        std::function<void(SDL_GPUCommandBuffer* commandBuffer)> preRenderPass;
        std::function<void(SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass)> onRenderPass;
    };

    struct Config {
        SDL_GPUShaderFormat shaderFormats = SDL_GPU_SHADERFORMAT_SPIRV;
        bool debugMode = true;
        const char* preferredBackend = "vulkan";
        SDL_FColor clearColor{0.08f, 0.11f, 0.16f, 1.0f};
    };

    struct GlobalShaderParams {
        // vec4: x=density, y=quality, z/w reserved
        float density = 1.0f;
        float quality = 1.0f;
        float reserved0 = 0.0f;
        float reserved1 = 0.0f;

        // vec4: x=viewportWidth, y=viewportHeight, z=timeSeconds, w=frameIndex
        float viewportWidth = 0.0f;
        float viewportHeight = 0.0f;
        float timeSeconds = 0.0f;
        float frameIndex = 0.0f;
    };

    struct ObjectShaderParams {
        float density = 1.0f;
        float quality = 1.0f;
        float particleEnabled = 0.0f;
        glm::vec2 particleCenter{0.0f, 0.0f};
        float particleRadius = 0.0f;
        float smoothingRadius = 0.0f;
    };

    // Compatibility aliases: primitives now live in DrawTypes2D.hpp
    using Color = sim::renderer::Color;
    using Triangle2D = sim::renderer::Triangle2D;
    using Rect2D = sim::renderer::Rect2D;
    using Circle2D = sim::renderer::Circle2D;
    using Polygon2D = sim::renderer::Polygon2D;
    using Arc2D = sim::renderer::Arc2D;

    GpuRenderer() = default;
    ~GpuRenderer();

    GpuRenderer(const GpuRenderer&) = delete;
    GpuRenderer& operator=(const GpuRenderer&) = delete;

    bool Initialize(SDL_Window* window, const Config& config);
    void Shutdown();
    bool RenderFrame();
    bool RenderFrame(const FrameHooks& hooks);

    void SetProjection2D(float left, float right, float bottom, float top);
    bool UseWindowResolutionProjection2D(bool originTopLeft = true);
    void DisableWindowResolutionProjection2D();
    bool GetWindowResolution(Uint32& outWidth, Uint32& outHeight) const;
    void SetClearColor(const SDL_FColor& clearColor);
    SDL_FColor GetClearColor() const;
    SDL_GPUTextureFormat GetSwapchainTextureFormat() const;
    void SetGlobalShaderParams(const GlobalShaderParams& params);
    const GlobalShaderParams& GetGlobalShaderParams() const;
    void SetGlobalDensity(float density);
    void SetGlobalQuality(float quality);
    void SetGlobalTimeSeconds(float timeSeconds);
    void SetGlobalFrameIndex(float frameIndex);
    void SetObjectShaderParams(const ObjectShaderParams& params);
    const ObjectShaderParams& GetObjectShaderParams() const;
    void SetObjectDensity(float density);
    void SetObjectQuality(float quality);
    void SetObjectParticle(const glm::vec2& center, float particleRadius, float smoothingRadius);
    void DisableObjectParticle();
    void ResetObjectShaderParams();
    void SetView2D(const glm::vec2& cameraPosition);
    void SetModel2D(
        const glm::vec2& position,
        float rotationRadians = 0.0f,
        glm::vec2 scale = glm::vec2(1.0f, 1.0f)
    );
    void ResetModel2D();

    void BeginScene2D();
    void DrawTriangle2D(const Triangle2D& triangle);
    void DrawTriangle2D(
        const glm::vec2& p0,
        const glm::vec2& p1,
        const glm::vec2& p2,
        Color color = Color(),
        bool wireframe = false,
        float lineThickness = 0.0f
    );
    void DrawRect2D(const Rect2D& rect);
    void DrawRect2D(
        const glm::vec2& position,
        const glm::vec2& size,
        Color color = Color(),
        bool wireframe = false,
        float rotationRadians = 0.0f,
        float lineThickness = 0.0f
    );
    void DrawCircle2D(const Circle2D& circle);
    void DrawCircle2D(
        const glm::vec2& center,
        float radius,
        Color color = Color(),
        bool wireframe = false,
        int segments = 32,
        float lineThickness = 0.0f
    );
    void DrawPolygon2D(const Polygon2D& polygon);
    void DrawPolygon2D(
        const std::vector<glm::vec2>& points,
        Color color = Color(),
        bool wireframe = false,
        float lineThickness = 0.0f
    );
    void DrawArc2D(const Arc2D& arc);
    void DrawArc2D(
        const glm::vec2& center,
        float radius,
        float startRadians,
        float endRadians,
        Color color = Color(),
        bool wireframe = true,
        int segments = 32,
        float lineThickness = 0.0f
    );
    void DrawLine2D(
        const glm::vec2& start,
        const glm::vec2& end,
        Color color = Color(),
        float thickness = 0.0f
    );
    void DrawDensityFieldOverlay2D(
        const DensityFieldGrid2D& grid,
        const DensityFieldOverlayStyle2D& style
    );

    size_t GetQueuedDrawCount() const;

    SDL_GPUDevice* GetDevice() const { return m_device; }
    const char* GetBackendName() const;

private:
    using Vertex = PrimitiveVertex2D;

    static constexpr Uint32 kMaxBatchVertices = 131072;

    bool CreateRenderResources();
    bool CreateShaders();
    bool CreateGraphicsPipelines();
    bool CreateVertexBuffers();
    void UpdateMVP();

    bool BeginFrame(SDL_GPUCommandBuffer*& outCommandBuffer, SDL_GPUTexture*& outSwapchainTexture) const;
    SDL_GPURenderPass* BeginMainRenderPass(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* swapchainTexture) const;
    bool UploadVertices(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUBuffer* vertexBuffer, const std::vector<Vertex>& vertices) const;
    void RecordShapeBatchDraw(SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass);
    bool EndFrame(SDL_GPUCommandBuffer* commandBuffer) const;

    bool SyncWindowResolutionProjection2D();
    bool ValidateBatchCapacity(size_t fillStartSize, size_t lineStartSize);
    void LogBatchOverflowOnce();

    SDL_Window* m_window = nullptr;
    SDL_GPUDevice* m_device = nullptr;
    SDL_GPUShader* m_vertexShader = nullptr;
    SDL_GPUShader* m_fragmentShader = nullptr;
    SDL_GPUGraphicsPipeline* m_fillPipeline = nullptr;
    SDL_GPUGraphicsPipeline* m_linePipeline = nullptr;
    SDL_GPUBuffer* m_fillVertexBuffer = nullptr;
    SDL_GPUBuffer* m_lineVertexBuffer = nullptr;
    glm::mat4 m_model{1.0f};
    glm::mat4 m_view{1.0f};
    glm::mat4 m_projection{1.0f};
    glm::mat4 m_mvp{1.0f};
    bool m_mvpDirty = true;
    std::vector<Vertex> m_fillBatch;
    std::vector<Vertex> m_lineBatch;
    size_t m_queuedPrimitiveCount = 0;
    bool m_overflowLogged = false;
    bool m_useWindowResolutionProjection = false;
    bool m_originTopLeft = true;
    Uint32 m_cachedWindowWidth = 0;
    Uint32 m_cachedWindowHeight = 0;
    GlobalShaderParams m_globalShaderParams{};
    ObjectShaderParams m_objectShaderParams{};
    Config m_config;
    bool m_windowClaimed = false;
};

}  // namespace sim::renderer
