#include "GpuRenderer.hpp"

#include <algorithm>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_log.h>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ShaderCompile.hpp"

namespace sim::renderer {

namespace {

static_assert(
    sizeof(GpuRenderer::GlobalShaderParams) == sizeof(float) * 8,
    "GlobalShaderParams must match two vec4 values (std140)."
);

constexpr const char* kVertexShaderPath = "shaders/common.vert.glsl";
constexpr const char* kFragmentShaderPath = "shaders/common.frag.glsl";

glm::mat4 BuildModel2D(const glm::vec2& position, float rotationRadians, const glm::vec2& scale) {
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(position, 0.0f));
    model = glm::rotate(model, rotationRadians, glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, glm::vec3(scale, 1.0f));
    return model;
}

PrimitiveColor2D ToPrimitiveColor(const GpuRenderer::Color& color) {
    PrimitiveColor2D out{};
    out.r = color.r;
    out.g = color.g;
    out.b = color.b;
    return out;
}

PrimitiveShaderParams2D ToPrimitiveShaderParams(const GpuRenderer::ObjectShaderParams& shaderParams) {
    PrimitiveShaderParams2D out{};
    out.density = shaderParams.density;
    out.quality = shaderParams.quality;
    out.particleEnabled = shaderParams.particleEnabled;
    out.smoothingRadius = shaderParams.smoothingRadius;
    out.particleCenter[0] = shaderParams.particleCenter.x;
    out.particleCenter[1] = shaderParams.particleCenter.y;
    out.particleRadius = shaderParams.particleRadius;
    return out;
}

bool CompileShaderFromFile(
    const char* path,
    shaderc_shader_kind kind,
    const char* sourceName,
    std::vector<uint32_t>& outSpirv,
    std::string& outError
) {
    const std::string source = ReadTextFile(path);
    if (source.empty()) {
        outError = std::string("Shader source is empty or missing: ") + path;
        return false;
    }
    return CompileGLSLToSPIRV(source, kind, sourceName, outSpirv, outError);
}

SDL_GPUShader* CreateSDLShaderFromSpv(
    SDL_GPUDevice* device,
    const std::vector<uint32_t>& spirv,
    SDL_GPUShaderStage stage
) {
    SDL_GPUShaderCreateInfo ci{};
    ci.code = reinterpret_cast<const Uint8*>(spirv.data());
    ci.code_size = spirv.size() * sizeof(uint32_t);
    ci.entrypoint = "main";
    ci.stage = stage;
    ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
    ci.num_samplers = 0;
    ci.num_storage_buffers = 0;
    ci.num_storage_textures = 0;
    ci.num_uniform_buffers =
        (stage == SDL_GPU_SHADERSTAGE_VERTEX || stage == SDL_GPU_SHADERSTAGE_FRAGMENT) ? 1u : 0u;
    return SDL_CreateGPUShader(device, &ci);
}

bool CreateDynamicVertexBuffer(SDL_GPUDevice* device, Uint32 sizeInBytes, SDL_GPUBuffer** outVertexBuffer) {
    if (device == nullptr || outVertexBuffer == nullptr) {
        return false;
    }

    *outVertexBuffer = nullptr;

    SDL_GPUBufferCreateInfo bufferInfo{};
    bufferInfo.size = sizeInBytes;
    bufferInfo.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
    bufferInfo.props = 0;

    SDL_GPUBuffer* vertexBuffer = SDL_CreateGPUBuffer(device, &bufferInfo);
    if (vertexBuffer == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_CreateGPUBuffer failed: %s", SDL_GetError());
        return false;
    }

    *outVertexBuffer = vertexBuffer;
    return true;
}

}  // namespace

GpuRenderer::~GpuRenderer() {
    Shutdown();
}

bool GpuRenderer::CreateRenderResources() {
    m_fillBatch.reserve(kMaxBatchVertices);
    m_lineBatch.reserve(kMaxBatchVertices);
    return CreateShaders() && CreateGraphicsPipelines() && CreateVertexBuffers();
}

bool GpuRenderer::CreateShaders() {
    std::string error;
    std::vector<uint32_t> vertexSpv;
    if (!CompileShaderFromFile(kVertexShaderPath, shaderc_vertex_shader, "triangle.vert.glsl", vertexSpv, error)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to compile vertex shader: %s", error.c_str());
        return false;
    }

    std::vector<uint32_t> fragmentSpv;
    if (!CompileShaderFromFile(kFragmentShaderPath, shaderc_fragment_shader, "triangle.frag.glsl", fragmentSpv, error)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to compile fragment shader: %s", error.c_str());
        return false;
    }

    m_vertexShader = CreateSDLShaderFromSpv(m_device, vertexSpv, SDL_GPU_SHADERSTAGE_VERTEX);
    if (m_vertexShader == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create vertex shader: %s", SDL_GetError());
        return false;
    }

    m_fragmentShader = CreateSDLShaderFromSpv(m_device, fragmentSpv, SDL_GPU_SHADERSTAGE_FRAGMENT);
    if (m_fragmentShader == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create fragment shader: %s", SDL_GetError());
        return false;
    }

    return true;
}

bool GpuRenderer::CreateGraphicsPipelines() {
    SDL_GPUVertexBufferDescription vertexBufferDesc{};
    vertexBufferDesc.slot = 0;
    vertexBufferDesc.pitch = sizeof(Vertex);
    vertexBufferDesc.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    vertexBufferDesc.instance_step_rate = 0;

    SDL_GPUVertexAttribute vertexAttributes[4]{};
    vertexAttributes[0].location = 0;
    vertexAttributes[0].buffer_slot = 0;
    vertexAttributes[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
    vertexAttributes[0].offset = static_cast<Uint32>(offsetof(Vertex, position));

    vertexAttributes[1].location = 1;
    vertexAttributes[1].buffer_slot = 0;
    vertexAttributes[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
    vertexAttributes[1].offset = static_cast<Uint32>(offsetof(Vertex, color));

    vertexAttributes[2].location = 2;
    vertexAttributes[2].buffer_slot = 0;
    vertexAttributes[2].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4;
    vertexAttributes[2].offset = static_cast<Uint32>(offsetof(Vertex, shaderParams0));

    vertexAttributes[3].location = 3;
    vertexAttributes[3].buffer_slot = 0;
    vertexAttributes[3].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4;
    vertexAttributes[3].offset = static_cast<Uint32>(offsetof(Vertex, shaderParams1));

    SDL_GPUColorTargetDescription colorTargetDesc{};
    colorTargetDesc.format = SDL_GetGPUSwapchainTextureFormat(m_device, m_window);
    colorTargetDesc.blend_state.enable_blend = true;
    colorTargetDesc.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    colorTargetDesc.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    colorTargetDesc.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
    colorTargetDesc.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
    colorTargetDesc.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    colorTargetDesc.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
    colorTargetDesc.blend_state.color_write_mask = SDL_GPU_COLORCOMPONENT_R
                                                  | SDL_GPU_COLORCOMPONENT_G
                                                  | SDL_GPU_COLORCOMPONENT_B
                                                  | SDL_GPU_COLORCOMPONENT_A;

    SDL_GPUGraphicsPipelineTargetInfo targetInfo{};
    targetInfo.num_color_targets = 1;
    targetInfo.color_target_descriptions = &colorTargetDesc;

    auto createPipeline = [&](SDL_GPUPrimitiveType primitiveType) {
        SDL_GPUGraphicsPipelineCreateInfo ci{};
        ci.vertex_shader = m_vertexShader;
        ci.fragment_shader = m_fragmentShader;
        ci.vertex_input_state.vertex_buffer_descriptions = &vertexBufferDesc;
        ci.vertex_input_state.num_vertex_buffers = 1;
        ci.vertex_input_state.vertex_attributes = vertexAttributes;
        ci.vertex_input_state.num_vertex_attributes = 4;
        ci.primitive_type = primitiveType;
        ci.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
        ci.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
        ci.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
        ci.rasterizer_state.enable_depth_clip = true;
        ci.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
        ci.multisample_state.sample_mask = 0;
        ci.multisample_state.enable_mask = false;
        ci.target_info = targetInfo;
        return SDL_CreateGPUGraphicsPipeline(m_device, &ci);
    };

    m_fillPipeline = createPipeline(SDL_GPU_PRIMITIVETYPE_TRIANGLELIST);
    if (m_fillPipeline == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create fill pipeline: %s", SDL_GetError());
        return false;
    }

    m_linePipeline = createPipeline(SDL_GPU_PRIMITIVETYPE_LINELIST);
    if (m_linePipeline == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create line pipeline: %s", SDL_GetError());
        return false;
    }

    return true;
}

bool GpuRenderer::CreateVertexBuffers() {
    const Uint32 vertexBufferSize = static_cast<Uint32>(sizeof(Vertex) * kMaxBatchVertices);
    if (!CreateDynamicVertexBuffer(m_device, vertexBufferSize, &m_fillVertexBuffer)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create fill vertex buffer");
        return false;
    }

    if (!CreateDynamicVertexBuffer(m_device, vertexBufferSize, &m_lineVertexBuffer)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "Failed to create line vertex buffer");
        return false;
    }

    return true;
}

bool GpuRenderer::BeginFrame(SDL_GPUCommandBuffer*& outCommandBuffer, SDL_GPUTexture*& outSwapchainTexture) const {
    outCommandBuffer = SDL_AcquireGPUCommandBuffer(m_device);
    if (outCommandBuffer == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_AcquireGPUCommandBuffer failed: %s", SDL_GetError());
        return false;
    }

    Uint32 width = 0;
    Uint32 height = 0;
    outSwapchainTexture = nullptr;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(outCommandBuffer, m_window, &outSwapchainTexture, &width, &height)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_WaitAndAcquireGPUSwapchainTexture failed: %s", SDL_GetError());
        SDL_CancelGPUCommandBuffer(outCommandBuffer);
        outCommandBuffer = nullptr;
        return false;
    }

    return true;
}

SDL_GPURenderPass* GpuRenderer::BeginMainRenderPass(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* swapchainTexture) const {
    SDL_GPUColorTargetInfo colorTarget{};
    colorTarget.texture = swapchainTexture;
    colorTarget.mip_level = 0;
    colorTarget.layer_or_depth_plane = 0;
    colorTarget.clear_color = m_config.clearColor;
    colorTarget.load_op = SDL_GPU_LOADOP_CLEAR;
    colorTarget.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorTarget, 1, nullptr);
    if (renderPass == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_BeginGPURenderPass failed: %s", SDL_GetError());
    }

    return renderPass;
}

bool GpuRenderer::UploadVertices(
    SDL_GPUCommandBuffer* commandBuffer,
    SDL_GPUBuffer* vertexBuffer,
    const std::vector<Vertex>& vertices
) const {
    if (vertices.empty()) {
        return true;
    }

    const Uint32 uploadSize = static_cast<Uint32>(vertices.size() * sizeof(Vertex));

    SDL_GPUTransferBufferCreateInfo transferInfo{};
    transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    transferInfo.size = uploadSize;
    transferInfo.props = 0;

    SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(m_device, &transferInfo);
    if (transferBuffer == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_CreateGPUTransferBuffer failed: %s", SDL_GetError());
        return false;
    }

    void* mapped = SDL_MapGPUTransferBuffer(m_device, transferBuffer, false);
    if (mapped == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_MapGPUTransferBuffer failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
        return false;
    }
    std::memcpy(mapped, vertices.data(), uploadSize);
    SDL_UnmapGPUTransferBuffer(m_device, transferBuffer);

    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
    if (copyPass == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_BeginGPUCopyPass failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
        return false;
    }

    SDL_GPUTransferBufferLocation source{};
    source.transfer_buffer = transferBuffer;
    source.offset = 0;

    SDL_GPUBufferRegion destination{};
    destination.buffer = vertexBuffer;
    destination.offset = 0;
    destination.size = uploadSize;

    SDL_UploadToGPUBuffer(copyPass, &source, &destination, false);
    SDL_EndGPUCopyPass(copyPass);
    SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
    return true;
}

void GpuRenderer::RecordShapeBatchDraw(SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass) {
    UpdateMVP();

    if (!m_fillBatch.empty()) {
        SDL_BindGPUGraphicsPipeline(renderPass, m_fillPipeline);

        SDL_GPUBufferBinding vertexBinding{};
        vertexBinding.buffer = m_fillVertexBuffer;
        vertexBinding.offset = 0;
        SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);

        SDL_PushGPUVertexUniformData(commandBuffer, 0, glm::value_ptr(m_mvp), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(commandBuffer, 0, &m_globalShaderParams, sizeof(GlobalShaderParams));
        SDL_DrawGPUPrimitives(renderPass, static_cast<Uint32>(m_fillBatch.size()), 1, 0, 0);
    }

    if (!m_lineBatch.empty()) {
        SDL_BindGPUGraphicsPipeline(renderPass, m_linePipeline);

        SDL_GPUBufferBinding vertexBinding{};
        vertexBinding.buffer = m_lineVertexBuffer;
        vertexBinding.offset = 0;
        SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);

        SDL_PushGPUVertexUniformData(commandBuffer, 0, glm::value_ptr(m_mvp), sizeof(glm::mat4));
        SDL_PushGPUFragmentUniformData(commandBuffer, 0, &m_globalShaderParams, sizeof(GlobalShaderParams));
        SDL_DrawGPUPrimitives(renderPass, static_cast<Uint32>(m_lineBatch.size()), 1, 0, 0);
    }
}

bool GpuRenderer::EndFrame(SDL_GPUCommandBuffer* commandBuffer) const {
    if (!SDL_SubmitGPUCommandBuffer(commandBuffer)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_SubmitGPUCommandBuffer failed: %s", SDL_GetError());
        return false;
    }

    return true;
}

void GpuRenderer::UpdateMVP() {
    if (!m_mvpDirty) {
        return;
    }
    m_mvp = m_projection * m_view * m_model;
    m_mvpDirty = false;
}

void GpuRenderer::LogBatchOverflowOnce() {
    if (m_overflowLogged) {
        return;
    }
    SDL_LogWarn(
        SDL_LOG_CATEGORY_RENDER,
        "Draw batch vertex capacity exceeded (max=%u). Remaining draw requests are skipped for this frame.",
        kMaxBatchVertices
    );
    m_overflowLogged = true;
}

bool GpuRenderer::ValidateBatchCapacity(size_t fillStartSize, size_t lineStartSize) {
    if (m_fillBatch.size() <= kMaxBatchVertices && m_lineBatch.size() <= kMaxBatchVertices) {
        return true;
    }
    m_fillBatch.resize(fillStartSize);
    m_lineBatch.resize(lineStartSize);
    LogBatchOverflowOnce();
    return false;
}

void GpuRenderer::SetProjection2D(float left, float right, float bottom, float top) {
    m_projection = glm::ortho(left, right, bottom, top, -1.0f, 1.0f);
    m_mvpDirty = true;
}

bool GpuRenderer::GetWindowResolution(Uint32& outWidth, Uint32& outHeight) const {
    outWidth = 0;
    outHeight = 0;
    if (m_window == nullptr) {
        return false;
    }

    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(m_window, &width, &height)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "SDL_GetWindowSizeInPixels failed: %s", SDL_GetError());
        return false;
    }

    if (width <= 0 || height <= 0) {
        return false;
    }

    outWidth = static_cast<Uint32>(width);
    outHeight = static_cast<Uint32>(height);
    return true;
}

void GpuRenderer::SetClearColor(const SDL_FColor& clearColor) {
    m_config.clearColor = clearColor;
}

SDL_FColor GpuRenderer::GetClearColor() const {
    return m_config.clearColor;
}

SDL_GPUTextureFormat GpuRenderer::GetSwapchainTextureFormat() const {
    if (m_device == nullptr || m_window == nullptr) {
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
    return SDL_GetGPUSwapchainTextureFormat(m_device, m_window);
}

void GpuRenderer::SetGlobalShaderParams(const GlobalShaderParams& params) {
    m_globalShaderParams = params;
}

const GpuRenderer::GlobalShaderParams& GpuRenderer::GetGlobalShaderParams() const {
    return m_globalShaderParams;
}

void GpuRenderer::SetGlobalDensity(float density) {
    m_globalShaderParams.density = std::max(0.0f, density);
}

void GpuRenderer::SetGlobalQuality(float quality) {
    m_globalShaderParams.quality = std::max(0.001f, quality);
}

void GpuRenderer::SetGlobalTimeSeconds(float timeSeconds) {
    m_globalShaderParams.timeSeconds = timeSeconds;
}

void GpuRenderer::SetGlobalFrameIndex(float frameIndex) {
    m_globalShaderParams.frameIndex = frameIndex;
}

void GpuRenderer::SetObjectShaderParams(const ObjectShaderParams& params) {
    m_objectShaderParams = params;
}

const GpuRenderer::ObjectShaderParams& GpuRenderer::GetObjectShaderParams() const {
    return m_objectShaderParams;
}

void GpuRenderer::SetObjectDensity(float density) {
    m_objectShaderParams.density = std::max(0.0f, density);
}

void GpuRenderer::SetObjectQuality(float quality) {
    m_objectShaderParams.quality = std::max(0.001f, quality);
}

void GpuRenderer::SetObjectParticle(const glm::vec2& center, float particleRadius, float smoothingRadius) {
    m_objectShaderParams.particleEnabled = 1.0f;
    m_objectShaderParams.particleCenter = center;
    m_objectShaderParams.particleRadius = std::max(0.0f, particleRadius);
    m_objectShaderParams.smoothingRadius = std::max(0.0f, smoothingRadius);
}

void GpuRenderer::DisableObjectParticle() {
    m_objectShaderParams.particleEnabled = 0.0f;
    m_objectShaderParams.particleRadius = 0.0f;
    m_objectShaderParams.smoothingRadius = 0.0f;
}

void GpuRenderer::ResetObjectShaderParams() {
    m_objectShaderParams = ObjectShaderParams{};
}

bool GpuRenderer::UseWindowResolutionProjection2D(bool originTopLeft) {
    m_useWindowResolutionProjection = true;
    m_originTopLeft = originTopLeft;
    m_cachedWindowWidth = 0;
    m_cachedWindowHeight = 0;
    return SyncWindowResolutionProjection2D();
}

void GpuRenderer::DisableWindowResolutionProjection2D() {
    m_useWindowResolutionProjection = false;
}

bool GpuRenderer::SyncWindowResolutionProjection2D() {
    if (!m_useWindowResolutionProjection) {
        return true;
    }

    Uint32 width = 0;
    Uint32 height = 0;
    if (!GetWindowResolution(width, height)) {
        return false;
    }

    if (width == m_cachedWindowWidth && height == m_cachedWindowHeight) {
        return true;
    }

    if (m_originTopLeft) {
        SetProjection2D(0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f);
    } else {
        SetProjection2D(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height));
    }

    m_cachedWindowWidth = width;
    m_cachedWindowHeight = height;
    return true;
}

void GpuRenderer::SetView2D(const glm::vec2& cameraPosition) {
    m_view = glm::translate(glm::mat4(1.0f), glm::vec3(-cameraPosition, 0.0f));
    m_mvpDirty = true;
}

void GpuRenderer::SetModel2D(const glm::vec2& position, float rotationRadians, glm::vec2 scale) {
    m_model = BuildModel2D(position, rotationRadians, scale);
    m_mvpDirty = true;
}

void GpuRenderer::ResetModel2D() {
    m_model = glm::mat4(1.0f);
    m_mvpDirty = true;
}

void GpuRenderer::BeginScene2D() {
    m_fillBatch.clear();
    m_lineBatch.clear();
    m_queuedPrimitiveCount = 0;
    m_overflowLogged = false;
}

void GpuRenderer::DrawTriangle2D(const Triangle2D& triangle) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendTriangle(
            triangle.p0,
            triangle.p1,
            triangle.p2,
            ToPrimitiveColor(triangle.color),
            shaderParams,
            triangle.wireframe,
            triangle.lineThickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

void GpuRenderer::DrawTriangle2D(
    const glm::vec2& p0,
    const glm::vec2& p1,
    const glm::vec2& p2,
    Color color,
    bool wireframe,
    float lineThickness
) {
    Triangle2D triangle;
    triangle.p0 = p0;
    triangle.p1 = p1;
    triangle.p2 = p2;
    triangle.color = color;
    triangle.wireframe = wireframe;
    triangle.lineThickness = lineThickness;
    DrawTriangle2D(triangle);
}

void GpuRenderer::DrawRect2D(const Rect2D& rect) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendRect(
            rect.center,
            rect.size,
            rect.rotationRadians,
            ToPrimitiveColor(rect.color),
            shaderParams,
            rect.wireframe,
            rect.lineThickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

void GpuRenderer::DrawRect2D(
    const glm::vec2& position,
    const glm::vec2& size,
    Color color,
    bool wireframe,
    float rotationRadians,
    float lineThickness
) {
    Rect2D rect;
    rect.center = position;
    rect.size = size;
    rect.rotationRadians = rotationRadians;
    rect.color = color;
    rect.wireframe = wireframe;
    rect.lineThickness = lineThickness;
    DrawRect2D(rect);
}

void GpuRenderer::DrawCircle2D(const Circle2D& circle) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendCircle(
            circle.center,
            circle.radius,
            circle.segments,
            ToPrimitiveColor(circle.color),
            shaderParams,
            circle.wireframe,
            circle.lineThickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

void GpuRenderer::DrawCircle2D(
    const glm::vec2& center,
    float radius,
    Color color,
    bool wireframe,
    int segments,
    float lineThickness
) {
    Circle2D circle;
    circle.center = center;
    circle.radius = radius;
    circle.color = color;
    circle.wireframe = wireframe;
    circle.segments = segments;
    circle.lineThickness = lineThickness;
    DrawCircle2D(circle);
}

void GpuRenderer::DrawPolygon2D(const Polygon2D& polygon) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendPolygon(
            polygon.points,
            ToPrimitiveColor(polygon.color),
            shaderParams,
            polygon.wireframe,
            polygon.lineThickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

void GpuRenderer::DrawPolygon2D(
    const std::vector<glm::vec2>& points,
    Color color,
    bool wireframe,
    float lineThickness
) {
    Polygon2D polygon;
    polygon.points = points;
    polygon.color = color;
    polygon.wireframe = wireframe;
    polygon.lineThickness = lineThickness;
    DrawPolygon2D(polygon);
}

void GpuRenderer::DrawArc2D(const Arc2D& arc) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendArc(
            arc.center,
            arc.radius,
            arc.startRadians,
            arc.endRadians,
            arc.segments,
            ToPrimitiveColor(arc.color),
            shaderParams,
            arc.wireframe,
            arc.lineThickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

void GpuRenderer::DrawArc2D(
    const glm::vec2& center,
    float radius,
    float startRadians,
    float endRadians,
    Color color,
    bool wireframe,
    int segments,
    float lineThickness
) {
    Arc2D arc;
    arc.center = center;
    arc.radius = radius;
    arc.startRadians = startRadians;
    arc.endRadians = endRadians;
    arc.color = color;
    arc.wireframe = wireframe;
    arc.segments = segments;
    arc.lineThickness = lineThickness;
    DrawArc2D(arc);
}

void GpuRenderer::DrawLine2D(const glm::vec2& start, const glm::vec2& end, Color color, float thickness) {
    const size_t fillStart = m_fillBatch.size();
    const size_t lineStart = m_lineBatch.size();
    const PrimitiveShaderParams2D shaderParams = ToPrimitiveShaderParams(m_objectShaderParams);
    if (!PrimitiveBuilder2D::AppendLine(
            start,
            end,
            ToPrimitiveColor(color),
            shaderParams,
            thickness,
            m_fillBatch,
            m_lineBatch
        )) {
        m_fillBatch.resize(fillStart);
        m_lineBatch.resize(lineStart);
        return;
    }
    if (ValidateBatchCapacity(fillStart, lineStart)) {
        ++m_queuedPrimitiveCount;
    }
}

size_t GpuRenderer::GetQueuedDrawCount() const {
    return m_queuedPrimitiveCount;
}

bool GpuRenderer::Initialize(SDL_Window* window, const Config& config) {
    if (m_device != nullptr) {
        return true;
    }
    if (window == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "GpuRenderer::Initialize received null window");
        return false;
    }

    m_window = window;
    m_config = config;

    m_device = SDL_CreateGPUDevice(m_config.shaderFormats, m_config.debugMode, m_config.preferredBackend);
    if (m_device == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_CreateGPUDevice failed: %s", SDL_GetError());
        return false;
    }

    if (!SDL_ClaimWindowForGPUDevice(m_device, m_window)) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
        Shutdown();
        return false;
    }
    m_windowClaimed = true;

    if (!SDL_SetGPUAllowedFramesInFlight(m_device, 2)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "SDL_SetGPUAllowedFramesInFlight failed: %s", SDL_GetError());
    }

    const char* backend = SDL_GetGPUDeviceDriver(m_device);
    SDL_Log("SDL GPU backend: %s", backend != nullptr ? backend : "unknown");

    SetProjection2D(-1.0f, 1.0f, -1.0f, 1.0f);
    SetView2D(glm::vec2(0.0f, 0.0f));
    ResetModel2D();

    Uint32 width = 0;
    Uint32 height = 0;
    if (GetWindowResolution(width, height)) {
        m_globalShaderParams.viewportWidth = static_cast<float>(width);
        m_globalShaderParams.viewportHeight = static_cast<float>(height);
    }

    if (!CreateRenderResources()) {
        Shutdown();
        return false;
    }

    return true;
}

void GpuRenderer::Shutdown() {
    if (m_device != nullptr && !SDL_WaitForGPUIdle(m_device)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_RENDER, "SDL_WaitForGPUIdle failed: %s", SDL_GetError());
    }

    if (m_fillVertexBuffer != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUBuffer(m_device, m_fillVertexBuffer);
        m_fillVertexBuffer = nullptr;
    }

    if (m_lineVertexBuffer != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUBuffer(m_device, m_lineVertexBuffer);
        m_lineVertexBuffer = nullptr;
    }

    if (m_fillPipeline != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUGraphicsPipeline(m_device, m_fillPipeline);
        m_fillPipeline = nullptr;
    }

    if (m_linePipeline != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUGraphicsPipeline(m_device, m_linePipeline);
        m_linePipeline = nullptr;
    }

    if (m_fragmentShader != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUShader(m_device, m_fragmentShader);
        m_fragmentShader = nullptr;
    }

    if (m_vertexShader != nullptr && m_device != nullptr) {
        SDL_ReleaseGPUShader(m_device, m_vertexShader);
        m_vertexShader = nullptr;
    }

    if (m_windowClaimed && m_device != nullptr && m_window != nullptr) {
        SDL_ReleaseWindowFromGPUDevice(m_device, m_window);
        m_windowClaimed = false;
    }

    if (m_device != nullptr) {
        SDL_DestroyGPUDevice(m_device);
        m_device = nullptr;
    }

    m_fillBatch.clear();
    m_lineBatch.clear();
    m_queuedPrimitiveCount = 0;
    m_useWindowResolutionProjection = false;
    m_originTopLeft = true;
    m_cachedWindowWidth = 0;
    m_cachedWindowHeight = 0;
    m_globalShaderParams = GlobalShaderParams{};
    m_objectShaderParams = ObjectShaderParams{};
    m_window = nullptr;
}

bool GpuRenderer::RenderFrame() {
    return RenderFrame(FrameHooks{});
}

bool GpuRenderer::RenderFrame(const FrameHooks& hooks) {
    if (m_device == nullptr || m_window == nullptr || m_fillPipeline == nullptr || m_linePipeline == nullptr
        || m_fillVertexBuffer == nullptr || m_lineVertexBuffer == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_RENDER, "GpuRenderer is not initialized");
        return false;
    }

    if (m_useWindowResolutionProjection) {
        (void)SyncWindowResolutionProjection2D();
    }

    Uint32 width = 0;
    Uint32 height = 0;
    if (GetWindowResolution(width, height)) {
        m_globalShaderParams.viewportWidth = static_cast<float>(width);
        m_globalShaderParams.viewportHeight = static_cast<float>(height);
    }

    SDL_GPUCommandBuffer* commandBuffer = nullptr;
    SDL_GPUTexture* swapchainTexture = nullptr;
    if (!BeginFrame(commandBuffer, swapchainTexture)) {
        return false;
    }

    if (swapchainTexture == nullptr) {
        const bool submitted = EndFrame(commandBuffer);
        m_fillBatch.clear();
        m_lineBatch.clear();
        m_queuedPrimitiveCount = 0;
        return submitted;
    }

    if (!UploadVertices(commandBuffer, m_fillVertexBuffer, m_fillBatch)
        || !UploadVertices(commandBuffer, m_lineVertexBuffer, m_lineBatch)) {
        // Once swapchain acquisition succeeds, command buffers must be submitted
        // instead of canceled.
        (void)EndFrame(commandBuffer);
        m_fillBatch.clear();
        m_lineBatch.clear();
        m_queuedPrimitiveCount = 0;
        return false;
    }

    if (hooks.preRenderPass) {
        hooks.preRenderPass(commandBuffer);
    }

    SDL_GPURenderPass* renderPass = BeginMainRenderPass(commandBuffer, swapchainTexture);
    if (renderPass == nullptr) {
        (void)EndFrame(commandBuffer);
        m_fillBatch.clear();
        m_lineBatch.clear();
        m_queuedPrimitiveCount = 0;
        return false;
    }

    RecordShapeBatchDraw(commandBuffer, renderPass);
    if (hooks.onRenderPass) {
        hooks.onRenderPass(commandBuffer, renderPass);
    }
    SDL_EndGPURenderPass(renderPass);

    const bool submitted = EndFrame(commandBuffer);
    if (submitted) {
        m_globalShaderParams.frameIndex += 1.0f;
    }
    m_fillBatch.clear();
    m_lineBatch.clear();
    m_queuedPrimitiveCount = 0;
    return submitted;
}

const char* GpuRenderer::GetBackendName() const {
    if (m_device == nullptr) {
        return nullptr;
    }

    return SDL_GetGPUDeviceDriver(m_device);
}

}  // namespace sim::renderer
