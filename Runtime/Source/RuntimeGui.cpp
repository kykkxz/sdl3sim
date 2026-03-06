#include "RuntimeGui.hpp"

#include <SDL3/SDL_log.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>

#include "RuntimeProfiling.hpp"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlgpu3.h"

namespace sim::runtime {
namespace {

bool LoadChineseFont(ImGuiIO& io) {
    constexpr float kFontSizePx = 18.0f;
    const ImWchar* glyphRanges = io.Fonts->GetGlyphRangesChineseFull();
    ImFontConfig fontConfig{};
    fontConfig.OversampleH = 2;
    fontConfig.OversampleV = 1;
    fontConfig.PixelSnapH = true;

    static constexpr std::array<const char*, 7> kFontCandidates = {
        "C:/Windows/Fonts/msyh.ttc",      // Microsoft YaHei
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",    // SimHei
        "C:/Windows/Fonts/simsun.ttc",    // SimSun
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/Deng.ttf",      // DengXian
        "C:/Windows/Fonts/simfang.ttf"    // FangSong
    };

    for (const char* fontPath : kFontCandidates) {
        if (!std::filesystem::exists(fontPath)) {
            continue;
        }
        ImFont* font = io.Fonts->AddFontFromFileTTF(fontPath, kFontSizePx, &fontConfig, glyphRanges);
        if (font != nullptr) {
            io.FontDefault = font;
            SDL_Log("Loaded ImGui CJK font: %s", fontPath);
            return true;
        }
    }

    io.Fonts->AddFontDefault();
    SDL_LogWarn(
        SDL_LOG_CATEGORY_APPLICATION,
        "No CJK-capable font was loaded for ImGui. Chinese text may render as '?'."
    );
    return false;
}

}  // namespace

RuntimeGui::~RuntimeGui() {
    Shutdown();
}

bool RuntimeGui::Initialize(SDL_Window* window, sim::renderer::GpuRenderer& renderer) {
    if (m_initialized) {
        return true;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    LoadChineseFont(io);

    if (!ImGui_ImplSDL3_InitForSDLGPU(window)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize ImGui SDL3 backend");
        ImGui::DestroyContext();
        return false;
    }

    ImGui_ImplSDLGPU3_InitInfo initInfo{};
    initInfo.Device = renderer.GetDevice();
    initInfo.ColorTargetFormat = renderer.GetSwapchainTextureFormat();
    initInfo.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    if (!ImGui_ImplSDLGPU3_Init(&initInfo)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize ImGui SDLGPU3 backend");
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    m_initialized = true;
    return true;
}

void RuntimeGui::Shutdown() {
    if (!m_initialized) {
        return;
    }

    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    m_initialized = false;
}

void RuntimeGui::ProcessEvent(const SDL_Event& event) const {
    if (!m_initialized) {
        return;
    }
    ImGui_ImplSDL3_ProcessEvent(&event);
}

void RuntimeGui::BeginFrame() const {
    if (!m_initialized) {
        return;
    }
    ImGui_ImplSDLGPU3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

bool RuntimeGui::WantsMouseCapture() const {
    if (!m_initialized) {
        return false;
    }
    return ImGui::GetIO().WantCaptureMouse;
}

EditorActions RuntimeGui::BuildRuntimeEditor(
    UiState& state,
    float deltaSeconds,
    float fps
) const {
    EditorActions actions;
    if (!m_initialized) {
        return actions;
    }

    const RuntimeFrameProfiling& profiling = GetRuntimeFrameProfiling();
    const float safeTargetDensityErrorRatio = std::clamp(state.update.targetDensityErrorRatio, 0.001f, 0.2f);
    const float safeTargetDensityErrorPercent = safeTargetDensityErrorRatio * 100.0f;
    ImGui::SetNextWindowSize(ImVec2(560.0f, 820.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("运行时编辑器");
    ImGui::Text("帧时间: %.2f ms (%.1f FPS)", deltaSeconds * 1000.0f, fps);
    ImGui::SeparatorText("仿真控制");
    if (ImGui::RadioButton("运行中", state.runtime.state == RuntimeState::Running)) {
        state.runtime.state = RuntimeState::Running;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("暂停", state.runtime.state == RuntimeState::Paused)) {
        state.runtime.state = RuntimeState::Paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("重置到初始状态")) {
        actions.resetToInitialState = true;
    }
    if (ImGui::BeginTabBar("RuntimeEditorTabs")) {
        if (ImGui::BeginTabItem("性能")) {
            ImGui::Text(
                "总计: 更新 %.3f ms | 绘制 %.3f ms",
                profiling.updateTotalMs,
                profiling.drawTotalMs
            );
            ImGui::SeparatorText("更新阶段");
            ImGui::BulletText("场景同步: %.3f ms", profiling.updateSceneSyncMs);
            ImGui::BulletText("粒子校验: %.3f ms", profiling.updateBallValidationMs);
            ImGui::BulletText("盒体校验: %.3f ms", profiling.updateBoxValidationMs);
            ImGui::BulletText("粒子物理: %.3f ms", profiling.updateBallPhysicsMs);
            ImGui::BulletText("粒子密度: %.3f ms", profiling.updateParticleDensityMs);
            ImGui::BulletText("变换归一化: %.3f ms", profiling.updateTransformNormalizeMs);
            ImGui::SeparatorText("绘制阶段");
            ImGui::BulletText("密度网格: %.3f ms", profiling.drawDensityViewGridMs);
            ImGui::BulletText("密度覆盖: %.3f ms", profiling.drawDensityViewOverlayMs);
            ImGui::BulletText("粒子绘制: %.3f ms", profiling.drawBallMs);
            ImGui::BulletText("边界绘制: %.3f ms", profiling.drawBoxMs);
            constexpr std::size_t kRendererFillVertexBudget = 120000;
            const int safeSegments = std::clamp(state.world.ballSegments, 6, 96);
            const std::size_t particleCountHint = static_cast<std::size_t>(
                std::max(0, state.world.particleCount)
            );
            const std::size_t estimatedCircleFillVertices =
                particleCountHint * static_cast<std::size_t>(safeSegments) * 3u;
            const std::size_t estimatedBillboardFillVertices = particleCountHint * 6u;
            const bool useParticleBillboards = estimatedCircleFillVertices > kRendererFillVertexBudget;
            ImGui::SeparatorText("粒子渲染预算估算");
            ImGui::BulletText(
                "圆形估算顶点: %llu | Billboard估算顶点: %llu | 预算: %llu",
                static_cast<unsigned long long>(estimatedCircleFillVertices),
                static_cast<unsigned long long>(estimatedBillboardFillVertices),
                static_cast<unsigned long long>(kRendererFillVertexBudget)
            );
            ImGui::BulletText(
                "当前粒子渲染路径: %s",
                useParticleBillboards ? "Billboard(6顶点/粒子)" : "Circle(segments*3顶点/粒子)"
            );
            ImGui::SeparatorText("物理诊断");
            ImGui::Text(
                "子步数: %d | 子步时长: %.4f ms | 最大密度误差: %.3f",
                profiling.physicsSubsteps,
                profiling.physicsSubstepDtMs,
                profiling.physicsMaxDensityErrorRatio
            );
            ImGui::BulletText(
                "建议重力上限(%.2f%%密度误差): %.6f px/s^2",
                safeTargetDensityErrorPercent,
                profiling.physicsRecommendedGravityMax
            );
            ImGui::BulletText(
                "求解刚度(自动校准后): %.3e",
                profiling.physicsEffectiveStiffness
            );
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("物理")) {
            ImGui::SeparatorText("核心参数（可调）");
            ImGui::SliderFloat("质量密度参考", &state.update.massDensityReference, 0.1f, 300.0f, "%.2f");
            ImGui::SliderFloat("目标密度参考（静止密度）", &state.update.restDensity, 0.1f, 300.0f, "%.2f");
            ImGui::SliderFloat(
                "压力刚度",
                &state.update.stiffness,
                10.0f,
                50000.0f,
                "%.2f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::SliderFloat("状态方程指数", &state.update.gamma, 1.0f, 10.0f, "%.2f");
            ImGui::SliderFloat("允许密度误差(重力建议)", &state.update.targetDensityErrorRatio, 0.001f, 0.2f, "%.3f");
            ImGui::SliderFloat("压力上限（0=关闭）", &state.update.maxPressureLimit, 0.0f, 300000.0f, "%.0f");
            ImGui::SliderInt("最大子步预算", &state.update.maxSubsteps, 1, 32);
            ImGui::SliderFloat(
                "最小子步(s)",
                &state.update.minTimeStep,
                1.0f / 5000.0f,
                1.0f / 120.0f,
                "%.5f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::SliderFloat(
                "最大子步(s)",
                &state.update.maxTimeStep,
                std::max(state.update.minTimeStep, 1.0f / 5000.0f),
                1.0f / 30.0f,
                "%.5f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::Checkbox("启用重力", &state.update.enableGravity);
            const float recommendedGravity = std::max(0.0f, profiling.physicsRecommendedGravityMax);
            const float gravityReference = std::max(0.001f, recommendedGravity);
            float gravityRatio = state.update.gravity / gravityReference;
            gravityRatio = std::clamp(gravityRatio, 0.01f, 2000.0f);
            if (ImGui::SliderFloat(
                "重力倍率(相对建议)",
                &gravityRatio,
                0.01f,
                2000.0f,
                "%.2fx",
                ImGuiSliderFlags_Logarithmic)) {
                state.update.gravity = gravityRatio * gravityReference;
            }
            if (ImGui::Button("重力归零")) {
                state.update.gravity = 0.0f;
            }
            ImGui::SameLine();
            if (ImGui::Button("设为建议值")) {
                state.update.gravity = recommendedGravity;
            }
            ImGui::DragFloat("重力绝对值", &state.update.gravity, 1.0f, 0.0f, 20000.0f, "%.3f px/s^2");
            if (recommendedGravity > 0.0f) {
                ImGui::Text("当前/建议: %.2fx", state.update.gravity / recommendedGravity);
            } else {
                ImGui::TextDisabled("当前建议值不可用，请使用绝对值调节。");
            }
            ImGui::SliderFloat("速度上限（0=关闭）", &state.update.maxVelocityLimit, 0.0f, 6000.0f, "%.0f px/s");
            ImGui::SliderFloat("鼠标作用半径", &state.update.mouseForceRadius, 20.0f, 360.0f, "%.0f px");
            ImGui::SliderFloat(
                "鼠标作用强度（0=关闭）",
                &state.update.mouseForceStrength,
                0.0f,
                30000.0f,
                "%.0f px/s^2"
            );
            ImGui::SliderFloat(
                "左键推开声速比上限",
                &state.update.mousePushMachCap,
                1.0f,
                80.0f,
                "%.2f",
                ImGuiSliderFlags_Logarithmic
            );
            ImGui::SliderFloat("XSPH速度混合", &state.update.xsphVelocityBlend, 0.0f, 0.5f, "%.3f");
            ImGui::SliderFloat(
                "平滑半径",
                &state.update.smoothingRadius,
                state.world.ballRadius,
                std::max(state.world.ballRadius + 1.0f, 180.0f),
                "%.1f px"
            );

            ImGui::SeparatorText("稳定性参数（诊断）");
            ImGui::Text(
                "压力刚度(基础/求解): %.2f / %.3e",
                state.update.stiffness,
                profiling.physicsEffectiveStiffness
            );
            ImGui::Text("状态方程指数: %.2f", state.update.gamma);
            ImGui::Text("粘性: %.3f", state.update.viscosity);
            ImGui::Text("CFL 因子: %.2f", state.update.cflFactor);
            ImGui::Text("最大子步数: %d", state.update.maxSubsteps);
            ImGui::Text("最小步长: %.4f s", state.update.minTimeStep);
            ImGui::Text("最大步长: %.4f s", state.update.maxTimeStep);
            ImGui::Text("碰撞恢复系数: %.3f", state.update.restitution);
            ImGui::Text("线性阻尼: %.3f", state.update.linearDamping);
            ImGui::Text(
                "当前刚度建议重力上限(%.2f%%密度误差): %.6g px/s^2",
                safeTargetDensityErrorPercent,
                profiling.physicsRecommendedGravityMax
            );
            if (state.update.enableGravity
                && profiling.physicsRecommendedGravityMax > 0.0f
                && state.update.gravity > profiling.physicsRecommendedGravityMax) {
                ImGui::TextColored(
                    ImVec4(1.0f, 0.56f, 0.18f, 1.0f),
                    "当前重力超出建议上限，系统将自动提高求解刚度。"
                );
            }
            ImGui::Text("密度采样步长: %.1f px", state.update.densityViewSampleStep);
            ImGui::Text("密度着色混合: %.2f", state.update.densityColorMix);
            ImGui::TextDisabled("粒子质量 = 质量密度参考 x (盒体面积 / 粒子数)，与静止密度独立。");
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("场景")) {
            ImGui::SeparatorText("粒子");
            ImGui::InputInt("粒子数", &state.world.particleCount, 100, 1000);
            ImGui::SameLine();
            if (ImGui::Button("应用粒子数")) {
                actions.rebuildParticleEntities = true;
            }
            ImGui::TextDisabled("应用后会重建粒子并重置到初始分布。");
            ImGui::TextDisabled("范围: %d - %d", kMinParticleCount, kMaxParticleCount);
            ImGui::Text("粒子偏移 X: %.1f px", state.world.ballOffsetX);
            ImGui::Text("粒子偏移 Y: %.1f px", state.world.ballOffsetY);
            ImGui::Text("粒子半径: %.2f px", state.world.ballRadius);
            ImGui::Text("粒子分段: %d", state.world.ballSegments);
            ImGui::Text(
                "粒子颜色: (%.2f, %.2f, %.2f)",
                state.world.ballColor[0],
                state.world.ballColor[1],
                state.world.ballColor[2]
            );

            ImGui::SeparatorText("边界盒");
            ImGui::Text("边界内缩: %.1f px", state.world.boxInset);
            ImGui::Text("边框线宽: %.2f px", state.world.boxLineThickness);
            ImGui::Text(
                "边框颜色: (%.2f, %.2f, %.2f)",
                state.world.boxColor[0],
                state.world.boxColor[1],
                state.world.boxColor[2]
            );
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("渲染")) {
            ImGui::SeparatorText("可视化开关");
            ImGui::Checkbox("绘制粒子", &state.render.ball);
            ImGui::Checkbox("绘制边界盒", &state.render.boundingBox);
            ImGui::Checkbox("渲染密度场", &state.render.densityField);
            ImGui::Checkbox("密度视图", &state.render.densityView);
            ImGui::SeparatorText("着色器全局参数");
            ImGui::SliderFloat("全局着色密度", &state.shaderGlobals.density, 0.0f, 3.0f, "%.3f");
            ImGui::SliderFloat("全局着色质量", &state.shaderGlobals.quality, 0.001f, 3.0f, "%.3f");
            ImGui::ColorEdit3("清屏颜色", state.clearColor);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

    state.world.ballRadius = std::max(1.0f, state.world.ballRadius);
    state.world.particleCount = std::clamp(state.world.particleCount, kMinParticleCount, kMaxParticleCount);
    state.world.ballSegments = std::clamp(state.world.ballSegments, 6, 96);
    state.world.boxInset = std::max(0.0f, state.world.boxInset);
    state.world.boxLineThickness = std::max(0.0f, state.world.boxLineThickness);
    state.update.gravity = std::max(0.0f, state.update.gravity);
    state.update.massDensityReference = std::max(0.01f, state.update.massDensityReference);
    state.update.restDensity = std::max(0.01f, state.update.restDensity);
    state.update.stiffness = std::max(0.0f, state.update.stiffness);
    state.update.gamma = std::max(1.0f, state.update.gamma);
    state.update.targetDensityErrorRatio = std::clamp(state.update.targetDensityErrorRatio, 0.001f, 0.2f);
    state.update.viscosity = std::max(0.0f, state.update.viscosity);
    state.update.cflFactor = std::clamp(state.update.cflFactor, 0.05f, 1.0f);
    state.update.maxSubsteps = std::clamp(state.update.maxSubsteps, 1, 32);
    state.update.minTimeStep = std::max(0.0001f, state.update.minTimeStep);
    state.update.maxTimeStep = std::max(state.update.minTimeStep, state.update.maxTimeStep);
    state.update.restitution = std::max(0.0f, state.update.restitution);
    state.update.linearDamping = std::max(0.0f, state.update.linearDamping);
    state.update.maxVelocityLimit = std::max(0.0f, state.update.maxVelocityLimit);
    state.update.mouseForceRadius = std::max(1.0f, state.update.mouseForceRadius);
    state.update.mouseForceStrength = std::max(0.0f, state.update.mouseForceStrength);
    state.update.mousePushMachCap = std::clamp(state.update.mousePushMachCap, 1.0f, 80.0f);
    state.update.xsphVelocityBlend = std::clamp(state.update.xsphVelocityBlend, 0.0f, 0.5f);
    state.update.maxPressureLimit = std::max(0.0f, state.update.maxPressureLimit);
    state.update.smoothingRadius = std::max(state.update.smoothingRadius, state.world.ballRadius);
    state.update.densityViewSampleStep = std::max(1.0f, state.update.densityViewSampleStep);
    state.update.densityColorMix = std::clamp(state.update.densityColorMix, 0.0f, 1.0f);
    state.shaderGlobals.density = std::max(0.0f, state.shaderGlobals.density);
    state.shaderGlobals.quality = std::max(0.001f, state.shaderGlobals.quality);

    return actions;
}

void RuntimeGui::EndFrame() const {
    if (!m_initialized) {
        return;
    }
    ImGui::Render();
}

void RuntimeGui::PrepareDrawData(SDL_GPUCommandBuffer* commandBuffer) const {
    if (!m_initialized) {
        return;
    }
    Imgui_ImplSDLGPU3_PrepareDrawData(ImGui::GetDrawData(), commandBuffer);
}

void RuntimeGui::RenderDrawData(SDL_GPUCommandBuffer* commandBuffer, SDL_GPURenderPass* renderPass) const {
    if (!m_initialized) {
        return;
    }
    ImGui_ImplSDLGPU3_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderPass);
}

}  // namespace sim::runtime
