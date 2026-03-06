#include <shaderc/shaderc.hpp>

#include <string>
#include <vector>
#include <fstream>

namespace sim::renderer {
static std::string ReadTextFile(const std::string& filePath) {
    std::ifstream ifs(filePath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        return "";
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return std::string(buffer.data(), buffer.size());
}

static bool CompileGLSLToSPIRV(
    const std::string& src,
    shaderc_shader_kind kind,
    const std::string& sourceName,
    std::vector<uint32_t>& spirvOut,
    std::string& errorOut
)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    options.SetTargetSpirv(shaderc_spirv_version_1_3);

    auto result = compiler.CompileGlslToSpv(src, kind, sourceName.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        errorOut = result.GetErrorMessage();
        return false;
    }
    spirvOut.assign(result.cbegin(), result.cend());
    return true;
}

}
