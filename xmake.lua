set_project("SDL3Simulator")
set_version("0.1.0")
set_xmakever("2.8.0")

-- Build modes
add_rules("mode.debug", "mode.release")

-- C++20 standard
set_languages("c++20")
set_runtimes("MD")

option("enable_cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA backend (requires local CUDA toolkit)")
option_end()

option("cuda_arch")
    set_default("native")
    set_showmenu(true)
    set_description("CUDA architecture gencode (e.g. native, sm_86)")
option_end()

-- Use vcpkg package manager
add_requires("vcpkg::sdl3", {alias = "sdl3"})
add_requires("vcpkg::glm", {alias = "glm"})
add_requires("vcpkg::shaderc", {alias = "shaderc"})
add_requires("vcpkg::spirv-tools", {alias = "spirv_tools"})
add_requires("vcpkg::glslang", {alias = "glslang"})
add_requires("vcpkg::entt", {alias = "entt"})
add_requires("vcpkg::imgui[sdl3-binding,docking-experimental]", {alias = "imgui"})

-- Output directories
set_targetdir("$(builddir)/$(plat)/$(arch)/$(mode)")
set_objectdir("$(builddir)/.objs/$(plat)/$(arch)/$(mode)")

includes("Core/xmake.lua")
includes("Runtime/xmake.lua")
