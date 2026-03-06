target("Runtime")
    set_kind("binary")
    set_group("Runtime")
    add_options("enable_cuda", "cuda_arch")

    add_files("Source/**.cpp")
    add_files("ThirdParty/imgui/backends/imgui_impl_sdlgpu3.cpp")
    add_headerfiles("Include/**.h", "Include/**.hpp")
    add_headerfiles("ThirdParty/imgui/backends/**.h")
    add_includedirs("Include")
    add_includedirs("ThirdParty/imgui/backends")

    add_syslinks("advapi32", "user32", "gdi32", "shell32", "ole32", "oleaut32", "imm32", "version", "setupapi", "winmm")

    add_packages("imgui")
    add_deps("Core")

    if is_plat("windows") then
        add_cxxflags("/utf-8", {force = true})
    end

    if has_config("enable_cuda") then
        add_rules("cuda")
        add_files("Source/Cuda/**.cu")
        add_cuflags("--allow-unsupported-compiler", {force = true})
        add_defines("SIM_ENABLE_CUDA=1")
        add_cugencodes(get_config("cuda_arch"))
    else
        add_defines("SIM_ENABLE_CUDA=0")
    end
target_end()
