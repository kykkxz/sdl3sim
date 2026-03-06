target("Renderer")
    set_kind("static")
    set_group("Core")

    add_files("Source/**.cpp")
    add_headerfiles("Include/**.h", "Include/**.hpp")
    add_includedirs("Include", {public = true})

    add_packages("sdl3", {public = true})
    add_packages("glm", {public = true})
    add_packages("shaderc", {public = true})
    add_packages("spirv_tools", {public = true})
    add_packages("glslang", {public = true})
target_end()
