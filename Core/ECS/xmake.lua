target("ECS")
    set_kind("static")
    set_group("Core")

    add_files("Source/**.cpp")
    add_headerfiles("Include/**.h", "Include/**.hpp")
    add_includedirs("Include", {public = true})

    add_packages("glm", {public = true})
    add_packages("entt", {public = true})
target_end()
