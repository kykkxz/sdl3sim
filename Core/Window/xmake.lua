target("Window")
    set_kind("static")
    set_group("Core")

    add_files("Source/**.cpp")
    add_headerfiles("Include/**.h", "Include/**.hpp")
    add_includedirs("Include", {public = true})

    add_packages("sdl3", {public = true})
target_end()