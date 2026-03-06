includes("Renderer/xmake.lua")
includes("Window/xmake.lua")
includes("ECS/xmake.lua")

target("Core")
    set_kind("phony")
    set_group("Core")
    add_deps("Window", "Renderer", "ECS")
target_end()
