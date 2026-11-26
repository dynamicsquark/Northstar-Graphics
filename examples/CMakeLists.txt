# =====================================================================
#  Northstar Graphics — Example Applications
# =====================================================================

cmake_minimum_required(VERSION 3.20)

# -------------------- Example 01: HelloPBR --------------------
add_executable(HelloPBR
    01_HelloPBR/main.cpp
)

target_include_directories(HelloPBR PRIVATE
    ${NORTHSTAR_REPO_ROOT}/northstar/include
    ${NORTHSTAR_REPO_ROOT}/common
    ${NORTHSTAR_REPO_ROOT}/libs/NSResourceLoader/include
    ${NORTHSTAR_REPO_ROOT}/libs/NSWindow/include
)

target_link_libraries(HelloPBR PRIVATE
    QuarkDynMath
    NSResourceLoader
    NSWindow
)

target_compile_features(HelloPBR PRIVATE cxx_std_20)
set_target_properties(HelloPBR PROPERTIES FOLDER "Examples")
northstar_configure_target_outputs(HelloPBR FALSE)
northstar_apply_strict_warnings(HelloPBR)
add_dependencies(HelloPBR NorthstarCore NorthstarD3D12 NorthstarStandardShaders)

# -------------------- Example 02: VirtualGeometryStudio --------------------
add_executable(VirtualGeometryStudio
    02_VirtualGeometryStudio/main.cpp
)

target_include_directories(VirtualGeometryStudio PRIVATE
    ${NORTHSTAR_REPO_ROOT}/northstar/include
    ${NORTHSTAR_REPO_ROOT}/common
    ${NORTHSTAR_REPO_ROOT}/libs/NSResourceLoader/include
    ${NORTHSTAR_REPO_ROOT}/libs/NSWindow/include
    ${NORTHSTAR_REPO_ROOT}/thirdparty
    ${NORTHSTAR_REPO_ROOT}/thirdparty/assimp/include
)

target_link_libraries(VirtualGeometryStudio PRIVATE
    QuarkDynMath
    NSResourceLoader
    NSWindow
    NorthstarImGui
    ${NORTHSTAR_REPO_ROOT}/thirdparty/assimp/lib/assimp-vc143-mt.lib
    d3d12
    dxgi
    comdlg32
)

target_compile_features(VirtualGeometryStudio PRIVATE cxx_std_20)
set_target_properties(VirtualGeometryStudio PROPERTIES FOLDER "Examples")
northstar_configure_target_outputs(VirtualGeometryStudio FALSE)
northstar_apply_strict_warnings(VirtualGeometryStudio)
add_dependencies(VirtualGeometryStudio NorthstarCore NorthstarD3D12 NorthstarStandardShaders)

add_custom_command(TARGET VirtualGeometryStudio POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${NORTHSTAR_REPO_ROOT}/thirdparty/assimp/bin/assimp.dll"
        "$<TARGET_FILE_DIR:VirtualGeometryStudio>/assimp.dll"
    COMMENT "Deploying assimp.dll next to VirtualGeometryStudio"
    VERBATIM
)
