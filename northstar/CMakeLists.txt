# =====================================================================
#  Northstar Graphics — Runtime, Hardware Backends & Shader Packaging
# =====================================================================

cmake_minimum_required(VERSION 3.20)

set(NORTHSTAR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(NORTHSTAR_SHADERS_SRC_DIR "${NORTHSTAR_ROOT}/../shaders")
set(NORTHSTAR_GEN_ROOT_DIR    "${CMAKE_BINARY_DIR}/intermediate/generated")

file(GLOB_RECURSE _NORTHSTAR_LIBRARY_NSL_FILES CONFIGURE_DEPENDS
    "${NORTHSTAR_SHADERS_SRC_DIR}/NorthstarLibrary/*.nsl"
)

# -------------------- 1. NorthstarRuntime Internal Shaders (.nsl -> build/generated/.nsc) --------------------
set(NORTHSTAR_RUNTIME_SHADER_ENTRIES
    "Compute/RuntimeBufferClear"
    "Compute/RuntimeDirtyDepthSliceBuild"
    "Compute/RuntimeGPUSceneDiagnostic"
    "Compute/RuntimeNVSPageResidency"
    "Compute/RuntimeNVSFeedback"
    "Compute/RuntimeNVSPageProjection"
    "Compute/RuntimeNVSCasterCull"
    "Compute/RuntimeNVSIndirectDraw"
    "Compute/RuntimeNVGHZBBuild"
    "Compute/RuntimeNVGResidencyMask"
    "Compute/RuntimeNVGInstanceCull"
    "Compute/RuntimeNVGHierarchyCull"
    "Compute/RuntimeNVGCompactionAndDecode"
    "Compute/RuntimeNVGIndirectDraw"
    "Compute/RuntimeNVGShadowCull"
    "Compute/RuntimeNVGShadowMicropolyRaster"
    "Compute/RuntimeNVGDepthExport"
    "Compute/RuntimeReflectionMaterialize"
    "Depth/RuntimeLayeredDepthClearVS"
    "Depth/RuntimeLayeredDepthPassGS"
    "Depth/RuntimeNVGShadowSWMergeVS"
    "Depth/RuntimeNVGShadowSWMergeGS"
    "Depth/RuntimeNVGShadowSWMergePS"
)

set(_NORTHSTAR_RUNTIME_GEN_DIR "${NORTHSTAR_GEN_ROOT_DIR}/NorthstarRuntime")
set(_NORTHSTAR_RUNTIME_GEN_OUTPUTS "")

foreach(_RT_ENTRY ${NORTHSTAR_RUNTIME_SHADER_ENTRIES})
    get_filename_component(_RT_CATEGORY "${_RT_ENTRY}" DIRECTORY)
    get_filename_component(_RT_SHADER   "${_RT_ENTRY}" NAME)

    set(_RT_NSL       "${NORTHSTAR_SHADERS_SRC_DIR}/NorthstarRuntime/${_RT_CATEGORY}/${_RT_SHADER}.nsl")
    set(_RT_GEN_NSC11 "${_NORTHSTAR_RUNTIME_GEN_DIR}/${_RT_CATEGORY}/${_RT_SHADER}_hlsl11.nsc")
    set(_RT_GEN_NSC12 "${_NORTHSTAR_RUNTIME_GEN_DIR}/${_RT_CATEGORY}/${_RT_SHADER}_hlsl12.nsc")

    add_custom_command(
        OUTPUT
            "${_RT_GEN_NSC11}"
            "${_RT_GEN_NSC12}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${_NORTHSTAR_RUNTIME_GEN_DIR}/${_RT_CATEGORY}"
        COMMAND $<TARGET_FILE:NSLCompiler>
            --input "${_RT_NSL}"
            --target hlsl11
            --output "${_RT_GEN_NSC11}"
            -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        COMMAND $<TARGET_FILE:NSLCompiler>
            --input "${_RT_NSL}"
            --target hlsl12
            --output "${_RT_GEN_NSC12}"
            -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        DEPENDS
            "${_RT_NSL}"
            ${_NORTHSTAR_LIBRARY_NSL_FILES}
            NSLCompiler
        COMMENT "Compiling NorthstarRuntime/${_RT_CATEGORY}/${_RT_SHADER}.nsl -> .nsc"
        VERBATIM
    )

    list(APPEND _NORTHSTAR_RUNTIME_GEN_OUTPUTS "${_RT_GEN_NSC11}" "${_RT_GEN_NSC12}")
endforeach()

# Unified Universal Camera Micropolygon Rasterizer specializations from ONE authoritative NSL source:
#   - NVG_RASTER_ATOMIC64=1 -> RuntimeNVGMicropolyRasterAtomic64_hlsl12.nsc (SM 6.6 DXIL single-pass 64-bit InterlockedMax ONLY; never HLSL11/SM 5.0)
#   - NVG_RASTER_ATOMIC64=0 -> RuntimeNVGMicropolyRasterAtomic32_{hlsl11,hlsl12}.nsc (universal two-pass 2x32-bit InterlockedMax)
set(_NVG_RASTER_UNIFIED_NSL "${NORTHSTAR_SHADERS_SRC_DIR}/NorthstarRuntime/Compute/RuntimeNVGMicropolyRaster.nsl")
set(_NVG_A64_NSC12 "${_NORTHSTAR_RUNTIME_GEN_DIR}/Compute/RuntimeNVGMicropolyRasterAtomic64_hlsl12.nsc")
add_custom_command(
    OUTPUT
        "${_NVG_A64_NSC12}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${_NORTHSTAR_RUNTIME_GEN_DIR}/Compute"
    COMMAND $<TARGET_FILE:NSLCompiler>
        --input "${_NVG_RASTER_UNIFIED_NSL}"
        --target hlsl12
        --output "${_NVG_A64_NSC12}"
        -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        -D "NVG_RASTER_ATOMIC64=1"
    DEPENDS
        "${_NVG_RASTER_UNIFIED_NSL}"
        ${_NORTHSTAR_LIBRARY_NSL_FILES}
        NSLCompiler
    COMMENT "Compiling unified RuntimeNVGMicropolyRaster.nsl (NVG_RASTER_ATOMIC64=1, hlsl12) -> RuntimeNVGMicropolyRasterAtomic64_hlsl12.nsc"
    VERBATIM
)
list(APPEND _NORTHSTAR_RUNTIME_GEN_OUTPUTS "${_NVG_A64_NSC12}")

set(_NVG_A32_NSC11 "${_NORTHSTAR_RUNTIME_GEN_DIR}/Compute/RuntimeNVGMicropolyRasterAtomic32_hlsl11.nsc")
set(_NVG_A32_NSC12 "${_NORTHSTAR_RUNTIME_GEN_DIR}/Compute/RuntimeNVGMicropolyRasterAtomic32_hlsl12.nsc")
add_custom_command(
    OUTPUT
        "${_NVG_A32_NSC11}"
        "${_NVG_A32_NSC12}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${_NORTHSTAR_RUNTIME_GEN_DIR}/Compute"
    COMMAND $<TARGET_FILE:NSLCompiler>
        --input "${_NVG_RASTER_UNIFIED_NSL}"
        --target hlsl11
        --output "${_NVG_A32_NSC11}"
        -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        -D "NVG_RASTER_ATOMIC64=0"
    COMMAND $<TARGET_FILE:NSLCompiler>
        --input "${_NVG_RASTER_UNIFIED_NSL}"
        --target hlsl12
        --output "${_NVG_A32_NSC12}"
        -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        -D "NVG_RASTER_ATOMIC64=0"
    DEPENDS
        "${_NVG_RASTER_UNIFIED_NSL}"
        ${_NORTHSTAR_LIBRARY_NSL_FILES}
        NSLCompiler
    COMMENT "Compiling unified RuntimeNVGMicropolyRaster.nsl (NVG_RASTER_ATOMIC64=0) -> RuntimeNVGMicropolyRasterAtomic32.nsc"
    VERBATIM
)
list(APPEND _NORTHSTAR_RUNTIME_GEN_OUTPUTS "${_NVG_A32_NSC11}" "${_NVG_A32_NSC12}")

add_custom_target(NorthstarRuntimeShaders ALL
    DEPENDS ${_NORTHSTAR_RUNTIME_GEN_OUTPUTS}
)
set_target_properties(NorthstarRuntimeShaders PROPERTIES FOLDER "Engine")

# -------------------- 2. StandardLit Built-in Shaders (.nsl -> build/generated/shaders/StandardLit/.nsc) --------------------
set(NORTHSTAR_STANDARD_LIT_ENTRIES
    "StandardLitVS"
    "StandardLitInstancedVS"
    "StandardLitSkinnedVS"
    "StandardLitInstancedSkinnedVS"
    "StandardLitNVGVS"
    "StandardLitPS"
    "StandardLitShadowVS"
    "StandardLitShadowInstancedVS"
    "StandardLitShadowSkinnedVS"
    "StandardLitShadowInstancedSkinnedVS"
    "StandardLitNVGShadowVS"
    "StandardLitShadowPS"
)

set(_NORTHSTAR_STDLIT_GEN_DIR "${NORTHSTAR_GEN_ROOT_DIR}/shaders/StandardLit")
set(_NORTHSTAR_STDLIT_GEN_OUTPUTS "")

foreach(_STD_SHADER ${NORTHSTAR_STANDARD_LIT_ENTRIES})
    set(_STD_NSL       "${NORTHSTAR_SHADERS_SRC_DIR}/StandardLit/${_STD_SHADER}.nsl")
    set(_STD_GEN_NSC11 "${_NORTHSTAR_STDLIT_GEN_DIR}/${_STD_SHADER}_hlsl11.nsc")
    set(_STD_GEN_NSC12 "${_NORTHSTAR_STDLIT_GEN_DIR}/${_STD_SHADER}_hlsl12.nsc")

    add_custom_command(
        OUTPUT
            "${_STD_GEN_NSC11}"
            "${_STD_GEN_NSC12}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${_NORTHSTAR_STDLIT_GEN_DIR}"
        COMMAND $<TARGET_FILE:NSLCompiler>
            --input "${_STD_NSL}"
            --target hlsl11
            --output "${_STD_GEN_NSC11}"
            -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        COMMAND $<TARGET_FILE:NSLCompiler>
            --input "${_STD_NSL}"
            --target hlsl12
            --output "${_STD_GEN_NSC12}"
            -I "${NORTHSTAR_SHADERS_SRC_DIR}"
        DEPENDS
            "${_STD_NSL}"
            ${_NORTHSTAR_LIBRARY_NSL_FILES}
            NSLCompiler
        COMMENT "Compiling StandardLit/${_STD_SHADER}.nsl -> .nsc"
        VERBATIM
    )

    list(APPEND _NORTHSTAR_STDLIT_GEN_OUTPUTS "${_STD_GEN_NSC11}" "${_STD_GEN_NSC12}")
endforeach()

add_custom_target(NorthstarStandardShaders ALL
    DEPENDS ${_NORTHSTAR_STDLIT_GEN_OUTPUTS}
)
set_target_properties(NorthstarStandardShaders PROPERTIES FOLDER "Engine")

# -------------------- 3. NorthstarCore Shared Library --------------------
add_library(NorthstarCore SHARED
    core/runtime/northstar_core.cpp
    core/runtime/core_shader_manager.cpp
    core/runtime/core_texture_manager.cpp
    core/runtime/core_buffer_manager.cpp
    core/runtime/core_geometry_manager.cpp
    core/runtime/core_material_manager.cpp
    core/runtime/core_compute_manager.cpp
    core/runtime/core_scene_manager.cpp
    core/runtime/core_gpu_scene_manager.cpp
    core/runtime/core_resource_binding_resolver.cpp
    core/runtime/core_render_pipeline.cpp
    core/runtime/core_nvs_page_manager.cpp
    core/runtime/core_nvs_shadow_rasterizer.cpp
    core/runtime/core_nvs_feedback_manager.cpp
    core/runtime/internal_resource_manager.cpp
    core/nvg/nvg_codec_lz4.cpp
    core/nvg/nvg_qem_simplifier.cpp
    core/nvg/nvg_builder.cpp
    core/nvg/nvg_async_stream_io.cpp
    core/nvg/nvg_sw_raster_ref.cpp
    core/nvg/core_nvg_residency_manager.cpp
    core/nvg/core_nvg_manager.cpp
    core/shader_loader/shader_loader.cpp
)

target_compile_definitions(NorthstarCore PRIVATE NORTHSTAR_CORE_EXPORTS)

target_include_directories(NorthstarCore PUBLIC
    ${NORTHSTAR_ROOT}/include
    ${NORTHSTAR_ROOT}/../common
    ${NORTHSTAR_ROOT}/../libs
)

target_include_directories(NorthstarCore PRIVATE
    ${NORTHSTAR_ROOT}
)

target_link_libraries(NorthstarCore PUBLIC
    QuarkDynMath
)
target_link_libraries(NorthstarCore PRIVATE
    NSCReader
)

target_compile_features(NorthstarCore PUBLIC cxx_std_20)
set_target_properties(NorthstarCore PROPERTIES FOLDER "Engine")
northstar_configure_target_outputs(NorthstarCore TRUE)
northstar_apply_strict_warnings(NorthstarCore)

add_dependencies(NorthstarCore NorthstarRuntimeShaders NorthstarStandardShaders)

add_custom_command(TARGET NorthstarCore POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${_NORTHSTAR_RUNTIME_GEN_DIR}"
        "$<TARGET_FILE_DIR:NorthstarCore>/NorthstarRuntime"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${_NORTHSTAR_STDLIT_GEN_DIR}"
        "$<TARGET_FILE_DIR:NorthstarCore>/shaders/StandardLit"
    COMMENT "Deploying compiled NorthstarRuntime and StandardLit .nsc packages to active output directory"
    VERBATIM
)

# -------------------- 4. NorthstarD3D12 Shared Library --------------------
add_library(NorthstarD3D12 SHARED
    platform/d3d12/runtime/northstar_d3d12.cpp
    platform/d3d12/runtime/d3d12_device.cpp
    platform/d3d12/runtime/d3d12_swapchain.cpp
    platform/d3d12/runtime/d3d12_descriptor_pool.cpp
    platform/d3d12/runtime/d3d12_resource_manager.cpp
    platform/d3d12/runtime/d3d12_resource_state_tracker.cpp
    platform/d3d12/runtime/d3d12_pipeline.cpp
)

target_compile_definitions(NorthstarD3D12 PRIVATE NORTHSTAR_D3D12_EXPORTS)

target_include_directories(NorthstarD3D12 PUBLIC
    ${NORTHSTAR_ROOT}/include
    ${NORTHSTAR_ROOT}/../common
    ${NORTHSTAR_ROOT}/../libs
)

target_include_directories(NorthstarD3D12 PRIVATE
    ${NORTHSTAR_ROOT}
)

target_link_libraries(NorthstarD3D12 PRIVATE
    QuarkDynMath
    d3d12
    dxgi
    dxguid
    d3dcompiler
)

target_compile_features(NorthstarD3D12 PUBLIC cxx_std_20)
set_target_properties(NorthstarD3D12 PROPERTIES FOLDER "Engine")
northstar_configure_target_outputs(NorthstarD3D12 TRUE)
northstar_apply_strict_warnings(NorthstarD3D12)

if(WIN32)
    set(_NS_DXC_DLL "")
    set(_NS_DXIL_DLL "")
    set(_NS_WINSDK_HINTS "")
    if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        list(APPEND _NS_WINSDK_HINTS
            "C:/Program Files (x86)/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        )
    endif()
    file(GLOB _NS_WINSDK_BINS "C:/Program Files (x86)/Windows Kits/10/bin/10.0.*/x64")
    if(_NS_WINSDK_BINS)
        list(SORT _NS_WINSDK_BINS ORDER DESCENDING)
        list(APPEND _NS_WINSDK_HINTS ${_NS_WINSDK_BINS})
    endif()
    foreach(_SDK_BIN ${_NS_WINSDK_HINTS})
        if(EXISTS "${_SDK_BIN}/dxcompiler.dll" AND EXISTS "${_SDK_BIN}/dxil.dll")
            set(_NS_DXC_DLL "${_SDK_BIN}/dxcompiler.dll")
            set(_NS_DXIL_DLL "${_SDK_BIN}/dxil.dll")
            break()
        endif()
    endforeach()
    if(_NS_DXC_DLL AND _NS_DXIL_DLL)
        add_custom_command(TARGET NorthstarD3D12 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_NS_DXIL_DLL}"
                "$<TARGET_FILE_DIR:NorthstarD3D12>/dxil.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_NS_DXC_DLL}"
                "$<TARGET_FILE_DIR:NorthstarD3D12>/dxcompiler.dll"
            COMMENT "Deploying DXC SM 6.6 runtime libraries (dxil.dll, dxcompiler.dll)"
            VERBATIM
        )
        install(FILES "${_NS_DXIL_DLL}" "${_NS_DXC_DLL}" DESTINATION bin)
    endif()
endif()

# -------------------- 5. NorthstarD3D11 Shared Library (Stub) --------------------
add_library(NorthstarD3D11 SHARED
    platform/d3d11/runtime/northstar_d3d11.cpp
)

target_compile_definitions(NorthstarD3D11 PRIVATE NORTHSTAR_D3D11_EXPORTS)

target_include_directories(NorthstarD3D11 PUBLIC
    ${NORTHSTAR_ROOT}/include
    ${NORTHSTAR_ROOT}/../common
    ${NORTHSTAR_ROOT}/../libs
)

target_include_directories(NorthstarD3D11 PRIVATE
    ${NORTHSTAR_ROOT}
)

target_link_libraries(NorthstarD3D11 PRIVATE
    QuarkDynMath
    d3d11
    dxgi
)

target_compile_features(NorthstarD3D11 PUBLIC cxx_std_20)
set_target_properties(NorthstarD3D11 PROPERTIES FOLDER "Engine")
northstar_configure_target_outputs(NorthstarD3D11 TRUE)
northstar_apply_strict_warnings(NorthstarD3D11)

# -------------------- 6. Third-Party Dear ImGui Helper Library --------------------
add_library(NorthstarImGui STATIC
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/imgui.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/imgui_draw.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/imgui_tables.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/imgui_widgets.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/backends/imgui_impl_win32.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/backends/imgui_impl_dx12.cpp
    ${NORTHSTAR_ROOT}/../thirdparty/ImGuizmo/ImGuizmo.cpp
)

target_include_directories(NorthstarImGui PUBLIC
    ${NORTHSTAR_ROOT}/../thirdparty
    ${NORTHSTAR_ROOT}/../thirdparty/imgui
    ${NORTHSTAR_ROOT}/../thirdparty/imgui/backends
    ${NORTHSTAR_ROOT}/../thirdparty/ImGuizmo
)

target_compile_features(NorthstarImGui PUBLIC cxx_std_20)
set_target_properties(NorthstarImGui PROPERTIES FOLDER "ThirdParty")
northstar_configure_target_outputs(NorthstarImGui TRUE)
if(MSVC)
    target_compile_options(NorthstarImGui PRIVATE /MP /W3 /wd4189 /utf-8 /permissive- /Zc:__cplusplus)
endif()

# -------------------- 7. SDK Installations --------------------
install(TARGETS NorthstarCore NorthstarD3D12 NorthstarD3D11
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(DIRECTORY "${NORTHSTAR_ROOT}/include/"
    DESTINATION include
)

install(DIRECTORY "${_NORTHSTAR_RUNTIME_GEN_DIR}"
    DESTINATION bin
)

install(DIRECTORY "${NORTHSTAR_GEN_ROOT_DIR}/shaders"
    DESTINATION bin
)
