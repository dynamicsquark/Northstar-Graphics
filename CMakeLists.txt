# =====================================================================
#  Northstar Graphics — Unified Root CMakeLists.txt
# =====================================================================

cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif()

project(NorthstarGraphics VERSION 0.7.0 LANGUAGES CXX)

if(MSVC AND NOT POLICY CMP0141)
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(NORTHSTAR_BUILD_EXAMPLES "Build Northstar Graphics example applications" ON)

set(NORTHSTAR_REPO_ROOT   "${CMAKE_CURRENT_SOURCE_DIR}")
set(NORTHSTAR_OUTPUT_ROOT "${CMAKE_BINARY_DIR}")

function(northstar_configure_target_outputs target_name apply_debug_postfix)
    if(apply_debug_postfix)
        set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "d")
    endif()
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY                "${NORTHSTAR_OUTPUT_ROOT}/bin/win64"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-release"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-relwithdebinfo"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-minsizerel"
        LIBRARY_OUTPUT_DIRECTORY                "${NORTHSTAR_OUTPUT_ROOT}/bin/win64"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-debug"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-release"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-relwithdebinfo"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     "${NORTHSTAR_OUTPUT_ROOT}/bin/win64-minsizerel"
        ARCHIVE_OUTPUT_DIRECTORY                "${NORTHSTAR_OUTPUT_ROOT}/lib/win64"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG          "${NORTHSTAR_OUTPUT_ROOT}/lib/win64-debug"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE        "${NORTHSTAR_OUTPUT_ROOT}/lib/win64-release"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NORTHSTAR_OUTPUT_ROOT}/lib/win64-relwithdebinfo"
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL     "${NORTHSTAR_OUTPUT_ROOT}/lib/win64-minsizerel"
    )
endfunction()

function(northstar_apply_strict_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /MP /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

# -------------------- Core SDK, Libraries & Shader Toolchain --------------------
add_subdirectory(common/QuarkDynSDK    "${CMAKE_BINARY_DIR}/intermediate/QuarkDynSDK")
add_subdirectory(libs/NSCReader        "${CMAKE_BINARY_DIR}/intermediate/NSCReader")
add_subdirectory(libs/NSResourceLoader "${CMAKE_BINARY_DIR}/intermediate/NSResourceLoader")
add_subdirectory(libs/NSWindow         "${CMAKE_BINARY_DIR}/intermediate/NSWindow")
add_subdirectory(tools/NSLCompiler     "${CMAKE_BINARY_DIR}/intermediate/NSLCompiler")

# -------------------- Northstar Runtime & Hardware Backends --------------------
add_subdirectory(northstar             "${CMAKE_BINARY_DIR}/intermediate/northstar")

# -------------------- Example Applications --------------------
if(NORTHSTAR_BUILD_EXAMPLES)
    add_subdirectory(examples          "${CMAKE_BINARY_DIR}/intermediate/examples")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT VirtualGeometryStudio)
endif()
