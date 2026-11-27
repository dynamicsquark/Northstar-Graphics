cmake_minimum_required(VERSION 3.20)
project(QuarkDynSDK VERSION 0.7.0 LANGUAGES CXX)

add_library(QuarkDynMath STATIC
    math/src/vec2.cpp
    math/src/vec3.cpp
    math/src/vec4.cpp
    math/src/color.cpp
    math/src/mat3.cpp
    math/src/mat4.cpp
    math/src/quat.cpp
    math/src/transform.cpp
    math/src/aabb.cpp
    math/src/ray.cpp
    math/src/plane.cpp
    math/src/sphere.cpp
    math/src/frustum.cpp
)

target_include_directories(QuarkDynMath PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/math/include
    ${CMAKE_CURRENT_SOURCE_DIR}/header_only
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(QuarkDynMath PUBLIC cxx_std_20)
set_target_properties(QuarkDynMath PROPERTIES POSITION_INDEPENDENT_CODE ON FOLDER "SDK")

if(COMMAND northstar_configure_target_outputs)
    northstar_configure_target_outputs(QuarkDynMath TRUE)
    northstar_apply_strict_warnings(QuarkDynMath)
else()
    if(MSVC)
        target_compile_options(QuarkDynMath PRIVATE /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(QuarkDynMath PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

install(TARGETS QuarkDynMath
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
