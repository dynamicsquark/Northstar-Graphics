cmake_minimum_required(VERSION 3.20)
project(NSResourceLoader VERSION 0.7.0 LANGUAGES CXX)

add_library(NSResourceLoader STATIC
    src/ns_resource_loader.cpp
    src/stb_image_impl.cpp
)

target_include_directories(NSResourceLoader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../northstar/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty
)

target_compile_features(NSResourceLoader PUBLIC cxx_std_20)
set_target_properties(NSResourceLoader PROPERTIES FOLDER "Libraries")

if(COMMAND northstar_configure_target_outputs)
    northstar_configure_target_outputs(NSResourceLoader TRUE)
    northstar_apply_strict_warnings(NSResourceLoader)
else()
    if(MSVC)
        target_compile_options(NSResourceLoader PRIVATE /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(NSResourceLoader PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

install(TARGETS NSResourceLoader
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
