cmake_minimum_required(VERSION 3.20)
project(NSCReader VERSION 0.7.0 LANGUAGES CXX)

add_library(NSCReader STATIC
    src/nsc_reader.cpp
)

target_include_directories(NSCReader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_compile_features(NSCReader PUBLIC cxx_std_20)
set_target_properties(NSCReader PROPERTIES FOLDER "Libraries")

if(COMMAND northstar_configure_target_outputs)
    northstar_configure_target_outputs(NSCReader TRUE)
    northstar_apply_strict_warnings(NSCReader)
else()
    if(MSVC)
        target_compile_options(NSCReader PRIVATE /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(NSCReader PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

install(TARGETS NSCReader
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
