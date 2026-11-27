cmake_minimum_required(VERSION 3.20)
project(NSWindow VERSION 0.7.0 LANGUAGES CXX)

set(NS_WINDOW_SOURCES)
if(WIN32)
    list(APPEND NS_WINDOW_SOURCES
        src/platform/windows/ns_win32_window.cpp
    )
endif()

add_library(NSWindow STATIC
    ${NS_WINDOW_SOURCES}
)

target_include_directories(NSWindow PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../northstar/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_compile_features(NSWindow PUBLIC cxx_std_20)
set_target_properties(NSWindow PROPERTIES FOLDER "Libraries")

if(WIN32)
    target_link_libraries(NSWindow PUBLIC dwmapi)
endif()

if(COMMAND northstar_configure_target_outputs)
    northstar_configure_target_outputs(NSWindow TRUE)
    northstar_apply_strict_warnings(NSWindow)
else()
    if(MSVC)
        target_compile_options(NSWindow PRIVATE /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(NSWindow PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

install(TARGETS NSWindow
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
