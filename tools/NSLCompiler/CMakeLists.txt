cmake_minimum_required(VERSION 3.20)
project(NSLCompiler VERSION 3.2.0 LANGUAGES CXX)

add_executable(NSLCompiler
    main.cpp
    core/nsl_preprocessor.cpp
    core/nsl_file_parser.cpp
    core/nsl_library_resolver.cpp
    property/nsl_property_parser.cpp
    property/nsl_binding_resolver.cpp
    stage_io/nsl_stage_io_parser.cpp
    transpiler/lexer/nsl_lexer.cpp
    transpiler/parser/nsl_parser.cpp
    transpiler/sema/nsl_sema.cpp
    transpiler/backend/hlsl/nsl_hlsl_backend.cpp
    aligner/hlsl/nsl_hlsl_aligner.cpp
    writer/nsl_nsc_writer.cpp
)

target_include_directories(NSLCompiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_compile_features(NSLCompiler PRIVATE cxx_std_20)
set_target_properties(NSLCompiler PROPERTIES FOLDER "Tools")

if(COMMAND northstar_configure_target_outputs)
    northstar_configure_target_outputs(NSLCompiler FALSE)
    northstar_apply_strict_warnings(NSLCompiler)
else()
    if(MSVC)
        target_compile_options(NSLCompiler PRIVATE /W4 /WX /utf-8 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(NSLCompiler PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

install(TARGETS NSLCompiler
    RUNTIME DESTINATION bin
)
