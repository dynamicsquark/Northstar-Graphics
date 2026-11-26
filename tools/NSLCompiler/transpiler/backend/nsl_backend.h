#pragma once

#include "../ast/nsl_ast.h"
#include "../../property/nsl_property_parser.h"
#include "../../aligner/nsl_aligner.h"
#include <string>
#include <unordered_map>

// =====================================================================
//  IBackend — Abstract code generation interface
//
//  Implement this interface to add a new target API.
//  The Generate() call receives the fully-analyzed Program AST plus
//  the resolved ParsedProperties (with assigned slots) and returns
//  the complete target-language source code as a string.
// =====================================================================

namespace NSL
{
    class IBackend
    {
    public:
        virtual ~IBackend() = default;

        // Generate target source code from the given AST and property data.
        // Called after Sema passes — AST is guaranteed to be valid.
        virtual std::string Generate(
            const Program&          program,
            const ParsedProperties& properties,
            const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers
        ) = 0;

        // Human-readable name of the target (e.g. "HLSL", "GLSL").
        virtual const char* GetTargetName() const = 0;
    };

} // namespace NSL
