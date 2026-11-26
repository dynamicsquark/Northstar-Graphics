#pragma once

#include "../property/nsl_property_parser.h"
#include "../transpiler/ast/nsl_ast_declarations.h"
#include <string>
#include <vector>
#include <cstdint>

// =====================================================================
//  Aligner — Abstract Interface
//
//  Computes the memory layout of Buffer and StructuredBuffer groups
//  according to target-specific packing rules.
//
//  Output structures:
//    AlignedField   — single field with byte offset, size, and padding
//    AlignedBuffer  — fully laid-out Buffer with total size
//    AlignedStruct  — fully laid-out StructuredBuffer element struct
//
//  Concrete implementations live under aligner/<api>/:
//    HLSLAligner — HLSL constant-buffer packing + tight-packed structs
// =====================================================================

namespace NSL
{
    // ========================= OUTPUT TYPES ==========================

    struct AlignedField
    {
        std::string       name;
        std::string       typeName;
        DataTypeID        typeID        = DataTypeID::UNKNOWN;
        uint32_t          offset        = 0;    // Byte offset within the buffer
        uint32_t          size          = 0;    // Byte size of the field
        uint32_t          padding       = 0;    // Trailing padding bytes added after this field
        bool              isPadding     = false; // True if this is a synthetic padding field
        bool              isGet         = false; // True if engine-provided (Get)
        std::string       defaultValue;          // For Add() fields
        std::string       propertyName;          // Engine property name for Get() fields
    };

    struct AlignedBuffer
    {
        std::string              name;
        uint16_t                 set        = 0;
        uint16_t                 binding    = 0xFFFF;
        std::vector<AlignedField> fields;        // All real fields, no padding
        std::vector<AlignedField> paddedFields;  // Fields + padding fields interleaved
        uint32_t                 totalSize  = 0; // Target-padded total size
    };

    struct AlignedStructField
    {
        std::string  name;
        std::string  typeName;
        DataTypeID   typeID  = DataTypeID::UNKNOWN;
        uint32_t     offset  = 0;
        uint32_t     size    = 0;
        bool         isGet   = false;
        std::string  propertyName;
    };

    struct AlignedStruct
    {
        std::string                   name;
        uint32_t                      stride = 0;  // Total element size (tight packed)
        std::vector<AlignedStructField> fields;
    };

    // ========================= ABSTRACT INTERFACE ====================

    class IAligner
    {
    public:
        virtual ~IAligner() = default;

        // Align a Buffer PROPERTY group (constant-buffer packing rules).
        virtual AlignedBuffer AlignBuffer(
            const PropertyGroup& group,
            const std::vector<StructDecl>& allStructs) = 0;

        // Align a StructuredBuffer PROPERTY group with its element struct.
        virtual AlignedStruct AlignStructuredBuffer(
            const PropertyGroup& group,
            const std::vector<StructDecl>& allStructs = {}) = 0;

        // Align a user-defined struct declaration from the SHADER block.
        // Used for StructuredBuffer types defined in the shader.
        virtual AlignedStruct AlignStruct(
            const StructDecl&              decl,
            const std::vector<StructDecl>& allStructs
        ) = 0;
    };

} // namespace NSL
