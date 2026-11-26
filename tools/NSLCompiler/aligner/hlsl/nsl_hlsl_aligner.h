#pragma once

#include "../nsl_aligner.h"

// =====================================================================
//  HLSL Aligner
//
//  CBuffer (cbuffer) packing rules — HLSL / D3D constant buffer:
//    - Fields are packed into 16-byte "register" slots.
//    - A field may NOT straddle a 16-byte boundary.
//    - Vec3 (12 bytes) followed by Float (4 bytes) fits in one register.
//    - Vec3 (12 bytes) followed by Vec2 (8 bytes) does NOT fit →
//      Vec2 moves to the next register (4 bytes of padding added after Vec3).
//    - Mat3 occupies 3 × 16-byte slots (each row is a float3 padded to float4).
//    - Mat4 occupies 4 × 16-byte slots.
//    - Total buffer size is rounded up to 16 bytes.
//
//  StructuredBuffer element packing rules — tight / natural alignment:
//    - Natural alignment: size of the field (max 4 bytes for scalars/vectors).
//    - Vec3 = 12 bytes, aligned to 4 — does NOT round to 16.
//    - No 16-byte boundary enforcement.
//    - stride = sum of all field sizes (+ any natural padding).
// =====================================================================

namespace NSL
{
    class HLSLAligner : public IAligner
    {
    public:
        // ==================== IAligner ====================
        AlignedBuffer AlignBuffer(
            const PropertyGroup& group,
            const std::vector<StructDecl>& allStructs) override;
        AlignedStruct AlignStructuredBuffer(
            const PropertyGroup& group,
            const std::vector<StructDecl>& allStructs = {}) override;
        AlignedStruct AlignStruct(
            const StructDecl&              decl,
            const std::vector<StructDecl>& allStructs
        ) override;

    private:
        // ==================== CBuffer helpers ====================

        // Returns the HLSL cbuffer alignment for a given field size and type.
        // Fields must not straddle a 16-byte boundary; Double is 8-byte aligned.
        static uint32_t CBufferAlign(uint32_t currentOffset, uint32_t fieldSize, DataTypeID id = DataTypeID::UNKNOWN);

        // Returns the size of a type in a CBuffer context.
        // Mat3 = 48 (3 × 16 with row padding), Mat4 = 64 (4 × 16), others = raw size.
        static uint32_t CBufferSize(DataTypeID id, uint32_t rawSize);

        // Resolves HLSL cbuffer size of a user-defined struct (or struct array).
        uint32_t ResolveCBufferStructSize(
            const std::string&             typeName,
            const std::string&             arraySizeExpr,
            const std::vector<StructDecl>& allStructs,
            uint32_t                       depth = 0
        ) const;

        // ==================== Struct helpers ====================

        // Returns natural C/HLSL tight-packing alignment (8 for Double / structs containing Double, else up to 4).
        uint32_t GetTightFieldAlignment(
            const std::string&             typeName,
            DataTypeID                     typeID,
            uint32_t                       rawSize,
            const std::vector<StructDecl>& allStructs,
            uint32_t                       depth = 0
        ) const;

        // Resolves field type from a StructField in the SHADER block.
        // Handles nested struct lookup for composed structs.
        uint32_t ResolveStructFieldSize(
            const std::string&             typeName,
            const std::string&             arraySizeExpr,
            const std::vector<StructDecl>& allStructs,
            uint32_t                       depth = 0
        ) const;
    };

} // namespace NSL
