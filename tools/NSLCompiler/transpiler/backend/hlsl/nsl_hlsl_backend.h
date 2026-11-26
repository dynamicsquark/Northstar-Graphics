#pragma once

#include "../nsl_backend.h"
#include <sstream>
#include <unordered_map>
#include <unordered_set>

// =====================================================================
//  HLSL Backend
//  NSL → HLSL transpiler.
//
//  Output structure:
//    1. CBuffer declarations (one per Buffer PROPERTY group)
//    2. Texture declarations  (one per Texture entry)
//    3. Sampler declarations  (one per Sampler entry)
//    4. StructuredBuffer declarations
//    5. UAV declarations (RWStructuredBuffer, RWTexture2D)
//    6. Struct declarations (from SHADER block)
//    7. Global variables (from SHADER block)
//    8. Function definitions (from SHADER block)
//
//  Name mangling:
//    Buffer field:             GroupName__FieldName
//    Texture/Sampler:          GroupName__EntryName
//    StructuredBuffer element: GroupName__ElementStruct (generated)
//
//  Type mapping:
//    Vec2→float2, Vec3→float3, Vec4→float4
//    Int/Int2/Int3/Int4 → int/int2/int3/int4
//    UInt/UInt2/UInt3/UInt4 → uint/uint2/uint3/uint4
//    Mat3→float3x3, Mat4→float4x4
//    Bool→bool, Double→double, Color→float4, Quat→float4
//    Float→float
// =====================================================================

namespace NSL
{
    class HLSLBackend : public IBackend
    {
    public:
        explicit HLSLBackend(TargetAPI api) : m_api(api) {}

        std::string Generate(
            const Program&          program,
            const ParsedProperties& properties,
            const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers
        ) override;

        const char* GetTargetName() const override { return "HLSL"; }

    private:
        // ==================== Resource emitters ====================
        void EmitPropertyResources(
            std::ostringstream&     out,
            const ParsedProperties& props,
            const Program&          program,
            const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers
        );

        void EmitCBuffer(
            std::ostringstream&  out,
            const PropertyGroup& group,
            const AlignedBuffer& alignedBuf
        );

        void EmitTextureGroup(
            std::ostringstream&  out,
            const PropertyGroup& group
        );

        void EmitSamplerGroup(
            std::ostringstream&  out,
            const PropertyGroup& group
        );

        void EmitStructuredBufferGroup(
            std::ostringstream&  out,
            const PropertyGroup& group,
            const Program&       program
        );

        void EmitUAVGroup(
            std::ostringstream&  out,
            const PropertyGroup& group,
            const Program&       program
        );

        // ==================== SHADER block emitters ====================
        void EmitStructs(std::ostringstream& out, const Program& program);
        void EmitGlobals(std::ostringstream& out, const Program& program);
        void EmitFunctions(
            std::ostringstream&     out,
            const Program&          program,
            const ParsedProperties& props
        );

        // ==================== Statement / Expression emitters ====================
        void     EmitStatement(std::ostringstream& out, const Statement* stmt,
                               int indent, const ParsedProperties& props);
        void     EmitBlock(std::ostringstream& out, const std::vector<StmtPtr>& stmts,
                           int indent, const ParsedProperties& props);
        std::string EmitExpr(const Expression* expr, const ParsedProperties& props);
        std::string EmitBinaryOp(BinaryExpr::Op op);
        std::string EmitUnaryOp(UnaryExpr::Op op);

        // ==================== Type & name helpers ====================
        std::string ToHLSLType(const std::string& nslType);
        std::string MangledName(const std::string& groupName, const std::string& field);
        std::string Indent(int level);

        // ==================== State ====================
        // Alias name -> mangled HLSL name (Buffer/texture/sampler/RWTexture).
        std::unordered_map<std::string, std::string> m_aliasMap;
        // Structured field selector alias -> (qualified group, field).
        // `alias[index]` lowers to `Group__data[index].field`.
        std::unordered_map<std::string, std::pair<std::string, std::string>>
            m_structuredAliasMap;
        // Qualified PROPERTY group names.
        std::unordered_set<std::string> m_groupNames;
        bool      m_depthOnlyPixel = false;
        TargetAPI m_api = TargetAPI::HLSL11;
    };

} // namespace NSL
