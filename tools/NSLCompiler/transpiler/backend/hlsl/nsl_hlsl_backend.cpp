#include "nsl_hlsl_backend.h"
#include "nsl_hlsl_stage_io.h"
#include <cassert>
#include <iomanip>
#include <sstream>

namespace NSL
{

// =====================================================================
//  Type conversion table
// =====================================================================

static std::string TypeIDToHLSL(DataTypeID typeID)
{
    switch (typeID)
    {
        case DataTypeID::FLOAT:  return "float";
        case DataTypeID::DOUBLE: return "double";
        case DataTypeID::BOOL:   return "bool";
        case DataTypeID::INT:    return "int";
        case DataTypeID::UINT:   return "uint";
        case DataTypeID::VEC2:   return "float2";
        case DataTypeID::VEC3:   return "float3";
        case DataTypeID::VEC4:   return "float4";
        case DataTypeID::INT2:   return "int2";
        case DataTypeID::INT3:   return "int3";
        case DataTypeID::INT4:   return "int4";
        case DataTypeID::UINT2:  return "uint2";
        case DataTypeID::UINT3:  return "uint3";
        case DataTypeID::UINT4:  return "uint4";
        case DataTypeID::MAT3:   return "column_major float3x3";
        case DataTypeID::MAT4:   return "column_major float4x4";
        case DataTypeID::COLOR:  return "float4";
        case DataTypeID::QUAT:   return "float4";
        default:                 return "";
    }
}

std::string HLSLBackend::ToHLSLType(const std::string& nsl)
{
    static const std::unordered_map<std::string, std::string> kMap =
    {
        { "Float",  "float"    }, { "Double",  "double"    },
        { "Bool",   "bool"     }, { "Int",     "int"       }, { "UInt",   "uint"     },
        { "Vec2",   "float2"   }, { "Vec3",    "float3"    }, { "Vec4",    "float4"  },
        { "Int2",   "int2"     }, { "Int3",    "int3"      }, { "Int4",    "int4"    },
        { "UInt2",  "uint2"    }, { "UInt3",   "uint3"     }, { "UInt4",   "uint4"   },
        { "Mat3",   "float3x3" }, { "Mat4",    "float4x4"  },
        { "Color",  "float4"   }, { "Quat",    "float4"    },
        { "Texture2D",              "Texture2D"              },
        { "TextureCube",            "TextureCube"            },
        { "Texture2DArray",         "Texture2DArray"         },
        { "TextureCubeArray",       "TextureCubeArray"       },
        { "SamplerState",           "SamplerState"           },
        { "SamplerComparisonState", "SamplerComparisonState" },
        { "void",                   "void"                   },
        { "Void",                   "void"                   },
    };
    auto it = kMap.find(nsl);
    return (it != kMap.end()) ? it->second : nsl; // pass through if unknown
}

std::string HLSLBackend::MangledName(const std::string& qualifiedGroup, const std::string& field)
{
    return qualifiedGroup + "__" + field;
}

std::string HLSLBackend::Indent(int level)
{
    return std::string(static_cast<size_t>(level * 4), ' ');
}

// =====================================================================
//  Generate — main entry point
// =====================================================================

std::string HLSLBackend::Generate(
    const Program&          program,
    const ParsedProperties& properties,
    const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers)
{
    // ==================== Build internal maps ====================
    m_groupNames.clear();
    m_aliasMap.clear();
    m_structuredAliasMap.clear();
    m_depthOnlyPixel = false;
    if (program.stage == ShaderStage::PIXEL)
    {
        for (const auto& s : program.structs)
        {
            if (s.name == "STAGE_OUTPUT" && s.fields.empty())
            {
                m_depthOnlyPixel = true;
                break;
            }
        }
    }

    for (const auto& g : properties.groups)
    {
        m_groupNames.insert(g.qualifiedName); // Store qualified names
    }

    // Build alias maps from semantically validated qualified sources.
    for (const auto& alias : program.aliases)
    {
        auto itq = properties.byQualifiedName.find(alias.sourceBinding);
        if (itq == properties.byQualifiedName.end()) continue;

        const PropertyGroup* group = itq->second;
        if (group->resourceKind == ResourceKind::STRUCTURED_BUFFER ||
            group->resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
        {
            m_structuredAliasMap[alias.aliasName] =
                { group->qualifiedName, alias.sourceField };
        }
        else
        {
            m_aliasMap[alias.aliasName] =
                MangledName(group->qualifiedName, alias.sourceField);
        }
    }

    // ==================== Emit HLSL sections ====================
    std::ostringstream out;

    out << "// ============================================================\n";
    out << "// Auto-generated HLSL — NSL Compiler v3.2.0\n";
    out << "// Target: " << (m_api == TargetAPI::HLSL12 ? "HLSL (DirectX 12 / SM 5.1+)" : "HLSL (DirectX 11 / SM 5.0)") << "\n";
    out << "// Coordinate System: Left-Handed (X+ Right, Y+ Up, Z+ Forward)\n";
    out << "// Matrix Layout: Column-Major (v' = M * v)\n";
    out << "// DO NOT EDIT MANUALLY\n";
    out << "// ============================================================\n\n";

    // Pass object-like preprocessor directives through so macros (e.g. MAX_LIGHTS)
    // are visible to the HLSL compiler. Order is preserved from the NSL source.
    if (!program.preprocessors.empty())
    {
        for (const auto& pp : program.preprocessors)
            out << pp.line << "\n";
        out << "\n";
    }

    // Emit struct declarations and typedefs BEFORE property resources so Buffer /
    // StructuredBuffer / RWStructuredBuffer groups can reference user-defined structs.
    EmitStructs(out, program);

    if (!program.typedefs.empty())
    {
        for (const auto& td : program.typedefs)
            out << "typedef " << ToHLSLType(td.originalType) << " " << td.aliasName << ";\n";
        out << "\n";
    }

    EmitPropertyResources(out, properties, program, alignedBuffers);

    EmitGlobals(out, program);
    EmitFunctions(out, program, properties);

    return out.str();
}

// =====================================================================
//  EmitPropertyResources
// =====================================================================

void HLSLBackend::EmitPropertyResources(
    std::ostringstream&     out,
    const ParsedProperties& props,
    const Program&          program,
    const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers)
{
    for (const auto& group : props.groups)
    {
        switch (group.resourceKind)
        {
            case ResourceKind::BUFFER:
            {
                auto it = alignedBuffers.find(group.qualifiedName);
                if (it == alignedBuffers.end()) it = alignedBuffers.find(group.name); // fallback
                if (it != alignedBuffers.end())
                    EmitCBuffer(out, group, it->second);
                break;
            }
            case ResourceKind::TEXTURE_2D:
            case ResourceKind::TEXTURE_CUBE:
            case ResourceKind::TEXTURE_2D_ARRAY:
            case ResourceKind::TEXTURE_CUBE_ARRAY:
                EmitTextureGroup(out, group);
                break;
            case ResourceKind::SAMPLER_STATE:
                EmitSamplerGroup(out, group);
                break;
            case ResourceKind::STRUCTURED_BUFFER:
                EmitStructuredBufferGroup(out, group, program);
                break;
            case ResourceKind::RW_STRUCTURED_BUFFER:
            case ResourceKind::RW_TEXTURE_2D:
                EmitUAVGroup(out, group, program);
                break;
            default:
                break;
        }
        out << "\n";
    }
}

// =====================================================================
//  EmitCBuffer
// =====================================================================

void HLSLBackend::EmitCBuffer(
    std::ostringstream&  out,
    const PropertyGroup& group,
    const AlignedBuffer& alignedBuf)
{
    out << "cbuffer " << group.qualifiedName
        << " : register(b" << group.assignedBinding
        << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(group.assignedSet) : "") << ")\n{\n";

    for (const auto& field : alignedBuf.paddedFields)
    {
        std::string hlslType = TypeIDToHLSL(field.typeID);
        if (hlslType.empty() || (hlslType == "float" && field.typeID != DataTypeID::FLOAT))
        {
            if (field.isPadding) hlslType = "float";
            else hlslType = ToHLSLType(field.typeName);
        }

        std::string fieldName;
        if (field.isPadding)
        {
            // Never emit `float pad[N]` in HLSL cbuffers: HLSL aligns every array element
            // to a 16-byte register boundary, inflating `float pad[2]` from 8B to 20B!
            // Instead emit exact scalar/vector padding chunks aligned to 16-byte slots.
            uint32_t remainingFloats = field.size / 4u;
            uint32_t curOff          = field.offset;
            uint32_t chunkIdx        = 0u;
            while (remainingFloats > 0u)
            {
                const uint32_t slotsLeftInReg = (16u - (curOff % 16u)) / 4u;
                const uint32_t take           = (remainingFloats < slotsLeftInReg) ? remainingFloats : slotsLeftInReg;
                const char*    padType        = (take == 1u) ? "float"
                                              : (take == 2u) ? "float2"
                                              : (take == 3u) ? "float3" : "float4";
                out << "    " << padType << " " << field.name;
                if (chunkIdx > 0u) out << "_" << chunkIdx;
                out << ";\n";
                remainingFloats -= take;
                curOff          += take * 4u;
                ++chunkIdx;
            }
            continue;
        }
        else
        {
            fieldName = MangledName(group.qualifiedName, field.name);
            std::string arraySizeExpr;
            if (!field.isGet)
            {
                for (const auto& entry : group.entries)
                {
                    if (entry.name == field.name)
                    {
                        arraySizeExpr = entry.arraySizeExpr;
                        break;
                    }
                }
            }
            if (!arraySizeExpr.empty())
                fieldName += "[" + arraySizeExpr + "]";
        }

        out << "    " << hlslType << " " << fieldName << ";\n";

        // HLSL cbuffer packs float3x3 as 44 bytes (3 rows x 16B minus the last 4B tail),
        // leaving offsets [44..47] open for a subsequent float, whereas NSL HLSLAligner
        // and CPU/GPU runtime writers treat Mat3 as 48 bytes (3 x float4). Emit an explicit
        // 4-byte scalar pad immediately after Mat3 so FXC/DXC layout matches reflection 100%.
        if (field.typeID == DataTypeID::MAT3)
        {
            out << "    float " << MangledName(group.qualifiedName, field.name) << "__mat3pad;\n";
        }
    }

    out << "};\n";
}

// =====================================================================
//  EmitTextureGroup
// =====================================================================

void HLSLBackend::EmitTextureGroup(
    std::ostringstream&  out,
    const PropertyGroup& group)
{
    std::string hlslResType = "Texture2D";
    if (group.resourceKind == ResourceKind::TEXTURE_CUBE)        hlslResType = "TextureCube";
    else if (group.resourceKind == ResourceKind::TEXTURE_2D_ARRAY) hlslResType = "Texture2DArray";
    else if (group.resourceKind == ResourceKind::TEXTURE_CUBE_ARRAY) hlslResType = "TextureCubeArray";

    for (const auto& entry : group.entries)
    {
        std::string mangledName = MangledName(group.qualifiedName, entry.name);

        std::string resType = hlslResType;
        if (entry.kind == PropertyEntryKind::GET)
        {
            const EngineProperty* ep = FindEngineProperty(entry.name);
            if (ep)
            {
                if (ep->typeID == DataTypeID::TEXTURE_CUBE)       resType = "TextureCube";
                else if (ep->typeID == DataTypeID::TEXTURE_2D_ARRAY) resType = "Texture2DArray";
                else if (ep->typeID == DataTypeID::TEXTURE_CUBE_ARRAY) resType = "TextureCubeArray";
            }
        }

        std::string nameSuffix = "";
        if (!entry.arraySizeExpr.empty())
            nameSuffix = "[" + entry.arraySizeExpr + "]";

        out << resType << " " << mangledName << nameSuffix
            << " : register(t" << entry.assignedBinding
            << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(entry.assignedSet) : "") << ");\n";
    }
}

// =====================================================================
//  EmitSamplerGroup
// =====================================================================

void HLSLBackend::EmitSamplerGroup(
    std::ostringstream&  out,
    const PropertyGroup& group)
{
    for (const auto& entry : group.entries)
    {
        std::string mangledName = MangledName(group.qualifiedName, entry.name);

        std::string samplerType = "SamplerState";
        if (entry.kind == PropertyEntryKind::GET)
        {
            const EngineProperty* ep = FindEngineProperty(entry.name);
            if (ep && ep->typeID == DataTypeID::SAMPLER_COMPARISON_STATE)
                samplerType = "SamplerComparisonState";
        }
        else
        {
            if (entry.typeID == DataTypeID::SAMPLER_COMPARISON_STATE ||
                entry.typeName == "SamplerComparisonState" ||
                group.resourceTypeName == "SamplerComparisonState")
            {
                samplerType = "SamplerComparisonState";
            }
        }

        std::string nameSuffix = "";
        if (!entry.arraySizeExpr.empty())
            nameSuffix = "[" + entry.arraySizeExpr + "]";

        out << samplerType << " " << mangledName << nameSuffix
            << " : register(s" << entry.assignedBinding
            << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(entry.assignedSet) : "") << ");\n";
    }
}

// =====================================================================
//  EmitStructuredBufferGroup
// =====================================================================

void HLSLBackend::EmitStructuredBufferGroup(
    std::ostringstream&  out,
    const PropertyGroup& group,
    const Program&       /*program*/)
{
    std::string structName = group.qualifiedName + "__Struct";

    out << "struct " << structName << "\n{\n";
    for (const auto& entry : group.entries)
    {
        if (entry.kind == PropertyEntryKind::GET)
        {
            if (const EngineProperty* ep = FindEngineProperty(entry.name))
            {
                out << "    " << TypeIDToHLSL(ep->typeID) << " " << entry.name << ";\n";
                continue;
            }
        }

        std::string hlslType = TypeIDToHLSL(entry.typeID);
        if (hlslType.empty() || (hlslType == "float" && entry.typeID != DataTypeID::FLOAT))
        {
            hlslType = ToHLSLType(entry.typeName);
            if (hlslType.empty()) hlslType = "float";
        }

        std::string fieldName = entry.name;
        if (!entry.arraySizeExpr.empty())
            fieldName += "[" + entry.arraySizeExpr + "]";

        out << "    " << hlslType << " " << fieldName << ";\n";
    }
    out << "};\n\n";

    out << "StructuredBuffer<" << structName << "> " << group.qualifiedName
        << "__data : register(t" << group.assignedBinding
        << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(group.assignedSet) : "") << ");\n";
}

// =====================================================================
//  EmitUAVGroup
// =====================================================================

void HLSLBackend::EmitUAVGroup(
    std::ostringstream&  out,
    const PropertyGroup& group,
    const Program&       /*program*/)
{
    if (group.resourceKind == ResourceKind::RW_TEXTURE_2D)
    {
        for (const auto& entry : group.entries)
        {
            const std::string elemType = entry.typeName.empty() ? "float4" : ToHLSLType(entry.typeName);
            const std::string nameSuffix = entry.arraySizeExpr.empty() ? "" : ("[" + entry.arraySizeExpr + "]");
            out << "RWTexture2D<" << elemType << "> " << MangledName(group.qualifiedName, entry.name) << nameSuffix
                << " : register(u" << entry.assignedBinding
                << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(entry.assignedSet) : "") << ");\n";
        }
        return;
    }

    if (!group.entries.empty())
    {
        std::string structName = group.qualifiedName + "__Struct";
        out << "struct " << structName << "\n{\n";
        for (const auto& entry : group.entries)
        {
            if (entry.kind == PropertyEntryKind::GET)
            {
                if (const EngineProperty* ep = FindEngineProperty(entry.name))
                {
                    out << "    " << TypeIDToHLSL(ep->typeID) << " " << entry.name << ";\n";
                    continue;
                }
            }

            std::string hlslType = TypeIDToHLSL(entry.typeID);
            if (hlslType.empty() || (hlslType == "float" && entry.typeID != DataTypeID::FLOAT))
            {
                hlslType = ToHLSLType(entry.typeName);
                if (hlslType.empty()) hlslType = entry.typeName.empty() ? "float" : entry.typeName;
            }

            std::string fieldName = entry.name;
            if (!entry.arraySizeExpr.empty())
                fieldName += "[" + entry.arraySizeExpr + "]";

            out << "    " << hlslType << " " << fieldName << ";\n";
        }
        out << "};\n\n";

        out << "RWStructuredBuffer<" << structName << "> " << group.qualifiedName
            << "__data : register(u" << group.assignedBinding
            << (m_api == TargetAPI::HLSL12 ? ", space" + std::to_string(group.assignedSet) : "") << ");\n";
    }
}

// =====================================================================
//  EmitStructs / Globals / Functions (mostly unmodified)
// =====================================================================

void HLSLBackend::EmitStructs(std::ostringstream& out, const Program& program)
{
    for (const auto& s : program.structs)
    {
        out << "struct " << s.name << "\n{\n";
        if (m_depthOnlyPixel && s.name == "STAGE_OUTPUT" && s.fields.empty())
        {
            out << "    uint unusedDepthOnlyField;\n";
        }
        for (const auto& f : s.fields)
        {
            out << "    ";
            if (!f.interpolation.empty()) out << f.interpolation << " ";
            out << ToHLSLType(f.typeName) << " " << f.name;
            if (!f.arraySizeExpr.empty()) out << "[" << f.arraySizeExpr << "]";
            if (!f.semantic.empty())      out << " : " << MapNSLSemanticToHLSL(f.semantic);
            out << ";\n";
        }
        out << "};\n\n";
    }
}

void HLSLBackend::EmitGlobals(std::ostringstream& out, const Program& program)
{
    for (const auto& gv : program.globals)
    {
        if (gv.isGroupShared) out << "groupshared ";
        if (gv.isStatic)      out << "static ";
        if (gv.isConst)       out << "const ";

        out << ToHLSLType(gv.typeName) << " " << gv.name;
        if (!gv.arraySizeExpr.empty()) out << "[" << gv.arraySizeExpr << "]";

        if (gv.initializer)
        {
            out << " = " << EmitExpr(gv.initializer.get(), {});
        }
        out << ";\n";
    }
    if (!program.globals.empty()) out << "\n";
}

// Maps an NSL geometry [inputtopology("...")] value to the HLSL input-primitive
// keyword that must prefix the geometry shader's vertex-array parameter.
static std::string GeometryInputPrimitiveKeyword(const std::string& topology)
{
    if (topology == "point")       return "point";
    if (topology == "line")        return "line";
    if (topology == "triangle")    return "triangle";
    if (topology == "lineadj")     return "lineadj";
    if (topology == "triangleadj") return "triangleadj";
    return "";
}

static std::string ExtractAttributeValue(const Attribute& attr)
{
    size_t open  = attr.args.find('(');
    size_t close = attr.args.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1)
        return {};
    std::string v = attr.args.substr(open + 1, close - open - 1);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
        v = v.substr(1, v.size() - 2);
    return v;
}

void HLSLBackend::EmitFunctions(
    std::ostringstream&     out,
    const Program&          program,
    const ParsedProperties& props)
{
    for (const auto& fn : program.functions)
    {
        const bool isGeometry =
            program.stage == ShaderStage::GEOMETRY && fn.name == "Main";
        const bool isHullMain = program.stage == ShaderStage::HULL && fn.name == "Main";

        // Derive the geometry input-primitive keyword from [inputtopology(...)].
        std::string geoInputPrim;
        if (isGeometry)
        {
            for (const auto& attr : fn.attributes)
                if (attr.name == "inputtopology")
                    geoInputPrim = GeometryInputPrimitiveKeyword(ExtractAttributeValue(attr));
        }

        const bool isDepthOnlyPixelEntry =
            m_depthOnlyPixel && fn.name == "Main" && fn.returnType == "STAGE_OUTPUT";

        if (isDepthOnlyPixelEntry)
        {
            out << "static STAGE_OUTPUT NSLDepthOnlyPixelBody(";
            for (size_t i = 0; i < fn.parameters.size(); ++i)
            {
                const auto& p = fn.parameters[i];
                if (i > 0) out << ", ";
                if (p.qualifier != ParamQualifier::NONE)
                    out << ParamQualifierToString(p.qualifier) << " ";
                out << ToHLSLType(p.typeName) << " " << p.name;
                if (!p.arraySizeExpr.empty()) out << "[" << p.arraySizeExpr << "]";
            }
            out << ")\n{\n";
            EmitBlock(out, fn.body, 1, props);
            out << "}\n\n";

            for (const auto& attr : fn.attributes)
            {
                out << "[" << attr.args << "]\n";
            }
            out << "void Main(";
            for (size_t i = 0; i < fn.parameters.size(); ++i)
            {
                const auto& p = fn.parameters[i];
                if (i > 0) out << ", ";
                if (p.qualifier != ParamQualifier::NONE)
                    out << ParamQualifierToString(p.qualifier) << " ";
                out << ToHLSLType(p.typeName) << " " << p.name;
                if (!p.arraySizeExpr.empty()) out << "[" << p.arraySizeExpr << "]";
                if (!p.semantic.empty()) out << " : " << MapNSLSemanticToHLSL(p.semantic);
            }
            out << ")\n{\n    NSLDepthOnlyPixelBody(";
            for (size_t i = 0; i < fn.parameters.size(); ++i)
            {
                if (i > 0) out << ", ";
                out << fn.parameters[i].name;
            }
            out << ");\n}\n\n";
            continue;
        }

        // Emit attributes. Geometry's inputtopology/outputtopology are NSL-only
        // (HLSL derives them from the param prefix and stream type), so they are
        // dropped; everything else (maxvertexcount, hull/domain attrs) passes
        // through unchanged.
        bool hasPatchConstantFunc = false;
        for (const auto& attr : fn.attributes)
        {
            if (isGeometry && (attr.name == "inputtopology" || attr.name == "outputtopology"))
                continue;
            if (attr.name == "patchconstantfunc") hasPatchConstantFunc = true;
            out << "[" << attr.args << "]\n";
        }

        // The hull control-point entry must name its patch-constant function.
        if (isHullMain && !hasPatchConstantFunc)
            out << "[patchconstantfunc(\"PatchConstant\")]\n";

        out << ToHLSLType(fn.returnType) << " " << fn.name << "(";

        for (size_t i = 0; i < fn.parameters.size(); ++i)
        {
            const auto& p = fn.parameters[i];
            if (i > 0) out << ", ";
            if (p.qualifier != ParamQualifier::NONE)
                out << ParamQualifierToString(p.qualifier) << " ";
            // Geometry vertex-array param needs its input-primitive keyword.
            if (isGeometry && !geoInputPrim.empty() &&
                !p.arraySizeExpr.empty() && p.typeName == "STAGE_INPUT")
                out << geoInputPrim << " ";
            out << ToHLSLType(p.typeName) << " " << p.name;
            if (!p.arraySizeExpr.empty()) out << "[" << p.arraySizeExpr << "]";
            if (!p.semantic.empty()) out << " : " << MapNSLSemanticToHLSL(p.semantic);
        }

        out << ")";
        if (!fn.returnSemantic.empty()) out << " : " << MapNSLSemanticToHLSL(fn.returnSemantic);
        out << "\n{\n";

        EmitBlock(out, fn.body, 1, props);

        out << "}\n\n";
    }
}

// =====================================================================
//  Statements
// =====================================================================

void HLSLBackend::EmitBlock(
    std::ostringstream&         out,
    const std::vector<StmtPtr>& stmts,
    int                         indent,
    const ParsedProperties&     props)
{
    for (const auto& stmt : stmts)
        EmitStatement(out, stmt.get(), indent, props);
}

void HLSLBackend::EmitStatement(
    std::ostringstream&     out,
    const Statement*        stmt,
    int                     indent,
    const ParsedProperties& props)
{
    if (!stmt) return;
    std::string ind = Indent(indent);

    if (auto* blk = dynamic_cast<const BlockStmt*>(stmt))
    {
        out << ind << "{\n";
        EmitBlock(out, blk->statements, indent + 1, props);
        out << ind << "}\n";
    }
    else if (auto* es = dynamic_cast<const ExprStmt*>(stmt))
    {
        out << ind << EmitExpr(es->expr.get(), props) << ";\n";
    }
    else if (auto* rs = dynamic_cast<const ReturnStmt*>(stmt))
    {
        out << ind << "return";
        if (rs->value) out << " " << EmitExpr(rs->value.get(), props);
        out << ";\n";
    }
    else if (auto* vd = dynamic_cast<const VarDeclStmt*>(stmt))
    {
        out << ind;
        if (vd->isGroupShared) out << "groupshared ";
        if (vd->isStatic)      out << "static ";
        if (vd->isConst)       out << "const ";
        out << ToHLSLType(vd->typeName) << " " << vd->name;
        if (!vd->arraySizeExpr.empty()) out << "[" << vd->arraySizeExpr << "]";
        if (vd->initializer)
            out << " = " << EmitExpr(vd->initializer.get(), props);
        else if (m_depthOnlyPixel && vd->typeName == "STAGE_OUTPUT" && vd->arraySizeExpr.empty())
            out << " = (STAGE_OUTPUT)0";
        out << ";\n";
    }
    else if (auto* ifs = dynamic_cast<const IfStmt*>(stmt))
    {
        out << ind << "if (" << EmitExpr(ifs->condition.get(), props) << ")\n";
        EmitStatement(out, ifs->thenBranch.get(), indent, props);
        if (ifs->elseBranch)
        {
            out << ind << "else\n";
            EmitStatement(out, ifs->elseBranch.get(), indent, props);
        }
    }
    else if (auto* fs = dynamic_cast<const ForStmt*>(stmt))
    {
        out << ind << "for (";
        if (fs->init)
        {
            std::ostringstream initOut;
            EmitStatement(initOut, fs->init.get(), 0, props);
            std::string initStr = initOut.str();
            while (!initStr.empty() && (initStr.back() == '\n' || initStr.back() == ' ' || initStr.back() == ';'))
                initStr.pop_back();
            out << initStr;
        }
        out << "; ";
        if (fs->condition) out << EmitExpr(fs->condition.get(), props);
        out << "; ";
        if (fs->increment) out << EmitExpr(fs->increment.get(), props);
        out << ")\n";
        EmitStatement(out, fs->body.get(), indent, props);
    }
    else if (auto* ws = dynamic_cast<const WhileStmt*>(stmt))
    {
        out << ind << "while (" << EmitExpr(ws->condition.get(), props) << ")\n";
        EmitStatement(out, ws->body.get(), indent, props);
    }
    else if (auto* dw = dynamic_cast<const DoWhileStmt*>(stmt))
    {
        out << ind << "do\n";
        EmitStatement(out, dw->body.get(), indent, props);
        out << ind << "while (" << EmitExpr(dw->condition.get(), props) << ");\n";
    }
    else if (auto* sw = dynamic_cast<const SwitchStmt*>(stmt))
    {
        out << ind << "switch (" << EmitExpr(sw->condition.get(), props) << ")\n";
        out << ind << "{\n";
        for (const auto& sc : sw->cases)
        {
            if (sc.value)
                out << ind << "case " << EmitExpr(sc.value.get(), props) << ":\n";
            else
                out << ind << "default:\n";
            for (const auto& s : sc.statements)
                EmitStatement(out, s.get(), indent + 1, props);
        }
        out << ind << "}\n";
    }
    else if (dynamic_cast<const BreakStmt*>(stmt))    out << ind << "break;\n";
    else if (dynamic_cast<const ContinueStmt*>(stmt)) out << ind << "continue;\n";
    else if (dynamic_cast<const DiscardStmt*>(stmt))  out << ind << "discard;\n";
}

// =====================================================================
//  EmitExpr
// =====================================================================

std::string HLSLBackend::EmitExpr(const Expression* expr, const ParsedProperties& props)
{
    if (!expr) return "";

    if (auto* lit = dynamic_cast<const LiteralExpr*>(expr))
    {
        switch (lit->litKind)
        {
            case LiteralExpr::Kind::Int:
                return std::to_string(lit->intValue);
            case LiteralExpr::Kind::UInt:
                return std::to_string(lit->uintValue) + "u";
            case LiteralExpr::Kind::Float:
            {
                std::ostringstream ss;
                ss << std::setprecision(9) << lit->floatValue;
                std::string s = ss.str();
                if (s.find('.') == std::string::npos &&
                    s.find('e') == std::string::npos &&
                    s.find('E') == std::string::npos)
                {
                    s += ".0";
                }
                s += "f";
                return s;
            }
            case LiteralExpr::Kind::Bool:
                return lit->boolValue ? "true" : "false";
            case LiteralExpr::Kind::String:
                return "\"" + lit->strValue + "\"";
        }
    }

    if (auto* id = dynamic_cast<const IdentifierExpr*>(expr))
    {
        auto aliasIt = m_aliasMap.find(id->name);
        if (aliasIt != m_aliasMap.end()) return aliasIt->second;
        // A structured field alias is a selector and must be indexed.
        if (m_structuredAliasMap.count(id->name))
            return "/* structured alias requires [index] */" + id->name;
        return id->name;
    }

    if (auto* ma = dynamic_cast<const MemberAccessExpr*>(expr))
    {
        // Check for 3-level member access: (Property.Group).Field
        if (auto* innerMa = dynamic_cast<const MemberAccessExpr*>(ma->object.get()))
        {
            if (auto* objId = dynamic_cast<const IdentifierExpr*>(innerMa->object.get()))
            {
                std::string qualifiedName = objId->name + "__" + innerMa->member;
                if (m_groupNames.count(qualifiedName))
                    return MangledName(qualifiedName, ma->member);
            }
        }

        // Check for 2-level member access to a StructuredBuffer / RWStructuredBuffer:
        // Property.Group -> Property__Group__data
        if (auto* objId = dynamic_cast<const IdentifierExpr*>(ma->object.get()))
        {
            std::string qualifiedName = objId->name + "__" + ma->member;
            if (m_groupNames.count(qualifiedName))
            {
                auto it = props.byQualifiedName.find(qualifiedName);
                if (it != props.byQualifiedName.end() &&
                    (it->second->resourceKind == ResourceKind::STRUCTURED_BUFFER ||
                     it->second->resourceKind == ResourceKind::RW_STRUCTURED_BUFFER))
                {
                    return qualifiedName + "__data";
                }
            }
        }

        std::string op = ma->isArrow ? "->" : ".";
        return EmitExpr(ma->object.get(), props) + op + ma->member;
    }

    if (auto* ie = dynamic_cast<const IndexExpr*>(expr))
    {
        // Structured field selector:
        // Property.Group.Field[index] -> Property__Group__data[index].Field
        if (auto* fieldMa = dynamic_cast<const MemberAccessExpr*>(ie->object.get()))
        {
            if (auto* groupMa =
                    dynamic_cast<const MemberAccessExpr*>(fieldMa->object.get()))
            {
                if (auto* blockId =
                        dynamic_cast<const IdentifierExpr*>(groupMa->object.get()))
                {
                    const std::string qualifiedName =
                        blockId->name + "__" + groupMa->member;
                    auto groupIt = props.byQualifiedName.find(qualifiedName);
                    if (groupIt != props.byQualifiedName.end() &&
                        (groupIt->second->resourceKind ==
                             ResourceKind::STRUCTURED_BUFFER ||
                         groupIt->second->resourceKind ==
                             ResourceKind::RW_STRUCTURED_BUFFER))
                    {
                        return qualifiedName + "__data[" +
                               EmitExpr(ie->index.get(), props) + "]." +
                               fieldMa->member;
                    }
                }
            }
        }

        // Structured field alias: alias[index] -> Group__data[index].Field
        if (auto* aliasId = dynamic_cast<const IdentifierExpr*>(ie->object.get()))
        {
            auto aliasIt = m_structuredAliasMap.find(aliasId->name);
            if (aliasIt != m_structuredAliasMap.end())
            {
                return aliasIt->second.first + "__data[" +
                       EmitExpr(ie->index.get(), props) + "]." +
                       aliasIt->second.second;
            }
        }

        // 3-level index access: Property.Group[i] -> Property__Group__data[i]
        if (auto* ma = dynamic_cast<const MemberAccessExpr*>(ie->object.get()))
        {
            if (auto* objId = dynamic_cast<const IdentifierExpr*>(ma->object.get()))
            {
                std::string qualifiedName = objId->name + "__" + ma->member;
                if (m_groupNames.count(qualifiedName))
                {
                    return qualifiedName + "__data[" + EmitExpr(ie->index.get(), props) + "]";
                }
            }
        }

        return EmitExpr(ie->object.get(), props) + "[" + EmitExpr(ie->index.get(), props) + "]";
    }

    if (auto* ce = dynamic_cast<const CallExpr*>(expr))
    {
        std::string callee = EmitExpr(ce->callee.get(), props);

        static const std::unordered_map<std::string, std::string> kFuncMap =
        {
            { "Dot",        "dot"        }, { "Cross",      "cross"       },
            { "Normalize",  "normalize"  }, { "Pow",        "pow"         },
            { "Sqrt",       "sqrt"       }, { "Rsqrt",      "rsqrt"       },
            { "Abs",        "abs"        }, { "Sign",       "sign"        },
            { "Floor",      "floor"      }, { "Ceil",       "ceil"        },
            { "Round",      "round"      }, { "Saturate",   "saturate"    },
            { "Clamp",      "clamp"      }, { "Lerp",       "lerp"        },
            { "SmoothStep", "smoothstep" }, { "Max",        "max"         },
            { "Min",        "min"        }, { "Length",     "length"      },
            { "Distance",   "distance"   }, { "Mul",        "mul"         },
            { "Reflect",    "reflect"    }, { "Refract",    "refract"     },
            { "Transpose",  "transpose"  }, { "Determinant", "determinant" },
            { "Ddx",        "ddx"        }, { "Ddy",        "ddy"         },
            { "Frac",       "frac"       }, { "Trunc",      "trunc"       },
            { "Modf",       "modf"       }, { "Exp",        "exp"         },
            { "Exp2",       "exp2"       }, { "Log",        "log"         },
            { "Log2",       "log2"       }, { "Sin",        "sin"         },
            { "Cos",        "cos"        }, { "Tan",        "tan"         },
            { "Asin",       "asin"       }, { "Acos",       "acos"        },
            { "Atan",       "atan"       }, { "Atan2",      "atan2"       },
            { "All",        "all"        }, { "Any",        "any"         },
            { "CountBits",  "countbits"  }, { "FirstBitHigh", "firstbithigh" },
            { "FirstBitLow", "firstbitlow" },
            { "AsFloat",    "asfloat"    }, { "AsUInt",     "asuint"      },
            { "AsInt",      "asint"      }, { "Clip",       "clip"        },
            { "F16ToF32",   "f16tof32"   }, { "F32ToF16",   "f32tof16"    },
            { "IsNaN",      "isnan"      }, { "IsInf",      "isinf"       },
            { "IsFinite",   "isfinite"   },
        };

        auto funcIt = kFuncMap.find(callee);
        if (funcIt != kFuncMap.end()) callee = funcIt->second;

        // Geometry stream intrinsics: Append(stream, v) -> stream.Append(v),
        // RestartStrip(stream) -> stream.RestartStrip().
        if (callee == "Append" && ce->args.size() == 2)
        {
            std::string stream = EmitExpr(ce->args[0].get(), props);
            std::string vertex = EmitExpr(ce->args[1].get(), props);
            return stream + ".Append(" + vertex + ")";
        }
        if (callee == "RestartStrip" && ce->args.size() == 1)
        {
            std::string stream = EmitExpr(ce->args[0].get(), props);
            return stream + ".RestartStrip()";
        }

        if ((callee == "Sample" && ce->args.size() >= 3) ||
            (callee == "SampleLevel" && ce->args.size() >= 4) ||
            (callee == "SampleCmp" && ce->args.size() >= 4) ||
            (callee == "SampleCmpLevelZero" && ce->args.size() >= 4))
        {
            std::string tex = EmitExpr(ce->args[0].get(), props);
            std::string callArgs;
            for (size_t i = 1; i < ce->args.size(); ++i)
            {
                if (i > 1) callArgs += ", ";
                callArgs += EmitExpr(ce->args[i].get(), props);
            }
            return tex + "." + callee + "(" + callArgs + ")";
        }

        // Unfiltered texel/buffer load: Load(tex, location [, offset]) -> tex.Load(...)
        if (callee == "Load" && ce->args.size() == 2)
        {
            std::string tex = EmitExpr(ce->args[0].get(), props);
            std::string loc = EmitExpr(ce->args[1].get(), props);
            return tex + ".Load(" + loc + ")";
        }
        if (callee == "Load" && ce->args.size() == 3)
        {
            std::string tex = EmitExpr(ce->args[0].get(), props);
            std::string loc = EmitExpr(ce->args[1].get(), props);
            std::string off = EmitExpr(ce->args[2].get(), props);
            return tex + ".Load(" + loc + ", " + off + ")";
        }

        // Buffer dimension query: GetDimensions(buffer, count, stride) -> buffer.GetDimensions(count, stride)
        if (callee == "GetDimensions" && ce->args.size() == 3)
        {
            std::string buf   = EmitExpr(ce->args[0].get(), props);
            std::string count = EmitExpr(ce->args[1].get(), props);
            std::string strd  = EmitExpr(ce->args[2].get(), props);
            return buf + ".GetDimensions(" + count + ", " + strd + ")";
        }
        if (callee == "GetDimensions" && ce->args.size() == 2)
        {
            std::string buf   = EmitExpr(ce->args[0].get(), props);
            std::string count = EmitExpr(ce->args[1].get(), props);
            return buf + ".GetDimensions(" + count + ")";
        }

        std::string result = callee + "(";
        for (size_t i = 0; i < ce->args.size(); ++i)
        {
            if (i > 0) result += ", ";
            result += EmitExpr(ce->args[i].get(), props);
        }
        result += ")";
        return result;
    }

    if (auto* cast = dynamic_cast<const CastExpr*>(expr))
    {
        std::string hlslType = ToHLSLType(cast->typeName);
        std::string result = hlslType + "(";
        for (size_t i = 0; i < cast->args.size(); ++i)
        {
            if (i > 0) result += ", ";
            result += EmitExpr(cast->args[i].get(), props);
        }
        result += ")";
        return result;
    }

    if (auto* ue = dynamic_cast<const UnaryExpr*>(expr))
    {
        std::string op  = EmitUnaryOp(ue->op);
        std::string operand = EmitExpr(ue->operand.get(), props);
        return ue->isPostfix ? operand + op : op + operand;
    }

    if (auto* be = dynamic_cast<const BinaryExpr*>(expr))
    {
        std::string left  = EmitExpr(be->left.get(), props);
        std::string right = EmitExpr(be->right.get(), props);
        std::string op    = EmitBinaryOp(be->op);
        return left + " " + op + " " + right;
    }

    if (auto* pe = dynamic_cast<const ParenExpr*>(expr))
    {
        return "(" + EmitExpr(pe->inner.get(), props) + ")";
    }

    if (auto* te = dynamic_cast<const TernaryExpr*>(expr))
    {
        std::string cond = EmitExpr(te->condition.get(), props);
        std::string t    = EmitExpr(te->thenExpr.get(), props);
        std::string e    = EmitExpr(te->elseExpr.get(), props);
        return cond + " ? " + t + " : " + e;
    }

    if (auto* le = dynamic_cast<const LengthExpr*>(expr))
    {
        std::string obj = EmitExpr(le->object.get(), props);
        return obj + ".Length";
    }

    if (auto* se = dynamic_cast<const SwizzleExpr*>(expr))
    {
        std::string obj = EmitExpr(se->object.get(), props);
        return obj + "." + se->swizzle;
    }

    if (auto* ile = dynamic_cast<const InitializerListExpr*>(expr))
    {
        std::string result = "{ ";
        for (size_t i = 0; i < ile->elements.size(); ++i)
        {
            if (i > 0) result += ", ";
            result += EmitExpr(ile->elements[i].get(), props);
        }
        result += " }";
        return result;
    }

    return "";
}

std::string HLSLBackend::EmitBinaryOp(BinaryExpr::Op op)
{
    switch (op)
    {
        case BinaryExpr::Op::Add: return "+";
        case BinaryExpr::Op::Sub: return "-";
        case BinaryExpr::Op::Mul: return "*";
        case BinaryExpr::Op::Div: return "/";
        case BinaryExpr::Op::Mod: return "%";
        case BinaryExpr::Op::Eq:  return "==";
        case BinaryExpr::Op::Neq: return "!=";
        case BinaryExpr::Op::Lt:  return "<";
        case BinaryExpr::Op::Gt:  return ">";
        case BinaryExpr::Op::Le:  return "<=";
        case BinaryExpr::Op::Ge:  return ">=";
        case BinaryExpr::Op::And: return "&&";
        case BinaryExpr::Op::Or:  return "||";
        case BinaryExpr::Op::BitAnd: return "&";
        case BinaryExpr::Op::BitOr:  return "|";
        case BinaryExpr::Op::BitXor: return "^";
        case BinaryExpr::Op::Shl: return "<<";
        case BinaryExpr::Op::Shr: return ">>";
        case BinaryExpr::Op::Assign:    return "=";
        case BinaryExpr::Op::AddAssign: return "+=";
        case BinaryExpr::Op::SubAssign: return "-=";
        case BinaryExpr::Op::MulAssign: return "*=";
        case BinaryExpr::Op::DivAssign: return "/=";
        case BinaryExpr::Op::ModAssign: return "%=";
        case BinaryExpr::Op::AndAssign: return "&=";
        case BinaryExpr::Op::OrAssign:  return "|=";
        case BinaryExpr::Op::XorAssign: return "^=";
        case BinaryExpr::Op::ShlAssign: return "<<=";
        case BinaryExpr::Op::ShrAssign: return ">>=";
    }
    return "";
}

std::string HLSLBackend::EmitUnaryOp(UnaryExpr::Op op)
{
    switch (op)
    {
        case UnaryExpr::Op::Neg:      return "-";
        case UnaryExpr::Op::Not:      return "!";
        case UnaryExpr::Op::BitNot:   return "~";
        case UnaryExpr::Op::PreInc:   return "++";
        case UnaryExpr::Op::PreDec:   return "--";
        case UnaryExpr::Op::PostInc:  return "++";
        case UnaryExpr::Op::PostDec:  return "--";
    }
    return "";
}

} // namespace NSL
