#pragma once

#include "NSLCommon/nsl_stage_io.h"
#include <cstdint>
#include <string>
#include <string_view>

// =====================================================================
//  HLSL Stage I/O + system-value semantic maps
//
//  NSL catalog / source uses backend-agnostic names.
//  This file alone maps them to HLSL semantics for codegen / .nsc.
// =====================================================================

namespace NSL
{
    struct HLSLStageIOSemantic
    {
        const char*       name;       // NSL catalog name
        const char*       semantic;   // HLSL semantic
        StageIODirection  direction;
        ShaderStageMask   validStages;
    };

    inline constexpr HLSLStageIOSemantic HLSL_STAGE_IO_SEMANTICS[] =
    {
        { "VERTEX_POSITION",     "POSITION",      StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_NORMAL",       "NORMAL",        StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_TANGENT",      "TANGENT",       StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_BITANGENT",    "BINORMAL",      StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_UV0",          "TEXCOORD0",     StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_UV1",          "TEXCOORD1",     StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_UV2",          "TEXCOORD2",     StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_UV3",          "TEXCOORD3",     StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_COLOR",        "COLOR0",        StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_COLOR0",       "COLOR0",        StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_COLOR1",       "COLOR1",        StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_BONE_INDICES", "BLENDINDICES",  StageIODirection::INPUT,  ShaderStageMask::VERTEX },
        { "VERTEX_BONE_WEIGHTS", "BLENDWEIGHT",   StageIODirection::INPUT,  ShaderStageMask::VERTEX },

        { "VERTEX_ID",           "SV_VertexID",          StageIODirection::INPUT, ShaderStageMask::VERTEX },
        { "INSTANCE_ID",         "SV_InstanceID",        StageIODirection::INPUT, ShaderStageMask::VERTEX },
        { "CLIP_POSITION",       "SV_POSITION",          StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "IS_FRONT_FACE",       "SV_IsFrontFace",       StageIODirection::INPUT, ShaderStageMask::PIXEL },
        { "SAMPLE_INDEX",        "SV_SampleIndex",       StageIODirection::INPUT, ShaderStageMask::PIXEL },
        { "PRIMITIVE_ID",        "SV_PrimitiveID",       StageIODirection::INPUT, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY },
        { "DISPATCH_THREAD_ID",  "SV_DispatchThreadID",  StageIODirection::INPUT, ShaderStageMask::COMPUTE },
        { "GROUP_ID",            "SV_GroupID",           StageIODirection::INPUT, ShaderStageMask::COMPUTE },
        { "GROUP_THREAD_ID",     "SV_GroupThreadID",     StageIODirection::INPUT, ShaderStageMask::COMPUTE },
        { "GROUP_INDEX",         "SV_GroupIndex",        StageIODirection::INPUT, ShaderStageMask::COMPUTE },

        { "WORLD_POSITION", "TEXCOORD0", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "WORLD_NORMAL",   "TEXCOORD1", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV0",            "TEXCOORD2", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV1",            "TEXCOORD3", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV2",            "TEXCOORD4", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV3",            "TEXCOORD5", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "TANGENT",        "TEXCOORD6", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "BITANGENT",      "TEXCOORD7", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "COLOR",          "TEXCOORD8", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "VIEW_DIRECTION", "TEXCOORD9", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },

        { "WORLD_POSITION", "TEXCOORD0", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "WORLD_NORMAL",   "TEXCOORD1", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV0",            "TEXCOORD2", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV1",            "TEXCOORD3", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV2",            "TEXCOORD4", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV3",            "TEXCOORD5", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "TANGENT",        "TEXCOORD6", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "BITANGENT",      "TEXCOORD7", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "COLOR",          "TEXCOORD8", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "VIEW_DIRECTION", "TEXCOORD9", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },

        { "CLIP_POSITION", "SV_POSITION", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "RENDER_TARGET_ARRAY_INDEX", "SV_RenderTargetArrayIndex", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::DOMAIN },
        { "RENDER_TARGET_ARRAY_INDEX", "SV_RenderTargetArrayIndex", StageIODirection::INPUT,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY },
        { "VIEWPORT_ARRAY_INDEX", "SV_ViewportArrayIndex", StageIODirection::OUTPUT,
          ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::DOMAIN },
        { "COLOR_TARGET0", "SV_TARGET0", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET1", "SV_TARGET1", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET2", "SV_TARGET2", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET3", "SV_TARGET3", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET4", "SV_TARGET4", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET5", "SV_TARGET5", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET6", "SV_TARGET6", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "COLOR_TARGET7", "SV_TARGET7", StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
        { "FRAG_DEPTH",    "SV_DEPTH",   StageIODirection::OUTPUT, ShaderStageMask::PIXEL },
    };

    inline constexpr uint32_t HLSL_STAGE_IO_SEMANTIC_COUNT =
        static_cast<uint32_t>(sizeof(HLSL_STAGE_IO_SEMANTICS) / sizeof(HLSL_STAGE_IO_SEMANTICS[0]));

    inline const char* FindHLSLStageIOSemantic(
        const char* name,
        StageIODirection direction,
        ShaderStage stage)
    {
        const ShaderStageMask stageMask = ShaderStageToMask(stage);
        for (uint32_t i = 0; i < HLSL_STAGE_IO_SEMANTIC_COUNT; ++i)
        {
            const HLSLStageIOSemantic& e = HLSL_STAGE_IO_SEMANTICS[i];
            if (e.direction != direction) continue;
            if (std::string_view(e.name) != name) continue;
            if (!HasStage(e.validStages, stageMask)) continue;
            return e.semantic;
        }
        return nullptr;
    }

    // Maps NSL stage-I/O / entry-point semantic names → HLSL.
    // STAGE_INPUT/OUTPUT fields keep NSL names on the AST; emit calls this.
    inline std::string MapNSLSemanticToHLSL(const std::string& nslSemantic)
    {
        // Vertex attributes
        if (nslSemantic == "VERTEX_POSITION")     return "POSITION";
        if (nslSemantic == "VERTEX_NORMAL")       return "NORMAL";
        if (nslSemantic == "VERTEX_TANGENT")      return "TANGENT";
        if (nslSemantic == "VERTEX_BITANGENT")    return "BINORMAL";
        if (nslSemantic == "VERTEX_UV0")          return "TEXCOORD0";
        if (nslSemantic == "VERTEX_UV1")          return "TEXCOORD1";
        if (nslSemantic == "VERTEX_UV2")          return "TEXCOORD2";
        if (nslSemantic == "VERTEX_UV3")          return "TEXCOORD3";
        if (nslSemantic == "VERTEX_COLOR")        return "COLOR0";
        if (nslSemantic == "VERTEX_COLOR0")       return "COLOR0";
        if (nslSemantic == "VERTEX_COLOR1")       return "COLOR1";
        if (nslSemantic == "VERTEX_BONE_INDICES") return "BLENDINDICES";
        if (nslSemantic == "VERTEX_BONE_WEIGHTS") return "BLENDWEIGHT";

        // Varyings (fixed TEXCOORD slots)
        if (nslSemantic == "WORLD_POSITION") return "TEXCOORD0";
        if (nslSemantic == "WORLD_NORMAL")   return "TEXCOORD1";
        if (nslSemantic == "UV0")            return "TEXCOORD2";
        if (nslSemantic == "UV1")            return "TEXCOORD3";
        if (nslSemantic == "UV2")            return "TEXCOORD4";
        if (nslSemantic == "UV3")            return "TEXCOORD5";
        if (nslSemantic == "TANGENT")        return "TEXCOORD6";
        if (nslSemantic == "BITANGENT")      return "TEXCOORD7";
        if (nslSemantic == "COLOR")          return "TEXCOORD8";
        if (nslSemantic == "VIEW_DIRECTION") return "TEXCOORD9";

        // System values / entry-point params / tess inject
        if (nslSemantic == "IS_FRONT_FACE")       return "SV_IsFrontFace";
        if (nslSemantic == "SAMPLE_INDEX")        return "SV_SampleIndex";

        // System values / entry-point params / tess inject
        if (nslSemantic == "DISPATCH_THREAD_ID")  return "SV_DispatchThreadID";
        if (nslSemantic == "GROUP_ID")            return "SV_GroupID";
        if (nslSemantic == "GROUP_THREAD_ID")     return "SV_GroupThreadID";
        if (nslSemantic == "GROUP_INDEX")         return "SV_GroupIndex";
        if (nslSemantic == "PRIMITIVE_ID")        return "SV_PrimitiveID";
        if (nslSemantic == "GS_INSTANCE_ID")      return "SV_GSInstanceID";
        if (nslSemantic == "CONTROL_POINT_ID")    return "SV_OutputControlPointID";
        if (nslSemantic == "DOMAIN_LOCATION")     return "SV_DomainLocation";
        if (nslSemantic == "TESS_FACTOR")         return "SV_TessFactor";
        if (nslSemantic == "INSIDE_TESS_FACTOR")  return "SV_InsideTessFactor";
        if (nslSemantic == "CLIP_POSITION")       return "SV_POSITION";
        if (nslSemantic == "COLOR_TARGET0")       return "SV_TARGET0";
        if (nslSemantic == "COLOR_TARGET1")       return "SV_TARGET1";
        if (nslSemantic == "COLOR_TARGET2")       return "SV_TARGET2";
        if (nslSemantic == "COLOR_TARGET3")       return "SV_TARGET3";
        if (nslSemantic == "COLOR_TARGET4")       return "SV_TARGET4";
        if (nslSemantic == "COLOR_TARGET5")       return "SV_TARGET5";
        if (nslSemantic == "COLOR_TARGET6")       return "SV_TARGET6";
        if (nslSemantic == "COLOR_TARGET7")       return "SV_TARGET7";
        if (nslSemantic == "FRAG_DEPTH")                return "SV_DEPTH";
        if (nslSemantic == "RENDER_TARGET_ARRAY_INDEX") return "SV_RenderTargetArrayIndex";
        if (nslSemantic == "VIEWPORT_ARRAY_INDEX")      return "SV_ViewportArrayIndex";
        if (nslSemantic == "VERTEX_ID")                 return "SV_VertexID";
        if (nslSemantic == "INSTANCE_ID")               return "SV_InstanceID";
        return nslSemantic;
    }
}
