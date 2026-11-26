#pragma once

#include "nsl_common_types.h"
#include <cstdint>
#include <string>
#include <string_view>

// =====================================================================
//  NSL Stage I/O Catalog
//
//  Backend-agnostic INPUT_LAYOUT / OUTPUT_LAYOUT entries.
//  Target semantics (HLSL SV_*, GLSL locations, …) live in backends.
// =====================================================================

namespace NSL
{
    enum class StageIODirection : uint8_t
    {
        INPUT  = 0,
        OUTPUT = 1,
    };

    enum class StageIOCategory : uint8_t
    {
        VERTEX_ATTRIBUTE = 0,
        VARYING          = 1,
        SYSTEM           = 2, // CLIP_POSITION, COLOR_TARGET*, VERTEX_ID, ...
    };

    enum class ShaderStageMask : uint8_t
    {
        NONE     = 0,
        VERTEX   = 1 << 0,
        PIXEL    = 1 << 1,
        COMPUTE  = 1 << 2,
        GEOMETRY = 1 << 3,
        HULL     = 1 << 4,
        DOMAIN   = 1 << 5,
        ANY      = 0xFF,
    };

    inline constexpr ShaderStageMask operator|(ShaderStageMask a, ShaderStageMask b)
    {
        return static_cast<ShaderStageMask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    inline constexpr ShaderStageMask operator&(ShaderStageMask a, ShaderStageMask b)
    {
        return static_cast<ShaderStageMask>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

    inline constexpr bool HasStage(ShaderStageMask mask, ShaderStageMask flag)
    {
        return (static_cast<uint8_t>(mask) & static_cast<uint8_t>(flag)) != 0;
    }

    inline ShaderStageMask ShaderStageToMask(ShaderStage stage)
    {
        switch (stage)
        {
            case ShaderStage::VERTEX:   return ShaderStageMask::VERTEX;
            case ShaderStage::PIXEL:    return ShaderStageMask::PIXEL;
            case ShaderStage::COMPUTE:  return ShaderStageMask::COMPUTE;
            case ShaderStage::GEOMETRY: return ShaderStageMask::GEOMETRY;
            case ShaderStage::HULL:     return ShaderStageMask::HULL;
            case ShaderStage::DOMAIN:   return ShaderStageMask::DOMAIN;
            default:                    return ShaderStageMask::NONE;
        }
    }

    struct StageIOProperty
    {
        const char*       name;
        DataTypeID        typeID;
        StageIODirection  direction;
        StageIOCategory   category;
        ShaderStageMask   validStages;
    };

    inline constexpr StageIOProperty STAGE_IO_PROPERTIES[] =
    {
        // ---------- Vertex attributes (VS INPUT) ----------
        { "VERTEX_POSITION",     DataTypeID::VEC3,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_NORMAL",       DataTypeID::VEC3,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_TANGENT",      DataTypeID::VEC3,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_BITANGENT",    DataTypeID::VEC3,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_UV0",          DataTypeID::VEC2,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_UV1",          DataTypeID::VEC2,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_UV2",          DataTypeID::VEC2,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_UV3",          DataTypeID::VEC2,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_COLOR",        DataTypeID::VEC4,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_COLOR0",       DataTypeID::VEC4,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_COLOR1",       DataTypeID::VEC4,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_BONE_INDICES", DataTypeID::UINT4, StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },
        { "VERTEX_BONE_WEIGHTS", DataTypeID::VEC4,  StageIODirection::INPUT,  StageIOCategory::VERTEX_ATTRIBUTE, ShaderStageMask::VERTEX },

        // ---------- System inputs ----------
        { "VERTEX_ID",           DataTypeID::UINT,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::VERTEX },
        { "INSTANCE_ID",         DataTypeID::UINT,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::VERTEX },
        { "CLIP_POSITION",       DataTypeID::VEC4,  StageIODirection::INPUT, StageIOCategory::SYSTEM,
          ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "IS_FRONT_FACE",       DataTypeID::BOOL,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL },
        { "SAMPLE_INDEX",        DataTypeID::UINT,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL },
        { "PRIMITIVE_ID",        DataTypeID::UINT,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY },
        { "DISPATCH_THREAD_ID",  DataTypeID::UINT3, StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::COMPUTE },
        { "GROUP_ID",            DataTypeID::UINT3, StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::COMPUTE },
        { "GROUP_THREAD_ID",     DataTypeID::UINT3, StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::COMPUTE },
        { "GROUP_INDEX",         DataTypeID::UINT,  StageIODirection::INPUT, StageIOCategory::SYSTEM, ShaderStageMask::COMPUTE },

        // ---------- Varyings (Output) ----------
        { "WORLD_POSITION", DataTypeID::VEC3, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "WORLD_NORMAL",   DataTypeID::VEC3, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV0",            DataTypeID::VEC2, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV1",            DataTypeID::VEC2, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV2",            DataTypeID::VEC2, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV3",            DataTypeID::VEC2, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "TANGENT",        DataTypeID::VEC3, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "BITANGENT",      DataTypeID::VEC3, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "COLOR",          DataTypeID::VEC4, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "VIEW_DIRECTION", DataTypeID::VEC3, StageIODirection::OUTPUT, StageIOCategory::VARYING, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },

        // ---------- Varyings (Input) ----------
        { "WORLD_POSITION", DataTypeID::VEC3, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "WORLD_NORMAL",   DataTypeID::VEC3, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV0",            DataTypeID::VEC2, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV1",            DataTypeID::VEC2, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV2",            DataTypeID::VEC2, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "UV3",            DataTypeID::VEC2, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "TANGENT",        DataTypeID::VEC3, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "BITANGENT",      DataTypeID::VEC3, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "COLOR",          DataTypeID::VEC4, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "VIEW_DIRECTION", DataTypeID::VEC3, StageIODirection::INPUT, StageIOCategory::VARYING, ShaderStageMask::PIXEL | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },

        // ---------- System outputs ----------
        { "CLIP_POSITION",             DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::HULL | ShaderStageMask::DOMAIN },
        { "RENDER_TARGET_ARRAY_INDEX", DataTypeID::UINT,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::DOMAIN },
        { "RENDER_TARGET_ARRAY_INDEX", DataTypeID::UINT,  StageIODirection::INPUT,  StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  | ShaderStageMask::GEOMETRY },
        { "VIEWPORT_ARRAY_INDEX",      DataTypeID::UINT,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::VERTEX | ShaderStageMask::GEOMETRY | ShaderStageMask::DOMAIN },
        { "COLOR_TARGET0",             DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET1", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET2", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET3", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET4", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET5", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET6", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "COLOR_TARGET7", DataTypeID::VEC4,  StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
        { "FRAG_DEPTH",    DataTypeID::FLOAT, StageIODirection::OUTPUT, StageIOCategory::SYSTEM, ShaderStageMask::PIXEL  },
    };

    inline constexpr uint32_t STAGE_IO_PROPERTY_COUNT =
        static_cast<uint32_t>(sizeof(STAGE_IO_PROPERTIES) / sizeof(STAGE_IO_PROPERTIES[0]));

    inline const StageIOProperty* FindStageIOProperty(
        const char* name,
        StageIODirection direction,
        ShaderStage stage)
    {
        const ShaderStageMask stageMask = ShaderStageToMask(stage);
        for (uint32_t i = 0; i < STAGE_IO_PROPERTY_COUNT; ++i)
        {
            const StageIOProperty& p = STAGE_IO_PROPERTIES[i];
            if (p.direction != direction) continue;
            if (std::string_view(p.name) != name) continue;
            if (!HasStage(p.validStages, stageMask)) continue;
            return &p;
        }
        return nullptr;
    }

    inline const StageIOProperty* FindStageIOProperty(
        const std::string& name,
        StageIODirection direction,
        ShaderStage stage)
    {
        return FindStageIOProperty(name.c_str(), direction, stage);
    }

    inline bool IsStageIOName(const std::string& name)
    {
        for (uint32_t i = 0; i < STAGE_IO_PROPERTY_COUNT; ++i)
            if (std::string_view(STAGE_IO_PROPERTIES[i].name) == name)
                return true;
        return false;
    }

} // namespace NSL
