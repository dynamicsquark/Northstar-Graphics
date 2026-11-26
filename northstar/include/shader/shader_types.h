#pragma once

#include "NSLCommon/nsl_common_types.h"
#include "NSLCommon/nsl_shader_data.h"
#include "NSLCommon/nsl_engine_data.h"
#include "NSLCommon/nsl_stage_io.h"
#include "../types/northstar_types.h"
#include "../types/northstar_platform_guard.h"

namespace Northstar
{
    using ShaderStage          = NSL::ShaderStage;
    using DataTypeID           = NSL::DataTypeID;
    using TargetAPI            = NSL::TargetAPI;
    using ResourceKind         = NSL::ResourceKind;
    using PropertyEntryKind    = NSL::PropertyEntryKind;

    using NSCField             = NSL::NSCField;
    using NSCBuffer            = NSL::NSCBuffer;
    using NSCTexture           = NSL::NSCTexture;
    using NSCSampler           = NSL::NSCSampler;
    using NSCStructField       = NSL::NSCStructField;
    using NSCStructuredBuffer  = NSL::NSCStructuredBuffer;
    using NSCUAVBuffer         = NSL::NSCUAVBuffer;
    using NSCPropertyEntry     = NSL::NSCPropertyEntry;
    using NSCPropertyGroup     = NSL::NSCPropertyGroup;
    using NSCPropertyBlock     = NSL::NSCPropertyBlock;
    using NSCStageIOEntry      = NSL::NSCStageIOEntry;
    using ShaderData           = NSL::ShaderData;

    using EngineProperty       = NSL::EngineProperty;
    using StageIOProperty      = NSL::StageIOProperty;
    using StageIODirection     = NSL::StageIODirection;
    using StageIOCategory      = NSL::StageIOCategory;
    using ShaderStageMask      = NSL::ShaderStageMask;

    using ShaderReflection     = NSL::ShaderData;

    using NSL::FindEngineProperty;
    using NSL::IsEnginePropertyName;
    using NSL::FindStageIOProperty;
    using NSL::IsStageIOName;
    using NSL::ENGINE_PROPERTIES;
    using NSL::ENGINE_PROPERTY_COUNT;
    using NSL::STAGE_IO_PROPERTIES;
    using NSL::STAGE_IO_PROPERTY_COUNT;

    enum class ShaderDataSource : UINT8
    {
        MEMORY     = 0,
        REFLECTION = 1,

        Memory     = MEMORY,
        Reflection = REFLECTION
    };

    enum class ShaderVariant : UINT8
    {
        STANDARD          = 0,
        INSTANCED         = 1,
        SKINNED           = 2,
        INSTANCED_SKINNED = 3,
        COUNT             = 4,

        Standard          = STANDARD,
        Instanced         = INSTANCED,
        Skinned           = SKINNED,
        InstancedSkinned  = INSTANCED_SKINNED
    };

    enum class PassType : UINT8
    {
        GEOMETRY = 0,
        SHADOW   = 1,
        DEPTH    = 2,
        COUNT    = 3,

        Geometry = GEOMETRY,
        Shadow   = SHADOW,
        Depth    = DEPTH
    };

    enum class ShaderSlot : UINT8
    {
        VERTEX       = 0,
        HULL         = 1,
        DOMAIN_STAGE = 2,
        GEOMETRY     = 3,
        PIXEL        = 4,
        COUNT        = 5,

        Vertex       = VERTEX,
        Hull         = HULL,
        Domain       = DOMAIN_STAGE,
        Geometry     = GEOMETRY,
        Pixel        = PIXEL
    };

    using GraphicsShaderSlot = ShaderSlot;

    inline constexpr ShaderSlot ShaderStageToSlot(ShaderStage stage) noexcept
    {
        switch (stage)
        {
        case ShaderStage::VERTEX:   return ShaderSlot::VERTEX;
        case ShaderStage::HULL:     return ShaderSlot::HULL;
        case ShaderStage::DOMAIN:   return ShaderSlot::DOMAIN_STAGE;
        case ShaderStage::GEOMETRY: return ShaderSlot::GEOMETRY;
        case ShaderStage::PIXEL:    return ShaderSlot::PIXEL;
        default:                    return ShaderSlot::COUNT;
        }
    }

    inline constexpr ShaderStage SlotToShaderStage(ShaderSlot slot) noexcept
    {
        switch (slot)
        {
        case ShaderSlot::VERTEX:       return ShaderStage::VERTEX;
        case ShaderSlot::HULL:         return ShaderStage::HULL;
        case ShaderSlot::DOMAIN_STAGE: return ShaderStage::DOMAIN;
        case ShaderSlot::GEOMETRY:     return ShaderStage::GEOMETRY;
        case ShaderSlot::PIXEL:        return ShaderStage::PIXEL;
        default:                       return ShaderStage::NONE;
        }
    }
}
