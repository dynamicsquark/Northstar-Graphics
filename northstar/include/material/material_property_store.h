#pragma once

#include "../shader/shader_property_store.h"

namespace Northstar
{
    // Backward-compatible aliases pointing to the object-independent ShaderPropertyStore types
    using MaterialBufferSlot           = ShaderBufferSlot;
    using MaterialTextureSlot          = ShaderTextureSlot;
    using MaterialRWTextureSlot        = ShaderRWTextureSlot;
    using MaterialSamplerSlot          = ShaderSamplerSlot;
    using MaterialStructuredBufferSlot = ShaderStructuredBufferSlot;
    using MaterialStageProps           = ShaderStageProps;

    // ==================== FULL MATERIAL STORE ====================
    // Indexed: [PassType][GraphicsShaderSlot]
    struct MaterialPropertyStore
    {
        ShaderStageProps stages
            [static_cast<SIZE_T>(PassType::COUNT)]
            [static_cast<SIZE_T>(GraphicsShaderSlot::COUNT)];

        ShaderStageProps& Get(PassType pass, GraphicsShaderSlot slot)
        {
            return stages[static_cast<SIZE_T>(pass)][static_cast<SIZE_T>(slot)];
        }

        const ShaderStageProps& Get(PassType pass, GraphicsShaderSlot slot) const
        {
            return stages[static_cast<SIZE_T>(pass)][static_cast<SIZE_T>(slot)];
        }

        ShaderStageProps* Get(PassType pass, ShaderStage stage)
        {
            const GraphicsShaderSlot slot = ShaderStageToSlot(stage);
            if (pass >= PassType::COUNT || slot >= GraphicsShaderSlot::COUNT)
                return nullptr;
            return &Get(pass, slot);
        }

        const ShaderStageProps* Get(PassType pass, ShaderStage stage) const
        {
            const GraphicsShaderSlot slot = ShaderStageToSlot(stage);
            if (pass >= PassType::COUNT || slot >= GraphicsShaderSlot::COUNT)
                return nullptr;
            return &Get(pass, slot);
        }
    };
}
