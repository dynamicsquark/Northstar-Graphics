#pragma once

#include "../types/northstar_types.h"

namespace Northstar
{
    // ==================== RENDER OBJECT FLAGS ====================
    enum class RenderObjectFlags : UINT32
    {
        NONE               = 0,
        STATIC             = 1 << 0,
        VISIBLE            = 1 << 1,
        CAST_SHADOW        = 1 << 2,
        DISTANCE_CULL      = 1 << 3,
        FRUSTUM_CULL       = 1 << 4,
        SMALL_OBJECT_CULL  = 1 << 5,
        OCCLUSION_CULL     = 1 << 6,
        USE_LOD            = 1 << 7,
        SKINNED            = 1 << 8,
        BAKED_STATIC       = 1 << 9,
        RECEIVE_SHADOW     = 1 << 10
    };
}