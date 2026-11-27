#pragma once

#include "../types/northstar_types.h"
#include "../types/pipeline_state_types.h"

namespace Northstar
{
    enum class LightFlags : UINT32
    {
        ENABLED     = 1 << 0,
        CAST_SHADOW = 1 << 1,

        Enabled     = ENABLED,
        CastShadow  = CAST_SHADOW
    };

    enum class LightShadowResolution : UINT16
    {
        RES_512   = 512,
        RES_1024  = 1024,
        RES_2048  = 2048,
        RES_4096  = 4096,
        RES_8192  = 8192,
        RES_16384 = 16384,

        Res512    = RES_512,
        Res1024   = RES_1024,
        Res2048   = RES_2048,
        Res4096   = RES_4096,
        Res8192   = RES_8192,
        Res16384  = RES_16384
    };

    inline constexpr BOOL IsValidLightShadowResolution(LightShadowResolution res) noexcept
    {
        switch (res)
        {
            case LightShadowResolution::RES_512:
            case LightShadowResolution::RES_1024:
            case LightShadowResolution::RES_2048:
            case LightShadowResolution::RES_4096:
            case LightShadowResolution::RES_8192:
            case LightShadowResolution::RES_16384:
                return true;
            default:
                return false;
        }
    }
}
