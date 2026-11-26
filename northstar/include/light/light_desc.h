#pragma once

#include "light_types.h"
#include "light_bitwise.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"

namespace Northstar
{
    // ==================== LIGHT DESC ====================
    struct LightDesc
    {
        // ==================== CORE ====================
        LightType              type                   = LightType::POINT;
        Color                  color                  = Color::White();
        FLOAT32                intensity              = 1.0f;
        LightFlags             flags                  = LightFlags::ENABLED | LightFlags::CAST_SHADOW;

        // ==================== TRANSFORM ====================
        Vec3                   position               = Vec3::Zero();
        Quat                   rotation               = Quat::Identity();      // Left-handed orientation (+Z forward)

        // ==================== ATTENUATION (POINT / SPOT) ====================
        FLOAT32                range                  = 10.0f;
        FLOAT32                constantAttenuation    = 1.0f;
        FLOAT32                linearAttenuation      = 0.09f;
        FLOAT32                quadraticAttenuation   = 0.032f;

        // ==================== SPOT ====================
        FLOAT32                innerConeAngle         = 25.0f;                 // degrees
        FLOAT32                outerConeAngle         = 35.0f;                 // degrees

        // ==================== SHADOW PARAMETERS (NVS) ====================
        // shadowBias and shadowNormalBias use standardized dynamic texel-relative units in [0.0, 1.0]:
        //   0.0 = zero additional bias
        //   0.5 = 0.5x local virtual shadow texel footprint at the resolved view/mip level
        //   1.0 = 1.0x local virtual shadow texel footprint at the resolved view/mip level
        LightShadowResolution     shadowResolution       = LightShadowResolution::RES_16384;
        LightShadowSoftness       shadowSoftness         = LightShadowSoftness::HARD;
        FLOAT32                   shadowBias             = 0.5f;
        FLOAT32                   shadowNormalBias       = 0.5f;
        FLOAT32                   shadowNearPlane        = 0.1f;
        FLOAT32                   shadowMaxDistance      = 150.0f;

        // ==================== CULLING ====================
        UINT32                    cullingMask            = 0xFFFFFFFF;
    };
}
