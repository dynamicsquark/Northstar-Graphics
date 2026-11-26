#pragma once

#include "northstar_platform_guard.h"
#include "northstar_types.h"

namespace Northstar
{
    enum class CullMode : UINT8
    {
        NONE  = 0,
        FRONT = 1,
        BACK  = 2,

        None  = NONE,
        Front = FRONT,
        Back  = BACK
    };

    enum class BlendMode : UINT8
    {
        OPAQUE              = 0,
        ALPHA_BLEND         = 1,
        ADDITIVE            = 2,
        MULTIPLY            = 3,
        PREMULTIPLIED_ALPHA = 4,

        Opaque              = OPAQUE,
        AlphaBlend          = ALPHA_BLEND,
        Additive            = ADDITIVE,
        Multiply            = MULTIPLY,
        PremultipliedAlpha  = PREMULTIPLIED_ALPHA
    };

    enum class DepthCompare : UINT8
    {
        NEVER         = 0,
        LESS          = 1,
        EQUAL         = 2,
        LESS_EQUAL    = 3,
        GREATER       = 4,
        GREATER_EQUAL = 5,
        NOT_EQUAL     = 6,
        ALWAYS        = 7,

        Never         = NEVER,
        Less          = LESS,
        Equal         = EQUAL,
        LessEqual     = LESS_EQUAL,
        Greater       = GREATER,
        GreaterEqual  = GREATER_EQUAL,
        NotEqual      = NOT_EQUAL,
        Always        = ALWAYS
    };

    enum class FillMode : UINT8
    {
        SOLID     = 0,
        WIREFRAME = 1,

        Solid     = SOLID,
        Wireframe = WIREFRAME
    };

    struct DepthState
    {
        BOOL         test    = true;
        BOOL         write   = true;
        DepthCompare compare = DepthCompare::LessEqual;
    };

    enum class ComputePassPhase : UINT8
    {
        PRE_PASS    = 0,
        POST_PASS   = 1,
        POST_DEPTH  = 2,
        POST_OPAQUE = 3,

        PrePass     = PRE_PASS,
        PostPass    = POST_PASS,
        PostDepth   = POST_DEPTH,
        PostOpaque  = POST_OPAQUE
    };

    enum class ComputeExecutionScope : UINT8
    {
        PER_VIEW  = 0,
        PER_FRAME = 1,

        PerView   = PER_VIEW,
        PerFrame  = PER_FRAME
    };

    enum class LightType : UINT8
    {
        DIRECTIONAL = 0,
        POINT       = 1,
        SPOT        = 2,

        Directional = DIRECTIONAL,
        Point       = POINT,
        Spot        = SPOT
    };

    enum class LightShadowSoftness : UINT8
    {
        HARD        = 0,
        MEDIUM_HARD = 1,
        MEDIUM      = 2,
        MEDIUM_SOFT = 3,
        SOFT        = 4,

        Hard        = HARD,
        MediumHard  = MEDIUM_HARD,
        Medium      = MEDIUM,
        MediumSoft  = MEDIUM_SOFT,
        Soft        = SOFT
    };

    inline constexpr BOOL IsValidLightShadowSoftness(LightShadowSoftness softness) noexcept
    {
        switch (softness)
        {
            case LightShadowSoftness::HARD:
            case LightShadowSoftness::MEDIUM_HARD:
            case LightShadowSoftness::MEDIUM:
            case LightShadowSoftness::MEDIUM_SOFT:
            case LightShadowSoftness::SOFT:
                return true;
            default:
                return false;
        }
    }
}
