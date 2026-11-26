#pragma once

#include "../shader/shader_types.h"
#include "../types/northstar_types.h"

namespace Northstar
{
    enum class SamplerFilter : UINT8
    {
        NEAREST     = 0,
        LINEAR      = 1,
        ANISOTROPIC = 2,

        Nearest     = NEAREST,
        Point       = NEAREST,
        Linear      = LINEAR,
        Anisotropic = ANISOTROPIC
    };

    enum class SamplerWrap : UINT8
    {
        REPEAT = 0,
        CLAMP  = 1,
        MIRROR = 2,
        BORDER = 3,

        Repeat = REPEAT,
        Clamp  = CLAMP,
        Mirror = MIRROR,
        Border = BORDER
    };

    enum class SamplerComparisonMode : UINT8
    {
        NONE          = 0,
        LESS          = 1,
        EQUAL         = 2,
        LESS_EQUAL    = 3,
        GREATER       = 4,
        NOT_EQUAL     = 5,
        GREATER_EQUAL = 6,
        ALWAYS        = 7,

        None          = NONE,
        Less          = LESS,
        Equal         = EQUAL,
        LessEqual     = LESS_EQUAL,
        Greater       = GREATER,
        NotEqual      = NOT_EQUAL,
        GreaterEqual  = GREATER_EQUAL,
        Always        = ALWAYS
    };

    inline BOOL IsComparisonSampler(SamplerComparisonMode mode)
    {
        return mode != SamplerComparisonMode::NONE;
    }

    inline DataTypeID SamplerComparisonToDataTypeID(SamplerComparisonMode mode)
    {
        return IsComparisonSampler(mode)
            ? DataTypeID::SAMPLER_COMPARISON_STATE
            : DataTypeID::SAMPLER_STATE;
    }

    inline ResourceKind SamplerToResourceKind()
    {
        return ResourceKind::SAMPLER_STATE;
    }
}
