#pragma once

#include "sampler_types.h"
#include "../types/northstar_types.h"

namespace Northstar
{
    struct SamplerDesc
    {
        SamplerFilter         minFilter      = SamplerFilter::Linear;
        SamplerFilter         magFilter      = SamplerFilter::Linear;
        SamplerFilter         mipFilter      = SamplerFilter::Linear;
        SamplerWrap           wrapU          = SamplerWrap::Repeat;
        SamplerWrap           wrapV          = SamplerWrap::Repeat;
        SamplerWrap           wrapW          = SamplerWrap::Repeat;
        UINT8                 maxAnisotropy  = 1;
        SamplerComparisonMode comparisonMode = SamplerComparisonMode::None;

        [[nodiscard]] static constexpr SamplerDesc LinearWrap(UINT8 maxAniso = 1) noexcept
        {
            SamplerDesc d{};
            d.maxAnisotropy = maxAniso;
            return d;
        }

        [[nodiscard]] static constexpr SamplerDesc LinearClamp() noexcept
        {
            SamplerDesc d{};
            d.wrapU = SamplerWrap::Clamp;
            d.wrapV = SamplerWrap::Clamp;
            d.wrapW = SamplerWrap::Clamp;
            return d;
        }

        [[nodiscard]] static constexpr SamplerDesc PointClamp() noexcept
        {
            SamplerDesc d{};
            d.minFilter = SamplerFilter::Nearest;
            d.magFilter = SamplerFilter::Nearest;
            d.mipFilter = SamplerFilter::Nearest;
            d.wrapU     = SamplerWrap::Clamp;
            d.wrapV     = SamplerWrap::Clamp;
            d.wrapW     = SamplerWrap::Clamp;
            return d;
        }

        [[nodiscard]] static constexpr SamplerDesc AnisotropicWrap(UINT8 maxAniso = 16) noexcept
        {
            SamplerDesc d{};
            d.minFilter     = SamplerFilter::Anisotropic;
            d.magFilter     = SamplerFilter::Anisotropic;
            d.mipFilter     = SamplerFilter::Anisotropic;
            d.maxAnisotropy = maxAniso;
            return d;
        }
    };
}
