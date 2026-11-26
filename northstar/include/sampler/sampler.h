#pragma once

#include "sampler_desc.h"

namespace Northstar
{
    // ============ SamplerState ============
    class SamplerState
    {
    public:
        NativeHandle          GetNativeHandle()                           const { return m_nativeHandle;   }

        // ==================== GETTERS ====================
        SamplerFilter         GetMinFilter()                              const { return m_minFilter;      }
        SamplerFilter         GetMagFilter()                              const { return m_magFilter;      }
        SamplerFilter         GetMipFilter()                              const { return m_mipFilter;      }
        SamplerWrap           GetWrapU()                                  const { return m_wrapU;          }
        SamplerWrap           GetWrapV()                                  const { return m_wrapV;          }
        SamplerWrap           GetWrapW()                                  const { return m_wrapW;          }
        UINT8                 GetMaxAnisotropy()                          const { return m_maxAnisotropy;  }
        SamplerComparisonMode GetComparisonMode()                         const { return m_comparisonMode; }

        // NSL DataTypeID for PROPERTY sampler slots (State vs ComparisonState)
        DataTypeID            GetNSLDataTypeID()                          const
        {
            return SamplerComparisonToDataTypeID(m_comparisonMode);
        }
        ResourceKind          GetNSLResourceKind()                        const
        {
            return SamplerToResourceKind();
        }

        // ==================== NSL INTEGRATION ====================
        BOOL IsComparison() const
        {
            return IsComparisonSampler(m_comparisonMode);
        }

        BOOL MatchesNSCSampler(const NSCSampler& nscSampler) const
        {
            return IsComparison() == nscSampler.isComparison;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreTextureManager;

        SamplerState() = default;

        void SetNativeHandle(NativeHandle handle) { m_nativeHandle = handle; }
        void SetFilter(SamplerFilter minFilter, SamplerFilter magFilter, SamplerFilter mipFilter)
        {
            m_minFilter = minFilter;
            m_magFilter = magFilter;
            m_mipFilter = mipFilter;
        }
        void SetWrap(SamplerWrap u, SamplerWrap v, SamplerWrap w)
        {
            m_wrapU = u;
            m_wrapV = v;
            m_wrapW = w;
        }
        void SetMaxAnisotropy(UINT8 maxAniso)              { m_maxAnisotropy = maxAniso; }
        void SetComparisonMode(SamplerComparisonMode mode) { m_comparisonMode = mode; }

        NativeHandle          m_nativeHandle   = 0;

        SamplerFilter         m_minFilter      = SamplerFilter::LINEAR;
        SamplerFilter         m_magFilter      = SamplerFilter::LINEAR;
        SamplerFilter         m_mipFilter      = SamplerFilter::LINEAR;
        SamplerWrap           m_wrapU          = SamplerWrap::REPEAT;
        SamplerWrap           m_wrapV          = SamplerWrap::REPEAT;
        SamplerWrap           m_wrapW          = SamplerWrap::REPEAT;
        UINT8                 m_maxAnisotropy  = 1;
        SamplerComparisonMode m_comparisonMode = SamplerComparisonMode::NONE;
    };
}
