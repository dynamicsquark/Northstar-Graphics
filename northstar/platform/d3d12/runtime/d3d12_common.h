#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#define BOOL WIN_BOOL
#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#undef BOOL

#undef OPAQUE
#undef TRANSPARENT
#undef IN
#undef OUT
#endif

#include "texture/texture_desc.h"
#include "sampler/sampler_desc.h"
#include "types/northstar_types.h"

namespace Northstar
{
    using Microsoft::WRL::ComPtr;

    namespace D3D12Util
    {
        inline D3D12_HEAP_PROPERTIES GetDefaultHeapProperties()
        {
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type                 = D3D12_HEAP_TYPE_DEFAULT;
            hp.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            hp.CreationNodeMask     = 1;
            hp.VisibleNodeMask      = 1;
            return hp;
        }

        inline D3D12_HEAP_PROPERTIES GetUploadHeapProperties()
        {
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type                 = D3D12_HEAP_TYPE_UPLOAD;
            hp.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            hp.CreationNodeMask     = 1;
            hp.VisibleNodeMask      = 1;
            return hp;
        }

        inline D3D12_RESOURCE_DESC CreateBufferResourceDesc(UINT64 byteSize)
        {
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Alignment          = 0;
            desc.Width              = byteSize;
            desc.Height             = 1;
            desc.DepthOrArraySize   = 1;
            desc.MipLevels          = 1;
            desc.Format             = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count   = 1;
            desc.SampleDesc.Quality = 0;
            desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags              = D3D12_RESOURCE_FLAG_NONE;
            return desc;
        }

        inline D3D12_RESOURCE_BARRIER CreateTransitionBarrier(
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES stateBefore,
            D3D12_RESOURCE_STATES stateAfter)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource   = resource;
            barrier.Transition.StateBefore = stateBefore;
            barrier.Transition.StateAfter  = stateAfter;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            return barrier;
        }

        inline DXGI_FORMAT ToDxgiFormat(TextureFormat format)
        {
            switch (format)
            {
            case TextureFormat::R8_UNORM:        return DXGI_FORMAT_R8_UNORM;
            case TextureFormat::RG8_UNORM:       return DXGI_FORMAT_R8G8_UNORM;
            case TextureFormat::RGBA8_UNORM:     return DXGI_FORMAT_R8G8B8A8_UNORM;
            case TextureFormat::RGBA8_SRGB:      return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            case TextureFormat::R16_FLOAT:       return DXGI_FORMAT_R16_FLOAT;
            case TextureFormat::RG16_FLOAT:      return DXGI_FORMAT_R16G16_FLOAT;
            case TextureFormat::RGBA16_FLOAT:    return DXGI_FORMAT_R16G16B16A16_FLOAT;
            case TextureFormat::R32_FLOAT:       return DXGI_FORMAT_R32_FLOAT;
            case TextureFormat::RG32_FLOAT:      return DXGI_FORMAT_R32G32_FLOAT;
            case TextureFormat::RGBA32_FLOAT:    return DXGI_FORMAT_R32G32B32A32_FLOAT;
            case TextureFormat::R32_UINT:        return DXGI_FORMAT_R32_UINT;
            case TextureFormat::RG32_UINT:       return DXGI_FORMAT_R32G32_UINT;
            case TextureFormat::RGBA32_UINT:     return DXGI_FORMAT_R32G32B32A32_UINT;
            case TextureFormat::D16_UNORM:       return DXGI_FORMAT_D16_UNORM;
            case TextureFormat::D24_UNORM_S8:    return DXGI_FORMAT_D24_UNORM_S8_UINT;
            case TextureFormat::D32_FLOAT:       return DXGI_FORMAT_D32_FLOAT;
            case TextureFormat::D32_FLOAT_S8:    return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            case TextureFormat::BC1_UNORM:       return DXGI_FORMAT_BC1_UNORM;
            case TextureFormat::BC3_UNORM:       return DXGI_FORMAT_BC3_UNORM;
            case TextureFormat::BC5_UNORM:       return DXGI_FORMAT_BC5_UNORM;
            case TextureFormat::BC7_UNORM:       return DXGI_FORMAT_BC7_UNORM;
            case TextureFormat::BC7_SRGB:        return DXGI_FORMAT_BC7_UNORM_SRGB;
            default:                             return DXGI_FORMAT_R8G8B8A8_UNORM;
            }
        }

        inline D3D12_FILTER ToD3D12Filter(SamplerFilter minFilter, SamplerFilter magFilter, SamplerFilter mipFilter, BOOL isComparison)
        {
            if (minFilter == SamplerFilter::ANISOTROPIC || magFilter == SamplerFilter::ANISOTROPIC || mipFilter == SamplerFilter::ANISOTROPIC)
            {
                return isComparison ? D3D12_FILTER_COMPARISON_ANISOTROPIC : D3D12_FILTER_ANISOTROPIC;
            }

            const BOOL minLinear = (minFilter == SamplerFilter::LINEAR);
            const BOOL magLinear = (magFilter == SamplerFilter::LINEAR);
            const BOOL mipLinear = (mipFilter == SamplerFilter::LINEAR);

            if (isComparison)
            {
                if (minLinear && magLinear && mipLinear) return D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
                if (minLinear && magLinear)              return D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
                if (minLinear && mipLinear)              return D3D12_FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
                if (minLinear)                           return D3D12_FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT;
                if (magLinear && mipLinear)              return D3D12_FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR;
                if (magLinear)                           return D3D12_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT;
                if (mipLinear)                           return D3D12_FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR;
                return D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
            }
            else
            {
                if (minLinear && magLinear && mipLinear) return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                if (minLinear && magLinear)              return D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
                if (minLinear && mipLinear)              return D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR;
                if (minLinear)                           return D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT;
                if (magLinear && mipLinear)              return D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR;
                if (magLinear)                           return D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
                if (mipLinear)                           return D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR;
                return D3D12_FILTER_MIN_MAG_MIP_POINT;
            }
        }

        inline D3D12_TEXTURE_ADDRESS_MODE ToD3D12AddressMode(SamplerWrap mode)
        {
            switch (mode)
            {
            case SamplerWrap::REPEAT: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            case SamplerWrap::CLAMP:  return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case SamplerWrap::MIRROR: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case SamplerWrap::BORDER: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default:                  return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            }
        }

        inline D3D12_COMPARISON_FUNC ToD3D12ComparisonFunc(SamplerComparisonMode mode)
        {
            switch (mode)
            {
            case SamplerComparisonMode::NONE:          return D3D12_COMPARISON_FUNC_NEVER;
            case SamplerComparisonMode::LESS:          return D3D12_COMPARISON_FUNC_LESS;
            case SamplerComparisonMode::EQUAL:         return D3D12_COMPARISON_FUNC_EQUAL;
            case SamplerComparisonMode::LESS_EQUAL:    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case SamplerComparisonMode::GREATER:       return D3D12_COMPARISON_FUNC_GREATER;
            case SamplerComparisonMode::NOT_EQUAL:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case SamplerComparisonMode::GREATER_EQUAL: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            case SamplerComparisonMode::ALWAYS:        return D3D12_COMPARISON_FUNC_ALWAYS;
            default:                                   return D3D12_COMPARISON_FUNC_NEVER;
            }
        }
    }
}
