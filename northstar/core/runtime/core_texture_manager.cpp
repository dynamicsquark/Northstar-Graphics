#include "core_texture_manager.h"
#include "platform/northstar_backend.h"
#include <algorithm>

namespace Northstar
{
    CoreTextureManager::~CoreTextureManager()
    {
        Shutdown();
    }

    Texture* CoreTextureManager::CreateTexture(const TextureDesc& desc)
    {
        m_lastError.clear();

        if (desc.width == 0 || desc.height == 0)
        {
            m_lastError = "CreateTexture failed: width and height must be non-zero.";
            return nullptr;
        }

        if (desc.type == TextureType::TEXTURE_3D)
        {
            if (desc.depth == 0 || desc.width > 2048u || desc.height > 2048u || desc.depth > 2048u || desc.arrayLayers > 1u)
            {
                m_lastError = "CreateTexture failed: invalid TEXTURE_3D dimensions or arrayLayers.";
                return nullptr;
            }
        }
        else
        {
            if (desc.width > 16384u || desc.height > 16384u)
            {
                m_lastError = "CreateTexture failed: 2D/Cube dimensions exceed 16384.";
                return nullptr;
            }
            if ((desc.type == TextureType::TEXTURE_2D_ARRAY || desc.type == TextureType::TEXTURE_CUBE_ARRAY) && desc.arrayLayers == 0u)
            {
                m_lastError = "CreateTexture failed: arrayLayers must be >= 1 for array textures.";
                return nullptr;
            }
            if ((desc.type == TextureType::TEXTURE_CUBE || desc.type == TextureType::TEXTURE_CUBE_ARRAY) && desc.width != desc.height)
            {
                m_lastError = "CreateTexture failed: cubemap faces must be square (width == height).";
                return nullptr;
            }
        }

        const UINT32 arraySliceCount = ComputeTextureArraySliceCount(desc.type, desc.arrayLayers);
        if (arraySliceCount == 0u || arraySliceCount > 2048u)
        {
            m_lastError = "CreateTexture failed: total array slice count exceeds D3D12 2048 limit.";
            return nullptr;
        }

        const BOOL isDepthFmt = IsTextureFormatDepth(desc.format);
        const BOOL isBCnFmt   = IsTextureFormatBlockCompressed(desc.format);

        if (desc.usage == TextureUsage::DEPTH_STENCIL)
        {
            if (!isDepthFmt || desc.type == TextureType::TEXTURE_3D || desc.initialData != nullptr)
            {
                m_lastError = "CreateTexture failed: invalid format, dimension, or initialData for DEPTH_STENCIL usage.";
                return nullptr;
            }
        }
        else if (isDepthFmt)
        {
            m_lastError = "CreateTexture failed: depth format requires TextureUsage::DEPTH_STENCIL.";
            return nullptr;
        }

        if (isBCnFmt)
        {
            if (desc.usage != TextureUsage::SAMPLED || desc.type == TextureType::TEXTURE_3D)
            {
                m_lastError = "CreateTexture failed: block-compressed formats require 2D/Cube SAMPLED usage.";
                return nullptr;
            }
            if ((desc.width % 4u) != 0u || (desc.height % 4u) != 0u)
            {
                m_lastError = "CreateTexture failed: block-compressed Mip 0 dimensions must be multiples of 4.";
                return nullptr;
            }
            if (desc.mipLevels == 0u && desc.initialData != nullptr)
            {
                m_lastError = "CreateTexture failed: block-compressed textures require explicit mipLevels when initialData is supplied.";
                return nullptr;
            }
        }

        if (desc.usage == TextureUsage::STORAGE &&
            (desc.type == TextureType::TEXTURE_CUBE || desc.type == TextureType::TEXTURE_CUBE_ARRAY))
        {
            m_lastError = "CreateTexture failed: cubemap STORAGE UAV usage is unsupported.";
            return nullptr;
        }

        const UINT32 maxPossibleMips = ComputeTextureMaxMipLevels(desc.width, desc.height, desc.depth, desc.type);
        if (maxPossibleMips == 0u || desc.mipLevels > maxPossibleMips)
        {
            m_lastError = "CreateTexture failed: mipLevels exceeds maximum possible mip count for dimensions.";
            return nullptr;
        }

        const BOOL   autoMipChain      = (desc.mipLevels == 0u);
        const UINT32 resolvedMipLevels = autoMipChain
            ? ((isBCnFmt || isDepthFmt) ? 1u : maxPossibleMips)
            : desc.mipLevels;

        const SIZE_T mip0AllSlicesBytes      = ComputeTextureMip0AllSlicesTightBytes(desc.width, desc.height, desc.depth, desc.arrayLayers, desc.type, desc.format);
        const SIZE_T fullChainAllSlicesBytes = ComputeTextureFullChainTightBytes(desc.width, desc.height, desc.depth, resolvedMipLevels, desc.arrayLayers, desc.type, desc.format);
        SIZE_T pendingStoreBytes             = 0u;

        if (desc.initialData != nullptr)
        {
            if (mip0AllSlicesBytes == 0u)
            {
                m_lastError = "CreateTexture failed: invalid tight byte layout for format.";
                return nullptr;
            }
            if (desc.initialDataSize != 0u && desc.initialDataSize < mip0AllSlicesBytes)
            {
                m_lastError = "CreateTexture failed: initialDataSize is smaller than Mip 0 tight byte footprint.";
                return nullptr;
            }

            const BOOL callerProvidedFullChain =
                (resolvedMipLevels > 1u && desc.mipLevels > 1u && desc.initialDataSize >= fullChainAllSlicesBytes);
            if (resolvedMipLevels > 1u && !callerProvidedFullChain && !IsTextureFormatCPUFilterable(desc.format))
            {
                m_lastError = "CreateTexture failed: automatic mip generation from Mip 0 requires a CPU-filterable format.";
                return nullptr;
            }
            pendingStoreBytes = callerProvidedFullChain ? fullChainAllSlicesBytes : mip0AllSlicesBytes;
        }

        Texture* texture = new Texture();
        texture->SetDimensions(desc.width, desc.height, (desc.type == TextureType::TEXTURE_3D) ? desc.depth : 1u, resolvedMipLevels, desc.arrayLayers, autoMipChain);
        texture->SetFormatAndType(desc.format, desc.type, desc.usage, desc.dataSource, desc.allowSharedHandle);

        if (m_backend && texture->GetNativeHandle() == 0)
        {
            TextureDesc backendDesc = desc;
            if (autoMipChain && resolvedMipLevels == 1u)
            {
                backendDesc.mipLevels = 1u;
            }
            NativeHandle h = m_backend->CreateTexture(backendDesc, desc.initialData);
            if (h == 0)
            {
                m_lastError = "CreateTexture failed: backend CreateTexture returned 0.";
                delete texture;
                return nullptr;
            }
            texture->SetNativeHandle(h);
        }
        else if (!m_backend && desc.initialData != nullptr && pendingStoreBytes > 0u)
        {
            texture->SetPendingInitialData(desc.initialData, pendingStoreBytes);
        }

        m_textures.push_back(texture);
        return texture;
    }

    BOOL CoreTextureManager::IsManagedTexture(const Texture* texture) const
    {
        if (!texture) return false;
        return std::find(m_textures.begin(), m_textures.end(), texture) != m_textures.end();
    }

    BOOL CoreTextureManager::ResizeTexture(Texture* texture, UINT32 newWidth, UINT32 newHeight)
    {
        if (!IsManagedTexture(texture) || newWidth == 0 || newHeight == 0)
            return false;

        if (texture->GetType() == TextureType::TEXTURE_3D)
        {
            if (newWidth > 2048u || newHeight > 2048u) return false;
        }
        else
        {
            if (newWidth > 16384u || newHeight > 16384u) return false;
            if ((texture->GetType() == TextureType::TEXTURE_CUBE || texture->GetType() == TextureType::TEXTURE_CUBE_ARRAY) &&
                newWidth != newHeight)
            {
                return false;
            }
        }

        if (IsTextureFormatBlockCompressed(texture->GetFormat()) &&
            ((newWidth % 4u) != 0u || (newHeight % 4u) != 0u))
        {
            return false;
        }

        if (texture->GetWidth() == newWidth &&
            texture->GetHeight() == newHeight &&
            (!m_backend || texture->GetNativeHandle() != 0))
        {
            return true;
        }

        const UINT32 maxMips = ComputeTextureMaxMipLevels(newWidth, newHeight, texture->GetDepth(), texture->GetType());
        if (maxMips == 0u) return false;

        const BOOL autoMips = texture->UsesAutoMipChain();
        const UINT32 newMipLevels = autoMips
            ? ((IsTextureFormatBlockCompressed(texture->GetFormat()) || IsTextureFormatDepth(texture->GetFormat())) ? 1u : maxMips)
            : (std::min)(texture->GetMipLevels(), maxMips);

        if (m_backend)
        {
            TextureDesc desc{};
            desc.width             = newWidth;
            desc.height            = newHeight;
            desc.depth             = texture->GetDepth();
            desc.mipLevels         = newMipLevels;
            desc.arrayLayers       = texture->GetArrayLayers();
            desc.format            = texture->GetFormat();
            desc.type              = texture->GetType();
            desc.usage             = texture->GetUsage();
            desc.dataSource        = texture->GetDataSource();
            desc.allowSharedHandle = texture->AllowsSharedHandle();

            const NativeHandle newHandle = m_backend->CreateTexture(desc, nullptr);
            if (newHandle == 0)
            {
                return false;
            }

            if (texture->GetNativeHandle() != 0)
            {
                m_backend->DestroyTexture(texture->GetNativeHandle());
            }
            texture->SetDimensions(newWidth, newHeight, texture->GetDepth(), newMipLevels, texture->GetArrayLayers(), autoMips);
            texture->SetNativeHandle(newHandle);
            return true;
        }

        texture->SetDimensions(newWidth, newHeight, texture->GetDepth(), newMipLevels, texture->GetArrayLayers(), autoMips);
        texture->SetPendingInitialData(nullptr, 0u);
        return true;
    }

    BOOL CoreTextureManager::UpdateTextureSubresource(
        Texture* texture,
        UINT32 mipSlice,
        UINT32 arraySlice,
        const void* srcData,
        SIZE_T srcByteSize)
    {
        if (!m_backend || !IsManagedTexture(texture) || texture->GetNativeHandle() == 0 || !srcData || srcByteSize == 0u)
            return false;
        return m_backend->UpdateTextureSubresource(texture->GetNativeHandle(), mipSlice, arraySlice, srcData, srcByteSize);
    }

    BOOL CoreTextureManager::GenerateTextureMips(Texture* texture)
    {
        if (!m_backend || !IsManagedTexture(texture) || texture->GetNativeHandle() == 0)
            return false;
        return m_backend->GenerateTextureMips(texture->GetNativeHandle());
    }

    BOOL CoreTextureManager::ReadbackTextureSync(const Texture* texture, void* outData, UINT32 maxByteSize) const
    {
        if (!m_backend || !IsManagedTexture(texture) || texture->GetNativeHandle() == 0 || !outData || maxByteSize == 0u)
            return false;
        return m_backend->ReadbackTextureSync(texture->GetNativeHandle(), outData, maxByteSize);
    }

    BOOL CoreTextureManager::ReadbackTextureSubresourceSync(
        const Texture* texture,
        UINT32 mipSlice,
        UINT32 arraySlice,
        void* outData,
        UINT32 maxByteSize) const
    {
        if (!m_backend || !IsManagedTexture(texture) || texture->GetNativeHandle() == 0 || !outData || maxByteSize == 0u)
            return false;
        return m_backend->ReadbackTextureSubresourceSync(texture->GetNativeHandle(), mipSlice, arraySlice, outData, maxByteSize);
    }

    void CoreTextureManager::DestroyTexture(Texture* texture)
    {
        auto it = std::find(m_textures.begin(), m_textures.end(), texture);
        if (it != m_textures.end())
        {
            if (m_backend && texture->GetNativeHandle() != 0)
            {
                m_backend->DestroyTexture(texture->GetNativeHandle());
                texture->SetNativeHandle(0);
            }
            delete texture;
            m_textures.erase(it);
        }
    }

    SamplerState* CoreTextureManager::CreateSamplerState(const SamplerDesc& desc)
    {
        SamplerState* sampler = new SamplerState();
        sampler->SetFilter(desc.minFilter, desc.magFilter, desc.mipFilter);
        sampler->SetWrap(desc.wrapU, desc.wrapV, desc.wrapW);
        sampler->SetMaxAnisotropy(desc.maxAnisotropy);
        sampler->SetComparisonMode(desc.comparisonMode);

        if (m_backend && sampler->GetNativeHandle() == 0)
        {
            NativeHandle h = m_backend->CreateSampler(desc);
            if (h == 0)
            {
                delete sampler;
                return nullptr;
            }
            sampler->SetNativeHandle(h);
        }

        m_samplers.push_back(sampler);
        return sampler;
    }

    BOOL CoreTextureManager::IsManagedSampler(const SamplerState* sampler) const
    {
        if (!sampler) return false;
        return std::find(m_samplers.begin(), m_samplers.end(), sampler) != m_samplers.end();
    }

    void CoreTextureManager::DestroySamplerState(SamplerState* sampler)
    {
        if (!sampler) return;
        auto it = std::find(m_samplers.begin(), m_samplers.end(), sampler);
        if (it != m_samplers.end())
        {
            if (m_backend && sampler->GetNativeHandle() != 0)
            {
                m_backend->DestroySampler(sampler->GetNativeHandle());
                sampler->SetNativeHandle(0);
            }
            delete sampler;
            m_samplers.erase(it);
        }
    }

    void CoreTextureManager::RegisterWithBackend(NorthstarBackend* backend)
    {
        m_backend = backend;
        if (!m_backend) return;

        for (auto* tex : m_textures)
        {
            if (tex && tex->GetNativeHandle() == 0)
            {
                TextureDesc desc{};
                desc.width             = tex->GetWidth();
                desc.height            = tex->GetHeight();
                desc.depth             = tex->GetDepth();
                desc.mipLevels         = tex->GetMipLevels();
                desc.arrayLayers       = tex->GetArrayLayers();
                desc.format            = tex->GetFormat();
                desc.type              = tex->GetType();
                desc.usage             = tex->GetUsage();
                desc.dataSource        = tex->GetDataSource();
                desc.allowSharedHandle = tex->AllowsSharedHandle();
                desc.initialData       = tex->GetPendingInitialData();
                desc.initialDataSize   = tex->GetPendingInitialDataSize();
                NativeHandle h         = m_backend->CreateTexture(desc, desc.initialData);
                tex->SetNativeHandle(h);
            }
        }

        for (auto* samp : m_samplers)
        {
            if (samp && samp->GetNativeHandle() == 0)
            {
                SamplerDesc desc{};
                desc.minFilter      = samp->GetMinFilter();
                desc.magFilter      = samp->GetMagFilter();
                desc.mipFilter      = samp->GetMipFilter();
                desc.wrapU          = samp->GetWrapU();
                desc.wrapV          = samp->GetWrapV();
                desc.wrapW          = samp->GetWrapW();
                desc.maxAnisotropy  = samp->GetMaxAnisotropy();
                desc.comparisonMode = samp->GetComparisonMode();
                NativeHandle h      = m_backend->CreateSampler(desc);
                samp->SetNativeHandle(h);
            }
        }
    }

    void CoreTextureManager::UnregisterFromBackend(BOOL destroyNativeHandles)
    {
        if (destroyNativeHandles && m_backend)
        {
            for (auto* samp : m_samplers)
            {
                if (samp && samp->GetNativeHandle() != 0)
                {
                    m_backend->DestroySampler(samp->GetNativeHandle());
                    samp->SetNativeHandle(0);
                }
            }
            for (auto* tex : m_textures)
            {
                if (tex && tex->GetNativeHandle() != 0)
                {
                    m_backend->DestroyTexture(tex->GetNativeHandle());
                    tex->SetNativeHandle(0);
                }
            }
        }
        m_backend = nullptr;
    }

    void CoreTextureManager::Shutdown()
    {
        for (auto sampler : m_samplers)
        {
            if (m_backend && sampler->GetNativeHandle() != 0)
            {
                m_backend->DestroySampler(sampler->GetNativeHandle());
                sampler->SetNativeHandle(0);
            }
            delete sampler;
        }
        m_samplers.clear();

        for (auto texture : m_textures)
        {
            if (m_backend && texture->GetNativeHandle() != 0)
            {
                m_backend->DestroyTexture(texture->GetNativeHandle());
                texture->SetNativeHandle(0);
            }
            delete texture;
        }
        m_textures.clear();
        m_backend = nullptr;
    }
}
