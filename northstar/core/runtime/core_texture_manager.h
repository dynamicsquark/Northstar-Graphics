#pragma once

#include "texture/texture.h"
#include "texture/texture_desc.h"
#include "sampler/sampler.h"
#include "sampler/sampler_desc.h"
#include "sampler/sampler_types.h"
#include "platform/northstar_backend.h"
#include <string>
#include <vector>

namespace Northstar
{
    class CoreTextureManager
    {
    public:
        CoreTextureManager() = default;
        ~CoreTextureManager();

        const std::string& GetLastError() const { return m_lastError; }

        Texture*      CreateTexture(const TextureDesc& desc);
        BOOL          ResizeTexture(Texture* texture, UINT32 newWidth, UINT32 newHeight);
        BOOL          UpdateTextureSubresource(Texture* texture, UINT32 mipSlice, UINT32 arraySlice, const void* srcData, SIZE_T srcByteSize);
        BOOL          GenerateTextureMips(Texture* texture);
        BOOL          ReadbackTextureSync(const Texture* texture, void* outData, UINT32 maxByteSize) const;
        BOOL          ReadbackTextureSubresourceSync(const Texture* texture, UINT32 mipSlice, UINT32 arraySlice, void* outData, UINT32 maxByteSize) const;
        void          DestroyTexture(Texture* texture);
        BOOL          IsManagedTexture(const Texture* texture) const;

        SamplerState* CreateSamplerState(const SamplerDesc& desc);
        void          DestroySamplerState(SamplerState* sampler);
        BOOL          IsManagedSampler(const SamplerState* sampler) const;

        void          RegisterWithBackend(NorthstarBackend* backend);
        void          UnregisterFromBackend(BOOL destroyNativeHandles = false);
        void          Shutdown();

    private:
        std::vector<Texture*>      m_textures;
        std::vector<SamplerState*> m_samplers;
        NorthstarBackend*          m_backend = nullptr;
        std::string                m_lastError;
    };
}
