#pragma once

#include "material/material.h"
#include "material/material_desc.h"
#include <vector>

namespace Northstar
{
    class CoreMaterialManager
    {
    public:
        CoreMaterialManager() = default;
        ~CoreMaterialManager();

        Material*      CreateMaterial(const MaterialDesc& desc);
        Material*      CloneMaterial(const Material* source);
        void           DestroyMaterial(Material* material);
        BOOL           IsManagedMaterial(const Material* material) const;

        void           OnComputeKernelDestroyed(ComputeKernel* destroyedKernel);
        void           OnShaderSetDestroyed(ShaderSet* destroyedShaderSet);
        void           OnTextureDestroyed(Texture* destroyedTexture);
        void           OnSamplerDestroyed(SamplerState* destroyedSampler);
        void           OnGPUBufferDestroyed(GPUBuffer* destroyedBuffer);

        void           Shutdown();

    private:
        std::vector<Material*> m_materials;
        UINT32                 m_nextMaterialId = 1;
    };
}
