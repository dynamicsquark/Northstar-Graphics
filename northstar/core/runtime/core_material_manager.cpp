#include "core_material_manager.h"
#include "core_texture_manager.h"
#include "core_shader_manager.h"
#include <algorithm>

namespace Northstar
{
    CoreMaterialManager::~CoreMaterialManager()
    {
        Shutdown();
    }

    Material* CoreMaterialManager::CreateMaterial(const MaterialDesc& desc)
    {
        Material* material = new Material();
        if (m_nextMaterialId == 0u) m_nextMaterialId = 1u;
        material->SetGPUSceneMaterialId(m_nextMaterialId++);
        if (m_nextMaterialId == 0u) m_nextMaterialId = 1u;
        material->SetCullMode(desc.cullMode);
        material->SetBlendMode(desc.blendMode);
        material->SetFillMode(desc.fillMode);
        material->SetDepthState(desc.depthState);

        if (desc.geometryShaderSet)
            material->SetGeometryShaderSet(desc.geometryShaderSet);
        if (desc.shadowShaderSet)
            material->SetShadowShaderSet(desc.shadowShaderSet);
        if (desc.depthShaderSet)
            material->SetDepthShaderSet(desc.depthShaderSet);
        if (desc.computeKernel)
            material->SetComputeKernel(desc.computeKernel);

        m_materials.push_back(material);
        return material;
    }

    Material* CoreMaterialManager::CloneMaterial(const Material* source)
    {
        if (!source || !IsManagedMaterial(source))
        {
            return nullptr;
        }

        Material* cloned = new Material(*source);
        if (m_nextMaterialId == 0u) m_nextMaterialId = 1u;
        cloned->SetGPUSceneMaterialId(m_nextMaterialId++);
        if (m_nextMaterialId == 0u) m_nextMaterialId = 1u;

        m_materials.push_back(cloned);
        return cloned;
    }

    BOOL CoreMaterialManager::IsManagedMaterial(const Material* material) const
    {
        if (!material) return false;
        return std::find(m_materials.begin(), m_materials.end(), material) != m_materials.end();
    }

    void CoreMaterialManager::DestroyMaterial(Material* material)
    {
        if (!material) return;
        auto it = std::find(m_materials.begin(), m_materials.end(), material);
        if (it != m_materials.end())
        {
            delete material;
            m_materials.erase(it);
        }
    }

    void CoreMaterialManager::OnComputeKernelDestroyed(ComputeKernel* destroyedKernel)
    {
        if (!destroyedKernel) return;
        for (Material* mat : m_materials)
        {
            if (mat)
            {
                mat->RemoveComputeKernel(destroyedKernel);
            }
        }
    }

    void CoreMaterialManager::OnShaderSetDestroyed(ShaderSet* destroyedShaderSet)
    {
        if (!destroyedShaderSet) return;
        for (Material* mat : m_materials)
        {
            if (mat)
            {
                mat->OnShaderSetDestroyed(destroyedShaderSet);
            }
        }
    }

    void CoreMaterialManager::OnTextureDestroyed(Texture* destroyedTexture)
    {
        if (!destroyedTexture) return;
        for (Material* mat : m_materials)
        {
            if (mat)
            {
                mat->OnTextureDestroyed(destroyedTexture);
            }
        }
    }

    void CoreMaterialManager::OnSamplerDestroyed(SamplerState* destroyedSampler)
    {
        if (!destroyedSampler) return;
        for (Material* mat : m_materials)
        {
            if (mat)
            {
                mat->OnSamplerDestroyed(destroyedSampler);
            }
        }
    }

    void CoreMaterialManager::OnGPUBufferDestroyed(GPUBuffer* destroyedBuffer)
    {
        if (!destroyedBuffer) return;
        for (Material* mat : m_materials)
        {
            if (mat)
            {
                mat->OnGPUBufferDestroyed(destroyedBuffer);
            }
        }
    }

    void CoreMaterialManager::Shutdown()
    {
        for (auto material : m_materials) delete material;
        m_materials.clear();
    }
}
