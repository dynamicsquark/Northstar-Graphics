#include "core_shader_manager.h"
#include "../shader_loader/shader_loader.h"
#include "platform/northstar_backend.h"

#include <algorithm>
#include <string>

namespace Northstar
{
    CoreShaderManager::~CoreShaderManager()
    {
        Shutdown();
    }

    BOOL CoreShaderManager::RegisterSingleShaderWithBackend(Shader& shader)
    {
        if (!m_backend)
        {
            return true;
        }

        if (shader.GetNativeHandle() == 0)
        {
            const BOOL isHLSL12Backend = (m_backend->GetTargetAPI() == TargetAPI::HLSL12);
            if (!shader.HasBytecode() && shader.HasSourceCode() && isHLSL12Backend)
            {
                ShaderLoader::TryPopulateCachedDXBC(shader);
            }

            if (shader.HasBytecode())
            {
                NativeHandle h = m_backend->CreateShader(
                    shader.GetStage(),
                    shader.GetBytecodeData(),
                    shader.GetBytecodeSize()
                );
                shader.SetNativeHandle(h);
            }

            if (shader.GetNativeHandle() == 0 && shader.HasSourceCode())
            {
                NativeHandle h = m_backend->CreateShader(
                    shader.GetStage(),
                    shader.GetSourceCode().data(),
                    shader.GetSourceCode().size()
                );
                shader.SetNativeHandle(h);
                if (h != 0 && isHLSL12Backend)
                {
                    std::vector<BYTE> compiledDxbc;
                    if (m_backend->GetShaderCompiledBytecode(h, compiledDxbc) && !compiledDxbc.empty())
                    {
                        ShaderLoader::StoreCachedDXBC(shader, compiledDxbc.data(), compiledDxbc.size());
                    }
                }
            }

            if (m_backend->IsHardwareBackend() &&
                (!shader.IsValid() || ((shader.HasBytecode() || shader.HasSourceCode()) && shader.GetNativeHandle() == 0)))
            {
                m_lastError = "CreateShader failed: hardware backend rejected empty or invalid shader payload.";
                return false;
            }
        }

        if (shader.GetNativeHandle() != 0 && !shader.GetInputLayout().empty())
        {
            m_backend->SetShaderInputLayout(shader.GetNativeHandle(), shader.GetInputLayout());
        }
        return true;
    }

    Shader* CoreShaderManager::CreateShader(const ShaderDesc& desc)
    {
        m_lastError.clear();
        Shader* shader = new Shader();

        if (desc.initialData != nullptr || desc.preloadedReflection != nullptr)
        {
            if (!ShaderLoader::Load(desc, *shader))
            {
                m_lastError = "CreateShader failed: invalid or unreadable shader descriptor / .nsc payload.";
                delete shader;
                return nullptr;
            }
        }
        else if (!desc.name.empty())
        {
            shader->SetName(desc.name);
        }

        if (!m_backend && !shader->HasBytecode() && shader->HasSourceCode() &&
            shader->GetReflection().api == TargetAPI::HLSL12)
        {
            ShaderLoader::TryPopulateCachedDXBC(*shader);
        }

        if (m_backend)
        {
            if (!RegisterSingleShaderWithBackend(*shader))
            {
                delete shader;
                return nullptr;
            }
        }

        m_shaders.push_back(shader);
        return shader;
    }

    BOOL CoreShaderManager::IsManagedShader(const Shader* shader) const
    {
        if (!shader) return false;
        return std::find(m_shaders.begin(), m_shaders.end(), shader) != m_shaders.end();
    }

    void CoreShaderManager::DestroyShader(Shader* shader)
    {
        if (!shader) return;
        auto it = std::find(m_shaders.begin(), m_shaders.end(), shader);
        if (it != m_shaders.end())
        {
            for (ShaderSet* set : m_shaderSets)
            {
                if (set)
                {
                    set->OnShaderDestroyed(shader);
                }
            }

            if (m_backend && shader->GetNativeHandle() != 0)
            {
                m_backend->DestroyShader(shader->GetNativeHandle());
                shader->SetNativeHandle(0);
            }

            delete shader;
            m_shaders.erase(it);
        }
    }

    ShaderSet* CoreShaderManager::CreateShaderSet(const ShaderSetDesc& desc)
    {
        m_lastError.clear();

        auto checkStage = [&](const Shader* s, ShaderStage expected, const char* slotLabel) -> BOOL {
            if (s != nullptr && s->GetStage() != expected)
            {
                m_lastError = std::string("CreateShaderSet failed: ") + slotLabel + " stage mismatch.";
                return false;
            }
            return true;
        };

        if (!checkStage(desc.vertexShader,                 ShaderStage::VERTEX,   "vertexShader") ||
            !checkStage(desc.instancedVertexShader,        ShaderStage::VERTEX,   "instancedVertexShader") ||
            !checkStage(desc.skinnedVertexShader,          ShaderStage::VERTEX,   "skinnedVertexShader") ||
            !checkStage(desc.instancedSkinnedVertexShader, ShaderStage::VERTEX,   "instancedSkinnedVertexShader") ||
            !checkStage(desc.hullShader,                   ShaderStage::HULL,     "hullShader") ||
            !checkStage(desc.domainShader,                 ShaderStage::DOMAIN,   "domainShader") ||
            !checkStage(desc.geometryShader,               ShaderStage::GEOMETRY, "geometryShader") ||
            !checkStage(desc.pixelShader,                  ShaderStage::PIXEL,    "pixelShader") ||
            !checkStage(desc.computeShader,                ShaderStage::COMPUTE,  "computeShader"))
        {
            return nullptr;
        }

        const BOOL hasAnyGraphics =
            (desc.vertexShader != nullptr) ||
            (desc.instancedVertexShader != nullptr) ||
            (desc.skinnedVertexShader != nullptr) ||
            (desc.instancedSkinnedVertexShader != nullptr) ||
            (desc.hullShader != nullptr) ||
            (desc.domainShader != nullptr) ||
            (desc.geometryShader != nullptr) ||
            (desc.pixelShader != nullptr);
        if (hasAnyGraphics && desc.computeShader != nullptr)
        {
            m_lastError = "CreateShaderSet failed: cannot mix graphics stages and computeShader in the same ShaderSet.";
            return nullptr;
        }

        ShaderSet* set = new ShaderSet();
        if (!desc.name.empty())
        {
            set->SetName(desc.name);
        }

        if (desc.vertexShader)                 set->SetVertexShader(desc.vertexShader, ShaderVariant::STANDARD);
        if (desc.instancedVertexShader)        set->SetVertexShader(desc.instancedVertexShader, ShaderVariant::INSTANCED);
        if (desc.skinnedVertexShader)          set->SetVertexShader(desc.skinnedVertexShader, ShaderVariant::SKINNED);
        if (desc.instancedSkinnedVertexShader) set->SetVertexShader(desc.instancedSkinnedVertexShader, ShaderVariant::INSTANCED_SKINNED);

        if (desc.hullShader)     set->SetShader(ShaderStage::HULL,     desc.hullShader);
        if (desc.domainShader)   set->SetShader(ShaderStage::DOMAIN,   desc.domainShader);
        if (desc.geometryShader) set->SetShader(ShaderStage::GEOMETRY, desc.geometryShader);
        if (desc.pixelShader)    set->SetShader(ShaderStage::PIXEL,    desc.pixelShader);
        if (desc.computeShader)  set->SetShader(ShaderStage::COMPUTE,  desc.computeShader);

        m_shaderSets.push_back(set);
        return set;
    }

    BOOL CoreShaderManager::IsManagedShaderSet(const ShaderSet* shaderSet) const
    {
        if (!shaderSet) return false;
        return std::find(m_shaderSets.begin(), m_shaderSets.end(), shaderSet) != m_shaderSets.end();
    }

    void CoreShaderManager::DestroyShaderSet(ShaderSet* shaderSet)
    {
        if (!shaderSet) return;
        auto it = std::find(m_shaderSets.begin(), m_shaderSets.end(), shaderSet);
        if (it != m_shaderSets.end())
        {
            delete shaderSet;
            m_shaderSets.erase(it);
        }
    }

    void CoreShaderManager::RegisterShadersWithBackend(NorthstarBackend* backend)
    {
        m_backend = backend;
        if (!m_backend) return;

        for (auto shader : m_shaders)
        {
            if (shader)
            {
                RegisterSingleShaderWithBackend(*shader);
            }
        }
    }

    void CoreShaderManager::UnregisterFromBackend(BOOL destroyNativeHandles)
    {
        if (destroyNativeHandles && m_backend)
        {
            for (auto* shader : m_shaders)
            {
                if (shader && shader->GetNativeHandle() != 0)
                {
                    m_backend->DestroyShader(shader->GetNativeHandle());
                    shader->SetNativeHandle(0);
                }
            }
        }
        m_backend = nullptr;
    }

    void CoreShaderManager::Shutdown()
    {
        for (auto set : m_shaderSets) delete set;
        m_shaderSets.clear();

        for (auto shader : m_shaders)
        {
            if (m_backend && shader->GetNativeHandle() != 0)
            {
                m_backend->DestroyShader(shader->GetNativeHandle());
                shader->SetNativeHandle(0);
            }
            delete shader;
        }
        m_shaders.clear();
        m_backend = nullptr;
    }
}
