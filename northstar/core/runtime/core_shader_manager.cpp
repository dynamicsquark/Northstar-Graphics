#include "core_shader_manager.h"
#include "../shader_loader/shader_loader.h"
#include "platform/northstar_backend.h"

#include <algorithm>
#include <string>

namespace Northstar
{
    namespace
    {
        ShaderSpaceMask BuildShaderSpaceMaskFromReflection(const Shader& shader) noexcept
        {
            const ShaderReflection& refl = shader.GetReflection();
            if (refl.nscVersion == 0)
            {
                return ShaderSpaceMask{};
            }

            ShaderSpaceMask mask{};
            mask.cbvSrvMask  = (shader.GetStage() == ShaderStage::VERTEX) ? 0x0001u : 0x0000u;
            mask.samplerMask = 0x0000u;
            mask.uavMask     = 0x0000u;
            mask.isExplicit  = true;

            for (const auto& b : refl.buffers)
            {
                if (b.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << b.set));
            }
            for (const auto& sb : refl.structuredBuffers)
            {
                if (refl.FindUAV(sb.name) != nullptr) continue;
                if (sb.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << sb.set));
            }
            for (const auto& t : refl.textures)
            {
                if (t.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << t.set));
            }
            for (const auto& sm : refl.samplers)
            {
                if (sm.set < 16u) mask.samplerMask = static_cast<UINT16>(mask.samplerMask | (1u << sm.set));
            }
            for (const auto& uav : refl.uavBuffers)
            {
                if (uav.set < 16u) mask.uavMask = static_cast<UINT16>(mask.uavMask | (1u << uav.set));
                for (const auto& f : uav.fields)
                {
                    if (f.set < 16u && f.binding != 0xFFFFu)
                        mask.uavMask = static_cast<UINT16>(mask.uavMask | (1u << f.set));
                }
            }
            return mask;
        }

        UINT32 SelectTargetShaderModel(const NorthstarBackend* backend) noexcept
        {
            if (!backend || backend->GetTargetAPI() != TargetAPI::HLSL12)
                return 0x50u;
            const BackendCapabilities caps = backend->GetCapabilities();
            return (caps.maxShaderModel >= 0x66u && caps.supportsDXILRuntime) ? 0x66u : 0x51u;
        }
    }

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
            const ShaderReflection& shaderRefl = shader.GetReflection();
            if (m_backend->IsHardwareBackend())
            {
                const BackendCapabilities caps = m_backend->GetCapabilities();
                if (shaderRefl.UsesInt64Atomic() && !caps.SupportsNVGRasterAtomic64())
                {
                    m_lastError = "CreateShader failed: shader requires Shader Model 6.6 DXIL, Int64ShaderOps, and 64-bit descriptor-heap atomics.";
                    return false;
                }
                if (shaderRefl.UsesInt64() && !(caps.maxShaderModel >= 0x60u && caps.supportsDXILRuntime && caps.supportsInt64ShaderOps))
                {
                    m_lastError = "CreateShader failed: shader requires 64-bit integer shader operations (Int64ShaderOps + DXIL).";
                    return false;
                }
            }

            const BOOL   isHLSL12Backend = (m_backend->GetTargetAPI() == TargetAPI::HLSL12);
            const UINT32 targetSM        = SelectTargetShaderModel(m_backend);
            if (!shader.HasBytecode() && shader.HasSourceCode() && isHLSL12Backend)
            {
                ShaderLoader::TryPopulateCachedDXBC(shader, targetSM);
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
                        ShaderLoader::StoreCachedDXBC(shader, compiledDxbc.data(), compiledDxbc.size(), targetSM);
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

        if (shader.GetNativeHandle() != 0)
        {
            if (!shader.GetInputLayout().empty())
            {
                m_backend->SetShaderInputLayout(shader.GetNativeHandle(), shader.GetInputLayout());
            }
            const ShaderSpaceMask spaceMask = BuildShaderSpaceMaskFromReflection(shader);
            if (spaceMask.isExplicit)
            {
                m_backend->SetShaderReflectionSpaces(shader.GetNativeHandle(), spaceMask);
            }
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

        const BOOL hasAnyVertex =
            (desc.vertexShader != nullptr) ||
            (desc.instancedVertexShader != nullptr) ||
            (desc.skinnedVertexShader != nullptr) ||
            (desc.instancedSkinnedVertexShader != nullptr);

        const BOOL hasAnyGraphics =
            hasAnyVertex ||
            (desc.hullShader != nullptr) ||
            (desc.domainShader != nullptr) ||
            (desc.geometryShader != nullptr) ||
            (desc.pixelShader != nullptr);

        if (!hasAnyGraphics && desc.computeShader == nullptr)
        {
            m_lastError = "CreateShaderSet failed: empty ShaderSetDesc contains no shader stages.";
            return nullptr;
        }
        if (hasAnyGraphics && desc.computeShader != nullptr)
        {
            m_lastError = "CreateShaderSet failed: cannot mix graphics stages and computeShader in the same ShaderSet.";
            return nullptr;
        }
        if (hasAnyGraphics && !hasAnyVertex)
        {
            m_lastError = "CreateShaderSet failed: graphics ShaderSet requires at least one vertex shader variant.";
            return nullptr;
        }
        if ((desc.hullShader != nullptr) != (desc.domainShader != nullptr))
        {
            m_lastError = "CreateShaderSet failed: tessellation requires both hullShader and domainShader.";
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
