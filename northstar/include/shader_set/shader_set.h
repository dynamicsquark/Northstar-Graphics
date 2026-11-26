#pragma once

#include "shader_set_desc.h"
#include "../shader/shader.h"
#include "../shader/shader_types.h"
#include "../types/northstar_types.h"

#include <string>

namespace Northstar
{
    class ShaderSet
    {
    public:
        // ==================== STAGE SETTERS ====================
        // Automatically assigns shader to its appropriate stage slot based on shader->GetStage()
        void SetShader(Shader* shader)
        {
            if (!shader) return;
            const ShaderStage stage = shader->GetStage();
            if (stage == ShaderStage::COMPUTE)
            {
                m_computeShader = shader;
                ++m_stateRevision;
            }
            else if (stage == ShaderStage::VERTEX)
            {
                SetVertexShader(shader, ShaderVariant::STANDARD);
            }
            else
            {
                const ShaderSlot slot = ShaderStageToSlot(stage);
                if (slot < ShaderSlot::COUNT)
                {
                    m_graphicsShaders[static_cast<SIZE_T>(slot)] = shader;
                    ++m_stateRevision;
                }
            }
        }

        void SetShader(ShaderStage stage, Shader* shader)
        {
            if (shader != nullptr && shader->GetStage() != stage)
            {
                return;
            }
            if (stage == ShaderStage::COMPUTE)
            {
                m_computeShader = shader;
                ++m_stateRevision;
            }
            else if (stage == ShaderStage::VERTEX)
            {
                SetVertexShader(shader, ShaderVariant::STANDARD);
            }
            else
            {
                const ShaderSlot slot = ShaderStageToSlot(stage);
                if (slot < ShaderSlot::COUNT)
                {
                    m_graphicsShaders[static_cast<SIZE_T>(slot)] = shader;
                    ++m_stateRevision;
                }
            }
        }

        void RemoveShader(ShaderStage stage)
        {
            if (stage == ShaderStage::COMPUTE)
            {
                m_computeShader = nullptr;
                ++m_stateRevision;
            }
            else if (stage == ShaderStage::VERTEX)
            {
                for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderVariant::COUNT); ++i)
                {
                    m_vertexVariants[i] = nullptr;
                }
                m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::VERTEX)] = nullptr;
                ++m_stateRevision;
            }
            else
            {
                const ShaderSlot slot = ShaderStageToSlot(stage);
                if (slot < ShaderSlot::COUNT)
                {
                    m_graphicsShaders[static_cast<SIZE_T>(slot)] = nullptr;
                    ++m_stateRevision;
                }
            }
        }

        void Clear()
        {
            for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderSlot::COUNT); ++i)
            {
                m_graphicsShaders[i] = nullptr;
            }
            for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderVariant::COUNT); ++i)
            {
                m_vertexVariants[i] = nullptr;
            }
            m_computeShader = nullptr;
            ++m_stateRevision;
        }

        // ==================== VERTEX SHADER VARIANTS ====================
        void SetVertexShader(Shader* shader, ShaderVariant variant = ShaderVariant::STANDARD)
        {
            if (shader != nullptr && shader->GetStage() != ShaderStage::VERTEX)
            {
                return;
            }
            const SIZE_T idx = static_cast<SIZE_T>(variant);
            if (idx < static_cast<SIZE_T>(ShaderVariant::COUNT))
            {
                m_vertexVariants[idx] = shader;
            }
            if (variant == ShaderVariant::STANDARD)
            {
                m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::VERTEX)] = shader;
            }
            ++m_stateRevision;
        }

        void SetInstancedVertexShader(Shader* shader)         { SetVertexShader(shader, ShaderVariant::INSTANCED); }
        void SetSkinnedVertexShader(Shader* shader)           { SetVertexShader(shader, ShaderVariant::SKINNED); }
        void SetInstancedSkinnedVertexShader(Shader* shader)  { SetVertexShader(shader, ShaderVariant::INSTANCED_SKINNED); }

        void SetHullShader(Shader* shader)     { if (shader && shader->GetStage() != ShaderStage::HULL)     return; m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::HULL)]     = shader; ++m_stateRevision; }
        void SetDomainShader(Shader* shader)   { if (shader && shader->GetStage() != ShaderStage::DOMAIN)   return; m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::DOMAIN_STAGE)] = shader; ++m_stateRevision; }
        void SetGeometryShader(Shader* shader) { if (shader && shader->GetStage() != ShaderStage::GEOMETRY) return; m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::GEOMETRY)] = shader; ++m_stateRevision; }
        void SetPixelShader(Shader* shader)    { if (shader && shader->GetStage() != ShaderStage::PIXEL)    return; m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::PIXEL)]    = shader; ++m_stateRevision; }
        void SetComputeShader(Shader* shader)  { if (shader && shader->GetStage() != ShaderStage::COMPUTE)  return; m_computeShader = shader; ++m_stateRevision; }

        // ==================== STAGE GETTERS ====================
        Shader* GetShader(ShaderStage stage) const
        {
            if (stage == ShaderStage::COMPUTE)
            {
                return m_computeShader;
            }
            if (stage == ShaderStage::VERTEX)
            {
                Shader* s = GetVertexShader(ShaderVariant::STANDARD);
                if (s && s->IsValid()) return s;
                for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderVariant::COUNT); ++i)
                {
                    Shader* v = m_vertexVariants[i];
                    if (v && v->IsValid()) return v;
                }
                return s;
            }
            const ShaderSlot slot = ShaderStageToSlot(stage);
            if (slot < ShaderSlot::COUNT)
            {
                return m_graphicsShaders[static_cast<SIZE_T>(slot)];
            }
            return nullptr;
        }

        Shader* GetShader(ShaderSlot slot) const
        {
            if (slot == ShaderSlot::VERTEX)
            {
                return GetShader(ShaderStage::VERTEX);
            }
            if (slot < ShaderSlot::COUNT)
            {
                return m_graphicsShaders[static_cast<SIZE_T>(slot)];
            }
            return nullptr;
        }

        // Vertex variant getters
        Shader* GetVertexShader(ShaderVariant variant = ShaderVariant::STANDARD) const
        {
            const SIZE_T idx = static_cast<SIZE_T>(variant);
            if (idx < static_cast<SIZE_T>(ShaderVariant::COUNT))
            {
                return m_vertexVariants[idx];
            }
            return nullptr;
        }

        BOOL HasVertexVariant(ShaderVariant variant) const
        {
            Shader* s = GetVertexShader(variant);
            return s != nullptr && s->IsValid();
        }

        Shader* GetInstancedVertexShader()        const { return GetVertexShader(ShaderVariant::INSTANCED); }
        Shader* GetSkinnedVertexShader()          const { return GetVertexShader(ShaderVariant::SKINNED); }
        Shader* GetInstancedSkinnedVertexShader() const { return GetVertexShader(ShaderVariant::INSTANCED_SKINNED); }

        Shader* GetHullShader()     const { return m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::HULL)]; }
        Shader* GetDomainShader()   const { return m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::DOMAIN_STAGE)]; }
        Shader* GetGeometryShader() const { return m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::GEOMETRY)]; }
        Shader* GetPixelShader()    const { return m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::PIXEL)]; }
        Shader* GetComputeShader()  const { return m_computeShader; }

        BOOL HasShader(ShaderStage stage) const
        {
            Shader* s = GetShader(stage);
            return s != nullptr && s->IsValid();
        }

        // ==================== PIPELINE PROPERTIES ====================
        BOOL IsGraphics() const
        {
            return (HasVertexVariant(ShaderVariant::STANDARD) ||
                    HasVertexVariant(ShaderVariant::INSTANCED) ||
                    HasVertexVariant(ShaderVariant::SKINNED) ||
                    HasVertexVariant(ShaderVariant::INSTANCED_SKINNED)) &&
                   m_computeShader == nullptr;
        }

        BOOL IsCompute() const
        {
            return m_computeShader != nullptr &&
                   GetVertexShader() == nullptr &&
                   GetPixelShader() == nullptr &&
                   GetGeometryShader() == nullptr &&
                   GetHullShader() == nullptr &&
                   GetDomainShader() == nullptr;
        }

        BOOL HasTessellation() const
        {
            return GetHullShader() != nullptr || GetDomainShader() != nullptr;
        }

        BOOL IsValid() const
        {
            if (IsGraphics())
            {
                // Tessellation requires both hull and domain shaders
                if (GetHullShader() && !GetDomainShader()) return false;
                if (!GetHullShader() && GetDomainShader()) return false;
                return true;
            }
            if (IsCompute())
            {
                return true;
            }
            return false;
        }

        // ==================== METADATA & NATIVE HANDLE ====================
        NativeHandle       GetNativePipelineHandle() const { return m_nativePipelineHandle; }
        UINT64             GetStateRevision()        const { return m_stateRevision; }
        const std::string& GetName()                 const { return m_name; }

        void SetName(const std::string& name)             { m_name = name; }
        void BumpStateRevision()                          { ++m_stateRevision; }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreShaderManager;

        void SetNativePipelineHandle(NativeHandle handle) { m_nativePipelineHandle = handle; ++m_stateRevision; }

        void OnShaderDestroyed(Shader* destroyedShader)
        {
            if (!destroyedShader) return;
            bool changed = false;
            for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderSlot::COUNT); ++i)
            {
                if (m_graphicsShaders[i] == destroyedShader)
                {
                    m_graphicsShaders[i] = nullptr;
                    changed = true;
                }
            }
            for (SIZE_T i = 0; i < static_cast<SIZE_T>(ShaderVariant::COUNT); ++i)
            {
                if (m_vertexVariants[i] == destroyedShader)
                {
                    m_vertexVariants[i] = nullptr;
                    changed = true;
                }
            }
            if (m_computeShader == destroyedShader)
            {
                m_computeShader = nullptr;
                changed = true;
            }
            if (changed)
            {
                ++m_stateRevision;
            }
        }

        ShaderSet() = default;

        // Base pipeline slots for Graphics: Vertex, Hull, Domain, Geometry, Pixel
        Shader*      m_graphicsShaders[static_cast<SIZE_T>(ShaderSlot::COUNT)] = {};
        // Vertex Shader drawing mode variants (Standard, Instanced, Skinned, Instanced-Skinned)
        Shader*      m_vertexVariants[static_cast<SIZE_T>(ShaderVariant::COUNT)] = {};
        // Compute pipeline slot
        Shader*      m_computeShader                                            = nullptr;

        NativeHandle m_nativePipelineHandle = 0;
        UINT64       m_stateRevision        = 1u;
        std::string  m_name;
    };
}
