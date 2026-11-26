#pragma once

#include "../shader/shader.h"
#include "../shader/shader_types.h"
#include "../types/northstar_types.h"

#include <string>

namespace Northstar
{
    struct ShaderSetDesc
    {
        // ==================== VERTEX SHADER VARIANTS ====================
        Shader* vertexShader                  = nullptr; // STANDARD variant (Default single-object VS)
        Shader* instancedVertexShader         = nullptr; // INSTANCED variant (GPU instanced VS)
        Shader* skinnedVertexShader           = nullptr; // SKINNED variant (Skeletal mesh VS)
        Shader* instancedSkinnedVertexShader  = nullptr; // INSTANCED_SKINNED variant (Crowd / Instanced Skinned VS)

        // ==================== BASE PIPELINE GRAPHICS STAGES ====================
        Shader* hullShader     = nullptr;
        Shader* domainShader   = nullptr;
        Shader* geometryShader = nullptr;
        Shader* pixelShader    = nullptr;

        // ==================== BASE PIPELINE COMPUTE STAGE ====================
        Shader* computeShader  = nullptr;

        // Optional identification / profiling name
        std::string name;

        // ==================== CONVENIENCE HELPERS ====================
        void SetShader(Shader* shader)
        {
            if (!shader) return;
            switch (shader->GetStage())
            {
            case ShaderStage::VERTEX:   SetVertexShader(shader, ShaderVariant::STANDARD); break;
            case ShaderStage::HULL:     hullShader     = shader; break;
            case ShaderStage::DOMAIN:   domainShader   = shader; break;
            case ShaderStage::GEOMETRY: geometryShader = shader; break;
            case ShaderStage::PIXEL:    pixelShader    = shader; break;
            case ShaderStage::COMPUTE:  computeShader  = shader; break;
            default: break;
            }
        }

        void SetVertexShader(Shader* shader, ShaderVariant variant = ShaderVariant::STANDARD)
        {
            switch (variant)
            {
            case ShaderVariant::STANDARD:          vertexShader                 = shader; break;
            case ShaderVariant::INSTANCED:         instancedVertexShader        = shader; break;
            case ShaderVariant::SKINNED:           skinnedVertexShader          = shader; break;
            case ShaderVariant::INSTANCED_SKINNED: instancedSkinnedVertexShader = shader; break;
            default: break;
            }
        }

        Shader* GetVertexShader(ShaderVariant variant = ShaderVariant::STANDARD) const
        {
            switch (variant)
            {
            case ShaderVariant::STANDARD:          return vertexShader;
            case ShaderVariant::INSTANCED:         return instancedVertexShader;
            case ShaderVariant::SKINNED:           return skinnedVertexShader;
            case ShaderVariant::INSTANCED_SKINNED: return instancedSkinnedVertexShader;
            default:                               return nullptr;
            }
        }

        BOOL HasVertexVariant(ShaderVariant variant) const
        {
            return GetVertexShader(variant) != nullptr;
        }

        Shader* GetShader(ShaderStage stage) const
        {
            switch (stage)
            {
            case ShaderStage::VERTEX:   return GetVertexShader(ShaderVariant::STANDARD);
            case ShaderStage::HULL:     return hullShader;
            case ShaderStage::DOMAIN:   return domainShader;
            case ShaderStage::GEOMETRY: return geometryShader;
            case ShaderStage::PIXEL:    return pixelShader;
            case ShaderStage::COMPUTE:  return computeShader;
            default:                    return nullptr;
            }
        }

        BOOL HasAnyShader() const
        {
            return vertexShader || instancedVertexShader || skinnedVertexShader ||
                   instancedSkinnedVertexShader || hullShader || domainShader ||
                   geometryShader || pixelShader || computeShader;
        }

        BOOL IsGraphics() const
        {
            return (vertexShader != nullptr || instancedVertexShader != nullptr ||
                    skinnedVertexShader != nullptr || instancedSkinnedVertexShader != nullptr) &&
                   computeShader == nullptr;
        }

        BOOL IsCompute() const
        {
            return computeShader != nullptr && vertexShader == nullptr &&
                   pixelShader == nullptr && geometryShader == nullptr &&
                   hullShader == nullptr && domainShader == nullptr;
        }

        BOOL HasTessellation() const
        {
            return hullShader != nullptr || domainShader != nullptr;
        }

        BOOL IsValid() const
        {
            // Graphics pipeline requires at least a Vertex Shader.
            // If tessellation is enabled, both Hull and Domain shaders are required.
            if (IsGraphics())
            {
                if (hullShader && !domainShader) return false;
                if (!hullShader && domainShader) return false;
                return true;
            }
            // Compute pipeline requires Compute Shader and no graphics stages.
            if (IsCompute())
            {
                return true;
            }
            return false;
        }
    };
}
