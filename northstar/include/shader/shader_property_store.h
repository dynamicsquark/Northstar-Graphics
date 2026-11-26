#pragma once

#include "shader_types.h"
#include "../types/northstar_types.h"
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include "../texture/texture.h"
#include "../sampler/sampler.h"
#include "../buffer/gpu_buffer.h"

namespace Northstar
{

    // ==================== PER-RESOURCE SLOTS (ADD only) ====================
    // Object-independent, backend-agnostic shader property storage.
    // Shared across Material, ComputeKernel, and CoreRenderPipeline.

    struct ShaderBufferSlot
    {
        std::string       propertyName;
        std::string       resourceName;
        UINT32            byteSize = 0;
        UINT16            slotIndex = 0xFFFF;
        UINT16            registerSpace = 0;
        std::vector<BYTE> cpuData;
        BOOL              dirty = false;
    };

    struct ShaderTextureSlot
    {
        std::string  propertyName;
        std::string  groupName;
        std::string  resourceName;
        UINT16       slotIndex = 0xFFFF;
        UINT16       registerSpace = 0;
        Texture*     texture = nullptr;
        NativeHandle textureHandle = 0;
        UINT32       mipSlice = 0xFFFFFFFFu;
        BOOL         hasDestroyedResource = false;
    };

    struct ShaderRWTextureSlot
    {
        std::string       propertyName;
        std::string       groupName;
        std::string       resourceName;
        UINT16            slotIndex = 0xFFFF;
        UINT16            registerSpace = 0;
        Texture*          texture = nullptr;
        NativeHandle      textureHandle = 0;
        UINT32            mipSlice = 0u;
        GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        BOOL              hasDestroyedResource = false;
    };

    struct ShaderSamplerSlot
    {
        std::string   propertyName;
        std::string   groupName;
        std::string   resourceName;
        UINT16        slotIndex = 0xFFFF;
        UINT16        registerSpace = 0;
        BOOL          isComparison = false;
        SamplerState* sampler = nullptr;
        NativeHandle  samplerHandle = 0;
        BOOL          hasDestroyedResource = false;
    };

    struct ShaderStructuredBufferSlot
    {
        std::string       propertyName;
        std::string       resourceName;
        UINT16            slotIndex = 0xFFFF;
        UINT16            registerSpace = 0;
        UINT32            stride = 0;
        UINT32            elementCount = 0;
        std::vector<BYTE> cpuData;
        BOOL              dirty = false;
        BOOL              isUAV = false;

        // User-owned element management
        UINT32            userElementCount = 0;
        std::vector<BYTE> userData;

        // Persistent GPU-resident buffer binding (optional override for CPU upload ring)
        GPUBuffer*        gpuBuffer             = nullptr;
        NativeHandle      gpuBufferHandle       = 0;
        UINT32            gpuBufferElementCount = 0;
        UINT32            gpuBufferStride       = 0;
        UINT32            gpuBufferByteSize     = 0;
        GPUResourceAccess uavAccess             = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        BOOL              hasDestroyedResource  = false;
    };

    // ==================== PER-SHADER STAGE PROPERTY STORE ====================
    struct ShaderStageProps
    {
        std::vector<ShaderBufferSlot>           buffers;
        std::vector<ShaderTextureSlot>          textures;
        std::vector<ShaderRWTextureSlot>        rwTextures;
        std::vector<ShaderSamplerSlot>          samplers;
        std::vector<ShaderStructuredBufferSlot> structuredBuffers;

        void Clear()
        {
            buffers.clear();
            textures.clear();
            rwTextures.clear();
            samplers.clear();
            structuredBuffers.clear();
        }

        BOOL HasDestroyedResource() const
        {
            for (const auto& t : textures)
                if (t.hasDestroyedResource) return true;
            for (const auto& rwt : rwTextures)
                if (rwt.hasDestroyedResource) return true;
            for (const auto& s : samplers)
                if (s.hasDestroyedResource) return true;
            for (const auto& sb : structuredBuffers)
                if (sb.hasDestroyedResource) return true;
            return false;
        }

        BOOL OnTextureDestroyed(Texture* destroyedTex)
        {
            if (!destroyedTex) return false;
            BOOL modified = false;
            for (auto& t : textures)
            {
                if (t.texture == destroyedTex)
                {
                    t.texture              = nullptr;
                    t.textureHandle        = 0;
                    t.hasDestroyedResource = true;
                    modified               = true;
                }
            }
            for (auto& rwt : rwTextures)
            {
                if (rwt.texture == destroyedTex)
                {
                    rwt.texture              = nullptr;
                    rwt.textureHandle        = 0;
                    rwt.hasDestroyedResource = true;
                    modified                 = true;
                }
            }
            return modified;
        }

        BOOL OnSamplerDestroyed(SamplerState* destroyedSamp)
        {
            if (!destroyedSamp) return false;
            BOOL modified = false;
            for (auto& s : samplers)
            {
                if (s.sampler == destroyedSamp)
                {
                    s.sampler              = nullptr;
                    s.samplerHandle        = 0;
                    s.hasDestroyedResource = true;
                    modified               = true;
                }
            }
            return modified;
        }

        BOOL OnGPUBufferDestroyed(GPUBuffer* destroyedBuf)
        {
            if (!destroyedBuf) return false;
            BOOL modified = false;
            for (auto& sb : structuredBuffers)
            {
                if (sb.gpuBuffer == destroyedBuf)
                {
                    sb.gpuBuffer             = nullptr;
                    sb.gpuBufferHandle       = 0;
                    sb.gpuBufferElementCount = 0;
                    sb.gpuBufferStride       = 0;
                    sb.gpuBufferByteSize     = 0;
                    sb.hasDestroyedResource  = true;
                    modified                 = true;
                }
            }
            return modified;
        }
    };

    // ==================== SHARED REFLECTION SLOT POPULATION ====================
    inline const NSL::NSCTexture* FindReflectionTextureInShader(const ShaderReflection& r, const std::string& name)
    {
        for (const auto& t : r.textures)
            if (t.name == name || t.groupQualifiedName == name) return &t;
        return nullptr;
    }

    inline const NSL::NSCSampler* FindReflectionSamplerInShader(const ShaderReflection& r, const std::string& name)
    {
        for (const auto& s : r.samplers)
            if (s.name == name || s.groupQualifiedName == name) return &s;
        return nullptr;
    }

    inline void PopulateShaderStageProps(ShaderStageProps& stageProps, ShaderStage stage, const ShaderReflection& refl)
    {
        for (const NSL::NSCPropertyGroup& group : refl.propertyGroups)
        {
            if (group.resourceKind == ResourceKind::BUFFER)
            {
                const NSL::NSCBuffer* buf = refl.FindBuffer(group.qualifiedName);
                if (!buf) buf = refl.FindBuffer(group.name);

                // A buffer requires a property slot if it contains user-settable (Add) fields (OnlyAdd or Hybrid)
                BOOL hasAddFields = false;
                if (buf)
                {
                    for (const auto& f : buf->fields)
                    {
                        if (!f.isGet)
                        {
                            hasAddFields = true;
                            break;
                        }
                    }
                }

                if (buf && hasAddFields)
                {
                    BOOL alreadyExists = false;
                    for (const auto& existing : stageProps.buffers)
                    {
                        if (existing.propertyName == group.ownerPropertyName && existing.resourceName == group.name)
                        {
                            alreadyExists = true;
                            break;
                        }
                    }

                    if (!alreadyExists)
                    {
                        ShaderBufferSlot slot;
                        slot.propertyName  = group.ownerPropertyName;
                        slot.resourceName  = group.name;
                        slot.byteSize      = buf->totalSize;
                        slot.slotIndex     = buf->binding;
                        slot.registerSpace = buf->set;

                        slot.cpuData.assign(buf->totalSize, 0);

                        // Pre-populate defaults for Add fields from NSL reflection
                        for (const auto& field : buf->fields)
                        {
                            if (!field.isGet && !field.defaultData.empty() && field.offset + field.size <= buf->totalSize)
                            {
                                std::memcpy(slot.cpuData.data() + field.offset,
                                            field.defaultData.data(),
                                            std::min<SIZE_T>(field.size, field.defaultData.size()));
                            }
                        }
                        stageProps.buffers.push_back(std::move(slot));
                    }
                }
            }
            else if (group.resourceKind == ResourceKind::TEXTURE_2D ||
                     group.resourceKind == ResourceKind::TEXTURE_CUBE ||
                     group.resourceKind == ResourceKind::TEXTURE_2D_ARRAY ||
                     group.resourceKind == ResourceKind::TEXTURE_CUBE_ARRAY)
            {
                if (!group.entries.empty())
                {
                    for (const auto& entry : group.entries)
                    {
                        if (entry.kind == NSL::PropertyEntryKind::GET) continue;

                        const std::string entryResName = entry.name.empty() ? group.name : entry.name;
                        BOOL alreadyExists = false;
                        for (const auto& existing : stageProps.textures)
                        {
                            if (existing.propertyName == group.ownerPropertyName &&
                                existing.groupName == group.name &&
                                existing.resourceName == entryResName)
                            {
                                alreadyExists = true;
                                break;
                            }
                        }
                        if (alreadyExists) continue;

                        ShaderTextureSlot slot;
                        slot.propertyName = group.ownerPropertyName;
                        slot.groupName    = group.name;
                        slot.resourceName = entryResName;
                        slot.texture      = nullptr;

                        const NSL::NSCTexture* tex = refl.FindTexture(group.qualifiedName, entry.name);
                        if (!tex) tex = FindReflectionTextureInShader(refl, entry.name);
                        if (tex)
                        {
                            slot.slotIndex     = tex->binding;
                            slot.registerSpace = tex->set;
                        }

                        stageProps.textures.push_back(std::move(slot));
                    }
                }
                else
                {
                    const NSL::NSCTexture* tex = FindReflectionTextureInShader(refl, group.qualifiedName);
                    if (!tex) tex = FindReflectionTextureInShader(refl, group.name);
                    if (!tex || !tex->IsEngineProvided())
                    {
                        BOOL alreadyExists = false;
                        for (const auto& existing : stageProps.textures)
                        {
                            if (existing.propertyName == group.ownerPropertyName && existing.resourceName == group.name)
                            {
                                alreadyExists = true;
                                break;
                            }
                        }
                        if (!alreadyExists)
                        {
                            ShaderTextureSlot slot;
                            slot.propertyName = group.ownerPropertyName;
                            slot.groupName    = group.name;
                            slot.resourceName = group.name;
                            slot.texture      = nullptr;
                            if (tex)
                            {
                                slot.slotIndex     = tex->binding;
                                slot.registerSpace = tex->set;
                            }
                            stageProps.textures.push_back(std::move(slot));
                        }
                    }
                }
            }
            else if (group.resourceKind == ResourceKind::RW_TEXTURE_2D)
            {
                if (!group.entries.empty())
                {
                    for (const auto& entry : group.entries)
                    {
                        if (entry.kind == NSL::PropertyEntryKind::GET) continue;

                        const std::string entryResName = entry.name.empty() ? group.name : entry.name;
                        BOOL alreadyExists = false;
                        for (const auto& existing : stageProps.rwTextures)
                        {
                            if (existing.propertyName == group.ownerPropertyName &&
                                existing.groupName == group.name &&
                                existing.resourceName == entryResName)
                            {
                                alreadyExists = true;
                                break;
                            }
                        }
                        if (alreadyExists) continue;

                        ShaderRWTextureSlot slot;
                        slot.propertyName  = group.ownerPropertyName;
                        slot.groupName     = group.name;
                        slot.resourceName  = entryResName;
                        slot.texture       = nullptr;
                        slot.slotIndex     = (entry.binding != 0xFFFF) ? entry.binding : group.binding;
                        slot.registerSpace = (entry.binding != 0xFFFF) ? entry.set : group.set;
                        stageProps.rwTextures.push_back(std::move(slot));
                    }
                }
                else
                {
                    BOOL alreadyExists = false;
                    for (const auto& existing : stageProps.rwTextures)
                    {
                        if (existing.propertyName == group.ownerPropertyName && existing.resourceName == group.name)
                        {
                            alreadyExists = true;
                            break;
                        }
                    }
                    if (!alreadyExists)
                    {
                        ShaderRWTextureSlot slot;
                        slot.propertyName  = group.ownerPropertyName;
                        slot.groupName     = group.name;
                        slot.resourceName  = group.name;
                        slot.texture       = nullptr;
                        slot.slotIndex     = group.binding;
                        slot.registerSpace = group.set;
                        stageProps.rwTextures.push_back(std::move(slot));
                    }
                }
            }
            else if (group.resourceKind == ResourceKind::SAMPLER_STATE)
            {
                if (!group.entries.empty())
                {
                    for (const auto& entry : group.entries)
                    {
                        if (entry.kind == NSL::PropertyEntryKind::GET) continue;

                        const NSL::NSCSampler* smp = refl.FindSampler(group.qualifiedName, entry.name);
                        if (!smp) smp = FindReflectionSamplerInShader(refl, entry.name);
                        if (!smp) smp = FindReflectionSamplerInShader(refl, group.qualifiedName);
                        if (smp && smp->IsEngineProvided()) continue;

                        const std::string entryResName = entry.name.empty() ? group.name : entry.name;
                        BOOL alreadyExists = false;
                        for (const auto& existing : stageProps.samplers)
                        {
                            if (existing.propertyName == group.ownerPropertyName &&
                                existing.groupName == group.name &&
                                existing.resourceName == entryResName)
                            {
                                alreadyExists = true;
                                break;
                            }
                        }
                        if (alreadyExists) continue;

                        ShaderSamplerSlot slot;
                        slot.propertyName = group.ownerPropertyName;
                        slot.groupName    = group.name;
                        slot.resourceName = entryResName;
                        slot.sampler      = nullptr;

                        if (smp)
                        {
                            slot.isComparison  = smp->isComparison;
                            slot.slotIndex     = smp->binding;
                            slot.registerSpace = smp->set;
                        }

                        stageProps.samplers.push_back(std::move(slot));
                    }
                }
                else
                {
                    const NSL::NSCSampler* smp = FindReflectionSamplerInShader(refl, group.qualifiedName);
                    if (!smp) smp = FindReflectionSamplerInShader(refl, group.name);
                    if (!smp || !smp->IsEngineProvided())
                    {
                        BOOL alreadyExists = false;
                        for (const auto& existing : stageProps.samplers)
                        {
                            if (existing.propertyName == group.ownerPropertyName && existing.resourceName == group.name)
                            {
                                alreadyExists = true;
                                break;
                            }
                        }
                        if (!alreadyExists)
                        {
                            ShaderSamplerSlot slot;
                            slot.propertyName  = group.ownerPropertyName;
                            slot.groupName     = group.name;
                            slot.resourceName  = group.name;
                            slot.isComparison  = smp ? smp->isComparison : false;
                            slot.slotIndex     = smp ? smp->binding : group.binding;
                            slot.registerSpace = smp ? smp->set : 0;
                            slot.sampler       = nullptr;
                            stageProps.samplers.push_back(std::move(slot));
                        }
                    }
                }
            }
            else if (group.resourceKind == ResourceKind::STRUCTURED_BUFFER ||
                     group.resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
            {
                const NSL::NSCStructuredBuffer* sb = refl.FindStructuredBuffer(group.qualifiedName);
                if (!sb) sb = refl.FindStructuredBuffer(group.name);

                uint32_t stride  = sb ? sb->stride : 0;
                uint16_t binding = sb ? sb->binding : group.binding;
                uint16_t set     = sb ? sb->set : group.set;

                if (!sb && group.resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
                {
                    const NSL::NSCUAVBuffer* uav = refl.FindUAV(group.qualifiedName);
                    if (!uav) uav = refl.FindUAV(group.name);
                    if (uav)
                    {
                        binding = uav->binding;
                        set     = uav->set;
                        for (const auto& f : uav->fields)
                        {
                            stride = (std::max)(stride, f.offset + f.size);
                        }
                    }
                }

                // Pure Tier-1 GPU Instancing stream (Instances { Add(worldMatrix), ... } or Get(INSTANCE_*))
                // is driven by the scene instance batcher rather than a Material property slot.
                // Any other Vertex StructuredBuffer (including space0 user or CST hybrid Get+Add buffers)
                // MUST remain accessible as a Material property slot!
                if (stage == ShaderStage::VERTEX && sb != nullptr)
                {
                    BOOL hasInstGetField       = false;
                    BOOL hasOtherGetField      = false;
                    BOOL hasTier1WorldMatrix   = false;
                    BOOL hasNonTier1AddField   = false;
                    for (const auto& f : sb->fields)
                    {
                        if (f.isGet)
                        {
                            if (f.propertyName == "INSTANCE_WORLD_MATRIX" ||
                                f.propertyName == "INSTANCE_ID_VALUE" ||
                                f.propertyName == "INSTANCE_CUSTOM_DATA")
                            {
                                hasInstGetField = true;
                            }
                            else if (f.propertyName != "INSTANCE_BONE_OFFSET" &&
                                     f.propertyName != "INSTANCE_BONE_COUNT")
                            {
                                hasOtherGetField = true;
                            }
                        }
                        else
                        {
                            if (f.name == "worldMatrix" && f.size == 64u)
                            {
                                hasTier1WorldMatrix = true;
                            }
                            else if (f.name != "instanceId" && f.name != "customData")
                            {
                                hasNonTier1AddField = true;
                            }
                        }
                    }
                    const BOOL isPureTier1Instances =
                        !hasOtherGetField && !hasNonTier1AddField &&
                        (hasInstGetField || hasTier1WorldMatrix);
                    if (isPureTier1Instances)
                    {
                        continue;
                    }
                }

                BOOL alreadyExists = false;
                for (const auto& existing : stageProps.structuredBuffers)
                {
                    if (existing.propertyName == group.ownerPropertyName && existing.resourceName == group.name)
                    {
                        alreadyExists = true;
                        break;
                    }
                }
                if (!alreadyExists)
                {
                    ShaderStructuredBufferSlot slot;
                    slot.propertyName  = group.ownerPropertyName;
                    slot.resourceName  = group.name;
                    slot.stride        = stride;
                    slot.slotIndex     = binding;
                    slot.registerSpace = set;
                    slot.isUAV         = (group.resourceKind == ResourceKind::RW_STRUCTURED_BUFFER);
                    stageProps.structuredBuffers.push_back(std::move(slot));
                }
            }
        }

        // Fallback for direct reflection structures that do not populate propertyGroups
        if (refl.propertyGroups.empty())
        {
            auto SplitQualifiedName = [](const std::string& qName, std::string& outProp, std::string& outRes)
            {
                const SIZE_T sep = qName.find("__");
                if (sep != std::string::npos)
                {
                    outProp = qName.substr(0, sep);
                    outRes  = qName.substr(sep + 2);
                }
                else
                {
                    outProp.clear();
                    outRes = qName;
                }
            };

            for (const auto& buf : refl.buffers)
            {
                BOOL hasAddFields = false;
                for (const auto& f : buf.fields)
                {
                    if (!f.isGet) { hasAddFields = true; break; }
                }
                if (!hasAddFields || buf.totalSize == 0) continue;

                ShaderBufferSlot slot;
                SplitQualifiedName(buf.name, slot.propertyName, slot.resourceName);
                slot.byteSize      = buf.totalSize;
                slot.slotIndex     = buf.binding;
                slot.registerSpace = buf.set;
                slot.cpuData.assign(buf.totalSize, 0);
                for (const auto& field : buf.fields)
                {
                    if (!field.isGet && !field.defaultData.empty() && field.offset + field.size <= buf.totalSize)
                    {
                        std::memcpy(slot.cpuData.data() + field.offset,
                                    field.defaultData.data(),
                                    std::min<SIZE_T>(field.size, field.defaultData.size()));
                    }
                }
                stageProps.buffers.push_back(std::move(slot));
            }

            for (const auto& sb : refl.structuredBuffers)
            {
                if (sb.IsEngineProvided()) continue;
                if (stage == ShaderStage::VERTEX)
                {
                    BOOL hasTier1World  = false;
                    BOOL hasNonTier1Add = false;
                    for (const auto& f : sb.fields)
                    {
                        if (!f.isGet)
                        {
                            if (f.name == "worldMatrix" && f.size == 64u)
                                hasTier1World = true;
                            else if (f.name != "instanceId" && f.name != "customData")
                                hasNonTier1Add = true;
                        }
                    }
                    if (hasTier1World && !hasNonTier1Add) continue;
                }

                ShaderStructuredBufferSlot slot;
                SplitQualifiedName(sb.name, slot.propertyName, slot.resourceName);
                slot.stride        = sb.stride;
                slot.slotIndex     = sb.binding;
                slot.registerSpace = sb.set;
                slot.isUAV         = (refl.FindUAV(sb.name) != nullptr);
                stageProps.structuredBuffers.push_back(std::move(slot));
            }

            for (const auto& uav : refl.uavBuffers)
            {
                if (uav.resourceKind != ResourceKind::RW_STRUCTURED_BUFFER) continue;
                if (refl.FindStructuredBuffer(uav.name) != nullptr) continue;

                uint32_t stride = 0;
                for (const auto& f : uav.fields)
                {
                    stride = (std::max)(stride, f.offset + f.size);
                }

                ShaderStructuredBufferSlot slot;
                SplitQualifiedName(uav.name, slot.propertyName, slot.resourceName);
                slot.stride        = stride;
                slot.slotIndex     = uav.binding;
                slot.registerSpace = uav.set;
                slot.isUAV         = true;
                stageProps.structuredBuffers.push_back(std::move(slot));
            }

            for (const auto& tex : refl.textures)
            {
                if (tex.IsEngineProvided()) continue;

                ShaderTextureSlot slot;
                SplitQualifiedName(tex.groupQualifiedName.empty() ? tex.name : tex.groupQualifiedName,
                                   slot.propertyName, slot.groupName);
                slot.resourceName  = tex.name.empty() ? slot.groupName : tex.name;
                slot.slotIndex     = tex.binding;
                slot.registerSpace = tex.set;
                slot.texture       = nullptr;
                stageProps.textures.push_back(std::move(slot));
            }

            for (const auto& uav : refl.uavBuffers)
            {
                if (uav.resourceKind != ResourceKind::RW_TEXTURE_2D) continue;

                std::string propName;
                std::string grpName;
                SplitQualifiedName(uav.name, propName, grpName);

                if (!uav.fields.empty())
                {
                    for (const auto& field : uav.fields)
                    {
                        ShaderRWTextureSlot slot;
                        slot.propertyName  = propName;
                        slot.groupName     = grpName;
                        slot.resourceName  = field.name.empty() ? grpName : field.name;
                        slot.slotIndex     = (field.binding != 0xFFFF) ? field.binding : uav.binding;
                        slot.registerSpace = (field.binding != 0xFFFF) ? field.set : uav.set;
                        slot.texture       = nullptr;
                        stageProps.rwTextures.push_back(std::move(slot));
                    }
                }
                else
                {
                    ShaderRWTextureSlot slot;
                    slot.propertyName  = propName;
                    slot.groupName     = grpName;
                    slot.resourceName  = grpName;
                    slot.slotIndex     = uav.binding;
                    slot.registerSpace = uav.set;
                    slot.texture       = nullptr;
                    stageProps.rwTextures.push_back(std::move(slot));
                }
            }

            for (const auto& smp : refl.samplers)
            {
                if (smp.IsEngineProvided()) continue;

                ShaderSamplerSlot slot;
                SplitQualifiedName(smp.groupQualifiedName.empty() ? smp.name : smp.groupQualifiedName,
                                   slot.propertyName, slot.groupName);
                slot.resourceName  = smp.name.empty() ? slot.groupName : smp.name;
                slot.isComparison  = smp.isComparison;
                slot.slotIndex     = smp.binding;
                slot.registerSpace = smp.set;
                slot.sampler       = nullptr;
                stageProps.samplers.push_back(std::move(slot));
            }
        }
    }
}

