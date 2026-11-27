#include "core_nvs_shadow_rasterizer.h"
#include "core_resource_binding_resolver.h"
#include "../../include/material/material.h"
#include "../../include/scene/scene.h"
#include "../../include/camera/camera.h"
#include "../../include/light/light.h"
#include "../../include/shader/shader.h"
#include "../../include/shader_set/shader_set.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_set>

namespace Northstar
{
    namespace
    {
        inline UINT64 HashCombine64(UINT64 seed, UINT64 val) noexcept
        {
            seed ^= val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }

        inline bool EqualsIgnoreCase(const std::string& a, const char* b)
        {
            if (!b) return false;
            const size_t lenB = std::strlen(b);
            if (a.size() != lenB) return false;
            for (size_t i = 0; i < lenB; ++i)
            {
                if (std::toupper(static_cast<unsigned char>(a[i])) !=
                    std::toupper(static_cast<unsigned char>(b[i])))
                {
                    return false;
                }
            }
            return true;
        }

        enum class ShadowStructuredBufferRole : UINT8
        {
            USER_MATERIAL      = 0,
            GLOBAL_BONE_TABLE  = 1,
            BONE_PALETTE_SLICE = 2,
            INSTANCE_POOL      = 3,
            GPU_SCENE_TABLE    = 4,
            ENGINE_NULL_SRV    = 5,
            INVALID_LAYOUT     = 6
        };

        inline NVSShadowPatchSemantic MapCBufferFieldToNVSPatchSemantic(const std::string& propName)
        {
            const UINT32 sem = CoreResourceBindingResolver::MapPropertyNameToPatchSemantic(propName);
            if (IsNVSInstancePatchSemantic(sem))
            {
                return NVS_PATCH_SEMANTIC_NONE;
            }
            return static_cast<NVSShadowPatchSemantic>(sem);
        }

        inline NVSShadowPatchSemantic MapInstanceFieldToNVSPatchSemantic(
            const NSL::NSCStructField& f,
            bool                       allowTier1InstanceNames)
        {
            if (f.isGet)
            {
                const UINT32 sem = CoreResourceBindingResolver::MapPropertyNameToPatchSemantic(
                    f.propertyName.empty() ? f.name : f.propertyName);
                if (IsNVSInstancePatchSemantic(sem))
                {
                    return static_cast<NVSShadowPatchSemantic>(sem);
                }
                return NVS_PATCH_SEMANTIC_NONE;
            }
            if (allowTier1InstanceNames)
            {
                if (f.name == "worldMatrix" && f.size == 64u) return NVS_PATCH_INSTANCE_WORLD_MATRIX;
                if (f.name == "instanceId"  && f.size == 4u)  return NVS_PATCH_INSTANCE_ID;
                if (f.name == "customData"  && f.size == 4u)  return NVS_PATCH_INSTANCE_CUSTOM_DATA;
            }
            return NVS_PATCH_SEMANTIC_NONE;
        }

        inline bool IsTier1InstanceBufferName(const NSL::NSCStructuredBuffer& nscSb, ShaderStage stage)
        {
            return CoreResourceBindingResolver::IsTier1InstanceStructuredBuffer(nscSb, stage) != false;
        }

        inline ShadowStructuredBufferRole ClassifyShadowStructuredBuffer(
            const NSL::NSCStructuredBuffer& nscSb,
            ShaderStage                     stage,
            BOOL*                           outRequiresSkinning = nullptr)
        {
            const StructuredBufferLayoutPlan plan = CoreResourceBindingResolver::AnalyzeStructuredBuffer(nscSb, stage, 0u);
            if (!plan.valid || plan.domain == EngineResourceDomain::INVALID_MIXED_DOMAINS)
            {
                return ShadowStructuredBufferRole::INVALID_LAYOUT;
            }

            if (outRequiresSkinning)
            {
                if (plan.domain == EngineResourceDomain::PER_OBJECT_BONE_SLICE ||
                    plan.domain == EngineResourceDomain::INSTANCE_SKINNING ||
                    plan.hasInstanceSkinFields)
                {
                    *outRequiresSkinning = true;
                }
            }

            switch (plan.domain)
            {
                case EngineResourceDomain::INSTANCES:
                case EngineResourceDomain::INSTANCE_SKINNING:
                    return ShadowStructuredBufferRole::INSTANCE_POOL;
                case EngineResourceDomain::GLOBAL_BONE_PALETTE:
                    return ShadowStructuredBufferRole::GLOBAL_BONE_TABLE;
                case EngineResourceDomain::PER_OBJECT_BONE_SLICE:
                    return ShadowStructuredBufferRole::BONE_PALETTE_SLICE;
                case EngineResourceDomain::GPU_SCENE_OBJECTS:
                    return ShadowStructuredBufferRole::GPU_SCENE_TABLE;
                case EngineResourceDomain::NVG_PAGE_POOL:
                case EngineResourceDomain::NVG_BUCKET_RANGES:
                case EngineResourceDomain::NVG_VISIBLE_CLUSTERS:
                case EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS:
                case EngineResourceDomain::LIGHTS:
                case EngineResourceDomain::DECALS:
                case EngineResourceDomain::REFLECTION_PROBES:
                case EngineResourceDomain::PARTICLES:
                case EngineResourceDomain::NVS_SPARSE_PAGE_HIERARCHY:
                case EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE:
                case EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE:
                case EngineResourceDomain::NVS_LIGHT_VIEW_TABLE:
                    return ShadowStructuredBufferRole::ENGINE_NULL_SRV;
                case EngineResourceDomain::USER_ONLY:
                default:
                    return ShadowStructuredBufferRole::USER_MATERIAL;
            }
        }

        inline bool DoesCBufferRequirePerDrawPromotion(const NSL::NSCBuffer& nscBuf)
        {
            for (const auto& f : nscBuf.fields)
            {
                if (f.isGet && MapCBufferFieldToNVSPatchSemantic(f.propertyName) != NVS_PATCH_SEMANTIC_NONE)
                {
                    return true;
                }
            }
            return false;
        }

        inline void PopulateInitialCBufferBytes(
            const NSL::NSCBuffer&     nscBuf,
            const ShaderStageProps&   props,
            const ShaderStageProps&   fbProps,
            Scene*                    scene,
            FLOAT32                   timeSeconds,
            std::vector<BYTE>&        outBytes,
            BOOL&                     outDirty)
        {
            outBytes.assign(nscBuf.totalSize, 0);
            outDirty = false;

            const MaterialBufferSlot* matSlot = nullptr;
            for (const auto& mbs : props.buffers)
            {
                if ((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name || mbs.resourceName == nscBuf.name)
                {
                    matSlot = &mbs;
                    break;
                }
            }
            if (!matSlot || !matSlot->dirty)
            {
                for (const auto& mbs : fbProps.buffers)
                {
                    if (((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name || mbs.resourceName == nscBuf.name) &&
                        (mbs.dirty || !matSlot))
                    {
                        matSlot = &mbs;
                        break;
                    }
                }
            }

            if (matSlot && !matSlot->cpuData.empty())
            {
                const SIZE_T copySize = (std::min)(outBytes.size(), matSlot->cpuData.size());
                std::memcpy(outBytes.data(), matSlot->cpuData.data(), copySize);
                outDirty = matSlot->dirty;
            }
            else
            {
                for (const auto& field : nscBuf.fields)
                {
                    if (!field.isGet && !field.defaultData.empty() && field.offset + field.size <= nscBuf.totalSize)
                    {
                        const SIZE_T copyFieldSize = (std::min)(static_cast<SIZE_T>(field.size), field.defaultData.size());
                        std::memcpy(outBytes.data() + field.offset, field.defaultData.data(), copyFieldSize);
                    }
                }
            }

            // Populate scene-level Get() fields (TIME, ENGINE_TIME, AMBIENT_*, LIGHT_COUNT, CAMERA_POSITION)
            for (const auto& field : nscBuf.fields)
            {
                if (!field.isGet || field.offset + field.size > nscBuf.totalSize) continue;
                BYTE* dst = outBytes.data() + field.offset;

                if ((field.propertyName == "TIME" || field.propertyName == "ENGINE_TIME") && field.size >= sizeof(FLOAT32))
                {
                    std::memcpy(dst, &timeSeconds, sizeof(FLOAT32));
                    outDirty = true;
                }
                else if (field.propertyName == "AMBIENT_COLOR" && field.size >= sizeof(Vec3))
                {
                    Vec3 amb(0.1f, 0.1f, 0.1f);
                    if (scene)
                    {
                        const Color c = scene->GetAmbientColor();
                        amb = Vec3(c.r, c.g, c.b);
                    }
                    std::memcpy(dst, &amb, sizeof(Vec3));
                    outDirty = true;
                }
                else if (field.propertyName == "AMBIENT_INTENSITY" && field.size >= sizeof(FLOAT32))
                {
                    const FLOAT32 intensity = scene ? scene->GetAmbientIntensity() : 1.0f;
                    std::memcpy(dst, &intensity, sizeof(FLOAT32));
                    outDirty = true;
                }
                else if (field.propertyName == "LIGHT_COUNT" && field.size >= sizeof(UINT32))
                {
                    const UINT32 lightCount = scene ? static_cast<UINT32>(scene->GetLights().size()) : 0u;
                    std::memcpy(dst, &lightCount, sizeof(UINT32));
                    outDirty = true;
                }
                else if (field.propertyName == "CAMERA_POSITION" && field.size >= sizeof(Vec3))
                {
                    Vec3 camPos = Vec3::Zero();
                    if (scene && scene->GetMainCamera())
                    {
                        camPos = scene->GetMainCamera()->GetPosition();
                    }
                    std::memcpy(dst, &camPos, sizeof(Vec3));
                    outDirty = true;
                }
                else if ((field.propertyName == "NVG_DRAW_BUCKET_ID" ||
                          field.propertyName == "NVG_VISIBLE_CLUSTER_OFFSET" ||
                          field.propertyName == "GPU_SCENE_OBJECT_SLOT") &&
                         field.size >= sizeof(UINT32))
                {
                    const UINT32 inv = kNVGInvalidIndex;
                    std::memcpy(dst, &inv, sizeof(UINT32));
                    outDirty = true;
                }
                else if (field.propertyName == "GEOMETRY_MODE" && field.size >= sizeof(UINT32))
                {
                    const UINT32 gm = static_cast<UINT32>(GeometryMode::STANDARD);
                    std::memcpy(dst, &gm, sizeof(UINT32));
                    outDirty = true;
                }
                else if (field.propertyName == "NVG_VISIBLE_CLUSTER_COUNT" && field.size >= sizeof(UINT32))
                {
                    const UINT32 zeroU = 0u;
                    std::memcpy(dst, &zeroU, sizeof(UINT32));
                    outDirty = true;
                }
                else if (field.propertyName == "NVG_PAGE_WORD_COUNT" && field.size >= sizeof(UINT32))
                {
                    const UINT32 words = kNVGPageWordCount;
                    std::memcpy(dst, &words, sizeof(UINT32));
                    outDirty = true;
                }
            }
        }

        inline UINT32 AppendBytesAsWords(const BYTE* bytes, UINT32 byteCount, std::vector<UINT32>& ioWords)
        {
            const UINT32 wordOffset = static_cast<UINT32>(ioWords.size());
            const UINT32 wordCount  = (byteCount + 3u) / 4u;
            if (wordCount == 0u)
            {
                return wordOffset;
            }
            ioWords.resize(wordOffset + wordCount, 0u);
            if (bytes && byteCount > 0u)
            {
                std::memcpy(ioWords.data() + wordOffset, bytes, byteCount);
            }
            return wordOffset;
        }
    }

    void CoreNVSShadowRasterizer::Initialize(
        NorthstarBackend*        backend,
        CoreBufferManager*       bufferManager,
        InternalResourceManager* internalResourceManager)
    {
        m_backend                 = backend;
        m_bufferManager           = bufferManager;
        m_internalResourceManager = internalResourceManager;
        m_drawGroups.clear();
        m_hasExplicitDrawGroups            = false;
        m_productionSceneActive            = false;
        m_rejectedMissingRTArrayIndexCount = 0u;
        m_lastPassUniquePSOCount           = 0u;
        m_lastPassActiveDrawGroupCount     = 0u;
    }

    void CoreNVSShadowRasterizer::Shutdown()
    {
        m_drawGroups.clear();
        m_hasExplicitDrawGroups = false;
        m_productionSceneActive = false;
        m_backend                 = nullptr;
        m_bufferManager           = nullptr;
        m_internalResourceManager = nullptr;
    }

    void CoreNVSShadowRasterizer::ClearDrawGroups()
    {
        m_drawGroups.clear();
        m_hasExplicitDrawGroups = false;
        m_productionSceneActive = false;
    }

    void CoreNVSShadowRasterizer::OnMaterialDestroyed(const Material* material)
    {
        if (!material) return;
        for (NVSShadowDrawGroup& grp : m_drawGroups)
        {
            if (grp.material == material)
            {
                grp = NVSShadowDrawGroup{};
            }
        }
    }

    void CoreNVSShadowRasterizer::OnShaderSetDestroyed(const ShaderSet* shaderSet)
    {
        if (!shaderSet) return;
        for (NVSShadowDrawGroup& grp : m_drawGroups)
        {
            if (grp.shadowShaderSet == shaderSet)
            {
                grp = NVSShadowDrawGroup{};
            }
        }
    }

    void CoreNVSShadowRasterizer::OnSceneDestroyed(const Scene* scene)
    {
        if (!scene) return;
        for (NVSShadowDrawGroup& grp : m_drawGroups)
        {
            if (grp.scene == scene)
            {
                grp.scene = nullptr;
            }
        }
    }

    void CoreNVSShadowRasterizer::SetShadowBucketMaterial(
        UINT32        bucketIndex,
        Material*     material,
        ShaderVariant variant,
        Scene*        scene)
    {
        if (bucketIndex == kNVSInvalidShadowRasterBucket)
        {
            return;
        }
        if (m_drawGroups.size() <= bucketIndex)
        {
            m_drawGroups.resize(bucketIndex + 1u);
        }

        ShaderSet* shadowSet = material ? material->GetShaderSet(PassType::SHADOW) : nullptr;
        NVSShadowDrawGroup& grp  = m_drawGroups[bucketIndex];
        grp.material             = material;
        grp.shadowShaderSet      = shadowSet;
        grp.variant              = variant;
        grp.scene                = scene;
        grp.psoCompatibilityKey  = ComputePSOCompatibilityKey(material, shadowSet, variant);
        grp.resourceBindingKey   = ComputeMaterialResourceBindingKey(material, shadowSet, variant);
        grp.shaderSetRevision    = shadowSet ? shadowSet->GetStateRevision() : 0u;
        grp.hasUserResourceBinds = (grp.resourceBindingKey != 0u);

        m_hasExplicitDrawGroups = false;
        for (const auto& g : m_drawGroups)
        {
            if (g.material != nullptr)
            {
                m_hasExplicitDrawGroups = true;
                break;
            }
        }
    }

    UINT32 CoreNVSShadowRasterizer::RegisterOrAcquireDrawGroup(
        Material*     material,
        ShaderVariant variant,
        Scene*        scene)
    {
        if (!material || !material->HasPass(PassType::SHADOW))
        {
            return kNVSInvalidShadowRasterBucket;
        }

        ShaderSet* shadowSet = material->GetShaderSet(PassType::SHADOW);
        if (!shadowSet)
        {
            return kNVSInvalidShadowRasterBucket;
        }

        const UINT64 psoKey = ComputePSOCompatibilityKey(material, shadowSet, variant);
        const UINT64 resKey = ComputeMaterialResourceBindingKey(material, shadowSet, variant);
        const UINT64 rev    = shadowSet->GetStateRevision();

        for (UINT32 i = 0u; i < static_cast<UINT32>(m_drawGroups.size()); ++i)
        {
            const NVSShadowDrawGroup& existing = m_drawGroups[i];
            if (existing.scene == scene &&
                existing.variant == variant &&
                existing.psoCompatibilityKey == psoKey &&
                existing.resourceBindingKey == resKey &&
                existing.shaderSetRevision == rev)
            {
                return i;
            }
        }

        NVSShadowDrawGroup grp{};
        grp.material             = material;
        grp.shadowShaderSet      = shadowSet;
        grp.variant              = variant;
        grp.scene                = scene;
        grp.psoCompatibilityKey  = psoKey;
        grp.resourceBindingKey   = resKey;
        grp.shaderSetRevision    = rev;
        grp.hasUserResourceBinds = (resKey != 0u);

        const UINT32 idx = static_cast<UINT32>(m_drawGroups.size());
        m_drawGroups.push_back(grp);
        m_hasExplicitDrawGroups = true;
        return idx;
    }

    BOOL CoreNVSShadowRasterizer::DoesShaderDeclareRenderTargetArrayIndexOutput(const Shader* shader)
    {
        if (!shader || !shader->IsValid() || shader->GetNativeHandle() == 0)
        {
            return false;
        }

        const ShaderReflection& refl = shader->GetReflection();
        for (const auto& outEntry : refl.outputLayout)
        {
            if (EqualsIgnoreCase(outEntry.semantic, "SV_RenderTargetArrayIndex") ||
                EqualsIgnoreCase(outEntry.semantic, "RENDER_TARGET_ARRAY_INDEX") ||
                EqualsIgnoreCase(outEntry.name, "RENDER_TARGET_ARRAY_INDEX"))
            {
                return true;
            }
        }
        return false;
    }

    BOOL CoreNVSShadowRasterizer::ValidateShadowShaderSetContract(
        const Material*         material,
        const ShaderSet*        shadowSet,
        ShaderVariant           variant,
        const NorthstarBackend* backend,
        BOOL                    forceGSFallback,
        BOOL&                   outRequiresSkinningData,
        BOOL&                   outMissingRTArrayIndex)
    {
        outRequiresSkinningData = false;
        outMissingRTArrayIndex  = false;

        if (!material || !shadowSet)
        {
            return false;
        }

        const Shader* vs = shadowSet->GetVertexShader(variant);
        if (!vs || !vs->IsValid() || vs->GetNativeHandle() == 0)
        {
            return false;
        }

        const Shader* ps = shadowSet->GetPixelShader();
        if (ps && (!ps->IsValid() || ps->GetNativeHandle() == 0))
        {
            return false;
        }

        const Shader* gs = shadowSet->GetGeometryShader();
        if (gs && (!gs->IsValid() || gs->GetNativeHandle() == 0))
        {
            return false;
        }

        const Shader* hs = shadowSet->GetHullShader();
        if (hs && (!hs->IsValid() || hs->GetNativeHandle() == 0))
        {
            return false;
        }

        const Shader* ds = shadowSet->GetDomainShader();
        if (ds && (!ds->IsValid() || ds->GetNativeHandle() == 0))
        {
            return false;
        }

        if ((hs != nullptr) != (ds != nullptr))
        {
            return false;
        }

        // Enforce VS / DS / GS SV_RenderTargetArrayIndex contract for NVS layered Texture2DArray shadow pool
        if (gs != nullptr)
        {
            // Case B (GS != nullptr): User Geometry Shader is the final pre-rasterizer stage and must output SV_RenderTargetArrayIndex
            if (!DoesShaderDeclareRenderTargetArrayIndexOutput(gs))
            {
                outMissingRTArrayIndex = true;
                m_rejectedMissingRTArrayIndexCount++;
                return false;
            }
        }
        else
        {
            // Case A (GS == nullptr): Hardware must support VPAndRTArrayIndexWithoutGS and final pre-rasterizer stage (DS or VS) must output SV_RenderTargetArrayIndex
            const BOOL hwSupportsRTArrayWithoutGS =
                !forceGSFallback &&
                (!backend || (!backend->IsForceLayeredDepthFallback() &&
                              backend->SupportsVPAndRTArrayIndexWithoutGS()));
            const Shader* finalPreRasterStage = (ds != nullptr) ? ds : vs;
            if (!hwSupportsRTArrayWithoutGS || !DoesShaderDeclareRenderTargetArrayIndexOutput(finalPreRasterStage))
            {
                outMissingRTArrayIndex = true;
                m_rejectedMissingRTArrayIndexCount++;
                return false;
            }
        }

        const Shader*            stages[5]     = { vs, ps, gs, hs, ds };
        const ShaderStage        stageTypes[5] = {
            ShaderStage::VERTEX,
            ShaderStage::PIXEL,
            ShaderStage::GEOMETRY,
            ShaderStage::HULL,
            ShaderStage::DOMAIN
        };
        const GraphicsShaderSlot slots[5]      = {
            GraphicsShaderSlot::VERTEX,
            GraphicsShaderSlot::PIXEL,
            GraphicsShaderSlot::GEOMETRY,
            GraphicsShaderSlot::HULL,
            GraphicsShaderSlot::DOMAIN_STAGE
        };

        UINT32 promotedCBVCount = 0u;
        UINT32 promotedSRVCount = 0u;

        for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
        {
            const Shader* stShader = stages[stIdx];
            if (!stShader) continue;

            const ShaderStage        sType   = stageTypes[stIdx];
            const GraphicsShaderSlot gSlot   = slots[stIdx];
            const ShaderStageProps&  props   = material->GetPropertyStore().Get(PassType::SHADOW, gSlot);
            const ShaderStageProps&  fbProps = material->GetPropertyStore().Get(PassType::GEOMETRY, gSlot);

            if (props.HasDestroyedResource() || fbProps.HasDestroyedResource())
            {
                return false;
            }

            const ShaderReflection& refl = stShader->GetReflection();
            if (CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(refl))
            {
                outRequiresSkinningData = true;
            }

            for (const auto& b : refl.buffers)
            {
                if (b.binding == 0xFFFF || b.set >= 16u || b.totalSize == 0) continue;
                if (DoesCBufferRequirePerDrawPromotion(b))
                {
                    promotedCBVCount++;
                }
            }

            for (const auto& nscTex : refl.textures)
            {
                if (nscTex.binding == 0xFFFF || nscTex.set >= 16u) continue;
                if (nscTex.IsEngineProvided())
                {
                    // Engine-owned NVS_SHADOW_MAP cannot be sampled as SRV during PassType::SHADOW DSV rasterization
                    return false;
                }
                BOOL foundTex = false;
                auto checkTex = [&](const ShaderStageProps& sp) {
                    for (const auto& mt : sp.textures)
                    {
                        if ((mt.resourceName == nscTex.name ||
                             (mt.propertyName + "__" + mt.resourceName) == nscTex.groupQualifiedName ||
                             (mt.slotIndex == nscTex.binding && mt.registerSpace == nscTex.set)) &&
                            mt.texture != nullptr && mt.texture->GetNativeHandle() != 0)
                        {
                            foundTex = true;
                            return;
                        }
                    }
                };
                checkTex(props);
                if (!foundTex) checkTex(fbProps);
                if (!foundTex) return false;
            }

            for (const auto& nscSamp : refl.samplers)
            {
                if (nscSamp.binding == 0xFFFF || nscSamp.set >= 16u) continue;
                if (nscSamp.IsEngineProvided()) return false;
                BOOL foundSamp = false;
                auto checkSamp = [&](const ShaderStageProps& sp) {
                    for (const auto& ms : sp.samplers)
                    {
                        if ((ms.resourceName == nscSamp.name ||
                             (ms.propertyName + "__" + ms.resourceName) == nscSamp.groupQualifiedName ||
                             (ms.slotIndex == nscSamp.binding && ms.registerSpace == nscSamp.set)) &&
                            ms.sampler != nullptr && ms.sampler->GetNativeHandle() != 0 &&
                            ms.sampler->IsComparison() == nscSamp.isComparison)
                        {
                            foundSamp = true;
                            return;
                        }
                    }
                };
                checkSamp(props);
                if (!foundSamp) checkSamp(fbProps);
                if (!foundSamp) return false;
            }

            for (const auto& nscSb : refl.structuredBuffers)
            {
                if (nscSb.binding == 0xFFFF || nscSb.set >= 16u || nscSb.stride == 0) continue;
                const ShadowStructuredBufferRole role = ClassifyShadowStructuredBuffer(nscSb, sType, &outRequiresSkinningData);
                if (role == ShadowStructuredBufferRole::INVALID_LAYOUT)
                {
                    return false;
                }
                if (role == ShadowStructuredBufferRole::INSTANCE_POOL ||
                    role == ShadowStructuredBufferRole::BONE_PALETTE_SLICE)
                {
                    promotedSRVCount++;
                    continue;
                }
                if (role == ShadowStructuredBufferRole::GLOBAL_BONE_TABLE ||
                    role == ShadowStructuredBufferRole::GPU_SCENE_TABLE ||
                    role == ShadowStructuredBufferRole::ENGINE_NULL_SRV)
                {
                    continue;
                }

                BOOL foundSb = false;
                auto checkSb = [&](const ShaderStageProps& sp) {
                    for (const auto& ms : sp.structuredBuffers)
                    {
                        if ((ms.propertyName + "__" + ms.resourceName) == nscSb.name ||
                            ms.resourceName == nscSb.name ||
                            (ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set))
                        {
                            const NativeHandle h = ms.gpuBuffer ? ms.gpuBuffer->GetNativeHandle() : ms.gpuBufferHandle;
                            const UINT32 st      = ms.gpuBuffer ? ms.gpuBuffer->GetStride() : ms.gpuBufferStride;
                            if ((h != 0 && st == nscSb.stride) || (!ms.cpuData.empty() && ms.elementCount > 0))
                            {
                                foundSb = true;
                                return;
                            }
                        }
                    }
                };
                checkSb(props);
                if (!foundSb) checkSb(fbProps);
                if (!foundSb) return false;
            }
        }

        if (promotedCBVCount > kNVSMaxPromotedIndirectCBVs || promotedSRVCount > 1u)
        {
            return false;
        }

        return true;
    }

    UINT64 CoreNVSShadowRasterizer::ComputePSOCompatibilityKey(
        const Material*  material,
        const ShaderSet* shadowSet,
        ShaderVariant    variant)
    {
        if (!material || !shadowSet) return 0u;

        const Shader* vs = shadowSet->GetVertexShader(variant);
        const Shader* ps = shadowSet->GetPixelShader();
        const Shader* gs = shadowSet->GetGeometryShader();
        const Shader* hs = shadowSet->GetHullShader();
        const Shader* ds = shadowSet->GetDomainShader();

        UINT64 h = 14695981039346656037ULL;
        h = HashCombine64(h, vs ? vs->GetNativeHandle() : 0u);
        h = HashCombine64(h, vs ? vs->GetStateRevision() : 0u);
        h = HashCombine64(h, ps ? ps->GetNativeHandle() : 0u);
        h = HashCombine64(h, ps ? ps->GetStateRevision() : 0u);
        h = HashCombine64(h, gs ? gs->GetNativeHandle() : 0u);
        h = HashCombine64(h, gs ? gs->GetStateRevision() : 0u);
        h = HashCombine64(h, hs ? hs->GetNativeHandle() : 0u);
        h = HashCombine64(h, hs ? hs->GetStateRevision() : 0u);
        h = HashCombine64(h, ds ? ds->GetNativeHandle() : 0u);
        h = HashCombine64(h, ds ? ds->GetStateRevision() : 0u);
        h = HashCombine64(h, shadowSet->GetStateRevision());
        h = HashCombine64(h, static_cast<UINT64>(variant));
        h = HashCombine64(h, static_cast<UINT64>(material->GetCullMode()));
        h = HashCombine64(h, static_cast<UINT64>(material->GetFillMode()));
        const DepthState dsState = material->GetDepthState();
        h = HashCombine64(h, dsState.test ? 1u : 0u);
        h = HashCombine64(h, dsState.write ? 1u : 0u);
        h = HashCombine64(h, static_cast<UINT64>(dsState.compare));
        return h;
    }

    UINT64 CoreNVSShadowRasterizer::ComputeMaterialResourceBindingKey(
        const Material*  material,
        const ShaderSet* shadowSet,
        ShaderVariant    variant)
    {
        if (!material || !shadowSet) return 0u;

        const Shader* vs = shadowSet->GetVertexShader(variant);
        const Shader* ps = shadowSet->GetPixelShader();
        const Shader* gs = shadowSet->GetGeometryShader();
        const Shader* hs = shadowSet->GetHullShader();
        const Shader* ds = shadowSet->GetDomainShader();

        const Shader*            stages[5]     = { vs, ps, gs, hs, ds };
        const ShaderStage        stageTypes[5] = {
            ShaderStage::VERTEX,
            ShaderStage::PIXEL,
            ShaderStage::GEOMETRY,
            ShaderStage::HULL,
            ShaderStage::DOMAIN
        };
        const GraphicsShaderSlot slots[5]      = {
            GraphicsShaderSlot::VERTEX,
            GraphicsShaderSlot::PIXEL,
            GraphicsShaderSlot::GEOMETRY,
            GraphicsShaderSlot::HULL,
            GraphicsShaderSlot::DOMAIN_STAGE
        };

        UINT64 h           = 0u;
        BOOL   hasUserBind = false;

        for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
        {
            const Shader* stShader = stages[stIdx];
            if (!stShader) continue;

            const ShaderStage        sType   = stageTypes[stIdx];
            const GraphicsShaderSlot gSlot   = slots[stIdx];
            const ShaderStageProps&  props   = material->GetPropertyStore().Get(PassType::SHADOW, gSlot);
            const ShaderStageProps&  fbProps = material->GetPropertyStore().Get(PassType::GEOMETRY, gSlot);
            const ShaderReflection&  refl    = stShader->GetReflection();

            for (const auto& nscBuf : refl.buffers)
            {
                if (nscBuf.binding == 0xFFFF || nscBuf.set >= 16u || nscBuf.totalSize == 0) continue;
                bool hasNonGetField = false;
                for (const auto& f : nscBuf.fields)
                {
                    if (!f.isGet)
                    {
                        hasNonGetField = true;
                        break;
                    }
                }
                if (!hasNonGetField) continue;

                hasUserBind = true;
                const MaterialBufferSlot* matSlot = nullptr;
                for (const auto& mbs : props.buffers)
                {
                    if ((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name || mbs.resourceName == nscBuf.name)
                    {
                        matSlot = &mbs;
                        break;
                    }
                }
                if (!matSlot || !matSlot->dirty)
                {
                    for (const auto& mbs : fbProps.buffers)
                    {
                        if (((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name || mbs.resourceName == nscBuf.name) &&
                            (mbs.dirty || !matSlot))
                        {
                            matSlot = &mbs;
                            break;
                        }
                    }
                }
                if (matSlot && !matSlot->cpuData.empty())
                {
                    for (BYTE bVal : matSlot->cpuData)
                    {
                        h = HashCombine64(h, static_cast<UINT64>(bVal));
                    }
                }
                else
                {
                    h = HashCombine64(h, 0xD3FA017C0FFEEULL);
                    for (const auto& field : nscBuf.fields)
                    {
                        if (!field.isGet)
                        {
                            for (BYTE bVal : field.defaultData)
                            {
                                h = HashCombine64(h, static_cast<UINT64>(bVal));
                            }
                        }
                    }
                }
            }

            for (const auto& nscTex : refl.textures)
            {
                if (nscTex.binding == 0xFFFF || nscTex.set >= 16u || nscTex.IsEngineProvided()) continue;
                hasUserBind = true;
                NativeHandle texHandle = 0;
                auto findTex = [&](const ShaderStageProps& sp) {
                    for (const auto& mt : sp.textures)
                    {
                        if ((mt.resourceName == nscTex.name ||
                             (mt.propertyName + "__" + mt.resourceName) == nscTex.groupQualifiedName ||
                             (mt.slotIndex == nscTex.binding && mt.registerSpace == nscTex.set)) &&
                            mt.texture != nullptr)
                        {
                            texHandle = mt.texture->GetNativeHandle();
                            return;
                        }
                    }
                };
                findTex(props);
                if (texHandle == 0) findTex(fbProps);
                h = HashCombine64(h, texHandle);
            }

            for (const auto& nscSamp : refl.samplers)
            {
                if (nscSamp.binding == 0xFFFF || nscSamp.set >= 16u || nscSamp.IsEngineProvided()) continue;
                hasUserBind = true;
                NativeHandle sampHandle = 0;
                auto findSamp = [&](const ShaderStageProps& sp) {
                    for (const auto& ms : sp.samplers)
                    {
                        if ((ms.resourceName == nscSamp.name ||
                             (ms.propertyName + "__" + ms.resourceName) == nscSamp.groupQualifiedName ||
                             (ms.slotIndex == nscSamp.binding && ms.registerSpace == nscSamp.set)) &&
                            ms.sampler != nullptr)
                        {
                            sampHandle = ms.sampler->GetNativeHandle();
                            return;
                        }
                    }
                };
                findSamp(props);
                if (sampHandle == 0) findSamp(fbProps);
                h = HashCombine64(h, sampHandle);
            }

            for (const auto& nscSb : refl.structuredBuffers)
            {
                if (nscSb.binding == 0xFFFF || nscSb.stride == 0) continue;
                if (ClassifyShadowStructuredBuffer(nscSb, sType) != ShadowStructuredBufferRole::USER_MATERIAL) continue;
                hasUserBind = true;
                const ShaderStructuredBufferSlot* matSb = nullptr;
                auto findSb = [&](const ShaderStageProps& sp) -> const ShaderStructuredBufferSlot* {
                    for (const auto& ms : sp.structuredBuffers)
                    {
                        if ((ms.propertyName + "__" + ms.resourceName) == nscSb.name ||
                            ms.resourceName == nscSb.name ||
                            (ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set))
                        {
                            return &ms;
                        }
                    }
                    return nullptr;
                };
                matSb = findSb(props);
                if (!matSb || (matSb->gpuBuffer == nullptr && matSb->gpuBufferHandle == 0 && matSb->elementCount == 0))
                {
                    const ShaderStructuredBufferSlot* fbSb = findSb(fbProps);
                    if (fbSb) matSb = fbSb;
                }
                if (matSb)
                {
                    const NativeHandle sbH = matSb->gpuBuffer ? matSb->gpuBuffer->GetNativeHandle() : matSb->gpuBufferHandle;
                    if (sbH != 0)
                    {
                        h = HashCombine64(h, sbH);
                    }
                    else
                    {
                        for (BYTE bVal : matSb->cpuData)
                        {
                            h = HashCombine64(h, static_cast<UINT64>(bVal));
                        }
                    }
                }
            }
        }

        return hasUserBind ? (h != 0u ? h : 1u) : 0u;
    }

    BOOL CoreNVSShadowRasterizer::BindMaterialStageResources(
        const Shader*                               shader,
        ShaderStage                                 stage,
        const Material*                             material,
        PassType                                    primaryPass,
        Scene*                                      scene,
        FLOAT32                                     timeSeconds,
        std::vector<NativeConstantBufferBinding>&   outCBuffers,
        std::vector<NativeTextureBinding>&          outTextures,
        std::vector<NativeSamplerBinding>&          outSamplers,
        std::vector<NativeStructuredBufferBinding>& outStructuredBuffers)
    {
        if (!shader || !material) return true;

        const GraphicsShaderSlot gSlot = ShaderStageToSlot(stage);
        if (gSlot >= GraphicsShaderSlot::COUNT) return false;

        const ShaderStageProps& props   = material->GetPropertyStore().Get(primaryPass, gSlot);
        const ShaderStageProps& fbProps = material->GetPropertyStore().Get(PassType::GEOMETRY, gSlot);

        if (props.HasDestroyedResource() || fbProps.HasDestroyedResource())
        {
            return false;
        }

        const ShaderReflection& refl = shader->GetReflection();

        // 1. ConstantBuffers (non-promoted user Add() values + scene-level Get() values)
        for (const auto& nscBuf : refl.buffers)
        {
            if (nscBuf.binding == 0xFFFF || nscBuf.set >= 16u || nscBuf.totalSize == 0) continue;

            NativeConstantBufferBinding cb{};
            cb.slotIndex     = nscBuf.binding;
            cb.registerSpace = nscBuf.set;
            cb.stage         = stage;
            cb.byteSize      = nscBuf.totalSize;

            PopulateInitialCBufferBytes(nscBuf, props, fbProps, scene, timeSeconds, cb.ownedData, cb.dirty);
            cb.cpuData = cb.ownedData.data();
            outCBuffers.push_back(std::move(cb));
        }

        // 2. User Add() Textures
        for (const auto& nscTex : refl.textures)
        {
            if (nscTex.binding == 0xFFFF || nscTex.set >= 16u || nscTex.IsEngineProvided()) continue;

            NativeTextureBinding tb{};
            tb.slotIndex     = nscTex.binding;
            tb.registerSpace = nscTex.set;
            tb.stage         = stage;
            tb.texture       = 0;

            Texture* boundTex = nullptr;
            auto findTex = [&](const ShaderStageProps& sp) {
                for (const auto& mt : sp.textures)
                {
                    if (mt.resourceName == nscTex.name ||
                        (mt.propertyName + "__" + mt.resourceName) == nscTex.groupQualifiedName ||
                        (mt.slotIndex == nscTex.binding && mt.registerSpace == nscTex.set))
                    {
                        if (mt.texture)
                        {
                            boundTex   = mt.texture;
                            tb.texture = mt.texture->GetNativeHandle();
                            return;
                        }
                    }
                }
            };
            findTex(props);
            if (!boundTex) findTex(fbProps);
            if (!boundTex || tb.texture == 0)
            {
                return false;
            }
            outTextures.push_back(tb);
        }

        // 3. User Add() Samplers
        for (const auto& nscSamp : refl.samplers)
        {
            if (nscSamp.binding == 0xFFFF || nscSamp.set >= 16u) continue;
            if (nscSamp.IsEngineProvided())
            {
                return false;
            }

            NativeSamplerBinding sb{};
            sb.slotIndex     = nscSamp.binding;
            sb.registerSpace = nscSamp.set;
            sb.stage         = stage;
            sb.sampler       = 0;

            SamplerState* boundSamp = nullptr;
            auto findSamp = [&](const ShaderStageProps& sp) {
                for (const auto& ms : sp.samplers)
                {
                    if (ms.resourceName == nscSamp.name ||
                        (ms.propertyName + "__" + ms.resourceName) == nscSamp.groupQualifiedName ||
                        (ms.slotIndex == nscSamp.binding && ms.registerSpace == nscSamp.set))
                    {
                        if (ms.sampler)
                        {
                            boundSamp  = ms.sampler;
                            sb.sampler = ms.sampler->GetNativeHandle();
                            return;
                        }
                    }
                }
            };
            findSamp(props);
            if (!boundSamp) findSamp(fbProps);
            if (!boundSamp || sb.sampler == 0 || boundSamp->IsComparison() != nscSamp.isComparison)
            {
                return false;
            }
            outSamplers.push_back(sb);
        }

        // 4. StructuredBuffers (GPUScene LIVE, NVG-only CST Null SRV on STANDARD casters, and User Add() SBs)
        for (const auto& nscSb : refl.structuredBuffers)
        {
            if (nscSb.binding == 0xFFFF || nscSb.set >= 16u || nscSb.stride == 0) continue;
            const ShadowStructuredBufferRole role = ClassifyShadowStructuredBuffer(nscSb, stage);
            if (role == ShadowStructuredBufferRole::INVALID_LAYOUT)
            {
                return false;
            }
            if (role == ShadowStructuredBufferRole::INSTANCE_POOL ||
                role == ShadowStructuredBufferRole::GLOBAL_BONE_TABLE ||
                role == ShadowStructuredBufferRole::BONE_PALETTE_SLICE)
            {
                continue;
            }
            if (role == ShadowStructuredBufferRole::GPU_SCENE_TABLE)
            {
                NativeStructuredBufferBinding sbb{};
                sbb.slotIndex     = nscSb.binding;
                sbb.registerSpace = nscSb.set;
                sbb.stage         = stage;
                sbb.stride        = nscSb.stride;
                GPUBuffer* gsBuf  = scene ? scene->GetGPUSceneBuffer() : nullptr;
                const StructuredBufferLayoutPlan plan = CoreResourceBindingResolver::AnalyzeStructuredBuffer(nscSb, stage, 0u);
                if (gsBuf && gsBuf->IsValid() && plan.canDirectZeroCopy)
                {
                    const GPUSceneUpdateStats& gStat = scene->GetGPUSceneStats();
                    sbb.buffer       = gsBuf->GetNativeHandle();
                    sbb.elementCount = (gStat.highWaterMark > 0u) ? gStat.highWaterMark : gStat.capacitySlots;
                    sbb.byteSize     = sbb.elementCount * nscSb.stride;
                }
                else
                {
                    sbb.buffer       = 0;
                    sbb.elementCount = 0u;
                    sbb.byteSize     = 0u;
                }
                outStructuredBuffers.push_back(std::move(sbb));
                continue;
            }
            if (role == ShadowStructuredBufferRole::ENGINE_NULL_SRV)
            {
                // STANDARD shadow casters sharing a unified NVG/STANDARD shadow shader receive
                // 0-element CST Null Buffer SRVs for NVG-only domains without dropping the draw!
                NativeStructuredBufferBinding nullSb{};
                nullSb.slotIndex     = nscSb.binding;
                nullSb.registerSpace = nscSb.set;
                nullSb.stage         = stage;
                nullSb.buffer        = 0;
                nullSb.elementCount  = 0u;
                nullSb.stride        = nscSb.stride;
                nullSb.byteSize      = 0u;
                outStructuredBuffers.push_back(std::move(nullSb));
                continue;
            }

            const ShaderStructuredBufferSlot* matSb = nullptr;
            auto findSb = [&](const ShaderStageProps& sp) -> const ShaderStructuredBufferSlot* {
                for (const auto& ms : sp.structuredBuffers)
                {
                    if ((ms.propertyName + "__" + ms.resourceName) == nscSb.name ||
                        ms.resourceName == nscSb.name ||
                        (ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set))
                    {
                        return &ms;
                    }
                }
                return nullptr;
            };
            matSb = findSb(props);
            if (!matSb || (matSb->gpuBuffer == nullptr && matSb->gpuBufferHandle == 0 && matSb->userElementCount == 0 && matSb->elementCount == 0))
            {
                const ShaderStructuredBufferSlot* fbSb = findSb(fbProps);
                if (fbSb && (fbSb->gpuBuffer != nullptr || fbSb->gpuBufferHandle != 0 || fbSb->userElementCount > 0 || fbSb->elementCount > 0))
                {
                    matSb = fbSb;
                }
            }

            NativeStructuredBufferBinding sbb{};
            sbb.slotIndex     = nscSb.binding;
            sbb.registerSpace = nscSb.set;
            sbb.stage         = stage;
            sbb.stride        = nscSb.stride;

            const NativeHandle resolvedHandle = (matSb && matSb->gpuBuffer != nullptr)
                ? matSb->gpuBuffer->GetNativeHandle()
                : (matSb ? matSb->gpuBufferHandle : 0);
            const UINT32 resolvedStride = (matSb && matSb->gpuBuffer != nullptr)
                ? matSb->gpuBuffer->GetStride()
                : (matSb ? matSb->gpuBufferStride : 0);

            if (matSb && resolvedHandle != 0 && resolvedStride == nscSb.stride)
            {
                sbb.buffer       = resolvedHandle;
                sbb.elementCount = matSb->gpuBuffer ? matSb->gpuBuffer->GetElementCount() : matSb->gpuBufferElementCount;
                sbb.byteSize     = matSb->gpuBuffer ? matSb->gpuBuffer->GetByteSize() : matSb->gpuBufferByteSize;
                outStructuredBuffers.push_back(std::move(sbb));
            }
            else if (matSb && !matSb->cpuData.empty() && matSb->elementCount > 0)
            {
                sbb.elementCount = matSb->elementCount;
                sbb.byteSize     = static_cast<UINT32>(matSb->cpuData.size());
                sbb.ownedData    = matSb->cpuData;
                sbb.cpuData      = sbb.ownedData.data();
                outStructuredBuffers.push_back(std::move(sbb));
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    BOOL CoreNVSShadowRasterizer::BuildDrawGroupBindingTables(
        const NVSShadowRasterPassDesc&              passDesc,
        std::vector<NVSDrawGroupShaderBindingDesc>& outBindings,
        std::vector<NVSDrawGroupFieldPatch>&        outPatches,
        std::vector<UINT32>&                        outTemplateWords)
    {
        outBindings.clear();
        outPatches.clear();
        outTemplateWords.clear();

        // Reserve word 0 as zero sentinel
        outTemplateWords.push_back(0u);

        const UINT32 groupCount = static_cast<UINT32>(m_drawGroups.size());
        if (groupCount == 0u)
        {
            return true;
        }

        outBindings.resize(groupCount);

        for (UINT32 b = 0u; b < groupCount; ++b)
        {
            NVSDrawGroupShaderBindingDesc& desc = outBindings[b];
            desc                = NVSDrawGroupShaderBindingDesc{};
            desc.drawGroupIndex = b;

            const NVSShadowDrawGroup& grp = m_drawGroups[b];
            Material*     mat     = grp.material;
            ShaderVariant variant = grp.variant;
            Scene*        scene   = grp.scene;

            if (!mat || !mat->HasPass(PassType::SHADOW))
            {
                continue;
            }

            ShaderSet* shadowSet = mat->GetShaderSet(PassType::SHADOW);
            if (!shadowSet)
            {
                continue;
            }

            BOOL reqSkinning    = false;
            BOOL missingRTArray = false;
            if (!ValidateShadowShaderSetContract(
                    mat,
                    shadowSet,
                    variant,
                    m_backend,
                    passDesc.forceGeometryShaderFallback,
                    reqSkinning,
                    missingRTArray))
            {
                continue;
            }

            const Shader* userVS = shadowSet->GetVertexShader(variant);
            const Shader* userPS = shadowSet->GetPixelShader();
            const Shader* userGS = shadowSet->GetGeometryShader();
            const Shader* userHS = shadowSet->GetHullShader();
            const Shader* userDS = shadowSet->GetDomainShader();

            const Shader*     stageShaders[5] = { userVS, userPS, userGS, userHS, userDS };
            const ShaderStage stageTypes[5]   = {
                ShaderStage::VERTEX,
                ShaderStage::PIXEL,
                ShaderStage::GEOMETRY,
                ShaderStage::HULL,
                ShaderStage::DOMAIN
            };

            desc.patchFirstIndex = static_cast<UINT32>(outPatches.size());
            desc.shaderSetId     = static_cast<UINT32>(shadowSet->GetStateRevision() & 0xFFFFFFFFu);

            UINT32 promotedCBVs = 0u;

            for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
            {
                const Shader* stShader = stageShaders[stIdx];
                if (!stShader || !stShader->IsValid()) continue;

                desc.activeStageMask |= (1u << stIdx);

                const ShaderStage        sType   = stageTypes[stIdx];
                const GraphicsShaderSlot gSlot   = ShaderStageToSlot(sType);
                const ShaderStageProps&  props   = mat->GetPropertyStore().Get(PassType::SHADOW, gSlot);
                const ShaderStageProps&  fbProps = mat->GetPropertyStore().Get(PassType::GEOMETRY, gSlot);
                const ShaderReflection&  refl    = stShader->GetReflection();

                for (const auto& nscBuf : refl.buffers)
                {
                    if (nscBuf.binding == 0xFFFF || nscBuf.set >= 16u || nscBuf.totalSize == 0) continue;
                    if (!DoesCBufferRequirePerDrawPromotion(nscBuf)) continue;
                    if (promotedCBVs >= kNVSMaxPromotedIndirectCBVs) continue;

                    const UINT32 targetSlot = promotedCBVs++;
                    std::vector<BYTE> initBytes;
                    BOOL initDirty = false;
                    PopulateInitialCBufferBytes(nscBuf, props, fbProps, scene, passDesc.timeSeconds, initBytes, initDirty);

                    const UINT32 tplOffset = AppendBytesAsWords(initBytes.data(), nscBuf.totalSize, outTemplateWords);
                    if (targetSlot == 0u)
                    {
                        desc.cbv0ByteSize           = nscBuf.totalSize;
                        desc.cbv0TemplateWordOffset = tplOffset;
                    }
                    else if (targetSlot == 1u)
                    {
                        desc.cbv1ByteSize           = nscBuf.totalSize;
                        desc.cbv1TemplateWordOffset = tplOffset;
                    }
                    else
                    {
                        desc.cbv2ByteSize           = nscBuf.totalSize;
                        desc.cbv2TemplateWordOffset = tplOffset;
                    }

                    for (const auto& f : nscBuf.fields)
                    {
                        if (!f.isGet) continue;
                        const NVSShadowPatchSemantic sem = MapCBufferFieldToNVSPatchSemantic(f.propertyName);
                        if (sem != NVS_PATCH_SEMANTIC_NONE && f.offset + f.size <= nscBuf.totalSize)
                        {
                            NVSDrawGroupFieldPatch patch{};
                            patch.semanticId = sem;
                            patch.targetSlot = targetSlot;
                            patch.byteOffset = f.offset;
                            patch.byteSize   = f.size;
                            outPatches.push_back(patch);
                        }
                    }
                }

                for (const auto& nscSb : refl.structuredBuffers)
                {
                    if (nscSb.binding == 0xFFFF || nscSb.stride == 0) continue;
                    const ShadowStructuredBufferRole role = ClassifyShadowStructuredBuffer(nscSb, sType);
                    if (role == ShadowStructuredBufferRole::INSTANCE_POOL && desc.srvMode == NVS_INDIRECT_SRV_NONE)
                    {
                        desc.srvMode             = NVS_INDIRECT_SRV_INSTANCE_POOL;
                        desc.instanceStrideBytes = nscSb.stride;

                        std::vector<BYTE> instTpl(nscSb.stride, 0);
                        desc.instanceTemplateWordOffset = AppendBytesAsWords(instTpl.data(), nscSb.stride, outTemplateWords);

                        const bool allowTier1Names = IsTier1InstanceBufferName(nscSb, sType);
                        for (const auto& f : nscSb.fields)
                        {
                            const NVSShadowPatchSemantic sem = MapInstanceFieldToNVSPatchSemantic(f, allowTier1Names);
                            if (sem != NVS_PATCH_SEMANTIC_NONE && f.offset + f.size <= nscSb.stride)
                            {
                                NVSDrawGroupFieldPatch patch{};
                                patch.semanticId = sem;
                                patch.targetSlot = 3u;
                                patch.byteOffset = f.offset;
                                patch.byteSize   = f.size;
                                outPatches.push_back(patch);
                            }
                        }
                    }
                    else if (role == ShadowStructuredBufferRole::BONE_PALETTE_SLICE && desc.srvMode == NVS_INDIRECT_SRV_NONE)
                    {
                        desc.srvMode = NVS_INDIRECT_SRV_BONE_PALETTE_SLICE;
                    }
                }
            }

            desc.cbvCount     = promotedCBVs;
            desc.patchCount   = static_cast<UINT32>(outPatches.size()) - desc.patchFirstIndex;
            desc.bindingFlags = 1u;
        }

        return true;
    }

    BOOL CoreNVSShadowRasterizer::BuildShadowRasterizationPassView(
        const NVSShadowRasterPassDesc& passDesc,
        NativeHandle                   physicalPoolTexture,
        UINT32                         physicalPoolCapacity,
        GPUBuffer*                     indirectCmdBuf,
        GPUBuffer*                     bucketCountBuf,
        GPUBuffer*                     drawCtxBuf,
        GPUBuffer*                     effSceneBuf,
        GPUBuffer*                     renderDatBuf,
        GPUBuffer*                     skinPayBuf,
        UINT32                         effSceneSlots,
        LayeredDepthPassView&          outPassView,
        std::string&                   outError,
        GPUBuffer*                     perDrawCBVPoolBuf,
        GPUBuffer*                     perBatchInstancePoolBuf)
    {
        (void)drawCtxBuf;
        (void)effSceneBuf;
        (void)renderDatBuf;
        (void)effSceneSlots;

        if (!m_internalResourceManager || physicalPoolTexture == 0 || !indirectCmdBuf || !bucketCountBuf)
        {
            outError = "CoreNVSShadowRasterizer::BuildShadowRasterizationPassView: required resources missing.";
            return false;
        }

        const UINT32 maxCmdsPerBucket = (passDesc.maxCommandsPerBucketOverride > 0u &&
                                         passDesc.maxCommandsPerBucketOverride <= kNVSMaxIndirectCommandsPerBucket)
                                        ? passDesc.maxCommandsPerBucketOverride
                                        : kNVSMaxIndirectCommandsPerBucket;

        outPassView = LayeredDepthPassView{};
        outPassView.depthArrayTexture              = physicalPoolTexture;
        outPassView.arrayLayerCount                = physicalPoolCapacity;
        outPassView.viewportX                      = 0.0f;
        outPassView.viewportY                      = 0.0f;
        outPassView.viewportWidth                  = static_cast<FLOAT32>(kNVSPageTexelSize);
        outPassView.viewportHeight                 = static_cast<FLOAT32>(kNVSPageTexelSize);
        outPassView.scissorLeft                    = 0;
        outPassView.scissorTop                     = 0;
        outPassView.scissorRight                   = static_cast<INT32>(kNVSPageTexelSize);
        outPassView.scissorBottom                  = static_cast<INT32>(kNVSPageTexelSize);
        outPassView.depthBias                      = passDesc.rasterDepthBias;
        outPassView.depthBiasClamp                 = passDesc.rasterDepthBiasClamp;
        outPassView.slopeScaledDepthBias           = passDesc.rasterSlopeScaledDepthBias;
        outPassView.fallbackGeometryShader         = 0; // never inject internal GS into user shadow draws
        outPassView.indirectPerDrawCBVBuffer       = perDrawCBVPoolBuf ? perDrawCBVPoolBuf->GetNativeHandle() : 0;
        outPassView.indirectPerBatchInstanceBuffer = perBatchInstancePoolBuf ? perBatchInstancePoolBuf->GetNativeHandle() : 0;
        outPassView.selectiveClear.enabled         = false;

        m_lastPassUniquePSOCount       = 0u;
        m_lastPassActiveDrawGroupCount = 0u;

        // NVS Shadow Rasterization executes ONLY user-owned PassType::SHADOW ShaderSets.
        // If no draw groups are registered, emit 0 draw commands safely.
        if (m_drawGroups.empty())
        {
            return true;
        }

        const UINT32 availBucketCap = indirectCmdBuf->GetElementCount() / kNVSMaxIndirectCommandsPerBucket;
        const UINT32 groupCount     = (std::min)(static_cast<UINT32>(m_drawGroups.size()), availBucketCap);
        outPassView.drawCommands.reserve(groupCount);

        std::unordered_set<UINT64> uniquePSOKeys;

        for (UINT32 b = 0u; b < groupCount; ++b)
        {
            const NVSShadowDrawGroup& grp = m_drawGroups[b];
            Material*     mat     = grp.material;
            ShaderVariant variant = grp.variant;
            Scene*        scene   = grp.scene;

            if (!mat || !mat->HasPass(PassType::SHADOW))
            {
                continue;
            }

            ShaderSet* shadowSet = mat->GetShaderSet(PassType::SHADOW);
            if (!shadowSet)
            {
                continue;
            }

            BOOL reqSkinning    = false;
            BOOL missingRTArray = false;
            if (!ValidateShadowShaderSetContract(
                    mat,
                    shadowSet,
                    variant,
                    m_backend,
                    passDesc.forceGeometryShaderFallback,
                    reqSkinning,
                    missingRTArray))
            {
                continue;
            }

            const Shader* userVS = shadowSet->GetVertexShader(variant);
            const Shader* userPS = shadowSet->GetPixelShader();
            const Shader* userGS = shadowSet->GetGeometryShader();
            const Shader* userHS = shadowSet->GetHullShader();
            const Shader* userDS = shadowSet->GetDomainShader();

            DrawCommand cmd{};
            cmd.useNVSIndirectCommandSignature = true;
            cmd.indirectRootKey.enabled        = true;
            cmd.indirectArgsBuffer             = indirectCmdBuf->GetNativeHandle();
            cmd.indirectArgsOffset             = b * kNVSMaxIndirectCommandsPerBucket * static_cast<UINT32>(sizeof(NVSIndirectDrawCommand));
            cmd.maxIndirectCommandCount        = maxCmdsPerBucket;
            cmd.indirectCountBuffer            = bucketCountBuf->GetNativeHandle();
            cmd.indirectCountBufferOffset      = b * static_cast<UINT32>(sizeof(UINT32));

            cmd.vertexShader   = userVS->GetNativeHandle();
            cmd.pixelShader    = (userPS && userPS->IsValid()) ? userPS->GetNativeHandle() : 0;
            cmd.geometryShader = (userGS && userGS->IsValid()) ? userGS->GetNativeHandle() : 0;
            cmd.hullShader     = (userHS && userHS->IsValid()) ? userHS->GetNativeHandle() : 0;
            cmd.domainShader   = (userDS && userDS->IsValid()) ? userDS->GetNativeHandle() : 0;

            cmd.cullMode   = mat->GetCullMode();
            cmd.blendMode  = BlendMode::OPAQUE;
            cmd.fillMode   = mat->GetFillMode();
            cmd.depthState = mat->GetDepthState();
            if (cmd.depthState.test)
            {
                cmd.depthState.compare = (passDesc.depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
                    ? DepthCompare::GREATER_EQUAL
                    : DepthCompare::LESS_EQUAL;
            }

            GPUBuffer* bonePaletteBuf = passDesc.gpuSkinningPaletteBuffer;
            if (!bonePaletteBuf && scene && scene->GetGPUSkinningPaletteBuffer() != nullptr)
            {
                bonePaletteBuf = scene->GetGPUSkinningPaletteBuffer();
            }
            if (!bonePaletteBuf)
            {
                bonePaletteBuf = skinPayBuf;
            }
            if (bonePaletteBuf && bonePaletteBuf->IsValid() && outPassView.indirectBonePaletteBuffer == 0)
            {
                outPassView.indirectBonePaletteBuffer = bonePaletteBuf->GetNativeHandle();
            }

            const Shader*     stageShaders[5] = { userVS, userPS, userGS, userHS, userDS };
            const ShaderStage stageTypes[5]   = {
                ShaderStage::VERTEX,
                ShaderStage::PIXEL,
                ShaderStage::GEOMETRY,
                ShaderStage::HULL,
                ShaderStage::DOMAIN
            };

            BOOL resourcesValid = true;
            for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
            {
                const Shader* stShader = stageShaders[stIdx];
                if (!stShader || !stShader->IsValid()) continue;

                const ShaderStage      sType = stageTypes[stIdx];
                const ShaderReflection& refl = stShader->GetReflection();

                std::vector<NativeConstantBufferBinding>   stageCBs;
                std::vector<NativeTextureBinding>          stageTexs;
                std::vector<NativeSamplerBinding>          stageSamps;
                std::vector<NativeStructuredBufferBinding> stageSBs;

                if (!BindMaterialStageResources(
                        stShader,
                        sType,
                        mat,
                        PassType::SHADOW,
                        scene,
                        passDesc.timeSeconds,
                        stageCBs,
                        stageTexs,
                        stageSamps,
                        stageSBs))
                {
                    resourcesValid = false;
                    break;
                }

                // Promote per-draw/per-page Constant Buffers into Root CBV 0..2; keep pure Add()/scene CBs in descriptor table
                for (auto& cb : stageCBs)
                {
                    bool promoted = false;
                    for (const auto& nscBuf : refl.buffers)
                    {
                        if (nscBuf.binding == cb.slotIndex && nscBuf.set == cb.registerSpace && DoesCBufferRequirePerDrawPromotion(nscBuf))
                        {
                            if (cmd.indirectRootKey.cbvCount < kNVSMaxPromotedIndirectCBVs)
                            {
                                const UINT8 rIdx = cmd.indirectRootKey.cbvCount++;
                                cmd.indirectRootKey.cbvSpace[rIdx] = cb.registerSpace;
                                cmd.indirectRootKey.cbvSlot[rIdx]  = cb.slotIndex;
                                cmd.indirectRootKey.cbvStage[rIdx] = sType;
                                promoted = true;
                            }
                            break;
                        }
                    }
                    if (!promoted)
                    {
                        cmd.constantBuffers.push_back(std::move(cb));
                    }
                }

                for (auto& t : stageTexs)  cmd.textures.push_back(t);
                for (auto& s : stageSamps) cmd.samplers.push_back(s);
                for (auto& sb : stageSBs)  cmd.structuredBuffers.push_back(std::move(sb));

                // Bind GLOBAL_BONE_MATRIX via descriptor table; promote INSTANCE_POOL / BONE_PALETTE_SLICE to Root SRV 0
                for (const auto& nscSb : refl.structuredBuffers)
                {
                    if (nscSb.binding == 0xFFFF || nscSb.stride == 0) continue;
                    const ShadowStructuredBufferRole role = ClassifyShadowStructuredBuffer(nscSb, sType);
                    if (role == ShadowStructuredBufferRole::GLOBAL_BONE_TABLE)
                    {
                        NativeStructuredBufferBinding bsb{};
                        bsb.slotIndex     = nscSb.binding;
                        bsb.registerSpace = nscSb.set;
                        bsb.stage         = sType;
                        bsb.isUAV         = false;
                        if (bonePaletteBuf && bonePaletteBuf->IsValid())
                        {
                            bsb.buffer       = bonePaletteBuf->GetNativeHandle();
                            bsb.elementCount = bonePaletteBuf->GetElementCount();
                            bsb.stride       = bonePaletteBuf->GetStride();
                            bsb.byteSize     = bonePaletteBuf->GetByteSize();
                        }
                        else if (!reqSkinning)
                        {
                            bsb.buffer       = 0;
                            bsb.elementCount = 0u;
                            bsb.stride       = nscSb.stride;
                            bsb.byteSize     = 0u;
                        }
                        else
                        {
                            resourcesValid = false;
                            break;
                        }
                        cmd.structuredBuffers.push_back(std::move(bsb));
                    }
                    else if ((role == ShadowStructuredBufferRole::INSTANCE_POOL ||
                              role == ShadowStructuredBufferRole::BONE_PALETTE_SLICE) &&
                             !cmd.indirectRootKey.hasPromotedSRV)
                    {
                        cmd.indirectRootKey.hasPromotedSRV = true;
                        cmd.indirectRootKey.srvSpace       = nscSb.set;
                        cmd.indirectRootKey.srvSlot        = nscSb.binding;
                        cmd.indirectRootKey.srvStage       = sType;
                    }
                }

                if (!resourcesValid)
                {
                    break;
                }
            }

            if (!resourcesValid)
            {
                continue;
            }

            uniquePSOKeys.insert(ComputePSOCompatibilityKey(mat, shadowSet, variant));
            m_lastPassActiveDrawGroupCount++;
            outPassView.drawCommands.push_back(std::move(cmd));
        }

        m_lastPassUniquePSOCount = static_cast<UINT32>(uniquePSOKeys.size());
        return true;
    }
}
