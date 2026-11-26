#pragma once

#include "../../include/types/northstar_types.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/platform/frame_view.h"
#include "../../include/buffer/gpu_buffer.h"
#include "../../include/material/material.h"
#include "../../include/scene/scene.h"
#include "../../include/shader/shader.h"
#include "../../include/shader_set/shader_set.h"
#include "core_buffer_manager.h"
#include "internal_resource_manager.h"

#include <string>
#include <vector>

namespace Northstar
{
    struct NVSShadowDrawGroup
    {
        Material*     material             = nullptr;
        ShaderSet*    shadowShaderSet      = nullptr;
        ShaderVariant variant              = ShaderVariant::STANDARD;
        Scene*        scene                = nullptr;
        UINT64        psoCompatibilityKey  = 0u;
        UINT64        resourceBindingKey   = 0u;
        UINT64        shaderSetRevision    = 0u;
        BOOL          hasUserResourceBinds = false;
    };

    class CoreNVSShadowRasterizer
    {
    public:
        CoreNVSShadowRasterizer() = default;
        ~CoreNVSShadowRasterizer() = default;

        void Initialize(
            NorthstarBackend*        backend,
            CoreBufferManager*       bufferManager,
            InternalResourceManager* internalResourceManager);

        void Shutdown();

        // Reset draw groups for a new scene descriptor synchronization
        void ClearDrawGroups();

        // Invalidate draw groups referencing destroyed engine objects (prevents dangling pointer dereference)
        void OnMaterialDestroyed(const Material* material);
        void OnShaderSetDestroyed(const ShaderSet* shaderSet);
        void OnSceneDestroyed(const Scene* scene);

        // Mark whether a production Scene is active
        void SetProductionSceneActive(BOOL active) { m_productionSceneActive = active; }
        BOOL IsProductionSceneActive() const { return m_productionSceneActive; }

        // Explicit bucket assignment for direct NVS draw group mapping
        void SetShadowBucketMaterial(UINT32 bucketIndex, Material* material, ShaderVariant variant, Scene* scene = nullptr);

        // Dynamic draw group registration separating PSO compatibility from Material resource bindings
        UINT32 RegisterOrAcquireDrawGroup(Material* material, ShaderVariant variant, Scene* scene);

        UINT32 GetActiveDrawGroupCount() const { return static_cast<UINT32>(m_drawGroups.size()); }
        const std::vector<NVSShadowDrawGroup>& GetDrawGroups() const { return m_drawGroups; }
        BOOL HasExplicitDrawGroups() const { return m_hasExplicitDrawGroups; }

        // Static / helper validation of VS / GS / DS / PS contract for NVS Shadow ShaderSets
        static BOOL DoesShaderDeclareRenderTargetArrayIndexOutput(const Shader* shader);

        BOOL ValidateShadowShaderSetContract(
            const Material*         material,
            const ShaderSet*        shadowSet,
            ShaderVariant           variant,
            const NorthstarBackend* backend,
            BOOL                    forceGSFallback,
            BOOL&                   outRequiresSkinningData,
            BOOL&                   outMissingRTArrayIndex);

        static UINT64 ComputePSOCompatibilityKey(
            const Material*  material,
            const ShaderSet* shadowSet,
            ShaderVariant    variant);

        static UINT64 ComputeMaterialResourceBindingKey(
            const Material*  material,
            const ShaderSet* shadowSet,
            ShaderVariant    variant);

        // Shared stage resource binding helper usable across graphics stages (VS, PS, GS, HS, DS)
        static BOOL BindMaterialStageResources(
            const Shader*                               shader,
            ShaderStage                                 stage,
            const Material*                             material,
            PassType                                    primaryPass,
            Scene*                                      scene,
            FLOAT32                                     timeSeconds,
            std::vector<NativeConstantBufferBinding>&   outCBuffers,
            std::vector<NativeTextureBinding>&          outTextures,
            std::vector<NativeSamplerBinding>&          outSamplers,
            std::vector<NativeStructuredBufferBinding>& outStructuredBuffers);

        // Builds GPU-driven per-draw-group shader binding metadata, field patch descriptors,
        // and initial CPU CBuffer/Instance template words for RuntimeNVSIndirectDraw.nsl.
        BOOL BuildDrawGroupBindingTables(
            const NVSShadowRasterPassDesc&              passDesc,
            std::vector<NVSDrawGroupShaderBindingDesc>& outBindings,
            std::vector<NVSDrawGroupFieldPatch>&        outPatches,
            std::vector<UINT32>&                        outTemplateWords);

        // Build LayeredDepthPassView for ExecuteIndirect NVS Shadow Page Rasterization
        BOOL BuildShadowRasterizationPassView(
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
            GPUBuffer*                     perDrawCBVPoolBuf       = nullptr,
            GPUBuffer*                     perBatchInstancePoolBuf = nullptr);

        // Diagnostics & Telemetry
        UINT32 GetRejectedMissingRTArrayIndexCount() const { return m_rejectedMissingRTArrayIndexCount; }
        void   ResetRejectedMissingRTArrayIndexCount() { m_rejectedMissingRTArrayIndexCount = 0u; }
        UINT32 GetLastPassUniquePSOCount() const { return m_lastPassUniquePSOCount; }
        UINT32 GetLastPassActiveDrawGroupCount() const { return m_lastPassActiveDrawGroupCount; }

    private:
        NorthstarBackend*        m_backend                 = nullptr;
        CoreBufferManager*       m_bufferManager           = nullptr;
        InternalResourceManager* m_internalResourceManager = nullptr;

        std::vector<NVSShadowDrawGroup> m_drawGroups;
        BOOL                            m_hasExplicitDrawGroups            = false;
        BOOL                            m_productionSceneActive            = false;
        UINT32                          m_rejectedMissingRTArrayIndexCount = 0u;
        UINT32                          m_lastPassUniquePSOCount           = 0u;
        UINT32                          m_lastPassActiveDrawGroupCount     = 0u;
    };
}
