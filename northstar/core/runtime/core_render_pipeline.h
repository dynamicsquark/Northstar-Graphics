#pragma once

#include "../../include/platform/frame_view.h"
#include "../../include/stats/northstar_stats.h"
#include "../../include/scene/scene.h"
#include "../../include/camera/camera.h"
#include "../../include/light/light.h"
#include "../../include/render_object/render_object.h"
#include "../../include/shader/shader_property_store.h"
#include "../../include/material/material.h"
#include "../../include/compute/compute_kernel.h"
#include "../../include/mesh/mesh.h"
#include "../../include/lod_group/lod_group.h"
#include "../../include/shader_set/shader_set.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "internal_resource_manager.h"
#include "core_resource_binding_resolver.h"
#include <unordered_set>

namespace Northstar
{
    struct LightNVSViewAssignment
    {
        const Light* light           = nullptr;
        UINT32       shadowViewBase  = 0u;
        UINT32       shadowViewCount = 0u;
        UINT32       nvsLightId      = 0u;
    };

    struct FrameNVSSamplingContext
    {
        NativeHandle                        shadowMapTexture                      = 0;
        NativeHandle                        comparisonSampler                     = 0;
        NativeHandle                        depthSampler                          = 0;
        NativeHandle                        sparsePageHierarchyBuffer             = 0;
        UINT32                              sparsePageHierarchyElementCount       = 0u;
        UINT32                              sparseRootCapacity                    = 0u;
        UINT32                              sparseDirStride                       = kNVSSparseDirStride;
        UINT32                              sparseDirCapacity                     = 0u;
        Vec4                                screenPixelWorldScale                 = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        NativeHandle                        virtualPageTableBuffer                = 0;
        UINT32                              virtualPageTableElementCount          = 0u;
        NativeHandle                        physicalPageTableBuffer               = 0;
        UINT32                              physicalPageTableElementCount         = 0u;
        UINT32                              pagesPerAxis                          = 64u;
        UINT32                              physicalPoolCapacity                  = 0u;
        UINT32                              pageTexelSize                         = 128u;
        UINT32                              depthMode                             = 0u;
        BOOL                                requiresDepthPrepass                  = false;

        NativeHandle                        opaqueLightViewTableBuffer            = 0;
        UINT32                              opaqueLightViewTableElementCount      = 0u;
        UINT32                              opaqueActiveViewCount                 = 0u;
        UINT32                              opaqueSamplingAvailable               = 0u;
        UINT32                              opaqueSnapshotFrameIndex              = 0u;
        const NVSClipmapLevelGPUData*       opaqueViewTableCpuMirror              = nullptr;
        UINT32                              opaqueViewTableCpuMirrorCount         = 0u;
        std::vector<LightNVSViewAssignment> opaqueLightAssignments;
    };

    class CoreRenderPipeline
    {
    public:
        CoreRenderPipeline() = default;
        ~CoreRenderPipeline() = default;

        void SetResourceMaterializer(CoreGPUResourceMaterializer* materializer) { m_resourceMaterializer = materializer; }
        CoreGPUResourceMaterializer* GetResourceMaterializer() const { return m_resourceMaterializer; }

        void BuildFrameView(Scene* scene, Camera* camera, const NativeRenderTarget& defaultRenderTarget,
                            RenderStats& stats, FrameView& outView,
                            const FrameNVSSamplingContext* nvsContext = nullptr,
                            std::unordered_set<const ComputeKernel*>* executedPerFrameKernels = nullptr,
                            const InternalResourceManager* internalResources = nullptr);

        BOOL BuildComputeDispatchCommand(const ComputeKernel* kernel,
                                         const FrameView& frameView,
                                         ComputeDispatchCommand& outCmd,
                                         std::vector<ComputeDispatchCommand>* outPreDispatches = nullptr);

        void RefreshFrameViewSceneContext(Scene* scene, Camera* camera,
                                          const NativeRenderTarget& defaultRenderTarget,
                                          FrameView& ioView,
                                          const FrameNVSSamplingContext* nvsContext = nullptr);

        BOOL BuildDrawCommand(RenderObject* obj, PassType pass, const FrameView& frameView, const Vec3& cameraPos,
                              DrawCommand& outCmd, ShaderVariant overrideVertexVariant = ShaderVariant::COUNT,
                              const InternalResourceManager* internalResources = nullptr,
                              std::vector<ComputeDispatchCommand>* outPreDrawDispatches = nullptr,
                              const Scene* scene = nullptr);

    private:
        void ApplyNVSSamplingPhase(Scene* scene, FrameView& ioView, const FrameNVSSamplingContext* nvsContext);
        void BuildCameraData(Camera* camera, const NativeRenderTarget& defaultRenderTarget, FrameCameraData& outCamera);
        void BuildEnvironmentData(Scene* scene, FrameEnvironmentData& outEnv);
        void BuildDirectionalLightData(Scene* scene, BOOL& outHasDirLight, FrameLightData& outDirLight, const std::vector<LightNVSViewAssignment>* nvsAssignments = nullptr);
        void BuildLightsData(Scene* scene, std::vector<FrameLightData>& outLights, const std::vector<LightNVSViewAssignment>* nvsAssignments = nullptr);
        Mesh* ResolveEffectiveMesh(RenderObject* obj, const FrameCameraData& cameraData) const;
        void BatchAndCategorizeDrawCommands(Scene* scene,
                                            const std::vector<RenderObject*>& objects, PassType pass,
                                            const FrameView& frameView, const Vec3& cameraPos,
                                            RenderStats& stats,
                                            std::vector<DrawCommand>& outCommands,
                                            const InternalResourceManager* internalResources = nullptr,
                                            std::vector<ComputeDispatchCommand>* outPreDrawDispatches = nullptr);

        struct CachedEngineLightsSBEntry
        {
            UINT64                             lightsRevision = 0u;
            UINT64                             layoutHash     = 0u;
            UINT64                             sharedDataId   = 0u;
            std::unique_ptr<std::vector<BYTE>> payload;
        };

        void PopulateSharedEngineLightsPayload(
            const NSL::NSCStructuredBuffer& nscSb,
            const FrameView&                frameView,
            NativeStructuredBufferBinding&  ioSbb);

        CoreGPUResourceMaterializer*           m_resourceMaterializer   = nullptr;
        UINT64                                 m_nextLightsDataRevision = 0u;
        UINT64                                 m_nextSharedSbDataId     = 0u;
        std::vector<CachedEngineLightsSBEntry> m_cachedLightsPayloads;
    };
}
