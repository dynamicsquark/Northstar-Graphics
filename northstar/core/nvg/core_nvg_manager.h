#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../../include/mesh/mesh_desc.h"
#include "../../include/material/material.h"
#include "../../include/scene/scene.h"
#include "../../include/camera/camera.h"
#include "../../include/platform/frame_view.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "../../include/stats/northstar_stats.h"
#include "../runtime/core_buffer_manager.h"
#include "../runtime/core_texture_manager.h"
#include "../runtime/core_compute_manager.h"
#include "../runtime/internal_resource_manager.h"
#include "../runtime/core_render_pipeline.h"
#include "../runtime/core_resource_binding_resolver.h"
#include "core_nvg_residency_manager.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Northstar
{
    enum class NVGInternalExecutionStrategy : UINT8
    {
        VERTEX_PULL = 0, // Zero-decode indirect vertex pulling (no IA vertex attribs, reads NVG pulling buffers)
        DECODED_IA  = 1  // GPU-decoded IA vertex/index stream (+ live NVG/GPUScene SBs for hybrid & procedural shaders)
    };

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
    struct NVGPromotedCBVTemplateSpan
    {
        const NSL::NSCBuffer* nscBuf             = nullptr;
        ShaderStage           stage              = ShaderStage::VERTEX;
        UINT32                templateWordOffset = 0u;
        UINT32                byteSize           = 0u;
    };

    struct NVGDrawBucketEntry
    {
        UINT32                                  bucketIndex               = 0u;
        Material*                               material                  = nullptr;
        ShaderSet*                              shaderSet                 = nullptr;
        PassType                                passType                  = PassType::GEOMETRY;
        ShaderVariant                           variant                   = ShaderVariant::STANDARD;
        BOOL                                    isVertexPull              = false; // True only for unindexed raw Path-A
        BOOL                                    isIndexedPull             = false; // Path-A Indexed Cluster Pull (0 VB words, unpacked IB)
        BOOL                                    isTransparent             = false;
        BOOL                                    isShadowPass              = false;
        BOOL                                    isDepthPass               = false;
        BOOL                                    isDepthCompanion          = false;
        BOOL                                    hasRejectedCustomDepth    = false;
        UINT32                                  depthCompanionBucketIndex = kNVGInvalidIndex;
        UINT32                                  decodeVertexWords         = 28u;
        UINT32                                  decodeAttrMask            = 0x1FFu;
        IndirectRootBindingKey                  indirectRootKey{};
        NVGDrawGroupShaderBindingDesc           bindingDesc{};
        std::vector<NVGPromotedCBVTemplateSpan> promotedCBVSpans;
    };
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    class CoreNVGManager
    {
    public:
        CoreNVGManager() = default;
        ~CoreNVGManager();

        BOOL Initialize(
            NorthstarBackend*        backend,
            CoreBufferManager*       bufferManager,
            CoreTextureManager*      textureManager,
            CoreComputeManager*      computeManager,
            InternalResourceManager* internalResources);

        void Shutdown();

        BOOL IsInitialized() const { return m_initialized; }

        BOOL             SetConfig(const NVGConfig& config);
        const NVGConfig& GetConfig() const { return m_config; }
        const NVGStats&  GetStats()  const { return m_stats; }
        NVGStats&        GetMutableStats() { return m_stats; }

        CoreNVGResidencyManager&       GetResidencyManager()       { return m_residencyManager; }
        const CoreNVGResidencyManager& GetResidencyManager() const { return m_residencyManager; }
        UINT64                         GetResidencyRevision() const { return m_residencyManager.GetResidencyRevision(); }
        void                           SetResourceMaterializer(CoreGPUResourceMaterializer* materializer) { m_resourceMaterializer = materializer; }

        // Virtual mesh build / deserialize + residency registration
        BOOL CreateAndRegisterVirtualMesh(
            const MeshDesc&                          desc,
            UINT32                                   meshId,
            NVGMeshInfo&                             outInfo,
            AABB&                                    outLocalBounds,
            std::string&                             outError,
            std::shared_ptr<INVGPageStreamProvider>* outResolvedStreamProvider = nullptr);

        void UnregisterVirtualMesh(UINT32 meshId);
        BOOL SerializeVirtualMesh(UINT32 meshId, std::vector<BYTE>& outBlob) const;

        // Explicit deterministic compatibility checker (zero silent fallback)
        NVGCompatibilityStatus CheckMaterialCompatibility(
            const Material* material,
            PassType        pass,
            BOOL            isSkinned,
            std::string*    outReason = nullptr) const;

        // Per-frame streaming, telemetry readback polling, and scene material/bucket sync
        void BeginFrameTick(
            UINT64                                        frameIndex,
            Scene*                                        activeScene,
            std::vector<NVSDynamicCasterInvalidationBox>& outNvsInvalidations);

        BOOL HasVirtualObjectsInScene(const Scene* scene) const;

        // Injects full GPU-driven NVG compute + graphics passes (Depth Prepass ON/OFF,
        // HZB occlusion, two-phase disocclusion, transparent sorting) into FrameView.
        BOOL InjectCameraNVGPasses(
            Scene*                    scene,
            Camera*                   camera,
            const NativeRenderTarget& defaultRenderTarget,
            FrameView&                ioFrameView,
            GPUBuffer*                gpuSceneBuffer,
            UINT32                    gpuSceneSlotCount,
            CoreRenderPipeline*       renderPipeline,
            RenderStats&              stats,
            BOOL                      isFirstCameraInFrame = true,
            BOOL                      isFinalCameraInFrame = true);

        // Injects independent NVS + NVG shadow caster culling, shadow-texel LOD,
        // compaction/decode, and shadow draw commands directly into NVS's LayeredDepthPassView.
        BOOL InjectNVSShadowPasses(
            Scene*                         scene,
            Camera*                        camera,
            const NVSShadowRasterPassDesc& rasterDesc,
            GPUBuffer*                     gpuSceneBuffer,
            UINT32                         gpuSceneSlotCount,
            GPUBuffer*                     projCounterBuf,
            GPUBuffer*                     activeRenderPageListBuf,
            GPUBuffer*                     pageRenderDataBuf,
            UINT32                         activePageCap,
            std::vector<FramePassItem>&    outPreRasterItems,
            LayeredDepthPassView&          ioShadowPassView,
            CoreRenderPipeline*            renderPipeline,
            const FrameView&               cameraFrameView,
            GPUBuffer*                     pageCommitRecordsBuf = nullptr);

        // Queues async readbacks (page faults + GPU telemetry) after ExecuteFrame submits.
        void OnAfterExecuteFrame(Camera* camera);

        void AccumulateBackendTiming(const BackendFrameMetrics& bm);

        void OnCameraDestroyed(const Camera* camera);
        void OnMaterialDestroyed(const Material* material);
        void OnShaderSetDestroyed(const ShaderSet* shaderSet);
        void OnSceneDestroyed(const Scene* scene);
        void OnResize();

        BOOL                    SetMockCapabilitiesOverride(BOOL enableOverride, const BackendCapabilities& mockCaps);
        BOOL                    DebugReloadRasterKernels(
            BOOL    bypassDiskCache,
            BYTE*   outAtomic32Bytecode,
            UINT32  maxAtomic32Bytes,
            UINT32* outAtomic32ActualBytes,
            BYTE*   outAtomic64Bytecode,
            UINT32  maxAtomic64Bytes,
            UINT32* outAtomic64ActualBytes);
        BackendCapabilities     GetEffectiveCapabilities() const noexcept;
        NVGRasterAtomicStrategy ResolveActiveCameraRasterStrategy() const noexcept;

    private:
        struct CameraNVGContext
        {
            const Camera* camera            = nullptr;
            Texture*      hzbTexture        = nullptr;
            UINT32        hzbWidth          = 0u;
            UINT32        hzbHeight         = 0u;
            UINT32        hzbMipCount       = 0u;
            UINT32        lastViewportW     = 0u;
            UINT32        lastViewportH     = 0u;
            BOOL          hasValidPrevHZB   = false;
            Mat4          prevViewMatrix    = Mat4::Identity();
            Mat4          prevProjMatrix    = Mat4::Identity();
            Mat4          prevViewProj      = Mat4::Identity();
            Vec3          prevCameraPos     = Vec3::Zero();
            UINT64        lastSceneMutationEpoch = 0ull;
            UINT64        lastResidencyRevision  = 0ull;
            Vec3          lodHysteresisPos       = Vec3::Zero();
            BOOL          hasLodHysteresis  = false;
            BOOL          telemetryPending        = false;
            BOOL          hasLastTelemetry        = false;
            UINT32        lastTelemetryWords[48]  = {};
            BOOL          usedThisFrame           = false;
            BOOL          cameraPassRanThisFrame  = false;
            BOOL          lastUsedSplitAtomic32   = false;

            // Compute Software Micropoly Rasterizer isolated resources (Decision 4C)
            Texture*      swVisTexture            = nullptr;
            Texture*      swDepthTexture          = nullptr;
            UINT32        swTexWidth              = 0u;
            UINT32        swTexHeight             = 0u;
            GPUBuffer*    visBuffer64             = nullptr;
            GPUBuffer*    swAtomicDepthKey32      = nullptr;
            GPUBuffer*    swAtomicPayload32       = nullptr;
            GPUBuffer*    swClusterList           = nullptr;
            GPUBuffer*    swScheduledMask         = nullptr;
            GPUBuffer*    swWinningClusterMask    = nullptr;

            // Autonomous dynamic GPU queue & Path-B pool capacities (2x growth on watermark/overflow)
            UINT32        dynNodeQueueCap         = 524288u;
            UINT32        dynCandidateCap         = 1048576u;
            UINT32        dynVisibleCap           = 524288u;
            UINT32        dynShadowVisibleCap     = 1048576u;
            UINT32        dynOccludedClusterCap   = 524288u;
            UINT32        dynDecodeClusterCap     = 32768u;
            UINT32        dynCameraCmdsPerBucket  = 65536u;
            UINT32        dynShadowCmdsPerBucket  = 65536u;
            UINT32        dynCBVWords             = 8388608u;

            // GPU working buffers for culling, compaction, Path-B decode, indirect draw & telemetry
            GPUBuffer* nodeQueueA               = nullptr;
            GPUBuffer* nodeQueueB               = nullptr;
            GPUBuffer* candidateClusters        = nullptr;
            GPUBuffer* occludedObjects          = nullptr;
            GPUBuffer* occludedNodes            = nullptr;
            GPUBuffer* occludedClusters         = nullptr;
            GPUBuffer* rawVisibleClusters       = nullptr;
            GPUBuffer* compactedVisibleClusters = nullptr;
            GPUBuffer* bucketRanges             = nullptr;
            GPUBuffer* bucketCounts             = nullptr;
            GPUBuffer* bucketWriteCursors       = nullptr;
            GPUBuffer* pathAIndirectArgs        = nullptr;
            GPUBuffer* queueCounters            = nullptr;
            GPUBuffer* dispatchArgs             = nullptr;
            GPUBuffer* decodedClusterTable      = nullptr;
            GPUBuffer* uniqueDecodeJobs         = nullptr;
            GPUBuffer* decodedVertexPool        = nullptr;
            GPUBuffer* decodedIndexPool         = nullptr;
            GPUBuffer* indirectDrawCommands     = nullptr;
            GPUBuffer* perDrawCBVWords          = nullptr;
            GPUBuffer* perBatchInstanceWords    = nullptr;
            GPUBuffer* bucketCommandCounts      = nullptr;
            GPUBuffer* telemetryBuffer          = nullptr;

            // Dedicated Phase-2 disocclusion draw output buffers (preserves Phase-1 camera Opaque & Transparent draws!)
            GPUBuffer* p2CompactedVisible       = nullptr;
            GPUBuffer* p2BucketRanges           = nullptr;
            GPUBuffer* p2PathAIndirectArgs      = nullptr;
            GPUBuffer* p2IndirectCommands       = nullptr;
            GPUBuffer* p2BucketCmdCounts        = nullptr;
            GPUBuffer* p2PerDrawCBVWords        = nullptr;
            GPUBuffer* p2PerBatchInstWords      = nullptr;
            GPUBuffer* p2DecodedVertexPool      = nullptr;
            GPUBuffer* p2DecodedIndexPool       = nullptr;

            // Dedicated shadow working buffers (so camera visible clusters stay intact across NVS shadow pass!)
            GPUBuffer* shadowPageViews          = nullptr;
            GPUBuffer* shadowViewMasks          = nullptr;
            GPUBuffer* shadowPageMasks          = nullptr;
            GPUBuffer* shadowRawVisible         = nullptr;
            GPUBuffer* shadowCompactedVisible   = nullptr;
            GPUBuffer* shadowBucketRanges       = nullptr;
            GPUBuffer* shadowBucketCounts       = nullptr;
            GPUBuffer* shadowBucketWriteCursors = nullptr;
            GPUBuffer* shadowPathAIndirectArgs  = nullptr;
            GPUBuffer* shadowIndirectCommands   = nullptr;
            GPUBuffer* shadowBucketCmdCounts    = nullptr;
            GPUBuffer* shadowPerDrawCBVWords    = nullptr;
            GPUBuffer* shadowPerBatchInstWords  = nullptr;
            GPUBuffer* shadowDecodedVertexPool  = nullptr;
            GPUBuffer* shadowDecodedIndexPool   = nullptr;
            GPUBuffer* shadowDecodedClusterTbl  = nullptr;
            GPUBuffer* shadowUniqueDecodeJobs   = nullptr;
            GPUBuffer* shadowSWDepthKeyPool     = nullptr;
            GPUBuffer* shadowSWClusterList      = nullptr;
            GPUBuffer* shadowSWScheduledMask    = nullptr;
            GPUBuffer* shadowSWFallbackMask     = nullptr;
            GPUBuffer* shadowSWPageNeedsTile    = nullptr;
            GPUBuffer* shadowSWPageTileMap      = nullptr;
            GPUBuffer* shadowSWDirtyTileSlices  = nullptr;
            GPUBuffer* shadowSWMergeIndirectArgs= nullptr;

            UINT32 frameTagCounter = 1u;
        };

        BOOL EnsureRuntimeKernelsLoaded();
        BOOL EnsureCameraContextResources(CameraNVGContext& ctx, UINT32 vpW, UINT32 vpH, UINT32 gpuSceneSlotCount = 0u);
        void DestroyCameraContextResources(CameraNVGContext& ctx);
        BOOL EnsureBuffer(GPUBuffer*& buf, UINT32 stride, UINT32 elemCount, GPUBufferUsage usage);

        void SyncSceneMaterialsAndBuckets(Scene* scene);
        UINT32 RegisterDrawBucket(
            Material*     material,
            PassType      passType,
            ShaderVariant variant,
            BOOL          isVertexPull,
            BOOL          isTransparent,
            BOOL          isShadowPass,
            BOOL          isDepthPass,
            Scene*        scene,
            FLOAT32       timeSeconds);

        BOOL BuildReflectedBindingForBucket(
            NVGDrawBucketEntry&                  bucket,
            Scene*                               scene,
            FLOAT32                              timeSeconds,
            std::vector<NVSDrawGroupFieldPatch>& ioPatches,
            std::vector<UINT32>&                 ioTemplateWords);

        void RefreshBucketTemplatesForView(
            const FrameView& frameView,
            Scene*           scene,
            BOOL             isShadowPass);

        BOOL AppendHZBBuildDispatches(
            CameraNVGContext&           ctx,
            Texture*                    sceneDepthTex,
            UINT32                      vpX,
            UINT32                      vpY,
            UINT32                      vpW,
            UINT32                      vpH,
            BOOL                        isReversedZ,
            std::vector<FramePassItem>& outItems,
            Texture*                    swDepthTex = nullptr);

        BOOL AppendResidencyMaskDispatch(std::vector<FramePassItem>& outItems);

        BOOL AppendPrepareDispatchArgs(
            CameraNVGContext&           ctx,
            UINT32                      srcCounterIdx,
            UINT32                      dstArgsWordOffset,
            UINT32                      groupSizeDivisor,
            UINT32                      resetNodeCounterIdx,
            std::vector<FramePassItem>& outItems,
            ComputeDispatchProfileGroup profileGroup);

        BOOL BuildVertexPullDrawCommand(
            const NVGDrawBucketEntry&            bucket,
            const FrameView&                     frameView,
            Scene*                               scene,
            GPUBuffer*                           gpuSceneBuffer,
            GPUBuffer*                           bucketRangesBuf,
            GPUBuffer*                           compactedVisBuf,
            GPUBuffer*                           pathAIndirectArgsBuf,
            GPUBuffer*                           shadowPageViewsBuf,
            UINT64                               passRevision,
            BOOL                                 depthWriteOverride,
            DepthCompare                         depthCompareOverride,
            BOOL                                 applyDepthOverride,
            std::vector<ComputeDispatchCommand>& outMaterializeDispatches,
            DrawCommand&                         outCmd);

        BOOL BuildDecodedIADrawCommand(
            const NVGDrawBucketEntry&            bucket,
            const FrameView&                     frameView,
            Scene*                               scene,
            GPUBuffer*                           gpuSceneBuffer,
            GPUBuffer*                           bucketRangesBuf,
            GPUBuffer*                           compactedVisBuf,
            GPUBuffer*                           shadowPageViewsBuf,
            GPUBuffer*                           indirectCmdBuf,
            GPUBuffer*                           bucketCmdCountBuf,
            GPUBuffer*                           decodedVertexPoolBuf,
            GPUBuffer*                           decodedIndexPoolBuf,
            GPUBuffer*                           perDrawCBVBuf,
            GPUBuffer*                           perBatchInstBuf,
            UINT32                               maxCmdsPerBucket,
            UINT64                               passRevision,
            BOOL                                 depthWriteOverride,
            DepthCompare                         depthCompareOverride,
            BOOL                                 applyDepthOverride,
            std::vector<ComputeDispatchCommand>& outMaterializeDispatches,
            DrawCommand&                         outCmd);

        FLOAT32 ResolveEffectiveScreenErrorPixels() const noexcept;
        FLOAT32 ResolveEffectiveShadowTexelError() const noexcept;

        static NVGInternalExecutionStrategy PlanExecutionStrategy(
            const ShaderSet*         shaderSet,
            ShaderVariant            variant,
            const NVGMaterialPolicy& policy,
            BOOL*                    outUsesIAVertexAttribs = nullptr,
            BOOL*                    outUsesNVGPullBuffers  = nullptr);

        NorthstarBackend*            m_backend              = nullptr;
        CoreBufferManager*           m_bufferManager        = nullptr;
        CoreTextureManager*          m_textureManager       = nullptr;
        CoreComputeManager*          m_computeManager       = nullptr;
        InternalResourceManager*     m_internalRes          = nullptr;
        CoreGPUResourceMaterializer* m_resourceMaterializer = nullptr;

        CoreNVGResidencyManager  m_residencyManager;
        NVGConfig                m_config{};
        NVGStats                 m_stats{};

        BOOL   m_initialized        = false;
        BOOL   m_kernelsLoaded      = false;
        UINT64 m_currentFrameIndex  = 0ull;
        UINT64 m_lastMaskRevision   = 0ull;
        UINT32 m_maxMeshBVHDepth    = 1u;

        // Incremental material/bucket sync cache (O(M) steady-state instead of O(N) per-frame scene scan)
        struct CachedSyncMaterialState
        {
            const Material* material       = nullptr;
            UINT32          matRev         = 0u;
            UINT64          geomShaderRev  = 0ull;
            UINT64          depthShaderRev = 0ull;
            UINT64          shShaderRev    = 0ull;
            UINT32          gpuMatId       = 0u;
        };
        const Scene*                         m_cachedSyncScene      = nullptr;
        SIZE_T                               m_cachedSyncObjCount   = 0u;
        UINT64                               m_cachedSyncSceneEpoch = ~0ull;
        std::vector<CachedSyncMaterialState> m_cachedSyncMaterials;

        // Cached feature-dependent internal NVG compute kernels & shadow SW merge shaders
        ComputeKernel* m_kHZBBuild          = nullptr;
        ComputeKernel* m_kResidencyMask     = nullptr;
        ComputeKernel* m_kInstanceCull      = nullptr;
        ComputeKernel* m_kHierarchyCull     = nullptr;
        ComputeKernel* m_kCompactionDecode  = nullptr;
        ComputeKernel* m_kIndirectDraw      = nullptr;
        ComputeKernel* m_kShadowCull              = nullptr;
        ComputeKernel* m_kShadowMicropolyRaster   = nullptr;
        ComputeKernel* m_kMicropolyRasterAtomic64 = nullptr;
        ComputeKernel* m_kMicropolyRasterAtomic32 = nullptr;
        ComputeKernel* m_kDepthExport             = nullptr;
        Shader*        m_shadowSWMergeVS          = nullptr;
        Shader*        m_shadowSWMergeGS          = nullptr;
        Shader*        m_shadowSWMergePS          = nullptr;
        BOOL                m_hasMockCapsOverride = false;
        BackendCapabilities m_mockCapabilities{};

        // Global per-frame material / draw-bucket / reflection tables
        std::vector<NVGMaterialGPUData>            m_materialTable;
        std::vector<NVGDrawBucketEntry>            m_cameraBuckets;
        std::vector<NVGDrawBucketEntry>            m_shadowBuckets;
        std::vector<UINT32>                        m_cameraBucketFlags;
        std::vector<UINT32>                        m_shadowBucketFlags;
        std::vector<NVGDrawGroupShaderBindingDesc> m_cameraBindingDescs;
        std::vector<NVSDrawGroupFieldPatch>        m_cameraFieldPatches;
        std::vector<UINT32>                        m_cameraTemplateWords;
        std::vector<NVGDrawGroupShaderBindingDesc> m_shadowBindingDescs;
        std::vector<NVSDrawGroupFieldPatch>        m_shadowFieldPatches;
        std::vector<UINT32>                        m_shadowTemplateWords;

        GPUBuffer* m_materialTableBuffer    = nullptr;
        GPUBuffer* m_cameraBucketFlagsBuf   = nullptr;
        GPUBuffer* m_shadowBucketFlagsBuf   = nullptr;
        GPUBuffer* m_cameraBindingsBuf      = nullptr;
        GPUBuffer* m_cameraPatchesBuf       = nullptr;
        GPUBuffer* m_cameraTemplatesBuf     = nullptr;
        GPUBuffer* m_shadowBindingsBuf      = nullptr;
        GPUBuffer* m_shadowPatchesBuf       = nullptr;
        GPUBuffer* m_shadowTemplatesBuf     = nullptr;
        GPUBuffer* m_dummyCommitRecordsBuf  = nullptr;
        GPUBuffer* m_dummySkinnedBoundsBuf  = nullptr;
        GPUBuffer* m_dummyObjectMatSlotsBuf = nullptr;
        GPUBuffer* m_dummyBonePaletteBuf    = nullptr;
        GPUBuffer* m_dummySWMaskBuf         = nullptr;
        BOOL       m_sceneHasAnimatedDeformation = false;

        // Shadow configuration & per-material shadow policy tracking for automatic NVS cache invalidation
        BOOL    m_hasLastShadowConfigState = false;
        BOOL    m_lastShadowEnabled        = true;
        FLOAT32 m_lastShadowBaseTexelError = 1.0f;
        FLOAT32 m_lastThinShadowBoost      = 3.0f;
        std::unordered_map<const Material*, UINT32> m_lastMaterialShadowHash;

        std::unordered_map<const Camera*, std::unique_ptr<CameraNVGContext>> m_cameraContexts;
        BOOL m_residencyMaskDispatchedThisFrame = false;
    };
}
