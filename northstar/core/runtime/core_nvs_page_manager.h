#pragma once

#include "../../include/types/northstar_types.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/platform/frame_view.h"
#include "../../include/buffer/gpu_buffer.h"
#include "../../include/material/material.h"
#include "../../include/scene/scene.h"
#include "../../include/shader/shader.h"
#include "core_buffer_manager.h"
#include "core_compute_manager.h"
#include "core_render_pipeline.h"
#include "core_nvs_shadow_rasterizer.h"
#include "internal_resource_manager.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Northstar
{
    class CoreNVSPageManager
    {
    public:
        CoreNVSPageManager() = default;
        ~CoreNVSPageManager();

        BOOL Initialize(
            NorthstarBackend*        backend,
            CoreBufferManager*       bufferManager,
            CoreComputeManager*      computeManager,
            CoreRenderPipeline*      renderPipeline,
            InternalResourceManager* internalResourceManager,
            const NVSPageManagerConfig& config = NVSPageManagerConfig{});

        void Shutdown();

        BOOL IsInitialized() const { return m_initialized; }
        const std::string& GetLastError() const { return m_lastError; }
        const NVSPageManagerConfig& GetConfig() const { return m_config; }

        BOOL IsSparseHierarchyEnabled() const { return m_config.enableSparseHierarchy; }
        UINT32 GetTotalVirtualPageCount() const
        {
            if (m_config.enableSparseHierarchy)
            {
                return m_sparseBlockCapacity * kNVSSparsePagesPerBlock;
            }
            return m_config.maxLightCount * m_config.clipmapLevelCount *
                   m_config.pagesPerAxis * m_config.pagesPerAxis;
        }
        UINT32 GetPhysicalPoolCapacity() const { return m_config.physicalPoolCapacity; }
        void SetMinInFlightProtectFrames(UINT32 protectFrames) { m_config.minInFlightProtectFrames = protectFrames; }
        UINT32 GetMinInFlightProtectFrames() const { return m_config.minInFlightProtectFrames; }
        NativeHandle GetPhysicalPoolTextureHandle() const { return m_physicalPoolTexture; }

        GPUBuffer* GetVirtualPageTableBuffer() const { return m_virtualPageTableBuffer; }
        GPUBuffer* GetPhysicalPageTableBuffer() const { return m_physicalPageTableBuffer; }
        GPUBuffer* GetPageRequestBuffer() const { return m_pageRequestBuffer; }
        GPUBuffer* GetDirtyPhysicalSlicesBuffer() const { return m_dirtyPhysicalSlicesBuffer; }
        GPUBuffer* GetDirtySliceCountBuffer() const { return m_dirtySliceCountBuffer; }
        GPUBuffer* GetSelectiveClearIndirectArgsBuffer() const { return m_selectiveClearIndirectArgsBuffer; }
        GPUBuffer* GetTelemetryBuffer() const { return m_telemetryBuffer; }
        GPUBuffer* GetSparsePageHierarchyBuffer() const { return m_sparsePageHierarchyBuffer; }
        GPUBuffer* GetMissRecordsBuffer() const { return m_missRecordsBuffer; }
        GPUBuffer* GetInvalidationCommandsBuffer() const { return m_invalidationCommandsBuffer; }
        UINT32     GetSparseRootCapacity() const { return m_sparseRootCapacity; }
        UINT32     GetSparseDirCapacity() const { return m_sparseDirCapacity; }
        UINT32     GetSparseBlockCapacity() const { return m_sparseBlockCapacity; }

        // Per-Page Projection & Clipmap Level Table Buffers (active frame slot or explicit slot [0..2])
        GPUBuffer* GetClipmapLevelTableBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetPageRenderDataBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetActiveRenderPageListBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetProjectionTelemetryBuffer(Camera* camera = nullptr) const;

        GPUBuffer* GetClipmapLevelTableBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetPageRenderDataBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetActiveRenderPageListBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetProjectionTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        UINT32     GetActiveProjectionFrameSlot(Camera* camera = nullptr) const;

        // Per-Page Caster Ranges, Compacted Caster Worklist & Telemetry Buffers
        GPUBuffer* GetCasterPageRangeBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetCasterWorklistBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetCasterCullTelemetryBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetWorklistIndirectArgsBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetFilteredCasterBuffer() const { return m_filteredCasterBuffer; }
        GPUBuffer* GetCasterValidBitmaskBuffer() const { return m_casterValidBitmaskBuffer; }
        GPUBuffer* GetDynamicInvalidationBoxesBuffer() const { return m_dynamicInvalidationBoxesBuffer; }

        GPUBuffer* GetCasterPageRangeBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetCasterWorklistBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetCasterCullTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetWorklistIndirectArgsBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        UINT32     GetCasterWorklistCapacity(Camera* camera = nullptr) const;
        BOOL       ResizeCasterWorklistCapacity(UINT32 newMaxWorklistCapacity, Camera* camera = nullptr);

        // GPU Mesh Draw Descriptors, Material Shadow Descriptors & Indirect Draw Buffers
        GPUBuffer* GetMeshDrawDescriptorBuffer() const { return m_meshDrawDescriptorBuffer; }
        GPUBuffer* GetMaterialShadowDescriptorBuffer() const { return m_materialShadowDescriptorBuffer; }
        GPUBuffer* GetSkinningPayloadBuffer() const { return m_skinningPayloadBuffer; }
        GPUBuffer* GetShadowDrawContextBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetIndirectDrawCommandBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetBucketCommandCountBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetPageCommitRecordBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetIndirectDrawTelemetryBuffer(Camera* camera = nullptr) const;

        GPUBuffer* GetShadowDrawContextBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetIndirectDrawCommandBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetBucketCommandCountBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetPageCommitRecordBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetIndirectDrawTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera = nullptr) const;
        GPUBuffer* GetPerDrawCBVWordsBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetPerBatchInstanceWordsBuffer(Camera* camera = nullptr) const;
        GPUBuffer* GetAllocationCountersBuffer(Camera* camera = nullptr) const;

        // Dynamic Committed Shadow View Table Snapshot & CPU Mirror Accessors
        GPUBuffer*                    GetCommittedSamplingViewTableBuffer(Camera* camera = nullptr) const;
        const NVSClipmapLevelGPUData* GetCpuClipmapLevelTable(Camera* camera = nullptr) const;
        const NVSClipmapLevelGPUData* GetCommittedCpuViewTable(Camera* camera = nullptr) const;
        UINT32                        GetActiveViewTableCount(Camera* camera = nullptr) const;
        UINT32                        GetAllocatedViewTableCapacity(Camera* camera = nullptr) const;
        UINT32                        GetCommittedViewTableCount(Camera* camera = nullptr) const;
        BOOL                          HasCommittedShadowSnapshot(Camera* camera = nullptr) const;
        UINT32                        GetCommittedSnapshotFrameIndex(Camera* camera = nullptr) const;
        BOOL                          CommitSamplingViewTableSnapshot(UINT32 currentFrame, Camera* camera = nullptr);
        void                          ResetCommittedShadowSnapshots();

        // Reconfigure / resize physical pool capacity (64 .. 2048 slices) safely
        BOOL ResizePhysicalPool(UINT32 newPhysicalPoolCapacity);

        // Reconfigure virtual grid or full config
        BOOL Reconfigure(const NVSPageManagerConfig& newConfig);

        // Reset all virtual & physical page tables on the GPU (MODE 3)
        BOOL ResetTablesOnGPU();

        // Upload page request array into m_pageRequestBuffer
        BOOL UploadPageRequests(const NVSVirtualPageRequest* requests, UINT32 requestCount);

        // Sparse Virtual Page Hierarchy growth & O(1) GPU batch view invalidation / reclamation
        BOOL EnsureSparseRootCapacity(UINT32 minRootViews);
        BOOL EnsureSparsePoolCapacities(UINT32 minDirCapacity, UINT32 minBlockCapacity);
        BOOL PollSparsePoolHeadroomAndGrowIfNeeded();
        BOOL PollResidencyOverflowNonBlocking(UINT32& outOverflowPages, UINT32& outDirtySlices, BOOL& outFreshSample);
        void QueueInvalidationCommand(const NVSInvalidationCommand& cmd);
        void QueueInvalidationCommands(const NVSInvalidationCommand* cmds, UINT32 count);
        BOOL HasPendingInvalidationCommands() const { return !m_pendingInvalidationCommands.empty(); }
        void ClearPendingInvalidationCommands() { m_pendingInvalidationCommands.clear(); }
        BOOL BuildPendingInvalidationDispatches(
            UINT32                              currentFrame,
            std::vector<ComputeDispatchCommand>& outDispatches,
            Camera*                             camera = nullptr);
        BOOL ExecutePendingInvalidationsOnGPU(UINT32 currentFrame = 0u, Camera* camera = nullptr);

        // Upload directional clipmap hierarchy or dynamic shadow view table (full or incremental subrange)
        BOOL EnsureViewTableCapacity(UINT32 minEntries, Camera* camera = nullptr);
        BOOL UploadClipmapHierarchy(
            const NVSClipmapHierarchyState& hierarchy,
            Camera*                         camera = nullptr);

        BOOL UploadClipmapLevelTable(
            const NVSClipmapLevelGPUData* entries,
            UINT32                        entryCount,
            Camera*                       camera = nullptr);

        BOOL UpdateClipmapLevelTableRange(
            const NVSClipmapLevelGPUData* entries,
            UINT32                        startSlot,
            UINT32                        entryCount,
            UINT32                        totalActiveCount,
            Camera*                       camera = nullptr);

        void SetActiveProjectionFrame(UINT32 currentFrame, Camera* camera = nullptr);
        BOOL FlushViewTableToGPU(Camera* camera = nullptr);

        // Build a ComputeDispatchCommand for RuntimeNVSPageResidency
        BOOL BuildResidencyComputeDispatch(
            NVSResidencyDispatchMode    dispatchMode,
            const NVSResidencyPassDesc& passDesc,
            ComputeDispatchCommand&     outCmd,
            GPUBuffer*                  overridePageRequestBuffer = nullptr,
            Camera*                     camera                    = nullptr);

        // Build a ComputeDispatchCommand for RuntimeNVSPageProjection
        BOOL BuildPageProjectionComputeDispatch(
            NVSProjectionDispatchMode        dispatchMode,
            const NVSPageProjectionPassDesc& passDesc,
            ComputeDispatchCommand&          outCmd,
            Camera*                          camera = nullptr);

        // Execute GPU per-page projection & render metadata pass
        BOOL ExecutePageProjectionPass(
            const NVSPageProjectionPassDesc& passDesc = NVSPageProjectionPassDesc{},
            Camera*                          camera   = nullptr);

        // GPU-Driven Shadow Caster Culling, Worklist Compaction & Dynamic Invalidation
        BOOL EnsureFilteredCasterCapacity(UINT32 minSceneSlots);

        BOOL UploadDynamicCasterInvalidationBoxes(
            const NVSDynamicCasterInvalidationBox* boxes,
            UINT32                                 boxCount);

        BOOL BuildCasterCullComputeDispatch(
            NVSCasterCullDispatchMode    dispatchMode,
            GPUBuffer*                   gpuSceneBuffer,
            UINT32                       sceneSlotCount,
            const NVSCasterCullPassDesc& passDesc,
            ComputeDispatchCommand&      outCmd,
            Camera*                      camera                      = nullptr,
            UINT32                       dynamicInvalidationBoxCount = 0u,
            GPUBuffer*                   overrideClipmapTableBuffer  = nullptr);

        BOOL BuildCasterCullPipelineDispatches(
            GPUBuffer*                             gpuSceneBuffer,
            UINT32                                 sceneSlotCount,
            const NVSCasterCullPassDesc&           passDesc,
            std::vector<ComputeDispatchCommand>&   outDispatches,
            Camera*                                camera          = nullptr,
            const NVSDynamicCasterInvalidationBox* dynamicBoxes    = nullptr,
            UINT32                                 dynamicBoxCount = 0u);

        BOOL ExecuteCasterCullPass(
            GPUBuffer*                             gpuSceneBuffer,
            UINT32                                 sceneSlotCount,
            const NVSCasterCullPassDesc&           passDesc        = NVSCasterCullPassDesc{},
            Camera*                                camera          = nullptr,
            const NVSDynamicCasterInvalidationBox* dynamicBoxes    = nullptr,
            UINT32                                 dynamicBoxCount = 0u);

        BOOL ExecuteDynamicCasterInvalidationPass(
            const NVSDynamicCasterInvalidationBox* boxes,
            UINT32                                 boxCount,
            Camera*                                camera                      = nullptr,
            BOOL                                   invalidateAllResidentStates = false);

        BOOL VerifyMatrixClipTransformOnGPU(
            const Vec3*                           worldPoints,
            const UINT32*                         physicalPageIndices,
            UINT32                                pointCount,
            GPUBuffer*                            gpuSceneBuffer,
            std::vector<NVSFilteredCasterObject>& outResults,
            Camera*                               camera = nullptr);

        // GPU Mesh Draw Descriptors, Material Shadow Descriptors & ExecuteIndirect Rasterization
        BOOL RegisterOrUpdateMeshDrawDescriptor(const GPUMeshDrawDescriptor& desc);
        BOOL RegisterOrUpdateMeshVariantDescriptor(UINT32 slotIndex, const GPUMeshDrawDescriptor& desc);
        BOOL RegisterOrUpdateMeshVariantDescriptor(
            UINT32                       baseMeshId,
            UINT32                       submeshIndex,
            UINT32                       lodLevel,
            const GPUMeshDrawDescriptor& desc);
        BOOL MarkMeshDrawDescriptorDestroyed(UINT32 meshId);
        BOOL UploadMeshDrawDescriptorTable(
            const GPUMeshDrawDescriptor* entries,
            UINT32                       entryCount,
            UINT32                       startSlot = 0u);
        BOOL RegisterMeshFromHandles(
            UINT32       meshId,
            NativeHandle vertexBufferHandle,
            NativeHandle indexBufferHandle,
            UINT32       vertexCount,
            UINT32       indexCount,
            UINT32       startIndex      = 0u,
            INT32        baseVertex      = 0,
            UINT32       submeshIndex    = 0u,
            UINT32       lodLevel        = 0u,
            UINT32       descriptorFlags = GPU_MESH_DRAW_DESC_FLAG_VALID,
            UINT32       meshGeneration  = 1u);

        BOOL RegisterOrUpdateMaterialShadowDescriptor(const GPUMaterialShadowDescriptor& desc);
        BOOL MarkMaterialShadowDescriptorDestroyed(UINT32 materialId);
        BOOL UploadMaterialShadowDescriptorTable(
            const GPUMaterialShadowDescriptor* entries,
            UINT32                             entryCount,
            UINT32                             startSlot = 0u);

        BOOL UploadSkinningPayload(UINT32 payloadSlot, const NVSShadowSkinningPayload& payload);
        BOOL FlushDescriptorTablesToGPU();

        void ClearShadowBucketMaterials();
        void SetShadowBucketMaterial(
            UINT32        bucketIndex,
            Material*     material,
            ShaderVariant variant = ShaderVariant::STANDARD,
            Scene*        scene   = nullptr);

        CoreNVSShadowRasterizer&       GetShadowRasterizer() { return m_shadowRasterizer; }
        const CoreNVSShadowRasterizer& GetShadowRasterizer() const { return m_shadowRasterizer; }

        BOOL HasRegisteredMeshDescriptor(UINT32 meshId) const noexcept
        {
            return meshId < m_cpuMeshDescriptors.size() &&
                   (m_cpuMeshDescriptors[meshId].meshGeneration != 0u ||
                    (m_cpuMeshDescriptors[meshId].descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_DESTROYED) != 0u);
        }

        BOOL HasRegisteredMaterialDescriptor(UINT32 materialId) const noexcept
        {
            return materialId < m_cpuMaterialDescriptors.size() &&
                   (m_cpuMaterialDescriptors[materialId].materialGeneration != 0u ||
                    (m_cpuMaterialDescriptors[materialId].materialFlags & (GPU_MAT_SHADOW_FLAG_DESTROYED_DEPENDENCY | GPU_MAT_SHADOW_FLAG_MISSING_VARIANT)) != 0u);
        }

        BOOL BuildIndirectDrawComputeDispatch(
            NVSIndirectDrawDispatchMode    dispatchMode,
            GPUBuffer*                     gpuSceneBuffer,
            UINT32                         sceneSlotCount,
            const NVSShadowRasterPassDesc& passDesc,
            ComputeDispatchCommand&        outCmd,
            Camera*                        camera = nullptr);

        BOOL BuildIndirectDrawPipelineDispatches(
            GPUBuffer*                           gpuSceneBuffer,
            UINT32                               sceneSlotCount,
            const NVSShadowRasterPassDesc&       passDesc,
            std::vector<ComputeDispatchCommand>& outDispatches,
            Camera*                              camera = nullptr);

        BOOL BuildShadowRasterizationPassView(
            GPUBuffer*                     gpuSceneBuffer,
            UINT32                         sceneSlotCount,
            const NVSShadowRasterPassDesc& passDesc,
            LayeredDepthPassView&          outPassView,
            Camera*                        camera = nullptr);

        BOOL BuildGPUCommitRenderedPagesDispatch(
            UINT32                  currentFrame,
            ComputeDispatchCommand& outCmd,
            Camera*                 camera = nullptr);

        BOOL ExecuteShadowRasterizationPass(
            GPUBuffer*                     gpuSceneBuffer,
            UINT32                         sceneSlotCount,
            const NVSShadowRasterPassDesc& passDesc = NVSShadowRasterPassDesc{},
            Camera*                        camera   = nullptr);

        // Build the LayeredDepthPassView for Selective Layered Depth Clear of dirty physical pages
        BOOL BuildSelectiveClearPassView(
            FLOAT32               clearDepth,
            LayeredDepthPassView& outClearPass) const;

        // Execute the GPU residency pass and (optionally) GPU-Driven Selective Layered Depth Clear
        // in a single submission without any intermediate CPU readback.
        BOOL ExecuteResidencyPass(const NVSResidencyPassDesc& passDesc);

        // Transition ALLOCATED_NOT_CLEARED (1) -> CLEARED_NOT_RENDERED (2) on the GPU
        BOOL MarkClearedPagesOnGPU(UINT32 currentFrame);

        // Transition CLEARED_NOT_RENDERED (2) -> VALID_SHADOW_DATA (3) on the GPU
        // NOTE: Promotes EVERY cleared page unconditionally. Never use in the production frame path;
        //       production code must use CommitRenderedPagesOnGPU (page validity contract).
        BOOL PromoteClearedPagesToValidOnGPU(UINT32 currentFrame);

        // Page validity contract: per-page, generation-checked CLEARED_NOT_RENDERED -> VALID_SHADOW_DATA
        // commit (NVS_DISPATCH_COMMIT_RENDERED_PAGES). Records are uploaded into the PageRequests buffer.
        BOOL CommitRenderedPagesOnGPU(
            const NVSPageRenderCommitRecord* records,
            UINT32                           recordCount,
            UINT32                           currentFrame);

        // Clear specific physical slices in the 128x128 Texture2DArray pool to a specified depth value
        // using the GPU selective layered depth clear pipeline (used to seed resident pages in tests).
        BOOL ClearSpecificPhysicalSlicesOnGPU(
            const UINT32* sliceIndices,
            UINT32        sliceCount,
            FLOAT32       depthValue);

        // Camera lifecycle notification (releases per-camera 3-frame-in-flight projection ring buffers)
        void OnCameraDestroyed(Camera* camera);
        UINT32 GetActiveCameraProjectionSlotCount() const;

        // Synchronous GPU readback helpers for verification & diagnostics
        BOOL ReadbackTelemetrySync(NVSResidencyTelemetry& outTelemetry) const;
        BOOL ReadbackVirtualPageTableSync(std::vector<NVSVirtualPageEntry>& outEntries) const;
        BOOL ReadbackPhysicalPageTableSync(std::vector<NVSPhysicalPageEntry>& outEntries) const;
        BOOL ReadbackDirtySlicesSync(std::vector<UINT32>& outSlices, UINT32 outCount = 0u) const;
        BOOL ReadbackDirtySliceStatusSync(UINT32 outStatus[4]) const;
        BOOL ReadbackSelectiveClearIndirectArgsSync(UINT32 outArgs[4]) const;

        // Synchronous GPU readback helpers for per-page projection & render metadata
        BOOL ReadbackPageRenderDataSync(
            std::vector<NVSPageRenderData>& outEntries,
            Camera*                         camera = nullptr) const;
        BOOL ReadbackActiveRenderPageListSync(
            std::vector<UINT32>& outPhysicalIndices,
            UINT32               outCount = 0u,
            Camera*              camera   = nullptr) const;
        BOOL ReadbackProjectionTelemetrySync(
            NVSPageProjectionTelemetry& outTelemetry,
            Camera*                     camera = nullptr) const;
        BOOL ReadbackClipmapLevelTableSync(
            std::vector<NVSClipmapLevelGPUData>& outTable,
            Camera*                              camera = nullptr) const;

        // Synchronous GPU readback helpers for shadow caster culling & compacted worklists
        BOOL ReadbackCasterCullTelemetrySync(
            NVSCasterCullTelemetry& outTelemetry,
            Camera*                 camera = nullptr) const;
        BOOL ReadbackCasterPageRangesSync(
            std::vector<NVSCasterPageRange>& outRanges,
            UINT32                           outCount = 0u,
            Camera*                          camera   = nullptr) const;
        BOOL ReadbackCasterWorklistSync(
            std::vector<NVSCasterDrawCandidate>& outCandidates,
            UINT32                               outCount = 0u,
            Camera*                              camera   = nullptr) const;
        BOOL ReadbackWorklistIndirectArgsSync(
            UINT32  outArgs[8],
            Camera* camera = nullptr) const;
        BOOL ReadbackFilteredCastersSync(
            std::vector<NVSFilteredCasterObject>& outFiltered,
            UINT32                                outCount = 0u) const;

        // Synchronous GPU readback helpers for indirect draw commands, draw contexts & depth slices
        BOOL ReadbackIndirectDrawTelemetrySync(
            NVSIndirectDrawTelemetry& outTelemetry,
            Camera*                   camera = nullptr) const;
        BOOL ReadbackBucketCommandCountsSync(
            UINT32  outCounts[16],
            Camera* camera = nullptr) const;
        BOOL ReadbackIndirectDrawCommandsSync(
            UINT32                               bucketIndex,
            std::vector<NVSIndirectDrawCommand>& outCommands,
            UINT32                               outCount = 0u,
            Camera*                              camera   = nullptr) const;
        BOOL ReadbackShadowDrawContextsSync(
            std::vector<NVSShadowDrawContext>& outContexts,
            UINT32                             outCount = 0u,
            Camera*                            camera   = nullptr) const;
        BOOL ReadbackPageCommitRecordsSync(
            std::vector<NVSPageRenderCommitRecord>& outRecords,
            UINT32                                  outCount = 0u,
            Camera*                                 camera   = nullptr) const;
        BOOL ReadbackPhysicalPoolSliceSync(
            UINT32                sliceIndex,
            std::vector<FLOAT32>& outDepthTexels) const;
        BOOL ReadbackSparseHierarchyMetricsSync(
            NVSSparseHierarchyMetrics& outMetrics,
            Camera*                    camera = nullptr) const;
        BOOL ReadbackSparsePageHierarchySync(
            std::vector<UINT32>& outNodes) const;
        UINT64 GetMetadataVramBytes() const noexcept;

    private:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
        struct CameraProjectionSlot
        {
            Camera*                camera                    = nullptr;
            BOOL                   allocated                 = false;
            UINT32                 activeSlot                = 0u;
            UINT32                 worklistCapacity              = 65536u;
            UINT32                 allocatedWorklistCapacity     = 0u;
            UINT32                 allocatedShadowBucketCapacity = 0u;
            UINT32                 activeViewTableCount          = kNVSClipmapTableMaxEntries;
            UINT32                 committedViewTableCount       = 0u;
            UINT32                 allocatedViewTableCapacity    = 0u;
            UINT32                 dirtyMinViewSlot[kNVSProjectionFrameSlots]     = { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
            UINT32                 dirtyMaxViewSlot[kNVSProjectionFrameSlots]     = { 0u, 0u, 0u };
            UINT32                 committedDirtyMin[kNVSProjectionFrameSlots]    = { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
            UINT32                 committedDirtyMax[kNVSProjectionFrameSlots]    = { 0u, 0u, 0u };
            GPUBuffer*             clipmapLevelTableBuffers[kNVSProjectionFrameSlots]     = { nullptr, nullptr, nullptr };
            GPUBuffer*             pageRenderDataBuffers[kNVSProjectionFrameSlots]        = { nullptr, nullptr, nullptr };
            GPUBuffer*             activeRenderPageListBuffers[kNVSProjectionFrameSlots]  = { nullptr, nullptr, nullptr };
            GPUBuffer*             projectionTelemetryBuffers[kNVSProjectionFrameSlots]   = { nullptr, nullptr, nullptr };
            GPUBuffer*             casterPageRangeBuffers[kNVSProjectionFrameSlots]       = { nullptr, nullptr, nullptr };
            GPUBuffer*             casterWorklistBuffers[kNVSProjectionFrameSlots]        = { nullptr, nullptr, nullptr };
            GPUBuffer*             casterCullTelemetryBuffers[kNVSProjectionFrameSlots]   = { nullptr, nullptr, nullptr };
            GPUBuffer*             worklistIndirectArgsBuffers[kNVSProjectionFrameSlots]  = { nullptr, nullptr, nullptr };
            GPUBuffer*             shadowDrawContextBuffers[kNVSProjectionFrameSlots]     = { nullptr, nullptr, nullptr };
            GPUBuffer*             indirectDrawCommandBuffers[kNVSProjectionFrameSlots]   = { nullptr, nullptr, nullptr };
            GPUBuffer*             bucketCommandCountBuffers[kNVSProjectionFrameSlots]    = { nullptr, nullptr, nullptr };
            GPUBuffer*             pageCommitRecordBuffers[kNVSProjectionFrameSlots]      = { nullptr, nullptr, nullptr };
            GPUBuffer*             indirectDrawTelemetryBuffers[kNVSProjectionFrameSlots] = { nullptr, nullptr, nullptr };
            GPUBuffer*             drawGroupBindingsBuffers[kNVSProjectionFrameSlots]     = { nullptr, nullptr, nullptr };
            GPUBuffer*             drawGroupFieldPatchesBuffers[kNVSProjectionFrameSlots] = { nullptr, nullptr, nullptr };
            GPUBuffer*             templateWordsBuffers[kNVSProjectionFrameSlots]         = { nullptr, nullptr, nullptr };
            GPUBuffer*             perDrawCBVWordsBuffers[kNVSProjectionFrameSlots]       = { nullptr, nullptr, nullptr };
            GPUBuffer*             perBatchInstanceWordsBuffers[kNVSProjectionFrameSlots] = { nullptr, nullptr, nullptr };
            GPUBuffer*             allocationCountersBuffers[kNVSProjectionFrameSlots]    = { nullptr, nullptr, nullptr };
            GPUBuffer*             committedSamplingViewTableBuffers[kNVSProjectionFrameSlots] = { nullptr, nullptr, nullptr };
            UINT32                 committedActiveSlot                                    = 0u;
            BOOL                   hasCommittedShadowSnapshot                             = false;
            UINT32                 committedSnapshotFrameIndex                            = 0u;
            UINT32                 lastUploadedDrawGroupCount                             = 0u;
            std::vector<NVSClipmapLevelGPUData> cpuClipmapTable;
            std::vector<NVSClipmapLevelGPUData> committedCpuClipmapTable;
        };
#ifdef _MSC_VER
#pragma warning(pop)
#endif

        BOOL CreateResources(const NVSPageManagerConfig& cfg);
        void DestroyResources();
        BOOL AllocateProjectionSlotResources(CameraProjectionSlot& slot, const NVSPageManagerConfig& cfg);
        void DestroyProjectionSlotResources(CameraProjectionSlot& slot);
        BOOL EnsureSlotViewTableCapacity(CameraProjectionSlot& slot, UINT32 minViewCapacity);
        BOOL FlushSlotViewTableBuffer(CameraProjectionSlot& slot, UINT32 fSlot);
        BOOL EnsureSlotShadowBucketCapacity(CameraProjectionSlot& slot, UINT32 minBuckets);
        BOOL UploadSlotIndirectBindingTables(CameraProjectionSlot& slot, UINT32 fSlot, const NVSShadowRasterPassDesc& passDesc);
        BOOL EnsureMaterialShadowDescriptorCapacity(UINT32 minSlots);
        BOOL EnsureMeshDrawDescriptorCapacity(UINT32 minSlots);
        void InitializeDefaultCpuClipmapTable(std::vector<NVSClipmapLevelGPUData>& outTable, const NVSPageManagerConfig& cfg) const;
        BOOL BuildResidencyComputeDispatchInternal(
            NVSResidencyDispatchMode    dispatchMode,
            const NVSResidencyPassDesc& passDesc,
            ComputeDispatchCommand&     outCmd,
            GPUBuffer*                  overridePageRequestBuffer,
            UINT32                      subMode,
            UINT32                      invCmdCount,
            UINT32                      dispatchGroupsX,
            UINT32                      dispatchGroupsY = 1u,
            Camera*                     camera          = nullptr);
        CameraProjectionSlot* AcquireCameraProjectionSlot(Camera* camera);
        const CameraProjectionSlot* FindCameraProjectionSlot(Camera* camera) const;

        NorthstarBackend*        m_backend                          = nullptr;
        CoreBufferManager*       m_bufferManager                    = nullptr;
        CoreComputeManager*      m_computeManager                   = nullptr;
        CoreRenderPipeline*      m_renderPipeline                   = nullptr;
        InternalResourceManager* m_internalResourceManager          = nullptr;
        NVSPageManagerConfig     m_config{};
        BOOL                     m_initialized                      = false;
        std::string              m_lastError;

        NativeHandle             m_physicalPoolTexture              = 0;
        GPUBuffer*               m_virtualPageTableBuffer           = nullptr;
        GPUBuffer*               m_physicalPageTableBuffer          = nullptr;
        GPUBuffer*               m_pageRequestBuffer                = nullptr;
        GPUBuffer*               m_dirtyPhysicalSlicesBuffer        = nullptr;
        GPUBuffer*               m_dirtySliceCountBuffer            = nullptr;
        GPUBuffer*               m_selectiveClearIndirectArgsBuffer = nullptr;
        GPUBuffer*               m_telemetryBuffer                  = nullptr;
        GPUBuffer*               m_sparsePageHierarchyBuffer        = nullptr;
        GPUBuffer*               m_invalidationCommandsBuffer       = nullptr;
        GPUBuffer*               m_missRecordsBuffer                = nullptr;
        UINT32                   m_sparseRootCapacity               = 0u;
        UINT32                   m_sparseDirCapacity                = 0u;
        UINT32                   m_sparseBlockCapacity              = 0u;
        std::vector<NVSInvalidationCommand> m_pendingInvalidationCommands;
        GPUBuffer*               m_seedCandidateBuffer              = nullptr;
        GPUBuffer*               m_filteredCasterBuffer             = nullptr;
        UINT32                   m_filteredCasterCapacity           = 0u;
        GPUBuffer*               m_casterValidBitmaskBuffer         = nullptr;
        GPUBuffer*               m_dynamicInvalidationBoxesBuffer   = nullptr;
        GPUBuffer*               m_dummySceneObjectBuffer           = nullptr;

        GPUBuffer*                               m_meshDrawDescriptorBuffer       = nullptr;
        GPUBuffer*                               m_materialShadowDescriptorBuffer = nullptr;
        GPUBuffer*                               m_skinningPayloadBuffer          = nullptr;
        std::vector<GPUMeshDrawDescriptor>       m_cpuMeshDescriptors;
        std::vector<GPUMaterialShadowDescriptor> m_cpuMaterialDescriptors;
        std::vector<NVSShadowSkinningPayload>    m_cpuSkinningPayloads;
        BOOL                                     m_meshDescriptorsDirty           = false;
        BOOL                                     m_materialDescriptorsDirty       = false;
        BOOL                                     m_skinningPayloadsDirty          = false;

        CoreNVSShadowRasterizer                  m_shadowRasterizer;
        BOOL                                     m_suppressSingleBindingUpload    = false;
        BOOL                                     m_sparseHeadroomReadbackInFlight = false;
        BOOL                                     m_dirtyCountReadbackInFlight     = false;
        UINT32                                   m_lastResidencyOverflowPages     = 0u;
        UINT32                                   m_lastResidencyDirtySlices       = 0u;

        CameraProjectionSlot              m_defaultProjectionSlot{};
        std::vector<CameraProjectionSlot> m_cameraProjectionSlots;
    };
}
