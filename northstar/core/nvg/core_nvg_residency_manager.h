#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../runtime/core_buffer_manager.h"
#include "nvg_builder.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Northstar
{
    struct NVGResidencyChangedRegion
    {
        UINT32  meshId                     = 0u;
        AABB    localBounds                = {};
        FLOAT32 maxShadowScaledParentError = 1.0e20f;
    };

    class CoreNVGResidencyManager
    {
    public:
        CoreNVGResidencyManager() = default;
        ~CoreNVGResidencyManager();

        BOOL Initialize(
            NorthstarBackend*  backend,
            CoreBufferManager* bufferManager,
            const NVGConfig&   config);

        void Shutdown();

        BOOL IsInitialized() const { return m_initialized; }

        // Updates runtime budgets/pool sizing cleanly (may re-create physical pool if capacity changed).
        BOOL ApplyConfig(const NVGConfig& config, UINT64 currentFrameIndex);

        // Registers a built/deserialized virtual mesh asset under Mesh::GetGPUSceneMeshId().
        // Immediately allocates & uploads pinned Root Pages without CPU fence stalls.
        BOOL RegisterVirtualMesh(
            UINT32                    meshId,
            NVGVirtualMeshAsset&&     asset,
            const INVGPageDataSource* customSource,
            UINT64                    currentFrameIndex,
            std::string&              outError);

        // Invalidates virtual mesh metadata & places its physical pages into fence-safe deferred retirement.
        void UnregisterVirtualMesh(UINT32 meshId, UINT64 currentFrameIndex);

        BOOL HasVirtualMesh(UINT32 meshId) const;
        BOOL GetVirtualMeshBounds(UINT32 meshId, AABB& outLocalBounds) const;
        BOOL SerializeVirtualMesh(UINT32 meshId, std::vector<BYTE>& outBlob) const;

        // Per-frame non-blocking streaming tick:
        //  1) Reclaims fence-safe retired physical slots.
        //  2) Syncs dirty global metadata buffers.
        //  3) Polls async GPU page-fault readback, propagates ancestor priorities & LRU,
        //     elastically grows physical pool within budget, and runs strict leaf-first eviction.
        //  4) Returns meshIds & localized sub-mesh AABBs whose page residency changed.
        void ProcessStreamingFrame(
            UINT64                                  frameIndex,
            std::vector<UINT32>&                    outResidencyChangedMeshIds,
            NVGStats&                               ioStats,
            std::vector<NVGResidencyChangedRegion>* outChangedRegions = nullptr);

        // Schedules async GPU->CPU readback of m_pageRequestBuffer after NVG compute dispatches complete.
        void QueuePageRequestReadback();

        // Returns true ONLY when m_pageRequestBuffer was just read back or resized, allowing frames
        // while a readback is in flight to accumulate InterlockedMax requests losslessly.
        BOOL ConsumeClearPageRequestsFlag() noexcept
        {
            const BOOL clearNow = m_clearRequestsNextFrame;
            m_clearRequestsNextFrame = false;
            return clearNow;
        }

        FLOAT32 GetAdaptiveErrorScale() const noexcept { return m_adaptiveErrorScale; }
        UINT32  GetMaxPoolBudgetPages() const noexcept { return m_maxPoolBudgetPages; }
        UINT32  GetCurrentUploadBudget() const noexcept { return m_lastUploadBudgetPages; }

        // Ensures global metadata GPU buffers reflect any newly registered/unregistered meshes before dispatch.
        BOOL SyncDirtyMetadataToGPU();

        // Buffer accessors for GPU compute & vertex-pulling shaders
        GPUBuffer* GetPhysicalPagePoolBuffer()  const { return m_physicalPagePoolBuffer; }
        GPUBuffer* GetVirtualPageTableBuffer()  const { return m_virtualPageTableBuffer; }
        GPUBuffer* GetPageRequestBuffer()       const { return m_pageRequestBuffer; }
        GPUBuffer* GetMeshTableBuffer()         const { return m_meshTableBuffer; }
        GPUBuffer* GetBVHNodeBuffer()           const { return m_bvhNodeBuffer; }
        GPUBuffer* GetClusterMetaBuffer()       const { return m_clusterMetaBuffer; }
        GPUBuffer* GetClusterGroupBuffer()      const { return m_clusterGroupBuffer; }
        GPUBuffer* GetGroupActiveMaskBuffer()   const { return m_groupActiveMaskBuffer; }

        UINT32 GetRegisteredMeshCount()    const { return static_cast<UINT32>(m_registeredMeshes.size()); }
        UINT32 GetTotalVirtualPageCount()  const { return static_cast<UINT32>(m_virtualPages.size()); }
        UINT32 GetTotalClusterCount()      const { return static_cast<UINT32>(m_clusters.size()); }
        UINT32 GetTotalClusterGroupCount() const { return static_cast<UINT32>(m_groups.size()); }
        UINT32 GetTotalBVHNodeCount()      const { return static_cast<UINT32>(m_bvhNodes.size()); }
        UINT32 GetMeshTableCapacity()      const { return static_cast<UINT32>(m_meshTable.size()); }
        UINT32 GetPhysicalPoolCapacity()   const { return m_physicalPoolCapacity; }
        UINT32 GetResidentPageCount()      const { return m_residentPageCount; }
        UINT32 GetPinnedRootPageCount()    const { return m_pinnedRootPageCount; }
        FLOAT32 GetGeometryPoolVramMB()    const;
        FLOAT32 GetMetadataVramMB()        const;
        UINT64 GetResidencyRevision()      const { return m_residencyRevision; }

    private:
        struct RegisteredMeshRecord
        {
            UINT32                    meshId             = 0u;
            NVGVirtualMeshAsset       asset;
            const INVGPageDataSource* pageSource         = nullptr; // Points to custom source or &m_defaultMemorySource
            UINT32                    nodeBase           = 0u;
            UINT32                    nodeCount          = 0u;
            UINT32                    clusterBase        = 0u;
            UINT32                    clusterCount       = 0u;
            UINT32                    groupBase          = 0u;
            UINT32                    groupCount         = 0u;
            UINT32                    virtualPageBase    = 0u;
            UINT32                    virtualPageCount   = 0u;
            UINT32                    rootPageCount      = 0u;
        };

        struct PhysicalPageSlot
        {
            BOOL   occupied           = false;
            BOOL   pinnedRoot         = false;
            UINT32 virtualPageIndex   = kNVGInvalidIndex;
            UINT32 meshId             = 0u;
            UINT64 lastUsedFrame      = 0ull;
            UINT64 uploadedFrame      = 0ull;
            UINT32 lastActivePriority = 0u;
        };

        struct DeferredFreeSlot
        {
            UINT32 physicalSlotIndex = kNVGInvalidIndex;
            UINT64 retireFrame       = 0ull;
        };

        struct MetadataFreeRange
        {
            UINT32 offset = 0u;
            UINT32 count  = 0u;
        };

        struct DeferredMetadataRetirement
        {
            UINT32 nodeBase         = 0u;
            UINT32 nodeCount        = 0u;
            UINT32 clusterBase      = 0u;
            UINT32 clusterCount     = 0u;
            UINT32 groupBase        = 0u;
            UINT32 groupCount       = 0u;
            UINT32 virtualPageBase  = 0u;
            UINT32 virtualPageCount = 0u;
            UINT64 retireFrame      = 0ull;
        };

        struct VirtualPageHostMeta
        {
            UINT32              meshId                     = 0u;
            UINT32              localPageIndex             = 0u;
            UINT32              lodLevel                   = 0u;
            UINT32              residentChildRefCount      = 0u;
            BOOL                isRootPage                 = false;
            BOOL                valid                      = false;
            FLOAT32             maxShadowScaledParentError = 1.0e20f;
            AABB                pageLocalBounds            = {};
            std::vector<UINT32> parentVPs;
        };

        BOOL EnsurePhysicalPoolAllocated(UINT32 newCapacity, UINT64 currentFrameIndex);
        BOOL GrowPhysicalPoolPreservingResidents(UINT32 newCapacity);
        BOOL EnsureStructuredBufferCapacity(
            GPUBuffer*&    buffer,
            UINT32         stride,
            UINT32         requiredElements,
            GPUBufferUsage usage);

        UINT32 AcquirePhysicalSlot(
            UINT64                                  currentFrameIndex,
            UINT32                                  incomingPriority,
            std::vector<UINT32>&                    outEvictedMeshIds,
            std::vector<NVGResidencyChangedRegion>* outChangedRegions,
            UINT32&                                 outEvictedCount);

        void IncrementParentRefCounts(UINT32 childGlobalVP);
        void DecrementParentRefCounts(UINT32 childGlobalVP);
        void PropagateResidentAncestorLRU(UINT32 startGlobalVP, UINT64 frameIndex, UINT32 activePriority);
        static void RecordChangedRegion(
            std::vector<NVGResidencyChangedRegion>* ioRegions,
            UINT32                                  meshId,
            const AABB&                             localBounds,
            FLOAT32                                 maxShadowScaledParentError);

        void ReclaimDeferredSlots(UINT64 currentFrameIndex);
        void UploadVirtualPageEntry(UINT32 globalVirtualPageIndex);

        NorthstarBackend*  m_backend       = nullptr;
        CoreBufferManager* m_bufferManager = nullptr;
        BOOL               m_initialized   = false;

        BOOL    m_autoPoolMode             = true;
        BOOL    m_enableAutoQualityScaling = true;
        UINT32  m_physicalPoolCapacity     = kNVGDefaultPhysicalPageCapacity;
        UINT32  m_maxPoolBudgetPages       = 16384u;
        UINT32  m_configuredMaxUploads     = kNVGAutoUploadRate;
        UINT32  m_lastUploadBudgetPages    = kNVGDefaultMaxUploadsPerFrame;
        UINT32  m_minInFlightProtect       = 4u;
        FLOAT32 m_adaptiveErrorScale       = 1.0f;

        // GPU buffers owned via CoreBufferManager
        GPUBuffer* m_physicalPagePoolBuffer = nullptr; // UINT32[poolCapacity * 8192]
        GPUBuffer* m_virtualPageTableBuffer = nullptr; // NVGVirtualPageGPUData[]
        GPUBuffer* m_pageRequestBuffer      = nullptr; // UINT32[] (UAV + CPU_READBACK)
        GPUBuffer* m_meshTableBuffer        = nullptr; // NVGMeshGPUData[]
        GPUBuffer* m_bvhNodeBuffer          = nullptr; // NVGHierarchyNodeGPUData[]
        GPUBuffer* m_clusterMetaBuffer      = nullptr; // NVGClusterMetaGPUData[]
        GPUBuffer* m_clusterGroupBuffer     = nullptr; // NVGClusterGroupGPUData[]
        GPUBuffer* m_groupActiveMaskBuffer  = nullptr; // UINT32[] (UAV + SRV, 1 per group)

        // CPU mirrors of global tables
        std::vector<NVGMeshGPUData>          m_meshTable;
        std::vector<NVGHierarchyNodeGPUData> m_bvhNodes;
        std::vector<NVGClusterMetaGPUData>   m_clusters;
        std::vector<NVGClusterGroupGPUData>  m_groups;
        std::vector<NVGVirtualPageGPUData>   m_virtualPages;
        std::vector<VirtualPageHostMeta>     m_virtualPageHost;

        std::unordered_map<UINT32, RegisteredMeshRecord> m_registeredMeshes;
        NVGMemoryPageDataSource                          m_defaultMemorySource;

        // Physical VRAM page pool allocator & fence-safe metadata free-range allocators
        std::vector<PhysicalPageSlot>           m_physicalSlots;
        std::vector<UINT32>                     m_freePhysicalSlots;
        std::vector<DeferredFreeSlot>           m_deferredFreeSlots;
        std::vector<MetadataFreeRange>          m_freeNodeRanges;
        std::vector<MetadataFreeRange>          m_freeClusterRanges;
        std::vector<MetadataFreeRange>          m_freeGroupRanges;
        std::vector<MetadataFreeRange>          m_freeVirtualPageRanges;
        std::vector<DeferredMetadataRetirement> m_deferredMetadataRetirements;
        std::vector<BYTE>                       m_scratchPageBytes;
        std::vector<UINT32>                     m_scratchReadbackRequests;

        UINT32 m_residentPageCount   = 0u;
        UINT32 m_pinnedRootPageCount = 0u;
        UINT64 m_residencyRevision   = 1ull;

        BOOL m_meshTableDirty         = false;
        BOOL m_bvhNodesDirty          = false;
        BOOL m_clustersDirty          = false;
        BOOL m_groupsDirty            = false;
        BOOL m_virtualPageTableDirty  = false;
        BOOL m_readbackPending        = false;
        BOOL m_clearRequestsNextFrame = true;
    };
}
