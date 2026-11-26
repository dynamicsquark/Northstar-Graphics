#pragma once

#include "scene/scene.h"
#include "scene/gpu_scene_types.h"
#include "shadow/nvs_virtual_shadow_types.h"
#include "core_buffer_manager.h"
#include <vector>
#include <string>

namespace Northstar
{
    struct GPUSceneSlotEntry
    {
        RenderObject* object                = nullptr;
        UINT32        nextGeneration        = 1u;
        UINT32        currentGeneration     = 0u;
        BOOL          occupied              = false;
        BOOL          queuedDirty           = false;
        UINT32        dirtyMask             = 0u;
        Mesh*         boundMesh             = nullptr;
        UINT32        boundMeshRevision     = 0u;
        Material*     boundMaterial         = nullptr;
        UINT32        boundMaterialRevision = 0u;
        LODGroup*     boundLODGroup           = nullptr;
        UINT32        boundLODGroupRevision   = 0u;
        UINT32        boundSkinningRevision   = 0u;
        UINT32        skinningPaletteOffset   = 0u;
        UINT32        skinningPaletteCapacity = 0u;
        UINT32        skinningBoneCount       = 0u;
        BOOL          skinningPoseDirty       = false;
        BOOL          trackedVirtual          = false;
        BOOL          trackedMaterialKernel   = false;
    };

    struct GPUSceneWatchedMesh
    {
        Mesh*  mesh         = nullptr;
        UINT32 lastRevision = 0u;
    };

    struct GPUSceneWatchedMaterial
    {
        Material* material     = nullptr;
        UINT32    lastRevision = 0u;
    };

    struct GPUSceneWatchedLODGroup
    {
        LODGroup* lodGroup     = nullptr;
        UINT32    lastRevision = 0u;
    };

    struct SkinningPaletteFreeRange
    {
        UINT32 offset = 0u;
        UINT32 count  = 0u;
    };

    struct GPUSceneInstance
    {
        Scene*                             scene                  = nullptr;
        GPUBuffer*                         gpuBuffer              = nullptr;
        BOOL                               initialized            = false;
        BOOL                               lastUpdateSucceeded    = false;
        UINT32                             capacitySlots          = 0u;
        UINT32                             activeObjectCount          = 0u;
        UINT32                             virtualActiveCount         = 0u;
        UINT32                             materialKernelActiveCount  = 0u;
        UINT64                             mutationEpoch              = 0ull;
        UINT32                             highWaterMark              = 0u;
        UINT32                             totalBufferResizes     = 0u;
        UINT64                             lastUpdatedFrameNumber = 0ull;
        std::vector<GPUSceneObjectData>    cpuRecords;
        std::vector<GPUSceneObjectData>    gpuCommittedRecords;
        std::vector<GPUSceneSlotEntry>     slots;
        std::vector<UINT32>                freeSlots;
        std::vector<UINT32>                dirtySlotIndices;
        std::vector<GPUSceneUploadRange>   lastUploadRanges;
        std::vector<GPUSceneWatchedMesh>                 watchedMeshes;
        std::vector<GPUSceneWatchedMaterial>             watchedMaterials;
        std::vector<GPUSceneWatchedLODGroup>             watchedLODGroups;
        std::vector<NVSDynamicCasterInvalidationBox>     lastShadowInvalidationBoxes;
        std::vector<NVSDynamicCasterInvalidationBox>     pendingShadowInvalidationBoxes;
        GPUSceneUpdateStats                              lastStats{};

        static constexpr UINT32                          kSkinningRingCount = 3u;
        GPUBuffer*                                       skinningRingBuffers[kSkinningRingCount]    = { nullptr, nullptr, nullptr };
        UINT32                                           skinningRingCapacities[kSkinningRingCount] = { 0u, 0u, 0u };
        UINT32                                           skinningRingDirtyMin[kSkinningRingCount]   = { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
        UINT32                                           skinningRingDirtyMax[kSkinningRingCount]   = { 0u, 0u, 0u };
        UINT32                                           skinningRingIndex                          = 0u;
        GPUBuffer*                                       activeSkinningBuffer                       = nullptr;
        std::vector<Mat4>                                cpuSkinningPalette;
        std::vector<SkinningPaletteFreeRange>            skinningFreeRanges;
    };

    class CoreGPUSceneManager
    {
    public:
        CoreGPUSceneManager() = default;
        ~CoreGPUSceneManager();

        void Initialize(CoreBufferManager* bufferManager);
        void Shutdown();

        void OnSceneDestroyed(Scene* scene);
        void OnRenderObjectDestroyed(RenderObject* destroyedObject);
        void OnMeshDestroyed(Mesh* destroyedMesh);
        void OnMaterialDestroyed(Material* destroyedMaterial);
        void OnLODGroupDestroyed(LODGroup* destroyedLODGroup);

        // Incremental GPU Scene update (processes only dirty objects & removed slots; O(1) when 0% dirty)
        BOOL UpdateScene(Scene* scene, UINT64 frameNumber = 0ull);

        // Baseline full-scene rebuild & full-buffer upload (used for benchmark comparison & full resync)
        BOOL ForceFullRebuildAndUpload(Scene* scene, GPUSceneUpdateStats* outStats = nullptr);

        // Reserve or grow scene slot capacity explicitly
        BOOL ReserveSceneCapacity(Scene* scene, UINT32 minCapacitySlots);

        // Queries & Diagnostics
        GPUBuffer*                             GetSceneGPUBuffer(const Scene* scene) const;
        GPUBuffer*                             GetSceneSkinningPaletteBuffer(const Scene* scene) const;
        const std::vector<Mat4>*               GetSceneSkinningPaletteCPUData(const Scene* scene) const;
        GPUSceneHandle                         GetObjectHandle(const Scene* scene, const RenderObject* obj) const;
        BOOL                                   IsHandleValid(const Scene* scene, GPUSceneHandle handle) const;
        const GPUSceneObjectData*              GetCPURecord(const Scene* scene, UINT32 slotIndex) const;
        const GPUSceneUpdateStats*             GetSceneStats(const Scene* scene) const;
        const std::vector<GPUSceneUploadRange>*                  GetLastUploadRanges(const Scene* scene) const;
        const std::vector<NVSDynamicCasterInvalidationBox>*      GetLastShadowInvalidationBoxes(const Scene* scene) const;
        const std::vector<NVSDynamicCasterInvalidationBox>*      GetPendingShadowInvalidationBoxes(const Scene* scene) const;
        BOOL                                                     ConsumePendingShadowInvalidationBoxes(Scene* scene, std::vector<NVSDynamicCasterInvalidationBox>& outBoxes);
        void                                                     ClearPendingShadowInvalidationBoxes(Scene* scene);
        UINT32                                                   GetTotalBufferResizes(const Scene* scene) const;
        const std::string&                     GetLastError() const { return m_lastError; }

        void SetSimulateAllocationFailure(BOOL simulateFailure) { m_simulateAllocationFailure = simulateFailure; }
        BOOL IsSimulatingAllocationFailure() const { return m_simulateAllocationFailure; }

        void SetSimulateUploadFailure(BOOL simulateFailure)
        {
            m_simulateUploadFailure           = simulateFailure;
            m_simulateUploadFailureRangeIndex = -1;
        }
        void SetSimulateUploadFailureAtRange(BOOL simulateFailure, INT32 failAtRangeIndex)
        {
            m_simulateUploadFailure           = simulateFailure;
            m_simulateUploadFailureRangeIndex = failAtRangeIndex;
        }
        BOOL  IsSimulatingUploadFailure() const { return m_simulateUploadFailure; }
        INT32 GetSimulatedUploadFailureRangeIndex() const { return m_simulateUploadFailureRangeIndex; }

        void   SetMaxSlotCapacity(UINT32 maxSlots) { m_maxSlotCapacity = (maxSlots > 0u) ? maxSlots : kMaxGPUSceneSlotCapacity; }
        UINT32 GetMaxSlotCapacity() const { return m_maxSlotCapacity; }

        void ForceSlotNextGeneration(Scene* scene, UINT32 slotIndex, UINT32 nextGen);

    private:
        GPUSceneInstance*       FindInstance(const Scene* scene);
        const GPUSceneInstance* FindInstance(const Scene* scene) const;
        GPUSceneInstance&       GetOrCreateInstance(Scene* scene);

        static UINT32           ComputeTargetCapacity(UINT32 currentCapacity, UINT32 requiredSlots);
        static void             BuildObjectRecord(
            const RenderObject*    obj,
            const SceneObjectLink* link,
            UINT32                 slotIndex,
            UINT32                 generation,
            GPUSceneObjectData&    outRecord);

        static void             BuildInvalidSlotRecord(
            UINT32              slotIndex,
            GPUSceneObjectData& outRecord);

        void                    WatchSlotResources(GPUSceneInstance& inst, GPUSceneSlotEntry& se);
        void                    CheckWatchedResourceMutations(GPUSceneInstance& inst);

        CoreBufferManager*            m_bufferManager                   = nullptr;
        std::vector<GPUSceneInstance> m_instances;
        std::string                   m_lastError;
        BOOL                          m_simulateAllocationFailure       = false;
        BOOL                          m_simulateUploadFailure           = false;
        INT32                         m_simulateUploadFailureRangeIndex = -1;
        UINT32                        m_maxSlotCapacity                 = kMaxGPUSceneSlotCapacity;
    };
}

