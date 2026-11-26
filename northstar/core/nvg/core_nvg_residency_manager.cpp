#include "core_nvg_residency_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace Northstar
{
    namespace
    {
        inline UINT32 NextPow2AtLeast(UINT32 value, UINT32 minCap) noexcept
        {
            UINT32 cap = (std::max)(minCap, 1u);
            while (cap < value && cap < 0x40000000u)
            {
                cap <<= 1u;
            }
            return (std::max)(cap, value);
        }

        inline void AppendUniqueU32(std::vector<UINT32>& vec, UINT32 val)
        {
            if (std::find(vec.begin(), vec.end(), val) == vec.end())
            {
                vec.push_back(val);
            }
        }

        template <typename FreeRangeT>
        inline UINT32 AllocateContiguousRange(std::vector<FreeRangeT>& freeRanges, SIZE_T currentVecSize, UINT32 count)
        {
            if (count == 0u) return static_cast<UINT32>(currentVecSize);
            for (SIZE_T i = 0u; i < freeRanges.size(); ++i)
            {
                if (freeRanges[i].count >= count)
                {
                    const UINT32 base = freeRanges[i].offset;
                    freeRanges[i].offset += count;
                    freeRanges[i].count  -= count;
                    if (freeRanges[i].count == 0u)
                    {
                        freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(i));
                    }
                    return base;
                }
            }
            return static_cast<UINT32>(currentVecSize);
        }

        template <typename FreeRangeT, typename VecT>
        inline void ReleaseAndTrimRange(std::vector<FreeRangeT>& freeRanges, VecT& vec, UINT32 offset, UINT32 count)
        {
            if (count == 0u) return;
            freeRanges.push_back({ offset, count });
            std::sort(freeRanges.begin(), freeRanges.end(), [](const FreeRangeT& a, const FreeRangeT& b) {
                return a.offset < b.offset;
            });
            SIZE_T w = 0u;
            for (SIZE_T i = 0u; i < freeRanges.size(); ++i)
            {
                if (w > 0u && freeRanges[w - 1u].offset + freeRanges[w - 1u].count >= freeRanges[i].offset)
                {
                    const UINT32 mergedEnd = (std::max)(
                        freeRanges[w - 1u].offset + freeRanges[w - 1u].count,
                        freeRanges[i].offset + freeRanges[i].count);
                    freeRanges[w - 1u].count = mergedEnd - freeRanges[w - 1u].offset;
                }
                else
                {
                    freeRanges[w++] = freeRanges[i];
                }
            }
            freeRanges.resize(w);

            while (!freeRanges.empty() &&
                   static_cast<SIZE_T>(freeRanges.back().offset) + static_cast<SIZE_T>(freeRanges.back().count) == vec.size())
            {
                vec.resize(freeRanges.back().offset);
                freeRanges.pop_back();
            }
        }
    }

    CoreNVGResidencyManager::~CoreNVGResidencyManager()
    {
        Shutdown();
    }

    BOOL CoreNVGResidencyManager::Initialize(
        NorthstarBackend*  backend,
        CoreBufferManager* bufferManager,
        const NVGConfig&   config)
    {
        if (m_initialized) return true;
        if (!backend || !bufferManager) return false;

        m_backend       = backend;
        m_bufferManager = bufferManager;

        m_autoPoolMode             = (config.physicalPagePoolCapacity == kNVGAutoPoolCapacity &&
                                      config.memoryBudget != NVGMemoryBudget::CUSTOM_PAGES);
        m_enableAutoQualityScaling = config.enableAutoQualityScaling;
        m_configuredMaxUploads     = config.maxPageUploadsPerFrame;
        m_lastUploadBudgetPages    = (m_configuredMaxUploads == kNVGAutoUploadRate)
            ? kNVGDefaultMaxUploadsPerFrame
            : m_configuredMaxUploads;
        m_minInFlightProtect       = (std::max)(1u, config.minInFlightProtectFrames);
        m_adaptiveErrorScale       = 1.0f;

        if (m_autoPoolMode)
        {
            m_maxPoolBudgetPages   = ResolveNVGMemoryBudgetPages(config.memoryBudget);
            m_physicalPoolCapacity = (std::min)(kNVGDefaultPhysicalPageCapacity, m_maxPoolBudgetPages);
        }
        else
        {
            const UINT32 rawCap    = (config.physicalPagePoolCapacity == 0u)
                ? kNVGDefaultPhysicalPageCapacity
                : config.physicalPagePoolCapacity;
            m_physicalPoolCapacity = (std::clamp)(rawCap, kNVGMinPhysicalPageCapacity, kNVGMaxPhysicalPageCapacity);
            m_maxPoolBudgetPages   = m_physicalPoolCapacity;
        }

        m_scratchPageBytes.resize(kNVGPageSizeBytes, 0u);

        // Reserve slot 0 in m_meshTable as an invalid sentinel so meshId == 0 is safely invalid
        m_meshTable.clear();
        m_meshTable.push_back(NVGMeshGPUData{});
        m_bvhNodes.clear();
        m_clusters.clear();
        m_groups.clear();
        m_virtualPages.clear();
        m_virtualPageHost.clear();
        m_registeredMeshes.clear();
        m_defaultMemorySource.Clear();

        if (!EnsurePhysicalPoolAllocated(m_physicalPoolCapacity, 0ull))
        {
            Shutdown();
            return false;
        }

        // Pre-create minimal valid structured buffers (1 element sentinel) so shaders always have valid descriptors
        const NVGMeshGPUData dummyMesh{};
        const NVGHierarchyNodeGPUData dummyNode{};
        const NVGClusterMetaGPUData dummyCluster{};
        const NVGClusterGroupGPUData dummyGroup{};
        const NVGVirtualPageGPUData dummyPage{};
        const UINT32 dummyZero = 0u;

        if (!EnsureStructuredBufferCapacity(m_meshTableBuffer, sizeof(NVGMeshGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureStructuredBufferCapacity(m_bvhNodeBuffer, sizeof(NVGHierarchyNodeGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureStructuredBufferCapacity(m_clusterMetaBuffer, sizeof(NVGClusterMetaGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureStructuredBufferCapacity(m_clusterGroupBuffer, sizeof(NVGClusterGroupGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureStructuredBufferCapacity(m_groupActiveMaskBuffer, sizeof(UINT32), 16u, GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV) ||
            !EnsureStructuredBufferCapacity(m_virtualPageTableBuffer, sizeof(NVGVirtualPageGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureStructuredBufferCapacity(m_pageRequestBuffer, sizeof(UINT32), 16u,
                GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK))
        {
            Shutdown();
            return false;
        }

        m_bufferManager->UpdateGPUBuffer(m_meshTableBuffer, &dummyMesh, sizeof(dummyMesh), 0u);
        m_bufferManager->UpdateGPUBuffer(m_bvhNodeBuffer, &dummyNode, sizeof(dummyNode), 0u);
        m_bufferManager->UpdateGPUBuffer(m_clusterMetaBuffer, &dummyCluster, sizeof(dummyCluster), 0u);
        m_bufferManager->UpdateGPUBuffer(m_clusterGroupBuffer, &dummyGroup, sizeof(dummyGroup), 0u);
        m_bufferManager->UpdateGPUBuffer(m_groupActiveMaskBuffer, &dummyZero, sizeof(dummyZero), 0u);
        m_bufferManager->UpdateGPUBuffer(m_virtualPageTableBuffer, &dummyPage, sizeof(dummyPage), 0u);
        m_bufferManager->UpdateGPUBuffer(m_pageRequestBuffer, &dummyZero, sizeof(dummyZero), 0u);

        m_residentPageCount      = 0u;
        m_pinnedRootPageCount    = 0u;
        m_residencyRevision      = 1ull;
        m_meshTableDirty         = false;
        m_bvhNodesDirty          = false;
        m_clustersDirty          = false;
        m_groupsDirty            = false;
        m_virtualPageTableDirty  = false;
        m_readbackPending        = false;
        m_clearRequestsNextFrame = true;
        m_initialized            = true;
        return true;
    }

    void CoreNVGResidencyManager::Shutdown()
    {
        if (m_bufferManager)
        {
            if (m_physicalPagePoolBuffer) { m_bufferManager->DestroyGPUBuffer(m_physicalPagePoolBuffer); m_physicalPagePoolBuffer = nullptr; }
            if (m_virtualPageTableBuffer) { m_bufferManager->DestroyGPUBuffer(m_virtualPageTableBuffer); m_virtualPageTableBuffer = nullptr; }
            if (m_pageRequestBuffer)      { m_bufferManager->DestroyGPUBuffer(m_pageRequestBuffer);      m_pageRequestBuffer      = nullptr; }
            if (m_meshTableBuffer)        { m_bufferManager->DestroyGPUBuffer(m_meshTableBuffer);        m_meshTableBuffer        = nullptr; }
            if (m_bvhNodeBuffer)          { m_bufferManager->DestroyGPUBuffer(m_bvhNodeBuffer);          m_bvhNodeBuffer          = nullptr; }
            if (m_clusterMetaBuffer)      { m_bufferManager->DestroyGPUBuffer(m_clusterMetaBuffer);      m_clusterMetaBuffer      = nullptr; }
            if (m_clusterGroupBuffer)     { m_bufferManager->DestroyGPUBuffer(m_clusterGroupBuffer);     m_clusterGroupBuffer     = nullptr; }
            if (m_groupActiveMaskBuffer)  { m_bufferManager->DestroyGPUBuffer(m_groupActiveMaskBuffer);  m_groupActiveMaskBuffer  = nullptr; }
        }

        m_meshTable.clear();
        m_bvhNodes.clear();
        m_clusters.clear();
        m_groups.clear();
        m_virtualPages.clear();
        m_virtualPageHost.clear();
        m_registeredMeshes.clear();
        m_defaultMemorySource.Clear();
        m_physicalSlots.clear();
        m_freePhysicalSlots.clear();
        m_deferredFreeSlots.clear();
        m_freeNodeRanges.clear();
        m_freeClusterRanges.clear();
        m_freeGroupRanges.clear();
        m_freeVirtualPageRanges.clear();
        m_deferredMetadataRetirements.clear();
        m_scratchPageBytes.clear();
        m_scratchReadbackRequests.clear();

        m_residentPageCount      = 0u;
        m_pinnedRootPageCount    = 0u;
        m_adaptiveErrorScale     = 1.0f;
        m_readbackPending        = false;
        m_clearRequestsNextFrame = true;
        m_backend                = nullptr;
        m_bufferManager          = nullptr;
        m_initialized            = false;
    }

    BOOL CoreNVGResidencyManager::ApplyConfig(const NVGConfig& config, UINT64 currentFrameIndex)
    {
        m_enableAutoQualityScaling = config.enableAutoQualityScaling;
        if (!m_enableAutoQualityScaling)
        {
            m_adaptiveErrorScale = 1.0f;
        }
        m_configuredMaxUploads  = config.maxPageUploadsPerFrame;
        m_lastUploadBudgetPages = (m_configuredMaxUploads == kNVGAutoUploadRate)
            ? kNVGDefaultMaxUploadsPerFrame
            : m_configuredMaxUploads;
        m_minInFlightProtect    = (std::max)(1u, config.minInFlightProtectFrames);

        const BOOL newAutoMode = (config.physicalPagePoolCapacity == kNVGAutoPoolCapacity &&
                                  config.memoryBudget != NVGMemoryBudget::CUSTOM_PAGES);
        UINT32 desiredCap = m_physicalPoolCapacity;
        if (newAutoMode)
        {
            const UINT32 newBudget = ResolveNVGMemoryBudgetPages(config.memoryBudget);
            m_autoPoolMode       = true;
            m_maxPoolBudgetPages = newBudget;
            const UINT32 maxAllowedNow = (std::max)(newBudget, m_pinnedRootPageCount);
            if (!m_initialized || m_physicalPoolCapacity <= maxAllowedNow)
            {
                if (newBudget > m_physicalPoolCapacity)
                {
                    m_adaptiveErrorScale = 1.0f;
                }
                return true;
            }
            desiredCap = maxAllowedNow;
        }
        else
        {
            const UINT32 rawCap = (config.physicalPagePoolCapacity == 0u)
                ? kNVGDefaultPhysicalPageCapacity
                : config.physicalPagePoolCapacity;
            desiredCap           = (std::clamp)(rawCap, kNVGMinPhysicalPageCapacity, kNVGMaxPhysicalPageCapacity);
            m_autoPoolMode       = false;
            m_maxPoolBudgetPages = desiredCap;
            if (!m_initialized || desiredCap == m_physicalPoolCapacity)
            {
                return true;
            }
            if (desiredCap > m_physicalPoolCapacity)
            {
                m_adaptiveErrorScale = 1.0f;
                return GrowPhysicalPoolPreservingResidents(desiredCap);
            }
        }

        m_adaptiveErrorScale = 1.0f;

        // Shrinking pool capacity: cannot shrink below currently pinned root pages
        if (desiredCap < m_pinnedRootPageCount)
        {
            return false;
        }

        // Rebuild physical pool cleanly on shrink: wait for GPU, reset child refcounts,
        // re-pin all registered root pages, and mark streaming pages non-resident so they re-stream cleanly.
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }

        if (!EnsurePhysicalPoolAllocated(desiredCap, currentFrameIndex))
        {
            return false;
        }

        // Clear all virtual page residency entries & child refcounts first
        for (SIZE_T i = 0; i < m_virtualPages.size(); ++i)
        {
            if (m_virtualPageHost[i].valid)
            {
                m_virtualPageHost[i].residentChildRefCount = 0u;
                m_virtualPages[i].stateAndFlags            = NVG_VPAGE_FLAG_VALID | (m_virtualPageHost[i].isRootPage ? NVG_VPAGE_FLAG_ROOT_PINNED : 0u);
                m_virtualPages[i].physicalPageIndex        = kNVGInvalidIndex;
                m_virtualPages[i].pageGeneration++;
            }
        }

        // Re-upload pinned root pages for every registered mesh
        std::vector<UINT32> dummyEvicted;
        UINT32 dummyCount = 0u;
        for (auto& kv : m_registeredMeshes)
        {
            RegisteredMeshRecord& rec = kv.second;
            for (UINT32 rp = 0u; rp < rec.rootPageCount; ++rp)
            {
                const UINT32 gvp  = rec.virtualPageBase + rp;
                const UINT32 slot = AcquirePhysicalSlot(currentFrameIndex, UINT32_MAX, dummyEvicted, nullptr, dummyCount);
                if (slot == kNVGInvalidIndex)
                {
                    return false;
                }

                if (rec.pageSource && rec.pageSource->ReadVirtualPageData(rec.meshId, rp, m_scratchPageBytes.data(), kNVGPageSizeBytes))
                {
                    m_bufferManager->UpdateGPUBuffer(
                        m_physicalPagePoolBuffer,
                        m_scratchPageBytes.data(),
                        kNVGPageSizeBytes,
                        slot * kNVGPageSizeBytes);
                }

                m_physicalSlots[slot].occupied         = true;
                m_physicalSlots[slot].pinnedRoot       = true;
                m_physicalSlots[slot].virtualPageIndex = gvp;
                m_physicalSlots[slot].meshId           = rec.meshId;
                m_physicalSlots[slot].lastUsedFrame    = currentFrameIndex;
                m_physicalSlots[slot].uploadedFrame    = currentFrameIndex;

                m_virtualPages[gvp].stateAndFlags     = NVG_VPAGE_FLAG_VALID | NVG_VPAGE_FLAG_RESIDENT | NVG_VPAGE_FLAG_ROOT_PINNED;
                m_virtualPages[gvp].physicalPageIndex = slot;
                m_virtualPages[gvp].pageGeneration++;
                ++m_residentPageCount;
                ++m_pinnedRootPageCount;
            }
        }

        m_virtualPageTableDirty  = true;
        m_readbackPending        = false;
        m_clearRequestsNextFrame = true;
        ++m_residencyRevision;
        SyncDirtyMetadataToGPU();
        return true;
    }

    BOOL CoreNVGResidencyManager::EnsurePhysicalPoolAllocated(UINT32 newCapacity, UINT64 /*currentFrameIndex*/)
    {
        if (!m_bufferManager) return false;

        if (m_physicalPagePoolBuffer)
        {
            m_bufferManager->DestroyGPUBuffer(m_physicalPagePoolBuffer);
            m_physicalPagePoolBuffer = nullptr;
        }

        GPUBufferDesc desc{};
        desc.elementCount = newCapacity * kNVGPageWordCount;
        desc.stride       = sizeof(UINT32);
        desc.usage        = GPUBufferUsage::STRUCTURED_SRV;
        desc.initialData  = nullptr;

        m_physicalPagePoolBuffer = m_bufferManager->CreateGPUBuffer(desc);
        if (!m_physicalPagePoolBuffer)
        {
            return false;
        }

        m_physicalPoolCapacity = newCapacity;
        m_physicalSlots.assign(newCapacity, PhysicalPageSlot{});
        m_freePhysicalSlots.clear();
        m_freePhysicalSlots.reserve(newCapacity);
        for (UINT32 i = newCapacity; i > 0u; --i)
        {
            m_freePhysicalSlots.push_back(i - 1u);
        }
        m_deferredFreeSlots.clear();
        m_residentPageCount   = 0u;
        m_pinnedRootPageCount = 0u;
        return true;
    }

    BOOL CoreNVGResidencyManager::GrowPhysicalPoolPreservingResidents(UINT32 newCapacity)
    {
        const UINT32 clampedNew = (std::min)(newCapacity, kNVGMaxPhysicalPageCapacity);
        if (clampedNew <= m_physicalPoolCapacity)
        {
            return true;
        }
        if (!m_bufferManager || !m_physicalPagePoolBuffer)
        {
            return false;
        }

        const UINT32 oldCapacity = m_physicalPoolCapacity;
        if (!m_bufferManager->ResizeGPUBuffer(m_physicalPagePoolBuffer, clampedNew * kNVGPageWordCount, nullptr))
        {
            return false;
        }

        // Re-upload all currently occupied physical slots at their identical [0 .. oldCapacity-1] indices
        // (zero WaitForGPU() CPU stall; old buffer handle retires safely via fence-tracked deferred deletion).
        for (UINT32 slot = 0u; slot < oldCapacity; ++slot)
        {
            const PhysicalPageSlot& ps = m_physicalSlots[slot];
            if (!ps.occupied || ps.virtualPageIndex >= m_virtualPageHost.size()) continue;
            const VirtualPageHostMeta& hm = m_virtualPageHost[ps.virtualPageIndex];
            if (!hm.valid) continue;
            const auto itMesh = m_registeredMeshes.find(hm.meshId);
            if (itMesh == m_registeredMeshes.end() || !itMesh->second.pageSource) continue;

            if (itMesh->second.pageSource->ReadVirtualPageData(
                    hm.meshId, hm.localPageIndex, m_scratchPageBytes.data(), kNVGPageSizeBytes))
            {
                m_bufferManager->UpdateGPUBuffer(
                    m_physicalPagePoolBuffer,
                    m_scratchPageBytes.data(),
                    kNVGPageSizeBytes,
                    slot * kNVGPageSizeBytes);
            }
        }

        m_physicalSlots.resize(clampedNew, PhysicalPageSlot{});
        m_freePhysicalSlots.reserve(m_freePhysicalSlots.size() + (clampedNew - oldCapacity));
        for (UINT32 i = clampedNew; i > oldCapacity; --i)
        {
            m_freePhysicalSlots.push_back(i - 1u);
        }
        m_physicalPoolCapacity = clampedNew;
        ++m_residencyRevision;
        return true;
    }

    void CoreNVGResidencyManager::IncrementParentRefCounts(UINT32 childGlobalVP)
    {
        if (childGlobalVP >= m_virtualPageHost.size()) return;
        for (UINT32 pvp : m_virtualPageHost[childGlobalVP].parentVPs)
        {
            if (pvp < m_virtualPageHost.size() && m_virtualPageHost[pvp].valid)
            {
                ++m_virtualPageHost[pvp].residentChildRefCount;
            }
        }
    }

    void CoreNVGResidencyManager::DecrementParentRefCounts(UINT32 childGlobalVP)
    {
        if (childGlobalVP >= m_virtualPageHost.size()) return;
        for (UINT32 pvp : m_virtualPageHost[childGlobalVP].parentVPs)
        {
            if (pvp < m_virtualPageHost.size() && m_virtualPageHost[pvp].residentChildRefCount > 0u)
            {
                --m_virtualPageHost[pvp].residentChildRefCount;
            }
        }
    }

    void CoreNVGResidencyManager::PropagateResidentAncestorLRU(UINT32 startGlobalVP, UINT64 frameIndex, UINT32 activePriority)
    {
        if (startGlobalVP >= m_virtualPageHost.size()) return;
        UINT32 stack[32]{};
        UINT32 sp = 0u;
        stack[sp++] = startGlobalVP;

        UINT32 visited[64]{};
        UINT32 visitedCnt = 0u;

        while (sp > 0u)
        {
            const UINT32 curVP = stack[--sp];
            if (curVP >= m_virtualPageHost.size() || !m_virtualPageHost[curVP].valid) continue;

            const UINT32 physSlot = m_virtualPages[curVP].physicalPageIndex;
            if (physSlot < m_physicalSlots.size() && m_physicalSlots[physSlot].occupied)
            {
                if (frameIndex > m_physicalSlots[physSlot].lastUsedFrame)
                {
                    m_physicalSlots[physSlot].lastUsedFrame = frameIndex;
                }
                if (activePriority > m_physicalSlots[physSlot].lastActivePriority)
                {
                    m_physicalSlots[physSlot].lastActivePriority = activePriority;
                }
            }

            for (UINT32 pvp : m_virtualPageHost[curVP].parentVPs)
            {
                BOOL seen = false;
                for (UINT32 v = 0u; v < visitedCnt; ++v)
                {
                    if (visited[v] == pvp) { seen = true; break; }
                }
                if (!seen && visitedCnt < 64u && sp < 32u)
                {
                    visited[visitedCnt++] = pvp;
                    stack[sp++]           = pvp;
                }
            }
        }
    }

    void CoreNVGResidencyManager::RecordChangedRegion(
        std::vector<NVGResidencyChangedRegion>* ioRegions,
        UINT32                                  meshId,
        const AABB&                             localBounds,
        FLOAT32                                 maxShadowScaledParentError)
    {
        if (!ioRegions || meshId == 0u) return;

        // Keep up to 8 spatially distinct changed sub-boxes per mesh; merge into closest overlapping/nearby box beyond that
        UINT32 meshRegionCount = 0u;
        SIZE_T mergeTargetIdx  = static_cast<SIZE_T>(-1);
        for (SIZE_T i = 0u; i < ioRegions->size(); ++i)
        {
            NVGResidencyChangedRegion& r = (*ioRegions)[i];
            if (r.meshId != meshId) continue;
            ++meshRegionCount;
            const BOOL overlaps =
                (localBounds.m_Min.x <= r.localBounds.m_Max.x && localBounds.m_Max.x >= r.localBounds.m_Min.x) &&
                (localBounds.m_Min.y <= r.localBounds.m_Max.y && localBounds.m_Max.y >= r.localBounds.m_Min.y) &&
                (localBounds.m_Min.z <= r.localBounds.m_Max.z && localBounds.m_Max.z >= r.localBounds.m_Min.z);
            if (overlaps)
            {
                r.localBounds.Expand(localBounds.m_Min);
                r.localBounds.Expand(localBounds.m_Max);
                if (maxShadowScaledParentError > r.maxShadowScaledParentError)
                {
                    r.maxShadowScaledParentError = maxShadowScaledParentError;
                }
                return;
            }
            mergeTargetIdx = i;
        }

        if (meshRegionCount >= 8u && mergeTargetIdx != static_cast<SIZE_T>(-1))
        {
            (*ioRegions)[mergeTargetIdx].localBounds.Expand(localBounds.m_Min);
            (*ioRegions)[mergeTargetIdx].localBounds.Expand(localBounds.m_Max);
            if (maxShadowScaledParentError > (*ioRegions)[mergeTargetIdx].maxShadowScaledParentError)
            {
                (*ioRegions)[mergeTargetIdx].maxShadowScaledParentError = maxShadowScaledParentError;
            }
            return;
        }

        NVGResidencyChangedRegion nr{};
        nr.meshId                     = meshId;
        nr.localBounds                = localBounds;
        nr.maxShadowScaledParentError = maxShadowScaledParentError;
        ioRegions->push_back(nr);
    }

    BOOL CoreNVGResidencyManager::EnsureStructuredBufferCapacity(
        GPUBuffer*&    buffer,
        UINT32         stride,
        UINT32         requiredElements,
        GPUBufferUsage usage)
    {
        if (!m_bufferManager) return false;
        const UINT32 needed = (std::max)(requiredElements, 1u);

        if (buffer != nullptr)
        {
            if (buffer->GetElementCount() >= needed)
            {
                return true;
            }
            const UINT32 newCap = NextPow2AtLeast(needed, 16u);
            return m_bufferManager->ResizeGPUBuffer(buffer, newCap, nullptr);
        }

        GPUBufferDesc desc{};
        desc.elementCount = NextPow2AtLeast(needed, 16u);
        desc.stride       = stride;
        desc.usage        = usage;
        desc.initialData  = nullptr;
        buffer = m_bufferManager->CreateGPUBuffer(desc);
        return (buffer != nullptr);
    }

    void CoreNVGResidencyManager::ReclaimDeferredSlots(UINT64 currentFrameIndex)
    {
        SIZE_T writeIdx = 0u;
        for (SIZE_T i = 0u; i < m_deferredFreeSlots.size(); ++i)
        {
            if (currentFrameIndex >= m_deferredFreeSlots[i].retireFrame)
            {
                const UINT32 slotIdx = m_deferredFreeSlots[i].physicalSlotIndex;
                if (slotIdx < m_physicalSlots.size())
                {
                    m_physicalSlots[slotIdx] = PhysicalPageSlot{};
                    m_freePhysicalSlots.push_back(slotIdx);
                }
            }
            else
            {
                m_deferredFreeSlots[writeIdx++] = m_deferredFreeSlots[i];
            }
        }
        m_deferredFreeSlots.resize(writeIdx);

        SIZE_T metaWriteIdx = 0u;
        for (SIZE_T i = 0u; i < m_deferredMetadataRetirements.size(); ++i)
        {
            const DeferredMetadataRetirement& mr = m_deferredMetadataRetirements[i];
            if (currentFrameIndex >= mr.retireFrame)
            {
                ReleaseAndTrimRange(m_freeNodeRanges,        m_bvhNodes,     mr.nodeBase,        mr.nodeCount);
                ReleaseAndTrimRange(m_freeClusterRanges,     m_clusters,     mr.clusterBase,     mr.clusterCount);
                ReleaseAndTrimRange(m_freeGroupRanges,       m_groups,       mr.groupBase,       mr.groupCount);
                ReleaseAndTrimRange(m_freeVirtualPageRanges, m_virtualPages, mr.virtualPageBase, mr.virtualPageCount);
                if (m_virtualPageHost.size() > m_virtualPages.size())
                {
                    m_virtualPageHost.resize(m_virtualPages.size());
                }
            }
            else
            {
                m_deferredMetadataRetirements[metaWriteIdx++] = mr;
            }
        }
        m_deferredMetadataRetirements.resize(metaWriteIdx);

        while (m_meshTable.size() > 1u && m_meshTable.back().meshId == 0u && m_meshTable.back().flags == 0u)
        {
            m_meshTable.pop_back();
        }
    }

    UINT32 CoreNVGResidencyManager::AcquirePhysicalSlot(
        UINT64                                  currentFrameIndex,
        UINT32                                  incomingPriority,
        std::vector<UINT32>&                    outEvictedMeshIds,
        std::vector<NVGResidencyChangedRegion>* outChangedRegions,
        UINT32&                                 outEvictedCount)
    {
        if (!m_freePhysicalSlots.empty())
        {
            const UINT32 slot = m_freePhysicalSlots.back();
            m_freePhysicalSlots.pop_back();
            return slot;
        }

        // Strict Leaf-First LRU + Priority-Preemptive Eviction:
        //   1) Never evict pinnedRoot pages.
        //   2) Never evict any page with residentChildRefCount > 0 (ancestor protection — eliminates LOD 0 <-> Root implosion!).
        //   3) Never evict pages uploaded within the async readback protection window (prevents ping-pong before readback).
        //   4) Pass 1: Evict unused leaves (lastUsedFrame + effectiveProtect < currentFrameIndex), oldest first.
        //   5) Pass 2 (Over-Committed Pool): If all leaves are currently visible, evict the lowest-priority leaf
        //      ONLY when incomingPriority strictly exceeds victim.lastActivePriority + 32u (reaches zero-churn equilibrium!).
        const UINT64 effectiveProtect = (std::max)(static_cast<UINT64>(m_minInFlightProtect), 4ull);
        UINT32 bestSlot       = kNVGInvalidIndex;
        UINT64 oldestUseFrame = UINT64_MAX;
        UINT32 bestLodLevel   = UINT32_MAX;

        for (UINT32 i = 0u; i < m_physicalSlots.size(); ++i)
        {
            const PhysicalPageSlot& s = m_physicalSlots[i];
            if (!s.occupied || s.pinnedRoot) continue;
            if (s.virtualPageIndex < m_virtualPageHost.size() &&
                m_virtualPageHost[s.virtualPageIndex].residentChildRefCount > 0u)
            {
                continue;
            }
            if (s.uploadedFrame + effectiveProtect >= currentFrameIndex) continue;
            if (s.lastUsedFrame + effectiveProtect >= currentFrameIndex) continue;

            const UINT32 candLod = (s.virtualPageIndex < m_virtualPageHost.size())
                ? m_virtualPageHost[s.virtualPageIndex].lodLevel
                : 0u;
            if (s.lastUsedFrame < oldestUseFrame ||
                (s.lastUsedFrame == oldestUseFrame && candLod < bestLodLevel))
            {
                oldestUseFrame = s.lastUsedFrame;
                bestLodLevel   = candLod;
                bestSlot       = i;
            }
        }

        if (bestSlot == kNVGInvalidIndex && incomingPriority > 0u)
        {
            UINT32 lowestActivePrio = UINT32_MAX;
            UINT32 preemptLod       = UINT32_MAX;
            UINT64 preemptOldestUse = UINT64_MAX;
            for (UINT32 i = 0u; i < m_physicalSlots.size(); ++i)
            {
                const PhysicalPageSlot& s = m_physicalSlots[i];
                if (!s.occupied || s.pinnedRoot) continue;
                if (s.virtualPageIndex < m_virtualPageHost.size() &&
                    m_virtualPageHost[s.virtualPageIndex].residentChildRefCount > 0u)
                {
                    continue;
                }
                if (s.uploadedFrame + effectiveProtect >= currentFrameIndex) continue;
                const UINT32 signalPrio = (s.lastActivePriority >= 262144u)
                    ? (s.lastActivePriority - 262144u)
                    : s.lastActivePriority;
                const UINT32 hysteresisMargin = (signalPrio >> 4u) + 32u;
                if (incomingPriority != UINT32_MAX && incomingPriority <= s.lastActivePriority + hysteresisMargin)
                {
                    continue;
                }

                const UINT32 candLod = (s.virtualPageIndex < m_virtualPageHost.size())
                    ? m_virtualPageHost[s.virtualPageIndex].lodLevel
                    : 0u;
                if (s.lastActivePriority < lowestActivePrio ||
                    (s.lastActivePriority == lowestActivePrio &&
                     (candLod < preemptLod || (candLod == preemptLod && s.lastUsedFrame < preemptOldestUse))))
                {
                    lowestActivePrio = s.lastActivePriority;
                    preemptLod       = candLod;
                    preemptOldestUse = s.lastUsedFrame;
                    bestSlot         = i;
                }
            }
        }

        if (bestSlot == kNVGInvalidIndex)
        {
            return kNVGInvalidIndex;
        }

        // Evict victim leaf streaming page cleanly & decrement parent child-refcounts
        PhysicalPageSlot& victim = m_physicalSlots[bestSlot];
        const UINT32 victimVP    = victim.virtualPageIndex;
        const UINT32 victimMesh  = victim.meshId;

        if (victimVP < m_virtualPages.size())
        {
            DecrementParentRefCounts(victimVP);
            if (victimVP < m_virtualPageHost.size() && m_virtualPageHost[victimVP].valid)
            {
                RecordChangedRegion(
                    outChangedRegions,
                    victimMesh,
                    m_virtualPageHost[victimVP].pageLocalBounds,
                    m_virtualPageHost[victimVP].maxShadowScaledParentError);
            }
            m_virtualPages[victimVP].stateAndFlags &= ~NVG_VPAGE_FLAG_RESIDENT;
            m_virtualPages[victimVP].physicalPageIndex = kNVGInvalidIndex;
            m_virtualPages[victimVP].pageGeneration++;
            UploadVirtualPageEntry(victimVP);
        }

        if (m_residentPageCount > 0u) --m_residentPageCount;
        victim = PhysicalPageSlot{};
        AppendUniqueU32(outEvictedMeshIds, victimMesh);
        ++outEvictedCount;
        ++m_residencyRevision;
        return bestSlot;
    }

    void CoreNVGResidencyManager::UploadVirtualPageEntry(UINT32 globalVirtualPageIndex)
    {
        if (!m_virtualPageTableBuffer || m_virtualPageTableDirty) return;
        if (globalVirtualPageIndex >= m_virtualPages.size()) return;
        if (globalVirtualPageIndex >= m_virtualPageTableBuffer->GetElementCount())
        {
            m_virtualPageTableDirty = true;
            return;
        }

        m_bufferManager->UpdateGPUBuffer(
            m_virtualPageTableBuffer,
            &m_virtualPages[globalVirtualPageIndex],
            sizeof(NVGVirtualPageGPUData),
            globalVirtualPageIndex * sizeof(NVGVirtualPageGPUData));
    }

    BOOL CoreNVGResidencyManager::RegisterVirtualMesh(
        UINT32                    meshId,
        NVGVirtualMeshAsset&&     asset,
        const INVGPageDataSource* customSource,
        UINT64                    currentFrameIndex,
        std::string&              outError)
    {
        outError.clear();
        if (!m_initialized)
        {
            outError = "NVG residency manager is not initialized.";
            return false;
        }
        if (meshId == 0u || asset.clusters.empty() || asset.bvhNodes.empty() || asset.pages.empty() || asset.rootPageCount == 0u)
        {
            outError = "Invalid NVG virtual mesh asset (empty clusters, BVH, or root pages).";
            return false;
        }

        if (m_registeredMeshes.find(meshId) != m_registeredMeshes.end())
        {
            UnregisterVirtualMesh(meshId, currentFrameIndex);
        }

        ReclaimDeferredSlots(currentFrameIndex);

        // In autonomous elastic pool mode, grow pool proactively if free slots are low for incoming root pages
        if (m_autoPoolMode &&
            (asset.rootPageCount > static_cast<UINT32>(m_freePhysicalSlots.size()) ||
             m_residentPageCount + asset.rootPageCount > (m_physicalPoolCapacity * 85u) / 100u) &&
            m_physicalPoolCapacity < kNVGMaxPhysicalPageCapacity)
        {
            const UINT32 targetCap = (std::min)(
                kNVGMaxPhysicalPageCapacity,
                NextPow2AtLeast(m_residentPageCount + asset.rootPageCount + 64u, m_physicalPoolCapacity * 2u));
            GrowPhysicalPoolPreservingResidents(targetCap);
        }

        // Check physical pool capacity for pinned root pages BEFORE mutating tables
        // (PropagateResidentAncestorLRU guarantees ancestor.recentFrame >= descendant.recentFrame, so every
        //  non-root slot outside the protect window peels leaf-first cleanly during sequential AcquirePhysicalSlot).
        const UINT64 effectiveProtect = (std::max)(static_cast<UINT64>(m_minInFlightProtect), 4ull);
        UINT32 evictableCount = 0u;
        for (const PhysicalPageSlot& s : m_physicalSlots)
        {
            if (!s.occupied || s.pinnedRoot) continue;
            if (s.uploadedFrame + effectiveProtect < currentFrameIndex)
            {
                ++evictableCount;
            }
        }
        const UINT32 availableForRoots = static_cast<UINT32>(m_freePhysicalSlots.size()) + evictableCount;
        if (asset.rootPageCount > availableForRoots)
        {
            outError = "NVG physical page pool exhausted: insufficient capacity to pin root pages for virtual mesh.";
            return false;
        }

        const UINT32 nodeCount        = static_cast<UINT32>(asset.bvhNodes.size());
        const UINT32 clusterCount     = static_cast<UINT32>(asset.clusters.size());
        const UINT32 groupCount       = static_cast<UINT32>(asset.clusterGroups.size());
        const UINT32 virtualPageCount = static_cast<UINT32>(asset.pages.size());
        const UINT32 rootPageCount    = asset.rootPageCount;

        const UINT32 nodeBase        = AllocateContiguousRange(m_freeNodeRanges,        m_bvhNodes.size(),     nodeCount);
        const UINT32 clusterBase     = AllocateContiguousRange(m_freeClusterRanges,     m_clusters.size(),     clusterCount);
        const UINT32 groupBase       = AllocateContiguousRange(m_freeGroupRanges,       m_groups.size(),       groupCount);
        const UINT32 virtualPageBase = AllocateContiguousRange(m_freeVirtualPageRanges, m_virtualPages.size(), virtualPageCount);

        // Allocate physical slots for all root pages first (guaranteed to succeed by check above)
        std::vector<UINT32> acquiredRootSlots;
        acquiredRootSlots.reserve(rootPageCount);
        std::vector<UINT32> evictedMeshes;
        UINT32 evictedCnt = 0u;
        for (UINT32 rp = 0u; rp < rootPageCount; ++rp)
        {
            const UINT32 slot = AcquirePhysicalSlot(currentFrameIndex, UINT32_MAX, evictedMeshes, nullptr, evictedCnt);
            if (slot == kNVGInvalidIndex)
            {
                for (UINT32 s : acquiredRootSlots) m_freePhysicalSlots.push_back(s);
                if (nodeBase < m_bvhNodes.size())         ReleaseAndTrimRange(m_freeNodeRanges,        m_bvhNodes,     nodeBase,        nodeCount);
                if (clusterBase < m_clusters.size())      ReleaseAndTrimRange(m_freeClusterRanges,     m_clusters,     clusterBase,     clusterCount);
                if (groupBase < m_groups.size())          ReleaseAndTrimRange(m_freeGroupRanges,       m_groups,       groupBase,       groupCount);
                if (virtualPageBase < m_virtualPages.size()) ReleaseAndTrimRange(m_freeVirtualPageRanges, m_virtualPages, virtualPageBase, virtualPageCount);
                outError = "NVG failed to acquire physical page slot for root page.";
                return false;
            }
            acquiredRootSlots.push_back(slot);
        }

        // Insert record and wire page data source
        asset.virtualMeshId = meshId;
        RegisteredMeshRecord& rec = m_registeredMeshes[meshId];
        rec.meshId           = meshId;
        rec.asset            = std::move(asset);
        rec.nodeBase         = nodeBase;
        rec.nodeCount        = nodeCount;
        rec.clusterBase      = clusterBase;
        rec.clusterCount     = clusterCount;
        rec.groupBase        = groupBase;
        rec.groupCount       = groupCount;
        rec.virtualPageBase  = virtualPageBase;
        rec.virtualPageCount = virtualPageCount;
        rec.rootPageCount    = rootPageCount;

        if (customSource != nullptr)
        {
            rec.pageSource = customSource;
        }
        else
        {
            m_defaultMemorySource.RegisterAsset(meshId, &rec.asset.pages);
            rec.pageSource = &m_defaultMemorySource;
        }

        // Write globally remapped BVH nodes into [nodeBase .. nodeBase + nodeCount)
        if (static_cast<SIZE_T>(nodeBase) + static_cast<SIZE_T>(nodeCount) > m_bvhNodes.size())
        {
            m_bvhNodes.resize(static_cast<SIZE_T>(nodeBase) + static_cast<SIZE_T>(nodeCount));
        }
        for (UINT32 i = 0u; i < nodeCount; ++i)
        {
            NVGHierarchyNodeGPUData n = rec.asset.bvhNodes[i];
            n.meshId = meshId;
            const BOOL isLeaf = ((n.childCountAndFlags & (1u << 16u)) != 0u);
            n.childStart += isLeaf ? clusterBase : nodeBase;
            m_bvhNodes[nodeBase + i] = n;
        }

        // Write globally remapped Cluster metadata into [clusterBase .. clusterBase + clusterCount)
        if (static_cast<SIZE_T>(clusterBase) + static_cast<SIZE_T>(clusterCount) > m_clusters.size())
        {
            m_clusters.resize(static_cast<SIZE_T>(clusterBase) + static_cast<SIZE_T>(clusterCount));
        }
        for (UINT32 i = 0u; i < clusterCount; ++i)
        {
            NVGClusterMetaGPUData c = rec.asset.clusters[i];
            c.meshId            = meshId;
            c.virtualPageId     = virtualPageBase + c.virtualPageId;
            c.selfGroupIndex    = groupBase + c.selfGroupIndex;
            c.parentGroupIndex  = (c.parentGroupIndex == kNVGInvalidIndex) ? kNVGInvalidIndex : (groupBase + c.parentGroupIndex);
            if (c.refinedByChildGroupCount > 0u && c.refinedByChildGroupStart != kNVGInvalidIndex)
            {
                c.refinedByChildGroupStart = groupBase + c.refinedByChildGroupStart;
            }
            m_clusters[clusterBase + i] = c;
        }

        // Write globally remapped ClusterGroup metadata into [groupBase .. groupBase + groupCount)
        if (static_cast<SIZE_T>(groupBase) + static_cast<SIZE_T>(groupCount) > m_groups.size())
        {
            m_groups.resize(static_cast<SIZE_T>(groupBase) + static_cast<SIZE_T>(groupCount));
        }
        for (UINT32 i = 0u; i < groupCount; ++i)
        {
            NVGClusterGroupGPUData g = rec.asset.clusterGroups[i];
            g.meshId           = meshId;
            g.clusterStart     = clusterBase + g.clusterStart;
            g.parentGroupIndex = (g.parentGroupIndex == kNVGInvalidIndex) ? kNVGInvalidIndex : (groupBase + g.parentGroupIndex);
            g.firstVirtualPageId = (g.firstVirtualPageId == kNVGInvalidIndex)
                ? kNVGInvalidIndex
                : (virtualPageBase + g.firstVirtualPageId);
            m_groups[groupBase + i] = g;
        }

        // Compute tight per-page local AABB bounds & conservative shadow parent error from member clusters
        std::vector<AABB>    localPageBounds(virtualPageCount, AABB(Vec3(1.0e30f, 1.0e30f, 1.0e30f), Vec3(-1.0e30f, -1.0e30f, -1.0e30f)));
        std::vector<FLOAT32> localPageMaxShErr(virtualPageCount, 0.0f);
        std::vector<UINT8>   localPageHasBounds(virtualPageCount, 0u);
        for (UINT32 ci = 0u; ci < clusterCount; ++ci)
        {
            const NVGClusterMetaGPUData& lc = rec.asset.clusters[ci];
            if (lc.virtualPageId < virtualPageCount)
            {
                localPageBounds[lc.virtualPageId].Expand(lc.aabbMin);
                localPageBounds[lc.virtualPageId].Expand(lc.aabbMax);
                localPageHasBounds[lc.virtualPageId] = 1u;

                const FLOAT32 pThin = ((lc.clusterFlags & 2u) != 0u)
                    ? (static_cast<FLOAT32>((lc.clusterFlags >> 16u) & 0x3FFFu) * (4.0f / 16383.0f))
                    : lc.thinFeatureMetric;
                const FLOAT32 shParentErr = (lc.parentGroupError >= 1.0e20f)
                    ? 1.0e20f
                    : (lc.parentGroupError * (1.0f + 8.0f * (std::max)(pThin, 0.0f)));
                if (shParentErr > localPageMaxShErr[lc.virtualPageId])
                {
                    localPageMaxShErr[lc.virtualPageId] = shParentErr;
                }
            }
        }

        // Write Virtual Page Table entries into [virtualPageBase .. virtualPageBase + virtualPageCount)
        // and immediately upload pinned Root Pages (0 .. rootPageCount-1)
        if (static_cast<SIZE_T>(virtualPageBase) + static_cast<SIZE_T>(virtualPageCount) > m_virtualPages.size())
        {
            m_virtualPages.resize(static_cast<SIZE_T>(virtualPageBase) + static_cast<SIZE_T>(nodeCount == 0u ? virtualPageCount : virtualPageCount));
            m_virtualPageHost.resize(static_cast<SIZE_T>(virtualPageBase) + static_cast<SIZE_T>(virtualPageCount));
        }
        for (UINT32 p = 0u; p < virtualPageCount; ++p)
        {
            const NVGVirtualPagePayload& page = rec.asset.pages[p];
            const BOOL isRoot = (p < rootPageCount);
            const UINT32 gvp  = virtualPageBase + p;

            NVGVirtualPageGPUData vpe{};
            vpe.stateAndFlags     = NVG_VPAGE_FLAG_VALID | (isRoot ? (NVG_VPAGE_FLAG_ROOT_PINNED | NVG_VPAGE_FLAG_RESIDENT) : 0u);
            vpe.physicalPageIndex = isRoot ? acquiredRootSlots[p] : kNVGInvalidIndex;
            vpe.pageGeneration    = (m_virtualPages[gvp].pageGeneration == 0u) ? 1u : (m_virtualPages[gvp].pageGeneration + 1u);
            vpe.meshIdAndLod      = (meshId & 0x00FFFFFFu) | ((page.lodLevel & 0xFFu) << 24u);
            m_virtualPages[gvp]   = vpe;

            VirtualPageHostMeta hm{};
            hm.meshId                     = meshId;
            hm.localPageIndex             = p;
            hm.lodLevel                   = page.lodLevel;
            hm.residentChildRefCount      = 0u;
            hm.isRootPage                 = isRoot;
            hm.valid                      = true;
            hm.maxShadowScaledParentError = isRoot ? 1.0e20f : (localPageHasBounds[p] ? localPageMaxShErr[p] : 1.0e20f);
            hm.pageLocalBounds            = localPageHasBounds[p] ? localPageBounds[p] : rec.asset.localBounds;
            m_virtualPageHost[gvp]        = std::move(hm);

            if (isRoot)
            {
                const UINT32 slot = acquiredRootSlots[p];
                if (rec.pageSource->ReadVirtualPageData(meshId, p, m_scratchPageBytes.data(), kNVGPageSizeBytes))
                {
                    m_bufferManager->UpdateGPUBuffer(
                        m_physicalPagePoolBuffer,
                        m_scratchPageBytes.data(),
                        kNVGPageSizeBytes,
                        slot * kNVGPageSizeBytes);
                }

                m_physicalSlots[slot].occupied         = true;
                m_physicalSlots[slot].pinnedRoot       = true;
                m_physicalSlots[slot].virtualPageIndex = gvp;
                m_physicalSlots[slot].meshId           = meshId;
                m_physicalSlots[slot].lastUsedFrame    = currentFrameIndex;
                m_physicalSlots[slot].uploadedFrame    = currentFrameIndex;
                ++m_residentPageCount;
                ++m_pinnedRootPageCount;
            }
        }

        // Build strict child-page -> parent-page dependency links from ClusterGroup hierarchy
        const UINT32 meshVPEnd = virtualPageBase + virtualPageCount;
        for (UINT32 gi = 0u; gi < groupCount; ++gi)
        {
            const NVGClusterGroupGPUData& g = m_groups[groupBase + gi];
            if (g.parentGroupIndex == kNVGInvalidIndex || g.parentGroupIndex >= m_groups.size()) continue;

            const UINT32 gFirstVP = g.firstVirtualPageId;
            const UINT32 gPageCnt = UnpackNVGGroupPageCount(g.lodLevel);
            const NVGClusterGroupGPUData& pg = m_groups[g.parentGroupIndex];
            const UINT32 pFirstVP = pg.firstVirtualPageId;
            const UINT32 pPageCnt = UnpackNVGGroupPageCount(pg.lodLevel);
            if (gFirstVP == kNVGInvalidIndex || pFirstVP == kNVGInvalidIndex) continue;

            for (UINT32 gp = 0u; gp < gPageCnt; ++gp)
            {
                const UINT32 childGVP = gFirstVP + gp;
                if (childGVP < virtualPageBase || childGVP >= meshVPEnd) continue;
                VirtualPageHostMeta& childHm = m_virtualPageHost[childGVP];
                if (childHm.isRootPage) continue;

                for (UINT32 pp = 0u; pp < pPageCnt; ++pp)
                {
                    const UINT32 parentGVP = pFirstVP + pp;
                    if (parentGVP == childGVP || parentGVP < virtualPageBase || parentGVP >= meshVPEnd) continue;
                    const VirtualPageHostMeta& parentHm = m_virtualPageHost[parentGVP];
                    if (parentHm.isRootPage || parentHm.lodLevel > childHm.lodLevel || parentGVP < childGVP)
                    {
                        AppendUniqueU32(childHm.parentVPs, parentGVP);
                    }
                }
            }
        }

        // Populate global MeshTable[meshId]
        if (meshId >= m_meshTable.size())
        {
            m_meshTable.resize(static_cast<SIZE_T>(meshId) + 1u, NVGMeshGPUData{});
        }
        NVGMeshGPUData meshGpu{};
        meshGpu.meshId           = meshId;
        meshGpu.flags            = NVG_MESH_FLAG_VALID | (rec.asset.hasSkinning ? NVG_MESH_FLAG_HAS_SKINNING : 0u);
        meshGpu.rootNodeIndex    = nodeBase + rec.asset.rootNodeIndex;
        meshGpu.nodeCount        = nodeCount;
        meshGpu.clusterBaseIndex = clusterBase;
        meshGpu.clusterCount     = clusterCount;
        meshGpu.groupBaseIndex   = groupBase;
        meshGpu.groupCount       = groupCount;
        meshGpu.localBoundsMin   = rec.asset.localBounds.m_Min;
        meshGpu.virtualPageBase  = virtualPageBase;
        meshGpu.localBoundsMax   = rec.asset.localBounds.m_Max;
        meshGpu.virtualPageCount = virtualPageCount;
        m_meshTable[meshId]      = meshGpu;

        m_meshTableDirty        = true;
        m_bvhNodesDirty         = true;
        m_clustersDirty         = true;
        m_groupsDirty           = true;
        m_virtualPageTableDirty = true;
        ++m_residencyRevision;
        return true;
    }

    void CoreNVGResidencyManager::UnregisterVirtualMesh(UINT32 meshId, UINT64 currentFrameIndex)
    {
        auto it = m_registeredMeshes.find(meshId);
        if (it == m_registeredMeshes.end()) return;

        const RegisteredMeshRecord& rec = it->second;

        // Immediately invalidate MeshTable[meshId] so future GPU frames skip it
        if (meshId < m_meshTable.size())
        {
            m_meshTable[meshId] = NVGMeshGPUData{};
            m_meshTableDirty    = true;
        }

        // Invalidate virtual pages & defer physical slot + metadata range recycling past in-flight GPU frames
        const UINT64 effectiveProtect = (std::max)(static_cast<UINT64>(m_minInFlightProtect), 4ull);
        const UINT64 retireAt         = currentFrameIndex + effectiveProtect + 1ull;
        for (UINT32 p = 0u; p < rec.virtualPageCount; ++p)
        {
            const UINT32 gvp = rec.virtualPageBase + p;
            if (gvp >= m_virtualPages.size()) continue;

            const UINT32 physSlot = m_virtualPages[gvp].physicalPageIndex;
            const BOOL   wasRoot  = m_virtualPageHost[gvp].isRootPage;
            const BOOL   wasRes   = ((m_virtualPages[gvp].stateAndFlags & NVG_VPAGE_FLAG_RESIDENT) != 0u);

            m_virtualPages[gvp].stateAndFlags     = 0u;
            m_virtualPages[gvp].physicalPageIndex = kNVGInvalidIndex;
            m_virtualPages[gvp].pageGeneration++;
            m_virtualPageHost[gvp]                = VirtualPageHostMeta{};

            if (wasRes && physSlot < m_physicalSlots.size())
            {
                if (m_residentPageCount > 0u) --m_residentPageCount;
                if (wasRoot && m_pinnedRootPageCount > 0u) --m_pinnedRootPageCount;
                m_physicalSlots[physSlot].pinnedRoot = false;
                m_deferredFreeSlots.push_back({ physSlot, retireAt });
            }
        }

        for (UINT32 i = 0u; i < rec.nodeCount && (rec.nodeBase + i) < m_bvhNodes.size(); ++i)
        {
            m_bvhNodes[rec.nodeBase + i] = NVGHierarchyNodeGPUData{};
        }
        for (UINT32 i = 0u; i < rec.clusterCount && (rec.clusterBase + i) < m_clusters.size(); ++i)
        {
            m_clusters[rec.clusterBase + i] = NVGClusterMetaGPUData{};
        }
        for (UINT32 i = 0u; i < rec.groupCount && (rec.groupBase + i) < m_groups.size(); ++i)
        {
            m_groups[rec.groupBase + i] = NVGClusterGroupGPUData{};
        }

        DeferredMetadataRetirement mr{};
        mr.nodeBase         = rec.nodeBase;
        mr.nodeCount        = rec.nodeCount;
        mr.clusterBase      = rec.clusterBase;
        mr.clusterCount     = rec.clusterCount;
        mr.groupBase        = rec.groupBase;
        mr.groupCount       = rec.groupCount;
        mr.virtualPageBase  = rec.virtualPageBase;
        mr.virtualPageCount = rec.virtualPageCount;
        mr.retireFrame      = retireAt;
        m_deferredMetadataRetirements.push_back(mr);

        m_defaultMemorySource.UnregisterAsset(meshId);
        m_registeredMeshes.erase(it);

        m_bvhNodesDirty         = true;
        m_clustersDirty         = true;
        m_groupsDirty           = true;
        m_virtualPageTableDirty = true;
        ++m_residencyRevision;
    }

    BOOL CoreNVGResidencyManager::HasVirtualMesh(UINT32 meshId) const
    {
        return (m_registeredMeshes.find(meshId) != m_registeredMeshes.end());
    }

    BOOL CoreNVGResidencyManager::GetVirtualMeshBounds(UINT32 meshId, AABB& outLocalBounds) const
    {
        auto it = m_registeredMeshes.find(meshId);
        if (it == m_registeredMeshes.end()) return false;
        outLocalBounds = it->second.asset.localBounds;
        return true;
    }

    BOOL CoreNVGResidencyManager::SerializeVirtualMesh(UINT32 meshId, std::vector<BYTE>& outBlob) const
    {
        auto it = m_registeredMeshes.find(meshId);
        if (it == m_registeredMeshes.end()) return false;
        return NVGBuilder::Serialize(it->second.asset, outBlob);
    }

    BOOL CoreNVGResidencyManager::SyncDirtyMetadataToGPU()
    {
        if (!m_initialized || !m_bufferManager) return false;

        if (m_meshTableDirty && !m_meshTable.empty())
        {
            if (!EnsureStructuredBufferCapacity(m_meshTableBuffer, sizeof(NVGMeshGPUData), static_cast<UINT32>(m_meshTable.size()), GPUBufferUsage::STRUCTURED_SRV))
                return false;
            m_bufferManager->UpdateGPUBuffer(
                m_meshTableBuffer, m_meshTable.data(),
                static_cast<UINT32>(m_meshTable.size() * sizeof(NVGMeshGPUData)), 0u);
            m_meshTableDirty = false;
        }

        if (m_bvhNodesDirty && !m_bvhNodes.empty())
        {
            if (!EnsureStructuredBufferCapacity(m_bvhNodeBuffer, sizeof(NVGHierarchyNodeGPUData), static_cast<UINT32>(m_bvhNodes.size()), GPUBufferUsage::STRUCTURED_SRV))
                return false;
            m_bufferManager->UpdateGPUBuffer(
                m_bvhNodeBuffer, m_bvhNodes.data(),
                static_cast<UINT32>(m_bvhNodes.size() * sizeof(NVGHierarchyNodeGPUData)), 0u);
            m_bvhNodesDirty = false;
        }

        if (m_clustersDirty && !m_clusters.empty())
        {
            if (!EnsureStructuredBufferCapacity(m_clusterMetaBuffer, sizeof(NVGClusterMetaGPUData), static_cast<UINT32>(m_clusters.size()), GPUBufferUsage::STRUCTURED_SRV))
                return false;
            m_bufferManager->UpdateGPUBuffer(
                m_clusterMetaBuffer, m_clusters.data(),
                static_cast<UINT32>(m_clusters.size() * sizeof(NVGClusterMetaGPUData)), 0u);
            m_clustersDirty = false;
        }

        if (m_groupsDirty && !m_groups.empty())
        {
            const UINT32 groupCount = static_cast<UINT32>(m_groups.size());
            if (!EnsureStructuredBufferCapacity(m_clusterGroupBuffer, sizeof(NVGClusterGroupGPUData), groupCount, GPUBufferUsage::STRUCTURED_SRV))
                return false;
            if (!EnsureStructuredBufferCapacity(m_groupActiveMaskBuffer, sizeof(UINT32), groupCount, GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV))
                return false;
            m_bufferManager->UpdateGPUBuffer(
                m_clusterGroupBuffer, m_groups.data(),
                static_cast<UINT32>(m_groups.size() * sizeof(NVGClusterGroupGPUData)), 0u);
            m_groupsDirty = false;
        }

        if (m_virtualPageTableDirty && !m_virtualPages.empty())
        {
            const UINT32 vpCount = static_cast<UINT32>(m_virtualPages.size());
            if (!EnsureStructuredBufferCapacity(m_virtualPageTableBuffer, sizeof(NVGVirtualPageGPUData), vpCount, GPUBufferUsage::STRUCTURED_SRV))
                return false;
            const UINT32 prevReqCap = m_pageRequestBuffer ? m_pageRequestBuffer->GetElementCount() : 0u;
            if (!EnsureStructuredBufferCapacity(m_pageRequestBuffer, sizeof(UINT32), vpCount,
                GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK))
                return false;
            if (m_pageRequestBuffer && m_pageRequestBuffer->GetElementCount() != prevReqCap)
            {
                m_readbackPending        = false;
                m_clearRequestsNextFrame = true;
            }
            m_bufferManager->UpdateGPUBuffer(
                m_virtualPageTableBuffer, m_virtualPages.data(),
                static_cast<UINT32>(m_virtualPages.size() * sizeof(NVGVirtualPageGPUData)), 0u);
            m_virtualPageTableDirty = false;
        }

        return true;
    }

    void CoreNVGResidencyManager::ProcessStreamingFrame(
        UINT64                                  frameIndex,
        std::vector<UINT32>&                    outResidencyChangedMeshIds,
        NVGStats&                               ioStats,
        std::vector<NVGResidencyChangedRegion>* outChangedRegions)
    {
        const auto tStart = std::chrono::high_resolution_clock::now();
        outResidencyChangedMeshIds.clear();
        if (outChangedRegions) outChangedRegions->clear();

        if (!m_initialized) return;

        // 1. Reclaim physical slots retired past the in-flight GPU frame window
        ReclaimDeferredSlots(frameIndex);

        // 2. Sync any dirty metadata buffers before GPU dispatch
        SyncDirtyMetadataToGPU();

        // 3. Poll async non-blocking readback of GPU page requests from prior completed frame
        const UINT32 totalVP = static_cast<UINT32>(m_virtualPages.size());
        if (m_readbackPending && totalVP > 0u && m_pageRequestBuffer != nullptr)
        {
            m_scratchReadbackRequests.assign(totalVP, 0u);
            const UINT32 readBytes = totalVP * sizeof(UINT32);
            if (m_bufferManager->TryGetGPUBufferReadbackData(m_pageRequestBuffer, m_scratchReadbackRequests.data(), readBytes))
            {
                m_readbackPending = false;

                // Gently decay active priorities before applying fresh readback priorities:
                //   - Camera pages (>= 262144) re-evaluate every frame -> 0.75x decay prevents 1-frame HZB blips from thrashing.
                //   - Shadow-only pages (< 262144) belong to cached NVS shadow pages that do NOT re-run ShadowCull when static ->
                //     15/16 decay prevents 4-frame eviction/invalidation ping-pong while still yielding to camera pages!
                for (PhysicalPageSlot& ps : m_physicalSlots)
                {
                    if (ps.occupied && !ps.pinnedRoot)
                    {
                        if (ps.lastActivePriority >= 262144u)
                        {
                            ps.lastActivePriority = (ps.lastActivePriority * 3u) / 4u;
                        }
                        else
                        {
                            ps.lastActivePriority = (ps.lastActivePriority * 15u) / 16u;
                        }
                    }
                }

                // Unified symmetric DAG priority propagation across BOTH resident and missing virtual pages:
                //   1) Seed effPrioMap with all GPU-requested virtual pages and discover all DAG ancestors up to root.
                //   2) Propagate parentPrio = max(parentPrio, childPrio + 1) bottom-up (finest LOD -> coarsest LOD).
                //   3) Apply identical final priorities to resident slots and missing upload candidates (zero asymmetry!).
                std::unordered_map<UINT32, UINT32> effPrioMap;
                effPrioMap.reserve(512u);

                for (UINT32 vp = 0u; vp < totalVP; ++vp)
                {
                    const UINT32 prio = m_scratchReadbackRequests[vp];
                    if (prio == 0u) continue;
                    if (!m_virtualPageHost[vp].valid) continue;

                    UINT32& curP = effPrioMap[vp];
                    if (prio > curP) curP = prio;

                    UINT32 stack[32]{};
                    UINT32 sp = 0u;
                    stack[sp++] = vp;
                    while (sp > 0u)
                    {
                        const UINT32 curVP = stack[--sp];
                        const VirtualPageHostMeta& curHm = m_virtualPageHost[curVP];
                        for (UINT32 pvp : curHm.parentVPs)
                        {
                            if (pvp >= totalVP || !m_virtualPageHost[pvp].valid) continue;
                            auto [itAnc, inserted] = effPrioMap.emplace(pvp, 0u);
                            (void)itAnc;
                            if (inserted && sp < 32u)
                            {
                                stack[sp++] = pvp;
                            }
                        }
                    }
                }

                struct PageReq
                {
                    UINT32 globalVP = 0u;
                    UINT32 priority = 0u;
                    UINT32 lodLevel = 0u;
                };
                std::vector<PageReq> activeDAGPages;
                activeDAGPages.reserve(effPrioMap.size());
                for (const auto& kv : effPrioMap)
                {
                    activeDAGPages.push_back({ kv.first, kv.second, m_virtualPageHost[kv.first].lodLevel });
                }

                std::sort(activeDAGPages.begin(), activeDAGPages.end(), [](const PageReq& a, const PageReq& b) {
                    if (a.lodLevel != b.lodLevel) return a.lodLevel < b.lodLevel;
                    return a.globalVP < b.globalVP;
                });
                for (const PageReq& r : activeDAGPages)
                {
                    const UINT32 childPrio = effPrioMap[r.globalVP];
                    if (childPrio == 0u) continue;
                    for (UINT32 pvp : m_virtualPageHost[r.globalVP].parentVPs)
                    {
                        auto itP = effPrioMap.find(pvp);
                        if (itP != effPrioMap.end() && itP->second <= childPrio)
                        {
                            itP->second = childPrio + 1u;
                        }
                    }
                }

                std::vector<PageReq> missingReqs;
                missingReqs.reserve(activeDAGPages.size());
                for (const PageReq& r : activeDAGPages)
                {
                    const UINT32 finalPrio = effPrioMap[r.globalVP];
                    if (finalPrio == 0u) continue;

                    const NVGVirtualPageGPUData& entry = m_virtualPages[r.globalVP];
                    if ((entry.stateAndFlags & NVG_VPAGE_FLAG_RESIDENT) != 0u)
                    {
                        const UINT32 physIdx = entry.physicalPageIndex;
                        if (physIdx < m_physicalSlots.size() && m_physicalSlots[physIdx].occupied)
                        {
                            m_physicalSlots[physIdx].lastUsedFrame = frameIndex;
                            if (finalPrio > m_physicalSlots[physIdx].lastActivePriority)
                            {
                                m_physicalSlots[physIdx].lastActivePriority = finalPrio;
                            }
                        }
                    }
                    else
                    {
                        missingReqs.push_back({ r.globalVP, finalPrio, r.lodLevel });
                    }
                }

                ioStats.streamingRequestsQueued = static_cast<UINT32>(missingReqs.size());

                // Sort highest priority first; tie-break coarser LOD first so ancestors precede children
                std::sort(missingReqs.begin(), missingReqs.end(), [](const PageReq& a, const PageReq& b) {
                    if (a.priority != b.priority) return a.priority > b.priority;
                    if (a.lodLevel != b.lodLevel) return a.lodLevel > b.lodLevel;
                    return a.globalVP < b.globalVP;
                });

                // Resolve effective upload budget (autonomous adaptive burst vs explicit user override)
                const UINT32 missingCount = static_cast<UINT32>(missingReqs.size());
                UINT32 effectiveMaxUploads = m_configuredMaxUploads;
                if (m_configuredMaxUploads == kNVGAutoUploadRate)
                {
                    if (missingCount > 512u)      effectiveMaxUploads = 512u;
                    else if (missingCount > 128u) effectiveMaxUploads = 256u;
                    else if (missingCount > 64u)  effectiveMaxUploads = 128u;
                    else                          effectiveMaxUploads = kNVGDefaultMaxUploadsPerFrame;
                }
                m_lastUploadBudgetPages = effectiveMaxUploads;

                // Autonomous elastic physical pool growth up to m_maxPoolBudgetPages (zero CPU stall, preserves all residents!)
                if (m_autoPoolMode && missingCount > 0u && effectiveMaxUploads > 0u && m_physicalPoolCapacity < m_maxPoolBudgetPages)
                {
                    const UINT32 activeDemand = m_residentPageCount + missingCount;
                    if (activeDemand > (m_physicalPoolCapacity * 85u) / 100u ||
                        static_cast<UINT32>(m_freePhysicalSlots.size()) < (std::min)(effectiveMaxUploads, 64u))
                    {
                        const UINT32 desiredPool = (std::min)(
                            m_maxPoolBudgetPages,
                            NextPow2AtLeast(activeDemand + 128u, m_physicalPoolCapacity * 2u));
                        GrowPhysicalPoolPreservingResidents(desiredPool);
                    }
                }

                UINT32 uploadsThisFrame   = 0u;
                UINT32 evictionsThisFrame = 0u;
                UINT32 blockedThisFrame   = 0u;

                for (const PageReq& req : missingReqs)
                {
                    if (uploadsThisFrame >= effectiveMaxUploads) break;

                    const VirtualPageHostMeta& hm = m_virtualPageHost[req.globalVP];
                    if (!hm.valid) continue;

                    // Prefix-closed residency invariant: only upload a child once all its parent pages are resident
                    BOOL parentsReady = true;
                    for (UINT32 pvp : hm.parentVPs)
                    {
                        if (pvp < m_virtualPages.size() &&
                            (m_virtualPages[pvp].stateAndFlags & NVG_VPAGE_FLAG_RESIDENT) == 0u)
                        {
                            parentsReady = false;
                            break;
                        }
                    }
                    if (!parentsReady) continue;

                    auto itMesh = m_registeredMeshes.find(hm.meshId);
                    if (itMesh == m_registeredMeshes.end() || !itMesh->second.pageSource) continue;

                    const UINT32 slot = AcquirePhysicalSlot(
                        frameIndex, req.priority, outResidencyChangedMeshIds, outChangedRegions, evictionsThisFrame);
                    if (slot == kNVGInvalidIndex)
                    {
                        // Pool is saturated with pinned roots or higher-priority active subtrees; fallback stays crack-free
                        ++blockedThisFrame;
                        break;
                    }

                    if (!itMesh->second.pageSource->ReadVirtualPageData(hm.meshId, hm.localPageIndex, m_scratchPageBytes.data(), kNVGPageSizeBytes))
                    {
                        m_freePhysicalSlots.push_back(slot);
                        continue;
                    }

                    // Async GPU upload into physical pool at exact 32 KiB slot offset (zero CPU wait!)
                    m_bufferManager->UpdateGPUBuffer(
                        m_physicalPagePoolBuffer,
                        m_scratchPageBytes.data(),
                        kNVGPageSizeBytes,
                        slot * kNVGPageSizeBytes);

                    m_physicalSlots[slot].occupied           = true;
                    m_physicalSlots[slot].pinnedRoot         = hm.isRootPage;
                    m_physicalSlots[slot].virtualPageIndex   = req.globalVP;
                    m_physicalSlots[slot].meshId             = hm.meshId;
                    m_physicalSlots[slot].lastUsedFrame      = frameIndex;
                    m_physicalSlots[slot].uploadedFrame      = frameIndex;
                    m_physicalSlots[slot].lastActivePriority = req.priority;

                    m_virtualPages[req.globalVP].stateAndFlags     = NVG_VPAGE_FLAG_VALID | NVG_VPAGE_FLAG_RESIDENT | (hm.isRootPage ? NVG_VPAGE_FLAG_ROOT_PINNED : 0u);
                    m_virtualPages[req.globalVP].physicalPageIndex = slot;
                    m_virtualPages[req.globalVP].pageGeneration++;
                    UploadVirtualPageEntry(req.globalVP);

                    IncrementParentRefCounts(req.globalVP);
                    PropagateResidentAncestorLRU(req.globalVP, frameIndex, req.priority);
                    RecordChangedRegion(outChangedRegions, hm.meshId, hm.pageLocalBounds, hm.maxShadowScaledParentError);

                    ++m_residentPageCount;
                    ++uploadsThisFrame;
                    ++m_residencyRevision;
                    AppendUniqueU32(outResidencyChangedMeshIds, hm.meshId);
                }

                // Smooth autonomous quality scaling when working set genuinely exceeds physical VRAM budget:
                //   - Grows ONLY when AcquirePhysicalSlot is blocked (active visible cut cannot fit in pool at current error),
                //     so normal camera-pan LRU leaf streaming (evictionsThisFrame > 0, blockedThisFrame == 0) stays sharp!
                //   - Caps at 4.0x (16x shadow texel harmonization) so constrained 2 MiB pools preserve crisp silhouettes
                //     and active close-up leaf streaming while settling to 0-eviction static equilibrium.
                if (!m_enableAutoQualityScaling)
                {
                    m_adaptiveErrorScale = 1.0f;
                }
                else
                {
                    const BOOL atPoolCeiling = (!m_autoPoolMode || m_physicalPoolCapacity >= m_maxPoolBudgetPages);
                    if (atPoolCeiling && blockedThisFrame > 0u)
                    {
                        m_adaptiveErrorScale = (std::min)(4.0f, m_adaptiveErrorScale * 1.06f + 0.02f);
                    }
                    else if (missingCount == 0u && blockedThisFrame == 0u && evictionsThisFrame == 0u &&
                             m_adaptiveErrorScale > 1.0f &&
                             (!atPoolCeiling || m_residentPageCount * 100u < m_physicalPoolCapacity * 82u))
                    {
                        m_adaptiveErrorScale = (std::max)(1.0f, m_adaptiveErrorScale * 0.94f - 0.02f);
                    }
                }

                ioStats.pagesUploadedThisFrame = uploadsThisFrame;
                ioStats.pagesEvictedThisFrame  = evictionsThisFrame;
                ioStats.evictionBlockedCount   = blockedThisFrame;
                ioStats.streamingUploadedBytes = uploadsThisFrame * kNVGPageSizeBytes;
            }
        }

        // Update residency & budget telemetry
        ioStats.registeredVirtualMeshCount  = static_cast<UINT32>(m_registeredMeshes.size());
        ioStats.totalVirtualPages           = totalVP;
        ioStats.residentPhysicalPages       = m_residentPageCount;
        ioStats.pinnedRootPages             = m_pinnedRootPageCount;
        ioStats.physicalPoolCapacity        = m_physicalPoolCapacity;
        ioStats.physicalPoolBudgetPages     = m_maxPoolBudgetPages;
        ioStats.currentUploadBudgetPages    = m_lastUploadBudgetPages;
        ioStats.effectiveAdaptiveErrorScale = m_adaptiveErrorScale;
        ioStats.geometryPoolVramMB          = GetGeometryPoolVramMB();
        ioStats.metadataVramMB              = GetMetadataVramMB();

        const auto tEnd = std::chrono::high_resolution_clock::now();
        ioStats.cpuStreamingMs = std::chrono::duration<FLOAT32, std::milli>(tEnd - tStart).count();
    }

    void CoreNVGResidencyManager::QueuePageRequestReadback()
    {
        if (!m_initialized || !m_bufferManager || !m_pageRequestBuffer || m_readbackPending) return;
        const UINT32 totalVP = static_cast<UINT32>(m_virtualPages.size());
        if (totalVP == 0u) return;

        const UINT32 readBytes = totalVP * sizeof(UINT32);
        if (m_bufferManager->RequestGPUBufferReadback(m_pageRequestBuffer, 0u, readBytes))
        {
            m_readbackPending        = true;
            m_clearRequestsNextFrame = true;
        }
    }

    FLOAT32 CoreNVGResidencyManager::GetGeometryPoolVramMB() const
    {
        const UINT64 bytes = static_cast<UINT64>(m_physicalPoolCapacity) * static_cast<UINT64>(kNVGPageSizeBytes);
        return static_cast<FLOAT32>(static_cast<double>(bytes) / (1024.0 * 1024.0));
    }

    FLOAT32 CoreNVGResidencyManager::GetMetadataVramMB() const
    {
        UINT64 bytes = 0ull;
        if (m_virtualPageTableBuffer) bytes += m_virtualPageTableBuffer->GetByteSize();
        if (m_pageRequestBuffer)      bytes += m_pageRequestBuffer->GetByteSize();
        if (m_meshTableBuffer)        bytes += m_meshTableBuffer->GetByteSize();
        if (m_bvhNodeBuffer)          bytes += m_bvhNodeBuffer->GetByteSize();
        if (m_clusterMetaBuffer)      bytes += m_clusterMetaBuffer->GetByteSize();
        if (m_clusterGroupBuffer)     bytes += m_clusterGroupBuffer->GetByteSize();
        if (m_groupActiveMaskBuffer)  bytes += m_groupActiveMaskBuffer->GetByteSize();
        return static_cast<FLOAT32>(static_cast<double>(bytes) / (1024.0 * 1024.0));
    }
}
