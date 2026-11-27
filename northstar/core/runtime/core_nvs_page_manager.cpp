#include "core_nvs_page_manager.h"
#include "../../include/material/material.h"
#include "../../include/scene/scene.h"
#include "../../include/shader/shader.h"
#include <algorithm>
#include <cstring>
#include <sstream>

namespace Northstar
{
    CoreNVSPageManager::~CoreNVSPageManager()
    {
        Shutdown();
    }

    BOOL CoreNVSPageManager::Initialize(
        NorthstarBackend*           backend,
        CoreBufferManager*          bufferManager,
        CoreComputeManager*         computeManager,
        CoreRenderPipeline*         renderPipeline,
        InternalResourceManager*    internalResourceManager,
        const NVSPageManagerConfig& config)
    {
        if (m_initialized)
        {
            Shutdown();
        }

        if (!backend || !bufferManager || !computeManager || !renderPipeline || !internalResourceManager)
        {
            m_lastError = "CoreNVSPageManager::Initialize: null subsystem dependency.";
            return false;
        }

        m_backend                 = backend;
        m_bufferManager           = bufferManager;
        m_computeManager          = computeManager;
        m_renderPipeline          = renderPipeline;
        m_internalResourceManager = internalResourceManager;
        m_shadowRasterizer.Initialize(backend, bufferManager, internalResourceManager);

        if (!CreateResources(config))
        {
            DestroyResources();
            return false;
        }

        m_initialized = true;
        if (!ResetTablesOnGPU())
        {
            m_initialized = false;
            DestroyResources();
            return false;
        }

        return true;
    }

    void CoreNVSPageManager::Shutdown()
    {
        m_shadowRasterizer.Shutdown();
        DestroyResources();
        m_initialized             = false;
        m_backend                 = nullptr;
        m_bufferManager           = nullptr;
        m_computeManager          = nullptr;
        m_renderPipeline          = nullptr;
        m_internalResourceManager = nullptr;
        m_config                  = NVSPageManagerConfig{};
    }

    static UINT32 NextPow2AtLeast(UINT32 v, UINT32 minVal) noexcept
    {
        UINT32 p = std::max(minVal, 1u);
        while (p < v && p < 0x80000000u)
        {
            p <<= 1u;
        }
        return p;
    }

    static void ComputeDispatchGroups2D(UINT32 totalGroups, UINT32& outGroupsX, UINT32& outGroupsY) noexcept
    {
        if (totalGroups == 0u)
        {
            outGroupsX = 1u;
            outGroupsY = 1u;
            return;
        }
        if (totalGroups <= 65535u)
        {
            outGroupsX = totalGroups;
            outGroupsY = 1u;
            return;
        }
        outGroupsX = 65535u;
        outGroupsY = (totalGroups + 65534u) / 65535u;
    }

    BOOL CoreNVSPageManager::CreateResources(const NVSPageManagerConfig& cfg)
    {
        if (cfg.pagesPerAxis == 0u || cfg.pagesPerAxis > kNVSMaxPagesPerAxis ||
            cfg.clipmapLevelCount == 0u || cfg.clipmapLevelCount > kNVSMaxClipmapLevels ||
            cfg.physicalPoolCapacity == 0u || cfg.physicalPoolCapacity > kNVSMaxPhysicalPoolPages ||
            cfg.maxPageRequestsCapacity == 0u)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: invalid NVSPageManagerConfig bounds.";
            return false;
        }

        UINT32 totalVirtualPages = 0u;
        if (cfg.enableSparseHierarchy)
        {
            m_sparseRootCapacity  = NextPow2AtLeast(cfg.initialSparseRootCapacity, 4096u);
            m_sparseDirCapacity   = NextPow2AtLeast(cfg.initialSparseDirCapacity, 512u);
            m_sparseBlockCapacity = NextPow2AtLeast(cfg.initialSparseBlockCapacity, 1024u);
            totalVirtualPages     = m_sparseBlockCapacity * kNVSSparsePagesPerBlock;
        }
        else
        {
            if (cfg.maxLightCount == 0u || cfg.maxLightCount > kNVSMaxSupportedLights)
            {
                m_lastError = "CoreNVSPageManager::CreateResources: invalid maxLightCount in legacy flat mode.";
                return false;
            }
            totalVirtualPages =
                cfg.maxLightCount * cfg.clipmapLevelCount * cfg.pagesPerAxis * cfg.pagesPerAxis;
            if (totalVirtualPages == 0u || totalVirtualPages > kNVSDefaultTotalVirtualPages)
            {
                m_lastError = "CoreNVSPageManager::CreateResources: totalVirtualPages exceeds 65536 limit in legacy flat mode.";
                return false;
            }
            m_sparseRootCapacity  = 0u;
            m_sparseDirCapacity   = 0u;
            m_sparseBlockCapacity = 0u;
        }

        m_config = cfg;
        m_pendingInvalidationCommands.clear();

        // 1. Physical 128x128 D32_FLOAT Texture2DArray Pool
        TextureDesc poolTexDesc{};
        poolTexDesc.dataSource  = TextureDataSource::EMPTY;
        poolTexDesc.width       = kNVSPageTexelSize;
        poolTexDesc.height      = kNVSPageTexelSize;
        poolTexDesc.depth       = 1u;
        poolTexDesc.mipLevels   = 1u;
        poolTexDesc.arrayLayers = cfg.physicalPoolCapacity;
        poolTexDesc.format      = TextureFormat::D32_FLOAT;
        poolTexDesc.type        = TextureType::TEXTURE_2D_ARRAY;
        poolTexDesc.usage       = TextureUsage::DEPTH_STENCIL;

        m_physicalPoolTexture = m_backend->CreateTexture(poolTexDesc, nullptr);
        if (m_physicalPoolTexture == 0)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create physical pool Texture2DArray.";
            return false;
        }

        // 2. Persistent Virtual Page Table Buffer (flat window in legacy mode; 64-page leaf block pool in sparse mode)
        GPUBufferDesc vptDesc{};
        vptDesc.elementCount = totalVirtualPages;
        vptDesc.stride       = sizeof(NVSVirtualPageEntry);
        vptDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        vptDesc.debugName    = "Northstar.NVS.VirtualPageTable";
        m_virtualPageTableBuffer = m_bufferManager->CreateGPUBuffer(vptDesc);
        if (!m_virtualPageTableBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create VirtualPageTable buffer.";
            return false;
        }

        // 2B. 2-Level Demand-Driven Sparse Page Hierarchy + Ring Free-List Buffer + Miss Records + Batch Invalidation Commands
        const UINT32 sparseWords = cfg.enableSparseHierarchy
            ? (m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride) + 16u + m_sparseDirCapacity + m_sparseBlockCapacity)
            : 64u;
        GPUBufferDesc sphDesc{};
        sphDesc.elementCount = sparseWords;
        sphDesc.stride       = sizeof(UINT32);
        sphDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        sphDesc.debugName    = "Northstar.NVS.SparsePageHierarchy";
        m_sparsePageHierarchyBuffer = m_bufferManager->CreateGPUBuffer(sphDesc);
        if (!m_sparsePageHierarchyBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create SparsePageHierarchy buffer.";
            return false;
        }

        GPUBufferDesc invCmdDesc{};
        invCmdDesc.elementCount = cfg.enableSparseHierarchy ? 4096u : 16u;
        invCmdDesc.stride       = sizeof(NVSInvalidationCommand);
        invCmdDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                  GPUBufferUsage::STRUCTURED_UAV;
        invCmdDesc.debugName    = "Northstar.NVS.InvalidationCommands";
        m_invalidationCommandsBuffer = m_bufferManager->CreateGPUBuffer(invCmdDesc);
        if (!m_invalidationCommandsBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create InvalidationCommands buffer.";
            return false;
        }

        GPUBufferDesc missDesc{};
        missDesc.elementCount = cfg.enableSparseHierarchy
            ? NextPow2AtLeast(std::max(cfg.maxPageRequestsCapacity, 8192u), 8192u)
            : 64u;
        missDesc.stride       = sizeof(NVSCacheMissRecord);
        missDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                GPUBufferUsage::STRUCTURED_UAV |
                                GPUBufferUsage::CPU_READBACK;
        missDesc.debugName    = "Northstar.NVS.CacheMissRecords";
        m_missRecordsBuffer = m_bufferManager->CreateGPUBuffer(missDesc);
        if (!m_missRecordsBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create CacheMissRecords buffer.";
            return false;
        }

        // 3. Persistent Physical Page Metadata Buffer
        GPUBufferDesc pptDesc{};
        pptDesc.elementCount = cfg.physicalPoolCapacity;
        pptDesc.stride       = sizeof(NVSPhysicalPageEntry);
        pptDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        pptDesc.debugName    = "Northstar.NVS.PhysicalPageTable";
        m_physicalPageTableBuffer = m_bufferManager->CreateGPUBuffer(pptDesc);
        if (!m_physicalPageTableBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create PhysicalPageTable buffer.";
            return false;
        }

        // 4. Page Request Buffer
        GPUBufferDesc reqDesc{};
        reqDesc.elementCount = cfg.maxPageRequestsCapacity;
        reqDesc.stride       = sizeof(NVSVirtualPageRequest);
        reqDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        reqDesc.debugName    = "Northstar.NVS.PageRequests";
        m_pageRequestBuffer = m_bufferManager->CreateGPUBuffer(reqDesc);
        if (!m_pageRequestBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create PageRequest buffer.";
            return false;
        }

        // 5. Dirty Physical Slice Indices Buffer (feeds RuntimeLayeredDepthClearVS)
        GPUBufferDesc dirtyDesc{};
        dirtyDesc.elementCount = cfg.physicalPoolCapacity;
        dirtyDesc.stride       = sizeof(UINT32);
        dirtyDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                 GPUBufferUsage::STRUCTURED_UAV |
                                 GPUBufferUsage::CPU_READBACK;
        dirtyDesc.debugName    = "Northstar.NVS.DirtyPhysicalSlices";
        m_dirtyPhysicalSlicesBuffer = m_bufferManager->CreateGPUBuffer(dirtyDesc);
        if (!m_dirtyPhysicalSlicesBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create DirtyPhysicalSlices buffer.";
            return false;
        }

        // 6. Dirty Slice Status / Count Buffer (4 x UINT32)
        GPUBufferDesc countDesc{};
        countDesc.elementCount = 4u;
        countDesc.stride       = sizeof(UINT32);
        countDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                 GPUBufferUsage::STRUCTURED_UAV |
                                 GPUBufferUsage::INDIRECT_ARGUMENT |
                                 GPUBufferUsage::CPU_READBACK;
        countDesc.debugName    = "Northstar.NVS.DirtySliceCount";
        m_dirtySliceCountBuffer = m_bufferManager->CreateGPUBuffer(countDesc);
        if (!m_dirtySliceCountBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create DirtySliceCount buffer.";
            return false;
        }

        // 7. Selective Clear Indirect Draw Args Buffer (4 x UINT32)
        GPUBufferDesc indDesc{};
        indDesc.elementCount = 4u;
        indDesc.stride       = sizeof(UINT32);
        indDesc.usage        = GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::INDIRECT_ARGUMENT |
                               GPUBufferUsage::CPU_READBACK;
        indDesc.debugName    = "Northstar.NVS.SelectiveClearIndirectArgs";
        m_selectiveClearIndirectArgsBuffer = m_bufferManager->CreateGPUBuffer(indDesc);
        if (!m_selectiveClearIndirectArgsBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create SelectiveClearIndirectArgs buffer.";
            return false;
        }

        // 8. GPU Residency Telemetry Buffer
        GPUBufferDesc telDesc{};
        telDesc.elementCount = 1u;
        telDesc.stride       = sizeof(NVSResidencyTelemetry);
        telDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        telDesc.debugName    = "Northstar.NVS.ResidencyTelemetry";
        m_telemetryBuffer = m_bufferManager->CreateGPUBuffer(telDesc);
        if (!m_telemetryBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create Telemetry buffer.";
            return false;
        }

        // 9. Helper candidate buffer for seeding specific physical slices in tests
        GPUBufferDesc seedDesc{};
        seedDesc.elementCount = cfg.physicalPoolCapacity;
        seedDesc.stride       = sizeof(UINT32);
        seedDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        seedDesc.debugName    = "Northstar.NVS.SeedCandidateSlices";
        m_seedCandidateBuffer = m_bufferManager->CreateGPUBuffer(seedDesc);
        if (!m_seedCandidateBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create SeedCandidateSlices buffer.";
            return false;
        }

        // 10. Shared Filtered Caster Cache, Hierarchical Bitmask & Dynamic Invalidation Buffers
        m_filteredCasterCapacity = 4096u;
        GPUBufferDesc fcDesc{};
        fcDesc.elementCount = m_filteredCasterCapacity;
        fcDesc.stride       = sizeof(NVSFilteredCasterObject);
        fcDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                              GPUBufferUsage::STRUCTURED_UAV |
                              GPUBufferUsage::CPU_READBACK;
        fcDesc.debugName    = "Northstar.NVS.FilteredCasters";
        m_filteredCasterBuffer = m_bufferManager->CreateGPUBuffer(fcDesc);
        if (!m_filteredCasterBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create FilteredCasters buffer.";
            return false;
        }

        GPUBufferDesc cbmDesc{};
        cbmDesc.elementCount = kNVSCasterBitmaskWordCount + 64u; // 2048 words + 64 superblock words
        cbmDesc.stride       = sizeof(UINT32);
        cbmDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        cbmDesc.debugName    = "Northstar.NVS.CasterValidBitmask";
        m_casterValidBitmaskBuffer = m_bufferManager->CreateGPUBuffer(cbmDesc);
        if (!m_casterValidBitmaskBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create CasterValidBitmask buffer.";
            return false;
        }

        GPUBufferDesc dibDesc{};
        dibDesc.elementCount = kNVSMaxDynamicInvalidationBoxes;
        dibDesc.stride       = sizeof(NVSDynamicCasterInvalidationBox);
        dibDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        dibDesc.debugName    = "Northstar.NVS.DynamicInvalidationBoxes";
        m_dynamicInvalidationBoxesBuffer = m_bufferManager->CreateGPUBuffer(dibDesc);
        if (!m_dynamicInvalidationBoxesBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create DynamicInvalidationBoxes buffer.";
            return false;
        }

        GPUSceneObjectData dummySceneObj{};
        GPUBufferDesc      dummyDesc{};
        dummyDesc.elementCount = 1u;
        dummyDesc.stride       = sizeof(GPUSceneObjectData);
        dummyDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        dummyDesc.initialData  = &dummySceneObj;
        dummyDesc.debugName    = "Northstar.NVS.DummySceneObjectBuffer";
        m_dummySceneObjectBuffer = m_bufferManager->CreateGPUBuffer(dummyDesc);
        if (!m_dummySceneObjectBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create DummySceneObjectBuffer.";
            return false;
        }

        // 11. Shared GPU Mesh Draw Descriptor, Material Shadow Descriptor & Skinning Payload Tables
        m_cpuMeshDescriptors.assign(kNVSMaxMeshDrawDescriptors, GPUMeshDrawDescriptor{});
        for (UINT32 i = 0u; i < kNVSMaxMeshDrawDescriptors; ++i)
        {
            m_cpuMeshDescriptors[i].meshId = i;
        }
        GPUBufferDesc mddDesc{};
        mddDesc.elementCount = kNVSMaxMeshDrawDescriptors;
        mddDesc.stride       = sizeof(GPUMeshDrawDescriptor);
        mddDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        mddDesc.initialData  = m_cpuMeshDescriptors.data();
        mddDesc.debugName    = "Northstar.NVS.MeshDrawDescriptors";
        m_meshDrawDescriptorBuffer = m_bufferManager->CreateGPUBuffer(mddDesc);
        if (!m_meshDrawDescriptorBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create MeshDrawDescriptors buffer.";
            return false;
        }
        m_meshDescriptorsDirty = false;

        m_cpuMaterialDescriptors.assign(kNVSMaxMaterialShadowDescriptors, GPUMaterialShadowDescriptor{});
        for (UINT32 i = 0u; i < kNVSMaxMaterialShadowDescriptors; ++i)
        {
            m_cpuMaterialDescriptors[i].materialId         = i;
            m_cpuMaterialDescriptors[i].materialFlags      = GPU_MAT_SHADOW_FLAG_VALID |
                                                             GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT |
                                                             GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
            m_cpuMaterialDescriptors[i].materialGeneration = 1u;
        }
        GPUBufferDesc msdDesc{};
        msdDesc.elementCount = kNVSMaxMaterialShadowDescriptors;
        msdDesc.stride       = sizeof(GPUMaterialShadowDescriptor);
        msdDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        msdDesc.initialData  = m_cpuMaterialDescriptors.data();
        msdDesc.debugName    = "Northstar.NVS.MaterialShadowDescriptors";
        m_materialShadowDescriptorBuffer = m_bufferManager->CreateGPUBuffer(msdDesc);
        if (!m_materialShadowDescriptorBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create MaterialShadowDescriptors buffer.";
            return false;
        }
        m_materialDescriptorsDirty = false;

        m_cpuSkinningPayloads.assign(kNVSMaxSkinningPayloads, NVSShadowSkinningPayload{});
        GPUBufferDesc skpDesc{};
        skpDesc.elementCount = kNVSMaxSkinningPayloads * 3u;
        skpDesc.stride       = sizeof(Mat4);
        skpDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                               GPUBufferUsage::STRUCTURED_UAV |
                               GPUBufferUsage::CPU_READBACK;
        skpDesc.initialData  = m_cpuSkinningPayloads.data();
        skpDesc.debugName    = "Northstar.NVS.SkinningPayloads";
        m_skinningPayloadBuffer = m_bufferManager->CreateGPUBuffer(skpDesc);
        if (!m_skinningPayloadBuffer)
        {
            m_lastError = "CoreNVSPageManager::CreateResources: failed to create SkinningPayloads buffer.";
            return false;
        }
        m_skinningPayloadsDirty = false;

        // 12. Default Per-Page Projection, Clipmap, Caster Worklist & Indirect Draw Ring Buffers (3 frame slots)
        m_defaultProjectionSlot.camera = nullptr;
        if (!AllocateProjectionSlotResources(m_defaultProjectionSlot, cfg))
        {
            return false;
        }

        for (auto& camSlot : m_cameraProjectionSlots)
        {
            if (camSlot.camera != nullptr)
            {
                if (!AllocateProjectionSlotResources(camSlot, cfg))
                {
                    return false;
                }
            }
        }

        return true;
    }

    void CoreNVSPageManager::InitializeDefaultCpuClipmapTable(
        std::vector<NVSClipmapLevelGPUData>& outTable,
        const NVSPageManagerConfig&          cfg) const
    {
        const UINT32 initCap = std::max(cfg.initialViewTableCapacity, kNVSClipmapTableMaxEntries);
        outTable.assign(initCap, NVSClipmapLevelGPUData{});
        for (UINT32 lightId = 0u; lightId < kNVSMaxClipmapLights; ++lightId)
        {
            FLOAT32 curPageSize = 2.0f;
            for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
            {
                NVSClipmapLevelState ls{};
                ls.lightId                   = lightId;
                ls.clipmapLevel              = lvl;
                ls.pagesPerAxis              = cfg.pagesPerAxis;
                ls.pageWorldSize             = curPageSize;
                ls.worldExtent               = curPageSize * static_cast<FLOAT32>(cfg.pagesPerAxis);
                ls.texelWorldSize            = curPageSize / static_cast<FLOAT32>(kNVSPageTexelSize);
                ls.lightRight                = Vec3(1.0f, 0.0f, 0.0f);
                ls.lightUp                   = Vec3(0.0f, 1.0f, 0.0f);
                ls.lightDir                  = Vec3(0.0f, 0.0f, 1.0f);
                ls.snappedOriginPageX        = 0;
                ls.snappedOriginPageY        = 0;
                ls.snappedLightCenterX       = 0.0f;
                ls.snappedLightCenterY       = 0.0f;
                ls.receiverMinZ              = -0.5f * ls.worldExtent;
                ls.receiverMaxZ              = +0.5f * ls.worldExtent;
                ls.casterExtrusionDistance   = std::max(256.0f, 2.0f * ls.worldExtent);
                ls.casterDownstreamMargin    = std::max(16.0f, 0.25f * ls.worldExtent);
                ls.casterConservativePadding = 0.5f * ls.pageWorldSize;
                ls.casterMinZ                = ls.receiverMinZ - ls.casterExtrusionDistance - ls.casterConservativePadding;
                ls.casterMaxZ                = ls.receiverMaxZ + ls.casterDownstreamMargin + ls.casterConservativePadding;
                const FLOAT32 zSpan          = std::max(ls.casterMaxZ - ls.casterMinZ, 1e-4f);
                ls.normalBiasWorld           = 1.5f * ls.texelWorldSize;
                ls.depthBiasNdc              = (0.5f * ls.texelWorldSize) / zSpan;
                ComputeNVSLightViewMatrices(ls.lightRight, ls.lightUp, ls.lightDir, ls.worldToLightView, ls.lightViewToWorld);

                const BOOL levelValid = (lightId < cfg.maxLightCount) && (lvl < cfg.clipmapLevelCount);
                outTable[lightId * kNVSMaxClipmapLevels + lvl] = BuildNVSClipmapLevelGPUData(
                    ls,
                    NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
                    levelValid);
                curPageSize *= 2.0f;
            }
        }
    }

    BOOL CoreNVSPageManager::EnsureSlotViewTableCapacity(CameraProjectionSlot& slot, UINT32 minViewCapacity)
    {
        if (!m_bufferManager || !slot.allocated)
        {
            return false;
        }
        if (minViewCapacity <= slot.allocatedViewTableCapacity)
        {
            return true;
        }
        UINT32 targetCap = std::max(slot.allocatedViewTableCapacity, kNVSClipmapTableMaxEntries);
        while (targetCap < minViewCapacity)
        {
            targetCap *= 2u;
        }

        if (m_config.enableSparseHierarchy && targetCap > m_sparseRootCapacity)
        {
            if (!EnsureSparseRootCapacity(targetCap))
            {
                return false;
            }
        }

        slot.cpuClipmapTable.resize(targetCap, NVSClipmapLevelGPUData{});
        slot.committedCpuClipmapTable.resize(targetCap, NVSClipmapLevelGPUData{});

        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            if (!slot.clipmapLevelTableBuffers[s] ||
                !m_bufferManager->ResizeGPUBuffer(
                    slot.clipmapLevelTableBuffers[s],
                    targetCap,
                    slot.cpuClipmapTable.data()))
            {
                m_lastError = "CoreNVSPageManager::EnsureSlotViewTableCapacity: failed to resize ClipmapLevelTable buffer.";
                return false;
            }
            slot.dirtyMinViewSlot[s] = 0xFFFFFFFFu;
            slot.dirtyMaxViewSlot[s] = 0u;

            if (!slot.committedSamplingViewTableBuffers[s] ||
                !m_bufferManager->ResizeGPUBuffer(
                    slot.committedSamplingViewTableBuffers[s],
                    targetCap,
                    slot.committedCpuClipmapTable.data()))
            {
                m_lastError = "CoreNVSPageManager::EnsureSlotViewTableCapacity: failed to resize CommittedSamplingViewTable buffer.";
                return false;
            }
            slot.committedDirtyMin[s] = 0xFFFFFFFFu;
            slot.committedDirtyMax[s] = 0u;
        }

        slot.allocatedViewTableCapacity = targetCap;
        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            slot.committedDirtyMin[s] = 0u;
            slot.committedDirtyMax[s] = targetCap;
        }
        return true;
    }

    BOOL CoreNVSPageManager::FlushSlotViewTableBuffer(CameraProjectionSlot& slot, UINT32 fSlot)
    {
        if (!m_bufferManager || !slot.allocated || fSlot >= kNVSProjectionFrameSlots)
        {
            return false;
        }
        GPUBuffer* dstBuf = slot.clipmapLevelTableBuffers[fSlot];
        if (!dstBuf)
        {
            return false;
        }
        const UINT32 dMin = slot.dirtyMinViewSlot[fSlot];
        const UINT32 dMax = slot.dirtyMaxViewSlot[fSlot];
        if (dMin >= dMax || dMin >= static_cast<UINT32>(slot.cpuClipmapTable.size()))
        {
            return true;
        }
        const UINT32 clampedMax = std::min<UINT32>(dMax, static_cast<UINT32>(slot.cpuClipmapTable.size()));
        const UINT32 count      = clampedMax - dMin;
        if (count > 0u)
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    dstBuf,
                    slot.cpuClipmapTable.data() + dMin,
                    count * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    dMin * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData))))
            {
                return false;
            }
        }
        slot.dirtyMinViewSlot[fSlot] = 0xFFFFFFFFu;
        slot.dirtyMaxViewSlot[fSlot] = 0u;
        return true;
    }

    BOOL CoreNVSPageManager::AllocateProjectionSlotResources(
        CameraProjectionSlot&       slot,
        const NVSPageManagerConfig& cfg)
    {
        Camera* savedCamera = slot.camera;
        DestroyProjectionSlotResources(slot);
        slot.camera     = savedCamera;
        slot.activeSlot = 0u;
        InitializeDefaultCpuClipmapTable(slot.cpuClipmapTable, cfg);
        const UINT32 initViewCap        = static_cast<UINT32>(slot.cpuClipmapTable.size());
        slot.allocatedViewTableCapacity = initViewCap;
        slot.activeViewTableCount       = kNVSClipmapTableMaxEntries;
        slot.committedViewTableCount    = 0u;
        slot.committedCpuClipmapTable.assign(initViewCap, NVSClipmapLevelGPUData{});

        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            slot.dirtyMinViewSlot[s]  = 0xFFFFFFFFu;
            slot.dirtyMaxViewSlot[s]  = 0u;
            slot.committedDirtyMin[s] = 0xFFFFFFFFu;
            slot.committedDirtyMax[s] = 0u;

            GPUBufferDesc cmDesc{};
            cmDesc.elementCount = initViewCap;
            cmDesc.stride       = sizeof(NVSClipmapLevelGPUData);
            cmDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                  GPUBufferUsage::STRUCTURED_UAV |
                                  GPUBufferUsage::CPU_READBACK;
            cmDesc.debugName    = "Northstar.NVS.ClipmapLevelTable";
            slot.clipmapLevelTableBuffers[s] = m_bufferManager->CreateGPUBuffer(cmDesc);
            if (!slot.clipmapLevelTableBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create ClipmapLevelTable buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            if (!m_bufferManager->UpdateGPUBuffer(
                    slot.clipmapLevelTableBuffers[s],
                    slot.cpuClipmapTable.data(),
                    initViewCap * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    0u))
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to upload initial ClipmapLevelTable.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            GPUBufferDesc prdDesc{};
            prdDesc.elementCount = cfg.physicalPoolCapacity;
            prdDesc.stride       = sizeof(NVSPageRenderData);
            prdDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            prdDesc.debugName    = "Northstar.NVS.PageRenderData";
            slot.pageRenderDataBuffers[s] = m_bufferManager->CreateGPUBuffer(prdDesc);
            if (!slot.pageRenderDataBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create PageRenderData buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            GPUBufferDesc actDesc{};
            actDesc.elementCount = cfg.physicalPoolCapacity;
            actDesc.stride       = sizeof(UINT32);
            actDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::INDIRECT_ARGUMENT |
                                   GPUBufferUsage::CPU_READBACK;
            actDesc.debugName    = "Northstar.NVS.ActiveRenderPageList";
            slot.activeRenderPageListBuffers[s] = m_bufferManager->CreateGPUBuffer(actDesc);
            if (!slot.activeRenderPageListBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create ActiveRenderPageList buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            GPUBufferDesc ptelDesc{};
            ptelDesc.elementCount = 1u;
            ptelDesc.stride       = sizeof(NVSPageProjectionTelemetry);
            ptelDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                    GPUBufferUsage::STRUCTURED_UAV |
                                    GPUBufferUsage::INDIRECT_ARGUMENT |
                                    GPUBufferUsage::CPU_READBACK;
            ptelDesc.debugName    = "Northstar.NVS.ProjectionTelemetry";
            slot.projectionTelemetryBuffers[s] = m_bufferManager->CreateGPUBuffer(ptelDesc);
            if (!slot.projectionTelemetryBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create ProjectionTelemetry buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Per-Page Caster Ranges Buffer
            GPUBufferDesc cprDesc{};
            cprDesc.elementCount = cfg.physicalPoolCapacity;
            cprDesc.stride       = sizeof(NVSCasterPageRange);
            cprDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            cprDesc.debugName    = "Northstar.NVS.CasterPageRanges";
            slot.casterPageRangeBuffers[s] = m_bufferManager->CreateGPUBuffer(cprDesc);
            if (!slot.casterPageRangeBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create CasterPageRanges buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Compacted Caster Worklist Buffer
            const UINT32 initWlCap = (slot.worklistCapacity > 0u) ? slot.worklistCapacity : 131072u;
            slot.worklistCapacity          = initWlCap;
            slot.allocatedWorklistCapacity = std::max(initWlCap, 1024u);
            GPUBufferDesc wlDesc{};
            wlDesc.elementCount = slot.allocatedWorklistCapacity;
            wlDesc.stride       = sizeof(NVSCasterDrawCandidate);
            wlDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                  GPUBufferUsage::STRUCTURED_UAV |
                                  GPUBufferUsage::CPU_READBACK;
            wlDesc.debugName    = "Northstar.NVS.CasterWorklist";
            slot.casterWorklistBuffers[s] = m_bufferManager->CreateGPUBuffer(wlDesc);
            if (!slot.casterWorklistBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create CasterWorklist buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Caster Culling Telemetry Buffer
            GPUBufferDesc ctelDesc{};
            ctelDesc.elementCount = 1u;
            ctelDesc.stride       = sizeof(NVSCasterCullTelemetry);
            ctelDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                    GPUBufferUsage::STRUCTURED_UAV |
                                    GPUBufferUsage::INDIRECT_ARGUMENT |
                                    GPUBufferUsage::CPU_READBACK;
            ctelDesc.debugName    = "Northstar.NVS.CasterCullTelemetry";
            slot.casterCullTelemetryBuffers[s] = m_bufferManager->CreateGPUBuffer(ctelDesc);
            if (!slot.casterCullTelemetryBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create CasterCullTelemetry buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Worklist Indirect Args Buffer (16 x UINT32: [0..7] indirect args, [8..15] 3D spatial union AABB)
            GPUBufferDesc windDesc{};
            windDesc.elementCount = 16u;
            windDesc.stride       = sizeof(UINT32);
            windDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                    GPUBufferUsage::STRUCTURED_UAV |
                                    GPUBufferUsage::INDIRECT_ARGUMENT |
                                    GPUBufferUsage::CPU_READBACK;
            windDesc.debugName    = "Northstar.NVS.WorklistIndirectArgs";
            slot.worklistIndirectArgsBuffers[s] = m_bufferManager->CreateGPUBuffer(windDesc);
            if (!slot.worklistIndirectArgsBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create WorklistIndirectArgs buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Per-Draw / Per-Instance Shadow Draw Contexts Buffer
            GPUBufferDesc sdcDesc{};
            sdcDesc.elementCount = kNVSMaxShadowDrawContextsCapacity;
            sdcDesc.stride       = sizeof(NVSShadowDrawContext);
            sdcDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            sdcDesc.debugName    = "Northstar.NVS.ShadowDrawContexts";
            slot.shadowDrawContextBuffers[s] = m_bufferManager->CreateGPUBuffer(sdcDesc);
            if (!slot.shadowDrawContextBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create ShadowDrawContexts buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // 64-Byte ExecuteIndirect Draw Commands Buffer (Dynamic Bucket Capacity)
            GPUBufferDesc idcDesc{};
            idcDesc.elementCount = kNVSInitialShadowRasterBucketCapacity * kNVSMaxIndirectCommandsPerBucket;
            idcDesc.stride       = sizeof(NVSIndirectDrawCommand);
            idcDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::INDIRECT_ARGUMENT |
                                   GPUBufferUsage::CPU_READBACK;
            idcDesc.debugName    = "Northstar.NVS.IndirectDrawCommands";
            slot.indirectDrawCommandBuffers[s] = m_bufferManager->CreateGPUBuffer(idcDesc);
            if (!slot.indirectDrawCommandBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create IndirectDrawCommands buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Per-Bucket Command Count Buffer ((allocatedShadowBucketCapacity + 1) x UINT32, ExecuteIndirect CountBuffer)
            GPUBufferDesc bccDesc{};
            bccDesc.elementCount = kNVSInitialShadowRasterBucketCapacity + 1u;
            bccDesc.stride       = sizeof(UINT32);
            bccDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::INDIRECT_ARGUMENT |
                                   GPUBufferUsage::CPU_READBACK;
            bccDesc.debugName    = "Northstar.NVS.BucketCommandCounts";
            slot.bucketCommandCountBuffers[s] = m_bufferManager->CreateGPUBuffer(bccDesc);
            if (!slot.bucketCommandCountBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create BucketCommandCounts buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Per-Page Render Commit Records Buffer (feeds RuntimeNVSPageResidency Mode 4)
            GPUBufferDesc pcrDesc{};
            pcrDesc.elementCount = cfg.physicalPoolCapacity;
            pcrDesc.stride       = sizeof(NVSPageRenderCommitRecord);
            pcrDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            pcrDesc.debugName    = "Northstar.NVS.PageCommitRecords";
            slot.pageCommitRecordBuffers[s] = m_bufferManager->CreateGPUBuffer(pcrDesc);
            if (!slot.pageCommitRecordBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create PageCommitRecords buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Indirect Draw Generation Telemetry Buffer
            GPUBufferDesc idtDesc{};
            idtDesc.elementCount = 1u;
            idtDesc.stride       = sizeof(NVSIndirectDrawTelemetry);
            idtDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            idtDesc.debugName    = "Northstar.NVS.IndirectDrawTelemetry";
            slot.indirectDrawTelemetryBuffers[s] = m_bufferManager->CreateGPUBuffer(idtDesc);
            if (!slot.indirectDrawTelemetryBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create IndirectDrawTelemetry buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Reflected Per-Draw-Group Shader Binding Metadata Buffers
            GPUBufferDesc dgbDesc{};
            dgbDesc.elementCount = 64u;
            dgbDesc.stride       = sizeof(NVSDrawGroupShaderBindingDesc);
            dgbDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
            dgbDesc.debugName    = "Northstar.NVS.DrawGroupBindings";
            slot.drawGroupBindingsBuffers[s] = m_bufferManager->CreateGPUBuffer(dgbDesc);
            if (!slot.drawGroupBindingsBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create DrawGroupBindings buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            GPUBufferDesc dgpDesc{};
            dgpDesc.elementCount = 512u;
            dgpDesc.stride       = sizeof(NVSDrawGroupFieldPatch);
            dgpDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
            dgpDesc.debugName    = "Northstar.NVS.DrawGroupFieldPatches";
            slot.drawGroupFieldPatchesBuffers[s] = m_bufferManager->CreateGPUBuffer(dgpDesc);
            if (!slot.drawGroupFieldPatchesBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create DrawGroupFieldPatches buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            GPUBufferDesc tplDesc{};
            tplDesc.elementCount = 4096u;
            tplDesc.stride       = sizeof(UINT32);
            tplDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
            tplDesc.debugName    = "Northstar.NVS.TemplateWords";
            slot.templateWordsBuffers[s] = m_bufferManager->CreateGPUBuffer(tplDesc);
            if (!slot.templateWordsBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create TemplateWords buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // GPU-Driven Per-Draw Constant Buffer Pool (256B-aligned blocks, 2097152 words = 8 MB)
            GPUBufferDesc cbvPoolDesc{};
            cbvPoolDesc.elementCount = 2097152u;
            cbvPoolDesc.stride       = sizeof(UINT32);
            cbvPoolDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                       GPUBufferUsage::STRUCTURED_UAV |
                                       GPUBufferUsage::INDIRECT_ARGUMENT;
            cbvPoolDesc.debugName    = "Northstar.NVS.PerDrawCBVWords";
            slot.perDrawCBVWordsBuffers[s] = m_bufferManager->CreateGPUBuffer(cbvPoolDesc);
            if (!slot.perDrawCBVWordsBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create PerDrawCBVWords buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // GPU-Driven Per-Batch Instance StructuredBuffer Pool (stride-aligned blocks, 1048576 words = 4 MB)
            GPUBufferDesc instPoolDesc{};
            instPoolDesc.elementCount = 1048576u;
            instPoolDesc.stride       = sizeof(UINT32);
            instPoolDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                        GPUBufferUsage::STRUCTURED_UAV;
            instPoolDesc.debugName    = "Northstar.NVS.PerBatchInstanceWords";
            slot.perBatchInstanceWordsBuffers[s] = m_bufferManager->CreateGPUBuffer(instPoolDesc);
            if (!slot.perBatchInstanceWordsBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create PerBatchInstanceWords buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // GPU Allocation Counters ([0] = CBV word offset, [1] = Instance word offset)
            GPUBufferDesc allocCntDesc{};
            allocCntDesc.elementCount = 4u;
            allocCntDesc.stride       = sizeof(UINT32);
            allocCntDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                        GPUBufferUsage::STRUCTURED_UAV |
                                        GPUBufferUsage::CPU_READBACK;
            allocCntDesc.debugName    = "Northstar.NVS.AllocationCounters";
            slot.allocationCountersBuffers[s] = m_bufferManager->CreateGPUBuffer(allocCntDesc);
            if (!slot.allocationCountersBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create AllocationCounters buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }

            // Committed Shadow View Table Snapshot Buffer (race-free DepthPrepass OFF / ON sampling)
            GPUBufferDesc csvDesc{};
            csvDesc.elementCount = initViewCap;
            csvDesc.stride       = sizeof(NVSClipmapLevelGPUData);
            csvDesc.usage        = GPUBufferUsage::STRUCTURED_SRV |
                                   GPUBufferUsage::STRUCTURED_UAV |
                                   GPUBufferUsage::CPU_READBACK;
            csvDesc.initialData  = slot.committedCpuClipmapTable.data();
            csvDesc.debugName    = "Northstar.NVS.CommittedSamplingViewTable";
            slot.committedSamplingViewTableBuffers[s] = m_bufferManager->CreateGPUBuffer(csvDesc);
            if (!slot.committedSamplingViewTableBuffers[s])
            {
                m_lastError = "CoreNVSPageManager::AllocateProjectionSlotResources: failed to create CommittedSamplingViewTable buffer.";
                DestroyProjectionSlotResources(slot);
                slot.camera = savedCamera;
                return false;
            }
        }

        slot.committedActiveSlot           = 0u;
        slot.hasCommittedShadowSnapshot    = false;
        slot.committedSnapshotFrameIndex   = 0u;
        slot.lastUploadedDrawGroupCount    = 0u;
        slot.allocated                     = true;
        slot.allocatedShadowBucketCapacity = kNVSInitialShadowRasterBucketCapacity;
        return true;
    }

    void CoreNVSPageManager::DestroyProjectionSlotResources(CameraProjectionSlot& slot)
    {
        auto destroyBuf = [&](GPUBuffer*& buf) {
            if (buf && m_bufferManager)
            {
                if (m_computeManager)
                {
                    m_computeManager->OnGPUBufferDestroyed(buf);
                }
                m_bufferManager->DestroyGPUBuffer(buf);
                buf = nullptr;
            }
        };

        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            destroyBuf(slot.committedSamplingViewTableBuffers[s]);
            destroyBuf(slot.allocationCountersBuffers[s]);
            destroyBuf(slot.perBatchInstanceWordsBuffers[s]);
            destroyBuf(slot.perDrawCBVWordsBuffers[s]);
            destroyBuf(slot.templateWordsBuffers[s]);
            destroyBuf(slot.drawGroupFieldPatchesBuffers[s]);
            destroyBuf(slot.drawGroupBindingsBuffers[s]);
            destroyBuf(slot.indirectDrawTelemetryBuffers[s]);
            destroyBuf(slot.pageCommitRecordBuffers[s]);
            destroyBuf(slot.bucketCommandCountBuffers[s]);
            destroyBuf(slot.indirectDrawCommandBuffers[s]);
            destroyBuf(slot.shadowDrawContextBuffers[s]);
            destroyBuf(slot.worklistIndirectArgsBuffers[s]);
            destroyBuf(slot.casterCullTelemetryBuffers[s]);
            destroyBuf(slot.casterWorklistBuffers[s]);
            destroyBuf(slot.casterPageRangeBuffers[s]);
            destroyBuf(slot.projectionTelemetryBuffers[s]);
            destroyBuf(slot.activeRenderPageListBuffers[s]);
            destroyBuf(slot.pageRenderDataBuffers[s]);
            destroyBuf(slot.clipmapLevelTableBuffers[s]);
        }
        slot.allocated                     = false;
        slot.activeSlot                    = 0u;
        slot.committedActiveSlot           = 0u;
        slot.hasCommittedShadowSnapshot    = false;
        slot.committedSnapshotFrameIndex   = 0u;
        slot.lastUploadedDrawGroupCount    = 0u;
        slot.allocatedWorklistCapacity     = 0u;
        slot.allocatedShadowBucketCapacity = 0u;
        slot.activeViewTableCount          = 0u;
        slot.committedViewTableCount       = 0u;
        slot.allocatedViewTableCapacity    = 0u;
        slot.cpuClipmapTable.clear();
        slot.committedCpuClipmapTable.clear();
        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            slot.dirtyMinViewSlot[s]  = 0xFFFFFFFFu;
            slot.dirtyMaxViewSlot[s]  = 0u;
            slot.committedDirtyMin[s] = 0xFFFFFFFFu;
            slot.committedDirtyMax[s] = 0u;
        }
    }

    BOOL CoreNVSPageManager::UploadSlotIndirectBindingTables(
        CameraProjectionSlot&          slot,
        UINT32                         fSlot,
        const NVSShadowRasterPassDesc& passDesc)
    {
        if (!m_bufferManager || !slot.allocated || fSlot >= kNVSProjectionFrameSlots)
        {
            return false;
        }

        std::vector<NVSDrawGroupShaderBindingDesc> bindings;
        std::vector<NVSDrawGroupFieldPatch>        patches;
        std::vector<UINT32>                        templateWords;
        if (!m_shadowRasterizer.BuildDrawGroupBindingTables(passDesc, bindings, patches, templateWords))
        {
            m_lastError = "CoreNVSPageManager::UploadSlotIndirectBindingTables: BuildDrawGroupBindingTables failed.";
            return false;
        }

        slot.lastUploadedDrawGroupCount = static_cast<UINT32>(bindings.size());

        auto ensureAndUpload = [&](GPUBuffer*& buf, UINT32 elemCount, const void* srcData, UINT32 elemStride) -> BOOL {
            if (!buf)
            {
                return false;
            }
            if (elemCount == 0u)
            {
                return true;
            }
            if (elemCount > buf->GetElementCount())
            {
                UINT32 newCap = std::max(buf->GetElementCount(), 16u);
                while (newCap < elemCount)
                {
                    newCap *= 2u;
                }
                if (!m_bufferManager->ResizeGPUBuffer(buf, newCap, nullptr))
                {
                    return false;
                }
            }
            return m_bufferManager->UpdateGPUBuffer(buf, srcData, elemCount * elemStride, 0u);
        };

        if (!ensureAndUpload(
                slot.drawGroupBindingsBuffers[fSlot],
                static_cast<UINT32>(bindings.size()),
                bindings.data(),
                static_cast<UINT32>(sizeof(NVSDrawGroupShaderBindingDesc))) ||
            !ensureAndUpload(
                slot.drawGroupFieldPatchesBuffers[fSlot],
                static_cast<UINT32>(patches.size()),
                patches.data(),
                static_cast<UINT32>(sizeof(NVSDrawGroupFieldPatch))) ||
            !ensureAndUpload(
                slot.templateWordsBuffers[fSlot],
                static_cast<UINT32>(templateWords.size()),
                templateWords.data(),
                static_cast<UINT32>(sizeof(UINT32))))
        {
            m_lastError = "CoreNVSPageManager::UploadSlotIndirectBindingTables: failed to upload draw group binding tables.";
            return false;
        }

        return true;
    }

    BOOL CoreNVSPageManager::EnsureSlotShadowBucketCapacity(CameraProjectionSlot& slot, UINT32 minBuckets)
    {
        if (!m_bufferManager || !slot.allocated)
        {
            return false;
        }
        if (minBuckets <= slot.allocatedShadowBucketCapacity)
        {
            return true;
        }
        UINT32 targetCap = (slot.allocatedShadowBucketCapacity > 0u)
            ? slot.allocatedShadowBucketCapacity
            : kNVSInitialShadowRasterBucketCapacity;
        while (targetCap < minBuckets)
        {
            targetCap *= 2u;
        }
        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            if (!slot.indirectDrawCommandBuffers[s] ||
                !m_bufferManager->ResizeGPUBuffer(
                    slot.indirectDrawCommandBuffers[s],
                    targetCap * kNVSMaxIndirectCommandsPerBucket,
                    nullptr))
            {
                m_lastError = "CoreNVSPageManager::EnsureSlotShadowBucketCapacity: failed to resize IndirectDrawCommands buffer.";
                return false;
            }
            if (!slot.bucketCommandCountBuffers[s] ||
                !m_bufferManager->ResizeGPUBuffer(
                    slot.bucketCommandCountBuffers[s],
                    targetCap + 1u,
                    nullptr))
            {
                m_lastError = "CoreNVSPageManager::EnsureSlotShadowBucketCapacity: failed to resize BucketCommandCounts buffer.";
                return false;
            }
        }
        slot.allocatedShadowBucketCapacity = targetCap;
        return true;
    }

    BOOL CoreNVSPageManager::EnsureMaterialShadowDescriptorCapacity(UINT32 minSlots)
    {
        if (!m_bufferManager || !m_materialShadowDescriptorBuffer)
        {
            return false;
        }
        const UINT32 oldCap = static_cast<UINT32>(m_cpuMaterialDescriptors.size());
        if (minSlots <= oldCap)
        {
            return true;
        }
        UINT32 targetCap = (oldCap > 0u) ? oldCap : kNVSMaxMaterialShadowDescriptors;
        while (targetCap < minSlots)
        {
            targetCap *= 2u;
        }
        m_cpuMaterialDescriptors.resize(targetCap);
        for (UINT32 i = oldCap; i < targetCap; ++i)
        {
            m_cpuMaterialDescriptors[i]                    = GPUMaterialShadowDescriptor{};
            m_cpuMaterialDescriptors[i].materialId         = i;
            m_cpuMaterialDescriptors[i].materialFlags      = GPU_MAT_SHADOW_FLAG_VALID |
                                                             GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT |
                                                             GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
            m_cpuMaterialDescriptors[i].materialGeneration = 1u;
        }
        if (!m_bufferManager->ResizeGPUBuffer(
                m_materialShadowDescriptorBuffer,
                targetCap,
                m_cpuMaterialDescriptors.data()))
        {
            m_lastError = "CoreNVSPageManager::EnsureMaterialShadowDescriptorCapacity: failed to resize MaterialShadowDescriptors buffer.";
            return false;
        }
        return true;
    }

    CoreNVSPageManager::CameraProjectionSlot* CoreNVSPageManager::AcquireCameraProjectionSlot(Camera* camera)
    {
        if (!m_initialized)
        {
            return nullptr;
        }
        if (camera == nullptr)
        {
            return m_defaultProjectionSlot.allocated ? &m_defaultProjectionSlot : nullptr;
        }

        for (auto& s : m_cameraProjectionSlots)
        {
            if (s.camera == camera && s.allocated)
            {
                return &s;
            }
        }

        for (auto& s : m_cameraProjectionSlots)
        {
            if (s.camera == nullptr && !s.allocated)
            {
                s.camera = camera;
                if (!AllocateProjectionSlotResources(s, m_config))
                {
                    s.camera = nullptr;
                    return nullptr;
                }
                return &s;
            }
        }

        CameraProjectionSlot newSlot{};
        newSlot.camera = camera;
        if (!AllocateProjectionSlotResources(newSlot, m_config))
        {
            return nullptr;
        }
        m_cameraProjectionSlots.push_back(newSlot);
        return &m_cameraProjectionSlots.back();
    }

    const CoreNVSPageManager::CameraProjectionSlot* CoreNVSPageManager::FindCameraProjectionSlot(Camera* camera) const
    {
        if (!m_initialized)
        {
            return nullptr;
        }
        if (camera == nullptr)
        {
            return m_defaultProjectionSlot.allocated ? &m_defaultProjectionSlot : nullptr;
        }
        for (const auto& s : m_cameraProjectionSlots)
        {
            if (s.camera == camera && s.allocated)
            {
                return &s;
            }
        }
        return m_defaultProjectionSlot.allocated ? &m_defaultProjectionSlot : nullptr;
    }

    void CoreNVSPageManager::OnCameraDestroyed(Camera* camera)
    {
        if (camera == nullptr)
        {
            return;
        }
        for (auto& s : m_cameraProjectionSlots)
        {
            if (s.camera == camera)
            {
                DestroyProjectionSlotResources(s);
                s.camera = nullptr;
            }
        }
    }

    UINT32 CoreNVSPageManager::GetActiveCameraProjectionSlotCount() const
    {
        UINT32 count = 0u;
        for (const auto& s : m_cameraProjectionSlots)
        {
            if (s.camera != nullptr && s.allocated)
            {
                ++count;
            }
        }
        return count;
    }

    GPUBuffer* CoreNVSPageManager::GetClipmapLevelTableBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->clipmapLevelTableBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPageRenderDataBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->pageRenderDataBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetActiveRenderPageListBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->activeRenderPageListBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetProjectionTelemetryBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->projectionTelemetryBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetClipmapLevelTableBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->clipmapLevelTableBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPageRenderDataBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->pageRenderDataBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetActiveRenderPageListBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->activeRenderPageListBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetProjectionTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->projectionTelemetryBuffers[frameSlot] : nullptr;
    }

    UINT32 CoreNVSPageManager::GetActiveProjectionFrameSlot(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? (s->activeSlot % kNVSProjectionFrameSlots) : 0u;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterPageRangeBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->casterPageRangeBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterWorklistBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->casterWorklistBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterCullTelemetryBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->casterCullTelemetryBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetWorklistIndirectArgsBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->worklistIndirectArgsBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterPageRangeBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->casterPageRangeBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterWorklistBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->casterWorklistBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCasterCullTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->casterCullTelemetryBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetWorklistIndirectArgsBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->worklistIndirectArgsBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetShadowDrawContextBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->shadowDrawContextBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetIndirectDrawCommandBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->indirectDrawCommandBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetBucketCommandCountBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->bucketCommandCountBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPageCommitRecordBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->pageCommitRecordBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetIndirectDrawTelemetryBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->indirectDrawTelemetryBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetShadowDrawContextBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->shadowDrawContextBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetIndirectDrawCommandBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->indirectDrawCommandBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetBucketCommandCountBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->bucketCommandCountBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPageCommitRecordBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->pageCommitRecordBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetIndirectDrawTelemetryBufferForSlot(UINT32 frameSlot, Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && frameSlot < kNVSProjectionFrameSlots) ? s->indirectDrawTelemetryBuffers[frameSlot] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPerDrawCBVWordsBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->perDrawCBVWordsBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetPerBatchInstanceWordsBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->perBatchInstanceWordsBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetAllocationCountersBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->allocationCountersBuffers[s->activeSlot % kNVSProjectionFrameSlots] : nullptr;
    }

    GPUBuffer* CoreNVSPageManager::GetCommittedSamplingViewTableBuffer(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        if (!s)
        {
            return nullptr;
        }
        return s->committedSamplingViewTableBuffers[s->committedActiveSlot % kNVSProjectionFrameSlots];
    }

    const NVSClipmapLevelGPUData* CoreNVSPageManager::GetCpuClipmapLevelTable(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && !s->cpuClipmapTable.empty()) ? s->cpuClipmapTable.data() : nullptr;
    }

    const NVSClipmapLevelGPUData* CoreNVSPageManager::GetCommittedCpuViewTable(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return (s && !s->committedCpuClipmapTable.empty()) ? s->committedCpuClipmapTable.data() : nullptr;
    }

    UINT32 CoreNVSPageManager::GetActiveViewTableCount(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? std::max(s->activeViewTableCount, 1u) : 0u;
    }

    UINT32 CoreNVSPageManager::GetAllocatedViewTableCapacity(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->allocatedViewTableCapacity : 0u;
    }

    UINT32 CoreNVSPageManager::GetCommittedViewTableCount(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->committedViewTableCount : 0u;
    }

    BOOL CoreNVSPageManager::HasCommittedShadowSnapshot(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->hasCommittedShadowSnapshot : false;
    }

    UINT32 CoreNVSPageManager::GetCommittedSnapshotFrameIndex(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->committedSnapshotFrameIndex : 0u;
    }

    BOOL CoreNVSPageManager::CommitSamplingViewTableSnapshot(UINT32 currentFrame, Camera* camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated || slot->cpuClipmapTable.empty())
        {
            return false;
        }

        const UINT32 activeCount = std::min<UINT32>(
            std::max(slot->activeViewTableCount, kNVSClipmapTableMaxEntries),
            static_cast<UINT32>(slot->cpuClipmapTable.size()));

        auto commitSlotSnapshot = [&](CameraProjectionSlot& targetSlot, UINT32 targetFSlot) -> BOOL
        {
            const UINT32 tgtActive = std::min<UINT32>(
                std::max(targetSlot.activeViewTableCount, kNVSClipmapTableMaxEntries),
                static_cast<UINT32>(targetSlot.cpuClipmapTable.size()));
            if (targetSlot.committedCpuClipmapTable.size() < targetSlot.cpuClipmapTable.size())
            {
                targetSlot.committedCpuClipmapTable.resize(targetSlot.cpuClipmapTable.size(), NVSClipmapLevelGPUData{});
            }

            GPUBuffer* dstBuf = targetSlot.committedSamplingViewTableBuffers[targetFSlot];
            if (!dstBuf)
            {
                return false;
            }

            UINT32 upMin = targetSlot.committedDirtyMin[targetFSlot];
            UINT32 upMax = targetSlot.committedDirtyMax[targetFSlot];
            if (!targetSlot.hasCommittedShadowSnapshot)
            {
                const UINT32 fullEnd = std::max(tgtActive, targetSlot.committedViewTableCount);
                for (UINT32 f = 0u; f < kNVSProjectionFrameSlots; ++f)
                {
                    targetSlot.committedDirtyMin[f] = 0u;
                    targetSlot.committedDirtyMax[f] = std::max(targetSlot.committedDirtyMax[f], fullEnd);
                }
                upMin = 0u;
                upMax = fullEnd;
            }
            if (tgtActive < targetSlot.committedViewTableCount)
            {
                upMin = std::min(upMin, tgtActive);
                upMax = std::max(upMax, targetSlot.committedViewTableCount);
            }

            const UINT32 bufCap = dstBuf->GetElementCount();
            if (upMin < upMax && upMin < bufCap)
            {
                const UINT32 clampedMax = std::min<UINT32>(upMax, std::min<UINT32>(bufCap, static_cast<UINT32>(targetSlot.committedCpuClipmapTable.size())));
                const UINT32 copyActiveEnd = std::min<UINT32>(clampedMax, tgtActive);
                if (upMin < copyActiveEnd)
                {
                    std::memcpy(
                        targetSlot.committedCpuClipmapTable.data() + upMin,
                        targetSlot.cpuClipmapTable.data() + upMin,
                        (copyActiveEnd - upMin) * sizeof(NVSClipmapLevelGPUData));
                }
                for (UINT32 i = std::max(upMin, copyActiveEnd); i < clampedMax; ++i)
                {
                    targetSlot.committedCpuClipmapTable[i] = NVSClipmapLevelGPUData{};
                }
                if (clampedMax > upMin)
                {
                    if (!m_bufferManager->UpdateGPUBuffer(
                            dstBuf,
                            targetSlot.committedCpuClipmapTable.data() + upMin,
                            (clampedMax - upMin) * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                            upMin * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData))))
                    {
                        return false;
                    }
                }
            }

            targetSlot.committedDirtyMin[targetFSlot] = 0xFFFFFFFFu;
            targetSlot.committedDirtyMax[targetFSlot] = 0u;
            targetSlot.committedViewTableCount        = tgtActive;
            targetSlot.committedActiveSlot            = targetFSlot;
            targetSlot.hasCommittedShadowSnapshot     = true;
            targetSlot.committedSnapshotFrameIndex    = currentFrame;
            return true;
        };

        UINT32 fSlot = currentFrame % kNVSProjectionFrameSlots;
        if (slot->hasCommittedShadowSnapshot && currentFrame != 0u && slot->committedSnapshotFrameIndex == currentFrame)
        {
            fSlot = slot->committedActiveSlot % kNVSProjectionFrameSlots;
            if (slot->committedViewTableCount == activeCount &&
                slot->committedDirtyMin[fSlot] >= slot->committedDirtyMax[fSlot])
            {
                return true;
            }
        }
        else if (slot->hasCommittedShadowSnapshot && fSlot == (slot->committedActiveSlot % kNVSProjectionFrameSlots))
        {
            fSlot = (slot->committedActiveSlot + 1u) % kNVSProjectionFrameSlots;
        }

        if (!commitSlotSnapshot(*slot, fSlot))
        {
            return false;
        }

        if (camera != nullptr && m_defaultProjectionSlot.allocated)
        {
            if (EnsureSlotViewTableCapacity(m_defaultProjectionSlot, activeCount))
            {
                commitSlotSnapshot(m_defaultProjectionSlot, fSlot);
            }
        }
        return true;
    }

    void CoreNVSPageManager::ResetCommittedShadowSnapshots()
    {
        auto resetSlot = [&](CameraProjectionSlot& s) {
            if (!s.allocated)
            {
                return;
            }
            const UINT32 prevCount = std::max(s.committedViewTableCount, kNVSClipmapTableMaxEntries);
            const UINT32 clrCount  = std::min<UINT32>(prevCount, static_cast<UINT32>(s.committedCpuClipmapTable.size()));
            for (UINT32 i = 0u; i < clrCount; ++i)
            {
                s.committedCpuClipmapTable[i] = NVSClipmapLevelGPUData{};
            }
            s.committedViewTableCount     = 0u;
            s.committedActiveSlot         = 0u;
            s.hasCommittedShadowSnapshot  = false;
            s.committedSnapshotFrameIndex = 0u;
            const UINT32 dirtyResetEnd    = std::max(clrCount, static_cast<UINT32>(s.cpuClipmapTable.size()));
            for (UINT32 f = 0u; f < kNVSProjectionFrameSlots; ++f)
            {
                s.committedDirtyMin[f] = 0u;
                s.committedDirtyMax[f] = dirtyResetEnd;
            }
            if (m_bufferManager && clrCount > 0u)
            {
                for (UINT32 f = 0u; f < kNVSProjectionFrameSlots; ++f)
                {
                    if (s.committedSamplingViewTableBuffers[f])
                    {
                        const UINT32 uploadCnt = std::min<UINT32>(clrCount, s.committedSamplingViewTableBuffers[f]->GetElementCount());
                        if (uploadCnt > 0u)
                        {
                            m_bufferManager->UpdateGPUBuffer(
                                s.committedSamplingViewTableBuffers[f],
                                s.committedCpuClipmapTable.data(),
                                uploadCnt * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                                0u);
                        }
                    }
                }
            }
        };

        resetSlot(m_defaultProjectionSlot);
        for (auto& camSlot : m_cameraProjectionSlots)
        {
            resetSlot(camSlot);
        }
    }

    UINT32 CoreNVSPageManager::GetCasterWorklistCapacity(Camera* camera) const
    {
        const CameraProjectionSlot* s = FindCameraProjectionSlot(camera);
        return s ? s->worklistCapacity : 0u;
    }

    BOOL CoreNVSPageManager::ResizeCasterWorklistCapacity(UINT32 newMaxWorklistCapacity, Camera* camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        if (newMaxWorklistCapacity == 0u || newMaxWorklistCapacity > kNVSMaxCasterWorklistCapacity)
        {
            m_lastError = "CoreNVSPageManager::ResizeCasterWorklistCapacity: capacity out of valid bounds [1 .. 1048576].";
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }

        if (newMaxWorklistCapacity > slot->allocatedWorklistCapacity)
        {
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                if (!m_bufferManager->ResizeGPUBuffer(slot->casterWorklistBuffers[s], newMaxWorklistCapacity, nullptr))
                {
                    m_lastError = "CoreNVSPageManager::ResizeCasterWorklistCapacity: failed to resize CasterWorklist GPUBuffer.";
                    return false;
                }
            }
            slot->allocatedWorklistCapacity = newMaxWorklistCapacity;
        }
        slot->worklistCapacity = newMaxWorklistCapacity;
        return true;
    }

    BOOL CoreNVSPageManager::EnsureFilteredCasterCapacity(UINT32 minSceneSlots)
    {
        if (!m_initialized || !m_bufferManager || !m_filteredCasterBuffer)
        {
            return false;
        }
        const UINT32 clampedReq = std::min(std::max(minSceneSlots, 1u), kNVSMaxFilteredCastersCapacity);
        if (clampedReq <= m_filteredCasterCapacity)
        {
            return true;
        }
        UINT32 newCap = (m_filteredCasterCapacity > 0u) ? m_filteredCasterCapacity : 4096u;
        while (newCap < clampedReq)
        {
            newCap *= 2u;
        }
        if (newCap > kNVSMaxFilteredCastersCapacity)
        {
            newCap = kNVSMaxFilteredCastersCapacity;
        }
        if (!m_bufferManager->ResizeGPUBuffer(m_filteredCasterBuffer, newCap, nullptr))
        {
            m_lastError = "CoreNVSPageManager::EnsureFilteredCasterCapacity: failed to resize FilteredCasters GPUBuffer.";
            return false;
        }
        m_filteredCasterCapacity = newCap;
        return true;
    }

    void CoreNVSPageManager::DestroyResources()
    {
        for (auto& camSlot : m_cameraProjectionSlots)
        {
            DestroyProjectionSlotResources(camSlot);
            camSlot.camera = nullptr;
        }
        m_cameraProjectionSlots.clear();
        DestroyProjectionSlotResources(m_defaultProjectionSlot);

        auto destroyBuf = [&](GPUBuffer*& buf) {
            if (buf && m_bufferManager)
            {
                if (m_computeManager)
                {
                    m_computeManager->OnGPUBufferDestroyed(buf);
                }
                m_bufferManager->DestroyGPUBuffer(buf);
                buf = nullptr;
            }
        };

        destroyBuf(m_skinningPayloadBuffer);
        destroyBuf(m_materialShadowDescriptorBuffer);
        destroyBuf(m_meshDrawDescriptorBuffer);
        m_cpuSkinningPayloads.clear();
        m_cpuMaterialDescriptors.clear();
        m_cpuMeshDescriptors.clear();
        m_skinningPayloadsDirty    = false;
        m_materialDescriptorsDirty = false;
        m_meshDescriptorsDirty     = false;

        destroyBuf(m_dummySceneObjectBuffer);
        destroyBuf(m_dynamicInvalidationBoxesBuffer);
        destroyBuf(m_casterValidBitmaskBuffer);
        destroyBuf(m_filteredCasterBuffer);
        m_filteredCasterCapacity = 0u;
        destroyBuf(m_missRecordsBuffer);
        destroyBuf(m_invalidationCommandsBuffer);
        destroyBuf(m_sparsePageHierarchyBuffer);
        m_sparseRootCapacity  = 0u;
        m_sparseDirCapacity   = 0u;
        m_sparseBlockCapacity = 0u;
        m_pendingInvalidationCommands.clear();
        destroyBuf(m_seedCandidateBuffer);
        destroyBuf(m_telemetryBuffer);
        destroyBuf(m_selectiveClearIndirectArgsBuffer);
        destroyBuf(m_dirtySliceCountBuffer);
        destroyBuf(m_dirtyPhysicalSlicesBuffer);
        destroyBuf(m_pageRequestBuffer);
        destroyBuf(m_physicalPageTableBuffer);
        destroyBuf(m_virtualPageTableBuffer);

        if (m_physicalPoolTexture != 0 && m_backend)
        {
            m_backend->DestroyTexture(m_physicalPoolTexture);
            m_physicalPoolTexture = 0;
        }

        ClearShadowBucketMaterials();
        ResetCommittedShadowSnapshots();
        m_sparseHeadroomReadbackInFlight = false;
        m_dirtyCountReadbackInFlight     = false;
        m_lastResidencyOverflowPages     = 0u;
        m_lastResidencyDirtySlices       = 0u;
    }

    BOOL CoreNVSPageManager::ResizePhysicalPool(UINT32 newPhysicalPoolCapacity)
    {
        if (!m_initialized)
        {
            m_lastError = "CoreNVSPageManager::ResizePhysicalPool: manager not initialized.";
            return false;
        }
        NVSPageManagerConfig newCfg = m_config;
        newCfg.physicalPoolCapacity = newPhysicalPoolCapacity;
        return Reconfigure(newCfg);
    }

    BOOL CoreNVSPageManager::Reconfigure(const NVSPageManagerConfig& newConfig)
    {
        if (!m_backend || !m_bufferManager || !m_computeManager || !m_renderPipeline || !m_internalResourceManager)
        {
            return false;
        }

        const std::vector<GPUMeshDrawDescriptor>       savedMeshDescs = m_cpuMeshDescriptors;
        const std::vector<GPUMaterialShadowDescriptor> savedMatDescs  = m_cpuMaterialDescriptors;
        const std::vector<NVSShadowSkinningPayload>    savedSkinPays  = m_cpuSkinningPayloads;

        m_initialized = false;
        DestroyResources();

        if (!CreateResources(newConfig))
        {
            DestroyResources();
            return false;
        }

        m_initialized = true;
        if (!savedMeshDescs.empty() && m_cpuMeshDescriptors.size() == savedMeshDescs.size())
        {
            m_cpuMeshDescriptors   = savedMeshDescs;
            m_meshDescriptorsDirty = true;
        }
        if (!savedMatDescs.empty() && m_cpuMaterialDescriptors.size() == savedMatDescs.size())
        {
            m_cpuMaterialDescriptors   = savedMatDescs;
            m_materialDescriptorsDirty = true;
        }
        if (!savedSkinPays.empty() && m_cpuSkinningPayloads.size() == savedSkinPays.size())
        {
            m_cpuSkinningPayloads   = savedSkinPays;
            m_skinningPayloadsDirty = true;
        }
        FlushDescriptorTablesToGPU();

        if (!ResetTablesOnGPU())
        {
            m_initialized = false;
            DestroyResources();
            return false;
        }

        return true;
    }

    BOOL CoreNVSPageManager::UploadPageRequests(const NVSVirtualPageRequest* requests, UINT32 requestCount)
    {
        if (!m_initialized || !m_pageRequestBuffer)
        {
            return false;
        }
        if (requestCount == 0u)
        {
            return true;
        }
        if (!requests || requestCount > m_config.maxPageRequestsCapacity)
        {
            m_lastError = "CoreNVSPageManager::UploadPageRequests: null pointer or requestCount exceeds capacity.";
            return false;
        }

        const UINT32 byteSize = requestCount * static_cast<UINT32>(sizeof(NVSVirtualPageRequest));
        return m_bufferManager->UpdateGPUBuffer(m_pageRequestBuffer, requests, byteSize, 0u);
    }

    BOOL CoreNVSPageManager::EnsureSparseRootCapacity(UINT32 minRootViews)
    {
        if (!m_initialized || !m_bufferManager || !m_config.enableSparseHierarchy || !m_sparsePageHierarchyBuffer)
        {
            return !m_config.enableSparseHierarchy;
        }
        if (minRootViews <= m_sparseRootCapacity)
        {
            return true;
        }

        const UINT32 oldRoot  = m_sparseRootCapacity;
        const UINT32 newRoot  = NextPow2AtLeast(minRootViews, std::max(oldRoot * 2u, 4096u));
        const UINT32 oldWords = oldRoot + (m_sparseDirCapacity * kNVSSparseDirStride) + 16u + m_sparseDirCapacity + m_sparseBlockCapacity;
        const UINT32 newWords = newRoot + (m_sparseDirCapacity * kNVSSparseDirStride) + 16u + m_sparseDirCapacity + m_sparseBlockCapacity;
        const UINT32 tailWords = oldWords - oldRoot;

        std::vector<UINT32> fillData(newWords, kNVSSparseUnmappedIndex);
        const GPUBufferCopySpan copySpans[2] = {
            { 0ull,                                        0ull,                                        static_cast<UINT64>(oldRoot) * sizeof(UINT32) },
            { static_cast<UINT64>(newRoot) * sizeof(UINT32), static_cast<UINT64>(oldRoot) * sizeof(UINT32), static_cast<UINT64>(tailWords) * sizeof(UINT32) }
        };
        if (!m_bufferManager->ResizeGPUBufferWithSpans(
                m_sparsePageHierarchyBuffer,
                newWords,
                fillData.data(),
                copySpans,
                2u))
        {
            m_lastError = "CoreNVSPageManager::EnsureSparseRootCapacity: failed to resize SparsePageHierarchy buffer.";
            return false;
        }

        m_sparseRootCapacity = newRoot;
        return true;
    }

    BOOL CoreNVSPageManager::EnsureSparsePoolCapacities(UINT32 minDirCapacity, UINT32 minBlockCapacity)
    {
        if (!m_initialized || !m_bufferManager || !m_config.enableSparseHierarchy ||
            !m_sparsePageHierarchyBuffer || !m_virtualPageTableBuffer)
        {
            return !m_config.enableSparseHierarchy;
        }
        if (minDirCapacity <= m_sparseDirCapacity && minBlockCapacity <= m_sparseBlockCapacity)
        {
            return true;
        }

        const UINT32 oldRoot   = m_sparseRootCapacity;
        const UINT32 oldDirCap = m_sparseDirCapacity;
        const UINT32 oldBlkCap = m_sparseBlockCapacity;
        const UINT32 newDirCap = (minDirCapacity > oldDirCap) ? NextPow2AtLeast(minDirCapacity, oldDirCap * 2u) : oldDirCap;
        const UINT32 newBlkCap = (minBlockCapacity > oldBlkCap) ? NextPow2AtLeast(minBlockCapacity, oldBlkCap * 2u) : oldBlkCap;

        const UINT32 oldAllocBase   = oldRoot + (oldDirCap * kNVSSparseDirStride);
        const UINT32 oldDirFreeBase = oldAllocBase + 16u;
        const UINT32 oldBlkFreeBase = oldDirFreeBase + oldDirCap;
        const UINT32 oldTotalWords  = oldBlkFreeBase + oldBlkCap;

        std::vector<UINT32> oldData(oldTotalWords, kNVSSparseUnmappedIndex);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_sparsePageHierarchyBuffer,
                oldData.data(),
                oldTotalWords * static_cast<UINT32>(sizeof(UINT32))))
        {
            m_lastError = "CoreNVSPageManager::EnsureSparsePoolCapacities: failed to read back SparsePageHierarchy.";
            return false;
        }

        const UINT32 newAllocBase   = oldRoot + (newDirCap * kNVSSparseDirStride);
        const UINT32 newDirFreeBase = newAllocBase + 16u;
        const UINT32 newBlkFreeBase = newDirFreeBase + newDirCap;
        const UINT32 newTotalWords  = newBlkFreeBase + newBlkCap;

        std::vector<UINT32> newData(newTotalWords, kNVSSparseUnmappedIndex);
        std::memcpy(newData.data(), oldData.data(), oldRoot * sizeof(UINT32));
        std::memcpy(newData.data() + oldRoot, oldData.data() + oldRoot, (oldDirCap * kNVSSparseDirStride) * sizeof(UINT32));
        std::memcpy(newData.data() + newAllocBase, oldData.data() + oldAllocBase, 16u * sizeof(UINT32));

        // Rebuild dense linear ring queues starting at head = 0 (32-bit wrap-safe)
        const UINT32 oldDirHead  = oldData[oldAllocBase + 0u];
        const UINT32 oldDirTail  = oldData[oldAllocBase + 1u];
        const UINT32 oldFreeDirs = std::min(static_cast<UINT32>(oldDirTail - oldDirHead), oldDirCap);
        UINT32 newDirTail = 0u;
        for (UINT32 i = 0u; i < oldFreeDirs; ++i)
        {
            newData[newDirFreeBase + newDirTail++] = oldData[oldDirFreeBase + ((oldDirHead + i) % oldDirCap)];
        }
        for (UINT32 d = oldDirCap; d < newDirCap; ++d)
        {
            newData[newDirFreeBase + newDirTail++] = d;
        }
        newData[newAllocBase + 0u] = 0u;
        newData[newAllocBase + 1u] = newDirTail;

        const UINT32 oldBlkHead  = oldData[oldAllocBase + 2u];
        const UINT32 oldBlkTail  = oldData[oldAllocBase + 3u];
        const UINT32 oldFreeBlks = std::min(static_cast<UINT32>(oldBlkTail - oldBlkHead), oldBlkCap);
        UINT32 newBlkTail = 0u;
        for (UINT32 i = 0u; i < oldFreeBlks; ++i)
        {
            newData[newBlkFreeBase + newBlkTail++] = oldData[oldBlkFreeBase + ((oldBlkHead + i) % oldBlkCap)];
        }
        for (UINT32 b = oldBlkCap; b < newBlkCap; ++b)
        {
            newData[newBlkFreeBase + newBlkTail++] = b;
        }
        newData[newAllocBase + 2u] = 0u;
        newData[newAllocBase + 3u] = newBlkTail;
        newData[newAllocBase + 4u] = 0u;
        newData[newAllocBase + 5u] = 0u;

        if (newBlkCap > oldBlkCap)
        {
            const UINT32 oldVptCount = oldBlkCap * kNVSSparsePagesPerBlock;
            const UINT32 newVptCount = newBlkCap * kNVSSparsePagesPerBlock;
            NVSVirtualPageEntry initVEntry{};
            initVEntry.stateAndFlags        = NVS_PAGE_FLAG_EXTENDED_VIEW_ADDR;
            initVEntry.physicalPageIndex    = kNVSInvalidPhysicalPage;
            initVEntry.virtualAddressPacked = 1u << 20u;
            initVEntry.lastTouchedFrame     = 0u;

            std::vector<NVSVirtualPageEntry> vptFill(newVptCount, initVEntry);
            const GPUBufferCopySpan vptSpan{
                0ull,
                0ull,
                static_cast<UINT64>(oldVptCount) * sizeof(NVSVirtualPageEntry)
            };
            if (!m_bufferManager->ResizeGPUBufferWithSpans(
                    m_virtualPageTableBuffer,
                    newVptCount,
                    vptFill.data(),
                    &vptSpan,
                    1u))
            {
                m_lastError = "CoreNVSPageManager::EnsureSparsePoolCapacities: failed to resize VirtualPageTable.";
                return false;
            }
        }

        if (!m_bufferManager->ResizeGPUBuffer(m_sparsePageHierarchyBuffer, newTotalWords, newData.data()))
        {
            m_lastError = "CoreNVSPageManager::EnsureSparsePoolCapacities: failed to resize SparsePageHierarchy.";
            return false;
        }

        m_sparseDirCapacity   = newDirCap;
        m_sparseBlockCapacity = newBlkCap;
        return true;
    }

    BOOL CoreNVSPageManager::PollSparsePoolHeadroomAndGrowIfNeeded()
    {
        if (!m_initialized || !m_bufferManager || !m_config.enableSparseHierarchy ||
            !m_sparsePageHierarchyBuffer || !m_virtualPageTableBuffer)
        {
            return true;
        }

        if (m_sparseHeadroomReadbackInFlight)
        {
            UINT32 headerWords[8]{};
            if (!m_bufferManager->TryGetGPUBufferReadbackData(
                    m_sparsePageHierarchyBuffer,
                    headerWords,
                    static_cast<UINT32>(sizeof(headerWords))))
            {
                return true;
            }

            m_sparseHeadroomReadbackInFlight = false;
            const UINT32 freeDirs = std::min(static_cast<UINT32>(headerWords[1] - headerWords[0]), m_sparseDirCapacity);
            const UINT32 freeBlks = std::min(static_cast<UINT32>(headerWords[3] - headerWords[2]), m_sparseBlockCapacity);
            const UINT32 dirOvf   = headerWords[4];
            const UINT32 blkOvf   = headerWords[5];

            UINT32 targetDirCap = m_sparseDirCapacity;
            UINT32 targetBlkCap = m_sparseBlockCapacity;
            if ((dirOvf > 0u || freeDirs < (m_sparseDirCapacity / 4u)) && targetDirCap < 65536u)
            {
                targetDirCap = std::min(targetDirCap * 2u, 65536u);
            }
            if ((blkOvf > 0u || freeBlks < (m_sparseBlockCapacity / 4u)) && targetBlkCap < 65536u)
            {
                targetBlkCap = std::min(targetBlkCap * 2u, 65536u);
            }
            if (targetDirCap > m_sparseDirCapacity || targetBlkCap > m_sparseBlockCapacity)
            {
                if (!EnsureSparsePoolCapacities(targetDirCap, targetBlkCap))
                {
                    return false;
                }
            }
        }

        const UINT32 allocBase = m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride);
        if (m_bufferManager->RequestGPUBufferReadback(
                m_sparsePageHierarchyBuffer,
                allocBase * static_cast<UINT32>(sizeof(UINT32)),
                8u * static_cast<UINT32>(sizeof(UINT32))))
        {
            m_sparseHeadroomReadbackInFlight = true;
        }
        return true;
    }

    BOOL CoreNVSPageManager::PollResidencyOverflowNonBlocking(
        UINT32& outOverflowPages,
        UINT32& outDirtySlices,
        BOOL&   outFreshSample)
    {
        outOverflowPages = m_lastResidencyOverflowPages;
        outDirtySlices   = m_lastResidencyDirtySlices;
        outFreshSample   = false;

        if (!m_initialized || !m_bufferManager || !m_dirtySliceCountBuffer)
        {
            return false;
        }

        if (m_dirtyCountReadbackInFlight)
        {
            UINT32 counts[4]{};
            if (!m_bufferManager->TryGetGPUBufferReadbackData(
                    m_dirtySliceCountBuffer,
                    counts,
                    static_cast<UINT32>(sizeof(counts))))
            {
                return true;
            }
            m_dirtyCountReadbackInFlight = false;
            m_lastResidencyDirtySlices   = counts[0];
            m_lastResidencyOverflowPages = counts[3];
            outDirtySlices               = m_lastResidencyDirtySlices;
            outOverflowPages             = m_lastResidencyOverflowPages;
            outFreshSample               = true;
        }

        if (m_bufferManager->RequestGPUBufferReadback(
                m_dirtySliceCountBuffer,
                0u,
                4u * static_cast<UINT32>(sizeof(UINT32))))
        {
            m_dirtyCountReadbackInFlight = true;
        }
        return true;
    }

    void CoreNVSPageManager::QueueInvalidationCommand(const NVSInvalidationCommand& cmd)
    {
        if (cmd.mode == NVS_INVALIDATE_NONE)
        {
            return;
        }
        for (NVSInvalidationCommand& existing : m_pendingInvalidationCommands)
        {
            if (existing.viewIndex == cmd.viewIndex && existing.viewCount == cmd.viewCount)
            {
                if (existing.mode == NVS_INVALIDATE_RECLAIM_VIEW || cmd.mode == NVS_INVALIDATE_RECLAIM_VIEW)
                {
                    existing.mode = NVS_INVALIDATE_RECLAIM_VIEW;
                }
                else if (existing.mode == NVS_INVALIDATE_VIEW_SLOT || cmd.mode == NVS_INVALIDATE_VIEW_SLOT)
                {
                    existing.mode = NVS_INVALIDATE_VIEW_SLOT;
                }
                else if (existing.mode == NVS_INVALIDATE_VIEW_ORIGIN && cmd.mode == NVS_INVALIDATE_VIEW_ORIGIN)
                {
                    const INT64 sumDX = static_cast<INT64>(existing.originShiftDeltaX) + static_cast<INT64>(cmd.originShiftDeltaX);
                    const INT64 sumDY = static_cast<INT64>(existing.originShiftDeltaY) + static_cast<INT64>(cmd.originShiftDeltaY);
                    const INT64 pAxis = static_cast<INT64>(std::max(cmd.regionOrPagesAxis, 1u));
                    if (std::abs(sumDX) >= pAxis || std::abs(sumDY) >= pAxis)
                    {
                        existing.mode = NVS_INVALIDATE_VIEW_SLOT;
                    }
                    else
                    {
                        existing.originShiftDeltaX = static_cast<INT32>(sumDX);
                        existing.originShiftDeltaY = static_cast<INT32>(sumDY);
                        existing.snappedOriginX    = cmd.snappedOriginX;
                        existing.snappedOriginY    = cmd.snappedOriginY;
                        existing.regionOrPagesAxis = cmd.regionOrPagesAxis;
                    }
                }
                else
                {
                    existing.mode = NVS_INVALIDATE_VIEW_SLOT;
                }
                return;
            }
        }
        m_pendingInvalidationCommands.push_back(cmd);
    }

    void CoreNVSPageManager::QueueInvalidationCommands(const NVSInvalidationCommand* cmds, UINT32 count)
    {
        if (!cmds || count == 0u)
        {
            return;
        }
        m_pendingInvalidationCommands.reserve(m_pendingInvalidationCommands.size() + count);
        for (UINT32 i = 0u; i < count; ++i)
        {
            QueueInvalidationCommand(cmds[i]);
        }
    }

    BOOL CoreNVSPageManager::BuildPendingInvalidationDispatches(
        UINT32                              currentFrame,
        std::vector<ComputeDispatchCommand>& outDispatches,
        Camera*                             camera)
    {
        (void)camera;
        if (m_pendingInvalidationCommands.empty())
        {
            return true;
        }
        if (!m_initialized || !m_bufferManager || !m_invalidationCommandsBuffer || !m_config.enableSparseHierarchy)
        {
            m_pendingInvalidationCommands.clear();
            return true;
        }

        const UINT32 cmdCount = static_cast<UINT32>(m_pendingInvalidationCommands.size());
        if (cmdCount > m_invalidationCommandsBuffer->GetElementCount())
        {
            const UINT32 newCap = NextPow2AtLeast(cmdCount, 4096u);
            if (!m_bufferManager->ResizeGPUBuffer(m_invalidationCommandsBuffer, newCap, nullptr))
            {
                m_lastError = "CoreNVSPageManager::BuildPendingInvalidationDispatches: failed to resize InvalidationCommands buffer.";
                return false;
            }
        }

        if (!m_bufferManager->UpdateGPUBuffer(
                m_invalidationCommandsBuffer,
                m_pendingInvalidationCommands.data(),
                cmdCount * static_cast<UINT32>(sizeof(NVSInvalidationCommand)),
                0u))
        {
            m_lastError = "CoreNVSPageManager::BuildPendingInvalidationDispatches: failed to upload InvalidationCommands.";
            return false;
        }
        m_pendingInvalidationCommands.clear();

        NVSResidencyPassDesc invPass{};
        invPass.currentFrame          = currentFrame;
        invPass.executeSelectiveClear = false;

        UINT32 gx0 = 1u, gy0 = 1u;
        ComputeDispatchGroups2D((cmdCount + 63u) / 64u, gx0, gy0);
        const UINT32 gx1 = (m_config.physicalPoolCapacity + 63u) / 64u;
        UINT32 gx2 = 1u, gy2 = 1u;
        ComputeDispatchGroups2D((cmdCount + 63u) / 64u, gx2, gy2);

        ComputeDispatchCommand cmdSub0{};
        ComputeDispatchCommand cmdSub1{};
        ComputeDispatchCommand cmdSub2{};
        if (!BuildResidencyComputeDispatchInternal(NVS_DISPATCH_BATCH_INVALIDATE_VIEWS, invPass, cmdSub0, nullptr, 0u, cmdCount, gx0, gy0, camera) ||
            !BuildResidencyComputeDispatchInternal(NVS_DISPATCH_BATCH_INVALIDATE_VIEWS, invPass, cmdSub1, nullptr, 1u, cmdCount, gx1, 1u, camera) ||
            !BuildResidencyComputeDispatchInternal(NVS_DISPATCH_BATCH_INVALIDATE_VIEWS, invPass, cmdSub2, nullptr, 2u, cmdCount, gx2, gy2, camera))
        {
            return false;
        }

        outDispatches.push_back(std::move(cmdSub0));
        outDispatches.push_back(std::move(cmdSub1));
        outDispatches.push_back(std::move(cmdSub2));
        return true;
    }

    BOOL CoreNVSPageManager::ExecutePendingInvalidationsOnGPU(UINT32 currentFrame, Camera* camera)
    {
        if (m_pendingInvalidationCommands.empty())
        {
            return true;
        }
        if (!m_initialized || !m_backend)
        {
            return false;
        }
        std::vector<ComputeDispatchCommand> dispatches;
        dispatches.reserve(3u);
        if (!BuildPendingInvalidationDispatches(currentFrame, dispatches, camera))
        {
            return false;
        }
        if (dispatches.empty())
        {
            return true;
        }
        return m_backend->ExecuteComputeDispatches(dispatches);
    }

    BOOL CoreNVSPageManager::BuildResidencyComputeDispatchInternal(
        NVSResidencyDispatchMode    dispatchMode,
        const NVSResidencyPassDesc& passDesc,
        ComputeDispatchCommand&     outCmd,
        GPUBuffer*                  overridePageRequestBuffer,
        UINT32                      subMode,
        UINT32                      invCmdCount,
        UINT32                      dispatchGroupsX,
        UINT32                      dispatchGroupsY,
        Camera*                     camera)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_internalResourceManager || !m_renderPipeline)
        {
            return false;
        }

        GPUBuffer* reqBuf = overridePageRequestBuffer ? overridePageRequestBuffer : m_pageRequestBuffer;
        if (!reqBuf || !reqBuf->IsValid() || !reqBuf->HasUsage(GPUBufferUsage::STRUCTURED_SRV))
        {
            m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: invalid PageRequests buffer.";
            return false;
        }

        if (dispatchMode == NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE &&
            !passDesc.useGPURequestCountFromTelemetry &&
            passDesc.requestCount > reqBuf->GetElementCount())
        {
            m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: requestCount exceeds maxPageRequestsCapacity.";
            return false;
        }

        if (dispatchMode == NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE &&
            m_config.enableSparseHierarchy &&
            m_missRecordsBuffer &&
            m_bufferManager)
        {
            const UINT32 reqPow2Cap = NextPow2AtLeast(std::min<UINT32>(reqBuf->GetElementCount(), 65536u), 8192u);
            if (reqPow2Cap > m_missRecordsBuffer->GetElementCount())
            {
                if (!m_bufferManager->ResizeGPUBuffer(m_missRecordsBuffer, reqPow2Cap, nullptr))
                {
                    m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: failed to resize CacheMissRecords buffer.";
                    return false;
                }
            }
        }

        ComputeKernel* kernel = m_internalResourceManager->GetComputeKernel(InternalComputeKernelId::NVS_PAGE_RESIDENCY);
        if (!kernel || !kernel->IsValid())
        {
            m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: Runtime.NVSPageResidency kernel unavailable.";
            return false;
        }

        CameraProjectionSlot* projSlot = AcquireCameraProjectionSlot(camera);
        if (projSlot && projSlot->allocated)
        {
            FlushSlotViewTableBuffer(*projSlot, projSlot->activeSlot % kNVSProjectionFrameSlots);
        }
        GPUBuffer* cmBuf = GetClipmapLevelTableBuffer(camera);
        if (!cmBuf || !cmBuf->IsValid())
        {
            m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: invalid ClipmapLevelTable buffer.";
            return false;
        }
        FlushSlotViewTableBuffer(m_defaultProjectionSlot, m_defaultProjectionSlot.activeSlot % kNVSProjectionFrameSlots);

        const UINT32 protectFrames = (passDesc.overrideInFlightProtectFrames > 0u)
            ? passDesc.overrideInFlightProtectFrames
            : m_config.minInFlightProtectFrames;
        const UINT32 markClearedInPass =
            (dispatchMode == NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE &&
             passDesc.executeSelectiveClear &&
             passDesc.markClearedAfterPass) ? 1u : 0u;
        const UINT32 gpuReqCfg = passDesc.useGPURequestCountFromTelemetry
            ? ((dispatchMode == NVS_DISPATCH_COMMIT_RENDERED_PAGES) ? 1u : ((reqBuf->GetElementCount() << 1u) | 1u))
            : 0u;
        const UINT32 viewTableCap = m_config.enableSparseHierarchy
            ? GetActiveViewTableCount(camera)
            : 0u;
        const UINT32 maxMissCap = m_missRecordsBuffer ? m_missRecordsBuffer->GetElementCount() : 0u;

        kernel->SetUInt("NVSResidencyParams", "Config", "dispatchMode",             static_cast<UINT32>(dispatchMode));
        kernel->SetUInt("NVSResidencyParams", "Config", "currentFrame",             passDesc.currentFrame);
        kernel->SetUInt("NVSResidencyParams", "Config", "minInFlightProtectFrames", protectFrames);
        kernel->SetUInt("NVSResidencyParams", "Config", "requestCount",             passDesc.requestCount);
        kernel->SetUInt("NVSResidencyParams", "Config", "pagesPerAxis",             m_config.pagesPerAxis);
        kernel->SetUInt("NVSResidencyParams", "Config", "clipmapLevelCount",        m_config.clipmapLevelCount);
        kernel->SetUInt("NVSResidencyParams", "Config", "maxLightCount",            m_config.maxLightCount);
        kernel->SetUInt("NVSResidencyParams", "Config", "physicalPoolCapacity",     m_config.physicalPoolCapacity);
        kernel->SetUInt("NVSResidencyParams", "Config", "invalidationMode",         static_cast<UINT32>(passDesc.invalidationMode));
        kernel->SetUInt("NVSResidencyParams", "Config", "invalidateLightId",        passDesc.invalidateLightId);
        kernel->SetUInt("NVSResidencyParams", "Config", "invalidateClipmapLevel",   passDesc.invalidateClipmapLevel);
        kernel->SetUInt("NVSResidencyParams", "Config", "invalidateRegionPacked",   passDesc.invalidateRegionPacked);
        kernel->SetInt ("NVSResidencyParams", "Config", "originShiftDeltaX",        passDesc.originShiftDeltaX);
        kernel->SetInt ("NVSResidencyParams", "Config", "originShiftDeltaY",        passDesc.originShiftDeltaY);
        kernel->SetUInt("NVSResidencyParams", "Config", "maxDirtyClearOutput",      m_config.physicalPoolCapacity);
        kernel->SetUInt("NVSResidencyParams", "Config", "clearTailSentinel",        1u);
        kernel->SetInt ("NVSResidencyParams", "Config", "snappedOriginPageX",       passDesc.snappedOriginPageX);
        kernel->SetInt ("NVSResidencyParams", "Config", "snappedOriginPageY",       passDesc.snappedOriginPageY);
        kernel->SetUInt("NVSResidencyParams", "Config", "markClearedInPass",        markClearedInPass);
        kernel->SetUInt("NVSResidencyParams", "Config", "gpuRequestConfig",         gpuReqCfg);
        kernel->SetUInt("NVSResidencyParams", "Config", "sparseMode",               m_config.enableSparseHierarchy ? 1u : 0u);
        kernel->SetUInt("NVSResidencyParams", "Config", "sparseRootCapacity",       m_sparseRootCapacity);
        kernel->SetUInt("NVSResidencyParams", "Config", "sparseDirCapacity",        m_sparseDirCapacity);
        kernel->SetUInt("NVSResidencyParams", "Config", "sparseBlockCapacity",      m_sparseBlockCapacity);
        kernel->SetUInt("NVSResidencyParams", "Config", "invalidationCmdCount",     invCmdCount);
        kernel->SetUInt("NVSResidencyParams", "Config", "maxMissRecordsCapacity",   maxMissCap);
        kernel->SetUInt("NVSResidencyParams", "Config", "clipmapTableSize",         viewTableCap);
        kernel->SetUInt("NVSResidencyParams", "Config", "residencySubMode",         subMode);

        auto unbindAll = [&]() {
            kernel->SetGPUBuffer("NVSResidencyParams", "PageRequests",                 nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "InInvalidationCommands",       nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "InClipmapLevels",              nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "VirtualPageTable",             nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "PhysicalPageTable",            nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "OutDirtyPhysicalSlices",       nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "OutDirtySliceCount",           nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "OutSelectiveClearIndirectArgs", nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "OutTelemetry",                 nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "InOutSparsePageHierarchy",     nullptr);
            kernel->SetGPUBuffer("NVSResidencyParams", "InOutMissRecords",             nullptr);
        };

        const GPUResourceAccess telemetryAccess =
            (passDesc.useGPURequestCountFromTelemetry || dispatchMode == NVS_DISPATCH_BATCH_INVALIDATE_VIEWS)
                ? GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE
                : GPUResourceAccess::UNORDERED_ACCESS_WRITE;

        if (!m_invalidationCommandsBuffer || !m_sparsePageHierarchyBuffer || !m_missRecordsBuffer ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "PageRequests",                 reqBuf,                             GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "InInvalidationCommands",       m_invalidationCommandsBuffer,       GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "InClipmapLevels",              cmBuf,                              GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "VirtualPageTable",             m_virtualPageTableBuffer,           GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "PhysicalPageTable",            m_physicalPageTableBuffer,          GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "OutDirtyPhysicalSlices",       m_dirtyPhysicalSlicesBuffer,        GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "OutDirtySliceCount",           m_dirtySliceCountBuffer,            GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "OutSelectiveClearIndirectArgs", m_selectiveClearIndirectArgsBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "OutTelemetry",                 m_telemetryBuffer,                  telemetryAccess) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "InOutSparsePageHierarchy",     m_sparsePageHierarchyBuffer,        GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSResidencyParams", "InOutMissRecords",             m_missRecordsBuffer,                GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE))
        {
            unbindAll();
            m_lastError = "CoreNVSPageManager::BuildResidencyComputeDispatch: failed to bind GPUBuffers.";
            return false;
        }

        kernel->SetThreadGroupCount(std::max(dispatchGroupsX, 1u), std::max(dispatchGroupsY, 1u), 1u);

        FrameView emptyView{};
        const BOOL built = m_renderPipeline->BuildComputeDispatchCommand(kernel, emptyView, outCmd);
        if (built)
        {
            outCmd.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
        }
        unbindAll();
        return built;
    }

    BOOL CoreNVSPageManager::BuildResidencyComputeDispatch(
        NVSResidencyDispatchMode    dispatchMode,
        const NVSResidencyPassDesc& passDesc,
        ComputeDispatchCommand&     outCmd,
        GPUBuffer*                  overridePageRequestBuffer,
        Camera*                     camera)
    {
        UINT32 subMode  = 0u;
        UINT32 groupsX  = 1u;
        UINT32 groupsY  = 1u;
        if (dispatchMode == NVS_DISPATCH_RESET_ALL_TABLES && m_config.enableSparseHierarchy)
        {
            const UINT32 treeWords = m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride);
            const UINT32 maxItems  = std::max(treeWords, GetTotalVirtualPageCount());
            const UINT32 totalGrp  = std::min<UINT32>(std::max((maxItems + 63u) / 64u, 1u), 4096u);
            subMode = totalGrp;
            groupsX = totalGrp;
        }
        return BuildResidencyComputeDispatchInternal(
            dispatchMode,
            passDesc,
            outCmd,
            overridePageRequestBuffer,
            subMode,
            0u,
            groupsX,
            groupsY,
            camera);
    }

    BOOL CoreNVSPageManager::BuildSelectiveClearPassView(
        FLOAT32               clearDepth,
        LayeredDepthPassView& outClearPass) const
    {
        outClearPass = LayeredDepthPassView{};
        if (!m_initialized || !m_internalResourceManager ||
            !m_dirtyPhysicalSlicesBuffer || !m_selectiveClearIndirectArgsBuffer)
        {
            return false;
        }

        Shader* clearVs = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_CLEAR_VS);
        Shader* passGs  = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_PASS_GS);
        if (!clearVs || clearVs->GetNativeHandle() == 0 || !passGs || passGs->GetNativeHandle() == 0)
        {
            return false;
        }

        outClearPass.depthArrayTexture                      = m_physicalPoolTexture;
        outClearPass.fallbackGeometryShader                 = 0;
        outClearPass.selectiveClear.enabled                 = true;
        outClearPass.selectiveClear.clearDepth              = clearDepth;
        outClearPass.selectiveClear.dirtySliceIndicesBuffer = m_dirtyPhysicalSlicesBuffer->GetNativeHandle();
        outClearPass.selectiveClear.indirectDrawArgsBuffer  = m_selectiveClearIndirectArgsBuffer->GetNativeHandle();
        outClearPass.selectiveClear.maxDirtySlices          = m_config.physicalPoolCapacity;
        outClearPass.selectiveClear.clearVertexShader       = clearVs->GetNativeHandle();
        outClearPass.selectiveClear.clearGeometryShader     = passGs->GetNativeHandle();
        return true;
    }

    BOOL CoreNVSPageManager::EnsureViewTableCapacity(UINT32 minEntries, Camera* camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }
        if (!EnsureSlotViewTableCapacity(*slot, minEntries))
        {
            return false;
        }
        if (camera != nullptr && m_defaultProjectionSlot.allocated)
        {
            return EnsureSlotViewTableCapacity(m_defaultProjectionSlot, minEntries);
        }
        return true;
    }

    BOOL CoreNVSPageManager::UploadClipmapHierarchy(
        const NVSClipmapHierarchyState& hierarchy,
        Camera*                         camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }

        const UINT32 baseSlot = hierarchy.lightId * kNVSMaxClipmapLevels;
        if (!m_config.enableSparseHierarchy && hierarchy.lightId >= kNVSMaxClipmapLights)
        {
            m_lastError = "CoreNVSPageManager::UploadClipmapHierarchy: lightId exceeds kNVSMaxClipmapLights (4) in legacy mode.";
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }
        if (!EnsureSlotViewTableCapacity(*slot, baseSlot + kNVSMaxClipmapLevels))
        {
            return false;
        }

        BuildNVSClipmapHierarchyGPUDataAtSlot(
            hierarchy,
            slot->cpuClipmapTable.data(),
            static_cast<UINT32>(slot->cpuClipmapTable.size()),
            baseSlot);
        slot->activeViewTableCount = std::max(slot->activeViewTableCount, baseSlot + kNVSMaxClipmapLevels);
        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            slot->committedDirtyMin[s] = std::min(slot->committedDirtyMin[s], baseSlot);
            slot->committedDirtyMax[s] = std::max(slot->committedDirtyMax[s], baseSlot + kNVSMaxClipmapLevels);
        }

        if (camera != nullptr && m_defaultProjectionSlot.allocated && EnsureSlotViewTableCapacity(m_defaultProjectionSlot, baseSlot + kNVSMaxClipmapLevels))
        {
            BuildNVSClipmapHierarchyGPUDataAtSlot(
                hierarchy,
                m_defaultProjectionSlot.cpuClipmapTable.data(),
                static_cast<UINT32>(m_defaultProjectionSlot.cpuClipmapTable.size()),
                baseSlot);
            m_defaultProjectionSlot.activeViewTableCount = std::max(m_defaultProjectionSlot.activeViewTableCount, baseSlot + kNVSMaxClipmapLevels);
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                m_defaultProjectionSlot.committedDirtyMin[s] = std::min(m_defaultProjectionSlot.committedDirtyMin[s], baseSlot);
                m_defaultProjectionSlot.committedDirtyMax[s] = std::max(m_defaultProjectionSlot.committedDirtyMax[s], baseSlot + kNVSMaxClipmapLevels);
                m_bufferManager->UpdateGPUBuffer(
                    m_defaultProjectionSlot.clipmapLevelTableBuffers[s],
                    m_defaultProjectionSlot.cpuClipmapTable.data() + baseSlot,
                    kNVSMaxClipmapLevels * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    baseSlot * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)));
            }
        }

        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    slot->clipmapLevelTableBuffers[s],
                    slot->cpuClipmapTable.data() + baseSlot,
                    kNVSMaxClipmapLevels * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    baseSlot * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData))))
            {
                return false;
            }
        }
        return true;
    }

    BOOL CoreNVSPageManager::UploadClipmapLevelTable(
        const NVSClipmapLevelGPUData* entries,
        UINT32                        entryCount,
        Camera*                       camera)
    {
        if (!m_initialized || !m_bufferManager || !entries || entryCount == 0u)
        {
            return false;
        }
        if (!m_config.enableSparseHierarchy && entryCount > kNVSClipmapTableMaxEntries)
        {
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }

        const UINT32 reqCap = std::max(entryCount, kNVSClipmapTableMaxEntries);
        if (!EnsureSlotViewTableCapacity(*slot, reqCap))
        {
            return false;
        }

        const UINT32 prevActive = slot->activeViewTableCount;
        const UINT32 clearEnd   = std::max(entryCount, prevActive);
        std::memcpy(slot->cpuClipmapTable.data(), entries, entryCount * sizeof(NVSClipmapLevelGPUData));
        for (UINT32 i = entryCount; i < clearEnd && i < static_cast<UINT32>(slot->cpuClipmapTable.size()); ++i)
        {
            slot->cpuClipmapTable[i] = NVSClipmapLevelGPUData{};
        }
        slot->activeViewTableCount = entryCount;

        const UINT32 uploadCount = std::min<UINT32>(std::max(clearEnd, kNVSClipmapTableMaxEntries), static_cast<UINT32>(slot->cpuClipmapTable.size()));

        if (camera != nullptr && m_defaultProjectionSlot.allocated && EnsureSlotViewTableCapacity(m_defaultProjectionSlot, reqCap))
        {
            const UINT32 defClearEnd = std::max(entryCount, m_defaultProjectionSlot.activeViewTableCount);
            std::memcpy(
                m_defaultProjectionSlot.cpuClipmapTable.data(),
                slot->cpuClipmapTable.data(),
                entryCount * sizeof(NVSClipmapLevelGPUData));
            for (UINT32 i = entryCount; i < defClearEnd && i < static_cast<UINT32>(m_defaultProjectionSlot.cpuClipmapTable.size()); ++i)
            {
                m_defaultProjectionSlot.cpuClipmapTable[i] = NVSClipmapLevelGPUData{};
            }
            m_defaultProjectionSlot.activeViewTableCount = entryCount;
            const UINT32 defUpload = std::min<UINT32>(std::max(defClearEnd, kNVSClipmapTableMaxEntries), static_cast<UINT32>(m_defaultProjectionSlot.cpuClipmapTable.size()));
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                m_bufferManager->UpdateGPUBuffer(
                    m_defaultProjectionSlot.clipmapLevelTableBuffers[s],
                    m_defaultProjectionSlot.cpuClipmapTable.data(),
                    defUpload * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    0u);
                m_defaultProjectionSlot.dirtyMinViewSlot[s]  = 0xFFFFFFFFu;
                m_defaultProjectionSlot.dirtyMaxViewSlot[s]  = 0u;
                m_defaultProjectionSlot.committedDirtyMin[s] = 0u;
                m_defaultProjectionSlot.committedDirtyMax[s] = defUpload;
            }
        }
        for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    slot->clipmapLevelTableBuffers[s],
                    slot->cpuClipmapTable.data(),
                    uploadCount * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)),
                    0u))
            {
                return false;
            }
            slot->dirtyMinViewSlot[s]  = 0xFFFFFFFFu;
            slot->dirtyMaxViewSlot[s]  = 0u;
            slot->committedDirtyMin[s] = 0u;
            slot->committedDirtyMax[s] = uploadCount;
        }
        return true;
    }

    BOOL CoreNVSPageManager::UpdateClipmapLevelTableRange(
        const NVSClipmapLevelGPUData* entries,
        UINT32                        startSlot,
        UINT32                        entryCount,
        UINT32                        totalActiveCount,
        Camera*                       camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }

        const UINT32 neededCap = std::max(std::max(startSlot + entryCount, totalActiveCount), kNVSClipmapTableMaxEntries);
        if (!EnsureSlotViewTableCapacity(*slot, neededCap))
        {
            return false;
        }

        if (entryCount > 0u && entries != nullptr)
        {
            std::memcpy(slot->cpuClipmapTable.data() + startSlot, entries, entryCount * sizeof(NVSClipmapLevelGPUData));
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                slot->dirtyMinViewSlot[s]  = std::min(slot->dirtyMinViewSlot[s], startSlot);
                slot->dirtyMaxViewSlot[s]  = std::max(slot->dirtyMaxViewSlot[s], startSlot + entryCount);
                slot->committedDirtyMin[s] = std::min(slot->committedDirtyMin[s], startSlot);
                slot->committedDirtyMax[s] = std::max(slot->committedDirtyMax[s], startSlot + entryCount);
            }
        }

        if (totalActiveCount < slot->activeViewTableCount)
        {
            const UINT32 clrEnd = std::min<UINT32>(slot->activeViewTableCount, static_cast<UINT32>(slot->cpuClipmapTable.size()));
            for (UINT32 i = totalActiveCount; i < clrEnd; ++i)
            {
                slot->cpuClipmapTable[i] = NVSClipmapLevelGPUData{};
            }
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                slot->dirtyMinViewSlot[s]  = std::min(slot->dirtyMinViewSlot[s], totalActiveCount);
                slot->dirtyMaxViewSlot[s]  = std::max(slot->dirtyMaxViewSlot[s], clrEnd);
                slot->committedDirtyMin[s] = std::min(slot->committedDirtyMin[s], totalActiveCount);
                slot->committedDirtyMax[s] = std::max(slot->committedDirtyMax[s], clrEnd);
            }
        }
        slot->activeViewTableCount = std::max(totalActiveCount, 1u);

        if (camera != nullptr && m_defaultProjectionSlot.allocated && EnsureSlotViewTableCapacity(m_defaultProjectionSlot, neededCap))
        {
            if (entryCount > 0u && entries != nullptr)
            {
                std::memcpy(m_defaultProjectionSlot.cpuClipmapTable.data() + startSlot, entries, entryCount * sizeof(NVSClipmapLevelGPUData));
                for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
                {
                    m_defaultProjectionSlot.dirtyMinViewSlot[s]  = std::min(m_defaultProjectionSlot.dirtyMinViewSlot[s], startSlot);
                    m_defaultProjectionSlot.dirtyMaxViewSlot[s]  = std::max(m_defaultProjectionSlot.dirtyMaxViewSlot[s], startSlot + entryCount);
                    m_defaultProjectionSlot.committedDirtyMin[s] = std::min(m_defaultProjectionSlot.committedDirtyMin[s], startSlot);
                    m_defaultProjectionSlot.committedDirtyMax[s] = std::max(m_defaultProjectionSlot.committedDirtyMax[s], startSlot + entryCount);
                }
            }
            if (totalActiveCount < m_defaultProjectionSlot.activeViewTableCount)
            {
                const UINT32 clrEnd = std::min<UINT32>(m_defaultProjectionSlot.activeViewTableCount, static_cast<UINT32>(m_defaultProjectionSlot.cpuClipmapTable.size()));
                for (UINT32 i = totalActiveCount; i < clrEnd; ++i)
                {
                    m_defaultProjectionSlot.cpuClipmapTable[i] = NVSClipmapLevelGPUData{};
                }
                for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
                {
                    m_defaultProjectionSlot.dirtyMinViewSlot[s]  = std::min(m_defaultProjectionSlot.dirtyMinViewSlot[s], totalActiveCount);
                    m_defaultProjectionSlot.dirtyMaxViewSlot[s]  = std::max(m_defaultProjectionSlot.dirtyMaxViewSlot[s], clrEnd);
                    m_defaultProjectionSlot.committedDirtyMin[s] = std::min(m_defaultProjectionSlot.committedDirtyMin[s], totalActiveCount);
                    m_defaultProjectionSlot.committedDirtyMax[s] = std::max(m_defaultProjectionSlot.committedDirtyMax[s], clrEnd);
                }
            }
            m_defaultProjectionSlot.activeViewTableCount = std::max(totalActiveCount, 1u);
        }
        return true;
    }

    void CoreNVSPageManager::SetActiveProjectionFrame(UINT32 currentFrame, Camera* camera)
    {
        if (!m_initialized)
        {
            return;
        }
        const UINT32 fSlot = currentFrame % kNVSProjectionFrameSlots;
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (slot && slot->allocated)
        {
            slot->activeSlot = fSlot;
        }
        if (camera != nullptr && m_defaultProjectionSlot.allocated)
        {
            m_defaultProjectionSlot.activeSlot = fSlot;
        }
    }

    BOOL CoreNVSPageManager::FlushViewTableToGPU(Camera* camera)
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }
        const UINT32 fSlot = slot->activeSlot % kNVSProjectionFrameSlots;
        if (!FlushSlotViewTableBuffer(*slot, fSlot))
        {
            return false;
        }
        if (camera != nullptr && m_defaultProjectionSlot.allocated)
        {
            FlushSlotViewTableBuffer(
                m_defaultProjectionSlot,
                m_defaultProjectionSlot.activeSlot % kNVSProjectionFrameSlots);
        }
        return true;
    }

    BOOL CoreNVSPageManager::BuildPageProjectionComputeDispatch(
        NVSProjectionDispatchMode        dispatchMode,
        const NVSPageProjectionPassDesc& passDesc,
        ComputeDispatchCommand&          outCmd,
        Camera*                          camera)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_internalResourceManager || !m_renderPipeline || !m_bufferManager)
        {
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            m_lastError = "CoreNVSPageManager::BuildPageProjectionComputeDispatch: failed to acquire projection slot.";
            return false;
        }

        slot->activeSlot = passDesc.currentFrame % kNVSProjectionFrameSlots;
        const UINT32 fSlot = slot->activeSlot;

        GPUBuffer* cmBuf   = slot->clipmapLevelTableBuffers[fSlot];
        GPUBuffer* prdBuf  = slot->pageRenderDataBuffers[fSlot];
        GPUBuffer* actBuf  = slot->activeRenderPageListBuffers[fSlot];
        GPUBuffer* ptelBuf = slot->projectionTelemetryBuffers[fSlot];

        if (!cmBuf || !prdBuf || !actBuf || !ptelBuf)
        {
            return false;
        }

        if (passDesc.uploadClipmapTable && dispatchMode == NVS_PROJECTION_DISPATCH_RESET)
        {
            if (!FlushSlotViewTableBuffer(*slot, fSlot))
            {
                return false;
            }
        }

        ComputeKernel* kernel = m_internalResourceManager->GetComputeKernel(InternalComputeKernelId::NVS_PAGE_PROJECTION);
        if (!kernel || !kernel->IsValid())
        {
            m_lastError = "CoreNVSPageManager::BuildPageProjectionComputeDispatch: Runtime.NVSPageProjection kernel unavailable.";
            return false;
        }

        const UINT32 totalVirtualPages = GetTotalVirtualPageCount();
        const UINT32 maxLights         = m_config.enableSparseHierarchy
            ? std::max(1u, (slot->activeViewTableCount + 3u) / 4u)
            : std::min(m_config.maxLightCount, kNVSMaxClipmapLights);
        const UINT32 effTableSize      = std::max(slot->activeViewTableCount, kNVSClipmapTableMaxEntries);

        kernel->SetUInt("NVSPageProjectionParams", "Config", "dispatchMode",         static_cast<UINT32>(dispatchMode));
        kernel->SetUInt("NVSPageProjectionParams", "Config", "currentFrame",         passDesc.currentFrame);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "physicalPoolCapacity", m_config.physicalPoolCapacity);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "totalVirtualPages",    totalVirtualPages);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "clipmapTableSize",     effTableSize);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "maxLights",            maxLights);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "clipmapLevels",        m_config.clipmapLevelCount);
        kernel->SetUInt("NVSPageProjectionParams", "Config", "depthCompareMode",     static_cast<UINT32>(passDesc.depthMode));

        auto unbindAll = [&]() {
            kernel->SetGPUBuffer("NVSPageProjectionParams", "InClipmapLevels",         nullptr);
            kernel->SetGPUBuffer("NVSPageProjectionParams", "InVirtualPageTable",      nullptr);
            kernel->SetGPUBuffer("NVSPageProjectionParams", "InPhysicalPageTable",     nullptr);
            kernel->SetGPUBuffer("NVSPageProjectionParams", "OutPageRenderData",       nullptr);
            kernel->SetGPUBuffer("NVSPageProjectionParams", "OutActiveRenderPageList", nullptr);
            kernel->SetGPUBuffer("NVSPageProjectionParams", "OutProjectionTelemetry",  nullptr);
        };

        const GPUResourceAccess telAccess = (dispatchMode == NVS_PROJECTION_DISPATCH_RESET)
            ? GPUResourceAccess::UNORDERED_ACCESS_WRITE
            : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;

        if (!kernel->SetGPUBuffer("NVSPageProjectionParams", "InClipmapLevels",         cmBuf,                     GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSPageProjectionParams", "InVirtualPageTable",      m_virtualPageTableBuffer,  GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSPageProjectionParams", "InPhysicalPageTable",     m_physicalPageTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSPageProjectionParams", "OutPageRenderData",       prdBuf,                    GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSPageProjectionParams", "OutActiveRenderPageList", actBuf,                    GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSPageProjectionParams", "OutProjectionTelemetry",  ptelBuf,                   telAccess))
        {
            unbindAll();
            m_lastError = "CoreNVSPageManager::BuildPageProjectionComputeDispatch: failed to bind GPUBuffers.";
            return false;
        }

        const UINT32 threadCountX = (dispatchMode == NVS_PROJECTION_DISPATCH_RESET)
            ? 64u
            : m_config.physicalPoolCapacity;
        if (!kernel->SetDispatchThreadCount(threadCountX, 1u, 1u))
        {
            unbindAll();
            return false;
        }

        FrameView emptyView{};
        const BOOL built = m_renderPipeline->BuildComputeDispatchCommand(kernel, emptyView, outCmd);
        if (built)
        {
            outCmd.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
        }
        unbindAll();
        return built;
    }

    BOOL CoreNVSPageManager::ExecutePageProjectionPass(
        const NVSPageProjectionPassDesc& passDesc,
        Camera*                          camera)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds;
        cmds.reserve(2u);

        if (passDesc.resetBeforeBuild)
        {
            ComputeDispatchCommand resetCmd{};
            if (!BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, passDesc, resetCmd, camera))
            {
                return false;
            }
            cmds.push_back(resetCmd);
        }

        ComputeDispatchCommand buildCmd{};
        if (!BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, passDesc, buildCmd, camera))
        {
            return false;
        }
        cmds.push_back(buildCmd);

        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL CoreNVSPageManager::ResetTablesOnGPU()
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }

        ResetCommittedShadowSnapshots();

        NVSResidencyPassDesc desc{};
        desc.currentFrame          = 0u;
        desc.requestCount          = 0u;
        desc.executeSelectiveClear = false;

        ComputeDispatchCommand cmd{};
        if (!BuildResidencyComputeDispatch(NVS_DISPATCH_RESET_ALL_TABLES, desc, cmd))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { cmd };

        NVSPageProjectionPassDesc projDesc{};
        projDesc.currentFrame       = 0u;
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;
        ComputeDispatchCommand projResetCmd{};
        ComputeDispatchCommand projBuildCmd{};
        if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, nullptr) &&
            BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, nullptr))
        {
            cmds.push_back(projResetCmd);
            cmds.push_back(projBuildCmd);
        }

        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL CoreNVSPageManager::ExecuteResidencyPass(const NVSResidencyPassDesc& passDesc)
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager)
        {
            return false;
        }

        if (passDesc.invalidationMode == NVS_INVALIDATE_ORIGIN_SHIFT &&
            passDesc.invalidateClipmapLevel < kNVSMaxClipmapLevels &&
            m_defaultProjectionSlot.allocated)
        {
            const UINT32 idx = passDesc.invalidateLightId * kNVSMaxClipmapLevels + passDesc.invalidateClipmapLevel;
            if (idx < static_cast<UINT32>(m_defaultProjectionSlot.cpuClipmapTable.size()))
            {
                NVSClipmapLevelGPUData& cm = m_defaultProjectionSlot.cpuClipmapTable[idx];
                cm.snappedOriginPageX      = passDesc.snappedOriginPageX;
                cm.snappedOriginPageY      = passDesc.snappedOriginPageY;
                cm.lightRightAndCenterX.w  = static_cast<FLOAT32>(passDesc.snappedOriginPageX) * cm.pageWorldSize;
                cm.lightUpAndCenterY.w     = static_cast<FLOAT32>(passDesc.snappedOriginPageY) * cm.pageWorldSize;
                for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
                {
                    m_defaultProjectionSlot.dirtyMinViewSlot[s] = std::min(m_defaultProjectionSlot.dirtyMinViewSlot[s], idx);
                    m_defaultProjectionSlot.dirtyMaxViewSlot[s] = std::max(m_defaultProjectionSlot.dirtyMaxViewSlot[s], idx + 1u);
                }
            }
        }

        std::vector<ComputeDispatchCommand> preDispatches;
        if (!BuildPendingInvalidationDispatches(passDesc.currentFrame, preDispatches, nullptr))
        {
            return false;
        }

        ComputeDispatchCommand allocCmd{};
        if (!BuildResidencyComputeDispatch(NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE, passDesc, allocCmd))
        {
            return false;
        }

        preDispatches.push_back(std::move(allocCmd));

        if (passDesc.executePageProjection)
        {
            NVSPageProjectionPassDesc projDesc{};
            projDesc.currentFrame       = passDesc.currentFrame;
            projDesc.depthMode          = (passDesc.clearDepth < 0.5f)
                ? NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO
                : NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
            projDesc.resetBeforeBuild   = true;
            projDesc.uploadClipmapTable = true;
            ComputeDispatchCommand projResetCmd{};
            ComputeDispatchCommand projBuildCmd{};
            if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, nullptr) &&
                BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, nullptr))
            {
                preDispatches.push_back(projResetCmd);
                preDispatches.push_back(projBuildCmd);
            }
        }

        if (!passDesc.executeSelectiveClear)
        {
            return m_backend->ExecuteComputeDispatches(preDispatches);
        }

        Shader* clearVs = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_CLEAR_VS);
        Shader* passGs  = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_PASS_GS);
        if (!clearVs || clearVs->GetNativeHandle() == 0 || !passGs || passGs->GetNativeHandle() == 0)
        {
            m_lastError = "CoreNVSPageManager::ExecuteResidencyPass: layered depth clear shaders unavailable.";
            return false;
        }

        LayeredDepthPassView clearPass{};
        clearPass.depthArrayTexture                      = m_physicalPoolTexture;
        clearPass.fallbackGeometryShader                 = 0;
        clearPass.selectiveClear.enabled                 = true;
        clearPass.selectiveClear.clearDepth              = passDesc.clearDepth;
        clearPass.selectiveClear.dirtySliceIndicesBuffer = m_dirtyPhysicalSlicesBuffer->GetNativeHandle();
        clearPass.selectiveClear.indirectDrawArgsBuffer  = m_selectiveClearIndirectArgsBuffer->GetNativeHandle();
        clearPass.selectiveClear.maxDirtySlices          = m_config.physicalPoolCapacity;
        clearPass.selectiveClear.clearVertexShader       = clearVs->GetNativeHandle();
        clearPass.selectiveClear.clearGeometryShader     = passGs->GetNativeHandle();

        std::vector<LayeredDepthPassView> passes = { clearPass };
        if (!m_backend->ExecuteLayeredDepthPasses(passes, &preDispatches))
        {
            m_lastError = "CoreNVSPageManager::ExecuteResidencyPass: ExecuteLayeredDepthPasses failed.";
            return false;
        }

        return true;
    }

    BOOL CoreNVSPageManager::MarkClearedPagesOnGPU(UINT32 currentFrame)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }

        NVSResidencyPassDesc desc{};
        desc.currentFrame          = currentFrame;
        desc.requestCount          = 0u;
        desc.executeSelectiveClear = false;

        ComputeDispatchCommand cmd{};
        if (!BuildResidencyComputeDispatch(NVS_DISPATCH_MARK_CLEARED_PAGES, desc, cmd))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { cmd };

        NVSPageProjectionPassDesc projDesc{};
        projDesc.currentFrame       = currentFrame;
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;
        ComputeDispatchCommand projResetCmd{};
        ComputeDispatchCommand projBuildCmd{};
        if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, nullptr) &&
            BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, nullptr))
        {
            cmds.push_back(projResetCmd);
            cmds.push_back(projBuildCmd);
        }

        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL CoreNVSPageManager::PromoteClearedPagesToValidOnGPU(UINT32 currentFrame)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }

        NVSResidencyPassDesc desc{};
        desc.currentFrame          = currentFrame;
        desc.requestCount          = 0u;
        desc.executeSelectiveClear = false;

        ComputeDispatchCommand cmd{};
        if (!BuildResidencyComputeDispatch(NVS_DISPATCH_PROMOTE_CLEARED_TO_VALID, desc, cmd))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { cmd };

        NVSPageProjectionPassDesc projDesc{};
        projDesc.currentFrame       = currentFrame;
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;
        ComputeDispatchCommand projResetCmd{};
        ComputeDispatchCommand projBuildCmd{};
        if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, nullptr) &&
            BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, nullptr))
        {
            cmds.push_back(projResetCmd);
            cmds.push_back(projBuildCmd);
        }

        if (!m_backend->ExecuteComputeDispatches(cmds))
        {
            return false;
        }
        CommitSamplingViewTableSnapshot(currentFrame, nullptr);
        return true;
    }

    BOOL CoreNVSPageManager::CommitRenderedPagesOnGPU(
        const NVSPageRenderCommitRecord* records,
        UINT32                           recordCount,
        UINT32                           currentFrame)
    {
        if (!m_initialized || !m_backend || !m_pageRequestBuffer)
        {
            return false;
        }
        if (recordCount > 0u && !records)
        {
            m_lastError = "CoreNVSPageManager::CommitRenderedPagesOnGPU: null commit record pointer.";
            return false;
        }
        if (recordCount > m_pageRequestBuffer->GetElementCount())
        {
            m_lastError = "CoreNVSPageManager::CommitRenderedPagesOnGPU: recordCount exceeds PageRequests capacity.";
            return false;
        }

        if (recordCount > 0u)
        {
            const UINT32 byteSize = recordCount * static_cast<UINT32>(sizeof(NVSPageRenderCommitRecord));
            if (!m_bufferManager->UpdateGPUBuffer(m_pageRequestBuffer, records, byteSize, 0u))
            {
                m_lastError = "CoreNVSPageManager::CommitRenderedPagesOnGPU: failed to upload commit records.";
                return false;
            }
        }

        NVSResidencyPassDesc desc{};
        desc.currentFrame          = currentFrame;
        desc.requestCount          = recordCount;
        desc.invalidationMode      = NVS_INVALIDATE_NONE;
        desc.executeSelectiveClear = false;

        ComputeDispatchCommand cmd{};
        if (!BuildResidencyComputeDispatch(NVS_DISPATCH_COMMIT_RENDERED_PAGES, desc, cmd))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { cmd };

        NVSPageProjectionPassDesc projDesc{};
        projDesc.currentFrame       = currentFrame;
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;
        ComputeDispatchCommand projResetCmd{};
        ComputeDispatchCommand projBuildCmd{};
        if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, nullptr) &&
            BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, nullptr))
        {
            cmds.push_back(projResetCmd);
            cmds.push_back(projBuildCmd);
        }

        if (!m_backend->ExecuteComputeDispatches(cmds))
        {
            return false;
        }
        CommitSamplingViewTableSnapshot(currentFrame, nullptr);
        return true;
    }

    BOOL CoreNVSPageManager::ClearSpecificPhysicalSlicesOnGPU(
        const UINT32* sliceIndices,
        UINT32        sliceCount,
        FLOAT32       depthValue)
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager || !m_seedCandidateBuffer)
        {
            return false;
        }
        if (sliceCount == 0u)
        {
            return true;
        }
        if (!sliceIndices || sliceCount > m_config.physicalPoolCapacity)
        {
            return false;
        }

        if (!m_bufferManager->UpdateGPUBuffer(
                m_seedCandidateBuffer,
                sliceIndices,
                sliceCount * static_cast<UINT32>(sizeof(UINT32)),
                0u))
        {
            return false;
        }

        ComputeKernel* buildKernel = m_internalResourceManager->GetComputeKernel(InternalComputeKernelId::DIRTY_DEPTH_SLICE_BUILD);
        Shader* clearVs            = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_CLEAR_VS);
        Shader* passGs             = m_internalResourceManager->GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_PASS_GS);
        if (!buildKernel || !buildKernel->IsValid() || !clearVs || !passGs)
        {
            return false;
        }

        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "candidateCount",       sliceCount);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "maxArrayLayers",       m_config.physicalPoolCapacity);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "maxOutputSlices",      m_config.physicalPoolCapacity);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "inputMode",            1u);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "outputBufferCapacity", m_config.physicalPoolCapacity);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "statusBufferCapacity", 4u);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "overflowPolicy",       0u);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "clearTailSentinel",    1u);

        auto unbindBuild = [&]() {
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "CandidateSlices",      nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtySliceIndices", nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtyCount",        nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutIndirectDrawArgs",  nullptr);
        };

        if (!buildKernel->SetGPUBuffer("DirtySliceBuildParams", "CandidateSlices",      m_seedCandidateBuffer,              GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtySliceIndices", m_dirtyPhysicalSlicesBuffer,        GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtyCount",        m_dirtySliceCountBuffer,            GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutIndirectDrawArgs",  m_selectiveClearIndirectArgsBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !buildKernel->SetDispatchThreadCount(64u, 1u, 1u))
        {
            unbindBuild();
            return false;
        }

        FrameView emptyView{};
        ComputeDispatchCommand buildCmd{};
        const BOOL built = m_renderPipeline->BuildComputeDispatchCommand(buildKernel, emptyView, buildCmd);
        if (built)
        {
            buildCmd.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
        }
        unbindBuild();
        if (!built)
        {
            return false;
        }

        LayeredDepthPassView clearPass{};
        clearPass.depthArrayTexture                      = m_physicalPoolTexture;
        clearPass.fallbackGeometryShader                 = 0;
        clearPass.selectiveClear.enabled                 = true;
        clearPass.selectiveClear.clearDepth              = depthValue;
        clearPass.selectiveClear.dirtySliceIndicesBuffer = m_dirtyPhysicalSlicesBuffer->GetNativeHandle();
        clearPass.selectiveClear.indirectDrawArgsBuffer  = m_selectiveClearIndirectArgsBuffer->GetNativeHandle();
        clearPass.selectiveClear.maxDirtySlices          = m_config.physicalPoolCapacity;
        clearPass.selectiveClear.clearVertexShader       = clearVs->GetNativeHandle();
        clearPass.selectiveClear.clearGeometryShader     = passGs->GetNativeHandle();

        std::vector<ComputeDispatchCommand> pre = { buildCmd };
        std::vector<LayeredDepthPassView> passes = { clearPass };
        return m_backend->ExecuteLayeredDepthPasses(passes, &pre);
    }

    BOOL CoreNVSPageManager::ReadbackTelemetrySync(NVSResidencyTelemetry& outTelemetry) const
    {
        if (!m_initialized || !m_bufferManager || !m_telemetryBuffer)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            m_telemetryBuffer,
            &outTelemetry,
            static_cast<UINT32>(sizeof(NVSResidencyTelemetry)));
    }

    BOOL CoreNVSPageManager::ReadbackVirtualPageTableSync(std::vector<NVSVirtualPageEntry>& outEntries) const
    {
        if (!m_initialized || !m_bufferManager || !m_virtualPageTableBuffer)
        {
            return false;
        }

        if (!m_config.enableSparseHierarchy)
        {
            const UINT32 count = GetTotalVirtualPageCount();
            outEntries.resize(count);
            return m_bufferManager->ReadbackGPUBufferSync(
                m_virtualPageTableBuffer,
                outEntries.data(),
                count * static_cast<UINT32>(sizeof(NVSVirtualPageEntry)));
        }

        // In sparse hierarchy mode, reconstruct the logical Mip-0 flat window [0 .. maxLightCount * clipmapLevelCount * pagesPerAxis^2 - 1]
        // from SparsePageHierarchy + VirtualPageTable so C-export test queries inspect logical virtual pages transparently.
        const UINT32 logicalViews      = std::min(std::max(m_config.maxLightCount * m_config.clipmapLevelCount, kNVSClipmapTableMaxEntries), 64u);
        const UINT32 logicalAxis       = std::max(m_config.pagesPerAxis, 1u);
        const UINT32 pagesPerView      = logicalAxis * logicalAxis;
        const UINT32 logicalTotalPages = logicalViews * pagesPerView;

        outEntries.assign(logicalTotalPages, NVSVirtualPageEntry{});
        for (UINT32 v = 0u; v < logicalViews; ++v)
        {
            const UINT32 lightId = v / m_config.clipmapLevelCount;
            const UINT32 lvl     = v % m_config.clipmapLevelCount;
            for (UINT32 py = 0u; py < logicalAxis; ++py)
            {
                for (UINT32 px = 0u; px < logicalAxis; ++px)
                {
                    NVSVirtualPageEntry& dst = outEntries[v * pagesPerView + py * logicalAxis + px];
                    dst.stateAndFlags        = 0u;
                    dst.physicalPageIndex    = kNVSInvalidPhysicalPage;
                    dst.virtualAddressPacked = PackNVSVirtualPageAddress(lightId, lvl, px, py, 1u);
                    dst.lastTouchedFrame     = 0u;
                }
            }
        }

        if (!m_sparsePageHierarchyBuffer)
        {
            return true;
        }

        const UINT32 treeWords = m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride);
        std::vector<UINT32> treeData(treeWords, kNVSSparseUnmappedIndex);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_sparsePageHierarchyBuffer,
                treeData.data(),
                treeWords * static_cast<UINT32>(sizeof(UINT32))))
        {
            return false;
        }

        const UINT32 leafCount = m_sparseBlockCapacity * kNVSSparsePagesPerBlock;
        std::vector<NVSVirtualPageEntry> leafPool(leafCount);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_virtualPageTableBuffer,
                leafPool.data(),
                leafCount * static_cast<UINT32>(sizeof(NVSVirtualPageEntry))))
        {
            return false;
        }

        const UINT32 maxScanViews = std::min(logicalViews, m_sparseRootCapacity);
        for (UINT32 v = 0u; v < maxScanViews; ++v)
        {
            const UINT32 dirSlot = treeData[v];
            if (dirSlot >= m_sparseDirCapacity)
            {
                continue;
            }
            const UINT32 lightId = v / m_config.clipmapLevelCount;
            const UINT32 lvl     = v % m_config.clipmapLevelCount;
            for (UINT32 py = 0u; py < logicalAxis; ++py)
            {
                for (UINT32 px = 0u; px < logicalAxis; ++px)
                {
                    UINT32 tileOff = 0u, localIdx = 0u;
                    if (!ComputeNVSSparseDirectoryOffset(0u, px, py, tileOff, localIdx))
                    {
                        continue;
                    }
                    const UINT32 blkSlot = treeData[m_sparseRootCapacity + (dirSlot * kNVSSparseDirStride) + tileOff];
                    if (blkSlot >= m_sparseBlockCapacity)
                    {
                        continue;
                    }
                    const UINT32 vFlat = (blkSlot << 6u) + localIdx;
                    if (vFlat >= leafCount)
                    {
                        continue;
                    }
                    NVSVirtualPageEntry mapped = leafPool[vFlat];
                    if ((mapped.stateAndFlags & NVS_PAGE_FLAG_EXTENDED_VIEW_ADDR) != 0u)
                    {
                        const NVSSparsePageAddress sp = UnpackNVSSparsePageCoord(v, mapped.virtualAddressPacked);
                        mapped.virtualAddressPacked   = PackNVSVirtualPageAddress(lightId, lvl, sp.virtualPageX, sp.virtualPageY, sp.generation & 0xFFu);
                    }
                    outEntries[v * pagesPerView + py * logicalAxis + px] = mapped;
                }
            }
        }
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackPhysicalPageTableSync(std::vector<NVSPhysicalPageEntry>& outEntries) const
    {
        if (!m_initialized || !m_bufferManager || !m_physicalPageTableBuffer)
        {
            return false;
        }
        const UINT32 count = m_config.physicalPoolCapacity;
        outEntries.resize(count);
        return m_bufferManager->ReadbackGPUBufferSync(
            m_physicalPageTableBuffer,
            outEntries.data(),
            count * static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)));
    }

    BOOL CoreNVSPageManager::ReadbackDirtySlicesSync(std::vector<UINT32>& outSlices, UINT32 outCount) const
    {
        if (!m_initialized || !m_bufferManager || !m_dirtyPhysicalSlicesBuffer)
        {
            return false;
        }
        const UINT32 cap = m_config.physicalPoolCapacity;
        const UINT32 cnt = (outCount > 0u && outCount <= cap) ? outCount : cap;
        std::vector<UINT32> fullBuf(cap, kNVSInvalidPhysicalPage);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_dirtyPhysicalSlicesBuffer,
                fullBuf.data(),
                cap * static_cast<UINT32>(sizeof(UINT32))))
        {
            return false;
        }
        outSlices.assign(fullBuf.begin(), fullBuf.begin() + cnt);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackDirtySliceStatusSync(UINT32 outStatus[4]) const
    {
        if (!m_initialized || !m_bufferManager || !m_dirtySliceCountBuffer || !outStatus)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            m_dirtySliceCountBuffer,
            outStatus,
            4u * static_cast<UINT32>(sizeof(UINT32)));
    }

    BOOL CoreNVSPageManager::ReadbackSelectiveClearIndirectArgsSync(UINT32 outArgs[4]) const
    {
        if (!m_initialized || !m_bufferManager || !m_selectiveClearIndirectArgsBuffer || !outArgs)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            m_selectiveClearIndirectArgsBuffer,
            outArgs,
            4u * static_cast<UINT32>(sizeof(UINT32)));
    }

    BOOL CoreNVSPageManager::ReadbackPageRenderDataSync(
        std::vector<NVSPageRenderData>& outEntries,
        Camera*                         camera) const
    {
        GPUBuffer* buf = GetPageRenderDataBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 count = m_config.physicalPoolCapacity;
        outEntries.resize(count);
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            outEntries.data(),
            count * static_cast<UINT32>(sizeof(NVSPageRenderData)));
    }

    BOOL CoreNVSPageManager::ReadbackActiveRenderPageListSync(
        std::vector<UINT32>& outPhysicalIndices,
        UINT32               outCount,
        Camera*              camera) const
    {
        GPUBuffer* buf = GetActiveRenderPageListBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 cap = m_config.physicalPoolCapacity;
        const UINT32 cnt = (outCount > 0u && outCount <= cap) ? outCount : cap;
        std::vector<UINT32> fullBuf(cap, kNVSInvalidPhysicalPage);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                buf,
                fullBuf.data(),
                cap * static_cast<UINT32>(sizeof(UINT32))))
        {
            return false;
        }
        outPhysicalIndices.assign(fullBuf.begin(), fullBuf.begin() + cnt);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackProjectionTelemetrySync(
        NVSPageProjectionTelemetry& outTelemetry,
        Camera*                     camera) const
    {
        GPUBuffer* buf = GetProjectionTelemetryBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            &outTelemetry,
            static_cast<UINT32>(sizeof(NVSPageProjectionTelemetry)));
    }

    BOOL CoreNVSPageManager::ReadbackClipmapLevelTableSync(
        std::vector<NVSClipmapLevelGPUData>& outTable,
        Camera*                              camera) const
    {
        GPUBuffer* buf = GetClipmapLevelTableBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const CameraProjectionSlot* slot = FindCameraProjectionSlot(camera);
        const UINT32 count = std::min<UINT32>(
            slot ? std::max(slot->activeViewTableCount, kNVSClipmapTableMaxEntries) : kNVSClipmapTableMaxEntries,
            buf->GetElementCount());
        outTable.resize(count);
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            outTable.data(),
            count * static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)));
    }

    BOOL CoreNVSPageManager::UploadDynamicCasterInvalidationBoxes(
        const NVSDynamicCasterInvalidationBox* boxes,
        UINT32                                 boxCount)
    {
        if (!m_initialized || !m_bufferManager || !m_dynamicInvalidationBoxesBuffer)
        {
            return false;
        }
        if (boxCount == 0u)
        {
            return true;
        }
        if (!boxes || boxCount > (kMaxGPUSceneSlotCapacity * 2u))
        {
            m_lastError = "CoreNVSPageManager::UploadDynamicCasterInvalidationBoxes: invalid box pointer or count exceeds maximum scene bounds.";
            return false;
        }
        const UINT32 curCap = m_dynamicInvalidationBoxesBuffer->GetElementCount();
        if (boxCount > curCap)
        {
            UINT32 newCap = (curCap > 0u) ? curCap : kNVSMaxDynamicInvalidationBoxes;
            while (newCap < boxCount)
            {
                if (newCap > ((kMaxGPUSceneSlotCapacity * 2u) / 2u))
                {
                    newCap = kMaxGPUSceneSlotCapacity * 2u;
                    break;
                }
                newCap *= 2u;
            }
            if (!m_bufferManager->ResizeGPUBuffer(m_dynamicInvalidationBoxesBuffer, newCap, nullptr))
            {
                m_lastError = "CoreNVSPageManager::UploadDynamicCasterInvalidationBoxes: failed to grow DynamicInvalidationBoxes GPUBuffer.";
                return false;
            }
        }
        return m_bufferManager->UpdateGPUBuffer(
            m_dynamicInvalidationBoxesBuffer,
            boxes,
            boxCount * static_cast<UINT32>(sizeof(NVSDynamicCasterInvalidationBox)),
            0u);
    }

    BOOL CoreNVSPageManager::BuildCasterCullComputeDispatch(
        NVSCasterCullDispatchMode    dispatchMode,
        GPUBuffer*                   gpuSceneBuffer,
        UINT32                       sceneSlotCount,
        const NVSCasterCullPassDesc& passDesc,
        ComputeDispatchCommand&      outCmd,
        Camera*                      camera,
        UINT32                       dynamicInvalidationBoxCount,
        GPUBuffer*                   overrideClipmapTableBuffer)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_internalResourceManager || !m_renderPipeline)
        {
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }

        if (sceneSlotCount > m_filteredCasterCapacity)
        {
            if (!EnsureFilteredCasterCapacity(sceneSlotCount))
            {
                return false;
            }
        }

        const UINT32 fSlot  = slot->activeSlot % kNVSProjectionFrameSlots;
        GPUBuffer*   cmBuf  = overrideClipmapTableBuffer ? overrideClipmapTableBuffer : slot->clipmapLevelTableBuffers[fSlot];
        GPUBuffer*   prdBuf = slot->pageRenderDataBuffers[fSlot];
        GPUBuffer*   actBuf = slot->activeRenderPageListBuffers[fSlot];
        GPUBuffer*   cprBuf = slot->casterPageRangeBuffers[fSlot];
        GPUBuffer*   wlBuf  = slot->casterWorklistBuffers[fSlot];
        GPUBuffer*   telBuf = slot->casterCullTelemetryBuffers[fSlot];
        GPUBuffer*   indBuf = slot->worklistIndirectArgsBuffers[fSlot];

        GPUBuffer* effSceneBuf = (gpuSceneBuffer && gpuSceneBuffer->IsValid())
            ? gpuSceneBuffer
            : m_dummySceneObjectBuffer;
        const UINT32 effSceneSlots = (gpuSceneBuffer && gpuSceneBuffer->IsValid())
            ? std::min(sceneSlotCount, gpuSceneBuffer->GetElementCount())
            : 0u;

        if (!effSceneBuf || !cmBuf || !prdBuf || !actBuf || !cprBuf || !wlBuf || !telBuf || !indBuf ||
            !m_filteredCasterBuffer || !m_casterValidBitmaskBuffer || !m_dynamicInvalidationBoxesBuffer ||
            !m_virtualPageTableBuffer || !m_physicalPageTableBuffer)
        {
            m_lastError = "CoreNVSPageManager::BuildCasterCullComputeDispatch: required GPUBuffer is null.";
            return false;
        }

        ComputeKernel* kernel = m_internalResourceManager->GetComputeKernel(InternalComputeKernelId::NVS_CASTER_CULL);
        if (!kernel || !kernel->IsValid())
        {
            m_lastError = "CoreNVSPageManager::BuildCasterCullComputeDispatch: Runtime.NVSCasterCull kernel unavailable.";
            return false;
        }

        const UINT32 totalVirtualPages = GetTotalVirtualPageCount();
        const UINT32 effTableSize      = overrideClipmapTableBuffer
            ? overrideClipmapTableBuffer->GetElementCount()
            : std::max(slot->activeViewTableCount, kNVSClipmapTableMaxEntries);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "dispatchMode",                  static_cast<UINT32>(dispatchMode));
        kernel->SetUInt("NVSCasterCullParams",  "Config", "currentFrame",                  passDesc.currentFrame);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "sceneSlotCount",                effSceneSlots);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "activePageInputCount",          m_config.physicalPoolCapacity);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "physicalPoolCapacity",          m_config.physicalPoolCapacity);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "totalVirtualPages",             totalVirtualPages);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "clipmapTableSize",              effTableSize);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "maxWorklistCapacity",           slot->worklistCapacity);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "staticBatchPolicy",             passDesc.staticBatchPolicy);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "skipTransparentCasters",        passDesc.skipTransparentCasters ? 1u : 0u);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "disablePrepassFilter",          passDesc.disablePrepassFilter ? 1u : 0u);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "skipAlreadyRenderedValidPages", passDesc.skipAlreadyRenderedValidPages ? 1u : 0u);
        kernel->SetFloat("NVSCasterCullParams", "Config", "minCasterProjectedTexels",      passDesc.minCasterProjectedTexels);
        kernel->SetFloat("NVSCasterCullParams", "Config", "lodThresholdScale",             passDesc.lodThresholdScale);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "forceLodOverride",              passDesc.forceLodOverride);
        kernel->SetUInt("NVSCasterCullParams",  "Config", "dynamicInvalidationBoxCount",   dynamicInvalidationBoxCount);

        auto unbindAll = [&]() {
            kernel->SetGPUBuffer("NVSCasterCullParams", "InGPUSceneObjects",          nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InClipmapLevels",            nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InPageRenderData",           nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InActiveRenderPageList",     nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InDynamicInvalidationBoxes", nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutFilteredCasters",         nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterValidBitmask",      nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterPageRanges",        nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterWorklist",          nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutWorklistIndirectArgs",    nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterTelemetry",         nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InOutVirtualPageTable",      nullptr);
            kernel->SetGPUBuffer("NVSCasterCullParams", "InOutPhysicalPageTable",     nullptr);
        };

        if (!kernel->SetGPUBuffer("NVSCasterCullParams", "InGPUSceneObjects",          effSceneBuf,                      GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InClipmapLevels",            cmBuf,                            GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InPageRenderData",           prdBuf,                           GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InActiveRenderPageList",     actBuf,                           GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InDynamicInvalidationBoxes", m_dynamicInvalidationBoxesBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutFilteredCasters",         m_filteredCasterBuffer,           GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterValidBitmask",      m_casterValidBitmaskBuffer,       GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterPageRanges",        cprBuf,                           GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterWorklist",          wlBuf,                            GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutWorklistIndirectArgs",    indBuf,                           GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "OutCasterTelemetry",         telBuf,                           GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InOutVirtualPageTable",      m_virtualPageTableBuffer,         GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSCasterCullParams", "InOutPhysicalPageTable",     m_physicalPageTableBuffer,        GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE))
        {
            unbindAll();
            m_lastError = "CoreNVSPageManager::BuildCasterCullComputeDispatch: failed to bind GPUBuffers.";
            return false;
        }

        UINT32 totalGroups = 1u;
        switch (dispatchMode)
        {
            case NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES:
            case NVS_CASTER_CULL_MODE_PREFIX_SUM_RANGES:
                totalGroups = 1u;
                break;
            case NVS_CASTER_CULL_MODE_FILTER_SCENE_CASTERS:
                totalGroups = std::max((effSceneSlots + 63u) / 64u, 1u);
                break;
            case NVS_CASTER_CULL_MODE_COUNT_PAGE_CASTERS:
            case NVS_CASTER_CULL_MODE_SCATTER_WORKLIST:
                totalGroups = std::max(m_config.physicalPoolCapacity, 1u); // 1 workgroup (64 threads) per active page slot
                break;
            case NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION:
                totalGroups = std::max((m_config.physicalPoolCapacity + 63u) / 64u, 1u);
                break;
            case NVS_CASTER_CULL_MODE_VERIFY_MATRIX_CLIP:
                totalGroups = std::max((dynamicInvalidationBoxCount + 63u) / 64u, 1u);
                break;
            default:
                totalGroups = 1u;
                break;
        }

        UINT32 groupsX = 1u, groupsY = 1u;
        ComputeDispatchGroups2D(totalGroups, groupsX, groupsY);
        kernel->SetThreadGroupCount(groupsX, groupsY, 1u);

        FrameView emptyView{};
        const BOOL built = m_renderPipeline->BuildComputeDispatchCommand(kernel, emptyView, outCmd);
        if (built)
        {
            outCmd.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
        }
        unbindAll();
        return built;
    }

    BOOL CoreNVSPageManager::BuildCasterCullPipelineDispatches(
        GPUBuffer*                             gpuSceneBuffer,
        UINT32                                 sceneSlotCount,
        const NVSCasterCullPassDesc&           passDesc,
        std::vector<ComputeDispatchCommand>&   outDispatches,
        Camera*                                camera,
        const NVSDynamicCasterInvalidationBox* dynamicBoxes,
        UINT32                                 dynamicBoxCount)
    {
        if (!m_initialized)
        {
            return false;
        }

        NVSCasterCullPassDesc effDesc = passDesc;
        if (dynamicBoxes != nullptr && dynamicBoxCount > 0u)
        {
            if (!UploadDynamicCasterInvalidationBoxes(dynamicBoxes, dynamicBoxCount))
            {
                return false;
            }

            // Reset dynamicInvalidatedPageCount before Mode 5 by running a quick Mode 0 reset first, then Mode 5,
            // then rebuild PageRenderData so invalidated pages are excluded from active shadow pages!
            ComputeDispatchCommand preResetCmd{};
            if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES, gpuSceneBuffer, sceneSlotCount, effDesc, preResetCmd, camera, 0u))
            {
                return false;
            }
            outDispatches.push_back(preResetCmd);

            ComputeDispatchCommand dynInvCmd{};
            if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION, gpuSceneBuffer, sceneSlotCount, effDesc, dynInvCmd, camera, dynamicBoxCount))
            {
                return false;
            }
            outDispatches.push_back(dynInvCmd);

            NVSPageProjectionPassDesc projDesc{};
            if (CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera))
            {
                projDesc.currentFrame = slot->activeSlot;
                const UINT32 vLim = std::min<UINT32>(slot->activeViewTableCount, static_cast<UINT32>(slot->cpuClipmapTable.size()));
                for (UINT32 vi = 0u; vi < vLim; ++vi)
                {
                    if ((slot->cpuClipmapTable[vi].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u &&
                        (slot->cpuClipmapTable[vi].flags & NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z) != 0u)
                    {
                        projDesc.depthMode = NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO;
                        break;
                    }
                }
            }
            projDesc.resetBeforeBuild   = true;
            projDesc.uploadClipmapTable = true;
            ComputeDispatchCommand projResetCmd{};
            ComputeDispatchCommand projBuildCmd{};
            if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, camera) &&
                BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, camera))
            {
                outDispatches.push_back(projResetCmd);
                outDispatches.push_back(projBuildCmd);
            }

            // Preserve dynamicInvalidatedPageCount during subsequent Mode 0
            effDesc.forceLodOverride |= 0x80000000u;
        }

        ComputeDispatchCommand mode0Cmd{};
        if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES, gpuSceneBuffer, sceneSlotCount, effDesc, mode0Cmd, camera, 0u))
        {
            return false;
        }
        outDispatches.push_back(mode0Cmd);

        const UINT32 effSceneSlots = (gpuSceneBuffer && gpuSceneBuffer->IsValid())
            ? std::min(sceneSlotCount, gpuSceneBuffer->GetElementCount())
            : 0u;

        if (effSceneSlots > 0u && (effDesc.forceLodOverride & 0x10000000u) == 0u)
        {
            ComputeDispatchCommand mode1Cmd{};
            if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_FILTER_SCENE_CASTERS, gpuSceneBuffer, effSceneSlots, effDesc, mode1Cmd, camera, 0u))
            {
                return false;
            }
            outDispatches.push_back(mode1Cmd);
        }

        ComputeDispatchCommand mode2Cmd{};
        if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_COUNT_PAGE_CASTERS, gpuSceneBuffer, effSceneSlots, effDesc, mode2Cmd, camera, 0u))
        {
            return false;
        }
        outDispatches.push_back(mode2Cmd);

        ComputeDispatchCommand mode3Cmd{};
        if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_PREFIX_SUM_RANGES, gpuSceneBuffer, effSceneSlots, effDesc, mode3Cmd, camera, 0u))
        {
            return false;
        }
        outDispatches.push_back(mode3Cmd);

        if (effSceneSlots > 0u)
        {
            ComputeDispatchCommand mode4Cmd{};
            if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_SCATTER_WORKLIST, gpuSceneBuffer, effSceneSlots, effDesc, mode4Cmd, camera, 0u))
            {
                return false;
            }
            outDispatches.push_back(mode4Cmd);
        }

        return true;
    }

    BOOL CoreNVSPageManager::ExecuteCasterCullPass(
        GPUBuffer*                             gpuSceneBuffer,
        UINT32                                 sceneSlotCount,
        const NVSCasterCullPassDesc&           passDesc,
        Camera*                                camera,
        const NVSDynamicCasterInvalidationBox* dynamicBoxes,
        UINT32                                 dynamicBoxCount)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds;
        cmds.reserve(8u);
        if (!BuildCasterCullPipelineDispatches(
                gpuSceneBuffer,
                sceneSlotCount,
                passDesc,
                cmds,
                camera,
                dynamicBoxes,
                dynamicBoxCount))
        {
            return false;
        }

        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL CoreNVSPageManager::ExecuteDynamicCasterInvalidationPass(
        const NVSDynamicCasterInvalidationBox* boxes,
        UINT32                                 boxCount,
        Camera*                                camera,
        BOOL                                   invalidateAllResidentStates)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }
        if (boxCount == 0u)
        {
            return true;
        }
        if (!UploadDynamicCasterInvalidationBoxes(boxes, boxCount))
        {
            return false;
        }

        NVSCasterCullPassDesc desc{};
        if (invalidateAllResidentStates)
        {
            desc.forceLodOverride |= 0x20000000u; // bit 29 = invalidate any resident state 1..3
        }

        ComputeDispatchCommand resetCmd{};
        ComputeDispatchCommand invCmd{};
        if (!BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES, nullptr, 0u, desc, resetCmd, camera, 0u) ||
            !BuildCasterCullComputeDispatch(NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION, nullptr, 0u, desc, invCmd, camera, boxCount))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { resetCmd, invCmd };

        NVSPageProjectionPassDesc projDesc{};
        if (CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera))
        {
            projDesc.currentFrame = slot->activeSlot;
            const UINT32 vLim = std::min<UINT32>(slot->activeViewTableCount, static_cast<UINT32>(slot->cpuClipmapTable.size()));
            for (UINT32 vi = 0u; vi < vLim; ++vi)
            {
                if ((slot->cpuClipmapTable[vi].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u &&
                    (slot->cpuClipmapTable[vi].flags & NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z) != 0u)
                {
                    projDesc.depthMode = NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO;
                    break;
                }
            }
        }
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;
        ComputeDispatchCommand projResetCmd{};
        ComputeDispatchCommand projBuildCmd{};
        if (BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_RESET, projDesc, projResetCmd, camera) &&
            BuildPageProjectionComputeDispatch(NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS, projDesc, projBuildCmd, camera))
        {
            cmds.push_back(projResetCmd);
            cmds.push_back(projBuildCmd);
        }

        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL CoreNVSPageManager::VerifyMatrixClipTransformOnGPU(
        const Vec3*                           worldPoints,
        const UINT32*                         physicalPageIndices,
        UINT32                                pointCount,
        GPUBuffer*                            gpuSceneBuffer,
        std::vector<NVSFilteredCasterObject>& outResults,
        Camera*                               camera)
    {
        outResults.clear();
        if (!m_initialized || !m_backend || !worldPoints || pointCount == 0u || pointCount > kNVSMaxDynamicInvalidationBoxes)
        {
            return false;
        }
        if (!EnsureFilteredCasterCapacity(pointCount))
        {
            return false;
        }

        std::vector<NVSDynamicCasterInvalidationBox> packedPoints(pointCount);
        for (UINT32 i = 0u; i < pointCount; ++i)
        {
            packedPoints[i].worldMin          = worldPoints[i];
            packedPoints[i].gpuSceneSlotIndex = physicalPageIndices ? physicalPageIndices[i] : 0u;
            packedPoints[i].worldMax          = worldPoints[i];
            packedPoints[i].invalidationFlags = 0u;
        }

        if (!UploadDynamicCasterInvalidationBoxes(packedPoints.data(), pointCount))
        {
            return false;
        }

        NVSCasterCullPassDesc desc{};
        ComputeDispatchCommand verifyCmd{};
        if (!BuildCasterCullComputeDispatch(
                NVS_CASTER_CULL_MODE_VERIFY_MATRIX_CLIP,
                gpuSceneBuffer,
                pointCount,
                desc,
                verifyCmd,
                camera,
                pointCount))
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> cmds = { verifyCmd };
        if (!m_backend->ExecuteComputeDispatches(cmds))
        {
            return false;
        }

        return ReadbackFilteredCastersSync(outResults, pointCount);
    }

    BOOL CoreNVSPageManager::ReadbackCasterCullTelemetrySync(
        NVSCasterCullTelemetry& outTelemetry,
        Camera*                 camera) const
    {
        GPUBuffer* buf = GetCasterCullTelemetryBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            &outTelemetry,
            static_cast<UINT32>(sizeof(NVSCasterCullTelemetry)));
    }

    BOOL CoreNVSPageManager::ReadbackCasterPageRangesSync(
        std::vector<NVSCasterPageRange>& outRanges,
        UINT32                           outCount,
        Camera*                          camera) const
    {
        GPUBuffer* buf = GetCasterPageRangeBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 cap = m_config.physicalPoolCapacity;
        const UINT32 cnt = (outCount > 0u && outCount <= cap) ? outCount : cap;
        std::vector<NVSCasterPageRange> fullBuf(cap);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                buf,
                fullBuf.data(),
                cap * static_cast<UINT32>(sizeof(NVSCasterPageRange))))
        {
            return false;
        }
        outRanges.assign(fullBuf.begin(), fullBuf.begin() + cnt);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackCasterWorklistSync(
        std::vector<NVSCasterDrawCandidate>& outCandidates,
        UINT32                               outCount,
        Camera*                              camera) const
    {
        GPUBuffer* buf = GetCasterWorklistBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 allocCap = buf->GetElementCount();
        if (outCount == 0u)
        {
            NVSCasterCullTelemetry tel{};
            if (ReadbackCasterCullTelemetrySync(tel, camera))
            {
                outCount = tel.totalAssignedCasterDraws;
            }
        }
        if (outCount == 0u)
        {
            outCandidates.clear();
            return true;
        }
        const UINT32 cnt = std::min(outCount, allocCap);
        outCandidates.resize(cnt);
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            outCandidates.data(),
            cnt * static_cast<UINT32>(sizeof(NVSCasterDrawCandidate)));
    }

    BOOL CoreNVSPageManager::ReadbackWorklistIndirectArgsSync(
        UINT32  outArgs[8],
        Camera* camera) const
    {
        GPUBuffer* buf = GetWorklistIndirectArgsBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf || !outArgs)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            outArgs,
            8u * static_cast<UINT32>(sizeof(UINT32)));
    }

    BOOL CoreNVSPageManager::ReadbackFilteredCastersSync(
        std::vector<NVSFilteredCasterObject>& outFiltered,
        UINT32                                outCount) const
    {
        if (!m_initialized || !m_bufferManager || !m_filteredCasterBuffer)
        {
            return false;
        }
        const UINT32 cap = m_filteredCasterBuffer->GetElementCount();
        const UINT32 cnt = (outCount > 0u && outCount <= cap) ? outCount : cap;
        outFiltered.resize(cnt);
        return m_bufferManager->ReadbackGPUBufferSync(
            m_filteredCasterBuffer,
            outFiltered.data(),
            cnt * static_cast<UINT32>(sizeof(NVSFilteredCasterObject)));
    }

    // =====================================================================
    //  GPU Mesh Draw Descriptors, Material Shadow Descriptors,
    //              Indirect Command Generation & Shadow Rasterization
    // =====================================================================

    BOOL CoreNVSPageManager::EnsureMeshDrawDescriptorCapacity(UINT32 minSlots)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer)
        {
            return false;
        }
        const UINT32 oldCap = static_cast<UINT32>(m_cpuMeshDescriptors.size());
        if (minSlots <= oldCap)
        {
            return true;
        }
        if (minSlots > 65535u)
        {
            return false;
        }
        UINT32 newCap = std::max(oldCap, kNVSMaxMeshDrawDescriptors);
        while (newCap < minSlots && newCap < 65535u)
        {
            newCap = std::min<UINT32>(newCap * 2u, 65535u);
        }
        m_cpuMeshDescriptors.resize(newCap, GPUMeshDrawDescriptor{});
        for (UINT32 i = oldCap; i < newCap; ++i)
        {
            m_cpuMeshDescriptors[i].meshId = i;
        }
        if (!m_bufferManager->ResizeGPUBuffer(m_meshDrawDescriptorBuffer, newCap, m_cpuMeshDescriptors.data()))
        {
            m_lastError = "CoreNVSPageManager::EnsureMeshDrawDescriptorCapacity: failed to resize MeshDrawDescriptors buffer.";
            return false;
        }
        return true;
    }

    BOOL CoreNVSPageManager::RegisterOrUpdateMeshDrawDescriptor(const GPUMeshDrawDescriptor& desc)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer)
        {
            return false;
        }
        if (desc.meshId >= static_cast<UINT32>(m_cpuMeshDescriptors.size()))
        {
            if (!EnsureMeshDrawDescriptorCapacity(desc.meshId + 1u))
            {
                m_lastError = "CoreNVSPageManager::RegisterOrUpdateMeshDrawDescriptor: meshId out of bounds.";
                return false;
            }
        }
        if (std::memcmp(&m_cpuMeshDescriptors[desc.meshId], &desc, sizeof(GPUMeshDrawDescriptor)) == 0)
        {
            return true;
        }
        m_cpuMeshDescriptors[desc.meshId] = desc;
        return m_bufferManager->UpdateGPUBuffer(
            m_meshDrawDescriptorBuffer,
            &m_cpuMeshDescriptors[desc.meshId],
            static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)),
            desc.meshId * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)));
    }

    BOOL CoreNVSPageManager::RegisterOrUpdateMeshVariantDescriptor(
        UINT32                       slotIndex,
        const GPUMeshDrawDescriptor& desc)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer)
        {
            return false;
        }
        if (slotIndex >= static_cast<UINT32>(m_cpuMeshDescriptors.size()))
        {
            if (!EnsureMeshDrawDescriptorCapacity(slotIndex + 1u))
            {
                m_lastError = "CoreNVSPageManager::RegisterOrUpdateMeshVariantDescriptor: slotIndex out of bounds.";
                return false;
            }
        }
        if (std::memcmp(&m_cpuMeshDescriptors[slotIndex], &desc, sizeof(GPUMeshDrawDescriptor)) == 0)
        {
            return true;
        }
        m_cpuMeshDescriptors[slotIndex] = desc;
        return m_bufferManager->UpdateGPUBuffer(
            m_meshDrawDescriptorBuffer,
            &m_cpuMeshDescriptors[slotIndex],
            static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)),
            slotIndex * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)));
    }

    BOOL CoreNVSPageManager::RegisterOrUpdateMeshVariantDescriptor(
        UINT32                       baseMeshId,
        UINT32                       submeshIndex,
        UINT32                       lodLevel,
        const GPUMeshDrawDescriptor& desc)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer)
        {
            return false;
        }
        if (baseMeshId == 0u || !EnsureMeshDrawDescriptorCapacity(baseMeshId + 1u))
        {
            m_lastError = "CoreNVSPageManager::RegisterOrUpdateMeshVariantDescriptor: baseMeshId out of bounds.";
            return false;
        }

        const UINT32 reqPacked = PackGPUMeshSubmeshAndLod(submeshIndex, lodLevel);
        GPUMeshDrawDescriptor& baseDesc = m_cpuMeshDescriptors[baseMeshId];
        if (baseDesc.submeshAndLodPacked == reqPacked && baseDesc.meshId == baseMeshId)
        {
            GPUMeshDrawDescriptor updated = desc;
            updated.meshId                = baseMeshId;
            updated.submeshAndLodPacked   = reqPacked;
            const UINT32 existingVarOff   = (baseDesc.meshGeneration >> 16u) & 0xFFFFu;
            const UINT32 lowGen           = ((updated.meshGeneration & 0xFFFFu) == 0u) ? 1u : (updated.meshGeneration & 0xFFFFu);
            updated.meshGeneration        = lowGen | (existingVarOff << 16u);
            return RegisterOrUpdateMeshDrawDescriptor(updated);
        }

        UINT32 varOffset = (baseDesc.meshGeneration >> 16u) & 0xFFFFu;
        if (varOffset == 0u)
        {
            const UINT32 blockStart = std::max<UINT32>(
                static_cast<UINT32>(m_cpuMeshDescriptors.size()),
                kNVSMaxMeshDrawDescriptors);
            if (blockStart + 16u > 65535u || !EnsureMeshDrawDescriptorCapacity(blockStart + 16u))
            {
                return false;
            }
            varOffset = blockStart;
            GPUMeshDrawDescriptor& refreshedBase = m_cpuMeshDescriptors[baseMeshId];
            const UINT32 lowGen   = ((refreshedBase.meshGeneration & 0xFFFFu) == 0u) ? 1u : (refreshedBase.meshGeneration & 0xFFFFu);
            refreshedBase.meshGeneration = lowGen | (varOffset << 16u);
            if (!m_bufferManager->UpdateGPUBuffer(
                    m_meshDrawDescriptorBuffer,
                    &refreshedBase,
                    static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)),
                    baseMeshId * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor))))
            {
                return false;
            }
        }

        const UINT32 maxCap   = static_cast<UINT32>(m_cpuMeshDescriptors.size());
        UINT32 targetSlot     = maxCap;
        for (UINT32 k = 0u; k < 16u; ++k)
        {
            const UINT32 probe = varOffset + k;
            if (probe >= maxCap) break;
            const GPUMeshDrawDescriptor& pDesc = m_cpuMeshDescriptors[probe];
            if ((pDesc.meshId == baseMeshId && pDesc.submeshAndLodPacked == reqPacked) ||
                (pDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_VALID) == 0u ||
                pDesc.meshId != baseMeshId)
            {
                targetSlot = probe;
                break;
            }
        }

        if (targetSlot >= maxCap)
        {
            m_lastError = "CoreNVSPageManager::RegisterOrUpdateMeshVariantDescriptor: variant block full.";
            return false;
        }

        GPUMeshDrawDescriptor varDesc = desc;
        varDesc.meshId                = baseMeshId;
        varDesc.submeshAndLodPacked   = reqPacked;
        if ((varDesc.meshGeneration & 0xFFFFu) == 0u)
        {
            varDesc.meshGeneration = 1u;
        }
        return RegisterOrUpdateMeshVariantDescriptor(targetSlot, varDesc);
    }

    BOOL CoreNVSPageManager::MarkMeshDrawDescriptorDestroyed(UINT32 meshId)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer ||
            meshId >= static_cast<UINT32>(m_cpuMeshDescriptors.size()))
        {
            return false;
        }
        GPUMeshDrawDescriptor& d = m_cpuMeshDescriptors[meshId];
        d.descriptorFlags = (d.descriptorFlags & ~GPU_MESH_DRAW_DESC_FLAG_VALID) | GPU_MESH_DRAW_DESC_FLAG_DESTROYED;
        return m_bufferManager->UpdateGPUBuffer(
            m_meshDrawDescriptorBuffer,
            &d,
            static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)),
            meshId * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)));
    }

    BOOL CoreNVSPageManager::UploadMeshDrawDescriptorTable(
        const GPUMeshDrawDescriptor* entries,
        UINT32                       entryCount,
        UINT32                       startSlot)
    {
        if (!m_initialized || !m_bufferManager || !m_meshDrawDescriptorBuffer ||
            !entries || entryCount == 0u || !EnsureMeshDrawDescriptorCapacity(startSlot + entryCount))
        {
            return false;
        }
        for (UINT32 i = 0u; i < entryCount; ++i)
        {
            m_cpuMeshDescriptors[startSlot + i] = entries[i];
        }
        return m_bufferManager->UpdateGPUBuffer(
            m_meshDrawDescriptorBuffer,
            entries,
            entryCount * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)),
            startSlot * static_cast<UINT32>(sizeof(GPUMeshDrawDescriptor)));
    }

    BOOL CoreNVSPageManager::RegisterMeshFromHandles(
        UINT32       meshId,
        NativeHandle vertexBufferHandle,
        NativeHandle indexBufferHandle,
        UINT32       vertexCount,
        UINT32       indexCount,
        UINT32       startIndex,
        INT32        baseVertex,
        UINT32       submeshIndex,
        UINT32       lodLevel,
        UINT32       descriptorFlags,
        UINT32       meshGeneration)
    {
        if (!m_initialized || !m_backend || !EnsureMeshDrawDescriptorCapacity(meshId + 1u))
        {
            return false;
        }

        GeometryBufferGPUViews gpuViews{};
        if (!m_backend->GetGeometryBufferGPUViews(vertexBufferHandle, indexBufferHandle, gpuViews) ||
            !gpuViews.valid)
        {
            m_lastError = "CoreNVSPageManager::RegisterMeshFromHandles: failed to query D3D12 VBV/IBV GPU virtual addresses.";
            return false;
        }

        GPUMeshDrawDescriptor desc{};
        desc.meshId              = meshId;
        desc.submeshAndLodPacked = (submeshIndex & 0xFFFFu) | ((lodLevel & 0xFFFFu) << 16u);
        desc.descriptorFlags     = descriptorFlags;
        desc.meshGeneration      = (meshGeneration == 0u) ? 1u : meshGeneration;
        desc.indexCount          = indexCount;
        desc.startIndex          = startIndex;
        desc.baseVertex          = baseVertex;
        desc.vertexCount         = vertexCount;
        desc.vbAddressLo         = static_cast<UINT32>(gpuViews.vbGPUVirtualAddress & 0xFFFFFFFFull);
        desc.vbAddressHi         = static_cast<UINT32>((gpuViews.vbGPUVirtualAddress >> 32ull) & 0xFFFFFFFFull);
        desc.vbSizeInBytes       = gpuViews.vbSizeInBytes;
        desc.vbStrideInBytes     = gpuViews.vbStrideInBytes;
        desc.ibAddressLo         = static_cast<UINT32>(gpuViews.ibGPUVirtualAddress & 0xFFFFFFFFull);
        desc.ibAddressHi         = static_cast<UINT32>((gpuViews.ibGPUVirtualAddress >> 32ull) & 0xFFFFFFFFull);
        desc.ibSizeInBytes       = gpuViews.ibSizeInBytes;
        desc.ibFormat            = gpuViews.ibFormat;

        if ((submeshIndex > 0u || lodLevel > 0u) &&
            (m_cpuMeshDescriptors[meshId].descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_VALID) != 0u &&
            m_cpuMeshDescriptors[meshId].meshId == meshId)
        {
            return RegisterOrUpdateMeshVariantDescriptor(meshId, submeshIndex, lodLevel, desc);
        }

        const UINT32 preservedVarBits = (m_cpuMeshDescriptors[meshId].meshId == meshId)
            ? (m_cpuMeshDescriptors[meshId].meshGeneration & 0xFFFF0000u)
            : 0u;
        const UINT32 lowGen = ((desc.meshGeneration & 0xFFFFu) == 0u) ? 1u : (desc.meshGeneration & 0xFFFFu);
        desc.meshGeneration = lowGen | preservedVarBits;
        return RegisterOrUpdateMeshDrawDescriptor(desc);
    }

    BOOL CoreNVSPageManager::RegisterOrUpdateMaterialShadowDescriptor(const GPUMaterialShadowDescriptor& desc)
    {
        if (!m_initialized || !m_bufferManager || !m_materialShadowDescriptorBuffer)
        {
            return false;
        }
        if (!EnsureMaterialShadowDescriptorCapacity(desc.materialId + 1u))
        {
            return false;
        }
        if (std::memcmp(&m_cpuMaterialDescriptors[desc.materialId], &desc, sizeof(GPUMaterialShadowDescriptor)) == 0)
        {
            return true;
        }
        m_cpuMaterialDescriptors[desc.materialId] = desc;
        return m_bufferManager->UpdateGPUBuffer(
            m_materialShadowDescriptorBuffer,
            &m_cpuMaterialDescriptors[desc.materialId],
            static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)),
            desc.materialId * static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)));
    }

    BOOL CoreNVSPageManager::MarkMaterialShadowDescriptorDestroyed(UINT32 materialId)
    {
        if (!m_initialized || !m_bufferManager || !m_materialShadowDescriptorBuffer ||
            materialId >= static_cast<UINT32>(m_cpuMaterialDescriptors.size()))
        {
            return false;
        }
        GPUMaterialShadowDescriptor& d = m_cpuMaterialDescriptors[materialId];
        d.materialFlags = (d.materialFlags & ~GPU_MAT_SHADOW_FLAG_VALID) | GPU_MAT_SHADOW_FLAG_DESTROYED_DEPENDENCY;
        return m_bufferManager->UpdateGPUBuffer(
            m_materialShadowDescriptorBuffer,
            &d,
            static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)),
            materialId * static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)));
    }

    BOOL CoreNVSPageManager::UploadMaterialShadowDescriptorTable(
        const GPUMaterialShadowDescriptor* entries,
        UINT32                             entryCount,
        UINT32                             startSlot)
    {
        if (!m_initialized || !m_bufferManager || !m_materialShadowDescriptorBuffer ||
            !entries || entryCount == 0u || !EnsureMaterialShadowDescriptorCapacity(startSlot + entryCount))
        {
            return false;
        }
        for (UINT32 i = 0u; i < entryCount; ++i)
        {
            m_cpuMaterialDescriptors[startSlot + i] = entries[i];
        }
        return m_bufferManager->UpdateGPUBuffer(
            m_materialShadowDescriptorBuffer,
            entries,
            entryCount * static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)),
            startSlot * static_cast<UINT32>(sizeof(GPUMaterialShadowDescriptor)));
    }

    BOOL CoreNVSPageManager::UploadSkinningPayload(
        UINT32                         payloadSlot,
        const NVSShadowSkinningPayload& payload)
    {
        if (!m_initialized || !m_bufferManager || !m_skinningPayloadBuffer ||
            payloadSlot >= kNVSMaxSkinningPayloads)
        {
            return false;
        }
        m_cpuSkinningPayloads[payloadSlot] = payload;
        return m_bufferManager->UpdateGPUBuffer(
            m_skinningPayloadBuffer,
            &m_cpuSkinningPayloads[payloadSlot],
            static_cast<UINT32>(sizeof(NVSShadowSkinningPayload)),
            payloadSlot * static_cast<UINT32>(sizeof(NVSShadowSkinningPayload)));
    }

    BOOL CoreNVSPageManager::FlushDescriptorTablesToGPU()
    {
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        if (m_meshDescriptorsDirty && m_meshDrawDescriptorBuffer && !m_cpuMeshDescriptors.empty())
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    m_meshDrawDescriptorBuffer,
                    m_cpuMeshDescriptors.data(),
                    static_cast<UINT32>(m_cpuMeshDescriptors.size() * sizeof(GPUMeshDrawDescriptor)),
                    0u))
            {
                return false;
            }
            m_meshDescriptorsDirty = false;
        }
        if (m_materialDescriptorsDirty && m_materialShadowDescriptorBuffer && !m_cpuMaterialDescriptors.empty())
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    m_materialShadowDescriptorBuffer,
                    m_cpuMaterialDescriptors.data(),
                    static_cast<UINT32>(m_cpuMaterialDescriptors.size() * sizeof(GPUMaterialShadowDescriptor)),
                    0u))
            {
                return false;
            }
            m_materialDescriptorsDirty = false;
        }
        if (m_skinningPayloadsDirty && m_skinningPayloadBuffer && !m_cpuSkinningPayloads.empty())
        {
            if (!m_bufferManager->UpdateGPUBuffer(
                    m_skinningPayloadBuffer,
                    m_cpuSkinningPayloads.data(),
                    static_cast<UINT32>(m_cpuSkinningPayloads.size() * sizeof(NVSShadowSkinningPayload)),
                    0u))
            {
                return false;
            }
            m_skinningPayloadsDirty = false;
        }
        return true;
    }

    void CoreNVSPageManager::ClearShadowBucketMaterials()
    {
        m_shadowRasterizer.ClearDrawGroups();
    }

    void CoreNVSPageManager::SetShadowBucketMaterial(
        UINT32        bucketIndex,
        Material*     material,
        ShaderVariant variant,
        Scene*        scene)
    {
        m_shadowRasterizer.SetShadowBucketMaterial(bucketIndex, material, variant, scene);
    }

    BOOL CoreNVSPageManager::BuildIndirectDrawComputeDispatch(
        NVSIndirectDrawDispatchMode    dispatchMode,
        GPUBuffer*                     gpuSceneBuffer,
        UINT32                         sceneSlotCount,
        const NVSShadowRasterPassDesc& passDesc,
        ComputeDispatchCommand&        outCmd,
        Camera*                        camera)
    {
        if (!m_initialized || !m_internalResourceManager)
        {
            m_lastError = "CoreNVSPageManager::BuildIndirectDrawComputeDispatch: manager not initialized.";
            return false;
        }

        Shader* cs = m_internalResourceManager->GetShader(InternalComputeKernelId::NVS_INDIRECT_DRAW_BUILD);
        if (!cs || cs->GetNativeHandle() == 0)
        {
            m_lastError = "CoreNVSPageManager::BuildIndirectDrawComputeDispatch: RuntimeNVSIndirectDraw shader missing.";
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            m_lastError = "CoreNVSPageManager::BuildIndirectDrawComputeDispatch: camera projection slot unavailable.";
            return false;
        }

        const UINT32 minShadowBuckets = std::max(
            kNVSInitialShadowRasterBucketCapacity,
            m_shadowRasterizer.GetActiveDrawGroupCount());
        if (!EnsureSlotShadowBucketCapacity(*slot, minShadowBuckets))
        {
            return false;
        }

        const UINT32 fSlot = slot->activeSlot % kNVSProjectionFrameSlots;
        if (!m_suppressSingleBindingUpload)
        {
            if (!UploadSlotIndirectBindingTables(*slot, fSlot, passDesc))
            {
                return false;
            }
        }

        GPUBuffer* projCntBuf          = slot->worklistIndirectArgsBuffers[fSlot];
        GPUBuffer* pageRangeBuf        = slot->casterPageRangeBuffers[fSlot];
        GPUBuffer* worklistBuf         = slot->casterWorklistBuffers[fSlot];
        GPUBuffer* renderDatBuf        = slot->pageRenderDataBuffers[fSlot];
        GPUBuffer* drawGroupBindBuf    = slot->drawGroupBindingsBuffers[fSlot];
        GPUBuffer* drawGroupPatchBuf   = slot->drawGroupFieldPatchesBuffers[fSlot];
        GPUBuffer* templateWordsBuf    = slot->templateWordsBuffers[fSlot];
        GPUBuffer* drawCtxBuf          = slot->shadowDrawContextBuffers[fSlot];
        GPUBuffer* drawCmdBuf          = slot->indirectDrawCommandBuffers[fSlot];
        GPUBuffer* perDrawCBVBuf       = slot->perDrawCBVWordsBuffers[fSlot];
        GPUBuffer* perBatchInstBuf     = slot->perBatchInstanceWordsBuffers[fSlot];
        GPUBuffer* allocCountersBuf    = slot->allocationCountersBuffers[fSlot];
        GPUBuffer* bktCountBuf         = slot->bucketCommandCountBuffers[fSlot];
        GPUBuffer* commitRecBuf        = slot->pageCommitRecordBuffers[fSlot];
        GPUBuffer* indTelBuf           = slot->indirectDrawTelemetryBuffers[fSlot];

        GPUBuffer* effSceneBuf = passDesc.overrideGPUSceneBuffer
            ? passDesc.overrideGPUSceneBuffer
            : (gpuSceneBuffer ? gpuSceneBuffer : m_dummySceneObjectBuffer);
        const UINT32 effSceneCount = (passDesc.overrideSceneSlotCount > 0u)
            ? passDesc.overrideSceneSlotCount
            : (sceneSlotCount > 0u ? sceneSlotCount : (effSceneBuf ? effSceneBuf->GetElementCount() : 1u));

        if (!projCntBuf || !pageRangeBuf || !worklistBuf || !renderDatBuf ||
            !effSceneBuf || !m_meshDrawDescriptorBuffer || !m_materialShadowDescriptorBuffer ||
            !drawGroupBindBuf || !drawGroupPatchBuf || !templateWordsBuf ||
            !drawCtxBuf || !drawCmdBuf || !perDrawCBVBuf || !perBatchInstBuf ||
            !allocCountersBuf || !bktCountBuf || !commitRecBuf || !indTelBuf)
        {
            m_lastError = "CoreNVSPageManager::BuildIndirectDrawComputeDispatch: required GPU buffer missing.";
            return false;
        }

        GPUBuffer* bonePaletteBuf = passDesc.gpuSkinningPaletteBuffer;
        if (!bonePaletteBuf)
        {
            for (const auto& dg : m_shadowRasterizer.GetDrawGroups())
            {
                if (dg.scene && dg.scene->GetGPUSkinningPaletteBuffer() != nullptr)
                {
                    bonePaletteBuf = dg.scene->GetGPUSkinningPaletteBuffer();
                    break;
                }
            }
        }
        if (!bonePaletteBuf)
        {
            bonePaletteBuf = m_skinningPayloadBuffer;
        }

        const UINT64 cbvPoolGpuAddr = m_backend
            ? m_backend->GetGPUBufferGPUVirtualAddress(perDrawCBVBuf->GetNativeHandle())
            : 0ull;
        const UINT64 instPoolGpuAddr = m_backend
            ? m_backend->GetGPUBufferGPUVirtualAddress(perBatchInstBuf->GetNativeHandle())
            : 0ull;
        const UINT64 bonePaletteGpuAddr = (m_backend && bonePaletteBuf)
            ? m_backend->GetGPUBufferGPUVirtualAddress(bonePaletteBuf->GetNativeHandle())
            : 0ull;

        struct alignas(16) IndirectDrawConfigCB
        {
            UINT32 dispatchMode             = 0u;
            UINT32 currentFrame             = 1u;
            UINT32 maxActivePages           = 2048u;
            UINT32 maxWorklistEntries       = 65536u;
            UINT32 maxSceneSlots            = 65536u;
            UINT32 maxMeshDescriptors       = kNVSMaxMeshDrawDescriptors;
            UINT32 maxMaterialDescriptors   = kNVSMaxMaterialShadowDescriptors;
            UINT32 maxRenderDataSlots       = 2048u;
            UINT32 maxCommandsPerBucket     = kNVSMaxIndirectCommandsPerBucket;
            UINT32 maxDrawContexts          = kNVSMaxShadowDrawContextsCapacity;
            UINT32 physicalPoolCapacity     = 2048u;
            UINT32 enableHardwareInstancing = 1u;
            UINT32 useGPUActivePageCount    = 1u;
            UINT32 explicitActivePageCount  = 0u;
            UINT32 maxShadowBuckets         = kNVSInitialShadowRasterBucketCapacity;
            UINT32 drawGroupBindingCount    = 0u;
            UINT32 cbvPoolBaseLo            = 0u;
            UINT32 cbvPoolBaseHi            = 0u;
            UINT32 instancePoolBaseLo       = 0u;
            UINT32 instancePoolBaseHi       = 0u;
            UINT32 bonePaletteBaseLo        = 0u;
            UINT32 bonePaletteBaseHi        = 0u;
            UINT32 maxPerDrawCBVWords       = 0u;
            UINT32 maxPerBatchInstanceWords = 0u;
            UINT32 applyShaderBias          = 0u;
            FLOAT32 depthBiasScale          = 1.0f;
            FLOAT32 normalBiasScale         = 1.0f;
            FLOAT32 passTimeSeconds         = 0.0f;
        } cb{};
        static_assert(sizeof(IndirectDrawConfigCB) == 112, "IndirectDrawConfigCB must be 112 bytes (28 x 4B).");

        cb.dispatchMode             = static_cast<UINT32>(dispatchMode);
        cb.currentFrame             = passDesc.currentFrame;
        cb.maxActivePages           = m_config.physicalPoolCapacity;
        cb.maxWorklistEntries       = worklistBuf->GetElementCount();
        cb.maxSceneSlots            = std::max(effSceneCount, 1u);
        cb.maxMeshDescriptors       = std::max<UINT32>(1u, static_cast<UINT32>(m_cpuMeshDescriptors.size()));
        cb.maxMaterialDescriptors   = std::max<UINT32>(1u, static_cast<UINT32>(m_cpuMaterialDescriptors.size()));
        cb.maxRenderDataSlots       = m_config.physicalPoolCapacity;
        cb.maxCommandsPerBucket     = (passDesc.maxCommandsPerBucketOverride > 0u &&
                                       passDesc.maxCommandsPerBucketOverride <= kNVSMaxIndirectCommandsPerBucket)
                                      ? passDesc.maxCommandsPerBucketOverride
                                      : kNVSMaxIndirectCommandsPerBucket;
        cb.maxDrawContexts          = (passDesc.maxDrawContextsOverride > 0u &&
                                       passDesc.maxDrawContextsOverride <= kNVSMaxShadowDrawContextsCapacity)
                                      ? passDesc.maxDrawContextsOverride
                                      : kNVSMaxShadowDrawContextsCapacity;
        cb.physicalPoolCapacity     = m_config.physicalPoolCapacity;
        cb.enableHardwareInstancing = (passDesc.enableHardwareInstancing && passDesc.allowInstancedMerge) ? 1u : 0u;
        cb.useGPUActivePageCount    = 1u;
        cb.explicitActivePageCount  = 0u;
        cb.maxShadowBuckets         = slot->allocatedShadowBucketCapacity;
        cb.drawGroupBindingCount    = slot->lastUploadedDrawGroupCount;
        cb.cbvPoolBaseLo            = static_cast<UINT32>(cbvPoolGpuAddr & 0xFFFFFFFFull);
        cb.cbvPoolBaseHi            = static_cast<UINT32>((cbvPoolGpuAddr >> 32ull) & 0xFFFFFFFFull);
        cb.instancePoolBaseLo       = static_cast<UINT32>(instPoolGpuAddr & 0xFFFFFFFFull);
        cb.instancePoolBaseHi       = static_cast<UINT32>((instPoolGpuAddr >> 32ull) & 0xFFFFFFFFull);
        cb.bonePaletteBaseLo        = static_cast<UINT32>(bonePaletteGpuAddr & 0xFFFFFFFFull);
        cb.bonePaletteBaseHi        = static_cast<UINT32>((bonePaletteGpuAddr >> 32ull) & 0xFFFFFFFFull);
        cb.maxPerDrawCBVWords       = perDrawCBVBuf->GetElementCount();
        cb.maxPerBatchInstanceWords = perBatchInstBuf->GetElementCount();
        cb.applyShaderBias          = passDesc.applyShaderBias ? 1u : 0u;
        cb.depthBiasScale           = passDesc.depthBiasScale;
        cb.normalBiasScale          = passDesc.normalBiasScale;
        cb.passTimeSeconds          = passDesc.timeSeconds;

        outCmd = ComputeDispatchCommand{};
        outCmd.computeShader = cs->GetNativeHandle();
        outCmd.profileGroup  = ComputeDispatchProfileGroup::RESIDENCY_CULL;
        if (dispatchMode == NVS_INDIRECT_DRAW_MODE_BUILD_COMMANDS)
        {
            outCmd.threadGroupCountX = (m_config.physicalPoolCapacity + 63u) / 64u;
        }
        else if (dispatchMode == NVS_INDIRECT_DRAW_MODE_RESET_COUNTERS)
        {
            const UINT32 resetWorkItems = std::max(slot->allocatedShadowBucketCapacity + 1u, 64u);
            outCmd.threadGroupCountX    = (resetWorkItems + 63u) / 64u;
        }
        else
        {
            outCmd.threadGroupCountX = 1u;
        }
        outCmd.threadGroupCountY = 1u;
        outCmd.threadGroupCountZ = 1u;

        NativeConstantBufferBinding cbBind{};
        cbBind.slotIndex     = 0;
        cbBind.registerSpace = 0;
        cbBind.stage         = ShaderStage::COMPUTE;
        cbBind.byteSize      = sizeof(cb);
        cbBind.ownedData.resize(sizeof(cb));
        std::memcpy(cbBind.ownedData.data(), &cb, sizeof(cb));
        cbBind.cpuData       = cbBind.ownedData.data();
        cbBind.dirty         = true;
        outCmd.constantBuffers.push_back(std::move(cbBind));

        auto bindSrv = [&](UINT16 slotIdx, GPUBuffer* buf) {
            NativeStructuredBufferBinding b{};
            b.slotIndex     = slotIdx;
            b.registerSpace = 0;
            b.stage         = ShaderStage::COMPUTE;
            b.buffer        = buf->GetNativeHandle();
            b.elementCount  = buf->GetElementCount();
            b.stride        = buf->GetStride();
            b.byteSize      = buf->GetByteSize();
            b.isUAV         = false;
            outCmd.structuredBuffers.push_back(std::move(b));
        };

        auto bindUav = [&](UINT16 slotIdx, GPUBuffer* buf) {
            NativeStructuredBufferBinding b{};
            b.slotIndex     = slotIdx;
            b.registerSpace = 0;
            b.stage         = ShaderStage::COMPUTE;
            b.buffer        = buf->GetNativeHandle();
            b.elementCount  = buf->GetElementCount();
            b.stride        = buf->GetStride();
            b.byteSize      = buf->GetByteSize();
            b.isUAV         = true;
            b.uavAccess     = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
            outCmd.structuredBuffers.push_back(std::move(b));
        };

        // SRVs t0..t9
        bindSrv(0, projCntBuf);
        bindSrv(1, pageRangeBuf);
        bindSrv(2, worklistBuf);
        bindSrv(3, renderDatBuf);
        bindSrv(4, effSceneBuf);
        bindSrv(5, m_meshDrawDescriptorBuffer);
        bindSrv(6, m_materialShadowDescriptorBuffer);
        bindSrv(7, drawGroupBindBuf);
        bindSrv(8, drawGroupPatchBuf);
        bindSrv(9, templateWordsBuf);

        // UAVs u0..u7
        bindUav(0, drawCtxBuf);
        bindUav(1, drawCmdBuf);
        bindUav(2, perDrawCBVBuf);
        bindUav(3, perBatchInstBuf);
        bindUav(4, allocCountersBuf);
        bindUav(5, bktCountBuf);
        bindUav(6, commitRecBuf);
        bindUav(7, indTelBuf);

        return true;
    }

    BOOL CoreNVSPageManager::BuildIndirectDrawPipelineDispatches(
        GPUBuffer*                           gpuSceneBuffer,
        UINT32                               sceneSlotCount,
        const NVSShadowRasterPassDesc&       passDesc,
        std::vector<ComputeDispatchCommand>& outDispatches,
        Camera*                              camera)
    {
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }
        const UINT32 fSlot = slot->activeSlot % kNVSProjectionFrameSlots;
        if (!UploadSlotIndirectBindingTables(*slot, fSlot, passDesc))
        {
            return false;
        }

        ComputeDispatchCommand cmdReset{};
        ComputeDispatchCommand cmdBuild{};
        ComputeDispatchCommand cmdFinalize{};

        m_suppressSingleBindingUpload = true;
        const BOOL ok =
            BuildIndirectDrawComputeDispatch(
                NVS_INDIRECT_DRAW_MODE_RESET_COUNTERS, gpuSceneBuffer, sceneSlotCount, passDesc, cmdReset, camera) &&
            BuildIndirectDrawComputeDispatch(
                NVS_INDIRECT_DRAW_MODE_BUILD_COMMANDS, gpuSceneBuffer, sceneSlotCount, passDesc, cmdBuild, camera) &&
            BuildIndirectDrawComputeDispatch(
                NVS_INDIRECT_DRAW_MODE_FINALIZE_PAGE_COMMIT, gpuSceneBuffer, sceneSlotCount, passDesc, cmdFinalize, camera);
        m_suppressSingleBindingUpload = false;
        if (!ok)
        {
            return false;
        }

        outDispatches.push_back(std::move(cmdReset));
        outDispatches.push_back(std::move(cmdBuild));
        outDispatches.push_back(std::move(cmdFinalize));
        return true;
    }

    BOOL CoreNVSPageManager::BuildShadowRasterizationPassView(
        GPUBuffer*                     gpuSceneBuffer,
        UINT32                         sceneSlotCount,
        const NVSShadowRasterPassDesc& passDesc,
        LayeredDepthPassView&          outPassView,
        Camera*                        camera)
    {
        if (!m_initialized || !m_internalResourceManager || m_physicalPoolTexture == 0)
        {
            m_lastError = "CoreNVSPageManager::BuildShadowRasterizationPassView: manager not initialized.";
            return false;
        }

        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            m_lastError = "CoreNVSPageManager::BuildShadowRasterizationPassView: camera projection slot unavailable.";
            return false;
        }

        const UINT32 minShadowBuckets = std::max(
            kNVSInitialShadowRasterBucketCapacity,
            m_shadowRasterizer.GetActiveDrawGroupCount());
        if (!EnsureSlotShadowBucketCapacity(*slot, minShadowBuckets))
        {
            return false;
        }

        const UINT32 fSlot = slot->activeSlot % kNVSProjectionFrameSlots;
        GPUBuffer* renderDatBuf    = slot->pageRenderDataBuffers[fSlot];
        GPUBuffer* drawCtxBuf      = slot->shadowDrawContextBuffers[fSlot];
        GPUBuffer* drawCmdBuf      = slot->indirectDrawCommandBuffers[fSlot];
        GPUBuffer* bktCountBuf     = slot->bucketCommandCountBuffers[fSlot];
        GPUBuffer* perDrawCBVBuf   = slot->perDrawCBVWordsBuffers[fSlot];
        GPUBuffer* perBatchInstBuf = slot->perBatchInstanceWordsBuffers[fSlot];

        GPUBuffer* effSceneBuf = passDesc.overrideGPUSceneBuffer
            ? passDesc.overrideGPUSceneBuffer
            : (gpuSceneBuffer ? gpuSceneBuffer : m_dummySceneObjectBuffer);
        const UINT32 effSceneCount = (passDesc.overrideSceneSlotCount > 0u)
            ? passDesc.overrideSceneSlotCount
            : (sceneSlotCount > 0u ? sceneSlotCount : (effSceneBuf ? effSceneBuf->GetElementCount() : 1u));

        if (!renderDatBuf || !drawCtxBuf || !drawCmdBuf || !bktCountBuf ||
            !perDrawCBVBuf || !perBatchInstBuf || !effSceneBuf || !m_skinningPayloadBuffer)
        {
            m_lastError = "CoreNVSPageManager::BuildShadowRasterizationPassView: required GPU buffers missing.";
            return false;
        }

        return m_shadowRasterizer.BuildShadowRasterizationPassView(
            passDesc,
            m_physicalPoolTexture,
            m_config.physicalPoolCapacity,
            drawCmdBuf,
            bktCountBuf,
            drawCtxBuf,
            effSceneBuf,
            renderDatBuf,
            m_skinningPayloadBuffer,
            effSceneCount,
            outPassView,
            m_lastError,
            perDrawCBVBuf,
            perBatchInstBuf);
    }

    BOOL CoreNVSPageManager::BuildGPUCommitRenderedPagesDispatch(
        UINT32                  currentFrame,
        ComputeDispatchCommand& outCmd,
        Camera*                 camera)
    {
        CameraProjectionSlot* slot = AcquireCameraProjectionSlot(camera);
        if (!slot || !slot->allocated)
        {
            return false;
        }
        const UINT32 fSlot = slot->activeSlot % kNVSProjectionFrameSlots;
        GPUBuffer* commitRecBuf = slot->pageCommitRecordBuffers[fSlot];
        if (!commitRecBuf)
        {
            return false;
        }

        NVSResidencyPassDesc resDesc{};
        resDesc.currentFrame                    = currentFrame;
        resDesc.requestCount                    = m_config.physicalPoolCapacity;
        resDesc.invalidationMode                = NVS_INVALIDATE_NONE;
        resDesc.executeSelectiveClear           = false;
        resDesc.useGPURequestCountFromTelemetry = true;

        if (!BuildResidencyComputeDispatch(
                NVS_DISPATCH_COMMIT_RENDERED_PAGES,
                resDesc,
                outCmd,
                commitRecBuf,
                camera))
        {
            return false;
        }

        CommitSamplingViewTableSnapshot(currentFrame, camera);
        return true;
    }

    BOOL CoreNVSPageManager::ExecuteShadowRasterizationPass(
        GPUBuffer*                     gpuSceneBuffer,
        UINT32                         sceneSlotCount,
        const NVSShadowRasterPassDesc& passDesc,
        Camera*                        camera)
    {
        if (!m_initialized || !m_backend)
        {
            m_lastError = "CoreNVSPageManager::ExecuteShadowRasterizationPass: manager not initialized.";
            return false;
        }

        if (!FlushDescriptorTablesToGPU())
        {
            return false;
        }

        std::vector<ComputeDispatchCommand> preDispatches;
        if (!BuildIndirectDrawPipelineDispatches(gpuSceneBuffer, sceneSlotCount, passDesc, preDispatches, camera))
        {
            return false;
        }

        LayeredDepthPassView passView{};
        if (!BuildShadowRasterizationPassView(gpuSceneBuffer, sceneSlotCount, passDesc, passView, camera))
        {
            return false;
        }

        std::vector<LayeredDepthPassView> passes = { std::move(passView) };
        if (!m_backend->ExecuteLayeredDepthPasses(passes, &preDispatches))
        {
            m_lastError = "CoreNVSPageManager::ExecuteShadowRasterizationPass: ExecuteLayeredDepthPasses failed.";
            return false;
        }

        if (passDesc.commitRenderedPagesOnGPU && passDesc.commitRenderedPages)
        {
            ComputeDispatchCommand commitCmd{};
            if (!BuildGPUCommitRenderedPagesDispatch(passDesc.currentFrame, commitCmd, camera))
            {
                return false;
            }
            std::vector<ComputeDispatchCommand> commitDispatches = { std::move(commitCmd) };
            if (!m_backend->ExecuteComputeDispatches(commitDispatches))
            {
                m_lastError = "CoreNVSPageManager::ExecuteShadowRasterizationPass: GPU page commit dispatch failed.";
                return false;
            }
        }

        return true;
    }

    BOOL CoreNVSPageManager::ReadbackIndirectDrawTelemetrySync(
        NVSIndirectDrawTelemetry& outTelemetry,
        Camera*                   camera) const
    {
        GPUBuffer* buf = GetIndirectDrawTelemetryBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            &outTelemetry,
            static_cast<UINT32>(sizeof(NVSIndirectDrawTelemetry)));
    }

    BOOL CoreNVSPageManager::ReadbackBucketCommandCountsSync(
        UINT32  outCounts[16],
        Camera* camera) const
    {
        GPUBuffer* buf = GetBucketCommandCountBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf || !outCounts)
        {
            return false;
        }
        const UINT32 elemCount = buf->GetElementCount();
        if (elemCount < 2u)
        {
            return false;
        }
        const UINT32 bucketCap = elemCount - 1u;
        std::vector<UINT32> fullCounts(elemCount, 0u);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                buf,
                fullCounts.data(),
                elemCount * static_cast<UINT32>(sizeof(UINT32))))
        {
            return false;
        }
        for (UINT32 i = 0u; i < 16u; ++i)
        {
            outCounts[i] = (i < bucketCap) ? fullCounts[i] : 0u;
        }
        // Preserve legacy index 8 as total DrawContext count only when <= 8 buckets are active
        if (m_shadowRasterizer.GetActiveDrawGroupCount() <= 8u && outCounts[8] == 0u)
        {
            outCounts[8] = fullCounts[bucketCap];
        }
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackIndirectDrawCommandsSync(
        UINT32                               bucketIndex,
        std::vector<NVSIndirectDrawCommand>& outCommands,
        UINT32                               outCount,
        Camera*                              camera) const
    {
        GPUBuffer* buf = GetIndirectDrawCommandBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 availBuckets = buf->GetElementCount() / kNVSMaxIndirectCommandsPerBucket;
        if (bucketIndex >= availBuckets)
        {
            return false;
        }

        if (outCount == 0u)
        {
            GPUBuffer* cntBuf = GetBucketCommandCountBuffer(camera);
            const UINT32 elemCount = cntBuf ? cntBuf->GetElementCount() : 0u;
            if (cntBuf && bucketIndex + 1u < elemCount)
            {
                std::vector<UINT32> fullCounts(elemCount, 0u);
                if (m_bufferManager->ReadbackGPUBufferSync(
                        cntBuf,
                        fullCounts.data(),
                        elemCount * static_cast<UINT32>(sizeof(UINT32))))
                {
                    outCount = fullCounts[bucketIndex];
                }
            }
        }
        if (outCount == 0u)
        {
            outCommands.clear();
            return true;
        }

        const UINT32 clampedCount = std::min(outCount, kNVSMaxIndirectCommandsPerBucket);
        const UINT32 totalNeeded  = bucketIndex * kNVSMaxIndirectCommandsPerBucket + clampedCount;
        std::vector<NVSIndirectDrawCommand> temp(totalNeeded);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                buf,
                temp.data(),
                totalNeeded * static_cast<UINT32>(sizeof(NVSIndirectDrawCommand))))
        {
            return false;
        }

        const UINT32 baseOffset = bucketIndex * kNVSMaxIndirectCommandsPerBucket;
        outCommands.assign(temp.begin() + baseOffset, temp.begin() + baseOffset + clampedCount);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackShadowDrawContextsSync(
        std::vector<NVSShadowDrawContext>& outContexts,
        UINT32                             outCount,
        Camera*                            camera) const
    {
        GPUBuffer* buf = GetShadowDrawContextBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        if (outCount == 0u)
        {
            GPUBuffer* cntBuf = GetBucketCommandCountBuffer(camera);
            const UINT32 elemCount = cntBuf ? cntBuf->GetElementCount() : 0u;
            if (cntBuf && elemCount >= 2u)
            {
                std::vector<UINT32> fullCounts(elemCount, 0u);
                if (m_bufferManager->ReadbackGPUBufferSync(
                        cntBuf,
                        fullCounts.data(),
                        elemCount * static_cast<UINT32>(sizeof(UINT32))))
                {
                    outCount = fullCounts[elemCount - 1u];
                }
            }
        }
        if (outCount == 0u)
        {
            outContexts.clear();
            return true;
        }
        const UINT32 clampedCount = std::min(outCount, kNVSMaxShadowDrawContextsCapacity);
        outContexts.resize(clampedCount);
        return m_bufferManager->ReadbackGPUBufferSync(
            buf,
            outContexts.data(),
            clampedCount * static_cast<UINT32>(sizeof(NVSShadowDrawContext)));
    }

    BOOL CoreNVSPageManager::ReadbackPageCommitRecordsSync(
        std::vector<NVSPageRenderCommitRecord>& outRecords,
        UINT32                                  outCount,
        Camera*                                 camera) const
    {
        GPUBuffer* buf = GetPageCommitRecordBuffer(camera);
        if (!m_initialized || !m_bufferManager || !buf)
        {
            return false;
        }
        const UINT32 cap = m_config.physicalPoolCapacity;
        const UINT32 cnt = (outCount > 0u && outCount <= cap) ? outCount : cap;
        std::vector<NVSPageRenderCommitRecord> fullBuf(cap);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                buf,
                fullBuf.data(),
                cap * static_cast<UINT32>(sizeof(NVSPageRenderCommitRecord))))
        {
            return false;
        }
        outRecords.assign(fullBuf.begin(), fullBuf.begin() + cnt);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackPhysicalPoolSliceSync(
        UINT32                sliceIndex,
        std::vector<FLOAT32>& outDepthTexels) const
    {
        if (!m_initialized || !m_backend || m_physicalPoolTexture == 0 ||
            sliceIndex >= m_config.physicalPoolCapacity)
        {
            return false;
        }

        const UINT32 texelsPerSlice = kNVSPageTexelSize * kNVSPageTexelSize;
        const UINT32 totalTexels    = m_config.physicalPoolCapacity * texelsPerSlice;
        std::vector<FLOAT32> fullPool(totalTexels, 0.0f);
        if (!m_backend->ReadbackTextureSync(
                m_physicalPoolTexture,
                fullPool.data(),
                totalTexels * static_cast<UINT32>(sizeof(FLOAT32))))
        {
            return false;
        }

        const UINT32 sliceOffset = sliceIndex * texelsPerSlice;
        outDepthTexels.assign(
            fullPool.begin() + sliceOffset,
            fullPool.begin() + sliceOffset + texelsPerSlice);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackSparseHierarchyMetricsSync(
        NVSSparseHierarchyMetrics& outMetrics,
        Camera*                    camera) const
    {
        outMetrics = NVSSparseHierarchyMetrics{};
        if (!m_initialized || !m_bufferManager)
        {
            return false;
        }
        const CameraProjectionSlot* slot = FindCameraProjectionSlot(camera);
        outMetrics.viewTableCapacity   = slot ? slot->allocatedViewTableCapacity : 0u;
        outMetrics.activeViewCount     = slot ? slot->activeViewTableCount : 0u;
        outMetrics.sparseRootCapacity  = m_sparseRootCapacity;
        outMetrics.sparseDirCapacity   = m_sparseDirCapacity;
        outMetrics.sparseBlockCapacity = m_sparseBlockCapacity;

        if (!m_config.enableSparseHierarchy || !m_sparsePageHierarchyBuffer)
        {
            outMetrics.sparseMemoryBytes = m_virtualPageTableBuffer ? m_virtualPageTableBuffer->GetByteSize() : 0ull;
            return true;
        }

        const UINT32 totalWords = m_sparsePageHierarchyBuffer->GetElementCount();
        const UINT32 allocBase  = m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride);
        if (totalWords < allocBase + 8u)
        {
            return false;
        }

        std::vector<UINT32> fullHierarchy(totalWords, 0u);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_sparsePageHierarchyBuffer,
                fullHierarchy.data(),
                totalWords * static_cast<UINT32>(sizeof(UINT32))))
        {
            return false;
        }

        const UINT32 dirHead = fullHierarchy[allocBase + 0u];
        const UINT32 dirTail = fullHierarchy[allocBase + 1u];
        const UINT32 blkHead = fullHierarchy[allocBase + 2u];
        const UINT32 blkTail = fullHierarchy[allocBase + 3u];

        const UINT32 freeDirs = std::min(static_cast<UINT32>(dirTail - dirHead), m_sparseDirCapacity);
        const UINT32 freeBlks = std::min(static_cast<UINT32>(blkTail - blkHead), m_sparseBlockCapacity);

        outMetrics.freeDirCount        = freeDirs;
        outMetrics.allocatedDirCount   = (m_sparseDirCapacity >= freeDirs) ? (m_sparseDirCapacity - freeDirs) : 0u;
        outMetrics.freeBlockCount      = freeBlks;
        outMetrics.allocatedBlockCount = (m_sparseBlockCapacity >= freeBlks) ? (m_sparseBlockCapacity - freeBlks) : 0u;
        outMetrics.dirOverflowCount    = fullHierarchy[allocBase + 4u];
        outMetrics.blkOverflowCount    = fullHierarchy[allocBase + 5u];
        outMetrics.sparseMemoryBytes   =
            static_cast<UINT64>(m_sparsePageHierarchyBuffer->GetByteSize()) +
            static_cast<UINT64>(m_virtualPageTableBuffer ? m_virtualPageTableBuffer->GetByteSize() : 0u) +
            static_cast<UINT64>(m_missRecordsBuffer ? m_missRecordsBuffer->GetByteSize() : 0u);
        return true;
    }

    BOOL CoreNVSPageManager::ReadbackSparsePageHierarchySync(std::vector<UINT32>& outNodes) const
    {
        outNodes.clear();
        if (!m_initialized || !m_bufferManager || !m_sparsePageHierarchyBuffer || !m_config.enableSparseHierarchy)
        {
            return false;
        }
        const UINT32 count = m_sparseRootCapacity + (m_sparseDirCapacity * kNVSSparseDirStride);
        if (m_sparsePageHierarchyBuffer->GetElementCount() < count)
        {
            return false;
        }
        std::vector<UINT32> fullBuf(m_sparsePageHierarchyBuffer->GetElementCount(), kNVSSparseUnmappedIndex);
        if (!m_bufferManager->ReadbackGPUBufferSync(
                m_sparsePageHierarchyBuffer,
                fullBuf.data(),
                static_cast<UINT32>(fullBuf.size() * sizeof(UINT32))))
        {
            return false;
        }
        outNodes.assign(fullBuf.begin(), fullBuf.begin() + count);
        return true;
    }

    UINT64 CoreNVSPageManager::GetMetadataVramBytes() const noexcept
    {
        if (!m_initialized)
        {
            return 0ull;
        }
        auto addBuf = [](UINT64& sum, const GPUBuffer* b) noexcept {
            if (b) sum += b->GetByteSize();
        };
        UINT64 total = 0ull;
        addBuf(total, m_virtualPageTableBuffer);
        addBuf(total, m_physicalPageTableBuffer);
        addBuf(total, m_pageRequestBuffer);
        addBuf(total, m_dirtyPhysicalSlicesBuffer);
        addBuf(total, m_dirtySliceCountBuffer);
        addBuf(total, m_selectiveClearIndirectArgsBuffer);
        addBuf(total, m_telemetryBuffer);
        addBuf(total, m_sparsePageHierarchyBuffer);
        addBuf(total, m_invalidationCommandsBuffer);
        addBuf(total, m_missRecordsBuffer);
        addBuf(total, m_seedCandidateBuffer);
        addBuf(total, m_filteredCasterBuffer);
        addBuf(total, m_casterValidBitmaskBuffer);
        addBuf(total, m_dynamicInvalidationBoxesBuffer);
        addBuf(total, m_meshDrawDescriptorBuffer);
        addBuf(total, m_materialShadowDescriptorBuffer);
        addBuf(total, m_skinningPayloadBuffer);

        auto sumSlot = [&](const CameraProjectionSlot& sl) noexcept {
            if (!sl.allocated) return;
            for (UINT32 s = 0u; s < kNVSProjectionFrameSlots; ++s)
            {
                addBuf(total, sl.clipmapLevelTableBuffers[s]);
                addBuf(total, sl.pageRenderDataBuffers[s]);
                addBuf(total, sl.activeRenderPageListBuffers[s]);
                addBuf(total, sl.projectionTelemetryBuffers[s]);
                addBuf(total, sl.casterPageRangeBuffers[s]);
                addBuf(total, sl.casterWorklistBuffers[s]);
                addBuf(total, sl.casterCullTelemetryBuffers[s]);
                addBuf(total, sl.worklistIndirectArgsBuffers[s]);
                addBuf(total, sl.shadowDrawContextBuffers[s]);
                addBuf(total, sl.indirectDrawCommandBuffers[s]);
                addBuf(total, sl.bucketCommandCountBuffers[s]);
                addBuf(total, sl.pageCommitRecordBuffers[s]);
                addBuf(total, sl.indirectDrawTelemetryBuffers[s]);
                addBuf(total, sl.drawGroupBindingsBuffers[s]);
                addBuf(total, sl.drawGroupFieldPatchesBuffers[s]);
                addBuf(total, sl.templateWordsBuffers[s]);
                addBuf(total, sl.perDrawCBVWordsBuffers[s]);
                addBuf(total, sl.perBatchInstanceWordsBuffers[s]);
                addBuf(total, sl.allocationCountersBuffers[s]);
                addBuf(total, sl.committedSamplingViewTableBuffers[s]);
            }
        };
        sumSlot(m_defaultProjectionSlot);
        for (const auto& sl : m_cameraProjectionSlots)
        {
            sumSlot(sl);
        }
        return total;
    }
}


