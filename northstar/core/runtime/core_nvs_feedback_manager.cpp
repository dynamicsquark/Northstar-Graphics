#include "core_nvs_feedback_manager.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Northstar
{
    CoreNVSFeedbackManager::CoreNVSFeedbackManager() = default;

    CoreNVSFeedbackManager::~CoreNVSFeedbackManager()
    {
        Shutdown();
    }

    BOOL CoreNVSFeedbackManager::Initialize(
        NorthstarBackend*               backend,
        CoreBufferManager*              bufferManager,
        CoreTextureManager*             textureManager,
        CoreComputeManager*             computeManager,
        CoreRenderPipeline*             renderPipeline,
        InternalResourceManager*        internalResourceManager,
        CoreNVSPageManager*             pageManager,
        const NVSFeedbackManagerConfig& config)
    {
        Shutdown();
        m_lastError.clear();

        if (!backend || !bufferManager || !textureManager || !computeManager || !renderPipeline || !internalResourceManager)
        {
            m_lastError = "CoreNVSFeedbackManager::Initialize requires valid backend, buffer, texture, compute, pipeline, and internal resource managers.";
            return false;
        }

        if (config.maxRequestBudget == 0u || config.maxRequestBudget > 131072u)
        {
            m_lastError = "CoreNVSFeedbackManager::Initialize: invalid maxRequestBudget.";
            return false;
        }

        m_backend                 = backend;
        m_bufferManager           = bufferManager;
        m_textureManager          = textureManager;
        m_computeManager          = computeManager;
        m_renderPipeline          = renderPipeline;
        m_internalResourceManager = internalResourceManager;
        m_pageManager             = pageManager;
        m_config                  = config;

        if (!CreateDummyDepthTexture())
        {
            ReleaseAllResources();
            return false;
        }

        if (!AllocateSlotResources(m_defaultSlot, m_config.maxRequestBudget, "NVS_FeedbackDefault"))
        {
            ReleaseAllResources();
            return false;
        }

        m_initialized = true;
        ResetBuffersOnGPU(nullptr);
        return true;
    }

    void CoreNVSFeedbackManager::Shutdown()
    {
        ReleaseAllResources();
        m_backend                 = nullptr;
        m_bufferManager           = nullptr;
        m_textureManager          = nullptr;
        m_computeManager          = nullptr;
        m_renderPipeline          = nullptr;
        m_internalResourceManager = nullptr;
        m_pageManager             = nullptr;
        m_initialized             = false;
    }

    BOOL CoreNVSFeedbackManager::Reconfigure(const NVSFeedbackManagerConfig& newConfig)
    {
        if (!m_initialized)
        {
            m_lastError = "CoreNVSFeedbackManager::Reconfigure called before Initialize.";
            return false;
        }

        if (newConfig.maxRequestBudget == 0u || newConfig.maxRequestBudget > 131072u)
        {
            m_lastError = "CoreNVSFeedbackManager::Reconfigure: invalid maxRequestBudget.";
            return false;
        }

        NorthstarBackend*        backend                 = m_backend;
        CoreBufferManager*       bufferManager           = m_bufferManager;
        CoreTextureManager*      textureManager          = m_textureManager;
        CoreComputeManager*      computeManager          = m_computeManager;
        CoreRenderPipeline*      renderPipeline          = m_renderPipeline;
        InternalResourceManager* internalResourceManager = m_internalResourceManager;
        CoreNVSPageManager*      pageManager             = m_pageManager;

        return Initialize(
            backend,
            bufferManager,
            textureManager,
            computeManager,
            renderPipeline,
            internalResourceManager,
            pageManager,
            newConfig);
    }

    void CoreNVSFeedbackManager::OnCameraDestroyed(Camera* camera)
    {
        if (!camera)
            return;

        auto it = m_cameraSlots.find(camera);
        if (it != m_cameraSlots.end())
        {
            ReleaseSlotResources(it->second);
            m_cameraSlots.erase(it);
        }
    }

    BOOL CoreNVSFeedbackManager::CreateDummyDepthTexture()
    {
        if (!m_textureManager)
            return false;

        FLOAT32 clearPixel = 1.0f;
        TextureDesc texDesc{};
        texDesc.dataSource      = TextureDataSource::MEMORY;
        texDesc.initialData     = reinterpret_cast<const BYTE*>(&clearPixel);
        texDesc.initialDataSize = sizeof(FLOAT32);
        texDesc.width           = 1u;
        texDesc.height          = 1u;
        texDesc.depth           = 1u;
        texDesc.mipLevels       = 1u;
        texDesc.arrayLayers     = 1u;
        texDesc.format          = TextureFormat::R32_FLOAT;
        texDesc.type            = TextureType::TEXTURE_2D;
        texDesc.usage           = TextureUsage::SAMPLED;

        m_dummyDepthTexture = m_textureManager->CreateTexture(texDesc);
        if (!m_dummyDepthTexture || m_dummyDepthTexture->GetNativeHandle() == 0)
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create 1x1 fallback depth texture.";
            return false;
        }

        if (m_bufferManager)
        {
            NVSResidencyTelemetry zeroResTel{};
            GPUBufferDesc resTelDesc{};
            resTelDesc.elementCount = 1u;
            resTelDesc.stride       = sizeof(NVSResidencyTelemetry);
            resTelDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
            resTelDesc.initialData  = &zeroResTel;
            resTelDesc.debugName    = "NVS_FeedbackDummyResidencyTelemetry";
            m_dummyResidencyTelemetryBuffer = m_bufferManager->CreateGPUBuffer(resTelDesc);
            if (!m_dummyResidencyTelemetryBuffer || !m_dummyResidencyTelemetryBuffer->IsValid())
            {
                m_lastError = "CoreNVSFeedbackManager: failed to create fallback residency telemetry buffer.";
                return false;
            }

            std::vector<NVSClipmapLevelGPUData> zeroLevels(kNVSClipmapTableMaxEntries);
            GPUBufferDesc clipDesc{};
            clipDesc.elementCount = kNVSClipmapTableMaxEntries;
            clipDesc.stride       = sizeof(NVSClipmapLevelGPUData);
            clipDesc.usage        = GPUBufferUsage::STRUCTURED_SRV;
            clipDesc.initialData  = zeroLevels.data();
            clipDesc.debugName    = "NVS_FeedbackDummyClipmapLevels";
            m_dummyClipmapLevelsBuffer = m_bufferManager->CreateGPUBuffer(clipDesc);
            if (!m_dummyClipmapLevelsBuffer || !m_dummyClipmapLevelsBuffer->IsValid())
            {
                m_lastError = "CoreNVSFeedbackManager: failed to create fallback clipmap levels buffer.";
                return false;
            }
        }
        return true;
    }

    BOOL CoreNVSFeedbackManager::AllocateSlotResources(
        CameraFeedbackSlot& slot,
        UINT32              budgetCapacity,
        const std::string&  namePrefix)
    {
        ReleaseSlotResources(slot);

        if (!m_bufferManager || budgetCapacity == 0u)
            return false;

        const UINT32 allocBudget = (std::max)(
            (std::max)(budgetCapacity, m_config.maxCompactedRequestsCapacity),
            kNVSDefaultMaxPageRequests);

        // 1. Page Bitmask Buffer (2048 x UINT32 = 8 KB)
        std::vector<UINT32> zeroBitmask(kNVSFeedbackBitmaskWordCount, 0u);
        GPUBufferDesc bitmaskDesc{};
        bitmaskDesc.elementCount = kNVSFeedbackBitmaskWordCount;
        bitmaskDesc.stride       = sizeof(UINT32);
        bitmaskDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;
        bitmaskDesc.initialData  = zeroBitmask.data();
        bitmaskDesc.debugName    = namePrefix + "_PageBitmask";

        slot.pageBitmaskBuffer = m_bufferManager->CreateGPUBuffer(bitmaskDesc);
        if (!slot.pageBitmaskBuffer || !slot.pageBitmaskBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create PageBitmask GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 2. Compacted Request Buffer (allocBudget x 4 bytes)
        std::vector<NVSVirtualPageRequest> zeroReqs(allocBudget);
        GPUBufferDesc reqDesc{};
        reqDesc.elementCount = allocBudget;
        reqDesc.stride       = sizeof(NVSVirtualPageRequest);
        reqDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;
        reqDesc.initialData  = zeroReqs.data();
        reqDesc.debugName    = namePrefix + "_CompactedRequests";

        slot.compactedRequestBuffer = m_bufferManager->CreateGPUBuffer(reqDesc);
        if (!slot.compactedRequestBuffer || !slot.compactedRequestBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create CompactedRequests GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 3. Feedback Telemetry Buffer (1 x 64 bytes)
        NVSFeedbackTelemetry zeroTel{};
        GPUBufferDesc telDesc{};
        telDesc.elementCount = 1u;
        telDesc.stride       = sizeof(NVSFeedbackTelemetry);
        telDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;
        telDesc.initialData  = &zeroTel;
        telDesc.debugName    = namePrefix + "_FeedbackTelemetry";

        slot.feedbackTelemetryBuffer = m_bufferManager->CreateGPUBuffer(telDesc);
        if (!slot.feedbackTelemetryBuffer || !slot.feedbackTelemetryBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create FeedbackTelemetry GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 4. Sparse Global Request Hash Table Buffer (power-of-two capacity >= 2x budget)
        UINT32 initHashCap = kNVSSparseFeedbackHashCapacity;
        while (initHashCap < (allocBudget * 2u) && initHashCap < 262144u)
        {
            initHashCap *= 2u;
        }
        const UINT32 initMacroTiles   = 2048u;
        const UINT32 initViewsPerTile = 1024u;
        const UINT32 initVisViewsCap  = 8192u;

        GPUBufferDesc hashDesc{};
        hashDesc.elementCount = initHashCap;
        hashDesc.stride       = sizeof(NVSVirtualPageRequest);
        hashDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        hashDesc.debugName    = namePrefix + "_RequestHashTable";
        slot.requestHashTableBuffer = m_bufferManager->CreateGPUBuffer(hashDesc);
        if (!slot.requestHashTableBuffer || !slot.requestHashTableBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create RequestHashTable GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 5. 32x32 Macro-Tile World AABB Bounds Buffer ((maxMacroTiles + 1) x 32 bytes; [maxMacroTiles] = global screen AABB)
        GPUBufferDesc mtbDesc{};
        mtbDesc.elementCount = initMacroTiles + 1u;
        mtbDesc.stride       = 32u;
        mtbDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        mtbDesc.debugName    = namePrefix + "_MacroTileBounds";
        slot.macroTileBoundsBuffer = m_bufferManager->CreateGPUBuffer(mtbDesc);
        if (!slot.macroTileBoundsBuffer || !slot.macroTileBoundsBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create MacroTileBounds GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 6. Per-Macro-Tile Culled View Indices Buffer (maxMacroTiles * 512 x 4 bytes)
        GPUBufferDesc mtvDesc{};
        mtvDesc.elementCount = initMacroTiles * initViewsPerTile;
        mtvDesc.stride       = sizeof(UINT32);
        mtvDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        mtvDesc.debugName    = namePrefix + "_MacroTileViewIndices";
        slot.macroTileViewIndicesBuffer = m_bufferManager->CreateGPUBuffer(mtvDesc);
        if (!slot.macroTileViewIndicesBuffer || !slot.macroTileViewIndicesBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create MacroTileViewIndices GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        // 7. 4-Bucket Priority Counters + Global Visible View List Buffer ((16 + maxVisibleViewsCapacity) x 4 bytes)
        GPUBufferDesc bktDesc{};
        bktDesc.elementCount = 16u + initVisViewsCap;
        bktDesc.stride       = sizeof(UINT32);
        bktDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        bktDesc.debugName    = namePrefix + "_BucketAndVisibleViews";
        slot.bucketCountersBuffer = m_bufferManager->CreateGPUBuffer(bktDesc);
        if (!slot.bucketCountersBuffer || !slot.bucketCountersBuffer->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: failed to create BucketAndVisibleViews GPUBuffer.";
            ReleaseSlotResources(slot);
            return false;
        }

        slot.budgetCapacity          = allocBudget;
        slot.hashCapacity            = initHashCap;
        slot.maxMacroTiles           = initMacroTiles;
        slot.maxViewsPerMacroTile    = initViewsPerTile;
        slot.maxVisibleViewsCapacity = initVisViewsCap;
        return true;
    }

    BOOL CoreNVSFeedbackManager::EnsureSlotMacroTileAndViewCapacity(
        CameraFeedbackSlot& slot,
        UINT32              viewportW,
        UINT32              viewportH,
        UINT32              activeViews)
    {
        if (!m_bufferManager)
        {
            return false;
        }
        const UINT32 macroX      = (viewportW + 31u) / 32u;
        const UINT32 macroY      = (viewportH + 31u) / 32u;
        const UINT32 neededMacro = macroX * macroY;
        if (neededMacro > slot.maxMacroTiles && slot.macroTileBoundsBuffer && slot.macroTileViewIndicesBuffer)
        {
            UINT32 newMacro = std::max(slot.maxMacroTiles, 4096u);
            while (newMacro < neededMacro)
            {
                newMacro *= 2u;
            }
            if (!m_bufferManager->ResizeGPUBuffer(slot.macroTileBoundsBuffer, newMacro + 1u, nullptr) ||
                !m_bufferManager->ResizeGPUBuffer(slot.macroTileViewIndicesBuffer, newMacro * slot.maxViewsPerMacroTile, nullptr))
            {
                m_lastError = "CoreNVSFeedbackManager: failed to resize macro-tile culling buffers.";
                return false;
            }
            slot.maxMacroTiles = newMacro;
        }

        if (activeViews > slot.maxVisibleViewsCapacity && slot.bucketCountersBuffer)
        {
            UINT32 newVisCap = std::max(slot.maxVisibleViewsCapacity, 8192u);
            while (newVisCap < activeViews)
            {
                newVisCap *= 2u;
            }
            if (!m_bufferManager->ResizeGPUBuffer(slot.bucketCountersBuffer, 16u + newVisCap, nullptr))
            {
                m_lastError = "CoreNVSFeedbackManager: failed to resize visible view list buffer.";
                return false;
            }
            slot.maxVisibleViewsCapacity = newVisCap;
        }
        return true;
    }

    void CoreNVSFeedbackManager::ReleaseSlotResources(CameraFeedbackSlot& slot)
    {
        if (m_bufferManager)
        {
            if (slot.bucketCountersBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.bucketCountersBuffer);
                slot.bucketCountersBuffer = nullptr;
            }
            if (slot.macroTileViewIndicesBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.macroTileViewIndicesBuffer);
                slot.macroTileViewIndicesBuffer = nullptr;
            }
            if (slot.macroTileBoundsBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.macroTileBoundsBuffer);
                slot.macroTileBoundsBuffer = nullptr;
            }
            if (slot.requestHashTableBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.requestHashTableBuffer);
                slot.requestHashTableBuffer = nullptr;
            }
            if (slot.feedbackTelemetryBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.feedbackTelemetryBuffer);
                slot.feedbackTelemetryBuffer = nullptr;
            }
            if (slot.compactedRequestBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.compactedRequestBuffer);
                slot.compactedRequestBuffer = nullptr;
            }
            if (slot.pageBitmaskBuffer)
            {
                m_bufferManager->DestroyGPUBuffer(slot.pageBitmaskBuffer);
                slot.pageBitmaskBuffer = nullptr;
            }
        }
        slot.budgetCapacity          = 0u;
        slot.hashCapacity            = 0u;
        slot.maxMacroTiles           = 0u;
        slot.maxViewsPerMacroTile    = 0u;
        slot.maxVisibleViewsCapacity = 0u;
    }

    void CoreNVSFeedbackManager::ReleaseAllResources()
    {
        for (auto& kv : m_cameraSlots)
        {
            ReleaseSlotResources(kv.second);
        }
        m_cameraSlots.clear();
        ReleaseSlotResources(m_defaultSlot);

        if (m_bufferManager && m_dummyClipmapLevelsBuffer)
        {
            m_bufferManager->DestroyGPUBuffer(m_dummyClipmapLevelsBuffer);
            m_dummyClipmapLevelsBuffer = nullptr;
        }

        if (m_bufferManager && m_dummyResidencyTelemetryBuffer)
        {
            m_bufferManager->DestroyGPUBuffer(m_dummyResidencyTelemetryBuffer);
            m_dummyResidencyTelemetryBuffer = nullptr;
        }

        if (m_textureManager && m_dummyDepthTexture)
        {
            m_textureManager->DestroyTexture(m_dummyDepthTexture);
            m_dummyDepthTexture = nullptr;
        }
    }

    CoreNVSFeedbackManager::CameraFeedbackSlot* CoreNVSFeedbackManager::AcquireSlot(Camera* camera)
    {
        if (!m_initialized)
            return nullptr;
        if (!camera)
            return &m_defaultSlot;

        auto it = m_cameraSlots.find(camera);
        if (it != m_cameraSlots.end())
            return &it->second;

        CameraFeedbackSlot newSlot{};
        const std::string prefix = "NVS_FeedbackCam_" + std::to_string(reinterpret_cast<uintptr_t>(camera));
        if (!AllocateSlotResources(newSlot, m_config.maxRequestBudget, prefix))
            return nullptr;

        auto inserted = m_cameraSlots.emplace(camera, newSlot);
        return &inserted.first->second;
    }

    const CoreNVSFeedbackManager::CameraFeedbackSlot* CoreNVSFeedbackManager::FindSlot(Camera* camera) const
    {
        if (!m_initialized)
            return nullptr;
        if (!camera)
            return &m_defaultSlot;

        auto it = m_cameraSlots.find(camera);
        if (it != m_cameraSlots.end())
            return &it->second;
        return &m_defaultSlot;
    }

    GPUBuffer* CoreNVSFeedbackManager::GetPageBitmaskBuffer(Camera* camera)
    {
        CameraFeedbackSlot* slot = AcquireSlot(camera);
        return slot ? slot->pageBitmaskBuffer : nullptr;
    }

    GPUBuffer* CoreNVSFeedbackManager::GetCompactedRequestBuffer(Camera* camera)
    {
        CameraFeedbackSlot* slot = AcquireSlot(camera);
        return slot ? slot->compactedRequestBuffer : nullptr;
    }

    GPUBuffer* CoreNVSFeedbackManager::GetFeedbackTelemetryBuffer(Camera* camera)
    {
        CameraFeedbackSlot* slot = AcquireSlot(camera);
        return slot ? slot->feedbackTelemetryBuffer : nullptr;
    }

    const GPUBuffer* CoreNVSFeedbackManager::GetPageBitmaskBuffer(Camera* camera) const
    {
        const CameraFeedbackSlot* slot = FindSlot(camera);
        return slot ? slot->pageBitmaskBuffer : nullptr;
    }

    const GPUBuffer* CoreNVSFeedbackManager::GetCompactedRequestBuffer(Camera* camera) const
    {
        const CameraFeedbackSlot* slot = FindSlot(camera);
        return slot ? slot->compactedRequestBuffer : nullptr;
    }

    const GPUBuffer* CoreNVSFeedbackManager::GetFeedbackTelemetryBuffer(Camera* camera) const
    {
        const CameraFeedbackSlot* slot = FindSlot(camera);
        return slot ? slot->feedbackTelemetryBuffer : nullptr;
    }

    UINT64 CoreNVSFeedbackManager::GetMetadataVramBytes() const noexcept
    {
        if (!m_initialized)
        {
            return 0ull;
        }
        auto sumSlot = [](const CameraFeedbackSlot& s) noexcept -> UINT64 {
            UINT64 b = 0ull;
            if (s.pageBitmaskBuffer)          b += s.pageBitmaskBuffer->GetByteSize();
            if (s.compactedRequestBuffer)     b += s.compactedRequestBuffer->GetByteSize();
            if (s.feedbackTelemetryBuffer)    b += s.feedbackTelemetryBuffer->GetByteSize();
            if (s.requestHashTableBuffer)     b += s.requestHashTableBuffer->GetByteSize();
            if (s.macroTileBoundsBuffer)      b += s.macroTileBoundsBuffer->GetByteSize();
            if (s.macroTileViewIndicesBuffer) b += s.macroTileViewIndicesBuffer->GetByteSize();
            if (s.bucketCountersBuffer)       b += s.bucketCountersBuffer->GetByteSize();
            return b;
        };
        UINT64 total = sumSlot(m_defaultSlot);
        for (const auto& kv : m_cameraSlots)
        {
            total += sumSlot(kv.second);
        }
        return total;
    }

    static Vec4 ComputeEffectiveScreenPixelWorldScale(const NVSFeedbackPassDesc& passDesc, const Vec3& /*camPos*/)
    {
        if (passDesc.screenPixelWorldScale.x != 0.0f ||
            passDesc.screenPixelWorldScale.y != 0.0f ||
            passDesc.screenPixelWorldScale.z != 0.0f ||
            passDesc.screenPixelWorldScale.w != 0.0f)
        {
            return passDesc.screenPixelWorldScale;
        }
        const FLOAT32 vpW = static_cast<FLOAT32>((std::max)(passDesc.viewportWidth, 1u));
        const FLOAT32 vpH = static_cast<FLOAT32>((std::max)(passDesc.viewportHeight, 1u));
        const Mat4&   ivp = passDesc.invViewProj;

        auto unproject = [&](FLOAT32 ndcX, FLOAT32 ndcY, FLOAT32 ndcZ) -> Vec3 {
            const Vec4 clip(ndcX, ndcY, ndcZ, 1.0f);
            const Vec4 wp = ivp * clip;
            const FLOAT32 invW = (std::fabs(wp.w) > 1e-7f) ? (1.0f / wp.w) : 1.0f;
            return Vec3(wp.x * invW, wp.y * invW, wp.z * invW);
        };

        const BOOL isOrtho =
            (std::fabs(ivp.m[3]) < 1e-6f) &&
            (std::fabs(ivp.m[7]) < 1e-6f) &&
            (std::fabs(ivp.m[11]) < 1e-6f) &&
            (std::fabs(ivp.m[15] - 1.0f) < 1e-4f);

        const Vec3 pLeft1   = unproject(-1.0f,  0.0f, 0.1f);
        const Vec3 pRight1  = unproject(+1.0f,  0.0f, 0.1f);
        const Vec3 pBottom1 = unproject( 0.0f, -1.0f, 0.1f);
        const Vec3 pTop1    = unproject( 0.0f, +1.0f, 0.1f);
        const Vec3 pCenter1 = unproject( 0.0f,  0.0f, 0.1f);

        const Vec3    spanX1  = pRight1 - pLeft1;
        const Vec3    spanY1  = pTop1 - pBottom1;
        const FLOAT32 worldW1 = std::sqrt(spanX1.Dot(spanX1));
        const FLOAT32 worldH1 = std::sqrt(spanY1.Dot(spanY1));

        if (isOrtho)
        {
            const FLOAT32 orthoPx = (std::max)(worldW1 / vpW, worldH1 / vpH);
            return Vec4(0.0f, (std::max)(orthoPx, 1e-5f), 0.0f, 1.0f);
        }

        // Exact 2-plane perspective frustum slope (2 * tan(fov/2)), independent of camera world position
        // (including exact origin (0,0,0)), Standard-Z / Reversed-Z, and infinite far planes.
        const Vec3 pLeft2   = unproject(-1.0f,  0.0f, 0.9f);
        const Vec3 pRight2  = unproject(+1.0f,  0.0f, 0.9f);
        const Vec3 pBottom2 = unproject( 0.0f, -1.0f, 0.9f);
        const Vec3 pTop2    = unproject( 0.0f, +1.0f, 0.9f);
        const Vec3 pCenter2 = unproject( 0.0f,  0.0f, 0.9f);

        const Vec3    spanX2  = pRight2 - pLeft2;
        const Vec3    spanY2  = pTop2 - pBottom2;
        const FLOAT32 worldW2 = std::sqrt(spanX2.Dot(spanX2));
        const FLOAT32 worldH2 = std::sqrt(spanY2.Dot(spanY2));
        const Vec3    dCenter = pCenter2 - pCenter1;
        const FLOAT32 dZ      = std::sqrt(dCenter.Dot(dCenter));
        if (dZ <= 1e-5f || !std::isfinite(dZ))
        {
            return Vec4((std::max)(2.0f / vpW, 2.0f / vpH), 0.0f, 0.0f, 1.0f);
        }
        const FLOAT32 kX = std::fabs(worldW2 - worldW1) / dZ;
        const FLOAT32 kY = std::fabs(worldH2 - worldH1) / dZ;
        return Vec4(
            (std::max)(kX / vpW, kY / vpH),
            0.0f,
            0.0f,
            1.0f);
    }

    static void ComputeFeedbackDispatch2D(UINT32 totalGroups, UINT32& outGroupsX, UINT32& outGroupsY) noexcept
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

    static void PopulateFeedbackConfigOnKernel(
        ComputeKernel*             kernel,
        NVSFeedbackDispatchMode    mode,
        UINT32                     cullSubMode,
        const UINT32               hashAndTileParams[4],
        const NVSFeedbackPassDesc& passDesc,
        UINT32                     effectiveBudget,
        UINT32                     compactedBufferCapacity)
    {
        const NVSClipmapHierarchyState& h = passDesc.clipmapHierarchy;
        const Vec4 levelExtents(
            h.levelWorldExtents[0],
            h.levelWorldExtents[1],
            h.levelWorldExtents[2],
            h.levelWorldExtents[3]);
        const Vec3 camPos = (passDesc.cameraPosition.Dot(passDesc.cameraPosition) > 0.0f)
            ? passDesc.cameraPosition
            : passDesc.cameraWorldPos;
        const UINT32 depthMode = (passDesc.depthCompareMode != NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE)
            ? static_cast<UINT32>(passDesc.depthCompareMode)
            : static_cast<UINT32>(passDesc.depthMode);
        const Vec3 effectiveRight = h.lightValid ? h.lightRight : Vec3(0.0f, 0.0f, 0.0f);
        const Vec3 effectiveUp    = h.lightValid ? h.lightUp    : Vec3(0.0f, 0.0f, 0.0f);
        const Vec3 effectiveDir   = h.lightValid ? h.lightDir   : Vec3(0.0f, 0.0f, 0.0f);
        const Vec4 pixelScale     = ComputeEffectiveScreenPixelWorldScale(passDesc, camPos);

        kernel->SetUInt  ("NVSFeedbackParams", "Config", "dispatchMode",            static_cast<UINT32>(mode));
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "viewportOffsetX",         passDesc.viewportOffsetX);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "viewportOffsetY",         passDesc.viewportOffsetY);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "viewportWidth",           passDesc.viewportWidth);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "viewportHeight",          passDesc.viewportHeight);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "depthMode",               depthMode);
        kernel->SetFloat ("NVSFeedbackParams", "Config", "clearDepth",              passDesc.clearDepth);
        kernel->SetFloat ("NVSFeedbackParams", "Config", "depthEpsilon",            passDesc.depthEpsilon);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "lightId",                 h.lightId);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "activeLevelCount",        h.activeLevelCount);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "pagesPerAxis",            h.pagesPerAxis);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "footprintMode",           static_cast<UINT32>(passDesc.footprintMode));
        kernel->SetFloat ("NVSFeedbackParams", "Config", "footprintBorderMargin",   passDesc.footprintBorderMargin);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "maxRequestBudget",        effectiveBudget);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "compactedBufferCapacity", compactedBufferCapacity);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "disableGroupSharedDedup", passDesc.disableGroupSharedDedup ? 1u : 0u);
        kernel->SetVec3  ("NVSFeedbackParams", "Config", "cameraWorldPos",          camPos);
        kernel->SetFloat ("NVSFeedbackParams", "Config", "maxShadowDistance",       h.maxShadowDistance);
        kernel->SetVec3  ("NVSFeedbackParams", "Config", "lightRight",              effectiveRight);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "multiLightViewMode",      passDesc.multiLightViewMode ? 1u : 0u);
        kernel->SetVec3  ("NVSFeedbackParams", "Config", "lightUp",                 effectiveUp);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "activeViewTableCount",    passDesc.activeViewTableCount);
        kernel->SetVec3  ("NVSFeedbackParams", "Config", "lightDir",                effectiveDir);
        kernel->SetUInt  ("NVSFeedbackParams", "Config", "cullSubMode",             cullSubMode);
        kernel->SetVec4  ("NVSFeedbackParams", "Config", "levelWorldExtents",       levelExtents);
        kernel->SetStruct("snappedOriginPageX", h.snappedOriginPageX, sizeof(INT32) * 4u);
        kernel->SetStruct("snappedOriginPageY", h.snappedOriginPageY, sizeof(INT32) * 4u);
        kernel->SetVec4  ("NVSFeedbackParams", "Config", "screenPixelWorldScale",   pixelScale);
        kernel->SetStruct("hashAndTileParams",  hashAndTileParams, sizeof(UINT32) * 4u);
        kernel->SetMat4  ("NVSFeedbackParams", "Config", "invViewProj",             passDesc.invViewProj);
    }

    static void UnbindFeedbackKernelResources(ComputeKernel* kernel)
    {
        if (!kernel) return;
        kernel->SetTexture  ("NVSFeedbackParams", "SceneDepth",              nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "InClipmapLevels",         nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutPageBitmask",          nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutCompactedRequests",    nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutFeedbackTelemetry",    nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutResidencyTelemetry",   nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutRequestHashTable",     nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutMacroTileBounds",      nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutMacroTileViewIndices", nullptr);
        kernel->SetGPUBuffer("NVSFeedbackParams", "OutBucketCounters",       nullptr);
    }

    BOOL CoreNVSFeedbackManager::BuildFeedbackStageDispatchInternal(
        NVSFeedbackDispatchMode    mode,
        UINT32                     cullSubMode,
        BOOL                       enableMacroTileBinning,
        UINT32                     groupsX,
        UINT32                     groupsY,
        const NVSFeedbackPassDesc& passDesc,
        ComputeDispatchCommand&    outCmd,
        GPUBuffer*                 overrideCompactedBuf,
        GPUBuffer*                 residencyTelemetryBuf,
        Camera*                    camera)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_internalResourceManager || !m_renderPipeline)
            return false;

        CameraFeedbackSlot* slot = AcquireSlot(camera);
        if (!slot || !slot->pageBitmaskBuffer || !slot->compactedRequestBuffer || !slot->feedbackTelemetryBuffer ||
            !slot->requestHashTableBuffer || !slot->macroTileBoundsBuffer || !slot->macroTileViewIndicesBuffer || !slot->bucketCountersBuffer)
        {
            return false;
        }

        GPUBuffer* compactedBuf = overrideCompactedBuf ? overrideCompactedBuf : slot->compactedRequestBuffer;
        if (!compactedBuf || !compactedBuf->IsValid() || !compactedBuf->HasUsage(GPUBufferUsage::STRUCTURED_UAV))
            return false;

        ComputeKernel* kernel = m_internalResourceManager->GetComputeKernel(InternalComputeKernelId::NVS_SCREEN_SPACE_FEEDBACK);
        if (!kernel || !kernel->IsValid())
        {
            m_lastError = "CoreNVSFeedbackManager: Runtime.NVSScreenSpaceFeedback kernel unavailable.";
            return false;
        }

        const BOOL needsRealDepth =
            (mode == NVS_FEEDBACK_DISPATCH_MARK_SCREEN_DEPTH) ||
            (mode == NVS_FEEDBACK_DISPATCH_REDUCE_MACRO_TILE_BOUNDS);
        Texture* boundDepth = needsRealDepth ? passDesc.sceneDepthTexture : m_dummyDepthTexture;
        if (!boundDepth || boundDepth->GetNativeHandle() == 0)
        {
            m_lastError = "CoreNVSFeedbackManager: valid SceneDepth Texture is required.";
            return false;
        }

        GPUBuffer* resTel = residencyTelemetryBuf;
        if (!resTel && m_pageManager && m_pageManager->IsInitialized() &&
            (mode == NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY || mode == NVS_FEEDBACK_DISPATCH_COMPACT_PAGE_REQUESTS))
        {
            resTel = m_pageManager->GetTelemetryBuffer();
        }
        if (!resTel)
        {
            resTel = m_dummyResidencyTelemetryBuffer;
        }

        const UINT32 requestedBudget = passDesc.maxRequestBudget > 0u ? passDesc.maxRequestBudget : m_config.maxRequestBudget;
        const UINT32 effectiveBudget = (std::min)(requestedBudget, compactedBuf->GetElementCount());

        const UINT32 hashAndTileParams[4] = {
            slot->hashCapacity,
            enableMacroTileBinning ? slot->maxMacroTiles : 0u,
            slot->maxViewsPerMacroTile,
            slot->maxVisibleViewsCapacity
        };

        GPUBuffer* boundViewTable = passDesc.lightViewTableBuffer;
        if (!boundViewTable && m_pageManager && m_pageManager->IsInitialized())
        {
            m_pageManager->FlushViewTableToGPU(camera);
            boundViewTable = m_pageManager->GetClipmapLevelTableBuffer(camera);
        }
        if (!boundViewTable)
        {
            boundViewTable = m_dummyClipmapLevelsBuffer;
        }

        NVSFeedbackPassDesc effectivePassDesc = passDesc;
        if (boundViewTable && effectivePassDesc.activeViewTableCount > boundViewTable->GetElementCount())
        {
            effectivePassDesc.activeViewTableCount = boundViewTable->GetElementCount();
        }

        PopulateFeedbackConfigOnKernel(
            kernel,
            mode,
            cullSubMode,
            hashAndTileParams,
            effectivePassDesc,
            effectiveBudget,
            compactedBuf->GetElementCount());

        const GPUResourceAccess bitmaskAccess = (mode == NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY)
            ? GPUResourceAccess::UNORDERED_ACCESS_WRITE
            : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        const GPUResourceAccess fbTelAccess = (mode == NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY)
            ? GPUResourceAccess::UNORDERED_ACCESS_WRITE
            : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        const GPUResourceAccess hashAccess = (mode == NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY)
            ? GPUResourceAccess::UNORDERED_ACCESS_WRITE
            : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;

        kernel->SetTexture("NVSFeedbackParams", "SceneDepth", boundDepth);
        if (!boundViewTable || !boundViewTable->IsValid() ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "InClipmapLevels",         boundViewTable,                   GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutPageBitmask",          slot->pageBitmaskBuffer,          bitmaskAccess) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutCompactedRequests",    compactedBuf,                     GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutFeedbackTelemetry",    slot->feedbackTelemetryBuffer,    fbTelAccess) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutRequestHashTable",     slot->requestHashTableBuffer,     hashAccess) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutMacroTileBounds",      slot->macroTileBoundsBuffer,      GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutMacroTileViewIndices", slot->macroTileViewIndicesBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE) ||
            !kernel->SetGPUBuffer("NVSFeedbackParams", "OutBucketCounters",       slot->bucketCountersBuffer,       GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE))
        {
            UnbindFeedbackKernelResources(kernel);
            return false;
        }

        if (resTel && resTel->IsValid())
        {
            kernel->SetGPUBuffer("NVSFeedbackParams", "OutResidencyTelemetry", resTel, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        }

        kernel->SetThreadGroupCount((std::max)(groupsX, 1u), (std::max)(groupsY, 1u), 1u);

        FrameView emptyView{};
        const BOOL built = m_renderPipeline->BuildComputeDispatchCommand(kernel, emptyView, outCmd);
        if (built)
        {
            outCmd.profileGroup = ComputeDispatchProfileGroup::FEEDBACK;
        }
        UnbindFeedbackKernelResources(kernel);
        return built;
    }

    BOOL CoreNVSFeedbackManager::BuildFeedbackResetDispatch(
        const NVSFeedbackPassDesc& passDesc,
        ComputeDispatchCommand&    outCmd,
        GPUBuffer*                 residencyTelemetryBuf,
        Camera*                    camera)
    {
        CameraFeedbackSlot* slot = AcquireSlot(camera);
        if (!slot) return false;
        UINT32 groupsX = 1u, groupsY = 1u;
        if (passDesc.multiLightViewMode)
        {
            ComputeFeedbackDispatch2D((slot->hashCapacity + 63u) / 64u, groupsX, groupsY);
        }
        return BuildFeedbackStageDispatchInternal(
            NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY,
            0u,
            passDesc.multiLightViewMode,
            groupsX,
            groupsY,
            passDesc,
            outCmd,
            nullptr,
            residencyTelemetryBuf,
            camera);
    }

    BOOL CoreNVSFeedbackManager::BuildFeedbackMarkDispatch(
        const NVSFeedbackPassDesc& passDesc,
        ComputeDispatchCommand&    outCmd,
        Camera*                    camera)
    {
        if (!passDesc.sceneDepthTexture || passDesc.sceneDepthTexture->GetNativeHandle() == 0)
        {
            m_lastError = "CoreNVSFeedbackManager::BuildFeedbackMarkDispatch: valid SceneDepth Texture is required.";
            return false;
        }
        if (passDesc.viewportWidth == 0u || passDesc.viewportHeight == 0u)
        {
            m_lastError = "CoreNVSFeedbackManager::BuildFeedbackMarkDispatch: viewportWidth and viewportHeight must be > 0.";
            return false;
        }
        const UINT32 tilesX = (passDesc.viewportWidth  + kNVSFeedbackTileSize - 1u) / kNVSFeedbackTileSize;
        const UINT32 tilesY = (passDesc.viewportHeight + kNVSFeedbackTileSize - 1u) / kNVSFeedbackTileSize;
        // Standalone BuildFeedbackMarkDispatch disables macro-tile list lookup (enableMacroTileBinning = false)
        // because Mode 3/4 were not chained before it; BuildFeedbackPipelineDispatches enables full macro-tile culling.
        return BuildFeedbackStageDispatchInternal(
            NVS_FEEDBACK_DISPATCH_MARK_SCREEN_DEPTH,
            0u,
            false,
            tilesX,
            tilesY,
            passDesc,
            outCmd,
            nullptr,
            m_dummyResidencyTelemetryBuffer,
            camera);
    }

    BOOL CoreNVSFeedbackManager::BuildFeedbackCompactDispatch(
        const NVSFeedbackPassDesc& passDesc,
        ComputeDispatchCommand&    outCmd,
        GPUBuffer*                 overrideCompactedBuf,
        GPUBuffer*                 residencyTelemetryBuf,
        Camera*                    camera)
    {
        return BuildFeedbackStageDispatchInternal(
            NVS_FEEDBACK_DISPATCH_COMPACT_REQUESTS_AND_TELEMETRY,
            0u,
            false,
            1u,
            1u,
            passDesc,
            outCmd,
            overrideCompactedBuf,
            residencyTelemetryBuf,
            camera);
    }

    BOOL CoreNVSFeedbackManager::BuildFeedbackPipelineDispatches(
        const NVSFeedbackPassDesc&           passDesc,
        std::vector<ComputeDispatchCommand>& outDispatches,
        GPUBuffer*                           overrideCompactedBuf,
        GPUBuffer*                           residencyTelemetryBuf,
        Camera*                              camera)
    {
        if (!m_initialized)
        {
            return false;
        }

        CameraFeedbackSlot* slot = AcquireSlot(camera);
        if (!slot)
        {
            return false;
        }

        const BOOL hasDepthWork =
            passDesc.sceneDepthTexture &&
            passDesc.sceneDepthTexture->GetNativeHandle() != 0 &&
            passDesc.viewportWidth > 0u &&
            passDesc.viewportHeight > 0u;

        if (!passDesc.multiLightViewMode)
        {
            if (passDesc.resetBeforeMark)
            {
                ComputeDispatchCommand resetCmd{};
                if (!BuildFeedbackResetDispatch(passDesc, resetCmd, residencyTelemetryBuf, camera))
                    return false;
                outDispatches.push_back(std::move(resetCmd));
            }
            if (hasDepthWork)
            {
                ComputeDispatchCommand markCmd{};
                if (!BuildFeedbackMarkDispatch(passDesc, markCmd, camera))
                    return false;
                outDispatches.push_back(std::move(markCmd));
            }
            if (passDesc.executeCompaction)
            {
                ComputeDispatchCommand compactCmd{};
                if (!BuildFeedbackCompactDispatch(passDesc, compactCmd, overrideCompactedBuf, residencyTelemetryBuf, camera))
                    return false;
                outDispatches.push_back(std::move(compactCmd));
            }
            return true;
        }

        if (!EnsureSlotMacroTileAndViewCapacity(*slot, passDesc.viewportWidth, passDesc.viewportHeight, passDesc.activeViewTableCount))
        {
            return false;
        }

        const UINT32 hashGroups = (slot->hashCapacity + 63u) / 64u;
        UINT32 hashGX = 1u, hashGY = 1u;
        ComputeFeedbackDispatch2D(hashGroups, hashGX, hashGY);

        if (passDesc.resetBeforeMark)
        {
            ComputeDispatchCommand resetCmd{};
            if (!BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY,
                    0u,
                    true,
                    hashGX,
                    hashGY,
                    passDesc,
                    resetCmd,
                    overrideCompactedBuf,
                    residencyTelemetryBuf,
                    camera))
            {
                return false;
            }
            outDispatches.push_back(std::move(resetCmd));
        }

        if (hasDepthWork && passDesc.activeViewTableCount > 0u)
        {
            const UINT32 macroTilesX = (passDesc.viewportWidth  + 31u) / 32u;
            const UINT32 macroTilesY = (passDesc.viewportHeight + 31u) / 32u;
            const UINT32 totalMacros = (std::max)(macroTilesX * macroTilesY, 1u);
            UINT32 macroGX = 1u, macroGY = 1u;
            ComputeFeedbackDispatch2D(totalMacros, macroGX, macroGY);

            UINT32 viewFilterGX = 1u, viewFilterGY = 1u;
            ComputeFeedbackDispatch2D((passDesc.activeViewTableCount + 63u) / 64u, viewFilterGX, viewFilterGY);

            const UINT32 tiles8X = (passDesc.viewportWidth  + kNVSFeedbackTileSize - 1u) / kNVSFeedbackTileSize;
            const UINT32 tiles8Y = (passDesc.viewportHeight + kNVSFeedbackTileSize - 1u) / kNVSFeedbackTileSize;

            ComputeDispatchCommand reduceCmd{};
            ComputeDispatchCommand cullGlobalCmd{};
            ComputeDispatchCommand cullTileCmd{};
            ComputeDispatchCommand markCmd{};

            if (!BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_REDUCE_MACRO_TILE_BOUNDS,
                    0u,
                    true,
                    macroGX,
                    macroGY,
                    passDesc,
                    reduceCmd,
                    overrideCompactedBuf,
                    m_dummyResidencyTelemetryBuffer,
                    camera) ||
                !BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_CULL_VIEWS_TO_MACRO_TILES,
                    0u,
                    true,
                    viewFilterGX,
                    viewFilterGY,
                    passDesc,
                    cullGlobalCmd,
                    overrideCompactedBuf,
                    m_dummyResidencyTelemetryBuffer,
                    camera) ||
                !BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_CULL_VIEWS_TO_MACRO_TILES,
                    1u,
                    true,
                    macroGX,
                    macroGY,
                    passDesc,
                    cullTileCmd,
                    overrideCompactedBuf,
                    m_dummyResidencyTelemetryBuffer,
                    camera) ||
                !BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_MARK_SCREEN_DEPTH,
                    0u,
                    true,
                    tiles8X,
                    tiles8Y,
                    passDesc,
                    markCmd,
                    overrideCompactedBuf,
                    m_dummyResidencyTelemetryBuffer,
                    camera))
            {
                return false;
            }

            outDispatches.push_back(std::move(reduceCmd));
            outDispatches.push_back(std::move(cullGlobalCmd));
            outDispatches.push_back(std::move(cullTileCmd));
            outDispatches.push_back(std::move(markCmd));
        }

        if (passDesc.executeCompaction)
        {
            ComputeDispatchCommand histCmd{};
            ComputeDispatchCommand prefixCmd{};
            ComputeDispatchCommand scatterCmd{};

            if (!BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_COMPACT_REQUESTS_AND_TELEMETRY,
                    0u,
                    true,
                    hashGX,
                    hashGY,
                    passDesc,
                    histCmd,
                    overrideCompactedBuf,
                    residencyTelemetryBuf,
                    camera) ||
                !BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_COMPACT_REQUESTS_AND_TELEMETRY,
                    1u,
                    true,
                    1u,
                    1u,
                    passDesc,
                    prefixCmd,
                    overrideCompactedBuf,
                    residencyTelemetryBuf,
                    camera) ||
                !BuildFeedbackStageDispatchInternal(
                    NVS_FEEDBACK_DISPATCH_COMPACT_REQUESTS_AND_TELEMETRY,
                    2u,
                    true,
                    hashGX,
                    hashGY,
                    passDesc,
                    scatterCmd,
                    overrideCompactedBuf,
                    residencyTelemetryBuf,
                    camera))
            {
                return false;
            }

            outDispatches.push_back(std::move(histCmd));
            outDispatches.push_back(std::move(prefixCmd));
            outDispatches.push_back(std::move(scatterCmd));
        }

        return true;
    }

    BOOL CoreNVSFeedbackManager::ExecuteFeedbackPass(const NVSFeedbackPassDesc& passDesc, Camera* camera)
    {
        if (!m_initialized || !m_backend)
        {
            m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass called before Initialize.";
            return false;
        }

        CameraFeedbackSlot* slot = AcquireSlot(camera);
        if (!slot)
            return false;

        GPUBuffer* residencyTelemetryBuf = (m_pageManager && m_pageManager->IsInitialized())
            ? m_pageManager->GetTelemetryBuffer()
            : nullptr;

        FrameView execView{};

        const BOOL hasDepthWork =
            passDesc.sceneDepthTexture &&
            passDesc.sceneDepthTexture->GetNativeHandle() != 0 &&
            passDesc.viewportWidth > 0u &&
            passDesc.viewportHeight > 0u;

        if (hasDepthWork)
        {
            if (passDesc.multiLightViewMode)
            {
                if (passDesc.activeViewTableCount == 0u)
                {
                    m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: activeViewTableCount must be > 0 in multiLightViewMode.";
                    return false;
                }
            }
            else if (passDesc.clipmapHierarchy.pagesPerAxis == 0u ||
                     passDesc.clipmapHierarchy.pagesPerAxis > 64u ||
                     passDesc.clipmapHierarchy.activeLevelCount == 0u ||
                     passDesc.clipmapHierarchy.activeLevelCount > kNVSMaxClipmapLevels ||
                     passDesc.clipmapHierarchy.lightId >= 4u)
            {
                m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: invalid clipmapHierarchy bounds (pagesPerAxis <= 64, activeLevelCount <= 4, lightId < 4).";
                return false;
            }
        }
        else if (!passDesc.resetBeforeMark && !passDesc.executeCompaction)
        {
            m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: missing valid SceneDepth or viewport dimensions.";
            return false;
        }

        std::vector<ComputeDispatchCommand> fbDispatches;
        if (!BuildFeedbackPipelineDispatches(passDesc, fbDispatches, slot->compactedRequestBuffer, residencyTelemetryBuf, camera))
        {
            return false;
        }
        for (auto& cmd : fbDispatches)
        {
            FramePassItem item{};
            item.type            = FramePassItemType::COMPUTE_DISPATCH;
            item.computeDispatch = std::move(cmd);
            execView.interleavedPassItems.push_back(std::move(item));
        }

        // 4. Optional Zero-Readback GPU Residency + Selective Layered Depth Clear Chain
        if (passDesc.executeResidencyAndClear)
        {
            if (!m_pageManager || !m_pageManager->IsInitialized())
            {
                m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: executeResidencyAndClear requires initialized CoreNVSPageManager.";
                return false;
            }

            const NVSPageManagerConfig& pmCfg = m_pageManager->GetConfig();
            if (passDesc.sceneDepthTexture && !passDesc.multiLightViewMode && !pmCfg.enableSparseHierarchy &&
                (passDesc.clipmapHierarchy.pagesPerAxis != pmCfg.pagesPerAxis ||
                 passDesc.clipmapHierarchy.activeLevelCount > pmCfg.clipmapLevelCount ||
                 passDesc.clipmapHierarchy.lightId >= pmCfg.maxLightCount))
            {
                m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: clipmapHierarchy mismatch with CoreNVSPageManager capacity.";
                return false;
            }

            NVSResidencyPassDesc resDesc = passDesc.residencyPassDesc;
            resDesc.useGPURequestCountFromTelemetry = true;

            std::vector<ComputeDispatchCommand> pendingInvCmds;
            if (!m_pageManager->BuildPendingInvalidationDispatches(resDesc.currentFrame, pendingInvCmds, camera))
            {
                return false;
            }
            for (auto& invCmd : pendingInvCmds)
            {
                FramePassItem invItem{};
                invItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                invItem.computeDispatch = std::move(invCmd);
                execView.interleavedPassItems.push_back(std::move(invItem));
            }

            FramePassItem residencyItem{};
            residencyItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_pageManager->BuildResidencyComputeDispatch(
                    NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE,
                    resDesc,
                    residencyItem.computeDispatch,
                    slot->compactedRequestBuffer,
                    camera))
            {
                m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: failed to build chained NVS Residency dispatch.";
                return false;
            }
            execView.interleavedPassItems.push_back(std::move(residencyItem));

            if (resDesc.executeSelectiveClear)
            {
                FramePassItem clearItem{};
                clearItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
                if (!m_pageManager->BuildSelectiveClearPassView(resDesc.clearDepth, clearItem.layeredDepthPass))
                {
                    m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: failed to build chained NVS Selective Clear pass.";
                    return false;
                }
                execView.interleavedPassItems.push_back(std::move(clearItem));
            }

            if (resDesc.markClearedAfterPass)
            {
                FramePassItem markClearedItem{};
                markClearedItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!m_pageManager->BuildResidencyComputeDispatch(
                        NVS_DISPATCH_MARK_CLEARED_PAGES,
                        resDesc,
                        markClearedItem.computeDispatch,
                        nullptr,
                        camera))
                {
                    m_lastError = "CoreNVSFeedbackManager::ExecuteFeedbackPass: failed to build chained NVS MarkCleared dispatch.";
                    return false;
                }
                execView.interleavedPassItems.push_back(std::move(markClearedItem));
            }

            if (passDesc.executePageProjection || resDesc.executePageProjection)
            {
                if (!passDesc.multiLightViewMode && passDesc.clipmapHierarchy.lightId < kNVSMaxClipmapLights)
                {
                    m_pageManager->UploadClipmapHierarchy(passDesc.clipmapHierarchy, camera);
                }

                NVSPageProjectionPassDesc projDesc{};
                projDesc.currentFrame = resDesc.currentFrame;
                projDesc.depthMode    = (passDesc.depthCompareMode != NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE)
                    ? passDesc.depthCompareMode
                    : passDesc.depthMode;
                projDesc.resetBeforeBuild   = true;
                projDesc.uploadClipmapTable = true;

                FramePassItem projResetItem{};
                projResetItem.type = FramePassItemType::COMPUTE_DISPATCH;
                FramePassItem projBuildItem{};
                projBuildItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (m_pageManager->BuildPageProjectionComputeDispatch(
                        NVS_PROJECTION_DISPATCH_RESET,
                        projDesc,
                        projResetItem.computeDispatch,
                        camera) &&
                    m_pageManager->BuildPageProjectionComputeDispatch(
                        NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS,
                        projDesc,
                        projBuildItem.computeDispatch,
                        camera))
                {
                    execView.interleavedPassItems.push_back(std::move(projResetItem));
                    execView.interleavedPassItems.push_back(std::move(projBuildItem));

                    if (passDesc.executeCasterCulling && passDesc.gpuSceneBuffer != nullptr)
                    {
                        NVSCasterCullPassDesc cullDesc{};
                        cullDesc.currentFrame                  = resDesc.currentFrame;
                        cullDesc.staticBatchPolicy             = passDesc.casterStaticBatchPolicy;
                        cullDesc.skipAlreadyRenderedValidPages = passDesc.skipValidCachedPagesInCull;
                        cullDesc.minCasterProjectedTexels      = passDesc.minCasterProjectedTexels;

                        const UINT32 slotCount = (passDesc.gpuSceneSlotCount > 0u)
                            ? passDesc.gpuSceneSlotCount
                            : passDesc.gpuSceneBuffer->GetElementCount();

                        std::vector<ComputeDispatchCommand> cullCmds;
                        if (m_pageManager->BuildCasterCullPipelineDispatches(
                                passDesc.gpuSceneBuffer,
                                slotCount,
                                cullDesc,
                                cullCmds,
                                camera))
                        {
                            for (auto& cmd : cullCmds)
                            {
                                FramePassItem cullItem{};
                                cullItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                                cullItem.computeDispatch = std::move(cmd);
                                execView.interleavedPassItems.push_back(std::move(cullItem));
                            }

                            if (passDesc.executeShadowRasterization)
                            {
                                NVSShadowRasterPassDesc rasterDesc{};
                                rasterDesc.currentFrame        = resDesc.currentFrame;
                                rasterDesc.depthMode           = projDesc.depthMode;
                                rasterDesc.allowInstancedMerge = passDesc.allowShadowInstancedMerge;
                                rasterDesc.applyShaderBias     = passDesc.applyShadowShaderBias;
                                rasterDesc.commitRenderedPages = passDesc.commitRenderedShadowPages;
                                rasterDesc.timeSeconds         = passDesc.shadowTimeSeconds;

                                std::vector<ComputeDispatchCommand> indDrawCmds;
                                if (m_pageManager->BuildIndirectDrawPipelineDispatches(
                                        passDesc.gpuSceneBuffer,
                                        slotCount,
                                        rasterDesc,
                                        indDrawCmds,
                                        camera))
                                {
                                    for (auto& icmd : indDrawCmds)
                                    {
                                        FramePassItem indItem{};
                                        indItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                                        indItem.computeDispatch = std::move(icmd);
                                        execView.interleavedPassItems.push_back(std::move(indItem));
                                    }

                                    FramePassItem shadowPassItem{};
                                    shadowPassItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
                                    if (m_pageManager->BuildShadowRasterizationPassView(
                                            passDesc.gpuSceneBuffer,
                                            slotCount,
                                            rasterDesc,
                                            shadowPassItem.layeredDepthPass,
                                            camera))
                                    {
                                        execView.interleavedPassItems.push_back(std::move(shadowPassItem));

                                        if (rasterDesc.commitRenderedPages)
                                        {
                                            FramePassItem commitItem{};
                                            commitItem.type = FramePassItemType::COMPUTE_DISPATCH;
                                            if (m_pageManager->BuildGPUCommitRenderedPagesDispatch(
                                                    resDesc.currentFrame,
                                                    commitItem.computeDispatch,
                                                    camera))
                                            {
                                                execView.interleavedPassItems.push_back(std::move(commitItem));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (execView.interleavedPassItems.empty())
            return false;

        m_backend->ExecuteFrame(execView);
        return true;
    }

    BOOL CoreNVSFeedbackManager::ResetBuffersOnGPU(Camera* camera)
    {
        if (!m_initialized || !m_backend)
            return false;

        NVSFeedbackPassDesc desc{};
        desc.resetBeforeMark          = true;
        desc.executeCompaction        = false;
        desc.executeResidencyAndClear = false;
        desc.maxRequestBudget         = m_config.maxRequestBudget;

        FrameView execView{};
        FramePassItem resetItem{};
        resetItem.type = FramePassItemType::COMPUTE_DISPATCH;
        if (!BuildFeedbackResetDispatch(desc, resetItem.computeDispatch, nullptr, camera))
            return false;
        execView.interleavedPassItems.push_back(std::move(resetItem));
        m_backend->ExecuteFrame(execView);
        return true;
    }

    BOOL CoreNVSFeedbackManager::ReadbackFeedbackTelemetrySync(NVSFeedbackTelemetry& outTelemetry, Camera* camera) const
    {
        outTelemetry = NVSFeedbackTelemetry{};
        const CameraFeedbackSlot* slot = FindSlot(camera);
        if (!m_initialized || !m_bufferManager || !slot || !slot->feedbackTelemetryBuffer)
            return false;

        return m_bufferManager->ReadbackGPUBufferSync(
            slot->feedbackTelemetryBuffer,
            &outTelemetry,
            sizeof(NVSFeedbackTelemetry));
    }

    BOOL CoreNVSFeedbackManager::ReadbackCompactedRequestsSync(
        std::vector<NVSVirtualPageRequest>& outRequests,
        UINT32                              count,
        Camera*                             camera) const
    {
        outRequests.clear();
        const CameraFeedbackSlot* slot = FindSlot(camera);
        if (!m_initialized || !m_bufferManager || !slot || !slot->compactedRequestBuffer)
            return false;

        UINT32 readCount = count;
        if (readCount == 0u)
        {
            NVSFeedbackTelemetry tel{};
            if (!ReadbackFeedbackTelemetrySync(tel, camera))
                return false;
            readCount = tel.compactedRequestCount;
        }

        if (readCount == 0u)
            return true;

        readCount = (std::min)(readCount, slot->compactedRequestBuffer->GetElementCount());
        outRequests.resize(readCount);
        return m_bufferManager->ReadbackGPUBufferSync(
            slot->compactedRequestBuffer,
            outRequests.data(),
            readCount * static_cast<UINT32>(sizeof(NVSVirtualPageRequest)));
    }

    BOOL CoreNVSFeedbackManager::ReadbackPageBitmaskSync(std::vector<UINT32>& outWords, Camera* camera) const
    {
        outWords.clear();
        const CameraFeedbackSlot* slot = FindSlot(camera);
        if (!m_initialized || !m_bufferManager || !slot || !slot->pageBitmaskBuffer)
            return false;

        outWords.resize(kNVSFeedbackBitmaskWordCount, 0u);
        return m_bufferManager->ReadbackGPUBufferSync(
            slot->pageBitmaskBuffer,
            outWords.data(),
            kNVSFeedbackBitmaskWordCount * static_cast<UINT32>(sizeof(UINT32)));
    }
}
