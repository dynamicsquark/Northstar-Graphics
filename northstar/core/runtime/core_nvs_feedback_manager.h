#pragma once

#include "../../include/types/northstar_types.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "../../include/buffer/gpu_buffer.h"
#include "../../include/texture/texture.h"
#include "../../include/camera/camera.h"
#include "core_buffer_manager.h"
#include "core_texture_manager.h"
#include "core_compute_manager.h"
#include "core_render_pipeline.h"
#include "core_nvs_page_manager.h"
#include "internal_resource_manager.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Northstar
{
    class CoreNVSFeedbackManager
    {
    public:
        CoreNVSFeedbackManager();
        ~CoreNVSFeedbackManager();

        BOOL Initialize(
            NorthstarBackend*              backend,
            CoreBufferManager*             bufferManager,
            CoreTextureManager*            textureManager,
            CoreComputeManager*            computeManager,
            CoreRenderPipeline*            renderPipeline,
            InternalResourceManager*       internalResourceManager,
            CoreNVSPageManager*            pageManager,
            const NVSFeedbackManagerConfig& config = NVSFeedbackManagerConfig{});

        void Shutdown();

        BOOL Reconfigure(const NVSFeedbackManagerConfig& newConfig);

        // Releases any per-camera feedback resources when a Camera is destroyed.
        void OnCameraDestroyed(Camera* camera);

        BOOL                            IsInitialized() const { return m_initialized; }
        const NVSFeedbackManagerConfig& GetConfig()     const { return m_config; }
        const std::string&              GetLastError()  const { return m_lastError; }

        void SetPageManager(CoreNVSPageManager* pageManager) { m_pageManager = pageManager; }
        CoreNVSPageManager* GetPageManager() const { return m_pageManager; }

        // ==================== GPU RESOURCES ====================
        GPUBuffer* GetPageBitmaskBuffer(Camera* camera = nullptr);
        GPUBuffer* GetCompactedRequestBuffer(Camera* camera = nullptr);
        GPUBuffer* GetFeedbackTelemetryBuffer(Camera* camera = nullptr);

        const GPUBuffer* GetPageBitmaskBuffer(Camera* camera = nullptr) const;
        const GPUBuffer* GetCompactedRequestBuffer(Camera* camera = nullptr) const;
        const GPUBuffer* GetFeedbackTelemetryBuffer(Camera* camera = nullptr) const;
        UINT64           GetMetadataVramBytes() const noexcept;

        // ==================== COMPUTE DISPATCH BUILDERS ====================
        BOOL BuildFeedbackResetDispatch(
            const NVSFeedbackPassDesc& passDesc,
            ComputeDispatchCommand&    outCmd,
            GPUBuffer*                 residencyTelemetryBuf = nullptr,
            Camera*                    camera                = nullptr);

        BOOL BuildFeedbackMarkDispatch(
            const NVSFeedbackPassDesc& passDesc,
            ComputeDispatchCommand&    outCmd,
            Camera*                    camera = nullptr);

        BOOL BuildFeedbackCompactDispatch(
            const NVSFeedbackPassDesc& passDesc,
            ComputeDispatchCommand&    outCmd,
            GPUBuffer*                 overrideCompactedBuf  = nullptr,
            GPUBuffer*                 residencyTelemetryBuf = nullptr,
            Camera*                    camera                = nullptr);

        BOOL BuildFeedbackPipelineDispatches(
            const NVSFeedbackPassDesc&           passDesc,
            std::vector<ComputeDispatchCommand>& outDispatches,
            GPUBuffer*                           overrideCompactedBuf  = nullptr,
            GPUBuffer*                           residencyTelemetryBuf = nullptr,
            Camera*                              camera                = nullptr);

        // Executes the GPU Screen-Space Shadow Feedback pass (Reset + Mark + Priority Compaction)
        // and optionally chains GPU Residency + Selective Layered Depth Clear in a single submission
        // with zero CPU readback.
        BOOL ExecuteFeedbackPass(const NVSFeedbackPassDesc& passDesc, Camera* camera = nullptr);

        // Resets the feedback bitmask and telemetry buffers on the GPU.
        BOOL ResetBuffersOnGPU(Camera* camera = nullptr);

        // ==================== SYNCHRONOUS GPU READBACK (DIAGNOSTICS & TESTS) ====================
        BOOL ReadbackFeedbackTelemetrySync(NVSFeedbackTelemetry& outTelemetry, Camera* camera = nullptr) const;
        BOOL ReadbackCompactedRequestsSync(
            std::vector<NVSVirtualPageRequest>& outRequests,
            UINT32                              count  = 0u,
            Camera*                             camera = nullptr) const;
        BOOL ReadbackPageBitmaskSync(std::vector<UINT32>& outWords, Camera* camera = nullptr) const;

    private:
        struct CameraFeedbackSlot
        {
            GPUBuffer* pageBitmaskBuffer          = nullptr;
            GPUBuffer* compactedRequestBuffer     = nullptr;
            GPUBuffer* feedbackTelemetryBuffer    = nullptr;
            GPUBuffer* requestHashTableBuffer     = nullptr;
            GPUBuffer* macroTileBoundsBuffer      = nullptr;
            GPUBuffer* macroTileViewIndicesBuffer = nullptr;
            GPUBuffer* bucketCountersBuffer       = nullptr;
            UINT32     budgetCapacity             = 0u;
            UINT32     hashCapacity               = 0u;
            UINT32     maxMacroTiles              = 0u;
            UINT32     maxViewsPerMacroTile       = 0u;
            UINT32     maxVisibleViewsCapacity    = 0u;
        };

        BOOL AllocateSlotResources(CameraFeedbackSlot& slot, UINT32 budgetCapacity, const std::string& namePrefix);
        BOOL EnsureSlotMacroTileAndViewCapacity(CameraFeedbackSlot& slot, UINT32 viewportW, UINT32 viewportH, UINT32 activeViews);
        BOOL BuildFeedbackStageDispatchInternal(
            NVSFeedbackDispatchMode    mode,
            UINT32                     cullSubMode,
            BOOL                       enableMacroTileBinning,
            UINT32                     groupsX,
            UINT32                     groupsY,
            const NVSFeedbackPassDesc& passDesc,
            ComputeDispatchCommand&    outCmd,
            GPUBuffer*                 overrideCompactedBuf,
            GPUBuffer*                 residencyTelemetryBuf,
            Camera*                    camera);
        void ReleaseSlotResources(CameraFeedbackSlot& slot);
        CameraFeedbackSlot*       AcquireSlot(Camera* camera);
        const CameraFeedbackSlot* FindSlot(Camera* camera) const;

        BOOL CreateDummyDepthTexture();
        void ReleaseAllResources();

        NorthstarBackend*        m_backend                 = nullptr;
        CoreBufferManager*       m_bufferManager           = nullptr;
        CoreTextureManager*      m_textureManager          = nullptr;
        CoreComputeManager*      m_computeManager          = nullptr;
        CoreRenderPipeline*      m_renderPipeline          = nullptr;
        InternalResourceManager* m_internalResourceManager = nullptr;
        CoreNVSPageManager*      m_pageManager             = nullptr;
        BOOL                     m_initialized             = false;
        NVSFeedbackManagerConfig m_config{};
        std::string              m_lastError;

        Texture*                                        m_dummyDepthTexture             = nullptr;
        GPUBuffer*                                      m_dummyResidencyTelemetryBuffer = nullptr;
        GPUBuffer*                                      m_dummyClipmapLevelsBuffer      = nullptr;
        CameraFeedbackSlot                              m_defaultSlot{};
        std::unordered_map<Camera*, CameraFeedbackSlot> m_cameraSlots;
    };
}

