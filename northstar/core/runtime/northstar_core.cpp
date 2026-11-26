#include "northstar_core.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <cmath>

namespace Northstar
{
    NorthstarCore::NorthstarCore()
    {
    }

    NorthstarCore::~NorthstarCore()
    {
        Shutdown();
    }

    BOOL NorthstarCore::Initialize(NorthstarBackend* backend, const NativeRenderTarget& renderTarget)
    {
        m_lastError.clear();

        // 1. If NorthstarCore is already initialized (or holds an active backend reference),
        //    cleanly tear down the active runtime session before attempting (re-)initialization.
        if (m_initialized || m_backend != nullptr)
        {
            const BOOL backendChanged = (m_backend != backend);
            if (m_backend)
            {
                m_backend->WaitForGPU();
            }

            if (!m_executedPerFrameKernels.empty())
            {
                m_executedPerFrameKernels.clear();
            }
            m_pendingLayeredDepthPasses.clear();
            m_frameView = FrameView{};

            m_geometryManager.SetNVGManager(nullptr);
            m_renderPipeline.SetResourceMaterializer(nullptr);
            m_nvgManager.SetResourceMaterializer(nullptr);
            m_nvgManager.Shutdown();
            m_resourceMaterializer.Shutdown();
            m_nvsFeedbackManager.Shutdown();
            m_nvsPageManager.Shutdown();
            ClearAllCameraNVSTrackingStates();
            m_sceneMaterialShadowStates.clear();
            m_nvsFramePipelineEnabled    = false;
            m_nvsHasCustomPipelineConfig = false;
            DestroyInternalNVSSamplers();
            m_internalResourceManager.Shutdown();
            m_gpuSceneManager.Shutdown();
            ReleaseAllCameraSceneDepthTextures();

            m_shaderManager.UnregisterFromBackend(backendChanged);
            m_textureManager.UnregisterFromBackend(backendChanged);
            m_bufferManager.UnregisterFromBackend(backendChanged);
            m_geometryManager.UnregisterFromBackend(backendChanged);

            if (m_backend)
            {
                m_bufferManager.FlushRetiredNativeHandles(m_backend);
            }

            m_initialized = false;
            m_backend     = nullptr;
        }

        if (!backend)
        {
            m_lastError = "NorthstarCore::Initialize failed: backend is nullptr.";
            return false;
        }

        // Track whether the caller's backend was already initialized prior to this call so we preserve
        // the backend ownership contract on rollback if a subsequent Core initialization step fails.
        const BOOL backendWasInitialized = backend->IsInitialized();

        if (!backend->Initialize(renderTarget))
        {
            m_lastError = "NorthstarCore::Initialize failed: backend->Initialize(renderTarget) returned false.";
            return false;
        }

        m_backend      = backend;
        m_renderTarget = renderTarget;
        m_backend->SetVSync(m_vsyncEnabled);
        if (m_prePresentCallback)
        {
            m_backend->SetPrePresentCallback(m_prePresentCallback);
        }
        if (m_postFrameExecutionCallback)
        {
            m_backend->SetPostFrameExecutionCallback(m_postFrameExecutionCallback, m_postFrameWaitForGPU);
        }

#ifdef _WIN32
        if (renderTarget.type == RenderTargetType::WINDOW && renderTarget.surface != nullptr)
        {
            HWND hwnd = static_cast<HWND>(renderTarget.surface);
            RECT clientRect{};
            if (::IsWindow(hwnd) && ::GetClientRect(hwnd, &clientRect))
            {
                const LONG w = clientRect.right - clientRect.left;
                const LONG h = clientRect.bottom - clientRect.top;
                if (w > 0 && h > 0)
                {
                    m_windowWidth  = static_cast<UINT32>(w);
                    m_windowHeight = static_cast<UINT32>(h);
                }
            }
        }
#endif

        // 2. Initialize mandatory engine-internal runtime resources BEFORE allocating NVS resources
        //    or registering user managers with the backend.
        if (!m_internalResourceManager.Initialize(m_backend, &m_computeManager))
        {
            m_lastError = m_internalResourceManager.GetLastError();
            m_internalResourceManager.Shutdown();
            if (!backendWasInitialized)
            {
                m_backend->Shutdown();
            }
            m_backend = nullptr;
            return false;
        }

        // 3. Register user resource managers with the active backend
        const BOOL isHLSL12Backend = (m_backend->GetTargetAPI() == TargetAPI::HLSL12);
        m_shaderManager.RegisterShadersWithBackend(m_backend);
        m_textureManager.RegisterWithBackend(m_backend);
        m_bufferManager.RegisterWithBackend(m_backend);
        m_geometryManager.RegisterWithBackend(m_backend);
        m_gpuSceneManager.Initialize(&m_bufferManager);

        // 4. Initialize authoritative Northstar Virtual Shadows (NVS) physical pool, page tables, feedback manager & engine-owned samplers
        if (isHLSL12Backend)
        {
            NVSPageManagerConfig defaultNvsCfg{};
            defaultNvsCfg.pagesPerAxis             = 64u;
            defaultNvsCfg.clipmapLevelCount        = kNVSMaxClipmapLevels;
            defaultNvsCfg.maxLightCount            = kNVSMaxClipmapLights;
            defaultNvsCfg.physicalPoolCapacity     = kNVSMaxPhysicalPoolPages;
            defaultNvsCfg.maxPageRequestsCapacity  = 32768u;
            defaultNvsCfg.minInFlightProtectFrames = 2u;
            defaultNvsCfg.enableSparseHierarchy    = true;

            const BOOL pageMgrOk = m_nvsPageManager.Initialize(
                m_backend,
                &m_bufferManager,
                &m_computeManager,
                &m_renderPipeline,
                &m_internalResourceManager,
                defaultNvsCfg);

            NVSFeedbackManagerConfig defaultFbCfg{};
            defaultFbCfg.maxRequestBudget             = 32768u;
            defaultFbCfg.maxCompactedRequestsCapacity = 32768u;

            const BOOL fbMgrOk = pageMgrOk && m_nvsFeedbackManager.Initialize(
                m_backend,
                &m_bufferManager,
                &m_textureManager,
                &m_computeManager,
                &m_renderPipeline,
                &m_internalResourceManager,
                &m_nvsPageManager,
                defaultFbCfg);

            const BOOL sampOk = pageMgrOk && fbMgrOk && EnsureInternalNVSSamplers();

            if (!pageMgrOk || !fbMgrOk || !sampOk)
            {
                if (!pageMgrOk)
                {
                    m_lastError = m_nvsPageManager.GetLastError();
                }
                else if (!fbMgrOk)
                {
                    m_lastError = m_nvsFeedbackManager.GetLastError();
                }
                else
                {
                    m_lastError = "NorthstarCore::Initialize failed: unable to create engine-owned NVS samplers.";
                }
                DestroyInternalNVSSamplers();
                m_nvsFeedbackManager.Shutdown();
                m_nvsPageManager.Shutdown();
                m_internalResourceManager.Shutdown();
                m_gpuSceneManager.Shutdown();
                m_shaderManager.UnregisterFromBackend(!backendWasInitialized);
                m_textureManager.UnregisterFromBackend(!backendWasInitialized);
                m_bufferManager.UnregisterFromBackend(!backendWasInitialized);
                m_geometryManager.UnregisterFromBackend(!backendWasInitialized);
                m_bufferManager.FlushRetiredNativeHandles(m_backend);
                if (!backendWasInitialized)
                {
                    m_backend->Shutdown();
                }
                m_backend = nullptr;
                return false;
            }
        }
        else
        {
            EnsureInternalNVSSamplers();
        }

        m_nvsFramePipelineEnabled    = isHLSL12Backend;
        m_nvsHasCustomPipelineConfig = false;

        // 5. Initialize Reflection-Driven GPU Resource Materializer & NVG subsystem
        if (isHLSL12Backend)
        {
            if (m_resourceMaterializer.Initialize(&m_bufferManager, &m_internalResourceManager))
            {
                m_renderPipeline.SetResourceMaterializer(&m_resourceMaterializer);
                m_nvgManager.SetResourceMaterializer(&m_resourceMaterializer);
            }

            if (!m_nvgManager.Initialize(
                    m_backend,
                    &m_bufferManager,
                    &m_textureManager,
                    &m_computeManager,
                    &m_internalResourceManager))
            {
                m_lastError = "NorthstarCore::Initialize failed: unable to initialize Northstar Virtual Geometry (NVG) subsystem.";
                m_renderPipeline.SetResourceMaterializer(nullptr);
                m_nvgManager.SetResourceMaterializer(nullptr);
                m_resourceMaterializer.Shutdown();
                DestroyInternalNVSSamplers();
                m_nvsFeedbackManager.Shutdown();
                m_nvsPageManager.Shutdown();
                m_internalResourceManager.Shutdown();
                m_gpuSceneManager.Shutdown();
                m_shaderManager.UnregisterFromBackend(!backendWasInitialized);
                m_textureManager.UnregisterFromBackend(!backendWasInitialized);
                m_bufferManager.UnregisterFromBackend(!backendWasInitialized);
                m_geometryManager.UnregisterFromBackend(!backendWasInitialized);
                m_bufferManager.FlushRetiredNativeHandles(m_backend);
                if (!backendWasInitialized)
                {
                    m_backend->Shutdown();
                }
                m_backend = nullptr;
                return false;
            }
            m_geometryManager.SetNVGManager(&m_nvgManager);
        }

        // 6. Ensure any pre-existing cameras have valid SceneDepth resources registered on the backend
        for (Camera* cam : m_sceneManager.GetCameras())
        {
            EnsureCameraSceneDepthTexture(cam);
        }

        for (Scene* sc : m_sceneManager.GetScenes())
        {
            if (sc && !sc->GetRenderObjects().empty())
            {
                m_gpuSceneManager.UpdateScene(sc, 0ull);
            }
        }

        m_hasLastBeginFrameTime   = false;
        m_accumulatedTimeSeconds  = 0.0;
        m_currentDeltaTimeSeconds = 0.0f;
        m_stats.deltaTimeSeconds  = 0.0f;
        m_stats.elapsedTimeSeconds = 0.0;
        m_initialized = true;
        return true;
    }

    void NorthstarCore::Shutdown()
    {
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }

        if (!m_executedPerFrameKernels.empty())
        {
            m_executedPerFrameKernels.clear();
        }
        m_pendingLayeredDepthPasses.clear();
        m_frameView = FrameView{};

        for (Scene* sc : m_sceneManager.GetScenes())
        {
            FlushRetiredSceneStaticBatches(sc);
        }

        // Release NVG, Materializer, NVS Feedback & Page Managers, internal samplers, ComputeKernels and Shaders first while backend & CoreComputeManager are active
        m_geometryManager.SetNVGManager(nullptr);
        m_renderPipeline.SetResourceMaterializer(nullptr);
        m_nvgManager.SetResourceMaterializer(nullptr);
        m_nvgManager.Shutdown();
        m_resourceMaterializer.Shutdown();
        m_nvsFeedbackManager.Shutdown();
        m_nvsPageManager.Shutdown();
        ClearAllCameraNVSTrackingStates();
        m_sceneMaterialShadowStates.clear();
        m_nvsFramePipelineEnabled    = false;
        m_nvsHasCustomPipelineConfig = false;
        DestroyInternalNVSSamplers();
        m_internalResourceManager.Shutdown();
        m_gpuSceneManager.Shutdown();
        ReleaseAllCameraSceneDepthTextures();
        m_sceneManager.Shutdown();
        m_materialManager.Shutdown();
        m_computeManager.Shutdown();
        m_geometryManager.Shutdown();
        m_bufferManager.Shutdown();
        m_textureManager.Shutdown();
        m_shaderManager.Shutdown();

        if (m_backend)
        {
            m_bufferManager.FlushRetiredNativeHandles(m_backend);
        }

        m_initialized = false;
        m_backend     = nullptr;
        m_hasLastBeginFrameTime    = false;
        m_accumulatedTimeSeconds   = 0.0;
        m_currentDeltaTimeSeconds  = 0.0f;
        m_stats.frameIndex         = 0ull;
        m_stats.deltaTimeSeconds   = 0.0f;
        m_stats.elapsedTimeSeconds = 0.0;
        m_stats.Reset();
    }

    void NorthstarCore::BeginFrame()
    {
        if (!m_initialized) return;

        if (!m_executedPerFrameKernels.empty())
        {
            m_executedPerFrameKernels.clear();
        }
        m_bufferManager.PruneRetiredNativeHandles(m_backend);

        const auto steadyNow = std::chrono::steady_clock::now();
        if (!m_hasLastBeginFrameTime)
        {
            m_hasLastBeginFrameTime   = true;
            m_lastBeginFrameTime      = steadyNow;
            m_currentDeltaTimeSeconds = 0.0f;
        }
        else
        {
            FLOAT64 dtSec = std::chrono::duration<FLOAT64>(steadyNow - m_lastBeginFrameTime).count();
            m_lastBeginFrameTime = steadyNow;
            if (!std::isfinite(dtSec) || dtSec < 0.0)
                dtSec = 0.0;
            else if (dtSec > 10.0)
                dtSec = 10.0;
            m_currentDeltaTimeSeconds  = static_cast<FLOAT32>(dtSec);
            m_accumulatedTimeSeconds  += dtSec;
        }

        const UINT64 nextFrameIdx = m_stats.frameIndex + 1;
        m_stats.Reset();
        m_stats.frameIndex         = nextFrameIdx;
        m_stats.deltaTimeSeconds   = m_currentDeltaTimeSeconds;
        m_stats.elapsedTimeSeconds = m_accumulatedTimeSeconds;
        m_resourceMaterializer.BeginFrame(m_stats.frameIndex);

        m_frameStartTime = std::chrono::high_resolution_clock::now();
    }

    void NorthstarCore::RenderFrame()
    {
        if (!m_initialized) return;

        Scene* activeScene = m_sceneManager.GetActiveScene();
        if (activeScene)
        {
            FlushRetiredSceneStaticBatches(activeScene);
        }

        if (activeScene && m_gpuSceneUpdateEnabled)
        {
            m_gpuSceneManager.UpdateScene(activeScene, m_stats.frameIndex);
            const GPUSceneUpdateStats& gs   = activeScene->GetGPUSceneStats();
            m_stats.gpuSceneDirtyCpuMs      = static_cast<FLOAT32>(gs.cpuDirtyCollectMs);
            m_stats.gpuSceneRebuildCpuMs    = static_cast<FLOAT32>(gs.cpuRecordBuildMs);
            m_stats.gpuSceneUploadCpuMs     = static_cast<FLOAT32>(gs.cpuUploadSubmitMs);
            m_stats.gpuSceneUploadedBytes   = gs.uploadedBytes;
            m_stats.gpuSceneCoalescedRanges = gs.coalescedUploadCount;
            m_stats.gpuSceneCapacitySlots   = gs.capacitySlots;
            m_stats.gpuSceneActiveSlots     = gs.activeObjectCount;
            m_stats.gpuSceneInvalidSlots    = gs.invalidSlotCount;
            m_stats.gpuSceneBufferResizes   = gs.bufferResizeCount;
        }

        std::vector<NVSDynamicCasterInvalidationBox> frameInvalidationBoxes;
        if (activeScene && m_nvsFramePipelineEnabled)
        {
            m_gpuSceneManager.ConsumePendingShadowInvalidationBoxes(activeScene, frameInvalidationBoxes);
            CollectSceneMaterialShadowInvalidations(activeScene, frameInvalidationBoxes);
        }

        if (m_nvgManager.IsInitialized())
        {
            m_nvgManager.BeginFrameTick(m_stats.frameIndex, activeScene, frameInvalidationBoxes);
        }

        auto accumulateBackendMetrics = [&](const BackendFrameMetrics& bm) {
            m_stats.fenceWaitCpuMs             += bm.fenceWaitCpuMs;
            m_stats.cmdRecordCpuMs             += bm.cmdRecordCpuMs;
            m_stats.cmdSubmitCpuMs             += bm.cmdSubmitCpuMs;
            m_stats.gpuBufferUploadRecordCpuMs += bm.gpuBufferUploadRecordCpuMs;
            m_stats.gpuSceneUploadMs           += bm.gpuSceneUploadMs;
            m_stats.gpuComputePassMs           += bm.gpuComputePassMs;
            m_stats.gpuNVSFeedbackMs           += bm.gpuNVSFeedbackMs;
            m_stats.gpuNVSResidencyCullMs      += bm.gpuNVSResidencyCullMs;
            m_stats.gpuLayeredDepthPassMs      += bm.gpuLayeredDepthPassMs;
            m_stats.gpuShadowPassMs            += (bm.gpuShadowPassMs > 0.0f) ? bm.gpuShadowPassMs : bm.gpuLayeredDepthPassMs;
            m_stats.gpuDepthPrepassMs          += bm.gpuDepthPrepassMs;
            m_stats.gpuGeometryPassMs          += bm.gpuGeometryPassMs;
            m_stats.gpuTotalFrameMs            += bm.gpuTotalFrameMs;
            m_stats.descriptorTablesUsed       += bm.descriptorTablesUsed;
            m_stats.gpuBufferUploadedBytes     += bm.uploadedBytesThisFrame;
            m_stats.gpuBufferAllocations       += bm.gpuBufferAllocationsThisFrame;
            m_stats.pendingRetiredResources     = bm.pendingRetiredResources;
            m_stats.framesInFlightCount         = bm.framesInFlightCount;
        };

        std::vector<Camera*> cameras;
        if (activeScene)
        {
            for (Camera* c : activeScene->GetCameras())
            {
                if (c) cameras.push_back(c);
            }
            if (cameras.empty() && activeScene->GetMainCamera())
            {
                cameras.push_back(activeScene->GetMainCamera());
            }

            // Sort cameras by depth ascending (lower depth renders earlier, e.g. -1 for OSR texture, 0 for main window)
            std::sort(cameras.begin(), cameras.end(), [](Camera* a, Camera* b) {
                return a->GetDepth() < b->GetDepth();
            });
        }

        if (cameras.empty())
        {
            FrameNVSSamplingContext emptyNvsCtx{};
            PopulateBaseNVSSamplingContext(NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE, emptyNvsCtx);
            m_renderPipeline.BuildFrameView(
                activeScene, nullptr, m_renderTarget, m_stats, m_frameView, &emptyNvsCtx, &m_executedPerFrameKernels, &m_internalResourceManager);
            if (!m_pendingLayeredDepthPasses.empty())
            {
                m_frameView.layeredDepthPasses = std::move(m_pendingLayeredDepthPasses);
                m_pendingLayeredDepthPasses.clear();
            }
            if (m_backend)
            {
                m_backend->ExecuteFrame(m_frameView);
                accumulateBackendMetrics(m_backend->GetLastFrameMetrics());
            }
            return;
        }

        for (SIZE_T camIdx = 0; camIdx < cameras.size(); ++camIdx)
        {
            Camera* cam = cameras[camIdx];
            if (!cam) continue;
            if (cam->GetRenderType() == CameraRenderType::TO_TEXTURE && cam->GetRenderTarget() == nullptr)
            {
                // Never silently hijack the window SwapChain when a TO_TEXTURE camera lacks a bound RenderTarget.
                continue;
            }
            if (camIdx > 0 && activeScene && m_gpuSceneUpdateEnabled)
            {
                // Enforce single per-frame GPU Scene snapshot consistency across multi-camera views:
                // UpdateScene with the same frameIndex preserves the frame's snapshot and leaves any
                // mid-frame mutations queued for frameIndex + 1.
                m_gpuSceneManager.UpdateScene(activeScene, m_stats.frameIndex);
            }
            EnsureCameraSceneDepthTexture(cam);

            if (m_nvsFramePipelineEnabled && !m_nvsHasCustomPipelineConfig)
            {
                RenderCameraWithProductionNVS(activeScene, cam, frameInvalidationBoxes);
            }
            else
            {
                FrameNVSSamplingContext customNvsCtx{};
                BuildStandaloneNVSSamplingContext(activeScene, cam, customNvsCtx);
                if (m_nvsFramePipelineEnabled && cam && cam->GetSceneDepthTexture() != nullptr)
                {
                    customNvsCtx.requiresDepthPrepass = true;
                }
                m_renderPipeline.BuildFrameView(
                    activeScene, cam, m_renderTarget, m_stats, m_frameView, &customNvsCtx, &m_executedPerFrameKernels, &m_internalResourceManager);
                if (m_nvsFramePipelineEnabled)
                {
                    InjectCustomNVSFramePipelinePasses(activeScene, cam, frameInvalidationBoxes, m_frameView);
                }
            }

            if (activeScene && m_nvgManager.IsInitialized() && m_nvgManager.HasVirtualObjectsInScene(activeScene))
            {
                GPUBuffer* sceneObjBuf = m_gpuSceneManager.GetSceneGPUBuffer(activeScene);
                const UINT32 sceneSlots = activeScene->GetGPUSceneStats().highWaterMark;
                m_nvgManager.InjectCameraNVGPasses(
                    activeScene,
                    cam,
                    m_renderTarget,
                    m_frameView,
                    sceneObjBuf,
                    sceneSlots,
                    &m_renderPipeline,
                    m_stats);
            }

            if (camIdx == 0 && !m_pendingLayeredDepthPasses.empty())
            {
                m_frameView.layeredDepthPasses = std::move(m_pendingLayeredDepthPasses);
                m_pendingLayeredDepthPasses.clear();
            }
            if (m_backend)
            {
                m_backend->ExecuteFrame(m_frameView);
                const BackendFrameMetrics& bm = m_backend->GetLastFrameMetrics();
                accumulateBackendMetrics(bm);
                if (m_nvgManager.IsInitialized())
                {
                    m_nvgManager.AccumulateBackendTiming(bm);
                    m_nvgManager.OnAfterExecuteFrame(cam);
                    m_stats.nvgStats = m_nvgManager.GetStats();
                }
                m_stats.reflectionMaterializationsThisFrame = m_resourceMaterializer.GetMaterializationsThisFrame();
                m_stats.reflectionMaterializationCacheHits  = m_resourceMaterializer.GetCacheHitsThisFrame();
                m_stats.reflectionMaterializationVramMB     = m_resourceMaterializer.GetCurrentVramMB();
            }
            if (m_postCameraCallback)
            {
                m_postCameraCallback(this, activeScene, cam, static_cast<UINT32>(camIdx), m_postCameraCallbackUserData);
            }
        }
    }

    void NorthstarCore::EndFrame()
    {
        if (!m_initialized) return;

        if (m_backend)
        {
            const auto presentStart = std::chrono::high_resolution_clock::now();
            m_backend->Present();
            const auto presentEnd = std::chrono::high_resolution_clock::now();
            m_stats.presentCpuMs = std::chrono::duration<float, std::milli>(presentEnd - presentStart).count();
        }

        const auto now = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<float, std::milli> elapsed = now - m_frameStartTime;
        m_stats.frameTimeMs = elapsed.count();
        m_stats.fps = (m_stats.frameTimeMs > 0.0f) ? (1000.0f / m_stats.frameTimeMs) : 0.0f;
    }

    void NorthstarCore::SetVSync(BOOL enabled)
    {
        m_vsyncEnabled = enabled;
        if (m_backend)
        {
            m_backend->SetVSync(enabled);
        }
    }

    BOOL NorthstarCore::IsVSyncEnabled() const
    {
        return m_backend ? m_backend->IsVSyncEnabled() : m_vsyncEnabled;
    }

    void NorthstarCore::OnResize(UINT32 width, UINT32 height)
    {
        if (width == 0 || height == 0) return;

        m_windowWidth  = width;
        m_windowHeight = height;

        if (m_backend)
        {
            m_backend->OnResize(width, height);
        }

        Scene* activeScene = m_sceneManager.GetActiveScene();
        if (activeScene)
        {
            const FLOAT32 winAspect = static_cast<FLOAT32>(width) / static_cast<FLOAT32>(height);
            auto updateFullWindowCamAspect = [&](Camera* cam) {
                if (!cam) return;
                if (cam->GetRenderType() == CameraRenderType::TO_WINDOW &&
                    cam->GetViewportWidth() <= 0.0f &&
                    cam->GetViewportHeight() <= 0.0f)
                {
                    cam->SetAspectRatio(winAspect);
                }
            };

            Camera* mainCam = activeScene->GetMainCamera();
            updateFullWindowCamAspect(mainCam);
            for (Camera* cam : activeScene->GetCameras())
            {
                if (cam != mainCam)
                {
                    updateFullWindowCamAspect(cam);
                }
            }
        }

        for (Camera* cam : m_sceneManager.GetCameras())
        {
            EnsureCameraSceneDepthTexture(cam);
        }
        if (m_nvgManager.IsInitialized())
        {
            m_nvgManager.OnResize();
        }
    }

    Shader* NorthstarCore::CreateShader(const ShaderDesc& desc)
    {
        Shader* s = m_shaderManager.CreateShader(desc);
        if (!s && !m_shaderManager.GetLastError().empty())
        {
            m_lastError = m_shaderManager.GetLastError();
        }
        return s;
    }

    void NorthstarCore::DestroyShader(Shader* shader)
    {
        if (!m_shaderManager.IsManagedShader(shader)) return;
        m_computeManager.OnShaderDestroyed(shader);
        m_shaderManager.DestroyShader(shader);
    }

    ShaderSet* NorthstarCore::CreateShaderSet(const ShaderSetDesc& desc)
    {
        ShaderSet* ss = m_shaderManager.CreateShaderSet(desc);
        if (!ss && !m_shaderManager.GetLastError().empty())
        {
            m_lastError = m_shaderManager.GetLastError();
        }
        return ss;
    }

    void NorthstarCore::DestroyShaderSet(ShaderSet* shaderSet)
    {
        if (!m_shaderManager.IsManagedShaderSet(shaderSet)) return;
        m_nvsPageManager.GetShadowRasterizer().OnShaderSetDestroyed(shaderSet);
        m_nvgManager.OnShaderSetDestroyed(shaderSet);
        m_materialManager.OnShaderSetDestroyed(shaderSet);
        m_shaderManager.DestroyShaderSet(shaderSet);
    }

    Mesh* NorthstarCore::CreateMesh(const MeshDesc& desc)
    {
        Mesh* mesh = m_geometryManager.CreateMesh(desc);
        if (!mesh && !m_geometryManager.GetLastError().empty())
        {
            m_lastError = m_geometryManager.GetLastError();
        }
        return mesh;
    }

    void NorthstarCore::DestroyMesh(Mesh* mesh)
    {
        if (!m_geometryManager.IsManagedMesh(mesh)) return;
        if (m_nvsPageManager.IsInitialized() && mesh->GetGPUSceneMeshId() != 0u)
        {
            m_nvsPageManager.MarkMeshDrawDescriptorDestroyed(mesh->GetGPUSceneMeshId());
        }
        m_sceneManager.OnMeshDestroyed(mesh);
        m_gpuSceneManager.OnMeshDestroyed(mesh);
        m_geometryManager.DestroyMesh(mesh);
    }

    BOOL NorthstarCore::SerializeVirtualMesh(const Mesh* mesh, std::vector<BYTE>& outBlob) const
    {
        if (!mesh || !mesh->IsVirtualized() || !m_nvgManager.IsInitialized())
            return false;
        return m_nvgManager.SerializeVirtualMesh(mesh->GetGPUSceneMeshId(), outBlob);
    }

    void NorthstarCore::SetNVGConfig(const NVGConfig& config)
    {
        m_nvgManager.SetConfig(config);
    }

    const NVGConfig& NorthstarCore::GetNVGConfig() const
    {
        return m_nvgManager.GetConfig();
    }

    const NVGStats& NorthstarCore::GetNVGStats() const
    {
        return m_nvgManager.GetStats();
    }

    NVGCompatibilityStatus NorthstarCore::CheckMaterialNVGCompatibility(
        const Material* material,
        PassType        pass,
        BOOL            isSkinned,
        std::string*    outReason) const
    {
        return m_nvgManager.CheckMaterialCompatibility(material, pass, isSkinned, outReason);
    }

    Material* NorthstarCore::CreateMaterial(const MaterialDesc& desc)
    {
        return m_materialManager.CreateMaterial(desc);
    }

    Material* NorthstarCore::CloneMaterial(const Material* source)
    {
        return m_materialManager.CloneMaterial(source);
    }

    void NorthstarCore::DestroyMaterial(Material* material)
    {
        if (!m_materialManager.IsManagedMaterial(material)) return;
        for (auto& kv : m_sceneMaterialShadowStates)
        {
            kv.second.erase(material);
        }
        m_nvsPageManager.GetShadowRasterizer().OnMaterialDestroyed(material);
        m_nvgManager.OnMaterialDestroyed(material);
        if (m_nvsPageManager.IsInitialized() && material->GetGPUSceneMaterialId() != 0u)
        {
            m_nvsPageManager.MarkMaterialShadowDescriptorDestroyed(material->GetGPUSceneMaterialId());
        }
        m_sceneManager.OnMaterialDestroyed(material);
        m_gpuSceneManager.OnMaterialDestroyed(material);
        m_materialManager.DestroyMaterial(material);
    }

    ComputeKernel* NorthstarCore::CreateComputeKernel(const ComputeKernelDesc& desc)
    {
        ComputeKernel* kernel = m_computeManager.CreateComputeKernel(desc);
        if (!kernel && !m_computeManager.GetLastError().empty())
        {
            m_lastError = m_computeManager.GetLastError();
        }
        return kernel;
    }

    ComputeKernel* NorthstarCore::CloneComputeKernel(const ComputeKernel* source)
    {
        return m_computeManager.CloneComputeKernel(source);
    }

    void NorthstarCore::DestroyComputeKernel(ComputeKernel* kernel)
    {
        if (!m_computeManager.IsManagedKernel(kernel) || m_computeManager.IsInternalKernel(kernel)) return;
        m_executedPerFrameKernels.erase(kernel);
        m_sceneManager.OnComputeKernelDestroyed(kernel);
        m_materialManager.OnComputeKernelDestroyed(kernel);
        m_computeManager.DestroyComputeKernel(kernel);
    }

    BOOL NorthstarCore::ExecuteInternalBufferClear(GPUBuffer* dstBuffer, UINT32 clearValue, UINT32 elementCount, UINT32 dstElementOffset)
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
            return false;

        if (!dstBuffer || !dstBuffer->IsValid())
            return false;

        if (!dstBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV) || dstBuffer->GetStride() != sizeof(UINT32))
            return false;

        if (dstElementOffset >= dstBuffer->GetElementCount())
            return false;

        if (elementCount == 0)
        {
            elementCount = dstBuffer->GetElementCount() - dstElementOffset;
        }

        const UINT64 endElement = static_cast<UINT64>(dstElementOffset) + static_cast<UINT64>(elementCount);
        if (elementCount == 0 || endElement > dstBuffer->GetElementCount())
            return false;

        ComputeKernel* clearKernel = m_internalResourceManager.GetComputeKernel(InternalComputeKernelId::BUFFER_CLEAR_UINT);
        if (!clearKernel || !clearKernel->IsValid())
            return false;

        clearKernel->SetUInt("ClearParams", "Config", "clearValue", clearValue);
        clearKernel->SetUInt("ClearParams", "Config", "elementCount", elementCount);
        clearKernel->SetUInt("ClearParams", "Config", "dstOffset", dstElementOffset);

        if (!clearKernel->SetGPUBuffer("ClearParams", "OutBuffer", dstBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE))
            return false;

        if (!clearKernel->SetDispatchThreadCount(elementCount, 1u, 1u))
        {
            clearKernel->SetGPUBuffer("ClearParams", "OutBuffer", nullptr);
            return false;
        }

        ComputeDispatchCommand cmd{};
        const BOOL built = BuildComputeDispatchCommand(clearKernel, cmd);

        // Immediately unbind the transient user GPUBuffer from the persistent internal kernel
        // after capturing the isolated ComputeDispatchCommand snapshot.
        clearKernel->SetGPUBuffer("ClearParams", "OutBuffer", nullptr);

        if (!built)
            return false;

        std::vector<ComputeDispatchCommand> cmds;
        cmds.push_back(std::move(cmd));
        return m_backend->ExecuteComputeDispatches(cmds);
    }

    BOOL NorthstarCore::BuildInternalDirtyDepthSliceDispatch(
        GPUBuffer*              candidateSlicesBuffer,
        UINT32                  candidateCount,
        UINT32                  maxArrayLayers,
        UINT32                  maxOutputSlices,
        UINT32                  inputMode,
        GPUBuffer*              outDirtySliceIndicesBuffer,
        GPUBuffer*              outDirtyCountBuffer,
        GPUBuffer*              outIndirectDrawArgsBuffer,
        ComputeDispatchCommand& outCmd)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
            return false;

        if (!outDirtySliceIndicesBuffer || !outDirtySliceIndicesBuffer->IsValid() ||
            !outDirtySliceIndicesBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV) ||
            outDirtySliceIndicesBuffer->GetStride() != sizeof(UINT32))
        {
            return false;
        }

        if (!outDirtyCountBuffer || !outDirtyCountBuffer->IsValid() ||
            !outDirtyCountBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV) ||
            outDirtyCountBuffer->GetStride() != sizeof(UINT32) ||
            outDirtyCountBuffer->GetElementCount() < 1u)
        {
            return false;
        }

        if (!outIndirectDrawArgsBuffer || !outIndirectDrawArgsBuffer->IsValid() ||
            !outIndirectDrawArgsBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV) ||
            outIndirectDrawArgsBuffer->GetStride() != sizeof(UINT32) ||
            outIndirectDrawArgsBuffer->GetElementCount() < 4u)
        {
            return false;
        }

        const UINT32 actualInputMode = inputMode & 0xFFu;
        if (actualInputMode != 0u && candidateCount > 0u)
        {
            if (!candidateSlicesBuffer || !candidateSlicesBuffer->IsValid() ||
                !candidateSlicesBuffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV) ||
                candidateSlicesBuffer->GetStride() != sizeof(UINT32) ||
                candidateCount > candidateSlicesBuffer->GetElementCount())
            {
                return false;
            }
        }
        else if (candidateSlicesBuffer != nullptr &&
                 (!candidateSlicesBuffer->IsValid() ||
                  !candidateSlicesBuffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV) ||
                  candidateSlicesBuffer->GetStride() != sizeof(UINT32)))
        {
            return false;
        }

        ComputeKernel* buildKernel = m_internalResourceManager.GetComputeKernel(InternalComputeKernelId::DIRTY_DEPTH_SLICE_BUILD);
        if (!buildKernel || !buildKernel->IsValid())
            return false;

        const UINT32 outBufferCap  = outDirtySliceIndicesBuffer->GetElementCount();
        const UINT32 statusBufCap  = outDirtyCountBuffer->GetElementCount();
        const UINT32 clampedMaxOut = (maxOutputSlices > outBufferCap) ? outBufferCap : maxOutputSlices;
        const UINT32 overflowPol   = ((inputMode & DIRTY_SLICE_INPUT_FLAG_ABORT_ON_OVERFLOW) != 0u) ? 1u : 0u;
        const UINT32 clearTailSent = ((inputMode & DIRTY_SLICE_INPUT_FLAG_PRESERVE_TAIL_CANARY) != 0u) ? 0u : 1u;

        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "candidateCount", candidateCount);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "maxArrayLayers", maxArrayLayers);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "maxOutputSlices", clampedMaxOut);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "inputMode", actualInputMode);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "outputBufferCapacity", outBufferCap);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "statusBufferCapacity", statusBufCap);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "overflowPolicy", overflowPol);
        buildKernel->SetUInt("DirtySliceBuildParams", "Config", "clearTailSentinel", clearTailSent);

        if (candidateSlicesBuffer != nullptr)
        {
            if (!buildKernel->SetGPUBuffer("DirtySliceBuildParams", "CandidateSlices", candidateSlicesBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE))
                return false;
        }
        else
        {
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "CandidateSlices", nullptr);
        }

        auto unbindTransientBuffers = [&]() {
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "CandidateSlices", nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtySliceIndices", nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtyCount", nullptr);
            buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutIndirectDrawArgs", nullptr);
        };

        if (!buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtySliceIndices", outDirtySliceIndicesBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutDirtyCount", outDirtyCountBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE) ||
            !buildKernel->SetGPUBuffer("DirtySliceBuildParams", "OutIndirectDrawArgs", outIndirectDrawArgsBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE))
        {
            unbindTransientBuffers();
            return false;
        }

        if (!buildKernel->SetDispatchThreadCount(64u, 1u, 1u))
        {
            unbindTransientBuffers();
            return false;
        }

        const BOOL built = BuildComputeDispatchCommand(buildKernel, outCmd);
        unbindTransientBuffers();
        return built;
    }

    BOOL NorthstarCore::ConfigureInternalLayeredDepthPass(LayeredDepthPassView& ioPass) const
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
            return false;

        Shader* clearVs = m_internalResourceManager.GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_CLEAR_VS);
        Shader* passGs  = m_internalResourceManager.GetGraphicsShader(InternalShaderId::LAYERED_DEPTH_PASS_GS);
        if (!clearVs || clearVs->GetNativeHandle() == 0 || !passGs || passGs->GetNativeHandle() == 0)
            return false;

        if (ioPass.fallbackGeometryShader == 0)
        {
            ioPass.fallbackGeometryShader = passGs->GetNativeHandle();
        }

        if (ioPass.selectiveClear.enabled && !ioPass.selectiveClear.useCpuPerSliceClearFallback)
        {
            if (ioPass.selectiveClear.clearVertexShader == 0)
            {
                ioPass.selectiveClear.clearVertexShader = clearVs->GetNativeHandle();
            }
            if (ioPass.selectiveClear.clearGeometryShader == 0)
            {
                ioPass.selectiveClear.clearGeometryShader = passGs->GetNativeHandle();
            }
        }
        return true;
    }

    BOOL NorthstarCore::ExecuteInternalLayeredDepthPass(
        LayeredDepthPassView&                      pass,
        const std::vector<ComputeDispatchCommand>* preDispatches)
    {
        if (!ConfigureInternalLayeredDepthPass(pass))
            return false;

        std::vector<LayeredDepthPassView> passes;
        passes.push_back(pass);
        return m_backend->ExecuteLayeredDepthPasses(passes, preDispatches);
    }

    BOOL NorthstarCore::EnsureInternalNVSSamplers()
    {
        if (!m_backend)
        {
            return false;
        }

        if (m_nvsComparisonSamplerStandard == 0)
        {
            SamplerDesc cmpStdDesc{};
            cmpStdDesc.minFilter      = SamplerFilter::LINEAR;
            cmpStdDesc.magFilter      = SamplerFilter::LINEAR;
            cmpStdDesc.mipFilter      = SamplerFilter::NEAREST;
            cmpStdDesc.wrapU          = SamplerWrap::CLAMP;
            cmpStdDesc.wrapV          = SamplerWrap::CLAMP;
            cmpStdDesc.wrapW          = SamplerWrap::CLAMP;
            cmpStdDesc.maxAnisotropy  = 1u;
            cmpStdDesc.comparisonMode = SamplerComparisonMode::LESS_EQUAL;
            m_nvsComparisonSamplerStandard = m_backend->CreateSampler(cmpStdDesc);
        }

        if (m_nvsComparisonSamplerReversedZ == 0)
        {
            SamplerDesc cmpRevDesc{};
            cmpRevDesc.minFilter      = SamplerFilter::LINEAR;
            cmpRevDesc.magFilter      = SamplerFilter::LINEAR;
            cmpRevDesc.mipFilter      = SamplerFilter::NEAREST;
            cmpRevDesc.wrapU          = SamplerWrap::CLAMP;
            cmpRevDesc.wrapV          = SamplerWrap::CLAMP;
            cmpRevDesc.wrapW          = SamplerWrap::CLAMP;
            cmpRevDesc.maxAnisotropy  = 1u;
            cmpRevDesc.comparisonMode = SamplerComparisonMode::GREATER_EQUAL;
            m_nvsComparisonSamplerReversedZ = m_backend->CreateSampler(cmpRevDesc);
        }

        if (m_nvsDepthSampler == 0)
        {
            SamplerDesc depthSampDesc{};
            depthSampDesc.minFilter      = SamplerFilter::NEAREST;
            depthSampDesc.magFilter      = SamplerFilter::NEAREST;
            depthSampDesc.mipFilter      = SamplerFilter::NEAREST;
            depthSampDesc.wrapU          = SamplerWrap::CLAMP;
            depthSampDesc.wrapV          = SamplerWrap::CLAMP;
            depthSampDesc.wrapW          = SamplerWrap::CLAMP;
            depthSampDesc.maxAnisotropy  = 1u;
            depthSampDesc.comparisonMode = SamplerComparisonMode::NONE;
            m_nvsDepthSampler = m_backend->CreateSampler(depthSampDesc);
        }

        return (m_nvsComparisonSamplerStandard != 0) &&
               (m_nvsComparisonSamplerReversedZ != 0) &&
               (m_nvsDepthSampler != 0);
    }

    void NorthstarCore::DestroyInternalNVSSamplers()
    {
        if (m_backend)
        {
            if (m_nvsComparisonSamplerStandard != 0)
            {
                m_backend->DestroySampler(m_nvsComparisonSamplerStandard);
            }
            if (m_nvsComparisonSamplerReversedZ != 0)
            {
                m_backend->DestroySampler(m_nvsComparisonSamplerReversedZ);
            }
            if (m_nvsDepthSampler != 0)
            {
                m_backend->DestroySampler(m_nvsDepthSampler);
            }
        }
        m_nvsComparisonSamplerStandard  = 0;
        m_nvsComparisonSamplerReversedZ = 0;
        m_nvsDepthSampler               = 0;
    }

    void NorthstarCore::ClearAllCameraNVSTrackingStates()
    {
        m_cameraNVSClipmapStates.clear();
        m_cameraNVSProductionStates.clear();
        m_nvsLocalLightStates.clear();
        m_nvsDirectionalLightStates.clear();
        m_nvsViewSlotOccupied.clear();
        m_nvsViewSlotWatermark = 0u;
        m_nvsFirstFreeSlotHint = 0u;
        ++m_nvsViewTableEpoch;
        for (UINT32 s = 0u; s < kNVSClipmapTableMaxEntries; ++s)
        {
            m_globalNVSSlotSignatures[s] = CameraNVSSlotSignature{};
        }
    }

    void NorthstarCore::ResetInternalNVSTrackingStates()
    {
        ClearAllCameraNVSTrackingStates();
    }

    void NorthstarCore::MarkAllNVSSlotsExternallyModified()
    {
        for (UINT32 s = 0u; s < kNVSClipmapTableMaxEntries; ++s)
        {
            m_globalNVSSlotSignatures[s]          = CameraNVSSlotSignature{};
            m_globalNVSSlotSignatures[s].occupied = true;
        }
        for (auto& kv : m_nvsLocalLightStates)
        {
            kv.second.initialized = false;
        }
        for (auto& kv : m_nvsDirectionalLightStates)
        {
            kv.second.initialized = false;
        }
        for (auto& camKv : m_cameraNVSProductionStates)
        {
            camKv.second.uploadedSlotVersion.clear();
        }
        ++m_nvsViewTableEpoch;
    }

    UINT32 NorthstarCore::AllocateNVSViewSlots(UINT32 viewCount, BOOL require4Alignment)
    {
        if (viewCount == 0u)
        {
            return 0u;
        }

        const UINT32 step = require4Alignment ? 4u : 1u;
        UINT32 cand = require4Alignment
            ? ((m_nvsFirstFreeSlotHint + 3u) & ~3u)
            : m_nvsFirstFreeSlotHint;

        const UINT32 occupiedCap = static_cast<UINT32>(m_nvsViewSlotOccupied.size());
        for (;; cand += step)
        {
            BOOL runFree = true;
            if (cand < occupiedCap)
            {
                const UINT32 checkEnd = (std::min)(cand + viewCount, occupiedCap);
                for (UINT32 s = cand; s < checkEnd; ++s)
                {
                    if (m_nvsViewSlotOccupied[s] != 0u)
                    {
                        runFree = false;
                        if (!require4Alignment)
                        {
                            cand = s; // loop increment advances to s + 1
                        }
                        break;
                    }
                }
            }
            if (runFree)
            {
                break;
            }
        }

        if (m_nvsViewSlotOccupied.size() < static_cast<SIZE_T>(cand + viewCount))
        {
            m_nvsViewSlotOccupied.resize(static_cast<SIZE_T>(cand + viewCount), 0u);
        }
        for (UINT32 i = 0u; i < viewCount; ++i)
        {
            m_nvsViewSlotOccupied[cand + i] = 1u;
        }

        m_nvsViewSlotWatermark = (std::max)(m_nvsViewSlotWatermark, cand + viewCount);
        while (m_nvsFirstFreeSlotHint < static_cast<UINT32>(m_nvsViewSlotOccupied.size()) &&
               m_nvsViewSlotOccupied[m_nvsFirstFreeSlotHint] != 0u)
        {
            ++m_nvsFirstFreeSlotHint;
        }
        return cand;
    }

    void NorthstarCore::ReclaimNVSViewSlots(UINT32 viewSlotBase, UINT32 viewCount, BOOL queueGpuReclaim)
    {
        if (viewCount == 0u)
        {
            return;
        }

        if (viewSlotBase < static_cast<UINT32>(m_nvsViewSlotOccupied.size()))
        {
            const UINT32 clrEnd = (std::min)(
                viewSlotBase + viewCount,
                static_cast<UINT32>(m_nvsViewSlotOccupied.size()));
            for (UINT32 s = viewSlotBase; s < clrEnd; ++s)
            {
                m_nvsViewSlotOccupied[s] = 0u;
            }
        }
        m_nvsFirstFreeSlotHint = (std::min)(m_nvsFirstFreeSlotHint, viewSlotBase);
        while (m_nvsViewSlotWatermark > 0u &&
               m_nvsViewSlotWatermark <= static_cast<UINT32>(m_nvsViewSlotOccupied.size()) &&
               m_nvsViewSlotOccupied[m_nvsViewSlotWatermark - 1u] == 0u)
        {
            --m_nvsViewSlotWatermark;
        }
        ++m_nvsViewTableEpoch;

        if (queueGpuReclaim && m_nvsPageManager.IsInitialized() && m_nvsPageManager.GetConfig().enableSparseHierarchy)
        {
            NVSInvalidationCommand cmd{};
            cmd.mode      = NVS_INVALIDATE_RECLAIM_VIEW;
            cmd.viewIndex = viewSlotBase;
            cmd.viewCount = viewCount;
            m_nvsPageManager.QueueInvalidationCommand(cmd);
        }
    }

    Vec4 NorthstarCore::ComputeCameraNVSScreenPixelWorldScale(const Camera* camera, BOOL sparseEnabled) const
    {
        const FLOAT32 sparseFlag = sparseEnabled ? 1.0f : 0.0f;
        FLOAT32 effVpH = static_cast<FLOAT32>((std::max)(m_windowHeight, 1u));
        if (camera != nullptr)
        {
            if (camera->GetViewportHeight() > 0.0f)
            {
                effVpH = camera->GetViewportHeight();
            }
            else if (camera->GetSceneDepthTexture() != nullptr && camera->GetSceneDepthTexture()->GetHeight() > 0u)
            {
                effVpH = static_cast<FLOAT32>(camera->GetSceneDepthTexture()->GetHeight());
            }
            else if (camera->GetRenderType() == CameraRenderType::TO_TEXTURE &&
                     camera->GetRenderTarget() != nullptr &&
                     camera->GetRenderTarget()->GetHeight() > 0u)
            {
                effVpH = static_cast<FLOAT32>(camera->GetRenderTarget()->GetHeight());
            }

            if (camera->GetProjectionType() == ProjectionType::ORTHOGRAPHIC)
            {
                const FLOAT32 orthoH = (camera->GetOrthoHeight() > 1e-4f) ? camera->GetOrthoHeight() : 10.0f;
                return Vec4(0.0f, (std::max)(orthoH / (std::max)(effVpH, 1.0f), 1e-5f), 0.0f, sparseFlag);
            }

            const FLOAT32 fovDeg = (camera->GetFOV() > 1.0f) ? camera->GetFOV() : 60.0f;
            const FLOAT32 fovRad = fovDeg * DEG2RAD;
            return Vec4((2.0f * std::tan(0.5f * fovRad)) / (std::max)(effVpH, 1.0f), 0.0f, 0.0f, sparseFlag);
        }

        const FLOAT32 defaultFovRad = 60.0f * DEG2RAD;
        return Vec4((2.0f * std::tan(0.5f * defaultFovRad)) / (std::max)(effVpH, 1.0f), 0.0f, 0.0f, sparseFlag);
    }

    BOOL NorthstarCore::InitializeInternalNVSPageManager(const NVSPageManagerConfig& config)
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
        {
            return false;
        }
        if (m_nvsPageManager.IsInitialized())
        {
            m_backend->WaitForGPU();
        }
        ClearAllCameraNVSTrackingStates();
        EnsureInternalNVSSamplers();
        return m_nvsPageManager.Initialize(
            m_backend,
            &m_bufferManager,
            &m_computeManager,
            &m_renderPipeline,
            &m_internalResourceManager,
            config);
    }

    void NorthstarCore::ShutdownInternalNVSPageManager()
    {
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }
        ClearAllCameraNVSTrackingStates();
        m_nvsPageManager.Shutdown();
    }

    BOOL NorthstarCore::InitializeInternalNVSFeedbackManager(const NVSFeedbackManagerConfig& config)
    {
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
        {
            return false;
        }
        if (m_nvsFeedbackManager.IsInitialized())
        {
            m_backend->WaitForGPU();
        }
        ClearAllCameraNVSTrackingStates();
        return m_nvsFeedbackManager.Initialize(
            m_backend,
            &m_bufferManager,
            &m_textureManager,
            &m_computeManager,
            &m_renderPipeline,
            &m_internalResourceManager,
            &m_nvsPageManager,
            config);
    }

    void NorthstarCore::ShutdownInternalNVSFeedbackManager()
    {
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }
        ClearAllCameraNVSTrackingStates();
        m_nvsFeedbackManager.Shutdown();
    }

    void NorthstarCore::SetInternalNVSFramePipelineEnabled(BOOL enabled, const NVSFeedbackPassDesc* config)
    {
        m_nvsHasCustomPipelineConfig = enabled;
        if (enabled)
        {
            m_nvsFramePipelineEnabled = true;
            if (config)
            {
                m_nvsFramePipelineConfig = *config;
            }
            else if (!m_nvsFramePipelineConfig.clipmapHierarchy.lightValid)
            {
                InitializeNVSClipmapHierarchy(
                    m_nvsFramePipelineConfig.clipmapHierarchy,
                    Vec3(0.0f, -1.0f, 0.0f),
                    0u);
            }
        }
        else
        {
            m_nvsFramePipelineEnabled = (m_backend && m_backend->GetTargetAPI() == TargetAPI::HLSL12);
        }
        ClearAllCameraNVSTrackingStates();
    }

    void NorthstarCore::CollectSceneMaterialShadowInvalidations(
        Scene*                                       scene,
        std::vector<NVSDynamicCasterInvalidationBox>& ioBoxes)
    {
        if (!scene)
        {
            return;
        }

        // Fast O(L) guard: skip scanning 100K scene objects when no shadow-casting light is active.
        BOOL hasShadowLight = false;
        const Light* dirLight = scene->GetDirectionalLight();
        if (dirLight && dirLight->HasFlag(LightFlags::ENABLED) && dirLight->HasFlag(LightFlags::CAST_SHADOW))
        {
            hasShadowLight = true;
        }
        if (!hasShadowLight)
        {
            for (const Light* l : scene->GetLights())
            {
                if (l && l->HasFlag(LightFlags::ENABLED) && l->HasFlag(LightFlags::CAST_SHADOW))
                {
                    hasShadowLight = true;
                    break;
                }
            }
        }
        if (!hasShadowLight)
        {
            return;
        }

        auto& matSnapshots = m_sceneMaterialShadowStates[scene];
        std::unordered_set<const Material*> mutatedMaterials;

        for (RenderObject* obj : scene->GetRenderObjects())
        {
            if (!obj || !obj->GetCastShadows())
            {
                continue;
            }
            const Material* mat = obj->GetMaterial();
            if (!mat)
            {
                continue;
            }

            MaterialShadowStateSnapshot currentSnap{};
            currentSnap.stateRevision           = mat->GetStateRevision();
            currentSnap.shadowShaderSet         = mat->HasPass(PassType::SHADOW) ? mat->GetShaderSet(PassType::SHADOW) : nullptr;
            currentSnap.shadowShaderSetRevision = currentSnap.shadowShaderSet ? currentSnap.shadowShaderSet->GetStateRevision() : 0u;
            currentSnap.cullMode                = mat->GetCullMode();
            currentSnap.alphaTested             = mat->IsAlphaTested();
            currentSnap.twoSided                = mat->IsTwoSided();

            auto it = matSnapshots.find(mat);
            if (it == matSnapshots.end())
            {
                matSnapshots.emplace(mat, currentSnap);
            }
            else if (it->second.stateRevision != currentSnap.stateRevision ||
                     it->second.shadowShaderSet != currentSnap.shadowShaderSet ||
                     it->second.shadowShaderSetRevision != currentSnap.shadowShaderSetRevision ||
                     it->second.cullMode != currentSnap.cullMode ||
                     it->second.alphaTested != currentSnap.alphaTested ||
                     it->second.twoSided != currentSnap.twoSided)
            {
                it->second = currentSnap;
                mutatedMaterials.insert(mat);
            }
        }

        if (mutatedMaterials.empty())
        {
            return;
        }

        for (RenderObject* obj : scene->GetRenderObjects())
        {
            if (!obj || !obj->GetCastShadows())
            {
                continue;
            }
            const Material* mat = obj->GetMaterial();
            if (!mat || mutatedMaterials.find(mat) == mutatedMaterials.end())
            {
                continue;
            }
            const AABB wb = obj->GetWorldAABB();
            if (!wb.IsValid())
            {
                continue;
            }
            NVSDynamicCasterInvalidationBox box{};
            box.worldMin          = wb.m_Min;
            box.worldMax          = wb.m_Max;
            box.gpuSceneSlotIndex = obj->GetGPUSceneHandle().slotIndex;
            box.invalidationFlags = NVS_DYN_INV_FLAG_NEW_BOUNDS | NVS_DYN_INV_FLAG_MATERIAL_OR_MESH;
            ioBoxes.push_back(box);
        }
    }

    void NorthstarCore::PopulateBaseNVSSamplingContext(
        NVSDepthCompareMode      depthMode,
        FrameNVSSamplingContext& outContext) const
    {
        outContext = FrameNVSSamplingContext{};
        outContext.depthMode         = static_cast<UINT32>(depthMode);
        outContext.comparisonSampler = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
            ? m_nvsComparisonSamplerReversedZ
            : m_nvsComparisonSamplerStandard;
        outContext.depthSampler      = m_nvsDepthSampler;
        outContext.pagesPerAxis      = 64u;
        outContext.pageTexelSize     = kNVSPageTexelSize;

        if (!m_nvsPageManager.IsInitialized())
        {
            return;
        }

        GPUBuffer* vptBuf  = m_nvsPageManager.GetVirtualPageTableBuffer();
        GPUBuffer* pptBuf  = m_nvsPageManager.GetPhysicalPageTableBuffer();
        GPUBuffer* sphBuf  = m_nvsPageManager.GetSparsePageHierarchyBuffer();
        const auto& cfg    = m_nvsPageManager.GetConfig();

        outContext.shadowMapTexture              = m_nvsPageManager.GetPhysicalPoolTextureHandle();
        outContext.virtualPageTableBuffer        = vptBuf ? vptBuf->GetNativeHandle() : 0;
        outContext.virtualPageTableElementCount  = vptBuf ? vptBuf->GetElementCount() : 0u;
        outContext.physicalPageTableBuffer       = pptBuf ? pptBuf->GetNativeHandle() : 0;
        outContext.physicalPageTableElementCount = pptBuf ? pptBuf->GetElementCount() : 0u;
        outContext.pagesPerAxis                  = cfg.pagesPerAxis;
        outContext.physicalPoolCapacity          = cfg.physicalPoolCapacity;
        if (cfg.enableSparseHierarchy && sphBuf && sphBuf->IsValid())
        {
            outContext.sparsePageHierarchyBuffer       = sphBuf->GetNativeHandle();
            outContext.sparsePageHierarchyElementCount = sphBuf->GetElementCount();
            outContext.sparseRootCapacity              = m_nvsPageManager.GetSparseRootCapacity();
            outContext.sparseDirStride                 = kNVSSparseDirStride;
            outContext.sparseDirCapacity               = m_nvsPageManager.GetSparseDirCapacity();
        }
    }

    void NorthstarCore::BuildStandaloneNVSSamplingContext(
        Scene*                   scene,
        Camera*                  camera,
        FrameNVSSamplingContext& outContext) const
    {
        const NVSDepthCompareMode depthMode = m_nvsHasCustomPipelineConfig
            ? ((m_nvsFramePipelineConfig.depthCompareMode != NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE)
                ? m_nvsFramePipelineConfig.depthCompareMode
                : m_nvsFramePipelineConfig.depthMode)
            : NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;

        PopulateBaseNVSSamplingContext(depthMode, outContext);
        outContext.screenPixelWorldScale = ComputeCameraNVSScreenPixelWorldScale(
            camera,
            outContext.sparsePageHierarchyBuffer != 0 && outContext.sparseRootCapacity > 0u);
        if (!m_nvsPageManager.IsInitialized())
        {
            return;
        }

        if (m_nvsPageManager.HasCommittedShadowSnapshot(camera))
        {
            GPUBuffer* committedBuf = m_nvsPageManager.GetCommittedSamplingViewTableBuffer(camera);
            outContext.opaqueLightViewTableBuffer       = committedBuf ? committedBuf->GetNativeHandle() : 0;
            outContext.opaqueLightViewTableElementCount = committedBuf ? committedBuf->GetElementCount() : 0u;
            outContext.opaqueViewTableCpuMirror         = m_nvsPageManager.GetCommittedCpuViewTable(camera);
            outContext.opaqueViewTableCpuMirrorCount    = (std::max)(m_nvsPageManager.GetCommittedViewTableCount(camera), kNVSClipmapTableMaxEntries);
            outContext.opaqueSamplingAvailable          = 1u;
            outContext.opaqueSnapshotFrameIndex         = m_nvsPageManager.GetCommittedSnapshotFrameIndex(camera);

            auto prodIt = m_cameraNVSProductionStates.find(camera);
            if (prodIt != m_cameraNVSProductionStates.end())
            {
                outContext.opaqueActiveViewCount  = prodIt->second.committedActiveViewCount;
                outContext.opaqueLightAssignments = prodIt->second.committedLightAssignments;
            }
            else
            {
                outContext.opaqueActiveViewCount = kNVSMaxClipmapLevels;
                if (scene && scene->GetDirectionalLight())
                {
                    LightNVSViewAssignment assign{};
                    assign.light           = scene->GetDirectionalLight();
                    assign.shadowViewBase  = 0u;
                    assign.shadowViewCount = kNVSMaxClipmapLevels;
                    assign.nvsLightId      = 0u;
                    outContext.opaqueLightAssignments.push_back(assign);
                }
            }
        }
    }

    namespace
    {
        BOOL ShaderReflectionReferencesNVS(const ShaderReflection& refl) noexcept
        {
            for (const auto& tex : refl.textures)
            {
                if (tex.IsEngineProvided() && tex.propertyName == "NVS_SHADOW_MAP")
                {
                    return true;
                }
            }
            for (const auto& samp : refl.samplers)
            {
                if (samp.IsEngineProvided() &&
                    (samp.propertyName == "NVS_COMPARISON_SAMPLER" || samp.propertyName == "NVS_DEPTH_SAMPLER"))
                {
                    return true;
                }
            }
            for (const auto& sb : refl.structuredBuffers)
            {
                for (const auto& f : sb.fields)
                {
                    if (f.isGet && f.propertyName.compare(0u, 4u, "NVS_") == 0)
                    {
                        return true;
                    }
                }
            }
            for (const auto& cb : refl.buffers)
            {
                for (const auto& f : cb.fields)
                {
                    if (f.isGet && f.propertyName.compare(0u, 4u, "NVS_") == 0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        BOOL ShaderSetReferencesNVS(const ShaderSet* set) noexcept
        {
            if (!set)
            {
                return false;
            }
            const Shader* ps = set->GetPixelShader();
            if (ps && ps->IsValid() && ShaderReflectionReferencesNVS(ps->GetReflection()))
            {
                return true;
            }
            for (UINT32 v = 0u; v < static_cast<UINT32>(ShaderVariant::COUNT); ++v)
            {
                const Shader* vs = set->GetVertexShader(static_cast<ShaderVariant>(v));
                if (vs && vs->IsValid() && ShaderReflectionReferencesNVS(vs->GetReflection()))
                {
                    return true;
                }
            }
            return false;
        }

        BOOL ShaderReflectionReferencesProperty(const ShaderReflection& refl, const char* propName) noexcept
        {
            if (!propName)
            {
                return false;
            }
            for (const auto& cb : refl.buffers)
            {
                for (const auto& f : cb.fields)
                {
                    if (f.isGet && f.propertyName == propName)
                    {
                        return true;
                    }
                }
            }
            for (const auto& sb : refl.structuredBuffers)
            {
                for (const auto& f : sb.fields)
                {
                    if (f.isGet && f.propertyName == propName)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        BOOL ShaderSetReferencesProperty(const ShaderSet* set, const char* propName) noexcept
        {
            if (!set || !propName)
            {
                return false;
            }
            const Shader* fixedStages[4] = {
                set->GetPixelShader(),
                set->GetGeometryShader(),
                set->GetHullShader(),
                set->GetDomainShader()
            };
            for (const Shader* st : fixedStages)
            {
                if (st && st->IsValid() && ShaderReflectionReferencesProperty(st->GetReflection(), propName))
                {
                    return true;
                }
            }
            for (UINT32 v = 0u; v < static_cast<UINT32>(ShaderVariant::COUNT); ++v)
            {
                const Shader* vs = set->GetVertexShader(static_cast<ShaderVariant>(v));
                if (vs && vs->IsValid() && ShaderReflectionReferencesProperty(vs->GetReflection(), propName))
                {
                    return true;
                }
            }
            return false;
        }

        BOOL ShaderSetVariantReferencesProperty(const ShaderSet* set, ShaderVariant variant, const char* propName) noexcept
        {
            if (!set || !propName)
            {
                return false;
            }
            const Shader* vs = set->GetVertexShader(variant);
            if (vs && vs->IsValid() && ShaderReflectionReferencesProperty(vs->GetReflection(), propName))
            {
                return true;
            }
            const Shader* fixedStages[4] = {
                set->GetPixelShader(),
                set->GetGeometryShader(),
                set->GetHullShader(),
                set->GetDomainShader()
            };
            for (const Shader* st : fixedStages)
            {
                if (st && st->IsValid() && ShaderReflectionReferencesProperty(st->GetReflection(), propName))
                {
                    return true;
                }
            }
            return false;
        }

        BOOL NearlyEqualFloat(FLOAT32 a, FLOAT32 b, FLOAT32 eps = 1e-4f) noexcept
        {
            return std::fabs(a - b) <= eps;
        }

        BOOL NearlyEqualVec3(const Vec3& a, const Vec3& b, FLOAT32 eps = 1e-4f) noexcept
        {
            return NearlyEqualFloat(a.x, b.x, eps) &&
                   NearlyEqualFloat(a.y, b.y, eps) &&
                   NearlyEqualFloat(a.z, b.z, eps);
        }
    }

    void NorthstarCore::RenderCameraWithProductionNVS(
        Scene*                                              scene,
        Camera*                                             camera,
        const std::vector<NVSDynamicCasterInvalidationBox>& frameInvalidationBoxes)
    {
        if (!scene || !camera || !m_nvsPageManager.IsInitialized() || !m_nvsFeedbackManager.IsInitialized())
        {
            FrameNVSSamplingContext fallbackCtx{};
            PopulateBaseNVSSamplingContext(NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE, fallbackCtx);
            m_renderPipeline.BuildFrameView(
                scene, camera, m_renderTarget, m_stats, m_frameView, &fallbackCtx, &m_executedPerFrameKernels, &m_internalResourceManager);
            return;
        }

        const auto tPrepStart = std::chrono::high_resolution_clock::now();

        // 1. Gather all active shadow-casting lights FIRST so scenes with 0 shadow lights
        //    skip O(N) caster/material inspection completely!
        std::vector<const Light*> shadowLights;
        shadowLights.reserve(scene->GetLights().size() + 1u);
        const Light* primaryDir = scene->GetDirectionalLight();
        if (primaryDir && primaryDir->HasFlag(LightFlags::ENABLED) && primaryDir->HasFlag(LightFlags::CAST_SHADOW))
        {
            shadowLights.push_back(primaryDir);
        }
        for (const Light* l : scene->GetLights())
        {
            if (!l || l == primaryDir) continue;
            if (!l->HasFlag(LightFlags::ENABLED) || !l->HasFlag(LightFlags::CAST_SHADOW)) continue;
            shadowLights.push_back(l);
        }

        // 2. Inspect scene casters, active caster variants & receiver shaders (only when shadow lights exist)
        UINT32 shadowCasterCount          = 0u;
        BOOL   sceneHasNVSReceiver        = false;
        BOOL   allCastersReadDepthBias    = true;
        BOOL   allCastersReadNormalBias   = true;
        std::unordered_set<const Material*> checkedMaterials;

        struct CasterGroupKey
        {
            const Mesh*     mesh    = nullptr;
            const Material* mat     = nullptr;
            BOOL            skinned = false;
            bool operator==(const CasterGroupKey& o) const noexcept
            {
                return mesh == o.mesh && mat == o.mat && skinned == o.skinned;
            }
        };
        struct CasterGroupKeyHash
        {
            size_t operator()(const CasterGroupKey& k) const noexcept
            {
                const size_t h1 = std::hash<const void*>{}(k.mesh);
                const size_t h2 = std::hash<const void*>{}(k.mat);
                return h1 ^ (h2 * 0x9e3779b97f4a7c15ull) ^ (k.skinned ? 0x5555ull : 0ull);
            }
        };
        std::unordered_map<CasterGroupKey, UINT32, CasterGroupKeyHash> casterGroupCounts;

        if (!shadowLights.empty())
        {
            for (RenderObject* obj : scene->GetRenderObjects())
            {
                if (!obj) continue;
                const Material* mat = obj->GetMaterial();
                if (obj->GetCastShadows() && mat && mat->HasPass(PassType::SHADOW) && mat->GetShadowShaderSet() != nullptr)
                {
                    shadowCasterCount++;
                    const BOOL isSkinned = obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices();
                    casterGroupCounts[CasterGroupKey{ obj->GetMesh(), mat, isSkinned }]++;
                }
                if (mat && checkedMaterials.insert(mat).second)
                {
                    if (ShaderSetReferencesNVS(mat->GetGeometryShaderSet()))
                    {
                        sceneHasNVSReceiver = true;
                    }
                    if (!sceneHasNVSReceiver && mat->HasComputeKernels())
                    {
                        for (const ComputeKernel* k : mat->GetComputeKernels())
                        {
                            if (k && k->IsValid() && k->GetComputeShader() &&
                                ShaderReflectionReferencesNVS(k->GetComputeShader()->GetReflection()))
                            {
                                sceneHasNVSReceiver = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (shadowCasterCount == 0u)
        {
            allCastersReadDepthBias  = false;
            allCastersReadNormalBias = false;
        }
        else
        {
            for (const auto& kv : casterGroupCounts)
            {
                const ShaderSet* shadowSet = kv.first.mat->GetShadowShaderSet();
                const ShaderVariant singleVar = kv.first.skinned ? ShaderVariant::SKINNED : ShaderVariant::STANDARD;
                const ShaderVariant instVar   = kv.first.skinned ? ShaderVariant::INSTANCED_SKINNED : ShaderVariant::INSTANCED;
                const BOOL hasSingle = (shadowSet->GetVertexShader(singleVar) != nullptr);
                const BOOL hasInst   = (shadowSet->GetVertexShader(instVar) != nullptr);
                if (!hasSingle && !hasInst)
                {
                    allCastersReadDepthBias  = false;
                    allCastersReadNormalBias = false;
                }
                if (hasSingle)
                {
                    if (!ShaderSetVariantReferencesProperty(shadowSet, singleVar, "NVS_DEPTH_BIAS_NDC"))
                        allCastersReadDepthBias = false;
                    if (!ShaderSetVariantReferencesProperty(shadowSet, singleVar, "NVS_NORMAL_BIAS_WORLD"))
                        allCastersReadNormalBias = false;
                }
                if (hasInst && (kv.second > 1u || !hasSingle))
                {
                    if (!ShaderSetVariantReferencesProperty(shadowSet, instVar, "NVS_DEPTH_BIAS_NDC"))
                        allCastersReadDepthBias = false;
                    if (!ShaderSetVariantReferencesProperty(shadowSet, instVar, "NVS_NORMAL_BIAS_WORLD"))
                        allCastersReadNormalBias = false;
                }
            }
        }
        const BOOL sceneRasterReadsDepthBias  = allCastersReadDepthBias;
        const BOOL sceneRasterReadsNormalBias = allCastersReadNormalBias;

        if (!shadowLights.empty() && !sceneHasNVSReceiver)
        {
            for (const auto& entry : scene->GetComputeKernelEntries())
            {
                if (entry.kernel && entry.kernel->IsValid() && entry.kernel->GetComputeShader() &&
                    ShaderReflectionReferencesNVS(entry.kernel->GetComputeShader()->GetReflection()))
                {
                    sceneHasNVSReceiver = true;
                    break;
                }
            }
        }

        // NVS physical shadow pool always uses Reversed-Z (D32_FLOAT 1.0 -> 0.0) for maximum
        // logarithmic depth precision across both orthographic clipmaps and perspective Spot/Point views.
        const NVSDepthCompareMode depthMode  = NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO;
        const FLOAT32             clearDepth = 0.0f;

        CameraNVSProductionState& camState = m_cameraNVSProductionStates[camera];
        const Vec3   camWorldPos           = camera->GetPosition();
        const BOOL   isPerspectiveCam      = (camera->GetProjectionType() == ProjectionType::PERSPECTIVE);
        const auto&  pageCfg               = m_nvsPageManager.GetConfig();
        const BOOL   useSparseHierarchy    = pageCfg.enableSparseHierarchy;
        const UINT32 pagesPerAxis          = (pageCfg.pagesPerAxis > 0u) ? pageCfg.pagesPerAxis : 64u;
        const UINT32 rasterBiasFlags       =
            (sceneRasterReadsDepthBias  ? NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_DEPTH_BIAS  : 0u) |
            (sceneRasterReadsNormalBias ? NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_NORMAL_BIAS : 0u);

        std::vector<LightNVSViewAssignment> currentAssignments;
        currentAssignments.reserve(shadowLights.size());
        UINT32 activeViewSlotHighWater = 0u;
        UINT32 totalActiveShadowViews  = 0u;
        UINT32 logicalLightCounter     = 0u;

        struct OriginShiftInvalidation
        {
            UINT32 lightId            = 0u;
            UINT32 clipmapLevel       = 0u;
            INT32  deltaPageX         = 0;
            INT32  deltaPageY         = 0;
            INT32  snappedOriginPageX = 0;
            INT32  snappedOriginPageY = 0;
            BOOL   fullLevelReset     = false;
        };
        struct CameraActiveSlotSpan
        {
            UINT32                        baseSlot    = 0u;
            UINT32                        viewCount   = 0u;
            UINT64                        dataVersion = 0ull;
            const NVSClipmapLevelGPUData* views       = nullptr;
        };

        std::vector<OriginShiftInvalidation> legacyOriginShifts;
        std::vector<UINT32>                  legacyInvalidatedSlots;
        NVSClipmapLevelGPUData               legacyViewTable[kNVSClipmapTableMaxEntries]{};
        std::vector<CameraActiveSlotSpan>    cameraActiveSpans;
        cameraActiveSpans.reserve(shadowLights.size());

        if (useSparseHierarchy)
        {
            const UINT64 frameStamp = (std::max)(1ull, m_stats.frameIndex);

            // Pass 2A (Sparse): Dynamic 4-aligned blocks per (Camera*, Directional Light UID) + 1/6-slot blocks per Local Light UID
            std::unordered_set<UINT64> activeDirUids;
            std::unordered_set<UINT64> activeLocalUids;
            activeDirUids.reserve(8u);
            activeLocalUids.reserve(shadowLights.size());

            for (const Light* l : shadowLights)
            {
                const UINT64 uid = l->GetRuntimeUid();
                if (l->GetType() == LightType::DIRECTIONAL)
                {
                    activeDirUids.insert(uid);
                }
                else if (l->GetType() == LightType::SPOT || l->GetType() == LightType::POINT)
                {
                    activeLocalUids.insert(uid);
                }
            }

            // Reclaim any directional light views on this camera that were removed/disabled/type-changed in this scene
            for (auto it = m_nvsDirectionalLightStates.begin(); it != m_nvsDirectionalLightStates.end();)
            {
                if (it->second.cameraPtr == camera &&
                    it->second.scenePtr == scene &&
                    activeDirUids.find(it->second.lightRuntimeUid) == activeDirUids.end())
                {
                    ReclaimNVSViewSlots(it->second.viewSlotBase, kNVSMaxClipmapLevels, true);
                    it = m_nvsDirectionalLightStates.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // Reclaim any local light views in this scene that were removed/disabled/type-changed
            for (auto it = m_nvsLocalLightStates.begin(); it != m_nvsLocalLightStates.end();)
            {
                if (it->second.scenePtr == scene &&
                    activeLocalUids.find(it->second.lightRuntimeUid) == activeLocalUids.end())
                {
                    ReclaimNVSViewSlots(it->second.viewSlotBase, it->second.viewCount, true);
                    it = m_nvsLocalLightStates.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // Schedule Directional Lights (4-aligned blocks)
            for (const Light* l : shadowLights)
            {
                if (l->GetType() != LightType::DIRECTIONAL)
                {
                    continue;
                }

                const UINT64                uid             = l->GetRuntimeUid();
                const UINT32                logicalLightId  = logicalLightCounter++;
                (void)logicalLightId;
                const Vec3                  lightDir        = l->GetDirection();
                const LightShadowResolution shadowRes       = l->GetShadowResolution();
                // Unified 16K virtual clipmap pyramid (128x128 pages, 8 mips); shadowRes controls per-light detail tier.
                const UINT32                dirPagesAxis    = kNVSMaxPagesPerAxis;
                UINT32                      dirMinMip       = 0u;
                FLOAT32                     dirLodBias      = 0.0f;
                ComputeNVSDetailTierFromResolution(shadowRes, dirMinMip, dirLodBias);
                const FLOAT32               maxDist         = (std::max)(0.05f, l->GetShadowMaxDistance());
                const FLOAT32               nearPlane       = l->GetShadowNearPlane();
                const FLOAT32               nearPlaneOffset = nearPlane - 0.1f;
                const FLOAT32               sBias           = l->GetShadowBias();
                const FLOAT32               nBias           = l->GetShadowNormalBias();

                FLOAT32 basePageWorld    = 2.0f;
                FLOAT32 levelScaleFactor = 2.0f;
                ComputeNVSDirectionalClipmapScalingFromResolution(
                    shadowRes,
                    maxDist,
                    dirPagesAxis,
                    basePageWorld,
                    levelScaleFactor);

                NVSDirectionalCameraKey dirKey{ camera, uid };
                auto dirIt = m_nvsDirectionalLightStates.find(dirKey);
                if (dirIt == m_nvsDirectionalLightStates.end())
                {
                    NVSDirectionalLightCameraState newState{};
                    newState.scenePtr        = scene;
                    newState.cameraPtr       = camera;
                    newState.lightRuntimeUid = uid;
                    newState.viewSlotBase    = AllocateNVSViewSlots(kNVSMaxClipmapLevels, true);
                    newState.viewGeneration  = 1u;
                    dirIt = m_nvsDirectionalLightStates.emplace(dirKey, std::move(newState)).first;
                }

                NVSDirectionalLightCameraState& st = dirIt->second;
                const UINT32 baseSlot              = st.viewSlotBase;
                const UINT32 blockIdx              = baseSlot >> 2u;

                const BOOL resolutionChanged = st.initialized && (st.shadowResolution != shadowRes);
                const BOOL basisOrProjChanged =
                    !st.initialized ||
                    st.scenePtr != scene ||
                    resolutionChanged ||
                    !NearlyEqualVec3(st.direction, lightDir) ||
                    !NearlyEqualFloat(st.shadowNearPlane, nearPlane, 1e-6f) ||
                    !NearlyEqualFloat(st.shadowMaxDistance, maxDist, 1e-4f) ||
                    st.depthMode != static_cast<UINT32>(depthMode) ||
                    st.hierarchy.pagesPerAxis != dirPagesAxis;
                const BOOL rasterBiasInvalidated =
                    st.initialized &&
                    ((!NearlyEqualFloat(st.shadowBias, sBias, 1e-6f) && (st.rasterAppliedDepthBias || sceneRasterReadsDepthBias)) ||
                     (st.rasterAppliedDepthBias != sceneRasterReadsDepthBias) ||
                     (!NearlyEqualFloat(st.shadowNormalBias, nBias, 1e-6f) && (st.rasterAppliedNormalBias || sceneRasterReadsNormalBias)) ||
                     (st.rasterAppliedNormalBias != sceneRasterReadsNormalBias));
                const BOOL samplingOnlyBiasOrCamChanged =
                    !NearlyEqualFloat(st.shadowBias, sBias, 1e-6f) ||
                    !NearlyEqualFloat(st.shadowNormalBias, nBias, 1e-6f) ||
                    (st.isPerspectiveCam != isPerspectiveCam);

                BOOL anyOriginOrViewDirty = basisOrProjChanged || rasterBiasInvalidated || samplingOnlyBiasOrCamChanged;

                if (basisOrProjChanged)
                {
                    if (st.initialized)
                    {
                        st.viewGeneration = (st.viewGeneration >= 255u) ? 1u : (st.viewGeneration + 1u);
                        NVSInvalidationCommand invCmd{};
                        invCmd.mode      = resolutionChanged ? NVS_INVALIDATE_RECLAIM_VIEW : NVS_INVALIDATE_VIEW_SLOT;
                        invCmd.viewIndex = baseSlot;
                        invCmd.viewCount = kNVSMaxClipmapLevels;
                        m_nvsPageManager.QueueInvalidationCommand(invCmd);
                    }

                    NVSOriginShiftResult initShifts[kNVSMaxClipmapLevels]{};
                    ComputeNVSDirectionalClipmapHierarchy(
                        st.hierarchy,
                        camWorldPos,
                        lightDir,
                        blockIdx,
                        basePageWorld,
                        dirPagesAxis,
                        kNVSMaxClipmapLevels,
                        maxDist,
                        depthMode,
                        0.0f,
                        initShifts,
                        levelScaleFactor,
                        nearPlaneOffset);
                }
                else
                {
                    if (rasterBiasInvalidated)
                    {
                        st.viewGeneration = (st.viewGeneration >= 255u) ? 1u : (st.viewGeneration + 1u);
                        NVSInvalidationCommand invCmd{};
                        invCmd.mode      = NVS_INVALIDATE_VIEW_SLOT;
                        invCmd.viewIndex = baseSlot;
                        invCmd.viewCount = kNVSMaxClipmapLevels;
                        m_nvsPageManager.QueueInvalidationCommand(invCmd);
                    }

                    const Vec3 prevCamCheck(
                        st.cachedViews[0].lightViewToWorld.m[12],
                        st.cachedViews[0].lightViewToWorld.m[13],
                        st.cachedViews[0].lightViewToWorld.m[14]);
                    if (isPerspectiveCam && !NearlyEqualVec3(prevCamCheck, camWorldPos, 1e-4f))
                    {
                        anyOriginOrViewDirty = true;
                    }

                    NVSOriginShiftResult l0Shift{};
                    NVSOriginShiftResult allShifts[kNVSMaxClipmapLevels]{};
                    UpdateNVSClipmapHierarchySnappedOrigins(st.hierarchy, camWorldPos, l0Shift, allShifts);
                    for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
                    {
                        if (allShifts[lvl].depthRangeChanged || allShifts[lvl].fullWindowReset)
                        {
                            anyOriginOrViewDirty = true;
                            if (!rasterBiasInvalidated)
                            {
                                NVSInvalidationCommand invCmd{};
                                invCmd.mode      = NVS_INVALIDATE_VIEW_SLOT;
                                invCmd.viewIndex = baseSlot + lvl;
                                invCmd.viewCount = 1u;
                                m_nvsPageManager.QueueInvalidationCommand(invCmd);
                            }
                        }
                        else if (allShifts[lvl].deltaPageX != 0 || allShifts[lvl].deltaPageY != 0)
                        {
                            anyOriginOrViewDirty = true;
                            if (!rasterBiasInvalidated)
                            {
                                NVSInvalidationCommand shiftCmd{};
                                shiftCmd.mode              = NVS_INVALIDATE_VIEW_ORIGIN;
                                shiftCmd.viewIndex         = baseSlot + lvl;
                                shiftCmd.viewCount         = 1u;
                                shiftCmd.regionOrPagesAxis = dirPagesAxis;
                                shiftCmd.originShiftDeltaX = allShifts[lvl].deltaPageX;
                                shiftCmd.originShiftDeltaY = allShifts[lvl].deltaPageY;
                                shiftCmd.snappedOriginX    = allShifts[lvl].snappedOriginPageX;
                                shiftCmd.snappedOriginY    = allShifts[lvl].snappedOriginPageY;
                                m_nvsPageManager.QueueInvalidationCommand(shiftCmd);
                            }
                        }
                    }
                }

                if (anyOriginOrViewDirty)
                {
                    BuildNVSClipmapHierarchyGPUDataAtSlot(st.hierarchy, st.cachedViews, kNVSMaxClipmapLevels, 0u);
                    for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
                    {
                        NVSClipmapLevelGPUData& gv = st.cachedViews[lvl];
                        gv.lightId                 = blockIdx;
                        gv.clipmapLevel            = lvl;
                        if ((gv.flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                        {
                            const UINT32 dirMipCount = ComputeNVSMipCountFromPagesPerAxis(gv.pagesPerAxis);
                            ComputeNVSTexelRelativeOrthographicBias(
                                gv.texelWorldSize,
                                gv.casterMinZ,
                                gv.casterMaxZ,
                                sBias,
                                nBias,
                                depthMode,
                                gv.depthBiasNdc,
                                gv.normalBiasWorld);
                            gv.flags |= NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS |
                                        rasterBiasFlags |
                                        (((dirMipCount - 1u) & 0xFu) << NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_SHIFT) |
                                        PackNVSViewDetailTierFlags(dirMinMip, dirLodBias) |
                                        ((st.viewGeneration & 0xFFu) << NVS_CLIPMAP_LEVEL_VIEW_GEN_SHIFT);
                            if (isPerspectiveCam)
                            {
                                const FLOAT32 radialRadius = (lvl + 1u < kNVSMaxClipmapLevels)
                                    ? (0.45f * gv.worldExtent)
                                    : maxDist;
                                gv.flags |= NVS_CLIPMAP_LEVEL_FLAG_RADIAL_SELECT;
                                gv.lightViewToWorld.m[12] = camWorldPos.x;
                                gv.lightViewToWorld.m[13] = camWorldPos.y;
                                gv.lightViewToWorld.m[14] = camWorldPos.z;
                                gv.lightViewToWorld.m[15] = radialRadius;
                            }
                        }
                    }

                    st.scenePtr                = scene;
                    st.direction               = lightDir;
                    st.shadowResolution        = shadowRes;
                    st.shadowNearPlane         = nearPlane;
                    st.shadowMaxDistance       = maxDist;
                    st.shadowBias              = sBias;
                    st.shadowNormalBias        = nBias;
                    st.depthMode               = static_cast<UINT32>(depthMode);
                    st.isPerspectiveCam        = isPerspectiveCam;
                    st.rasterAppliedDepthBias  = sceneRasterReadsDepthBias;
                    st.rasterAppliedNormalBias = sceneRasterReadsNormalBias;
                    st.initialized             = true;
                    st.dataVersion             = ++m_nvsViewTableEpoch;
                }
                st.lastActiveFrame = frameStamp;

                CameraActiveSlotSpan span{};
                span.baseSlot    = baseSlot;
                span.viewCount   = kNVSMaxClipmapLevels;
                span.dataVersion = st.dataVersion;
                span.views       = st.cachedViews;
                cameraActiveSpans.push_back(span);

                LightNVSViewAssignment assign{};
                assign.light           = l;
                assign.shadowViewBase  = baseSlot;
                assign.shadowViewCount = kNVSMaxClipmapLevels;
                assign.nvsLightId      = blockIdx;
                currentAssignments.push_back(assign);
                totalActiveShadowViews += kNVSMaxClipmapLevels;
                activeViewSlotHighWater = (std::max)(activeViewSlotHighWater, baseSlot + kNVSMaxClipmapLevels);
            }

            // Schedule Spot (1 view) and Point (6 views) Lights with per-light incremental caching
            for (const Light* l : shadowLights)
            {
                const LightType lType = l->GetType();
                if (lType != LightType::SPOT && lType != LightType::POINT)
                {
                    continue;
                }

                const UINT64                uid            = l->GetRuntimeUid();
                const UINT32                logicalLightId = logicalLightCounter++;
                const UINT32                reqViews       = (lType == LightType::POINT) ? 6u : 1u;
                const LightShadowResolution shadowRes      = l->GetShadowResolution();
                const Vec3                  pos            = l->GetPosition();
                const Vec3                  dir            = (lType == LightType::SPOT) ? l->GetDirection() : Vec3::Zero();
                const FLOAT32               range          = l->GetRange();
                const FLOAT32               maxDist        = l->GetShadowMaxDistance();
                const FLOAT32               effRange       = (std::min)(range, maxDist);
                const FLOAT32               nearPlane      = l->GetShadowNearPlane();
                const FLOAT32               innerAngle     = (lType == LightType::SPOT) ? l->GetInnerConeAngle() : 0.0f;
                const FLOAT32               outerAngle     = (lType == LightType::SPOT) ? l->GetOuterConeAngle() : 90.0f;
                // Unified 16K virtual pyramid (128x128 pages, 8 mips); shadowRes controls per-light (minMip, lodBias).
                const UINT32                perspPagesAxis = kNVSMaxPagesPerAxis;
                UINT32                      localMinMip    = 0u;
                FLOAT32                     localLodBias   = 0.0f;
                if (lType == LightType::SPOT)
                {
                    ComputeNVSSpotDetailTier(shadowRes, outerAngle * DEG2RAD, localMinMip, localLodBias);
                }
                else
                {
                    ComputeNVSDetailTierFromResolution(shadowRes, localMinMip, localLodBias);
                }
                const FLOAT32               sBias          = l->GetShadowBias();
                const FLOAT32               nBias          = l->GetShadowNormalBias();

                auto locIt = m_nvsLocalLightStates.find(uid);
                if (locIt != m_nvsLocalLightStates.end() && locIt->second.viewCount != reqViews)
                {
                    ReclaimNVSViewSlots(locIt->second.viewSlotBase, locIt->second.viewCount, true);
                    m_nvsLocalLightStates.erase(locIt);
                    locIt = m_nvsLocalLightStates.end();
                }
                if (locIt == m_nvsLocalLightStates.end())
                {
                    NVSLocalLightState newLoc{};
                    newLoc.scenePtr        = scene;
                    newLoc.lightRuntimeUid = uid;
                    newLoc.lightType       = lType;
                    newLoc.viewSlotBase    = AllocateNVSViewSlots(reqViews, false);
                    newLoc.viewCount       = reqViews;
                    newLoc.viewGeneration  = 1u;
                    locIt = m_nvsLocalLightStates.emplace(uid, std::move(newLoc)).first;
                }

                NVSLocalLightState& st = locIt->second;
                const UINT32 baseSlot  = st.viewSlotBase;

                if (st.lastActiveFrame != frameStamp || !st.initialized)
                {
                    const BOOL resChanged = st.initialized &&
                        (st.shadowResolution != shadowRes || st.effectivePagesPerAxis != perspPagesAxis);
                    const BOOL geomChanged =
                        !st.initialized ||
                        st.scenePtr != scene ||
                        st.lightType != lType ||
                        resChanged ||
                        !NearlyEqualVec3(st.position, pos) ||
                        (lType == LightType::SPOT && !NearlyEqualVec3(st.direction, dir)) ||
                        !NearlyEqualFloat(st.range, range, 1e-4f) ||
                        (lType == LightType::SPOT && !NearlyEqualFloat(st.innerConeAngle, innerAngle, 1e-4f)) ||
                        (lType == LightType::SPOT && !NearlyEqualFloat(st.outerConeAngle, outerAngle, 1e-4f)) ||
                        !NearlyEqualFloat(st.shadowNearPlane, nearPlane, 1e-6f) ||
                        !NearlyEqualFloat(st.shadowMaxDistance, maxDist, 1e-4f) ||
                        st.depthMode != static_cast<UINT32>(depthMode) ||
                        !st.enableVirtualMipChain;
                    const BOOL rasterBiasInvalidated =
                        st.initialized &&
                        ((!NearlyEqualFloat(st.shadowBias, sBias, 1e-6f) && (st.rasterAppliedDepthBias || sceneRasterReadsDepthBias)) ||
                         (st.rasterAppliedDepthBias != sceneRasterReadsDepthBias) ||
                         (!NearlyEqualFloat(st.shadowNormalBias, nBias, 1e-6f) && (st.rasterAppliedNormalBias || sceneRasterReadsNormalBias)) ||
                         (st.rasterAppliedNormalBias != sceneRasterReadsNormalBias));
                    const BOOL samplingOnlyBiasChanged =
                        !NearlyEqualFloat(st.shadowBias, sBias, 1e-6f) ||
                        !NearlyEqualFloat(st.shadowNormalBias, nBias, 1e-6f);

                    if (geomChanged || rasterBiasInvalidated)
                    {
                        if (st.initialized)
                        {
                            st.viewGeneration = (st.viewGeneration >= 255u) ? 1u : (st.viewGeneration + 1u);
                            NVSInvalidationCommand invCmd{};
                            invCmd.mode      = resChanged ? NVS_INVALIDATE_RECLAIM_VIEW : NVS_INVALIDATE_VIEW_SLOT;
                            invCmd.viewIndex = baseSlot;
                            invCmd.viewCount = reqViews;
                            m_nvsPageManager.QueueInvalidationCommand(invCmd);
                        }
                    }

                    if (geomChanged || rasterBiasInvalidated || samplingOnlyBiasChanged)
                    {
                        if (lType == LightType::SPOT)
                        {
                            st.cachedViews[0] = BuildNVSSpotShadowViewGPUData(
                                baseSlot,
                                logicalLightId,
                                pos,
                                dir,
                                effRange,
                                innerAngle * DEG2RAD,
                                outerAngle * DEG2RAD,
                                perspPagesAxis,
                                depthMode,
                                sBias,
                                nBias,
                                nearPlane,
                                true,
                                true,
                                st.viewGeneration,
                                true,
                                localMinMip,
                                localLodBias);
                            if ((st.cachedViews[0].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                            {
                                st.cachedViews[0].flags |= rasterBiasFlags;
                            }
                        }
                        else
                        {
                            BuildNVSPointCubeShadowViewsGPUData(
                                baseSlot,
                                logicalLightId,
                                pos,
                                effRange,
                                st.cachedViews,
                                perspPagesAxis,
                                depthMode,
                                sBias,
                                nBias,
                                nearPlane,
                                true,
                                true,
                                st.viewGeneration,
                                true,
                                localMinMip,
                                localLodBias);
                            for (UINT32 f = 0u; f < 6u; ++f)
                            {
                                if ((st.cachedViews[f].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                                {
                                    st.cachedViews[f].flags |= rasterBiasFlags;
                                }
                            }
                        }

                        st.scenePtr                = scene;
                        st.lightType               = lType;
                        st.shadowResolution        = shadowRes;
                        st.effectivePagesPerAxis   = perspPagesAxis;
                        st.position                = pos;
                        st.direction               = dir;
                        st.range                   = range;
                        st.innerConeAngle          = innerAngle;
                        st.outerConeAngle          = outerAngle;
                        st.shadowNearPlane         = nearPlane;
                        st.shadowMaxDistance       = maxDist;
                        st.shadowBias              = sBias;
                        st.shadowNormalBias        = nBias;
                        st.depthMode               = static_cast<UINT32>(depthMode);
                        st.enableVirtualMipChain   = true;
                        st.rasterAppliedDepthBias  = sceneRasterReadsDepthBias;
                        st.rasterAppliedNormalBias = sceneRasterReadsNormalBias;
                        st.initialized             = true;
                        st.dataVersion             = ++m_nvsViewTableEpoch;
                    }
                    st.lastActiveFrame = frameStamp;
                }

                CameraActiveSlotSpan span{};
                span.baseSlot    = baseSlot;
                span.viewCount   = reqViews;
                span.dataVersion = st.dataVersion;
                span.views       = st.cachedViews;
                cameraActiveSpans.push_back(span);

                LightNVSViewAssignment assign{};
                assign.light           = l;
                assign.shadowViewBase  = baseSlot;
                assign.shadowViewCount = reqViews;
                assign.nvsLightId      = baseSlot >> 2u;
                currentAssignments.push_back(assign);
                totalActiveShadowViews += reqViews;
                activeViewSlotHighWater = (std::max)(activeViewSlotHighWater, baseSlot + reqViews);
            }

            if (currentAssignments.empty())
            {
                activeViewSlotHighWater = 0u;
            }
        }
        else
        {
            // Legacy flat non-sparse fallback (< 16 shadow views)
            const UINT32 maxLightsCap      = (std::min)(pageCfg.maxLightCount, kNVSMaxClipmapLights);
            const UINT32 maxTotalViewSlots = maxLightsCap * kNVSMaxClipmapLevels;
            CameraNVSSlotSignature currentSlots[kNVSClipmapTableMaxEntries]{};
            BOOL                   slotOccupied[kNVSClipmapTableMaxEntries]{};
            BOOL                   forceSlotFullInvalidation[kNVSClipmapTableMaxEntries]{};

            for (const Light* l : shadowLights)
            {
                if (l->GetType() != LightType::DIRECTIONAL) continue;
                UINT32 foundBlock = kNVSMaxClipmapLights;
                // Pass 1: reuse block already owned by this (scene, camera, light) tuple across frames
                for (UINT32 b = 0u; b < maxLightsCap; ++b)
                {
                    const UINT32 baseS = b * kNVSMaxClipmapLevels;
                    if (!slotOccupied[baseS] && !slotOccupied[baseS + 1u] &&
                        !slotOccupied[baseS + 2u] && !slotOccupied[baseS + 3u] &&
                        m_globalNVSSlotSignatures[baseS].occupied &&
                        m_globalNVSSlotSignatures[baseS].scenePtr == scene &&
                        m_globalNVSSlotSignatures[baseS].cameraPtr == camera &&
                        m_globalNVSSlotSignatures[baseS].lightRuntimeUid == l->GetRuntimeUid())
                    {
                        foundBlock = b;
                        break;
                    }
                }
                // Pass 2: pick a globally unoccupied block so multiple cameras partition legacy slots cleanly
                if (foundBlock >= maxLightsCap)
                {
                    for (UINT32 b = 0u; b < maxLightsCap; ++b)
                    {
                        const UINT32 baseS = b * kNVSMaxClipmapLevels;
                        if (!slotOccupied[baseS] && !slotOccupied[baseS + 1u] &&
                            !slotOccupied[baseS + 2u] && !slotOccupied[baseS + 3u] &&
                            !m_globalNVSSlotSignatures[baseS].occupied)
                        {
                            foundBlock = b;
                            break;
                        }
                    }
                }
                // Pass 3: fallback to any block not yet claimed in this camera pass
                if (foundBlock >= maxLightsCap)
                {
                    for (UINT32 b = 0u; b < maxLightsCap; ++b)
                    {
                        const UINT32 baseS = b * kNVSMaxClipmapLevels;
                        if (!slotOccupied[baseS] && !slotOccupied[baseS + 1u] &&
                            !slotOccupied[baseS + 2u] && !slotOccupied[baseS + 3u])
                        {
                            foundBlock = b;
                            break;
                        }
                    }
                }
                if (foundBlock >= maxLightsCap) continue;

                const UINT32                baseSlot        = foundBlock * kNVSMaxClipmapLevels;
                const UINT32                logicalLightId  = logicalLightCounter++;
                const Vec3                  lightDir        = l->GetDirection();
                const LightShadowResolution shadowRes       = l->GetShadowResolution();
                const FLOAT32               maxDist         = (std::max)(0.05f, l->GetShadowMaxDistance());
                const FLOAT32               nearPlane       = l->GetShadowNearPlane();
                const FLOAT32               nearPlaneOffset = nearPlane - 0.1f;
                FLOAT32 basePageWorld    = 2.0f;
                FLOAT32 levelScaleFactor = 2.0f;
                ComputeNVSDirectionalClipmapScalingFromResolution(shadowRes, maxDist, pagesPerAxis, basePageWorld, levelScaleFactor);

                NVSClipmapHierarchyState& hier        = camState.directionalHierarchies[foundBlock];
                const CameraNVSSlotSignature& prevSig = m_globalNVSSlotSignatures[baseSlot];
                const BOOL sameDirOwnerAndBasis =
                    camState.directionalInitialized[foundBlock] &&
                    prevSig.occupied &&
                    prevSig.scenePtr == scene &&
                    prevSig.cameraPtr == camera &&
                    prevSig.lightRuntimeUid == l->GetRuntimeUid() &&
                    prevSig.lightType == LightType::DIRECTIONAL &&
                    prevSig.shadowResolution == shadowRes &&
                    NearlyEqualVec3(prevSig.direction, lightDir) &&
                    NearlyEqualFloat(prevSig.shadowNearPlane, nearPlane, 1e-6f) &&
                    NearlyEqualFloat(prevSig.shadowMaxDistance, maxDist, 1e-4f) &&
                    prevSig.depthMode == static_cast<UINT32>(depthMode) &&
                    hier.pagesPerAxis == pagesPerAxis;

                if (!sameDirOwnerAndBasis)
                {
                    NVSOriginShiftResult initShifts[kNVSMaxClipmapLevels]{};
                    ComputeNVSDirectionalClipmapHierarchy(
                        hier, camWorldPos, lightDir, foundBlock, basePageWorld, pagesPerAxis,
                        kNVSMaxClipmapLevels, maxDist, depthMode, 0.0f, initShifts, levelScaleFactor, nearPlaneOffset);
                    camState.directionalInitialized[foundBlock] = true;
                    for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
                    {
                        if (m_globalNVSSlotSignatures[baseSlot + lvl].occupied)
                            forceSlotFullInvalidation[baseSlot + lvl] = true;
                    }
                }
                else
                {
                    NVSOriginShiftResult l0Shift{};
                    NVSOriginShiftResult allShifts[kNVSMaxClipmapLevels]{};
                    UpdateNVSClipmapHierarchySnappedOrigins(hier, camWorldPos, l0Shift, allShifts);
                    for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
                    {
                        if (allShifts[lvl].depthRangeChanged || allShifts[lvl].fullWindowReset ||
                            allShifts[lvl].deltaPageX != 0 || allShifts[lvl].deltaPageY != 0)
                        {
                            OriginShiftInvalidation osi{};
                            osi.lightId            = foundBlock;
                            osi.clipmapLevel       = lvl;
                            osi.deltaPageX         = allShifts[lvl].deltaPageX;
                            osi.deltaPageY         = allShifts[lvl].deltaPageY;
                            osi.snappedOriginPageX = allShifts[lvl].snappedOriginPageX;
                            osi.snappedOriginPageY = allShifts[lvl].snappedOriginPageY;
                            osi.fullLevelReset     = allShifts[lvl].depthRangeChanged || allShifts[lvl].fullWindowReset;
                            legacyOriginShifts.push_back(osi);
                        }
                    }
                }

                BuildNVSClipmapHierarchyGPUData(hier, legacyViewTable);
                for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
                {
                    const UINT32 s  = baseSlot + lvl;
                    slotOccupied[s] = true;
                    if ((legacyViewTable[s].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                    {
                        ComputeNVSTexelRelativeOrthographicBias(
                            legacyViewTable[s].texelWorldSize,
                            legacyViewTable[s].casterMinZ,
                            legacyViewTable[s].casterMaxZ,
                            l->GetShadowBias(),
                            l->GetShadowNormalBias(),
                            depthMode,
                            legacyViewTable[s].depthBiasNdc,
                            legacyViewTable[s].normalBiasWorld);
                        legacyViewTable[s].flags |= NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS |
                                                    rasterBiasFlags |
                                                    ((logicalLightId & 0x7Fu) << NVS_CLIPMAP_LEVEL_LOGICAL_LIGHT_SHIFT);
                        if (isPerspectiveCam)
                        {
                            const FLOAT32 radialRadius = (lvl + 1u < kNVSMaxClipmapLevels) ? (0.45f * legacyViewTable[s].worldExtent) : maxDist;
                            legacyViewTable[s].flags |= NVS_CLIPMAP_LEVEL_FLAG_RADIAL_SELECT;
                            legacyViewTable[s].lightViewToWorld.m[12] = camWorldPos.x;
                            legacyViewTable[s].lightViewToWorld.m[13] = camWorldPos.y;
                            legacyViewTable[s].lightViewToWorld.m[14] = camWorldPos.z;
                            legacyViewTable[s].lightViewToWorld.m[15] = radialRadius;
                        }
                    }
                    CameraNVSSlotSignature sig{};
                    sig.occupied              = true;
                    sig.scenePtr              = scene;
                    sig.cameraPtr             = camera;
                    sig.lightPtr              = l;
                    sig.lightRuntimeUid       = l->GetRuntimeUid();
                    sig.lightType             = LightType::DIRECTIONAL;
                    sig.shadowResolution      = shadowRes;
                    sig.direction             = lightDir;
                    sig.shadowNearPlane       = nearPlane;
                    sig.shadowMaxDistance     = maxDist;
                    sig.shadowBias            = l->GetShadowBias();
                    sig.shadowNormalBias      = l->GetShadowNormalBias();
                    sig.depthMode             = static_cast<UINT32>(depthMode);
                    sig.rasterReadsDepthBias  = sceneRasterReadsDepthBias;
                    sig.rasterReadsNormalBias = sceneRasterReadsNormalBias;
                    currentSlots[s]           = sig;
                }

                LightNVSViewAssignment assign{};
                assign.light           = l;
                assign.shadowViewBase  = baseSlot;
                assign.shadowViewCount = kNVSMaxClipmapLevels;
                assign.nvsLightId      = foundBlock;
                currentAssignments.push_back(assign);
                totalActiveShadowViews += kNVSMaxClipmapLevels;
                activeViewSlotHighWater = (std::max)(activeViewSlotHighWater, baseSlot + kNVSMaxClipmapLevels);
            }

            for (const Light* l : shadowLights)
            {
                if (l->GetType() == LightType::SPOT)
                {
                    UINT32 freeSlot = maxTotalViewSlots;
                    for (UINT32 s = 0u; s < maxTotalViewSlots; ++s)
                    {
                        if (!slotOccupied[s]) { freeSlot = s; break; }
                    }
                    if (freeSlot >= maxTotalViewSlots) continue;

                    const UINT32                logicalLightId = logicalLightCounter++;
                    const LightShadowResolution shadowRes      = l->GetShadowResolution();
                    const UINT32                spotPagesAxis  = ComputeNVSSpotVirtualPagesPerAxis(shadowRes, l->GetOuterConeAngle(), pagesPerAxis);
                    const FLOAT32               effRange       = (std::min)(l->GetRange(), l->GetShadowMaxDistance());

                    legacyViewTable[freeSlot] = BuildNVSSpotShadowViewGPUData(
                        freeSlot, logicalLightId, l->GetPosition(), l->GetDirection(), effRange,
                        l->GetInnerConeAngle() * DEG2RAD, l->GetOuterConeAngle() * DEG2RAD,
                        spotPagesAxis, depthMode, l->GetShadowBias(), l->GetShadowNormalBias(),
                        l->GetShadowNearPlane(), false, true, 1u);
                    if ((legacyViewTable[freeSlot].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                    {
                        legacyViewTable[freeSlot].flags |= rasterBiasFlags;
                    }
                    slotOccupied[freeSlot] = true;

                    CameraNVSSlotSignature sig{};
                    sig.occupied              = true;
                    sig.scenePtr              = scene;
                    sig.cameraPtr             = camera;
                    sig.lightPtr              = l;
                    sig.lightRuntimeUid       = l->GetRuntimeUid();
                    sig.lightType             = LightType::SPOT;
                    sig.shadowResolution      = shadowRes;
                    sig.direction             = l->GetDirection();
                    sig.position              = l->GetPosition();
                    sig.range                 = l->GetRange();
                    sig.innerConeAngle        = l->GetInnerConeAngle();
                    sig.outerConeAngle        = l->GetOuterConeAngle();
                    sig.shadowNearPlane       = l->GetShadowNearPlane();
                    sig.shadowMaxDistance     = l->GetShadowMaxDistance();
                    sig.shadowBias            = l->GetShadowBias();
                    sig.shadowNormalBias      = l->GetShadowNormalBias();
                    sig.depthMode             = static_cast<UINT32>(depthMode);
                    sig.rasterReadsDepthBias  = sceneRasterReadsDepthBias;
                    sig.rasterReadsNormalBias = sceneRasterReadsNormalBias;
                    currentSlots[freeSlot]    = sig;

                    LightNVSViewAssignment assign{};
                    assign.light           = l;
                    assign.shadowViewBase  = freeSlot;
                    assign.shadowViewCount = 1u;
                    assign.nvsLightId      = freeSlot >> 2u;
                    currentAssignments.push_back(assign);
                    totalActiveShadowViews += 1u;
                    activeViewSlotHighWater = (std::max)(activeViewSlotHighWater, freeSlot + 1u);
                }
                else if (l->GetType() == LightType::POINT)
                {
                    if (maxTotalViewSlots < 6u) continue;
                    UINT32 freeBase = maxTotalViewSlots;
                    for (UINT32 s = 0u; s + 6u <= maxTotalViewSlots; ++s)
                    {
                        BOOL allFree = true;
                        for (UINT32 f = 0u; f < 6u; ++f)
                        {
                            if (slotOccupied[s + f]) { allFree = false; break; }
                        }
                        if (allFree) { freeBase = s; break; }
                    }
                    if (freeBase >= maxTotalViewSlots) continue;

                    const UINT32                logicalLightId = logicalLightCounter++;
                    const LightShadowResolution shadowRes      = l->GetShadowResolution();
                    const UINT32                pointPagesAxis = ComputeNVSPerspectivePagesPerAxisFromResolution(shadowRes, pagesPerAxis);
                    const FLOAT32               effRange       = (std::min)(l->GetRange(), l->GetShadowMaxDistance());

                    NVSClipmapLevelGPUData cubeViews[6]{};
                    BuildNVSPointCubeShadowViewsGPUData(
                        freeBase, logicalLightId, l->GetPosition(), effRange, cubeViews,
                        pointPagesAxis, depthMode, l->GetShadowBias(), l->GetShadowNormalBias(),
                        l->GetShadowNearPlane(), false, true, 1u);

                    for (UINT32 f = 0u; f < 6u; ++f)
                    {
                        const UINT32 s     = freeBase + f;
                        legacyViewTable[s] = cubeViews[f];
                        if ((legacyViewTable[s].flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u)
                        {
                            legacyViewTable[s].flags |= rasterBiasFlags;
                        }
                        slotOccupied[s] = true;

                        CameraNVSSlotSignature sig{};
                        sig.occupied              = true;
                        sig.scenePtr              = scene;
                        sig.cameraPtr             = camera;
                        sig.lightPtr              = l;
                        sig.lightRuntimeUid       = l->GetRuntimeUid();
                        sig.lightType             = LightType::POINT;
                        sig.shadowResolution      = shadowRes;
                        sig.position              = l->GetPosition();
                        sig.range                 = l->GetRange();
                        sig.shadowNearPlane       = l->GetShadowNearPlane();
                        sig.shadowMaxDistance     = l->GetShadowMaxDistance();
                        sig.shadowBias            = l->GetShadowBias();
                        sig.shadowNormalBias      = l->GetShadowNormalBias();
                        sig.depthMode             = static_cast<UINT32>(depthMode);
                        sig.rasterReadsDepthBias  = sceneRasterReadsDepthBias;
                        sig.rasterReadsNormalBias = sceneRasterReadsNormalBias;
                        currentSlots[s]           = sig;
                    }

                    LightNVSViewAssignment assign{};
                    assign.light           = l;
                    assign.shadowViewBase  = freeBase;
                    assign.shadowViewCount = 6u;
                    assign.nvsLightId      = freeBase >> 2u;
                    currentAssignments.push_back(assign);
                    totalActiveShadowViews += 6u;
                    activeViewSlotHighWater = (std::max)(activeViewSlotHighWater, freeBase + 6u);
                }
            }

            for (UINT32 s = 0u; s < maxTotalViewSlots; ++s)
            {
                const CameraNVSSlotSignature& prev = m_globalNVSSlotSignatures[s];
                const CameraNVSSlotSignature& curr = currentSlots[s];
                if (prev.occupied)
                {
                    const BOOL depthBiasRequiresRasterInvalidation =
                        (!NearlyEqualFloat(prev.shadowBias, curr.shadowBias, 1e-6f) &&
                         (prev.rasterReadsDepthBias || curr.rasterReadsDepthBias)) ||
                        (prev.rasterReadsDepthBias != curr.rasterReadsDepthBias);
                    const BOOL normalBiasRequiresRasterInvalidation =
                        (!NearlyEqualFloat(prev.shadowNormalBias, curr.shadowNormalBias, 1e-6f) &&
                         (prev.rasterReadsNormalBias || curr.rasterReadsNormalBias)) ||
                        (prev.rasterReadsNormalBias != curr.rasterReadsNormalBias);

                    const BOOL changed =
                        !curr.occupied ||
                        prev.scenePtr != curr.scenePtr ||
                        prev.cameraPtr != curr.cameraPtr ||
                        prev.lightRuntimeUid != curr.lightRuntimeUid ||
                        prev.lightType != curr.lightType ||
                        prev.shadowResolution != curr.shadowResolution ||
                        !NearlyEqualVec3(prev.direction, curr.direction) ||
                        !NearlyEqualVec3(prev.position, curr.position) ||
                        !NearlyEqualFloat(prev.range, curr.range, 1e-4f) ||
                        !NearlyEqualFloat(prev.innerConeAngle, curr.innerConeAngle, 1e-4f) ||
                        !NearlyEqualFloat(prev.outerConeAngle, curr.outerConeAngle, 1e-4f) ||
                        !NearlyEqualFloat(prev.shadowNearPlane, curr.shadowNearPlane, 1e-6f) ||
                        !NearlyEqualFloat(prev.shadowMaxDistance, curr.shadowMaxDistance, 1e-4f) ||
                        depthBiasRequiresRasterInvalidation ||
                        normalBiasRequiresRasterInvalidation ||
                        prev.depthMode != curr.depthMode ||
                        forceSlotFullInvalidation[s];
                    if (changed)
                    {
                        legacyInvalidatedSlots.push_back(s);
                    }
                }
                if (!curr.occupied && (s % kNVSMaxClipmapLevels) == 0u && (s / kNVSMaxClipmapLevels) < kNVSMaxClipmapLights)
                {
                    camState.directionalInitialized[s / kNVSMaxClipmapLevels] = false;
                }
                m_globalNVSSlotSignatures[s] = curr;
            }
        }

        const UINT32 currentFrame = static_cast<UINT32>((std::max)(1ull, m_stats.frameIndex));
        const BOOL shouldExecuteNVSPipeline =
            (camera->GetSceneDepthTexture() != nullptr) &&
            (!currentAssignments.empty() && activeViewSlotHighWater > 0u) &&
            (shadowCasterCount > 0u || sceneHasNVSReceiver);

        if (!shouldExecuteNVSPipeline)
        {
            m_nvsPageManager.SetActiveProjectionFrame(currentFrame, camera);
            if (useSparseHierarchy && (!camState.uploadedSlotVersion.empty() || !camState.cameraViewTableInitialized))
            {
                const UINT32 prevCap = (std::max)(
                    static_cast<UINT32>(camState.uploadedSlotVersion.size()),
                    kNVSClipmapTableMaxEntries);
                std::vector<NVSClipmapLevelGPUData> zeroBlock(prevCap, NVSClipmapLevelGPUData{});
                m_nvsPageManager.UpdateClipmapLevelTableRange(zeroBlock.data(), 0u, prevCap, 0u, camera);
                m_nvsPageManager.FlushViewTableToGPU(camera);
                m_nvsPageManager.CommitSamplingViewTableSnapshot(currentFrame, camera);
                camState.uploadedSlotVersion.clear();
                camState.cameraViewTableInitialized = true;
            }

            FrameNVSSamplingContext idleNvsCtx{};
            PopulateBaseNVSSamplingContext(depthMode, idleNvsCtx);
            idleNvsCtx.screenPixelWorldScale = ComputeCameraNVSScreenPixelWorldScale(
                camera,
                idleNvsCtx.sparsePageHierarchyBuffer != 0 && idleNvsCtx.sparseRootCapacity > 0u);
            camState.committedLightAssignments.clear();
            camState.committedActiveViewCount     = 0u;
            camState.hadActivePipelineLastFrame   = false;
            camState.adaptiveLodBias              = 0.0f;
            camState.adaptiveLowPressureSamples   = 0u;
            camState.adaptiveBiasLastChangeFrame  = 0u;
            camState.lastFeedbackUniquePages      = 0u;
            camState.feedbackReadbackRequestFrame = 0u;
            camState.feedbackReadbackInFlight     = false;

            m_renderPipeline.BuildFrameView(
                scene, camera, m_renderTarget, m_stats, m_frameView, &idleNvsCtx, &m_executedPerFrameKernels, &m_internalResourceManager);
            m_stats.shadowCasters    = shadowCasterCount;
            m_stats.nvsMetadataVramMB = static_cast<FLOAT32>(
                static_cast<FLOAT64>(m_nvsPageManager.GetMetadataVramBytes() + m_nvsFeedbackManager.GetMetadataVramBytes()) / (1024.0 * 1024.0));

            std::vector<ComputeDispatchCommand> cleanupCmds;
            if (useSparseHierarchy)
            {
                m_nvsPageManager.BuildPendingInvalidationDispatches(currentFrame, cleanupCmds, camera);
            }
            else
            {
                for (UINT32 s : legacyInvalidatedSlots)
                {
                    NVSResidencyPassDesc invDesc{};
                    invDesc.currentFrame           = currentFrame;
                    invDesc.invalidationMode       = NVS_INVALIDATE_CLIPMAP_LEVEL;
                    invDesc.invalidateLightId      = s >> 2u;
                    invDesc.invalidateClipmapLevel = s & 3u;
                    invDesc.executeSelectiveClear  = false;
                    invDesc.markClearedAfterPass   = false;
                    invDesc.executePageProjection  = false;
                    ComputeDispatchCommand invCmd{};
                    if (m_nvsPageManager.BuildResidencyComputeDispatch(
                            NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE, invDesc, invCmd, nullptr, camera))
                    {
                        cleanupCmds.push_back(std::move(invCmd));
                    }
                }
            }
            if (!cleanupCmds.empty())
            {
                m_frameView.prePassComputeDispatches.insert(
                    m_frameView.prePassComputeDispatches.begin(),
                    std::make_move_iterator(cleanupCmds.begin()),
                    std::make_move_iterator(cleanupCmds.end()));
            }
            return;
        }

        // 3. Ensure dynamic GPU view-table capacity and non-blocking demand-driven sparse pool growth
        const UINT32 requiredTableCap = (std::max)(activeViewSlotHighWater, kNVSClipmapTableMaxEntries);
        m_nvsPageManager.EnsureViewTableCapacity(requiredTableCap, camera);
        m_nvsPageManager.SetActiveProjectionFrame(currentFrame, camera);
        if (useSparseHierarchy)
        {
            m_nvsPageManager.PollSparsePoolHeadroomAndGrowIfNeeded();

            // Non-blocking GPU pool-pressure adaptive LOD bias controller:
            // Keeps active multi-light virtual page working set safely inside the physical pool
            // with quantized 0.25-mip steps and hysteresis (zero CPU-GPU stalls, zero oscillation).
            UINT32 residencyOverflowPages = 0u;
            UINT32 residencyDirtySlices   = 0u;
            BOOL   residencyFresh         = false;
            m_nvsPageManager.PollResidencyOverflowNonBlocking(residencyOverflowPages, residencyDirtySlices, residencyFresh);

            BOOL feedbackFresh = false;
            GPUBuffer* fbTelBuf = m_nvsFeedbackManager.GetFeedbackTelemetryBuffer(camera);
            if (fbTelBuf && fbTelBuf->IsValid())
            {
                if (camState.feedbackReadbackInFlight)
                {
                    NVSFeedbackTelemetry fbTel{};
                    if (m_bufferManager.TryGetGPUBufferReadbackData(
                            fbTelBuf,
                            &fbTel,
                            static_cast<UINT32>(sizeof(fbTel))))
                    {
                        camState.feedbackReadbackInFlight = false;
                        if (camState.feedbackReadbackRequestFrame >= camState.adaptiveBiasLastChangeFrame)
                        {
                            camState.lastFeedbackUniquePages = fbTel.uniquePageRequestCount;
                            feedbackFresh                    = true;
                        }
                    }
                }
                if (!camState.feedbackReadbackInFlight)
                {
                    if (m_bufferManager.RequestGPUBufferReadback(
                            fbTelBuf,
                            0u,
                            static_cast<UINT32>(sizeof(NVSFeedbackTelemetry))))
                    {
                        camState.feedbackReadbackInFlight     = true;
                        camState.feedbackReadbackRequestFrame = currentFrame;
                    }
                }
            }

            if (feedbackFresh || (residencyFresh && currentFrame >= (camState.adaptiveBiasLastChangeFrame + 4u)))
            {
                const UINT32 poolCap      = (std::max)(m_nvsPageManager.GetPhysicalPoolCapacity(), 64u);
                const UINT32 highWater    = (poolCap * 78u) / 100u;
                const UINT32 targetPages  = (std::max)((poolCap * 65u) / 100u, 1u);
                const UINT32 lowWater     = (poolCap * 48u) / 100u;
                const UINT32 demanded     = camState.lastFeedbackUniquePages;
                const UINT32 overflowCnt  = residencyFresh ? residencyOverflowPages : 0u;
                const BOOL   settled      = (currentFrame >= (camState.adaptiveBiasLastChangeFrame + 3u));

                if (settled && (overflowCnt > 0u || demanded > highWater))
                {
                    const UINT32  effDemand = (std::max)(demanded, (overflowCnt > 0u) ? (poolCap + overflowCnt) : demanded);
                    const FLOAT32 ratio     = static_cast<FLOAT32>(effDemand) / static_cast<FLOAT32>(targetPages);
                    const FLOAT32 rawDelta  = 0.5f * std::log2((std::max)(ratio, 1.0f));
                    const FLOAT32 quantStep = (std::clamp)(std::ceil(rawDelta * 4.0f) * 0.25f, 0.25f, 1.5f);
                    const FLOAT32 nextBias  = (std::min)(camState.adaptiveLodBias + quantStep, 4.0f);
                    if (nextBias > camState.adaptiveLodBias + 1e-4f)
                    {
                        camState.adaptiveLodBias             = nextBias;
                        camState.adaptiveBiasLastChangeFrame = currentFrame;
                    }
                    camState.adaptiveLowPressureSamples = 0u;
                }
                else if (settled && feedbackFresh && camState.adaptiveLodBias > 0.0f &&
                         overflowCnt == 0u && demanded > 0u && demanded < lowWater)
                {
                    camState.adaptiveLowPressureSamples++;
                    if (camState.adaptiveLowPressureSamples >= 10u)
                    {
                        camState.adaptiveLodBias             = (std::max)(0.0f, camState.adaptiveLodBias - 0.25f);
                        camState.adaptiveBiasLastChangeFrame = currentFrame;
                        camState.adaptiveLowPressureSamples  = 0u;
                    }
                }
                else if (feedbackFresh)
                {
                    camState.adaptiveLowPressureSamples = 0u;
                }
            }
        }

        camState.hadActivePipelineLastFrame = true;

        // 4. Upload dirty view ranges incrementally per-(Scene, Camera) in sparse mode (or 16-entry table in legacy mode)
        if (useSparseHierarchy)
        {
            if (!camState.cameraViewTableInitialized)
            {
                camState.uploadedSlotVersion.assign(
                    static_cast<SIZE_T>((std::max)(activeViewSlotHighWater, kNVSClipmapTableMaxEntries)),
                    ~0ull);
                camState.cameraViewTableInitialized = true;
            }
            const UINT32 prevSlotCap = static_cast<UINT32>(camState.uploadedSlotVersion.size());
            const UINT32 trackCap    = (std::max)(activeViewSlotHighWater, prevSlotCap);
            if (camState.uploadedSlotVersion.size() < static_cast<SIZE_T>(activeViewSlotHighWater))
            {
                camState.uploadedSlotVersion.resize(static_cast<SIZE_T>(activeViewSlotHighWater), 0ull);
            }

            std::vector<UINT8> slotStillActive(static_cast<SIZE_T>(trackCap), 0u);
            BOOL anyRangeUploaded = false;

            for (const CameraActiveSlotSpan& span : cameraActiveSpans)
            {
                BOOL spanDirty = false;
                for (UINT32 k = 0u; k < span.viewCount; ++k)
                {
                    const UINT32 slotIdx = span.baseSlot + k;
                    slotStillActive[slotIdx] = 1u;
                    if (camState.uploadedSlotVersion[slotIdx] != span.dataVersion)
                    {
                        spanDirty = true;
                    }
                }
                if (spanDirty && span.views != nullptr)
                {
                    m_nvsPageManager.UpdateClipmapLevelTableRange(
                        span.views,
                        span.baseSlot,
                        span.viewCount,
                        activeViewSlotHighWater,
                        camera);
                    for (UINT32 k = 0u; k < span.viewCount; ++k)
                    {
                        camState.uploadedSlotVersion[span.baseSlot + k] = span.dataVersion;
                    }
                    anyRangeUploaded = true;
                }
            }

            // Deterministically zero any slots previously valid on this camera that are not active in (scene, camera)
            std::vector<NVSClipmapLevelGPUData> zeroScratch;
            UINT32 s = 0u;
            while (s < prevSlotCap)
            {
                if (camState.uploadedSlotVersion[s] != 0ull && slotStillActive[s] == 0u)
                {
                    const UINT32 runStart = s;
                    while (s < prevSlotCap && camState.uploadedSlotVersion[s] != 0ull && slotStillActive[s] == 0u)
                    {
                        camState.uploadedSlotVersion[s] = 0ull;
                        ++s;
                    }
                    const UINT32 runCount = s - runStart;
                    if (zeroScratch.size() < static_cast<SIZE_T>(runCount))
                    {
                        zeroScratch.assign(static_cast<SIZE_T>(runCount), NVSClipmapLevelGPUData{});
                    }
                    m_nvsPageManager.UpdateClipmapLevelTableRange(
                        zeroScratch.data(),
                        runStart,
                        runCount,
                        activeViewSlotHighWater,
                        camera);
                    anyRangeUploaded = true;
                }
                else
                {
                    ++s;
                }
            }

            while (!camState.uploadedSlotVersion.empty() && camState.uploadedSlotVersion.back() == 0ull)
            {
                camState.uploadedSlotVersion.pop_back();
            }

            if (!anyRangeUploaded && camState.committedActiveViewCount != activeViewSlotHighWater)
            {
                m_nvsPageManager.UpdateClipmapLevelTableRange(nullptr, 0u, 0u, activeViewSlotHighWater, camera);
            }
            m_nvsPageManager.FlushViewTableToGPU(camera);
        }
        else
        {
            m_nvsPageManager.UploadClipmapLevelTable(legacyViewTable, kNVSClipmapTableMaxEntries, camera);
        }

        m_nvsPageManager.CommitSamplingViewTableSnapshot(currentFrame, camera);
        GPUBuffer* newCommittedBuf         = m_nvsPageManager.GetCommittedSamplingViewTableBuffer(camera);
        const auto* newCommittedCpuPtr     = m_nvsPageManager.GetCommittedCpuViewTable(camera);
        const UINT32 newCommittedCpuCount  = (std::max)(m_nvsPageManager.GetCommittedViewTableCount(camera), kNVSClipmapTableMaxEntries);
        camState.committedLightAssignments = currentAssignments;
        camState.committedActiveViewCount  = activeViewSlotHighWater;

        // 5. Build unified same-frame FrameNVSSamplingContext (engine automatically executes DEPTH_PREPASS -> POST_DEPTH NVS -> Opaque)
        FrameNVSSamplingContext nvsCtx{};
        PopulateBaseNVSSamplingContext(depthMode, nvsCtx);
        nvsCtx.requiresDepthPrepass    = true;
        nvsCtx.screenPixelWorldScale   = ComputeCameraNVSScreenPixelWorldScale(
            camera,
            nvsCtx.sparsePageHierarchyBuffer != 0 && nvsCtx.sparseRootCapacity > 0u);
        nvsCtx.screenPixelWorldScale.z          = camState.adaptiveLodBias;
        nvsCtx.opaqueLightViewTableBuffer       = newCommittedBuf ? newCommittedBuf->GetNativeHandle() : 0;
        nvsCtx.opaqueLightViewTableElementCount = newCommittedBuf ? newCommittedBuf->GetElementCount() : 0u;
        nvsCtx.opaqueActiveViewCount            = activeViewSlotHighWater;
        nvsCtx.opaqueSamplingAvailable          = (newCommittedBuf != nullptr) ? 1u : 0u;
        nvsCtx.opaqueSnapshotFrameIndex         = currentFrame;
        nvsCtx.opaqueViewTableCpuMirror         = newCommittedCpuPtr;
        nvsCtx.opaqueViewTableCpuMirrorCount    = newCommittedCpuCount;
        nvsCtx.opaqueLightAssignments           = currentAssignments;

        const auto tPrepPause = std::chrono::high_resolution_clock::now();
        const FLOAT32 preBuildPrepMs = std::chrono::duration<FLOAT32, std::milli>(tPrepPause - tPrepStart).count();

        // 6. Build FrameView with exact Opaque & Post-Opaque NVS sampling states
        m_renderPipeline.BuildFrameView(
            scene, camera, m_renderTarget, m_stats, m_frameView, &nvsCtx, &m_executedPerFrameKernels, &m_internalResourceManager);

        const auto tInjectStart = std::chrono::high_resolution_clock::now();
        m_stats.shadowCasters             += shadowCasterCount;
        const UINT64 logicalTiles64        = static_cast<UINT64>(m_stats.logicalTileCount) +
                                             static_cast<UINT64>(totalActiveShadowViews) * static_cast<UINT64>(pagesPerAxis) * static_cast<UINT64>(pagesPerAxis);
        m_stats.logicalTileCount           = (logicalTiles64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<UINT32>(logicalTiles64);
        m_stats.physicalTileCount          = m_nvsPageManager.GetPhysicalPoolCapacity();
        m_stats.shadowPoolVramMB           = static_cast<FLOAT32>(m_nvsPageManager.GetPhysicalPoolCapacity()) * 0.0625f;
        m_stats.nvsActiveShadowLightCount  = (std::max)(m_stats.nvsActiveShadowLightCount, static_cast<UINT32>(currentAssignments.size()));
        m_stats.nvsActiveShadowViewCount  += totalActiveShadowViews;
        m_stats.nvsMetadataVramMB          = static_cast<FLOAT32>(
            static_cast<FLOAT64>(m_nvsPageManager.GetMetadataVramBytes() + m_nvsFeedbackManager.GetMetadataVramBytes()) / (1024.0 * 1024.0));

        // 7. Build slot/origin & dynamic/skinned caster invalidation dispatches
        std::vector<ComputeDispatchCommand> slotAndOriginInvCmds;
        if (useSparseHierarchy)
        {
            m_nvsPageManager.BuildPendingInvalidationDispatches(currentFrame, slotAndOriginInvCmds, camera);
        }
        else
        {
            for (UINT32 s : legacyInvalidatedSlots)
            {
                NVSResidencyPassDesc invDesc{};
                invDesc.currentFrame           = currentFrame;
                invDesc.invalidationMode       = NVS_INVALIDATE_CLIPMAP_LEVEL;
                invDesc.invalidateLightId      = s >> 2u;
                invDesc.invalidateClipmapLevel = s & 3u;
                invDesc.executeSelectiveClear  = false;
                invDesc.markClearedAfterPass   = false;
                invDesc.executePageProjection  = false;

                ComputeDispatchCommand invCmd{};
                if (m_nvsPageManager.BuildResidencyComputeDispatch(
                        NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE, invDesc, invCmd, nullptr, camera))
                {
                    slotAndOriginInvCmds.push_back(std::move(invCmd));
                }
            }

            for (const auto& osi : legacyOriginShifts)
            {
                NVSResidencyPassDesc shiftDesc{};
                shiftDesc.currentFrame           = currentFrame;
                shiftDesc.invalidationMode       = osi.fullLevelReset ? NVS_INVALIDATE_CLIPMAP_LEVEL : NVS_INVALIDATE_ORIGIN_SHIFT;
                shiftDesc.invalidateLightId      = osi.lightId;
                shiftDesc.invalidateClipmapLevel = osi.clipmapLevel;
                shiftDesc.originShiftDeltaX      = osi.deltaPageX;
                shiftDesc.originShiftDeltaY      = osi.deltaPageY;
                shiftDesc.snappedOriginPageX     = osi.snappedOriginPageX;
                shiftDesc.snappedOriginPageY     = osi.snappedOriginPageY;
                shiftDesc.executeSelectiveClear  = false;
                shiftDesc.markClearedAfterPass   = false;
                shiftDesc.executePageProjection  = false;

                ComputeDispatchCommand shiftCmd{};
                if (m_nvsPageManager.BuildResidencyComputeDispatch(
                        NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE, shiftDesc, shiftCmd, nullptr, camera))
                {
                    slotAndOriginInvCmds.push_back(std::move(shiftCmd));
                }
            }
        }

        BOOL ranDynamicInvalidation = false;
        std::vector<ComputeDispatchCommand> dynInvCmds;
        if (!frameInvalidationBoxes.empty())
        {
            std::vector<NVSDynamicCasterInvalidationBox> boundedBoxes;
            const NVSDynamicCasterInvalidationBox* uploadPtr = frameInvalidationBoxes.data();
            UINT32 uploadCnt = static_cast<UINT32>(frameInvalidationBoxes.size());
            if (uploadCnt > kNVSMaxDynamicInvalidationBoxes)
            {
                // Coalesce excess invalidation boxes into a conservative union box in the last slot so zero invalidations are lost
                boundedBoxes.assign(
                    frameInvalidationBoxes.begin(),
                    frameInvalidationBoxes.begin() + static_cast<std::ptrdiff_t>(kNVSMaxDynamicInvalidationBoxes));
                NVSDynamicCasterInvalidationBox& merged = boundedBoxes.back();
                merged.gpuSceneSlotIndex = 0xFFFFFFFFu;
                for (SIZE_T bi = kNVSMaxDynamicInvalidationBoxes; bi < frameInvalidationBoxes.size(); ++bi)
                {
                    const NVSDynamicCasterInvalidationBox& ex = frameInvalidationBoxes[bi];
                    merged.worldMin.x = (std::min)(merged.worldMin.x, ex.worldMin.x);
                    merged.worldMin.y = (std::min)(merged.worldMin.y, ex.worldMin.y);
                    merged.worldMin.z = (std::min)(merged.worldMin.z, ex.worldMin.z);
                    merged.worldMax.x = (std::max)(merged.worldMax.x, ex.worldMax.x);
                    merged.worldMax.y = (std::max)(merged.worldMax.y, ex.worldMax.y);
                    merged.worldMax.z = (std::max)(merged.worldMax.z, ex.worldMax.z);
                    merged.invalidationFlags |= ex.invalidationFlags;
                }
                uploadPtr = boundedBoxes.data();
                uploadCnt = kNVSMaxDynamicInvalidationBoxes;
            }

            if (m_nvsPageManager.UploadDynamicCasterInvalidationBoxes(uploadPtr, uploadCnt))
            {
                NVSCasterCullPassDesc preInvDesc{};
                preInvDesc.currentFrame = currentFrame;
                const UINT32 boxCount   = uploadCnt;

                GPUBuffer* primaryInvViewBuf = m_nvsPageManager.GetClipmapLevelTableBuffer(camera);

                ComputeDispatchCommand preResetCmd{};
                ComputeDispatchCommand dynInvCmd{};
                if (m_nvsPageManager.BuildCasterCullComputeDispatch(
                        NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES,
                        nullptr,
                        0u,
                        preInvDesc,
                        preResetCmd,
                        camera,
                        0u,
                        primaryInvViewBuf) &&
                    m_nvsPageManager.BuildCasterCullComputeDispatch(
                        NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION,
                        nullptr,
                        0u,
                        preInvDesc,
                        dynInvCmd,
                        camera,
                        boxCount,
                        primaryInvViewBuf))
                {
                    dynInvCmds.push_back(std::move(preResetCmd));
                    dynInvCmds.push_back(std::move(dynInvCmd));
                    ranDynamicInvalidation = true;
                }
            }
        }

        // 8. Locate POST_DEPTH insertion point in interleavedPassItems (immediately before Main Opaque draws)
        BOOL hasDepthPrepassItem = false;
        for (const auto& item : m_frameView.interleavedPassItems)
        {
            if (item.type == FramePassItemType::DEPTH_PREPASS)
            {
                hasDepthPrepassItem = true;
                break;
            }
        }
        if (!hasDepthPrepassItem)
        {
            FramePassItem emptyDepthItem{};
            emptyDepthItem.type = FramePassItemType::DEPTH_PREPASS;
            m_frameView.interleavedPassItems.insert(m_frameView.interleavedPassItems.begin(), std::move(emptyDepthItem));
        }

        auto insertIt = m_frameView.interleavedPassItems.end();
        for (auto it = m_frameView.interleavedPassItems.begin(); it != m_frameView.interleavedPassItems.end(); ++it)
        {
            if (it->type == FramePassItemType::GRAPHICS_DRAWS)
            {
                insertIt = it;
                break;
            }
        }

        // 9. Build NVS Feedback -> Residency -> Clear -> Projection -> CasterCull -> IndirectGen -> ShadowRaster -> Commit
        Texture* sceneDepthTex = camera->GetSceneDepthTexture();
        UINT32 vpX = 0u;
        UINT32 vpY = 0u;
        UINT32 vpW = sceneDepthTex->GetWidth();
        UINT32 vpH = sceneDepthTex->GetHeight();
        if (camera->GetViewportWidth() > 0.0f && camera->GetViewportHeight() > 0.0f)
        {
            vpX = static_cast<UINT32>((std::max)(0.0f, camera->GetViewportX()));
            vpY = static_cast<UINT32>((std::max)(0.0f, camera->GetViewportY()));
            vpW = static_cast<UINT32>((std::max)(1.0f, camera->GetViewportWidth()));
            vpH = static_cast<UINT32>((std::max)(1.0f, camera->GetViewportHeight()));
            if (vpX + vpW > sceneDepthTex->GetWidth())
                vpW = (sceneDepthTex->GetWidth() > vpX) ? (sceneDepthTex->GetWidth() - vpX) : 1u;
            if (vpY + vpH > sceneDepthTex->GetHeight())
                vpH = (sceneDepthTex->GetHeight() > vpY) ? (sceneDepthTex->GetHeight() - vpY) : 1u;
        }

        NVSFeedbackPassDesc fbDesc{};
        fbDesc.sceneDepthTexture                  = sceneDepthTex;
        fbDesc.viewportOffsetX                    = vpX;
        fbDesc.viewportOffsetY                    = vpY;
        fbDesc.viewportWidth                      = vpW;
        fbDesc.viewportHeight                     = vpH;
        fbDesc.cameraPosition                     = m_frameView.camera.position;
        fbDesc.cameraWorldPos                     = m_frameView.camera.position;
        fbDesc.invViewProj                        = m_frameView.camera.viewProjectionMatrix.Inverted();
        fbDesc.depthCompareMode                   = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        fbDesc.depthMode                          = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        fbDesc.clearDepth                         = 1.0f;
        fbDesc.footprintMode                      = NVS_FOOTPRINT_BORDER_NEIGHBORS;
        fbDesc.footprintBorderMargin              = 0.0625f;
        fbDesc.multiLightViewMode                 = true;
        fbDesc.activeViewTableCount               = activeViewSlotHighWater;
        fbDesc.lightViewTableBuffer               = newCommittedBuf;
        fbDesc.screenPixelWorldScale              = m_frameView.nvsScreenPixelWorldScale;
        fbDesc.screenPixelWorldScale.z            = camState.adaptiveLodBias;
        fbDesc.clipmapHierarchy.pagesPerAxis      = pagesPerAxis;
        fbDesc.clipmapHierarchy.activeLevelCount  = kNVSMaxClipmapLevels;
        fbDesc.clipmapHierarchy.maxShadowDistance = 0.0f;

        std::vector<FramePassItem> nvsItems;
        nvsItems.reserve(24u);

        for (auto& invCmd : slotAndOriginInvCmds)
        {
            FramePassItem item{};
            item.type            = FramePassItemType::COMPUTE_DISPATCH;
            item.computeDispatch = std::move(invCmd);
            nvsItems.push_back(std::move(item));
        }

        GPUBuffer* residencyTelemetryBuf = m_nvsPageManager.GetTelemetryBuffer();
        GPUBuffer* compactedReqBuf       = m_nvsFeedbackManager.GetCompactedRequestBuffer(camera);

        {
            std::vector<ComputeDispatchCommand> fbPipelineCmds;
            if (!m_nvsFeedbackManager.BuildFeedbackPipelineDispatches(
                    fbDesc,
                    fbPipelineCmds,
                    compactedReqBuf,
                    residencyTelemetryBuf,
                    camera))
            {
                return;
            }
            for (auto& fbCmd : fbPipelineCmds)
            {
                FramePassItem item{};
                item.type            = FramePassItemType::COMPUTE_DISPATCH;
                item.computeDispatch = std::move(fbCmd);
                nvsItems.push_back(std::move(item));
            }
        }

        for (auto& dCmd : dynInvCmds)
        {
            FramePassItem item{};
            item.type            = FramePassItemType::COMPUTE_DISPATCH;
            item.computeDispatch = std::move(dCmd);
            nvsItems.push_back(std::move(item));
        }

        NVSResidencyPassDesc resDesc{};
        resDesc.currentFrame                    = currentFrame;
        resDesc.useGPURequestCountFromTelemetry = true;
        resDesc.clearDepth                      = clearDepth;
        resDesc.executeSelectiveClear           = true;
        resDesc.markClearedAfterPass            = true;
        resDesc.executePageProjection           = true;

        {
            FramePassItem residencyItem{};
            residencyItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsPageManager.BuildResidencyComputeDispatch(
                    NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE,
                    resDesc,
                    residencyItem.computeDispatch,
                    compactedReqBuf,
                    camera))
            {
                return;
            }
            nvsItems.push_back(std::move(residencyItem));
        }
        {
            FramePassItem clearItem{};
            clearItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
            if (!m_nvsPageManager.BuildSelectiveClearPassView(clearDepth, clearItem.layeredDepthPass))
                return;
            nvsItems.push_back(std::move(clearItem));
        }
        {
            FramePassItem markClearedItem{};
            markClearedItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsPageManager.BuildResidencyComputeDispatch(
                    NVS_DISPATCH_MARK_CLEARED_PAGES,
                    resDesc,
                    markClearedItem.computeDispatch,
                    nullptr,
                    camera))
            {
                return;
            }
            nvsItems.push_back(std::move(markClearedItem));
        }

        NVSPageProjectionPassDesc projDesc{};
        projDesc.currentFrame       = currentFrame;
        projDesc.depthMode          = depthMode;
        projDesc.resetBeforeBuild   = true;
        projDesc.uploadClipmapTable = true;

        FramePassItem projResetItem{};
        projResetItem.type = FramePassItemType::COMPUTE_DISPATCH;
        FramePassItem projBuildItem{};
        projBuildItem.type = FramePassItemType::COMPUTE_DISPATCH;
        if (m_nvsPageManager.BuildPageProjectionComputeDispatch(
                NVS_PROJECTION_DISPATCH_RESET,
                projDesc,
                projResetItem.computeDispatch,
                camera) &&
            m_nvsPageManager.BuildPageProjectionComputeDispatch(
                NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS,
                projDesc,
                projBuildItem.computeDispatch,
                camera))
        {
            nvsItems.push_back(std::move(projResetItem));
            nvsItems.push_back(std::move(projBuildItem));

            GPUBuffer* sceneObjBuf  = scene->GetGPUSceneBuffer();
            const UINT32 sceneSlots = scene->GetGPUSceneStats().highWaterMark;

            NVSCasterCullPassDesc cullDesc{};
            cullDesc.currentFrame                  = currentFrame;
            cullDesc.staticBatchPolicy             = NVS_STATIC_BATCH_POLICY_PREFER_SUBMESH_MEMBERS;
            cullDesc.skipAlreadyRenderedValidPages = true;
            cullDesc.minCasterProjectedTexels      = 0.0f;
            if (ranDynamicInvalidation)
            {
                cullDesc.forceLodOverride |= 0x80000000u;
            }

            std::vector<ComputeDispatchCommand> cullCmds;
            if (m_nvsPageManager.BuildCasterCullPipelineDispatches(
                    sceneObjBuf,
                    sceneSlots,
                    cullDesc,
                    cullCmds,
                    camera))
            {
                for (auto& cmd : cullCmds)
                {
                    FramePassItem cullItem{};
                    cullItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                    cullItem.computeDispatch = std::move(cmd);
                    nvsItems.push_back(std::move(cullItem));
                }

                SyncSceneNVSDescriptors(scene);

                const BOOL isRevZShadowPass = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO);
                NVSShadowRasterPassDesc rasterDesc{};
                rasterDesc.currentFrame               = currentFrame;
                rasterDesc.allowInstancedMerge        = true;
                rasterDesc.applyShaderBias            = (sceneRasterReadsDepthBias || sceneRasterReadsNormalBias);
                rasterDesc.depthBiasScale             = sceneRasterReadsDepthBias  ? 1.0f : 0.0f;
                rasterDesc.normalBiasScale            = sceneRasterReadsNormalBias ? 1.0f : 0.0f;
                rasterDesc.rasterDepthBias            = isRevZShadowPass ? -4 : +4;
                rasterDesc.rasterSlopeScaledDepthBias = isRevZShadowPass ? -1.5f : +1.5f;
                rasterDesc.rasterDepthBiasClamp       = 0.0f;
                rasterDesc.commitRenderedPages        = true;
                rasterDesc.depthMode                  = depthMode;
                rasterDesc.timeSeconds                = m_frameView.hasAuthoritativeTime
                    ? m_frameView.elapsedTime
                    : (static_cast<FLOAT32>(m_frameView.frameIndex) * 0.016666667f);

                std::vector<ComputeDispatchCommand> indDrawCmds;
                if (m_nvsPageManager.BuildIndirectDrawPipelineDispatches(
                        sceneObjBuf,
                        sceneSlots,
                        rasterDesc,
                        indDrawCmds,
                        camera))
                {
                    for (auto& icmd : indDrawCmds)
                    {
                        FramePassItem indItem{};
                        indItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                        indItem.computeDispatch = std::move(icmd);
                        nvsItems.push_back(std::move(indItem));
                    }

                    FramePassItem shadowPassItem{};
                    shadowPassItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
                    if (m_nvsPageManager.BuildShadowRasterizationPassView(
                            sceneObjBuf,
                            sceneSlots,
                            rasterDesc,
                            shadowPassItem.layeredDepthPass,
                            camera))
                    {
                        if (m_nvgManager.IsInitialized() && m_nvgManager.HasVirtualObjectsInScene(scene))
                        {
                            m_nvgManager.InjectNVSShadowPasses(
                                scene,
                                camera,
                                rasterDesc,
                                sceneObjBuf,
                                sceneSlots,
                                m_nvsPageManager.GetWorklistIndirectArgsBuffer(camera),
                                m_nvsPageManager.GetCasterPageRangeBuffer(camera),
                                m_nvsPageManager.GetPageRenderDataBuffer(camera),
                                m_nvsPageManager.GetConfig().physicalPoolCapacity,
                                nvsItems,
                                shadowPassItem.layeredDepthPass,
                                &m_renderPipeline,
                                m_frameView,
                                m_nvsPageManager.GetPageCommitRecordBuffer(camera));
                        }
                        m_stats.shadowDrawCalls += static_cast<UINT32>(shadowPassItem.layeredDepthPass.drawCommands.size());
                        nvsItems.push_back(std::move(shadowPassItem));

                        FramePassItem commitItem{};
                        commitItem.type = FramePassItemType::COMPUTE_DISPATCH;
                        if (m_nvsPageManager.BuildGPUCommitRenderedPagesDispatch(
                                currentFrame,
                                commitItem.computeDispatch,
                                camera))
                        {
                            nvsItems.push_back(std::move(commitItem));
                        }
                    }
                }
            }
        }

        m_frameView.interleavedPassItems.insert(
            insertIt,
            std::make_move_iterator(nvsItems.begin()),
            std::make_move_iterator(nvsItems.end()));

        const auto tInjectEnd = std::chrono::high_resolution_clock::now();
        const FLOAT32 nvsPrepMs = preBuildPrepMs +
            std::chrono::duration<FLOAT32, std::milli>(tInjectEnd - tInjectStart).count();
        m_stats.computePrepCpuMs += nvsPrepMs;
        m_stats.shadowPrepCpuMs  += nvsPrepMs;
    }

    void NorthstarCore::InjectCustomNVSFramePipelinePasses(
        Scene*                                              scene,
        Camera*                                             camera,
        const std::vector<NVSDynamicCasterInvalidationBox>& frameInvalidationBoxes,
        FrameView&                                          ioView)
    {
        if (!m_nvsFramePipelineEnabled || !camera || !camera->GetSceneDepthTexture())
            return;

        if (!m_nvsPageManager.IsInitialized())
        {
            NVSPageManagerConfig autoPageCfg{};
            autoPageCfg.pagesPerAxis      = (m_nvsFramePipelineConfig.clipmapHierarchy.pagesPerAxis > 0u)
                ? m_nvsFramePipelineConfig.clipmapHierarchy.pagesPerAxis
                : 64u;
            autoPageCfg.clipmapLevelCount = kNVSMaxClipmapLevels;
            const UINT32 pagesPerLight    = autoPageCfg.clipmapLevelCount * autoPageCfg.pagesPerAxis * autoPageCfg.pagesPerAxis;
            autoPageCfg.maxLightCount     = (pagesPerLight > 0u)
                ? (std::max)(1u, (std::min)(4u, kNVSDefaultTotalVirtualPages / pagesPerLight))
                : 1u;
            if (!InitializeInternalNVSPageManager(autoPageCfg))
                return;
        }
        if (!m_nvsFeedbackManager.IsInitialized())
        {
            if (!InitializeInternalNVSFeedbackManager())
                return;
        }

        const auto tPrepStart = std::chrono::high_resolution_clock::now();

        Texture* sceneDepthTex = camera->GetSceneDepthTexture();
        UINT32 vpX = 0u;
        UINT32 vpY = 0u;
        UINT32 vpW = sceneDepthTex->GetWidth();
        UINT32 vpH = sceneDepthTex->GetHeight();

        if (camera->GetViewportWidth() > 0.0f && camera->GetViewportHeight() > 0.0f)
        {
            vpX = static_cast<UINT32>((std::max)(0.0f, camera->GetViewportX()));
            vpY = static_cast<UINT32>((std::max)(0.0f, camera->GetViewportY()));
            vpW = static_cast<UINT32>((std::max)(1.0f, camera->GetViewportWidth()));
            vpH = static_cast<UINT32>((std::max)(1.0f, camera->GetViewportHeight()));
            if (vpX + vpW > sceneDepthTex->GetWidth())
                vpW = (sceneDepthTex->GetWidth() > vpX) ? (sceneDepthTex->GetWidth() - vpX) : 1u;
            if (vpY + vpH > sceneDepthTex->GetHeight())
                vpH = (sceneDepthTex->GetHeight() > vpY) ? (sceneDepthTex->GetHeight() - vpY) : 1u;
        }

        auto stateIt = m_cameraNVSClipmapStates.find(camera);
        if (stateIt == m_cameraNVSClipmapStates.end())
        {
            NVSClipmapHierarchyState initState = m_nvsFramePipelineConfig.clipmapHierarchy;
            if (!initState.lightValid && m_nvsFramePipelineConfig.clipmapHierarchy.activeLevelCount == 0u)
            {
                const Vec3 lightDir = (scene && scene->GetDirectionalLight())
                    ? scene->GetDirectionalLight()->GetDirection()
                    : Vec3(0.0f, -1.0f, 0.0f);
                InitializeNVSClipmapHierarchy(initState, lightDir, 0u);
            }
            stateIt = m_cameraNVSClipmapStates.emplace(camera, initState).first;
        }

        NVSClipmapHierarchyState& camHierarchy = stateIt->second;
        if (scene && scene->GetDirectionalLight() && m_nvsFramePipelineConfig.clipmapHierarchy.lightValid)
        {
            camHierarchy.lightValid = ComputeNVSOrthonormalLightBasis(
                scene->GetDirectionalLight()->GetDirection(),
                camHierarchy.lightRight,
                camHierarchy.lightUp,
                camHierarchy.lightDir);
            camHierarchy.lightForward = camHierarchy.lightDir;
        }

        NVSOriginShiftResult shiftL0{};
        NVSOriginShiftResult allCustomShifts[kNVSMaxClipmapLevels]{};
        UpdateNVSClipmapHierarchySnappedOrigins(camHierarchy, ioView.camera.position, shiftL0, allCustomShifts);

        NVSFeedbackPassDesc passDesc = m_nvsFramePipelineConfig;
        passDesc.sceneDepthTexture   = sceneDepthTex;
        passDesc.viewportWidth       = vpW;
        passDesc.viewportHeight      = vpH;
        passDesc.viewportOffsetX     = vpX;
        passDesc.viewportOffsetY     = vpY;
        passDesc.cameraPosition      = ioView.camera.position;
        passDesc.invViewProj         = ioView.camera.viewProjectionMatrix.Inverted();
        passDesc.clipmapHierarchy    = camHierarchy;

        auto insertIt = ioView.interleavedPassItems.end();
        if (ioView.hasDepthPrepass)
        {
            BOOL hasDepthPrepassItem = false;
            for (const auto& item : ioView.interleavedPassItems)
            {
                if (item.type == FramePassItemType::DEPTH_PREPASS)
                {
                    hasDepthPrepassItem = true;
                    break;
                }
            }
            if (!hasDepthPrepassItem)
            {
                FramePassItem emptyDepthItem{};
                emptyDepthItem.type = FramePassItemType::DEPTH_PREPASS;
                ioView.interleavedPassItems.insert(ioView.interleavedPassItems.begin(), std::move(emptyDepthItem));
            }
            insertIt = ioView.interleavedPassItems.end();
            for (auto it = ioView.interleavedPassItems.begin(); it != ioView.interleavedPassItems.end(); ++it)
            {
                if (it->type == FramePassItemType::GRAPHICS_DRAWS)
                {
                    insertIt = it;
                    break;
                }
            }
        }
        else
        {
            if (ioView.interleavedPassItems.empty())
            {
                if (!ioView.geometryPass.opaqueCommands.empty())
                {
                    FramePassItem opaqueItem{};
                    opaqueItem.type         = FramePassItemType::GRAPHICS_DRAWS;
                    opaqueItem.drawCommands = std::move(ioView.geometryPass.opaqueCommands);
                    ioView.geometryPass.opaqueCommands.clear();
                    ioView.interleavedPassItems.push_back(std::move(opaqueItem));
                }
                if (!ioView.geometryPass.transparentCommands.empty())
                {
                    FramePassItem transItem{};
                    transItem.type         = FramePassItemType::GRAPHICS_DRAWS;
                    transItem.drawCommands = std::move(ioView.geometryPass.transparentCommands);
                    ioView.geometryPass.transparentCommands.clear();
                    ioView.interleavedPassItems.push_back(std::move(transItem));
                }
            }

            insertIt = ioView.interleavedPassItems.end();
            for (auto it = ioView.interleavedPassItems.begin(); it != ioView.interleavedPassItems.end(); ++it)
            {
                if (it->type == FramePassItemType::GRAPHICS_DRAWS)
                {
                    insertIt = std::next(it);
                    break;
                }
            }
        }

        std::vector<FramePassItem> nvsItems;
        nvsItems.reserve(8u);

        GPUBuffer* residencyTelemetryBuf = m_nvsPageManager.GetTelemetryBuffer();
        GPUBuffer* compactedReqBuf       = m_nvsFeedbackManager.GetCompactedRequestBuffer(camera);

        {
            FramePassItem resetItem{};
            resetItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsFeedbackManager.BuildFeedbackResetDispatch(passDesc, resetItem.computeDispatch, residencyTelemetryBuf, camera))
                return;
            nvsItems.push_back(std::move(resetItem));
        }
        {
            FramePassItem markItem{};
            markItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsFeedbackManager.BuildFeedbackMarkDispatch(passDesc, markItem.computeDispatch, camera))
                return;
            nvsItems.push_back(std::move(markItem));
        }
        {
            FramePassItem compactItem{};
            compactItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsFeedbackManager.BuildFeedbackCompactDispatch(passDesc, compactItem.computeDispatch, compactedReqBuf, residencyTelemetryBuf, camera))
                return;
            nvsItems.push_back(std::move(compactItem));
        }

        if (passDesc.executeResidencyAndClear)
        {
            const UINT32 customFrame = (passDesc.residencyPassDesc.currentFrame > 0u)
                ? passDesc.residencyPassDesc.currentFrame
                : static_cast<UINT32>((std::max)(1ull, m_stats.frameIndex));
            const UINT32 activeLvls = (std::min)(camHierarchy.activeLevelCount, kNVSMaxClipmapLevels);
            for (UINT32 lvl = 0u; lvl < activeLvls; ++lvl)
            {
                const BOOL fullLvl = allCustomShifts[lvl].depthRangeChanged || allCustomShifts[lvl].fullWindowReset;
                const BOOL xyShift = (lvl > 0u) && (allCustomShifts[lvl].deltaPageX != 0 || allCustomShifts[lvl].deltaPageY != 0);
                if (fullLvl || xyShift)
                {
                    NVSResidencyPassDesc preInv{};
                    preInv.currentFrame                    = customFrame;
                    preInv.requestCount                    = 0u;
                    preInv.useGPURequestCountFromTelemetry = false;
                    preInv.invalidationMode                = fullLvl ? NVS_INVALIDATE_CLIPMAP_LEVEL : NVS_INVALIDATE_ORIGIN_SHIFT;
                    preInv.invalidateLightId               = camHierarchy.lightId;
                    preInv.invalidateClipmapLevel          = lvl;
                    preInv.originShiftDeltaX               = allCustomShifts[lvl].deltaPageX;
                    preInv.originShiftDeltaY               = allCustomShifts[lvl].deltaPageY;
                    preInv.snappedOriginPageX              = allCustomShifts[lvl].snappedOriginPageX;
                    preInv.snappedOriginPageY              = allCustomShifts[lvl].snappedOriginPageY;
                    preInv.executeSelectiveClear           = false;
                    preInv.markClearedAfterPass            = false;
                    preInv.executePageProjection           = false;

                    FramePassItem preInvItem{};
                    preInvItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (m_nvsPageManager.BuildResidencyComputeDispatch(
                            NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE,
                            preInv,
                            preInvItem.computeDispatch))
                    {
                        nvsItems.insert(nvsItems.begin(), std::move(preInvItem));
                    }
                }
            }

            NVSResidencyPassDesc resDesc = passDesc.residencyPassDesc;
            resDesc.currentFrame = customFrame;
            resDesc.useGPURequestCountFromTelemetry = true;
            if (!allCustomShifts[0].depthRangeChanged && !allCustomShifts[0].fullWindowReset &&
                (shiftL0.deltaPageX != 0 || shiftL0.deltaPageY != 0))
            {
                resDesc.invalidationMode   = NVS_INVALIDATE_ORIGIN_SHIFT;
                resDesc.originShiftDeltaX  = shiftL0.deltaPageX;
                resDesc.originShiftDeltaY  = shiftL0.deltaPageY;
                resDesc.snappedOriginPageX = shiftL0.snappedOriginPageX;
                resDesc.snappedOriginPageY = shiftL0.snappedOriginPageY;
            }
            else
            {
                resDesc.snappedOriginPageX = camHierarchy.snappedOriginPageX[0];
                resDesc.snappedOriginPageY = camHierarchy.snappedOriginPageY[0];
            }

            if (!passDesc.multiLightViewMode && camHierarchy.lightId < kNVSMaxClipmapLights)
            {
                m_nvsPageManager.UploadClipmapHierarchy(camHierarchy, camera);
            }

            BOOL ranDynamicInvalidationPreResidency = false;
            if (!frameInvalidationBoxes.empty())
            {
                if (m_nvsPageManager.UploadDynamicCasterInvalidationBoxes(
                        frameInvalidationBoxes.data(),
                        static_cast<UINT32>(frameInvalidationBoxes.size())))
                {
                    NVSCasterCullPassDesc preInvDesc{};
                    preInvDesc.currentFrame = resDesc.currentFrame;

                    FramePassItem preResetItem{};
                    preResetItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    FramePassItem dynInvItem{};
                    dynInvItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (m_nvsPageManager.BuildCasterCullComputeDispatch(
                            NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES,
                            nullptr,
                            0u,
                            preInvDesc,
                            preResetItem.computeDispatch,
                            camera,
                            0u) &&
                        m_nvsPageManager.BuildCasterCullComputeDispatch(
                            NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION,
                            nullptr,
                            0u,
                            preInvDesc,
                            dynInvItem.computeDispatch,
                            camera,
                            static_cast<UINT32>(frameInvalidationBoxes.size())))
                    {
                        nvsItems.push_back(std::move(preResetItem));
                        nvsItems.push_back(std::move(dynInvItem));
                        ranDynamicInvalidationPreResidency = true;
                    }
                }
            }

            FramePassItem residencyItem{};
            residencyItem.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!m_nvsPageManager.BuildResidencyComputeDispatch(
                    NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE,
                    resDesc,
                    residencyItem.computeDispatch,
                    compactedReqBuf))
            {
                return;
            }
            nvsItems.push_back(std::move(residencyItem));

            if (resDesc.executeSelectiveClear)
            {
                FramePassItem clearItem{};
                clearItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
                if (!m_nvsPageManager.BuildSelectiveClearPassView(resDesc.clearDepth, clearItem.layeredDepthPass))
                    return;
                nvsItems.push_back(std::move(clearItem));
            }

            if (resDesc.markClearedAfterPass)
            {
                FramePassItem markClearedItem{};
                markClearedItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!m_nvsPageManager.BuildResidencyComputeDispatch(
                        NVS_DISPATCH_MARK_CLEARED_PAGES,
                        resDesc,
                        markClearedItem.computeDispatch))
                {
                    return;
                }
                nvsItems.push_back(std::move(markClearedItem));
            }

            if (passDesc.executePageProjection || resDesc.executePageProjection)
            {
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
                if (m_nvsPageManager.BuildPageProjectionComputeDispatch(
                        NVS_PROJECTION_DISPATCH_RESET,
                        projDesc,
                        projResetItem.computeDispatch,
                        camera) &&
                    m_nvsPageManager.BuildPageProjectionComputeDispatch(
                        NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS,
                        projDesc,
                        projBuildItem.computeDispatch,
                        camera))
                {
                    nvsItems.push_back(std::move(projResetItem));
                    nvsItems.push_back(std::move(projBuildItem));

                    if (passDesc.executeCasterCulling)
                    {
                        GPUBuffer* sceneObjBuf = passDesc.gpuSceneBuffer
                            ? passDesc.gpuSceneBuffer
                            : (scene ? scene->GetGPUSceneBuffer() : nullptr);
                        const UINT32 sceneSlots = (passDesc.gpuSceneSlotCount > 0u)
                            ? passDesc.gpuSceneSlotCount
                            : (scene ? scene->GetGPUSceneStats().highWaterMark : 0u);

                        NVSCasterCullPassDesc cullDesc{};
                        cullDesc.currentFrame                  = resDesc.currentFrame;
                        cullDesc.staticBatchPolicy             = passDesc.casterStaticBatchPolicy;
                        cullDesc.skipAlreadyRenderedValidPages = passDesc.skipValidCachedPagesInCull;
                        cullDesc.minCasterProjectedTexels      = passDesc.minCasterProjectedTexels;
                        if (ranDynamicInvalidationPreResidency)
                        {
                            cullDesc.forceLodOverride |= 0x80000000u;
                        }

                        std::vector<ComputeDispatchCommand> cullCmds;
                        if (m_nvsPageManager.BuildCasterCullPipelineDispatches(
                                sceneObjBuf,
                                sceneSlots,
                                cullDesc,
                                cullCmds,
                                camera))
                        {
                            for (auto& cmd : cullCmds)
                            {
                                FramePassItem cullItem{};
                                cullItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                                cullItem.computeDispatch = std::move(cmd);
                                nvsItems.push_back(std::move(cullItem));
                            }

                            if (passDesc.executeShadowRasterization)
                            {
                                SyncSceneNVSDescriptors(scene);

                                NVSShadowRasterPassDesc rasterDesc{};
                                rasterDesc.currentFrame        = resDesc.currentFrame;
                                rasterDesc.allowInstancedMerge = passDesc.allowShadowInstancedMerge;
                                rasterDesc.applyShaderBias     = passDesc.applyShadowShaderBias;
                                rasterDesc.commitRenderedPages = passDesc.commitRenderedShadowPages;
                                rasterDesc.timeSeconds         = passDesc.shadowTimeSeconds;

                                std::vector<ComputeDispatchCommand> indDrawCmds;
                                if (m_nvsPageManager.BuildIndirectDrawPipelineDispatches(
                                        sceneObjBuf,
                                        sceneSlots,
                                        rasterDesc,
                                        indDrawCmds,
                                        camera))
                                {
                                    for (auto& icmd : indDrawCmds)
                                    {
                                        FramePassItem indItem{};
                                        indItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                                        indItem.computeDispatch = std::move(icmd);
                                        nvsItems.push_back(std::move(indItem));
                                    }

                                    FramePassItem shadowPassItem{};
                                    shadowPassItem.type = FramePassItemType::LAYERED_DEPTH_PASS;
                                    if (m_nvsPageManager.BuildShadowRasterizationPassView(
                                            sceneObjBuf,
                                            sceneSlots,
                                            rasterDesc,
                                            shadowPassItem.layeredDepthPass,
                                            camera))
                                    {
                                        if (m_nvgManager.IsInitialized() && m_nvgManager.HasVirtualObjectsInScene(scene))
                                        {
                                            m_nvgManager.InjectNVSShadowPasses(
                                                scene,
                                                camera,
                                                rasterDesc,
                                                sceneObjBuf,
                                                sceneSlots,
                                                m_nvsPageManager.GetWorklistIndirectArgsBuffer(camera),
                                                m_nvsPageManager.GetCasterPageRangeBuffer(camera),
                                                m_nvsPageManager.GetPageRenderDataBuffer(camera),
                                                m_nvsPageManager.GetConfig().physicalPoolCapacity,
                                                nvsItems,
                                                shadowPassItem.layeredDepthPass,
                                                &m_renderPipeline,
                                                ioView,
                                                m_nvsPageManager.GetPageCommitRecordBuffer(camera));
                                        }
                                        nvsItems.push_back(std::move(shadowPassItem));

                                        if (rasterDesc.commitRenderedPages)
                                        {
                                            FramePassItem commitItem{};
                                            commitItem.type = FramePassItemType::COMPUTE_DISPATCH;
                                            if (m_nvsPageManager.BuildGPUCommitRenderedPagesDispatch(
                                                    resDesc.currentFrame,
                                                    commitItem.computeDispatch,
                                                    camera))
                                            {
                                                nvsItems.push_back(std::move(commitItem));
                                                m_nvsPageManager.CommitSamplingViewTableSnapshot(resDesc.currentFrame, camera);
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

        ioView.interleavedPassItems.insert(
            insertIt,
            std::make_move_iterator(nvsItems.begin()),
            std::make_move_iterator(nvsItems.end()));

        const auto tPrepEnd = std::chrono::high_resolution_clock::now();
        m_stats.computePrepCpuMs += std::chrono::duration<float, std::milli>(tPrepEnd - tPrepStart).count();
    }

    void NorthstarCore::SyncSceneNVSDescriptors(Scene* scene)
    {
        if (!scene || !m_nvsPageManager.IsInitialized() || !m_backend)
            return;

        m_nvsPageManager.ClearShadowBucketMaterials();
        auto& shadowRast = m_nvsPageManager.GetShadowRasterizer();
        shadowRast.SetProductionSceneActive(true);
        std::unordered_set<Material*> syncedMaterials;
        struct SyncedMeshKey
        {
            const Mesh*     mesh       = nullptr;
            const LODGroup* lod        = nullptr;
            UINT32          extraFlags = 0u;
            bool operator==(const SyncedMeshKey& o) const noexcept
            {
                return mesh == o.mesh && lod == o.lod && extraFlags == o.extraFlags;
            }
        };
        struct SyncedMeshKeyHash
        {
            size_t operator()(const SyncedMeshKey& k) const noexcept
            {
                const size_t h1 = std::hash<const void*>{}(k.mesh);
                const size_t h2 = std::hash<const void*>{}(k.lod);
                return h1 ^ (h2 * 0x9e3779b97f4a7c15ull) ^ (static_cast<size_t>(k.extraFlags) << 1u);
            }
        };
        std::unordered_set<SyncedMeshKey, SyncedMeshKeyHash> syncedMeshes;

        auto syncMesh = [&](const Mesh* mesh, UINT32 extraFlags, const LODGroup* lod) {
            if (!mesh) return;
            const UINT32 meshId = mesh->GetGPUSceneMeshId();
            if (meshId == 0u || mesh->GetVertexBufferHandle() == 0) return;
            if (!syncedMeshes.insert(SyncedMeshKey{ mesh, lod, extraFlags }).second)
            {
                return;
            }

            UINT32 flags = GPU_MESH_DRAW_DESC_FLAG_VALID | extraFlags;
            if (lod && !lod->GetLevels().empty())
            {
                flags |= GPU_MESH_DRAW_DESC_FLAG_HAS_LOD;
            }
            const UINT32 idxCount = mesh->HasIndices() ? mesh->GetIndexCount() : mesh->GetVertexCount();
            m_nvsPageManager.RegisterMeshFromHandles(
                meshId,
                mesh->GetVertexBufferHandle(),
                mesh->GetIndexBufferHandle(),
                mesh->GetVertexCount(),
                idxCount,
                0u,
                0,
                0u,
                0u,
                flags,
                (std::max)(1u, mesh->GetStateRevision()));

            if (lod)
            {
                const auto& levels = lod->GetLevels();
                for (UINT32 lvl = 1u; lvl < static_cast<UINT32>(levels.size()) && lvl < 4u; ++lvl)
                {
                    const Mesh* lodMesh = levels[lvl].mesh;
                    if (!lodMesh || lodMesh->GetVertexBufferHandle() == 0) continue;
                    GeometryBufferGPUViews views{};
                    if (m_backend->GetGeometryBufferGPUViews(
                            lodMesh->GetVertexBufferHandle(),
                            lodMesh->GetIndexBufferHandle(),
                            views))
                    {
                        GPUMeshDrawDescriptor varDesc{};
                        varDesc.meshId              = meshId;
                        varDesc.meshGeneration      = (std::max)(1u, lodMesh->GetStateRevision());
                        varDesc.descriptorFlags     = flags;
                        varDesc.submeshAndLodPacked = PackGPUMeshSubmeshAndLod(0u, lvl);
                        varDesc.vbAddressLo         = static_cast<UINT32>(views.vbGPUVirtualAddress & 0xFFFFFFFFull);
                        varDesc.vbAddressHi         = static_cast<UINT32>(views.vbGPUVirtualAddress >> 32ull);
                        varDesc.vbSizeInBytes       = views.vbSizeInBytes;
                        varDesc.vbStrideInBytes     = views.vbStrideInBytes;
                        varDesc.ibAddressLo         = static_cast<UINT32>(views.ibGPUVirtualAddress & 0xFFFFFFFFull);
                        varDesc.ibAddressHi         = static_cast<UINT32>(views.ibGPUVirtualAddress >> 32ull);
                        varDesc.ibSizeInBytes       = views.ibSizeInBytes;
                        varDesc.ibFormat            = views.ibFormat;
                        varDesc.vertexCount         = lodMesh->GetVertexCount();
                        varDesc.indexCount          = lodMesh->HasIndices() ? lodMesh->GetIndexCount() : lodMesh->GetVertexCount();
                        varDesc.startIndex          = 0u;
                        varDesc.baseVertex          = 0;
                        m_nvsPageManager.RegisterOrUpdateMeshVariantDescriptor(meshId, 0u, lvl, varDesc);
                    }
                }
            }
        };

        auto syncMaterial = [&](Material* mat) {
            if (!mat) return;
            const UINT32 matId = mat->GetGPUSceneMaterialId();
            if (matId == 0u) return;
            if (!syncedMaterials.insert(mat).second)
            {
                return;
            }

            GPUMaterialShadowDescriptor desc{};
            desc.materialId           = matId;
            desc.materialGeneration   = (std::max)(1u, mat->GetStateRevision());
            desc.defaultBucketIndex   = 0xFFFFFFFFu;
            desc.instancedBucketIndex = 0xFFFFFFFFu;
            desc.skinnedBucketIndex   = 0xFFFFFFFFu;
            desc.customBucketIndex    = 0xFFFFFFFFu;
            desc.alphaCutoff          = 0.5f;

            // A shadow caster material MUST explicitly provide PassType::SHADOW.
            // Never silently fall back to PassType::GEOMETRY; materials without PassType::SHADOW emit 0 shadow draws.
            ShaderSet* shadowSet = mat->HasPass(PassType::SHADOW) ? mat->GetShaderSet(PassType::SHADOW) : nullptr;
            if (!shadowSet)
            {
                desc.materialFlags = GPU_MAT_SHADOW_FLAG_NO_SHADOW_PASS | GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS;
                m_nvsPageManager.RegisterOrUpdateMaterialShadowDescriptor(desc);
                return;
            }

            BOOL stdReqSkin = false, instReqSkin = false, skinReqSkin = false, instSkinReqSkin = false;
            BOOL stdRejRT   = false, instRejRT   = false, skinRejRT   = false, instSkinRejRT   = false;
            const BOOL hasStd      = shadowRast.ValidateShadowShaderSetContract(mat, shadowSet, ShaderVariant::STANDARD,          m_backend, false, stdReqSkin,      stdRejRT);
            const BOOL hasInst     = shadowRast.ValidateShadowShaderSetContract(mat, shadowSet, ShaderVariant::INSTANCED,         m_backend, false, instReqSkin,     instRejRT);
            const BOOL hasSkin     = shadowRast.ValidateShadowShaderSetContract(mat, shadowSet, ShaderVariant::SKINNED,           m_backend, false, skinReqSkin,     skinRejRT);
            const BOOL hasInstSkin = shadowRast.ValidateShadowShaderSetContract(mat, shadowSet, ShaderVariant::INSTANCED_SKINNED, m_backend, false, instSkinReqSkin, instSkinRejRT);

            if (!hasStd && !hasInst && !hasSkin && !hasInstSkin)
            {
                desc.materialFlags = GPU_MAT_SHADOW_FLAG_MISSING_VARIANT | GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS;
                m_nvsPageManager.RegisterOrUpdateMaterialShadowDescriptor(desc);
                return;
            }

            UINT32 mFlags = GPU_MAT_SHADOW_FLAG_VALID |
                            GPU_MAT_SHADOW_FLAG_CUSTOM_SHADOW_PASS |
                            GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS;
            if (mat->IsAlphaTested())
                mFlags |= GPU_MAT_SHADOW_FLAG_ALPHA_TEST;
            if (mat->IsTwoSided() || mat->GetCullMode() == CullMode::NONE)
                mFlags |= GPU_MAT_SHADOW_FLAG_DOUBLE_SIDED;

            if (hasStd)
            {
                const UINT32 bStd = shadowRast.RegisterOrAcquireDrawGroup(mat, ShaderVariant::STANDARD, scene);
                desc.defaultBucketIndex = bStd;
                if (stdReqSkin)
                    mFlags |= GPU_MAT_SHADOW_FLAG_REQUIRES_SKINNING_DATA;
            }

            if (hasInst)
            {
                UINT32 bInst = desc.defaultBucketIndex;
                if (!hasStd || shadowSet->GetVertexShader(ShaderVariant::INSTANCED) != shadowSet->GetVertexShader(ShaderVariant::STANDARD))
                {
                    bInst = shadowRast.RegisterOrAcquireDrawGroup(mat, ShaderVariant::INSTANCED, scene);
                }
                desc.instancedBucketIndex = bInst;
                if (!hasStd)
                {
                    desc.defaultBucketIndex = bInst;
                }
                mFlags |= GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
                if (instReqSkin)
                    mFlags |= GPU_MAT_SHADOW_FLAG_REQUIRES_SKINNING_DATA;
            }
            else if (hasStd && !instRejRT)
            {
                // Compile-time NVS lowering supports per-instance SV_InstanceID context reads on STANDARD shaders
                desc.instancedBucketIndex = desc.defaultBucketIndex;
                mFlags |= GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
            }

            if (hasSkin)
            {
                const UINT32 bSkin = shadowRast.RegisterOrAcquireDrawGroup(mat, ShaderVariant::SKINNED, scene);
                desc.skinnedBucketIndex = bSkin;
                desc.customBucketIndex  = bSkin;
                mFlags |= GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT;
                if (!hasInstSkin && !instSkinRejRT)
                {
                    mFlags |= GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
                }
                if (skinReqSkin)
                    mFlags |= GPU_MAT_SHADOW_FLAG_SKINNED_REQUIRES_PALETTE;
            }

            if (hasInstSkin)
            {
                const UINT32 bInstSkin = (!hasSkin || shadowSet->GetVertexShader(ShaderVariant::INSTANCED_SKINNED) != shadowSet->GetVertexShader(ShaderVariant::SKINNED))
                    ? shadowRast.RegisterOrAcquireDrawGroup(mat, ShaderVariant::INSTANCED_SKINNED, scene)
                    : desc.skinnedBucketIndex;
                desc.customBucketIndex = bInstSkin;
                mFlags |= GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
                if (!hasSkin)
                {
                    desc.skinnedBucketIndex = bInstSkin;
                    mFlags |= GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT;
                }
                if (instSkinReqSkin)
                    mFlags |= GPU_MAT_SHADOW_FLAG_SKINNED_REQUIRES_PALETTE;
            }

            if (desc.defaultBucketIndex == 0xFFFFFFFFu &&
                desc.instancedBucketIndex == 0xFFFFFFFFu &&
                desc.skinnedBucketIndex == 0xFFFFFFFFu &&
                desc.customBucketIndex == 0xFFFFFFFFu)
            {
                desc.materialFlags = GPU_MAT_SHADOW_FLAG_MISSING_VARIANT | GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS;
            }
            else
            {
                desc.materialFlags = mFlags;
            }
            m_nvsPageManager.RegisterOrUpdateMaterialShadowDescriptor(desc);
        };

        for (RenderObject* obj : scene->GetRenderObjects())
        {
            if (!obj) continue;
            UINT32 extraMeshFlags = 0u;
            if (obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices())
                extraMeshFlags |= GPU_MESH_DRAW_DESC_FLAG_SKINNED_VERT;
            syncMesh(obj->GetMesh(), extraMeshFlags, obj->GetLODGroup());
            syncMaterial(obj->GetMaterial());
        }

        if (scene->IsStaticBatched())
        {
            for (const auto& batch : scene->GetBakedStaticBatches())
            {
                syncMesh(batch.combinedMesh, GPU_MESH_DRAW_DESC_FLAG_STATIC_BATCH, nullptr);
                syncMaterial(batch.material);
            }
        }

        m_nvsPageManager.FlushDescriptorTablesToGPU();
    }

    BOOL NorthstarCore::EnqueueInternalLayeredDepthPass(LayeredDepthPassView pass)
    {
        if (!ConfigureInternalLayeredDepthPass(pass))
            return false;
        m_pendingLayeredDepthPasses.push_back(std::move(pass));
        return true;
    }

    BOOL NorthstarCore::UpdateInternalGPUScene(Scene* scene, UINT64 frameNumber)
    {
        if (!m_initialized || !m_backend)
            return false;
        Scene* targetScene = scene ? scene : m_sceneManager.GetActiveScene();
        if (!targetScene)
            return false;
        return m_gpuSceneManager.UpdateScene(targetScene, frameNumber);
    }

    BOOL NorthstarCore::ForceInternalGPUSceneFullRebuildAndUpload(Scene* scene, GPUSceneUpdateStats* outStats)
    {
        if (!m_initialized || !m_backend)
            return false;
        Scene* targetScene = scene ? scene : m_sceneManager.GetActiveScene();
        if (!targetScene)
            return false;
        return m_gpuSceneManager.ForceFullRebuildAndUpload(targetScene, outStats);
    }

    BOOL NorthstarCore::BuildInternalGPUSceneDiagnosticDispatch(
        Scene*                  scene,
        GPUBuffer*              queriesBuffer,
        UINT32                  queryCount,
        GPUBuffer*              outDiagnosticsBuffer,
        ComputeDispatchCommand& outCmd)
    {
        outCmd = ComputeDispatchCommand{};
        if (!m_initialized || !m_backend || !m_internalResourceManager.IsInitialized())
            return false;

        Scene* targetScene = scene ? scene : m_sceneManager.GetActiveScene();
        if (!targetScene)
            return false;

        if (!targetScene->GetGPUSceneBuffer())
        {
            if (!m_gpuSceneManager.UpdateScene(targetScene, m_stats.frameIndex))
                return false;
        }

        GPUBuffer* sceneObjBuf = targetScene->GetGPUSceneBuffer();
        if (!sceneObjBuf || !sceneObjBuf->IsValid() ||
            !sceneObjBuf->HasUsage(GPUBufferUsage::STRUCTURED_SRV) ||
            sceneObjBuf->GetStride() != sizeof(GPUSceneObjectData))
        {
            return false;
        }

        if (queryCount == 0u ||
            !queriesBuffer || !queriesBuffer->IsValid() ||
            !queriesBuffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV) ||
            queriesBuffer->GetStride() != sizeof(GPUSceneDiagnosticQuery) ||
            queryCount > queriesBuffer->GetElementCount())
        {
            return false;
        }

        if (!outDiagnosticsBuffer || !outDiagnosticsBuffer->IsValid() ||
            !outDiagnosticsBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV) ||
            outDiagnosticsBuffer->GetStride() != sizeof(GPUSceneDiagnosticOutput) ||
            queryCount > outDiagnosticsBuffer->GetElementCount())
        {
            return false;
        }

        ComputeKernel* diagKernel = m_internalResourceManager.GetComputeKernel(InternalComputeKernelId::GPU_SCENE_DIAGNOSTIC);
        if (!diagKernel || !diagKernel->IsValid())
            return false;

        const GPUSceneUpdateStats& gStats = targetScene->GetGPUSceneStats();
        diagKernel->SetUInt("GPUSceneDiagParams", "Config", "queryCount", queryCount);
        diagKernel->SetUInt("GPUSceneDiagParams", "Config", "sceneCapacity", sceneObjBuf->GetElementCount());
        diagKernel->SetUInt("GPUSceneDiagParams", "Config", "sceneHighWater", gStats.highWaterMark);
        diagKernel->SetUInt("GPUSceneDiagParams", "Config", "sceneActiveCount", gStats.activeObjectCount);

        auto unbindTransientBuffers = [&]() {
            diagKernel->SetGPUBuffer("GPUSceneDiagParams", "SceneObjects", nullptr);
            diagKernel->SetGPUBuffer("GPUSceneDiagParams", "Queries", nullptr);
            diagKernel->SetGPUBuffer("GPUSceneDiagParams", "OutDiagnostics", nullptr);
        };

        if (!diagKernel->SetGPUBuffer("GPUSceneDiagParams", "SceneObjects", sceneObjBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !diagKernel->SetGPUBuffer("GPUSceneDiagParams", "Queries", queriesBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE) ||
            !diagKernel->SetGPUBuffer("GPUSceneDiagParams", "OutDiagnostics", outDiagnosticsBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE))
        {
            unbindTransientBuffers();
            return false;
        }

        if (!diagKernel->SetDispatchThreadCount(queryCount, 1u, 1u))
        {
            unbindTransientBuffers();
            return false;
        }

        const BOOL built = BuildComputeDispatchCommand(diagKernel, outCmd);
        unbindTransientBuffers();
        return built;
    }

    BOOL NorthstarCore::ExecuteInternalGPUSceneDiagnostic(
        Scene*     scene,
        GPUBuffer* queriesBuffer,
        UINT32     queryCount,
        GPUBuffer* outDiagnosticsBuffer)
    {
        if (!m_backend)
            return false;

        ComputeDispatchCommand cmd{};
        if (!BuildInternalGPUSceneDiagnosticDispatch(scene, queriesBuffer, queryCount, outDiagnosticsBuffer, cmd))
            return false;

        std::vector<ComputeDispatchCommand> cmds;
        cmds.push_back(std::move(cmd));
        if (!m_backend->ExecuteComputeDispatches(cmds))
            return false;
        m_backend->WaitForGPU();
        return true;
    }

    BOOL NorthstarCore::BuildComputeDispatchCommand(const ComputeKernel* kernel, ComputeDispatchCommand& outCmd)
    {
        if (!kernel || !m_computeManager.IsManagedKernel(kernel) || !kernel->IsValid())
        {
            outCmd = ComputeDispatchCommand{};
            return false;
        }

        FrameView standaloneView = m_frameView;
        standaloneView.frameIndex           = m_stats.frameIndex;
        standaloneView.deltaTime            = m_stats.deltaTimeSeconds;
        standaloneView.elapsedTime          = static_cast<FLOAT32>(m_stats.elapsedTimeSeconds);
        standaloneView.hasAuthoritativeTime = true;

        Scene* activeScene = m_sceneManager.GetActiveScene();
        if (activeScene)
        {
            FlushRetiredSceneStaticBatches(activeScene);
            if (!activeScene->GetRenderObjects().empty() || activeScene->GetGPUSceneBuffer() != nullptr)
            {
                m_gpuSceneManager.UpdateScene(activeScene, m_stats.frameIndex);
            }
            Camera* cam = activeScene->GetMainCamera();
            if (!cam && !activeScene->GetCameras().empty())
            {
                cam = activeScene->GetCameras().front();
            }
            if (cam)
            {
                EnsureCameraSceneDepthTexture(cam);
            }
            FrameNVSSamplingContext standaloneNvsCtx{};
            BuildStandaloneNVSSamplingContext(activeScene, cam, standaloneNvsCtx);
            m_renderPipeline.RefreshFrameViewSceneContext(activeScene, cam, m_renderTarget, standaloneView, &standaloneNvsCtx);
        }

        return m_renderPipeline.BuildComputeDispatchCommand(kernel, standaloneView, outCmd);
    }

    BOOL NorthstarCore::ExecuteComputeKernel(const ComputeKernel* kernel)
    {
        if (!m_backend || !kernel || !m_computeManager.IsManagedKernel(kernel))
            return false;

        FrameView standaloneView = m_frameView;
        standaloneView.frameIndex           = m_stats.frameIndex;
        standaloneView.deltaTime            = m_stats.deltaTimeSeconds;
        standaloneView.elapsedTime          = static_cast<FLOAT32>(m_stats.elapsedTimeSeconds);
        standaloneView.hasAuthoritativeTime = true;

        Scene* activeScene = m_sceneManager.GetActiveScene();
        if (activeScene)
        {
            FlushRetiredSceneStaticBatches(activeScene);
            if (!activeScene->GetRenderObjects().empty() || activeScene->GetGPUSceneBuffer() != nullptr)
            {
                m_gpuSceneManager.UpdateScene(activeScene, m_stats.frameIndex);
            }
            Camera* cam = activeScene->GetMainCamera();
            if (!cam && !activeScene->GetCameras().empty())
            {
                cam = activeScene->GetCameras().front();
            }
            if (cam)
            {
                EnsureCameraSceneDepthTexture(cam);
            }
            FrameNVSSamplingContext standaloneNvsCtx{};
            BuildStandaloneNVSSamplingContext(activeScene, cam, standaloneNvsCtx);
            m_renderPipeline.RefreshFrameViewSceneContext(activeScene, cam, m_renderTarget, standaloneView, &standaloneNvsCtx);
        }

        std::vector<ComputeDispatchCommand> cmds;
        ComputeDispatchCommand cmd{};
        if (!m_renderPipeline.BuildComputeDispatchCommand(kernel, standaloneView, cmd, &cmds))
            return false;

        cmds.push_back(std::move(cmd));
        const BOOL ok = m_backend->ExecuteComputeDispatches(cmds);
        m_stats.reflectionMaterializationsThisFrame = m_resourceMaterializer.GetMaterializationsThisFrame();
        m_stats.reflectionMaterializationCacheHits  = m_resourceMaterializer.GetCacheHitsThisFrame();
        m_stats.reflectionMaterializationVramMB     = m_resourceMaterializer.GetCurrentVramMB();
        return ok;
    }

    BOOL NorthstarCore::ExecuteComputeKernels(const std::vector<const ComputeKernel*>& kernels)
    {
        if (!m_backend)
            return false;
        if (kernels.empty())
            return true;
        for (const ComputeKernel* k : kernels)
        {
            if (!m_computeManager.IsManagedKernel(k))
                return false;
        }

        FrameView standaloneView = m_frameView;
        standaloneView.frameIndex           = m_stats.frameIndex;
        standaloneView.deltaTime            = m_stats.deltaTimeSeconds;
        standaloneView.elapsedTime          = static_cast<FLOAT32>(m_stats.elapsedTimeSeconds);
        standaloneView.hasAuthoritativeTime = true;

        Scene* activeScene = m_sceneManager.GetActiveScene();
        if (activeScene)
        {
            FlushRetiredSceneStaticBatches(activeScene);
            if (!activeScene->GetRenderObjects().empty() || activeScene->GetGPUSceneBuffer() != nullptr)
            {
                m_gpuSceneManager.UpdateScene(activeScene, m_stats.frameIndex);
            }
            Camera* cam = activeScene->GetMainCamera();
            if (!cam && !activeScene->GetCameras().empty())
            {
                cam = activeScene->GetCameras().front();
            }
            if (cam)
            {
                EnsureCameraSceneDepthTexture(cam);
            }
            FrameNVSSamplingContext standaloneNvsCtx{};
            BuildStandaloneNVSSamplingContext(activeScene, cam, standaloneNvsCtx);
            m_renderPipeline.RefreshFrameViewSceneContext(activeScene, cam, m_renderTarget, standaloneView, &standaloneNvsCtx);
        }

        std::vector<ComputeDispatchCommand> cmds;
        cmds.reserve(kernels.size());
        for (const ComputeKernel* k : kernels)
        {
            ComputeDispatchCommand cmd{};
            if (!m_renderPipeline.BuildComputeDispatchCommand(k, standaloneView, cmd, &cmds))
                return false;
            cmds.push_back(std::move(cmd));
        }
        const BOOL ok = m_backend->ExecuteComputeDispatches(cmds);
        m_stats.reflectionMaterializationsThisFrame = m_resourceMaterializer.GetMaterializationsThisFrame();
        m_stats.reflectionMaterializationCacheHits  = m_resourceMaterializer.GetCacheHitsThisFrame();
        m_stats.reflectionMaterializationVramMB     = m_resourceMaterializer.GetCurrentVramMB();
        return ok;
    }

    Texture* NorthstarCore::CreateTexture(const TextureDesc& desc)
    {
        Texture* tex = m_textureManager.CreateTexture(desc);
        if (!tex && !m_textureManager.GetLastError().empty())
        {
            m_lastError = m_textureManager.GetLastError();
        }
        return tex;
    }

    BOOL NorthstarCore::ResizeTexture(Texture* texture, UINT32 newWidth, UINT32 newHeight)
    {
        return m_textureManager.ResizeTexture(texture, newWidth, newHeight);
    }

    BOOL NorthstarCore::UpdateTextureSubresource(
        Texture* texture,
        UINT32 mipSlice,
        UINT32 arraySlice,
        const void* srcData,
        SIZE_T srcByteSize)
    {
        return m_textureManager.UpdateTextureSubresource(texture, mipSlice, arraySlice, srcData, srcByteSize);
    }

    BOOL NorthstarCore::GenerateTextureMips(Texture* texture)
    {
        return m_textureManager.GenerateTextureMips(texture);
    }

    BOOL NorthstarCore::ReadbackTextureSync(const Texture* texture, void* outData, UINT32 maxByteSize) const
    {
        return m_textureManager.ReadbackTextureSync(texture, outData, maxByteSize);
    }

    BOOL NorthstarCore::ReadbackTextureSubresourceSync(
        const Texture* texture,
        UINT32 mipSlice,
        UINT32 arraySlice,
        void* outData,
        UINT32 maxByteSize) const
    {
        return m_textureManager.ReadbackTextureSubresourceSync(texture, mipSlice, arraySlice, outData, maxByteSize);
    }

    void NorthstarCore::DestroyTexture(Texture* texture)
    {
        if (!m_textureManager.IsManagedTexture(texture)) return;
        m_sceneManager.OnTextureDestroyed(texture);
        m_computeManager.OnTextureDestroyed(texture);
        m_materialManager.OnTextureDestroyed(texture);
        m_textureManager.DestroyTexture(texture);
    }

    void* NorthstarCore::GetTextureGPUDescriptor(Texture* texture) const
    {
        if (!m_backend || !m_textureManager.IsManagedTexture(texture)) return nullptr;
        return m_backend->GetTextureGPUDescriptor(texture->GetNativeHandle());
    }

    void* NorthstarCore::GetTextureSharedHandle(Texture* texture) const
    {
        if (!m_backend || !m_textureManager.IsManagedTexture(texture)) return nullptr;
        return m_backend->GetTextureSharedHandle(texture->GetNativeHandle());
    }

    SharedHandleType NorthstarCore::GetTextureSharedHandleType(Texture* texture) const
    {
        if (!m_backend || !m_textureManager.IsManagedTexture(texture)) return SharedHandleType::NONE;
        return m_backend->GetTextureSharedHandleType(texture->GetNativeHandle());
    }

    void NorthstarCore::WaitForGPU()
    {
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }
    }

    BOOL NorthstarCore::EnqueueExternalFenceWait(void* sharedFenceNtHandle, UINT64 fenceValue)
    {
        if (!m_backend)
            return false;
        return m_backend->EnqueueExternalFenceWait(sharedFenceNtHandle, fenceValue);
    }

    SamplerState* NorthstarCore::CreateSamplerState(const SamplerDesc& desc)
    {
        return m_textureManager.CreateSamplerState(desc);
    }

    void NorthstarCore::DestroySamplerState(SamplerState* sampler)
    {
        if (!m_textureManager.IsManagedSampler(sampler)) return;
        m_computeManager.OnSamplerDestroyed(sampler);
        m_materialManager.OnSamplerDestroyed(sampler);
        m_textureManager.DestroySamplerState(sampler);
    }

    GPUBuffer* NorthstarCore::CreateGPUBuffer(const GPUBufferDesc& desc)
    {
        return m_bufferManager.CreateGPUBuffer(desc);
    }

    BOOL NorthstarCore::ResizeGPUBuffer(GPUBuffer* buffer, UINT32 newElementCount, const void* initialData)
    {
        return m_bufferManager.ResizeGPUBuffer(buffer, newElementCount, initialData);
    }

    BOOL NorthstarCore::UpdateGPUBuffer(GPUBuffer* buffer, const void* data, UINT32 byteSize, UINT32 dstOffset)
    {
        return m_bufferManager.UpdateGPUBuffer(buffer, data, byteSize, dstOffset);
    }

    BOOL NorthstarCore::RequestGPUBufferReadback(GPUBuffer* buffer, UINT32 srcOffset, UINT32 byteSize)
    {
        return m_bufferManager.RequestGPUBufferReadback(buffer, srcOffset, byteSize);
    }

    BOOL NorthstarCore::TryGetGPUBufferReadbackData(GPUBuffer* buffer, void* outData, UINT32 byteSize)
    {
        return m_bufferManager.TryGetGPUBufferReadbackData(buffer, outData, byteSize);
    }

    BOOL NorthstarCore::ReadbackGPUBufferSync(GPUBuffer* buffer, void* outData, UINT32 byteSize)
    {
        return m_bufferManager.ReadbackGPUBufferSync(buffer, outData, byteSize);
    }

    void NorthstarCore::DestroyGPUBuffer(GPUBuffer* buffer)
    {
        if (!m_bufferManager.IsManagedBuffer(buffer)) return;
        m_computeManager.OnGPUBufferDestroyed(buffer);
        m_materialManager.OnGPUBufferDestroyed(buffer);
        m_bufferManager.DestroyGPUBuffer(buffer);
    }

    Camera* NorthstarCore::CreateCamera(const CameraDesc& desc)
    {
        Camera* cam = m_sceneManager.CreateCamera(desc);
        if (!cam)
        {
            m_lastError = "CreateCamera failed: invalid projection, plane, or non-finite camera descriptor.";
            return nullptr;
        }
        EnsureCameraSceneDepthTexture(cam);
        return cam;
    }

    void NorthstarCore::DestroyCamera(Camera* camera)
    {
        if (!m_sceneManager.IsManagedCamera(camera)) return;
        for (auto it = m_nvsDirectionalLightStates.begin(); it != m_nvsDirectionalLightStates.end();)
        {
            if (it->second.cameraPtr == camera)
            {
                ReclaimNVSViewSlots(it->second.viewSlotBase, kNVSMaxClipmapLevels, true);
                it = m_nvsDirectionalLightStates.erase(it);
            }
            else
            {
                ++it;
            }
        }
        m_nvsFeedbackManager.OnCameraDestroyed(camera);
        m_nvsPageManager.OnCameraDestroyed(camera);
        m_nvgManager.OnCameraDestroyed(camera);
        m_cameraNVSClipmapStates.erase(camera);
        m_cameraNVSProductionStates.erase(camera);
        if (camera->m_sceneDepthTexture != nullptr)
        {
            Texture* sceneDepth = camera->m_sceneDepthTexture;
            camera->m_sceneDepthTexture = nullptr;
            DestroyTexture(sceneDepth);
        }
        m_sceneManager.DestroyCamera(camera);
    }

    void NorthstarCore::EnsureCameraSceneDepthTexture(Camera* camera)
    {
        if (!camera) return;
        if (camera->GetDepthStencilTarget() != nullptr)
        {
            return;
        }
        if (camera->GetRenderType() == CameraRenderType::TO_TEXTURE && camera->GetRenderTarget() == nullptr)
        {
            return;
        }

        UINT32 reqW = m_windowWidth > 0 ? m_windowWidth : 1280u;
        UINT32 reqH = m_windowHeight > 0 ? m_windowHeight : 720u;

        if (camera->GetRenderType() == CameraRenderType::TO_TEXTURE && camera->GetRenderTarget() != nullptr)
        {
            reqW = camera->GetRenderTarget()->GetWidth() > 0 ? camera->GetRenderTarget()->GetWidth() : 1u;
            reqH = camera->GetRenderTarget()->GetHeight() > 0 ? camera->GetRenderTarget()->GetHeight() : 1u;
        }

        if (camera->GetViewportWidth() > 0.0f && camera->GetViewportHeight() > 0.0f)
        {
            const float vpRight  = (camera->GetViewportX() > 0.0f ? camera->GetViewportX() : 0.0f) + camera->GetViewportWidth();
            const float vpBottom = (camera->GetViewportY() > 0.0f ? camera->GetViewportY() : 0.0f) + camera->GetViewportHeight();
            const UINT32 vpW = static_cast<UINT32>(std::ceil(vpRight));
            const UINT32 vpH = static_cast<UINT32>(std::ceil(vpBottom));
            if (vpW > 0 && vpH > 0)
            {
                reqW = vpW;
                reqH = vpH;
            }
        }

        if (camera->m_sceneDepthTexture == nullptr)
        {
            TextureDesc depthDesc{};
            depthDesc.width       = reqW;
            depthDesc.height      = reqH;
            depthDesc.depth       = 1;
            depthDesc.mipLevels   = 1;
            depthDesc.arrayLayers = 1;
            depthDesc.format      = TextureFormat::D32_FLOAT;
            depthDesc.type        = TextureType::TEXTURE_2D;
            depthDesc.usage       = TextureUsage::DEPTH_STENCIL;
            depthDesc.dataSource  = TextureDataSource::EMPTY;
            camera->m_sceneDepthTexture = m_textureManager.CreateTexture(depthDesc);
        }
        else if (camera->m_sceneDepthTexture->GetWidth() != reqW ||
                 camera->m_sceneDepthTexture->GetHeight() != reqH ||
                 (m_backend && camera->m_sceneDepthTexture->GetNativeHandle() == 0))
        {
            m_textureManager.ResizeTexture(camera->m_sceneDepthTexture, reqW, reqH);
        }
    }

    void NorthstarCore::ReleaseAllCameraSceneDepthTextures()
    {
        for (Camera* cam : m_sceneManager.GetCameras())
        {
            if (cam && cam->m_sceneDepthTexture != nullptr)
            {
                Texture* sceneDepth = cam->m_sceneDepthTexture;
                cam->m_sceneDepthTexture = nullptr;
                DestroyTexture(sceneDepth);
            }
        }
    }

    RenderObject* NorthstarCore::CreateRenderObject(const RenderObjectDesc& desc)
    {
        return m_sceneManager.CreateRenderObject(desc);
    }

    void NorthstarCore::DestroyRenderObject(RenderObject* renderObject)
    {
        if (!m_sceneManager.IsManagedRenderObject(renderObject)) return;
        m_gpuSceneManager.OnRenderObjectDestroyed(renderObject);
        m_sceneManager.DestroyRenderObject(renderObject);
        for (Scene* sc : m_sceneManager.GetScenes())
        {
            FlushRetiredSceneStaticBatches(sc);
        }
    }

    LODGroup* NorthstarCore::CreateLODGroup(const LODGroupDesc& desc)
    {
        return m_geometryManager.CreateLODGroup(desc);
    }

    void NorthstarCore::DestroyLODGroup(LODGroup* lodGroup)
    {
        if (!m_geometryManager.IsManagedLODGroup(lodGroup)) return;
        m_sceneManager.OnLODGroupDestroyed(lodGroup);
        m_gpuSceneManager.OnLODGroupDestroyed(lodGroup);
        m_geometryManager.DestroyLODGroup(lodGroup);
    }

    Light* NorthstarCore::CreateLight(const LightDesc& desc)
    {
        Light* l = m_sceneManager.CreateLight(desc);
        if (!l)
        {
            m_lastError = "CreateLight failed: invalid light descriptor parameters.";
        }
        return l;
    }

    void NorthstarCore::DestroyLight(Light* light)
    {
        if (!m_sceneManager.IsManagedLight(light)) return;
        const UINT64 uid = light->GetRuntimeUid();
        auto locIt = m_nvsLocalLightStates.find(uid);
        if (locIt != m_nvsLocalLightStates.end())
        {
            ReclaimNVSViewSlots(locIt->second.viewSlotBase, locIt->second.viewCount, true);
            m_nvsLocalLightStates.erase(locIt);
        }
        for (auto dirIt = m_nvsDirectionalLightStates.begin(); dirIt != m_nvsDirectionalLightStates.end();)
        {
            if (dirIt->second.lightRuntimeUid == uid)
            {
                ReclaimNVSViewSlots(dirIt->second.viewSlotBase, kNVSMaxClipmapLevels, true);
                dirIt = m_nvsDirectionalLightStates.erase(dirIt);
            }
            else
            {
                ++dirIt;
            }
        }
        for (UINT32 s = 0u; s < kNVSClipmapTableMaxEntries; ++s)
        {
            if (m_globalNVSSlotSignatures[s].lightPtr == light)
            {
                m_globalNVSSlotSignatures[s].lightPtr = nullptr;
            }
        }
        for (auto& kv : m_cameraNVSProductionStates)
        {
            auto& assigns = kv.second.committedLightAssignments;
            assigns.erase(
                std::remove_if(assigns.begin(), assigns.end(), [light](const LightNVSViewAssignment& a) {
                    return a.light == light;
                }),
                assigns.end());
        }
        m_sceneManager.DestroyLight(light);
    }

    void NorthstarCore::FlushRetiredSceneStaticBatches(Scene* scene)
    {
        if (!scene) return;

        std::vector<RenderObject*> retiredObjs;
        retiredObjs.swap(scene->m_retiredStaticBatchObjects);
        for (RenderObject* ro : retiredObjs)
        {
            if (ro && m_sceneManager.IsManagedRenderObject(ro))
            {
                m_gpuSceneManager.OnRenderObjectDestroyed(ro);
                m_sceneManager.DestroyRenderObject(ro);
            }
        }

        std::vector<Mesh*> retiredMeshes;
        retiredMeshes.swap(scene->m_retiredStaticBatchMeshes);
        for (Mesh* mesh : retiredMeshes)
        {
            if (mesh && m_geometryManager.IsManagedMesh(mesh))
            {
                DestroyMesh(mesh);
            }
        }
    }

    Scene* NorthstarCore::CreateScene(const SceneDesc& desc)
    {
        return m_sceneManager.CreateScene(desc);
    }

    void NorthstarCore::ClearScene(Scene* scene)
    {
        if (!m_sceneManager.IsManagedScene(scene)) return;

        if (m_backend)
        {
            m_backend->WaitForGPU();
        }

        for (auto locIt = m_nvsLocalLightStates.begin(); locIt != m_nvsLocalLightStates.end();)
        {
            if (locIt->second.scenePtr == scene)
            {
                ReclaimNVSViewSlots(locIt->second.viewSlotBase, locIt->second.viewCount, true);
                locIt = m_nvsLocalLightStates.erase(locIt);
            }
            else
            {
                ++locIt;
            }
        }
        for (auto dirIt = m_nvsDirectionalLightStates.begin(); dirIt != m_nvsDirectionalLightStates.end();)
        {
            if (dirIt->second.scenePtr == scene)
            {
                ReclaimNVSViewSlots(dirIt->second.viewSlotBase, kNVSMaxClipmapLevels, true);
                dirIt = m_nvsDirectionalLightStates.erase(dirIt);
            }
            else
            {
                ++dirIt;
            }
        }
        for (UINT32 s = 0u; s < kNVSClipmapTableMaxEntries; ++s)
        {
            if (m_globalNVSSlotSignatures[s].scenePtr == scene)
            {
                m_globalNVSSlotSignatures[s] = CameraNVSSlotSignature{};
            }
        }
        m_sceneMaterialShadowStates.erase(scene);

        scene->Clear();
        FlushRetiredSceneStaticBatches(scene);
        if (scene->GetGPUSceneBuffer() != nullptr)
        {
            m_gpuSceneManager.UpdateScene(scene, 0ull);
        }
    }

    void NorthstarCore::DestroyScene(Scene* scene)
    {
        if (!m_sceneManager.IsManagedScene(scene)) return;

        if (m_backend)
        {
            m_backend->WaitForGPU();
        }

        m_nvgManager.OnSceneDestroyed(scene);
        for (auto locIt = m_nvsLocalLightStates.begin(); locIt != m_nvsLocalLightStates.end();)
        {
            if (locIt->second.scenePtr == scene)
            {
                ReclaimNVSViewSlots(locIt->second.viewSlotBase, locIt->second.viewCount, true);
                locIt = m_nvsLocalLightStates.erase(locIt);
            }
            else
            {
                ++locIt;
            }
        }
        for (auto dirIt = m_nvsDirectionalLightStates.begin(); dirIt != m_nvsDirectionalLightStates.end();)
        {
            if (dirIt->second.scenePtr == scene)
            {
                ReclaimNVSViewSlots(dirIt->second.viewSlotBase, kNVSMaxClipmapLevels, true);
                dirIt = m_nvsDirectionalLightStates.erase(dirIt);
            }
            else
            {
                ++dirIt;
            }
        }
        for (UINT32 s = 0u; s < kNVSClipmapTableMaxEntries; ++s)
        {
            if (m_globalNVSSlotSignatures[s].scenePtr == scene)
            {
                m_globalNVSSlotSignatures[s] = CameraNVSSlotSignature{};
            }
        }
        m_nvsPageManager.GetShadowRasterizer().OnSceneDestroyed(scene);

        while (!scene->m_bakedStaticBatches.empty())
        {
            scene->RetireBakedStaticBatchAt(scene->m_bakedStaticBatches.size() - 1u);
        }
        FlushRetiredSceneStaticBatches(scene);
        m_sceneMaterialShadowStates.erase(scene);
        m_gpuSceneManager.OnSceneDestroyed(scene);
        m_sceneManager.DestroyScene(scene);
    }

    void NorthstarCore::SetActiveScene(Scene* scene)
    {
        m_sceneManager.SetActiveScene(scene);
    }

    Scene* NorthstarCore::GetActiveScene() const
    {
        return m_sceneManager.GetActiveScene();
    }

    void NorthstarCore::BakeStaticBatching(Scene* scene)
    {
        Scene* s = scene ? scene : m_sceneManager.GetActiveScene();
        if (!s) return;

        if (m_backend)
        {
            m_backend->WaitForGPU();
        }

        // Clear existing baked batches if rebaking
        while (!s->m_bakedStaticBatches.empty())
        {
            s->RetireBakedStaticBatchAt(s->m_bakedStaticBatches.size() - 1u);
        }
        FlushRetiredSceneStaticBatches(s);

        const auto& sceneObjects = s->GetRenderObjects();
        std::unordered_map<Material*, std::vector<RenderObject*>> groups;
        std::vector<Material*> materialOrder;

        for (RenderObject* obj : sceneObjects)
        {
            if (!obj) continue;
            if (!obj->HasFlag(RenderObjectFlags::STATIC)) continue;
            if (obj->HasFlag(RenderObjectFlags::BAKED_STATIC)) continue;
            if (obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices()) continue;
            if (obj->GetLODGroup() != nullptr) continue;

            Mesh* mesh = obj->GetMesh();
            Material* mat = obj->GetMaterial();
            if (!mesh || !mat || mesh->IsVirtualized() || !mesh->HasCPUData()) continue;
            if (mat->IsTransparent()) continue;

            if (groups.find(mat) == groups.end())
            {
                materialOrder.push_back(mat);
            }
            groups[mat].push_back(obj);
        }

        for (Material* mat : materialOrder)
        {
            const auto& objs = groups[mat];
            if (objs.empty()) continue;

            UINT64 totalVertices64 = 0ull;
            UINT64 totalIndices64  = 0ull;
            for (RenderObject* obj : objs)
            {
                Mesh* m = obj->GetMesh();
                totalVertices64 += m->GetVertexCount();
                totalIndices64  += m->HasIndices() ? m->GetIndexCount() : m->GetVertexCount();
            }
            if (totalVertices64 == 0ull ||
                (totalVertices64 * sizeof(Vertex)) > 0x7FFFFFFFULL ||
                (totalIndices64 * sizeof(UINT32)) > 0x7FFFFFFFULL)
            {
                continue;
            }

            std::vector<Vertex> combinedVertices;
            combinedVertices.reserve(static_cast<SIZE_T>(totalVertices64));
            std::vector<UINT32> combinedIndices;
            combinedIndices.reserve(static_cast<SIZE_T>(totalIndices64));

            AABB combinedAABB;
            BOOL hasAABB = false;
            BOOL anyCastsShadow = false;
            BOOL anyReceivesShadow = false;
            UINT32 combinedLayerMask = 0u;
            std::vector<BakedStaticSubmesh> batchSubmeshes;
            batchSubmeshes.reserve(objs.size());

            for (RenderObject* obj : objs)
            {
                Mesh* m = obj->GetMesh();
                const Mat4& world = obj->GetWorldMatrix();

                const Vec3 c0(world.m[0], world.m[1], world.m[2]);
                const Vec3 c1(world.m[4], world.m[5], world.m[6]);
                const Vec3 c2(world.m[8], world.m[9], world.m[10]);
                const Vec3 cof0 = c1.Cross(c2);
                const Vec3 cof1 = c2.Cross(c0);
                const Vec3 cof2 = c0.Cross(c1);
                const FLOAT32 det3x3 = c0.Dot(cof0);
                const BOOL flipWinding = (det3x3 < 0.0f);
                const FLOAT32 detSign = (det3x3 < 0.0f) ? -1.0f : 1.0f;
                const BOOL hasValidCof = std::isfinite(det3x3) && (std::fabs(det3x3) > 1e-12f);

                const UINT32 baseVertexIndex = static_cast<UINT32>(combinedVertices.size());
                const UINT32 submeshStartIndex = static_cast<UINT32>(combinedIndices.size());
                const Vertex* srcVerts = m->GetVertices();
                const UINT32 vertCount = m->GetVertexCount();

                for (UINT32 v = 0; v < vertCount; ++v)
                {
                    Vertex outV = srcVerts[v];
                    outV.position = world.TransformPoint(srcVerts[v].position);
                    if (hasValidCof)
                    {
                        const Vec3& sn = srcVerts[v].normal;
                        outV.normal = ((cof0 * sn.x + cof1 * sn.y + cof2 * sn.z) * detSign).Normalized();
                    }
                    else
                    {
                        outV.normal = world.TransformDirection(srcVerts[v].normal).Normalized();
                    }

                    // Tangents and bitangents are covariant surface derivatives (transform by M, not (M^-1)^T)
                    Vec3 tWorld = world.TransformDirection(srcVerts[v].tangent);
                    if (outV.normal.LengthSq() > 1e-6f && tWorld.LengthSq() > 1e-6f)
                    {
                        tWorld = (tWorld - outV.normal * tWorld.Dot(outV.normal));
                    }
                    outV.tangent   = tWorld.Normalized();
                    outV.bitangent = world.TransformDirection(srcVerts[v].bitangent).Normalized();
                    combinedVertices.push_back(outV);
                }

                if (m->HasIndices())
                {
                    const UINT32* srcIndices = m->GetIndices();
                    const UINT32 indexCount = m->GetIndexCount();
                    for (UINT32 i = 0; i < indexCount; ++i)
                    {
                        combinedIndices.push_back(baseVertexIndex + srcIndices[i]);
                    }
                    if (flipWinding)
                    {
                        for (UINT32 i = submeshStartIndex; i + 2u < static_cast<UINT32>(combinedIndices.size()); i += 3u)
                        {
                            std::swap(combinedIndices[i + 1u], combinedIndices[i + 2u]);
                        }
                    }
                }
                else
                {
                    for (UINT32 i = 0; i < vertCount; ++i)
                    {
                        combinedIndices.push_back(baseVertexIndex + i);
                    }
                    if (flipWinding)
                    {
                        for (UINT32 i = submeshStartIndex; i + 2u < static_cast<UINT32>(combinedIndices.size()); i += 3u)
                        {
                            std::swap(combinedIndices[i + 1u], combinedIndices[i + 2u]);
                        }
                    }
                }

                AABB objWorldAABB = obj->GetWorldAABB();
                if (!hasAABB)
                {
                    combinedAABB = objWorldAABB;
                    hasAABB = true;
                }
                else
                {
                    combinedAABB = combinedAABB.Merge(objWorldAABB);
                }

                const BOOL objCastsShadow = obj->HasFlag(RenderObjectFlags::CAST_SHADOW);
                if (objCastsShadow)
                {
                    anyCastsShadow = true;
                }
                if (obj->HasFlag(RenderObjectFlags::RECEIVE_SHADOW))
                {
                    anyReceivesShadow = true;
                }
                combinedLayerMask |= obj->GetLayerMask();

                BakedStaticSubmesh sub{};
                sub.startIndex  = submeshStartIndex;
                sub.indexCount  = static_cast<UINT32>(combinedIndices.size()) - submeshStartIndex;
                sub.localAABB   = m->GetBoundingBox();
                sub.worldMatrix = world;
                sub.worldAABB   = objWorldAABB;
                sub.castsShadow = objCastsShadow;
                batchSubmeshes.push_back(sub);

                // Mark original source object as pre-transformed & baked into combined mesh
                obj->SetBakedIntoStaticBatch(true);
            }

            // Create combined Mesh
            MeshDesc meshDesc{};
            meshDesc.vertices = combinedVertices.data();
            meshDesc.vertexCount = static_cast<UINT32>(combinedVertices.size());
            meshDesc.indices = combinedIndices.data();
            meshDesc.indexCount = static_cast<UINT32>(combinedIndices.size());
            meshDesc.boundingBox = combinedAABB;

            Mesh* combinedMesh = CreateMesh(meshDesc);
            if (!combinedMesh)
            {
                for (RenderObject* obj : objs)
                {
                    if (obj) obj->SetBakedIntoStaticBatch(false);
                }
                continue;
            }

            // Create baked RenderObject with Identity world matrix
            RenderObjectDesc objDesc{};
            objDesc.mesh = combinedMesh;
            objDesc.material = mat;
            objDesc.worldMatrix = Mat4::Identity();
            objDesc.worldAABB = combinedAABB;
            objDesc.layerMask = (combinedLayerMask != 0u) ? combinedLayerMask : 0x00000001u;
            objDesc.flags = RenderObjectFlags::STATIC | RenderObjectFlags::VISIBLE | RenderObjectFlags::FRUSTUM_CULL | RenderObjectFlags::BAKED_STATIC;
            if (anyCastsShadow)
            {
                objDesc.flags |= RenderObjectFlags::CAST_SHADOW;
            }
            if (anyReceivesShadow)
            {
                objDesc.flags |= RenderObjectFlags::RECEIVE_SHADOW;
            }

            RenderObject* batchObj = m_sceneManager.CreateRenderObject(objDesc);

            BakedStaticBatch batch{};
            batch.combinedMesh  = combinedMesh;
            batch.material      = mat;
            batch.worldAABB     = combinedAABB;
            batch.batchObject   = batchObj;
            batch.sourceObjects = objs;
            batch.submeshes     = std::move(batchSubmeshes);

            s->AddBakedStaticBatch(batch);
        }

        if (!s->GetBakedStaticBatches().empty())
        {
            s->SetStaticBatched(true);
        }
    }

    const RenderStats& NorthstarCore::GetStats() const
    {
        return m_stats;
    }

    void NorthstarCore::ResetStats()
    {
        m_stats.Reset();
    }

    NativeContext NorthstarCore::GetNativeContext() const
    {
        return m_backend ? m_backend->GetNativeContext() : NativeContext{};
    }

    void NorthstarCore::SetPrePresentCallback(PrePresentCallback cb)
    {
        m_prePresentCallback = cb;
        if (m_backend)
        {
            m_backend->SetPrePresentCallback(std::move(cb));
        }
    }

    void NorthstarCore::SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU)
    {
        m_postFrameExecutionCallback = cb;
        m_postFrameWaitForGPU        = waitForGPU;
        if (m_backend)
        {
            m_backend->SetPostFrameExecutionCallback(std::move(cb), waitForGPU);
        }
    }
}

Northstar::UINT32 Northstar_GetAPIVersion()
{
    return Northstar::NORTHSTAR_ENGINE_API_VERSION;
}

Northstar::NorthstarEngine* CreateNorthstarEngine()
{
    return new Northstar::NorthstarCore();
}

void DestroyNorthstarEngine(Northstar::NorthstarEngine* engine)
{
    delete engine;
}
