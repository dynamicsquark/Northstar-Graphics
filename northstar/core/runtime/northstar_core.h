#pragma once

#ifdef _WIN32
    #ifdef NORTHSTAR_CORE_EXPORTS
        #define NORTHSTAR_CORE __declspec(dllexport)
    #else
        #define NORTHSTAR_CORE __declspec(dllimport)
    #endif
#else
    #define NORTHSTAR_CORE
#endif

#include "../../include/api/northstar_engine.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/platform/frame_view.h"
#include "core_shader_manager.h"
#include "core_texture_manager.h"
#include "core_buffer_manager.h"
#include "core_geometry_manager.h"
#include "core_material_manager.h"
#include "core_compute_manager.h"
#include "core_scene_manager.h"
#include "core_gpu_scene_manager.h"
#include "core_render_pipeline.h"
#include "core_resource_binding_resolver.h"
#include "internal_resource_manager.h"
#include "core_nvs_page_manager.h"
#include "core_nvs_feedback_manager.h"
#include "../nvg/core_nvg_manager.h"
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace Northstar
{
    using PostCameraRenderCallback = void (*)(
        NorthstarEngine* engine,
        Scene*           scene,
        Camera*          camera,
        UINT32           cameraOrderIndex,
        void*            userData);

    class NorthstarCore : public NorthstarEngine
    {
    public:
        NorthstarCore();
        ~NorthstarCore() override;

        BOOL Initialize(NorthstarBackend* backend, const NativeRenderTarget& renderTarget) override;
        void Shutdown() override;
        BOOL IsInitialized() const override { return m_initialized; }
        const std::string& GetLastError() const override { return m_lastError; }
        const std::string& GetRuntimeDirectory() const override { return m_internalResourceManager.GetRuntimeDirectoryUTF8(); }

        // Engine-internal operations (not exposed on public NorthstarEngine user vtable)
        BOOL ExecuteInternalBufferClear(GPUBuffer* dstBuffer, UINT32 clearValue, UINT32 elementCount, UINT32 dstElementOffset = 0);
        BOOL BuildInternalDirtyDepthSliceDispatch(
            GPUBuffer*              candidateSlicesBuffer,
            UINT32                  candidateCount,
            UINT32                  maxArrayLayers,
            UINT32                  maxOutputSlices,
            UINT32                  inputMode,
            GPUBuffer*              outDirtySliceIndicesBuffer,
            GPUBuffer*              outDirtyCountBuffer,
            GPUBuffer*              outIndirectDrawArgsBuffer,
            ComputeDispatchCommand& outCmd);
        BOOL ConfigureInternalLayeredDepthPass(LayeredDepthPassView& ioPass) const;
        BOOL ExecuteInternalLayeredDepthPass(
            LayeredDepthPassView&                      pass,
            const std::vector<ComputeDispatchCommand>* preDispatches = nullptr);

        // Persistent GPU Scene internal operations
        BOOL UpdateInternalGPUScene(Scene* scene = nullptr, UINT64 frameNumber = 0ull);
        BOOL ForceInternalGPUSceneFullRebuildAndUpload(Scene* scene = nullptr, GPUSceneUpdateStats* outStats = nullptr);
        BOOL BuildInternalGPUSceneDiagnosticDispatch(
            Scene*                  scene,
            GPUBuffer*              queriesBuffer,
            UINT32                  queryCount,
            GPUBuffer*              outDiagnosticsBuffer,
            ComputeDispatchCommand& outCmd);
        BOOL ExecuteInternalGPUSceneDiagnostic(
            Scene*     scene,
            GPUBuffer* queriesBuffer,
            UINT32     queryCount,
            GPUBuffer* outDiagnosticsBuffer);

        CoreGPUSceneManager&           GetGPUSceneManager()                  { return m_gpuSceneManager; }
        const CoreGPUSceneManager&     GetGPUSceneManager()            const { return m_gpuSceneManager; }
        CoreNVSPageManager&            GetNVSPageManager()                   { return m_nvsPageManager; }
        const CoreNVSPageManager&      GetNVSPageManager()             const { return m_nvsPageManager; }
        CoreNVSFeedbackManager&        GetNVSFeedbackManager()               { return m_nvsFeedbackManager; }
        const CoreNVSFeedbackManager&  GetNVSFeedbackManager()         const { return m_nvsFeedbackManager; }
        CoreNVGManager&                GetNVGManager()                       { return m_nvgManager; }
        const CoreNVGManager&          GetNVGManager()                 const { return m_nvgManager; }
        NorthstarBackend*              GetBackend()                    const noexcept { return m_backend; }
        CoreBufferManager&             GetBufferManager()                    { return m_bufferManager; }
        CoreComputeManager&            GetComputeManager()                   { return m_computeManager; }
        CoreRenderPipeline&            GetRenderPipeline()                   { return m_renderPipeline; }
        CoreGPUResourceMaterializer&   GetResourceMaterializer()             { return m_resourceMaterializer; }
        const CoreGPUResourceMaterializer& GetResourceMaterializer()   const { return m_resourceMaterializer; }
        InternalResourceManager&       GetInternalResourceManager()          { return m_internalResourceManager; }
        const InternalResourceManager& GetInternalResourceManager()    const { return m_internalResourceManager; }
        BOOL                           InitializeInternalNVSPageManager(const NVSPageManagerConfig& config = NVSPageManagerConfig{});
        void                           ShutdownInternalNVSPageManager();
        BOOL                           InitializeInternalNVSFeedbackManager(const NVSFeedbackManagerConfig& config = NVSFeedbackManagerConfig{});
        void                           ShutdownInternalNVSFeedbackManager();
        void                           SetInternalNVSFramePipelineEnabled(BOOL enabled, const NVSFeedbackPassDesc* config = nullptr);
        BOOL                           IsInternalNVSFramePipelineEnabled() const { return m_nvsHasCustomPipelineConfig; }
        void                           SyncSceneNVSDescriptors(Scene* scene);

        void SetGPUSceneUpdateEnabled(BOOL enabled) { m_gpuSceneUpdateEnabled = enabled; }
        BOOL IsGPUSceneUpdateEnabled()        const { return m_gpuSceneUpdateEnabled; }
        BOOL EnqueueInternalLayeredDepthPass(LayeredDepthPassView pass);
        void SetPostCameraRenderCallback(PostCameraRenderCallback cb, void* userData = nullptr)
        {
            m_postCameraCallback         = cb;
            m_postCameraCallbackUserData = userData;
        }

        void BeginFrame() override;
        void RenderFrame() override;
        void EndFrame() override;

        void SetVSync(BOOL enabled) override;
        BOOL IsVSyncEnabled() const override;

        void OnResize(UINT32 width, UINT32 height) override;

        Shader*       CreateShader(const ShaderDesc& desc) override;
        void          DestroyShader(Shader* shader) override;
        ShaderSet*    CreateShaderSet(const ShaderSetDesc& desc) override;
        void          DestroyShaderSet(ShaderSet* shaderSet) override;

        Mesh*         CreateMesh(const MeshDesc& desc) override;
        void          DestroyMesh(Mesh* mesh) override;
        BOOL          SerializeVirtualMesh(const Mesh* mesh, std::vector<BYTE>& outBlob) const override;
        BOOL          SerializeVirtualMeshToBuffer(const Mesh* mesh, BYTE** outBuffer, SIZE_T* outByteSize) const override;
        void          FreeSerializedBuffer(BYTE* buffer) const override;

        BOOL             SetNVGConfig(const NVGConfig& config) override;
        const NVGConfig& GetNVGConfig() const override;
        const NVGStats&  GetNVGStats() const override;
        NVGCompatibilityStatus CheckMaterialNVGCompatibility(
            const Material* material,
            PassType        pass,
            BOOL            isSkinned = false,
            std::string*    outReason = nullptr) const override;

        Material*     CreateMaterial(const MaterialDesc& desc) override;
        Material*     CloneMaterial(const Material* source) override;
        void          DestroyMaterial(Material* material) override;

        ComputeKernel* CreateComputeKernel(const ComputeKernelDesc& desc) override;
        ComputeKernel* CloneComputeKernel(const ComputeKernel* source) override;
        void           DestroyComputeKernel(ComputeKernel* kernel) override;
        BOOL           BuildComputeDispatchCommand(const ComputeKernel* kernel, ComputeDispatchCommand& outCmd) override;
        BOOL           ExecuteComputeKernel(const ComputeKernel* kernel) override;
        BOOL           ExecuteComputeKernels(const std::vector<const ComputeKernel*>& kernels) override;

        Texture*      CreateTexture(const TextureDesc& desc) override;
        BOOL          ResizeTexture(Texture* texture, UINT32 newWidth, UINT32 newHeight) override;
        BOOL          UpdateTextureSubresource(Texture* texture, UINT32 mipSlice, UINT32 arraySlice, const void* srcData, SIZE_T srcByteSize) override;
        BOOL          GenerateTextureMips(Texture* texture) override;
        BOOL          ReadbackTextureSync(const Texture* texture, void* outData, UINT32 maxByteSize) const override;
        BOOL          ReadbackTextureSubresourceSync(const Texture* texture, UINT32 mipSlice, UINT32 arraySlice, void* outData, UINT32 maxByteSize) const override;
        void          DestroyTexture(Texture* texture) override;
        void*         GetTextureGPUDescriptor(Texture* texture) const override;
        void*         GetTextureSharedHandle(Texture* texture) const override;
        SharedHandleType GetTextureSharedHandleType(Texture* texture) const override;
        void          WaitForGPU() override;
        BOOL          EnqueueExternalFenceWait(void* sharedFenceNtHandle, UINT64 fenceValue) override;

        SamplerState* CreateSamplerState(const SamplerDesc& desc) override;
        void          DestroySamplerState(SamplerState* sampler) override;

        GPUBuffer*    CreateGPUBuffer(const GPUBufferDesc& desc) override;
        BOOL          ResizeGPUBuffer(GPUBuffer* buffer, UINT32 newElementCount, const void* initialData = nullptr) override;
        BOOL          UpdateGPUBuffer(GPUBuffer* buffer, const void* data, UINT32 byteSize, UINT32 dstOffset = 0) override;
        BOOL          RequestGPUBufferReadback(GPUBuffer* buffer, UINT32 srcOffset = 0, UINT32 byteSize = 0) override;
        BOOL          TryGetGPUBufferReadbackData(GPUBuffer* buffer, void* outData, UINT32 byteSize) override;
        BOOL          ReadbackGPUBufferSync(GPUBuffer* buffer, void* outData, UINT32 byteSize) override;
        void          DestroyGPUBuffer(GPUBuffer* buffer) override;

        Camera*       CreateCamera(const CameraDesc& desc) override;
        void          DestroyCamera(Camera* camera) override;

        RenderObject* CreateRenderObject(const RenderObjectDesc& desc) override;
        void          DestroyRenderObject(RenderObject* renderObject) override;

        LODGroup*     CreateLODGroup(const LODGroupDesc& desc) override;
        void          DestroyLODGroup(LODGroup* lodGroup) override;

        Light*        CreateLight(const LightDesc& desc) override;
        void          DestroyLight(Light* light) override;

        Scene*        CreateScene(const SceneDesc& desc) override;
        void          ClearScene(Scene* scene) override;
        void          DestroyScene(Scene* scene) override;
        void          SetActiveScene(Scene* scene) override;
        Scene*        GetActiveScene() const override;
        void          BakeStaticBatching(Scene* scene = nullptr) override;

        const RenderStats& GetStats() const override;
        void               ResetStats() override;

        NativeContext        GetNativeContext() const override;
        void                 SetPrePresentCallback(PrePresentCallback cb) override;
        void                 SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU = true) override;

        void                 ResetInternalNVSTrackingStates();
        void                 MarkAllNVSSlotsExternallyModified();

    private:
        NorthstarBackend*   m_backend = nullptr;
        NativeRenderTarget  m_renderTarget;
        RenderStats         m_stats;

        CoreShaderManager   m_shaderManager;
        CoreTextureManager  m_textureManager;
        CoreBufferManager   m_bufferManager;
        CoreGeometryManager m_geometryManager;
        CoreMaterialManager m_materialManager;
        CoreComputeManager      m_computeManager;
        CoreSceneManager        m_sceneManager;
        CoreGPUSceneManager         m_gpuSceneManager;
        CoreGPUResourceMaterializer m_resourceMaterializer;
        CoreRenderPipeline          m_renderPipeline;
        InternalResourceManager m_internalResourceManager;
        CoreNVSPageManager      m_nvsPageManager;
        CoreNVSFeedbackManager  m_nvsFeedbackManager;
        CoreNVGManager          m_nvgManager;

        NativeHandle m_nvsComparisonSamplerStandard  = 0;
        NativeHandle m_nvsComparisonSamplerReversedZ = 0;
        NativeHandle m_nvsDepthSampler               = 0;

        struct CameraNVSSlotSignature
        {
            BOOL                  occupied              = false;
            const Scene*          scenePtr              = nullptr;
            const Camera*         cameraPtr             = nullptr;
            const Light*          lightPtr              = nullptr;
            UINT64                lightRuntimeUid       = 0ull;
            LightType             lightType             = LightType::DIRECTIONAL;
            LightShadowResolution shadowResolution      = LightShadowResolution::RES_16384;
            Vec3                  direction             = Vec3::Zero();
            Vec3                  position              = Vec3::Zero();
            FLOAT32               range                 = 0.0f;
            FLOAT32               innerConeAngle        = 0.0f;
            FLOAT32               outerConeAngle        = 0.0f;
            FLOAT32               shadowNearPlane       = 0.1f;
            FLOAT32               shadowMaxDistance     = 0.0f;
            FLOAT32               shadowBias            = 0.0f;
            FLOAT32               shadowNormalBias      = 0.0f;
            UINT32                cullingMask           = 0xFFFFFFFFu;
            UINT32                depthMode             = 0u;
            BOOL                  rasterReadsDepthBias  = false;
            BOOL                  rasterReadsNormalBias = false;
        };

        struct NVSDirectionalCameraKey
        {
            const Camera* cameraPtr       = nullptr;
            UINT64        lightRuntimeUid = 0ull;

            bool operator==(const NVSDirectionalCameraKey& o) const noexcept
            {
                return cameraPtr == o.cameraPtr && lightRuntimeUid == o.lightRuntimeUid;
            }
        };

        struct NVSDirectionalCameraKeyHash
        {
            size_t operator()(const NVSDirectionalCameraKey& k) const noexcept
            {
                const size_t h1 = std::hash<const void*>{}(k.cameraPtr);
                const size_t h2 = std::hash<UINT64>{}(k.lightRuntimeUid);
                return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6u) + (h1 >> 2u));
            }
        };

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
        struct NVSDirectionalLightCameraState
        {
            const Scene*             scenePtr                = nullptr;
            const Camera*            cameraPtr               = nullptr;
            UINT64                   lightRuntimeUid         = 0ull;
            UINT32                   viewSlotBase            = 0u;
            UINT32                   viewGeneration          = 1u;
            UINT64                   dataVersion             = 1ull;
            UINT64                   lastActiveFrame         = 0ull;
            BOOL                     initialized             = false;
            BOOL                     isPerspectiveCam        = false;
            LightShadowResolution    shadowResolution        = LightShadowResolution::RES_16384;
            Vec3                     direction               = Vec3::Zero();
            FLOAT32                  shadowNearPlane         = 0.1f;
            FLOAT32                  shadowMaxDistance       = 0.0f;
            FLOAT32                  shadowBias              = 0.0f;
            FLOAT32                  shadowNormalBias        = 0.0f;
            UINT32                   cullingMask             = 0xFFFFFFFFu;
            UINT32                   depthMode               = 0u;
            BOOL                     rasterAppliedDepthBias  = false;
            BOOL                     rasterAppliedNormalBias = false;
            NVSClipmapHierarchyState hierarchy{};
            NVSClipmapLevelGPUData   cachedViews[kNVSMaxClipmapLevels]{};
        };

        struct NVSLocalLightState
        {
            NVSClipmapLevelGPUData cachedViews[6]{};
            const Scene*           scenePtr                = nullptr;
            UINT64                 lightRuntimeUid         = 0ull;
            UINT64                 dataVersion             = 1ull;
            UINT64                 lastActiveFrame         = 0ull;
            Vec3                   position                = Vec3::Zero();
            Vec3                   direction               = Vec3::Zero();
            LightType              lightType               = LightType::SPOT;
            UINT32                 viewSlotBase            = 0u;
            UINT32                 viewCount               = 1u;
            UINT32                 viewGeneration          = 1u;
            UINT32                 effectivePagesPerAxis   = 0u;
            LightShadowResolution  shadowResolution        = LightShadowResolution::RES_16384;
            FLOAT32                range                   = 0.0f;
            FLOAT32                innerConeAngle          = 0.0f;
            FLOAT32                outerConeAngle          = 0.0f;
            FLOAT32                shadowNearPlane         = 0.1f;
            FLOAT32                shadowMaxDistance       = 0.0f;
            FLOAT32                shadowBias              = 0.0f;
            FLOAT32                shadowNormalBias        = 0.0f;
            UINT32                 cullingMask             = 0xFFFFFFFFu;
            UINT32                 depthMode               = 0u;
            BOOL                   initialized             = false;
            BOOL                   enableVirtualMipChain   = true;
            BOOL                   rasterAppliedDepthBias  = false;
            BOOL                   rasterAppliedNormalBias = false;
        };
#ifdef _MSC_VER
#pragma warning(pop)
#endif

        struct CameraNVSProductionState
        {
            NVSClipmapHierarchyState             directionalHierarchies[kNVSMaxClipmapLights]{};
            BOOL                                 directionalInitialized[kNVSMaxClipmapLights]{};
            std::vector<LightNVSViewAssignment>  committedLightAssignments;
            std::vector<UINT64>                  uploadedSlotVersion;
            UINT32                               committedActiveViewCount     = 0u;
            BOOL                                 hadActivePipelineLastFrame   = false;
            BOOL                                 cameraViewTableInitialized   = false;
            FLOAT32                              adaptiveLodBias              = 0.0f;
            UINT32                               adaptiveLowPressureSamples   = 0u;
            UINT32                               adaptiveBiasLastChangeFrame  = 0u;
            UINT32                               lastFeedbackUniquePages      = 0u;
            UINT32                               feedbackReadbackRequestFrame = 0u;
            BOOL                                 feedbackReadbackInFlight     = false;
        };

        CameraNVSSlotSignature m_globalNVSSlotSignatures[kNVSClipmapTableMaxEntries]{};
        UINT32                 m_nvsViewSlotWatermark = 0u;
        UINT32                 m_nvsFirstFreeSlotHint = 0u;
        UINT64                 m_nvsViewTableEpoch    = 1ull;
        std::vector<UINT8>     m_nvsViewSlotOccupied;
        std::unordered_map<UINT64, NVSLocalLightState> m_nvsLocalLightStates;
        std::unordered_map<NVSDirectionalCameraKey, NVSDirectionalLightCameraState, NVSDirectionalCameraKeyHash> m_nvsDirectionalLightStates;

        struct MaterialShadowStateSnapshot
        {
            UINT32           stateRevision           = 0u;
            const ShaderSet* shadowShaderSet         = nullptr;
            UINT64           shadowShaderSetRevision = 0u;
            CullMode         cullMode                = CullMode::BACK;
            BOOL             alphaTested             = false;
            BOOL             twoSided                = false;
        };

        FrameView                                      m_frameView;
        std::vector<LayeredDepthPassView>              m_pendingLayeredDepthPasses;
        std::unordered_set<const ComputeKernel*>       m_executedPerFrameKernels;
        std::chrono::high_resolution_clock::time_point m_frameStartTime;
        std::chrono::steady_clock::time_point          m_lastBeginFrameTime{};
        BOOL                                           m_hasLastBeginFrameTime      = false;
        FLOAT64                                        m_accumulatedTimeSeconds     = 0.0;
        FLOAT32                                        m_currentDeltaTimeSeconds    = 0.0f;
        PostCameraRenderCallback                       m_postCameraCallback         = nullptr;
        void*                                          m_postCameraCallbackUserData = nullptr;
        PrePresentCallback                             m_prePresentCallback         = nullptr;
        PostFrameExecutionCallback                     m_postFrameExecutionCallback = nullptr;
        BOOL                                           m_postFrameWaitForGPU        = true;
        UINT32                                         m_windowWidth                = 1280;
        UINT32                                         m_windowHeight               = 720;
        BOOL                                           m_vsyncEnabled               = true;
        BOOL                                           m_gpuSceneUpdateEnabled      = true;
        BOOL                                           m_nvsFramePipelineEnabled    = false;
        BOOL                                           m_nvsHasCustomPipelineConfig = false;
        NVSFeedbackPassDesc                            m_nvsFramePipelineConfig{};
        std::unordered_map<Camera*, NVSClipmapHierarchyState> m_cameraNVSClipmapStates;
        std::unordered_map<Camera*, CameraNVSProductionState> m_cameraNVSProductionStates;
        std::unordered_map<Scene*, std::unordered_map<const Material*, MaterialShadowStateSnapshot>> m_sceneMaterialShadowStates;
        BOOL                                           m_initialized                = false;
        std::thread::id                                m_ownerThreadId{};
        std::string                                    m_lastError;

        void AssertOwnerThread() const noexcept;

        BOOL EnsureInternalNVSSamplers();
        void DestroyInternalNVSSamplers();
        void ClearAllCameraNVSTrackingStates();
        UINT32 AllocateNVSViewSlots(UINT32 viewCount, BOOL require4Alignment);
        void ReclaimNVSViewSlots(UINT32 viewSlotBase, UINT32 viewCount, BOOL queueGpuReclaim);
        Vec4 ComputeCameraNVSScreenPixelWorldScale(const Camera* camera, BOOL sparseEnabled) const;
        void CollectSceneMaterialShadowInvalidations(
            Scene*                                       scene,
            std::vector<NVSDynamicCasterInvalidationBox>& ioBoxes);
        void PopulateBaseNVSSamplingContext(
            NVSDepthCompareMode      depthMode,
            FrameNVSSamplingContext& outContext) const;
        void BuildStandaloneNVSSamplingContext(
            Scene*                   scene,
            Camera*                  camera,
            FrameNVSSamplingContext& outContext) const;
        void RenderCameraWithProductionNVS(
            Scene*                                              scene,
            Camera*                                             camera,
            const std::vector<NVSDynamicCasterInvalidationBox>& frameInvalidationBoxes);
        void EnsureCameraSceneDepthTexture(Camera* camera);
        void ReleaseAllCameraSceneDepthTextures();
        void FlushRetiredSceneStaticBatches(Scene* scene);
        void InjectCustomNVSFramePipelinePasses(
            Scene*                                              scene,
            Camera*                                             camera,
            const std::vector<NVSDynamicCasterInvalidationBox>& frameInvalidationBoxes,
            FrameView&                                          ioView);
    };
}

// ==================== DYNAMIC MODULE BOUNDARY ====================
extern "C" {
    NORTHSTAR_CORE Northstar::UINT32            Northstar_GetAPIVersion();
    NORTHSTAR_CORE Northstar::NorthstarEngine*  CreateNorthstarEngine();
    NORTHSTAR_CORE void                         DestroyNorthstarEngine(Northstar::NorthstarEngine* engine);
}
