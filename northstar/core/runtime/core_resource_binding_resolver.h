#pragma once

#include "../../include/types/northstar_types.h"
#include "../../include/types/northstar_math.h"
#include "../../include/platform/frame_view.h"
#include "../../include/shader/shader.h"
#include "../../include/shader/shader_property_store.h"
#include "../../include/material/material.h"
#include "../../include/render_object/render_object.h"
#include "../../include/scene/scene.h"
#include "../../include/scene/gpu_scene_types.h"
#include "../../include/nvg/nvg_types.h"
#include "../../include/shadow/nvs_virtual_shadow_types.h"
#include "../../include/buffer/gpu_buffer.h"
#include "../../include/stats/northstar_stats.h"
#include "core_buffer_manager.h"
#include "internal_resource_manager.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace Northstar
{
    // =====================================================================
    //  Execution Context Model
    //  Governs resource availability, neutral/invalid sentinels, and domain
    //  validity uniformly across Graphics, Shadow, NVG, and Compute passes.
    // =====================================================================
    enum ResourceExecutionContextBits : UINT32
    {
        EXEC_CTX_NONE             = 0u,
        EXEC_CTX_SCENE            = 1u << 0, // Active Scene exists
        EXEC_CTX_FRAME            = 1u << 1, // Active Frame snapshot exists
        EXEC_CTX_CAMERA_VIEW      = 1u << 2, // Camera/View matrices & frustum active
        EXEC_CTX_DRAW             = 1u << 3, // Graphics draw invocation
        EXEC_CTX_RENDER_OBJECT    = 1u << 4, // Single RenderObject bound (e.g. non-instanced STANDARD draw)
        EXEC_CTX_INSTANCE         = 1u << 5, // Per-instance hardware batch draw
        EXEC_CTX_CLUSTER          = 1u << 6, // NVG virtual geometry cluster draw (Pull or Decoded-IA)
        EXEC_CTX_SHADOW_PAGE      = 1u << 7, // NVS / NVG layered shadow page rasterization
        EXEC_CTX_COMPUTE_DISPATCH = 1u << 8  // ComputeKernel dispatch
    };

    // =====================================================================
    //  Physical Engine Resource Domains (Owned by real GPU/CPU providers,
    //  NEVER inferred by name prefix alone or hardcoded register slots).
    // =====================================================================
    enum class EngineResourceDomain : UINT8
    {
        USER_ONLY = 0,
        INSTANCES,                 // Tier-1 per-draw instance payload (Get(INSTANCE_*) or Add(worldMatrix))
        INSTANCE_SKINNING,         // Per-instance skinning metadata (INSTANCE_BONE_OFFSET / COUNT)
        LIGHTS,                    // Scene lights array (GLOBAL_LIGHT_*)
        DECALS,                    // Scene decals array (GLOBAL_DECAL_*)
        REFLECTION_PROBES,         // Scene probes array (GLOBAL_PROBE_*)
        PARTICLES,                 // Scene particles array (GLOBAL_PARTICLE_*)
        GLOBAL_BONE_PALETTE,       // Scene-wide bone matrix table (GLOBAL_BONE_MATRIX -> Descriptor Table SRV)
        PER_OBJECT_BONE_SLICE,     // Per-object bone palette slice (BONE_MATRIX -> offset slice / Root SRV)
        NVS_SPARSE_PAGE_HIERARCHY, // NVS sparse page directory (NVS_SPARSE_NODE_VALUE, 4B)
        NVS_VIRTUAL_PAGE_TABLE,    // NVS virtual page table (NVS_VPAGE_*, 16B)
        NVS_PHYSICAL_PAGE_TABLE,   // NVS physical page metadata (NVS_PPAGE_*, 32B)
        NVS_LIGHT_VIEW_TABLE,      // NVS light view / clipmap table (NVS_VIEW_*, 256B)
        GPU_SCENE_OBJECTS,         // Scene-wide GPUSceneObjectData (NVG_OBJ_*, 160B) — LIVE on STANDARD & VIRTUALIZED
        NVG_PAGE_POOL,             // NVG physical 32KiB compressed geometry page pool (NVG_PAGE_POOL_WORD, 4B) — NVG only
        NVG_BUCKET_RANGES,         // NVG draw bucket ranges (NVG_BUCKET_*, 16B) — NVG only
        NVG_VISIBLE_CLUSTERS,      // NVG compacted visible cluster records (NVG_VIS_*, 48B) — NVG only
        NVG_SHADOW_PAGE_VIEWS,     // NVG per-active-shadow-page views (NVG_SHADOW_PAGE_*, 80B) — NVG + Shadow only
        INVALID_MIXED_DOMAINS      // Deterministic rejection: struct mixes incompatible physical domains
    };

    struct EnginePropertyMetadata
    {
        const char*          propertyName            = nullptr;
        EngineResourceDomain domain                  = EngineResourceDomain::USER_ONLY;
        UINT32               validContextMask        = EXEC_CTX_NONE;
        UINT32               canonicalSrcByteOffset  = 0xFFFFFFFFu;
        UINT32               canonicalSrcByteSize    = 0u;
        UINT32               canonicalSrcStride      = 0u;
        BOOL                 isNVGOnly               = false;
        BOOL                 isShadowPageOnly        = false;
        BOOL                 isPerDrawDynamic        = false;
    };

    // GPU copy-span entry matching RuntimeReflectionMaterialize.nsl (16 bytes)
    struct alignas(16) MaterializeCopySpanGPU
    {
        UINT32 srcWordOffset = 0u;
        UINT32 dstWordOffset = 0u;
        UINT32 wordCount     = 0u;
        UINT32 flags         = 0u;
    };
    static_assert(sizeof(MaterializeCopySpanGPU) == 16, "MaterializeCopySpanGPU must be 16 bytes.");

    struct StructuredBufferLayoutPlan
    {
        EngineResourceDomain                domain                 = EngineResourceDomain::USER_ONLY;
        BOOL                                valid                  = true;
        BOOL                                canDirectZeroCopy      = false;
        BOOL                                requiresMaterialize    = false;
        BOOL                                hasAddFields           = false;
        BOOL                                hasGetFields           = false;
        BOOL                                hasInstanceSkinFields  = false;
        UINT32                              canonicalSrcStride     = 0u;
        UINT32                              dstStride              = 0u;
        UINT64                              layoutHash             = 0u;
        std::vector<MaterializeCopySpanGPU> copySpans;
        std::string                         diagnostic;
    };

    // =====================================================================
    //  CoreGPUResourceMaterializer (Decision 6C)
    //  Fence-safe 3-frame-in-flight GPU Gather/Pack materializer producing
    //  reflection-compatible StructuredBuffers with ZERO CPU readback and
    //  ZERO WaitForGPU() stalls.
    // =====================================================================
    class CoreGPUResourceMaterializer
    {
    public:
        static constexpr UINT32 kFramePoolCount             = 3u;
        static constexpr UINT64 kDefaultMaxVramBudgetBytes  = 64ull * 1024ull * 1024ull; // 64 MiB

        CoreGPUResourceMaterializer() = default;
        ~CoreGPUResourceMaterializer();

        BOOL Initialize(CoreBufferManager* bufferManager, InternalResourceManager* internalResources);
        void Shutdown();

        void BeginFrame(UINT64 frameIndex);

        void   SetMaxVramBudgetBytes(UINT64 maxBytes) { m_maxVramBudgetBytes = maxBytes; }
        UINT64 GetMaxVramBudgetBytes() const          { return m_maxVramBudgetBytes; }

        UINT32  GetMaterializationsThisFrame() const  { return m_materializationsThisFrame; }
        UINT32  GetCacheHitsThisFrame() const         { return m_cacheHitsThisFrame; }
        UINT32  GetOverflowRejectionsThisFrame() const { return m_overflowRejectionsThisFrame; }
        UINT64  GetCurrentVramBytes() const           { return m_totalAllocatedVramBytes; }
        FLOAT32 GetCurrentVramMB() const              { return static_cast<FLOAT32>(m_totalAllocatedVramBytes) / (1024.0f * 1024.0f); }

        struct TransactionSnapshot
        {
            UINT32 poolIndex        = 0u;
            UINT32 matCount         = 0u;
            UINT32 hitCount         = 0u;
            SIZE_T insertedKeysSize = 0u;
        };

        TransactionSnapshot BeginTransaction() const
        {
            TransactionSnapshot s{};
            s.poolIndex        = m_nextPoolIndexInSlot;
            s.matCount         = m_materializationsThisFrame;
            s.hitCount         = m_cacheHitsThisFrame;
            s.insertedKeysSize = m_frameInsertedKeys.size();
            return s;
        }

        void RollbackTransaction(const TransactionSnapshot& snap);

        // Materializes a reflection-compatible StructuredBuffer on GPU from `srcHandle`.
        // Reuses cached materialized buffer if (frameIndex, srcHandle, srcRevision, srcBaseElement,
        // engineCount, userCount, layoutHash, userDataHash) matches an earlier request this frame.
        // Appends the GPU ComputeDispatchCommand to `outDispatches` on cache miss.
        BOOL AcquireMaterializedBuffer(
            const StructuredBufferLayoutPlan&    plan,
            NativeHandle                         srcHandle,
            UINT32                               srcByteSize,
            UINT64                               srcRevision,
            UINT32                               srcBaseElement,
            UINT32                               engineElementCount,
            const BYTE*                          userTailData,
            UINT32                               userTailByteSize,
            UINT32                               userElementCount,
            std::vector<ComputeDispatchCommand>& outDispatches,
            NativeHandle&                        outDstBufferHandle,
            UINT32&                              outDstElementCount,
            UINT32&                              outDstByteSize,
            std::string*                         outDiagnostic = nullptr);

    private:
        struct PooledMaterializeBuffer
        {
            GPUBuffer* buffer    = nullptr; // stride = 4B UINT32 words (SRV + UAV)
            UINT32     wordCount = 0u;
        };

        struct CacheKey
        {
            UINT64       frameIndex         = 0ull;
            NativeHandle srcHandle          = 0u;
            UINT64       srcRevision        = 0ull;
            UINT32       srcBaseElement     = 0u;
            UINT32       engineElementCount = 0u;
            UINT32       userElementCount   = 0u;
            UINT64       layoutHash         = 0ull;
            UINT64       userDataHash       = 0ull;

            bool operator==(const CacheKey& o) const noexcept
            {
                return frameIndex         == o.frameIndex &&
                       srcHandle          == o.srcHandle &&
                       srcRevision        == o.srcRevision &&
                       srcBaseElement     == o.srcBaseElement &&
                       engineElementCount == o.engineElementCount &&
                       userElementCount   == o.userElementCount &&
                       layoutHash         == o.layoutHash &&
                       userDataHash       == o.userDataHash;
            }
        };

        struct CacheKeyHash
        {
            SIZE_T operator()(const CacheKey& k) const noexcept;
        };

        struct CachedMaterializeResult
        {
            NativeHandle bufferHandle = 0u;
            UINT32       elementCount = 0u;
            UINT32       byteSize     = 0u;
        };

        GPUBuffer* AllocatePooledWordBuffer(UINT32 requiredWords, std::string* outDiagnostic);

        CoreBufferManager*       m_bufferManager     = nullptr;
        InternalResourceManager* m_internalResources = nullptr;
        ComputeKernel*           m_materializeKernel = nullptr;
        BOOL                     m_initialized       = false;

        UINT64 m_currentFrameIndex           = 0ull;
        UINT32 m_activePoolSlot              = 0u;
        UINT32 m_nextPoolIndexInSlot         = 0u;
        UINT64 m_maxVramBudgetBytes          = kDefaultMaxVramBudgetBytes;
        UINT64 m_totalAllocatedVramBytes     = 0ull;
        UINT32 m_materializationsThisFrame   = 0u;
        UINT32 m_cacheHitsThisFrame          = 0u;
        UINT32 m_overflowRejectionsThisFrame = 0u;

        std::vector<PooledMaterializeBuffer>                                 m_framePools[kFramePoolCount];
        std::unordered_map<CacheKey, CachedMaterializeResult, CacheKeyHash>  m_frameCache;
        std::vector<CacheKey>                                                m_frameInsertedKeys;
    };

    // =====================================================================
    //  Stage Binding Context passed into CoreResourceBindingResolver
    // =====================================================================
    struct StageResolveContext
    {
        UINT32                            executionContextMask  = EXEC_CTX_FRAME;
        const FrameView*                  frameView             = nullptr;
        const Scene*                      scene                 = nullptr;
        const RenderObject*               renderObject          = nullptr;
        const Material*                   material              = nullptr;
        PassType                          passType              = PassType::GEOMETRY;
        Mat4                              worldMatrix           = Mat4::Identity();
        Mat4                              prevWorldMatrix       = Mat4::Identity();
        UINT32                            gpuSceneObjectSlot    = kInvalidGPUSceneSlot;
        UINT32                            geometryMode          = kNVGInvalidIndex; // 0 = STANDARD, 1 = VIRTUALIZED, 0xFFFFFFFF = non-draw (e.g. Compute)
        UINT32                            nvgDrawBucketId       = kNVGInvalidIndex;
        UINT32                            nvgDrawContextId      = kNVGInvalidIndex;
        UINT32                            nvgVisibleClusterOff  = kNVGInvalidIndex;
        UINT32                            nvgVisibleClusterCnt  = 0u;
        UINT32                            nvgDebugMode          = 0u;
        FLOAT32                           nvgMaxScreenErrorPx   = 1.0f;
        UINT32                            nvgPhysicalPoolCap    = 0u;
        NativeHandle                      nvgPagePoolBuffer     = 0u;
        UINT32                            nvgPagePoolWords      = 0u;
        NativeHandle                      nvgBucketRangesBuffer = 0u;
        UINT32                            nvgBucketRangesCount  = 0u;
        NativeHandle                      nvgVisibleClustersBuf = 0u;
        UINT32                            nvgVisibleClustersCnt = 0u;
        NativeHandle                      nvgShadowPageViewsBuf = 0u;
        UINT32                            nvgShadowPageViewsCnt = 0u;
        UINT64                            nvgPassRevision       = 0ull;
        const IndirectRootBindingKey*     promotedRootKey       = nullptr;
        CoreGPUResourceMaterializer*      materializer          = nullptr;
        std::vector<ComputeDispatchCommand>* outPreDrawDispatches = nullptr;
        BOOL                              allowEmptyUserSBAsNull = false; // True for Compute / non-strict optional SRVs
    };

    // =====================================================================
    //  CoreResourceBindingResolver — Single Source of Truth across
    //  CoreRenderPipeline, CoreNVGManager, CoreNVSShadowRasterizer, Compute
    // =====================================================================
    class CoreResourceBindingResolver
    {
    public:
        static const EnginePropertyMetadata* FindPropertyMetadata(const std::string& propertyName);

        static StructuredBufferLayoutPlan AnalyzeStructuredBuffer(
            const NSL::NSCStructuredBuffer& nscSb,
            ShaderStage                     stage,
            UINT32                          userElementCount = 0u);

        static EngineResourceDomain DetectStructuredBufferDomain(
            const NSL::NSCStructuredBuffer& nscSb,
            ShaderStage                     stage);

        static BOOL IsCBufferFieldPerDrawDynamic(const std::string& propertyName);
        static BOOL IsCBufferFieldShadowPageDynamic(const std::string& propertyName);
        static UINT32 MapPropertyNameToPatchSemantic(const std::string& propertyName);

        static void FillEngineConstantBufferField(
            const std::string&        propName,
            const StageResolveContext& ctx,
            BYTE*                     dest,
            UINT32                    offset,
            UINT32                    size,
            UINT32                    bufTotalSize);

        static BOOL FillEngineLightsElementField(
            const FrameLightData&       light,
            const NSL::NSCStructField&  field,
            BYTE*                       elemPtr,
            UINT32                      stride);

        static BOOL IsTier1InstanceStructuredBuffer(
            const NSL::NSCStructuredBuffer& nscSb,
            ShaderStage                     stage);

        static BOOL DoesShaderRequireEngineSkinning(const ShaderReflection& refl);
        static BOOL DoesShaderUseGlobalBoneOffset(const ShaderReflection& refl);

        // Checks whether a StructuredBuffer layout is valid (rejects mixed incompatible domains)
        // and reports whether it requires GPU materialization.
        static BOOL ValidateStructuredBufferLayout(
            const NSL::NSCStructuredBuffer& nscSb,
            ShaderStage                     stage,
            std::string*                    outDiagnostic = nullptr);

        // Resolves all CBVs, Textures, Samplers, and StructuredBuffers for a single graphics shader stage.
        // Enforces CST (N engine + M user, Null Buffer SRV when N+M==0), strict validation on missing Add()
        // resources, Direct Zero-Copy fast path, and Reflection-Driven GPU Materialization (Decision 6C).
        // Also populates reflected Tier-1 instance binding coordinates on `outHasReflectedInstance`.
        static BOOL ResolveGraphicsStageBindings(
            const Shader*                               shader,
            ShaderStage                                 stage,
            const ShaderStageProps&                     primaryProps,
            const ShaderStageProps*                     fallbackProps,
            const StageResolveContext&                  ctx,
            std::vector<NativeConstantBufferBinding>&   outCBuffers,
            std::vector<NativeTextureBinding>&          outTextures,
            std::vector<NativeSamplerBinding>&          outSamplers,
            std::vector<NativeStructuredBufferBinding>& outStructuredBuffers,
            BOOL*                                       outHasReflectedInstance = nullptr,
            UINT16*                                     outInstanceSpace        = nullptr,
            UINT16*                                     outInstanceSlot         = nullptr,
            std::string*                                outDiagnostic           = nullptr,
            std::vector<NativeRWTextureBinding>*        outRWTextures           = nullptr);

        // Resolves an engine-owned StructuredBuffer (SRV) for either Graphics or Compute stages,
        // applying Direct Zero-Copy, GPU Materialization, CPU Lights packing, or CST Null Buffer SRV.
        static BOOL ResolveEngineStructuredBuffer(
            const NSL::NSCStructuredBuffer&             nscSb,
            ShaderStage                                 stage,
            const StructuredBufferLayoutPlan&           plan,
            const ShaderStructuredBufferSlot*           matSb,
            const StageResolveContext&                  ctx,
            NativeStructuredBufferBinding&              outBinding,
            std::string*                                outDiagnostic = nullptr);
    };
}
