#pragma once

#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "../types/pipeline_state_types.h"
#include "../shader/shader_types.h"
#include "../mesh/vertex.h"
#include "../buffer/gpu_buffer_types.h"
#include "../buffer/gpu_indirect_args.h"
#include "../texture/texture_types.h"
#include "native_render_target.h"

#include <memory>
#include <vector>

namespace Northstar
{
    // ==================== RESOURCE BINDINGS ====================
    struct NativeTextureBinding
    {
        UINT16        slotIndex     = 0;
        UINT16        registerSpace = 0;
        ShaderStage   stage         = ShaderStage::PIXEL;
        NativeHandle  texture       = 0;
        TextureType   nullViewType  = TextureType::TEXTURE_2D;
        TextureFormat nullFormat    = TextureFormat::RGBA8_UNORM;
        UINT32        mipSlice      = 0xFFFFFFFFu; // 0xFFFFFFFF = full mip chain SRV; < mipLevels = single-mip SRV
    };

    struct NativeSamplerBinding
    {
        UINT16       slotIndex     = 0;
        UINT16       registerSpace = 0;
        ShaderStage  stage         = ShaderStage::PIXEL;
        NativeHandle sampler       = 0;
    };

    struct NativeConstantBufferBinding
    {
        UINT16            slotIndex     = 0;
        UINT16            registerSpace = 0;
        ShaderStage       stage         = ShaderStage::VERTEX;
        const BYTE*       cpuData       = nullptr;
        std::vector<BYTE> ownedData;
        UINT32            byteSize      = 0;
        BOOL              dirty         = false;
        UINT64            sharedDataId  = 0;

        NativeConstantBufferBinding() = default;
        NativeConstantBufferBinding(const NativeConstantBufferBinding& other)
            : slotIndex(other.slotIndex)
            , registerSpace(other.registerSpace)
            , stage(other.stage)
            , cpuData(other.cpuData)
            , ownedData(other.ownedData)
            , byteSize(other.byteSize)
            , dirty(other.dirty)
            , sharedDataId(other.sharedDataId)
        {
            if (!ownedData.empty()) cpuData = ownedData.data();
        }
        NativeConstantBufferBinding& operator=(const NativeConstantBufferBinding& other)
        {
            if (this == &other) return *this;
            slotIndex     = other.slotIndex;
            registerSpace = other.registerSpace;
            stage         = other.stage;
            cpuData       = other.cpuData;
            ownedData     = other.ownedData;
            byteSize      = other.byteSize;
            dirty         = other.dirty;
            sharedDataId  = other.sharedDataId;
            if (!ownedData.empty()) cpuData = ownedData.data();
            return *this;
        }
        NativeConstantBufferBinding(NativeConstantBufferBinding&& other) noexcept
            : slotIndex(other.slotIndex)
            , registerSpace(other.registerSpace)
            , stage(other.stage)
            , cpuData(other.cpuData)
            , ownedData(std::move(other.ownedData))
            , byteSize(other.byteSize)
            , dirty(other.dirty)
            , sharedDataId(other.sharedDataId)
        {
            if (!ownedData.empty()) cpuData = ownedData.data();
        }
        NativeConstantBufferBinding& operator=(NativeConstantBufferBinding&& other) noexcept
        {
            if (this == &other) return *this;
            slotIndex     = other.slotIndex;
            registerSpace = other.registerSpace;
            stage         = other.stage;
            cpuData       = other.cpuData;
            ownedData     = std::move(other.ownedData);
            byteSize      = other.byteSize;
            dirty         = other.dirty;
            sharedDataId  = other.sharedDataId;
            if (!ownedData.empty()) cpuData = ownedData.data();
            return *this;
        }

        const BYTE* GetData() const
        {
            if (!ownedData.empty()) return ownedData.data();
            return cpuData;
        }
    };

    struct NativeStructuredBufferBinding
    {
        UINT16            slotIndex     = 0;
        UINT16            registerSpace = 0;
        ShaderStage       stage         = ShaderStage::VERTEX;
        NativeHandle      buffer        = 0;
        const BYTE*       cpuData       = nullptr;
        std::vector<BYTE> ownedData;
        UINT32            elementCount  = 0;
        UINT32            firstElement  = 0;
        UINT32            stride        = 0;
        UINT32            byteSize      = 0;
        BOOL              isUAV         = false;
        GPUResourceAccess uavAccess     = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        UINT64            sharedDataId  = 0;

        NativeStructuredBufferBinding() = default;
        NativeStructuredBufferBinding(const NativeStructuredBufferBinding& other)
            : slotIndex(other.slotIndex)
            , registerSpace(other.registerSpace)
            , stage(other.stage)
            , buffer(other.buffer)
            , cpuData(other.cpuData)
            , ownedData(other.ownedData)
            , elementCount(other.elementCount)
            , firstElement(other.firstElement)
            , stride(other.stride)
            , byteSize(other.byteSize)
            , isUAV(other.isUAV)
            , uavAccess(other.uavAccess)
            , sharedDataId(other.sharedDataId)
        {
            if (ownedData.empty() && other.cpuData != nullptr && other.byteSize > 0u)
            {
                ownedData.assign(other.cpuData, other.cpuData + other.byteSize);
            }
            if (!ownedData.empty()) cpuData = ownedData.data();
        }
        NativeStructuredBufferBinding& operator=(const NativeStructuredBufferBinding& other)
        {
            if (this == &other) return *this;
            slotIndex     = other.slotIndex;
            registerSpace = other.registerSpace;
            stage         = other.stage;
            buffer        = other.buffer;
            cpuData       = other.cpuData;
            ownedData     = other.ownedData;
            elementCount  = other.elementCount;
            firstElement  = other.firstElement;
            stride        = other.stride;
            byteSize      = other.byteSize;
            isUAV         = other.isUAV;
            uavAccess     = other.uavAccess;
            sharedDataId  = other.sharedDataId;
            if (ownedData.empty() && other.cpuData != nullptr && other.byteSize > 0u)
            {
                ownedData.assign(other.cpuData, other.cpuData + other.byteSize);
            }
            if (!ownedData.empty()) cpuData = ownedData.data();
            return *this;
        }
        NativeStructuredBufferBinding(NativeStructuredBufferBinding&& other) noexcept
            : slotIndex(other.slotIndex)
            , registerSpace(other.registerSpace)
            , stage(other.stage)
            , buffer(other.buffer)
            , cpuData(other.cpuData)
            , ownedData(std::move(other.ownedData))
            , elementCount(other.elementCount)
            , firstElement(other.firstElement)
            , stride(other.stride)
            , byteSize(other.byteSize)
            , isUAV(other.isUAV)
            , uavAccess(other.uavAccess)
            , sharedDataId(other.sharedDataId)
        {
            if (!ownedData.empty()) cpuData = ownedData.data();
        }
        NativeStructuredBufferBinding& operator=(NativeStructuredBufferBinding&& other) noexcept
        {
            if (this == &other) return *this;
            slotIndex     = other.slotIndex;
            registerSpace = other.registerSpace;
            stage         = other.stage;
            buffer        = other.buffer;
            cpuData       = other.cpuData;
            ownedData     = std::move(other.ownedData);
            elementCount  = other.elementCount;
            firstElement  = other.firstElement;
            stride        = other.stride;
            byteSize      = other.byteSize;
            isUAV         = other.isUAV;
            uavAccess     = other.uavAccess;
            sharedDataId  = other.sharedDataId;
            if (!ownedData.empty()) cpuData = ownedData.data();
            return *this;
        }

        const BYTE* GetData() const
        {
            if (!ownedData.empty()) return ownedData.data();
            return cpuData;
        }
    };

    struct NativeRWTextureBinding
    {
        UINT16            slotIndex     = 0;
        UINT16            registerSpace = 0;
        ShaderStage       stage         = ShaderStage::COMPUTE;
        NativeHandle      texture       = 0;
        UINT32            mipSlice      = 0;
        GPUResourceAccess uavAccess     = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
    };

    enum class ComputeDispatchProfileGroup : UINT8
    {
        GENERIC              = 0,
        FEEDBACK             = 1,
        RESIDENCY_CULL       = 2,
        NVG_HZB              = 3,
        NVG_CULL_LOD         = 4,
        NVG_DRAW_GEN         = 5,
        NVG_SW_RASTER        = 6,
        NVG_SW_RASTER_PASS1  = 7,
        NVG_SW_RASTER_PASS2  = 8
    };

    struct ComputeDispatchCommand
    {
        NativeHandle                               computeShader      = 0;
        UINT32                                     threadGroupCountX  = 1;
        UINT32                                     threadGroupCountY  = 1;
        UINT32                                     threadGroupCountZ  = 1;
        NativeHandle                               indirectArgsBuffer = 0;
        UINT32                                     indirectArgsOffset = 0;
        BOOL                                       useRootConstants   = false;
        UINT32                                     rootConstants[4]   = { 0, 0, 0, 0 }; // b0, space16
        ComputeDispatchProfileGroup                profileGroup       = ComputeDispatchProfileGroup::GENERIC;
        std::vector<NativeConstantBufferBinding>   constantBuffers;
        std::vector<NativeTextureBinding>          textures;
        std::vector<NativeSamplerBinding>          samplers;
        std::vector<NativeStructuredBufferBinding> structuredBuffers; // isUAV=false -> SRV (t), isUAV=true -> UAV (u)
        std::vector<NativeRWTextureBinding>        rwTextures;        // UAV (u)
    };

    enum class BufferCategory : UINT8
    {
        DYNAMIC_OBJECT = 0,
        STATIC_OBJECT  = 1,
        INSTANCE       = 2
    };

    struct InstanceData
    {
        Mat4    worldMatrix;
        UINT32  instanceId = 0;
        UINT32  customData = 0;
    };

    struct InstanceSkinningData
    {
        UINT32  bonePaletteOffset = 0;
        UINT32  boneCount         = 0;
        UINT32  skinningRevision  = 0;
    };

    // Root descriptor promotion layout for GPU-driven NVS indirect shadow draws:
    // Up to 3 per-draw/per-page Constant Buffers and 1 per-batch/per-slice StructuredBuffer SRV
    // are bound directly via 64-bit GPU Virtual Addresses inside NVSIndirectDrawCommand (96 bytes),
    // while all remaining user Add() textures, samplers, CBVs, and SRVs stay bound via descriptor tables.
    struct IndirectRootBindingKey
    {
        BOOL        enabled        = false;
        BOOL        extended       = false; // False: 3 CBVs + 1 SRV (96B NVS stride); True: 8 CBVs + 2 SRVs (144B NVG extended stride)
        UINT8       cbvCount       = 0;
        UINT8       srvCount       = 0;
        UINT16      cbvSpace[8]    = { 0, 0, 0, 0, 0, 0, 0, 0 };
        UINT16      cbvSlot[8]     = { 0, 0, 0, 0, 0, 0, 0, 0 };
        ShaderStage cbvStage[8]    = {
            ShaderStage::VERTEX, ShaderStage::VERTEX, ShaderStage::VERTEX, ShaderStage::VERTEX,
            ShaderStage::VERTEX, ShaderStage::VERTEX, ShaderStage::VERTEX, ShaderStage::VERTEX
        };
        BOOL        hasPromotedSRV = false;
        UINT16      srvSpace       = 0;
        UINT16      srvSlot        = 0;
        ShaderStage srvStage       = ShaderStage::VERTEX;
        UINT16      srv2Space      = 0;
        UINT16      srv2Slot       = 0;
        ShaderStage srv2Stage      = ShaderStage::VERTEX;

        bool operator==(const IndirectRootBindingKey& o) const noexcept
        {
            const UINT8 effSrvA = (srvCount > 0u) ? srvCount : (hasPromotedSRV ? 1u : 0u);
            const UINT8 effSrvB = (o.srvCount > 0u) ? o.srvCount : (o.hasPromotedSRV ? 1u : 0u);
            if (enabled != o.enabled || extended != o.extended || cbvCount != o.cbvCount || effSrvA != effSrvB)
                return false;
            if (!enabled)
                return true;
            const UINT8 maxCbv = extended ? 8u : 3u;
            for (UINT8 i = 0; i < cbvCount && i < maxCbv; ++i)
            {
                if (cbvSpace[i] != o.cbvSpace[i] || cbvSlot[i] != o.cbvSlot[i] || cbvStage[i] != o.cbvStage[i])
                    return false;
            }
            if (effSrvA >= 1u)
            {
                if (srvSpace != o.srvSpace || srvSlot != o.srvSlot || srvStage != o.srvStage)
                    return false;
            }
            if (effSrvA >= 2u)
            {
                if (srv2Space != o.srv2Space || srv2Slot != o.srv2Slot || srv2Stage != o.srv2Stage)
                    return false;
            }
            return true;
        }

        bool operator!=(const IndirectRootBindingKey& o) const noexcept
        {
            return !(*this == o);
        }
    };

    struct DrawCommand
    {
        NativeHandle         vertexBuffer = 0;
        NativeHandle         indexBuffer  = 0;
        UINT32               vertexCount  = 0;
        UINT32               indexCount   = 0;
        UINT32               startIndex   = 0;
        UINT32               baseVertex   = 0;

        BufferCategory                    bufferCategory            = BufferCategory::DYNAMIC_OBJECT;
        UINT32                            instanceCount             = 1;
        UINT32                            startInstance             = 0;
        NativeHandle                      instanceBuffer              = 0; // GPU StructuredBuffer or VB instancing
        UINT64                            sharedInstanceBatchId       = 0; // Non-zero when instances payload is shared across views
        BOOL                              hasReflectedInstanceBinding = false; // True when VS declares a Tier-1 instance buffer via reflection
        UINT16                            instanceBindingSlot         = 0xFFFFu; // Reflected SRV register slot (tN) for Tier-1 instances
        UINT16                            instanceBindingSpace        = 0u;      // Reflected SRV register space (spaceS) for Tier-1 instances
        std::vector<InstanceData>         instances;                     // CPU payload of instances for GPU upload
        std::vector<Mat4>                 instancePrevWorldMatrices;     // Per-instance previous world matrices (parallel to instances)
        std::vector<InstanceSkinningData> instanceSkinning;              // Per-instance skinning metadata (parallel to instances)

        // GPU-Driven Indirect Execution (ExecuteIndirect)
        // - InstanceCount is read per-command from GPUIndirectDrawIndexedArgs::instanceCount / GPUIndirectDrawArgs::instanceCount.
        // - CommandCount is maxIndirectCommandCount (when indirectCountBuffer == 0), or min(maxIndirectCommandCount, CountBuffer[indirectCountBufferOffset]) (when indirectCountBuffer != 0).
        NativeHandle                 indirectArgsBuffer             = 0; // GPUBuffer (GPUBufferUsage::INDIRECT_ARGUMENT)
        UINT32                       indirectArgsOffset             = 0; // 4-byte aligned byte offset in indirectArgsBuffer
        UINT32                       maxIndirectCommandCount        = 1; // Upper bound of commands (MaxCommandCount)
        NativeHandle                 indirectCountBuffer            = 0; // Optional GPUBuffer (GPUBufferUsage::INDIRECT_ARGUMENT, 4B UINT32 GPU command count)
        UINT32                       indirectCountBufferOffset      = 0; // 4-byte aligned byte offset in indirectCountBuffer
        BOOL                         useNVSIndirectCommandSignature = false; // Uses 96B promoted indirect signature (VBV + IBV + 3 RootCBVs + 1 RootSRV + DrawIndexed + 12B metadata)
        BOOL                         useVertexPullingIndirect       = false; // Vertex-pulling indirect draw (vertexBuffer == 0, 16B GPUIndirectDrawArgs)
        IndirectRootBindingKey       indirectRootKey{};
        NativeHandle                 indirectPerDrawCBVBuffer       = 0; // Optional GPU-generated CBV pool transitioned to VERTEX_AND_CONSTANT_BUFFER
        NativeHandle                 indirectPerBatchInstanceBuffer = 0; // Optional GPU-generated instance pool transitioned to ALL_SHADER_RESOURCE
        NativeHandle                 indirectBonePaletteBuffer      = 0; // Optional GPU bone palette transitioned to ALL_SHADER_RESOURCE
        NativeHandle                 indirectVertexPoolBuffer       = 0; // Optional GPU-decoded IA vertex pool transitioned to VERTEX_AND_CONSTANT_BUFFER
        NativeHandle                 indirectIndexPoolBuffer        = 0; // Optional GPU-decoded IA index pool transitioned to INDEX_BUFFER
        BOOL                         useGraphicsRootConstants       = false;
        UINT32                       graphicsRootConstants[4]       = { 0, 0, 0, 0 }; // b0, space16 (Vertex Shader 32-bit root constants)

        // CPU Geometry pointers (available if not yet uploaded to GPU or for debug)
        const Vertex*        cpuVertices  = nullptr;
        const UINT32*        cpuIndices   = nullptr;

        // World transform (primary matrix / fallback if single instance)
        Mat4                 worldMatrix;
        Mat4                 prevWorldMatrix = Mat4::Identity();

        // Shader native handles
        NativeHandle         vertexShader   = 0;
        NativeHandle         hullShader     = 0;
        NativeHandle         domainShader   = 0;
        NativeHandle         geometryShader = 0;
        NativeHandle         pixelShader    = 0;

        // Pipeline / Rasterizer states
        CullMode             cullMode   = CullMode::BACK;
        BlendMode            blendMode  = BlendMode::OPAQUE;
        FillMode             fillMode   = FillMode::SOLID;
        DepthState           depthState = {};
        BOOL                 pixelShaderModifiesCoverage = false;

        // Resource Bindings (Textures, RWTextures, Samplers, Constant Buffers, Structured Buffers)
        std::vector<NativeTextureBinding>          textures;
        std::vector<NativeRWTextureBinding>        rwTextures;
        std::vector<NativeSamplerBinding>          samplers;
        std::vector<NativeConstantBufferBinding>   constantBuffers;
        std::vector<NativeStructuredBufferBinding> structuredBuffers;

        // Sorting & identification keys
        FLOAT32              depthKey   = 0.0f; // Distance to camera (for front-to-back or back-to-front sorting)
        UINT64               sortKey    = 0;    // Combined state hash for minimizing GPU state switches
    };

    // ==================== FRAME CAMERA DATA ====================
    struct FrameCameraData
    {
        Mat4                 viewMatrix;
        Mat4                 projectionMatrix;
        Mat4                 viewProjectionMatrix;
        Mat4                 invViewProjectionMatrix = Mat4::Identity();
        Mat4                 prevViewMatrix          = Mat4::Identity();
        Mat4                 prevProjectionMatrix    = Mat4::Identity();
        Mat4                 prevViewProjectionMatrix= Mat4::Identity();
        Vec3                 position;
        Quat                 rotation;
        Vec3                 direction;
        FLOAT32              nearPlane   = 0.1f;
        FLOAT32              farPlane    = 1000.0f;
        FLOAT32              fov         = 60.0f;
        FLOAT32              aspectRatio = 16.0f / 9.0f;
        Frustum              frustum;
        NativeRenderTarget   renderTarget;
        FLOAT32              viewportX      = 0.0f;
        FLOAT32              viewportY      = 0.0f;
        FLOAT32              viewportWidth  = 0.0f;
        FLOAT32              viewportHeight = 0.0f;
    };

    // ==================== FRAME LIGHT DATA ====================
    struct FrameLightData
    {
        LightType              type           = LightType::DIRECTIONAL;
        Color                  color          = { 1.0f, 1.0f, 1.0f, 1.0f };
        FLOAT32                intensity      = 1.0f;
        Vec3                   position       = Vec3::Zero();
        Quat                   rotation       = Quat::Identity();
        Vec3                   direction      = Vec3(0.0f, 0.0f, 1.0f);
        FLOAT32                range          = 10.0f;
        Vec3                   attenuation    = Vec3(1.0f, 0.09f, 0.032f);
        FLOAT32                innerConeAngle = 30.0f;
        FLOAT32                outerConeAngle = 45.0f;
        Mat4                   viewProjectionMatrix = Mat4::Identity();
        UINT32                 castsShadow       = 0;
        LightShadowSoftness    shadowSoftness    = LightShadowSoftness::HARD;
        FLOAT32                shadowBias        = 0.005f;
        FLOAT32                shadowNormalBias  = 0.02f;
        FLOAT32                shadowNearPlane   = 0.1f;
        FLOAT32                shadowMaxDistance = 150.0f;
        // Northstar Virtual Shadows (NVS) light addressing metadata
        UINT32                 nvsLightId        = 0;
        UINT32                 shadowViewBase    = 0;
        UINT32                 shadowViewCount   = 0;
        UINT32                 cullingMask       = 0xFFFFFFFFu;
    };

    // ==================== FRAME ENVIRONMENT DATA ====================
    struct FrameEnvironmentData
    {
        Color                  ambientColor     = { 0.1f, 0.1f, 0.1f, 1.0f };
        FLOAT32                ambientIntensity = 1.0f;
    };

    // ==================== GEOMETRY PASS VIEW ====================
    struct GeometryPassView
    {
        std::vector<DrawCommand> opaqueCommands;      // Sorted for state batching & front-to-back
        std::vector<DrawCommand> transparentCommands; // Sorted strictly back-to-front

        void Clear()
        {
            opaqueCommands.clear();
            transparentCommands.clear();
        }
    };

    using BasePassView = GeometryPassView;

    // ==================== DIRTY DEPTH SLICE BUILD STATUS & OVERFLOW CONTRACT ====================
    // Layout of OutDirtyCount buffer written by NorthstarRuntime/Compute/RuntimeDirtyDepthSliceBuild:
    //   - Word [0] (emittedDirtyCount): Number of dirty slices actually written to OutDirtySliceIndices
    //     and emitted as GPUIndirectDrawArgs::instanceCount (always <= maxOutputSlices).
    //   - Word [1] (totalUniqueDirtyCount): Total valid unique dirty slices discovered across all candidates
    //     before capacity clamping (written when OutDirtyCount elementCount >= 4).
    //   - Word [2] (overflowFlag): 1 if totalUniqueDirtyCount > maxOutputSlices, 0 otherwise.
    //   - Word [3] (droppedSliceCount): Number of dirty slices NOT cleared in this batch
    //     (totalUniqueDirtyCount - emittedDirtyCount).
    enum class DirtySliceOverflowPolicy : UINT32
    {
        CLAMP_AND_FLAG    = 0, // Emit up to maxOutputSlices, set overflowFlag=1 & droppedSliceCount; dropped slices are NOT cleared
        ABORT_ON_OVERFLOW = 1  // On overflow, emit 0 slices (instanceCount=0), set overflowFlag=1 & droppedSliceCount=totalUniqueDirtyCount
    };

    // Optional bit flags that can be OR'd into inputMode of BuildNorthstarInternalDirtyDepthSliceDispatch
    // without changing the C ABI signature:
    inline constexpr UINT32 DIRTY_SLICE_INPUT_FLAG_ABORT_ON_OVERFLOW    = 0x0100u;
    inline constexpr UINT32 DIRTY_SLICE_INPUT_FLAG_PRESERVE_TAIL_CANARY = 0x1000u;

    struct DirtySliceBuildStatus
    {
        UINT32 emittedDirtyCount     = 0;
        UINT32 totalUniqueDirtyCount = 0;
        UINT32 overflowFlag          = 0;
        UINT32 droppedSliceCount     = 0;

        constexpr BOOL HasOverflow() const noexcept
        {
            return (overflowFlag != 0u) || (droppedSliceCount != 0u) || (totalUniqueDirtyCount > emittedDirtyCount);
        }

        constexpr BOOL IsCompleteClear() const noexcept
        {
            return !HasOverflow();
        }
    };

    // NVS Page Reuse & Selective Clear Verification Contract:
    // Determines whether a specific array slice was actually cleared in a selective depth clear batch.
    // Prevents any downstream page reuse / virtual shadow system from treating an overflow-dropped
    // slice as cleared and valid.
    inline BOOL IsSliceClearedInDirtyBatch(
        UINT32                       sliceIndex,
        const DirtySliceBuildStatus& status,
        const UINT32*                emittedSliceIndices,
        UINT32                       emittedBufferCapacity) noexcept
    {
        if (!emittedSliceIndices || status.emittedDirtyCount == 0u)
            return false;
        const UINT32 count = (status.emittedDirtyCount < emittedBufferCapacity)
            ? status.emittedDirtyCount
            : emittedBufferCapacity;
        for (UINT32 i = 0u; i < count; ++i)
        {
            if (emittedSliceIndices[i] == sliceIndex)
                return true;
        }
        return false;
    }

    // ==================== LAYERED DEPTH PASS & SELECTIVE CLEAR VIEW ====================
    struct LayeredDepthClearCommand
    {
        BOOL         enabled                   = false;
        FLOAT32      clearDepth                = 1.0f;
        NativeHandle clearVertexShader         = 0;   // Internal NorthstarRuntime/Depth/RuntimeLayeredDepthClearVS
        NativeHandle clearGeometryShader       = 0;   // Optional internal GS fallback (RuntimeLayeredDepthPassGS)
        NativeHandle dirtySliceIndicesBuffer   = 0;   // GPUBuffer (STRUCTURED_SRV) containing unique dirty slice indices
        NativeHandle indirectDrawArgsBuffer    = 0;   // GPUBuffer (INDIRECT_ARGUMENT) containing GPUIndirectDrawArgs (vertexCount=3, instanceCount=dirtyCount)
        UINT32       indirectDrawArgsOffset    = 0;   // 4-byte aligned byte offset in indirectDrawArgsBuffer
        NativeHandle indirectCountBuffer       = 0;   // Optional GPUBuffer (INDIRECT_ARGUMENT) for command count clamping
        UINT32       indirectCountBufferOffset = 0;   // 4-byte aligned byte offset in indirectCountBuffer
        UINT32       maxIndirectCommandCount   = 1;   // Upper bound of commands when indirectCountBuffer is used
        UINT32       maxDirtySlices            = 256; // Upper bound on dirty slices validated by clear shader

        // Explicitly separated CPU Per-Slice DSV Clear Fallback / Baseline comparison mode:
        // Never mixed with the true GPU-driven layered clear path.
        BOOL                useCpuPerSliceClearFallback = false;
        std::vector<UINT32> cpuDirtySlices;
    };

    struct LayeredDepthPassView
    {
        NativeHandle             depthArrayTexture      = 0;    // D32_FLOAT Texture2DArray (DEPTH_STENCIL)
        UINT32                   arrayLayerCount        = 0;    // 0 = full texture arrayLayers; >0 = validated <= arrayLayers
        FLOAT32                  viewportX              = 0.0f;
        FLOAT32                  viewportY              = 0.0f;
        FLOAT32                  viewportWidth          = 0.0f; // 0 = full texture width
        FLOAT32                  viewportHeight         = 0.0f; // 0 = full texture height
        INT32                    scissorLeft            = 0;
        INT32                    scissorTop             = 0;
        INT32                    scissorRight           = 0;    // 0 = full texture width
        INT32                    scissorBottom          = 0;    // 0 = full texture height
        INT32                    depthBias              = 0;    // Rasterizer DepthBias
        FLOAT32                  depthBiasClamp         = 0.0f; // Rasterizer DepthBiasClamp
        FLOAT32                  slopeScaledDepthBias           = 0.0f; // Rasterizer SlopeScaledDepthBias
        NativeHandle             fallbackGeometryShader         = 0;    // Internal GS fallback for selective layered depth clear when VPAndRTArrayIndexWithoutGS is false/forced
        NativeHandle             indirectPerDrawCBVBuffer       = 0;    // Optional GPU-generated per-draw CBV pool buffer transitioned to VERTEX_AND_CONSTANT_BUFFER
        NativeHandle             indirectPerBatchInstanceBuffer = 0;    // Optional GPU-generated per-batch instance SRV pool buffer transitioned to SHADER_RESOURCE
        NativeHandle             indirectBonePaletteBuffer      = 0;    // Optional GPU skinning bone palette buffer transitioned to SHADER_RESOURCE for Root SRV slices
        NativeHandle             indirectVertexPoolBuffer       = 0;    // Optional GPU-decoded IA vertex pool buffer transitioned to VERTEX_AND_CONSTANT_BUFFER
        NativeHandle             indirectIndexPoolBuffer        = 0;    // Optional GPU-decoded IA index pool buffer transitioned to INDEX_BUFFER
        LayeredDepthClearCommand selectiveClear;                        // Optional GPU-driven selective layered depth clear before drawCommands
        std::vector<DrawCommand> drawCommands;                          // Depth-only layered rasterization draw commands
    };

    enum class FramePassItemType : UINT8
    {
        COMPUTE_DISPATCH   = 0,
        GRAPHICS_DRAWS     = 1,
        DEPTH_PREPASS      = 2,
        LAYERED_DEPTH_PASS = 3
    };

    struct FramePassItem
    {
        FramePassItemType        type = FramePassItemType::COMPUTE_DISPATCH;
        ComputeDispatchCommand   computeDispatch;
        std::vector<DrawCommand> drawCommands;
        LayeredDepthPassView     layeredDepthPass;
    };

    // ==================== FRAME VIEW ====================
    // GPU-level frame snapshot produced by NorthstarCore.
    // Contains geometry pass draw commands, camera, lights, and ambient data ready for execution.
    // Backend must not see ShaderSet* / Material* — only native layout data.
    struct FrameView
    {
        UINT64                      frameIndex           = 0;
        FLOAT32                     deltaTime            = 0.0f;
        FLOAT32                     elapsedTime          = 0.0f;
        BOOL                        hasAuthoritativeTime = false;
        NativeRenderTarget          renderTarget;

        FrameCameraData             camera;
        FrameEnvironmentData        environment;
        BOOL                        hasDirectionalLight = false;
        FrameLightData              directionalLight;
        std::vector<FrameLightData> directionalLights;
        std::vector<FrameLightData> lights;
        UINT64                      lightsDataRevision         = 0;
        NativeHandle                sceneDepthTexture          = 0;
        BOOL                        hasDepthPrepass            = false;
        NativeHandle                gpuSceneBuffer             = 0;
        UINT32                      gpuSceneCapacity           = 0;
        UINT32                      gpuSceneHighWaterMark      = 0;
        UINT32                      gpuSceneActiveObjectCount  = 0;
        UINT64                      gpuSceneMutationEpoch      = 0;
        NativeHandle                gpuSkinningPaletteBuffer   = 0;
        UINT32                      gpuSkinningPaletteCount    = 0;

        // Engine-Owned Northstar Virtual Shadows (NVS) Sampling Resources & Metadata
        NativeHandle                nvsShadowMapTexture            = 0;
        NativeHandle                nvsComparisonSampler           = 0;
        NativeHandle                nvsDepthSampler                = 0;
        NativeHandle                nvsSparsePageHierarchyBuffer   = 0;
        UINT32                      nvsSparsePageHierarchyCount    = 0;
        UINT32                      nvsSparseRootCapacity          = 0;
        UINT32                      nvsSparseDirStride             = 352;
        UINT32                      nvsSparseDirCapacity           = 0;
        Vec4                        nvsScreenPixelWorldScale       = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        NativeHandle                nvsVirtualPageTableBuffer      = 0;
        UINT32                      nvsVirtualPageTableCount       = 0;
        UINT32                      nvsTotalVirtualPages           = 0;
        NativeHandle                nvsPhysicalPageTableBuffer     = 0;
        UINT32                      nvsPhysicalPageTableCount      = 0;
        UINT32                      nvsPhysicalPoolCapacity        = 0;
        NativeHandle                nvsLightViewTableBuffer        = 0;
        UINT32                      nvsLightViewTableCount         = 0;
        UINT32                      nvsViewTableCount              = 0;
        UINT32                      nvsPagesPerAxis                = 64;
        UINT32                      nvsPageTexelSize               = 128;
        UINT32                      nvsDepthMode                   = 0;
        UINT32                      nvsSamplingAvailable           = 0;
        UINT32                      nvsSnapshotFrameIndex          = 0;
        const BYTE*                 nvsLightViewTableCpuMirrorPtr  = nullptr;
        UINT32                      nvsLightViewTableCpuMirrorCount = 0;
        std::vector<BYTE>           nvsLightViewTableCpuMirror;

        // Engine-Owned Northstar Virtual Geometry (NVG) Runtime Buffers & Metadata
        NativeHandle                nvgVisibilityTexture           = 0;
        NativeHandle                nvgPagePoolBuffer              = 0;
        UINT32                      nvgPagePoolWordCount           = 0;
        NativeHandle                nvgBucketRangesBuffer          = 0;
        UINT32                      nvgBucketRangesCount           = 0;
        NativeHandle                nvgVisibleClustersBuffer       = 0;
        UINT32                      nvgVisibleClustersCount        = 0;
        NativeHandle                nvgShadowPageViewsBuffer       = 0;
        UINT32                      nvgShadowPageViewsCount        = 0;
        UINT32                      nvgDrawBucketId                = 0xFFFFFFFFu;
        UINT32                      nvgVisibleClusterOffset        = 0xFFFFFFFFu;
        UINT32                      nvgVisibleClusterCount         = 0;
        UINT32                      nvgDebugMode                   = 0;
        FLOAT32                     nvgMaxScreenErrorPx            = 1.0f;
        UINT32                      nvgPhysicalPoolCapacity        = 0;
        UINT64                      nvgResidencyRevision           = 0;

        std::vector<ComputeDispatchCommand>    prePassComputeDispatches;
        std::vector<LayeredDepthPassView>      layeredDepthPasses;
        std::vector<ComputeDispatchCommand>    postPassComputeDispatches;
        std::vector<FramePassItem>             interleavedPassItems;

        GeometryPassView            geometryPass;
        GeometryPassView&           basePass = geometryPass;

        // SceneDepth Validity Contract:
        //   - PRE_PASS:    false (SceneDepth is cleared to 1.0f at frame start; geometry has not been rendered yet)
        //   - POST_DEPTH:  true ONLY when Depth Prepass is active (hasDepthPrepass == true) and sceneDepthTexture != 0
        //   - POST_OPAQUE: true whenever sceneDepthTexture != 0 (populated by Depth Prepass when ON, or by Opaque Geometry Pass when OFF)
        //   - POST_PASS:   true whenever sceneDepthTexture != 0
        BOOL IsSceneDepthValidAtPhase(ComputePassPhase phase) const
        {
            if (sceneDepthTexture == 0)
                return false;
            switch (phase)
            {
                case ComputePassPhase::PRE_PASS:
                    return false;
                case ComputePassPhase::POST_DEPTH:
                    return hasDepthPrepass;
                case ComputePassPhase::POST_OPAQUE:
                case ComputePassPhase::POST_PASS:
                    return true;
                default:
                    return false;
            }
        }

        FrameView() : basePass(geometryPass) {}
        FrameView(const FrameView& o)
            : frameIndex(o.frameIndex), deltaTime(o.deltaTime), elapsedTime(o.elapsedTime), hasAuthoritativeTime(o.hasAuthoritativeTime), renderTarget(o.renderTarget),
              camera(o.camera), environment(o.environment), hasDirectionalLight(o.hasDirectionalLight), directionalLight(o.directionalLight),
              directionalLights(o.directionalLights), lights(o.lights),
              lightsDataRevision(o.lightsDataRevision),
              sceneDepthTexture(o.sceneDepthTexture),
              hasDepthPrepass(o.hasDepthPrepass),
              gpuSceneBuffer(o.gpuSceneBuffer),
              gpuSceneCapacity(o.gpuSceneCapacity),
              gpuSceneHighWaterMark(o.gpuSceneHighWaterMark),
              gpuSceneActiveObjectCount(o.gpuSceneActiveObjectCount),
              gpuSceneMutationEpoch(o.gpuSceneMutationEpoch),
              gpuSkinningPaletteBuffer(o.gpuSkinningPaletteBuffer),
              gpuSkinningPaletteCount(o.gpuSkinningPaletteCount),
              nvsShadowMapTexture(o.nvsShadowMapTexture),
              nvsComparisonSampler(o.nvsComparisonSampler),
              nvsDepthSampler(o.nvsDepthSampler),
              nvsSparsePageHierarchyBuffer(o.nvsSparsePageHierarchyBuffer),
              nvsSparsePageHierarchyCount(o.nvsSparsePageHierarchyCount),
              nvsSparseRootCapacity(o.nvsSparseRootCapacity),
              nvsSparseDirStride(o.nvsSparseDirStride),
              nvsSparseDirCapacity(o.nvsSparseDirCapacity),
              nvsScreenPixelWorldScale(o.nvsScreenPixelWorldScale),
              nvsVirtualPageTableBuffer(o.nvsVirtualPageTableBuffer),
              nvsVirtualPageTableCount(o.nvsVirtualPageTableCount),
              nvsTotalVirtualPages(o.nvsTotalVirtualPages),
              nvsPhysicalPageTableBuffer(o.nvsPhysicalPageTableBuffer),
              nvsPhysicalPageTableCount(o.nvsPhysicalPageTableCount),
              nvsPhysicalPoolCapacity(o.nvsPhysicalPoolCapacity),
              nvsLightViewTableBuffer(o.nvsLightViewTableBuffer),
              nvsLightViewTableCount(o.nvsLightViewTableCount),
              nvsViewTableCount(o.nvsViewTableCount),
              nvsPagesPerAxis(o.nvsPagesPerAxis),
              nvsPageTexelSize(o.nvsPageTexelSize),
              nvsDepthMode(o.nvsDepthMode),
              nvsSamplingAvailable(o.nvsSamplingAvailable),
              nvsSnapshotFrameIndex(o.nvsSnapshotFrameIndex),
              nvsLightViewTableCpuMirrorPtr(nullptr),
              nvsLightViewTableCpuMirrorCount(o.nvsLightViewTableCpuMirrorCount),
              nvsLightViewTableCpuMirror(o.nvsLightViewTableCpuMirror),
              nvgVisibilityTexture(o.nvgVisibilityTexture),
              nvgPagePoolBuffer(o.nvgPagePoolBuffer),
              nvgPagePoolWordCount(o.nvgPagePoolWordCount),
              nvgBucketRangesBuffer(o.nvgBucketRangesBuffer),
              nvgBucketRangesCount(o.nvgBucketRangesCount),
              nvgVisibleClustersBuffer(o.nvgVisibleClustersBuffer),
              nvgVisibleClustersCount(o.nvgVisibleClustersCount),
              nvgShadowPageViewsBuffer(o.nvgShadowPageViewsBuffer),
              nvgShadowPageViewsCount(o.nvgShadowPageViewsCount),
              nvgDrawBucketId(o.nvgDrawBucketId),
              nvgVisibleClusterOffset(o.nvgVisibleClusterOffset),
              nvgVisibleClusterCount(o.nvgVisibleClusterCount),
              nvgDebugMode(o.nvgDebugMode),
              nvgMaxScreenErrorPx(o.nvgMaxScreenErrorPx),
              nvgPhysicalPoolCapacity(o.nvgPhysicalPoolCapacity),
              nvgResidencyRevision(o.nvgResidencyRevision),
              prePassComputeDispatches(o.prePassComputeDispatches),
              layeredDepthPasses(o.layeredDepthPasses),
              postPassComputeDispatches(o.postPassComputeDispatches),
              interleavedPassItems(o.interleavedPassItems),
              geometryPass(o.geometryPass), basePass(geometryPass)
        {
            if (!o.nvsLightViewTableCpuMirror.empty() &&
                o.nvsLightViewTableCpuMirrorPtr == o.nvsLightViewTableCpuMirror.data())
            {
                nvsLightViewTableCpuMirrorPtr = nvsLightViewTableCpuMirror.data();
            }
            else
            {
                nvsLightViewTableCpuMirrorPtr = o.nvsLightViewTableCpuMirrorPtr;
            }
        }
        FrameView& operator=(const FrameView& o)
        {
            if (this != &o)
            {
                frameIndex = o.frameIndex;
                deltaTime = o.deltaTime;
                elapsedTime = o.elapsedTime;
                hasAuthoritativeTime = o.hasAuthoritativeTime;
                renderTarget = o.renderTarget;
                camera = o.camera;
                environment = o.environment;
                hasDirectionalLight = o.hasDirectionalLight;
                directionalLight = o.directionalLight;
                directionalLights = o.directionalLights;
                lights = o.lights;
                lightsDataRevision = o.lightsDataRevision;
                sceneDepthTexture = o.sceneDepthTexture;
                hasDepthPrepass = o.hasDepthPrepass;
                gpuSceneBuffer = o.gpuSceneBuffer;
                gpuSceneCapacity = o.gpuSceneCapacity;
                gpuSceneHighWaterMark = o.gpuSceneHighWaterMark;
                gpuSceneActiveObjectCount = o.gpuSceneActiveObjectCount;
                gpuSceneMutationEpoch = o.gpuSceneMutationEpoch;
                gpuSkinningPaletteBuffer = o.gpuSkinningPaletteBuffer;
                gpuSkinningPaletteCount = o.gpuSkinningPaletteCount;
                nvsShadowMapTexture = o.nvsShadowMapTexture;
                nvsComparisonSampler = o.nvsComparisonSampler;
                nvsDepthSampler = o.nvsDepthSampler;
                nvsSparsePageHierarchyBuffer = o.nvsSparsePageHierarchyBuffer;
                nvsSparsePageHierarchyCount = o.nvsSparsePageHierarchyCount;
                nvsSparseRootCapacity = o.nvsSparseRootCapacity;
                nvsSparseDirStride = o.nvsSparseDirStride;
                nvsSparseDirCapacity = o.nvsSparseDirCapacity;
                nvsScreenPixelWorldScale = o.nvsScreenPixelWorldScale;
                nvsVirtualPageTableBuffer = o.nvsVirtualPageTableBuffer;
                nvsVirtualPageTableCount = o.nvsVirtualPageTableCount;
                nvsTotalVirtualPages = o.nvsTotalVirtualPages;
                nvsPhysicalPageTableBuffer = o.nvsPhysicalPageTableBuffer;
                nvsPhysicalPageTableCount = o.nvsPhysicalPageTableCount;
                nvsPhysicalPoolCapacity = o.nvsPhysicalPoolCapacity;
                nvsLightViewTableBuffer = o.nvsLightViewTableBuffer;
                nvsLightViewTableCount = o.nvsLightViewTableCount;
                nvsViewTableCount = o.nvsViewTableCount;
                nvsPagesPerAxis = o.nvsPagesPerAxis;
                nvsPageTexelSize = o.nvsPageTexelSize;
                nvsDepthMode = o.nvsDepthMode;
                nvsSamplingAvailable = o.nvsSamplingAvailable;
                nvsSnapshotFrameIndex = o.nvsSnapshotFrameIndex;
                nvsLightViewTableCpuMirrorCount = o.nvsLightViewTableCpuMirrorCount;
                nvsLightViewTableCpuMirror = o.nvsLightViewTableCpuMirror;
                if (!o.nvsLightViewTableCpuMirror.empty() &&
                    o.nvsLightViewTableCpuMirrorPtr == o.nvsLightViewTableCpuMirror.data())
                {
                    nvsLightViewTableCpuMirrorPtr = nvsLightViewTableCpuMirror.data();
                }
                else
                {
                    nvsLightViewTableCpuMirrorPtr = o.nvsLightViewTableCpuMirrorPtr;
                }
                nvgVisibilityTexture     = o.nvgVisibilityTexture;
                nvgPagePoolBuffer        = o.nvgPagePoolBuffer;
                nvgPagePoolWordCount     = o.nvgPagePoolWordCount;
                nvgBucketRangesBuffer    = o.nvgBucketRangesBuffer;
                nvgBucketRangesCount     = o.nvgBucketRangesCount;
                nvgVisibleClustersBuffer = o.nvgVisibleClustersBuffer;
                nvgVisibleClustersCount  = o.nvgVisibleClustersCount;
                nvgShadowPageViewsBuffer = o.nvgShadowPageViewsBuffer;
                nvgShadowPageViewsCount  = o.nvgShadowPageViewsCount;
                nvgDrawBucketId          = o.nvgDrawBucketId;
                nvgVisibleClusterOffset  = o.nvgVisibleClusterOffset;
                nvgVisibleClusterCount   = o.nvgVisibleClusterCount;
                nvgDebugMode             = o.nvgDebugMode;
                nvgMaxScreenErrorPx      = o.nvgMaxScreenErrorPx;
                nvgPhysicalPoolCapacity  = o.nvgPhysicalPoolCapacity;
                nvgResidencyRevision     = o.nvgResidencyRevision;
                prePassComputeDispatches = o.prePassComputeDispatches;
                layeredDepthPasses = o.layeredDepthPasses;
                postPassComputeDispatches = o.postPassComputeDispatches;
                interleavedPassItems = o.interleavedPassItems;
                geometryPass = o.geometryPass;
            }
            return *this;
        }

        void Clear()
        {
            deltaTime = 0.0f;
            elapsedTime = 0.0f;
            hasAuthoritativeTime = false;
            hasDirectionalLight = false;
            directionalLight = FrameLightData{};
            directionalLights.clear();
            lightsDataRevision = 0;
            sceneDepthTexture = 0;
            hasDepthPrepass = false;
            gpuSceneBuffer = 0;
            gpuSceneCapacity = 0;
            gpuSceneHighWaterMark = 0;
            gpuSceneActiveObjectCount = 0;
            gpuSceneMutationEpoch = 0;
            gpuSkinningPaletteBuffer = 0;
            gpuSkinningPaletteCount = 0;
            nvsShadowMapTexture = 0;
            nvsComparisonSampler = 0;
            nvsDepthSampler = 0;
            nvsSparsePageHierarchyBuffer = 0;
            nvsSparsePageHierarchyCount = 0;
            nvsSparseRootCapacity = 0;
            nvsSparseDirStride = 352;
            nvsSparseDirCapacity = 0;
            nvsScreenPixelWorldScale = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
            nvsVirtualPageTableBuffer = 0;
            nvsVirtualPageTableCount = 0;
            nvsTotalVirtualPages = 0;
            nvsPhysicalPageTableBuffer = 0;
            nvsPhysicalPageTableCount = 0;
            nvsPhysicalPoolCapacity = 0;
            nvsLightViewTableBuffer = 0;
            nvsLightViewTableCount = 0;
            nvsViewTableCount = 0;
            nvsPagesPerAxis = 64;
            nvsPageTexelSize = 128;
            nvsDepthMode = 0;
            nvsSamplingAvailable = 0;
            nvsSnapshotFrameIndex = 0;
            nvsLightViewTableCpuMirrorPtr = nullptr;
            nvsLightViewTableCpuMirrorCount = 0;
            nvsLightViewTableCpuMirror.clear();
            nvgVisibilityTexture = 0;
            nvgPagePoolBuffer = 0;
            nvgPagePoolWordCount = 0;
            nvgBucketRangesBuffer = 0;
            nvgBucketRangesCount = 0;
            nvgVisibleClustersBuffer = 0;
            nvgVisibleClustersCount = 0;
            nvgShadowPageViewsBuffer = 0;
            nvgShadowPageViewsCount = 0;
            nvgDrawBucketId = 0xFFFFFFFFu;
            nvgVisibleClusterOffset = 0xFFFFFFFFu;
            nvgVisibleClusterCount = 0;
            nvgDebugMode = 0;
            nvgMaxScreenErrorPx = 1.0f;
            nvgPhysicalPoolCapacity = 0;
            nvgResidencyRevision = 0;
            prePassComputeDispatches.clear();
            layeredDepthPasses.clear();
            postPassComputeDispatches.clear();
            interleavedPassItems.clear();
            lights.clear();
            geometryPass.Clear();
        }
    };
}
