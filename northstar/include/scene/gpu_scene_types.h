#pragma once

#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include <cstddef>

namespace Northstar
{
    inline constexpr UINT32 kInvalidGPUSceneSlot        = 0xFFFFFFFFu;
    inline constexpr UINT32 kDefaultGPUSceneMinCapacity = 64u;
    inline constexpr UINT32 kMaxGPUSceneSlotCapacity    = 1048576u;

    // =====================================================================
    //  Stable GPU Scene Object Handle (Slot Index + Generation)
    // =====================================================================
    struct GPUSceneHandle
    {
        UINT32 slotIndex  = kInvalidGPUSceneSlot;
        UINT32 generation = 0u;

        constexpr BOOL IsValid() const noexcept
        {
            return slotIndex != kInvalidGPUSceneSlot && generation != 0u;
        }

        constexpr UINT64 Packed() const noexcept
        {
            return (static_cast<UINT64>(generation) << 32u) | static_cast<UINT64>(slotIndex);
        }

        static constexpr GPUSceneHandle FromPacked(UINT64 packed) noexcept
        {
            GPUSceneHandle h{};
            h.slotIndex  = static_cast<UINT32>(packed & 0xFFFFFFFFull);
            h.generation = static_cast<UINT32>((packed >> 32u) & 0xFFFFFFFFull);
            return h;
        }

        constexpr bool operator==(const GPUSceneHandle& other) const noexcept
        {
            return slotIndex == other.slotIndex && generation == other.generation;
        }

        constexpr bool operator!=(const GPUSceneHandle& other) const noexcept
        {
            return !(*this == other);
        }
    };

    // =====================================================================
    //  GPU Scene Object Flags (Normalized Renderer + State Bitmask)
    // =====================================================================
    enum GPUSceneObjectFlags : UINT32
    {
        GPU_SCENE_OBJ_NONE                 = 0u,
        GPU_SCENE_OBJ_VALID                = 1u << 0,
        GPU_SCENE_OBJ_VISIBLE              = 1u << 1,
        GPU_SCENE_OBJ_STATIC               = 1u << 2,
        GPU_SCENE_OBJ_CAST_SHADOW          = 1u << 3,
        GPU_SCENE_OBJ_RECEIVE_SHADOW       = 1u << 4,
        GPU_SCENE_OBJ_FRUSTUM_CULL         = 1u << 5,
        GPU_SCENE_OBJ_DISTANCE_CULL        = 1u << 6,
        GPU_SCENE_OBJ_SMALL_OBJECT_CULL    = 1u << 7,
        GPU_SCENE_OBJ_OCCLUSION_CULL       = 1u << 8,
        GPU_SCENE_OBJ_USE_LOD              = 1u << 9,
        GPU_SCENE_OBJ_SKINNED              = 1u << 10,
        GPU_SCENE_OBJ_BAKED_STATIC         = 1u << 11,
        GPU_SCENE_OBJ_STATIC_BATCH_MEMBER  = 1u << 12,
        GPU_SCENE_OBJ_STATIC_BATCHED       = 1u << 12,
        GPU_SCENE_OBJ_INSTANCED_CANDIDATE  = 1u << 13,
        GPU_SCENE_OBJ_INSTANCED            = 1u << 13,
        GPU_SCENE_OBJ_ALPHA_TESTED         = 1u << 14,
        GPU_SCENE_OBJ_ALPHA_TEST           = 1u << 14,
        GPU_SCENE_OBJ_TRANSPARENT          = 1u << 15,
        GPU_SCENE_OBJ_TWO_SIDED            = 1u << 16,
        GPU_SCENE_OBJ_DEPTH_PREPASS        = 1u << 17,
        GPU_SCENE_OBJ_SHADOW_PASS          = 1u << 18,
        GPU_SCENE_OBJ_HAS_VALID_BOUNDS     = 1u << 19,
        GPU_SCENE_OBJ_VIRTUAL_GEOMETRY     = 1u << 20
    };

    // =====================================================================
    //  CPU Incremental Dirty Tracking Flags
    // =====================================================================
    enum GPUSceneDirtyFlags : UINT32
    {
        GPU_SCENE_DIRTY_NONE      = 0u,
        GPU_SCENE_DIRTY_TRANSFORM = 1u << 0,
        GPU_SCENE_DIRTY_BOUNDS    = 1u << 1,
        GPU_SCENE_DIRTY_MESH      = 1u << 2,
        GPU_SCENE_DIRTY_MATERIAL  = 1u << 3,
        GPU_SCENE_DIRTY_LOD_GROUP = 1u << 4,
        GPU_SCENE_DIRTY_FLAGS     = 1u << 5,
        GPU_SCENE_DIRTY_BATCHING  = 1u << 6,
        GPU_SCENE_DIRTY_LIFECYCLE = 1u << 7,
        GPU_SCENE_DIRTY_SKINNING  = 1u << 8,
        GPU_SCENE_DIRTY_ALL       = 0x1FFu
    };

    // =====================================================================
    //  GPU Scene Skinned Metadata Layout (Zero ABI Growth — 160B Preserved)
    //  - GPUSceneObjectData::batchGroupId (offset 28, when GPU_SCENE_OBJ_SKINNED):
    //      Full 32-bit bonePaletteOffset (0 .. 4,294,967,295 bones in scene)
    //  - GPUSceneObjectData::renderObjectFlags (offset 12):
    //      Bits [0..15]  (16 bits)                : RenderObjectFlags
    //      Bits [16..31] (16 bits, 0 .. 65,535)   : per-object skinningBoneCount
    // =====================================================================
    static constexpr UINT32 kGPUSceneRenderObjFlagsMask        = 0x0000FFFFu;
    static constexpr UINT32 kGPUSceneRenderObjBoneCountShift   = 16u;
    static constexpr UINT32 kGPUSceneRenderObjBoneCountMask    = 0x0000FFFFu;

    inline constexpr UINT32 PackGPUSceneRenderFlagsAndBoneCount(UINT32 renderObjFlags, UINT32 boneCount) noexcept
    {
        const UINT32 clampedCount = (boneCount > kGPUSceneRenderObjBoneCountMask)
            ? kGPUSceneRenderObjBoneCountMask : boneCount;
        return (renderObjFlags & kGPUSceneRenderObjFlagsMask)
             | ((clampedCount & kGPUSceneRenderObjBoneCountMask) << kGPUSceneRenderObjBoneCountShift);
    }

    inline constexpr UINT32 UnpackGPUSceneRenderObjectFlags(UINT32 packedRenderFlags) noexcept
    {
        return packedRenderFlags & kGPUSceneRenderObjFlagsMask;
    }

    inline constexpr UINT32 UnpackGPUSceneSkinnedBoneCountFromRenderFlags(UINT32 packedRenderFlags) noexcept
    {
        return (packedRenderFlags >> kGPUSceneRenderObjBoneCountShift) & kGPUSceneRenderObjBoneCountMask;
    }

    // Legacy 20/12-bit helper retained for optional InstanceData::customData convenience packing
    static constexpr UINT32 kGPUSceneSkinnedBoneOffsetMask  = 0x000FFFFFu;
    static constexpr UINT32 kGPUSceneSkinnedBoneCountShift  = 20u;
    static constexpr UINT32 kGPUSceneSkinnedBoneCountMask   = 0x00000FFFu;

    inline constexpr UINT32 PackGPUSceneSkinnedBoneRange(UINT32 bonePaletteOffset, UINT32 boneCount) noexcept
    {
        const UINT32 clampedOffset = (bonePaletteOffset > kGPUSceneSkinnedBoneOffsetMask)
            ? kGPUSceneSkinnedBoneOffsetMask : bonePaletteOffset;
        const UINT32 clampedCount  = (boneCount > kGPUSceneSkinnedBoneCountMask)
            ? kGPUSceneSkinnedBoneCountMask : boneCount;
        return (clampedOffset & kGPUSceneSkinnedBoneOffsetMask)
             | ((clampedCount & kGPUSceneSkinnedBoneCountMask) << kGPUSceneSkinnedBoneCountShift);
    }

    inline constexpr UINT32 UnpackGPUSceneSkinnedBoneOffset(UINT32 packedValue) noexcept
    {
        return packedValue & kGPUSceneSkinnedBoneOffsetMask;
    }

    inline constexpr UINT32 UnpackGPUSceneSkinnedBoneCount(UINT32 packedValue) noexcept
    {
        return (packedValue >> kGPUSceneSkinnedBoneCountShift) & kGPUSceneSkinnedBoneCountMask;
    }

    // =====================================================================
    //  Packed LOD + Material State Helpers (lodAndMaterialFlags)
    //    Bits [0..7]   (8 bits, 0..255, engine max MAX_LOD_LEVELS=8)        : active LOD level count
    //    Bits [8..9]   (2 bits, 0..2: 0=NONE, 1=FRONT, 2=BACK)              : CullMode
    //    Bits [10..12] (3 bits, 0..4: OPAQUE..PREMULTIPLIED_ALPHA)          : BlendMode
    //    Bit  [13]     (1 bit)                                              : DepthTest enabled
    //    Bit  [14]     (1 bit)                                              : DepthWrite enabled
    //    Bit  [15]     (1 bit)                                              : Wireframe fill mode
    //    Bit  [16]     (1 bit)                                              : Has Geometry Pass ShaderSet
    //    Bit  [17]     (1 bit)                                              : Has Shadow Pass ShaderSet
    //    Bit  [18]     (1 bit)                                              : Has Explicit Depth Pass ShaderSet
    //    Bits [19..31] (13 bits)                                            : Reserved (must be 0)
    //
    //  Note: lodGroupId is stored as a dedicated 32-bit UINT32 field at
    //  byte offset 24 (Row 1.z) of GPUSceneObjectData (range 0..4,294,967,295).
    // =====================================================================
    static constexpr UINT32 kGPUSceneLODCountShift       = 0u;
    static constexpr UINT32 kGPUSceneLODCountBits        = 8u;
    static constexpr UINT32 kGPUSceneLODCountMask        = 0xFFu;
    static constexpr UINT32 kGPUSceneMaxPackedLODCount   = 255u;

    static constexpr UINT32 kGPUSceneCullModeShift       = 8u;
    static constexpr UINT32 kGPUSceneCullModeBits        = 2u;
    static constexpr UINT32 kGPUSceneCullModeMask        = 0x3u;
    static constexpr UINT32 kGPUSceneMaxCullModeValue    = 2u;

    static constexpr UINT32 kGPUSceneBlendModeShift      = 10u;
    static constexpr UINT32 kGPUSceneBlendModeBits       = 3u;
    static constexpr UINT32 kGPUSceneBlendModeMask       = 0x7u;
    static constexpr UINT32 kGPUSceneMaxBlendModeValue   = 4u;

    static constexpr UINT32 kGPUSceneDepthTestBit        = 1u << 13u;
    static constexpr UINT32 kGPUSceneDepthWriteBit       = 1u << 14u;
    static constexpr UINT32 kGPUSceneWireframeBit        = 1u << 15u;
    static constexpr UINT32 kGPUSceneHasGeometryPassBit  = 1u << 16u;
    static constexpr UINT32 kGPUSceneHasShadowPassBit    = 1u << 17u;
    static constexpr UINT32 kGPUSceneHasDepthPassBit     = 1u << 18u;
    static constexpr UINT32 kGPUSceneLODMatDefinedMask   = 0x0007FFFFu;
    static constexpr UINT32 kGPUSceneLODMatReservedMask  = 0xFFF80000u;

    inline constexpr UINT32 PackGPUSceneLODAndMaterialFlags(
        UINT32 lodCount,
        UINT32 cullMode,
        UINT32 blendMode,
        BOOL   depthTest,
        BOOL   depthWrite,
        BOOL   wireframe,
        BOOL   hasGeometryPass,
        BOOL   hasShadowPass,
        BOOL   hasDepthPass) noexcept
    {
        const UINT32 clampedLod   = (lodCount > kGPUSceneMaxPackedLODCount) ? kGPUSceneMaxPackedLODCount : lodCount;
        const UINT32 clampedCull  = (cullMode > kGPUSceneMaxCullModeValue)  ? kGPUSceneMaxCullModeValue  : cullMode;
        const UINT32 clampedBlend = (blendMode > kGPUSceneMaxBlendModeValue) ? kGPUSceneMaxBlendModeValue : blendMode;
        return ((clampedLod & kGPUSceneLODCountMask) << kGPUSceneLODCountShift) |
               ((clampedCull & kGPUSceneCullModeMask) << kGPUSceneCullModeShift) |
               ((clampedBlend & kGPUSceneBlendModeMask) << kGPUSceneBlendModeShift) |
               (depthTest       ? kGPUSceneDepthTestBit       : 0u) |
               (depthWrite      ? kGPUSceneDepthWriteBit      : 0u) |
               (wireframe       ? kGPUSceneWireframeBit       : 0u) |
               (hasGeometryPass ? kGPUSceneHasGeometryPassBit : 0u) |
               (hasShadowPass   ? kGPUSceneHasShadowPassBit   : 0u) |
               (hasDepthPass    ? kGPUSceneHasDepthPassBit    : 0u);
    }

    inline constexpr UINT32 UnpackGPUSceneLODCount(UINT32 packedFlags) noexcept
    {
        return (packedFlags >> kGPUSceneLODCountShift) & kGPUSceneLODCountMask;
    }

    inline constexpr UINT32 UnpackGPUSceneCullMode(UINT32 packedFlags) noexcept
    {
        return (packedFlags >> kGPUSceneCullModeShift) & kGPUSceneCullModeMask;
    }

    inline constexpr UINT32 UnpackGPUSceneBlendMode(UINT32 packedFlags) noexcept
    {
        return (packedFlags >> kGPUSceneBlendModeShift) & kGPUSceneBlendModeMask;
    }

    inline constexpr BOOL UnpackGPUSceneDepthTest(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneDepthTestBit) != 0u;
    }

    inline constexpr BOOL UnpackGPUSceneDepthWrite(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneDepthWriteBit) != 0u;
    }

    inline constexpr BOOL UnpackGPUSceneWireframe(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneWireframeBit) != 0u;
    }

    inline constexpr BOOL UnpackGPUSceneHasGeometryPass(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneHasGeometryPassBit) != 0u;
    }

    inline constexpr BOOL UnpackGPUSceneHasShadowPass(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneHasShadowPassBit) != 0u;
    }

    inline constexpr BOOL UnpackGPUSceneHasDepthPass(UINT32 packedFlags) noexcept
    {
        return (packedFlags & kGPUSceneHasDepthPassBit) != 0u;
    }

    inline constexpr BOOL IsGPUSceneLODAndMaterialFlagsABIValid(UINT32 packedFlags) noexcept
    {
        return ((packedFlags & kGPUSceneLODMatReservedMask) == 0u) &&
               (UnpackGPUSceneCullMode(packedFlags) <= kGPUSceneMaxCullModeValue) &&
               (UnpackGPUSceneBlendMode(packedFlags) <= kGPUSceneMaxBlendModeValue);
    }

    // =====================================================================
    //  GPU Scene Object Metadata Record (160 Bytes, 16-Byte Aligned)
    //  Matches NSL StructuredBuffer<GPUSceneObjectData> 1:1 with zero
    //  CPU pointers or transient backend handles.
    // =====================================================================
    struct alignas(16) GPUSceneObjectData
    {
        // Row 0 (Offsets 0..15): Identity & Renderer Flags
        UINT32 slotIndex           = kInvalidGPUSceneSlot;
        UINT32 generation          = 0u;
        UINT32 objectFlags         = 0u;
        UINT32 renderObjectFlags   = 0u;

        // Row 1 (Offsets 16..31): Stable Resource & Batch Association IDs
        UINT32 meshId              = 0u;
        UINT32 materialId          = 0u;
        UINT32 lodGroupId          = 0u;
        UINT32 batchGroupId        = 0u;

        // Row 2 (Offsets 32..47): Geometry Draw Stream Topology
        UINT32 vertexCount         = 0u;
        UINT32 indexCount          = 0u;
        UINT32 startIndex          = 0u;
        UINT32 baseVertex          = 0u;

        // Row 3 (Offsets 48..63): World AABB Min + Submesh Index
        Vec3   aabbMin             = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 submeshIndex        = 0u;

        // Row 4 (Offsets 64..79): World AABB Max + Packed LOD/Material State
        Vec3   aabbMax             = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 lodAndMaterialFlags = 0u;

        // Row 5 (Offsets 80..95): World-Space Bounding Sphere (xyz = center, w = radius)
        Vec4   boundingSphere      = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

        // Rows 6..9 (Offsets 96..159): Column-Major World Matrix (4x4 FLOAT32)
        Mat4   worldMatrix         = Mat4::Identity();
    };

    static_assert(sizeof(GPUSceneObjectData) == 160, "GPUSceneObjectData must be exactly 160 bytes (10 x 16B).");
    static_assert(alignof(GPUSceneObjectData) == 16, "GPUSceneObjectData must be 16-byte aligned.");
    static_assert(offsetof(GPUSceneObjectData, slotIndex)           == 0,   "slotIndex offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, generation)          == 4,   "generation offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, objectFlags)         == 8,   "objectFlags offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, renderObjectFlags)   == 12,  "renderObjectFlags offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, meshId)              == 16,  "meshId offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, materialId)          == 20,  "materialId offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, lodGroupId)          == 24,  "lodGroupId offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, batchGroupId)        == 28,  "batchGroupId offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, vertexCount)         == 32,  "vertexCount offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, indexCount)          == 36,  "indexCount offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, startIndex)          == 40,  "startIndex offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, baseVertex)          == 44,  "baseVertex offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, aabbMin)             == 48,  "aabbMin offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, submeshIndex)        == 60,  "submeshIndex offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, aabbMax)             == 64,  "aabbMax offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, lodAndMaterialFlags) == 76,  "lodAndMaterialFlags offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, boundingSphere)      == 80,  "boundingSphere offset mismatch");
    static_assert(offsetof(GPUSceneObjectData, worldMatrix)         == 96,  "worldMatrix offset mismatch");

    // =====================================================================
    //  Coalesced Incremental Upload Range & Frame Update Telemetry
    // =====================================================================
    struct GPUSceneUploadRange
    {
        UINT32 firstSlot  = 0u;
        UINT32 slotCount  = 0u;
        UINT32 byteOffset = 0u;
        UINT32 byteSize   = 0u;
    };

    struct GPUSceneUpdateStats
    {
        BOOL    updateSucceeded      = true;
        UINT32  activeObjectCount         = 0u;
        UINT32  virtualActiveObjectCount  = 0u;
        UINT32  materialKernelObjectCount = 0u;
        UINT64  mutationEpoch             = 0ull;
        UINT32  highWaterMark        = 0u;
        UINT32  capacitySlots        = 0u;
        UINT32  freeSlotCount        = 0u;
        UINT32  invalidSlotCount     = 0u;
        UINT32  dirtyObjectCount     = 0u;
        UINT32  dirtySlotCount       = 0u;
        UINT32  coalescedUploadCount = 0u;
        UINT32  uploadedBytes        = 0u;
        UINT32  bufferResizeCount    = 0u;
        FLOAT64 cpuDirtyCollectMs    = 0.0;
        FLOAT64 cpuRecordBuildMs     = 0.0;
        FLOAT64 cpuUploadSubmitMs    = 0.0;
        FLOAT64 cpuTotalUpdateMs     = 0.0;
    };

    // =====================================================================
    //  GPU Diagnostic Compute Shader Query & Output Records
    // =====================================================================
    enum GPUSceneDiagnosticStatusBits : UINT32
    {
        GPU_SCENE_DIAG_NONE                  = 0u,
        GPU_SCENE_DIAG_SLOT_IN_BOUNDS        = 1u << 0,
        GPU_SCENE_DIAG_SLOT_VALID            = 1u << 1,
        GPU_SCENE_DIAG_GENERATION_MATCH      = 1u << 2,
        GPU_SCENE_DIAG_FLAGS_MATCH           = 1u << 3,
        GPU_SCENE_DIAG_BOUNDS_VALID          = 1u << 4,
        GPU_SCENE_DIAG_MESH_VALID            = 1u << 5,
        GPU_SCENE_DIAG_MATERIAL_VALID        = 1u << 6,
        GPU_SCENE_DIAG_LOD_MAT_ABI_VALID     = 1u << 7,
        GPU_SCENE_DIAG_REQ_VERIFY_LOD_MAT_ABI = 1u << 31
    };

    struct alignas(16) GPUSceneDiagnosticQuery
    {
        UINT32 slotIndex          = 0u;
        UINT32 expectedGeneration = 0u;
        UINT32 requiredFlagsMask  = 0u;
        UINT32 queryUserData      = 0u;
    };

    static_assert(sizeof(GPUSceneDiagnosticQuery) == 16, "GPUSceneDiagnosticQuery must be 16 bytes.");

    struct alignas(16) GPUSceneDiagnosticOutput
    {
        // Row 0 (0..15)
        UINT32 statusFlags         = 0u;
        UINT32 actualSlotIndex     = kInvalidGPUSceneSlot;
        UINT32 actualGeneration    = 0u;
        UINT32 objectFlags         = 0u;

        // Row 1 (16..31)
        UINT32 renderObjectFlags   = 0u;
        UINT32 meshId              = 0u;
        UINT32 materialId          = 0u;
        UINT32 lodGroupId          = 0u;

        // Row 2 (32..47)
        UINT32 batchGroupId        = 0u;
        UINT32 vertexCount         = 0u;
        UINT32 indexCount          = 0u;
        UINT32 submeshIndex        = 0u;

        // Row 3 (48..63)
        Vec3   aabbMin             = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 lodAndMaterialFlags = 0u;

        // Row 4 (64..79)
        Vec3   aabbMax             = Vec3(0.0f, 0.0f, 0.0f);
        FLOAT32 sphereRadius       = 0.0f;

        // Row 5 (80..95)
        Vec3   worldTranslation    = Vec3(0.0f, 0.0f, 0.0f);
        FLOAT32 queryEcho          = 0.0f;
    };

    static_assert(sizeof(GPUSceneDiagnosticOutput) == 96, "GPUSceneDiagnosticOutput must be 96 bytes (6 x 16B).");
    static_assert(offsetof(GPUSceneDiagnosticOutput, statusFlags)         == 0,  "statusFlags offset mismatch");
    static_assert(offsetof(GPUSceneDiagnosticOutput, renderObjectFlags)   == 16, "renderObjectFlags offset mismatch");
    static_assert(offsetof(GPUSceneDiagnosticOutput, batchGroupId)        == 32, "batchGroupId offset mismatch");
    static_assert(offsetof(GPUSceneDiagnosticOutput, aabbMin)             == 48, "aabbMin offset mismatch");
    static_assert(offsetof(GPUSceneDiagnosticOutput, aabbMax)             == 64, "aabbMax offset mismatch");
    static_assert(offsetof(GPUSceneDiagnosticOutput, worldTranslation)    == 80, "worldTranslation offset mismatch");
}
