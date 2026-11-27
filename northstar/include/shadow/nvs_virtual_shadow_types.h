#pragma once

#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "../light/light_types.h"
#include "../render_object/render_object_types.h"
#include "../scene/gpu_scene_types.h"
#include "../texture/texture.h"
#include "../buffer/gpu_buffer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace Northstar
{

    // =====================================================================
    //  Northstar Virtual Shadows (NVS) — Core Constants
    // =====================================================================
    static constexpr UINT32 kNVSPageTexelSize          = 128u;
    static constexpr UINT32 kNVSMaxClipmapLevels       = 4u;
    static constexpr UINT32 kNVSDefaultPagesPerAxis    = 32u;
    static constexpr UINT32 kNVSMaxPagesPerAxis        = 128u;
    static constexpr UINT32 kNVSDefaultMaxLights       = 1u;
    static constexpr UINT32 kNVSMaxSupportedLights     = 64u;
    static constexpr UINT32 kNVSMinPhysicalPoolPages   = 64u;
    static constexpr UINT32 kNVSMaxPhysicalPoolPages   = 2048u;
    static constexpr UINT32 kNVSInvalidPhysicalPage    = 0xFFFFFFFFu;
    static constexpr UINT32 kNVSInvalidVirtualPage     = 0xFFFFFFFFu;

    // =====================================================================
    //  Virtual Page Residency & Validity States
    //  Core Invariant: A newly allocated or cleared physical page MUST NEVER
    //  be marked NVS_PAGE_STATE_VALID_SHADOW_DATA before shadow rendering.
    // =====================================================================
    enum NVSPageState : UINT32
    {
        NVS_PAGE_STATE_UNMAPPED              = 0u, // No physical page assigned
        NVS_PAGE_STATE_ALLOCATED_NOT_CLEARED = 1u, // Physical page assigned, selective clear pending
        NVS_PAGE_STATE_CLEARED_NOT_RENDERED  = 2u, // Selective depth clear (1.0f) complete, awaiting caster pass
        NVS_PAGE_STATE_VALID_SHADOW_DATA     = 3u, // Resident with valid shadow depth data
        NVS_PAGE_STATE_INVALIDATED           = 4u, // Explicitly invalidated (light/origin shift/caster move)
        NVS_PAGE_STATE_EVICTED               = 5u  // Physical page reclaimed via age/LRU eviction
    };

    enum NVSPageFlags : UINT32
    {
        NVS_PAGE_STATE_MASK                = 0x0000000Fu,
        NVS_PAGE_FLAG_REQUESTED_THIS_FRAME = 1u << 4u,
        NVS_PAGE_FLAG_DIRTY_CLEAR_QUEUED   = 1u << 5u,
        NVS_PAGE_FLAG_CACHE_HIT_THIS_FRAME = 1u << 6u,
        NVS_PAGE_FLAG_NEWLY_ALLOCATED      = 1u << 7u,
        NVS_PAGE_FLAG_PIN_REQUESTED        = 1u << 8u,
        NVS_PAGE_FLAG_EXTENDED_VIEW_ADDR   = 1u << 9u
    };

    // =====================================================================
    //  Physical Page Slot States & Flags
    // =====================================================================
    enum NVSPhysicalPageState : UINT32
    {
        NVS_PHYS_STATE_FREE                  = 0u,
        NVS_PHYS_STATE_ALLOCATED_NOT_CLEARED = 1u,
        NVS_PHYS_STATE_CLEARED_NOT_RENDERED  = 2u,
        NVS_PHYS_STATE_RESIDENT_VALID        = 3u
    };

    enum NVSPhysicalPageFlags : UINT32
    {
        NVS_PHYS_STATE_MASK                = 0x0000000Fu,
        NVS_PHYS_FLAG_PINNED               = 1u << 4u,
        NVS_PHYS_FLAG_DIRTY_CLEAR          = 1u << 5u,
        NVS_PHYS_FLAG_EVICTED_REASSIGNED   = 1u << 6u,
        NVS_PHYS_FLAG_EXTENDED_VIEW_ADDR   = 1u << 7u
    };

    // =====================================================================
    //  GPU Residency Compute Dispatch Modes & Invalidation Modes
    // =====================================================================
    enum NVSResidencyDispatchMode : UINT32
    {
        NVS_DISPATCH_PROCESS_REQUESTS_AND_ALLOCATE = 0u, // Full pass: invalidate + dedup + allocate/evict + emit clear args
        NVS_DISPATCH_MARK_CLEARED_PAGES            = 1u, // Post-clear pass: ALLOCATED_NOT_CLEARED -> CLEARED_NOT_RENDERED
        NVS_DISPATCH_PROMOTE_CLEARED_TO_VALID      = 2u, // Direct state promotion: CLEARED_NOT_RENDERED -> VALID_SHADOW_DATA
        NVS_DISPATCH_RESET_ALL_TABLES              = 3u, // Reset all virtual & physical entries on GPU
        NVS_DISPATCH_COMMIT_RENDERED_PAGES         = 4u, // Page validity contract: per-page, generation-checked
                                                         // CLEARED_NOT_RENDERED -> VALID_SHADOW_DATA commit
        NVS_DISPATCH_BATCH_INVALIDATE_VIEWS        = 5u  // Batch selective view invalidation & sparse block/directory reclamation
    };

    // =====================================================================
    //  Page Validity Contract (GPU Rasterization & Commit)
    //
    //  The ONLY production path that may promote a page to
    //  NVS_PAGE_STATE_VALID_SHADOW_DATA is NVS_DISPATCH_COMMIT_RENDERED_PAGES.
    //  A commit record is accepted iff ALL of the following hold on the GPU:
    //    1. physicalPageIndex < physicalPoolCapacity
    //    2. Physical state == CLEARED_NOT_RENDERED (2)
    //       (ALLOCATED_NOT_CLEARED, FREE, already-VALID pages are rejected)
    //    3. physicalGeneration   == expectedPhysicalGeneration
    //       ownerVirtualFlatIndex == expectedVirtualFlatIndex
    //       VirtualPageTable[owner].physicalPageIndex == physicalPageIndex
    //       VirtualPageTable[owner] state == CLEARED_NOT_RENDERED (2)
    //    4. resultFlags has NVS_PAGE_RENDER_RESULT_SUCCESS and has neither
    //       NVS_PAGE_RENDER_RESULT_CASTER_LIST_INCOMPLETE nor
    //       NVS_PAGE_RENDER_RESULT_RASTER_FAILED.
    //  Records failing (4) leave the page in CLEARED_NOT_RENDERED, so the page
    //  keeps NVS_PAGE_RENDER_FLAG_NEEDS_CASTER_RENDER and is re-rendered on the
    //  next request. Records are processed serially in submission order, so a
    //  duplicate record for an already committed page is rejected (state 3).
    // =====================================================================
    enum NVSPageRenderResultFlags : UINT32
    {
        NVS_PAGE_RENDER_RESULT_NONE                  = 0u,
        NVS_PAGE_RENDER_RESULT_SUCCESS               = 1u << 0u, // Every caster in the page range was rasterized
        NVS_PAGE_RENDER_RESULT_CASTER_LIST_INCOMPLETE = 1u << 1u, // Range had NVS_CASTER_PAGE_FLAG_OVERFLOW_INCOMPLETE
        NVS_PAGE_RENDER_RESULT_RASTER_FAILED         = 1u << 2u, // Rasterization skipped / failed (PSO missing, etc.)

        // diagnostic commit flags emitted by RuntimeNVSIndirectDraw.nsl
        NVS_PAGE_COMMIT_FLAG_RENDER_COMPLETE          = 1u << 0u,
        NVS_PAGE_COMMIT_FLAG_INCOMPLETE_PAGE          = 1u << 1u,
        NVS_PAGE_COMMIT_FLAG_RASTER_FAILED            = 1u << 2u,
        NVS_PAGE_COMMIT_FLAG_EMPTY_CASTER_PAGE        = 1u << 3u,
        NVS_PAGE_COMMIT_FLAG_STALE_HANDLE_SKIPPED     = 1u << 4u,
        NVS_PAGE_COMMIT_FLAG_MISSING_MESH_DESC        = 1u << 5u,
        NVS_PAGE_COMMIT_FLAG_MISSING_MATERIAL_VARIANT = 1u << 6u,
        NVS_PAGE_COMMIT_FLAG_COMMAND_OVERFLOW         = 1u << 7u
    };

    // 16-byte commit record. Shares the NVSVirtualPageRequest stride so that it can be
    // uploaded into (or GPU-generated inside) the residency PageRequests buffer.
    struct alignas(16) NVSPageRenderCommitRecord
    {
        UINT32 physicalPageIndex          = 0xFFFFFFFFu;
        UINT32 resultFlags                = NVS_PAGE_RENDER_RESULT_NONE;
        UINT32 expectedPhysicalGeneration = 0u;
        UINT32 expectedVirtualFlatIndex   = 0xFFFFFFFFu;
    };

    // Commit outcome counters. NVS_DISPATCH_COMMIT_RENDERED_PAGES re-uses the 64-byte
    // NVSResidencyTelemetry record; use ExtractNVSPageCommitResult() to decode it.
    struct NVSPageCommitResult
    {
        UINT32 submittedRecords       = 0u; // telemetry.requestedCount
        UINT32 committedPages         = 0u; // telemetry.validRequestCount
        UINT32 rejectedInvalidState   = 0u; // telemetry.invalidRequestCount (out-of-range / state != 2)
        UINT32 rejectedIdentity       = 0u; // telemetry.duplicateRequestCount (generation / owner mismatch)
        UINT32 keptForRerender        = 0u; // telemetry.uniqueRequestCount (incomplete / failed / no SUCCESS)
        UINT32 residentValidPages     = 0u;
        UINT32 clearedNotRenderedPages = 0u;
    };

    enum NVSInvalidationMode : UINT32
    {
        NVS_INVALIDATE_NONE          = 0u,
        NVS_INVALIDATE_ALL           = 1u,
        NVS_INVALIDATE_LIGHT         = 2u,
        NVS_INVALIDATE_CLIPMAP_LEVEL = 3u,
        NVS_INVALIDATE_VIRTUAL_RANGE = 4u,
        NVS_INVALIDATE_ORIGIN_SHIFT  = 5u,
        NVS_INVALIDATE_VIEW_SLOT     = 6u,
        NVS_INVALIDATE_RECLAIM_VIEW  = 7u,
        NVS_INVALIDATE_VIEW_ORIGIN   = 8u
    };

    enum NVSPageRequestFlags : UINT32
    {
        NVS_REQ_FLAG_NONE               = 0u,
        NVS_REQ_FLAG_FORCE_INVALID      = 1u << 0u, // Explicit invalid request marker
        NVS_REQ_FLAG_PIN_PHYSICAL       = 1u << 1u, // Pin allocated physical page against eviction
        NVS_REQ_FLAG_EXTENDED_VIEW_ADDR = 1u << 2u, // 64-bit sparse view address: requestUserData = 32-bit viewIndex, virtualAddressPacked = PackNVSSparsePageCoord
        NVS_REQ_FLAG_COARSE_ANCHOR      = 1u << 3u  // Highest-priority coarse fallback mip page request
    };

    // =====================================================================
    //  2-Level Demand-Driven Sparse Virtual Page Hierarchy Constants & Helpers
    //
    //  Decouples unbounded virtual shadow address space (e.g. 10,000 Point Lights
    //  x 6 cube faces x 16,384x16,384 + 8 virtual mips = 1.31 billion virtual pages)
    //  from physical residency and metadata memory:
    //    Level 0 (Root)      : SparsePageHierarchy[0 .. rootCapacity - 1]
    //                          1 UINT32 per shadow view -> dirSlot (or 0xFFFFFFFF if unallocated)
    //    Level 1 (Directory) : SparsePageHierarchy[rootCapacity + dirSlot * 352 + mipTileOffset]
    //                          352 UINT32s per active view covering Mip 0..7 in 8x8 page blocks:
    //                            Mip 0 (128x128 pages = 16x16 blocks) : [0 .. 255]
    //                            Mip 1 (64x64 pages   = 8x8 blocks)   : [256 .. 319]
    //                            Mip 2 (32x32 pages   = 4x4 blocks)   : [320 .. 335]
    //                            Mip 3 (16x16 pages   = 2x2 blocks)   : [336 .. 339]
    //                            Mip 4 (8x8 pages     = 1x1 block)    : [340]
    //                            Mip 5..7 packed tail (4x4+2x2+1x1)   : [341]
    //                            View directory metadata              : [342 .. 351]
    //    Level 2 (Leaf Pool) : VirtualPageTable[blockSlot * 64 + localTileIdx]
    //                          64 NVSVirtualPageEntry records (1 KB) allocated on demand on GPU.
    // =====================================================================
    static constexpr UINT32 kNVSSparseUnmappedIndex           = 0xFFFFFFFFu;
    static constexpr UINT32 kNVSSparsePendingIndex            = 0xFFFFFFFEu;
    static constexpr UINT32 kNVSSparseBlockSidePages          = 8u;
    static constexpr UINT32 kNVSSparsePagesPerBlock           = 64u;
    static constexpr UINT32 kNVSSparseMaxMipLevels            = 8u;   // 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    static constexpr UINT32 kNVSSparseDirTileEntriesPerView   = 342u; // 256 + 64 + 16 + 4 + 1 + 1
    static constexpr UINT32 kNVSSparseDirStride               = 352u; // 16-word aligned directory slab per active view
    static constexpr UINT32 kNVSSparseInitialViewRootCapacity = 4096u;
    static constexpr UINT32 kNVSSparseInitialDirPoolCapacity  = 4096u;
    static constexpr UINT32 kNVSSparseInitialBlockPoolCapacity = 8192u; // 8,192 * 64 = 524,288 leaf slots (8 MB)
    static constexpr UINT32 kNVSSparseFeedbackHashCapacity    = 65536u;

    struct NVSSparsePageAddress
    {
        UINT32 viewIndex    = 0u;
        UINT32 mipLevel     = 0u;
        UINT32 virtualPageX = 0u;
        UINT32 virtualPageY = 0u;
        UINT32 generation   = 1u;
    };

    // Packs (vx:8 | vy:8 | mipLevel:4 | generation:12) for 64-bit (viewIndex, packedCoord) addressing.
    inline constexpr UINT32 PackNVSSparsePageCoord(
        UINT32 virtualPageX,
        UINT32 virtualPageY,
        UINT32 mipLevel   = 0u,
        UINT32 generation = 1u) noexcept
    {
        const UINT32 safeGen = (generation == 0u) ? 1u : (((generation - 1u) % 4095u) + 1u);
        return (virtualPageX & 0xFFu) |
               ((virtualPageY & 0xFFu) << 8u) |
               ((mipLevel & 0xFu) << 16u) |
               (safeGen << 20u);
    }

    inline constexpr NVSSparsePageAddress UnpackNVSSparsePageCoord(UINT32 viewIndex, UINT32 packedCoord) noexcept
    {
        NVSSparsePageAddress addr{};
        addr.viewIndex    = viewIndex;
        addr.virtualPageX = packedCoord & 0xFFu;
        addr.virtualPageY = (packedCoord >> 8u) & 0xFFu;
        addr.mipLevel     = (packedCoord >> 16u) & 0xFu;
        addr.generation   = (packedCoord >> 20u) & 0xFFFu;
        return addr;
    }

    // Computes exact (mipTileOffset in [0..341], localBlockIndex in [0..63]) inside a 352-word view directory.
    inline constexpr BOOL ComputeNVSSparseDirectoryOffset(
        UINT32  mipLevel,
        UINT32  virtualPageX,
        UINT32  virtualPageY,
        UINT32& outDirTileOffset,
        UINT32& outLocalBlockIndex) noexcept
    {
        if (mipLevel >= kNVSSparseMaxMipLevels)
        {
            outDirTileOffset   = kNVSSparseUnmappedIndex;
            outLocalBlockIndex = 0u;
            return false;
        }
        const UINT32 maxPagesAtMip = 128u >> mipLevel;
        if (virtualPageX >= maxPagesAtMip || virtualPageY >= maxPagesAtMip)
        {
            outDirTileOffset   = kNVSSparseUnmappedIndex;
            outLocalBlockIndex = 0u;
            return false;
        }

        if (mipLevel <= 4u)
        {
            const UINT32 tileX        = virtualPageX >> 3u;
            const UINT32 tileY        = virtualPageY >> 3u;
            const UINT32 localX       = virtualPageX & 7u;
            const UINT32 localY       = virtualPageY & 7u;
            const UINT32 tilesPerAxis = 16u >> mipLevel; // 16, 8, 4, 2, 1
            UINT32 baseOffset         = 0u;
            if (mipLevel == 1u)      baseOffset = 256u;
            else if (mipLevel == 2u) baseOffset = 320u;
            else if (mipLevel == 3u) baseOffset = 336u;
            else if (mipLevel == 4u) baseOffset = 340u;
            outDirTileOffset   = baseOffset + tileY * tilesPerAxis + tileX;
            outLocalBlockIndex = (localY << 3u) | localX;
            return true;
        }

        // Mips 5..7 share packed tail block [341]:
        //   Mip 5 (4x4 = 16 pages): local [0 .. 15]
        //   Mip 6 (2x2 = 4 pages) : local [16 .. 19]
        //   Mip 7 (1x1 = 1 page)  : local [20]
        outDirTileOffset = 341u;
        if (mipLevel == 5u)
        {
            outLocalBlockIndex = (virtualPageY << 2u) + virtualPageX;
        }
        else if (mipLevel == 6u)
        {
            outLocalBlockIndex = 16u + (virtualPageY << 1u) + virtualPageX;
        }
        else
        {
            outLocalBlockIndex = 20u;
        }
        return true;
    }

    // GPU Batch View Invalidation & Reclamation Command (32 Bytes, 16-Byte Aligned)
    struct alignas(16) NVSInvalidationCommand
    {
        UINT32 mode              = NVS_INVALIDATE_NONE;
        UINT32 viewIndex         = kNVSSparseUnmappedIndex;
        UINT32 viewCount         = 1u;
        UINT32 regionOrPagesAxis = 0u;
        INT32  originShiftDeltaX = 0;
        INT32  originShiftDeltaY = 0;
        INT32  snappedOriginX    = 0;
        INT32  snappedOriginY    = 0;
    };
    static_assert(sizeof(NVSInvalidationCommand) == 32, "NVSInvalidationCommand must be 32 bytes.");

    // GPU Sparse Allocator State Header (16 Bytes)
    struct alignas(16) NVSSparseAllocatorState
    {
        UINT32 freeDirCount      = 0u;
        UINT32 freeBlockCount    = 0u;
        UINT32 dirOverflowCount  = 0u;
        UINT32 blkOverflowCount  = 0u;
    };
    static_assert(sizeof(NVSSparseAllocatorState) == 16, "NVSSparseAllocatorState must be 16 bytes.");

    // GPU Cache Miss Record for 32-bit Unbounded Fair Priority Allocation (16 Bytes)
    struct alignas(16) NVSCacheMissRecord
    {
        UINT32 prioritySortKey   = 0xFFFFFFFFu;
        UINT32 virtualFlatIndex  = kNVSInvalidVirtualPage;
        UINT32 viewIndex         = kNVSSparseUnmappedIndex;
        UINT32 packedCoordAndPin = 0u;
    };
    static_assert(sizeof(NVSCacheMissRecord) == 16, "NVSCacheMissRecord must be 16 bytes.");

    // =====================================================================
    //  Virtual Page Address & Packed 32-Bit Encoding
    //    Bits [0..7]   (8 bits) : virtualPageX (0..255)
    //    Bits [8..15]  (8 bits) : virtualPageY (0..255)
    //    Bits [16..17] (2 bits) : clipmapLevel (0..3)
    //    Bits [18..23] (6 bits) : lightId      (0..63)
    //    Bits [24..31] (8 bits) : generation   (0..255)
    // =====================================================================
    struct NVSVirtualPageAddress
    {
        UINT32 lightId      = 0u;
        UINT32 clipmapLevel = 0u;
        UINT32 virtualPageX = 0u;
        UINT32 virtualPageY = 0u;
        UINT32 generation   = 1u;
    };

    inline constexpr UINT32 PackNVSVirtualPageAddress(
        UINT32 lightId,
        UINT32 clipmapLevel,
        UINT32 virtualPageX,
        UINT32 virtualPageY,
        UINT32 generation = 1u) noexcept
    {
        const UINT32 safeGen = (generation == 0u) ? 0u : (((generation - 1u) % 255u) + 1u);
        return (virtualPageX & 0xFFu) |
               ((virtualPageY & 0xFFu) << 8u) |
               ((clipmapLevel & 0x3u) << 16u) |
               ((lightId & 0x3Fu) << 18u) |
               ((safeGen & 0xFFu) << 24u);
    }

    inline constexpr UINT32 PackNVSVirtualPageAddress(const NVSVirtualPageAddress& addr) noexcept
    {
        return PackNVSVirtualPageAddress(
            addr.lightId,
            addr.clipmapLevel,
            addr.virtualPageX,
            addr.virtualPageY,
            addr.generation);
    }

    inline constexpr NVSVirtualPageAddress UnpackNVSVirtualPageAddress(UINT32 packed) noexcept
    {
        NVSVirtualPageAddress addr{};
        addr.virtualPageX = packed & 0xFFu;
        addr.virtualPageY = (packed >> 8u) & 0xFFu;
        addr.clipmapLevel = (packed >> 16u) & 0x3u;
        addr.lightId      = (packed >> 18u) & 0x3Fu;
        addr.generation   = (packed >> 24u) & 0xFFu;
        return addr;
    }

    inline constexpr BOOL IsValidNVSVirtualPageAddress(
        const NVSVirtualPageAddress& addr,
        UINT32                       pagesPerAxis,
        UINT32                       clipmapLevelCount = kNVSMaxClipmapLevels,
        UINT32                       maxLightCount     = kNVSDefaultMaxLights) noexcept
    {
        return (pagesPerAxis > 0u) &&
               (pagesPerAxis <= kNVSMaxPagesPerAxis) &&
               (clipmapLevelCount > 0u) &&
               (clipmapLevelCount <= kNVSMaxClipmapLevels) &&
               (maxLightCount > 0u) &&
               (maxLightCount <= kNVSMaxSupportedLights) &&
               (addr.lightId < maxLightCount) &&
               (addr.clipmapLevel < clipmapLevelCount) &&
               (addr.virtualPageX < pagesPerAxis) &&
               (addr.virtualPageY < pagesPerAxis);
    }

    inline constexpr UINT32 ComputeNVSVirtualPageFlatIndex(
        UINT32 lightId,
        UINT32 clipmapLevel,
        UINT32 virtualPageX,
        UINT32 virtualPageY,
        UINT32 pagesPerAxis,
        UINT32 clipmapLevelCount = kNVSMaxClipmapLevels) noexcept
    {
        const UINT32 pagesPerLevel = pagesPerAxis * pagesPerAxis;
        const UINT32 globalLevel   = (lightId * clipmapLevelCount) + clipmapLevel;
        return (globalLevel * pagesPerLevel) + (virtualPageY * pagesPerAxis) + virtualPageX;
    }

    inline constexpr UINT32 ComputeNVSVirtualPageFlatIndex(
        const NVSVirtualPageAddress& addr,
        UINT32                       pagesPerAxis,
        UINT32                       clipmapLevelCount = kNVSMaxClipmapLevels) noexcept
    {
        return ComputeNVSVirtualPageFlatIndex(
            addr.lightId,
            addr.clipmapLevel,
            addr.virtualPageX,
            addr.virtualPageY,
            pagesPerAxis,
            clipmapLevelCount);
    }

    inline constexpr UINT32 PackNVSInvalidationRegion(
        UINT32 minPageX,
        UINT32 minPageY,
        UINT32 maxPageX,
        UINT32 maxPageY) noexcept
    {
        return (minPageX & 0xFFu) |
               ((minPageY & 0xFFu) << 8u) |
               ((maxPageX & 0xFFu) << 16u) |
               ((maxPageY & 0xFFu) << 24u);
    }

    inline constexpr NVSPageState ExtractNVSPageState(UINT32 stateAndFlags) noexcept
    {
        return static_cast<NVSPageState>(stateAndFlags & NVS_PAGE_STATE_MASK);
    }

    inline constexpr NVSPhysicalPageState ExtractNVSPhysicalPageState(UINT32 stateAndFlags) noexcept
    {
        return static_cast<NVSPhysicalPageState>(stateAndFlags & NVS_PHYS_STATE_MASK);
    }

    // =====================================================================
    //  Directional Clipmap Light-Space & Quantized Origin Addressing
    // =====================================================================
    struct NVSClipmapLevelState
    {
        UINT32  lightId                   = 0u;
        UINT32  clipmapLevel              = 0u;
        UINT32  pagesPerAxis              = kNVSDefaultPagesPerAxis;
        FLOAT32 worldExtent               = 64.0f;  // Total width/height of clipmap level in world units (meters)
        FLOAT32 pageWorldSize             = 2.0f;   // worldExtent / pagesPerAxis
        FLOAT32 texelWorldSize            = 2.0f / 128.0f; // pageWorldSize / kNVSPageTexelSize
        Vec3    lightRight                = Vec3(1.0f, 0.0f, 0.0f);
        Vec3    lightUp                   = Vec3(0.0f, 1.0f, 0.0f);
        Vec3    lightDir                  = Vec3(0.0f, 0.0f, 1.0f);
        INT32   snappedOriginPageX        = 0;      // Quantized integer page coordinate X in light space
        INT32   snappedOriginPageY        = 0;      // Quantized integer page coordinate Y in light space
        FLOAT32 snappedLightCenterX       = 0.0f;
        FLOAT32 snappedLightCenterY       = 0.0f;
        FLOAT32 receiverMinZ              = -64.0f;
        FLOAT32 receiverMaxZ              = 64.0f;
        FLOAT32 casterMinZ                = -320.0f;
        FLOAT32 casterMaxZ                = 80.0f;
        FLOAT32 casterExtrusionDistance   = 256.0f;
        FLOAT32 casterDownstreamMargin    = 32.0f;
        FLOAT32 casterConservativePadding = 0.5f;
        FLOAT32 cameraLightZ              = 0.0f;
        FLOAT32 depthBiasNdc              = 0.0f;
        FLOAT32 normalBiasWorld           = 0.0f;
        FLOAT32 originSnapHysteresis      = 0.0f;   // Hysteresis margin in page units [0, 0.45] preventing boundary jitter
        FLOAT32 transitionStartRatio      = 0.85f;  // Start of smooth clipmap transition band [0.5 .. 0.99]
        Mat4    worldToLightView          = Mat4::Identity();
        Mat4    lightViewToWorld          = Mat4::Identity();
        BOOL    initialized               = false;
        BOOL    depthRangeAnchored        = false;
    };

    struct NVSOriginShiftResult
    {
        INT32  deltaPageX               = 0;
        INT32  deltaPageY               = 0;
        INT32  snappedOriginPageX       = 0;
        INT32  snappedOriginPageY       = 0;
        BOOL   originChanged            = false;
        BOOL   fullWindowReset          = false;
        BOOL   depthRangeChanged        = false;
        UINT32 expectedInvalidatedPages = 0u;
        UINT32 expectedPreservedPages   = 0u;
    };

    inline constexpr UINT32 NVSPositiveMod(INT32 value, UINT32 modulus) noexcept
    {
        if (modulus == 0u)
        {
            return 0u;
        }
        const INT32 m = static_cast<INT32>(modulus);
        const INT32 r = value % m;
        return static_cast<UINT32>((r < 0) ? (r + m) : r);
    }

    // Translates an absolute integer world-space page coordinate (worldPageX or worldPageY)
    // into a persistent toroidal VirtualPageTable coordinate [0 .. pagesPerAxis-1].
    // Adding halfPages (pagesPerAxis / 2) ensures that when snappedOriginPage == (0, 0),
    // toroidal coordinates [0 .. pagesPerAxis-1] match window-relative coordinates [0 .. pagesPerAxis-1] 1:1,
    // while remaining 100% invariant to camera movement (snappedOriginPage changes).
    inline constexpr UINT32 WorldPageToNVSToroidalPage(INT32 worldPageCoord, UINT32 pagesPerAxis) noexcept
    {
        const INT32 halfPages = static_cast<INT32>(pagesPerAxis / 2u);
        return NVSPositiveMod(worldPageCoord + halfPages, pagesPerAxis);
    }

    // Translates a toroidal VirtualPageTable coordinate [0 .. pagesPerAxis-1] back into
    // the window-relative coordinate [0 .. pagesPerAxis-1] for the given snappedOriginPage.
    inline constexpr UINT32 NVSToroidalPageToWindowRelative(
        UINT32 toroidalPageCoord,
        INT32  snappedOriginPage,
        UINT32 pagesPerAxis) noexcept
    {
        return NVSPositiveMod(static_cast<INT32>(toroidalPageCoord) - snappedOriginPage, pagesPerAxis);
    }

    // Translates a toroidal VirtualPageTable coordinate [0 .. pagesPerAxis-1] back into
    // the unique absolute integer world-space page coordinate covered by the active clipmap window.
    inline constexpr INT32 NVSToroidalPageToWorldPage(
        UINT32 toroidalPageCoord,
        INT32  snappedOriginPage,
        UINT32 pagesPerAxis) noexcept
    {
        const INT32  halfPages = static_cast<INT32>(pagesPerAxis / 2u);
        const UINT32 relPage   = NVSToroidalPageToWindowRelative(toroidalPageCoord, snappedOriginPage, pagesPerAxis);
        return (snappedOriginPage - halfPages) + static_cast<INT32>(relPage);
    }

    inline NVSOriginShiftResult UpdateNVSClipmapSnappedOrigin(
        NVSClipmapLevelState& levelState,
        const Vec3&           cameraWorldPos) noexcept
    {
        NVSOriginShiftResult res{};
        if (levelState.pagesPerAxis == 0u || levelState.worldExtent <= 0.0f)
        {
            return res;
        }

        levelState.pageWorldSize  = levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis);
        levelState.texelWorldSize = levelState.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        const FLOAT32 projX =
            cameraWorldPos.x * levelState.lightRight.x +
            cameraWorldPos.y * levelState.lightRight.y +
            cameraWorldPos.z * levelState.lightRight.z;
        const FLOAT32 projY =
            cameraWorldPos.x * levelState.lightUp.x +
            cameraWorldPos.y * levelState.lightUp.y +
            cameraWorldPos.z * levelState.lightUp.z;

        const FLOAT32 pageCoordX = projX / levelState.pageWorldSize;
        const FLOAT32 pageCoordY = projY / levelState.pageWorldSize;
        INT32 newOriginPageX     = static_cast<INT32>(std::floor(pageCoordX + 0.5f));
        INT32 newOriginPageY     = static_cast<INT32>(std::floor(pageCoordY + 0.5f));

        if (!levelState.initialized)
        {
            levelState.snappedOriginPageX  = newOriginPageX;
            levelState.snappedOriginPageY  = newOriginPageY;
            levelState.snappedLightCenterX = static_cast<FLOAT32>(newOriginPageX) * levelState.pageWorldSize;
            levelState.snappedLightCenterY = static_cast<FLOAT32>(newOriginPageY) * levelState.pageWorldSize;
            levelState.initialized         = true;
            res.snappedOriginPageX         = newOriginPageX;
            res.snappedOriginPageY         = newOriginPageY;
            res.expectedPreservedPages     = levelState.pagesPerAxis * levelState.pagesPerAxis;
            return res;
        }

        if (levelState.originSnapHysteresis > 0.0f)
        {
            const FLOAT32 h = std::clamp(levelState.originSnapHysteresis, 0.0f, 0.45f);
            if (std::fabs(pageCoordX - static_cast<FLOAT32>(levelState.snappedOriginPageX)) <= (0.5f + h))
            {
                newOriginPageX = levelState.snappedOriginPageX;
            }
            if (std::fabs(pageCoordY - static_cast<FLOAT32>(levelState.snappedOriginPageY)) <= (0.5f + h))
            {
                newOriginPageY = levelState.snappedOriginPageY;
            }
        }

        res.deltaPageX         = newOriginPageX - levelState.snappedOriginPageX;
        res.deltaPageY         = newOriginPageY - levelState.snappedOriginPageY;
        res.snappedOriginPageX = newOriginPageX;
        res.snappedOriginPageY = newOriginPageY;
        res.originChanged      = (res.deltaPageX != 0) || (res.deltaPageY != 0);

        levelState.snappedOriginPageX  = newOriginPageX;
        levelState.snappedOriginPageY  = newOriginPageY;
        levelState.snappedLightCenterX = static_cast<FLOAT32>(newOriginPageX) * levelState.pageWorldSize;
        levelState.snappedLightCenterY = static_cast<FLOAT32>(newOriginPageY) * levelState.pageWorldSize;

        const UINT32 N      = levelState.pagesPerAxis;
        const UINT32 total  = N * N;
        const UINT32 absDX  = static_cast<UINT32>((res.deltaPageX < 0) ? -res.deltaPageX : res.deltaPageX);
        const UINT32 absDY  = static_cast<UINT32>((res.deltaPageY < 0) ? -res.deltaPageY : res.deltaPageY);

        if (absDX >= N || absDY >= N)
        {
            res.fullWindowReset          = true;
            res.expectedInvalidatedPages = total;
            res.expectedPreservedPages   = 0u;
        }
        else
        {
            const UINT32 overlapX        = N - absDX;
            const UINT32 overlapY        = N - absDY;
            res.expectedPreservedPages   = overlapX * overlapY;
            res.expectedInvalidatedPages = total - res.expectedPreservedPages;
        }
        return res;
    }

    // Alias matching canonical coordinate helper naming
    inline NVSOriginShiftResult ComputeNVSSnappedClipmapConfig(
        NVSClipmapLevelState& levelState,
        const Vec3&           cameraWorldPos) noexcept
    {
        return UpdateNVSClipmapSnappedOrigin(levelState, cameraWorldPos);
    }

    // Maps a world-space point into a clipmap level's toroidal virtual page coordinates [0 .. pagesPerAxis-1]
    // and intra-page UV [0, 1). Also optionally outputs absolute integer world-space page coordinates.
    // Returns false if the point lies outside the clipmap level's active window coverage.
    inline BOOL WorldPositionToNVSVirtualPage(
        const NVSClipmapLevelState& levelState,
        const Vec3&                 worldPos,
        UINT32&                     outVirtualPageX,
        UINT32&                     outVirtualPageY,
        Vec2*                       outIntraPageUV = nullptr,
        INT32*                      outWorldPageX  = nullptr,
        INT32*                      outWorldPageY  = nullptr) noexcept
    {
        if (levelState.pagesPerAxis == 0u || levelState.worldExtent <= 0.0f)
        {
            outVirtualPageX = kNVSInvalidVirtualPage;
            outVirtualPageY = kNVSInvalidVirtualPage;
            return false;
        }

        const FLOAT32 pageSize = levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis);
        if (pageSize <= 0.0f)
        {
            outVirtualPageX = kNVSInvalidVirtualPage;
            outVirtualPageY = kNVSInvalidVirtualPage;
            return false;
        }

        const FLOAT32 projX =
            worldPos.x * levelState.lightRight.x +
            worldPos.y * levelState.lightRight.y +
            worldPos.z * levelState.lightRight.z;
        const FLOAT32 projY =
            worldPos.x * levelState.lightUp.x +
            worldPos.y * levelState.lightUp.y +
            worldPos.z * levelState.lightUp.z;

        const FLOAT32 exactWorldPageX = projX / pageSize;
        const FLOAT32 exactWorldPageY = projY / pageSize;
        const FLOAT32 flX             = std::floor(exactWorldPageX);
        const FLOAT32 flY             = std::floor(exactWorldPageY);
        const INT32   worldPageX      = static_cast<INT32>(flX);
        const INT32   worldPageY      = static_cast<INT32>(flY);

        const INT32 halfPages     = static_cast<INT32>(levelState.pagesPerAxis / 2u);
        const INT32 minWorldPageX = levelState.snappedOriginPageX - halfPages;
        const INT32 minWorldPageY = levelState.snappedOriginPageY - halfPages;
        const INT32 relPageX      = worldPageX - minWorldPageX;
        const INT32 relPageY      = worldPageY - minWorldPageY;
        const INT32 lim           = static_cast<INT32>(levelState.pagesPerAxis);

        if (relPageX < 0 || relPageX >= lim || relPageY < 0 || relPageY >= lim)
        {
            outVirtualPageX = kNVSInvalidVirtualPage;
            outVirtualPageY = kNVSInvalidVirtualPage;
            return false;
        }

        outVirtualPageX = WorldPageToNVSToroidalPage(worldPageX, levelState.pagesPerAxis);
        outVirtualPageY = WorldPageToNVSToroidalPage(worldPageY, levelState.pagesPerAxis);

        if (outIntraPageUV)
        {
            outIntraPageUV->x = exactWorldPageX - flX;
            outIntraPageUV->y = exactWorldPageY - flY;
        }
        if (outWorldPageX)
        {
            *outWorldPageX = worldPageX;
        }
        if (outWorldPageY)
        {
            *outWorldPageY = worldPageY;
        }
        return true;
    }

    using NVSClipmapLevelConfig = NVSClipmapLevelState;

    // Alias matching canonical coordinate helper naming
    inline BOOL WorldToNVSVirtualAddress(
        const NVSClipmapLevelState& levelState,
        const Vec3&                 worldPos,
        NVSVirtualPageAddress&      outAddress,
        Vec2*                       outIntraPageUV = nullptr,
        INT32*                      outWorldPageX  = nullptr,
        INT32*                      outWorldPageY  = nullptr) noexcept
    {
        UINT32 vx = kNVSInvalidVirtualPage;
        UINT32 vy = kNVSInvalidVirtualPage;
        if (!WorldPositionToNVSVirtualPage(levelState, worldPos, vx, vy, outIntraPageUV, outWorldPageX, outWorldPageY))
        {
            return false;
        }
        outAddress.lightId      = levelState.lightId;
        outAddress.clipmapLevel = levelState.clipmapLevel;
        outAddress.virtualPageX = vx;
        outAddress.virtualPageY = vy;
        outAddress.generation   = 1u;
        return true;
    }

    inline NVSVirtualPageAddress WorldToNVSVirtualAddress(
        const Vec3&                 worldPos,
        const NVSClipmapLevelState& levelState,
        UINT32                      generation = 1u) noexcept
    {
        NVSVirtualPageAddress addr{};
        if (WorldToNVSVirtualAddress(levelState, worldPos, addr))
        {
            addr.generation = generation;
        }
        return addr;
    }

    inline constexpr UINT32 ComputeNVSVirtualPageTableIndex(
        UINT32 lightId,
        UINT32 clipmapLevel,
        UINT32 virtualPageX,
        UINT32 virtualPageY,
        UINT32 pagesPerAxis,
        UINT32 clipmapLevelCount = kNVSMaxClipmapLevels) noexcept
    {
        return ComputeNVSVirtualPageFlatIndex(lightId, clipmapLevel, virtualPageX, virtualPageY, pagesPerAxis, clipmapLevelCount);
    }

    inline constexpr UINT32 ComputeNVSVirtualPageTableIndex(
        const NVSVirtualPageAddress& addr,
        UINT32                       pagesPerAxis,
        UINT32                       clipmapLevelCount = kNVSMaxClipmapLevels) noexcept
    {
        return ComputeNVSVirtualPageFlatIndex(addr, pagesPerAxis, clipmapLevelCount);
    }

    // Computes light-space XY bounds [minXY, maxXY] and world-space center of a toroidal virtual page
    // within the active clipmap window.
    inline BOOL NVSVirtualPageToLightSpaceBounds(
        const NVSClipmapLevelState& levelState,
        UINT32                      virtualPageX,
        UINT32                      virtualPageY,
        Vec2&                       outLightMinXY,
        Vec2&                       outLightMaxXY,
        Vec3*                       outWorldCenter = nullptr) noexcept
    {
        if (levelState.pagesPerAxis == 0u ||
            virtualPageX >= levelState.pagesPerAxis ||
            virtualPageY >= levelState.pagesPerAxis ||
            levelState.worldExtent <= 0.0f)
        {
            return false;
        }

        const FLOAT32 pageSize   = levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis);
        const INT32   worldPageX = NVSToroidalPageToWorldPage(virtualPageX, levelState.snappedOriginPageX, levelState.pagesPerAxis);
        const INT32   worldPageY = NVSToroidalPageToWorldPage(virtualPageY, levelState.snappedOriginPageY, levelState.pagesPerAxis);

        outLightMinXY.x = static_cast<FLOAT32>(worldPageX) * pageSize;
        outLightMinXY.y = static_cast<FLOAT32>(worldPageY) * pageSize;
        outLightMaxXY.x = outLightMinXY.x + pageSize;
        outLightMaxXY.y = outLightMinXY.y + pageSize;

        if (outWorldCenter)
        {
            const FLOAT32 cx = 0.5f * (outLightMinXY.x + outLightMaxXY.x);
            const FLOAT32 cy = 0.5f * (outLightMinXY.y + outLightMaxXY.y);
            outWorldCenter->x = levelState.lightRight.x * cx + levelState.lightUp.x * cy;
            outWorldCenter->y = levelState.lightRight.y * cx + levelState.lightUp.y * cy;
            outWorldCenter->z = levelState.lightRight.z * cx + levelState.lightUp.z * cy;
        }
        return true;
    }

    // =====================================================================
    //  Persistent GPU Virtual Page Table Entry (16 Bytes, 16-Byte Aligned)
    //  Matches NSL RWStructuredBuffer<NVSVirtualPageEntry> 1:1.
    //  Zero CPU pointers or native D3D12 handles.
    // =====================================================================
    struct alignas(16) NVSVirtualPageEntry
    {
        UINT32 stateAndFlags        = NVS_PAGE_STATE_UNMAPPED;
        UINT32 physicalPageIndex    = kNVSInvalidPhysicalPage;
        UINT32 virtualAddressPacked = 0u;
        UINT32 lastTouchedFrame     = 0u;
    };

    static_assert(sizeof(NVSVirtualPageEntry) == 16, "NVSVirtualPageEntry must be exactly 16 bytes.");
    static_assert(alignof(NVSVirtualPageEntry) == 16, "NVSVirtualPageEntry must be 16-byte aligned.");
    static_assert(offsetof(NVSVirtualPageEntry, stateAndFlags)        == 0,  "stateAndFlags offset mismatch");
    static_assert(offsetof(NVSVirtualPageEntry, physicalPageIndex)    == 4,  "physicalPageIndex offset mismatch");
    static_assert(offsetof(NVSVirtualPageEntry, virtualAddressPacked) == 8,  "virtualAddressPacked offset mismatch");
    static_assert(offsetof(NVSVirtualPageEntry, lastTouchedFrame)     == 12, "lastTouchedFrame offset mismatch");

    // =====================================================================
    //  Persistent GPU Physical Page Metadata Entry (32 Bytes, 16-Byte Aligned)
    //  Matches NSL RWStructuredBuffer<NVSPhysicalPageEntry> 1:1.
    // =====================================================================
    struct alignas(16) NVSPhysicalPageEntry
    {
        // Row 0 (Offsets 0..15)
        UINT32 stateAndFlags         = NVS_PHYS_STATE_FREE;
        UINT32 ownerVirtualFlatIndex = kNVSInvalidVirtualPage;
        UINT32 ownerVirtualAddress   = 0u;
        UINT32 physicalGeneration    = 0u;

        // Row 1 (Offsets 16..31)
        UINT32 lastUsedFrame         = 0u;
        UINT32 allocatedFrame        = 0u;
        UINT32 minEvictableFrame     = 0u;
        UINT32 reserved              = 0u;
    };

    static_assert(sizeof(NVSPhysicalPageEntry) == 32, "NVSPhysicalPageEntry must be exactly 32 bytes.");
    static_assert(alignof(NVSPhysicalPageEntry) == 16, "NVSPhysicalPageEntry must be 16-byte aligned.");
    static_assert(offsetof(NVSPhysicalPageEntry, stateAndFlags)         == 0,  "stateAndFlags offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, ownerVirtualFlatIndex) == 4,  "ownerVirtualFlatIndex offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, ownerVirtualAddress)   == 8,  "ownerVirtualAddress offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, physicalGeneration)    == 12, "physicalGeneration offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, lastUsedFrame)         == 16, "lastUsedFrame offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, allocatedFrame)        == 20, "allocatedFrame offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, minEvictableFrame)     == 24, "minEvictableFrame offset mismatch");
    static_assert(offsetof(NVSPhysicalPageEntry, reserved)              == 28, "reserved offset mismatch");

    // =====================================================================
    //  GPU Page Request Record (16 Bytes, 16-Byte Aligned)
    //  Matches NSL StructuredBuffer<NVSVirtualPageRequest> 1:1.
    // =====================================================================
    struct alignas(16) NVSVirtualPageRequest
    {
        UINT32 virtualAddressPacked = 0u;
        UINT32 requestFlags         = NVS_REQ_FLAG_NONE;
        UINT32 requestPriority      = 0u;
        UINT32 requestUserData      = 0u;
    };

    static_assert(sizeof(NVSVirtualPageRequest) == 16, "NVSVirtualPageRequest must be exactly 16 bytes.");
    static_assert(alignof(NVSVirtualPageRequest) == 16, "NVSVirtualPageRequest must be 16-byte aligned.");
    static_assert(offsetof(NVSVirtualPageRequest, virtualAddressPacked) == 0,  "virtualAddressPacked offset mismatch");
    static_assert(offsetof(NVSVirtualPageRequest, requestFlags)         == 4,  "requestFlags offset mismatch");
    static_assert(offsetof(NVSVirtualPageRequest, requestPriority)      == 8,  "requestPriority offset mismatch");
    static_assert(offsetof(NVSVirtualPageRequest, requestUserData)      == 12, "requestUserData offset mismatch");

    // =====================================================================
    //  GPU Residency Telemetry Record (64 Bytes, 16-Byte Aligned)
    //  Written on the GPU by RuntimeNVSPageResidency.nsl.
    // =====================================================================
    struct alignas(16) NVSResidencyTelemetry
    {
        // Row 0 (Offsets 0..15): Request Intake & Deduplication
        UINT32 requestedCount           = 0u;
        UINT32 validRequestCount        = 0u;
        UINT32 invalidRequestCount      = 0u;
        UINT32 uniqueRequestCount       = 0u;

        // Row 1 (Offsets 16..31): Cache Hit/Miss & Allocation
        UINT32 duplicateRequestCount    = 0u;
        UINT32 cacheHitCount            = 0u;
        UINT32 cacheMissCount           = 0u;
        UINT32 allocatedFromFreeCount   = 0u;

        // Row 2 (Offsets 32..47): Eviction, Overflow, Invalidation & Selective Clear
        UINT32 evictedCount             = 0u;
        UINT32 overflowCount            = 0u;
        UINT32 invalidatedCount         = 0u;
        UINT32 dirtyClearSliceCount     = 0u;

        // Row 3 (Offsets 48..63): Post-Pass Table Population Snapshot
        UINT32 residentValidPages       = 0u;
        UINT32 allocatedNotClearedPages = 0u;
        UINT32 clearedNotRenderedPages  = 0u;
        UINT32 freePhysicalPages        = 0u;
    };

    static_assert(sizeof(NVSResidencyTelemetry) == 64, "NVSResidencyTelemetry must be 64 bytes (4 x 16B).");
    static_assert(alignof(NVSResidencyTelemetry) == 16, "NVSResidencyTelemetry must be 16-byte aligned.");
    static_assert(offsetof(NVSResidencyTelemetry, requestedCount)        == 0,  "requestedCount offset mismatch");
    static_assert(offsetof(NVSResidencyTelemetry, duplicateRequestCount) == 16, "duplicateRequestCount offset mismatch");
    static_assert(offsetof(NVSResidencyTelemetry, evictedCount)          == 32, "evictedCount offset mismatch");
    static_assert(offsetof(NVSResidencyTelemetry, residentValidPages)    == 48, "residentValidPages offset mismatch");

    static_assert(sizeof(NVSPageRenderCommitRecord) == sizeof(NVSVirtualPageRequest),
                  "NVSPageRenderCommitRecord must share the 16-byte PageRequests stride.");
    static_assert(offsetof(NVSPageRenderCommitRecord, physicalPageIndex)          == 0,  "commit physicalPageIndex offset mismatch");
    static_assert(offsetof(NVSPageRenderCommitRecord, resultFlags)                == 4,  "commit resultFlags offset mismatch");
    static_assert(offsetof(NVSPageRenderCommitRecord, expectedPhysicalGeneration) == 8,  "commit expectedPhysicalGeneration offset mismatch");
    static_assert(offsetof(NVSPageRenderCommitRecord, expectedVirtualFlatIndex)   == 12, "commit expectedVirtualFlatIndex offset mismatch");

    inline NVSPageCommitResult ExtractNVSPageCommitResult(const NVSResidencyTelemetry& tel) noexcept
    {
        NVSPageCommitResult r{};
        r.submittedRecords        = tel.requestedCount;
        r.committedPages          = tel.validRequestCount;
        r.rejectedInvalidState    = tel.invalidRequestCount;
        r.rejectedIdentity        = tel.duplicateRequestCount;
        r.keptForRerender         = tel.uniqueRequestCount;
        r.residentValidPages      = tel.residentValidPages;
        r.clearedNotRenderedPages = tel.clearedNotRenderedPages;
        return r;
    }

    // =====================================================================
    //  NVS Page Manager Configuration & Pass Execution Descriptors
    // =====================================================================
    struct NVSPageManagerConfig
    {
        UINT32 pagesPerAxis               = kNVSDefaultPagesPerAxis; // e.g. 32, 64, or 128 virtual pages per axis
        UINT32 clipmapLevelCount          = kNVSMaxClipmapLevels;    // 4 levels (0..3) in legacy flat mode
        UINT32 maxLightCount              = kNVSDefaultMaxLights;    // Legacy flat mode light count
        UINT32 physicalPoolCapacity       = kNVSMaxPhysicalPoolPages; // 64..2048 physical 128x128 slices
        UINT32 maxPageRequestsCapacity    = 8192u;                   // Max requests per frame
        UINT32 minInFlightProtectFrames   = 2u;                      // Multi-frame-in-flight eviction protection age
        BOOL   enableSparseHierarchy      = false;                   // True in production runtime: 2-level sparse demand-driven VPT
        UINT32 initialViewTableCapacity   = 16u;                     // Initial shadow view table capacity (dynamically grows 2x)
        UINT32 initialSparseRootCapacity  = kNVSSparseInitialViewRootCapacity;
        UINT32 initialSparseDirCapacity   = kNVSSparseInitialDirPoolCapacity;
        UINT32 initialSparseBlockCapacity = kNVSSparseInitialBlockPoolCapacity;
    };

    struct NVSSparseHierarchyMetrics
    {
        UINT32 viewTableCapacity   = 0u;
        UINT32 activeViewCount     = 0u;
        UINT32 sparseRootCapacity  = 0u;
        UINT32 sparseDirCapacity   = 0u;
        UINT32 allocatedDirCount   = 0u;
        UINT32 freeDirCount        = 0u;
        UINT32 sparseBlockCapacity = 0u;
        UINT32 allocatedBlockCount = 0u;
        UINT32 freeBlockCount      = 0u;
        UINT32 dirOverflowCount    = 0u;
        UINT32 blkOverflowCount    = 0u;
        UINT64 sparseMemoryBytes   = 0ull;
    };

    struct NVSResidencyPassDesc
    {
        UINT32              currentFrame                    = 1u;
        UINT32              requestCount                    = 0u;
        BOOL                executeSelectiveClear           = true;
        BOOL                markClearedAfterPass            = true;
        FLOAT32             clearDepth                      = 1.0f;
        NVSInvalidationMode invalidationMode                = NVS_INVALIDATE_NONE;
        UINT32              invalidateLightId               = 0u;
        UINT32              invalidateClipmapLevel          = 0u;
        UINT32              invalidateRegionPacked          = 0u;
        INT32               originShiftDeltaX               = 0;
        INT32               originShiftDeltaY               = 0;
        INT32               snappedOriginPageX              = 0;
        INT32               snappedOriginPageY              = 0;
        UINT32              overrideInFlightProtectFrames   = 0u; // 0 = use manager default (minInFlightProtectFrames)
        BOOL                useGPURequestCountFromTelemetry = false; // read requestCount directly from OutTelemetry[0].requestedCount on GPU
        BOOL                executePageProjection           = true;  // build per-page projection & render metadata on GPU
    };

    // =====================================================================
    //  Screen-Space Shadow Feedback & Page Request Generation
    // =====================================================================
    static constexpr UINT32 kNVSFeedbackBitmaskWordsPerLevelLight = 128u;  // 64x64 = 4096 bits = 128 UINT32 words
    static constexpr UINT32 kNVSFeedbackBitmaskWordsPerLevel      = 512u;  // 4 lights * 128 words = 512 UINT32 words
    static constexpr UINT32 kNVSFeedbackBitmaskTotalWords         = 2048u; // 4 levels * 512 words = 2048 UINT32 words (65,536 bits)
    static constexpr UINT32 kNVSFeedbackBitmaskWordCount          = 2048u; // Alias for 2048 UINT32 words
    static constexpr UINT32 kNVSFeedbackTileSize                  = 8u;    // 8x8 screen pixel tile per 64-thread workgroup
    static constexpr UINT32 kNVSDefaultTotalVirtualPages          = 65536u;
    static constexpr UINT32 kNVSDefaultMaxPageRequests            = 4096u;

    enum NVSFeedbackDispatchMode : UINT32
    {
        NVS_FEEDBACK_DISPATCH_RESET_BITMASK_AND_TELEMETRY    = 0u,
        NVS_FEEDBACK_DISPATCH_MARK_SCREEN_DEPTH              = 1u,
        NVS_FEEDBACK_DISPATCH_COMPACT_PAGE_REQUESTS          = 2u,
        NVS_FEEDBACK_DISPATCH_COMPACT_REQUESTS_AND_TELEMETRY = 2u,
        NVS_FEEDBACK_DISPATCH_REDUCE_MACRO_TILE_BOUNDS       = 3u,
        NVS_FEEDBACK_DISPATCH_CULL_VIEWS_TO_MACRO_TILES      = 4u
    };

    enum NVSDepthCompareMode : UINT32
    {
        NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE = 0u, // Clear = 1.0f, valid surface depth in [0.0f, clearDepth - eps)
        NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO = 1u, // Clear = 0.0f, valid surface depth in (clearDepth + eps, 1.0f]
        NVS_DEPTH_COMPARE_STANDARD          = 0u,
        NVS_DEPTH_COMPARE_REVERSED_Z        = 1u
    };

    enum NVSFootprintMode : UINT32
    {
        NVS_FOOTPRINT_EXACT_PAGE       = 0u, // Request only the exact virtual page hit by the reconstructed surface point
        NVS_FOOTPRINT_BORDER_NEIGHBORS = 1u  // Also request adjacent neighbor pages within clipmap window when near page border
    };

    enum NVSDepthPixelClassification : UINT32
    {
        NVS_PIXEL_VALID_SURFACE       = 0u,
        NVS_PIXEL_REJECT_BACKGROUND   = 1u,
        NVS_PIXEL_REJECT_INVALID      = 2u,
        NVS_PIXEL_REJECT_FRUSTUM      = 3u,
        NVS_PIXEL_REJECT_OUT_OF_RANGE = 4u
    };

    using NVSVirtualAddress = NVSVirtualPageAddress;

    inline constexpr UINT32 PackNVSVirtualAddress(const NVSVirtualPageAddress& addr) noexcept
    {
        return PackNVSVirtualPageAddress(addr);
    }

    inline constexpr NVSVirtualPageAddress UnpackNVSVirtualAddress(UINT32 packed) noexcept
    {
        return UnpackNVSVirtualPageAddress(packed);
    }

    inline constexpr UINT32 ComputeNVSFeedbackBitmaskBitIndex(const NVSVirtualPageAddress& addr) noexcept
    {
        const UINT32 localBit = ((addr.virtualPageY & 63u) << 6u) | (addr.virtualPageX & 63u);
        return (((addr.clipmapLevel & 3u) * 4u + (addr.lightId & 3u)) << 12u) | localBit;
    }

    // =====================================================================
    //  4-Level Directional Light Clipmap Hierarchy Configuration
    // =====================================================================
    struct NVSClipmapHierarchyState
    {
        UINT32               lightId                                     = 0u;
        UINT32               activeLevelCount                            = kNVSMaxClipmapLevels;
        UINT32               pagesPerAxis                                = 64u;
        BOOL                 lightValid                                  = true;
        Vec3                 lightRight                                  = Vec3(1.0f, 0.0f, 0.0f);
        Vec3                 lightUp                                     = Vec3(0.0f, 1.0f, 0.0f);
        Vec3                 lightDir                                    = Vec3(0.0f, 0.0f, 1.0f);
        Vec3                 lightForward                                = Vec3(0.0f, 0.0f, 1.0f);
        Mat4                 worldToLightView                            = Mat4::Identity();
        Mat4                 lightViewToWorld                            = Mat4::Identity();
        FLOAT32              maxShadowDistance                           = 0.0f; // 0.0f = bounded only by coarsest clipmap window
        FLOAT32              levelScaleFactor                            = 2.0f; // Geometric inter-level extent ratio (2.0 at default RES_1024)
        FLOAT32              nearPlaneOffset                             = 0.0f; // Upstream casterMinZ near-plane shift relative to 0.1m default
        FLOAT32              originSnapHysteresis                        = 0.0f; // Hysteresis margin in page units [0, 0.45]
        FLOAT32              transitionStartRatio                        = 0.85f;
        NVSDepthCompareMode  depthMode                                   = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        FLOAT32              levelWorldExtents[kNVSMaxClipmapLevels]     = { 128.0f, 256.0f, 512.0f, 1024.0f };
        FLOAT32              levelPageWorldSizes[kNVSMaxClipmapLevels]   = { 2.0f, 4.0f, 8.0f, 16.0f };
        FLOAT32              levelTexelWorldSizes[kNVSMaxClipmapLevels]  = { 2.0f / 128.0f, 4.0f / 128.0f, 8.0f / 128.0f, 16.0f / 128.0f };
        FLOAT32              levelHalfExtents[kNVSMaxClipmapLevels]      = { 64.0f, 128.0f, 256.0f, 512.0f };
        INT32                snappedOriginPageX[kNVSMaxClipmapLevels]    = { 0, 0, 0, 0 };
        INT32                snappedOriginPageY[kNVSMaxClipmapLevels]    = { 0, 0, 0, 0 };
        NVSClipmapLevelState levels[kNVSMaxClipmapLevels]{};
    };

    inline BOOL ComputeNVSOrthonormalLightBasis(
        const Vec3& lightDirection,
        Vec3&       outLightRight,
        Vec3&       outLightUp,
        Vec3&       outLightDir) noexcept
    {
        const FLOAT32 lenSq = lightDirection.Dot(lightDirection);
        if (lenSq <= 1e-8f || std::isnan(lenSq) || std::isinf(lenSq))
        {
            outLightRight = Vec3(0.0f, 0.0f, 0.0f);
            outLightUp    = Vec3(0.0f, 0.0f, 0.0f);
            outLightDir   = Vec3(0.0f, 0.0f, 0.0f);
            return false;
        }

        outLightDir = lightDirection.Normalized();
        const Vec3 refUp = (std::fabs(outLightDir.y) < 0.99f)
            ? Vec3(0.0f, 1.0f, 0.0f)
            : Vec3(0.0f, 0.0f, 1.0f);
        Vec3 rightRaw = refUp.Cross(outLightDir);
        // Gram-Schmidt orthogonalization against outLightDir for numerical stability
        rightRaw      = rightRaw - outLightDir * rightRaw.Dot(outLightDir);
        outLightRight = rightRaw.Normalized();
        outLightUp    = outLightDir.Cross(outLightRight).Normalized();
        return true;
    }

    // Builds pure orthonormal rotation matrices World <-> LightView from orthonormal basis {R, U, D}.
    // Row 0 of worldToLightView = lightRight, Row 1 = lightUp, Row 2 = lightDir, Row 3 = (0,0,0,1).
    // lightViewToWorld = worldToLightView.Transposed() (exact inverse for orthonormal rotation).
    inline void ComputeNVSLightViewMatrices(
        const Vec3& lightRight,
        const Vec3& lightUp,
        const Vec3& lightDir,
        Mat4&       outWorldToLightView,
        Mat4&       outLightViewToWorld) noexcept
    {
        outWorldToLightView = Mat4::Identity();
        // Column 0
        outWorldToLightView.m[0]  = lightRight.x;
        outWorldToLightView.m[1]  = lightUp.x;
        outWorldToLightView.m[2]  = lightDir.x;
        outWorldToLightView.m[3]  = 0.0f;
        // Column 1
        outWorldToLightView.m[4]  = lightRight.y;
        outWorldToLightView.m[5]  = lightUp.y;
        outWorldToLightView.m[6]  = lightDir.y;
        outWorldToLightView.m[7]  = 0.0f;
        // Column 2
        outWorldToLightView.m[8]  = lightRight.z;
        outWorldToLightView.m[9]  = lightUp.z;
        outWorldToLightView.m[10] = lightDir.z;
        outWorldToLightView.m[11] = 0.0f;
        // Column 3
        outWorldToLightView.m[12] = 0.0f;
        outWorldToLightView.m[13] = 0.0f;
        outWorldToLightView.m[14] = 0.0f;
        outWorldToLightView.m[15] = 1.0f;

        outLightViewToWorld = outWorldToLightView.Transposed();
    }

    inline BOOL ComputeNVSLightViewMatrices(
        const Vec3& lightDirection,
        Vec3&       outLightRight,
        Vec3&       outLightUp,
        Vec3&       outLightDir,
        Mat4&       outWorldToLightView,
        Mat4&       outLightViewToWorld) noexcept
    {
        if (!ComputeNVSOrthonormalLightBasis(lightDirection, outLightRight, outLightUp, outLightDir))
        {
            outWorldToLightView = Mat4::Identity();
            outLightViewToWorld = Mat4::Identity();
            return false;
        }
        ComputeNVSLightViewMatrices(outLightRight, outLightUp, outLightDir, outWorldToLightView, outLightViewToWorld);
        return true;
    }

    // Maps LightShadowResolution -> base virtual pages per axis on 128x128 physical tiles:
    //   RES_512 -> 4, RES_1024 -> 8, RES_2048 -> 16, RES_4096 -> 32, RES_8192 -> 64, RES_16384 -> 128.
    inline constexpr UINT32 ComputeNVSVirtualPagesPerAxisFromResolution(LightShadowResolution resolution) noexcept
    {
        switch (resolution)
        {
            case LightShadowResolution::RES_512:   return 4u;
            case LightShadowResolution::RES_1024:  return 8u;
            case LightShadowResolution::RES_2048:  return 16u;
            case LightShadowResolution::RES_4096:  return 32u;
            case LightShadowResolution::RES_8192:  return 64u;
            case LightShadowResolution::RES_16384: return 128u;
            default:                               return 8u;
        }
    }

    inline constexpr UINT32 ComputeNVSMipCountFromPagesPerAxis(UINT32 pagesPerAxis) noexcept
    {
        UINT32 count = 1u;
        UINT32 p     = (pagesPerAxis > kNVSMaxPagesPerAxis) ? kNVSMaxPagesPerAxis : pagesPerAxis;
        while (p > 1u && count < kNVSSparseMaxMipLevels)
        {
            p >>= 1u;
            ++count;
        }
        return count;
    }

    inline constexpr UINT32 ComputeNVSPerspectivePagesPerAxisFromResolution(
        LightShadowResolution resolution,
        UINT32                maxTablePagesPerAxis = kNVSMaxPagesPerAxis) noexcept
    {
        const UINT32 cap     = (maxTablePagesPerAxis > 0u && maxTablePagesPerAxis <= kNVSMaxPagesPerAxis)
            ? maxTablePagesPerAxis
            : kNVSMaxPagesPerAxis;
        const UINT32 desired = ComputeNVSVirtualPagesPerAxisFromResolution(resolution);
        return (desired < cap) ? desired : cap;
    }

    // Unified 16K virtual shadow space detail tier mapping:
    // Every shadow view operates inside a unified 128x128-page (16384x16384, 8-mip) sparse virtual
    // hierarchy. LightShadowResolution selects the per-light detail tier (minimum permitted mip and
    // screen-space LOD bias): RES_16384 targets [0.5, 1.0] shadow texels per screen pixel (sub-pixel
    // crispness without 4x tile over-allocation), while lower tiers (RES_8192 .. RES_512) step down
    // cleanly by +0.5 LOD bias per tier (~2x tile reduction per step) without shrinking the virtual grid.
    inline void ComputeNVSDetailTierFromResolution(
        LightShadowResolution resolution,
        UINT32&               outMinMip,
        FLOAT32&              outLodBias) noexcept
    {
        switch (resolution)
        {
            case LightShadowResolution::RES_16384: outMinMip = 0u; outLodBias =  0.0f; break;
            case LightShadowResolution::RES_8192:  outMinMip = 1u; outLodBias = +0.5f; break;
            case LightShadowResolution::RES_4096:  outMinMip = 2u; outLodBias = +1.0f; break;
            case LightShadowResolution::RES_2048:  outMinMip = 3u; outLodBias = +1.5f; break;
            case LightShadowResolution::RES_1024:  outMinMip = 4u; outLodBias = +2.0f; break;
            case LightShadowResolution::RES_512:   outMinMip = 5u; outLodBias = +2.5f; break;
            default:                               outMinMip = 0u; outLodBias =  0.0f; break;
        }
    }

    // Spot Light detail tier inside the unified 16K virtual mip pyramid.
    // For wide cones (tan(outerCone) > 1.0), shifts minMip finer continuously by ceil(log2(tan(outerCone)))
    // so central 90-degree cone angular texel density remains invariant from 0.5 deg to 88 deg with zero
    // threshold discontinuities while screen-space mip selection naturally coarsens the outer skirt.
    inline void ComputeNVSSpotDetailTier(
        LightShadowResolution resolution,
        FLOAT32               outerConeRadians,
        UINT32&               outMinMip,
        FLOAT32&              outLodBias) noexcept
    {
        ComputeNVSDetailTierFromResolution(resolution, outMinMip, outLodBias);
        const FLOAT32 clampedOuter = std::isfinite(outerConeRadians)
            ? std::clamp(outerConeRadians, 0.00872665f, 1.53588974f)
            : (35.0f * DEG2RAD);
        const FLOAT32 tanHalf = std::tan(clampedOuter);
        if (tanHalf > 1.0f && outMinMip > 0u)
        {
            const UINT32 coneShift = static_cast<UINT32>(std::ceil(std::log2(tanHalf)));
            outMinMip = (outMinMip > coneShift) ? (outMinMip - coneShift) : 0u;
        }
    }

    // Legacy compatibility helper kept for standalone tests querying perspective pages per axis.
    inline UINT32 ComputeNVSSpotVirtualPagesPerAxis(
        LightShadowResolution resolution,
        FLOAT32               outerConeDegrees,
        UINT32                maxTablePagesPerAxis = kNVSMaxPagesPerAxis) noexcept
    {
        const UINT32 cap       = (maxTablePagesPerAxis > 0u && maxTablePagesPerAxis <= kNVSMaxPagesPerAxis)
            ? maxTablePagesPerAxis
            : kNVSMaxPagesPerAxis;
        const UINT32 basePages = ComputeNVSPerspectivePagesPerAxisFromResolution(resolution, cap);
        const FLOAT32 safeDeg  = std::isfinite(outerConeDegrees)
            ? std::clamp(outerConeDegrees, 0.5f, 88.0f)
            : 45.0f;
        if (safeDeg <= 60.5f)
        {
            return basePages;
        }
        const FLOAT32 tanHalf = std::tan(safeDeg * DEG2RAD);
        UINT32 mult = 1u;
        while (static_cast<FLOAT32>(mult) < tanHalf && (basePages * mult) < cap)
        {
            mult <<= 1u;
        }
        const UINT32 boosted = basePages * mult;
        return (boosted < cap) ? boosted : cap;
    }

    // Power-of-two camera-centered directional clipmap scaling (2x for standard ranges <= 256m,
    // 4x for open-world ranges > 256m). Exact power-of-two ratios ensure clipmap level k+1 Mip 0
    // aligns 1:1 with clipmap level k's internal mip pyramid across every clipmap boundary.
    inline void ComputeNVSDirectionalClipmapScalingFromResolution(
        LightShadowResolution /*resolution*/,
        FLOAT32               maxShadowDistance,
        UINT32                pagesPerAxis,
        FLOAT32&              outBasePageWorldSize,
        FLOAT32&              outLevelScaleFactor) noexcept
    {
        const FLOAT32 effMaxDist  = std::max(0.05f, maxShadowDistance);
        const UINT32  effPages    = (pagesPerAxis > 0u && pagesPerAxis <= kNVSMaxPagesPerAxis) ? pagesPerAxis : kNVSMaxPagesPerAxis;
        const FLOAT32 scaleFactor = (effMaxDist > 256.0f) ? 4.0f : 2.0f;
        const FLOAT32 l0Divisor   = scaleFactor * scaleFactor * scaleFactor;
        const FLOAT32 l0Extent    = (2.0f * effMaxDist) / l0Divisor;
        outBasePageWorldSize      = std::clamp(l0Extent / static_cast<FLOAT32>(effPages), 0.0001f, 4096.0f);
        outLevelScaleFactor       = scaleFactor;
    }

    inline void InitializeNVSClipmapHierarchy(
        NVSClipmapHierarchyState& outHierarchy,
        const Vec3&               lightDirection    = Vec3(0.0f, -1.0f, 0.0f),
        UINT32                    lightId           = 0u,
        FLOAT32                   basePageWorldSize = 2.0f,
        UINT32                    pagesPerAxis      = 64u,
        UINT32                    activeLevelCount  = kNVSMaxClipmapLevels,
        FLOAT32                   maxShadowDistance = 400.0f,
        FLOAT32                   levelScaleFactor  = 2.0f,
        FLOAT32                   nearPlaneOffset   = 0.0f) noexcept
    {
        outHierarchy                  = NVSClipmapHierarchyState{};
        outHierarchy.lightId          = lightId;
        outHierarchy.activeLevelCount = (activeLevelCount > 0u && activeLevelCount <= kNVSMaxClipmapLevels)
            ? activeLevelCount
            : kNVSMaxClipmapLevels;
        outHierarchy.pagesPerAxis      = (pagesPerAxis > 0u && pagesPerAxis <= kNVSMaxPagesPerAxis)
            ? pagesPerAxis
            : 64u;
        outHierarchy.maxShadowDistance = maxShadowDistance;
        outHierarchy.levelScaleFactor  = (std::isfinite(levelScaleFactor) && levelScaleFactor > 1.0f) ? levelScaleFactor : 2.0f;
        outHierarchy.nearPlaneOffset   = std::isfinite(nearPlaneOffset) ? nearPlaneOffset : 0.0f;

        outHierarchy.lightValid = ComputeNVSLightViewMatrices(
            lightDirection,
            outHierarchy.lightRight,
            outHierarchy.lightUp,
            outHierarchy.lightDir,
            outHierarchy.worldToLightView,
            outHierarchy.lightViewToWorld);
        outHierarchy.lightForward = outHierarchy.lightDir;

        const FLOAT32 l0PageSize = (basePageWorldSize > 0.0f) ? basePageWorldSize : 2.0f;
        FLOAT32       curExtent  = l0PageSize * static_cast<FLOAT32>(outHierarchy.pagesPerAxis);
        for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
        {
            NVSClipmapLevelState& ls     = outHierarchy.levels[lvl];
            ls.lightId                   = outHierarchy.lightId;
            ls.clipmapLevel              = lvl;
            ls.pagesPerAxis              = outHierarchy.pagesPerAxis;
            ls.worldExtent               = curExtent;
            ls.pageWorldSize             = curExtent / static_cast<FLOAT32>(outHierarchy.pagesPerAxis);
            ls.texelWorldSize            = ls.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
            ls.lightRight                = outHierarchy.lightRight;
            ls.lightUp                   = outHierarchy.lightUp;
            ls.lightDir                  = outHierarchy.lightDir;
            ls.worldToLightView          = outHierarchy.worldToLightView;
            ls.lightViewToWorld          = outHierarchy.lightViewToWorld;
            ls.snappedOriginPageX        = 0;
            ls.snappedOriginPageY        = 0;
            ls.snappedLightCenterX       = 0.0f;
            ls.snappedLightCenterY       = 0.0f;
            ls.cameraLightZ              = 0.0f;
            ls.receiverMinZ              = -0.5f * ls.worldExtent;
            ls.receiverMaxZ              = +0.5f * ls.worldExtent;
            ls.casterExtrusionDistance   = std::max(256.0f, 2.0f * ls.worldExtent);
            ls.casterDownstreamMargin    = std::max(64.0f, 0.5f * ls.worldExtent);
            ls.casterConservativePadding = 0.5f * ls.pageWorldSize;
            ls.casterMaxZ                = ls.receiverMaxZ + ls.casterDownstreamMargin + ls.casterConservativePadding;
            const FLOAT32 rawCasterMinZ  = ls.receiverMinZ - ls.casterExtrusionDistance - ls.casterConservativePadding;
            ls.casterMinZ                = std::min(rawCasterMinZ + outHierarchy.nearPlaneOffset, ls.casterMaxZ - 1.0f);
            const FLOAT32 zSpan          = std::max(ls.casterMaxZ - ls.casterMinZ, 1e-4f);
            ls.normalBiasWorld           = 1.5f * ls.texelWorldSize;
            ls.depthBiasNdc              = (0.5f * ls.texelWorldSize) / zSpan;
            ls.originSnapHysteresis      = outHierarchy.originSnapHysteresis;
            ls.transitionStartRatio      = outHierarchy.transitionStartRatio;
            ls.initialized               = false;
            ls.depthRangeAnchored        = false;

            outHierarchy.levelWorldExtents[lvl]    = ls.worldExtent;
            outHierarchy.levelPageWorldSizes[lvl]  = ls.pageWorldSize;
            outHierarchy.levelTexelWorldSizes[lvl] = ls.texelWorldSize;
            outHierarchy.levelHalfExtents[lvl]     = 0.5f * ls.worldExtent;
            outHierarchy.snappedOriginPageX[lvl]   = 0;
            outHierarchy.snappedOriginPageY[lvl]   = 0;

            curExtent *= outHierarchy.levelScaleFactor;
        }
    }

    inline void UpdateNVSClipmapHierarchySnappedOrigins(
        NVSClipmapHierarchyState& ioHierarchy,
        const Vec3&               cameraWorldPos,
        NVSOriginShiftResult&     outLevel0Shift,
        NVSOriginShiftResult      outAllShifts[kNVSMaxClipmapLevels] = nullptr) noexcept
    {
        outLevel0Shift = NVSOriginShiftResult{};
        if (ioHierarchy.lightValid)
        {
            ComputeNVSLightViewMatrices(
                ioHierarchy.lightRight,
                ioHierarchy.lightUp,
                ioHierarchy.lightDir,
                ioHierarchy.worldToLightView,
                ioHierarchy.lightViewToWorld);
        }

        const UINT32 count = (ioHierarchy.activeLevelCount <= kNVSMaxClipmapLevels)
            ? ioHierarchy.activeLevelCount
            : kNVSMaxClipmapLevels;
        const FLOAT32 camLightZ = cameraWorldPos.Dot(ioHierarchy.lightDir);

        for (UINT32 lvl = 0u; lvl < count; ++lvl)
        {
            NVSClipmapLevelState& ls = ioHierarchy.levels[lvl];
            ls.lightId               = ioHierarchy.lightId;
            ls.clipmapLevel          = lvl;
            ls.pagesPerAxis          = ioHierarchy.pagesPerAxis;
            ls.lightRight            = ioHierarchy.lightRight;
            ls.lightUp               = ioHierarchy.lightUp;
            ls.lightDir              = ioHierarchy.lightDir;
            ls.worldToLightView      = ioHierarchy.worldToLightView;
            ls.lightViewToWorld      = ioHierarchy.lightViewToWorld;
            if (ioHierarchy.originSnapHysteresis > 0.0f && ls.originSnapHysteresis <= 0.0f)
            {
                ls.originSnapHysteresis = ioHierarchy.originSnapHysteresis;
            }
            if (ls.worldExtent <= 0.0f)
            {
                ls.worldExtent = ioHierarchy.levelWorldExtents[lvl];
            }
            const BOOL           wasInitialized = ls.initialized;
            NVSOriginShiftResult shift          = UpdateNVSClipmapSnappedOrigin(ls, cameraWorldPos);

            // Quantized light-space Z anchoring with hysteresis:
            // Keeps [casterMinZ, casterMaxZ] bit-identical across cached physical shadow pages
            // while camera movement along lightDir remains within the level hysteresis band.
            // When exceeded (or on full window teleport), re-snaps cameraLightZ and marks depthRangeChanged
            // so the runtime invalidates all cached pages in this clipmap level.
            const FLOAT32 zSnapStep          = std::max(8.0f, 0.25f * ls.worldExtent);
            const FLOAT32 zRecenterThreshold = 0.75f * zSnapStep;
            const FLOAT32 snappedCamLightZ   = std::floor((camLightZ / zSnapStep) + 0.5f) * zSnapStep;

            if (!wasInitialized || !ls.depthRangeAnchored)
            {
                ls.cameraLightZ       = snappedCamLightZ;
                ls.depthRangeAnchored = true;
            }
            else if (shift.fullWindowReset || std::fabs(camLightZ - ls.cameraLightZ) > zRecenterThreshold)
            {
                if (std::fabs(snappedCamLightZ - ls.cameraLightZ) > 1e-4f)
                {
                    ls.cameraLightZ                = snappedCamLightZ;
                    shift.depthRangeChanged        = true;
                    shift.expectedInvalidatedPages = ls.pagesPerAxis * ls.pagesPerAxis;
                    shift.expectedPreservedPages   = 0u;
                }
            }

            const FLOAT32 halfExtent = 0.5f * ls.worldExtent;
            ls.receiverMinZ = ls.cameraLightZ - halfExtent;
            ls.receiverMaxZ = ls.cameraLightZ + halfExtent;
            if (ls.casterExtrusionDistance <= 0.0f)
            {
                ls.casterExtrusionDistance = std::max(256.0f, 2.0f * ls.worldExtent);
            }
            if (ls.casterDownstreamMargin <= 0.0f)
            {
                ls.casterDownstreamMargin = std::max(64.0f, 0.5f * ls.worldExtent);
            }
            if (ls.casterConservativePadding <= 0.0f)
            {
                ls.casterConservativePadding = 0.5f * ls.pageWorldSize;
            }
            ls.casterMaxZ = ls.receiverMaxZ + ls.casterDownstreamMargin + ls.casterConservativePadding;
            const FLOAT32 rawCasterMinZ = ls.receiverMinZ - ls.casterExtrusionDistance - ls.casterConservativePadding;
            ls.casterMinZ = std::min(rawCasterMinZ + ioHierarchy.nearPlaneOffset, ls.casterMaxZ - 1.0f);
            const FLOAT32 zSpan = std::max(ls.casterMaxZ - ls.casterMinZ, 1e-4f);
            ls.normalBiasWorld  = 1.5f * ls.texelWorldSize;
            const FLOAT32 absBiasNdc = (0.5f * ls.texelWorldSize) / zSpan;
            ls.depthBiasNdc = (ioHierarchy.depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
                ? -absBiasNdc
                : +absBiasNdc;

            ioHierarchy.snappedOriginPageX[lvl]   = ls.snappedOriginPageX;
            ioHierarchy.snappedOriginPageY[lvl]   = ls.snappedOriginPageY;
            ioHierarchy.levelWorldExtents[lvl]    = ls.worldExtent;
            ioHierarchy.levelPageWorldSizes[lvl]  = ls.pageWorldSize;
            ioHierarchy.levelTexelWorldSizes[lvl] = ls.texelWorldSize;
            ioHierarchy.levelHalfExtents[lvl]     = halfExtent;
            if (lvl == 0u)
            {
                outLevel0Shift = shift;
            }
            if (outAllShifts != nullptr)
            {
                outAllShifts[lvl] = shift;
            }
        }
    }

    inline void UpdateNVSClipmapHierarchySnappedOrigins(
        NVSClipmapHierarchyState& ioHierarchy,
        const Vec3&               cameraWorldPos) noexcept
    {
        NVSOriginShiftResult dummy{};
        UpdateNVSClipmapHierarchySnappedOrigins(ioHierarchy, cameraWorldPos, dummy, nullptr);
    }

    // Deterministic Finest-Level Clipmap Selection (Level 0 -> Level 1 -> Level 2 -> Level 3):
    // Evaluates active clipmap levels in strictly increasing order [0 .. activeLevelCount-1] and
    // selects the highest-resolution (lowest index) level whose snapped toroidal window covers surfaceWorldPos.
    inline BOOL SelectNVSClipmapLevelAndVirtualAddress(
        const NVSClipmapHierarchyState& hierarchy,
        const Vec3&                     cameraWorldPos,
        const Vec3&                     surfaceWorldPos,
        NVSVirtualPageAddress&          outAddress,
        Vec2*                           outIntraPageUV = nullptr,
        INT32*                          outWorldPageX  = nullptr,
        INT32*                          outWorldPageY  = nullptr) noexcept
    {
        outAddress = NVSVirtualPageAddress{};
        if (!hierarchy.lightValid ||
            hierarchy.activeLevelCount == 0u ||
            hierarchy.activeLevelCount > kNVSMaxClipmapLevels ||
            hierarchy.pagesPerAxis == 0u ||
            hierarchy.pagesPerAxis > 64u ||
            hierarchy.lightId >= kNVSMaxSupportedLights)
        {
            return false;
        }

        const FLOAT32 rightLenSq = hierarchy.lightRight.Dot(hierarchy.lightRight);
        const FLOAT32 upLenSq    = hierarchy.lightUp.Dot(hierarchy.lightUp);
        if (rightLenSq <= 1e-6f || upLenSq <= 1e-6f)
        {
            return false;
        }

        if (hierarchy.maxShadowDistance > 0.0f)
        {
            const Vec3    delta  = surfaceWorldPos - cameraWorldPos;
            const FLOAT32 distSq = delta.Dot(delta);
            if (distSq > (hierarchy.maxShadowDistance * hierarchy.maxShadowDistance))
            {
                return false;
            }
        }

        for (UINT32 lvl = 0u; lvl < hierarchy.activeLevelCount; ++lvl)
        {
            NVSClipmapLevelState ls = hierarchy.levels[lvl];
            ls.lightId              = hierarchy.lightId;
            ls.clipmapLevel         = lvl;
            ls.pagesPerAxis         = hierarchy.pagesPerAxis;
            ls.lightRight           = hierarchy.lightRight;
            ls.lightUp              = hierarchy.lightUp;
            ls.lightDir             = hierarchy.lightDir;
            ls.snappedOriginPageX   = hierarchy.snappedOriginPageX[lvl];
            ls.snappedOriginPageY   = hierarchy.snappedOriginPageY[lvl];
            if (hierarchy.levelWorldExtents[lvl] > 0.0f)
            {
                ls.worldExtent = hierarchy.levelWorldExtents[lvl];
            }
            if (ls.worldExtent <= 0.0f || ls.pagesPerAxis == 0u)
            {
                continue;
            }
            ls.pageWorldSize = ls.worldExtent / static_cast<FLOAT32>(ls.pagesPerAxis);
            if (WorldToNVSVirtualAddress(ls, surfaceWorldPos, outAddress, outIntraPageUV, outWorldPageX, outWorldPageY))
            {
                outAddress.lightId      = hierarchy.lightId;
                outAddress.clipmapLevel = lvl;
                return true;
            }
        }
        return false;
    }

    inline BOOL SelectNVSClipmapLevelAndVirtualAddress(
        const Vec3&                     surfaceWorldPos,
        const Vec3&                     cameraWorldPos,
        const NVSClipmapHierarchyState& hierarchy,
        UINT32&                         outSelectedLevel,
        NVSVirtualPageAddress&          outAddress) noexcept
    {
        outSelectedLevel = 0u;
        if (!SelectNVSClipmapLevelAndVirtualAddress(hierarchy, cameraWorldPos, surfaceWorldPos, outAddress))
        {
            return false;
        }
        outSelectedLevel = outAddress.clipmapLevel;
        return true;
    }

    // Reconstructs world-space surface position from a pixel coordinate (px, py) and SceneDepth sample
    // using the exact DirectX NDC -> InvViewProj -> Perspective-Divide chain executed in RuntimeNVSFeedback.nsl.
    inline NVSDepthPixelClassification ReconstructNVSWorldPositionFromDepth(
        UINT32              px,
        UINT32              py,
        UINT32              viewportWidth,
        UINT32              viewportHeight,
        FLOAT32             depthSample,
        const Mat4&         invViewProj,
        NVSDepthCompareMode depthMode,
        FLOAT32             clearDepth,
        FLOAT32             depthEpsilon,
        Vec3&               outWorldPos) noexcept
    {
        outWorldPos = Vec3::Zero();
        if (viewportWidth == 0u || viewportHeight == 0u || px >= viewportWidth || py >= viewportHeight)
        {
            return NVS_PIXEL_REJECT_INVALID;
        }

        if (std::isnan(depthSample) || std::isinf(depthSample) || depthSample < 0.0f || depthSample > 1.0f)
        {
            return NVS_PIXEL_REJECT_INVALID;
        }

        if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
        {
            if (depthSample <= (clearDepth + depthEpsilon))
            {
                return NVS_PIXEL_REJECT_BACKGROUND;
            }
        }
        else
        {
            if (depthSample >= (clearDepth - depthEpsilon))
            {
                return NVS_PIXEL_REJECT_BACKGROUND;
            }
        }

        const FLOAT32 u    = (static_cast<FLOAT32>(px) + 0.5f) / static_cast<FLOAT32>(viewportWidth);
        const FLOAT32 v    = (static_cast<FLOAT32>(py) + 0.5f) / static_cast<FLOAT32>(viewportHeight);
        const FLOAT32 ndcX = u * 2.0f - 1.0f;
        const FLOAT32 ndcY = (1.0f - v) * 2.0f - 1.0f;
        const FLOAT32 ndcZ = depthSample;

        const Vec4 worldH = invViewProj * Vec4(ndcX, ndcY, ndcZ, 1.0f);
        if (std::isnan(worldH.w) || std::isinf(worldH.w) || worldH.w <= 1e-7f)
        {
            return NVS_PIXEL_REJECT_FRUSTUM;
        }

        const FLOAT32 invW = 1.0f / worldH.w;
        const Vec3    wPos(worldH.x * invW, worldH.y * invW, worldH.z * invW);
        if (std::isnan(wPos.x) || std::isinf(wPos.x) ||
            std::isnan(wPos.y) || std::isinf(wPos.y) ||
            std::isnan(wPos.z) || std::isinf(wPos.z))
        {
            return NVS_PIXEL_REJECT_FRUSTUM;
        }

        outWorldPos = wPos;
        return NVS_PIXEL_VALID_SURFACE;
    }

    inline BOOL ReconstructNVSWorldPositionFromDepth(
        UINT32              px,
        UINT32              py,
        UINT32              viewportWidth,
        UINT32              viewportHeight,
        FLOAT32             depthSample,
        const Mat4&         invViewProj,
        Vec3&               outWorldPos,
        NVSDepthCompareMode depthMode    = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
        FLOAT32             clearDepth   = 1.0f,
        FLOAT32             depthEpsilon = 1e-6f) noexcept
    {
        return ReconstructNVSWorldPositionFromDepth(
            px, py, viewportWidth, viewportHeight, depthSample, invViewProj, depthMode, clearDepth, depthEpsilon, outWorldPos) == NVS_PIXEL_VALID_SURFACE;
    }

    // =====================================================================
    //  GPU Screen-Space Feedback Telemetry Record (64 Bytes, 16-Byte Aligned)
    //  Written on the GPU by RuntimeNVSFeedback.nsl.
    // =====================================================================
    struct alignas(16) NVSFeedbackTelemetry
    {
        // Row 0 (Offsets 0..15): Screen-Space Pixel Classification
        UINT32 examinedPixelCount           = 0u;
        UINT32 validDepthPixelCount         = 0u;
        UINT32 backgroundRejectedCount      = 0u;
        UINT32 invalidDepthRejectedCount    = 0u;

        // Row 1 (Offsets 16..31): Reconstruction & Candidate Generation
        UINT32 frustumRejectedCount         = 0u;
        UINT32 outOfRangeRejectedCount      = 0u;
        UINT32 acceptedSurfacePixelCount    = 0u;
        UINT32 totalPageCandidateCount      = 0u;

        // Row 2 (Offsets 32..47): Tile Deduplication & Global Compaction
        UINT32 tileDeduplicatedCount        = 0u;
        UINT32 globalDuplicateCount         = 0u;
        UINT32 uniquePageRequestCount       = 0u;
        UINT32 compactedRequestCount        = 0u;

        // Row 3 (Offsets 48..63): Capacity Overflow & Per-Clipmap-Level Breakdown
        UINT32 overflowRequestCount         = 0u;
        UINT32 level0RequestCount           = 0u;
        UINT32 level1RequestCount           = 0u;
        UINT32 level2And3RequestCountPacked = 0u; // Low 16 bits = Level 2 count, High 16 bits = Level 3 count

        constexpr UINT32 GetLevel2RequestCount() const noexcept
        {
            return level2And3RequestCountPacked & 0xFFFFu;
        }

        constexpr UINT32 GetLevel3RequestCount() const noexcept
        {
            return (level2And3RequestCountPacked >> 16u) & 0xFFFFu;
        }
    };

    static_assert(sizeof(NVSFeedbackTelemetry) == 64, "NVSFeedbackTelemetry must be 64 bytes (4 x 16B).");
    static_assert(alignof(NVSFeedbackTelemetry) == 16, "NVSFeedbackTelemetry must be 16-byte aligned.");
    static_assert(offsetof(NVSFeedbackTelemetry, examinedPixelCount)        == 0,  "examinedPixelCount offset mismatch");
    static_assert(offsetof(NVSFeedbackTelemetry, frustumRejectedCount)      == 16, "frustumRejectedCount offset mismatch");
    static_assert(offsetof(NVSFeedbackTelemetry, tileDeduplicatedCount)     == 32, "tileDeduplicatedCount offset mismatch");
    static_assert(offsetof(NVSFeedbackTelemetry, overflowRequestCount)      == 48, "overflowRequestCount offset mismatch");

    // =====================================================================
    //  NVS Feedback Manager Configuration & Pass Execution Descriptor
    // =====================================================================
    struct NVSFeedbackManagerConfig
    {
        UINT32 maxRequestBudget            = 4096u;
        UINT32 maxCompactedRequestsCapacity = 8192u;
        UINT32 maxCameraSlots              = 8u;
    };

    struct NVSFeedbackPassDesc
    {
        Texture*                 sceneDepthTexture        = nullptr;
        Mat4                     invViewProj              = Mat4::Identity();
        Vec3                     cameraPosition           = Vec3::Zero();
        Vec3                     cameraWorldPos           = Vec3::Zero();
        UINT32                   viewportOffsetX          = 0u;
        UINT32                   viewportOffsetY          = 0u;
        UINT32                   viewportWidth            = 0u;
        UINT32                   viewportHeight           = 0u;
        NVSDepthCompareMode      depthCompareMode         = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        NVSDepthCompareMode      depthMode                = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        FLOAT32                  clearDepth               = 1.0f;
        FLOAT32                  depthEpsilon             = 1e-6f;
        NVSFootprintMode         footprintMode            = NVS_FOOTPRINT_EXACT_PAGE;
        FLOAT32                  footprintBorderMargin    = 0.03125f; // Default 4/128 page border margin when NVS_FOOTPRINT_BORDER_NEIGHBORS is active
        UINT32                   maxRequestBudget         = 0u;       // 0 = use manager/residency capacity
        BOOL                     disableGroupSharedDedup  = false;    // True = pure per-pixel atomic baseline comparison mode
        BOOL                     resetBeforeMark          = true;     // True = execute Mode 0 reset before Mode 1 mark
        BOOL                     executeCompaction        = true;     // True = execute Mode 2 priority compaction after Mode 1 mark
        BOOL                     executeResidencyAndClear   = false;    // True = chain GPU Residency + Selective Layered Depth Clear without CPU readback
        BOOL                     executePageProjection      = true;     // chain GPU Per-Page Projection metadata generation after residency
        BOOL                     executeCasterCulling       = true;     // chain GPU Shadow Caster Culling & Worklist generation after Page Projection
        BOOL                     executeShadowRasterization = false;    // chain GPU Indirect Command Generation + Physical Page Rasterization + Page Validity Commit
        BOOL                     commitRenderedShadowPages  = true;     // chain GPU Mode 4 page validity commit after shadow rasterization
        BOOL                     allowShadowInstancedMerge  = true;     // merge identical instanced candidates per shadow page
        BOOL                     applyShadowShaderBias      = true;     // apply normal & depth bias in shadow VS
        FLOAT32                  shadowTimeSeconds          = 0.0f;     // time in seconds for custom vertex deformation
        GPUBuffer*               gpuSceneBuffer             = nullptr;  // Optional explicit GPU Scene buffer for ExecuteFeedbackPass
        UINT32                   gpuSceneSlotCount          = 0u;       // Optional explicit slot count (0 = use buffer/scene high-water mark)
        UINT32                   casterStaticBatchPolicy    = 0u;       // 0 = PREFER_INDIVIDUAL_MEMBERS, 1 = PREFER_COMBINED_UMBRELLA
        FLOAT32                  minCasterProjectedTexels   = 0.0f;     // Small-caster projection threshold in shadow texels
        BOOL                     skipValidCachedPagesInCull = true;     // True = skip VALID_SHADOW_DATA pages unless NeedsRaster is set
        BOOL                     multiLightViewMode         = false;    // evaluate active views [0 .. activeViewTableCount-1] from InClipmapLevels
        UINT32                   activeViewTableCount       = 0u;       // Active view count in InClipmapLevels (unbounded dynamic capacity)
        GPUBuffer*               lightViewTableBuffer       = nullptr;  // Optional explicit InClipmapLevels GPU buffer override
        Vec4                     screenPixelWorldScale      = Vec4::Zero(); // Optional explicit (tanHalfFovX*2/W, tanHalfFovY*2/H, isOrtho, orthoPixelWorldSize)
        NVSResidencyPassDesc     residencyPassDesc{};
        NVSClipmapHierarchyState clipmapHierarchy{};
    };

    // =====================================================================
    //  Directional Light Clipmaps, Stable Projection &
    //              Per-Page Rendering Data
    // =====================================================================
    static constexpr UINT32 kNVSMaxClipmapLights         = 4u;
    static constexpr UINT32 kNVSClipmapTableMaxEntries   = kNVSMaxClipmapLights * kNVSMaxClipmapLevels; // 16 entries (4 lights x 4 levels)
    static constexpr UINT32 kNVSProjectionFrameSlots     = 3u; // Per-camera 3-frame-in-flight buffer ring

    enum NVSProjectionDispatchMode : UINT32
    {
        NVS_PROJECTION_DISPATCH_RESET                  = 0u,
        NVS_PROJECTION_DISPATCH_BUILD_PAGE_PROJECTIONS = 1u
    };

    enum NVSClipmapLevelGPUFlags : UINT32
    {
        NVS_CLIPMAP_LEVEL_FLAG_NONE                       = 0u,
        NVS_CLIPMAP_LEVEL_FLAG_VALID                      = 1u << 0u,
        NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z                 = 1u << 1u,
        NVS_CLIPMAP_LEVEL_FLAG_PERSPECTIVE                = 1u << 2u, // Spot cone or Point cube-face perspective view
        NVS_CLIPMAP_LEVEL_FLAG_SPOT_CONE                  = 1u << 3u, // Radial spot cone boundary clipping enabled
        NVS_CLIPMAP_LEVEL_FLAG_RADIAL_SELECT              = 1u << 4u, // Camera-centered 3D radial clipmap level selection enabled
        NVS_CLIPMAP_LEVEL_FLAG_POINT_CUBE                 = 1u << 5u, // Point light 6-face omnidirectional cubemap view
        NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_DEPTH_BIAS  = 1u << 6u, // Caster shadow shader already applied raster depth bias
        NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_NORMAL_BIAS = 1u << 7u, // Caster shadow shader already applied raster normal bias
        NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS        = 1u << 8u, // Section 8A dynamic texel-relative bias contract active
        NVS_CLIPMAP_LEVEL_CUBE_FACE_SHIFT                 = 9u,
        NVS_CLIPMAP_LEVEL_CUBE_FACE_MASK                  = 0x00000E00u,
        NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_SHIFT       = 12u,
        NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_MASK        = 0x0000F000u,
        NVS_CLIPMAP_LEVEL_LOGICAL_LIGHT_SHIFT             = 16u,
        NVS_CLIPMAP_LEVEL_LOGICAL_LIGHT_MASK              = 0x007F0000u,
        // Per-light unified 16K detail tier packing (bits 16..23 when HAS_DETAIL_TIER bit 23 is set):
        //   bits [16..18]: minMip in [0..7]
        //   bits [19..22]: lodBiasEncoded in [0..15] -> lodBias = (encoded * 0.5f) - 1.5f (range [-1.5 .. +6.0])
        //   bit  [23]:     NVS_CLIPMAP_LEVEL_FLAG_HAS_DETAIL_TIER
        NVS_CLIPMAP_LEVEL_MIN_MIP_SHIFT                   = 16u,
        NVS_CLIPMAP_LEVEL_MIN_MIP_MASK                    = 0x00070000u,
        NVS_CLIPMAP_LEVEL_LOD_BIAS_SHIFT                  = 19u,
        NVS_CLIPMAP_LEVEL_LOD_BIAS_MASK                   = 0x00780000u,
        NVS_CLIPMAP_LEVEL_FLAG_HAS_DETAIL_TIER            = 1u << 23u,
        NVS_CLIPMAP_LEVEL_VIEW_GEN_SHIFT                  = 24u,
        NVS_CLIPMAP_LEVEL_VIEW_GEN_MASK                   = 0xFF000000u
    };

    inline UINT32 PackNVSViewDetailTierFlags(UINT32 minMip, FLOAT32 lodBias) noexcept
    {
        const UINT32  safeMinMip = (minMip <= 7u) ? minMip : 7u;
        const FLOAT32 clampedB   = std::isfinite(lodBias) ? std::clamp(lodBias, -1.5f, 6.0f) : 0.0f;
        const UINT32  biasEnc    = static_cast<UINT32>(std::clamp(static_cast<INT32>(std::lround((clampedB + 1.5f) * 2.0f)), 0, 15));
        return NVS_CLIPMAP_LEVEL_FLAG_HAS_DETAIL_TIER |
               ((safeMinMip & 0x7u) << NVS_CLIPMAP_LEVEL_MIN_MIP_SHIFT) |
               ((biasEnc & 0xFu) << NVS_CLIPMAP_LEVEL_LOD_BIAS_SHIFT);
    }

    inline constexpr UINT32 ExtractNVSViewMinMip(UINT32 viewFlags) noexcept
    {
        return ((viewFlags & NVS_CLIPMAP_LEVEL_FLAG_HAS_DETAIL_TIER) != 0u)
            ? ((viewFlags & NVS_CLIPMAP_LEVEL_MIN_MIP_MASK) >> NVS_CLIPMAP_LEVEL_MIN_MIP_SHIFT)
            : 0u;
    }

    inline constexpr FLOAT32 ExtractNVSViewLodBias(UINT32 viewFlags) noexcept
    {
        return ((viewFlags & NVS_CLIPMAP_LEVEL_FLAG_HAS_DETAIL_TIER) != 0u)
            ? (static_cast<FLOAT32>((viewFlags & NVS_CLIPMAP_LEVEL_LOD_BIAS_MASK) >> NVS_CLIPMAP_LEVEL_LOD_BIAS_SHIFT) * 0.5f - 1.5f)
            : 0.0f;
    }

    inline constexpr UINT32 ExtractNVSViewMipCount(UINT32 viewFlags) noexcept
    {
        const UINT32 m1 = (viewFlags & NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_MASK) >> NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_SHIFT;
        const UINT32 cnt = m1 + 1u;
        return (cnt <= kNVSSparseMaxMipLevels) ? cnt : kNVSSparseMaxMipLevels;
    }

    inline constexpr UINT32 ExtractNVSViewCubeFaceIndex(UINT32 viewFlags) noexcept
    {
        return (viewFlags & NVS_CLIPMAP_LEVEL_CUBE_FACE_MASK) >> NVS_CLIPMAP_LEVEL_CUBE_FACE_SHIFT;
    }

    inline constexpr UINT32 ExtractNVSViewGeneration(UINT32 viewFlags) noexcept
    {
        const UINT32 g = (viewFlags & NVS_CLIPMAP_LEVEL_VIEW_GEN_MASK) >> NVS_CLIPMAP_LEVEL_VIEW_GEN_SHIFT;
        return (g == 0u) ? 1u : g;
    }

    enum NVSPageRenderFlags : UINT32
    {
        NVS_PAGE_RENDER_FLAG_NONE                  = 0u,
        NVS_PAGE_RENDER_FLAG_VALID_RESIDENT        = 1u << 0u,
        NVS_PAGE_RENDER_FLAG_NEEDS_CASTER_RENDER   = 1u << 1u, // Physical state == CLEARED_NOT_RENDERED (2)
        NVS_PAGE_RENDER_FLAG_CACHED_VALID_SHADOW   = 1u << 2u, // Physical state == RESIDENT_VALID (3), skip caster rasterization
        NVS_PAGE_RENDER_FLAG_NEWLY_ALLOCATED       = 1u << 3u,
        NVS_PAGE_RENDER_FLAG_ALLOCATED_NOT_CLEARED = 1u << 4u, // Physical state == ALLOCATED_NOT_CLEARED (1)
        NVS_PAGE_RENDER_FLAG_REVERSED_Z            = 1u << 5u,
        NVS_PAGE_RENDER_FLAG_PERSPECTIVE           = 1u << 6u,
        NVS_PAGE_RENDER_FLAG_EXTENDED_VIEW         = 1u << 7u,
        NVS_PAGE_RENDER_FLAG_MIP_LEVEL_SHIFT       = 8u,
        NVS_PAGE_RENDER_FLAG_MIP_LEVEL_MASK        = 0x00000F00u
    };

    // =====================================================================
    //  GPU Clipmap Level Table Entry (256 Bytes = 16 x 16B, 16-Byte Aligned)
    //  Indexed on the GPU by (lightId * 4 + clipmapLevel) in [0 .. 15].
    //  Matches NSL StructuredBuffer<NVSClipmapLevelGPUData> 1:1.
    // =====================================================================
    struct alignas(16) NVSClipmapLevelGPUData
    {
        // Row 0 (Offsets 0..15)
        UINT32  lightId               = 0u;
        UINT32  clipmapLevel          = 0u;
        UINT32  pagesPerAxis          = 64u;
        UINT32  flags                 = NVS_CLIPMAP_LEVEL_FLAG_NONE;

        // Row 1 (Offsets 16..31)
        INT32   snappedOriginPageX    = 0;
        INT32   snappedOriginPageY    = 0;
        FLOAT32 worldExtent           = 128.0f;
        FLOAT32 pageWorldSize         = 2.0f;

        // Row 2 (Offsets 32..47)
        FLOAT32 texelWorldSize        = 2.0f / 128.0f;
        FLOAT32 receiverMinZ          = -64.0f;
        FLOAT32 receiverMaxZ          = 64.0f;
        FLOAT32 casterMinZ            = -320.0f;

        // Row 3 (Offsets 48..63)
        FLOAT32 casterMaxZ            = 80.0f;
        FLOAT32 depthBiasNdc          = 0.0f;
        FLOAT32 normalBiasWorld       = 0.0f;
        FLOAT32 cameraLightZ          = 0.0f;

        // Row 4 (Offsets 64..79): xyz = lightRight, w = snappedLightCenterX
        Vec4    lightRightAndCenterX  = Vec4(1.0f, 0.0f, 0.0f, 0.0f);
        // Row 5 (Offsets 80..95): xyz = lightUp, w = snappedLightCenterY
        Vec4    lightUpAndCenterY     = Vec4(0.0f, 1.0f, 0.0f, 0.0f);
        // Row 6 (Offsets 96..111): xyz = lightDir, w = casterExtrusionDistance
        Vec4    lightDirAndExtrusion  = Vec4(0.0f, 0.0f, 1.0f, 256.0f);
        // Row 7 (Offsets 112..127): x = transitionStartRatio (or cosOuterCone), y = transitionBandWidth (or cosInnerCone), z = casterDownstreamMargin, w = casterConservativePadding
        Vec4    transitionAndPadding  = Vec4(0.85f, 0.15f, 16.0f, 0.5f);

        // Rows 8..11 (Offsets 128..191): worldToLightView (64B)
        Mat4    worldToLightView      = Mat4::Identity();
        // Rows 12..15 (Offsets 192..255): lightViewToWorld (64B; col 3 xyz = lightWorldPos for perspective views)
        Mat4    lightViewToWorld      = Mat4::Identity();
    };

    static_assert(sizeof(NVSClipmapLevelGPUData) == 256, "NVSClipmapLevelGPUData must be exactly 256 bytes (16 x 16B).");
    static_assert(alignof(NVSClipmapLevelGPUData) == 16, "NVSClipmapLevelGPUData must be 16-byte aligned.");
    static_assert(offsetof(NVSClipmapLevelGPUData, lightId)              == 0,   "lightId offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, snappedOriginPageX)   == 16,  "snappedOriginPageX offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, texelWorldSize)       == 32,  "texelWorldSize offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, casterMaxZ)           == 48,  "casterMaxZ offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, lightRightAndCenterX) == 64,  "lightRightAndCenterX offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, lightUpAndCenterY)    == 80,  "lightUpAndCenterY offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, lightDirAndExtrusion) == 96,  "lightDirAndExtrusion offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, transitionAndPadding) == 112, "transitionAndPadding offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, worldToLightView)     == 128, "worldToLightView offset mismatch");
    static_assert(offsetof(NVSClipmapLevelGPUData, lightViewToWorld)     == 192, "lightViewToWorld offset mismatch");

    // =====================================================================
    //  GPU Per-Page Render & Projection Metadata (336 Bytes = 21 x 16B, 16-Byte Aligned)
    //  Indexed on the GPU by physicalPageIndex in [0 .. physicalPoolCapacity - 1].
    //  Written on the GPU by RuntimeNVSPageProjection.nsl with zero CPU readback.
    // =====================================================================
    struct alignas(16) NVSPageRenderData
    {
        // Row 0 (Offsets 0..15): Physical & Virtual Linkage
        UINT32  physicalPageIndex     = kNVSInvalidPhysicalPage;
        UINT32  virtualFlatIndex      = kNVSInvalidVirtualPage;
        UINT32  virtualAddressPacked  = 0u;
        UINT32  renderFlags           = NVS_PAGE_RENDER_FLAG_NONE;

        // Row 1 (Offsets 16..31): Unpacked Virtual Address
        UINT32  lightId               = 0u;
        UINT32  clipmapLevel          = 0u;
        UINT32  virtualPageX          = 0u;
        UINT32  virtualPageY          = 0u;

        // Row 2 (Offsets 32..47): World Page Identity & Generation
        INT32   worldPageX            = 0;
        INT32   worldPageY            = 0;
        UINT32  physicalGeneration    = 0u;
        UINT32  lastUsedFrame         = 0u;

        // Row 3 (Offsets 48..63): Light-Space XY / Tangent Bounds
        FLOAT32 lightMinX             = 0.0f;
        FLOAT32 lightMinY             = 0.0f;
        FLOAT32 lightMaxX             = 0.0f;
        FLOAT32 lightMaxY             = 0.0f;

        // Row 4 (Offsets 64..79): Receiver & Caster Light-Space Z Ranges
        FLOAT32 receiverMinZ          = 0.0f;
        FLOAT32 receiverMaxZ          = 0.0f;
        FLOAT32 casterMinZ            = 0.0f;
        FLOAT32 casterMaxZ            = 0.0f;

        // Row 5 (Offsets 80..95): World-Space Center & Page Size
        Vec3    worldCenter           = Vec3::Zero();
        FLOAT32 pageWorldSize         = 0.0f;

        // Row 6 (Offsets 96..111): World-Space AABB Min & Texel World Size
        Vec3    worldBoundsMin        = Vec3::Zero();
        FLOAT32 texelWorldSize        = 0.0f;

        // Row 7 (Offsets 112..127): World-Space AABB Max & Shadow Volume Bounding Radius
        Vec3    worldBoundsMax        = Vec3::Zero();
        FLOAT32 shadowVolumeRadius    = 0.0f;

        // Row 8 (Offsets 128..143): Bias & Physical Slice Viewport Dimensions (128 x 128)
        FLOAT32 depthBiasNdc          = 0.0f;
        FLOAT32 normalBiasWorld       = 0.0f;
        UINT32  viewportWidth         = kNVSPageTexelSize;
        UINT32  viewportHeight        = kNVSPageTexelSize;

        // Rows 9..12 (Offsets 144..207): Combined World -> Page Clip Matrix (64B)
        Mat4    worldToPageClipMatrix = Mat4::Identity();

        // Rows 13..16 (Offsets 208..271): Per-Page Light View Matrix (64B)
        Mat4    lightViewMatrix       = Mat4::Identity();

        // Rows 17..20 (Offsets 272..335): Per-Page Projection Matrix (64B)
        Mat4    pageProjMatrix        = Mat4::Identity();
    };

    static_assert(sizeof(NVSPageRenderData) == 336, "NVSPageRenderData must be exactly 336 bytes (21 x 16B).");
    static_assert(alignof(NVSPageRenderData) == 16, "NVSPageRenderData must be 16-byte aligned.");
    static_assert(offsetof(NVSPageRenderData, physicalPageIndex)     == 0,   "physicalPageIndex offset mismatch");
    static_assert(offsetof(NVSPageRenderData, lightId)               == 16,  "lightId offset mismatch");
    static_assert(offsetof(NVSPageRenderData, worldPageX)            == 32,  "worldPageX offset mismatch");
    static_assert(offsetof(NVSPageRenderData, lightMinX)             == 48,  "lightMinX offset mismatch");
    static_assert(offsetof(NVSPageRenderData, receiverMinZ)          == 64,  "receiverMinZ offset mismatch");
    static_assert(offsetof(NVSPageRenderData, worldCenter)           == 80,  "worldCenter offset mismatch");
    static_assert(offsetof(NVSPageRenderData, pageWorldSize)         == 92,  "pageWorldSize offset mismatch");
    static_assert(offsetof(NVSPageRenderData, worldBoundsMin)        == 96,  "worldBoundsMin offset mismatch");
    static_assert(offsetof(NVSPageRenderData, texelWorldSize)        == 108, "texelWorldSize offset mismatch");
    static_assert(offsetof(NVSPageRenderData, worldBoundsMax)        == 112, "worldBoundsMax offset mismatch");
    static_assert(offsetof(NVSPageRenderData, shadowVolumeRadius)    == 124, "shadowVolumeRadius offset mismatch");
    static_assert(offsetof(NVSPageRenderData, depthBiasNdc)          == 128, "depthBiasNdc offset mismatch");
    static_assert(offsetof(NVSPageRenderData, worldToPageClipMatrix) == 144, "worldToPageClipMatrix offset mismatch");
    static_assert(offsetof(NVSPageRenderData, lightViewMatrix)       == 208, "lightViewMatrix offset mismatch");
    static_assert(offsetof(NVSPageRenderData, pageProjMatrix)        == 272, "pageProjMatrix offset mismatch");

    // =====================================================================
    //  GPU Page Projection Telemetry Record (64 Bytes = 4 x 16B, 16-Byte Aligned)
    //  Written on the GPU by RuntimeNVSPageProjection.nsl.
    // =====================================================================
    struct alignas(16) NVSPageProjectionTelemetry
    {
        // Row 0 (Offsets 0..15): Physical Slot Classification
        UINT32 examinedPhysicalSlots        = 0u;
        UINT32 validResidentPageCount       = 0u; // CLEARED_NOT_RENDERED + RESIDENT_VALID
        UINT32 activeRenderPageCount        = 0u; // CLEARED_NOT_RENDERED (requires shadow caster render)
        UINT32 cachedValidPageCount         = 0u; // RESIDENT_VALID (cached valid shadow data, skip render)

        // Row 1 (Offsets 16..31): Unready, Free, Rejected & Multi-Light Breakdown
        UINT32 allocatedNotClearedCount     = 0u; // ALLOCATED_NOT_CLEARED (not yet cleared, excluded from active render)
        UINT32 freePhysicalSlotCount        = 0u; // FREE physical pages
        UINT32 invalidMetadataRejectedCount = 0u; // Stale/corrupted linkage or invalid clipmap level
        UINT32 multiLightActivePageCount    = 0u; // Active render pages belonging to lightId > 0

        // Row 2 (Offsets 32..47): Per-Clipmap-Level Active Render Page Counts (NEEDS_CASTER_RENDER)
        UINT32 level0ActiveRenderCount      = 0u;
        UINT32 level1ActiveRenderCount      = 0u;
        UINT32 level2ActiveRenderCount      = 0u;
        UINT32 level3ActiveRenderCount      = 0u;

        // Row 3 (Offsets 48..63): Per-Clipmap-Level Total Resident Page Counts (CLEARED + VALID)
        UINT32 level0ResidentCount          = 0u;
        UINT32 level1ResidentCount          = 0u;
        UINT32 level2ResidentCount          = 0u;
        UINT32 level3ResidentCount          = 0u;
    };

    static_assert(sizeof(NVSPageProjectionTelemetry) == 64, "NVSPageProjectionTelemetry must be exactly 64 bytes (4 x 16B).");
    static_assert(alignof(NVSPageProjectionTelemetry) == 16, "NVSPageProjectionTelemetry must be 16-byte aligned.");
    static_assert(offsetof(NVSPageProjectionTelemetry, examinedPhysicalSlots)    == 0,  "examinedPhysicalSlots offset mismatch");
    static_assert(offsetof(NVSPageProjectionTelemetry, allocatedNotClearedCount) == 16, "allocatedNotClearedCount offset mismatch");
    static_assert(offsetof(NVSPageProjectionTelemetry, level0ActiveRenderCount)  == 32, "level0ActiveRenderCount offset mismatch");
    static_assert(offsetof(NVSPageProjectionTelemetry, level0ResidentCount)      == 48, "level0ResidentCount offset mismatch");

    struct NVSPageProjectionPassDesc
    {
        UINT32              currentFrame         = 1u;
        UINT32              activeLightCount     = 1u;
        NVSDepthCompareMode depthMode            = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        BOOL                resetBeforeBuild     = true;
        BOOL                uploadClipmapTable   = true;
    };

    // =====================================================================
    //  Depth Precision & Bias Metrics (D32_FLOAT Orthographic Analysis)
    // =====================================================================
    struct NVSDepthPrecisionMetrics
    {
        FLOAT32 pageWorldSize                = 2.0f;
        FLOAT32 texelWorldSize               = 2.0f / 128.0f;
        FLOAT32 casterDepthSpan              = 400.0f;
        FLOAT32 ndcDepthPerWorldMeter        = 1.0f / 400.0f;
        FLOAT32 worldMetersPerUlpNear01      = 0.0f; // At NDC z = 0.1
        FLOAT32 worldMetersPerUlpMid05       = 0.0f; // At NDC z = 0.5
        FLOAT32 worldMetersPerUlpFar09       = 0.0f; // At NDC z = 0.9
        FLOAT32 recommendedConstantBiasWorld = 0.0f; // 0.5 * texelWorldSize
        FLOAT32 recommendedNormalBiasWorld   = 0.0f; // 1.5 * texelWorldSize
        FLOAT32 recommendedDepthBiasNdc      = 0.0f; // Signed NDC depth bias
    };

    inline NVSDepthPrecisionMetrics ComputeNVSDepthPrecisionAndBiasMetrics(
        FLOAT32             pageWorldSize,
        FLOAT32             casterMinZ,
        FLOAT32             casterMaxZ,
        NVSDepthCompareMode depthMode = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE) noexcept
    {
        NVSDepthPrecisionMetrics m{};
        m.pageWorldSize         = std::max(pageWorldSize, 1e-4f);
        m.texelWorldSize        = m.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        m.casterDepthSpan       = std::max(casterMaxZ - casterMinZ, 1e-4f);
        m.ndcDepthPerWorldMeter = 1.0f / m.casterDepthSpan;

        const FLOAT32 ulp01          = std::nextafter(0.1f, 2.0f) - 0.1f;
        const FLOAT32 ulp05          = std::nextafter(0.5f, 2.0f) - 0.5f;
        const FLOAT32 ulp09          = std::nextafter(0.9f, 2.0f) - 0.9f;
        m.worldMetersPerUlpNear01    = ulp01 * m.casterDepthSpan;
        m.worldMetersPerUlpMid05     = ulp05 * m.casterDepthSpan;
        m.worldMetersPerUlpFar09     = ulp09 * m.casterDepthSpan;

        m.recommendedConstantBiasWorld = 0.5f * m.texelWorldSize;
        m.recommendedNormalBiasWorld   = 1.5f * m.texelWorldSize;
        const FLOAT32 absBiasNdc       = m.recommendedConstantBiasWorld * m.ndcDepthPerWorldMeter;
        m.recommendedDepthBiasNdc      = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
            ? -absBiasNdc
            : +absBiasNdc;
        return m;
    }

    inline void UpdateNVSClipmapLevelDepthRanges(
        NVSClipmapLevelState& levelState,
        const Vec3&           cameraWorldPos,
        FLOAT32               casterExtrusionDistance   = 0.0f,
        FLOAT32               casterDownstreamMargin    = 0.0f,
        FLOAT32               casterConservativePadding = 0.0f,
        NVSDepthCompareMode   depthMode                 = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE) noexcept
    {
        if (levelState.pagesPerAxis > 0u && levelState.worldExtent > 0.0f)
        {
            levelState.pageWorldSize  = levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis);
            levelState.texelWorldSize = levelState.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        }

        levelState.cameraLightZ       = cameraWorldPos.Dot(levelState.lightDir);
        levelState.depthRangeAnchored = true;
        const FLOAT32 halfExt         = 0.5f * std::max(levelState.worldExtent, 1e-3f);
        levelState.receiverMinZ = levelState.cameraLightZ - halfExt;
        levelState.receiverMaxZ = levelState.cameraLightZ + halfExt;

        if (casterExtrusionDistance > 0.0f)
        {
            levelState.casterExtrusionDistance = casterExtrusionDistance;
        }
        else if (levelState.casterExtrusionDistance <= 0.0f)
        {
            levelState.casterExtrusionDistance = std::max(256.0f, 2.0f * levelState.worldExtent);
        }

        if (casterDownstreamMargin > 0.0f)
        {
            levelState.casterDownstreamMargin = casterDownstreamMargin;
        }
        else if (levelState.casterDownstreamMargin <= 0.0f)
        {
            levelState.casterDownstreamMargin = std::max(16.0f, 0.25f * levelState.worldExtent);
        }

        if (casterConservativePadding > 0.0f)
        {
            levelState.casterConservativePadding = casterConservativePadding;
        }
        else if (levelState.casterConservativePadding <= 0.0f)
        {
            levelState.casterConservativePadding = 0.5f * levelState.pageWorldSize;
        }

        levelState.casterMinZ = levelState.receiverMinZ - levelState.casterExtrusionDistance - levelState.casterConservativePadding;
        levelState.casterMaxZ = levelState.receiverMaxZ + levelState.casterDownstreamMargin + levelState.casterConservativePadding;

        const NVSDepthPrecisionMetrics metrics = ComputeNVSDepthPrecisionAndBiasMetrics(
            levelState.pageWorldSize,
            levelState.casterMinZ,
            levelState.casterMaxZ,
            depthMode);
        levelState.depthBiasNdc    = metrics.recommendedDepthBiasNdc;
        levelState.normalBiasWorld = metrics.recommendedNormalBiasWorld;
    }

    // Tightens [casterMinZ, casterMaxZ] using observed scene shadow caster light-space Z bounds
    // while guaranteeing full [receiverMinZ, receiverMaxZ] containment and non-degenerate depth span.
    inline BOOL TightenNVSClipmapCasterDepthRange(
        NVSClipmapLevelState& levelState,
        FLOAT32               observedCasterMinZ,
        FLOAT32               observedCasterMaxZ,
        NVSDepthCompareMode   depthMode = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE) noexcept
    {
        if (std::isnan(observedCasterMinZ) || std::isinf(observedCasterMinZ) ||
            std::isnan(observedCasterMaxZ) || std::isinf(observedCasterMaxZ) ||
            observedCasterMaxZ < observedCasterMinZ)
        {
            return false;
        }

        const FLOAT32 pad        = std::max(levelState.casterConservativePadding, 0.25f);
        const FLOAT32 clampedMin = std::max(levelState.casterMinZ, std::min(levelState.receiverMinZ, observedCasterMinZ - pad));
        const FLOAT32 clampedMax = std::min(levelState.casterMaxZ, std::max(levelState.receiverMaxZ, observedCasterMaxZ + pad));
        if ((clampedMax - clampedMin) <= 1e-3f)
        {
            return false;
        }

        levelState.casterMinZ = clampedMin;
        levelState.casterMaxZ = clampedMax;
        const NVSDepthPrecisionMetrics metrics = ComputeNVSDepthPrecisionAndBiasMetrics(
            levelState.pageWorldSize,
            levelState.casterMinZ,
            levelState.casterMaxZ,
            depthMode);
        levelState.depthBiasNdc    = metrics.recommendedDepthBiasNdc;
        levelState.normalBiasWorld = metrics.recommendedNormalBiasWorld;
        return true;
    }

    // Complete 1-step Directional Light Clipmap Hierarchy Computation (Section 3.2):
    // Builds orthonormal light basis + view matrices, 2^L scaled clipmap levels, snapped origins,
    // receiver/caster depth ranges, and depth/normal bias metrics.
    inline BOOL ComputeNVSDirectionalClipmapHierarchy(
        NVSClipmapHierarchyState& outHierarchy,
        const Vec3&               cameraWorldPos,
        const Vec3&               lightDirection       = Vec3(0.0f, -1.0f, 0.0f),
        UINT32                    lightId              = 0u,
        FLOAT32                   basePageWorldSize    = 2.0f,
        UINT32                    pagesPerAxis         = 64u,
        UINT32                    activeLevelCount     = kNVSMaxClipmapLevels,
        FLOAT32                   maxShadowDistance    = 400.0f,
        NVSDepthCompareMode       depthMode            = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
        FLOAT32                   originSnapHysteresis = 0.0f,
        NVSOriginShiftResult      outAllShifts[kNVSMaxClipmapLevels] = nullptr,
        FLOAT32                   levelScaleFactor     = 2.0f,
        FLOAT32                   nearPlaneOffset      = 0.0f) noexcept
    {
        InitializeNVSClipmapHierarchy(
            outHierarchy,
            lightDirection,
            lightId,
            basePageWorldSize,
            pagesPerAxis,
            activeLevelCount,
            maxShadowDistance,
            levelScaleFactor,
            nearPlaneOffset);
        outHierarchy.depthMode            = depthMode;
        outHierarchy.originSnapHysteresis = originSnapHysteresis;
        for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
        {
            outHierarchy.levels[lvl].originSnapHysteresis = originSnapHysteresis;
        }
        if (!outHierarchy.lightValid)
        {
            return false;
        }
        NVSOriginShiftResult l0Shift{};
        UpdateNVSClipmapHierarchySnappedOrigins(outHierarchy, cameraWorldPos, l0Shift, outAllShifts);
        return true;
    }

    // =====================================================================
    //  Virtual Page World-Space & Light-Space Bounds Contract
    // =====================================================================
    struct NVSPageWorldBounds
    {
        UINT32  lightId                 = 0u;
        UINT32  clipmapLevel            = 0u;
        UINT32  virtualPageX            = 0u;
        UINT32  virtualPageY            = 0u;
        INT32   worldPageX              = 0;
        INT32   worldPageY              = 0;
        FLOAT32 pageWorldSize           = 0.0f;
        FLOAT32 texelWorldSize          = 0.0f;
        Vec2    lightSpaceMinXY         = Vec2::Zero();
        Vec2    lightSpaceMaxXY         = Vec2::Zero();
        Vec3    lightSpaceCenter        = Vec3::Zero();
        Vec3    worldCenter             = Vec3::Zero();
        FLOAT32 receiverMinZ            = 0.0f;
        FLOAT32 receiverMaxZ            = 0.0f;
        FLOAT32 casterMinZ              = 0.0f;
        FLOAT32 casterMaxZ              = 0.0f;
        Vec3    planarWorldCorners[4]{};  // 4 corners on cameraLightZ plane: (minX,minY), (maxX,minY), (maxX,maxY), (minX,maxY)
        Vec3    volumeWorldCorners[8]{};  // 0..3 at casterMinZ, 4..7 at casterMaxZ
        Vec3    worldBoundsMin          = Vec3::Zero();
        Vec3    worldBoundsMax          = Vec3::Zero();
        FLOAT32 shadowVolumeRadius      = 0.0f;
        BOOL    valid                   = false;
    };

    inline BOOL ComputeNVSPageWorldAndLightBounds(
        const NVSClipmapLevelState& levelState,
        UINT32                      virtualPageX,
        UINT32                      virtualPageY,
        NVSPageWorldBounds&         outBounds) noexcept
    {
        outBounds = NVSPageWorldBounds{};
        if (levelState.pagesPerAxis == 0u ||
            virtualPageX >= levelState.pagesPerAxis ||
            virtualPageY >= levelState.pagesPerAxis ||
            levelState.worldExtent <= 0.0f ||
            (levelState.casterMaxZ - levelState.casterMinZ) <= 1e-5f)
        {
            return false;
        }

        const FLOAT32 rightLenSq = levelState.lightRight.Dot(levelState.lightRight);
        const FLOAT32 upLenSq    = levelState.lightUp.Dot(levelState.lightUp);
        const FLOAT32 dirLenSq   = levelState.lightDir.Dot(levelState.lightDir);
        if (rightLenSq <= 1e-6f || upLenSq <= 1e-6f || dirLenSq <= 1e-6f)
        {
            return false;
        }

        const FLOAT32 pageSize   = levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis);
        const FLOAT32 texelSize  = pageSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        const INT32   worldPageX = NVSToroidalPageToWorldPage(virtualPageX, levelState.snappedOriginPageX, levelState.pagesPerAxis);
        const INT32   worldPageY = NVSToroidalPageToWorldPage(virtualPageY, levelState.snappedOriginPageY, levelState.pagesPerAxis);

        const FLOAT32 minX = static_cast<FLOAT32>(worldPageX) * pageSize;
        const FLOAT32 minY = static_cast<FLOAT32>(worldPageY) * pageSize;
        const FLOAT32 maxX = minX + pageSize;
        const FLOAT32 maxY = minY + pageSize;
        const FLOAT32 cx   = 0.5f * (minX + maxX);
        const FLOAT32 cy   = 0.5f * (minY + maxY);
        const FLOAT32 cz   = 0.5f * (levelState.casterMinZ + levelState.casterMaxZ);

        outBounds.lightId          = levelState.lightId;
        outBounds.clipmapLevel     = levelState.clipmapLevel;
        outBounds.virtualPageX     = virtualPageX;
        outBounds.virtualPageY     = virtualPageY;
        outBounds.worldPageX       = worldPageX;
        outBounds.worldPageY       = worldPageY;
        outBounds.pageWorldSize    = pageSize;
        outBounds.texelWorldSize   = texelSize;
        outBounds.lightSpaceMinXY  = Vec2(minX, minY);
        outBounds.lightSpaceMaxXY  = Vec2(maxX, maxY);
        outBounds.lightSpaceCenter = Vec3(cx, cy, cz);
        outBounds.receiverMinZ     = levelState.receiverMinZ;
        outBounds.receiverMaxZ     = levelState.receiverMaxZ;
        outBounds.casterMinZ       = levelState.casterMinZ;
        outBounds.casterMaxZ       = levelState.casterMaxZ;

        auto lightToWorld = [&](FLOAT32 lx, FLOAT32 ly, FLOAT32 lz) noexcept -> Vec3
        {
            return levelState.lightRight * lx + levelState.lightUp * ly + levelState.lightDir * lz;
        };

        outBounds.worldCenter = lightToWorld(cx, cy, cz);

        const FLOAT32 planeZ          = levelState.cameraLightZ;
        outBounds.planarWorldCorners[0] = lightToWorld(minX, minY, planeZ);
        outBounds.planarWorldCorners[1] = lightToWorld(maxX, minY, planeZ);
        outBounds.planarWorldCorners[2] = lightToWorld(maxX, maxY, planeZ);
        outBounds.planarWorldCorners[3] = lightToWorld(minX, maxY, planeZ);

        outBounds.volumeWorldCorners[0] = lightToWorld(minX, minY, levelState.casterMinZ);
        outBounds.volumeWorldCorners[1] = lightToWorld(maxX, minY, levelState.casterMinZ);
        outBounds.volumeWorldCorners[2] = lightToWorld(maxX, maxY, levelState.casterMinZ);
        outBounds.volumeWorldCorners[3] = lightToWorld(minX, maxY, levelState.casterMinZ);
        outBounds.volumeWorldCorners[4] = lightToWorld(minX, minY, levelState.casterMaxZ);
        outBounds.volumeWorldCorners[5] = lightToWorld(maxX, minY, levelState.casterMaxZ);
        outBounds.volumeWorldCorners[6] = lightToWorld(maxX, maxY, levelState.casterMaxZ);
        outBounds.volumeWorldCorners[7] = lightToWorld(minX, maxY, levelState.casterMaxZ);

        // Exact AABB of the oriented box in world space:
        // center +/- 0.5 * (|R|*dx + |U|*dy + |D|*dz)
        const FLOAT32 hx = 0.5f * pageSize;
        const FLOAT32 hy = 0.5f * pageSize;
        const FLOAT32 hz = 0.5f * (levelState.casterMaxZ - levelState.casterMinZ);
        const Vec3 extentWorld(
            std::fabs(levelState.lightRight.x) * hx + std::fabs(levelState.lightUp.x) * hy + std::fabs(levelState.lightDir.x) * hz,
            std::fabs(levelState.lightRight.y) * hx + std::fabs(levelState.lightUp.y) * hy + std::fabs(levelState.lightDir.y) * hz,
            std::fabs(levelState.lightRight.z) * hx + std::fabs(levelState.lightUp.z) * hy + std::fabs(levelState.lightDir.z) * hz);

        outBounds.worldBoundsMin     = outBounds.worldCenter - extentWorld;
        outBounds.worldBoundsMax     = outBounds.worldCenter + extentWorld;
        outBounds.shadowVolumeRadius = std::sqrt(hx * hx + hy * hy + hz * hz);
        outBounds.valid              = true;
        return true;
    }

    inline BOOL IsWorldPositionInsideNVSPageLightXY(
        const NVSPageWorldBounds&   bounds,
        const NVSClipmapLevelState& levelState,
        const Vec3&                 worldPos,
        FLOAT32                     epsilon = 1e-5f) noexcept
    {
        if (!bounds.valid)
        {
            return false;
        }
        const FLOAT32 lx = worldPos.Dot(levelState.lightRight);
        const FLOAT32 ly = worldPos.Dot(levelState.lightUp);
        return (lx >= (bounds.lightSpaceMinXY.x - epsilon)) &&
               (lx <= (bounds.lightSpaceMaxXY.x + epsilon)) &&
               (ly >= (bounds.lightSpaceMinXY.y - epsilon)) &&
               (ly <= (bounds.lightSpaceMaxXY.y + epsilon));
    }

    inline BOOL IsWorldPositionInsideNVSPageShadowVolume(
        const NVSPageWorldBounds&   bounds,
        const NVSClipmapLevelState& levelState,
        const Vec3&                 worldPos,
        FLOAT32                     epsilon = 1e-5f) noexcept
    {
        if (!IsWorldPositionInsideNVSPageLightXY(bounds, levelState, worldPos, epsilon))
        {
            return false;
        }
        const FLOAT32 lz = worldPos.Dot(levelState.lightDir);
        return (lz >= (bounds.casterMinZ - epsilon)) &&
               (lz <= (bounds.casterMaxZ + epsilon));
    }

    // Verifies whether a cached virtual page mapping at toroidal coordinates (virtualPageX, virtualPageY)
    // still represents the exact same world-space page identity (worldPageX, worldPageY) and generation.
    // Prevents stale reuse after toroidal wrap-around (Section 5.3).
    inline BOOL VerifyNVSVirtualPageIdentityMatch(
        const NVSClipmapLevelState& levelState,
        UINT32                      virtualPageX,
        UINT32                      virtualPageY,
        INT32                       cachedWorldPageX,
        INT32                       cachedWorldPageY,
        UINT32                      currentGeneration = 1u,
        UINT32                      cachedGeneration  = 1u) noexcept
    {
        if (levelState.pagesPerAxis == 0u ||
            virtualPageX >= levelState.pagesPerAxis ||
            virtualPageY >= levelState.pagesPerAxis ||
            currentGeneration != cachedGeneration)
        {
            return false;
        }
        const INT32 currentWorldPageX = NVSToroidalPageToWorldPage(
            virtualPageX, levelState.snappedOriginPageX, levelState.pagesPerAxis);
        const INT32 currentWorldPageY = NVSToroidalPageToWorldPage(
            virtualPageY, levelState.snappedOriginPageY, levelState.pagesPerAxis);
        return (currentWorldPageX == cachedWorldPageX) && (currentWorldPageY == cachedWorldPageY);
    }

    // =====================================================================
    //  Per-Page Orthographic Projection & Matrix Contract
    // =====================================================================
    // Builds the per-page orthographic projection matrix (LightView -> PageClip) and
    // combined World -> PageClip matrix (plus optional inverse PageClip -> World matrix).
    // Maps:
    //   x_L in [lightMinX, lightMaxX] -> [-1, +1]
    //   y_L in [lightMinY, lightMaxY] -> [-1, +1]
    //   z_L in [casterMinZ, casterMaxZ] -> [0, 1] (Standard-Z) or [1, 0] (Reversed-Z)
    inline BOOL ComputeNVSPageOrthographicProjection(
        const NVSClipmapLevelState& levelState,
        UINT32                      virtualPageX,
        UINT32                      virtualPageY,
        NVSDepthCompareMode         depthMode,
        Mat4&                       outPageProjMatrix,
        Mat4&                       outWorldToPageClipMatrix,
        Mat4*                       outPageClipToWorldMatrix = nullptr) noexcept
    {
        outPageProjMatrix        = Mat4::Identity();
        outWorldToPageClipMatrix = Mat4::Identity();
        if (outPageClipToWorldMatrix)
        {
            *outPageClipToWorldMatrix = Mat4::Identity();
        }

        NVSPageWorldBounds bounds{};
        if (!ComputeNVSPageWorldAndLightBounds(levelState, virtualPageX, virtualPageY, bounds))
        {
            return false;
        }

        const FLOAT32 minX  = bounds.lightSpaceMinXY.x;
        const FLOAT32 maxX  = bounds.lightSpaceMaxXY.x;
        const FLOAT32 minY  = bounds.lightSpaceMinXY.y;
        const FLOAT32 maxY  = bounds.lightSpaceMaxXY.y;
        const FLOAT32 zNear = bounds.casterMinZ;
        const FLOAT32 zFar  = bounds.casterMaxZ;
        const FLOAT32 dx    = maxX - minX;
        const FLOAT32 dy    = maxY - minY;
        const FLOAT32 dz    = zFar - zNear;
        if (dx <= 1e-6f || dy <= 1e-6f || dz <= 1e-6f)
        {
            return false;
        }

        const FLOAT32 sx = 2.0f / dx;
        const FLOAT32 sy = 2.0f / dy;
        const FLOAT32 ox = -(maxX + minX) / dx;
        const FLOAT32 oy = -(maxY + minY) / dy;

        FLOAT32 sz = 1.0f / dz;
        FLOAT32 oz = -zNear / dz;
        if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
        {
            sz = -1.0f / dz;
            oz = zFar / dz;
        }

        // Column-major pageProjMatrix
        outPageProjMatrix       = Mat4::Identity();
        outPageProjMatrix.m[0]  = sx;
        outPageProjMatrix.m[5]  = sy;
        outPageProjMatrix.m[10] = sz;
        outPageProjMatrix.m[12] = ox;
        outPageProjMatrix.m[13] = oy;
        outPageProjMatrix.m[14] = oz;
        outPageProjMatrix.m[15] = 1.0f;

        Mat4 worldToLightView = levelState.worldToLightView;
        Mat4 lightViewToWorld = levelState.lightViewToWorld;
        ComputeNVSLightViewMatrices(
            levelState.lightRight,
            levelState.lightUp,
            levelState.lightDir,
            worldToLightView,
            lightViewToWorld);

        // Direct closed-form product: pageProjMatrix * worldToLightView
        // Matches RuntimeNVSPageProjection.nsl 1:1
        outWorldToPageClipMatrix.m[0]  = sx * levelState.lightRight.x;
        outWorldToPageClipMatrix.m[1]  = sy * levelState.lightUp.x;
        outWorldToPageClipMatrix.m[2]  = sz * levelState.lightDir.x;
        outWorldToPageClipMatrix.m[3]  = 0.0f;

        outWorldToPageClipMatrix.m[4]  = sx * levelState.lightRight.y;
        outWorldToPageClipMatrix.m[5]  = sy * levelState.lightUp.y;
        outWorldToPageClipMatrix.m[6]  = sz * levelState.lightDir.y;
        outWorldToPageClipMatrix.m[7]  = 0.0f;

        outWorldToPageClipMatrix.m[8]  = sx * levelState.lightRight.z;
        outWorldToPageClipMatrix.m[9]  = sy * levelState.lightUp.z;
        outWorldToPageClipMatrix.m[10] = sz * levelState.lightDir.z;
        outWorldToPageClipMatrix.m[11] = 0.0f;

        outWorldToPageClipMatrix.m[12] = ox;
        outWorldToPageClipMatrix.m[13] = oy;
        outWorldToPageClipMatrix.m[14] = oz;
        outWorldToPageClipMatrix.m[15] = 1.0f;

        if (outPageClipToWorldMatrix != nullptr)
        {
            Mat4 invProj       = Mat4::Identity();
            invProj.m[0]       = 0.5f * dx;
            invProj.m[5]       = 0.5f * dy;
            invProj.m[12]      = 0.5f * (maxX + minX);
            invProj.m[13]      = 0.5f * (maxY + minY);
            if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
            {
                invProj.m[10] = -dz;
                invProj.m[14] = zFar;
            }
            else
            {
                invProj.m[10] = dz;
                invProj.m[14] = zNear;
            }
            *outPageClipToWorldMatrix = lightViewToWorld * invProj;
        }
        return true;
    }

    // Projects a world-space position into page clip/NDC space, intra-page UV [0, 1], and shadow texel coordinates [0, 128].
    inline Vec3 ProjectWorldToNVSPageNDCAndUV(
        const Mat4& worldToPageClipMatrix,
        const Vec3& worldPos,
        Vec2*       outPageUV        = nullptr,
        Vec2*       outShadowTexelXY = nullptr) noexcept
    {
        const Vec4 clip = worldToPageClipMatrix * Vec4(worldPos.x, worldPos.y, worldPos.z, 1.0f);
        const FLOAT32 invW = (std::fabs(clip.w) > 1e-7f) ? (1.0f / clip.w) : 1.0f;
        const Vec3 ndc(clip.x * invW, clip.y * invW, clip.z * invW);
        const Vec2 uv(0.5f * ndc.x + 0.5f, 0.5f * ndc.y + 0.5f);
        if (outPageUV != nullptr)
        {
            *outPageUV = uv;
        }
        if (outShadowTexelXY != nullptr)
        {
            outShadowTexelXY->x = uv.x * static_cast<FLOAT32>(kNVSPageTexelSize);
            outShadowTexelXY->y = uv.y * static_cast<FLOAT32>(kNVSPageTexelSize);
        }
        return ndc;
    }

    // Builds a single NVSClipmapLevelGPUData entry from a CPU NVSClipmapLevelState.
    inline NVSClipmapLevelGPUData BuildNVSClipmapLevelGPUData(
        const NVSClipmapLevelState& levelState,
        NVSDepthCompareMode         depthMode = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
        BOOL                        levelValid = true) noexcept
    {
        NVSClipmapLevelGPUData gpu{};
        gpu.lightId            = levelState.lightId;
        gpu.clipmapLevel       = levelState.clipmapLevel;
        gpu.pagesPerAxis       = levelState.pagesPerAxis;
        gpu.flags              = levelValid ? NVS_CLIPMAP_LEVEL_FLAG_VALID : NVS_CLIPMAP_LEVEL_FLAG_NONE;
        if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
        {
            gpu.flags |= NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z;
        }

        gpu.snappedOriginPageX = levelState.snappedOriginPageX;
        gpu.snappedOriginPageY = levelState.snappedOriginPageY;
        gpu.worldExtent        = levelState.worldExtent;
        gpu.pageWorldSize      = (levelState.pagesPerAxis > 0u)
            ? (levelState.worldExtent / static_cast<FLOAT32>(levelState.pagesPerAxis))
            : levelState.pageWorldSize;
        gpu.texelWorldSize     = gpu.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        gpu.receiverMinZ       = levelState.receiverMinZ;
        gpu.receiverMaxZ       = levelState.receiverMaxZ;
        gpu.casterMinZ         = levelState.casterMinZ;
        gpu.casterMaxZ         = levelState.casterMaxZ;
        gpu.depthBiasNdc       = levelState.depthBiasNdc;
        gpu.normalBiasWorld    = levelState.normalBiasWorld;
        gpu.cameraLightZ       = levelState.cameraLightZ;

        gpu.lightRightAndCenterX = Vec4(
            levelState.lightRight.x,
            levelState.lightRight.y,
            levelState.lightRight.z,
            levelState.snappedLightCenterX);
        gpu.lightUpAndCenterY = Vec4(
            levelState.lightUp.x,
            levelState.lightUp.y,
            levelState.lightUp.z,
            levelState.snappedLightCenterY);
        gpu.lightDirAndExtrusion = Vec4(
            levelState.lightDir.x,
            levelState.lightDir.y,
            levelState.lightDir.z,
            levelState.casterExtrusionDistance);

        const FLOAT32 transStart = std::clamp(levelState.transitionStartRatio, 0.5f, 0.99f);
        const FLOAT32 transWidth = std::max(1.0f - transStart, 1e-4f);
        gpu.transitionAndPadding = Vec4(
            transStart,
            transWidth,
            levelState.casterDownstreamMargin,
            levelState.casterConservativePadding);

        ComputeNVSLightViewMatrices(
            levelState.lightRight,
            levelState.lightUp,
            levelState.lightDir,
            gpu.worldToLightView,
            gpu.lightViewToWorld);
        return gpu;
    }

    // =====================================================================
    //  Dynamic Texel-Relative Shadow Bias Contract
    //  Unified mathematical definitions shared across Directional orthographic
    //  clipmaps, Spot virtual perspective mip pyramids, and Point 6-face
    //  omnidirectional virtual cubemap mip pyramids.
    // =====================================================================
    inline void ComputeNVSTexelRelativeOrthographicBias(
        FLOAT32             localTexelWorldSize,
        FLOAT32             casterMinZ,
        FLOAT32             casterMaxZ,
        FLOAT32             depthBiasTexels,
        FLOAT32             normalBiasTexels,
        NVSDepthCompareMode depthMode,
        FLOAT32&            outSignedDepthBiasNdc,
        FLOAT32&            outNormalBiasWorld) noexcept
    {
        const FLOAT32 safeTexel = std::max(localTexelWorldSize, 0.0f);
        const FLOAT32 dBias     = std::isfinite(depthBiasTexels)  ? std::clamp(depthBiasTexels,  0.0f, 1.0f) : 0.0f;
        const FLOAT32 nBias     = std::isfinite(normalBiasTexels) ? std::clamp(normalBiasTexels, 0.0f, 1.0f) : 0.0f;
        const FLOAT32 zSpan     = std::max(casterMaxZ - casterMinZ, 1e-4f);

        outNormalBiasWorld = (nBias > 0.0f) ? (nBias * safeTexel) : 0.0f;
        if (dBias <= 0.0f)
        {
            outSignedDepthBiasNdc = 0.0f;
        }
        else
        {
            const FLOAT32 absBiasNdc = (dBias * safeTexel) / zSpan;
            outSignedDepthBiasNdc    = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -absBiasNdc : +absBiasNdc;
        }
    }

    inline void ComputeNVSTexelRelativePerspectiveBias(
        FLOAT32             baseMip0TexelTanSize,
        FLOAT32             nearZ,
        FLOAT32             farZ,
        FLOAT32             depthBiasTexels,
        FLOAT32             normalBiasTexels,
        NVSDepthCompareMode depthMode,
        FLOAT32&            outSignedDepthBiasCoeff,
        FLOAT32&            outBaseNormalBiasTanScale) noexcept
    {
        const FLOAT32 safeTan = std::max(baseMip0TexelTanSize, 0.0f);
        const FLOAT32 dBias   = std::isfinite(depthBiasTexels)  ? std::clamp(depthBiasTexels,  0.0f, 1.0f) : 0.0f;
        const FLOAT32 nBias   = std::isfinite(normalBiasTexels) ? std::clamp(normalBiasTexels, 0.0f, 1.0f) : 0.0f;
        const FLOAT32 zNear   = std::max(nearZ, 1e-4f);
        const FLOAT32 zFar    = std::max(farZ, zNear + 1e-4f);
        const FLOAT32 dz      = zFar - zNear;

        // At virtual mip m and light-space depth z, localTexelWorldSize(m, z) = (2^m * baseMip0TexelTanSize) * z.
        outBaseNormalBiasTanScale = (nBias > 0.0f) ? (nBias * safeTan) : 0.0f;
        if (dBias <= 0.0f)
        {
            outSignedDepthBiasCoeff = 0.0f;
        }
        else
        {
            // Nonlinear perspective |d(ndcZ)/dz| = (zNear * zFar) / (dz * z^2).
            // Multiplying by (dBias * 2^m * safeTan * z) cancels one power of z:
            //   |deltaNdc(m, z)| = (2^m * dBias * safeTan * zNear * zFar / dz) / z.
            const FLOAT32 absCoeff  = dBias * safeTan * ((zNear * zFar) / dz);
            outSignedDepthBiasCoeff = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -absCoeff : +absCoeff;
        }
    }

    // Populates the 4 clipmap level entries into an arbitrary dynamic viewSlotBase (4-aligned) inside outTable.
    inline void BuildNVSClipmapHierarchyGPUDataAtSlot(
        const NVSClipmapHierarchyState& hierarchy,
        NVSClipmapLevelGPUData*         outTable,
        UINT32                          tableCapacity,
        UINT32                          viewSlotBase) noexcept
    {
        if (outTable == nullptr || (viewSlotBase + kNVSMaxClipmapLevels) > tableCapacity)
        {
            return;
        }
        const UINT32 slotBlock = viewSlotBase >> 2u;
        for (UINT32 lvl = 0u; lvl < kNVSMaxClipmapLevels; ++lvl)
        {
            NVSClipmapLevelState ls = hierarchy.levels[lvl];
            ls.lightId              = slotBlock;
            ls.clipmapLevel         = lvl;
            ls.pagesPerAxis         = hierarchy.pagesPerAxis;
            ls.lightRight           = hierarchy.lightRight;
            ls.lightUp              = hierarchy.lightUp;
            ls.lightDir             = hierarchy.lightDir;
            ls.snappedOriginPageX   = hierarchy.snappedOriginPageX[lvl];
            ls.snappedOriginPageY   = hierarchy.snappedOriginPageY[lvl];
            if (hierarchy.levelWorldExtents[lvl] > 0.0f)
            {
                ls.worldExtent = hierarchy.levelWorldExtents[lvl];
            }
            if (ls.pagesPerAxis > 0u && ls.worldExtent > 0.0f)
            {
                ls.pageWorldSize  = ls.worldExtent / static_cast<FLOAT32>(ls.pagesPerAxis);
                ls.texelWorldSize = ls.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
            }
            ls.snappedLightCenterX = static_cast<FLOAT32>(ls.snappedOriginPageX) * ls.pageWorldSize;
            ls.snappedLightCenterY = static_cast<FLOAT32>(ls.snappedOriginPageY) * ls.pageWorldSize;
            if ((ls.casterMaxZ - ls.casterMinZ) <= 1e-4f)
            {
                ls.receiverMinZ              = -0.5f * ls.worldExtent;
                ls.receiverMaxZ              = +0.5f * ls.worldExtent;
                ls.casterExtrusionDistance   = std::max(256.0f, 2.0f * ls.worldExtent);
                ls.casterDownstreamMargin    = std::max(16.0f, 0.25f * ls.worldExtent);
                ls.casterConservativePadding = 0.5f * ls.pageWorldSize;
                ls.casterMinZ                = ls.receiverMinZ - ls.casterExtrusionDistance - ls.casterConservativePadding;
                ls.casterMaxZ                = ls.receiverMaxZ + ls.casterDownstreamMargin + ls.casterConservativePadding;
            }
            const BOOL valid = hierarchy.lightValid && (lvl < hierarchy.activeLevelCount);
            outTable[viewSlotBase + lvl] = BuildNVSClipmapLevelGPUData(
                ls,
                hierarchy.depthMode,
                valid);
        }
    }

    // Legacy 16-entry convenience wrapper for unit tests.
    inline void BuildNVSClipmapHierarchyGPUData(
        const NVSClipmapHierarchyState& hierarchy,
        NVSClipmapLevelGPUData          outTable[kNVSClipmapTableMaxEntries]) noexcept
    {
        if (outTable == nullptr || hierarchy.lightId >= kNVSMaxClipmapLights)
        {
            return;
        }
        BuildNVSClipmapHierarchyGPUDataAtSlot(
            hierarchy,
            outTable,
            kNVSClipmapTableMaxEntries,
            hierarchy.lightId * kNVSMaxClipmapLevels);
    }

    // CPU reference builder for NVSPageRenderData matching RuntimeNVSPageProjection.nsl 1:1.
    inline BOOL BuildNVSPageRenderDataCPU(
        UINT32                      physicalPageIndex,
        const NVSPhysicalPageEntry& physEntry,
        const NVSVirtualPageEntry&  virtEntry,
        const NVSClipmapLevelState& levelState,
        NVSDepthCompareMode         depthMode,
        NVSPageRenderData&          outRenderData) noexcept
    {
        outRenderData                   = NVSPageRenderData{};
        outRenderData.physicalPageIndex = physicalPageIndex;
        outRenderData.virtualFlatIndex  = kNVSInvalidVirtualPage;

        const NVSPhysicalPageState physState = ExtractNVSPhysicalPageState(physEntry.stateAndFlags);
        if (physState != NVS_PHYS_STATE_CLEARED_NOT_RENDERED &&
            physState != NVS_PHYS_STATE_RESIDENT_VALID)
        {
            if (physState == NVS_PHYS_STATE_ALLOCATED_NOT_CLEARED)
            {
                outRenderData.renderFlags = NVS_PAGE_RENDER_FLAG_ALLOCATED_NOT_CLEARED;
            }
            return false;
        }

        const NVSVirtualPageAddress addr = UnpackNVSVirtualPageAddress(physEntry.ownerVirtualAddress);
        if (!IsValidNVSVirtualPageAddress(addr, levelState.pagesPerAxis, kNVSMaxClipmapLevels, kNVSMaxSupportedLights))
        {
            return false;
        }

        const NVSPageState virtState = ExtractNVSPageState(virtEntry.stateAndFlags);
        if ((virtState != NVS_PAGE_STATE_CLEARED_NOT_RENDERED && virtState != NVS_PAGE_STATE_VALID_SHADOW_DATA) ||
            virtEntry.physicalPageIndex != physicalPageIndex ||
            virtEntry.virtualAddressPacked != physEntry.ownerVirtualAddress)
        {
            return false;
        }

        NVSPageWorldBounds bounds{};
        Mat4               pageProj{};
        Mat4               worldToPageClip{};
        if (!ComputeNVSPageWorldAndLightBounds(levelState, addr.virtualPageX, addr.virtualPageY, bounds) ||
            !ComputeNVSPageOrthographicProjection(levelState, addr.virtualPageX, addr.virtualPageY, depthMode, pageProj, worldToPageClip, nullptr))
        {
            return false;
        }

        UINT32 flags = NVS_PAGE_RENDER_FLAG_VALID_RESIDENT;
        if (physState == NVS_PHYS_STATE_CLEARED_NOT_RENDERED)
        {
            flags |= NVS_PAGE_RENDER_FLAG_NEEDS_CASTER_RENDER;
        }
        else if (physState == NVS_PHYS_STATE_RESIDENT_VALID)
        {
            flags |= NVS_PAGE_RENDER_FLAG_CACHED_VALID_SHADOW;
        }
        if ((virtEntry.stateAndFlags & NVS_PAGE_FLAG_NEWLY_ALLOCATED) != 0u)
        {
            flags |= NVS_PAGE_RENDER_FLAG_NEWLY_ALLOCATED;
        }
        if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
        {
            flags |= NVS_PAGE_RENDER_FLAG_REVERSED_Z;
        }

        outRenderData.physicalPageIndex     = physicalPageIndex;
        outRenderData.virtualFlatIndex      = physEntry.ownerVirtualFlatIndex;
        outRenderData.virtualAddressPacked  = physEntry.ownerVirtualAddress;
        outRenderData.renderFlags           = flags;
        outRenderData.lightId               = addr.lightId;
        outRenderData.clipmapLevel          = addr.clipmapLevel;
        outRenderData.virtualPageX          = addr.virtualPageX;
        outRenderData.virtualPageY          = addr.virtualPageY;
        outRenderData.worldPageX            = bounds.worldPageX;
        outRenderData.worldPageY            = bounds.worldPageY;
        outRenderData.physicalGeneration    = physEntry.physicalGeneration;
        outRenderData.lastUsedFrame         = physEntry.lastUsedFrame;
        outRenderData.lightMinX             = bounds.lightSpaceMinXY.x;
        outRenderData.lightMinY             = bounds.lightSpaceMinXY.y;
        outRenderData.lightMaxX             = bounds.lightSpaceMaxXY.x;
        outRenderData.lightMaxY             = bounds.lightSpaceMaxXY.y;
        outRenderData.receiverMinZ          = bounds.receiverMinZ;
        outRenderData.receiverMaxZ          = bounds.receiverMaxZ;
        outRenderData.casterMinZ            = bounds.casterMinZ;
        outRenderData.casterMaxZ            = bounds.casterMaxZ;
        outRenderData.worldCenter           = bounds.worldCenter;
        outRenderData.pageWorldSize         = bounds.pageWorldSize;
        outRenderData.worldBoundsMin        = bounds.worldBoundsMin;
        outRenderData.texelWorldSize        = bounds.texelWorldSize;
        outRenderData.worldBoundsMax        = bounds.worldBoundsMax;
        outRenderData.shadowVolumeRadius    = bounds.shadowVolumeRadius;
        outRenderData.depthBiasNdc          = levelState.depthBiasNdc;
        outRenderData.normalBiasWorld       = levelState.normalBiasWorld;
        outRenderData.viewportWidth         = kNVSPageTexelSize;
        outRenderData.viewportHeight        = kNVSPageTexelSize;
        outRenderData.worldToPageClipMatrix = worldToPageClip;
        Mat4 w2l = levelState.worldToLightView;
        Mat4 l2w = levelState.lightViewToWorld;
        ComputeNVSLightViewMatrices(levelState.lightRight, levelState.lightUp, levelState.lightDir, w2l, l2w);
        outRenderData.lightViewMatrix       = w2l;
        outRenderData.pageProjMatrix        = pageProj;
        return true;
    }

    // =====================================================================
    //  Perspective Shadow View Builders (Spot Cone & Point 6-Face Cube)
    // =====================================================================
    inline void ComputeNVSPerspectiveLightViewMatrices(
        const Vec3& lightWorldPos,
        const Vec3& lightRight,
        const Vec3& lightUp,
        const Vec3& lightDir,
        Mat4&       outWorldToLightView,
        Mat4&       outLightViewToWorld) noexcept
    {
        outWorldToLightView = Mat4::Identity();
        outWorldToLightView.m[0]  = lightRight.x;
        outWorldToLightView.m[1]  = lightUp.x;
        outWorldToLightView.m[2]  = lightDir.x;
        outWorldToLightView.m[3]  = 0.0f;

        outWorldToLightView.m[4]  = lightRight.y;
        outWorldToLightView.m[5]  = lightUp.y;
        outWorldToLightView.m[6]  = lightDir.y;
        outWorldToLightView.m[7]  = 0.0f;

        outWorldToLightView.m[8]  = lightRight.z;
        outWorldToLightView.m[9]  = lightUp.z;
        outWorldToLightView.m[10] = lightDir.z;
        outWorldToLightView.m[11] = 0.0f;

        outWorldToLightView.m[12] = -lightWorldPos.Dot(lightRight);
        outWorldToLightView.m[13] = -lightWorldPos.Dot(lightUp);
        outWorldToLightView.m[14] = -lightWorldPos.Dot(lightDir);
        outWorldToLightView.m[15] = 1.0f;

        outLightViewToWorld = Mat4::Identity();
        outLightViewToWorld.m[0]  = lightRight.x;
        outLightViewToWorld.m[1]  = lightRight.y;
        outLightViewToWorld.m[2]  = lightRight.z;
        outLightViewToWorld.m[3]  = 0.0f;

        outLightViewToWorld.m[4]  = lightUp.x;
        outLightViewToWorld.m[5]  = lightUp.y;
        outLightViewToWorld.m[6]  = lightUp.z;
        outLightViewToWorld.m[7]  = 0.0f;

        outLightViewToWorld.m[8]  = lightDir.x;
        outLightViewToWorld.m[9]  = lightDir.y;
        outLightViewToWorld.m[10] = lightDir.z;
        outLightViewToWorld.m[11] = 0.0f;

        outLightViewToWorld.m[12] = lightWorldPos.x;
        outLightViewToWorld.m[13] = lightWorldPos.y;
        outLightViewToWorld.m[14] = lightWorldPos.z;
        outLightViewToWorld.m[15] = 1.0f;
    }

    inline NVSClipmapLevelGPUData BuildNVSSpotShadowViewGPUData(
        UINT32              viewSlot,
        UINT32              logicalLightId,
        const Vec3&         lightWorldPos,
        const Vec3&         lightDirection,
        FLOAT32             range,
        FLOAT32             innerConeRadians,
        FLOAT32             outerConeRadians,
        UINT32              pagesPerAxis          = 64u,
        NVSDepthCompareMode depthMode             = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
        FLOAT32             customBias            = -1.0f,
        FLOAT32             normalBias            = -1.0f,
        FLOAT32             customNearPlane       = 0.0f,
        BOOL                enableVirtualMipChain = false,
        BOOL                useTexelRelativeBias  = false,
        UINT32              viewGeneration        = 1u,
        BOOL                hasDetailTier         = false,
        UINT32              detailMinMip          = 0u,
        FLOAT32             detailLodBias         = 0.0f) noexcept
    {
        NVSClipmapLevelGPUData gpu{};
        gpu.lightId      = viewSlot >> 2u;
        gpu.clipmapLevel = viewSlot & 3u;
        gpu.pagesPerAxis = (pagesPerAxis > 0u && pagesPerAxis <= kNVSMaxPagesPerAxis) ? pagesPerAxis : 64u;

        Vec3 R{}, U{}, D{};
        if (!ComputeNVSOrthonormalLightBasis(lightDirection, R, U, D) || range <= 0.05f)
        {
            gpu.flags = NVS_CLIPMAP_LEVEL_FLAG_NONE;
            return gpu;
        }

        const FLOAT32 clampedOuter = std::clamp(outerConeRadians, 0.00872665f, 1.53588974f); // 0.5 deg .. 88.0 deg half-angle (1 deg .. 176 deg full FOV)
        const FLOAT32 clampedInner = std::clamp(innerConeRadians, 0.0f, clampedOuter);
        const FLOAT32 tanHalf      = std::tan(clampedOuter);
        const FLOAT32 nearZ        = (std::isfinite(customNearPlane) && customNearPlane > 0.0f)
            ? std::clamp(customNearPlane, 0.001f, std::max(0.002f, 0.5f * range))
            : std::clamp(range * 0.005f, 0.05f, 0.5f);
        const FLOAT32 farZ         = std::max(range, nearZ + 0.05f);
        const UINT32  mipCount     = enableVirtualMipChain ? ComputeNVSMipCountFromPagesPerAxis(gpu.pagesPerAxis) : 1u;
        const UINT32  safeGen      = (viewGeneration == 0u) ? 1u : (((viewGeneration - 1u) % 255u) + 1u);

        const UINT32 tierOrIdBits  = hasDetailTier
            ? PackNVSViewDetailTierFlags(detailMinMip, detailLodBias)
            : ((logicalLightId & 0x7Fu) << NVS_CLIPMAP_LEVEL_LOGICAL_LIGHT_SHIFT);
        gpu.flags = NVS_CLIPMAP_LEVEL_FLAG_VALID |
                    NVS_CLIPMAP_LEVEL_FLAG_PERSPECTIVE |
                    NVS_CLIPMAP_LEVEL_FLAG_SPOT_CONE |
                    (((mipCount - 1u) & 0xFu) << NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_SHIFT) |
                    tierOrIdBits |
                    (safeGen << NVS_CLIPMAP_LEVEL_VIEW_GEN_SHIFT);
        if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
        {
            gpu.flags |= NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z;
        }

        gpu.snappedOriginPageX = 0;
        gpu.snappedOriginPageY = 0;
        gpu.worldExtent        = 2.0f * tanHalf;
        gpu.pageWorldSize      = gpu.worldExtent / static_cast<FLOAT32>(gpu.pagesPerAxis);
        gpu.texelWorldSize     = gpu.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
        gpu.receiverMinZ       = nearZ;
        gpu.receiverMaxZ       = farZ;
        gpu.casterMinZ         = nearZ;
        gpu.casterMaxZ         = farZ;

        if (useTexelRelativeBias)
        {
            gpu.flags |= NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS;
            ComputeNVSTexelRelativePerspectiveBias(
                gpu.texelWorldSize,
                nearZ,
                farZ,
                (customBias >= 0.0f) ? customBias : 0.005f,
                (normalBias >= 0.0f) ? normalBias : 0.02f,
                depthMode,
                gpu.depthBiasNdc,
                gpu.normalBiasWorld);
        }
        else
        {
            const FLOAT32 defaultBias = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -0.0005f : +0.0005f;
            gpu.depthBiasNdc          = (std::isfinite(customBias) && customBias >= 0.0f)
                ? ((depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -customBias : +customBias)
                : defaultBias;
            gpu.normalBiasWorld = (std::isfinite(normalBias) && normalBias >= 0.0f)
                ? normalBias
                : (1.5f * gpu.texelWorldSize * std::max(1.0f, 0.25f * farZ));
        }
        gpu.cameraLightZ    = 0.0f;

        gpu.lightRightAndCenterX = Vec4(R.x, R.y, R.z, 0.0f);
        gpu.lightUpAndCenterY    = Vec4(U.x, U.y, U.z, 0.0f);
        gpu.lightDirAndExtrusion = Vec4(D.x, D.y, D.z, farZ);
        gpu.transitionAndPadding = Vec4(std::cos(clampedOuter), std::cos(clampedInner), farZ, 0.05f);

        ComputeNVSPerspectiveLightViewMatrices(
            lightWorldPos,
            R,
            U,
            D,
            gpu.worldToLightView,
            gpu.lightViewToWorld);
        return gpu;
    }

    // Builds 6 omnidirectional 90-degree FOV perspective cube-face shadow views (+X, -X, +Y, -Y, +Z, -Z)
    // starting at viewSlotBase .. viewSlotBase + 5 with zero fixed view-table upper bound.
    inline void BuildNVSPointCubeShadowViewsGPUData(
        UINT32                 viewSlotBase,
        UINT32                 logicalLightId,
        const Vec3&            lightWorldPos,
        FLOAT32                range,
        NVSClipmapLevelGPUData outViews[6],
        UINT32                 pagesPerAxis          = 64u,
        NVSDepthCompareMode    depthMode             = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE,
        FLOAT32                customBias            = -1.0f,
        FLOAT32                normalBias            = -1.0f,
        FLOAT32                customNearPlane       = 0.0f,
        BOOL                   enableVirtualMipChain = false,
        BOOL                   useTexelRelativeBias  = false,
        UINT32                 viewGeneration        = 1u,
        BOOL                   hasDetailTier         = false,
        UINT32                 detailMinMip          = 0u,
        FLOAT32                detailLodBias         = 0.0f) noexcept
    {
        if (outViews == nullptr)
        {
            return;
        }

        // Orthonormal right-handed/left-handed consistent 6-face basis (det([R, U, D]) == +1 matching ComputeNVSOrthonormalLightBasis)
        static const Vec3 kFaceDirs[6] = {
            Vec3(+1.0f,  0.0f,  0.0f), // 0: +X
            Vec3(-1.0f,  0.0f,  0.0f), // 1: -X
            Vec3( 0.0f, +1.0f,  0.0f), // 2: +Y
            Vec3( 0.0f, -1.0f,  0.0f), // 3: -Y
            Vec3( 0.0f,  0.0f, +1.0f), // 4: +Z
            Vec3( 0.0f,  0.0f, -1.0f)  // 5: -Z
        };
        static const Vec3 kFaceRights[6] = {
            Vec3( 0.0f,  0.0f, -1.0f), // +X: R=(0,0,-1), U=(0,1,0), D=(1,0,0)
            Vec3( 0.0f,  0.0f, +1.0f), // -X: R=(0,0,+1), U=(0,1,0), D=(-1,0,0)
            Vec3(+1.0f,  0.0f,  0.0f), // +Y: R=(1,0,0),  U=(0,0,-1), D=(0,1,0)
            Vec3(+1.0f,  0.0f,  0.0f), // -Y: R=(1,0,0),  U=(0,0,+1), D=(0,-1,0)
            Vec3(+1.0f,  0.0f,  0.0f), // +Z: R=(1,0,0),  U=(0,1,0),  D=(0,0,1)
            Vec3(-1.0f,  0.0f,  0.0f)  // -Z: R=(-1,0,0), U=(0,1,0),  D=(0,0,-1)
        };
        static const Vec3 kFaceUps[6] = {
            Vec3( 0.0f, +1.0f,  0.0f),
            Vec3( 0.0f, +1.0f,  0.0f),
            Vec3( 0.0f,  0.0f, -1.0f),
            Vec3( 0.0f,  0.0f, +1.0f),
            Vec3( 0.0f, +1.0f,  0.0f),
            Vec3( 0.0f, +1.0f,  0.0f)
        };

        const UINT32  effPages = (pagesPerAxis > 0u && pagesPerAxis <= kNVSMaxPagesPerAxis) ? pagesPerAxis : 64u;
        const UINT32  mipCount = enableVirtualMipChain ? ComputeNVSMipCountFromPagesPerAxis(effPages) : 1u;
        const UINT32  safeGen  = (viewGeneration == 0u) ? 1u : (((viewGeneration - 1u) % 255u) + 1u);
        const FLOAT32 nearZ    = (std::isfinite(customNearPlane) && customNearPlane > 0.0f)
            ? std::clamp(customNearPlane, 0.001f, std::max(0.002f, 0.5f * range))
            : std::clamp(range * 0.005f, 0.05f, 0.5f);
        const FLOAT32 farZ     = std::max(range, nearZ + 0.05f);

        const FLOAT32 defaultBias = (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -0.0006f : +0.0006f;
        const FLOAT32 effBiasNdc  = (std::isfinite(customBias) && customBias >= 0.0f)
            ? ((depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO) ? -customBias : +customBias)
            : defaultBias;
        const UINT32 tierOrIdBits = hasDetailTier
            ? PackNVSViewDetailTierFlags(detailMinMip, detailLodBias)
            : ((logicalLightId & 0x7Fu) << NVS_CLIPMAP_LEVEL_LOGICAL_LIGHT_SHIFT);

        for (UINT32 f = 0u; f < 6u; ++f)
        {
            const UINT32 slot          = viewSlotBase + f;
            NVSClipmapLevelGPUData& gv = outViews[f];
            gv                         = NVSClipmapLevelGPUData{};
            gv.lightId                 = slot >> 2u;
            gv.clipmapLevel            = slot & 3u;
            gv.pagesPerAxis            = effPages;

            if (range <= 0.05f || slot == 0xFFFFFFFFu)
            {
                gv.flags = NVS_CLIPMAP_LEVEL_FLAG_NONE;
                continue;
            }

            gv.flags = NVS_CLIPMAP_LEVEL_FLAG_VALID |
                       NVS_CLIPMAP_LEVEL_FLAG_PERSPECTIVE |
                       NVS_CLIPMAP_LEVEL_FLAG_POINT_CUBE |
                       ((f & 0x7u) << NVS_CLIPMAP_LEVEL_CUBE_FACE_SHIFT) |
                       (((mipCount - 1u) & 0xFu) << NVS_CLIPMAP_LEVEL_MIP_COUNT_MINUS_ONE_SHIFT) |
                       tierOrIdBits |
                       (safeGen << NVS_CLIPMAP_LEVEL_VIEW_GEN_SHIFT);
            if (depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO)
            {
                gv.flags |= NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z;
            }

            gv.snappedOriginPageX    = 0;
            gv.snappedOriginPageY    = 0;
            gv.worldExtent           = 2.0f; // 90 deg FOV: tan(45 deg) = 1.0 -> [-1, +1] tangent span = 2.0
            gv.pageWorldSize         = gv.worldExtent / static_cast<FLOAT32>(effPages);
            gv.texelWorldSize        = gv.pageWorldSize / static_cast<FLOAT32>(kNVSPageTexelSize);
            gv.receiverMinZ          = nearZ;
            gv.receiverMaxZ          = farZ;
            gv.casterMinZ            = nearZ;
            gv.casterMaxZ            = farZ;
            if (useTexelRelativeBias)
            {
                gv.flags |= NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS;
                ComputeNVSTexelRelativePerspectiveBias(
                    gv.texelWorldSize,
                    nearZ,
                    farZ,
                    (customBias >= 0.0f) ? customBias : 0.005f,
                    (normalBias >= 0.0f) ? normalBias : 0.02f,
                    depthMode,
                    gv.depthBiasNdc,
                    gv.normalBiasWorld);
            }
            else
            {
                gv.depthBiasNdc    = effBiasNdc;
                gv.normalBiasWorld = (std::isfinite(normalBias) && normalBias >= 0.0f)
                    ? normalBias
                    : (1.5f * gv.texelWorldSize * std::max(1.0f, 0.25f * farZ));
            }
            gv.cameraLightZ          = 0.0f;
            gv.lightRightAndCenterX  = Vec4(kFaceRights[f].x, kFaceRights[f].y, kFaceRights[f].z, 0.0f);
            gv.lightUpAndCenterY     = Vec4(kFaceUps[f].x, kFaceUps[f].y, kFaceUps[f].z, 0.0f);
            gv.lightDirAndExtrusion  = Vec4(kFaceDirs[f].x, kFaceDirs[f].y, kFaceDirs[f].z, farZ);
            gv.transitionAndPadding  = Vec4(-2.0f, -2.0f, farZ, 0.05f);

            ComputeNVSPerspectiveLightViewMatrices(
                lightWorldPos,
                kFaceRights[f],
                kFaceUps[f],
                kFaceDirs[f],
                gv.worldToLightView,
                gv.lightViewToWorld);
        }
    }

    // =====================================================================
    //  Clipmap Transition & Seam-Avoidance Contract
    // =====================================================================
    // Computes a continuous blend factor in [0.0f, 1.0f] between clipmap level L (0.0)
    // and coarser level L+1 (1.0) inside the outer border band [transitionStartRatio .. 1.0].
    inline FLOAT32 ComputeNVSClipmapTransitionFactor(
        const NVSClipmapLevelState& levelState,
        const Vec3&                 worldPos,
        FLOAT32                     transitionStartRatio = 0.85f) noexcept
    {
        if (levelState.pagesPerAxis == 0u || levelState.worldExtent <= 0.0f)
        {
            return 1.0f;
        }
        const FLOAT32 halfExtent = 0.5f * levelState.worldExtent;
        const FLOAT32 projX      = worldPos.Dot(levelState.lightRight);
        const FLOAT32 projY      = worldPos.Dot(levelState.lightUp);
        const FLOAT32 dx         = std::fabs(projX - levelState.snappedLightCenterX) / halfExtent;
        const FLOAT32 dy         = std::fabs(projY - levelState.snappedLightCenterY) / halfExtent;
        const FLOAT32 maxNormDist = std::max(dx, dy);

        const FLOAT32 startRatio = std::clamp(transitionStartRatio, 0.5f, 0.99f);
        if (maxNormDist <= startRatio)
        {
            return 0.0f;
        }
        if (maxNormDist >= 1.0f)
        {
            return 1.0f;
        }
        const FLOAT32 t = (maxNormDist - startRatio) / (1.0f - startRatio);
        return t * t * (3.0f - 2.0f * t); // Smoothstep
    }

    // =====================================================================
    //  GPU-Driven Shadow Caster Culling Preparation Contract
    // =====================================================================
    using GPUSceneObjectHeader = GPUSceneObjectData;

    // Determines whether a persistent GPUSceneObjectData record is an active, visible shadow caster.
    inline constexpr BOOL IsGPUSceneObjectValidShadowCaster(const GPUSceneObjectData& obj) noexcept
    {
        constexpr UINT32 kRequiredFlags =
            GPU_SCENE_OBJ_VALID |
            GPU_SCENE_OBJ_VISIBLE |
            GPU_SCENE_OBJ_CAST_SHADOW;
        return (obj.slotIndex != kInvalidGPUSceneSlot) &&
               (obj.generation != 0u) &&
               ((obj.objectFlags & kRequiredFlags) == kRequiredFlags) &&
               (obj.indexCount > 0u || obj.vertexCount > 0u) &&
               (obj.boundingSphere.w > 0.0f);
    }

    // Projects a world-space AABB [worldBoundsMin, worldBoundsMax] into light-space AABB [outLightMin, outLightMax]
    // in O(1) using Arvo's matrix-AABB extent transform.
    inline void ProjectWorldBoundsToNVSLightSpace(
        const Vec3&                 worldBoundsMin,
        const Vec3&                 worldBoundsMax,
        const NVSClipmapLevelState& levelState,
        Vec3&                       outLightBoundsMin,
        Vec3&                       outLightBoundsMax) noexcept
    {
        const Vec3 center = (worldBoundsMin + worldBoundsMax) * 0.5f;
        const Vec3 half   = (worldBoundsMax - worldBoundsMin) * 0.5f;

        const FLOAT32 cx = center.Dot(levelState.lightRight);
        const FLOAT32 cy = center.Dot(levelState.lightUp);
        const FLOAT32 cz = center.Dot(levelState.lightDir);

        const FLOAT32 ex =
            std::fabs(levelState.lightRight.x) * half.x +
            std::fabs(levelState.lightRight.y) * half.y +
            std::fabs(levelState.lightRight.z) * half.z;
        const FLOAT32 ey =
            std::fabs(levelState.lightUp.x) * half.x +
            std::fabs(levelState.lightUp.y) * half.y +
            std::fabs(levelState.lightUp.z) * half.z;
        const FLOAT32 ez =
            std::fabs(levelState.lightDir.x) * half.x +
            std::fabs(levelState.lightDir.y) * half.y +
            std::fabs(levelState.lightDir.z) * half.z;

        outLightBoundsMin = Vec3(cx - ex, cy - ey, cz - ez);
        outLightBoundsMax = Vec3(cx + ex, cy + ey, cz + ez);
    }

    // Returns a 4-bit mask (bit L = 1) of clipmap levels whose active light-space XY window and
    // [casterMinZ, casterMaxZ] range intersect the shadow caster object's bounds.
    inline UINT32 ComputeNVSCandidateClipmapLevelMaskForObject(
        const GPUSceneObjectData&       obj,
        const NVSClipmapHierarchyState& hierarchy) noexcept
    {
        if (!IsGPUSceneObjectValidShadowCaster(obj) || !hierarchy.lightValid)
        {
            return 0u;
        }

        const Vec3& objMin = obj.aabbMin;
        const Vec3& objMax = obj.aabbMax;
        if (objMax.x < objMin.x || objMax.y < objMin.y || objMax.z < objMin.z)
        {
            return 0u;
        }

        UINT32 mask = 0u;
        const UINT32 count = std::min(hierarchy.activeLevelCount, kNVSMaxClipmapLevels);
        for (UINT32 lvl = 0u; lvl < count; ++lvl)
        {
            const NVSClipmapLevelState& ls = hierarchy.levels[lvl];
            if (ls.pagesPerAxis == 0u || ls.worldExtent <= 0.0f)
            {
                continue;
            }
            Vec3 lMin{}, lMax{};
            ProjectWorldBoundsToNVSLightSpace(objMin, objMax, ls, lMin, lMax);

            const INT32   halfPages = static_cast<INT32>(ls.pagesPerAxis / 2u);
            const FLOAT32 pageSize  = ls.worldExtent / static_cast<FLOAT32>(ls.pagesPerAxis);
            const FLOAT32 winMinX   = static_cast<FLOAT32>(ls.snappedOriginPageX - halfPages) * pageSize;
            const FLOAT32 winMinY   = static_cast<FLOAT32>(ls.snappedOriginPageY - halfPages) * pageSize;
            const FLOAT32 winMaxX   = winMinX + ls.worldExtent;
            const FLOAT32 winMaxY   = winMinY + ls.worldExtent;

            const BOOL overlapXY =
                (lMax.x >= winMinX) && (lMin.x <= winMaxX) &&
                (lMax.y >= winMinY) && (lMin.y <= winMaxY);
            const BOOL overlapZ =
                (lMax.z >= ls.casterMinZ) && (lMin.z <= ls.casterMaxZ);
            if (overlapXY && overlapZ)
            {
                mask |= (1u << lvl);
            }
        }
        return mask;
    }

    // Tests whether a persistent GPUSceneObjectData record intersects a resident shadow page's 3D shadow volume
    // using fast world-space sphere/AABB rejection followed by exact light-space 3D AABB overlap.
    inline BOOL DoesObjectIntersectNVSPageShadowVolume(
        const GPUSceneObjectData&     obj,
        const NVSPageRenderData&      pageData,
        const NVSClipmapLevelState&   levelState) noexcept
    {
        if (!IsGPUSceneObjectValidShadowCaster(obj) ||
            (pageData.renderFlags & NVS_PAGE_RENDER_FLAG_VALID_RESIDENT) == 0u)
        {
            return false;
        }

        // 1. Fast bounding sphere rejection in world space
        const Vec3    sphereCenter(obj.boundingSphere.x, obj.boundingSphere.y, obj.boundingSphere.z);
        const FLOAT32 sphereRadius = obj.boundingSphere.w;
        const Vec3    delta        = sphereCenter - pageData.worldCenter;
        const FLOAT32 maxDist      = sphereRadius + pageData.shadowVolumeRadius;
        if (delta.Dot(delta) > (maxDist * maxDist))
        {
            return false;
        }

        // 2. World-space AABB rejection
        const Vec3& objMin = obj.aabbMin;
        const Vec3& objMax = obj.aabbMax;
        if (objMax.x < pageData.worldBoundsMin.x || objMin.x > pageData.worldBoundsMax.x ||
            objMax.y < pageData.worldBoundsMin.y || objMin.y > pageData.worldBoundsMax.y ||
            objMax.z < pageData.worldBoundsMin.z || objMin.z > pageData.worldBoundsMax.z)
        {
            return false;
        }

        // 3. Light-space 3D AABB intersection against [lightMinX, lightMaxX] x [lightMinY, lightMaxY] x [casterMinZ, casterMaxZ]
        Vec3 lMin{}, lMax{};
        ProjectWorldBoundsToNVSLightSpace(objMin, objMax, levelState, lMin, lMax);
        return (lMax.x >= pageData.lightMinX) && (lMin.x <= pageData.lightMaxX) &&
               (lMax.y >= pageData.lightMinY) && (lMin.y <= pageData.lightMaxY) &&
               (lMax.z >= pageData.casterMinZ) && (lMin.z <= pageData.casterMaxZ);
    }

    // =====================================================================
    //  GPU-Driven Shadow Caster Culling & Virtual Page Assignment
    // =====================================================================
    static constexpr UINT32 kNVSCasterCullThreadGroupSize     = 64u;
    static constexpr UINT32 kNVSDefaultCasterWorklistCapacity = 262144u; // 262,144 candidates * 64B = 16 MB
    static constexpr UINT32 kNVSMaxCasterWorklistCapacity     = 1048576u;
    static constexpr UINT32 kNVSMaxFilteredCastersCapacity    = 65536u;
    static constexpr UINT32 kNVSCasterBitmaskWordCount        = (kNVSMaxFilteredCastersCapacity + 31u) / 32u; // 2048 UINT32 words
    static constexpr UINT32 kNVSMaxDynamicInvalidationBoxes   = 4096u;

    enum NVSCasterCullDispatchMode : UINT32
    {
        NVS_CASTER_CULL_MODE_RESET_AND_SORT_PAGES = 0u,
        NVS_CASTER_CULL_MODE_FILTER_SCENE_CASTERS = 1u,
        NVS_CASTER_CULL_MODE_COUNT_PAGE_CASTERS   = 2u,
        NVS_CASTER_CULL_MODE_PREFIX_SUM_RANGES    = 3u,
        NVS_CASTER_CULL_MODE_SCATTER_WORKLIST     = 4u,
        NVS_CASTER_CULL_MODE_DYNAMIC_INVALIDATION = 5u,
        NVS_CASTER_CULL_MODE_VERIFY_MATRIX_CLIP   = 6u
    };

    enum NVSCasterPageFlags : UINT32
    {
        NVS_CASTER_PAGE_FLAG_NONE                = 0u,
        NVS_CASTER_PAGE_FLAG_VALID_ACTIVE        = 1u << 0u,
        NVS_CASTER_PAGE_FLAG_EMPTY_VALID         = 1u << 1u,
        NVS_CASTER_PAGE_FLAG_OVERFLOW_INCOMPLETE = 1u << 2u,
        NVS_CASTER_PAGE_FLAG_NEWLY_ALLOCATED     = 1u << 3u,
        NVS_CASTER_PAGE_FLAG_RESIDENT_CACHE_HIT  = 1u << 4u,

        NVS_CASTER_RANGE_FLAG_VALID_PAGE         = 1u << 0u,
        NVS_CASTER_RANGE_FLAG_HAS_CASTERS        = 0u,
        NVS_CASTER_RANGE_FLAG_OVERFLOW           = 1u << 2u
    };

    enum NVSCasterVariantFlags : UINT32
    {
        NVS_CASTER_VARIANT_OPAQUE                = 0u,
        NVS_CASTER_VARIANT_FLAG_OPAQUE           = 0u,
        NVS_CASTER_VARIANT_ALPHA_TEST            = 1u << 0u,
        NVS_CASTER_VARIANT_DOUBLE_SIDED          = 1u << 1u,
        NVS_CASTER_VARIANT_CUSTOM_SHADOW_PASS    = 1u << 2u,
        NVS_CASTER_VARIANT_INSTANCED_CANDIDATE   = 1u << 3u,
        NVS_CASTER_VARIANT_STATIC_BATCH_MEMBER   = 1u << 4u,
        NVS_CASTER_VARIANT_COMBINED_STATIC_BATCH = 1u << 5u,
        NVS_CASTER_VARIANT_DYNAMIC_CASTER        = 1u << 6u,
        NVS_CASTER_VARIANT_SKINNED               = 1u << 7u
    };

    enum NVSStaticBatchCullPolicy : UINT32
    {
        NVS_STATIC_BATCH_POLICY_PREFER_SUBMESH_MEMBERS = 0u, // Default: cull individual static batch members; skip combined umbrella batch
        NVS_STATIC_BATCH_POLICY_PREFER_COMBINED_BATCH  = 1u  // Alternative: cull combined static batch umbrella; skip individual members
    };

    enum NVSDynamicInvalidationFlags : UINT32
    {
        NVS_DYN_INV_FLAG_NONE             = 0u,
        NVS_DYN_INV_FLAG_OLD_BOUNDS       = 1u << 0u,
        NVS_DYN_INV_FLAG_NEW_BOUNDS       = 1u << 1u,
        NVS_DYN_INV_FLAG_MOVED_OR_SCALED  = 1u << 2u,
        NVS_DYN_INV_FLAG_ADDED_CASTER     = 1u << 3u,
        NVS_DYN_INV_FLAG_ADDED            = 1u << 3u,
        NVS_DYN_INV_FLAG_REMOVED_CASTER   = 1u << 4u,
        NVS_DYN_INV_FLAG_REMOVED          = 1u << 4u,
        NVS_DYN_INV_FLAG_MATERIAL_OR_MESH = 1u << 5u
    };

    // =====================================================================
    //  Per-Page Compacted Caster Range Record (48 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSCasterPageRange
    {
        // Row 0 (Offsets 0..15)
        UINT32 virtualPageFlatIndex   = kNVSInvalidVirtualPage;
        UINT32 physicalPageIndex      = kNVSInvalidPhysicalPage;
        UINT32 virtualAddressPacked   = 0u;
        UINT32 lightId                = 0u;

        // Row 1 (Offsets 16..31)
        UINT32 clipmapLevel           = 0u;
        UINT32 firstCasterOffset      = 0u;
        UINT32 casterCount            = 0u;
        UINT32 rawCandidateCount      = 0u;

        // Row 2 (Offsets 32..47)
        UINT32 pageFlags              = 0u;
        UINT32 broadPhasePassedCount  = 0u;
        UINT32 narrowPhasePassedCount = 0u;
        UINT32 physicalGeneration     = 0u; // NVSPhysicalPageEntry::physicalGeneration snapshot (page validity contract)
    };

    static_assert(sizeof(NVSCasterPageRange) == 48, "NVSCasterPageRange must be 48 bytes");
    static_assert(alignof(NVSCasterPageRange) == 16, "NVSCasterPageRange must be 16-byte aligned");
    static_assert(offsetof(NVSCasterPageRange, virtualPageFlatIndex) == 0, "virtualPageFlatIndex offset mismatch");
    static_assert(offsetof(NVSCasterPageRange, clipmapLevel) == 16, "clipmapLevel offset mismatch");
    static_assert(offsetof(NVSCasterPageRange, firstCasterOffset) == 20, "firstCasterOffset offset mismatch");
    static_assert(offsetof(NVSCasterPageRange, casterCount) == 24, "casterCount offset mismatch");
    static_assert(offsetof(NVSCasterPageRange, pageFlags) == 32, "pageFlags offset mismatch");
    static_assert(offsetof(NVSCasterPageRange, physicalGeneration) == 44, "physicalGeneration offset mismatch");

    // Builds the page validity commit record submitted after rasterizing
    // (or failing to rasterize) the casters of one NVSCasterPageRange.
    // OVERFLOW_INCOMPLETE ranges are never committable, even if rasterization succeeded.
    inline NVSPageRenderCommitRecord BuildNVSPageRenderCommitRecord(
        const NVSCasterPageRange& range,
        BOOL                      rasterizationSucceeded) noexcept
    {
        NVSPageRenderCommitRecord rec{};
        rec.physicalPageIndex          = range.physicalPageIndex;
        rec.expectedPhysicalGeneration = range.physicalGeneration;
        rec.expectedVirtualFlatIndex   = range.virtualPageFlatIndex;
        rec.resultFlags                = rasterizationSucceeded ? NVS_PAGE_RENDER_RESULT_SUCCESS
                                                                : NVS_PAGE_RENDER_RESULT_RASTER_FAILED;
        if ((range.pageFlags & NVS_CASTER_PAGE_FLAG_OVERFLOW_INCOMPLETE) != 0u)
        {
            rec.resultFlags |= NVS_PAGE_RENDER_RESULT_CASTER_LIST_INCOMPLETE;
        }
        return rec;
    }

    // =====================================================================
    //  Compacted Page-Caster Draw Candidate Record (64 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSCasterDrawCandidate
    {
        // Row 0 (Offsets 0..15): Scene & Page Identity
        UINT32 gpuSceneSlotIndex      = kInvalidGPUSceneSlot;
        UINT32 renderObjectHandle     = 0u; // Generation (upper 16 bits) + Slot (lower 16 bits) or generation ID
        UINT32 virtualPageFlatIndex   = kNVSInvalidVirtualPage;
        UINT32 physicalPageIndex      = kNVSInvalidPhysicalPage;

        // Row 1 (Offsets 16..31): Virtual Address, Light/Level & Mesh
        UINT32 virtualAddressPacked   = 0u;
        UINT32 lightIdAndClipmapLevel = 0u; // (lightId & 0xFFFFu) | ((clipmapLevel & 0xFFFFu) << 16u)
        UINT32 meshId                 = 0u;
        UINT32 submeshIndex           = 0u;

        // Row 2 (Offsets 32..47): Material, Pipeline Variant, LOD & Instancing
        UINT32 materialId             = 0u;
        UINT32 pipelineVariantFlags   = 0u; // NVSCasterVariantFlags bitmask
        UINT32 selectedLODIndex       = 0u;
        UINT32 instanceGroupId        = 0u;

        // Row 3 (Offsets 48..63): Batch Group, Draw Topology & Depth Sort Key
        UINT32 batchGroupId           = 0u;
        UINT32 startIndex             = 0u;
        UINT32 indexCount             = 0u;
        FLOAT32 casterDepthSortKey    = 0.0f; // Light-space minimum Z (lMin.z)
    };

    static_assert(sizeof(NVSCasterDrawCandidate) == 64, "NVSCasterDrawCandidate must be 64 bytes");
    static_assert(alignof(NVSCasterDrawCandidate) == 16, "NVSCasterDrawCandidate must be 16-byte aligned");
    static_assert(offsetof(NVSCasterDrawCandidate, gpuSceneSlotIndex) == 0, "gpuSceneSlotIndex offset mismatch");
    static_assert(offsetof(NVSCasterDrawCandidate, virtualAddressPacked) == 16, "virtualAddressPacked offset mismatch");
    static_assert(offsetof(NVSCasterDrawCandidate, materialId) == 32, "materialId offset mismatch");
    static_assert(offsetof(NVSCasterDrawCandidate, batchGroupId) == 48, "batchGroupId offset mismatch");
    static_assert(offsetof(NVSCasterDrawCandidate, casterDepthSortKey) == 60, "casterDepthSortKey offset mismatch");

    // =====================================================================
    //  Pre-Filtered GPU Caster Cache Record (80 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSFilteredCasterObject
    {
        // Row 0 (Offsets 0..15): World AABB Min + Bounding Sphere Radius
        Vec3    aabbMin             = Vec3(0.0f, 0.0f, 0.0f);
        FLOAT32 boundingRadius      = 0.0f;

        // Row 1 (Offsets 16..31): World AABB Max + 16-bit Candidate Light/Level Mask
        Vec3    aabbMax             = Vec3(0.0f, 0.0f, 0.0f);
        UINT32  candidateLevelMask  = 0u; // Bit (lightId * 4 + level)

        // Row 2 (Offsets 32..47): World Sphere Center + GPU Scene Slot Index
        Vec3    sphereCenter        = Vec3(0.0f, 0.0f, 0.0f);
        UINT32  gpuSceneSlotIndex   = kInvalidGPUSceneSlot;

        // Row 3 (Offsets 48..63): Precomputed Light 0 Light-Space XY Bounds
        FLOAT32 light0MinX          = 0.0f;
        FLOAT32 light0MaxX          = 0.0f;
        FLOAT32 light0MinY          = 0.0f;
        FLOAT32 light0MaxY          = 0.0f;

        // Row 4 (Offsets 64..79): Precomputed Light 0 Light-Space Z Bounds + Packed Flags
        FLOAT32 light0MinZ          = 0.0f;
        FLOAT32 light0MaxZ          = 0.0f;
        UINT32  objectFlags         = 0u;
        UINT32  variantAndLodPacked = 0u; // Lower 16 bits: NVSCasterVariantFlags, Upper 16 bits: maxLodCount
    };

    static_assert(sizeof(NVSFilteredCasterObject) == 80, "NVSFilteredCasterObject must be 80 bytes");
    static_assert(alignof(NVSFilteredCasterObject) == 16, "NVSFilteredCasterObject must be 16-byte aligned");

    // =====================================================================
    //  Dynamic Shadow Caster Cache Invalidation Box (32 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSDynamicCasterInvalidationBox
    {
        Vec3   worldMin          = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 gpuSceneSlotIndex = kInvalidGPUSceneSlot;
        Vec3   worldMax          = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 invalidationFlags = 0u; // NVSDynamicInvalidationFlags bitmask
    };

    static_assert(sizeof(NVSDynamicCasterInvalidationBox) == 32, "NVSDynamicCasterInvalidationBox must be 32 bytes");
    static_assert(alignof(NVSDynamicCasterInvalidationBox) == 16, "NVSDynamicCasterInvalidationBox must be 16-byte aligned");

    // =====================================================================
    //  GPU Shadow Caster Culling & Worklist Telemetry (80 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSCasterCullTelemetry
    {
        // Row 0 (Offsets 0..15): Scene Caster Filtering
        UINT32 totalSceneObjectsScanned    = 0u;
        UINT32 validShadowCastersCount     = 0u;
        UINT32 rejectedDisabledOrHidden    = 0u;
        UINT32 rejectedNonShadowCaster     = 0u;

        // Row 1 (Offsets 16..31): Scene Rejection Reasons & Active Pages
        UINT32 rejectedInvalidBoundsOrGeom = 0u;
        UINT32 rejectedStaticBatchDedup    = 0u;
        UINT32 rejectedTransparentMaterial = 0u;
        UINT32 activeShadowPagesProcessed  = 0u;

        // Row 2 (Offsets 32..47): Broad-Phase & Narrow-Phase Intersection Counts
        UINT32 totalPageCasterTests        = 0u;
        UINT32 broadPhasePassedCount       = 0u;
        UINT32 broadPhaseRejectedCount     = 0u;
        UINT32 narrowPhasePassedCount      = 0u;

        // Row 3 (Offsets 48..63): Narrow-Phase Rejections & Worklist Emission
        UINT32 narrowPhaseRejectedCount    = 0u;
        UINT32 totalAssignedCasterDraws    = 0u;
        UINT32 overflowDroppedAssignments  = 0u;
        UINT32 emptyActivePageCount        = 0u;

        // Row 4 (Offsets 64..79): Page Summary & Dynamic Cache Invalidation
        UINT32 nonEmptyActivePageCount     = 0u;
        UINT32 overflowIncompletePageCount = 0u;
        UINT32 maxCastersPerPage           = 0u;
        UINT32 dynamicInvalidatedPageCount = 0u;
    };

    static_assert(sizeof(NVSCasterCullTelemetry) == 80, "NVSCasterCullTelemetry must be 80 bytes");
    static_assert(alignof(NVSCasterCullTelemetry) == 16, "NVSCasterCullTelemetry must be 16-byte aligned");

    // =====================================================================
    //  Caster Culling Pass Configuration Descriptor
    // =====================================================================
    struct NVSCasterCullPassDesc
    {
        UINT32     currentFrame                  = 1u;
        UINT32     staticBatchPolicy             = NVS_STATIC_BATCH_POLICY_PREFER_SUBMESH_MEMBERS;
        FLOAT32    minCasterProjectedTexels      = 0.0f;  // 0.0f = disabled; > 0.0f = cull SMALL_OBJECT_CULL casters smaller than threshold
        FLOAT32    lodThresholdScale             = 1.0f;
        UINT32     forceLodOverride              = 0u;    // (1u << 8) | forcedLodIndex to override automatic LOD selection
        BOOL       skipTransparentCasters        = true;
        BOOL       disablePrepassFilter          = false; // True = run baseline uncached O(Pages * RawSceneSlots) scan for A/B benchmark
        BOOL       skipAlreadyRenderedValidPages = false; // True = skip pages already in VALID_SHADOW_DATA state
        GPUBuffer* overrideGPUSceneBuffer        = nullptr;
        UINT32     overrideSceneSlotCount        = 0u;
    };

    // =====================================================================
    //  Preflight 2.1 Helper: Transform World Point by NVSPageRenderData Columns
    //  Matches NSL column-major matrix multiplication (col0*P.x + col1*P.y + col2*P.z + col3)
    // =====================================================================
    inline Vec4 TransformWorldPointByNVSPageClipColumns(
        const NVSPageRenderData& pageData,
        const Vec3&              worldPos) noexcept
    {
        const Vec4 col0 = pageData.worldToPageClipMatrix.GetColumn(0);
        const Vec4 col1 = pageData.worldToPageClipMatrix.GetColumn(1);
        const Vec4 col2 = pageData.worldToPageClipMatrix.GetColumn(2);
        const Vec4 col3 = pageData.worldToPageClipMatrix.GetColumn(3);
        return Vec4(
            col0.x * worldPos.x + col1.x * worldPos.y + col2.x * worldPos.z + col3.x,
            col0.y * worldPos.x + col1.y * worldPos.y + col2.y * worldPos.z + col3.y,
            col0.z * worldPos.x + col1.z * worldPos.y + col2.z * worldPos.z + col3.z,
            col0.w * worldPos.x + col1.w * worldPos.y + col2.w * worldPos.z + col3.w);
    }

    // =====================================================================
    //  Deterministic LOD Selection (Matches GPU RuntimeNVSCasterCull.nsl 1:1)
    // =====================================================================
    inline UINT32 SelectNVSShadowCasterLODLevel(
        UINT32  objectFlags,
        UINT32  lodAndMaterialFlags,
        FLOAT32 boundingRadius,
        UINT32  clipmapLevel,
        FLOAT32 pageTexelWorldSize,
        FLOAT32 lodThresholdScale,
        UINT32  forceLodOverride = 0u) noexcept
    {
        const UINT32 rawLodCount = UnpackGPUSceneLODCount(lodAndMaterialFlags);
        const UINT32 maxLod      = (rawLodCount > 1u) ? std::min(rawLodCount - 1u, 3u) : 0u;
        if ((forceLodOverride & 256u) != 0u)
        {
            const UINT32 forced = forceLodOverride & 255u;
            return std::min(forced, maxLod);
        }
        if ((objectFlags & GPU_SCENE_OBJ_USE_LOD) == 0u || maxLod == 0u)
        {
            return 0u;
        }

        const UINT32 levelLod = std::min(clipmapLevel, maxLod);
        if (lodThresholdScale > 0.0f && pageTexelWorldSize > 0.0f && boundingRadius > 0.0f)
        {
            const FLOAT32 denom      = pageTexelWorldSize * lodThresholdScale;
            const FLOAT32 diamTexels = (denom > 1e-8f) ? ((2.0f * boundingRadius) / denom) : 1024.0f;
            UINT32        texelLod   = 0u;
            if (diamTexels < 16.0f)
            {
                texelLod = 3u;
            }
            else if (diamTexels < 32.0f)
            {
                texelLod = 2u;
            }
            else if (diamTexels < 64.0f)
            {
                texelLod = 1u;
            }
            return std::min(std::max(levelLod, texelLod), maxLod);
        }
        return levelLod;
    }

    // =====================================================================
    //  Shadow Caster Eligibility & Variant Classification (CPU Reference)
    //  Matches GPU RuntimeNVSCasterCull.nsl Mode 1 1:1.
    // =====================================================================
    enum NVSCasterRejectionReason : UINT32
    {
        NVS_CASTER_ELIGIBLE                     = 0u,
        NVS_CASTER_REJECT_DISABLED_OR_HIDDEN    = 1u,
        NVS_CASTER_REJECT_NON_SHADOW_CASTER     = 2u,
        NVS_CASTER_REJECT_INVALID_BOUNDS_GEOM   = 3u,
        NVS_CASTER_REJECT_STATIC_BATCH_DEDUP    = 4u,
        NVS_CASTER_REJECT_TRANSPARENT_MATERIAL  = 5u
    };

    inline NVSCasterRejectionReason EvaluateNVSShadowCasterEligibilityCPU(
        UINT32                    expectedSlotIndex,
        const GPUSceneObjectData& obj,
        UINT32                    staticBatchPolicy     = NVS_STATIC_BATCH_POLICY_PREFER_SUBMESH_MEMBERS,
        BOOL                      skipTransparentCaster = true,
        UINT32*                   outVariantFlags       = nullptr) noexcept
    {
        if (outVariantFlags)
        {
            *outVariantFlags = NVS_CASTER_VARIANT_OPAQUE;
        }

        // 1. Slot occupancy, generation & scene-level visibility
        //    Note: GPU_SCENE_OBJ_FRUSTUM_CULL is NEVER used to reject off-screen shadow casters.
        if (obj.slotIndex != expectedSlotIndex ||
            obj.slotIndex == kInvalidGPUSceneSlot ||
            obj.generation == 0u ||
            (obj.objectFlags & GPU_SCENE_OBJ_VALID) == 0u ||
            (obj.objectFlags & GPU_SCENE_OBJ_VISIBLE) == 0u)
        {
            return NVS_CASTER_REJECT_DISABLED_OR_HIDDEN;
        }

        // 2. Shadow caster enable bit
        if ((obj.objectFlags & GPU_SCENE_OBJ_CAST_SHADOW) == 0u)
        {
            return NVS_CASTER_REJECT_NON_SHADOW_CASTER;
        }

        // 3. Valid finite bounds, mesh & material
        const BOOL finiteBounds =
            std::isfinite(obj.aabbMin.x) && std::isfinite(obj.aabbMin.y) && std::isfinite(obj.aabbMin.z) &&
            std::isfinite(obj.aabbMax.x) && std::isfinite(obj.aabbMax.y) && std::isfinite(obj.aabbMax.z) &&
            std::isfinite(obj.boundingSphere.x) && std::isfinite(obj.boundingSphere.y) &&
            std::isfinite(obj.boundingSphere.z) && std::isfinite(obj.boundingSphere.w);

        if (!finiteBounds ||
            (obj.objectFlags & GPU_SCENE_OBJ_HAS_VALID_BOUNDS) == 0u ||
            obj.aabbMax.x < obj.aabbMin.x ||
            obj.aabbMax.y < obj.aabbMin.y ||
            obj.aabbMax.z < obj.aabbMin.z ||
            !(obj.boundingSphere.w > 0.0f) ||
            obj.meshId == 0u ||
            (obj.indexCount == 0u && obj.vertexCount == 0u) ||
            obj.materialId == 0u)
        {
            return NVS_CASTER_REJECT_INVALID_BOUNDS_GEOM;
        }

        // 4. Static Batch deduplication (individual member vs. combined batch umbrella)
        const BOOL isBatchMember     = (obj.objectFlags & GPU_SCENE_OBJ_STATIC_BATCH_MEMBER) != 0u || (obj.batchGroupId != 0u);
        const BOOL isCombinedUmbrella = ((obj.objectFlags & GPU_SCENE_OBJ_BAKED_STATIC) != 0u) && !isBatchMember;
        if (staticBatchPolicy == NVS_STATIC_BATCH_POLICY_PREFER_SUBMESH_MEMBERS)
        {
            if (isCombinedUmbrella)
            {
                return NVS_CASTER_REJECT_STATIC_BATCH_DEDUP;
            }
        }
        else if (staticBatchPolicy == NVS_STATIC_BATCH_POLICY_PREFER_COMBINED_BATCH)
        {
            if (isBatchMember)
            {
                return NVS_CASTER_REJECT_STATIC_BATCH_DEDUP;
            }
        }

        // 5. Transparent material rejection (purely transparent without alpha-test or explicit shadow pass)
        const BOOL isTransparent = (obj.objectFlags & GPU_SCENE_OBJ_TRANSPARENT) != 0u;
        const BOOL isAlphaTest   = (obj.objectFlags & GPU_SCENE_OBJ_ALPHA_TESTED) != 0u;
        const BOOL hasShadowPass = ((obj.objectFlags & GPU_SCENE_OBJ_SHADOW_PASS) != 0u) ||
                                   UnpackGPUSceneHasShadowPass(obj.lodAndMaterialFlags);
        if (skipTransparentCaster && isTransparent && !isAlphaTest && !hasShadowPass)
        {
            return NVS_CASTER_REJECT_TRANSPARENT_MATERIAL;
        }

        if (outVariantFlags)
        {
            UINT32 vFlags = NVS_CASTER_VARIANT_OPAQUE;
            if (isAlphaTest)
            {
                vFlags |= NVS_CASTER_VARIANT_ALPHA_TEST;
            }
            if (((obj.objectFlags & GPU_SCENE_OBJ_TWO_SIDED) != 0u) ||
                (UnpackGPUSceneCullMode(obj.lodAndMaterialFlags) == 0u))
            {
                vFlags |= NVS_CASTER_VARIANT_DOUBLE_SIDED;
            }
            if (hasShadowPass)
            {
                vFlags |= NVS_CASTER_VARIANT_CUSTOM_SHADOW_PASS;
            }
            if ((obj.objectFlags & GPU_SCENE_OBJ_INSTANCED_CANDIDATE) != 0u)
            {
                vFlags |= NVS_CASTER_VARIANT_INSTANCED_CANDIDATE;
            }
            if (isBatchMember)
            {
                vFlags |= NVS_CASTER_VARIANT_STATIC_BATCH_MEMBER;
            }
            if (isCombinedUmbrella)
            {
                vFlags |= NVS_CASTER_VARIANT_COMBINED_STATIC_BATCH;
            }
            if ((obj.objectFlags & GPU_SCENE_OBJ_STATIC) == 0u)
            {
                vFlags |= NVS_CASTER_VARIANT_DYNAMIC_CASTER;
            }
            if ((obj.objectFlags & GPU_SCENE_OBJ_SKINNED) != 0u)
            {
                vFlags |= NVS_CASTER_VARIANT_SKINNED;
            }
            *outVariantFlags = vFlags;
        }

        return NVS_CASTER_ELIGIBLE;
    }

    // =====================================================================
    //  Broad-Phase & Narrow-Phase Page-Caster Intersection (CPU Reference)
    //  Matches GPU RuntimeNVSCasterCull.nsl Mode 2 & Mode 4 1:1.
    // =====================================================================
    inline BOOL DoesCasterPassNVSPageBroadAndNarrowPhaseCPU(
        const GPUSceneObjectData&   obj,
        UINT32                      candidateLevelMask,
        const NVSPageRenderData&    pageData,
        const NVSClipmapLevelState& levelState,
        FLOAT32                     minCasterProjectedTexels,
        BOOL&                       outBroadPhasePassed,
        FLOAT32&                    outCasterDepthSortKey) noexcept
    {
        outBroadPhasePassed   = false;
        outCasterDepthSortKey = 0.0f;

        if ((pageData.renderFlags & NVS_PAGE_RENDER_FLAG_VALID_RESIDENT) == 0u)
        {
            return false;
        }

        // Broad-Phase Step 1: 16-bit candidate (lightId, clipmapLevel) bitmask test
        const UINT32 bitIdx = (pageData.lightId & 3u) * 4u + (pageData.clipmapLevel & 3u);
        if ((candidateLevelMask & (1u << bitIdx)) == 0u)
        {
            return false;
        }

        // Broad-Phase Step 2: World-space bounding sphere vs. page 3D shadow volume sphere
        const Vec3    sphereCenter(obj.boundingSphere.x, obj.boundingSphere.y, obj.boundingSphere.z);
        const FLOAT32 sphereRadius = obj.boundingSphere.w;
        const Vec3    delta        = sphereCenter - pageData.worldCenter;
        const FLOAT32 maxDist      = sphereRadius + pageData.shadowVolumeRadius;
        if (delta.Dot(delta) > (maxDist * maxDist))
        {
            return false;
        }

        // Broad-Phase Step 3: World-space AABB vs. page 3D world AABB
        if (obj.aabbMax.x < pageData.worldBoundsMin.x || obj.aabbMin.x > pageData.worldBoundsMax.x ||
            obj.aabbMax.y < pageData.worldBoundsMin.y || obj.aabbMin.y > pageData.worldBoundsMax.y ||
            obj.aabbMax.z < pageData.worldBoundsMin.z || obj.aabbMin.z > pageData.worldBoundsMax.z)
        {
            return false;
        }

        outBroadPhasePassed = true;

        // Narrow-Phase Step 1: Arvo O(1) light-space 3D AABB intersection against
        // [lightMinX, lightMaxX] x [lightMinY, lightMaxY] x [casterMinZ, casterMaxZ]
        Vec3 lMin{}, lMax{};
        ProjectWorldBoundsToNVSLightSpace(obj.aabbMin, obj.aabbMax, levelState, lMin, lMax);
        if (lMax.x < pageData.lightMinX || lMin.x > pageData.lightMaxX ||
            lMax.y < pageData.lightMinY || lMin.y > pageData.lightMaxY ||
            lMax.z < pageData.casterMinZ || lMin.z > pageData.casterMaxZ)
        {
            return false;
        }

        // Narrow-Phase Step 2: Optional small-caster projected texel threshold
        if (minCasterProjectedTexels > 0.0f &&
            (obj.objectFlags & GPU_SCENE_OBJ_SMALL_OBJECT_CULL) != 0u &&
            pageData.texelWorldSize > 1e-8f)
        {
            const FLOAT32 projWidthTexels  = (lMax.x - lMin.x) / pageData.texelWorldSize;
            const FLOAT32 projHeightTexels = (lMax.y - lMin.y) / pageData.texelWorldSize;
            const FLOAT32 maxProjTexels    = std::max(projWidthTexels, projHeightTexels);
            if (maxProjTexels < minCasterProjectedTexels)
            {
                return false;
            }
        }

        outCasterDepthSortKey = lMin.z;
        return true;
    }

    // =====================================================================
    //  Full CPU Reference Worklist Builder (Matches GPU 1:1)
    // =====================================================================
    inline void BuildNVSCasterWorklistCPU(
        const GPUSceneObjectData*        sceneObjects,
        UINT32                           sceneSlotCount,
        const NVSPageRenderData*         pageRenderDataTable,
        UINT32                           pageRenderDataCapacity,
        const UINT32*                    activePhysicalPageIndices,
        UINT32                           activePageCount,
        const NVSClipmapHierarchyState*  lightHierarchies,
        UINT32                           lightHierarchyCount,
        const NVSCasterCullPassDesc&     passDesc,
        UINT32                           maxWorklistCapacity,
        NVSCasterPageRange*              outPageRanges,
        NVSCasterDrawCandidate*          outWorklist,
        NVSCasterCullTelemetry&          outTelemetry) noexcept
    {
        outTelemetry = NVSCasterCullTelemetry{};
        outTelemetry.totalSceneObjectsScanned = sceneSlotCount;

        // 1. Collect and deterministically sort active physical page indices in ascending order
        UINT32 sortedPhysPages[kNVSMaxPhysicalPoolPages]{};
        UINT32 validActivePageCount = 0u;
        const UINT32 clampedActiveCount = std::min(activePageCount, kNVSMaxPhysicalPoolPages);
        for (UINT32 i = 0u; i < clampedActiveCount; ++i)
        {
            const UINT32 physIdx = activePhysicalPageIndices ? activePhysicalPageIndices[i] : i;
            if (physIdx < pageRenderDataCapacity)
            {
                const NVSPageRenderData& pd = pageRenderDataTable[physIdx];
                if ((pd.renderFlags & NVS_PAGE_RENDER_FLAG_VALID_RESIDENT) != 0u)
                {
                    if (passDesc.skipAlreadyRenderedValidPages &&
                        (pd.renderFlags & NVS_PAGE_RENDER_FLAG_CACHED_VALID_SHADOW) != 0u &&
                        (pd.renderFlags & NVS_PAGE_RENDER_FLAG_NEEDS_CASTER_RENDER) == 0u)
                    {
                        continue;
                    }
                    sortedPhysPages[validActivePageCount++] = physIdx;
                }
            }
        }
        std::sort(sortedPhysPages, sortedPhysPages + validActivePageCount);
        outTelemetry.activeShadowPagesProcessed = validActivePageCount;

        // 2. Filter scene shadow casters & compute 16-bit candidate (lightId, level) bitmask
        std::vector<UINT32> validCasterSlots;
        std::vector<UINT32> casterCandidateMasks(sceneSlotCount, 0u);
        std::vector<UINT32> casterVariantFlags(sceneSlotCount, 0u);
        validCasterSlots.reserve(sceneSlotCount);

        for (UINT32 slot = 0u; slot < sceneSlotCount; ++slot)
        {
            const GPUSceneObjectData& obj = sceneObjects[slot];
            UINT32 variantFlags = 0u;
            const NVSCasterRejectionReason reason = EvaluateNVSShadowCasterEligibilityCPU(
                slot,
                obj,
                passDesc.staticBatchPolicy,
                passDesc.skipTransparentCasters,
                &variantFlags);

            if (reason == NVS_CASTER_REJECT_DISABLED_OR_HIDDEN)
            {
                outTelemetry.rejectedDisabledOrHidden++;
                continue;
            }
            if (reason == NVS_CASTER_REJECT_NON_SHADOW_CASTER)
            {
                outTelemetry.rejectedNonShadowCaster++;
                continue;
            }
            if (reason == NVS_CASTER_REJECT_INVALID_BOUNDS_GEOM)
            {
                outTelemetry.rejectedInvalidBoundsOrGeom++;
                continue;
            }
            if (reason == NVS_CASTER_REJECT_STATIC_BATCH_DEDUP)
            {
                outTelemetry.rejectedStaticBatchDedup++;
                continue;
            }
            if (reason == NVS_CASTER_REJECT_TRANSPARENT_MATERIAL)
            {
                outTelemetry.rejectedTransparentMaterial++;
                continue;
            }

            UINT32 mask16 = 0u;
            const UINT32 numLights = std::min(lightHierarchyCount, 4u);
            for (UINT32 l = 0u; l < numLights; ++l)
            {
                const NVSClipmapHierarchyState& h = lightHierarchies[l];
                if (!h.lightValid)
                {
                    continue;
                }
                const UINT32 numLevels = std::min(h.activeLevelCount, kNVSMaxClipmapLevels);
                for (UINT32 lvl = 0u; lvl < numLevels; ++lvl)
                {
                    const NVSClipmapLevelState& ls = h.levels[lvl];
                    if (ls.pagesPerAxis == 0u || ls.worldExtent <= 0.0f)
                    {
                        continue;
                    }
                    Vec3 lMin{}, lMax{};
                    ProjectWorldBoundsToNVSLightSpace(obj.aabbMin, obj.aabbMax, ls, lMin, lMax);
                    const INT32   halfPages = static_cast<INT32>(ls.pagesPerAxis / 2u);
                    const FLOAT32 pageSize  = ls.worldExtent / static_cast<FLOAT32>(ls.pagesPerAxis);
                    const FLOAT32 winMinX   = static_cast<FLOAT32>(ls.snappedOriginPageX - halfPages) * pageSize;
                    const FLOAT32 winMinY   = static_cast<FLOAT32>(ls.snappedOriginPageY - halfPages) * pageSize;
                    const FLOAT32 winMaxX   = winMinX + ls.worldExtent;
                    const FLOAT32 winMaxY   = winMinY + ls.worldExtent;
                    if (lMax.x >= winMinX && lMin.x <= winMaxX &&
                        lMax.y >= winMinY && lMin.y <= winMaxY &&
                        lMax.z >= ls.casterMinZ && lMin.z <= ls.casterMaxZ)
                    {
                        mask16 |= (1u << (l * 4u + lvl));
                    }
                }
            }

            outTelemetry.validShadowCastersCount++;
            casterCandidateMasks[slot] = mask16;
            casterVariantFlags[slot]   = variantFlags;
            validCasterSlots.push_back(slot);
        }

        // 3. Per-page broad-phase & narrow-phase evaluation + prefix-sum worklist compaction
        UINT32 runningOffset = 0u;
        for (UINT32 pIdx = 0u; pIdx < validActivePageCount; ++pIdx)
        {
            const UINT32             physIdx  = sortedPhysPages[pIdx];
            const NVSPageRenderData& pageData = pageRenderDataTable[physIdx];

            NVSClipmapLevelState levelState{};
            if (pageData.lightId < lightHierarchyCount && pageData.clipmapLevel < kNVSMaxClipmapLevels)
            {
                levelState = lightHierarchies[pageData.lightId].levels[pageData.clipmapLevel];
            }

            UINT32 pageBroadPassed  = 0u;
            UINT32 pageNarrowPassed = 0u;
            std::vector<NVSCasterDrawCandidate> pageCandidates;

            for (UINT32 cIdx = 0u; cIdx < static_cast<UINT32>(validCasterSlots.size()); ++cIdx)
            {
                const UINT32              slot = validCasterSlots[cIdx];
                const GPUSceneObjectData& obj  = sceneObjects[slot];
                outTelemetry.totalPageCasterTests++;

                BOOL    broadPassed  = false;
                FLOAT32 depthSortKey = 0.0f;
                const BOOL narrowPassed = DoesCasterPassNVSPageBroadAndNarrowPhaseCPU(
                    obj,
                    casterCandidateMasks[slot],
                    pageData,
                    levelState,
                    passDesc.minCasterProjectedTexels,
                    broadPassed,
                    depthSortKey);

                if (broadPassed)
                {
                    pageBroadPassed++;
                    outTelemetry.broadPhasePassedCount++;
                }
                else
                {
                    outTelemetry.broadPhaseRejectedCount++;
                }

                if (narrowPassed)
                {
                    pageNarrowPassed++;
                    outTelemetry.narrowPhasePassedCount++;

                    NVSCasterDrawCandidate cand{};
                    cand.gpuSceneSlotIndex      = slot;
                    cand.renderObjectHandle     = (obj.generation > 0xFFFFu)
                        ? obj.generation
                        : (((obj.generation & 0xFFFFu) << 16u) | (slot & 0xFFFFu));
                    cand.virtualPageFlatIndex   = pageData.virtualFlatIndex;
                    cand.physicalPageIndex      = pageData.physicalPageIndex;
                    cand.virtualAddressPacked   = pageData.virtualAddressPacked;
                    cand.lightIdAndClipmapLevel = (pageData.lightId & 0xFFFFu) | ((pageData.clipmapLevel & 0xFFFFu) << 16u);
                    cand.meshId                 = obj.meshId;
                    cand.submeshIndex           = obj.submeshIndex;
                    cand.materialId             = obj.materialId;
                    cand.pipelineVariantFlags   = casterVariantFlags[slot];
                    cand.selectedLODIndex       = SelectNVSShadowCasterLODLevel(
                        obj.objectFlags,
                        obj.lodAndMaterialFlags,
                        obj.boundingSphere.w,
                        pageData.clipmapLevel,
                        pageData.texelWorldSize,
                        passDesc.lodThresholdScale,
                        passDesc.forceLodOverride);
                    cand.instanceGroupId        = ((obj.objectFlags & GPU_SCENE_OBJ_INSTANCED_CANDIDATE) != 0u)
                        ? ((obj.meshId * 65537u) ^ obj.materialId)
                        : 0u;
                    cand.batchGroupId           = obj.batchGroupId;
                    cand.startIndex             = obj.startIndex;
                    cand.indexCount             = (obj.indexCount > 0u) ? obj.indexCount : obj.vertexCount;
                    cand.casterDepthSortKey     = depthSortKey;
                    pageCandidates.push_back(cand);
                }
                else if (broadPassed)
                {
                    outTelemetry.narrowPhaseRejectedCount++;
                }
            }

            const UINT32 rawCount  = pageNarrowPassed;
            UINT32       emitCount = rawCount;
            UINT32       pFlags    = NVS_CASTER_PAGE_FLAG_VALID_ACTIVE;
            if ((pageData.renderFlags & NVS_PAGE_RENDER_FLAG_NEWLY_ALLOCATED) != 0u)
            {
                pFlags |= NVS_CASTER_PAGE_FLAG_NEWLY_ALLOCATED;
            }
            if ((pageData.renderFlags & NVS_PAGE_RENDER_FLAG_CACHED_VALID_SHADOW) != 0u)
            {
                pFlags |= NVS_CASTER_PAGE_FLAG_RESIDENT_CACHE_HIT;
            }

            if (rawCount == 0u)
            {
                pFlags |= NVS_CASTER_PAGE_FLAG_EMPTY_VALID;
                outTelemetry.emptyActivePageCount++;
            }
            else
            {
                outTelemetry.nonEmptyActivePageCount++;
                if (runningOffset >= maxWorklistCapacity)
                {
                    emitCount = 0u;
                    pFlags |= NVS_CASTER_PAGE_FLAG_OVERFLOW_INCOMPLETE;
                    outTelemetry.overflowIncompletePageCount++;
                    outTelemetry.overflowDroppedAssignments += rawCount;
                }
                else if (runningOffset + rawCount > maxWorklistCapacity)
                {
                    emitCount = maxWorklistCapacity - runningOffset;
                    pFlags |= NVS_CASTER_PAGE_FLAG_OVERFLOW_INCOMPLETE;
                    outTelemetry.overflowIncompletePageCount++;
                    outTelemetry.overflowDroppedAssignments += (rawCount - emitCount);
                }
            }

            if (outPageRanges)
            {
                NVSCasterPageRange& range    = outPageRanges[pIdx];
                range.virtualPageFlatIndex   = pageData.virtualFlatIndex;
                range.physicalPageIndex      = pageData.physicalPageIndex;
                range.virtualAddressPacked   = pageData.virtualAddressPacked;
                range.lightId                = pageData.lightId;
                range.clipmapLevel           = pageData.clipmapLevel;
                range.firstCasterOffset      = runningOffset;
                range.casterCount            = emitCount;
                range.rawCandidateCount      = rawCount;
                range.pageFlags              = pFlags;
                range.broadPhasePassedCount  = pageBroadPassed;
                range.narrowPhasePassedCount = pageNarrowPassed;
                range.physicalGeneration     = pageData.physicalGeneration;
            }

            if (outWorklist && emitCount > 0u)
            {
                for (UINT32 k = 0u; k < emitCount; ++k)
                {
                    outWorklist[runningOffset + k] = pageCandidates[k];
                }
            }

            runningOffset += emitCount;
            outTelemetry.totalAssignedCasterDraws += emitCount;
            if (rawCount > outTelemetry.maxCastersPerPage)
            {
                outTelemetry.maxCastersPerPage = rawCount;
            }
        }
    }

    inline void BuildNVSCasterWorklistCPU(
        const GPUSceneObjectData*            sceneObjects,
        UINT32                               sceneSlotCount,
        const NVSClipmapLevelGPUData*        clipmapTable,
        UINT32                               clipmapTableCount,
        const NVSPageRenderData*             pageRenderDataTable,
        UINT32                               pageRenderDataCapacity,
        const NVSCasterCullPassDesc&         passDesc,
        UINT32                               maxWorklistCapacity,
        std::vector<NVSCasterPageRange>&     outPageRanges,
        std::vector<NVSCasterDrawCandidate>& outWorklist,
        NVSCasterCullTelemetry&              outTelemetry)
    {
        NVSClipmapHierarchyState lightHierarchies[kNVSMaxClipmapLights]{};
        const UINT32 clampedTableCount = std::min(clipmapTableCount, kNVSClipmapTableMaxEntries);
        for (UINT32 t = 0u; t < clampedTableCount; ++t)
        {
            const NVSClipmapLevelGPUData& gpuLvl = clipmapTable[t];
            const UINT32 lid = t / kNVSMaxClipmapLevels;
            const UINT32 lvl = t % kNVSMaxClipmapLevels;
            if (lid >= kNVSMaxClipmapLights)
            {
                continue;
            }
            NVSClipmapHierarchyState& h = lightHierarchies[lid];
            h.lightId = lid;
            if ((gpuLvl.flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) != 0u && gpuLvl.pagesPerAxis > 0u && gpuLvl.worldExtent > 0.0f)
            {
                h.lightValid       = true;
                h.pagesPerAxis     = gpuLvl.pagesPerAxis;
                h.activeLevelCount = std::max(h.activeLevelCount, lvl + 1u);
                h.lightRight       = Vec3(gpuLvl.lightRightAndCenterX.x, gpuLvl.lightRightAndCenterX.y, gpuLvl.lightRightAndCenterX.z);
                h.lightUp          = Vec3(gpuLvl.lightUpAndCenterY.x, gpuLvl.lightUpAndCenterY.y, gpuLvl.lightUpAndCenterY.z);
                h.lightDir         = Vec3(gpuLvl.lightDirAndExtrusion.x, gpuLvl.lightDirAndExtrusion.y, gpuLvl.lightDirAndExtrusion.z);

                NVSClipmapLevelState& ls     = h.levels[lvl];
                ls.lightId                   = lid;
                ls.clipmapLevel              = lvl;
                ls.pagesPerAxis              = gpuLvl.pagesPerAxis;
                ls.snappedOriginPageX        = gpuLvl.snappedOriginPageX;
                ls.snappedOriginPageY        = gpuLvl.snappedOriginPageY;
                ls.worldExtent               = gpuLvl.worldExtent;
                ls.pageWorldSize             = gpuLvl.pageWorldSize;
                ls.texelWorldSize            = gpuLvl.texelWorldSize;
                ls.receiverMinZ              = gpuLvl.receiverMinZ;
                ls.receiverMaxZ              = gpuLvl.receiverMaxZ;
                ls.casterMinZ                = gpuLvl.casterMinZ;
                ls.casterMaxZ                = gpuLvl.casterMaxZ;
                ls.depthBiasNdc              = gpuLvl.depthBiasNdc;
                ls.normalBiasWorld           = gpuLvl.normalBiasWorld;
                ls.cameraLightZ              = gpuLvl.cameraLightZ;
                ls.lightRight                = h.lightRight;
                ls.lightUp                   = h.lightUp;
                ls.lightDir                  = h.lightDir;
                ls.snappedLightCenterX       = gpuLvl.lightRightAndCenterX.w;
                ls.snappedLightCenterY       = gpuLvl.lightUpAndCenterY.w;
                ls.casterExtrusionDistance   = gpuLvl.lightDirAndExtrusion.w;
                ls.transitionStartRatio      = gpuLvl.transitionAndPadding.x;
                ls.casterDownstreamMargin    = gpuLvl.transitionAndPadding.z;
                ls.casterConservativePadding = gpuLvl.transitionAndPadding.w;
                ls.worldToLightView          = gpuLvl.worldToLightView;
                ls.lightViewToWorld          = gpuLvl.lightViewToWorld;
            }
        }

        outPageRanges.assign(pageRenderDataCapacity, NVSCasterPageRange{});
        outWorklist.assign(maxWorklistCapacity, NVSCasterDrawCandidate{});

        BuildNVSCasterWorklistCPU(
            sceneObjects,
            sceneSlotCount,
            pageRenderDataTable,
            pageRenderDataCapacity,
            nullptr,
            pageRenderDataCapacity,
            lightHierarchies,
            kNVSMaxClipmapLights,
            passDesc,
            maxWorklistCapacity,
            outPageRanges.empty() ? nullptr : outPageRanges.data(),
            outWorklist.empty() ? nullptr : outWorklist.data(),
            outTelemetry);

        outPageRanges.resize(outTelemetry.activeShadowPagesProcessed);
        outWorklist.resize(outTelemetry.totalAssignedCasterDraws);
    }

    // =====================================================================
    //  Dynamic Caster Bounds Selective Invalidation (CPU Reference)
    //  Checks whether a world-space invalidation box [worldMin, worldMax]
    //  intersects a resident page's 3D shadow volume.
    // =====================================================================
    inline BOOL DoesDynamicInvalidationBoxIntersectNVSPage(
        const NVSDynamicCasterInvalidationBox& box,
        const NVSClipmapLevelState&            levelState,
        UINT32                                 virtualPageX,
        UINT32                                 virtualPageY) noexcept
    {
        if (box.worldMax.x < box.worldMin.x ||
            box.worldMax.y < box.worldMin.y ||
            box.worldMax.z < box.worldMin.z)
        {
            return false;
        }

        NVSPageWorldBounds pageBounds{};
        if (!ComputeNVSPageWorldAndLightBounds(levelState, virtualPageX, virtualPageY, pageBounds))
        {
            return false;
        }

        // 1. World AABB overlap
        if (box.worldMax.x < pageBounds.worldBoundsMin.x || box.worldMin.x > pageBounds.worldBoundsMax.x ||
            box.worldMax.y < pageBounds.worldBoundsMin.y || box.worldMin.y > pageBounds.worldBoundsMax.y ||
            box.worldMax.z < pageBounds.worldBoundsMin.z || box.worldMin.z > pageBounds.worldBoundsMax.z)
        {
            return false;
        }

        // 2. Light-space 3D AABB overlap
        Vec3 lMin{}, lMax{};
        ProjectWorldBoundsToNVSLightSpace(box.worldMin, box.worldMax, levelState, lMin, lMax);
        return (lMax.x >= pageBounds.lightSpaceMinXY.x) && (lMin.x <= pageBounds.lightSpaceMaxXY.x) &&
               (lMax.y >= pageBounds.lightSpaceMinXY.y) && (lMin.y <= pageBounds.lightSpaceMaxXY.y) &&
               (lMax.z >= pageBounds.casterMinZ) && (lMin.z <= pageBounds.casterMaxZ);
    }

    // =====================================================================
    //  GPU-Driven Virtual Shadow Page Rasterization &
    //              ExecuteIndirect Rendering Types & Contracts
    // =====================================================================

    static constexpr UINT32 kNVSShadowRasterBucketCount          = 8u;
    static constexpr UINT32 kNVSInitialShadowRasterBucketCapacity = 8u;
    static constexpr UINT32 kNVSInvalidShadowRasterBucket        = 0xFFFFFFFFu;
    static constexpr UINT32 kNVSMaxMeshDrawDescriptors           = 4096u;
    static constexpr UINT32 kNVSMaxMaterialShadowDescriptors     = 1024u;
    static constexpr UINT32 kNVSMaxSkinningPayloads              = 256u;
    static constexpr UINT32 kNVSMaxIndirectCommandsPerBucket     = 24576u;
    static constexpr UINT32 kNVSMaxIndirectCommandsTotalCapacity = kNVSShadowRasterBucketCount * kNVSMaxIndirectCommandsPerBucket; // 196,608
    static constexpr UINT32 kNVSMaxShadowDrawContextsCapacity    = 131072u;

    enum NVSShadowRasterBucket : UINT32
    {
        NVS_SHADOW_BUCKET_OPAQUE_STATIC        = 0u,
        NVS_SHADOW_BUCKET_OPAQUE_INSTANCED     = 1u,
        NVS_SHADOW_BUCKET_OPAQUE_SKINNED       = 2u,
        NVS_SHADOW_BUCKET_ALPHA_TEST_STATIC    = 3u,
        NVS_SHADOW_BUCKET_ALPHA_TEST_INSTANCED = 4u,
        NVS_SHADOW_BUCKET_ALPHA_TEST_SKINNED   = 5u,
        NVS_SHADOW_BUCKET_DOUBLE_SIDED         = 6u,
        NVS_SHADOW_BUCKET_CUSTOM_SHADOW        = 7u,
        NVS_SHADOW_BUCKET_COUNT                = 8u
    };

    enum GPUMeshDrawDescriptorFlags : UINT32
    {
        GPU_MESH_DRAW_DESC_FLAG_NONE            = 0u,
        GPU_MESH_DRAW_DESC_FLAG_VALID           = 1u << 0u,
        GPU_MESH_DRAW_DESC_FLAG_DESTROYED       = 1u << 1u,
        GPU_MESH_DRAW_DESC_FLAG_SUBMESH         = 1u << 2u,
        GPU_MESH_DRAW_DESC_FLAG_LOD_VARIANT     = 1u << 3u,
        GPU_MESH_DRAW_DESC_FLAG_HAS_LOD         = 1u << 3u,
        GPU_MESH_DRAW_DESC_FLAG_STATIC_BATCH    = 1u << 4u,
        GPU_MESH_DRAW_DESC_FLAG_SKINNED_VERT    = 1u << 5u,

        GPU_MESH_DESC_FLAG_NONE                 = 0u,
        GPU_MESH_DESC_FLAG_VALID                = 1u << 0u,
        GPU_MESH_DESC_FLAG_DESTROYED            = 1u << 1u,
        GPU_MESH_DESC_FLAG_HAS_SUBMESHES        = 1u << 2u,
        GPU_MESH_DESC_FLAG_HAS_LOD              = 1u << 3u,
        GPU_MESH_DESC_FLAG_STATIC_BATCH_SUBMESH = 1u << 4u,
        GPU_MESH_DESC_FLAG_SKINNED_VERTICES     = 1u << 5u
    };

    enum GPUMaterialShadowFlags : UINT32
    {
        GPU_MAT_SHADOW_FLAG_NONE                  = 0u,
        GPU_MAT_SHADOW_FLAG_VALID                 = 1u << 0u,
        GPU_MAT_SHADOW_FLAG_ALPHA_TEST            = 1u << 1u,
        GPU_MAT_SHADOW_FLAG_DOUBLE_SIDED          = 1u << 2u,
        GPU_MAT_SHADOW_FLAG_CUSTOM_SHADOW_PASS    = 1u << 3u,
        GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT   = 1u << 4u,
        GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT = 1u << 5u,
        GPU_MAT_SHADOW_FLAG_MISSING_VARIANT        = 1u << 6u,
        GPU_MAT_SHADOW_FLAG_DESTROYED_DEPENDENCY   = 1u << 7u,
        GPU_MAT_SHADOW_FLAG_CUSTOM_VERTEX_DEFORM     = 1u << 8u,
        GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS         = 1u << 9u,
        GPU_MAT_SHADOW_FLAG_REQUIRES_SKINNING_DATA   = 1u << 10u,
        GPU_MAT_SHADOW_FLAG_SKINNED_REQUIRES_PALETTE = 1u << 11u,
        GPU_MAT_SHADOW_FLAG_NO_SHADOW_PASS           = 1u << 12u
    };

    enum NVSShadowDrawFlags : UINT32
    {
        NVS_SHADOW_DRAW_FLAG_NONE                 = 0u,
        NVS_SHADOW_DRAW_FLAG_VALID                = 1u << 0u,
        NVS_SHADOW_DRAW_FLAG_ALPHA_TEST           = 1u << 1u,
        NVS_SHADOW_DRAW_FLAG_DOUBLE_SIDED         = 1u << 2u,
        NVS_SHADOW_DRAW_FLAG_CUSTOM_SHADOW        = 1u << 3u,
        NVS_SHADOW_DRAW_FLAG_SKINNED              = 1u << 4u,
        NVS_SHADOW_DRAW_FLAG_INSTANCED            = 1u << 5u,
        NVS_SHADOW_DRAW_FLAG_STATIC_BATCH         = 1u << 6u,
        NVS_SHADOW_DRAW_FLAG_CUSTOM_VERTEX_DEFORM = 1u << 7u,
        NVS_SHADOW_DRAW_FLAG_PRETRANSFORMED_MESH  = 1u << 8u
    };

    enum NVSIndirectDrawDispatchMode : UINT32
    {
        NVS_INDIRECT_DRAW_MODE_RESET_COUNTERS       = 0u,
        NVS_INDIRECT_DRAW_MODE_BUILD_COMMANDS       = 1u,
        NVS_INDIRECT_DRAW_MODE_FINALIZE_PAGE_COMMIT = 2u
    };

    // =====================================================================
    //  GPU Mesh Draw Descriptor (64 Bytes, 16-Byte Aligned)
    //  Stores GPU-accessible geometry topology and D3D12 VBV / IBV GPU
    //  virtual addresses (no CPU pointers).
    // =====================================================================
    struct alignas(16) GPUMeshDrawDescriptor
    {
        // Row 0 (Offsets 0..15): Identity & Generation
        UINT32 meshId              = 0u;
        UINT32 submeshAndLodPacked = 0u; // (submeshIndex & 0xFFFFu) | ((lodLevel & 0xFFFFu) << 16u)
        UINT32 descriptorFlags     = 0u; // GPUMeshDrawDescriptorFlags bitmask
        UINT32 meshGeneration      = 0u; // Lower 16 bits: generation (>= 1); Upper 16 bits: optional variantTableOffset

        // Row 1 (Offsets 16..31): Draw Indexed Topology Range
        UINT32 indexCount          = 0u;
        UINT32 startIndex          = 0u;
        INT32  baseVertex          = 0;
        UINT32 vertexCount         = 0u;

        // Row 2 (Offsets 32..47): D3D12_VERTEX_BUFFER_VIEW (16 Bytes)
        UINT32 vbAddressLo         = 0u;
        UINT32 vbAddressHi         = 0u;
        UINT32 vbSizeInBytes       = 0u;
        UINT32 vbStrideInBytes     = 0u;

        // Row 3 (Offsets 48..63): D3D12_INDEX_BUFFER_VIEW (16 Bytes)
        UINT32 ibAddressLo         = 0u;
        UINT32 ibAddressHi         = 0u;
        UINT32 ibSizeInBytes       = 0u;
        UINT32 ibFormat            = 42u; // DXGI_FORMAT_R32_UINT = 42u, DXGI_FORMAT_R16_UINT = 57u
    };

    static_assert(sizeof(GPUMeshDrawDescriptor) == 64, "GPUMeshDrawDescriptor must be 64 bytes");
    static_assert(alignof(GPUMeshDrawDescriptor) == 16, "GPUMeshDrawDescriptor must be 16-byte aligned");
    static_assert(offsetof(GPUMeshDrawDescriptor, meshId) == 0, "meshId offset mismatch");
    static_assert(offsetof(GPUMeshDrawDescriptor, indexCount) == 16, "indexCount offset mismatch");
    static_assert(offsetof(GPUMeshDrawDescriptor, vbAddressLo) == 32, "vbAddressLo offset mismatch");
    static_assert(offsetof(GPUMeshDrawDescriptor, ibAddressLo) == 48, "ibAddressLo offset mismatch");

    inline constexpr UINT32 PackGPUMeshSubmeshAndLod(UINT32 submeshIndex, UINT32 lodLevel) noexcept
    {
        return (submeshIndex & 0xFFFFu) | ((lodLevel & 0xFFFFu) << 16u);
    }

    // =====================================================================
    //  GPU Material Shadow Descriptor (64 Bytes, 16-Byte Aligned)
    //  Resolves shadow rasterization bucket, alpha cutoff, double-sided
    //  cull override, and custom vertex deformation parameters on the GPU.
    // =====================================================================
    struct alignas(16) GPUMaterialShadowDescriptor
    {
        // Row 0 (Offsets 0..15): Identity & Variant Validity Flags
        UINT32 materialId           = 0u;
        UINT32 materialFlags        = GPU_MAT_SHADOW_FLAG_VALID | GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT | GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT;
        UINT32 materialGeneration   = 1u;
        UINT32 shaderSetId          = 0u;

        // Row 1 (Offsets 16..31): Pre-Resolved Bucket Routing
        UINT32 defaultBucketIndex   = NVS_SHADOW_BUCKET_OPAQUE_STATIC;
        UINT32 instancedBucketIndex = NVS_SHADOW_BUCKET_OPAQUE_INSTANCED;
        UINT32 skinnedBucketIndex   = NVS_SHADOW_BUCKET_OPAQUE_SKINNED;
        UINT32 customBucketIndex    = NVS_SHADOW_BUCKET_CUSTOM_SHADOW;

        // Row 2 (Offsets 32..47): Alpha Test & Custom Deformation Parameters
        FLOAT32 alphaCutoff         = 0.5f;
        FLOAT32 deformAmplitude     = 0.0f;
        FLOAT32 deformFrequency     = 1.0f;
        FLOAT32 normalBiasScale     = 1.0f;

        // Row 3 (Offsets 48..63): Procedural / UV Alpha Mask Pattern Parameters
        FLOAT32 uvScaleX            = 1.0f;
        FLOAT32 uvScaleY            = 1.0f;
        FLOAT32 uvOffsetX           = 0.0f;
        FLOAT32 uvOffsetY           = 0.0f;
    };

    static_assert(sizeof(GPUMaterialShadowDescriptor) == 64, "GPUMaterialShadowDescriptor must be 64 bytes");
    static_assert(alignof(GPUMaterialShadowDescriptor) == 16, "GPUMaterialShadowDescriptor must be 16-byte aligned");
    static_assert(offsetof(GPUMaterialShadowDescriptor, materialId) == 0, "materialId offset mismatch");
    static_assert(offsetof(GPUMaterialShadowDescriptor, defaultBucketIndex) == 16, "defaultBucketIndex offset mismatch");
    static_assert(offsetof(GPUMaterialShadowDescriptor, alphaCutoff) == 32, "alphaCutoff offset mismatch");
    static_assert(offsetof(GPUMaterialShadowDescriptor, uvScaleX) == 48, "uvScaleX offset mismatch");

    // =====================================================================
    //  GPU Skinned Shadow Payload (192 Bytes = 3 x Mat4, 16-Byte Aligned)
    //  Provides up to 3 bone matrices per skinned caster slot on the GPU.
    // =====================================================================
    struct alignas(16) NVSShadowSkinningPayload
    {
        Mat4 boneMatrix0 = Mat4::Identity();
        Mat4 boneMatrix1 = Mat4::Identity();
        Mat4 boneMatrix2 = Mat4::Identity();
    };

    static_assert(sizeof(NVSShadowSkinningPayload) == 192, "NVSShadowSkinningPayload must be 192 bytes");
    static_assert(alignof(NVSShadowSkinningPayload) == 16, "NVSShadowSkinningPayload must be 16-byte aligned");

    // =====================================================================
    //  Per-Draw / Per-Instance GPU Shadow Draw Context (64 Bytes, 16-Byte Aligned)
    //  Indexed in the Vertex Shader via (RootConstants.drawContextIndex + SV_InstanceID).
    // =====================================================================
    struct alignas(16) NVSShadowDrawContext
    {
        // Row 0 (Offsets 0..15): Scene Slot, Generation & Target Page
        UINT32 gpuSceneSlotIndex        = kInvalidGPUSceneSlot;
        UINT32 expectedObjectGeneration = 0u;
        UINT32 physicalPageIndex        = kNVSInvalidPhysicalPage;
        UINT32 renderDataIndex          = kNVSInvalidPhysicalPage;

        // Row 1 (Offsets 16..31): Virtual Page, Light/Level & Resource IDs
        UINT32 virtualPageFlatIndex     = kNVSInvalidVirtualPage;
        UINT32 lightIdAndClipmapLevel   = 0u;
        UINT32 meshDescriptorIndex      = 0u;
        UINT32 materialId               = 0u;

        // Row 2 (Offsets 32..47): Bucket, Flags, Bone Payload & Instance Metadata
        UINT32 bucketIndex              = NVS_SHADOW_BUCKET_OPAQUE_STATIC;
        UINT32 drawFlags                = NVS_SHADOW_DRAW_FLAG_NONE;
        UINT32 bonePayloadIndex         = 0u;
        UINT32 instanceOrdinalAndCount  = 1u << 16u; // (instanceOrdinal & 0xFFFFu) | ((instanceCount & 0xFFFFu) << 16u)

        // Row 3 (Offsets 48..63): Shader Material / Deformation Parameters
        FLOAT32 alphaCutoff             = 0.5f;
        FLOAT32 deformAmplitude         = 0.0f;
        FLOAT32 deformFrequency         = 1.0f;
        FLOAT32 uvScaleX                = 1.0f;
    };

    static_assert(sizeof(NVSShadowDrawContext) == 64, "NVSShadowDrawContext must be 64 bytes");
    static_assert(alignof(NVSShadowDrawContext) == 16, "NVSShadowDrawContext must be 16-byte aligned");
    static_assert(offsetof(NVSShadowDrawContext, gpuSceneSlotIndex) == 0, "gpuSceneSlotIndex offset mismatch");
    static_assert(offsetof(NVSShadowDrawContext, virtualPageFlatIndex) == 16, "virtualPageFlatIndex offset mismatch");
    static_assert(offsetof(NVSShadowDrawContext, bucketIndex) == 32, "bucketIndex offset mismatch");
    static_assert(offsetof(NVSShadowDrawContext, alphaCutoff) == 48, "alphaCutoff offset mismatch");

    // =====================================================================
    //  GPU-Driven User Shader Binding Metadata for ExecuteIndirect
    //  Allows RuntimeNVSIndirectDraw.nsl to populate 256B-aligned per-draw
    //  Constant Buffers and 64B-aligned per-batch Instance StructuredBuffers
    //  purely on the GPU from reflected NSC metadata (zero NSLCompiler hacks).
    // =====================================================================
    static constexpr UINT32 kNVSMaxPromotedIndirectCBVs = 3u;

    enum NVSShadowIndirectSRVMode : UINT32
    {
        NVS_INDIRECT_SRV_NONE               = 0u,
        NVS_INDIRECT_SRV_INSTANCE_POOL      = 1u,
        NVS_INDIRECT_SRV_BONE_PALETTE_SLICE = 2u
    };

    enum NVSShadowPatchSemantic : UINT32
    {
        NVS_PATCH_SEMANTIC_NONE            = 0u,
        NVS_PATCH_WORLD_MATRIX             = 1u,  // Mat4 (64B)
        NVS_PATCH_INV_WORLD_MATRIX         = 2u,  // Mat4 (64B)
        NVS_PATCH_NORMAL_MATRIX            = 3u,  // Mat3 in CBuffer (48B = 3 x Vec4)
        NVS_PATCH_VIEW_MATRIX              = 4u,  // Mat4 (64B, page lightViewMatrix)
        NVS_PATCH_INV_VIEW_MATRIX          = 5u,  // Mat4 (64B)
        NVS_PATCH_PROJECTION_MATRIX        = 6u,  // Mat4 (64B, page pageProjMatrix)
        NVS_PATCH_INV_PROJECTION_MATRIX    = 7u,  // Mat4 (64B)
        NVS_PATCH_VP_MATRIX                = 8u,  // Mat4 (64B, page worldToPageClipMatrix)
        NVS_PATCH_INV_VP_MATRIX            = 9u,  // Mat4 (64B)
        NVS_PATCH_MVP_MATRIX               = 10u, // Mat4 (64B, worldToPageClipMatrix * worldMatrix)
        NVS_PATCH_CAMERA_POSITION          = 11u, // Vec3 (12B)
        NVS_PATCH_CAMERA_DIRECTION         = 12u, // Vec3 (12B)
        NVS_PATCH_NEAR_PLANE               = 13u, // Float (4B, casterMinZ)
        NVS_PATCH_FAR_PLANE                = 14u, // Float (4B, casterMaxZ)
        NVS_PATCH_BONE_PALETTE_OFFSET      = 15u, // UInt (4B)
        NVS_PATCH_BONE_COUNT               = 16u, // UInt (4B)
        NVS_PATCH_SKINNING_REVISION        = 17u, // UInt (4B)
        NVS_PATCH_NVS_PHYSICAL_PAGE_INDEX  = 18u, // UInt (4B)
        NVS_PATCH_NVS_VIRTUAL_PAGE_ADDRESS = 19u, // UInt (4B)
        NVS_PATCH_NVS_CLIPMAP_LEVEL        = 20u, // UInt (4B)
        NVS_PATCH_NVS_DEPTH_BIAS_NDC       = 21u, // Float (4B)
        NVS_PATCH_NVS_NORMAL_BIAS_WORLD    = 22u, // Float (4B)
        NVS_PATCH_NVS_RECEIVER_MIN_Z       = 23u, // Float (4B)
        NVS_PATCH_NVS_RECEIVER_MAX_Z       = 24u, // Float (4B)
        NVS_PATCH_CAMERA_FOV                 = 25u, // Float (4B)
        NVS_PATCH_CAMERA_ASPECT              = 26u, // Float (4B)
        NVS_PATCH_NVG_DRAW_BUCKET_ID         = 27u, // UInt (4B)
        NVS_PATCH_NVG_VISIBLE_CLUSTER_COUNT  = 28u, // UInt (4B)
        NVS_PATCH_NVG_VISIBLE_CLUSTER_OFFSET = 29u, // UInt (4B)
        NVS_PATCH_GPU_SCENE_OBJECT_SLOT      = 30u, // UInt (4B)
        NVS_PATCH_NVG_DRAW_CONTEXT_ID        = 31u, // UInt (4B)

        NVS_PATCH_INSTANCE_WORLD_MATRIX      = 32u, // Mat4 (64B)
        NVS_PATCH_INSTANCE_ID                = 33u, // UInt (4B)
        NVS_PATCH_INSTANCE_CUSTOM_DATA       = 34u, // UInt (4B)
        NVS_PATCH_INSTANCE_BONE_OFFSET       = 35u, // UInt (4B)
        NVS_PATCH_INSTANCE_BONE_COUNT        = 36u, // UInt (4B)
        NVS_PATCH_INSTANCE_PREV_WORLD_MATRIX = 37u, // Mat4 (64B)

        NVS_PATCH_PREV_WORLD_MATRIX          = 40u, // Mat4 (64B)
        NVS_PATCH_PREV_VIEW_MATRIX           = 41u, // Mat4 (64B)
        NVS_PATCH_PREV_PROJECTION_MATRIX     = 42u, // Mat4 (64B)
        NVS_PATCH_PREV_VP_MATRIX             = 43u  // Mat4 (64B)
    };

    inline constexpr BOOL IsNVSInstancePatchSemantic(UINT32 semanticId) noexcept
    {
        return (semanticId >= NVS_PATCH_INSTANCE_WORLD_MATRIX) &&
               (semanticId <= NVS_PATCH_INSTANCE_PREV_WORLD_MATRIX);
    }

    struct alignas(16) NVSDrawGroupFieldPatch
    {
        UINT32 semanticId = NVS_PATCH_SEMANTIC_NONE;
        UINT32 targetSlot = 0u; // 0..2 = Promoted CBV 0..2, 3 = Promoted Instance SRV 0
        UINT32 byteOffset = 0u; // Reflected byte offset within CBuffer or Instance struct
        UINT32 byteSize   = 0u; // Reflected byte size
    };

    static_assert(sizeof(NVSDrawGroupFieldPatch) == 16, "NVSDrawGroupFieldPatch must be 16 bytes.");
    static_assert(alignof(NVSDrawGroupFieldPatch) == 16, "NVSDrawGroupFieldPatch must be 16-byte aligned.");

    struct alignas(16) NVSDrawGroupShaderBindingDesc
    {
        // Row 0 (Offsets 0..15)
        UINT32 bindingFlags               = 0u; // bit 0 = valid
        UINT32 patchFirstIndex            = 0u;
        UINT32 patchCount                 = 0u;
        UINT32 instanceStrideBytes        = 0u;

        // Row 1 (Offsets 16..31)
        UINT32 cbv0ByteSize               = 0u;
        UINT32 cbv1ByteSize               = 0u;
        UINT32 cbv2ByteSize               = 0u;
        UINT32 instanceTemplateWordOffset = 0u;

        // Row 2 (Offsets 32..47)
        UINT32 cbv0TemplateWordOffset     = 0u;
        UINT32 cbv1TemplateWordOffset     = 0u;
        UINT32 cbv2TemplateWordOffset     = 0u;
        UINT32 shaderSetId                = 0u;

        // Row 3 (Offsets 48..63)
        UINT32 activeStageMask            = 0u;
        UINT32 cbvCount                   = 0u;
        UINT32 srvMode                    = NVS_INDIRECT_SRV_NONE;
        UINT32 drawGroupIndex             = 0u;
    };

    static_assert(sizeof(NVSDrawGroupShaderBindingDesc) == 64, "NVSDrawGroupShaderBindingDesc must be 64 bytes.");
    static_assert(alignof(NVSDrawGroupShaderBindingDesc) == 16, "NVSDrawGroupShaderBindingDesc must be 16-byte aligned.");

    // =====================================================================
    //  D3D12 ExecuteIndirect Command Record (96 Bytes = 6 x 16B, 16-Byte Aligned)
    //  Exact binary layout for m_nvsDrawIndexedIndirectSignature:
    //    Bytes  0..15 : D3D12_VERTEX_BUFFER_VIEW (Slot 0)
    //    Bytes 16..31 : D3D12_INDEX_BUFFER_VIEW
    //    Bytes 32..39 : Root CBV 0 GPU Virtual Address (8B aligned)
    //    Bytes 40..47 : Root CBV 1 GPU Virtual Address (8B aligned)
    //    Bytes 48..55 : Root CBV 2 GPU Virtual Address (8B aligned)
    //    Bytes 56..63 : Root SRV 0 GPU Virtual Address (8B aligned)
    //    Bytes 64..83 : D3D12_DRAW_INDEXED_ARGUMENTS (5 x 32-bit values)
    //    Bytes 84..95 : Trailing diagnostic fields (drawContextIndex, meshDescriptorIndex, commandFlags)
    // =====================================================================
    struct alignas(16) NVSIndirectDrawCommand
    {
        // Row 0 (Offsets 0..15): Vertex Buffer View
        UINT32 vbAddressLo           = 0u;
        UINT32 vbAddressHi           = 0u;
        UINT32 vbSizeInBytes         = 0u;
        UINT32 vbStrideInBytes       = 0u;

        // Row 1 (Offsets 16..31): Index Buffer View
        UINT32 ibAddressLo           = 0u;
        UINT32 ibAddressHi           = 0u;
        UINT32 ibSizeInBytes         = 0u;
        UINT32 ibFormat              = 42u; // DXGI_FORMAT_R32_UINT = 42u

        // Row 2 (Offsets 32..47): Root CBV 0 & Root CBV 1 GPU Virtual Addresses
        UINT32 cbv0AddressLo         = 0u;
        UINT32 cbv0AddressHi         = 0u;
        UINT32 cbv1AddressLo         = 0u;
        UINT32 cbv1AddressHi         = 0u;

        // Row 3 (Offsets 48..63): Root CBV 2 & Root SRV 0 GPU Virtual Addresses
        UINT32 cbv2AddressLo         = 0u;
        UINT32 cbv2AddressHi         = 0u;
        UINT32 srv0AddressLo         = 0u;
        UINT32 srv0AddressHi         = 0u;

        // Row 4 (Offsets 64..79): DrawIndexed Arguments (First 4)
        UINT32 indexCountPerInstance = 0u;
        UINT32 instanceCount         = 0u;
        UINT32 startIndexLocation    = 0u;
        INT32  baseVertexLocation    = 0;

        // Row 5 (Offsets 80..95): StartInstanceLocation + Diagnostic Metadata
        UINT32 startInstanceLocation = 0u;
        UINT32 drawContextIndex      = 0u;
        UINT32 meshDescriptorIndex   = 0u;
        UINT32 commandFlags          = 0u;
    };

    static_assert(sizeof(NVSIndirectDrawCommand) == 96, "NVSIndirectDrawCommand must be 96 bytes (6 x 16B).");
    static_assert(alignof(NVSIndirectDrawCommand) == 16, "NVSIndirectDrawCommand must be 16-byte aligned.");
    static_assert(offsetof(NVSIndirectDrawCommand, vbAddressLo)           == 0,  "vbAddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, ibAddressLo)           == 16, "ibAddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, cbv0AddressLo)         == 32, "cbv0AddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, cbv1AddressLo)         == 40, "cbv1AddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, cbv2AddressLo)         == 48, "cbv2AddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, srv0AddressLo)         == 56, "srv0AddressLo offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, indexCountPerInstance) == 64, "indexCountPerInstance offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, startInstanceLocation) == 80, "startInstanceLocation offset mismatch");
    static_assert(offsetof(NVSIndirectDrawCommand, drawContextIndex)      == 84, "drawContextIndex offset mismatch");

    // =====================================================================
    //  GPU Indirect Command Generation Telemetry (80 Bytes, 16-Byte Aligned)
    // =====================================================================
    struct alignas(16) NVSIndirectDrawTelemetry
    {
        // Row 0 (Offsets 0..15): High-Level Execution Counts
        UINT32 activePagesEvaluated             = 0u;
        UINT32 candidateDrawsProcessed          = 0u;
        UINT32 validIndirectCommandsEmitted     = 0u;
        UINT32 totalInstancesEmitted            = 0u;

        // Row 1 (Offsets 16..31): GPU Validation Rejections
        UINT32 rejectedStaleObjectHandle        = 0u;
        UINT32 rejectedInvalidMeshDescriptor    = 0u;
        UINT32 rejectedInvalidMaterialOrVariant = 0u;
        UINT32 rejectedInvalidPageOrSlice       = 0u;

        // Row 2 (Offsets 32..47): Overflow, Instancing & Page Commit Summary
        UINT32 overflowDroppedCommands          = 0u;
        UINT32 instancedBatchMergedDraws        = 0u;
        UINT32 committablePagesCount            = 0u;
        UINT32 nonCommittablePagesCount         = 0u;

        // Rows 3..4 (Offsets 48..79): Per-Bucket Emitted Indirect Command Counts (First 8 Buckets)
        UINT32 bucketCommandCounts[kNVSShadowRasterBucketCount] = {};
    };

    static_assert(sizeof(NVSIndirectDrawTelemetry) == 80, "NVSIndirectDrawTelemetry must be 80 bytes");
    static_assert(alignof(NVSIndirectDrawTelemetry) == 16, "NVSIndirectDrawTelemetry must be 16-byte aligned");
    static_assert(offsetof(NVSIndirectDrawTelemetry, activePagesEvaluated) == 0, "activePagesEvaluated offset mismatch");
    static_assert(offsetof(NVSIndirectDrawTelemetry, rejectedStaleObjectHandle) == 16, "rejectedStaleObjectHandle offset mismatch");
    static_assert(offsetof(NVSIndirectDrawTelemetry, overflowDroppedCommands) == 32, "overflowDroppedCommands offset mismatch");
    static_assert(offsetof(NVSIndirectDrawTelemetry, bucketCommandCounts) == 48, "bucketCommandCounts offset mismatch");

    // =====================================================================
    //  Shadow Rasterization Pass Configuration Descriptor
    // =====================================================================
    struct NVSShadowRasterPassDesc
    {
        UINT32     currentFrame                 = 1u;
        BOOL       enableHardwareInstancing     = true;
        BOOL       allowInstancedMerge          = true;
        BOOL       commitRenderedPagesOnGPU     = true;
        BOOL       commitRenderedPages          = true;
        BOOL       applyShaderBias              = false;
        FLOAT32    depthBiasScale               = 1.0f;
        FLOAT32    normalBiasScale              = 1.0f;
        FLOAT32    timeSeconds                  = 0.0f;
        INT32      rasterDepthBias              = 0;
        FLOAT32    rasterDepthBiasClamp         = 0.0f;
        FLOAT32    rasterSlopeScaledDepthBias   = 0.0f;
        BOOL                forceGeometryShaderFallback  = false;
        NVSDepthCompareMode depthMode                    = NVS_DEPTH_MODE_STANDARD_ZERO_TO_ONE;
        UINT32              maxCommandsPerBucketOverride = 0u; // 0 = use kNVSMaxIndirectCommandsPerBucket
        UINT32     maxDrawContextsOverride      = 0u; // 0 = use kNVSMaxShadowDrawContextsCapacity
        GPUBuffer* overrideGPUSceneBuffer       = nullptr;
        UINT32     overrideSceneSlotCount       = 0u;
        GPUBuffer* gpuSkinningPaletteBuffer     = nullptr;
    };

    // =====================================================================
    //  CPU Reference Helpers for Bucket Selection & Command Build
    // =====================================================================
    inline BOOL ResolveNVSShadowBucketAndFlags(
        const NVSCasterDrawCandidate&      candidate,
        const GPUSceneObjectData&          sceneObj,
        const GPUMeshDrawDescriptor&       meshDesc,
        const GPUMaterialShadowDescriptor& matDesc,
        UINT32&                            outBucketIndex,
        UINT32&                            outDrawFlags) noexcept
    {
        outBucketIndex = NVS_SHADOW_BUCKET_OPAQUE_STATIC;
        outDrawFlags   = NVS_SHADOW_DRAW_FLAG_NONE;

        // 1. Validate Material & Shader Variant state
        if ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_VALID) == 0u ||
            (matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_MISSING_VARIANT) != 0u ||
            (matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_DESTROYED_DEPENDENCY) != 0u ||
            (matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_NO_SHADOW_PASS) != 0u)
        {
            return false;
        }

        // 2. Validate Mesh Draw Descriptor state
        if ((meshDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_VALID) == 0u ||
            (meshDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_DESTROYED) != 0u ||
            meshDesc.indexCount == 0u ||
            meshDesc.vbSizeInBytes == 0u ||
            meshDesc.vbStrideInBytes == 0u ||
            meshDesc.ibSizeInBytes == 0u ||
            (meshDesc.vbAddressLo == 0u && meshDesc.vbAddressHi == 0u) ||
            (meshDesc.ibAddressLo == 0u && meshDesc.ibAddressHi == 0u))
        {
            return false;
        }

        const UINT32 varFlags = candidate.pipelineVariantFlags;
        const BOOL isSkinned  = ((varFlags & NVS_CASTER_VARIANT_SKINNED) != 0u) ||
                                ((sceneObj.objectFlags & GPU_SCENE_OBJ_SKINNED) != 0u) ||
                                ((sceneObj.renderObjectFlags & static_cast<UINT32>(RenderObjectFlags::SKINNED)) != 0u) ||
                                ((meshDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_SKINNED_VERT) != 0u);
        const BOOL isInstanced = ((varFlags & NVS_CASTER_VARIANT_INSTANCED_CANDIDATE) != 0u) ||
                                 ((sceneObj.objectFlags & GPU_SCENE_OBJ_INSTANCED_CANDIDATE) != 0u);
        const BOOL isAlphaTest = ((varFlags & NVS_CASTER_VARIANT_ALPHA_TEST) != 0u) ||
                                 ((sceneObj.objectFlags & GPU_SCENE_OBJ_ALPHA_TESTED) != 0u) ||
                                 ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_ALPHA_TEST) != 0u);
        const BOOL isDoubleSided = ((varFlags & NVS_CASTER_VARIANT_DOUBLE_SIDED) != 0u) ||
                                   ((sceneObj.objectFlags & GPU_SCENE_OBJ_TWO_SIDED) != 0u) ||
                                   ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_DOUBLE_SIDED) != 0u);
        const BOOL isCustomShadow = ((varFlags & NVS_CASTER_VARIANT_CUSTOM_SHADOW_PASS) != 0u) ||
                                    ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_CUSTOM_SHADOW_PASS) != 0u);
        const BOOL isCustomDeform = ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_CUSTOM_VERTEX_DEFORM) != 0u);

        if (isSkinned && (matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_HAS_SKINNED_VARIANT) == 0u)
        {
            return false;
        }
        if (isInstanced && (matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_HAS_INSTANCED_VARIANT) == 0u)
        {
            return false;
        }
        const BOOL reqSkinData =
            ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_REQUIRES_SKINNING_DATA) != 0u) ||
            (isSkinned && ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_SKINNED_REQUIRES_PALETTE) != 0u));
        if (reqSkinData && ((sceneObj.renderObjectFlags >> 16u) & 0xFFFFu) == 0u)
        {
            return false;
        }

        outDrawFlags = NVS_SHADOW_DRAW_FLAG_VALID;
        if (isAlphaTest)    outDrawFlags |= NVS_SHADOW_DRAW_FLAG_ALPHA_TEST;
        if (isDoubleSided)  outDrawFlags |= NVS_SHADOW_DRAW_FLAG_DOUBLE_SIDED;
        if (isCustomShadow) outDrawFlags |= NVS_SHADOW_DRAW_FLAG_CUSTOM_SHADOW;
        if (isSkinned)      outDrawFlags |= NVS_SHADOW_DRAW_FLAG_SKINNED;
        if (isInstanced)    outDrawFlags |= NVS_SHADOW_DRAW_FLAG_INSTANCED;
        if (isCustomDeform) outDrawFlags |= NVS_SHADOW_DRAW_FLAG_CUSTOM_VERTEX_DEFORM;
        if ((varFlags & (NVS_CASTER_VARIANT_STATIC_BATCH_MEMBER | NVS_CASTER_VARIANT_COMBINED_STATIC_BATCH)) != 0u ||
            (meshDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_STATIC_BATCH) != 0u)
        {
            outDrawFlags |= NVS_SHADOW_DRAW_FLAG_STATIC_BATCH;
        }
        if ((meshDesc.descriptorFlags & GPU_MESH_DRAW_DESC_FLAG_STATIC_BATCH) != 0u)
        {
            outDrawFlags |= NVS_SHADOW_DRAW_FLAG_PRETRANSFORMED_MESH;
        }

        if ((matDesc.materialFlags & GPU_MAT_SHADOW_FLAG_EXPLICIT_BUCKETS) != 0u)
        {
            if (isSkinned && isInstanced)
            {
                outBucketIndex = (matDesc.customBucketIndex != kNVSInvalidShadowRasterBucket)
                    ? matDesc.customBucketIndex
                    : matDesc.skinnedBucketIndex;
            }
            else if (isSkinned)
            {
                outBucketIndex = (matDesc.skinnedBucketIndex != kNVSInvalidShadowRasterBucket)
                    ? matDesc.skinnedBucketIndex
                    : matDesc.customBucketIndex;
            }
            else if (isInstanced)
            {
                outBucketIndex = (matDesc.instancedBucketIndex != kNVSInvalidShadowRasterBucket)
                    ? matDesc.instancedBucketIndex
                    : matDesc.defaultBucketIndex;
            }
            else
            {
                outBucketIndex = (matDesc.defaultBucketIndex != kNVSInvalidShadowRasterBucket)
                    ? matDesc.defaultBucketIndex
                    : matDesc.instancedBucketIndex;
            }

            if (outBucketIndex == kNVSInvalidShadowRasterBucket)
            {
                return false;
            }
            return true;
        }

        // Priority bucket routing:
        if (isCustomShadow)
        {
            outBucketIndex = NVS_SHADOW_BUCKET_CUSTOM_SHADOW;
        }
        else if (isDoubleSided && !isSkinned && !isAlphaTest)
        {
            outBucketIndex = NVS_SHADOW_BUCKET_DOUBLE_SIDED;
        }
        else if (isAlphaTest)
        {
            if (isSkinned)        outBucketIndex = NVS_SHADOW_BUCKET_ALPHA_TEST_SKINNED;
            else if (isInstanced) outBucketIndex = NVS_SHADOW_BUCKET_ALPHA_TEST_INSTANCED;
            else                  outBucketIndex = NVS_SHADOW_BUCKET_ALPHA_TEST_STATIC;
        }
        else
        {
            if (isSkinned)        outBucketIndex = NVS_SHADOW_BUCKET_OPAQUE_SKINNED;
            else if (isInstanced) outBucketIndex = NVS_SHADOW_BUCKET_OPAQUE_INSTANCED;
            else                  outBucketIndex = NVS_SHADOW_BUCKET_OPAQUE_STATIC;
        }

        return true;
    }

    // =====================================================================
    //  Virtual-to-Physical Shadow Sampling ABI & CPU Reference
    //  Shared 1:1 mathematical contract across CPU, Compute & Pixel Shader
    //  (shaders/NorthstarLibrary/VirtualShadows/).
    // =====================================================================
    enum NVSShadowSampleStatus : UINT32
    {
        NVS_SAMPLE_STATUS_VALID_SHADOW_DATA   = 0u, // Resident valid physical shadow page (state == 3)
        NVS_SAMPLE_STATUS_MISSING             = 1u, // Virtual page unmapped or evicted (state == 0 or 5)
        NVS_SAMPLE_STATUS_NOT_RENDERED        = 2u, // Physical page allocated or cleared, not yet rendered (state == 1 or 2)
        NVS_SAMPLE_STATUS_INVALIDATED         = 3u, // Virtual page explicitly invalidated (state == 4)
        NVS_SAMPLE_STATUS_GENERATION_MISMATCH = 4u, // Bidirectional owner/address/generation mismatch
        NVS_SAMPLE_STATUS_AWAITING_UPDATE     = 5u, // Sampling snapshot not yet committed (e.g. Frame 0 DepthPrepass=OFF)
        NVS_SAMPLE_STATUS_OUT_OF_RANGE        = 6u, // Receiver outside shadow distance / cone / clipmap hierarchy
        NVS_SAMPLE_STATUS_INVALID_LIGHT       = 7u  // Invalid shadow view range or shadow disabled on light
    };

    struct NVSShadowSampleResult
    {
        NVSShadowSampleStatus status               = NVS_SAMPLE_STATUS_INVALID_LIGHT;
        UINT32                selectedViewIndex    = 0xFFFFFFFFu;
        UINT32                virtualFlatIndex     = kNVSInvalidVirtualPage;
        UINT32                physicalPageIndex    = kNVSInvalidPhysicalPage;
        UINT32                virtualPageX         = 0u;
        UINT32                virtualPageY         = 0u;
        UINT32                desiredMipLevel      = 0u;
        UINT32                resolvedMipLevel     = 0u;
        BOOL                  usedCoarseFallback   = false;
        INT32                 worldPageX           = 0;
        INT32                 worldPageY           = 0;
        Vec2                  rawSliceUV           = Vec2(0.5f, 0.5f); // Unclamped [0, 1] D3D12 slice UV
        Vec2                  borderSafeSliceUV    = Vec2(0.5f, 0.5f); // Clamped to [0.5/128, 127.5/128]
        FLOAT32               receiverNdcDepth     = 0.0f;
        FLOAT32               receiverCompareDepth = 0.0f;
        FLOAT32               clipmapTransition    = 0.0f;
        FLOAT32               mipTransition        = 0.0f;
    };

    inline UINT32 ComputeNVSPerspectiveDesiredMipCPU(
        FLOAT32  screenPixelWorldSize,
        FLOAT32  receiverLightZ,
        FLOAT32  baseMip0TexelTanSize,
        UINT32   mipCount,
        FLOAT32* outMipTransition = nullptr,
        UINT32   minMip           = 0u,
        FLOAT32  lodBias          = 0.0f) noexcept
    {
        if (outMipTransition) *outMipTransition = 0.0f;
        const UINT32 safeMaxMip = (mipCount > 0u) ? (mipCount - 1u) : 0u;
        const UINT32 safeMinMip = std::min(minMip, safeMaxMip);
        if (mipCount <= 1u || screenPixelWorldSize <= 0.0f || baseMip0TexelTanSize <= 1e-8f || receiverLightZ <= 1e-5f)
        {
            return safeMinMip;
        }
        const FLOAT32 mip0TexelWorld = receiverLightZ * baseMip0TexelTanSize;
        const FLOAT32 ratio          = screenPixelWorldSize / std::max(mip0TexelWorld, 1e-7f);
        const FLOAT32 rawLod         = std::log2(std::max(ratio, 0.0625f)) + lodBias;
        const FLOAT32 lodFloat       = std::clamp(rawLod, static_cast<FLOAT32>(safeMinMip), static_cast<FLOAT32>(safeMaxMip));
        return std::min(std::max(static_cast<UINT32>(std::floor(lodFloat)), safeMinMip), safeMaxMip);
    }

    inline NVSShadowSampleResult ResolveNVSVirtualPageLookupForViewCPU(
        UINT32                        viewIndex,
        const NVSClipmapLevelGPUData* viewTable,
        UINT32                        viewTableCount,
        const NVSVirtualPageEntry*    virtualPageTable,
        UINT32                        totalVirtualPages,
        const NVSPhysicalPageEntry*   physicalPageTable,
        UINT32                        physicalPoolCapacity,
        const Vec3&                   worldPos,
        const Vec3&                   worldNormal           = Vec3(0.0f, 1.0f, 0.0f),
        Vec2                          texelPageOffsetXY     = Vec2(0.0f, 0.0f),
        BOOL                          samplingAvailable     = true,
        const UINT32*                 sparsePageHierarchy   = nullptr,
        UINT32                        sparseRootCapacity    = 0u,
        UINT32                        sparseDirCapacity     = 0u,
        FLOAT32                       screenPixelWorldSize  = 0.0f,
        UINT32                        explicitMipOverride   = 0xFFFFFFFFu) noexcept
    {
        NVSShadowSampleResult res{};
        if (!samplingAvailable)
        {
            res.status = NVS_SAMPLE_STATUS_AWAITING_UPDATE;
            return res;
        }
        if (!viewTable || !virtualPageTable || !physicalPageTable || viewIndex >= viewTableCount)
        {
            res.status = NVS_SAMPLE_STATUS_INVALID_LIGHT;
            return res;
        }

        const NVSClipmapLevelGPUData& view = viewTable[viewIndex];
        if ((view.flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) == 0u ||
            view.pagesPerAxis == 0u ||
            view.pageWorldSize <= 1e-6f ||
            (view.casterMaxZ - view.casterMinZ) <= 1e-6f)
        {
            res.status = NVS_SAMPLE_STATUS_INVALID_LIGHT;
            return res;
        }

        res.selectedViewIndex = viewIndex;

        const Vec3 R(view.lightRightAndCenterX.x, view.lightRightAndCenterX.y, view.lightRightAndCenterX.z);
        const Vec3 U(view.lightUpAndCenterY.x, view.lightUpAndCenterY.y, view.lightUpAndCenterY.z);
        const Vec3 D(view.lightDirAndExtrusion.x, view.lightDirAndExtrusion.y, view.lightDirAndExtrusion.z);

        const FLOAT32 normalLenSq = worldNormal.Dot(worldNormal);
        const Vec3    safeNormal  = (normalLenSq > 1e-6f) ? worldNormal.Normalized() : Vec3::Zero();

        const BOOL   isPerspective     = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_PERSPECTIVE) != 0u;
        const BOOL   isSpotCone        = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_SPOT_CONE) != 0u;
        const BOOL   isReversedZ       = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_REVERSED_Z) != 0u;
        const BOOL   isTexelRelBias    = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_TEXEL_RELATIVE_BIAS) != 0u;
        const BOOL   rasterAppliedDB   = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_DEPTH_BIAS) != 0u;
        const BOOL   rasterAppliedNB   = (view.flags & NVS_CLIPMAP_LEVEL_FLAG_RASTER_APPLIED_NORMAL_BIAS) != 0u;
        const UINT32 mipCount          = isPerspective ? ExtractNVSViewMipCount(view.flags) : 1u;

        // First compute un-biased light-space depth for perspective mip & texel-relative scaling
        FLOAT32 rawLightZ = 0.0f;
        Vec3    lightPos  = Vec3::Zero();
        if (isPerspective)
        {
            lightPos = Vec3(view.lightViewToWorld.m[12], view.lightViewToWorld.m[13], view.lightViewToWorld.m[14]);
            const Vec3 rawRel = worldPos - lightPos;
            rawLightZ = rawRel.Dot(D);
            if (rawLightZ < view.casterMinZ || rawLightZ > view.casterMaxZ || rawLightZ <= 1e-5f)
            {
                res.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                return res;
            }
            if (isSpotCone)
            {
                const FLOAT32 dist = std::sqrt(rawRel.Dot(rawRel));
                if (dist <= 1e-5f || (rawLightZ / dist) < view.transitionAndPadding.x)
                {
                    res.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                    return res;
                }
            }
        }
        else if ((view.flags & NVS_CLIPMAP_LEVEL_FLAG_RADIAL_SELECT) != 0u && view.lightViewToWorld.m[15] > 1e-4f)
        {
            const Vec3 camPos(view.lightViewToWorld.m[12], view.lightViewToWorld.m[13], view.lightViewToWorld.m[14]);
            const Vec3 camRel   = worldPos - camPos;
            const FLOAT32 cDist = std::sqrt(camRel.Dot(camRel));
            if (cDist > view.lightViewToWorld.m[15])
            {
                res.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                return res;
            }
        }

        FLOAT32 mipTrans   = 0.0f;
        UINT32  desiredMip = 0u;
        if (isPerspective && mipCount > 1u)
        {
            if (explicitMipOverride < mipCount)
            {
                desiredMip = explicitMipOverride;
            }
            else
            {
                const UINT32  viewMinMip  = ExtractNVSViewMinMip(view.flags);
                const FLOAT32 viewLodBias = ExtractNVSViewLodBias(view.flags);
                desiredMip = ComputeNVSPerspectiveDesiredMipCPU(
                    screenPixelWorldSize,
                    rawLightZ,
                    view.texelWorldSize,
                    mipCount,
                    &mipTrans,
                    viewMinMip,
                    viewLodBias);
            }
        }
        res.desiredMipLevel  = desiredMip;
        res.resolvedMipLevel = desiredMip;
        res.mipTransition    = mipTrans;

        auto evaluateAtMip = [&](UINT32 evalMip, NVSShadowSampleResult& outCand) noexcept -> BOOL
        {
            outCand                   = res;
            outCand.resolvedMipLevel  = evalMip;
            outCand.usedCoarseFallback = (evalMip > desiredMip);

            const UINT32  mipPagesAxis = std::max(view.pagesPerAxis >> evalMip, 1u);
            const FLOAT32 mipScale     = static_cast<FLOAT32>(1u << evalMip);
            const FLOAT32 mipPageSize  = view.worldExtent / static_cast<FLOAT32>(mipPagesAxis);

            FLOAT32 effNormalOffsetWorld = rasterAppliedNB ? 0.0f : view.normalBiasWorld;
            if (isPerspective && isTexelRelBias)
            {
                const FLOAT32 fullOffset = (mipScale * view.normalBiasWorld) * rawLightZ;
                const FLOAT32 rasterPart = rasterAppliedNB ? ((mipScale * view.normalBiasWorld) * view.casterMinZ) : 0.0f;
                effNormalOffsetWorld     = std::max(fullOffset - rasterPart, 0.0f);
            }

            const Vec3 biasedPos = worldPos + safeNormal * effNormalOffsetWorld;
            Vec3       lightRayDir = D;
            FLOAT32 lx = 0.0f, ly = 0.0f, lz = 0.0f, projCoordX = 0.0f, projCoordY = 0.0f;
            if (isPerspective)
            {
                const Vec3    rel    = biasedPos - lightPos;
                const FLOAT32 distSq = std::max(rel.Dot(rel), 1e-10f);
                const FLOAT32 dist   = std::sqrt(distSq);
                lightRayDir          = rel * (1.0f / dist);
                lx = rel.Dot(R);
                ly = rel.Dot(U);
                lz = rel.Dot(D);
                if (lz < view.casterMinZ || lz > view.casterMaxZ || lz <= 1e-5f)
                {
                    outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                    return false;
                }
                if (isSpotCone)
                {
                    if (dist <= 1e-5f || (lz / dist) < view.transitionAndPadding.x)
                    {
                        outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                        return false;
                    }
                }
                projCoordX = lx / lz;
                projCoordY = ly / lz;
            }
            else
            {
                lx = biasedPos.Dot(R);
                ly = biasedPos.Dot(U);
                lz = biasedPos.Dot(D);
                if (lz < view.casterMinZ || lz > view.casterMaxZ)
                {
                    outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                    return false;
                }
                projCoordX = lx;
                projCoordY = ly;
            }

            const FLOAT32 invTexelCount = 1.0f / static_cast<FLOAT32>(kNVSPageTexelSize);
            const FLOAT32 centerShift   = (mipPagesAxis == 1u) ? 0.5f : 0.0f;
            const FLOAT32 exactPageX    = (projCoordX / mipPageSize) + centerShift + (texelPageOffsetXY.x * invTexelCount);
            const FLOAT32 exactPageY    = (projCoordY / mipPageSize) + centerShift - (texelPageOffsetXY.y * invTexelCount);
            const FLOAT32 flX           = std::floor(exactPageX);
            const FLOAT32 flY           = std::floor(exactPageY);
            const INT32   wPageX        = static_cast<INT32>(flX);
            const INT32   wPageY        = static_cast<INT32>(flY);
            const INT32   halfPages     = static_cast<INT32>(mipPagesAxis / 2u);
            const INT32   snapOrigX     = isPerspective ? 0 : view.snappedOriginPageX;
            const INT32   snapOrigY     = isPerspective ? 0 : view.snappedOriginPageY;
            const INT32   minPageX      = snapOrigX - halfPages;
            const INT32   minPageY      = snapOrigY - halfPages;
            const INT32   relX          = wPageX - minPageX;
            const INT32   relY          = wPageY - minPageY;
            const INT32   lim           = static_cast<INT32>(mipPagesAxis);

            if (relX < 0 || relX >= lim || relY < 0 || relY >= lim)
            {
                outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                return false;
            }

            const UINT32 vx = (mipPagesAxis == 1u) ? 0u : WorldPageToNVSToroidalPage(wPageX, mipPagesAxis);
            const UINT32 vy = (mipPagesAxis == 1u) ? 0u : WorldPageToNVSToroidalPage(wPageY, mipPagesAxis);

            outCand.virtualPageX = vx;
            outCand.virtualPageY = vy;
            outCand.worldPageX   = wPageX;
            outCand.worldPageY   = wPageY;

            const FLOAT32 fracX = exactPageX - flX;
            const FLOAT32 fracY = exactPageY - flY;
            const FLOAT32 rawU  = fracX;
            const FLOAT32 rawV  = 1.0f - fracY;
            const FLOAT32 minUV = 0.5f * invTexelCount;
            const FLOAT32 maxUV = 1.0f - minUV;
            outCand.rawSliceUV        = Vec2(rawU, rawV);
            outCand.borderSafeSliceUV = Vec2(std::clamp(rawU, minUV, maxUV), std::clamp(rawV, minUV, maxUV));

            const FLOAT32 dz = view.casterMaxZ - view.casterMinZ;
            FLOAT32 ndcZ = 0.0f;
            if (isPerspective)
            {
                const FLOAT32 pz = isReversedZ ? (-view.casterMinZ / dz) : (view.casterMaxZ / dz);
                const FLOAT32 qz = isReversedZ ? ((view.casterMinZ * view.casterMaxZ) / dz) : (-(view.casterMinZ * view.casterMaxZ) / dz);
                ndcZ = pz + (qz / lz);
            }
            else
            {
                ndcZ = isReversedZ ? ((view.casterMaxZ - lz) / dz) : ((lz - view.casterMinZ) / dz);
            }

            FLOAT32 slopeFactor = 1.0f;
            if (safeNormal.Dot(safeNormal) > 0.5f)
            {
                const FLOAT32 cosSlope = std::clamp(std::fabs(safeNormal.Dot(lightRayDir)), 0.4f, 1.0f);
                slopeFactor            = std::clamp(1.0f / cosSlope, 1.0f, 2.5f);
            }

            const FLOAT32 absViewDepthBias = std::fabs(view.depthBiasNdc);
            FLOAT32       effDepthBiasNdc  = 0.0f;
            if (absViewDepthBias > 0.0f)
            {
                if (isPerspective && isTexelRelBias)
                {
                    const FLOAT32 fullBias   = (mipScale * absViewDepthBias * slopeFactor) / std::max(lz, view.casterMinZ);
                    const FLOAT32 rasterPart = rasterAppliedDB ? ((mipScale * absViewDepthBias) / std::max(view.casterMaxZ, view.casterMinZ)) : 0.0f;
                    effDepthBiasNdc          = std::max(fullBias - rasterPart, 0.0f);
                }
                else if (isTexelRelBias)
                {
                    effDepthBiasNdc = rasterAppliedDB
                        ? (std::max(slopeFactor - 1.0f, 0.0f) * absViewDepthBias)
                        : (slopeFactor * absViewDepthBias);
                }
                else
                {
                    effDepthBiasNdc = rasterAppliedDB ? 0.0f : absViewDepthBias;
                }
            }

            const FLOAT32 biasedNdc      = isReversedZ ? (ndcZ + effDepthBiasNdc) : (ndcZ - effDepthBiasNdc);
            outCand.receiverNdcDepth     = std::clamp(ndcZ, 0.0f, 1.0f);
            outCand.receiverCompareDepth = std::clamp(biasedNdc, 0.0f, 1.0f);

            if (!isPerspective && view.worldExtent > 1e-5f)
            {
                const FLOAT32 halfExt = 0.5f * view.worldExtent;
                const FLOAT32 dxNorm  = std::fabs(lx - view.lightRightAndCenterX.w) / halfExt;
                const FLOAT32 dyNorm  = std::fabs(ly - view.lightUpAndCenterY.w) / halfExt;
                FLOAT32       maxNorm = std::max(dxNorm, dyNorm);
                if ((view.flags & NVS_CLIPMAP_LEVEL_FLAG_RADIAL_SELECT) != 0u && view.lightViewToWorld.m[15] > 1e-4f)
                {
                    const Vec3 camPos(view.lightViewToWorld.m[12], view.lightViewToWorld.m[13], view.lightViewToWorld.m[14]);
                    const Vec3 camRel = worldPos - camPos;
                    maxNorm = std::max(maxNorm, std::sqrt(camRel.Dot(camRel)) / view.lightViewToWorld.m[15]);
                }
                const FLOAT32 startR = std::clamp(view.transitionAndPadding.x, 0.5f, 0.99f);
                if (maxNorm > startR)
                {
                    const FLOAT32 t           = std::clamp((maxNorm - startR) / std::max(1.0f - startR, 1e-4f), 0.0f, 1.0f);
                    outCand.clipmapTransition = t * t * (3.0f - 2.0f * t);
                }
            }

            UINT32 vFlat = kNVSInvalidVirtualPage;
            if (sparsePageHierarchy != nullptr && sparseRootCapacity > 0u)
            {
                if (viewIndex >= sparseRootCapacity)
                {
                    outCand.status = NVS_SAMPLE_STATUS_MISSING;
                    return false;
                }
                const UINT32 dirSlot = sparsePageHierarchy[viewIndex];
                if (dirSlot == kNVSSparseUnmappedIndex || dirSlot >= sparseDirCapacity)
                {
                    outCand.status = NVS_SAMPLE_STATUS_MISSING;
                    return false;
                }
                UINT32 tileOffset = 0u, localIdx = 0u;
                if (!ComputeNVSSparseDirectoryOffset(evalMip, vx, vy, tileOffset, localIdx))
                {
                    outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                    return false;
                }
                const UINT32 blockSlot = sparsePageHierarchy[sparseRootCapacity + dirSlot * kNVSSparseDirStride + tileOffset];
                if (blockSlot == kNVSSparseUnmappedIndex)
                {
                    outCand.status = NVS_SAMPLE_STATUS_MISSING;
                    return false;
                }
                vFlat = (blockSlot << 6u) + localIdx;
            }
            else
            {
                UINT32 tablePagesAxis = view.pagesPerAxis;
                if (viewTableCount > 0u && totalVirtualPages >= viewTableCount)
                {
                    const UINT32 tablePagesPerSlot = totalVirtualPages / viewTableCount;
                    const UINT32 inferredAxis      = static_cast<UINT32>(std::sqrt(static_cast<FLOAT32>(tablePagesPerSlot)) + 0.5f);
                    if (inferredAxis * inferredAxis == tablePagesPerSlot && inferredAxis >= view.pagesPerAxis)
                    {
                        tablePagesAxis = inferredAxis;
                    }
                }
                vFlat = ComputeNVSVirtualPageFlatIndex(view.lightId, view.clipmapLevel, vx, vy, tablePagesAxis, kNVSMaxClipmapLevels);
            }

            if (vFlat >= totalVirtualPages)
            {
                outCand.status = NVS_SAMPLE_STATUS_OUT_OF_RANGE;
                return false;
            }
            outCand.virtualFlatIndex = vFlat;

            const NVSVirtualPageEntry& vEntry = virtualPageTable[vFlat];
            const NVSPageState         vState = ExtractNVSPageState(vEntry.stateAndFlags);
            if (vState == NVS_PAGE_STATE_UNMAPPED || vState == NVS_PAGE_STATE_EVICTED)
            {
                outCand.status = NVS_SAMPLE_STATUS_MISSING;
                return false;
            }
            if (vState == NVS_PAGE_STATE_ALLOCATED_NOT_CLEARED || vState == NVS_PAGE_STATE_CLEARED_NOT_RENDERED)
            {
                outCand.status = NVS_SAMPLE_STATUS_NOT_RENDERED;
                return false;
            }
            if (vState == NVS_PAGE_STATE_INVALIDATED)
            {
                outCand.status = NVS_SAMPLE_STATUS_INVALIDATED;
                return false;
            }
            if (vState != NVS_PAGE_STATE_VALID_SHADOW_DATA)
            {
                outCand.status = NVS_SAMPLE_STATUS_MISSING;
                return false;
            }

            const UINT32 physIdx = vEntry.physicalPageIndex;
            if (physIdx >= physicalPoolCapacity)
            {
                outCand.status = NVS_SAMPLE_STATUS_GENERATION_MISMATCH;
                return false;
            }

            const NVSPhysicalPageEntry& pEntry = physicalPageTable[physIdx];
            const NVSPhysicalPageState  pState = ExtractNVSPhysicalPageState(pEntry.stateAndFlags);
            if ((vEntry.stateAndFlags & NVS_PAGE_FLAG_EXTENDED_VIEW_ADDR) != 0u ||
                (pEntry.stateAndFlags & NVS_PHYS_FLAG_EXTENDED_VIEW_ADDR) != 0u)
            {
                const NVSSparsePageAddress spUnpacked = UnpackNVSSparsePageCoord(pEntry.reserved, vEntry.virtualAddressPacked);
                if (spUnpacked.generation == 0u ||
                    pEntry.reserved != viewIndex ||
                    spUnpacked.mipLevel != evalMip ||
                    spUnpacked.virtualPageX != vx ||
                    spUnpacked.virtualPageY != vy)
                {
                    outCand.status = NVS_SAMPLE_STATUS_GENERATION_MISMATCH;
                    return false;
                }
            }
            else
            {
                const NVSVirtualPageAddress unpacked = UnpackNVSVirtualPageAddress(vEntry.virtualAddressPacked);
                if (unpacked.generation == 0u ||
                    unpacked.lightId != view.lightId ||
                    unpacked.clipmapLevel != view.clipmapLevel ||
                    unpacked.virtualPageX != vx ||
                    unpacked.virtualPageY != vy)
                {
                    outCand.status = NVS_SAMPLE_STATUS_GENERATION_MISMATCH;
                    return false;
                }
            }

            if (pState != NVS_PHYS_STATE_RESIDENT_VALID ||
                pEntry.physicalGeneration == 0u ||
                pEntry.ownerVirtualFlatIndex != vFlat ||
                pEntry.ownerVirtualAddress != vEntry.virtualAddressPacked)
            {
                outCand.status = NVS_SAMPLE_STATUS_GENERATION_MISMATCH;
                return false;
            }

            outCand.physicalPageIndex = physIdx;
            outCand.status            = NVS_SAMPLE_STATUS_VALID_SHADOW_DATA;
            return true;
        };

        NVSShadowSampleResult primaryRes{};
        if (evaluateAtMip(desiredMip, primaryRes))
        {
            return primaryRes;
        }

        // Deterministic coarse-mip fallback for Spot and Point virtual mip hierarchies
        if (isPerspective && mipCount > 1u && explicitMipOverride == 0xFFFFFFFFu &&
            primaryRes.status != NVS_SAMPLE_STATUS_OUT_OF_RANGE &&
            primaryRes.status != NVS_SAMPLE_STATUS_INVALID_LIGHT)
        {
            for (UINT32 fallbackMip = desiredMip + 1u; fallbackMip < mipCount; ++fallbackMip)
            {
                NVSShadowSampleResult coarseRes{};
                if (evaluateAtMip(fallbackMip, coarseRes))
                {
                    return coarseRes;
                }
            }
        }
        return primaryRes;
    }

    inline NVSShadowSampleResult ResolveNVSVirtualShadowSampleCPU(
        UINT32                        shadowViewBase,
        UINT32                        shadowViewCount,
        const NVSClipmapLevelGPUData* viewTable,
        UINT32                        viewTableCount,
        const NVSVirtualPageEntry*    virtualPageTable,
        UINT32                        totalVirtualPages,
        const NVSPhysicalPageEntry*   physicalPageTable,
        UINT32                        physicalPoolCapacity,
        const Vec3&                   worldPos,
        const Vec3&                   worldNormal          = Vec3(0.0f, 1.0f, 0.0f),
        Vec2                          texelPageOffsetXY    = Vec2(0.0f, 0.0f),
        BOOL                          samplingAvailable    = true,
        const UINT32*                 sparsePageHierarchy  = nullptr,
        UINT32                        sparseRootCapacity   = 0u,
        UINT32                        sparseDirCapacity    = 0u,
        FLOAT32                       screenPixelWorldSize = 0.0f) noexcept
    {
        NVSShadowSampleResult res{};
        if (!samplingAvailable)
        {
            res.status = NVS_SAMPLE_STATUS_AWAITING_UPDATE;
            return res;
        }
        if (!viewTable || !virtualPageTable || !physicalPageTable ||
            shadowViewCount == 0u ||
            shadowViewBase >= viewTableCount ||
            (shadowViewBase + shadowViewCount) > viewTableCount)
        {
            res.status = NVS_SAMPLE_STATUS_INVALID_LIGHT;
            return res;
        }

        const NVSClipmapLevelGPUData& firstView = viewTable[shadowViewBase];
        const BOOL isPerspective = (firstView.flags & NVS_CLIPMAP_LEVEL_FLAG_PERSPECTIVE) != 0u;

        // Point light 6-face omnidirectional cube selection
        if (isPerspective && shadowViewCount == 6u)
        {
            UINT32  bestFace = shadowViewBase;
            FLOAT32 bestLz   = -1e30f;
            for (UINT32 f = 0u; f < 6u; ++f)
            {
                const NVSClipmapLevelGPUData& fv = viewTable[shadowViewBase + f];
                if ((fv.flags & NVS_CLIPMAP_LEVEL_FLAG_VALID) == 0u)
                {
                    continue;
                }
                const Vec3 lightPos(fv.lightViewToWorld.m[12], fv.lightViewToWorld.m[13], fv.lightViewToWorld.m[14]);
                const Vec3 D(fv.lightDirAndExtrusion.x, fv.lightDirAndExtrusion.y, fv.lightDirAndExtrusion.z);
                const FLOAT32 lz = (worldPos - lightPos).Dot(D);
                if (lz > bestLz)
                {
                    bestLz   = lz;
                    bestFace = shadowViewBase + f;
                }
            }
            NVSShadowSampleResult bestRes = ResolveNVSVirtualPageLookupForViewCPU(
                bestFace,
                viewTable,
                viewTableCount,
                virtualPageTable,
                totalVirtualPages,
                physicalPageTable,
                physicalPoolCapacity,
                worldPos,
                worldNormal,
                texelPageOffsetXY,
                samplingAvailable,
                sparsePageHierarchy,
                sparseRootCapacity,
                sparseDirCapacity,
                screenPixelWorldSize);
            if (bestRes.status == NVS_SAMPLE_STATUS_OUT_OF_RANGE)
            {
                for (UINT32 f = 0u; f < 6u; ++f)
                {
                    const UINT32 candView = shadowViewBase + f;
                    if (candView == bestFace)
                    {
                        continue;
                    }
                    const NVSShadowSampleResult nbrRes = ResolveNVSVirtualPageLookupForViewCPU(
                        candView,
                        viewTable,
                        viewTableCount,
                        virtualPageTable,
                        totalVirtualPages,
                        physicalPageTable,
                        physicalPoolCapacity,
                        worldPos,
                        worldNormal,
                        texelPageOffsetXY,
                        samplingAvailable,
                        sparsePageHierarchy,
                        sparseRootCapacity,
                        sparseDirCapacity,
                        screenPixelWorldSize);
                    if (nbrRes.status != NVS_SAMPLE_STATUS_OUT_OF_RANGE &&
                        nbrRes.status != NVS_SAMPLE_STATUS_INVALID_LIGHT)
                    {
                        return nbrRes;
                    }
                }
            }
            return bestRes;
        }

        // Spot light (1 view) or Directional light clipmaps (finest covering level 0 -> 3)
        NVSShadowSampleResult firstCandidate{};
        BOOL                  hasCandidate = false;
        for (UINT32 i = 0u; i < shadowViewCount; ++i)
        {
            const NVSShadowSampleResult cand = ResolveNVSVirtualPageLookupForViewCPU(
                shadowViewBase + i,
                viewTable,
                viewTableCount,
                virtualPageTable,
                totalVirtualPages,
                physicalPageTable,
                physicalPoolCapacity,
                worldPos,
                worldNormal,
                texelPageOffsetXY,
                samplingAvailable,
                sparsePageHierarchy,
                sparseRootCapacity,
                sparseDirCapacity,
                screenPixelWorldSize);
            if (cand.status != NVS_SAMPLE_STATUS_OUT_OF_RANGE &&
                cand.status != NVS_SAMPLE_STATUS_INVALID_LIGHT)
            {
                return cand;
            }
            if (!hasCandidate)
            {
                firstCandidate = cand;
                hasCandidate   = true;
            }
        }
        return hasCandidate ? firstCandidate : res;
    }
}
