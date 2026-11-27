#pragma once

#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Northstar
{
    // =====================================================================
    //  Northstar Virtual Geometry (NVG) — Core Constants & ABI
    // =====================================================================

    inline constexpr UINT32 kNVG1BinaryMagic                = 0x3147564Eu; // "NVG1"
    inline constexpr UINT32 kNVG2BinaryMagic                = 0x3247564Eu; // "NVG2"
    inline constexpr UINT32 kNVGBinaryMagic                 = kNVG1BinaryMagic;
    inline constexpr UINT32 kNVGBinaryVersion               = 1u;
    inline constexpr UINT32 kNVG2BinaryVersion              = 2u;

    inline constexpr UINT32 kNVGPageSizeBytes               = 32768u;      // 32 KiB physical geometry page
    inline constexpr UINT32 kNVGPageWordCount               = kNVGPageSizeBytes / sizeof(UINT32); // 8192 UINT32 words
    inline constexpr UINT32 kNVGMaxClusterTriangles         = 128u;
    inline constexpr UINT32 kNVG1MaxClusterVertices         = 128u;
    inline constexpr UINT32 kNVGMaxClusterVertices          = 256u;
    inline constexpr UINT32 kNVGMaxMaterialSlots            = 256u;
    inline constexpr UINT32 kNVGClusterHeaderWords          = 16u;         // 64 bytes
    inline constexpr UINT32 kNVGRigidVertexWords            = 7u;          // 28 bytes (4.0x vs 112B Northstar::Vertex)
    inline constexpr UINT32 kNVGSkinnedVertexWords          = 11u;         // 44 bytes (2.55x vs 112B Northstar::Vertex)
    inline constexpr UINT32 kNVGMaxLODLevels                = 16u;
    inline constexpr UINT32 kNVGMaxBVHDepth                 = 16u;
    inline constexpr UINT32 kNVGInvalidIndex                = 0xFFFFFFFFu;
    inline constexpr UINT32 kNVGDefaultPhysicalPageCapacity = 2048u;       // 64 MiB initial baseline physical VRAM pool
    inline constexpr UINT32 kNVGMinPhysicalPageCapacity     = 8u;          // 256 KiB minimum pool capacity
    inline constexpr UINT32 kNVGMaxPhysicalPageCapacity     = 32768u;      // 1024 MiB (1 GiB) maximum geometry page pool
    inline constexpr UINT32 kNVGAutoPoolCapacity            = 0u;          // Autonomous elastic pool managed up to NVGMemoryBudget
    inline constexpr UINT32 kNVGDefaultMaxUploadsPerFrame   = 64u;         // 2 MiB/frame baseline steady-state upload rate
    inline constexpr UINT32 kNVGAutoUploadRate              = 0xFFFFFFFFu; // Autonomous adaptive burst/steady upload scheduler
    inline constexpr UINT32 kNVGMaxDrawBuckets              = 256u;
    inline constexpr UINT32 kNVGMaxCommandsPerBucket        = 262144u;
    inline constexpr UINT32 kNVGDecodedHashTableCapacity    = 262144u;     // Fixed 4 MiB power-of-2 open-addressed GPU decode table

    enum NVGBucketRangeFlags : UINT32
    {
        NVG_BUCKET_FLAG_NONE                 = 0u,
        NVG_BUCKET_FLAG_IA_STREAM            = 1u << 0, // Path-B GPU-decoded IA vertex + index stream
        NVG_BUCKET_FLAG_TRANSPARENT          = 1u << 1, // Back-to-front sorted transparent bucket
        NVG_BUCKET_FLAG_INDEXED_PULL         = 1u << 2, // Path-A Indexed Cluster Pull (0 VB words, unpacked IB + promoted cluster offset)
        NVG_BUCKET_FLAG_SW_ELIGIBLE          = 1u << 3, // Eligible for Compute Software Micropoly Rasterization (Standard-Z LESS/LESS_EQUAL)
        NVG_BUCKET_FLAG_SH_READS_DEPTH_BIAS  = 1u << 4, // Shadow VS applies NVS_DEPTH_BIAS_NDC / NVG_SHADOW_PAGE_DEPTH_BIAS
        NVG_BUCKET_FLAG_SH_READS_NORMAL_BIAS = 1u << 5, // Shadow VS applies NVS_NORMAL_BIAS_WORLD / NVG_SHADOW_PAGE_NORMAL_BIAS
        NVG_BUCKET_FLAG_SW_ELIGIBLE_REVZ     = 1u << 6  // Eligible for Compute Software Micropoly Rasterization (Reversed-Z GREATER/GREATER_EQUAL)
    };

    enum class NVGRasterAtomicStrategy : UINT8
    {
        AUTO                       = 0u, // Hardware-capability-driven: ATOMIC64_SINGLE_PASS when supported, else ATOMIC32_TWO_PASS
        FORCE_ATOMIC32_TWO_PASS    = 1u, // Force universal two-pass 2x32-bit atomic commit (OutAtomicDepthKey32 + OutAtomicPayload32)
        FORCE_ATOMIC64_SINGLE_PASS = 2u, // Force single-pass 64-bit atomic commit (explicitly rejected if GPU lacks 64-bit descriptor-heap atomics)
        DISABLED                   = 3u, // Disable camera compute software rasterizer (HW-only baseline)

        Auto                       = AUTO,
        ForceAtomic32TwoPass       = FORCE_ATOMIC32_TWO_PASS,
        ForceAtomic64SinglePass    = FORCE_ATOMIC64_SINGLE_PASS,
        Disabled                   = DISABLED
    };

    inline constexpr BOOL IsValidNVGRasterAtomicStrategy(NVGRasterAtomicStrategy strategy) noexcept
    {
        switch (strategy)
        {
            case NVGRasterAtomicStrategy::AUTO:
            case NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS:
            case NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS:
            case NVGRasterAtomicStrategy::DISABLED:
                return true;
            default:
                return false;
        }
    }

    // =====================================================================
    //  High-Level User Quality & Memory Budget Presets (NVS-Aligned)
    // =====================================================================

    enum class NVGQualityPreset : UINT8
    {
        ULTRA       = 0u, // 0.50 px screen error / 0.50 tx shadow error
        HIGH        = 1u, // 1.00 px screen error / 1.00 tx shadow error
        BALANCED    = 2u, // 1.50 px screen error / 1.50 tx shadow error
        PERFORMANCE = 3u, // 2.25 px screen error / 2.25 tx shadow error
        CUSTOM      = 4u, // Uses explicit maxScreenErrorPixels & shadowMaxTexelError

        Ultra       = ULTRA,
        High        = HIGH,
        Balanced    = BALANCED,
        Performance = PERFORMANCE,
        Custom      = CUSTOM
    };

    inline constexpr BOOL IsValidNVGQualityPreset(NVGQualityPreset preset) noexcept
    {
        switch (preset)
        {
            case NVGQualityPreset::ULTRA:
            case NVGQualityPreset::HIGH:
            case NVGQualityPreset::BALANCED:
            case NVGQualityPreset::PERFORMANCE:
            case NVGQualityPreset::CUSTOM:
                return true;
            default:
                return false;
        }
    }

    inline constexpr FLOAT32 ResolveNVGQualityScreenError(NVGQualityPreset preset, FLOAT32 customPixels = 1.0f) noexcept
    {
        switch (preset)
        {
            case NVGQualityPreset::ULTRA:       return 0.50f;
            case NVGQualityPreset::HIGH:        return 1.00f;
            case NVGQualityPreset::BALANCED:    return 1.50f;
            case NVGQualityPreset::PERFORMANCE: return 2.25f;
            case NVGQualityPreset::CUSTOM:
            default:                            return (customPixels > 0.05f) ? customPixels : 0.05f;
        }
    }

    enum class NVGMemoryBudget : UINT16
    {
        CUSTOM_PAGES = 0u,
        AUTO_128MB   = 128u,
        AUTO_256MB   = 256u,
        AUTO_512MB   = 512u,
        AUTO_1024MB  = 1024u,

        CustomPages  = CUSTOM_PAGES,
        Auto128MB    = AUTO_128MB,
        Auto256MB    = AUTO_256MB,
        Auto512MB    = AUTO_512MB,
        Auto1024MB   = AUTO_1024MB
    };

    inline constexpr UINT32 ResolveNVGMemoryBudgetPages(NVGMemoryBudget budget) noexcept
    {
        const UINT32 mb = static_cast<UINT32>(budget);
        if (mb == 0u) return kNVGDefaultPhysicalPageCapacity;
        const UINT64 bytes = static_cast<UINT64>(mb) * 1024ull * 1024ull;
        const UINT32 pages = static_cast<UINT32>(bytes / kNVGPageSizeBytes);
        if (pages < kNVGMinPhysicalPageCapacity) return kNVGMinPhysicalPageCapacity;
        if (pages > kNVGMaxPhysicalPageCapacity) return kNVGMaxPhysicalPageCapacity;
        return pages;
    }

    // =====================================================================
    //  Geometry Mode & Opt-In Build Configuration
    // =====================================================================

    enum class GeometryMode : UINT8
    {
        STANDARD    = 0,
        VIRTUALIZED = 1,

        Standard    = STANDARD,
        Virtualized = VIRTUALIZED
    };

    enum class NVGDebugVisualizationMode : UINT32
    {
        NONE               = 0u,
        CLUSTER_ID         = 1u,
        LOD_LEVEL          = 2u,
        VIRTUAL_PAGE       = 3u,
        PHYSICAL_PAGE      = 4u,
        GEOMETRIC_ERROR    = 5u,
        RESIDENCY_FALLBACK = 6u,
        NORMAL_CONE        = 7u,

        None               = NONE,
        ClusterId          = CLUSTER_ID,
        LodLevel           = LOD_LEVEL,
        VirtualPage        = VIRTUAL_PAGE,
        PhysicalPage       = PHYSICAL_PAGE,
        GeometricError     = GEOMETRIC_ERROR,
        ResidencyFallback  = RESIDENCY_FALLBACK,
        NormalCone         = NORMAL_CONE
    };

    struct NVGSubmeshRange
    {
        UINT32 startIndex   = 0u;
        UINT32 indexCount   = 0u;
        UINT32 materialSlot = 0u;
    };

    enum class NVGCompatibilityStatus : UINT32
    {
        COMPATIBLE                       = 0u, // Material & ShaderSet satisfy hardware contracts
        REJECTED_MISSING_SHADERSET       = 2u, // Material lacks required PassType ShaderSet
        REJECTED_MISSING_VARIANT         = 3u, // ShaderSet lacks required STANDARD/INSTANCED/SKINNED vertex variant
        REJECTED_MISSING_USER_BINDING    = 4u, // User Add() texture, sampler, StructuredBuffer, or GPUBuffer missing or destroyed
        REJECTED_INCOMPLETE_TESSELLATION = 5u, // ShaderSet declares only Hull or only Domain shader (both required by hardware)
        REJECTED_ROOT_PROMOTION_LIMIT    = 7u, // Indirect shader exceeds promoted binding budget (>8 per-draw CBVs or >2 promoted SRVs)
        REJECTED_MISSING_RT_ARRAY_IDX    = 9u, // Shadow ShaderSet lacks SV_RenderTargetArrayIndex output for NVS layered pass
        REJECTED_INVALID_MESH            = 10u,// Virtual mesh build failed or destroyed
        REJECTED_UNSUPPORTED_LAYOUT      = 11u // StructuredBuffer mixes incompatible physical domains or unrepresentable field layout
    };

    inline constexpr BOOL IsNVGCompatible(NVGCompatibilityStatus status) noexcept
    {
        return status == NVGCompatibilityStatus::COMPATIBLE;
    }

    struct NVGBuildConfig
    {
        UINT32  targetTrianglesPerCluster    = 128u;   // Clamped to [32 .. 128]
        FLOAT32 normalPreservationWeight     = 2.0f;
        FLOAT32 uvSeamWeight                 = 8.0f;
        FLOAT32 boundaryLockWeight           = 1000.0f;
        FLOAT32 thinFeatureWeight            = 4.0f;   // Preserves thin wires, railings, foliage silhouettes
        FLOAT32 simplificationRatioPerLevel  = 0.5f;   // Target triangle reduction ratio per DAG level
        BOOL    preserveSkinningAttributes   = true;
        BOOL    allowNonManifoldGeometry     = true;
        BOOL    generateShadowThinMetrics    = true;
        UINT32  workerThreadCount            = 0u;     // 0 = hardware concurrency, 1 = single-threaded (bit-identical output)
    };

    struct NVGBuildReport
    {
        BOOL        succeeded                  = false;
        UINT32      sourceVertexCount          = 0u;
        UINT32      sourceTriangleCount        = 0u;
        UINT32      validTriangleCount         = 0u;
        UINT32      degenerateTrianglesRemoved = 0u;
        UINT32      sanitizedVertexCount       = 0u;
        UINT32      nonManifoldEdgeCount       = 0u;
        UINT32      boundaryEdgeCount          = 0u;
        UINT32      seamEdgeCount              = 0u;
        UINT32      disconnectedIslandCount    = 0u;
        UINT32      totalClusterCount          = 0u;
        UINT32      lod0ClusterCount           = 0u;
        UINT32      rootClusterCount           = 0u;
        UINT32      lodLevelCount              = 0u;
        UINT32      clusterGroupCount          = 0u;
        UINT32      bvhNodeCount               = 0u;
        UINT32      totalVirtualPageCount      = 0u;
        UINT32      rootVirtualPageCount       = 0u;
        UINT32      streamingVirtualPageCount  = 0u;
        UINT64      uncompressedSourceBytes    = 0ull;
        UINT64      compressedPagesBytes       = 0ull;
        FLOAT32     compressionRatio           = 1.0f;
        FLOAT32     maxQuantizationError       = 0.0f;
        FLOAT32     rootGeometricError         = 0.0f;
        FLOAT32     buildTimeMs                = 0.0f;
        BOOL        hasSkinning                = false;
        std::string diagnosticMessage;
    };

    struct NVGMeshInfo
    {
        BOOL           valid                = false;
        UINT32         virtualMeshId        = 0u;
        UINT32         totalClusterCount    = 0u;
        UINT32         lod0ClusterCount     = 0u;
        UINT32         lodLevelCount        = 0u;
        UINT32         virtualPageCount     = 0u;
        UINT32         rootPageCount        = 0u;
        UINT32         bvhNodeCount         = 0u;
        BOOL           hasSkinning          = false;
        FLOAT32        compressionRatio     = 1.0f;
        FLOAT32        maxQuantizationError = 0.0f;
        NVGBuildReport buildReport{};
    };

    // Physical bounds & LOD tuning policy on Material
    struct NVGMaterialPolicy
    {
        FLOAT32 maxWorldDisplacement = 0.0f;  // Conservative world-space bounds expansion for vertex displacement/skinning
        FLOAT32 lodErrorScale        = 1.0f;  // Multiplier on screen-space error threshold (<1 = finer, >1 = coarser)
        FLOAT32 shadowLodErrorScale  = 1.0f;  // Independent multiplier on NVS shadow texel error threshold
        BOOL    lockToLOD0           = false; // Locks automatic cluster LOD to LOD 0 while keeping GPU culling & paging
    };

    // Engine-wide runtime NVG configuration:
    // High-level user preferences (qualityPreset, memoryBudget, enableAutoQualityScaling, debugVisualizationMode)
    // drive all internal streaming, culling, queue sizing, and shadow synchronization autonomously.
    // Optional low-level fields remain available for explicit low-level tuning when explicitly overridden.
    struct NVGConfig
    {
        BOOL                      enabled                    = true;
        NVGQualityPreset          qualityPreset              = NVGQualityPreset::CUSTOM;
        NVGMemoryBudget           memoryBudget               = NVGMemoryBudget::AUTO_512MB;
        BOOL                      enableAutoQualityScaling   = true;  // Smoothly adapts effective error threshold if working set exceeds VRAM budget
        FLOAT32                   maxScreenErrorPixels       = 1.0f;
        FLOAT32                   shadowMaxTexelError        = 1.0f;
        FLOAT32                   lodHysteresis              = 0.08f; // Internal temporal hysteresis margin
        FLOAT32                   thinFeatureShadowBoost     = 3.0f;  // Internal thin-silhouette preservation factor
        BOOL                      enableHZBOcclusion         = true;
        BOOL                      enableTwoPhaseDisocclusion = true;
        BOOL                      enableBackfaceConeCulling  = true;
        BOOL                      enableSmallClusterCulling  = true;
        FLOAT32                   smallClusterPixelThreshold = 0.75f;
        UINT32                    physicalPagePoolCapacity   = kNVGAutoPoolCapacity; // 0 = Autonomous elastic pool up to memoryBudget
        UINT32                    maxPageUploadsPerFrame     = kNVGAutoUploadRate;   // kNVGAutoUploadRate = Autonomous adaptive burst/steady scheduler
        UINT32                    minInFlightProtectFrames   = 4u;
        BOOL                      enableSoftwareRasterizer   = true;
        NVGRasterAtomicStrategy   swRasterAtomicStrategy     = NVGRasterAtomicStrategy::AUTO;
        FLOAT32                   swRasterMaxEdgePixels      = 32.0f;
        NVGDebugVisualizationMode debugVisualizationMode     = NVGDebugVisualizationMode::NONE;
    };

    // Runtime NVG telemetry & performance statistics
    struct NVGStats
    {
        BOOL                    active                        = false;
        NVGRasterAtomicStrategy activeSWRasterStrategy        = NVGRasterAtomicStrategy::DISABLED;
        UINT32  registeredVirtualMeshCount    = 0u;
        UINT64  effectiveSceneTriangles       = 0ull; // Total source triangles across active virtualized RenderObjects in scene
        UINT32  totalVirtualPages             = 0u;
        UINT32  residentPhysicalPages         = 0u;
        UINT32  pinnedRootPages               = 0u;
        UINT32  physicalPoolCapacity          = 0u;
        UINT32  physicalPoolBudgetPages       = 0u;
        UINT32  currentUploadBudgetPages      = 0u;
        FLOAT32 effectiveAdaptiveErrorScale   = 1.0f;
        FLOAT32 geometryPoolVramMB            = 0.0f;
        FLOAT32 metadataVramMB                = 0.0f;
        UINT32  pageFaultsThisFrame           = 0u;
        UINT32  fallbackClustersThisFrame     = 0u;
        UINT32  streamingRequestsQueued       = 0u;
        UINT32  pagesUploadedThisFrame        = 0u;
        UINT32  pagesEvictedThisFrame         = 0u;
        UINT32  evictionBlockedCount          = 0u;
        UINT32  streamingUploadedBytes        = 0u;
        UINT32  nvgObjectsEvaluated           = 0u;
        UINT32  nvgObjectsFrustumCulled       = 0u;
        UINT32  nvgObjectsHZBCulled           = 0u;
        UINT32  clustersTested                = 0u;
        UINT32  clustersFrustumCulled         = 0u;
        UINT32  clustersConeCulled            = 0u;
        UINT32  clustersHZBCulled             = 0u;
        UINT32  clustersDisocclusionRecovered = 0u;
        UINT32  clustersSmallCulled           = 0u;
        UINT32  visibleOpaqueClusters         = 0u;
        UINT32  visibleTransparentClusters    = 0u;
        UINT32  swRasterDispatches            = 0u;
        UINT32  swRasterizedClusters          = 0u;
        UINT32  swRasterizedTriangles         = 0u;
        UINT32  swCoveredPixels               = 0u;
        UINT32  swFallbackToHWClusters        = 0u;
        UINT32  decodedUniqueClusters         = 0u;
        UINT32  indirectDrawCommandsGenerated = 0u;
        UINT32  overflowDroppedClusters       = 0u;
        UINT32  shadowCastersEvaluated        = 0u;
        UINT32  shadowClustersTested          = 0u;
        UINT32  shadowClustersVisible         = 0u;
        UINT32  shadowSWRasterizedClusters    = 0u;
        UINT32  shadowIndirectDrawsGenerated  = 0u;
        UINT32  shadowOverflowPages           = 0u;
        UINT32  shadowNodeQueueCapacity       = 0u;
        UINT32  shadowCandidateCapacity       = 0u;
        FLOAT32 cameraAverageLODLevel         = 0.0f;
        FLOAT32 shadowAverageLODLevel         = 0.0f;
        UINT32  cameraLODHistogram[8]         = { 0, 0, 0, 0, 0, 0, 0, 0 };
        UINT32  shadowLODHistogram[8]         = { 0, 0, 0, 0, 0, 0, 0, 0 };
        FLOAT32 gpuHZBBuildMs                 = 0.0f;
        FLOAT32 gpuCullAndLODMs               = 0.0f;
        FLOAT32 gpuDecodeAndDrawGenMs         = 0.0f;
        FLOAT32 gpuSWRasterMs                 = 0.0f;
        FLOAT32 gpuSWRasterPass1Ms            = 0.0f;
        FLOAT32 gpuSWRasterPass2Ms            = 0.0f;
        FLOAT32 gpuDepthPrepassMs             = 0.0f;
        FLOAT32 gpuGeometryRasterMs           = 0.0f;
        FLOAT32 cpuStreamingMs                = 0.0f;

        void ResetPerFrame()
        {
            activeSWRasterStrategy        = NVGRasterAtomicStrategy::DISABLED;
            effectiveSceneTriangles       = 0ull;
            pageFaultsThisFrame           = 0u;
            fallbackClustersThisFrame     = 0u;
            streamingRequestsQueued       = 0u;
            pagesUploadedThisFrame        = 0u;
            pagesEvictedThisFrame         = 0u;
            evictionBlockedCount          = 0u;
            streamingUploadedBytes        = 0u;
            nvgObjectsEvaluated           = 0u;
            nvgObjectsFrustumCulled       = 0u;
            nvgObjectsHZBCulled           = 0u;
            clustersTested                = 0u;
            clustersFrustumCulled         = 0u;
            clustersConeCulled            = 0u;
            clustersHZBCulled             = 0u;
            clustersDisocclusionRecovered = 0u;
            clustersSmallCulled           = 0u;
            visibleOpaqueClusters         = 0u;
            visibleTransparentClusters    = 0u;
            swRasterDispatches            = 0u;
            swRasterizedClusters          = 0u;
            swRasterizedTriangles         = 0u;
            swCoveredPixels               = 0u;
            swFallbackToHWClusters        = 0u;
            decodedUniqueClusters         = 0u;
            indirectDrawCommandsGenerated = 0u;
            overflowDroppedClusters       = 0u;
            shadowCastersEvaluated        = 0u;
            shadowClustersTested          = 0u;
            shadowClustersVisible         = 0u;
            shadowSWRasterizedClusters    = 0u;
            shadowIndirectDrawsGenerated  = 0u;
            shadowOverflowPages           = 0u;
            cameraAverageLODLevel         = 0.0f;
            shadowAverageLODLevel         = 0.0f;
            for (UINT32 i = 0u; i < 8u; ++i)
            {
                cameraLODHistogram[i] = 0u;
                shadowLODHistogram[i] = 0u;
            }
            gpuHZBBuildMs         = 0.0f;
            gpuCullAndLODMs       = 0.0f;
            gpuDecodeAndDrawGenMs = 0.0f;
            gpuSWRasterMs         = 0.0f;
            gpuSWRasterPass1Ms    = 0.0f;
            gpuSWRasterPass2Ms    = 0.0f;
            gpuDepthPrepassMs     = 0.0f;
            gpuGeometryRasterMs   = 0.0f;
            cpuStreamingMs        = 0.0f;
        }
    };

    // =====================================================================
    //  Bit Flags Shared Between C++ Core & NSL Compute/Vertex Shaders
    // =====================================================================

    enum NVGClusterAttributeFlags : UINT32
    {
        NVG_CLUSTER_ATTR_NONE        = 0u,
        NVG_CLUSTER_ATTR_UV1         = 1u << 0,
        NVG_CLUSTER_ATTR_COLOR       = 1u << 1,
        NVG_CLUSTER_ATTR_SKINNING    = 1u << 2,
        NVG_CLUSTER_ATTR_TANGENT     = 1u << 3,
        NVG_CLUSTER_ATTR_VERT_BIT8   = 1u << 4, // 9th bit of vertexCount (enables 1..256 unique cluster vertices)
        NVG_CLUSTER_ATTR_INT_LATTICE = 1u << 5  // Global 2^P integer lattice + INT32 intOrigin + lodShift encoding
    };

    enum NVGVirtualPageFlags : UINT32
    {
        NVG_VPAGE_FLAG_NONE        = 0u,
        NVG_VPAGE_FLAG_RESIDENT    = 1u << 0,
        NVG_VPAGE_FLAG_ROOT_PINNED = 1u << 1,
        NVG_VPAGE_FLAG_VALID       = 1u << 2
    };

    enum NVGMeshGPUFlags : UINT32
    {
        NVG_MESH_FLAG_NONE         = 0u,
        NVG_MESH_FLAG_VALID        = 1u << 0,
        NVG_MESH_FLAG_HAS_SKINNING = 1u << 1
    };

    enum NVGMaterialGPUFlags : UINT32
    {
        NVG_MAT_FLAG_NONE            = 0u,
        NVG_MAT_FLAG_VALID           = 1u << 0,
        NVG_MAT_FLAG_VERTEX_PULL     = 1u << 1,
        NVG_MAT_FLAG_IA_STREAM       = 1u << 2,
        NVG_MAT_FLAG_ALPHA_TESTED    = 1u << 3,
        NVG_MAT_FLAG_TWO_SIDED       = 1u << 4,
        NVG_MAT_FLAG_TRANSPARENT     = 1u << 5,
        NVG_MAT_FLAG_LOCK_LOD0       = 1u << 6,
        NVG_MAT_FLAG_HAS_SHADOW_PASS = 1u << 7,
        NVG_MAT_FLAG_SHADOW_PULL     = 1u << 8,
        NVG_MAT_FLAG_HAS_DEPTH_PASS  = 1u << 9,
        NVG_MAT_FLAG_FRONT_CULL      = 1u << 10,
        NVG_MAT_FLAG_HAS_SKINNING    = 1u << 11,
        NVG_MAT_FLAG_HAS_INSTANCED   = 1u << 12
    };

    enum NVGVisibleClusterFlags : UINT32
    {
        NVG_VIS_FLAG_NONE               = 0u,
        NVG_VIS_FLAG_RESIDENCY_FALLBACK = 1u << 0,
        NVG_VIS_FLAG_DISOCCLUDED        = 1u << 1,
        NVG_VIS_FLAG_SHADOW_CASTER      = 1u << 2,
        NVG_VIS_FLAG_TRANSPARENT        = 1u << 3,
        NVG_VIS_FLAG_SKINNED            = 1u << 4
    };

    // =====================================================================
    //  GPU-Aligned POD Structures (1:1 Match with NSL StructuredBuffers)
    // =====================================================================

    // 64 bytes — Per-Virtual-Mesh lookup table indexed by Mesh::GetGPUSceneMeshId()
    struct alignas(16) NVGMeshGPUData
    {
        UINT32 meshId           = 0u;
        UINT32 flags            = 0u;
        UINT32 rootNodeIndex    = 0u;
        UINT32 nodeCount        = 0u;

        UINT32 clusterBaseIndex = 0u;
        UINT32 clusterCount     = 0u;
        UINT32 groupBaseIndex   = 0u;
        UINT32 groupCount       = 0u;

        Vec3   localBoundsMin   = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 virtualPageBase  = 0u;

        Vec3   localBoundsMax   = Vec3(0.0f, 0.0f, 0.0f);
        UINT32 virtualPageCount = 0u;
    };
    static_assert(sizeof(NVGMeshGPUData) == 64, "NVGMeshGPUData must be 64 bytes.");

    // 64 bytes — Per-Material NVG routing & compatibility descriptor indexed by Material::GetGPUSceneMaterialId()
    struct alignas(16) NVGMaterialGPUData
    {
        UINT32  materialId               = 0u;
        UINT32  nvgFlags                 = 0u;
        UINT32  opaqueBucketIndex        = kNVGInvalidIndex;
        UINT32  depthBucketIndex         = kNVGInvalidIndex;

        UINT32  transparentBucketIndex   = kNVGInvalidIndex;
        UINT32  shadowBucketIndex        = kNVGInvalidIndex;
        UINT32  opaqueInstBucketIndex    = kNVGInvalidIndex;
        UINT32  shadowInstBucketIndex    = kNVGInvalidIndex;

        FLOAT32 maxWorldDisplacement     = 0.0f;
        FLOAT32 lodErrorScale            = 1.0f;
        FLOAT32 shadowLodErrorScale      = 1.0f;
        UINT32  compatStatus             = 0u;

        UINT32  depthInstBucketIndex     = kNVGInvalidIndex;
        UINT32  transInstBucketIndex     = kNVGInvalidIndex;
        UINT32  skinnedOpaqueBucketIndex = kNVGInvalidIndex;
        UINT32  skinnedShadowBucketIndex = kNVGInvalidIndex;
    };
    static_assert(sizeof(NVGMaterialGPUData) == 64, "NVGMaterialGPUData must be 64 bytes.");

    // 64 bytes — Cluster Hierarchy BVH Node for coarse-to-fine GPU traversal
    struct alignas(16) NVGHierarchyNodeGPUData
    {
        Vec3    boundsMin          = Vec3(0.0f, 0.0f, 0.0f);
        UINT32  childStart         = 0u;

        Vec3    boundsMax          = Vec3(0.0f, 0.0f, 0.0f);
        UINT32  childCountAndFlags = 0u; // [0..15] count, [16] isLeaf (points to clusters), [17..23] minLod, [24..31] maxLod

        Vec4    lodSphere          = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

        FLOAT32 maxParentError     = 0.0f;
        FLOAT32 minSelfError       = 0.0f;
        UINT32  meshId             = 0u;
        UINT32  reserved           = 0u;
    };
    static_assert(sizeof(NVGHierarchyNodeGPUData) == 64, "NVGHierarchyNodeGPUData must be 64 bytes.");

    // 48 bytes — Cluster Group metadata for crack-free residency & synchronized DAG cuts
    struct alignas(16) NVGClusterGroupGPUData
    {
        UINT32  meshId             = 0u;
        UINT32  lodLevel           = 0u;
        UINT32  clusterStart       = 0u; // Global cluster index of first cluster belonging to this group
        UINT32  clusterCount       = 0u;

        UINT32  parentGroupIndex   = kNVGInvalidIndex; // Global group index of parent group (kNVGInvalidIndex for root)
        UINT32  childGroupStart    = 0u;               // Offset into NVGChildGroupIndices list
        UINT32  childGroupCount    = 0u;               // 0 for LOD 0 groups
        UINT32  firstVirtualPageId = kNVGInvalidIndex;

        Vec4    groupSphere        = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    static_assert(sizeof(NVGClusterGroupGPUData) == 48, "NVGClusterGroupGPUData must be 48 bytes.");

    // 128 bytes — Per-Cluster geometric, error, bounds, normal cone, page & hierarchy metadata
    struct alignas(16) NVGClusterMetaGPUData
    {
        // Row 0 (0..15)
        UINT32  meshId                  = 0u;
        UINT32  clusterLocalIndex       = 0u;
        UINT32  virtualPageId           = kNVGInvalidIndex;
        UINT32  clusterWordOffsetInPage = 0u;

        // Row 1 (16..31)
        UINT32  countsAndLod            = 0u; // [0..7] vertCount, [8..15] triCount, [16..23] lodLevel, [24..31] attrFlags
        UINT32  selfGroupIndex          = 0u; // Global index in NVGClusterGroupGPUData
        UINT32  parentGroupIndex        = kNVGInvalidIndex;
        UINT32  refinedByChildGroupStart = 0u; // Offset into child group list that refines this cluster (if lodLevel > 0)

        // Row 2 (32..47)
        Vec3    aabbMin                 = Vec3(0.0f, 0.0f, 0.0f);
        FLOAT32 selfGroupError          = 0.0f;

        // Row 3 (48..63)
        Vec3    aabbMax                 = Vec3(0.0f, 0.0f, 0.0f);
        FLOAT32 parentGroupError        = 0.0f; // Very large (+1e30f) for root clusters

        // Row 4 (64..79)
        Vec4    selfGroupSphere         = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

        // Row 5 (80..95)
        Vec4    parentGroupSphere       = Vec4(0.0f, 0.0f, 0.0f, 0.0f);

        // Row 6 (96..111)
        Vec4    normalConeAxisAndSin    = Vec4(0.0f, 1.0f, 0.0f, -1.0f); // xyz = unit axis, w = sin(halfAngle) or <0 if disabled

        // Row 7 (112..127)
        FLOAT32 thinFeatureMetric       = 0.0f; // [0..1+] ratio measuring thin wires/railings/silhouettes
        FLOAT32 maxEdgeLength           = 0.0f;
        UINT32  refinedByChildGroupCount = 0u;
        UINT32  clusterFlags            = 0u;
    };
    static_assert(sizeof(NVGClusterMetaGPUData) == 128, "NVGClusterMetaGPUData must be 128 bytes.");

    // 16 bytes — Virtual-to-Physical geometry page table entry
    struct alignas(16) NVGVirtualPageGPUData
    {
        UINT32 stateAndFlags     = 0u; // NVGVirtualPageFlags
        UINT32 physicalPageIndex = kNVGInvalidIndex;
        UINT32 pageGeneration    = 0u;
        UINT32 meshIdAndLod      = 0u;
    };
    static_assert(sizeof(NVGVirtualPageGPUData) == 16, "NVGVirtualPageGPUData must be 16 bytes.");

    // 48 bytes — Compacted visible cluster output consumed by Vertex-Pulling shaders, Cluster Decode & Indirect Draw Builder
    struct alignas(16) NVGVisibleClusterGPUData
    {
        UINT32  objectSlotIndex       = 0u;
        UINT32  clusterId             = 0u;
        UINT32  physicalPageIndex     = kNVGInvalidIndex;
        UINT32  clusterWordOffset     = 0u;

        UINT32  virtualPageId         = kNVGInvalidIndex;
        UINT32  lodAndFlags           = 0u; // [0..7] lodLevel, [8..15] NVGVisibleClusterFlags, [16..31] shadowPageSlice
        FLOAT32 geometricError        = 0.0f;
        UINT32  materialId            = 0u;

        UINT32  bonePaletteOffset     = 0u;
        UINT32  boneCount             = 0u;
        UINT32  drawBucketIndex       = 0u;
        FLOAT32 viewDepthKey          = 0.0f; // View-space depth for transparent sorting / shadow page index in shadow pass
    };
    static_assert(sizeof(NVGVisibleClusterGPUData) == 48, "NVGVisibleClusterGPUData must be 48 bytes.");

    // 16 bytes — Per-Draw-Bucket range inside compacted visible cluster array
    struct alignas(16) NVGBucketRangeGPUData
    {
        UINT32 firstVisibleCluster = 0u;
        UINT32 visibleClusterCount = 0u;
        UINT32 maxTriangleCount    = 0u;
        UINT32 bucketFlags         = 0u;
    };
    static_assert(sizeof(NVGBucketRangeGPUData) == 16, "NVGBucketRangeGPUData must be 16 bytes.");

    // 80 bytes — Per-active-shadow-page projection & bias view for NVG vertex-pulling shadow shaders
    struct alignas(16) NVGShadowPageViewGPUData
    {
        Mat4    worldToPageClip  = Mat4::Identity();
        UINT32  physicalPageSlice = 0u;
        FLOAT32 depthBiasNdc     = 0.0f;
        FLOAT32 normalBiasWorld  = 0.0f;
        FLOAT32 texelWorldSize   = 0.0f;
    };
    static_assert(sizeof(NVGShadowPageViewGPUData) == 80, "NVGShadowPageViewGPUData must be 80 bytes.");

    // 16 bytes — Node/Cluster traversal queue item
    struct alignas(16) NVGTraversalItemGPUData
    {
        UINT32 objectSlotIndex = 0u;
        UINT32 nodeOrClusterId = 0u;
        UINT32 drawBucketHint  = 0u;
        UINT32 auxData         = 0u; // shadowPageRenderDataIndex in shadow pass, flags in camera pass
    };
    static_assert(sizeof(NVGTraversalItemGPUData) == 16, "NVGTraversalItemGPUData must be 16 bytes.");

    // 16 bytes — Decoded cluster cache entry for Path-B IA-compatible shaders
    struct alignas(16) NVGDecodedClusterRecordGPUData
    {
        UINT32 decodedFrameMarker = 0u; // Frame tag (0 = not yet decoded this frame)
        UINT32 vertexStartIndex   = 0u; // Start index in DecodedVertexPool (stride 112B)
        UINT32 indexStartOffset   = 0u; // Start index in DecodedIndexPool (stride 4B)
        UINT32 indexCount         = 0u; // triangleCount * 3
    };
    static_assert(sizeof(NVGDecodedClusterRecordGPUData) == 16, "NVGDecodedClusterRecordGPUData must be 16 bytes.");

    // 160 bytes — GPU-written telemetry counters (read back asynchronously via 3-slot non-blocking ring)
    struct alignas(16) NVGTelemetryGPUData
    {
        UINT32 objectsEvaluated            = 0u;
        UINT32 objectsFrustumCulled        = 0u;
        UINT32 objectsHZBCulled            = 0u;
        UINT32 clustersTested              = 0u;

        UINT32 clustersFrustumCulled       = 0u;
        UINT32 clustersConeCulled          = 0u;
        UINT32 clustersHZBCulled           = 0u;
        UINT32 clustersSmallCulled         = 0u;

        UINT32 visibleOpaqueClusters       = 0u;
        UINT32 visibleTransparentClusters  = 0u;
        UINT32 disocclusionRecovered       = 0u;
        UINT32 fallbackClusters            = 0u;

        UINT32 pageFaults                  = 0u;
        UINT32 decodedUniqueClusters       = 0u;
        UINT32 indirectDrawsGenerated      = 0u;
        UINT32 overflowDroppedClusters     = 0u;

        UINT32 shadowCastersEvaluated      = 0u;
        UINT32 shadowClustersTested        = 0u;
        UINT32 shadowClustersVisible       = 0u;
        UINT32 shadowIndirectDraws         = 0u;

        UINT32 cameraLODHistogram[8]       = { 0, 0, 0, 0, 0, 0, 0, 0 };
        UINT32 shadowLODHistogram[8]       = { 0, 0, 0, 0, 0, 0, 0, 0 };
        UINT32 cameraLODSum                = 0u;
        UINT32 shadowLODSum                = 0u;
        UINT32 reserved0                   = 0u;
        UINT32 reserved1                   = 0u;

        UINT32 swRasterizedClusters        = 0u;
        UINT32 shadowSWRasterizedClusters  = 0u;
        UINT32 swFallbackToHWClusters      = 0u;
        UINT32 swRasterizedTriangles       = 0u;

        UINT32 swCoveredPixels             = 0u;
        UINT32 reserved4                   = 0u;
        UINT32 reserved5                   = 0u;
        UINT32 reserved6                   = 0u;
    };
    static_assert(sizeof(NVGTelemetryGPUData) == 192, "NVGTelemetryGPUData must be 192 bytes.");

    // 112 bytes (7 x 16B) — Reflection-driven indirect draw binding descriptor for NVG buckets.
    // Supports up to 8 promoted per-draw CBVs, up to 2 promoted Root SRVs (Instance + Bone Slice),
    // Path-A Indexed Cluster Pull (decodeVertexWords == 0), and compact IA vertex strides (12B..112B).
    struct alignas(16) NVGDrawGroupShaderBindingDesc
    {
        // Row 0 (0..15)
        UINT32 bindingFlags               = 0u; // bit 0 = valid, bit 1 = extended 144B command, bit 2 = indexed pull (null VBV)
        UINT32 patchFirstIndex            = 0u;
        UINT32 patchCount                 = 0u;
        UINT32 instanceStrideBytes        = 0u;

        // Row 1 (16..31): Byte sizes for Promoted CBVs 0..3
        UINT32 cbvByteSize[4]             = { 0u, 0u, 0u, 0u };

        // Row 2 (32..47): Byte sizes for Promoted CBVs 4..7
        UINT32 cbvExtByteSize[4]          = { 0u, 0u, 0u, 0u };

        // Row 3 (48..63): Template word offsets for Promoted CBVs 0..3
        UINT32 cbvTemplateWordOffset[4]   = { 0u, 0u, 0u, 0u };

        // Row 4 (64..79): Template word offsets for Promoted CBVs 4..7
        UINT32 cbvExtTemplateWordOffset[4]= { 0u, 0u, 0u, 0u };

        // Row 5 (80..95)
        UINT32 instanceTemplateWordOffset = 0u;
        UINT32 activeStageMask            = 0u;
        UINT32 cbvCount                   = 0u; // 0..8
        UINT32 srv0Mode                   = 0u; // NVSShadowIndirectSRVMode

        // Row 6 (96..111)
        UINT32 srv1Mode                   = 0u; // NVSShadowIndirectSRVMode (second promoted Root SRV when extended)
        UINT32 decodeVertexWords          = 28u;// 0 (Indexed Pull) or 3, 6, 9, 12, 14, 16, 20, 24, 28
        UINT32 decodeAttrMask             = 0x1FFu;
        UINT32 drawGroupIndex             = 0u; // [0..15] bucketIndex, [16..31] optional depthCompanionBucket + 1 (0 = none)
    };
    static_assert(sizeof(NVGDrawGroupShaderBindingDesc) == 112, "NVGDrawGroupShaderBindingDesc must be 112 bytes.");

    // =====================================================================
    //  Pluggable Geometry Page Data Source Contract (Zero Disk Coupling)
    // =====================================================================

    class INVGPageDataSource
    {
    public:
        virtual ~INVGPageDataSource() = default;

        // Reads compressed page words for virtualPageLocalIndex within a virtual mesh into outPageBytes (kNVGPageSizeBytes).
        virtual BOOL ReadVirtualPageData(
            UINT32 virtualMeshId,
            UINT32 localPageIndex,
            BYTE*  outPageBytes,
            UINT32 maxBytes) const = 0;
    };

    // Shared-ownership async-capable page stream provider for disk/archive/LZ4 streaming.
    // Thread-safe for concurrent background worker reads across non-overlapping destination buffers.
    class INVGPageStreamProvider
    {
    public:
        virtual ~INVGPageStreamProvider() = default;

        virtual UINT32 GetVirtualPageCount() const noexcept = 0;

        // Reads raw compressed payload bytes from backing storage without decompressing on the caller thread.
        virtual BOOL ReadCompressedPage(
            UINT32  localPageIndex,
            BYTE*   outCompressedBytes,
            UINT32  maxBytes,
            UINT32& outBytesRead,
            UINT32& outUncompressedBytes,
            UINT32& outCodecFlags,
            UINT32& outExpectedCrc32c) const = 0;

        // Reads and decompresses a 32 KiB physical geometry page into outPageBytes (kNVGPageSizeBytes).
        virtual BOOL ReadDecompressedPage(
            UINT32 localPageIndex,
            BYTE*  outPageBytes,
            UINT32 maxBytes) const = 0;
    };

    // =====================================================================
    //  Packing Helpers
    // =====================================================================

    inline constexpr UINT32 PackNVGClusterCountsAndLod(
        UINT32 vertexCount,
        UINT32 triangleCount,
        UINT32 lodLevel,
        UINT32 attrFlags) noexcept
    {
        const UINT32 clampedVerts = (vertexCount > kNVGMaxClusterVertices) ? kNVGMaxClusterVertices : vertexCount;
        const UINT32 vertBit8     = ((clampedVerts >> 8u) & 1u) ? NVG_CLUSTER_ATTR_VERT_BIT8 : 0u;
        const UINT32 mergedFlags  = (attrFlags & ~NVG_CLUSTER_ATTR_VERT_BIT8) | vertBit8;
        return (clampedVerts & 0xFFu) |
               ((triangleCount & 0xFFu) << 8u) |
               ((lodLevel & 0xFFu) << 16u) |
               ((mergedFlags & 0xFFu) << 24u);
    }

    inline constexpr UINT32 UnpackNVGClusterVertexCount(UINT32 countsAndLod) noexcept
    {
        return (countsAndLod & 0xFFu) | (((countsAndLod >> 28u) & 1u) << 8u);
    }

    inline constexpr UINT32 UnpackNVGClusterTriangleCount(UINT32 countsAndLod) noexcept
    {
        return (countsAndLod >> 8u) & 0xFFu;
    }

    inline constexpr UINT32 PackNVGRefinedCountAndMaterialSlot(
        UINT32 childGroupCount,
        UINT32 materialSlot) noexcept
    {
        return (childGroupCount & 0xFFFFu) | ((materialSlot & 0xFFu) << 16u);
    }

    inline constexpr UINT32 UnpackNVGRefinedChildGroupCount(UINT32 packedValue) noexcept
    {
        return packedValue & 0xFFFFu;
    }

    inline constexpr UINT32 UnpackNVGClusterMaterialSlot(UINT32 packedValue) noexcept
    {
        return (packedValue >> 16u) & 0xFFu;
    }

    inline constexpr UINT32 PackNVGNodeChildCountAndFlags(
        UINT32 childCount,
        BOOL   isLeaf,
        UINT32 minLod,
        UINT32 maxLod) noexcept
    {
        return (childCount & 0xFFFFu) |
               (isLeaf ? (1u << 16u) : 0u) |
               ((minLod & 0x7Fu) << 17u) |
               ((maxLod & 0xFFu) << 24u);
    }

    // Packs [0..15] lodLevel and [16..31] virtualPageCount (0 in upper bits decodes to 1 for NVG1 backward compatibility).
    inline constexpr UINT32 PackNVGGroupLodAndPageCount(
        UINT32 lodLevel,
        UINT32 virtualPageCount) noexcept
    {
        const UINT32 clampedPages = (virtualPageCount == 0u)
            ? 1u
            : ((virtualPageCount > 65535u) ? 65535u : virtualPageCount);
        return (lodLevel & 0xFFFFu) | (clampedPages << 16u);
    }

    inline constexpr UINT32 UnpackNVGGroupLodLevel(UINT32 packedLodAndPages) noexcept
    {
        return packedLodAndPages & 0xFFFFu;
    }

    inline constexpr UINT32 UnpackNVGGroupPageCount(UINT32 packedLodAndPages) noexcept
    {
        const UINT32 pages = (packedLodAndPages >> 16u) & 0xFFFFu;
        return (pages > 0u) ? pages : 1u;
    }
}
