#include "core_resource_binding_resolver.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Northstar
{
    namespace
    {
        inline UINT64 Fnv1aInit() noexcept
        {
            return 14695981039346656037ull;
        }

        inline UINT64 Fnv1aMixU64(UINT64 h, UINT64 v) noexcept
        {
            h ^= v;
            h *= 1099511628211ull;
            return h;
        }

        inline UINT64 Fnv1aHashBytes(const BYTE* data, SIZE_T size) noexcept
        {
            UINT64 h = Fnv1aInit();
            if (!data || size == 0u) return h;
            for (SIZE_T i = 0u; i < size; ++i)
            {
                h ^= static_cast<UINT64>(data[i]);
                h *= 1099511628211ull;
            }
            return h;
        }

        inline const std::string& GetFieldPropertyName(const NSL::NSCStructField& f) noexcept
        {
            return f.propertyName.empty() ? f.name : f.propertyName;
        }

        inline const std::string& GetCBufferFieldPropertyName(const NSL::NSCField& f) noexcept
        {
            return f.propertyName.empty() ? f.name : f.propertyName;
        }

        inline BOOL IsPromotedCBVSlot(
            const IndirectRootBindingKey* key,
            ShaderStage                   stage,
            UINT16                        space,
            UINT16                        slot) noexcept
        {
            if (!key || !key->enabled) return false;
            for (UINT8 i = 0u; i < key->cbvCount && i < 3u; ++i)
            {
                if (key->cbvStage[i] == stage &&
                    key->cbvSpace[i] == space &&
                    key->cbvSlot[i] == slot)
                {
                    return true;
                }
            }
            return false;
        }

        inline BOOL IsPromotedSRVSlot(
            const IndirectRootBindingKey* key,
            ShaderStage                   stage,
            UINT16                        space,
            UINT16                        slot) noexcept
        {
            if (!key || !key->enabled || !key->hasPromotedSRV) return false;
            return (key->srvStage == stage && key->srvSpace == space && key->srvSlot == slot);
        }

        inline BOOL IsDomainGPUBacked(EngineResourceDomain domain) noexcept
        {
            switch (domain)
            {
                case EngineResourceDomain::GPU_SCENE_OBJECTS:
                case EngineResourceDomain::NVG_PAGE_POOL:
                case EngineResourceDomain::NVG_BUCKET_RANGES:
                case EngineResourceDomain::NVG_VISIBLE_CLUSTERS:
                case EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS:
                case EngineResourceDomain::NVS_SPARSE_PAGE_HIERARCHY:
                case EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE:
                case EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE:
                case EngineResourceDomain::NVS_LIGHT_VIEW_TABLE:
                case EngineResourceDomain::GLOBAL_BONE_PALETTE:
                case EngineResourceDomain::PER_OBJECT_BONE_SLICE:
                    return true;
                default:
                    return false;
            }
        }

        inline UINT32 GetCanonicalDomainStride(EngineResourceDomain domain) noexcept
        {
            switch (domain)
            {
                case EngineResourceDomain::GPU_SCENE_OBJECTS:         return static_cast<UINT32>(sizeof(GPUSceneObjectData));
                case EngineResourceDomain::NVG_PAGE_POOL:             return static_cast<UINT32>(sizeof(UINT32));
                case EngineResourceDomain::NVG_BUCKET_RANGES:         return static_cast<UINT32>(sizeof(NVGBucketRangeGPUData));
                case EngineResourceDomain::NVG_VISIBLE_CLUSTERS:      return static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData));
                case EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS:     return static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData));
                case EngineResourceDomain::NVS_SPARSE_PAGE_HIERARCHY: return static_cast<UINT32>(sizeof(UINT32));
                case EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE:    return static_cast<UINT32>(sizeof(NVSVirtualPageEntry));
                case EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE:   return static_cast<UINT32>(sizeof(NVSPhysicalPageEntry));
                case EngineResourceDomain::NVS_LIGHT_VIEW_TABLE:      return static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData));
                case EngineResourceDomain::GLOBAL_BONE_PALETTE:       return static_cast<UINT32>(sizeof(Mat4));
                case EngineResourceDomain::PER_OBJECT_BONE_SLICE:     return static_cast<UINT32>(sizeof(Mat4));
                default:                                              return 0u;
            }
        }

        static const EnginePropertyMetadata kPropertyMetadataTable[] =
        {
            // ==================== Camera (CBuffer) ====================
            { "CAMERA_POSITION",       EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "CAMERA_DIRECTION",      EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "VIEW_MATRIX",           EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "PROJECTION_MATRIX",     EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "VP_MATRIX",             EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "INV_VIEW_MATRIX",       EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "INV_PROJECTION_MATRIX", EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "INV_VP_MATRIX",         EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "NEAR_PLANE",            EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "FAR_PLANE",             EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "CAMERA_FOV",            EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "CAMERA_ASPECT",         EngineResourceDomain::USER_ONLY, EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu,  4u, 0u, false, false, false },

            // ==================== Transform & Object Identity (CBuffer) ====================
            { "WORLD_MATRIX",          EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 64u, 0u, false, false, true },
            { "NORMAL_MATRIX",         EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 48u, 0u, false, false, true },
            { "MVP_MATRIX",            EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 64u, 0u, false, false, true },
            { "INV_WORLD_MATRIX",      EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 64u, 0u, false, false, true },
            { "GPU_SCENE_OBJECT_SLOT", EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu,  4u, 0u, false, false, true },
            { "GEOMETRY_MODE",         EngineResourceDomain::USER_ONLY, EXEC_CTX_DRAW | EXEC_CTX_RENDER_OBJECT | EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, false },

            // ==================== Environment / Frame / Counters (CBuffer) ====================
            { "AMBIENT_COLOR",         EngineResourceDomain::USER_ONLY, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "AMBIENT_INTENSITY",     EngineResourceDomain::USER_ONLY, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "LIGHT_COUNT",           EngineResourceDomain::USER_ONLY, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "TIME",                  EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME,                  0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "ENGINE_TIME",           EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME,                  0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "DELTA_TIME",            EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME,                  0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "FRAME_INDEX",           EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME,                  0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "SCREEN_SIZE",           EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW, 0xFFFFFFFFu, 8u, 0u, false, false, false },

            // ==================== Skinning (CBuffer) ====================
            { "BONE_PALETTE_OFFSET",   EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },
            { "BONE_COUNT",            EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },
            { "SKINNING_REVISION",     EngineResourceDomain::USER_ONLY, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },

            // ==================== NVS Shadow Page Context (CBuffer) ====================
            { "NVS_PHYSICAL_PAGE_INDEX",  EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_VIRTUAL_PAGE_ADDRESS", EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_CLIPMAP_LEVEL",        EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_DEPTH_BIAS_NDC",       EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_NORMAL_BIAS_WORLD",    EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_RECEIVER_MIN_Z",       EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },
            { "NVS_RECEIVER_MAX_Z",       EngineResourceDomain::USER_ONLY, EXEC_CTX_SHADOW_PAGE, 0xFFFFFFFFu, 4u, 0u, false, true, true },

            // ==================== NVS Sampling Global Metadata (CBuffer) ====================
            { "NVS_PAGES_PER_AXIS",           EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_PHYSICAL_POOL_CAPACITY",   EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_PAGE_TEXEL_SIZE",          EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_VIEW_TABLE_COUNT",         EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_TOTAL_VIRTUAL_PAGES",      EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_DEPTH_MODE",               EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SAMPLING_AVAILABLE",       EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SNAPSHOT_FRAME_INDEX",     EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SPARSE_ROOT_CAPACITY",     EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SPARSE_DIR_STRIDE",        EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SPARSE_DIR_CAPACITY",      EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "NVS_SCREEN_PIXEL_WORLD_SCALE", EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_SCENE, 0xFFFFFFFFu, 16u, 0u, false, false, false },

            // ==================== NVG Draw & Telemetry Metadata (CBuffer) ====================
            { "NVG_DRAW_BUCKET_ID",           EngineResourceDomain::USER_ONLY, EXEC_CTX_CLUSTER,                0xFFFFFFFFu, 4u, 0u, true,  false, true  },
            { "NVG_VISIBLE_CLUSTER_COUNT",    EngineResourceDomain::USER_ONLY, EXEC_CTX_CLUSTER,                0xFFFFFFFFu, 4u, 0u, true,  false, true  },
            { "NVG_VISIBLE_CLUSTER_OFFSET",   EngineResourceDomain::USER_ONLY, EXEC_CTX_CLUSTER,                0xFFFFFFFFu, 4u, 0u, true,  false, true  },
            { "NVG_DEBUG_MODE",               EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, false },
            { "NVG_MAX_SCREEN_ERROR_PX",      EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, false },
            { "NVG_PHYSICAL_POOL_CAPACITY",   EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, false },
            { "NVG_PAGE_WORD_COUNT",          EngineResourceDomain::USER_ONLY, EXEC_CTX_FRAME | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, false },

            // ==================== Skinning & Instance StructuredBuffers ====================
            { "GLOBAL_BONE_MATRIX",    EngineResourceDomain::GLOBAL_BONE_PALETTE,   EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0u, 64u, 64u, false, false, false },
            { "BONE_MATRIX",           EngineResourceDomain::PER_OBJECT_BONE_SLICE, EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER, 0u, 64u, 64u, false, false, true },
            { "INSTANCE_WORLD_MATRIX", EngineResourceDomain::INSTANCES,             EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0u, 64u, 0u, false, false, true },
            { "INSTANCE_ID_VALUE",     EngineResourceDomain::INSTANCES,             EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },
            { "INSTANCE_CUSTOM_DATA",  EngineResourceDomain::INSTANCES,             EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },
            { "INSTANCE_BONE_OFFSET",  EngineResourceDomain::INSTANCE_SKINNING,     EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },
            { "INSTANCE_BONE_COUNT",   EngineResourceDomain::INSTANCE_SKINNING,     EXEC_CTX_INSTANCE | EXEC_CTX_CLUSTER, 0xFFFFFFFFu, 4u, 0u, false, false, true },

            // ==================== Global Lights StructuredBuffer ====================
            { "GLOBAL_LIGHT_POSITION",          EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "GLOBAL_LIGHT_COLOR",             EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "GLOBAL_LIGHT_DIRECTION",         EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "GLOBAL_LIGHT_INTENSITY",         EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_RANGE",             EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_TYPE",              EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_INNER_CONE",        EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_OUTER_CONE",        EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_ATTENUATION",       EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 12u, 0u, false, false, false },
            { "GLOBAL_LIGHT_VP_MATRIX",         EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu, 64u, 0u, false, false, false },
            { "GLOBAL_LIGHT_CASTS_SHADOW",      EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_SHADOW_VIEW_BASE",  EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_SHADOW_VIEW_COUNT", EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_NVS_LIGHT_ID",      EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_SHADOW_BIAS",       EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_NORMAL_BIAS",       EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },
            { "GLOBAL_LIGHT_SHADOW_SOFTNESS",   EngineResourceDomain::LIGHTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0xFFFFFFFFu,  4u, 0u, false, false, false },

            // ==================== NVS Sparse Page Hierarchy (4B) ====================
            { "NVS_SPARSE_NODE_VALUE", EngineResourceDomain::NVS_SPARSE_PAGE_HIERARCHY, EXEC_CTX_SCENE | EXEC_CTX_FRAME, 0u, 4u, 4u, false, false, false },

            // ==================== NVS Virtual Page Table (16B) ====================
            { "NVS_VPAGE_STATE_AND_FLAGS",    EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSVirtualPageEntry, stateAndFlags)),        4u, static_cast<UINT32>(sizeof(NVSVirtualPageEntry)), false, false, false },
            { "NVS_VPAGE_PHYSICAL_INDEX",     EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSVirtualPageEntry, physicalPageIndex)),    4u, static_cast<UINT32>(sizeof(NVSVirtualPageEntry)), false, false, false },
            { "NVS_VPAGE_VIRTUAL_ADDRESS",    EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSVirtualPageEntry, virtualAddressPacked)), 4u, static_cast<UINT32>(sizeof(NVSVirtualPageEntry)), false, false, false },
            { "NVS_VPAGE_LAST_TOUCHED_FRAME", EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSVirtualPageEntry, lastTouchedFrame)),     4u, static_cast<UINT32>(sizeof(NVSVirtualPageEntry)), false, false, false },

            // ==================== NVS Physical Page Table (32B) ====================
            { "NVS_PPAGE_STATE_AND_FLAGS",     EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, stateAndFlags)),         4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_OWNER_FLAT_INDEX",    EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, ownerVirtualFlatIndex)), 4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_OWNER_ADDRESS",       EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, ownerVirtualAddress)),   4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_GENERATION",          EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, physicalGeneration)),    4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_LAST_USED_FRAME",     EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, lastUsedFrame)),         4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_ALLOCATED_FRAME",     EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, allocatedFrame)),        4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_MIN_EVICTABLE_FRAME", EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, minEvictableFrame)),     4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },
            { "NVS_PPAGE_META_AUX",            EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSPhysicalPageEntry, reserved)),              4u, static_cast<UINT32>(sizeof(NVSPhysicalPageEntry)), false, false, false },

            // ==================== NVS Light View Table (256B) ====================
            { "NVS_VIEW_LIGHT_ID",              EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, lightId)),              4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_SUB_INDEX",             EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, clipmapLevel)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_PAGES_PER_AXIS",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, pagesPerAxis)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_FLAGS",                 EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, flags)),                4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_SNAPPED_ORIGIN_X",      EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, snappedOriginPageX)),   4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_SNAPPED_ORIGIN_Y",      EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, snappedOriginPageY)),   4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_WORLD_EXTENT",          EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, worldExtent)),          4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_PAGE_WORLD_SIZE",       EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, pageWorldSize)),        4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_TEXEL_WORLD_SIZE",      EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, texelWorldSize)),       4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_RECEIVER_MIN_Z",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, receiverMinZ)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_RECEIVER_MAX_Z",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, receiverMaxZ)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_CASTER_MIN_Z",          EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, casterMinZ)),           4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_CASTER_MAX_Z",          EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, casterMaxZ)),           4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_DEPTH_BIAS_NDC",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, depthBiasNdc)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_NORMAL_BIAS_WORLD",     EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, normalBiasWorld)),      4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_REF_LIGHT_Z",           EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, cameraLightZ)),         4u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_RIGHT_AND_CENTER_X",    EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, lightRightAndCenterX)), 16u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_UP_AND_CENTER_Y",       EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, lightUpAndCenterY)),    16u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_DIR_AND_EXTRUSION",     EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, lightDirAndExtrusion)), 16u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_TRANSITION_AND_PARAMS", EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, transitionAndPadding)), 16u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_WORLD_TO_LIGHT",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, worldToLightView)),     64u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },
            { "NVS_VIEW_LIGHT_TO_WORLD",        EngineResourceDomain::NVS_LIGHT_VIEW_TABLE, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(NVSClipmapLevelGPUData, lightViewToWorld)),     64u, static_cast<UINT32>(sizeof(NVSClipmapLevelGPUData)), false, false, false },

            // ==================== NVG Page Pool (4B — NVG Only) ====================
            { "NVG_PAGE_POOL_WORD", EngineResourceDomain::NVG_PAGE_POOL, EXEC_CTX_CLUSTER, 0u, 4u, 4u, true, false, false },

            // ==================== NVG Bucket Ranges (16B — NVG Only) ====================
            { "NVG_BUCKET_FIRST_CLUSTER", EngineResourceDomain::NVG_BUCKET_RANGES, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGBucketRangeGPUData, firstVisibleCluster)), 4u, static_cast<UINT32>(sizeof(NVGBucketRangeGPUData)), true, false, false },
            { "NVG_BUCKET_CLUSTER_COUNT", EngineResourceDomain::NVG_BUCKET_RANGES, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGBucketRangeGPUData, visibleClusterCount)), 4u, static_cast<UINT32>(sizeof(NVGBucketRangeGPUData)), true, false, false },
            { "NVG_BUCKET_MAX_TRIANGLES", EngineResourceDomain::NVG_BUCKET_RANGES, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGBucketRangeGPUData, maxTriangleCount)),    4u, static_cast<UINT32>(sizeof(NVGBucketRangeGPUData)), true, false, false },
            { "NVG_BUCKET_FLAGS",         EngineResourceDomain::NVG_BUCKET_RANGES, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGBucketRangeGPUData, bucketFlags)),         4u, static_cast<UINT32>(sizeof(NVGBucketRangeGPUData)), true, false, false },

            // ==================== NVG Visible Clusters (48B — NVG Only) ====================
            { "NVG_VIS_OBJECT_SLOT",         EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, objectSlotIndex)),   4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_CLUSTER_ID",          EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, clusterId)),         4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_PHYSICAL_PAGE_INDEX", EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, physicalPageIndex)), 4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_CLUSTER_WORD_OFFSET", EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, clusterWordOffset)), 4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_VIRTUAL_PAGE_ID",     EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, virtualPageId)),     4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_LOD_AND_FLAGS",       EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, lodAndFlags)),       4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_GEOMETRIC_ERROR",     EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, geometricError)),    4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_MATERIAL_ID",         EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, materialId)),        4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_BONE_OFFSET",         EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, bonePaletteOffset)), 4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_BONE_COUNT",          EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, boneCount)),         4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_DRAW_BUCKET_INDEX",   EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, drawBucketIndex)),   4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },
            { "NVG_VIS_VIEW_DEPTH_KEY",      EngineResourceDomain::NVG_VISIBLE_CLUSTERS, EXEC_CTX_CLUSTER, static_cast<UINT32>(offsetof(NVGVisibleClusterGPUData, viewDepthKey)),      4u, static_cast<UINT32>(sizeof(NVGVisibleClusterGPUData)), true, false, false },

            // ==================== GPUScene Object Records (160B — Scene-Wide, LIVE on STANDARD & VIRTUALIZED & Compute) ====================
            { "NVG_OBJ_SLOT_INDEX",         EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, slotIndex)),           4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_GENERATION",         EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, generation)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_FLAGS",              EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, objectFlags)),         4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_RENDER_FLAGS",       EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, renderObjectFlags)),   4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_MESH_ID",            EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, meshId)),              4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_MATERIAL_ID",        EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, materialId)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_LOD_GROUP_ID",       EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, lodGroupId)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_BATCH_GROUP_ID",     EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, batchGroupId)),        4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_VERTEX_COUNT",       EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, vertexCount)),         4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_INDEX_COUNT",        EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, indexCount)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_START_INDEX",        EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, startIndex)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_BASE_VERTEX",        EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, baseVertex)),          4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_AABB_MIN",           EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, aabbMin)),            12u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_SUBMESH_INDEX",      EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, submeshIndex)),        4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_AABB_MAX",           EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, aabbMax)),            12u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_LOD_MATERIAL_FLAGS", EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, lodAndMaterialFlags)), 4u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_BOUNDING_SPHERE",    EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, boundingSphere)),     16u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },
            { "NVG_OBJ_WORLD_MATRIX",       EngineResourceDomain::GPU_SCENE_OBJECTS, EXEC_CTX_SCENE | EXEC_CTX_FRAME, static_cast<UINT32>(offsetof(GPUSceneObjectData, worldMatrix)),        64u, static_cast<UINT32>(sizeof(GPUSceneObjectData)), false, false, false },

            // ==================== NVG Shadow Page Views (80B — NVG + Shadow Pass Only) ====================
            { "NVG_SHADOW_PAGE_VP_MATRIX",   EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS, EXEC_CTX_CLUSTER | EXEC_CTX_SHADOW_PAGE, static_cast<UINT32>(offsetof(NVGShadowPageViewGPUData, worldToPageClip)),   64u, static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData)), true, true, false },
            { "NVG_SHADOW_PAGE_SLICE",       EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS, EXEC_CTX_CLUSTER | EXEC_CTX_SHADOW_PAGE, static_cast<UINT32>(offsetof(NVGShadowPageViewGPUData, physicalPageSlice)),  4u, static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData)), true, true, false },
            { "NVG_SHADOW_PAGE_DEPTH_BIAS",  EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS, EXEC_CTX_CLUSTER | EXEC_CTX_SHADOW_PAGE, static_cast<UINT32>(offsetof(NVGShadowPageViewGPUData, depthBiasNdc)),       4u, static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData)), true, true, false },
            { "NVG_SHADOW_PAGE_NORMAL_BIAS", EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS, EXEC_CTX_CLUSTER | EXEC_CTX_SHADOW_PAGE, static_cast<UINT32>(offsetof(NVGShadowPageViewGPUData, normalBiasWorld)),    4u, static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData)), true, true, false },
            { "NVG_SHADOW_PAGE_TEXEL_SIZE",  EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS, EXEC_CTX_CLUSTER | EXEC_CTX_SHADOW_PAGE, static_cast<UINT32>(offsetof(NVGShadowPageViewGPUData, texelWorldSize)),     4u, static_cast<UINT32>(sizeof(NVGShadowPageViewGPUData)), true, true, false },
        };

        inline const std::unordered_map<std::string, const EnginePropertyMetadata*>& GetMetadataMap()
        {
            static const std::unordered_map<std::string, const EnginePropertyMetadata*> s_map = []() {
                std::unordered_map<std::string, const EnginePropertyMetadata*> m;
                m.reserve(sizeof(kPropertyMetadataTable) / sizeof(kPropertyMetadataTable[0]) * 2u);
                for (const auto& entry : kPropertyMetadataTable)
                {
                    m.emplace(entry.propertyName, &entry);
                }
                return m;
            }();
            return s_map;
        }
    } // anonymous namespace

    // =========================================================================
    //  CoreGPUResourceMaterializer (Decision 6C)
    // =========================================================================

    SIZE_T CoreGPUResourceMaterializer::CacheKeyHash::operator()(const CacheKey& k) const noexcept
    {
        UINT64 h = Fnv1aInit();
        h = Fnv1aMixU64(h, k.frameIndex);
        h = Fnv1aMixU64(h, static_cast<UINT64>(k.srcHandle));
        h = Fnv1aMixU64(h, k.srcRevision);
        h = Fnv1aMixU64(h, k.srcBaseElement);
        h = Fnv1aMixU64(h, k.engineElementCount);
        h = Fnv1aMixU64(h, k.userElementCount);
        h = Fnv1aMixU64(h, k.layoutHash);
        h = Fnv1aMixU64(h, k.userDataHash);
        return static_cast<SIZE_T>(h);
    }

    CoreGPUResourceMaterializer::~CoreGPUResourceMaterializer()
    {
        Shutdown();
    }

    BOOL CoreGPUResourceMaterializer::Initialize(
        CoreBufferManager*       bufferManager,
        InternalResourceManager* internalResources)
    {
        if (m_initialized) return true;
        if (!bufferManager || !internalResources) return false;

        m_bufferManager     = bufferManager;
        m_internalResources = internalResources;
        m_materializeKernel = nullptr;
        m_currentFrameIndex           = 0ull;
        m_activePoolSlot              = 0u;
        m_nextPoolIndexInSlot         = 0u;
        m_totalAllocatedVramBytes     = 0ull;
        m_materializationsThisFrame   = 0u;
        m_cacheHitsThisFrame          = 0u;
        m_overflowRejectionsThisFrame = 0u;
        m_frameCache.clear();
        m_initialized = true;
        return true;
    }

    void CoreGPUResourceMaterializer::Shutdown()
    {
        if (m_bufferManager)
        {
            for (UINT32 slot = 0u; slot < kFramePoolCount; ++slot)
            {
                for (auto& pooled : m_framePools[slot])
                {
                    if (pooled.buffer)
                    {
                        m_bufferManager->DestroyGPUBuffer(pooled.buffer);
                        pooled.buffer    = nullptr;
                        pooled.wordCount = 0u;
                    }
                }
                m_framePools[slot].clear();
            }
        }
        else
        {
            for (UINT32 slot = 0u; slot < kFramePoolCount; ++slot)
            {
                m_framePools[slot].clear();
            }
        }

        m_frameCache.clear();
        m_frameInsertedKeys.clear();
        m_materializeKernel           = nullptr;
        m_bufferManager               = nullptr;
        m_internalResources           = nullptr;
        m_totalAllocatedVramBytes     = 0ull;
        m_materializationsThisFrame   = 0u;
        m_cacheHitsThisFrame          = 0u;
        m_overflowRejectionsThisFrame = 0u;
        m_initialized                 = false;
    }

    void CoreGPUResourceMaterializer::BeginFrame(UINT64 frameIndex)
    {
        m_currentFrameIndex           = frameIndex;
        m_activePoolSlot              = static_cast<UINT32>(frameIndex % kFramePoolCount);
        m_nextPoolIndexInSlot         = 0u;
        m_materializationsThisFrame   = 0u;
        m_cacheHitsThisFrame          = 0u;
        m_overflowRejectionsThisFrame = 0u;
        m_frameCache.clear();
        m_frameInsertedKeys.clear();
    }

    void CoreGPUResourceMaterializer::RollbackTransaction(const TransactionSnapshot& snap)
    {
        while (m_frameInsertedKeys.size() > snap.insertedKeysSize)
        {
            m_frameCache.erase(m_frameInsertedKeys.back());
            m_frameInsertedKeys.pop_back();
        }
        m_nextPoolIndexInSlot       = snap.poolIndex;
        m_materializationsThisFrame = snap.matCount;
        m_cacheHitsThisFrame        = snap.hitCount;
    }

    GPUBuffer* CoreGPUResourceMaterializer::AllocatePooledWordBuffer(
        UINT32       requiredWords,
        std::string* outDiagnostic)
    {
        if (!m_initialized || !m_bufferManager || requiredWords == 0u)
        {
            if (outDiagnostic) *outDiagnostic = "Materializer is not initialized or requested 0 words.";
            return nullptr;
        }

        auto& pool = m_framePools[m_activePoolSlot];
        const UINT32 allocWords = (std::max)(requiredWords, 64u);
        const UINT64 allocBytes = static_cast<UINT64>(allocWords) * sizeof(UINT32);

        if (allocBytes > m_maxVramBudgetBytes)
        {
            ++m_overflowRejectionsThisFrame;
            if (outDiagnostic) *outDiagnostic = "Reflection GPU materializer VRAM budget exceeded.";
            return nullptr;
        }

        // 1. Check existing reusable pooled entry in the active (GPU-idle) slot FIRST
        //    whenever this request itself fits inside the configured VRAM budget.
        if (m_nextPoolIndexInSlot < pool.size())
        {
            PooledMaterializeBuffer& entry = pool[m_nextPoolIndexInSlot];
            if (entry.buffer && entry.wordCount >= requiredWords)
            {
                ++m_nextPoolIndexInSlot;
                return entry.buffer;
            }
        }

        // Helper: reclaim unused tail buffers in the active GPU-idle slot when approaching budget
        auto trimIdleTail = [&](UINT64 targetFreeBytes) {
            while (pool.size() > static_cast<SIZE_T>(m_nextPoolIndexInSlot) + 1u &&
                   m_totalAllocatedVramBytes + targetFreeBytes > m_maxVramBudgetBytes)
            {
                PooledMaterializeBuffer& tail = pool.back();
                if (tail.buffer)
                {
                    const UINT64 tailBytes = static_cast<UINT64>(tail.wordCount) * sizeof(UINT32);
                    m_bufferManager->DestroyGPUBuffer(tail.buffer);
                    m_totalAllocatedVramBytes = (m_totalAllocatedVramBytes >= tailBytes)
                        ? (m_totalAllocatedVramBytes - tailBytes)
                        : 0ull;
                }
                pool.pop_back();
            }
        };

        if (m_nextPoolIndexInSlot < pool.size())
        {
            PooledMaterializeBuffer& entry = pool[m_nextPoolIndexInSlot];
            const UINT64 oldBytes = entry.buffer ? static_cast<UINT64>(entry.wordCount) * sizeof(UINT32) : 0ull;
            UINT64 grownWords = (std::max)(static_cast<UINT64>(allocWords), static_cast<UINT64>(entry.wordCount) * 2ull);
            if (grownWords > 0x3FFFFFFFull)
            {
                grownWords = allocWords;
            }
            // Clamp speculative doubling to budget if exact allocWords fits
            if ((m_totalAllocatedVramBytes >= oldBytes ? (m_totalAllocatedVramBytes - oldBytes) : 0ull) + grownWords * sizeof(UINT32) > m_maxVramBudgetBytes)
            {
                grownWords = allocWords;
            }
            const UINT64 newBytes = grownWords * sizeof(UINT32);
            const UINT64 deltaBytes = (newBytes > oldBytes) ? (newBytes - oldBytes) : 0ull;
            if (m_totalAllocatedVramBytes + deltaBytes > m_maxVramBudgetBytes)
            {
                trimIdleTail(deltaBytes);
            }
            const UINT64 projectedVram = (m_totalAllocatedVramBytes >= oldBytes)
                ? (m_totalAllocatedVramBytes - oldBytes + newBytes)
                : newBytes;
            if (projectedVram > m_maxVramBudgetBytes)
            {
                ++m_overflowRejectionsThisFrame;
                if (outDiagnostic) *outDiagnostic = "Reflection GPU materializer VRAM budget exceeded.";
                return nullptr;
            }

            if (entry.buffer)
            {
                if (!m_bufferManager->ResizeGPUBuffer(entry.buffer, static_cast<UINT32>(grownWords), nullptr))
                {
                    ++m_overflowRejectionsThisFrame;
                    if (outDiagnostic) *outDiagnostic = "Failed to resize pooled GPU materialization buffer.";
                    return nullptr;
                }
            }
            else
            {
                GPUBufferDesc desc{};
                desc.elementCount = static_cast<UINT32>(grownWords);
                desc.stride       = sizeof(UINT32);
                desc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
                desc.initialData  = nullptr;
                entry.buffer = m_bufferManager->CreateGPUBuffer(desc);
                if (!entry.buffer)
                {
                    ++m_overflowRejectionsThisFrame;
                    if (outDiagnostic) *outDiagnostic = "Failed to allocate pooled GPU materialization buffer.";
                    return nullptr;
                }
            }

            m_totalAllocatedVramBytes = projectedVram;
            entry.wordCount           = static_cast<UINT32>(grownWords);
            ++m_nextPoolIndexInSlot;
            return entry.buffer;
        }

        if (m_totalAllocatedVramBytes + allocBytes > m_maxVramBudgetBytes)
        {
            trimIdleTail(allocBytes);
        }
        if (m_totalAllocatedVramBytes + allocBytes > m_maxVramBudgetBytes)
        {
            ++m_overflowRejectionsThisFrame;
            if (outDiagnostic) *outDiagnostic = "Reflection GPU materializer VRAM budget exceeded.";
            return nullptr;
        }

        GPUBufferDesc desc{};
        desc.elementCount = allocWords;
        desc.stride       = sizeof(UINT32);
        desc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        desc.initialData  = nullptr;
        GPUBuffer* newBuf = m_bufferManager->CreateGPUBuffer(desc);
        if (!newBuf)
        {
            ++m_overflowRejectionsThisFrame;
            if (outDiagnostic) *outDiagnostic = "Failed to create pooled GPU materialization buffer.";
            return nullptr;
        }

        PooledMaterializeBuffer entry{};
        entry.buffer    = newBuf;
        entry.wordCount = allocWords;
        pool.push_back(entry);
        m_totalAllocatedVramBytes += allocBytes;
        ++m_nextPoolIndexInSlot;
        return newBuf;
    }

    BOOL CoreGPUResourceMaterializer::AcquireMaterializedBuffer(
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
        std::string*                         outDiagnostic)
    {
        outDstBufferHandle = 0;
        outDstElementCount = 0u;
        outDstByteSize     = 0u;

        if (!m_initialized || !m_internalResources || !plan.valid)
        {
            if (outDiagnostic) *outDiagnostic = "GPU resource materializer not initialized or invalid layout plan.";
            return false;
        }

        const UINT32 effEngineCount = (srcHandle != 0 && srcByteSize > 0u) ? engineElementCount : 0u;
        const UINT32 totalElements  = effEngineCount + userElementCount;
        if (totalElements == 0u)
        {
            // CST 0-element contract: zero dispatch, zero allocation -> Null Buffer SRV!
            return true;
        }

        if (plan.dstStride == 0u || (plan.dstStride % 4u) != 0u ||
            plan.canonicalSrcStride == 0u || (plan.canonicalSrcStride % 4u) != 0u)
        {
            if (outDiagnostic) *outDiagnostic = "Materializer requires 4-byte aligned source and destination strides.";
            return false;
        }

        const UINT64 userDataHash = (userElementCount > 0u && userTailData != nullptr && userTailByteSize > 0u)
            ? Fnv1aHashBytes(userTailData, userTailByteSize)
            : 0ull;

        CacheKey key{};
        key.frameIndex         = m_currentFrameIndex;
        key.srcHandle          = (effEngineCount > 0u) ? srcHandle : 0;
        key.srcRevision        = (effEngineCount > 0u) ? srcRevision : 0ull;
        key.srcBaseElement     = (effEngineCount > 0u) ? srcBaseElement : 0u;
        key.engineElementCount = effEngineCount;
        key.userElementCount   = userElementCount;
        key.layoutHash         = plan.layoutHash;
        key.userDataHash       = userDataHash;

        auto itCache = m_frameCache.find(key);
        if (itCache != m_frameCache.end())
        {
            ++m_cacheHitsThisFrame;
            outDstBufferHandle = itCache->second.bufferHandle;
            outDstElementCount = itCache->second.elementCount;
            outDstByteSize     = itCache->second.byteSize;
            return true;
        }

        if (!m_materializeKernel)
        {
            m_materializeKernel = m_internalResources->AcquireFeatureKernel("Runtime.ReflectionMaterialize");
            if (!m_materializeKernel || !m_materializeKernel->IsValid())
            {
                if (outDiagnostic) *outDiagnostic = "Failed to acquire Runtime.ReflectionMaterialize internal compute kernel.";
                return false;
            }
        }

        const Shader* cs = m_materializeKernel->GetComputeShader();
        if (!cs || !cs->IsValid() || cs->GetNativeHandle() == 0)
        {
            if (outDiagnostic) *outDiagnostic = "Runtime.ReflectionMaterialize compute shader is invalid.";
            return false;
        }

        const UINT32 srcStrideWords = plan.canonicalSrcStride / 4u;
        const UINT32 dstStrideWords = plan.dstStride / 4u;
        const UINT64 requiredWords64 = static_cast<UINT64>(totalElements) * static_cast<UINT64>(dstStrideWords);
        if (requiredWords64 == 0ull || requiredWords64 > 0x3FFFFFFFull)
        {
            ++m_overflowRejectionsThisFrame;
            if (outDiagnostic) *outDiagnostic = "Materialized buffer byte size overflow.";
            return false;
        }

        GPUBuffer* dstBuf = AllocatePooledWordBuffer(static_cast<UINT32>(requiredWords64), outDiagnostic);
        if (!dstBuf || dstBuf->GetNativeHandle() == 0)
        {
            return false;
        }

        const ShaderReflection& refl = cs->GetReflection();
        const UINT64 totalGroups = (static_cast<UINT64>(totalElements) + 63ull) / 64ull;
        const UINT32 groupsX     = (totalGroups <= 65535ull) ? static_cast<UINT32>(totalGroups) : 65535u;
        const UINT32 groupsY     = static_cast<UINT32>((totalGroups + static_cast<UINT64>(groupsX) - 1ull) / static_cast<UINT64>(groupsX));
        if (groupsY > 65535u)
        {
            ++m_overflowRejectionsThisFrame;
            if (outDiagnostic) *outDiagnostic = "Materialized element count exceeds 2D compute dispatch limit.";
            return false;
        }

        ComputeDispatchCommand cmd{};
        cmd.computeShader     = cs->GetNativeHandle();
        cmd.threadGroupCountX = groupsX;
        cmd.threadGroupCountY = groupsY;
        cmd.threadGroupCountZ = 1u;
        cmd.profileGroup      = ComputeDispatchProfileGroup::GENERIC;

        const UINT32 srcTotalWords  = (effEngineCount > 0u) ? (srcByteSize / 4u) : 0u;
        const UINT32 userTotalWords = (userElementCount > 0u && userTailData != nullptr)
            ? (userTailByteSize / 4u)
            : 0u;

        // 1. Populate Config CBuffer from reflection
        for (const auto& nscBuf : refl.buffers)
        {
            if (nscBuf.binding == 0xFFFF || nscBuf.totalSize == 0u) continue;
            NativeConstantBufferBinding cb{};
            cb.slotIndex     = nscBuf.binding;
            cb.registerSpace = nscBuf.set;
            cb.stage         = ShaderStage::COMPUTE;
            cb.byteSize      = nscBuf.totalSize;
            cb.ownedData.assign(nscBuf.totalSize, 0u);

            auto writeU32 = [&](const char* fieldName, UINT32 value) {
                for (const auto& f : nscBuf.fields)
                {
                    if (f.name == fieldName && f.offset + sizeof(UINT32) <= cb.ownedData.size())
                    {
                        std::memcpy(cb.ownedData.data() + f.offset, &value, sizeof(UINT32));
                        break;
                    }
                }
            };
            writeU32("srcStrideWords",     srcStrideWords);
            writeU32("dstStrideWords",     dstStrideWords);
            writeU32("engineElementCount", effEngineCount);
            writeU32("userElementCount",   userElementCount);
            writeU32("copySpanCount",      static_cast<UINT32>(plan.copySpans.size()));
            writeU32("srcBaseElement",     (effEngineCount > 0u) ? srcBaseElement : 0u);
            writeU32("srcTotalWords",      srcTotalWords);
            writeU32("userTotalWords",     userTotalWords);
            writeU32("dispatchGroupsX",    groupsX);

            cb.cpuData = cb.ownedData.data();
            cmd.constantBuffers.push_back(std::move(cb));
        }

        // 2. Bind SRVs (InSrcWords, InCopySpans, InUserWords) and UAV (OutDstWords)
        for (const auto& nscSb : refl.structuredBuffers)
        {
            if (nscSb.binding == 0xFFFF || nscSb.stride == 0u) continue;
            const BOOL isUav = (refl.FindUAV(nscSb.name) != nullptr);

            NativeStructuredBufferBinding sbb{};
            sbb.slotIndex     = nscSb.binding;
            sbb.registerSpace = nscSb.set;
            sbb.stage         = ShaderStage::COMPUTE;
            sbb.stride        = nscSb.stride;
            sbb.isUAV         = isUav;

            if (isUav)
            {
                sbb.buffer       = dstBuf->GetNativeHandle();
                sbb.elementCount = dstBuf->GetElementCount();
                sbb.stride       = sizeof(UINT32);
                sbb.byteSize     = dstBuf->GetByteSize();
                sbb.uavAccess    = GPUResourceAccess::UNORDERED_ACCESS_WRITE;
            }
            else if (nscSb.name.find("InSrcWords") != std::string::npos)
            {
                if (effEngineCount > 0u && srcHandle != 0 && srcTotalWords > 0u)
                {
                    sbb.buffer       = srcHandle;
                    sbb.elementCount = srcTotalWords;
                    sbb.stride       = sizeof(UINT32);
                    sbb.byteSize     = srcTotalWords * static_cast<UINT32>(sizeof(UINT32));
                }
                else
                {
                    sbb.buffer       = 0;
                    sbb.elementCount = 0u;
                    sbb.stride       = sizeof(UINT32);
                    sbb.byteSize     = 0u;
                }
            }
            else if (nscSb.name.find("InCopySpans") != std::string::npos)
            {
                if (!plan.copySpans.empty())
                {
                    sbb.elementCount = static_cast<UINT32>(plan.copySpans.size());
                    sbb.stride       = static_cast<UINT32>(sizeof(MaterializeCopySpanGPU));
                    sbb.byteSize     = sbb.elementCount * sbb.stride;
                    sbb.ownedData.resize(sbb.byteSize);
                    std::memcpy(sbb.ownedData.data(), plan.copySpans.data(), sbb.byteSize);
                    sbb.cpuData      = sbb.ownedData.data();
                }
                else
                {
                    sbb.elementCount = 0u;
                    sbb.stride       = static_cast<UINT32>(sizeof(MaterializeCopySpanGPU));
                    sbb.byteSize     = 0u;
                }
            }
            else if (nscSb.name.find("InUserWords") != std::string::npos)
            {
                if (userElementCount > 0u && userTailData != nullptr && userTotalWords > 0u)
                {
                    sbb.elementCount = userTotalWords;
                    sbb.stride       = sizeof(UINT32);
                    sbb.byteSize     = userTotalWords * static_cast<UINT32>(sizeof(UINT32));
                    sbb.ownedData.resize(sbb.byteSize);
                    std::memcpy(sbb.ownedData.data(), userTailData, sbb.byteSize);
                    sbb.cpuData      = sbb.ownedData.data();
                }
                else
                {
                    sbb.elementCount = 0u;
                    sbb.stride       = sizeof(UINT32);
                    sbb.byteSize     = 0u;
                }
            }

            cmd.structuredBuffers.push_back(std::move(sbb));
        }

        outDispatches.push_back(std::move(cmd));

        CachedMaterializeResult res{};
        res.bufferHandle = dstBuf->GetNativeHandle();
        res.elementCount = totalElements;
        res.byteSize     = totalElements * plan.dstStride;
        m_frameCache.emplace(key, res);
        m_frameInsertedKeys.push_back(key);

        ++m_materializationsThisFrame;
        outDstBufferHandle = res.bufferHandle;
        outDstElementCount = res.elementCount;
        outDstByteSize     = res.byteSize;
        return true;
    }

    // =========================================================================
    //  CoreResourceBindingResolver
    // =========================================================================

    const EnginePropertyMetadata* CoreResourceBindingResolver::FindPropertyMetadata(const std::string& propertyName)
    {
        const auto& m = GetMetadataMap();
        const auto it = m.find(propertyName);
        return (it != m.end()) ? it->second : nullptr;
    }

    BOOL CoreResourceBindingResolver::IsTier1InstanceStructuredBuffer(
        const NSL::NSCStructuredBuffer& nscSb,
        ShaderStage                     stage)
    {
        if (stage != ShaderStage::VERTEX) return false;

        BOOL hasInstGetField     = false;
        BOOL hasOtherGetField    = false;
        BOOL hasTier1WorldMatrix = false;
        BOOL hasNonTier1AddField = false;

        for (const auto& f : nscSb.fields)
        {
            if (f.isGet)
            {
                const std::string& pName = GetFieldPropertyName(f);
                if (pName == "INSTANCE_WORLD_MATRIX" ||
                    pName == "INSTANCE_ID_VALUE" ||
                    pName == "INSTANCE_CUSTOM_DATA")
                {
                    hasInstGetField = true;
                }
                else if (pName != "INSTANCE_BONE_OFFSET" && pName != "INSTANCE_BONE_COUNT")
                {
                    hasOtherGetField = true;
                }
            }
            else
            {
                if (f.name == "worldMatrix" && f.size == sizeof(Mat4))
                {
                    hasTier1WorldMatrix = true;
                }
                else if (f.name != "instanceId" && f.name != "customData")
                {
                    hasNonTier1AddField = true;
                }
            }
        }

        if (hasOtherGetField) return false;
        if (hasInstGetField) return true;
        return (hasTier1WorldMatrix && !hasNonTier1AddField);
    }

    EngineResourceDomain CoreResourceBindingResolver::DetectStructuredBufferDomain(
        const NSL::NSCStructuredBuffer& nscSb,
        ShaderStage                     stage)
    {
        EngineResourceDomain detected = EngineResourceDomain::USER_ONLY;
        BOOL hasInstSkinning          = false;
        BOOL hasTier1WorldMatrix      = false;
        BOOL hasNonTier1Add           = false;

        for (const auto& f : nscSb.fields)
        {
            if (!f.isGet)
            {
                if (f.name == "worldMatrix" && f.size == sizeof(Mat4))
                {
                    hasTier1WorldMatrix = true;
                }
                else if (f.name != "instanceId" && f.name != "customData")
                {
                    hasNonTier1Add = true;
                }
                continue;
            }

            const std::string& pName = GetFieldPropertyName(f);
            const EnginePropertyMetadata* meta = FindPropertyMetadata(pName);
            EngineResourceDomain fieldDomain = EngineResourceDomain::USER_ONLY;
            if (meta)
            {
                fieldDomain = meta->domain;
            }
            else if (pName.rfind("GLOBAL_DECAL_", 0) == 0)
            {
                fieldDomain = EngineResourceDomain::DECALS;
            }
            else if (pName.rfind("GLOBAL_PROBE_", 0) == 0)
            {
                fieldDomain = EngineResourceDomain::REFLECTION_PROBES;
            }
            else if (pName.rfind("GLOBAL_PARTICLE_", 0) == 0)
            {
                fieldDomain = EngineResourceDomain::PARTICLES;
            }
            else
            {
                return EngineResourceDomain::INVALID_MIXED_DOMAINS;
            }

            if (fieldDomain == EngineResourceDomain::INSTANCE_SKINNING)
            {
                hasInstSkinning = true;
                if (detected == EngineResourceDomain::USER_ONLY)
                {
                    detected = EngineResourceDomain::INSTANCE_SKINNING;
                }
                else if (detected != EngineResourceDomain::INSTANCES &&
                         detected != EngineResourceDomain::INSTANCE_SKINNING)
                {
                    return EngineResourceDomain::INVALID_MIXED_DOMAINS;
                }
                continue;
            }

            if (fieldDomain == EngineResourceDomain::INSTANCES)
            {
                if (detected == EngineResourceDomain::USER_ONLY ||
                    detected == EngineResourceDomain::INSTANCE_SKINNING ||
                    detected == EngineResourceDomain::INSTANCES)
                {
                    detected = EngineResourceDomain::INSTANCES;
                }
                else
                {
                    return EngineResourceDomain::INVALID_MIXED_DOMAINS;
                }
                continue;
            }

            if (fieldDomain == EngineResourceDomain::USER_ONLY)
            {
                return EngineResourceDomain::INVALID_MIXED_DOMAINS;
            }

            if (detected == EngineResourceDomain::USER_ONLY)
            {
                detected = fieldDomain;
            }
            else if (detected != fieldDomain)
            {
                return EngineResourceDomain::INVALID_MIXED_DOMAINS;
            }
        }

        if (detected == EngineResourceDomain::INSTANCE_SKINNING)
        {
            if (stage == ShaderStage::VERTEX && hasTier1WorldMatrix && !hasNonTier1Add)
            {
                return EngineResourceDomain::INSTANCES;
            }
            return EngineResourceDomain::INSTANCE_SKINNING;
        }

        if (detected == EngineResourceDomain::USER_ONLY && stage == ShaderStage::VERTEX)
        {
            if (hasTier1WorldMatrix && !hasNonTier1Add)
            {
                return EngineResourceDomain::INSTANCES;
            }
        }

        (void)hasInstSkinning;
        return detected;
    }

    StructuredBufferLayoutPlan CoreResourceBindingResolver::AnalyzeStructuredBuffer(
        const NSL::NSCStructuredBuffer& nscSb,
        ShaderStage                     stage,
        UINT32                          userElementCount)
    {
        StructuredBufferLayoutPlan plan{};
        plan.domain    = DetectStructuredBufferDomain(nscSb, stage);
        plan.dstStride = nscSb.stride;

        for (const auto& f : nscSb.fields)
        {
            if (f.isGet)
            {
                plan.hasGetFields = true;
                const std::string& pName = GetFieldPropertyName(f);
                if (pName == "INSTANCE_BONE_OFFSET" || pName == "INSTANCE_BONE_COUNT")
                {
                    plan.hasInstanceSkinFields = true;
                }
            }
            else
            {
                plan.hasAddFields = true;
            }
        }

        if (plan.domain == EngineResourceDomain::INVALID_MIXED_DOMAINS)
        {
            plan.valid      = false;
            plan.diagnostic = "StructuredBuffer mixes incompatible physical engine domains or references an unsupported Get() property.";
            return plan;
        }

        if (!IsDomainGPUBacked(plan.domain))
        {
            plan.valid               = true;
            plan.canDirectZeroCopy   = false;
            plan.requiresMaterialize = false;
            return plan;
        }

        plan.canonicalSrcStride = GetCanonicalDomainStride(plan.domain);
        if (plan.canonicalSrcStride == 0u || nscSb.stride == 0u)
        {
            plan.valid      = false;
            plan.diagnostic = "GPU-backed StructuredBuffer has zero stride.";
            return plan;
        }

        // Check Direct Zero-Copy eligibility (exact canonical layout, no Add() fields, 0 user elements)
        BOOL exactCanonical = (userElementCount == 0u) &&
                              (!plan.hasAddFields) &&
                              (nscSb.stride == plan.canonicalSrcStride);

        UINT64 hash = Fnv1aInit();
        hash = Fnv1aMixU64(hash, static_cast<UINT64>(plan.domain));
        hash = Fnv1aMixU64(hash, plan.canonicalSrcStride);
        hash = Fnv1aMixU64(hash, nscSb.stride);

        plan.copySpans.reserve(nscSb.fields.size());

        for (const auto& f : nscSb.fields)
        {
            if (!f.isGet) continue;
            const std::string& pName = GetFieldPropertyName(f);
            const EnginePropertyMetadata* meta = FindPropertyMetadata(pName);
            if (!meta || meta->canonicalSrcByteOffset == 0xFFFFFFFFu || meta->canonicalSrcByteSize == 0u)
            {
                plan.valid      = false;
                plan.diagnostic = "Field '" + pName + "' lacks a valid physical GPU layout mapping.";
                return plan;
            }

            if (f.offset != meta->canonicalSrcByteOffset || f.size != meta->canonicalSrcByteSize)
            {
                exactCanonical = false;
            }

            if ((f.offset % 4u) != 0u || (f.size % 4u) != 0u ||
                (meta->canonicalSrcByteOffset % 4u) != 0u || (meta->canonicalSrcByteSize % 4u) != 0u ||
                f.size == 0u || f.size > meta->canonicalSrcByteSize ||
                f.offset + f.size > nscSb.stride)
            {
                plan.valid      = false;
                plan.diagnostic = "Field '" + pName + "' has unaligned or out-of-bounds GPU copy span.";
                return plan;
            }

            MaterializeCopySpanGPU span{};
            span.srcWordOffset = meta->canonicalSrcByteOffset / 4u;
            span.dstWordOffset = f.offset / 4u;
            span.wordCount     = f.size / 4u;
            span.flags         = 0u;
            plan.copySpans.push_back(span);

            hash = Fnv1aMixU64(hash, span.srcWordOffset);
            hash = Fnv1aMixU64(hash, span.dstWordOffset);
            hash = Fnv1aMixU64(hash, span.wordCount);
            hash = Fnv1aMixU64(hash, span.flags);
        }

        if ((nscSb.stride % 4u) != 0u || (plan.canonicalSrcStride % 4u) != 0u)
        {
            plan.valid      = false;
            plan.diagnostic = "GPU materialization requires 4-byte aligned StructuredBuffer stride.";
            return plan;
        }

        plan.layoutHash          = hash;
        plan.valid               = true;
        plan.canDirectZeroCopy   = exactCanonical;
        plan.requiresMaterialize = !exactCanonical;
        return plan;
    }

    BOOL CoreResourceBindingResolver::ValidateStructuredBufferLayout(
        const NSL::NSCStructuredBuffer& nscSb,
        ShaderStage                     stage,
        std::string*                    outDiagnostic)
    {
        const StructuredBufferLayoutPlan plan = AnalyzeStructuredBuffer(nscSb, stage, 0u);
        if (!plan.valid && outDiagnostic)
        {
            *outDiagnostic = plan.diagnostic;
        }
        return plan.valid;
    }

    BOOL CoreResourceBindingResolver::IsCBufferFieldPerDrawDynamic(const std::string& propertyName)
    {
        const UINT32 sem = MapPropertyNameToPatchSemantic(propertyName);
        return (sem == NVS_PATCH_WORLD_MATRIX ||
                sem == NVS_PATCH_INV_WORLD_MATRIX ||
                sem == NVS_PATCH_NORMAL_MATRIX ||
                sem == NVS_PATCH_MVP_MATRIX ||
                sem == NVS_PATCH_BONE_PALETTE_OFFSET ||
                sem == NVS_PATCH_BONE_COUNT ||
                sem == NVS_PATCH_SKINNING_REVISION ||
                sem == NVS_PATCH_NVG_DRAW_BUCKET_ID ||
                sem == NVS_PATCH_NVG_VISIBLE_CLUSTER_COUNT ||
                sem == NVS_PATCH_NVG_VISIBLE_CLUSTER_OFFSET ||
                sem == NVS_PATCH_GPU_SCENE_OBJECT_SLOT);
    }

    BOOL CoreResourceBindingResolver::IsCBufferFieldShadowPageDynamic(const std::string& propertyName)
    {
        const UINT32 sem = MapPropertyNameToPatchSemantic(propertyName);
        return (sem != NVS_PATCH_SEMANTIC_NONE);
    }

    UINT32 CoreResourceBindingResolver::MapPropertyNameToPatchSemantic(const std::string& name)
    {
        if (name == "WORLD_MATRIX")                 return NVS_PATCH_WORLD_MATRIX;
        if (name == "INV_WORLD_MATRIX")             return NVS_PATCH_INV_WORLD_MATRIX;
        if (name == "NORMAL_MATRIX")                return NVS_PATCH_NORMAL_MATRIX;
        if (name == "VIEW_MATRIX")                  return NVS_PATCH_VIEW_MATRIX;
        if (name == "INV_VIEW_MATRIX")              return NVS_PATCH_INV_VIEW_MATRIX;
        if (name == "PROJECTION_MATRIX")            return NVS_PATCH_PROJECTION_MATRIX;
        if (name == "INV_PROJECTION_MATRIX")        return NVS_PATCH_INV_PROJECTION_MATRIX;
        if (name == "VP_MATRIX")                    return NVS_PATCH_VP_MATRIX;
        if (name == "INV_VP_MATRIX")                return NVS_PATCH_INV_VP_MATRIX;
        if (name == "MVP_MATRIX")                   return NVS_PATCH_MVP_MATRIX;
        if (name == "CAMERA_POSITION")              return NVS_PATCH_CAMERA_POSITION;
        if (name == "CAMERA_DIRECTION")             return NVS_PATCH_CAMERA_DIRECTION;
        if (name == "NEAR_PLANE" || name == "CAMERA_NEAR_PLANE") return NVS_PATCH_NEAR_PLANE;
        if (name == "FAR_PLANE"  || name == "CAMERA_FAR_PLANE")  return NVS_PATCH_FAR_PLANE;
        if (name == "BONE_PALETTE_OFFSET")          return NVS_PATCH_BONE_PALETTE_OFFSET;
        if (name == "BONE_COUNT")                   return NVS_PATCH_BONE_COUNT;
        if (name == "SKINNING_REVISION")            return NVS_PATCH_SKINNING_REVISION;
        if (name == "NVS_PHYSICAL_PAGE_INDEX")      return NVS_PATCH_NVS_PHYSICAL_PAGE_INDEX;
        if (name == "NVS_VIRTUAL_PAGE_ADDRESS")     return NVS_PATCH_NVS_VIRTUAL_PAGE_ADDRESS;
        if (name == "NVS_CLIPMAP_LEVEL")            return NVS_PATCH_NVS_CLIPMAP_LEVEL;
        if (name == "NVS_DEPTH_BIAS_NDC")           return NVS_PATCH_NVS_DEPTH_BIAS_NDC;
        if (name == "NVS_NORMAL_BIAS_WORLD")        return NVS_PATCH_NVS_NORMAL_BIAS_WORLD;
        if (name == "NVS_RECEIVER_MIN_Z")           return NVS_PATCH_NVS_RECEIVER_MIN_Z;
        if (name == "NVS_RECEIVER_MAX_Z")           return NVS_PATCH_NVS_RECEIVER_MAX_Z;
        if (name == "CAMERA_FOV")                   return NVS_PATCH_CAMERA_FOV;
        if (name == "CAMERA_ASPECT")                return NVS_PATCH_CAMERA_ASPECT;
        if (name == "NVG_DRAW_BUCKET_ID")           return NVS_PATCH_NVG_DRAW_BUCKET_ID;
        if (name == "NVG_VISIBLE_CLUSTER_COUNT")    return NVS_PATCH_NVG_VISIBLE_CLUSTER_COUNT;
        if (name == "NVG_VISIBLE_CLUSTER_OFFSET")   return NVS_PATCH_NVG_VISIBLE_CLUSTER_OFFSET;
        if (name == "GPU_SCENE_OBJECT_SLOT")        return NVS_PATCH_GPU_SCENE_OBJECT_SLOT;
        if (name == "INSTANCE_WORLD_MATRIX")        return NVS_PATCH_INSTANCE_WORLD_MATRIX;
        if (name == "INSTANCE_ID_VALUE" || name == "INSTANCE_ID") return NVS_PATCH_INSTANCE_ID;
        if (name == "INSTANCE_CUSTOM_DATA")         return NVS_PATCH_INSTANCE_CUSTOM_DATA;
        if (name == "INSTANCE_BONE_OFFSET" || name == "INSTANCE_BONE_PALETTE_OFFSET") return NVS_PATCH_INSTANCE_BONE_OFFSET;
        if (name == "INSTANCE_BONE_COUNT")          return NVS_PATCH_INSTANCE_BONE_COUNT;
        return NVS_PATCH_SEMANTIC_NONE;
    }

    BOOL CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(const ShaderReflection& refl)
    {
        for (const auto& buf : refl.buffers)
        {
            for (const auto& f : buf.fields)
            {
                if (!f.isGet) continue;
                const std::string& pName = GetCBufferFieldPropertyName(f);
                if (pName == "BONE_PALETTE_OFFSET" ||
                    pName == "BONE_COUNT" ||
                    pName == "SKINNING_REVISION")
                {
                    return true;
                }
            }
        }
        for (const auto& sb : refl.structuredBuffers)
        {
            if (sb.name == "NVGResources__BonePalette") continue;
            for (const auto& f : sb.fields)
            {
                if (!f.isGet) continue;
                const std::string& pName = GetFieldPropertyName(f);
                if (pName == "GLOBAL_BONE_MATRIX" ||
                    pName == "BONE_MATRIX" ||
                    pName == "INSTANCE_BONE_OFFSET" ||
                    pName == "INSTANCE_BONE_COUNT")
                {
                    return true;
                }
            }
        }
        return false;
    }

    BOOL CoreResourceBindingResolver::DoesShaderUseGlobalBoneOffset(const ShaderReflection& refl)
    {
        for (const auto& buf : refl.buffers)
        {
            for (const auto& f : buf.fields)
            {
                if (f.isGet && GetCBufferFieldPropertyName(f) == "BONE_PALETTE_OFFSET")
                    return true;
            }
        }
        for (const auto& sb : refl.structuredBuffers)
        {
            if (sb.name == "NVGResources__BonePalette") continue;
            for (const auto& f : sb.fields)
            {
                if (!f.isGet) continue;
                const std::string& pName = GetFieldPropertyName(f);
                if (pName == "GLOBAL_BONE_MATRIX" || pName == "INSTANCE_BONE_OFFSET")
                    return true;
            }
        }
        return false;
    }

    void CoreResourceBindingResolver::FillEngineConstantBufferField(
        const std::string&         propName,
        const StageResolveContext& ctx,
        BYTE*                      dest,
        UINT32                     offset,
        UINT32                     size,
        UINT32                     bufTotalSize)
    {
        if (!dest || offset >= bufTotalSize) return;
        const UINT32 maxWrite = bufTotalSize - offset;

        auto copyBytes = [&](const void* src, SIZE_T numBytes) {
            const SIZE_T toCopy = (std::min)({
                static_cast<SIZE_T>(numBytes),
                static_cast<SIZE_T>(maxWrite),
                static_cast<SIZE_T>(size > 0u ? size : numBytes)
            });
            std::memcpy(dest + offset, src, toCopy);
        };

        const FrameView* frame = ctx.frameView;
        const Mat4& worldMat   = ctx.worldMatrix;

        if (propName == "WORLD_MATRIX")
        {
            copyBytes(&worldMat, sizeof(Mat4));
        }
        else if (propName == "NORMAL_MATRIX")
        {
            const Vec3 c0(worldMat.m[0], worldMat.m[1], worldMat.m[2]);
            const Vec3 c1(worldMat.m[4], worldMat.m[5], worldMat.m[6]);
            const Vec3 c2(worldMat.m[8], worldMat.m[9], worldMat.m[10]);
            const Vec3 cof0 = c1.Cross(c2);
            const Vec3 cof1 = c2.Cross(c0);
            const Vec3 cof2 = c0.Cross(c1);
            const FLOAT32 det = c0.Dot(cof0);
            FLOAT32 normalMat[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            };
            if (std::isfinite(det) && std::fabs(det) > 1e-12f)
            {
                const FLOAT32 invDet = 1.0f / det;
                normalMat[0]  = cof0.x * invDet; normalMat[1]  = cof0.y * invDet; normalMat[2]  = cof0.z * invDet;
                normalMat[4]  = cof1.x * invDet; normalMat[5]  = cof1.y * invDet; normalMat[6]  = cof1.z * invDet;
                normalMat[8]  = cof2.x * invDet; normalMat[9]  = cof2.y * invDet; normalMat[10] = cof2.z * invDet;
            }
            copyBytes(normalMat, sizeof(normalMat));
        }
        else if (propName == "INV_WORLD_MATRIX")
        {
            const Mat4 invW = worldMat.Inverted();
            copyBytes(&invW, sizeof(Mat4));
        }
        else if (propName == "MVP_MATRIX")
        {
            const Mat4 mvp = frame ? (frame->camera.viewProjectionMatrix * worldMat) : worldMat;
            copyBytes(&mvp, sizeof(Mat4));
        }
        else if (propName == "GPU_SCENE_OBJECT_SLOT")
        {
            const BOOL validObjCtx = ((ctx.executionContextMask & (EXEC_CTX_RENDER_OBJECT | EXEC_CTX_CLUSTER)) != 0u);
            const UINT32 slotVal   = validObjCtx ? ctx.gpuSceneObjectSlot : kInvalidGPUSceneSlot;
            copyBytes(&slotVal, sizeof(UINT32));
        }
        else if (propName == "GEOMETRY_MODE")
        {
            const BOOL validDrawCtx = ((ctx.executionContextMask & EXEC_CTX_DRAW) != 0u);
            UINT32 modeVal = kNVGInvalidIndex;
            if (validDrawCtx)
            {
                if (ctx.geometryMode != kNVGInvalidIndex)
                {
                    modeVal = ctx.geometryMode;
                }
                else
                {
                    modeVal = ((ctx.executionContextMask & EXEC_CTX_CLUSTER) != 0u)
                        ? static_cast<UINT32>(GeometryMode::VIRTUALIZED)
                        : static_cast<UINT32>(GeometryMode::STANDARD);
                }
            }
            copyBytes(&modeVal, sizeof(UINT32));
        }
        else if (propName == "VP_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.viewProjectionMatrix : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "INV_VP_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.viewProjectionMatrix.Inverted() : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "VIEW_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.viewMatrix : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "PROJECTION_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.projectionMatrix : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "INV_VIEW_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.viewMatrix.Inverted() : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "INV_PROJECTION_MATRIX")
        {
            const Mat4 m = frame ? frame->camera.projectionMatrix.Inverted() : Mat4::Identity();
            copyBytes(&m, sizeof(Mat4));
        }
        else if (propName == "CAMERA_POSITION")
        {
            const Vec3 v = frame ? frame->camera.position : Vec3::Zero();
            copyBytes(&v, sizeof(Vec3));
        }
        else if (propName == "CAMERA_DIRECTION")
        {
            const Vec3 v = frame ? frame->camera.direction : Vec3(0.0f, 0.0f, 1.0f);
            copyBytes(&v, sizeof(Vec3));
        }
        else if (propName == "NEAR_PLANE" || propName == "CAMERA_NEAR_PLANE")
        {
            const FLOAT32 v = frame ? frame->camera.nearPlane : 0.1f;
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "FAR_PLANE" || propName == "CAMERA_FAR_PLANE")
        {
            const FLOAT32 v = frame ? frame->camera.farPlane : 1000.0f;
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "CAMERA_FOV")
        {
            const FLOAT32 v = frame ? frame->camera.fov : 60.0f;
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "CAMERA_ASPECT")
        {
            const FLOAT32 v = frame ? frame->camera.aspectRatio : (16.0f / 9.0f);
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "AMBIENT_COLOR")
        {
            const Vec3 ac = frame
                ? Vec3(frame->environment.ambientColor.r, frame->environment.ambientColor.g, frame->environment.ambientColor.b)
                : Vec3(0.1f, 0.1f, 0.1f);
            copyBytes(&ac, sizeof(Vec3));
        }
        else if (propName == "AMBIENT_INTENSITY")
        {
            const FLOAT32 v = frame ? frame->environment.ambientIntensity : 1.0f;
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "DIR_LIGHT_DIRECTION")
        {
            const Vec3 v = frame ? frame->directionalLight.direction : Vec3(0.0f, -1.0f, 0.0f);
            copyBytes(&v, sizeof(Vec3));
        }
        else if (propName == "DIR_LIGHT_COLOR")
        {
            const Vec3 v = frame
                ? Vec3(frame->directionalLight.color.r, frame->directionalLight.color.g, frame->directionalLight.color.b)
                : Vec3(1.0f, 1.0f, 1.0f);
            copyBytes(&v, sizeof(Vec3));
        }
        else if (propName == "DIR_LIGHT_INTENSITY")
        {
            const FLOAT32 v = frame ? frame->directionalLight.intensity : 0.0f;
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "LIGHT_COUNT")
        {
            const UINT32 lc = frame ? static_cast<UINT32>(frame->lights.size()) : 0u;
            copyBytes(&lc, sizeof(UINT32));
        }
        else if (propName == "SCREEN_SIZE")
        {
            FLOAT32 w = 1920.0f;
            FLOAT32 h = 1080.0f;
            if (frame)
            {
                if (frame->camera.viewportWidth > 0.0f && frame->camera.viewportHeight > 0.0f)
                {
                    w = frame->camera.viewportWidth;
                    h = frame->camera.viewportHeight;
                }
                else if (frame->camera.aspectRatio > 0.0f)
                {
                    w = 1080.0f * frame->camera.aspectRatio;
                }
            }
            const Vec2 ss(w, h);
            copyBytes(&ss, sizeof(Vec2));
        }
        else if (propName == "TIME" || propName == "ENGINE_TIME")
        {
            const FLOAT32 t = frame
                ? (frame->hasAuthoritativeTime
                    ? frame->elapsedTime
                    : (static_cast<FLOAT32>(frame->frameIndex) * 0.016666667f))
                : 0.0f;
            copyBytes(&t, sizeof(FLOAT32));
        }
        else if (propName == "DELTA_TIME")
        {
            const FLOAT32 dt = frame ? frame->deltaTime : 0.0f;
            copyBytes(&dt, sizeof(FLOAT32));
        }
        else if (propName == "FRAME_INDEX")
        {
            const UINT32 fi = frame ? static_cast<UINT32>(frame->frameIndex) : 0u;
            copyBytes(&fi, sizeof(UINT32));
        }
        else if (propName == "BONE_PALETTE_OFFSET")
        {
            const UINT32 boneOff = ((ctx.executionContextMask & EXEC_CTX_RENDER_OBJECT) != 0u && ctx.renderObject != nullptr)
                ? ctx.renderObject->GetSkinningPaletteOffset(ctx.scene)
                : 0u;
            copyBytes(&boneOff, sizeof(UINT32));
        }
        else if (propName == "BONE_COUNT")
        {
            const UINT32 boneCnt = ((ctx.executionContextMask & EXEC_CTX_RENDER_OBJECT) != 0u && ctx.renderObject != nullptr)
                ? ctx.renderObject->GetSkinningMatrixCount()
                : 0u;
            copyBytes(&boneCnt, sizeof(UINT32));
        }
        else if (propName == "SKINNING_REVISION")
        {
            const UINT32 skinRev = ((ctx.executionContextMask & EXEC_CTX_RENDER_OBJECT) != 0u && ctx.renderObject != nullptr)
                ? ctx.renderObject->GetSkinningRevision()
                : 0u;
            copyBytes(&skinRev, sizeof(UINT32));
        }
        else if (propName == "NVS_PHYSICAL_PAGE_INDEX" ||
                 propName == "NVS_VIRTUAL_PAGE_ADDRESS" ||
                 propName == "NVS_CLIPMAP_LEVEL")
        {
            const UINT32 zeroU = 0u;
            copyBytes(&zeroU, sizeof(UINT32));
        }
        else if (propName == "NVS_DEPTH_BIAS_NDC" ||
                 propName == "NVS_NORMAL_BIAS_WORLD" ||
                 propName == "NVS_RECEIVER_MIN_Z" ||
                 propName == "NVS_RECEIVER_MAX_Z")
        {
            const FLOAT32 zeroF = 0.0f;
            copyBytes(&zeroF, sizeof(FLOAT32));
        }
        else if (propName == "NVS_PAGES_PER_AXIS")
        {
            const UINT32 v = frame ? frame->nvsPagesPerAxis : 64u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_PHYSICAL_POOL_CAPACITY")
        {
            const UINT32 v = frame ? frame->nvsPhysicalPoolCapacity : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_PAGE_TEXEL_SIZE")
        {
            const UINT32 v = frame ? frame->nvsPageTexelSize : kNVSPageTexelSize;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_VIEW_TABLE_COUNT")
        {
            const UINT32 v = frame ? frame->nvsViewTableCount : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_TOTAL_VIRTUAL_PAGES")
        {
            const UINT32 v = frame
                ? ((frame->nvsTotalVirtualPages > 0u) ? frame->nvsTotalVirtualPages : frame->nvsVirtualPageTableCount)
                : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_DEPTH_MODE")
        {
            const UINT32 v = frame ? frame->nvsDepthMode : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SAMPLING_AVAILABLE")
        {
            const UINT32 v = frame ? frame->nvsSamplingAvailable : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SNAPSHOT_FRAME_INDEX")
        {
            const UINT32 v = frame ? frame->nvsSnapshotFrameIndex : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SPARSE_ROOT_CAPACITY")
        {
            const UINT32 v = frame ? frame->nvsSparseRootCapacity : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SPARSE_DIR_STRIDE")
        {
            const UINT32 v = frame ? frame->nvsSparseDirStride : kNVSSparseDirStride;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SPARSE_DIR_CAPACITY")
        {
            const UINT32 v = frame ? frame->nvsSparseDirCapacity : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVS_SCREEN_PIXEL_WORLD_SCALE")
        {
            const Vec4 v = frame ? frame->nvsScreenPixelWorldScale : Vec4(0.0f, 0.0f, 0.0f, 0.0f);
            copyBytes(&v, sizeof(Vec4));
        }
        else if (propName == "NVG_DRAW_BUCKET_ID")
        {
            const BOOL clusterCtx = ((ctx.executionContextMask & EXEC_CTX_CLUSTER) != 0u);
            const UINT32 v = clusterCtx ? ctx.nvgDrawBucketId : kNVGInvalidIndex;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVG_VISIBLE_CLUSTER_COUNT")
        {
            const BOOL clusterCtx = ((ctx.executionContextMask & EXEC_CTX_CLUSTER) != 0u);
            const UINT32 v = clusterCtx ? ctx.nvgVisibleClusterCnt : 0u;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVG_VISIBLE_CLUSTER_OFFSET")
        {
            const BOOL clusterCtx = ((ctx.executionContextMask & EXEC_CTX_CLUSTER) != 0u);
            const UINT32 v = clusterCtx ? ctx.nvgVisibleClusterOff : kNVGInvalidIndex;
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVG_DEBUG_MODE")
        {
            const UINT32 v = (ctx.nvgDebugMode != 0u) ? ctx.nvgDebugMode : (frame ? frame->nvgDebugMode : 0u);
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVG_MAX_SCREEN_ERROR_PX")
        {
            const FLOAT32 v = (ctx.nvgMaxScreenErrorPx > 0.0f) ? ctx.nvgMaxScreenErrorPx : (frame ? frame->nvgMaxScreenErrorPx : 1.0f);
            copyBytes(&v, sizeof(FLOAT32));
        }
        else if (propName == "NVG_PHYSICAL_POOL_CAPACITY")
        {
            const UINT32 v = (ctx.nvgPhysicalPoolCap > 0u) ? ctx.nvgPhysicalPoolCap : (frame ? frame->nvgPhysicalPoolCapacity : 0u);
            copyBytes(&v, sizeof(UINT32));
        }
        else if (propName == "NVG_PAGE_WORD_COUNT")
        {
            const UINT32 v = kNVGPageWordCount;
            copyBytes(&v, sizeof(UINT32));
        }
    }

    BOOL CoreResourceBindingResolver::FillEngineLightsElementField(
        const FrameLightData&      light,
        const NSL::NSCStructField& f,
        BYTE*                      elemPtr,
        UINT32                     stride)
    {
        if (!elemPtr || f.offset >= stride) return false;
        const SIZE_T maxWrite = stride - f.offset;
        const std::string& pName = GetFieldPropertyName(f);

        auto w = [&](const void* src, SIZE_T sz) {
            std::memcpy(elemPtr + f.offset, src, (std::min<SIZE_T>)({ static_cast<SIZE_T>(f.size), sz, maxWrite }));
        };

        if (pName == "GLOBAL_LIGHT_POSITION")               { w(&light.position, sizeof(Vec3)); return true; }
        if (pName == "GLOBAL_LIGHT_COLOR")                  { const FLOAT32 c[3] = { light.color.r, light.color.g, light.color.b }; w(c, sizeof(c)); return true; }
        if (pName == "GLOBAL_LIGHT_DIRECTION")              { w(&light.direction, sizeof(Vec3)); return true; }
        if (pName == "GLOBAL_LIGHT_INTENSITY")              { w(&light.intensity, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_RANGE")                  { w(&light.range, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_TYPE")                   { const UINT32 t = static_cast<UINT32>(light.type); w(&t, sizeof(UINT32)); return true; }
        if (pName == "GLOBAL_LIGHT_INNER_CONE")             { const FLOAT32 o = std::clamp(light.outerConeAngle, 0.5f, 88.0f); const FLOAT32 r = std::clamp(light.innerConeAngle, 0.0f, o) * DEG2RAD; w(&r, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_OUTER_CONE")             { const FLOAT32 r = std::clamp(light.outerConeAngle, 0.5f, 88.0f) * DEG2RAD; w(&r, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_ATTENUATION")            { w(&light.attenuation, sizeof(Vec3)); return true; }
        if (pName == "GLOBAL_LIGHT_VP_MATRIX")              { w(&light.viewProjectionMatrix, sizeof(Mat4)); return true; }
        if (pName == "GLOBAL_LIGHT_CASTS_SHADOW")           { w(&light.castsShadow, sizeof(UINT32)); return true; }
        if (pName == "GLOBAL_LIGHT_SHADOW_VIEW_BASE")       { w(&light.shadowViewBase, sizeof(UINT32)); return true; }
        if (pName == "GLOBAL_LIGHT_SHADOW_VIEW_COUNT")      { w(&light.shadowViewCount, sizeof(UINT32)); return true; }
        if (pName == "GLOBAL_LIGHT_NVS_LIGHT_ID" || pName == "GLOBAL_LIGHT_NVS_ID") { w(&light.nvsLightId, sizeof(UINT32)); return true; }
        if (pName == "GLOBAL_LIGHT_SHADOW_BIAS")            { w(&light.shadowBias, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_NORMAL_BIAS")            { w(&light.shadowNormalBias, sizeof(FLOAT32)); return true; }
        if (pName == "GLOBAL_LIGHT_SHADOW_SOFTNESS")        { const UINT32 s = static_cast<UINT32>(light.shadowSoftness); w(&s, sizeof(UINT32)); return true; }
        return false;
    }

    BOOL CoreResourceBindingResolver::ResolveEngineStructuredBuffer(
        const NSL::NSCStructuredBuffer&   nscSb,
        ShaderStage                       stage,
        const StructuredBufferLayoutPlan& plan,
        const ShaderStructuredBufferSlot* matSb,
        const StageResolveContext&        ctx,
        NativeStructuredBufferBinding&    outBinding,
        std::string*                      outDiagnostic)
    {
        outBinding               = NativeStructuredBufferBinding{};
        outBinding.slotIndex     = nscSb.binding;
        outBinding.registerSpace = nscSb.set;
        outBinding.stage         = stage;
        outBinding.stride        = nscSb.stride;
        outBinding.isUAV         = false;

        if (!plan.valid || plan.domain == EngineResourceDomain::INVALID_MIXED_DOMAINS)
        {
            if (outDiagnostic) *outDiagnostic = plan.diagnostic;
            return false;
        }

        const FrameView* frame   = ctx.frameView;
        const UINT32 userCount   = matSb ? matSb->userElementCount : 0u;
        const BYTE*  userPtr     = (matSb && !matSb->userData.empty()) ? matSb->userData.data() : nullptr;
        const UINT32 userBytes   = (matSb && userCount > 0u)
            ? static_cast<UINT32>((std::min<SIZE_T>)(matSb->userData.size(), static_cast<SIZE_T>(userCount) * nscSb.stride))
            : 0u;

        auto emitNullSRV = [&]() -> BOOL {
            outBinding.buffer       = 0;
            outBinding.elementCount = 0u;
            outBinding.firstElement = 0u;
            outBinding.stride       = nscSb.stride;
            outBinding.byteSize     = 0u;
            outBinding.cpuData      = nullptr;
            outBinding.ownedData.clear();
            outBinding.isUAV        = false;
            return true;
        };

        // 1. CPU-packed Scene Lights (CST [0..N-1] engine + [N..N+M-1] user, 0-element Null SRV when N+M==0)
        if (plan.domain == EngineResourceDomain::LIGHTS)
        {
            const UINT32 engineCount = frame ? static_cast<UINT32>(frame->lights.size()) : 0u;
            const UINT32 totalCount  = engineCount + userCount;
            if (totalCount == 0u)
            {
                return emitNullSRV();
            }

            outBinding.elementCount = totalCount;
            outBinding.ownedData.assign(static_cast<SIZE_T>(totalCount) * nscSb.stride, 0u);
            for (UINT32 i = 0u; i < engineCount; ++i)
            {
                BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(i) * nscSb.stride;
                for (const auto& f : nscSb.fields)
                {
                    if (f.isGet)
                    {
                        FillEngineLightsElementField(frame->lights[i], f, elemPtr, nscSb.stride);
                    }
                }
            }
            for (UINT32 u = 0u; u < userCount; ++u)
            {
                BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(engineCount + u) * nscSb.stride;
                if (userPtr && (static_cast<SIZE_T>(u) + 1u) * nscSb.stride <= userBytes)
                {
                    std::memcpy(elemPtr, userPtr + static_cast<SIZE_T>(u) * nscSb.stride, nscSb.stride);
                }
                for (const auto& f : nscSb.fields)
                {
                    if (f.isGet && f.offset + f.size <= nscSb.stride)
                    {
                        std::memset(elemPtr + f.offset, 0, f.size);
                    }
                }
            }
            outBinding.cpuData  = outBinding.ownedData.data();
            outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
            return true;
        }

        // 2. Placeholder scene entity domains (Decals / Probes / Particles -> CST user tail or 0-elem Null SRV)
        if (plan.domain == EngineResourceDomain::DECALS ||
            plan.domain == EngineResourceDomain::REFLECTION_PROBES ||
            plan.domain == EngineResourceDomain::PARTICLES)
        {
            if (userCount == 0u)
            {
                return emitNullSRV();
            }
            outBinding.elementCount = userCount;
            outBinding.ownedData.assign(static_cast<SIZE_T>(userCount) * nscSb.stride, 0u);
            if (userPtr && userBytes > 0u)
            {
                std::memcpy(outBinding.ownedData.data(), userPtr, (std::min<SIZE_T>)(outBinding.ownedData.size(), userBytes));
            }
            for (UINT32 u = 0u; u < userCount; ++u)
            {
                BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(u) * nscSb.stride;
                for (const auto& f : nscSb.fields)
                {
                    if (f.isGet && f.offset + f.size <= nscSb.stride)
                    {
                        std::memset(elemPtr + f.offset, 0, f.size);
                    }
                }
            }
            outBinding.cpuData  = outBinding.ownedData.data();
            outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
            return true;
        }

        // 3. Per-Instance Skinning metadata (fallback for single-object draws)
        if (plan.domain == EngineResourceDomain::INSTANCE_SKINNING)
        {
            if (!ctx.renderObject || !ctx.renderObject->HasSkinningMatrices())
            {
                if ((ctx.executionContextMask & EXEC_CTX_RENDER_OBJECT) != 0u)
                {
                    if (outDiagnostic) *outDiagnostic = "Object lacks required skinning matrices for INSTANCE_SKINNING buffer.";
                    return false;
                }
                return emitNullSRV();
            }
            outBinding.elementCount = 1u;
            outBinding.ownedData.assign(nscSb.stride, 0u);
            const UINT32 boneOff = ctx.renderObject->GetSkinningPaletteOffset(ctx.scene);
            const UINT32 boneCnt = ctx.renderObject->GetSkinningMatrixCount();
            for (const auto& f : nscSb.fields)
            {
                if (!f.isGet || f.offset + sizeof(UINT32) > nscSb.stride) continue;
                const std::string& pName = GetFieldPropertyName(f);
                if (pName == "INSTANCE_BONE_OFFSET")
                    std::memcpy(outBinding.ownedData.data() + f.offset, &boneOff, sizeof(UINT32));
                else if (pName == "INSTANCE_BONE_COUNT")
                    std::memcpy(outBinding.ownedData.data() + f.offset, &boneCnt, sizeof(UINT32));
            }
            outBinding.cpuData  = outBinding.ownedData.data();
            outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
            return true;
        }

        // 4. GPU-backed physical domains (Direct Zero-Copy fast path + GPU Materializer Decision 6C)
        NativeHandle srcHandle      = 0;
        UINT32       srcElemCount   = 0u;
        UINT32       srcBaseElem    = 0u;
        UINT64       srcRevision    = frame ? frame->frameIndex : 0ull;
        const UINT32 canonicalStride = plan.canonicalSrcStride;

        const BOOL isClusterCtx = ((ctx.executionContextMask & EXEC_CTX_CLUSTER) != 0u);
        const BOOL isShadowCtx  = ((ctx.executionContextMask & EXEC_CTX_SHADOW_PAGE) != 0u);

        switch (plan.domain)
        {
            case EngineResourceDomain::GPU_SCENE_OBJECTS:
                // LIVE scene-wide across STANDARD, VIRTUALIZED, and Scene Compute!
                if (frame && frame->gpuSceneBuffer != 0)
                {
                    srcHandle    = frame->gpuSceneBuffer;
                    srcElemCount = (frame->gpuSceneHighWaterMark > 0u) ? frame->gpuSceneHighWaterMark : frame->gpuSceneCapacity;
                    srcRevision  = frame->gpuSceneMutationEpoch;
                }
                break;

            case EngineResourceDomain::NVG_PAGE_POOL:
                if (isClusterCtx)
                {
                    srcHandle    = (ctx.nvgPagePoolBuffer != 0) ? ctx.nvgPagePoolBuffer : (frame ? frame->nvgPagePoolBuffer : 0);
                    srcElemCount = (ctx.nvgPagePoolWords > 0u) ? ctx.nvgPagePoolWords : (frame ? frame->nvgPagePoolWordCount : 0u);
                    srcRevision  = (ctx.nvgPassRevision != 0ull) ? ctx.nvgPassRevision : (frame ? frame->nvgResidencyRevision : 0ull);
                }
                break;

            case EngineResourceDomain::NVG_BUCKET_RANGES:
                if (isClusterCtx)
                {
                    srcHandle    = (ctx.nvgBucketRangesBuffer != 0) ? ctx.nvgBucketRangesBuffer : (frame ? frame->nvgBucketRangesBuffer : 0);
                    srcElemCount = (ctx.nvgBucketRangesCount > 0u) ? ctx.nvgBucketRangesCount : (frame ? frame->nvgBucketRangesCount : 0u);
                    srcRevision  = (ctx.nvgPassRevision != 0ull) ? ctx.nvgPassRevision : (frame ? frame->frameIndex : 0ull);
                }
                break;

            case EngineResourceDomain::NVG_VISIBLE_CLUSTERS:
                if (isClusterCtx)
                {
                    srcHandle    = (ctx.nvgVisibleClustersBuf != 0) ? ctx.nvgVisibleClustersBuf : (frame ? frame->nvgVisibleClustersBuffer : 0);
                    srcElemCount = (ctx.nvgVisibleClustersCnt > 0u) ? ctx.nvgVisibleClustersCnt : (frame ? frame->nvgVisibleClustersCount : 0u);
                    srcRevision  = (ctx.nvgPassRevision != 0ull) ? ctx.nvgPassRevision : (frame ? frame->frameIndex : 0ull);
                }
                break;

            case EngineResourceDomain::NVG_SHADOW_PAGE_VIEWS:
                if (isClusterCtx && isShadowCtx)
                {
                    srcHandle    = (ctx.nvgShadowPageViewsBuf != 0) ? ctx.nvgShadowPageViewsBuf : (frame ? frame->nvgShadowPageViewsBuffer : 0);
                    srcElemCount = (ctx.nvgShadowPageViewsCnt > 0u) ? ctx.nvgShadowPageViewsCnt : (frame ? frame->nvgShadowPageViewsCount : 0u);
                    srcRevision  = (ctx.nvgPassRevision != 0ull) ? ctx.nvgPassRevision : (frame ? frame->frameIndex : 0ull);
                }
                break;

            case EngineResourceDomain::NVS_SPARSE_PAGE_HIERARCHY:
                if (frame && frame->nvsSparsePageHierarchyBuffer != 0)
                {
                    srcHandle    = frame->nvsSparsePageHierarchyBuffer;
                    srcElemCount = frame->nvsSparsePageHierarchyCount;
                    srcRevision  = frame->nvsSnapshotFrameIndex;
                }
                break;

            case EngineResourceDomain::NVS_VIRTUAL_PAGE_TABLE:
                if (frame && frame->nvsVirtualPageTableBuffer != 0)
                {
                    srcHandle    = frame->nvsVirtualPageTableBuffer;
                    srcElemCount = frame->nvsVirtualPageTableCount;
                    srcRevision  = frame->nvsSnapshotFrameIndex;
                }
                break;

            case EngineResourceDomain::NVS_PHYSICAL_PAGE_TABLE:
                if (frame && frame->nvsPhysicalPageTableBuffer != 0)
                {
                    srcHandle    = frame->nvsPhysicalPageTableBuffer;
                    srcElemCount = frame->nvsPhysicalPageTableCount;
                    srcRevision  = frame->nvsSnapshotFrameIndex;
                }
                break;

            case EngineResourceDomain::NVS_LIGHT_VIEW_TABLE:
                if (frame && frame->nvsLightViewTableBuffer != 0)
                {
                    srcHandle    = frame->nvsLightViewTableBuffer;
                    srcElemCount = frame->nvsLightViewTableCount;
                    srcRevision  = frame->nvsSnapshotFrameIndex;
                }
                break;

            case EngineResourceDomain::GLOBAL_BONE_PALETTE:
                if (frame && frame->gpuSkinningPaletteBuffer != 0 && frame->gpuSkinningPaletteCount > 0u)
                {
                    srcHandle    = frame->gpuSkinningPaletteBuffer;
                    srcElemCount = frame->gpuSkinningPaletteCount;
                    srcRevision  = frame->frameIndex;
                }
                else if (ctx.renderObject && ctx.renderObject->HasSkinningMatrices())
                {
                    // Single-object CPU skinning fallback when scene GPU palette hasn't been uploaded yet
                    const UINT32 palOffset  = ctx.renderObject->GetSkinningPaletteOffset(ctx.scene);
                    const UINT32 boneCount  = ctx.renderObject->GetSkinningMatrixCount();
                    const UINT32 totalCount = palOffset + boneCount + userCount;
                    if (totalCount == 0u) return emitNullSRV();

                    outBinding.elementCount = totalCount;
                    outBinding.ownedData.assign(static_cast<SIZE_T>(totalCount) * nscSb.stride, 0u);
                    const Mat4* boneData = ctx.renderObject->GetSkinningMatrixData();
                    for (UINT32 b = 0u; b < boneCount && boneData != nullptr; ++b)
                    {
                        BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(palOffset + b) * nscSb.stride;
                        for (const auto& f : nscSb.fields)
                        {
                            if (f.isGet && GetFieldPropertyName(f) == "GLOBAL_BONE_MATRIX" && f.offset + sizeof(Mat4) <= nscSb.stride)
                            {
                                std::memcpy(elemPtr + f.offset, &boneData[b], sizeof(Mat4));
                            }
                        }
                    }
                    for (UINT32 u = 0u; u < userCount; ++u)
                    {
                        BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(palOffset + boneCount + u) * nscSb.stride;
                        if (userPtr && (static_cast<SIZE_T>(u) + 1u) * nscSb.stride <= userBytes)
                            std::memcpy(elemPtr, userPtr + static_cast<SIZE_T>(u) * nscSb.stride, nscSb.stride);
                        for (const auto& f : nscSb.fields)
                            if (f.isGet && f.offset + f.size <= nscSb.stride)
                                std::memset(elemPtr + f.offset, 0, f.size);
                    }
                    outBinding.cpuData  = outBinding.ownedData.data();
                    outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
                    return true;
                }
                break;

            case EngineResourceDomain::PER_OBJECT_BONE_SLICE:
                if (ctx.renderObject && ctx.renderObject->HasSkinningMatrices())
                {
                    const UINT32 boneOff = ctx.renderObject->GetSkinningPaletteOffset(ctx.scene);
                    const UINT32 boneCnt = ctx.renderObject->GetSkinningMatrixCount();
                    if (frame && frame->gpuSkinningPaletteBuffer != 0 &&
                        boneOff + boneCnt <= frame->gpuSkinningPaletteCount)
                    {
                        srcHandle    = frame->gpuSkinningPaletteBuffer;
                        srcBaseElem  = boneOff;
                        srcElemCount = boneCnt;
                        srcRevision  = static_cast<UINT64>(ctx.renderObject->GetSkinningRevision()) ^ (frame->frameIndex << 32u);
                    }
                    else
                    {
                        const UINT32 totalCount = boneCnt + userCount;
                        if (totalCount == 0u) return emitNullSRV();
                        outBinding.elementCount = totalCount;
                        outBinding.ownedData.assign(static_cast<SIZE_T>(totalCount) * nscSb.stride, 0u);
                        const Mat4* boneData = ctx.renderObject->GetSkinningMatrixData();
                        for (UINT32 b = 0u; b < boneCnt && boneData != nullptr; ++b)
                        {
                            BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(b) * nscSb.stride;
                            for (const auto& f : nscSb.fields)
                            {
                                if (f.isGet && GetFieldPropertyName(f) == "BONE_MATRIX" && f.offset + sizeof(Mat4) <= nscSb.stride)
                                {
                                    std::memcpy(elemPtr + f.offset, &boneData[b], sizeof(Mat4));
                                }
                            }
                        }
                        for (UINT32 u = 0u; u < userCount; ++u)
                        {
                            BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(boneCnt + u) * nscSb.stride;
                            if (userPtr && (static_cast<SIZE_T>(u) + 1u) * nscSb.stride <= userBytes)
                                std::memcpy(elemPtr, userPtr + static_cast<SIZE_T>(u) * nscSb.stride, nscSb.stride);
                            for (const auto& f : nscSb.fields)
                                if (f.isGet && f.offset + f.size <= nscSb.stride)
                                    std::memset(elemPtr + f.offset, 0, f.size);
                        }
                        outBinding.cpuData  = outBinding.ownedData.data();
                        outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
                        return true;
                    }
                }
                break;

            default:
                break;
        }

        // If logical element count is 0 (and 0 user elements) -> CST Null Buffer SRV with ZERO allocation!
        if ((srcHandle == 0 || srcElemCount == 0u) && userCount == 0u)
        {
            return emitNullSRV();
        }

        // Fast Path: Direct Zero-Copy GPU SRV (exact layout, no Add() fields, 0 user elements)
        if (plan.canDirectZeroCopy && srcHandle != 0 && srcElemCount > 0u)
        {
            outBinding.buffer       = srcHandle;
            outBinding.firstElement = srcBaseElem;
            outBinding.elementCount = srcElemCount;
            outBinding.stride       = canonicalStride;
            outBinding.byteSize     = srcElemCount * canonicalStride;
            outBinding.cpuData      = nullptr;
            outBinding.ownedData.clear();
            outBinding.isUAV        = false;
            return true;
        }

        // Fallback for NVS_LIGHT_VIEW_TABLE CPU mirror if GPU materializer wasn't provided (e.g. non-D3D12)
        if (plan.domain == EngineResourceDomain::NVS_LIGHT_VIEW_TABLE &&
            (!ctx.materializer || !ctx.outPreDrawDispatches) &&
            frame != nullptr)
        {
            const NVSClipmapLevelGPUData* entries = nullptr;
            UINT32 mirrorCount = 0u;
            if (frame->nvsLightViewTableCpuMirrorPtr != nullptr && frame->nvsLightViewTableCpuMirrorCount > 0u)
            {
                entries     = reinterpret_cast<const NVSClipmapLevelGPUData*>(frame->nvsLightViewTableCpuMirrorPtr);
                mirrorCount = frame->nvsLightViewTableCpuMirrorCount;
            }
            else if (!frame->nvsLightViewTableCpuMirror.empty())
            {
                entries     = reinterpret_cast<const NVSClipmapLevelGPUData*>(frame->nvsLightViewTableCpuMirror.data());
                mirrorCount = static_cast<UINT32>(frame->nvsLightViewTableCpuMirror.size() / sizeof(NVSClipmapLevelGPUData));
            }

            const UINT32 totalCount = mirrorCount + userCount;
            if (totalCount == 0u) return emitNullSRV();

            outBinding.elementCount = totalCount;
            outBinding.ownedData.assign(static_cast<SIZE_T>(totalCount) * nscSb.stride, 0u);
            for (UINT32 i = 0u; i < mirrorCount && entries != nullptr; ++i)
            {
                const BYTE* srcBytes = reinterpret_cast<const BYTE*>(&entries[i]);
                BYTE* dstBytes       = outBinding.ownedData.data() + static_cast<SIZE_T>(i) * nscSb.stride;
                for (const auto& span : plan.copySpans)
                {
                    std::memcpy(dstBytes + span.dstWordOffset * 4u, srcBytes + span.srcWordOffset * 4u, span.wordCount * 4u);
                }
            }
            for (UINT32 u = 0u; u < userCount; ++u)
            {
                BYTE* elemPtr = outBinding.ownedData.data() + static_cast<SIZE_T>(mirrorCount + u) * nscSb.stride;
                if (userPtr && (static_cast<SIZE_T>(u) + 1u) * nscSb.stride <= userBytes)
                    std::memcpy(elemPtr, userPtr + static_cast<SIZE_T>(u) * nscSb.stride, nscSb.stride);
                for (const auto& f : nscSb.fields)
                    if (f.isGet && f.offset + f.size <= nscSb.stride)
                        std::memset(elemPtr + f.offset, 0, f.size);
            }
            outBinding.cpuData  = outBinding.ownedData.data();
            outBinding.byteSize = static_cast<UINT32>(outBinding.ownedData.size());
            return true;
        }

        // General Path (Decision 6C): GPU Gather/Pack Materialization (zero CPU readback, zero WaitForGPU!)
        if (!ctx.materializer || !ctx.outPreDrawDispatches)
        {
            if (outDiagnostic)
            {
                *outDiagnostic = "Custom reflection StructuredBuffer layout requires GPU materializer, which is unavailable in this pass.";
            }
            return false;
        }

        const UINT32 srcBufferTotalElems = (plan.domain == EngineResourceDomain::PER_OBJECT_BONE_SLICE && frame)
            ? frame->gpuSkinningPaletteCount
            : (srcBaseElem + srcElemCount);
        const UINT64 srcTotalByteSize64 = (srcHandle != 0)
            ? (static_cast<UINT64>(srcBufferTotalElems) * static_cast<UINT64>(canonicalStride))
            : 0ull;
        if (srcTotalByteSize64 > 0xFFFFFFFFull)
        {
            if (outDiagnostic) *outDiagnostic = "Materializer source byte size overflow.";
            return false;
        }
        const UINT32 srcTotalByteSize = static_cast<UINT32>(srcTotalByteSize64);

        NativeHandle matHandle    = 0;
        UINT32       matElemCount = 0u;
        UINT32       matByteSize  = 0u;
        if (!ctx.materializer->AcquireMaterializedBuffer(
                plan,
                srcHandle,
                srcTotalByteSize,
                srcRevision,
                srcBaseElem,
                (srcHandle != 0) ? srcElemCount : 0u,
                userPtr,
                userBytes,
                userCount,
                *ctx.outPreDrawDispatches,
                matHandle,
                matElemCount,
                matByteSize,
                outDiagnostic))
        {
            return false;
        }

        if (matHandle == 0 || matElemCount == 0u)
        {
            return emitNullSRV();
        }

        outBinding.buffer       = matHandle;
        outBinding.firstElement = 0u;
        outBinding.elementCount = matElemCount;
        outBinding.stride       = nscSb.stride;
        outBinding.byteSize     = matByteSize;
        outBinding.cpuData      = nullptr;
        outBinding.ownedData.clear();
        outBinding.isUAV        = false;
        return true;
    }

    BOOL CoreResourceBindingResolver::ResolveGraphicsStageBindings(
        const Shader*                               shader,
        ShaderStage                                 stage,
        const ShaderStageProps&                     primaryProps,
        const ShaderStageProps*                     fallbackProps,
        const StageResolveContext&                  ctx,
        std::vector<NativeConstantBufferBinding>&   outCBuffers,
        std::vector<NativeTextureBinding>&          outTextures,
        std::vector<NativeSamplerBinding>&          outSamplers,
        std::vector<NativeStructuredBufferBinding>& outStructuredBuffers,
        BOOL*                                       outHasReflectedInstance,
        UINT16*                                     outInstanceSpace,
        UINT16*                                     outInstanceSlot,
        std::string*                                outDiagnostic,
        std::vector<NativeRWTextureBinding>*        outRWTextures)
    {
        if (!shader || !shader->IsValid()) return true;

        if (primaryProps.HasDestroyedResource() ||
            (fallbackProps && fallbackProps->HasDestroyedResource()))
        {
            if (outDiagnostic) *outDiagnostic = "Material references a destroyed texture, sampler, or GPUBuffer.";
            return false;
        }

        const ShaderReflection& refl = shader->GetReflection();
        const FrameView* frame       = ctx.frameView;
        const BOOL isDepthPrepass    = (ctx.passType == PassType::DEPTH);
        const BOOL isShadowPass      = (ctx.passType == PassType::SHADOW);

        // -----------------------------------------------------------------
        // 1. Constant Buffers
        // -----------------------------------------------------------------
        for (const auto& nscBuf : refl.buffers)
        {
            if (nscBuf.binding == 0xFFFF || nscBuf.set >= 16u || nscBuf.totalSize == 0u) continue;
            if (nscBuf.set == 3u && nscBuf.binding == 1u) continue; // Reserved for graphics root constants
            if (IsPromotedCBVSlot(ctx.promotedRootKey, stage, nscBuf.set, nscBuf.binding)) continue;

            NativeConstantBufferBinding cb{};
            cb.slotIndex     = nscBuf.binding;
            cb.registerSpace = nscBuf.set;
            cb.stage         = stage;
            cb.byteSize      = nscBuf.totalSize;
            cb.ownedData.assign(nscBuf.totalSize, 0u);

            const ShaderBufferSlot* matSlot = nullptr;
            for (const auto& mbs : primaryProps.buffers)
            {
                if ((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name ||
                    mbs.resourceName == nscBuf.name)
                {
                    matSlot = &mbs;
                    break;
                }
            }
            if (!matSlot)
            {
                for (const auto& mbs : primaryProps.buffers)
                {
                    if (mbs.slotIndex == nscBuf.binding && mbs.registerSpace == nscBuf.set)
                    {
                        matSlot = &mbs;
                        break;
                    }
                }
            }
            if ((!matSlot || (!matSlot->dirty && fallbackProps)) && fallbackProps)
            {
                for (const auto& mbs : fallbackProps->buffers)
                {
                    if (((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name ||
                         mbs.resourceName == nscBuf.name ||
                         (mbs.slotIndex == nscBuf.binding && mbs.registerSpace == nscBuf.set)) &&
                        (mbs.dirty || !matSlot))
                    {
                        matSlot = &mbs;
                        break;
                    }
                }
            }

            if (matSlot && !matSlot->cpuData.empty())
            {
                const SIZE_T copySize = (std::min)(cb.ownedData.size(), matSlot->cpuData.size());
                std::memcpy(cb.ownedData.data(), matSlot->cpuData.data(), copySize);
                cb.dirty = matSlot->dirty;
            }
            else
            {
                for (const auto& field : nscBuf.fields)
                {
                    if (!field.isGet && !field.defaultData.empty() && field.offset + field.size <= nscBuf.totalSize)
                    {
                        const SIZE_T copyFieldSize = (std::min)(static_cast<SIZE_T>(field.size), field.defaultData.size());
                        std::memcpy(cb.ownedData.data() + field.offset, field.defaultData.data(), copyFieldSize);
                    }
                }
            }

            for (const auto& field : nscBuf.fields)
            {
                if (field.isGet && field.offset < nscBuf.totalSize)
                {
                    FillEngineConstantBufferField(
                        GetCBufferFieldPropertyName(field),
                        ctx,
                        cb.ownedData.data(),
                        field.offset,
                        field.size,
                        nscBuf.totalSize);
                }
            }

            cb.cpuData = cb.ownedData.data();
            outCBuffers.push_back(std::move(cb));
        }

        // -----------------------------------------------------------------
        // 2. Textures
        // -----------------------------------------------------------------
        for (const auto& nscTex : refl.textures)
        {
            if (nscTex.binding == 0xFFFF || nscTex.set >= 16u) continue;

            NativeTextureBinding tb{};
            tb.slotIndex     = nscTex.binding;
            tb.registerSpace = nscTex.set;
            tb.stage         = stage;
            tb.texture       = 0;

            if (nscTex.IsEngineProvided())
            {
                if (nscTex.propertyName == "NVS_SHADOW_MAP")
                {
                    if (isShadowPass)
                    {
                        // Cannot sample NVS shadow map while rasterizing into it as DSV!
                        if (outDiagnostic) *outDiagnostic = "Cannot bind NVS_SHADOW_MAP as SRV during PassType::SHADOW.";
                        return false;
                    }
                    tb.texture = frame ? frame->nvsShadowMapTexture : 0;
                    if (tb.texture == 0)
                    {
                        tb.nullViewType = TextureType::TEXTURE_2D_ARRAY;
                        tb.nullFormat   = TextureFormat::R32_FLOAT;
                    }
                }
                else
                {
                    if (outDiagnostic) *outDiagnostic = "Unavailable engine texture property: " + nscTex.propertyName;
                    return false;
                }
            }
            else
            {
                Texture* boundTex = nullptr;
                auto findTexInProps = [&](const ShaderStageProps& sp) {
                    // 1. Exact qualified (propertyName__group/resource) match first
                    for (const auto& mt : sp.textures)
                    {
                        if (!nscTex.groupQualifiedName.empty() &&
                            ((mt.propertyName + "__" + mt.groupName) == nscTex.groupQualifiedName ||
                             (mt.propertyName + "__" + mt.resourceName) == nscTex.groupQualifiedName) &&
                            mt.resourceName == nscTex.name)
                        {
                            if (mt.texture)
                            {
                                boundTex   = mt.texture;
                                tb.texture = mt.texture->GetNativeHandle();
                                tb.mipSlice = mt.mipSlice;
                            }
                            return;
                        }
                    }
                    // 2. Bare resourceName match
                    for (const auto& mt : sp.textures)
                    {
                        if (mt.resourceName == nscTex.name ||
                            (mt.propertyName + "__" + mt.resourceName) == nscTex.groupQualifiedName)
                        {
                            if (mt.texture)
                            {
                                boundTex   = mt.texture;
                                tb.texture = mt.texture->GetNativeHandle();
                                tb.mipSlice = mt.mipSlice;
                            }
                            return;
                        }
                    }
                    // 3. Slot/Space match
                    for (const auto& mt : sp.textures)
                    {
                        if (mt.slotIndex == nscTex.binding && mt.registerSpace == nscTex.set)
                        {
                            if (mt.texture)
                            {
                                boundTex   = mt.texture;
                                tb.texture = mt.texture->GetNativeHandle();
                                tb.mipSlice = mt.mipSlice;
                            }
                            return;
                        }
                    }
                };
                findTexInProps(primaryProps);
                if (!boundTex && fallbackProps) findTexInProps(*fallbackProps);

                if (!boundTex || tb.texture == 0)
                {
                    if (outDiagnostic) *outDiagnostic = "Missing required user Add() Texture: " + nscTex.name;
                    return false;
                }
            }

            if (frame && !isShadowPass)
            {
                if (frame->camera.renderTarget.type == RenderTargetType::TEXTURE &&
                    tb.texture != 0 &&
                    tb.texture == frame->camera.renderTarget.colorTexture)
                {
                    if (outDiagnostic) *outDiagnostic = "Render target feedback loop detected.";
                    return false;
                }
                if (isDepthPrepass &&
                    tb.texture != 0 &&
                    tb.texture == frame->camera.renderTarget.depthStencilTexture)
                {
                    if (outDiagnostic) *outDiagnostic = "SceneDepth SRV feedback loop during Depth Prepass.";
                    return false;
                }
            }

            outTextures.push_back(tb);
        }

        // -----------------------------------------------------------------
        // 3. Samplers
        // -----------------------------------------------------------------
        for (const auto& nscSamp : refl.samplers)
        {
            if (nscSamp.binding == 0xFFFF || nscSamp.set >= 16u) continue;

            NativeSamplerBinding sb{};
            sb.slotIndex     = nscSamp.binding;
            sb.registerSpace = nscSamp.set;
            sb.stage         = stage;
            sb.sampler       = 0;

            if (nscSamp.IsEngineProvided())
            {
                if (nscSamp.propertyName == "NVS_COMPARISON_SAMPLER" && nscSamp.isComparison)
                {
                    sb.sampler = frame ? frame->nvsComparisonSampler : 0;
                }
                else if (nscSamp.propertyName == "NVS_DEPTH_SAMPLER" && !nscSamp.isComparison)
                {
                    sb.sampler = frame ? frame->nvsDepthSampler : 0;
                }
                if (sb.sampler == 0)
                {
                    if (outDiagnostic) *outDiagnostic = "Unavailable engine sampler property: " + nscSamp.propertyName;
                    return false;
                }
                outSamplers.push_back(sb);
                continue;
            }

            SamplerState* boundSamp = nullptr;
            auto findSampInProps = [&](const ShaderStageProps& sp) {
                for (const auto& ms : sp.samplers)
                {
                    if (!nscSamp.groupQualifiedName.empty() &&
                        ((ms.propertyName + "__" + ms.groupName) == nscSamp.groupQualifiedName ||
                         (ms.propertyName + "__" + ms.resourceName) == nscSamp.groupQualifiedName) &&
                        ms.resourceName == nscSamp.name)
                    {
                        if (ms.sampler)
                        {
                            boundSamp  = ms.sampler;
                            sb.sampler = ms.sampler->GetNativeHandle();
                        }
                        return;
                    }
                }
                for (const auto& ms : sp.samplers)
                {
                    if (ms.resourceName == nscSamp.name ||
                        (ms.propertyName + "__" + ms.resourceName) == nscSamp.groupQualifiedName)
                    {
                        if (ms.sampler)
                        {
                            boundSamp  = ms.sampler;
                            sb.sampler = ms.sampler->GetNativeHandle();
                        }
                        return;
                    }
                }
                for (const auto& ms : sp.samplers)
                {
                    if (ms.slotIndex == nscSamp.binding && ms.registerSpace == nscSamp.set)
                    {
                        if (ms.sampler)
                        {
                            boundSamp  = ms.sampler;
                            sb.sampler = ms.sampler->GetNativeHandle();
                        }
                        return;
                    }
                }
            };
            findSampInProps(primaryProps);
            if (!boundSamp && fallbackProps) findSampInProps(*fallbackProps);

            if (!boundSamp || sb.sampler == 0 || boundSamp->IsComparison() != nscSamp.isComparison)
            {
                if (outDiagnostic) *outDiagnostic = "Missing or mismatched user Add() Sampler: " + nscSamp.name;
                return false;
            }
            outSamplers.push_back(sb);
        }

        // -----------------------------------------------------------------
        // 4. StructuredBuffers (transactional so failed binding never poisons materializer cache)
        // -----------------------------------------------------------------
        const BOOL hasMatTxn = (ctx.materializer != nullptr);
        const CoreGPUResourceMaterializer::TransactionSnapshot matSnap =
            hasMatTxn ? ctx.materializer->BeginTransaction() : CoreGPUResourceMaterializer::TransactionSnapshot{};
        const SIZE_T preDrawBaseSize = ctx.outPreDrawDispatches ? ctx.outPreDrawDispatches->size() : 0u;
        auto rollbackStageMat = [&]() {
            if (hasMatTxn)
            {
                ctx.materializer->RollbackTransaction(matSnap);
            }
            if (ctx.outPreDrawDispatches && ctx.outPreDrawDispatches->size() > preDrawBaseSize)
            {
                ctx.outPreDrawDispatches->resize(preDrawBaseSize);
            }
        };

        for (const auto& nscSb : refl.structuredBuffers)
        {
            if (nscSb.binding == 0xFFFF || nscSb.set >= 16u || nscSb.stride == 0u) continue;
            if (IsPromotedSRVSlot(ctx.promotedRootKey, stage, nscSb.set, nscSb.binding)) continue;

            const BOOL isReflUav = (refl.FindUAV(nscSb.name) != nullptr);

            const ShaderStructuredBufferSlot* matSb = nullptr;
            auto findSbInProps = [&](const ShaderStageProps& sp) -> const ShaderStructuredBufferSlot* {
                for (const auto& ms : sp.structuredBuffers)
                {
                    if (ms.isUAV == isReflUav && (ms.propertyName + "__" + ms.resourceName) == nscSb.name)
                    {
                        return &ms;
                    }
                }
                for (const auto& ms : sp.structuredBuffers)
                {
                    if (ms.isUAV == isReflUav && ms.resourceName == nscSb.name)
                    {
                        return &ms;
                    }
                }
                for (const auto& ms : sp.structuredBuffers)
                {
                    if (ms.isUAV == isReflUav && ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set)
                    {
                        return &ms;
                    }
                }
                return nullptr;
            };
            matSb = findSbInProps(primaryProps);
            if ((!matSb || (matSb->gpuBuffer == nullptr && matSb->gpuBufferHandle == 0 &&
                            matSb->userElementCount == 0u && matSb->elementCount == 0u)) &&
                fallbackProps)
            {
                const ShaderStructuredBufferSlot* fbSb = findSbInProps(*fallbackProps);
                if (fbSb && (fbSb->gpuBuffer != nullptr || fbSb->gpuBufferHandle != 0 ||
                             fbSb->userElementCount > 0u || fbSb->elementCount > 0u))
                {
                    matSb = fbSb;
                }
            }

            const UINT32 userCount = matSb ? matSb->userElementCount : 0u;
            const StructuredBufferLayoutPlan plan = AnalyzeStructuredBuffer(nscSb, stage, userCount);
            if (!plan.valid)
            {
                if (outDiagnostic) *outDiagnostic = plan.diagnostic;
                rollbackStageMat();
                return false;
            }

            if (plan.domain == EngineResourceDomain::INSTANCES)
            {
                if (outHasReflectedInstance) *outHasReflectedInstance = true;
                if (outInstanceSpace)        *outInstanceSpace        = nscSb.set;
                if (outInstanceSlot)         *outInstanceSlot         = nscSb.binding;
                continue;
            }

            // Explicit user GPUBuffer binding (USER_ONLY or direct override with matching stride)
            if (plan.domain == EngineResourceDomain::USER_ONLY)
            {
                const NativeHandle resolvedHandle = (matSb && matSb->gpuBuffer != nullptr)
                    ? matSb->gpuBuffer->GetNativeHandle()
                    : (matSb ? matSb->gpuBufferHandle : 0);
                const UINT32 resolvedStride = (matSb && matSb->gpuBuffer != nullptr)
                    ? matSb->gpuBuffer->GetStride()
                    : (matSb ? matSb->gpuBufferStride : 0u);

                NativeStructuredBufferBinding sbb{};
                sbb.slotIndex     = nscSb.binding;
                sbb.registerSpace = nscSb.set;
                sbb.stage         = stage;
                sbb.stride        = nscSb.stride;
                sbb.isUAV         = isReflUav;

                if (matSb && resolvedHandle != 0)
                {
                    if (resolvedStride != nscSb.stride)
                    {
                        if (outDiagnostic) *outDiagnostic = "User GPUBuffer stride mismatch on StructuredBuffer: " + nscSb.name;
                        rollbackStageMat();
                        return false;
                    }
                    sbb.buffer       = resolvedHandle;
                    sbb.elementCount = matSb->gpuBuffer ? matSb->gpuBuffer->GetElementCount() : matSb->gpuBufferElementCount;
                    sbb.byteSize     = matSb->gpuBuffer ? matSb->gpuBuffer->GetByteSize() : matSb->gpuBufferByteSize;
                    sbb.uavAccess    = matSb->uavAccess;
                    outStructuredBuffers.push_back(std::move(sbb));
                    continue;
                }

                const UINT32 count = matSb
                    ? (matSb->userElementCount > 0u ? matSb->userElementCount : matSb->elementCount)
                    : 0u;
                if (count == 0u && !isReflUav && !ctx.allowEmptyUserSBAsNull)
                {
                    if (outDiagnostic) *outDiagnostic = "Missing required user Add() StructuredBuffer: " + nscSb.name;
                    rollbackStageMat();
                    return false;
                }
                if (count == 0u)
                {
                    sbb.elementCount = 0u;
                    sbb.byteSize     = 0u;
                    outStructuredBuffers.push_back(std::move(sbb));
                    continue;
                }

                sbb.elementCount = count;
                sbb.ownedData.assign(static_cast<SIZE_T>(count) * nscSb.stride, 0u);
                if (matSb && !matSb->userData.empty())
                {
                    std::memcpy(sbb.ownedData.data(), matSb->userData.data(), (std::min)(sbb.ownedData.size(), matSb->userData.size()));
                }
                else if (matSb && !matSb->cpuData.empty())
                {
                    std::memcpy(sbb.ownedData.data(), matSb->cpuData.data(), (std::min)(sbb.ownedData.size(), matSb->cpuData.size()));
                }
                sbb.cpuData  = sbb.ownedData.data();
                sbb.byteSize = static_cast<UINT32>(sbb.ownedData.size());
                outStructuredBuffers.push_back(std::move(sbb));
                continue;
            }

            // Engine-owned or CST Hybrid StructuredBuffer domain
            NativeStructuredBufferBinding sbb{};
            if (!ResolveEngineStructuredBuffer(nscSb, stage, plan, matSb, ctx, sbb, outDiagnostic))
            {
                rollbackStageMat();
                return false;
            }
            outStructuredBuffers.push_back(std::move(sbb));
        }

        // -----------------------------------------------------------------
        // 5. UAV RWTextures & Unstructured UAV Buffers
        // -----------------------------------------------------------------
        if (!refl.uavBuffers.empty())
        {
            auto resolveSingleRwTexEntry = [&](const std::string& groupQualName,
                                               const std::string& entryName,
                                               UINT16             uavSet,
                                               UINT16             uavBinding) -> BOOL {
                if (uavBinding == 0xFFFF || uavSet >= 16u) return true;
                if (!outRWTextures)
                {
                    if (outDiagnostic) *outDiagnostic = "Pass does not support RWTexture UAV bindings: " + entryName;
                    rollbackStageMat();
                    return false;
                }

                auto findRwtInProps = [&](const ShaderStageProps& sp) -> const ShaderRWTextureSlot* {
                    for (const auto& mr : sp.rwTextures)
                    {
                        if (!groupQualName.empty() &&
                            ((mr.propertyName + "__" + mr.groupName) == groupQualName ||
                             (mr.propertyName + "__" + mr.resourceName) == groupQualName) &&
                            mr.resourceName == entryName)
                        {
                            if (mr.texture != nullptr || mr.textureHandle != 0) return &mr;
                        }
                    }
                    for (const auto& mr : sp.rwTextures)
                    {
                        if (mr.resourceName == entryName ||
                            (mr.propertyName + "__" + mr.resourceName) == groupQualName ||
                            (mr.groupName + "__" + mr.resourceName) == groupQualName)
                        {
                            if (mr.texture != nullptr || mr.textureHandle != 0) return &mr;
                        }
                    }
                    for (const auto& mr : sp.rwTextures)
                    {
                        if (mr.slotIndex == uavBinding && mr.registerSpace == uavSet)
                        {
                            if (mr.texture != nullptr || mr.textureHandle != 0) return &mr;
                        }
                    }
                    return nullptr;
                };

                const ShaderRWTextureSlot* boundRwt = findRwtInProps(primaryProps);
                if (!boundRwt && fallbackProps) boundRwt = findRwtInProps(*fallbackProps);

                const NativeHandle rwtHandle = boundRwt
                    ? (boundRwt->texture ? boundRwt->texture->GetNativeHandle() : boundRwt->textureHandle)
                    : 0;
                if (!boundRwt || rwtHandle == 0 ||
                    (boundRwt->texture && boundRwt->texture->GetUsage() != TextureUsage::STORAGE))
                {
                    if (outDiagnostic) *outDiagnostic = "Missing or non-STORAGE user Add() RWTexture: " + entryName;
                    rollbackStageMat();
                    return false;
                }

                if (frame && !isShadowPass &&
                    frame->camera.renderTarget.type == RenderTargetType::TEXTURE &&
                    rwtHandle == frame->camera.renderTarget.colorTexture)
                {
                    if (outDiagnostic) *outDiagnostic = "Simultaneous RTV + UAV binding on active render target.";
                    rollbackStageMat();
                    return false;
                }

                for (const auto& srvTex : outTextures)
                {
                    if (srvTex.texture == rwtHandle &&
                        (srvTex.mipSlice == 0xFFFFFFFFu || srvTex.mipSlice == boundRwt->mipSlice))
                    {
                        if (outDiagnostic) *outDiagnostic = "Simultaneous SRV + UAV binding on texture: " + entryName;
                        rollbackStageMat();
                        return false;
                    }
                }

                for (const auto& existingRwt : *outRWTextures)
                {
                    if (existingRwt.stage == stage &&
                        existingRwt.registerSpace == uavSet &&
                        existingRwt.slotIndex == uavBinding)
                    {
                        return true;
                    }
                }

                NativeRWTextureBinding rwtb{};
                rwtb.slotIndex     = uavBinding;
                rwtb.registerSpace = uavSet;
                rwtb.stage         = stage;
                rwtb.texture       = rwtHandle;
                rwtb.mipSlice      = boundRwt->mipSlice;
                rwtb.uavAccess     = boundRwt->uavAccess;
                outRWTextures->push_back(rwtb);
                return true;
            };

            for (const auto& nscUav : refl.uavBuffers)
            {
                if (nscUav.binding == 0xFFFF && nscUav.fields.empty()) continue;
                if (nscUav.set >= 16u) continue;

                const BOOL isRwTexKind = (nscUav.resourceKind == ResourceKind::RW_TEXTURE_2D);

                if (isRwTexKind)
                {
                    if (!nscUav.fields.empty())
                    {
                        for (const auto& f : nscUav.fields)
                        {
                            if (f.isGet) continue;
                            std::string shortEntry = f.name;
                            const SIZE_T sep = shortEntry.rfind("__");
                            if (sep != std::string::npos)
                            {
                                shortEntry = shortEntry.substr(sep + 2u);
                            }
                            const UINT16 entryBinding = (f.binding != 0xFFFF) ? f.binding : nscUav.binding;
                            const UINT16 entrySet     = (f.binding != 0xFFFF) ? f.set     : nscUav.set;
                            if (!resolveSingleRwTexEntry(nscUav.name, shortEntry.empty() ? nscUav.name : shortEntry, entrySet, entryBinding))
                            {
                                return false;
                            }
                        }
                    }
                    else
                    {
                        if (!resolveSingleRwTexEntry(nscUav.name, nscUav.name, nscUav.set, nscUav.binding))
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    // RWStructuredBuffer: ensure a valid UAV GPUBuffer was bound
                    BOOL foundBoundUavSb = false;
                    for (const auto& existingSb : outStructuredBuffers)
                    {
                        if (existingSb.stage == stage &&
                            existingSb.isUAV &&
                            existingSb.registerSpace == nscUav.set &&
                            existingSb.slotIndex == nscUav.binding &&
                            existingSb.buffer != 0)
                        {
                            foundBoundUavSb = true;
                            break;
                        }
                    }
                    if (!foundBoundUavSb)
                    {
                        auto findUavSbInProps = [&](const ShaderStageProps& sp) -> const ShaderStructuredBufferSlot* {
                            for (const auto& ms : sp.structuredBuffers)
                            {
                                if (ms.isUAV &&
                                    (ms.resourceName == nscUav.name ||
                                     (ms.propertyName + "__" + ms.resourceName) == nscUav.name ||
                                     (ms.slotIndex == nscUav.binding && ms.registerSpace == nscUav.set)))
                                {
                                    if ((ms.gpuBuffer && ms.gpuBuffer->IsValid()) || ms.gpuBufferHandle != 0)
                                        return &ms;
                                }
                            }
                            return nullptr;
                        };
                        const ShaderStructuredBufferSlot* uavSlot = findUavSbInProps(primaryProps);
                        if (!uavSlot && fallbackProps) uavSlot = findUavSbInProps(*fallbackProps);
                        if (!uavSlot)
                        {
                            if (outDiagnostic) *outDiagnostic = "Missing required user Add() RWStructuredBuffer: " + nscUav.name;
                            rollbackStageMat();
                            return false;
                        }
                        NativeStructuredBufferBinding sbb{};
                        sbb.slotIndex     = nscUav.binding;
                        sbb.registerSpace = nscUav.set;
                        sbb.stage         = stage;
                        sbb.buffer        = uavSlot->gpuBuffer ? uavSlot->gpuBuffer->GetNativeHandle() : uavSlot->gpuBufferHandle;
                        sbb.elementCount  = uavSlot->gpuBuffer ? uavSlot->gpuBuffer->GetElementCount() : uavSlot->gpuBufferElementCount;
                        sbb.stride        = uavSlot->gpuBuffer ? uavSlot->gpuBuffer->GetStride()       : uavSlot->gpuBufferStride;
                        sbb.byteSize      = uavSlot->gpuBuffer ? uavSlot->gpuBuffer->GetByteSize()     : uavSlot->gpuBufferByteSize;
                        sbb.isUAV         = true;
                        sbb.uavAccess     = uavSlot->uavAccess;
                        outStructuredBuffers.push_back(std::move(sbb));
                    }
                }
            }

            // Reject simultaneous SRV + UAV binding of the same GPUBuffer across stages
            for (const auto& uavSb : outStructuredBuffers)
            {
                if (!uavSb.isUAV || uavSb.buffer == 0) continue;
                for (const auto& srvSb : outStructuredBuffers)
                {
                    if (!srvSb.isUAV && srvSb.buffer == uavSb.buffer)
                    {
                        if (outDiagnostic) *outDiagnostic = "Simultaneous SRV + UAV binding on GPUBuffer.";
                        rollbackStageMat();
                        return false;
                    }
                }
            }
        }

        return true;
    }
}
