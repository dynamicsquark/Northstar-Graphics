#pragma once

#include "nsl_common_types.h"
#include <cstdint>
#include <string>
#include <string_view>

// =====================================================================
//  NSL Engine Data
//
//  Central definition of all engine-provided properties.
//  Included by BOTH the NSLCompiler (for validation) and the engine
//  renderer (for runtime data upload).
//
//  EngineProperty::size is always the RAW / tight size (same as
//  FindDataType). Constant-buffer Mat3 packing (36 → 48) is applied
//  only by the active backend aligner (HLSLAligner today) — never
//  store padded buffer sizes here.
//
//  Model:
//    Each engine Get() name binds to exactly ONE ResourceKind.
//    Compiler rejects Get(NAME) inside any other PROPERTY group kind.
//      Get(CAMERA_POSITION)     → only inside Buffer
//      Get(GLOBAL_LIGHT_COLOR)  → only inside StructuredBuffer
//      Get(NVS_SHADOW_MAP)      → only inside Texture2DArray
//    Users still compose packages by mixing Add() + Get() inside that
//    single allowed kind.
//
//  To add a new engine property:
//    1. Add a constexpr entry to ENGINE_PROPERTIES[] with its ResourceKind
//    Both compiler and runtime will pick it up automatically.
// =====================================================================

namespace NSL
{
    // ========================= ENGINE PROPERTY =======================

    struct EngineProperty
    {
        const char*  name;         // NSL name:  "CAMERA_POSITION"
        DataTypeID   typeID;       // DataTypeID::VEC3
        uint32_t     size;         // Raw byte size (0 for resource types)
        ResourceKind resourceKind; // Exact PROPERTY group kind that may Get() this
    };

    // ========================= ENGINE PROPERTIES TABLE ===============
    //  All engine-provided properties. Order is not significant.
    //  Only flat atomic values — no nested / package engine properties.

    inline constexpr EngineProperty ENGINE_PROPERTIES[] =
    {
        // ==================== Camera (Buffer) ====================
        { "CAMERA_POSITION",       DataTypeID::VEC3,  12, ResourceKind::BUFFER },
        { "CAMERA_DIRECTION",      DataTypeID::VEC3,  12, ResourceKind::BUFFER },
        { "VIEW_MATRIX",           DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "PROJECTION_MATRIX",     DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "VP_MATRIX",             DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "INV_VIEW_MATRIX",       DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "INV_PROJECTION_MATRIX", DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "NEAR_PLANE",            DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "FAR_PLANE",             DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "CAMERA_FOV",            DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "CAMERA_ASPECT",         DataTypeID::FLOAT,  4, ResourceKind::BUFFER },

        // ==================== Transform (Buffer) ====================
        { "WORLD_MATRIX",          DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "NORMAL_MATRIX",         DataTypeID::MAT3,  36, ResourceKind::BUFFER },
        { "MVP_MATRIX",            DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "INV_WORLD_MATRIX",      DataTypeID::MAT4,  64, ResourceKind::BUFFER },
        { "GPU_SCENE_OBJECT_SLOT", DataTypeID::UINT,   4, ResourceKind::BUFFER },
        { "GEOMETRY_MODE",         DataTypeID::UINT,   4, ResourceKind::BUFFER },

        // ==================== Ambient Lighting (Buffer) ====================
        { "AMBIENT_COLOR",         DataTypeID::VEC3,  12, ResourceKind::BUFFER },
        { "AMBIENT_INTENSITY",     DataTypeID::FLOAT,  4, ResourceKind::BUFFER },

        // ==================== Counters (Buffer) ====================
        { "LIGHT_COUNT",           DataTypeID::UINT,   4, ResourceKind::BUFFER },

        // ==================== Lighting — Array Element (StructuredBuffer) ====================
        { "GLOBAL_LIGHT_POSITION",          DataTypeID::VEC3,  12, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_COLOR",             DataTypeID::VEC3,  12, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_DIRECTION",         DataTypeID::VEC3,  12, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_INTENSITY",         DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_RANGE",             DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_TYPE",              DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_INNER_CONE",        DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_OUTER_CONE",        DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_ATTENUATION",       DataTypeID::VEC3,  12, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_VP_MATRIX",         DataTypeID::MAT4,  64, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_CASTS_SHADOW",      DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_SHADOW_VIEW_BASE",  DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_SHADOW_VIEW_COUNT", DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_NVS_LIGHT_ID",      DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_SHADOW_BIAS",       DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_NORMAL_BIAS",       DataTypeID::FLOAT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "GLOBAL_LIGHT_SHADOW_SOFTNESS",   DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== Time / Frame / Screen (Buffer) ====================
        { "TIME",                  DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "ENGINE_TIME",           DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "DELTA_TIME",            DataTypeID::FLOAT,  4, ResourceKind::BUFFER },
        { "FRAME_INDEX",           DataTypeID::UINT,   4, ResourceKind::BUFFER },
        { "SCREEN_SIZE",           DataTypeID::VEC2,   8, ResourceKind::BUFFER },

        // ==================== Skinning & Instance Metadata ====================
        { "BONE_PALETTE_OFFSET",   DataTypeID::UINT,   4, ResourceKind::BUFFER },
        { "BONE_COUNT",            DataTypeID::UINT,   4, ResourceKind::BUFFER },
        { "SKINNING_REVISION",     DataTypeID::UINT,   4, ResourceKind::BUFFER },

        { "GLOBAL_BONE_MATRIX",    DataTypeID::MAT4,  64, ResourceKind::STRUCTURED_BUFFER },
        { "BONE_MATRIX",           DataTypeID::MAT4,  64, ResourceKind::STRUCTURED_BUFFER },
        { "INSTANCE_WORLD_MATRIX", DataTypeID::MAT4,  64, ResourceKind::STRUCTURED_BUFFER },
        { "INSTANCE_ID_VALUE",     DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "INSTANCE_CUSTOM_DATA",  DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "INSTANCE_BONE_OFFSET",  DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "INSTANCE_BONE_COUNT",   DataTypeID::UINT,   4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVS Shadow Rasterization Page Context (Buffer) ====================
        { "NVS_PHYSICAL_PAGE_INDEX",  DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVS_VIRTUAL_PAGE_ADDRESS", DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVS_CLIPMAP_LEVEL",        DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVS_DEPTH_BIAS_NDC",       DataTypeID::FLOAT, 4, ResourceKind::BUFFER },
        { "NVS_NORMAL_BIAS_WORLD",    DataTypeID::FLOAT, 4, ResourceKind::BUFFER },
        { "NVS_RECEIVER_MIN_Z",       DataTypeID::FLOAT, 4, ResourceKind::BUFFER },
        { "NVS_RECEIVER_MAX_Z",       DataTypeID::FLOAT, 4, ResourceKind::BUFFER },

        // ==================== NVS Engine-Owned Sampling Texture & Samplers ====================
        { "NVS_SHADOW_MAP",           DataTypeID::TEXTURE_2D_ARRAY,         0, ResourceKind::TEXTURE_2D_ARRAY },
        { "NVS_COMPARISON_SAMPLER",   DataTypeID::SAMPLER_COMPARISON_STATE, 0, ResourceKind::SAMPLER_STATE },
        { "NVS_DEPTH_SAMPLER",        DataTypeID::SAMPLER_STATE,            0, ResourceKind::SAMPLER_STATE },

        // ==================== NVS Sampling Global Metadata (Buffer) ====================
        { "NVS_PAGES_PER_AXIS",           DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_PHYSICAL_POOL_CAPACITY",   DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_PAGE_TEXEL_SIZE",          DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_VIEW_TABLE_COUNT",         DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_TOTAL_VIRTUAL_PAGES",      DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_DEPTH_MODE",               DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SAMPLING_AVAILABLE",       DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SNAPSHOT_FRAME_INDEX",     DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SPARSE_ROOT_CAPACITY",     DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SPARSE_DIR_STRIDE",        DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SPARSE_DIR_CAPACITY",      DataTypeID::UINT, 4,  ResourceKind::BUFFER },
        { "NVS_SCREEN_PIXEL_WORLD_SCALE", DataTypeID::VEC4, 16, ResourceKind::BUFFER },

        // ==================== NVS Sparse Page Hierarchy (StructuredBuffer, 4B) ====================
        { "NVS_SPARSE_NODE_VALUE",        DataTypeID::UINT, 4,  ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVS Virtual Page Table (StructuredBuffer, 16B) ====================
        { "NVS_VPAGE_STATE_AND_FLAGS",    DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VPAGE_PHYSICAL_INDEX",     DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VPAGE_VIRTUAL_ADDRESS",    DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VPAGE_LAST_TOUCHED_FRAME", DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVS Physical Page Metadata Table (StructuredBuffer, 32B) ====================
        { "NVS_PPAGE_STATE_AND_FLAGS",     DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_OWNER_FLAT_INDEX",    DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_OWNER_ADDRESS",       DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_GENERATION",          DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_LAST_USED_FRAME",     DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_ALLOCATED_FRAME",     DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_MIN_EVICTABLE_FRAME", DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_PPAGE_META_AUX",            DataTypeID::UINT, 4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVS Light View / Clipmap Projection Table (StructuredBuffer, 256B) ====================
        { "NVS_VIEW_LIGHT_ID",              DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_SUB_INDEX",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_PAGES_PER_AXIS",        DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_FLAGS",                 DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_SNAPPED_ORIGIN_X",      DataTypeID::INT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_SNAPPED_ORIGIN_Y",      DataTypeID::INT,   4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_WORLD_EXTENT",          DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_PAGE_WORLD_SIZE",       DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_TEXEL_WORLD_SIZE",      DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_RECEIVER_MIN_Z",        DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_RECEIVER_MAX_Z",        DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_CASTER_MIN_Z",          DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_CASTER_MAX_Z",          DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_DEPTH_BIAS_NDC",        DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_NORMAL_BIAS_WORLD",     DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_REF_LIGHT_Z",           DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_RIGHT_AND_CENTER_X",    DataTypeID::VEC4, 16, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_UP_AND_CENTER_Y",       DataTypeID::VEC4, 16, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_DIR_AND_EXTRUSION",     DataTypeID::VEC4, 16, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_TRANSITION_AND_PARAMS", DataTypeID::VEC4, 16, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_WORLD_TO_LIGHT",        DataTypeID::MAT4, 64, ResourceKind::STRUCTURED_BUFFER },
        { "NVS_VIEW_LIGHT_TO_WORLD",        DataTypeID::MAT4, 64, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVG Virtual Geometry Draw & Telemetry Metadata (Buffer) ====================
        { "NVG_DRAW_BUCKET_ID",             DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVG_VISIBLE_CLUSTER_COUNT",      DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVG_VISIBLE_CLUSTER_OFFSET",     DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVG_DEBUG_MODE",                 DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVG_MAX_SCREEN_ERROR_PX",        DataTypeID::FLOAT, 4, ResourceKind::BUFFER },
        { "NVG_PHYSICAL_POOL_CAPACITY",     DataTypeID::UINT,  4, ResourceKind::BUFFER },
        { "NVG_PAGE_WORD_COUNT",            DataTypeID::UINT,  4, ResourceKind::BUFFER },

        // ==================== NVG Physical Geometry Page Pool (StructuredBuffer, 4B) ====================
        { "NVG_PAGE_POOL_WORD",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVG Draw Bucket Ranges (StructuredBuffer, 16B) ====================
        { "NVG_BUCKET_FIRST_CLUSTER",       DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_BUCKET_CLUSTER_COUNT",       DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_BUCKET_MAX_TRIANGLES",       DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_BUCKET_FLAGS",               DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVG Visible Cluster Records (StructuredBuffer, 48B) ====================
        { "NVG_VIS_OBJECT_SLOT",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_CLUSTER_ID",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_PHYSICAL_PAGE_INDEX",    DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_CLUSTER_WORD_OFFSET",    DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_VIRTUAL_PAGE_ID",        DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_LOD_AND_FLAGS",          DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_GEOMETRIC_ERROR",        DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_MATERIAL_ID",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_BONE_OFFSET",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_BONE_COUNT",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_DRAW_BUCKET_INDEX",      DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_VIS_VIEW_DEPTH_KEY",         DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVG GPUScene Object Records (StructuredBuffer, 160B Zero-Copy) ====================
        { "NVG_OBJ_SLOT_INDEX",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_GENERATION",             DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_FLAGS",                  DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_RENDER_FLAGS",           DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_MESH_ID",                DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_MATERIAL_ID",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_LOD_GROUP_ID",           DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_BATCH_GROUP_ID",         DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_VERTEX_COUNT",           DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_INDEX_COUNT",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_START_INDEX",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_BASE_VERTEX",            DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_AABB_MIN",               DataTypeID::VEC3, 12, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_SUBMESH_INDEX",          DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_AABB_MAX",               DataTypeID::VEC3, 12, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_LOD_MATERIAL_FLAGS",     DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_BOUNDING_SPHERE",        DataTypeID::VEC4, 16, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_OBJ_WORLD_MATRIX",           DataTypeID::MAT4, 64, ResourceKind::STRUCTURED_BUFFER },

        // ==================== NVG NVS Shadow Page Views (StructuredBuffer, 80B) ====================
        { "NVG_SHADOW_PAGE_VP_MATRIX",      DataTypeID::MAT4, 64, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_SHADOW_PAGE_SLICE",          DataTypeID::UINT,  4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_SHADOW_PAGE_DEPTH_BIAS",     DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_SHADOW_PAGE_NORMAL_BIAS",    DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
        { "NVG_SHADOW_PAGE_TEXEL_SIZE",     DataTypeID::FLOAT, 4, ResourceKind::STRUCTURED_BUFFER },
    };

    inline constexpr uint32_t ENGINE_PROPERTY_COUNT =
        static_cast<uint32_t>(sizeof(ENGINE_PROPERTIES) / sizeof(ENGINE_PROPERTIES[0]));

    // ========================= LOOKUP HELPERS ========================

    // Returns nullptr if not found.
    inline const EngineProperty* FindEngineProperty(const char* name)
    {
        for (uint32_t i = 0; i < ENGINE_PROPERTY_COUNT; ++i)
            if (std::string_view(ENGINE_PROPERTIES[i].name) == name)
                return &ENGINE_PROPERTIES[i];
        return nullptr;
    }

    inline const EngineProperty* FindEngineProperty(const std::string& name)
    {
        return FindEngineProperty(name.c_str());
    }

    // Returns true if name is a reserved engine property identifier.
    inline bool IsEnginePropertyName(const std::string& name)
    {
        return FindEngineProperty(name.c_str()) != nullptr;
    }

} // namespace NSL
