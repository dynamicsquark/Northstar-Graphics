#pragma once

#include <string>
#include <cstdint>
#include <unordered_map>

// =====================================================================
//  NSL Common Types
//
//  Single source of truth for NSL / Northstar shared enums and type info.
//  Used by: NSLCompiler, NSCReader, and Northstar engine headers.
//
//  Related:
//    nsl_shader_data.h  — shared .nsc reflection payload (ShaderData)
//    nsl_engine_data.h  — engine PROPERTY Get() table
//    nsl_stage_io.h     — INPUT_LAYOUT Get() / OUTPUT_LAYOUT Add() table
//  Do NOT include compiler-internal headers here.
// =====================================================================

#ifdef DOMAIN
#undef DOMAIN
#endif

namespace NSL
{
    // ========================= SHADER STAGES =========================

    enum class ShaderStage : uint8_t
    {
        NONE     = 0,
        VERTEX   = 1,
        PIXEL    = 2,
        COMPUTE  = 3,
        GEOMETRY = 4,
        HULL     = 5,
        DOMAIN   = 6,
    };

    inline const char* ShaderStageToString(ShaderStage stage)
    {
        switch (stage)
        {
            case ShaderStage::VERTEX:   return "vertex";
            case ShaderStage::PIXEL:    return "pixel";
            case ShaderStage::COMPUTE:  return "compute";
            case ShaderStage::GEOMETRY: return "geometry";
            case ShaderStage::HULL:     return "hull";
            case ShaderStage::DOMAIN:   return "domain";
            default:                    return "none";
        }
    }

    inline ShaderStage ShaderStageFromString(const std::string& s)
    {
        if (s == "VERTEX")   return ShaderStage::VERTEX;
        if (s == "PIXEL")    return ShaderStage::PIXEL;
        if (s == "COMPUTE")  return ShaderStage::COMPUTE;
        if (s == "GEOMETRY") return ShaderStage::GEOMETRY;
        if (s == "HULL")     return ShaderStage::HULL;
        if (s == "DOMAIN")   return ShaderStage::DOMAIN;
        return ShaderStage::NONE;
    }

    // ========================= TARGET API ============================

    enum class TargetAPI : uint8_t
    {
        HLSL11  = 0,
        GLSL    = 1,    // future
        MSL     = 2,    // future
        SPIRV   = 3,    // future
        HLSL12  = 4,
        UNKNOWN = 255,
    };

    inline const char* TargetAPIToString(TargetAPI api)
    {
        switch (api)
        {
            case TargetAPI::HLSL11: return "hlsl11";
            case TargetAPI::HLSL12: return "hlsl12";
            case TargetAPI::GLSL:   return "glsl";
            case TargetAPI::MSL:    return "msl";
            case TargetAPI::SPIRV:  return "spirv";
            default:                return "unknown";
        }
    }

    // ========================= DATA TYPE IDS =========================

    enum class DataTypeID : uint8_t
    {
        UNKNOWN = 0,

        // Scalars
        FLOAT   = 1,
        INT     = 5,
        UINT    = 9,
        BOOL    = 15,
        DOUBLE  = 16,
        UINT64  = 19,

        // Float vectors
        VEC2    = 2,
        VEC3    = 3,
        VEC4    = 4,

        // Int vectors
        INT2    = 6,
        INT3    = 7,
        INT4    = 8,

        // Uint vectors
        UINT2   = 10,
        UINT3   = 11,
        UINT4   = 12,

        // Matrices
        MAT3    = 13,
        MAT4    = 14,

        // High-level (raw size matches Vec4)
        COLOR   = 17,
        QUAT    = 18,

        // Resource types
        TEXTURE_2D               = 20,
        TEXTURE_CUBE             = 21,
        TEXTURE_2D_ARRAY         = 22,
        TEXTURE_CUBE_ARRAY       = 23,
        RW_TEXTURE_2D            = 24,
        SAMPLER_STATE            = 25,
        SAMPLER_COMPARISON_STATE = 26,
        STRUCTURED_BUFFER        = 27,
        RW_STRUCTURED_BUFFER     = 28,
    };

    // ========================= RESOURCE KIND =========================

    enum class ResourceKind : uint8_t
    {
        UNKNOWN             = 0,
        BUFFER              = 1,
        TEXTURE_2D          = 2,
        TEXTURE_CUBE        = 3,
        SAMPLER_STATE       = 4,
        STRUCTURED_BUFFER   = 5,
        RW_STRUCTURED_BUFFER = 6,
        RW_TEXTURE_2D       = 7,
        TEXTURE_2D_ARRAY    = 8,
        TEXTURE_CUBE_ARRAY  = 9,
    };

    inline ResourceKind ResourceKindFromString(const std::string& s)
    {
        if (s == "Buffer")              return ResourceKind::BUFFER;
        if (s == "Texture2D")           return ResourceKind::TEXTURE_2D;
        if (s == "TextureCube")         return ResourceKind::TEXTURE_CUBE;
        if (s == "SamplerState" || s == "SamplerComparisonState") return ResourceKind::SAMPLER_STATE;
        if (s == "StructuredBuffer")    return ResourceKind::STRUCTURED_BUFFER;
        if (s == "RWStructuredBuffer")  return ResourceKind::RW_STRUCTURED_BUFFER;
        if (s == "RWTexture2D")         return ResourceKind::RW_TEXTURE_2D;
        if (s == "Texture2DArray")      return ResourceKind::TEXTURE_2D_ARRAY;
        if (s == "TextureCubeArray")    return ResourceKind::TEXTURE_CUBE_ARRAY;
        return ResourceKind::UNKNOWN;
    }

    inline const char* ResourceKindToString(ResourceKind kind)
    {
        switch (kind)
        {
            case ResourceKind::BUFFER:               return "Buffer";
            case ResourceKind::TEXTURE_2D:           return "Texture2D";
            case ResourceKind::TEXTURE_CUBE:         return "TextureCube";
            case ResourceKind::SAMPLER_STATE:        return "SamplerState";
            case ResourceKind::STRUCTURED_BUFFER:    return "StructuredBuffer";
            case ResourceKind::RW_STRUCTURED_BUFFER: return "RWStructuredBuffer";
            case ResourceKind::RW_TEXTURE_2D:        return "RWTexture2D";
            case ResourceKind::TEXTURE_2D_ARRAY:     return "Texture2DArray";
            case ResourceKind::TEXTURE_CUBE_ARRAY:   return "TextureCubeArray";
            default:                                  return "Unknown";
        }
    }

    // ========================= PROPERTY ENTRY KIND ===================

    enum class PropertyEntryKind : uint8_t
    {
        ADD = 0,    // User-defined, has default value
        GET = 1,    // Engine-provided, read-only
    };

    // ========================= PARAMETER QUALIFIERS ==================

    enum class ParamQualifier : uint8_t
    {
        NONE  = 0,
        IN    = 1,
        OUT   = 2,
        INOUT = 3,
    };

    inline const char* ParamQualifierToString(ParamQualifier q)
    {
        switch (q)
        {
            case ParamQualifier::IN:    return "in";
            case ParamQualifier::OUT:   return "out";
            case ParamQualifier::INOUT: return "inout";
            default:                    return "";
        }
    }

    // ========================= DATA TYPE INFO ========================
    // Size only — alignment and target type names are backend-specific
    // (IAligner / HLSLBackend / future GLSL|MSL backends).

    struct DataTypeInfo
    {
        DataTypeID  id;
        const char* nslName;    // "Vec3"
        uint32_t    size;       // Byte size (0 for resource types)
    };

    // Shared type table. Sizes are RAW / tight packing sizes.
    // Constant-buffer Mat3 row-padding (36 → 48) is applied only by
    // the active backend aligner — never store padded sizes here.
    inline const DataTypeInfo* DataTypeTable(size_t& count)
    {
        static const DataTypeInfo kTypes[] =
        {
            // Scalars
            { DataTypeID::FLOAT,  "Float",  4  },
            { DataTypeID::INT,    "Int",    4  },
            { DataTypeID::UINT,   "UInt",   4  },
            { DataTypeID::BOOL,   "Bool",   4  },
            { DataTypeID::DOUBLE, "Double", 8  },
            { DataTypeID::UINT64, "UInt64", 8  },

            // Float vectors
            { DataTypeID::VEC2,   "Vec2",   8  },
            { DataTypeID::VEC3,   "Vec3",   12 },
            { DataTypeID::VEC4,   "Vec4",   16 },

            // Int vectors
            { DataTypeID::INT2,   "Int2",   8  },
            { DataTypeID::INT3,   "Int3",   12 },
            { DataTypeID::INT4,   "Int4",   16 },

            // Uint vectors
            { DataTypeID::UINT2,  "UInt2",  8  },
            { DataTypeID::UINT3,  "UInt3",  12 },
            { DataTypeID::UINT4,  "UInt4",  16 },

            // Matrices (raw / StructuredBuffer sizes)
            { DataTypeID::MAT3,   "Mat3",   36 },
            { DataTypeID::MAT4,   "Mat4",   64 },

            // High-level (raw size matches Vec4)
            { DataTypeID::COLOR,  "Color",  16 },
            { DataTypeID::QUAT,   "Quat",   16 },
        };
        count = sizeof(kTypes) / sizeof(kTypes[0]);
        return kTypes;
    }

    // Returns nullptr if not found.
    inline const DataTypeInfo* FindDataType(const std::string& nslName)
    {
        size_t count = 0;
        const DataTypeInfo* kTypes = DataTypeTable(count);
        for (size_t i = 0; i < count; ++i)
            if (nslName == kTypes[i].nslName) return &kTypes[i];
        return nullptr;
    }

    // Returns nullptr if not found.
    inline const DataTypeInfo* FindDataTypeByID(DataTypeID id)
    {
        size_t count = 0;
        const DataTypeInfo* kTypes = DataTypeTable(count);
        for (size_t i = 0; i < count; ++i)
            if (kTypes[i].id == id) return &kTypes[i];
        return nullptr;
    }

    inline DataTypeID GetDataTypeID(const std::string& nslName)
    {
        const DataTypeInfo* info = FindDataType(nslName);
        return info ? info->id : DataTypeID::UNKNOWN;
    }

    inline DataTypeID GetResourceTypeID(const std::string& typeName)
    {
        if (typeName == "Texture2D")           return DataTypeID::TEXTURE_2D;
        if (typeName == "TextureCube")         return DataTypeID::TEXTURE_CUBE;
        if (typeName == "Texture2DArray")      return DataTypeID::TEXTURE_2D_ARRAY;
        if (typeName == "TextureCubeArray")    return DataTypeID::TEXTURE_CUBE_ARRAY;
        if (typeName == "RWTexture2D")         return DataTypeID::RW_TEXTURE_2D;
        if (typeName == "SamplerState")        return DataTypeID::SAMPLER_STATE;
        if (typeName == "SamplerComparisonState")  return DataTypeID::SAMPLER_COMPARISON_STATE;
        if (typeName == "StructuredBuffer")    return DataTypeID::STRUCTURED_BUFFER;
        if (typeName == "RWStructuredBuffer")  return DataTypeID::RW_STRUCTURED_BUFFER;
        return DataTypeID::UNKNOWN;
    }

} // namespace NSL
