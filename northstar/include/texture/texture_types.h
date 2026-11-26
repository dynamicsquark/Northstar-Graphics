#pragma once

#include "../shader/shader_types.h"
#include "../types/northstar_types.h"

namespace Northstar
{
    // ==================== TEXTURE FORMAT ====================
    // Engine-only pixel formats (NSL does not encode format).
    enum class TextureFormat : UINT16
    {
        UNKNOWN         = 0,

        R8_UNORM        = 1,
        RG8_UNORM       = 2,
        RGBA8_UNORM     = 3,
        RGBA8_SRGB      = 4,

        R16_FLOAT       = 10,
        RG16_FLOAT      = 11,
        RGBA16_FLOAT    = 12,

        R32_FLOAT       = 20,
        RG32_FLOAT      = 21,
        RGBA32_FLOAT    = 22,
        R32_UINT        = 23,
        RG32_UINT       = 24,
        RGBA32_UINT     = 25,

        D16_UNORM       = 30,
        D24_UNORM_S8    = 31,
        D32_FLOAT       = 32,
        D32_FLOAT_S8    = 33,

        BC1_UNORM       = 40,
        BC3_UNORM       = 41,
        BC5_UNORM       = 42,
        BC7_UNORM       = 43,
        BC7_SRGB        = 44
    };

    enum class TextureDataSource : UINT8
    {
        MEMORY         = 0,
        EMPTY          = 1
    };

    // Dimension / view kind. Values are engine-local; convert to NSL via helpers.
    enum class TextureType : UINT8
    {
        TEXTURE_2D          = 0,
        TEXTURE_CUBE        = 1,
        TEXTURE_3D          = 2, // engine-only (no NSL DataTypeID yet)
        TEXTURE_2D_ARRAY    = 3,
        TEXTURE_CUBE_ARRAY  = 4
    };

    enum class TextureUsage : UINT8
    {
        SAMPLED         = 0, // Texture2D / Cube / arrays
        RENDER_TARGET   = 1,
        DEPTH_STENCIL   = 2,
        STORAGE         = 3  // RWTexture2D in NSL
    };

    // ---- NSL bridges (include-level) ----

    inline DataTypeID TextureTypeToDataTypeID(TextureType type, TextureUsage usage = TextureUsage::SAMPLED)
    {
        if (usage == TextureUsage::STORAGE)
        {
            if (type == TextureType::TEXTURE_2D)
                return DataTypeID::RW_TEXTURE_2D;
            return DataTypeID::UNKNOWN;
        }

        switch (type)
        {
            case TextureType::TEXTURE_2D:         return DataTypeID::TEXTURE_2D;
            case TextureType::TEXTURE_CUBE:       return DataTypeID::TEXTURE_CUBE;
            case TextureType::TEXTURE_2D_ARRAY:   return DataTypeID::TEXTURE_2D_ARRAY;
            case TextureType::TEXTURE_CUBE_ARRAY: return DataTypeID::TEXTURE_CUBE_ARRAY;
            case TextureType::TEXTURE_3D:         return DataTypeID::UNKNOWN;
            default:                              return DataTypeID::UNKNOWN;
        }
    }

    inline ResourceKind TextureTypeToResourceKind(TextureType type, TextureUsage usage = TextureUsage::SAMPLED)
    {
        if (usage == TextureUsage::STORAGE)
        {
            if (type == TextureType::TEXTURE_2D)
                return ResourceKind::RW_TEXTURE_2D;
            return ResourceKind::UNKNOWN;
        }

        switch (type)
        {
            case TextureType::TEXTURE_2D:         return ResourceKind::TEXTURE_2D;
            case TextureType::TEXTURE_CUBE:       return ResourceKind::TEXTURE_CUBE;
            case TextureType::TEXTURE_2D_ARRAY:   return ResourceKind::TEXTURE_2D_ARRAY;
            case TextureType::TEXTURE_CUBE_ARRAY: return ResourceKind::TEXTURE_CUBE_ARRAY;
            case TextureType::TEXTURE_3D:         return ResourceKind::UNKNOWN;
            default:                              return ResourceKind::UNKNOWN;
        }
    }

    inline BOOL TextureTypeFromDataTypeID(DataTypeID id, TextureType& outType, TextureUsage& outUsage)
    {
        outUsage = TextureUsage::SAMPLED;
        switch (id)
        {
            case DataTypeID::TEXTURE_2D:
                outType = TextureType::TEXTURE_2D; return true;
            case DataTypeID::TEXTURE_CUBE:
                outType = TextureType::TEXTURE_CUBE; return true;
            case DataTypeID::TEXTURE_2D_ARRAY:
                outType = TextureType::TEXTURE_2D_ARRAY; return true;
            case DataTypeID::TEXTURE_CUBE_ARRAY:
                outType = TextureType::TEXTURE_CUBE_ARRAY; return true;
            case DataTypeID::RW_TEXTURE_2D:
                outType = TextureType::TEXTURE_2D;
                outUsage = TextureUsage::STORAGE;
                return true;
            default:
                return false;
        }
    }

    // ==================== TEXTURE FORMAT & SUBRESOURCE LAYOUT HELPERS ====================

    inline constexpr BOOL IsTextureFormatDepth(TextureFormat fmt) noexcept
    {
        return fmt == TextureFormat::D16_UNORM ||
               fmt == TextureFormat::D24_UNORM_S8 ||
               fmt == TextureFormat::D32_FLOAT ||
               fmt == TextureFormat::D32_FLOAT_S8;
    }

    inline constexpr BOOL IsTextureFormatDepthStencil(TextureFormat fmt) noexcept
    {
        return fmt == TextureFormat::D24_UNORM_S8 ||
               fmt == TextureFormat::D32_FLOAT_S8;
    }

    inline constexpr BOOL IsTextureFormatBlockCompressed(TextureFormat fmt) noexcept
    {
        return fmt == TextureFormat::BC1_UNORM ||
               fmt == TextureFormat::BC3_UNORM ||
               fmt == TextureFormat::BC5_UNORM ||
               fmt == TextureFormat::BC7_UNORM ||
               fmt == TextureFormat::BC7_SRGB;
    }

    inline constexpr BOOL IsTextureFormatSRGB(TextureFormat fmt) noexcept
    {
        return fmt == TextureFormat::RGBA8_SRGB ||
               fmt == TextureFormat::BC7_SRGB;
    }

    inline constexpr BOOL IsTextureFormatInteger(TextureFormat fmt) noexcept
    {
        return fmt == TextureFormat::R32_UINT ||
               fmt == TextureFormat::RG32_UINT ||
               fmt == TextureFormat::RGBA32_UINT;
    }

    inline constexpr BOOL IsTextureFormatCPUFilterable(TextureFormat fmt) noexcept
    {
        switch (fmt)
        {
            case TextureFormat::R8_UNORM:
            case TextureFormat::RG8_UNORM:
            case TextureFormat::RGBA8_UNORM:
            case TextureFormat::RGBA8_SRGB:
            case TextureFormat::R16_FLOAT:
            case TextureFormat::RG16_FLOAT:
            case TextureFormat::RGBA16_FLOAT:
            case TextureFormat::R32_FLOAT:
            case TextureFormat::RG32_FLOAT:
            case TextureFormat::RGBA32_FLOAT:
                return true;
            default:
                return false;
        }
    }

    // Returns bytes per pixel for uncompressed formats, or bytes per 4x4 block for BCn formats.
    inline constexpr UINT32 GetTextureFormatBytesPerUnit(TextureFormat fmt) noexcept
    {
        switch (fmt)
        {
            case TextureFormat::R8_UNORM:        return 1u;
            case TextureFormat::RG8_UNORM:       return 2u;
            case TextureFormat::RGBA8_UNORM:     return 4u;
            case TextureFormat::RGBA8_SRGB:      return 4u;
            case TextureFormat::R16_FLOAT:       return 2u;
            case TextureFormat::RG16_FLOAT:      return 4u;
            case TextureFormat::RGBA16_FLOAT:    return 8u;
            case TextureFormat::R32_FLOAT:       return 4u;
            case TextureFormat::RG32_FLOAT:      return 8u;
            case TextureFormat::RGBA32_FLOAT:    return 16u;
            case TextureFormat::R32_UINT:        return 4u;
            case TextureFormat::RG32_UINT:       return 8u;
            case TextureFormat::RGBA32_UINT:     return 16u;
            case TextureFormat::D16_UNORM:       return 2u;
            case TextureFormat::D24_UNORM_S8:    return 4u;
            case TextureFormat::D32_FLOAT:       return 4u;
            case TextureFormat::D32_FLOAT_S8:    return 8u;
            case TextureFormat::BC1_UNORM:       return 8u;
            case TextureFormat::BC3_UNORM:       return 16u;
            case TextureFormat::BC5_UNORM:       return 16u;
            case TextureFormat::BC7_UNORM:       return 16u;
            case TextureFormat::BC7_SRGB:        return 16u;
            default:                             return 0u;
        }
    }

    inline constexpr UINT32 ComputeTextureArraySliceCount(TextureType type, UINT32 arrayLayers) noexcept
    {
        switch (type)
        {
            case TextureType::TEXTURE_CUBE:       return 6u;
            case TextureType::TEXTURE_CUBE_ARRAY: return 6u * (arrayLayers > 0u ? arrayLayers : 1u);
            case TextureType::TEXTURE_2D_ARRAY:   return (arrayLayers > 0u ? arrayLayers : 1u);
            default:                              return 1u;
        }
    }

    inline constexpr UINT32 ComputeTextureMaxMipLevels(
        UINT32      width,
        UINT32      height,
        UINT32      depth = 1u,
        TextureType type  = TextureType::TEXTURE_2D) noexcept
    {
        if (width == 0u || height == 0u)
            return 0u;
        UINT32 maxDim = (width > height) ? width : height;
        if (type == TextureType::TEXTURE_3D && depth > maxDim)
            maxDim = depth;
        UINT32 mips = 1u;
        while (maxDim > 1u)
        {
            maxDim >>= 1u;
            ++mips;
        }
        return mips;
    }

    inline constexpr BOOL ComputeTextureSubresourceTightLayout(
        UINT32        baseWidth,
        UINT32        baseHeight,
        UINT32        baseDepth,
        TextureType   type,
        TextureFormat format,
        UINT32        mipSlice,
        SIZE_T&       outRowBytes,
        UINT32&       outNumRows,
        UINT32&       outDepthSlices,
        SIZE_T&       outSubresourceBytes) noexcept
    {
        outRowBytes         = 0u;
        outNumRows          = 0u;
        outDepthSlices      = 0u;
        outSubresourceBytes = 0u;

        const UINT32 unitBytes = GetTextureFormatBytesPerUnit(format);
        if (baseWidth == 0u || baseHeight == 0u || unitBytes == 0u)
            return false;
        if (type == TextureType::TEXTURE_3D && baseDepth == 0u)
            return false;

        const UINT32 mipW = (baseWidth >> mipSlice) > 0u ? (baseWidth >> mipSlice) : 1u;
        const UINT32 mipH = (baseHeight >> mipSlice) > 0u ? (baseHeight >> mipSlice) : 1u;
        const UINT32 mipD = (type == TextureType::TEXTURE_3D)
            ? (((baseDepth >> mipSlice) > 0u) ? (baseDepth >> mipSlice) : 1u)
            : 1u;

        if (IsTextureFormatBlockCompressed(format))
        {
            const UINT32 blocksX = ((mipW + 3u) / 4u) > 0u ? ((mipW + 3u) / 4u) : 1u;
            const UINT32 blocksY = ((mipH + 3u) / 4u) > 0u ? ((mipH + 3u) / 4u) : 1u;
            outRowBytes = static_cast<SIZE_T>(blocksX) * static_cast<SIZE_T>(unitBytes);
            outNumRows  = blocksY;
        }
        else
        {
            outRowBytes = static_cast<SIZE_T>(mipW) * static_cast<SIZE_T>(unitBytes);
            outNumRows  = mipH;
        }

        outDepthSlices      = mipD;
        outSubresourceBytes = outRowBytes * static_cast<SIZE_T>(outNumRows) * static_cast<SIZE_T>(outDepthSlices);
        return outSubresourceBytes > 0u;
    }

    inline constexpr SIZE_T ComputeTextureMip0AllSlicesTightBytes(
        UINT32        width,
        UINT32        height,
        UINT32        depth,
        UINT32        arrayLayers,
        TextureType   type,
        TextureFormat format) noexcept
    {
        SIZE_T rowBytes = 0u, subBytes = 0u;
        UINT32 numRows = 0u, depthSlices = 0u;
        if (!ComputeTextureSubresourceTightLayout(width, height, depth, type, format, 0u, rowBytes, numRows, depthSlices, subBytes))
            return 0u;
        return subBytes * static_cast<SIZE_T>(ComputeTextureArraySliceCount(type, arrayLayers));
    }

    inline constexpr SIZE_T ComputeTextureFullChainTightBytes(
        UINT32        width,
        UINT32        height,
        UINT32        depth,
        UINT32        mipLevels,
        UINT32        arrayLayers,
        TextureType   type,
        TextureFormat format) noexcept
    {
        if (mipLevels == 0u)
            return 0u;
        SIZE_T perSliceBytes = 0u;
        for (UINT32 m = 0u; m < mipLevels; ++m)
        {
            SIZE_T rowBytes = 0u, subBytes = 0u;
            UINT32 numRows = 0u, depthSlices = 0u;
            if (!ComputeTextureSubresourceTightLayout(width, height, depth, type, format, m, rowBytes, numRows, depthSlices, subBytes))
                return 0u;
            perSliceBytes += subBytes;
        }
        return perSliceBytes * static_cast<SIZE_T>(ComputeTextureArraySliceCount(type, arrayLayers));
    }
}
