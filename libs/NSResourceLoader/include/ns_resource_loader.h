#pragma once

#include "types/northstar_types.h"
#include "texture/texture_desc.h"
#include "shader/shader_desc.h"

#include <vector>
#include <string>
#include <cstdint>

namespace Northstar::Tools
{
    // ==================== LOADED TEXTURE ====================
    // RAII container that owns raw pixel/block memory and produces a valid TextureDesc.
    // Rvalue AsDesc()/conversion overloads are deleted so temporary LoadedTexture instances
    // can never produce a dangling TextureDesc::initialData pointer across statements.
    struct LoadedTexture
    {
        std::vector<BYTE> pixels;
        UINT32            width       = 0;
        UINT32            height      = 0;
        UINT32            depth       = 1;
        UINT32            mipLevels   = 1;
        UINT32            arrayLayers = 1;
        UINT32            channels    = 0;
        TextureFormat     format      = TextureFormat::RGBA8_UNORM;
        TextureType       type        = TextureType::TEXTURE_2D;
        TextureUsage      usage       = TextureUsage::SAMPLED;
        BOOL              isValid     = false;

        TextureDesc AsDesc() const &
        {
            TextureDesc desc{};
            if (!isValid || pixels.empty())
            {
                desc.dataSource = TextureDataSource::EMPTY;
                return desc;
            }

            desc.dataSource      = TextureDataSource::MEMORY;
            desc.initialData     = pixels.data();
            desc.initialDataSize = pixels.size();
            desc.width           = width;
            desc.height          = height;
            desc.depth           = depth > 0u ? depth : 1u;
            desc.mipLevels       = mipLevels;
            desc.arrayLayers     = arrayLayers > 0u ? arrayLayers : 1u;
            desc.format          = format;
            desc.type            = type;
            desc.usage           = usage;
            return desc;
        }

        operator TextureDesc() const &
        {
            return AsDesc();
        }

        TextureDesc AsDesc() const && = delete;
        operator TextureDesc() const && = delete;
    };

    // ==================== LOADED SHADER ====================
    // RAII container that owns raw bytecode/.nsc binary and produces a valid ShaderDesc.
    struct LoadedShader
    {
        std::vector<BYTE> bytecode;
        std::string       name;
        BOOL              isValid = false;

        ShaderDesc AsDesc() const &
        {
            ShaderDesc desc{};
            if (!isValid || bytecode.empty())
            {
                desc.dataSource      = ShaderDataSource::MEMORY;
                desc.initialData     = nullptr;
                desc.initialDataSize = 0;
                desc.name.clear();
                return desc;
            }

            desc.dataSource      = ShaderDataSource::MEMORY;
            desc.initialData     = bytecode.data();
            desc.initialDataSize = bytecode.size();
            desc.name            = name;
            return desc;
        }

        operator ShaderDesc() const &
        {
            return AsDesc();
        }

        ShaderDesc AsDesc() const && = delete;
        operator ShaderDesc() const && = delete;
    };

    // ==================== RESOURCE LOADER ====================
    class ResourceLoader
    {
    public:
        // Texture loading: supports .dds (BC1/BC3/BC5/BC7/BC7_SRGB/RGBA8/RGBA16F/RGBA32F/R32F, multi-mip,
        // Cubemap, 2D Array, 3D Volume), .hdr (RGBA32_FLOAT), and LDR formats via stb_image (RGBA8_UNORM / RGBA8_SRGB).
        static LoadedTexture LoadTextureFromFile(
            const std::string& path,
            TextureUsage       usage        = TextureUsage::SAMPLED,
            BOOL               srgb         = false,
            BOOL               generateMips = false);
        static LoadedTexture LoadTextureFromFile(
            const std::wstring& path,
            TextureUsage        usage        = TextureUsage::SAMPLED,
            BOOL                srgb         = false,
            BOOL                generateMips = false);
        static LoadedTexture LoadTextureFromMemory(
            const BYTE*  data,
            SIZE_T       size,
            TextureUsage usage        = TextureUsage::SAMPLED,
            BOOL         srgb         = false,
            BOOL         generateMips = false);

        // Shader / bytecode loading (.nsc, compiled DXBC, etc.)
        static LoadedShader  LoadShaderFromFile(const std::string& path, const std::string& name = "");
        static LoadedShader  LoadShaderFromFile(const std::wstring& path, const std::string& name = "");
        static LoadedShader  LoadShaderFromMemory(const BYTE* data, SIZE_T size, const std::string& name = "");

        // 64-bit verified binary file reader helper
        static std::vector<BYTE> ReadBinaryFile(const std::string& path);
        static std::vector<BYTE> ReadBinaryFile(const std::wstring& path);
    };
}
