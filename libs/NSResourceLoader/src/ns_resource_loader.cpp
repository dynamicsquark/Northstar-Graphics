#include "ns_resource_loader.h"
#include "stb/stb_image.h"

#include <cstdio>
#include <cstring>
#include <climits>

namespace Northstar::Tools
{
    namespace
    {
        static std::string WideToUTF8Name(const std::wstring& wstr)
        {
            std::string out;
            out.reserve(wstr.size());
            for (wchar_t wc : wstr)
            {
                if (wc >= 0 && wc < 128)
                {
                    out.push_back(static_cast<char>(wc));
                }
                else
                {
                    out.push_back('_');
                }
            }
            return out;
        }

        #pragma pack(push, 1)
        struct DDSPixelFormat
        {
            UINT32 size;
            UINT32 flags;
            UINT32 fourCC;
            UINT32 rgbBitCount;
            UINT32 rBitMask;
            UINT32 gBitMask;
            UINT32 bBitMask;
            UINT32 aBitMask;
        };

        struct DDSHeader
        {
            UINT32         size;
            UINT32         flags;
            UINT32         height;
            UINT32         width;
            UINT32         pitchOrLinearSize;
            UINT32         depth;
            UINT32         mipMapCount;
            UINT32         reserved1[11];
            DDSPixelFormat ddspf;
            UINT32         caps;
            UINT32         caps2;
            UINT32         caps3;
            UINT32         caps4;
            UINT32         reserved2;
        };

        struct DDSHeaderDXT10
        {
            UINT32 dxgiFormat;
            UINT32 resourceDimension;
            UINT32 miscFlag;
            UINT32 arraySize;
            UINT32 miscFlags2;
        };
        #pragma pack(pop)

        static constexpr UINT32 MakeFourCC(char a, char b, char c, char d) noexcept
        {
            return static_cast<UINT32>(static_cast<UINT8>(a)) |
                   (static_cast<UINT32>(static_cast<UINT8>(b)) << 8u) |
                   (static_cast<UINT32>(static_cast<UINT8>(c)) << 16u) |
                   (static_cast<UINT32>(static_cast<UINT8>(d)) << 24u);
        }

        static BOOL TryParseDDSFromMemory(
            const BYTE*    data,
            SIZE_T         size,
            TextureUsage   usage,
            BOOL           forceSRGB,
            LoadedTexture& outTex)
        {
            outTex = LoadedTexture{};
            if (!data || size < (4u + sizeof(DDSHeader)))
                return false;

            UINT32 magic = 0u;
            std::memcpy(&magic, data, sizeof(UINT32));
            if (magic != MakeFourCC('D', 'D', 'S', ' '))
                return false;

            DDSHeader hdr{};
            std::memcpy(&hdr, data + 4u, sizeof(DDSHeader));
            if (hdr.size != 124u || hdr.ddspf.size != 32u || hdr.width == 0u || hdr.height == 0u)
                return false;

            SIZE_T payloadOffset = 4u + sizeof(DDSHeader);
            TextureFormat fmt = TextureFormat::UNKNOWN;
            TextureType   texType = TextureType::TEXTURE_2D;
            UINT32        depth   = 1u;
            UINT32        layers  = 1u;
            const UINT32  mips    = (hdr.mipMapCount > 0u) ? hdr.mipMapCount : 1u;

            constexpr UINT32 DDPF_FOURCC      = 0x00000004u;
            constexpr UINT32 DDPF_RGB         = 0x00000040u;
            constexpr UINT32 DDSCAPS2_CUBEMAP = 0x00000200u;
            constexpr UINT32 DDSCAPS2_VOLUME  = 0x00200000u;
            constexpr UINT32 DDSCAPS2_CUBEMAP_ALLFACES = 0x0000FC00u;

            if ((hdr.caps2 & DDSCAPS2_VOLUME) != 0u && hdr.depth > 0u)
            {
                texType = TextureType::TEXTURE_3D;
                depth   = hdr.depth;
            }
            else if ((hdr.caps2 & DDSCAPS2_CUBEMAP) != 0u)
            {
                if ((hdr.caps2 & DDSCAPS2_CUBEMAP_ALLFACES) != DDSCAPS2_CUBEMAP_ALLFACES)
                    return false; // Partial cubemaps unsupported in D3D12
                texType = TextureType::TEXTURE_CUBE;
                layers  = 1u;
            }

            if ((hdr.ddspf.flags & DDPF_FOURCC) != 0u)
            {
                const UINT32 fcc = hdr.ddspf.fourCC;
                if (fcc == MakeFourCC('D', 'X', 'T', '1'))
                {
                    fmt = TextureFormat::BC1_UNORM;
                }
                else if (fcc == MakeFourCC('D', 'X', 'T', '5'))
                {
                    fmt = TextureFormat::BC3_UNORM;
                }
                else if (fcc == MakeFourCC('A', 'T', 'I', '2') || fcc == MakeFourCC('B', 'C', '5', 'U'))
                {
                    fmt = TextureFormat::BC5_UNORM;
                }
                else if (fcc == 113u) // D3DFMT_A16B16G16R16F
                {
                    fmt = TextureFormat::RGBA16_FLOAT;
                }
                else if (fcc == 114u) // D3DFMT_R32F
                {
                    fmt = TextureFormat::R32_FLOAT;
                }
                else if (fcc == 116u) // D3DFMT_A32B32G32R32F
                {
                    fmt = TextureFormat::RGBA32_FLOAT;
                }
                else if (fcc == MakeFourCC('D', 'X', '1', '0'))
                {
                    if (size < payloadOffset + sizeof(DDSHeaderDXT10))
                        return false;
                    DDSHeaderDXT10 dx10{};
                    std::memcpy(&dx10, data + payloadOffset, sizeof(DDSHeaderDXT10));
                    payloadOffset += sizeof(DDSHeaderDXT10);

                    const UINT32 arraySize = (dx10.arraySize > 0u) ? dx10.arraySize : 1u;
                    constexpr UINT32 DDS_RESOURCE_MISC_TEXTURECUBE = 0x4u;
                    if (dx10.resourceDimension == 4u) // D3D10_RESOURCE_DIMENSION_TEXTURE3D
                    {
                        texType = TextureType::TEXTURE_3D;
                        depth   = (hdr.depth > 0u) ? hdr.depth : 1u;
                        layers  = 1u;
                    }
                    else if ((dx10.miscFlag & DDS_RESOURCE_MISC_TEXTURECUBE) != 0u)
                    {
                        texType = (arraySize > 1u) ? TextureType::TEXTURE_CUBE_ARRAY : TextureType::TEXTURE_CUBE;
                        layers  = arraySize;
                    }
                    else if (arraySize > 1u)
                    {
                        texType = TextureType::TEXTURE_2D_ARRAY;
                        layers  = arraySize;
                    }

                    switch (dx10.dxgiFormat)
                    {
                        case 61u: fmt = TextureFormat::R8_UNORM;     break; // DXGI_FORMAT_R8_UNORM
                        case 49u: fmt = TextureFormat::RG8_UNORM;    break; // DXGI_FORMAT_R8G8_UNORM
                        case 28u: fmt = forceSRGB ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8_UNORM; break; // R8G8B8A8_UNORM
                        case 29u: fmt = TextureFormat::RGBA8_SRGB;   break; // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                        case 54u: fmt = TextureFormat::R16_FLOAT;    break; // DXGI_FORMAT_R16_FLOAT
                        case 34u: fmt = TextureFormat::RG16_FLOAT;   break; // DXGI_FORMAT_R16G16_FLOAT
                        case 10u: fmt = TextureFormat::RGBA16_FLOAT; break; // DXGI_FORMAT_R16G16B16A16_FLOAT
                        case 41u: fmt = TextureFormat::R32_FLOAT;    break; // DXGI_FORMAT_R32_FLOAT
                        case 16u: fmt = TextureFormat::RG32_FLOAT;   break; // DXGI_FORMAT_R32G32_FLOAT
                        case 2u:  fmt = TextureFormat::RGBA32_FLOAT; break; // DXGI_FORMAT_R32G32B32A32_FLOAT
                        case 71u: fmt = TextureFormat::BC1_UNORM;    break; // DXGI_FORMAT_BC1_UNORM
                        case 77u: fmt = TextureFormat::BC3_UNORM;    break; // DXGI_FORMAT_BC3_UNORM
                        case 83u: fmt = TextureFormat::BC5_UNORM;    break; // DXGI_FORMAT_BC5_UNORM
                        case 98u: fmt = forceSRGB ? TextureFormat::BC7_SRGB : TextureFormat::BC7_UNORM; break; // BC7_UNORM
                        case 99u: fmt = TextureFormat::BC7_SRGB;     break; // DXGI_FORMAT_BC7_UNORM_SRGB
                        default:  return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            else if ((hdr.ddspf.flags & DDPF_RGB) != 0u && hdr.ddspf.rgbBitCount == 32u)
            {
                if (hdr.ddspf.rBitMask == 0x000000FFu &&
                    hdr.ddspf.gBitMask == 0x0000FF00u &&
                    hdr.ddspf.bBitMask == 0x00FF0000u)
                {
                    fmt = forceSRGB ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8_UNORM;
                }
                else if (hdr.ddspf.rBitMask == 0x00FF0000u &&
                         hdr.ddspf.gBitMask == 0x0000FF00u &&
                         hdr.ddspf.bBitMask == 0x000000FFu)
                {
                    // BGRA8 legacy DDS -> swizzle to RGBA8 below
                    fmt = forceSRGB ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8_UNORM;
                    const SIZE_T expectedBytes = ComputeTextureFullChainTightBytes(
                        hdr.width, hdr.height, depth, mips, layers, texType, fmt);
                    if (expectedBytes == 0u || payloadOffset + expectedBytes > size)
                        return false;
                    outTex.pixels.assign(data + payloadOffset, data + payloadOffset + expectedBytes);
                    for (SIZE_T p = 0u; p + 3u < expectedBytes; p += 4u)
                    {
                        const BYTE b = outTex.pixels[p + 0u];
                        outTex.pixels[p + 0u] = outTex.pixels[p + 2u];
                        outTex.pixels[p + 2u] = b;
                    }
                    outTex.width       = hdr.width;
                    outTex.height      = hdr.height;
                    outTex.depth       = depth;
                    outTex.mipLevels   = mips;
                    outTex.arrayLayers = layers;
                    outTex.channels    = 4u;
                    outTex.format      = fmt;
                    outTex.type        = texType;
                    outTex.usage       = usage;
                    outTex.isValid     = true;
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            if (IsTextureFormatBlockCompressed(fmt) && ((hdr.width % 4u) != 0u || (hdr.height % 4u) != 0u))
                return false;

            const SIZE_T expectedBytes = ComputeTextureFullChainTightBytes(
                hdr.width, hdr.height, depth, mips, layers, texType, fmt);
            if (expectedBytes == 0u || payloadOffset + expectedBytes > size)
                return false;

            outTex.pixels.assign(data + payloadOffset, data + payloadOffset + expectedBytes);
            outTex.width       = hdr.width;
            outTex.height      = hdr.height;
            outTex.depth       = depth;
            outTex.mipLevels   = mips;
            outTex.arrayLayers = layers;
            outTex.channels    = 4u;
            outTex.format      = fmt;
            outTex.type        = texType;
            outTex.usage       = usage;
            outTex.isValid     = true;
            return true;
        }
    } // anonymous namespace

    std::vector<BYTE> ResourceLoader::ReadBinaryFile(const std::string& path)
    {
        std::vector<BYTE> buffer;
        if (path.empty()) return buffer;

        FILE* file = nullptr;
        if (fopen_s(&file, path.c_str(), "rb") == 0 && file != nullptr)
        {
            if (_fseeki64(file, 0, SEEK_END) == 0)
            {
                const INT64 fileSize = _ftelli64(file);
                if (fileSize > 0 && static_cast<UINT64>(fileSize) <= static_cast<UINT64>(SIZE_MAX) &&
                    _fseeki64(file, 0, SEEK_SET) == 0)
                {
                    const SIZE_T targetBytes = static_cast<SIZE_T>(fileSize);
                    buffer.resize(targetBytes);
                    const SIZE_T bytesRead = std::fread(buffer.data(), 1, targetBytes, file);
                    if (bytesRead != targetBytes)
                    {
                        buffer.clear();
                    }
                }
            }
            std::fclose(file);
        }
        return buffer;
    }

    std::vector<BYTE> ResourceLoader::ReadBinaryFile(const std::wstring& path)
    {
        std::vector<BYTE> buffer;
        if (path.empty()) return buffer;

        FILE* file = nullptr;
        if (_wfopen_s(&file, path.c_str(), L"rb") == 0 && file != nullptr)
        {
            if (_fseeki64(file, 0, SEEK_END) == 0)
            {
                const INT64 fileSize = _ftelli64(file);
                if (fileSize > 0 && static_cast<UINT64>(fileSize) <= static_cast<UINT64>(SIZE_MAX) &&
                    _fseeki64(file, 0, SEEK_SET) == 0)
                {
                    const SIZE_T targetBytes = static_cast<SIZE_T>(fileSize);
                    buffer.resize(targetBytes);
                    const SIZE_T bytesRead = std::fread(buffer.data(), 1, targetBytes, file);
                    if (bytesRead != targetBytes)
                    {
                        buffer.clear();
                    }
                }
            }
            std::fclose(file);
        }
        return buffer;
    }

    LoadedTexture ResourceLoader::LoadTextureFromMemory(
        const BYTE*  data,
        SIZE_T       size,
        TextureUsage usage,
        BOOL         srgb,
        BOOL         generateMips)
    {
        LoadedTexture result{};
        if (data == nullptr || size == 0u)
        {
            return result;
        }

        // 1. Native DDS container fast-path (supports BC1-BC7, multi-mip chains, Cubemaps, Arrays, 3D Volumes)
        if (size >= 128u && std::memcmp(data, "DDS ", 4) == 0)
        {
            TryParseDDSFromMemory(data, size, usage, srgb, result);
            return result;
        }

        // stb_image takes a signed 32-bit length
        if (size > static_cast<SIZE_T>(INT_MAX))
        {
            return result;
        }

        int width    = 0;
        int height   = 0;
        int channels = 0;

        // 2. Radiance HDR float image path (RGBA32_FLOAT)
        if (stbi_is_hdr_from_memory(reinterpret_cast<const stbi_uc*>(data), static_cast<int>(size)) != 0)
        {
            float* hdrDecoded = stbi_loadf_from_memory(
                reinterpret_cast<const stbi_uc*>(data),
                static_cast<int>(size),
                &width,
                &height,
                &channels,
                STBI_rgb_alpha);

            if (hdrDecoded == nullptr || width <= 0 || height <= 0)
            {
                if (hdrDecoded) stbi_image_free(hdrDecoded);
                return result;
            }

            const UINT64 totalBytes64 = static_cast<UINT64>(width) * static_cast<UINT64>(height) * (4ULL * sizeof(float));
            if (totalBytes64 == 0ULL || totalBytes64 > 0x7FFFFFFFULL)
            {
                stbi_image_free(hdrDecoded);
                return result;
            }

            const SIZE_T totalBytes = static_cast<SIZE_T>(totalBytes64);
            result.pixels.resize(totalBytes);
            std::memcpy(result.pixels.data(), hdrDecoded, totalBytes);
            stbi_image_free(hdrDecoded);

            result.width       = static_cast<UINT32>(width);
            result.height      = static_cast<UINT32>(height);
            result.depth       = 1u;
            result.mipLevels   = generateMips ? 0u : 1u;
            result.arrayLayers = 1u;
            result.channels    = 4u;
            result.format      = TextureFormat::RGBA32_FLOAT;
            result.type        = TextureType::TEXTURE_2D;
            result.usage       = usage;
            result.isValid     = true;
            return result;
        }

        // 3. Standard LDR image path (PNG, JPEG, TGA, BMP -> RGBA8_UNORM or RGBA8_SRGB)
        stbi_uc* decoded = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(data),
            static_cast<int>(size),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha
        );

        if (decoded == nullptr || width <= 0 || height <= 0)
        {
            if (decoded) stbi_image_free(decoded);
            return result;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(width) * static_cast<UINT64>(height) * 4ULL;
        if (totalBytes64 == 0ULL || totalBytes64 > 0x7FFFFFFFULL)
        {
            stbi_image_free(decoded);
            return result;
        }

        const SIZE_T totalBytes = static_cast<SIZE_T>(totalBytes64);
        result.pixels.resize(totalBytes);
        std::memcpy(result.pixels.data(), decoded, totalBytes);
        stbi_image_free(decoded);

        result.width       = static_cast<UINT32>(width);
        result.height      = static_cast<UINT32>(height);
        result.depth       = 1u;
        result.mipLevels   = generateMips ? 0u : 1u;
        result.arrayLayers = 1u;
        result.channels    = 4u;
        result.format      = srgb ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8_UNORM;
        result.type        = TextureType::TEXTURE_2D;
        result.usage       = usage;
        result.isValid     = true;

        return result;
    }

    LoadedTexture ResourceLoader::LoadTextureFromFile(
        const std::string& path,
        TextureUsage       usage,
        BOOL               srgb,
        BOOL               generateMips)
    {
        const std::vector<BYTE> bytes = ReadBinaryFile(path);
        if (bytes.empty())
        {
            return {};
        }
        return LoadTextureFromMemory(bytes.data(), bytes.size(), usage, srgb, generateMips);
    }

    LoadedTexture ResourceLoader::LoadTextureFromFile(
        const std::wstring& path,
        TextureUsage        usage,
        BOOL                srgb,
        BOOL                generateMips)
    {
        const std::vector<BYTE> bytes = ReadBinaryFile(path);
        if (bytes.empty())
        {
            return {};
        }
        return LoadTextureFromMemory(bytes.data(), bytes.size(), usage, srgb, generateMips);
    }

    LoadedShader ResourceLoader::LoadShaderFromMemory(const BYTE* data, SIZE_T size, const std::string& name)
    {
        LoadedShader result{};
        if (data == nullptr || size == 0u)
        {
            return result;
        }

        result.bytecode.assign(data, data + size);
        result.name    = name;
        result.isValid = true;
        return result;
    }

    LoadedShader ResourceLoader::LoadShaderFromFile(const std::string& path, const std::string& name)
    {
        const std::vector<BYTE> bytes = ReadBinaryFile(path);
        if (bytes.empty())
        {
            return {};
        }
        return LoadShaderFromMemory(bytes.data(), bytes.size(), name.empty() ? path : name);
    }

    LoadedShader ResourceLoader::LoadShaderFromFile(const std::wstring& path, const std::string& name)
    {
        const std::vector<BYTE> bytes = ReadBinaryFile(path);
        if (bytes.empty())
        {
            return {};
        }
        return LoadShaderFromMemory(bytes.data(), bytes.size(), name.empty() ? WideToUTF8Name(path) : name);
    }
}
