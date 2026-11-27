#include "shader_loader.h"
#include "NSCReader/include/nsc_reader.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define BOOL WIN_BOOL
#include <windows.h>
#include <d3dcompiler.h>
#undef BOOL
#undef OPAQUE
#undef TRANSPARENT
#undef IN
#undef OUT
#endif

#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>

namespace Northstar
{
    namespace
    {
        struct CoreDXBCCacheKey
        {
            UINT64 h1 = 0;
            UINT64 h2 = 0;
        };

        struct CoreDXBCCacheFileHeader
        {
            UINT32 magic    = 0x4344534Eu; // "NSDC"
            UINT32 version  = 3u;          // v3: salted with DXC (dxcompiler.dll + dxil.dll) + FXC + active ShaderModel (6.6 vs 5.1)
            UINT64 h1       = 0;
            UINT64 h2       = 0;
            UINT32 dxbcSize = 0;
            UINT32 reserved = 0;
        };

        const char* StageToHLSL12Target(ShaderStage stage, UINT32 shaderModel) noexcept
        {
            const bool sm66 = (shaderModel >= 0x66u);
            switch (stage)
            {
            case ShaderStage::VERTEX:   return sm66 ? "vs_6_6" : "vs_5_1";
            case ShaderStage::PIXEL:    return sm66 ? "ps_6_6" : "ps_5_1";
            case ShaderStage::COMPUTE:  return sm66 ? "cs_6_6" : "cs_5_1";
            case ShaderStage::GEOMETRY: return sm66 ? "gs_6_6" : "gs_5_1";
            case ShaderStage::HULL:     return sm66 ? "hs_6_6" : "hs_5_1";
            case ShaderStage::DOMAIN:   return sm66 ? "ds_6_6" : "ds_5_1";
            default:                    return nullptr;
            }
        }

        void MixModuleFileAttributes(const wchar_t* dllName, UINT64& ioSalt) noexcept
        {
#ifdef _WIN32
            wchar_t dllPath[MAX_PATH]{};
            HMODULE hMod  = ::GetModuleHandleW(dllName);
            DWORD pathLen = 0;
            if (hMod)
            {
                pathLen = ::GetModuleFileNameW(hMod, dllPath, MAX_PATH);
            }
            if (pathLen == 0 || pathLen >= MAX_PATH)
            {
                pathLen = ::SearchPathW(nullptr, dllName, nullptr, MAX_PATH, dllPath, nullptr);
            }
            if (pathLen > 0 && pathLen < MAX_PATH)
            {
                WIN32_FILE_ATTRIBUTE_DATA fad{};
                if (::GetFileAttributesExW(dllPath, GetFileExInfoStandard, &fad))
                {
                    const UINT64 fSize = (static_cast<UINT64>(fad.nFileSizeHigh) << 32) | fad.nFileSizeLow;
                    const UINT64 fTime = (static_cast<UINT64>(fad.ftLastWriteTime.dwHighDateTime) << 32) |
                                         fad.ftLastWriteTime.dwLowDateTime;
                    ioSalt ^= fSize * 1099511628211ULL;
                    ioSalt ^= fTime + 0x9E3779B97F4A7C15ULL + (ioSalt << 6) + (ioSalt >> 2);
                }
            }
#else
            (void)dllName;
            (void)ioSalt;
#endif
        }

        UINT64 GetD3DCompilerBinarySalt() noexcept
        {
            static std::once_flag s_once;
            static UINT64 s_salt = 0xD3DC066000000003ULL;
            std::call_once(s_once, []() {
                MixModuleFileAttributes(L"dxcompiler.dll",     s_salt);
                MixModuleFileAttributes(L"dxil.dll",           s_salt);
                MixModuleFileAttributes(L"d3dcompiler_47.dll", s_salt);
            });
            return s_salt;
        }

        CoreDXBCCacheKey ComputeCoreDXBCCacheKey(ShaderStage stage, const std::string& sourceCode, UINT32 targetShaderModel) noexcept
        {
            const char* target = StageToHLSL12Target(stage, targetShaderModel);
            if (!target || sourceCode.empty())
            {
                return {};
            }

            UINT64 h1 = 14695981039346656037ULL;
            UINT64 h2 = GetD3DCompilerBinarySalt();
            auto mixByte = [&](BYTE b) {
                h1 ^= static_cast<UINT64>(b);
                h1 *= 1099511628211ULL;
                h2 ^= static_cast<UINT64>(b) + 0x9E3779B97F4A7C15ULL + (h2 << 6) + (h2 >> 2);
            };
            for (const char* p = target; *p; ++p) mixByte(static_cast<BYTE>(*p));
            mixByte(0xFFu);
            const UINT32 compileFlags = (targetShaderModel >= 0x66u)
                ? 0x00662021u
#ifdef _WIN32
                : static_cast<UINT32>(D3DCOMPILE_OPTIMIZATION_LEVEL1);
#else
                : (1u << 14);
#endif
            for (UINT32 i = 0; i < 4u; ++i) mixByte(static_cast<BYTE>((compileFlags >> (i * 8u)) & 0xFFu));
            const UINT64 sz64 = static_cast<UINT64>(sourceCode.size());
            for (UINT32 i = 0; i < 8u; ++i) mixByte(static_cast<BYTE>((sz64 >> (i * 8u)) & 0xFFu));
            const BYTE* bytes = reinterpret_cast<const BYTE*>(sourceCode.data());
            for (SIZE_T i = 0; i < sourceCode.size(); ++i) mixByte(bytes[i]);
            return { h1, h2 };
        }

        std::filesystem::path GetCoreDXBCCacheDir()
        {
            static std::once_flag s_once;
            static std::filesystem::path s_dir;
            std::call_once(s_once, []() {
#ifdef _WIN32
                wchar_t envBuf[MAX_PATH]{};
                const DWORD len = ::GetEnvironmentVariableW(L"LOCALAPPDATA", envBuf, MAX_PATH);
                if (len > 0 && len < MAX_PATH)
                {
                    s_dir = std::filesystem::path(envBuf) / L"NorthstarGraphics" / L"DXBCCache";
                }
                else
#endif
                {
                    std::error_code ec;
                    s_dir = std::filesystem::temp_directory_path(ec) / L"NorthstarDXBCCache";
                }
                std::error_code ec;
                std::filesystem::create_directories(s_dir, ec);
            });
            return s_dir;
        }
    }

    BOOL ShaderLoader::TryPopulateCachedDXBC(Shader& ioShader, UINT32 targetShaderModel)
    {
        if (ioShader.GetTargetAPI() != TargetAPI::HLSL12 ||
            ioShader.HasBytecode() ||
            !ioShader.HasSourceCode())
        {
            return false;
        }

        const CoreDXBCCacheKey key = ComputeCoreDXBCCacheKey(ioShader.GetStage(), ioShader.GetSourceCode(), targetShaderModel);
        if (key.h1 == 0 && key.h2 == 0)
            return false;

        const auto cacheDir = GetCoreDXBCCacheDir();
        if (cacheDir.empty())
            return false;

        wchar_t fname[64]{};
        std::swprintf(fname, sizeof(fname) / sizeof(fname[0]), L"%016llx_%016llx.nsdc",
                      static_cast<unsigned long long>(key.h1), static_cast<unsigned long long>(key.h2));
        const auto filePath = cacheDir / fname;

        std::ifstream ifs(filePath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open())
            return false;

        const std::streamoff totalSize = ifs.tellg();
        if (totalSize < static_cast<std::streamoff>(sizeof(CoreDXBCCacheFileHeader) + 4u) ||
            totalSize > static_cast<std::streamoff>(64 * 1024 * 1024))
        {
            return false;
        }

        ifs.seekg(0);
        CoreDXBCCacheFileHeader hdr{};
        if (!ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr)))
            return false;
        if (hdr.magic != 0x4344534Eu || hdr.version != 3u || hdr.h1 != key.h1 || hdr.h2 != key.h2)
            return false;
        if (static_cast<std::streamoff>(sizeof(hdr) + hdr.dxbcSize) != totalSize || hdr.dxbcSize < 4u)
            return false;

        std::vector<BYTE> data(hdr.dxbcSize);
        if (!ifs.read(reinterpret_cast<char*>(data.data()), hdr.dxbcSize))
            return false;
        if (std::memcmp(data.data(), "DXBC", 4) != 0)
            return false;

        ioShader.SetBytecode(std::move(data));
        return true;
    }

    void ShaderLoader::StoreCachedDXBC(const Shader& shader, const void* dxbcData, SIZE_T dxbcSize, UINT32 targetShaderModel)
    {
        if (shader.GetTargetAPI() != TargetAPI::HLSL12 ||
            !shader.HasSourceCode() ||
            !dxbcData ||
            dxbcSize < 4u ||
            dxbcSize > (64u * 1024u * 1024u))
        {
            return;
        }
        if (std::memcmp(dxbcData, "DXBC", 4) != 0)
            return;

        const CoreDXBCCacheKey key = ComputeCoreDXBCCacheKey(shader.GetStage(), shader.GetSourceCode(), targetShaderModel);
        if (key.h1 == 0 && key.h2 == 0)
            return;

        const auto cacheDir = GetCoreDXBCCacheDir();
        if (cacheDir.empty())
            return;

        wchar_t fname[64]{};
        std::swprintf(fname, sizeof(fname) / sizeof(fname[0]), L"%016llx_%016llx.nsdc",
                      static_cast<unsigned long long>(key.h1), static_cast<unsigned long long>(key.h2));
        const auto finalPath = cacheDir / fname;

        unsigned long pid = 0;
        unsigned long tid = 0;
#ifdef _WIN32
        pid = static_cast<unsigned long>(::GetCurrentProcessId());
        tid = static_cast<unsigned long>(::GetCurrentThreadId());
#endif
        wchar_t tmpName[96]{};
        std::swprintf(tmpName, sizeof(tmpName) / sizeof(tmpName[0]), L"%016llx_%016llx.%lu.%lu.tmp",
                      static_cast<unsigned long long>(key.h1), static_cast<unsigned long long>(key.h2), pid, tid);
        const auto tmpPath = cacheDir / tmpName;

        CoreDXBCCacheFileHeader hdr{};
        hdr.h1       = key.h1;
        hdr.h2       = key.h2;
        hdr.dxbcSize = static_cast<UINT32>(dxbcSize);

        std::ofstream ofs(tmpPath, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open())
            return;
        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        ofs.write(reinterpret_cast<const char*>(dxbcData), static_cast<std::streamsize>(dxbcSize));
        ofs.close();

        if (ofs.good())
        {
#ifdef _WIN32
            if (!::MoveFileExW(tmpPath.c_str(), finalPath.c_str(), MOVEFILE_REPLACE_EXISTING))
            {
                std::error_code ec;
                std::filesystem::remove(tmpPath, ec);
            }
#else
            std::error_code ec;
            std::filesystem::rename(tmpPath, finalPath, ec);
            if (ec) std::filesystem::remove(tmpPath, ec);
#endif
        }
        else
        {
            std::error_code ec;
            std::filesystem::remove(tmpPath, ec);
        }
    }

    BOOL ShaderLoader::Load(const ShaderDesc& desc, Shader& outShader)
    {
        switch (desc.dataSource)
        {
        case ShaderDataSource::MEMORY:
        {
            if (!desc.initialData || desc.initialDataSize == 0)
                return false;
            return LoadFromMemory(desc.initialData,
                                  desc.initialDataSize,
                                  outShader,
                                  desc.name);
        }
        case ShaderDataSource::REFLECTION:
        {
            if (!desc.preloadedReflection)
                return false;
            outShader.SetReflection(*desc.preloadedReflection);
            if (desc.initialData && desc.initialDataSize > 0)
                outShader.SetBytecode(desc.initialData, desc.initialDataSize);
            if (!desc.name.empty())
                outShader.SetName(desc.name);
            return outShader.IsValid();
        }
        default:
            return false;
        }
    }

    BOOL ShaderLoader::LoadFromMemory(const BYTE* data, SIZE_T size, Shader& outShader, const std::string& name)
    {
        if (!data || size == 0)
            return false;

        NSL::ShaderData shaderData;
        if (!NSL::Reader::LoadFromMemory(data, size, shaderData))
        {
            std::cerr << "[ShaderLoader] Failed to parse .nsc from memory (" << size << " bytes)\n";
            return false;
        }

        outShader.SetReflection(std::move(shaderData));
        if (!name.empty())
            outShader.SetName(name);
        return outShader.IsValid();
    }
}
