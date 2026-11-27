#include "d3d12_resource_manager.h"
#ifdef _WIN32
#define BOOL WIN_BOOL
#include <dxcapi.h>
#undef BOOL
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <iostream>
#include <mutex>

namespace Northstar
{
    using namespace D3D12Util;

    namespace
    {
        static constexpr SIZE_T kMaxInMemoryDXBCCacheEntries = 512u;

        struct DXBCCacheKey
        {
            UINT64 h1 = 0;
            UINT64 h2 = 0;
            bool operator==(const DXBCCacheKey& o) const noexcept { return h1 == o.h1 && h2 == o.h2; }
        };

        struct DXBCCacheKeyHash
        {
            SIZE_T operator()(const DXBCCacheKey& k) const noexcept
            {
                return static_cast<SIZE_T>(k.h1 ^ (k.h2 * 0x9E3779B97F4A7C15ULL));
            }
        };

        static DXBCCacheKey ComputeDXBCCacheKey(const char* target, UINT32 compileFlags, const void* src, SIZE_T srcSize) noexcept
        {
            UINT64 h1 = 14695981039346656037ULL;
            UINT64 h2 = 0x243F6A8885A308D3ULL;
            auto mixByte = [&](BYTE b) {
                h1 ^= static_cast<UINT64>(b);
                h1 *= 1099511628211ULL;
                h2 ^= static_cast<UINT64>(b) + 0x9E3779B97F4A7C15ULL + (h2 << 6) + (h2 >> 2);
            };
            if (target)
            {
                for (const char* p = target; *p; ++p) mixByte(static_cast<BYTE>(*p));
            }
            mixByte(0xFFu);
            for (UINT32 i = 0; i < 4u; ++i) mixByte(static_cast<BYTE>((compileFlags >> (i * 8u)) & 0xFFu));
            const UINT64 sz64 = static_cast<UINT64>(srcSize);
            for (UINT32 i = 0; i < 8u; ++i) mixByte(static_cast<BYTE>((sz64 >> (i * 8u)) & 0xFFu));
            const BYTE* bytes = static_cast<const BYTE*>(src);
            for (SIZE_T i = 0; i < srcSize; ++i) mixByte(bytes[i]);
            return { h1, h2 };
        }

        static std::mutex& GetDXBCCacheMutex()
        {
            static std::mutex s_mtx;
            return s_mtx;
        }

        static std::unordered_map<DXBCCacheKey, std::vector<BYTE>, DXBCCacheKeyHash>& GetDXBCMemoryCache()
        {
            static std::unordered_map<DXBCCacheKey, std::vector<BYTE>, DXBCCacheKeyHash> s_cache;
            return s_cache;
        }

        static BOOL TryLoadCachedDXBC(const DXBCCacheKey& key, std::vector<BYTE>& outBytes)
        {
            std::lock_guard<std::mutex> lock(GetDXBCCacheMutex());
            auto& mem = GetDXBCMemoryCache();
            auto it = mem.find(key);
            if (it != mem.end() && it->second.size() >= 4u)
            {
                outBytes = it->second;
                return true;
            }
            return false;
        }

        static void StoreCachedDXBC(const DXBCCacheKey& key, const void* dxbcPtr, SIZE_T dxbcSize)
        {
            if (!dxbcPtr || dxbcSize < 4u || dxbcSize > (64u * 1024u * 1024u)) return;
            if (std::memcmp(dxbcPtr, "DXBC", 4) != 0) return;

            std::vector<BYTE> copy(dxbcSize);
            std::memcpy(copy.data(), dxbcPtr, dxbcSize);
            std::lock_guard<std::mutex> lock(GetDXBCCacheMutex());
            auto& mem = GetDXBCMemoryCache();
            if (mem.size() >= kMaxInMemoryDXBCCacheEntries && mem.find(key) == mem.end())
            {
                mem.clear();
            }
            mem[key] = std::move(copy);
        }

        // ==================== EXACT LINEAR / sRGB / FP16 / FP32 POLYPHASE MIP GENERATOR ====================

        static float HalfBitsToFloat(UINT16 h) noexcept
        {
            const UINT32 sign = static_cast<UINT32>(h & 0x8000u) << 16u;
            UINT32 exp        = (h >> 10u) & 0x1Fu;
            UINT32 mant       = h & 0x03FFu;
            UINT32 bits       = 0u;

            if (exp == 0u)
            {
                if (mant == 0u)
                {
                    bits = sign;
                }
                else
                {
                    exp = 1u;
                    while ((mant & 0x0400u) == 0u)
                    {
                        mant <<= 1u;
                        --exp;
                    }
                    mant &= 0x03FFu;
                    bits = sign | ((exp + (127u - 15u)) << 23u) | (mant << 13u);
                }
            }
            else if (exp == 0x1Fu)
            {
                bits = sign | 0x7F800000u | (mant << 13u);
            }
            else
            {
                bits = sign | ((exp + (127u - 15u)) << 23u) | (mant << 13u);
            }

            float f = 0.0f;
            std::memcpy(&f, &bits, sizeof(float));
            return f;
        }

        static UINT16 FloatToHalfBits(float f) noexcept
        {
            UINT32 bits = 0u;
            std::memcpy(&bits, &f, sizeof(UINT32));
            const UINT16 sign = static_cast<UINT16>((bits >> 16u) & 0x8000u);
            const INT32  rawE = static_cast<INT32>((bits >> 23u) & 0xFFu);
            UINT32       mant = bits & 0x007FFFFFu;

            if (rawE == 0xFF)
            {
                return (mant != 0u) ? static_cast<UINT16>(sign | 0x7E00u) : static_cast<UINT16>(sign | 0x7C00u);
            }

            const INT32 exp = rawE - 127 + 15;
            if (exp <= 0)
            {
                if (exp < -10) return sign;
                const UINT32 shift = static_cast<UINT32>(1 - exp);
                const UINT32 fullM = mant | 0x00800000u;
                const UINT32 half  = 1u << (shift + 12u);
                const UINT32 rem   = fullM & ((half << 1u) - 1u);
                const UINT32 tie   = (rem > half || (rem == half && ((fullM >> (shift + 13u)) & 1u) != 0u)) ? 1u : 0u;
                return static_cast<UINT16>(sign | static_cast<UINT16>((fullM >> (shift + 13u)) + tie));
            }
            if (exp >= 31)
            {
                return static_cast<UINT16>(sign | 0x7C00u);
            }

            const UINT32 rem     = mant & 0x1FFFu;
            const UINT32 tie     = (rem > 0x1000u || (rem == 0x1000u && ((mant >> 13u) & 1u) != 0u)) ? 1u : 0u;
            const UINT32 rounded = (static_cast<UINT32>(exp) << 10u) + (mant >> 13u) + tie;
            return static_cast<UINT16>(sign | static_cast<UINT16>((std::min)(rounded, 0x7C00u)));
        }

        static float SrgbByteToLinear(BYTE v) noexcept
        {
            const float s = static_cast<float>(v) * (1.0f / 255.0f);
            return (s <= 0.04045f) ? (s * (1.0f / 12.92f)) : std::pow((s + 0.055f) * (1.0f / 1.055f), 2.4f);
        }

        static BYTE LinearToSrgbByte(float lin) noexcept
        {
            const float c = (std::max)(0.0f, (std::min)(1.0f, lin));
            const float s = (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
            const float q = (std::max)(0.0f, (std::min)(255.0f, s * 255.0f + 0.5f));
            return static_cast<BYTE>(q);
        }

        static BYTE UnormFloatToByte(float v) noexcept
        {
            const float q = (std::max)(0.0f, (std::min)(255.0f, v * 255.0f + 0.5f));
            return static_cast<BYTE>(q);
        }

        static void ReadTexelLinearRGBA(TextureFormat fmt, const BYTE* texelPtr, float outRGBA[4]) noexcept
        {
            outRGBA[0] = 0.0f; outRGBA[1] = 0.0f; outRGBA[2] = 0.0f; outRGBA[3] = 1.0f;
            switch (fmt)
            {
                case TextureFormat::R8_UNORM:
                    outRGBA[0] = static_cast<float>(texelPtr[0]) * (1.0f / 255.0f);
                    break;
                case TextureFormat::RG8_UNORM:
                    outRGBA[0] = static_cast<float>(texelPtr[0]) * (1.0f / 255.0f);
                    outRGBA[1] = static_cast<float>(texelPtr[1]) * (1.0f / 255.0f);
                    break;
                case TextureFormat::RGBA8_UNORM:
                    outRGBA[0] = static_cast<float>(texelPtr[0]) * (1.0f / 255.0f);
                    outRGBA[1] = static_cast<float>(texelPtr[1]) * (1.0f / 255.0f);
                    outRGBA[2] = static_cast<float>(texelPtr[2]) * (1.0f / 255.0f);
                    outRGBA[3] = static_cast<float>(texelPtr[3]) * (1.0f / 255.0f);
                    break;
                case TextureFormat::RGBA8_SRGB:
                    outRGBA[0] = SrgbByteToLinear(texelPtr[0]);
                    outRGBA[1] = SrgbByteToLinear(texelPtr[1]);
                    outRGBA[2] = SrgbByteToLinear(texelPtr[2]);
                    outRGBA[3] = static_cast<float>(texelPtr[3]) * (1.0f / 255.0f);
                    break;
                case TextureFormat::R16_FLOAT:
                {
                    UINT16 h0 = 0u;
                    std::memcpy(&h0, texelPtr, 2u);
                    outRGBA[0] = HalfBitsToFloat(h0);
                    break;
                }
                case TextureFormat::RG16_FLOAT:
                {
                    UINT16 h[2]{};
                    std::memcpy(h, texelPtr, 4u);
                    outRGBA[0] = HalfBitsToFloat(h[0]);
                    outRGBA[1] = HalfBitsToFloat(h[1]);
                    break;
                }
                case TextureFormat::RGBA16_FLOAT:
                {
                    UINT16 h[4]{};
                    std::memcpy(h, texelPtr, 8u);
                    outRGBA[0] = HalfBitsToFloat(h[0]);
                    outRGBA[1] = HalfBitsToFloat(h[1]);
                    outRGBA[2] = HalfBitsToFloat(h[2]);
                    outRGBA[3] = HalfBitsToFloat(h[3]);
                    break;
                }
                case TextureFormat::R32_FLOAT:
                    std::memcpy(&outRGBA[0], texelPtr, 4u);
                    break;
                case TextureFormat::RG32_FLOAT:
                    std::memcpy(&outRGBA[0], texelPtr, 8u);
                    break;
                case TextureFormat::RGBA32_FLOAT:
                    std::memcpy(outRGBA, texelPtr, 16u);
                    break;
                default:
                    break;
            }
        }

        static void WriteTexelLinearRGBA(TextureFormat fmt, BYTE* dstPtr, const float inRGBA[4]) noexcept
        {
            switch (fmt)
            {
                case TextureFormat::R8_UNORM:
                    dstPtr[0] = UnormFloatToByte(inRGBA[0]);
                    break;
                case TextureFormat::RG8_UNORM:
                    dstPtr[0] = UnormFloatToByte(inRGBA[0]);
                    dstPtr[1] = UnormFloatToByte(inRGBA[1]);
                    break;
                case TextureFormat::RGBA8_UNORM:
                    dstPtr[0] = UnormFloatToByte(inRGBA[0]);
                    dstPtr[1] = UnormFloatToByte(inRGBA[1]);
                    dstPtr[2] = UnormFloatToByte(inRGBA[2]);
                    dstPtr[3] = UnormFloatToByte(inRGBA[3]);
                    break;
                case TextureFormat::RGBA8_SRGB:
                    dstPtr[0] = LinearToSrgbByte(inRGBA[0]);
                    dstPtr[1] = LinearToSrgbByte(inRGBA[1]);
                    dstPtr[2] = LinearToSrgbByte(inRGBA[2]);
                    dstPtr[3] = UnormFloatToByte(inRGBA[3]);
                    break;
                case TextureFormat::R16_FLOAT:
                {
                    const UINT16 h0 = FloatToHalfBits(inRGBA[0]);
                    std::memcpy(dstPtr, &h0, 2u);
                    break;
                }
                case TextureFormat::RG16_FLOAT:
                {
                    const UINT16 h[2] = { FloatToHalfBits(inRGBA[0]), FloatToHalfBits(inRGBA[1]) };
                    std::memcpy(dstPtr, h, 4u);
                    break;
                }
                case TextureFormat::RGBA16_FLOAT:
                {
                    const UINT16 h[4] = {
                        FloatToHalfBits(inRGBA[0]),
                        FloatToHalfBits(inRGBA[1]),
                        FloatToHalfBits(inRGBA[2]),
                        FloatToHalfBits(inRGBA[3])
                    };
                    std::memcpy(dstPtr, h, 8u);
                    break;
                }
                case TextureFormat::R32_FLOAT:
                    std::memcpy(dstPtr, &inRGBA[0], 4u);
                    break;
                case TextureFormat::RG32_FLOAT:
                    std::memcpy(dstPtr, &inRGBA[0], 8u);
                    break;
                case TextureFormat::RGBA32_FLOAT:
                    std::memcpy(dstPtr, inRGBA, 16u);
                    break;
                default:
                    break;
            }
        }

        // Exact D3D/ARB polyphase NPOT box filter tap weights: preserves energy and prevents
        // odd-dimension border texel truncation across arbitrary POT and NPOT mip pyramids.
        static void ComputePolyphaseAxisTaps(
            UINT32  srcLen,
            UINT32  dstLen,
            UINT32  dstCoord,
            UINT32  outIdx[3],
            float   outWeight[3],
            UINT32& outTapCount) noexcept
        {
            if (srcLen <= 1u || dstLen >= srcLen)
            {
                outTapCount  = 1u;
                outIdx[0]    = (srcLen > 0u) ? (std::min)(dstCoord, srcLen - 1u) : 0u;
                outWeight[0] = 1.0f;
                return;
            }
            if ((srcLen & 1u) == 0u && srcLen == 2u * dstLen)
            {
                outTapCount  = 2u;
                outIdx[0]    = 2u * dstCoord;
                outIdx[1]    = (std::min)(2u * dstCoord + 1u, srcLen - 1u);
                outWeight[0] = 0.5f;
                outWeight[1] = 0.5f;
                return;
            }

            const float invSrc = 1.0f / static_cast<float>(srcLen);
            outTapCount  = 3u;
            outIdx[0]    = (std::min)(2u * dstCoord, srcLen - 1u);
            outIdx[1]    = (std::min)(2u * dstCoord + 1u, srcLen - 1u);
            outIdx[2]    = (std::min)(2u * dstCoord + 2u, srcLen - 1u);
            outWeight[0] = static_cast<float>(dstLen - dstCoord) * invSrc;
            outWeight[1] = static_cast<float>(dstLen) * invSrc;
            outWeight[2] = static_cast<float>(dstCoord + 1u) * invSrc;
        }

        static void DownsampleMipSlicePolyphase(
            TextureFormat fmt,
            const BYTE*   srcBytes,
            UINT32        srcW,
            UINT32        srcH,
            UINT32        srcD,
            BYTE*         dstBytes,
            UINT32        dstW,
            UINT32        dstH,
            UINT32        dstD) noexcept
        {
            const UINT32 bpp          = GetTextureFormatBytesPerUnit(fmt);
            const SIZE_T srcRowPitch  = static_cast<SIZE_T>(srcW) * bpp;
            const SIZE_T srcSliceStep = srcRowPitch * static_cast<SIZE_T>(srcH);
            const SIZE_T dstRowPitch  = static_cast<SIZE_T>(dstW) * bpp;
            const SIZE_T dstSliceStep = dstRowPitch * static_cast<SIZE_T>(dstH);

            for (UINT32 z = 0u; z < dstD; ++z)
            {
                UINT32 zIdx[3]{}; float zW[3]{}; UINT32 zTaps = 0u;
                ComputePolyphaseAxisTaps(srcD, dstD, z, zIdx, zW, zTaps);

                for (UINT32 y = 0u; y < dstH; ++y)
                {
                    UINT32 yIdx[3]{}; float yW[3]{}; UINT32 yTaps = 0u;
                    ComputePolyphaseAxisTaps(srcH, dstH, y, yIdx, yW, yTaps);

                    BYTE* dstRow = dstBytes + static_cast<SIZE_T>(z) * dstSliceStep + static_cast<SIZE_T>(y) * dstRowPitch;
                    for (UINT32 x = 0u; x < dstW; ++x)
                    {
                        UINT32 xIdx[3]{}; float xW[3]{}; UINT32 xTaps = 0u;
                        ComputePolyphaseAxisTaps(srcW, dstW, x, xIdx, xW, xTaps);

                        float accum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        for (UINT32 tz = 0u; tz < zTaps; ++tz)
                        {
                            const BYTE* srcZ = srcBytes + static_cast<SIZE_T>(zIdx[tz]) * srcSliceStep;
                            for (UINT32 ty = 0u; ty < yTaps; ++ty)
                            {
                                const float zyWeight = zW[tz] * yW[ty];
                                const BYTE* srcY = srcZ + static_cast<SIZE_T>(yIdx[ty]) * srcRowPitch;
                                for (UINT32 tx = 0u; tx < xTaps; ++tx)
                                {
                                    const float w = zyWeight * xW[tx];
                                    float sampleRGBA[4];
                                    ReadTexelLinearRGBA(fmt, srcY + static_cast<SIZE_T>(xIdx[tx]) * bpp, sampleRGBA);
                                    accum[0] += sampleRGBA[0] * w;
                                    accum[1] += sampleRGBA[1] * w;
                                    accum[2] += sampleRGBA[2] * w;
                                    accum[3] += sampleRGBA[3] * w;
                                }
                            }
                        }
                        WriteTexelLinearRGBA(fmt, dstRow + static_cast<SIZE_T>(x) * bpp, accum);
                    }
                }
            }
        }

        static DXGI_FORMAT ResolveCopyableReadbackFormat(DXGI_FORMAT resourceFormat, TextureFormat engineFormat) noexcept
        {
            switch (resourceFormat)
            {
                case DXGI_FORMAT_R32_TYPELESS:      return DXGI_FORMAT_R32_FLOAT;
                case DXGI_FORMAT_R16_TYPELESS:      return DXGI_FORMAT_R16_UNORM;
                case DXGI_FORMAT_R24G8_TYPELESS:    return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                case DXGI_FORMAT_R8G8B8A8_TYPELESS:
                    return (engineFormat == TextureFormat::RGBA8_SRGB)
                        ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                        : DXGI_FORMAT_R8G8B8A8_UNORM;
                default:
                    return resourceFormat;
            }
        }
    } // anonymous namespace

    NativeHandle D3D12ResourceManager::CreateVertexBuffer(
        D3D12DeviceContext& ctx,
        const void* data,
        UINT32 byteSize,
        UINT32 stride)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || byteSize == 0) return 0;

        D3D12_HEAP_PROPERTIES defaultHeap = GetDefaultHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(byteSize);

        ComPtr<ID3D12Resource> defaultBuffer;
        if (FAILED(device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&defaultBuffer))))
        {
            return 0;
        }

        if (data)
        {
            D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
            ComPtr<ID3D12Resource> uploadBuffer;
            if (SUCCEEDED(device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer))))
            {
                void* pMapped = nullptr;
                D3D12_RANGE readRange{ 0, 0 };
                if (SUCCEEDED(uploadBuffer->Map(0, &readRange, &pMapped)))
                {
                    std::memcpy(pMapped, data, byteSize);
                    uploadBuffer->Unmap(0, nullptr);

                    ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
                        D3D12_RESOURCE_BARRIER toCopy = CreateTransitionBarrier(
                            defaultBuffer.Get(),
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST
                        );
                        cmdList->ResourceBarrier(1, &toCopy);
                        cmdList->CopyBufferRegion(defaultBuffer.Get(), 0, uploadBuffer.Get(), 0, byteSize);
                        D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(
                            defaultBuffer.Get(),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                        );
                        cmdList->ResourceBarrier(1, &barrier);
                    });
                }
            }
        }

        D3D12VertexBufferResource res{};
        res.resource            = defaultBuffer;
        res.byteSize            = byteSize;
        res.stride              = stride;
        res.vertexCount         = (stride > 0) ? (byteSize / stride) : 0;
        res.view.BufferLocation = defaultBuffer->GetGPUVirtualAddress();
        res.view.SizeInBytes    = byteSize;
        res.view.StrideInBytes  = stride;

        NativeHandle handle = GenerateHandle();
        m_vertexBuffers[handle] = res;
        return handle;
    }

    void D3D12ResourceManager::DestroyVertexBuffer(NativeHandle handle, UINT64 fenceValue)
    {
        auto it = m_vertexBuffers.find(handle);
        if (it != m_vertexBuffers.end())
        {
            for (auto cIt = m_cpuMeshToGpuVb.begin(); cIt != m_cpuMeshToGpuVb.end(); ++cIt)
            {
                if (cIt->second == handle)
                {
                    m_cpuMeshToGpuVb.erase(cIt);
                    break;
                }
            }

            if (it->second.resource)
            {
                D3D12RetiredResource retired{};
                retired.fenceValue = fenceValue;
                retired.resource   = std::move(it->second.resource);
                m_retiredResources.push_back(std::move(retired));
            }
            m_vertexBuffers.erase(it);
        }
    }

    NativeHandle D3D12ResourceManager::CreateIndexBuffer(
        D3D12DeviceContext& ctx,
        const void* data,
        UINT32 byteSize,
        BOOL is32Bit)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || byteSize == 0) return 0;

        D3D12_HEAP_PROPERTIES defaultHeap = GetDefaultHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(byteSize);

        ComPtr<ID3D12Resource> defaultBuffer;
        if (FAILED(device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&defaultBuffer))))
        {
            return 0;
        }

        if (data)
        {
            D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
            ComPtr<ID3D12Resource> uploadBuffer;
            if (SUCCEEDED(device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &bufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer))))
            {
                void* pMapped = nullptr;
                D3D12_RANGE readRange{ 0, 0 };
                if (SUCCEEDED(uploadBuffer->Map(0, &readRange, &pMapped)))
                {
                    std::memcpy(pMapped, data, byteSize);
                    uploadBuffer->Unmap(0, nullptr);

                    ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
                        D3D12_RESOURCE_BARRIER toCopy = CreateTransitionBarrier(
                            defaultBuffer.Get(),
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST
                        );
                        cmdList->ResourceBarrier(1, &toCopy);
                        cmdList->CopyBufferRegion(defaultBuffer.Get(), 0, uploadBuffer.Get(), 0, byteSize);
                        D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(
                            defaultBuffer.Get(),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_INDEX_BUFFER
                        );
                        cmdList->ResourceBarrier(1, &barrier);
                    });
                }
            }
        }

        const DXGI_FORMAT format = is32Bit ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
        const UINT32 indexStride = is32Bit ? 4 : 2;

        D3D12IndexBufferResource res{};
        res.resource            = defaultBuffer;
        res.byteSize            = byteSize;
        res.format              = format;
        res.indexCount          = byteSize / indexStride;
        res.view.BufferLocation = defaultBuffer->GetGPUVirtualAddress();
        res.view.SizeInBytes    = byteSize;
        res.view.Format         = format;

        NativeHandle handle = GenerateHandle();
        m_indexBuffers[handle] = res;
        return handle;
    }

    void D3D12ResourceManager::DestroyIndexBuffer(NativeHandle handle, UINT64 fenceValue)
    {
        auto it = m_indexBuffers.find(handle);
        if (it != m_indexBuffers.end())
        {
            for (auto cIt = m_cpuMeshToGpuIb.begin(); cIt != m_cpuMeshToGpuIb.end(); ++cIt)
            {
                if (cIt->second == handle)
                {
                    m_cpuMeshToGpuIb.erase(cIt);
                    break;
                }
            }

            if (it->second.resource)
            {
                D3D12RetiredResource retired{};
                retired.fenceValue = fenceValue;
                retired.resource   = std::move(it->second.resource);
                m_retiredResources.push_back(std::move(retired));
            }
            m_indexBuffers.erase(it);
        }
    }

    NativeHandle D3D12ResourceManager::CreateConstantBuffer(
        D3D12DeviceContext& ctx,
        D3D12DescriptorPool& cbvPool,
        UINT32 byteSize)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || byteSize == 0) return 0;

        const UINT32 alignedSize = (byteSize + 255) & ~255;

        D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(alignedSize);

        ComPtr<ID3D12Resource> uploadBuffer;
        if (FAILED(device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer))))
        {
            return 0;
        }

        void* mappedData = nullptr;
        D3D12_RANGE readRange{ 0, 0 };
        if (FAILED(uploadBuffer->Map(0, &readRange, &mappedData)))
        {
            return 0;
        }

        UINT32 descIndex = cbvPool.Allocate();
        if (descIndex == UINT32_MAX)
        {
            uploadBuffer->Unmap(0, nullptr);
            return 0;
        }

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
        cbvDesc.BufferLocation = uploadBuffer->GetGPUVirtualAddress();
        cbvDesc.SizeInBytes    = alignedSize;

        device->CreateConstantBufferView(&cbvDesc, cbvPool.GetCPUHandle(descIndex));

        D3D12ConstantBufferResource res{};
        res.uploadBuffer    = uploadBuffer;
        res.mappedData      = mappedData;
        res.byteSize        = byteSize;
        res.alignedSize     = alignedSize;
        res.gpuAddress      = uploadBuffer->GetGPUVirtualAddress();
        res.descriptorIndex = descIndex;

        NativeHandle handle = GenerateHandle();
        m_constantBuffers[handle] = res;
        return handle;
    }

    void D3D12ResourceManager::UpdateConstantBuffer(NativeHandle handle, const void* data, UINT32 byteSize)
    {
        if (!data || byteSize == 0) return;

        auto it = m_constantBuffers.find(handle);
        if (it != m_constantBuffers.end() && it->second.mappedData)
        {
            std::memcpy(it->second.mappedData, data, (std::min)(byteSize, it->second.byteSize));
        }
    }

    void D3D12ResourceManager::DestroyConstantBuffer(NativeHandle handle, D3D12DescriptorPool& cbvPool, UINT64 fenceValue)
    {
        (void)cbvPool;
        auto it = m_constantBuffers.find(handle);
        if (it != m_constantBuffers.end())
        {
            D3D12RetiredResource retired{};
            retired.fenceValue         = fenceValue;
            retired.resource           = std::move(it->second.uploadBuffer);
            retired.mappedData         = it->second.mappedData;
            retired.cbvDescriptorIndex = it->second.descriptorIndex;
            m_retiredResources.push_back(std::move(retired));

            m_constantBuffers.erase(it);
        }
    }

    NativeHandle D3D12ResourceManager::CreateTexture(
        D3D12DeviceContext& ctx,
        D3D12DescriptorPool& srvPool,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        const TextureDesc& desc,
        const void* initialData)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || desc.width == 0u || desc.height == 0u || desc.format == TextureFormat::UNKNOWN)
            return 0;
        if (desc.type == TextureType::TEXTURE_3D && desc.depth == 0u)
            return 0;
        if ((desc.type == TextureType::TEXTURE_2D_ARRAY || desc.type == TextureType::TEXTURE_CUBE_ARRAY) && desc.arrayLayers == 0u)
            return 0;
        if ((desc.type == TextureType::TEXTURE_CUBE || desc.type == TextureType::TEXTURE_CUBE_ARRAY) && desc.width != desc.height)
            return 0;

        // Enforce D3D12 hardware dimension limits BEFORE CreateCommittedResource so invalid parameters never trigger RemoveDevice
        if (desc.type == TextureType::TEXTURE_3D)
        {
            if (desc.width > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
                desc.height > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION ||
                desc.depth > D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION)
            {
                return 0;
            }
        }
        else
        {
            if (desc.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
                desc.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            {
                return 0;
            }
            const UINT32 totalArraySlicesCheck = ComputeTextureArraySliceCount(desc.type, desc.arrayLayers);
            if (totalArraySlicesCheck == 0u || totalArraySlicesCheck > D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION)
            {
                return 0;
            }
        }

        // Strict Format x Usage pre-validation
        const BOOL isDepthFmt = IsTextureFormatDepth(desc.format);
        const BOOL isBCnFmt   = IsTextureFormatBlockCompressed(desc.format);

        if (desc.usage == TextureUsage::DEPTH_STENCIL)
        {
            if (!isDepthFmt || desc.type == TextureType::TEXTURE_3D || initialData != nullptr)
                return 0;
        }
        else
        {
            if (isDepthFmt)
                return 0;
        }

        if (isBCnFmt)
        {
            if (desc.usage != TextureUsage::SAMPLED || desc.type == TextureType::TEXTURE_3D)
                return 0;
            if ((desc.width % 4u) != 0u || (desc.height % 4u) != 0u)
                return 0;
            if (desc.mipLevels == 0u && initialData != nullptr)
                return 0; // Block-compressed mips cannot be auto-generated from Mip 0 without a BCn encoder
        }

        if (desc.usage == TextureUsage::STORAGE &&
            (desc.type == TextureType::TEXTURE_CUBE || desc.type == TextureType::TEXTURE_CUBE_ARRAY))
        {
            return 0;
        }

        const UINT32 maxPossibleMips = ComputeTextureMaxMipLevels(desc.width, desc.height, desc.depth, desc.type);
        if (maxPossibleMips == 0u || desc.mipLevels > maxPossibleMips)
            return 0;

        const UINT32 resolvedMipLevels = (desc.mipLevels == 0u) ? maxPossibleMips : desc.mipLevels;
        const UINT32 arraySliceCount   = ComputeTextureArraySliceCount(desc.type, desc.arrayLayers);

        // Validate initialData byte size & prepare full-chain CPU upload staging if needed
        const SIZE_T mip0AllSlicesBytes      = ComputeTextureMip0AllSlicesTightBytes(desc.width, desc.height, desc.depth, desc.arrayLayers, desc.type, desc.format);
        const SIZE_T fullChainAllSlicesBytes = ComputeTextureFullChainTightBytes(desc.width, desc.height, desc.depth, resolvedMipLevels, desc.arrayLayers, desc.type, desc.format);
        BOOL callerProvidedFullChain         = false;

        if (initialData != nullptr)
        {
            if (mip0AllSlicesBytes == 0u)
                return 0;
            if (desc.initialDataSize != 0u && desc.initialDataSize < mip0AllSlicesBytes)
                return 0;

            if (resolvedMipLevels > 1u && desc.mipLevels > 1u && desc.initialDataSize >= fullChainAllSlicesBytes)
            {
                callerProvidedFullChain = true;
            }
            else if (resolvedMipLevels > 1u && !IsTextureFormatCPUFilterable(desc.format))
            {
                // Multi-mip upload with only Mip 0 provided requires a CPU-filterable format
                return 0;
            }
        }

        const DXGI_FORMAT baseFormat = ToDxgiFormat(desc.format);
        DXGI_FORMAT resFormat = baseFormat;
        DXGI_FORMAT srvFormat = baseFormat;
        DXGI_FORMAT uavFormat = baseFormat;
        DXGI_FORMAT rtvFormat = baseFormat;
        DXGI_FORMAT dsvFormat = DXGI_FORMAT_UNKNOWN;

        if (desc.usage == TextureUsage::DEPTH_STENCIL)
        {
            rtvFormat = DXGI_FORMAT_UNKNOWN;
            uavFormat = DXGI_FORMAT_UNKNOWN;
            switch (desc.format)
            {
                case TextureFormat::D32_FLOAT:
                    resFormat = DXGI_FORMAT_R32_TYPELESS;
                    dsvFormat = DXGI_FORMAT_D32_FLOAT;
                    srvFormat = DXGI_FORMAT_R32_FLOAT;
                    break;
                case TextureFormat::D16_UNORM:
                    resFormat = DXGI_FORMAT_R16_TYPELESS;
                    dsvFormat = DXGI_FORMAT_D16_UNORM;
                    srvFormat = DXGI_FORMAT_R16_UNORM;
                    break;
                case TextureFormat::D24_UNORM_S8:
                    resFormat = DXGI_FORMAT_R24G8_TYPELESS;
                    dsvFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
                    srvFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                    break;
                case TextureFormat::D32_FLOAT_S8:
                    resFormat = DXGI_FORMAT_R32G8X24_TYPELESS;
                    dsvFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                    srvFormat = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                    break;
                default:
                    return 0;
            }
        }
        else if (desc.usage == TextureUsage::STORAGE && desc.format == TextureFormat::RGBA8_SRGB)
        {
            // D3D12 forbids SRGB typed UAVs directly, so create a TYPELESS resource with SRGB SRV and UNORM UAV
            resFormat = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            srvFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            uavFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
            rtvFormat = DXGI_FORMAT_UNKNOWN;
        }
        else
        {
            if (desc.usage != TextureUsage::RENDER_TARGET) rtvFormat = DXGI_FORMAT_UNKNOWN;
            if (desc.usage != TextureUsage::STORAGE)       uavFormat = DXGI_FORMAT_UNKNOWN;
        }

        // Hardware CheckFeatureSupport validation before resource creation
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT srvSupport{ srvFormat, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE };
            if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &srvSupport, sizeof(srvSupport))))
                return 0;

            const D3D12_FORMAT_SUPPORT1 reqDimFlag =
                (desc.type == TextureType::TEXTURE_3D) ? D3D12_FORMAT_SUPPORT1_TEXTURE3D :
                ((desc.type == TextureType::TEXTURE_CUBE || desc.type == TextureType::TEXTURE_CUBE_ARRAY)
                    ? D3D12_FORMAT_SUPPORT1_TEXTURECUBE
                    : D3D12_FORMAT_SUPPORT1_TEXTURE2D);
            if ((srvSupport.Support1 & reqDimFlag) == 0)
                return 0;

            if (desc.usage == TextureUsage::RENDER_TARGET)
            {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT rtvSupport{ rtvFormat, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE };
                if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &rtvSupport, sizeof(rtvSupport))) ||
                    (rtvSupport.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) == 0)
                {
                    return 0;
                }
            }
            else if (desc.usage == TextureUsage::DEPTH_STENCIL)
            {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT dsvSupport{ dsvFormat, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE };
                if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &dsvSupport, sizeof(dsvSupport))) ||
                    (dsvSupport.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) == 0)
                {
                    return 0;
                }
            }
            else if (desc.usage == TextureUsage::STORAGE)
            {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT uavSupport{ uavFormat, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE };
                if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &uavSupport, sizeof(uavSupport))) ||
                    (uavSupport.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) == 0)
                {
                    return 0;
                }
            }
        }

        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension        = (desc.type == TextureType::TEXTURE_3D) ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Alignment        = 0;
        texDesc.Width            = desc.width;
        texDesc.Height           = desc.height;
        texDesc.DepthOrArraySize = static_cast<UINT16>((desc.type == TextureType::TEXTURE_3D) ? desc.depth : arraySliceCount);
        texDesc.MipLevels        = static_cast<UINT16>(resolvedMipLevels);
        texDesc.Format           = resFormat;
        texDesc.SampleDesc.Count   = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags              = D3D12_RESOURCE_FLAG_NONE;

        if (desc.usage == TextureUsage::RENDER_TARGET)
            texDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (desc.usage == TextureUsage::DEPTH_STENCIL)
            texDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (desc.usage == TextureUsage::STORAGE)
            texDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;

        D3D12_CLEAR_VALUE optClear{};
        D3D12_CLEAR_VALUE* pClearValue = nullptr;
        if (desc.usage == TextureUsage::RENDER_TARGET)
        {
            optClear.Format   = rtvFormat;
            optClear.Color[0] = 0.0f; optClear.Color[1] = 0.0f; optClear.Color[2] = 0.0f; optClear.Color[3] = 1.0f;
            pClearValue       = &optClear;
        }
        else if (desc.usage == TextureUsage::DEPTH_STENCIL)
        {
            optClear.Format               = dsvFormat;
            optClear.DepthStencil.Depth   = 1.0f;
            optClear.DepthStencil.Stencil = 0;
            pClearValue                   = &optClear;
        }

        D3D12_HEAP_PROPERTIES defaultHeap = GetDefaultHeapProperties();
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
        if (desc.usage == TextureUsage::RENDER_TARGET && desc.allowSharedHandle != FALSE)
        {
            texDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            heapFlags = D3D12_HEAP_FLAG_SHARED;
        }

        ComPtr<ID3D12Resource> texResource;
        HRESULT hrCreate = device->CreateCommittedResource(
            &defaultHeap,
            heapFlags,
            &texDesc,
            initialState,
            pClearValue,
            IID_PPV_ARGS(&texResource));

        if (FAILED(hrCreate) && heapFlags != D3D12_HEAP_FLAG_NONE)
        {
            texDesc.Flags &= ~D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            heapFlags = D3D12_HEAP_FLAG_NONE;
            hrCreate = device->CreateCommittedResource(
                &defaultHeap,
                heapFlags,
                &texDesc,
                initialState,
                pClearValue,
                IID_PPV_ARGS(&texResource));
        }

        if (FAILED(hrCreate) || !texResource)
        {
            return 0;
        }

        // Upload all subresources across all array/cubemap faces, 3D depth slices, and mip levels
        if (initialData != nullptr)
        {
            const UINT32 totalSubresources = resolvedMipLevels * arraySliceCount;
            std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(totalSubresources);
            std::vector<UINT32> numRows(totalSubresources, 0u);
            std::vector<UINT64> rowSizeInBytes(totalSubresources, 0ull);
            UINT64 uploadBufferSize = 0ull;

            device->GetCopyableFootprints(
                &texDesc,
                0u,
                totalSubresources,
                0ull,
                footprints.data(),
                numRows.data(),
                rowSizeInBytes.data(),
                &uploadBufferSize);

            if (uploadBufferSize == 0ull)
                return 0;

            // Build CPU subresource pointer table (either pointing directly into caller's full chain
            // or generating Mips 1..N-1 via exact polyphase linear/sRGB filtering per slice)
            std::vector<const BYTE*> subresSrcPtrs(totalSubresources, nullptr);
            std::vector<std::vector<BYTE>> generatedMips;

            const BYTE* callerBytes = static_cast<const BYTE*>(initialData);
            if (callerProvidedFullChain)
            {
                SIZE_T cursor = 0u;
                for (UINT32 slice = 0u; slice < arraySliceCount; ++slice)
                {
                    for (UINT32 mip = 0u; mip < resolvedMipLevels; ++mip)
                    {
                        SIZE_T tightRow = 0u, tightSub = 0u;
                        UINT32 tightRows = 0u, tightDepth = 0u;
                        if (!ComputeTextureSubresourceTightLayout(
                                desc.width, desc.height, desc.depth, desc.type, desc.format, mip,
                                tightRow, tightRows, tightDepth, tightSub))
                        {
                            return 0;
                        }
                        const UINT32 subIdx = mip + slice * resolvedMipLevels;
                        subresSrcPtrs[subIdx] = callerBytes + cursor;
                        cursor += tightSub;
                    }
                }
            }
            else
            {
                SIZE_T mip0TightRow = 0u, mip0TightSub = 0u;
                UINT32 mip0TightRows = 0u, mip0TightDepth = 0u;
                if (!ComputeTextureSubresourceTightLayout(
                        desc.width, desc.height, desc.depth, desc.type, desc.format, 0u,
                        mip0TightRow, mip0TightRows, mip0TightDepth, mip0TightSub))
                {
                    return 0;
                }

                if (resolvedMipLevels > 1u)
                {
                    generatedMips.reserve(static_cast<SIZE_T>(arraySliceCount) * static_cast<SIZE_T>(resolvedMipLevels - 1u));
                }

                for (UINT32 slice = 0u; slice < arraySliceCount; ++slice)
                {
                    const BYTE* prevMipPtr = callerBytes + static_cast<SIZE_T>(slice) * mip0TightSub;
                    subresSrcPtrs[slice * resolvedMipLevels] = prevMipPtr;

                    UINT32 prevW = desc.width;
                    UINT32 prevH = desc.height;
                    UINT32 prevD = (desc.type == TextureType::TEXTURE_3D) ? desc.depth : 1u;

                    for (UINT32 mip = 1u; mip < resolvedMipLevels; ++mip)
                    {
                        const UINT32 curW = (std::max)(1u, desc.width >> mip);
                        const UINT32 curH = (std::max)(1u, desc.height >> mip);
                        const UINT32 curD = (desc.type == TextureType::TEXTURE_3D) ? (std::max)(1u, desc.depth >> mip) : 1u;

                        SIZE_T curTightRow = 0u, curTightSub = 0u;
                        UINT32 curTightRows = 0u, curTightDepth = 0u;
                        if (!ComputeTextureSubresourceTightLayout(
                                desc.width, desc.height, desc.depth, desc.type, desc.format, mip,
                                curTightRow, curTightRows, curTightDepth, curTightSub))
                        {
                            return 0;
                        }

                        generatedMips.emplace_back(curTightSub);
                        BYTE* dstMipPtr = generatedMips.back().data();
                        DownsampleMipSlicePolyphase(
                            desc.format,
                            prevMipPtr, prevW, prevH, prevD,
                            dstMipPtr,  curW,  curH,  curD);

                        subresSrcPtrs[mip + slice * resolvedMipLevels] = dstMipPtr;
                        prevMipPtr = dstMipPtr;
                        prevW      = curW;
                        prevH      = curH;
                        prevD      = curD;
                    }
                }
            }

            D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
            D3D12_RESOURCE_DESC upBufferDesc = CreateBufferResourceDesc(uploadBufferSize);
            ComPtr<ID3D12Resource> uploadBuffer;

            if (FAILED(device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &upBufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer))) || !uploadBuffer)
            {
                return 0;
            }

            BYTE* pData = nullptr;
            D3D12_RANGE readRange{ 0, 0 };
            if (FAILED(uploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pData))) || !pData)
            {
                return 0;
            }

            for (UINT32 subIdx = 0u; subIdx < totalSubresources; ++subIdx)
            {
                const BYTE* pSrcBase = subresSrcPtrs[subIdx];
                if (!pSrcBase) continue;

                const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp = footprints[subIdx];
                const UINT32 rows        = numRows[subIdx];
                const SIZE_T rowBytes    = static_cast<SIZE_T>(rowSizeInBytes[subIdx]);
                const SIZE_T dstRowPitch = static_cast<SIZE_T>(fp.Footprint.RowPitch);
                const UINT32 depthSlices = (std::max)(1u, static_cast<UINT32>(fp.Footprint.Depth));
                const SIZE_T srcSliceStep = rowBytes * static_cast<SIZE_T>(rows);
                const SIZE_T dstSliceStep = dstRowPitch * static_cast<SIZE_T>(rows);
                BYTE* pDstBase = pData + fp.Offset;

                for (UINT32 z = 0u; z < depthSlices; ++z)
                {
                    const BYTE* srcSlice = pSrcBase + static_cast<SIZE_T>(z) * srcSliceStep;
                    BYTE*       dstSlice = pDstBase + static_cast<SIZE_T>(z) * dstSliceStep;
                    for (UINT32 r = 0u; r < rows; ++r)
                    {
                        std::memcpy(dstSlice + static_cast<SIZE_T>(r) * dstRowPitch,
                                    srcSlice + static_cast<SIZE_T>(r) * rowBytes,
                                    rowBytes);
                    }
                }
            }
            uploadBuffer->Unmap(0, nullptr);

            const D3D12_RESOURCE_STATES postUploadState =
                (desc.usage == TextureUsage::STORAGE)
                    ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                    : (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
                D3D12_RESOURCE_BARRIER toCopy = CreateTransitionBarrier(
                    texResource.Get(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST
                );
                cmdList->ResourceBarrier(1, &toCopy);

                for (UINT32 subIdx = 0u; subIdx < totalSubresources; ++subIdx)
                {
                    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
                    dstLoc.pResource        = texResource.Get();
                    dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    dstLoc.SubresourceIndex = subIdx;

                    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
                    srcLoc.pResource       = uploadBuffer.Get();
                    srcLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    srcLoc.PlacedFootprint = footprints[subIdx];

                    cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
                }

                D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(
                    texResource.Get(),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    postUploadState
                );
                cmdList->ResourceBarrier(1, &barrier);
            });
            initialState = postUploadState;
        }

        UINT32 srvDescIdx          = UINT32_MAX;
        UINT32 uavDescIdx          = UINT32_MAX;
        UINT32 rtvDescIdx          = UINT32_MAX;
        UINT32 dsvDescIdx          = UINT32_MAX;
        UINT32 readOnlyDsvDescIdx  = UINT32_MAX;
        UINT32 fullArrayDsvDescIdx = UINT32_MAX;
        std::vector<UINT32> mipSrvDescriptors;
        std::vector<UINT32> mipUavDescriptors;
        std::vector<UINT32> dsvLayerDescriptors;

        auto rollbackDescriptors = [&]() {
            if (srvDescIdx != UINT32_MAX) srvPool.Free(srvDescIdx);
            if (uavDescIdx != UINT32_MAX) srvPool.Free(uavDescIdx);
            for (UINT32 mSrv : mipSrvDescriptors)
            {
                if (mSrv != UINT32_MAX) srvPool.Free(mSrv);
            }
            for (SIZE_T m = 1u; m < mipUavDescriptors.size(); ++m)
            {
                if (mipUavDescriptors[m] != UINT32_MAX) srvPool.Free(mipUavDescriptors[m]);
            }
            if (rtvDescIdx != UINT32_MAX) rtvPool.Free(rtvDescIdx);
            if (dsvDescIdx != UINT32_MAX && dsvLayerDescriptors.empty()) dsvPool.Free(dsvDescIdx);
            if (readOnlyDsvDescIdx != UINT32_MAX) dsvPool.Free(readOnlyDsvDescIdx);
            if (fullArrayDsvDescIdx != UINT32_MAX) dsvPool.Free(fullArrayDsvDescIdx);
            for (UINT32 sliceDsv : dsvLayerDescriptors)
            {
                if (sliceDsv != UINT32_MAX) dsvPool.Free(sliceDsv);
            }
        };

        if (desc.usage == TextureUsage::SAMPLED ||
            desc.usage == TextureUsage::RENDER_TARGET ||
            desc.usage == TextureUsage::DEPTH_STENCIL ||
            desc.usage == TextureUsage::STORAGE)
        {
            srvDescIdx = srvPool.Allocate();
            if (srvDescIdx == UINT32_MAX)
            {
                rollbackDescriptors();
                return 0;
            }

            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
            srvDesc.Format                  = srvFormat;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            if (desc.type == TextureType::TEXTURE_3D)
            {
                srvDesc.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE3D;
                srvDesc.Texture3D.MipLevels = resolvedMipLevels;
            }
            else if (desc.type == TextureType::TEXTURE_CUBE)
            {
                srvDesc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURECUBE;
                srvDesc.TextureCube.MipLevels           = resolvedMipLevels;
                srvDesc.TextureCube.MostDetailedMip     = 0;
                srvDesc.TextureCube.ResourceMinLODClamp = 0.0f;
            }
            else if (desc.type == TextureType::TEXTURE_2D_ARRAY)
            {
                srvDesc.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                srvDesc.Texture2DArray.MipLevels       = resolvedMipLevels;
                srvDesc.Texture2DArray.ArraySize       = desc.arrayLayers;
            }
            else if (desc.type == TextureType::TEXTURE_CUBE_ARRAY)
            {
                srvDesc.ViewDimension              = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                srvDesc.TextureCubeArray.MipLevels = resolvedMipLevels;
                srvDesc.TextureCubeArray.NumCubes  = desc.arrayLayers;
            }
            else
            {
                srvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
                srvDesc.Texture2D.MipLevels           = resolvedMipLevels;
                srvDesc.Texture2D.MostDetailedMip     = 0;
                srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
            }

            device->CreateShaderResourceView(texResource.Get(), &srvDesc, srvPool.GetCpuStagingHandle(srvDescIdx));
            device->CreateShaderResourceView(texResource.Get(), &srvDesc, srvPool.GetCPUHandle(srvDescIdx));
        }

        if (desc.usage == TextureUsage::STORAGE)
        {
            uavDescIdx = srvPool.Allocate();
            if (uavDescIdx == UINT32_MAX)
            {
                rollbackDescriptors();
                return 0;
            }

            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
            uavDesc.Format = uavFormat;
            if (desc.type == TextureType::TEXTURE_3D)
            {
                uavDesc.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
                uavDesc.Texture3D.MipSlice    = 0;
                uavDesc.Texture3D.FirstWSlice = 0;
                uavDesc.Texture3D.WSize       = desc.depth > 0 ? desc.depth : 1;
            }
            else if (desc.type == TextureType::TEXTURE_2D_ARRAY)
            {
                uavDesc.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                uavDesc.Texture2DArray.MipSlice        = 0;
                uavDesc.Texture2DArray.FirstArraySlice = 0;
                uavDesc.Texture2DArray.ArraySize       = desc.arrayLayers > 0 ? desc.arrayLayers : 1;
                uavDesc.Texture2DArray.PlaneSlice      = 0;
            }
            else
            {
                uavDesc.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE2D;
                uavDesc.Texture2D.MipSlice   = 0;
                uavDesc.Texture2D.PlaneSlice = 0;
            }

            device->CreateUnorderedAccessView(texResource.Get(), nullptr, &uavDesc, srvPool.GetCpuStagingHandle(uavDescIdx));
            device->CreateUnorderedAccessView(texResource.Get(), nullptr, &uavDesc, srvPool.GetCPUHandle(uavDescIdx));

            // Allocate per-mip SRV and UAV descriptors for multi-mip storage textures (2D, 2D Array, 3D)
            if (resolvedMipLevels > 1u)
            {
                mipSrvDescriptors.assign(resolvedMipLevels, UINT32_MAX);
                mipUavDescriptors.assign(resolvedMipLevels, UINT32_MAX);
                mipUavDescriptors[0] = uavDescIdx;

                for (UINT32 m = 0u; m < resolvedMipLevels; ++m)
                {
                    const UINT32 mSrv = srvPool.Allocate();
                    if (mSrv == UINT32_MAX)
                    {
                        rollbackDescriptors();
                        return 0;
                    }
                    mipSrvDescriptors[m] = mSrv;

                    D3D12_SHADER_RESOURCE_VIEW_DESC mSrvDesc{};
                    mSrvDesc.Format                  = srvFormat;
                    mSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    if (desc.type == TextureType::TEXTURE_3D)
                    {
                        mSrvDesc.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE3D;
                        mSrvDesc.Texture3D.MostDetailedMip = m;
                        mSrvDesc.Texture3D.MipLevels       = 1u;
                    }
                    else if (desc.type == TextureType::TEXTURE_2D_ARRAY)
                    {
                        mSrvDesc.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                        mSrvDesc.Texture2DArray.MostDetailedMip = m;
                        mSrvDesc.Texture2DArray.MipLevels       = 1u;
                        mSrvDesc.Texture2DArray.ArraySize       = desc.arrayLayers;
                    }
                    else
                    {
                        mSrvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
                        mSrvDesc.Texture2D.MostDetailedMip     = m;
                        mSrvDesc.Texture2D.MipLevels           = 1u;
                        mSrvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
                    }
                    device->CreateShaderResourceView(texResource.Get(), &mSrvDesc, srvPool.GetCpuStagingHandle(mSrv));
                    device->CreateShaderResourceView(texResource.Get(), &mSrvDesc, srvPool.GetCPUHandle(mSrv));

                    if (m > 0u)
                    {
                        const UINT32 mUav = srvPool.Allocate();
                        if (mUav == UINT32_MAX)
                        {
                            rollbackDescriptors();
                            return 0;
                        }
                        mipUavDescriptors[m] = mUav;

                        D3D12_UNORDERED_ACCESS_VIEW_DESC mUavDesc{};
                        mUavDesc.Format = uavFormat;
                        if (desc.type == TextureType::TEXTURE_3D)
                        {
                            mUavDesc.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
                            mUavDesc.Texture3D.MipSlice    = m;
                            mUavDesc.Texture3D.FirstWSlice = 0u;
                            mUavDesc.Texture3D.WSize       = (std::max)(1u, desc.depth >> m);
                        }
                        else if (desc.type == TextureType::TEXTURE_2D_ARRAY)
                        {
                            mUavDesc.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                            mUavDesc.Texture2DArray.MipSlice        = m;
                            mUavDesc.Texture2DArray.FirstArraySlice = 0u;
                            mUavDesc.Texture2DArray.ArraySize       = desc.arrayLayers;
                            mUavDesc.Texture2DArray.PlaneSlice      = 0u;
                        }
                        else
                        {
                            mUavDesc.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE2D;
                            mUavDesc.Texture2D.MipSlice   = m;
                            mUavDesc.Texture2D.PlaneSlice = 0u;
                        }
                        device->CreateUnorderedAccessView(texResource.Get(), nullptr, &mUavDesc, srvPool.GetCpuStagingHandle(mUav));
                        device->CreateUnorderedAccessView(texResource.Get(), nullptr, &mUavDesc, srvPool.GetCPUHandle(mUav));
                    }
                }
            }
        }

        if (desc.usage == TextureUsage::RENDER_TARGET)
        {
            rtvDescIdx = rtvPool.Allocate();
            if (rtvDescIdx == UINT32_MAX)
            {
                rollbackDescriptors();
                return 0;
            }
            D3D12_RENDER_TARGET_VIEW_DESC rtvDesc{};
            rtvDesc.Format        = rtvFormat;
            rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            device->CreateRenderTargetView(texResource.Get(), &rtvDesc, rtvPool.GetCPUHandle(rtvDescIdx));
        }

        if (desc.usage == TextureUsage::DEPTH_STENCIL)
        {
            if (desc.type == TextureType::TEXTURE_2D_ARRAY)
            {
                fullArrayDsvDescIdx = dsvPool.Allocate();
                if (fullArrayDsvDescIdx == UINT32_MAX)
                {
                    rollbackDescriptors();
                    return 0;
                }
                D3D12_DEPTH_STENCIL_VIEW_DESC fullDsvDesc{};
                fullDsvDesc.Format                         = dsvFormat;
                fullDsvDesc.ViewDimension                  = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                fullDsvDesc.Texture2DArray.MipSlice        = 0;
                fullDsvDesc.Texture2DArray.FirstArraySlice = 0;
                fullDsvDesc.Texture2DArray.ArraySize       = desc.arrayLayers;
                device->CreateDepthStencilView(texResource.Get(), &fullDsvDesc, dsvPool.GetCPUHandle(fullArrayDsvDescIdx));

                dsvLayerDescriptors.reserve(desc.arrayLayers);
                for (UINT32 slice = 0; slice < desc.arrayLayers; ++slice)
                {
                    const UINT32 sliceDsv = dsvPool.Allocate();
                    if (sliceDsv == UINT32_MAX)
                    {
                        rollbackDescriptors();
                        return 0;
                    }
                    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
                    dsvDesc.Format                         = dsvFormat;
                    dsvDesc.ViewDimension                  = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                    dsvDesc.Texture2DArray.MipSlice        = 0;
                    dsvDesc.Texture2DArray.FirstArraySlice = slice;
                    dsvDesc.Texture2DArray.ArraySize       = 1;
                    device->CreateDepthStencilView(texResource.Get(), &dsvDesc, dsvPool.GetCPUHandle(sliceDsv));
                    dsvLayerDescriptors.push_back(sliceDsv);
                }
                if (!dsvLayerDescriptors.empty())
                {
                    dsvDescIdx = dsvLayerDescriptors[0];
                }
            }
            else
            {
                dsvDescIdx = dsvPool.Allocate();
                readOnlyDsvDescIdx = dsvPool.Allocate();
                if (dsvDescIdx == UINT32_MAX || readOnlyDsvDescIdx == UINT32_MAX)
                {
                    rollbackDescriptors();
                    return 0;
                }

                D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
                dsvDesc.Format        = dsvFormat;
                dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                device->CreateDepthStencilView(texResource.Get(), &dsvDesc, dsvPool.GetCPUHandle(dsvDescIdx));

                D3D12_DEPTH_STENCIL_VIEW_DESC roDsvDesc{};
                roDsvDesc.Format        = dsvFormat;
                roDsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                roDsvDesc.Flags         = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
                if (IsTextureFormatDepthStencil(desc.format))
                {
                    roDsvDesc.Flags |= D3D12_DSV_FLAG_READ_ONLY_STENCIL;
                }
                device->CreateDepthStencilView(texResource.Get(), &roDsvDesc, dsvPool.GetCPUHandle(readOnlyDsvDescIdx));
            }
        }

        HANDLE sharedHandle = nullptr;
        if (desc.usage == TextureUsage::RENDER_TARGET && desc.allowSharedHandle != FALSE && (heapFlags & D3D12_HEAP_FLAG_SHARED))
        {
            device->CreateSharedHandle(texResource.Get(), nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        }

        D3D12TextureResource res{};
        res.resource                    = texResource;
        res.desc                        = desc;
        res.desc.mipLevels              = resolvedMipLevels;
        res.format                      = resFormat;
        res.srvFormat                   = srvFormat;
        res.uavFormat                   = uavFormat;
        res.rtvFormat                   = rtvFormat;
        res.dsvFormat                   = dsvFormat;
        res.currentState                = initialState;
        res.srvDescriptorIndex          = srvDescIdx;
        res.uavDescriptorIndex          = uavDescIdx;
        res.mipSrvDescriptors           = std::move(mipSrvDescriptors);
        res.mipUavDescriptors           = std::move(mipUavDescriptors);
        res.rtvDescriptorIndex          = rtvDescIdx;
        res.dsvDescriptorIndex          = dsvDescIdx;
        res.readOnlyDsvDescriptorIndex  = readOnlyDsvDescIdx;
        res.fullArrayDsvDescriptorIndex = fullArrayDsvDescIdx;
        res.dsvLayerDescriptors         = std::move(dsvLayerDescriptors);
        res.sharedHandle                = sharedHandle;

        NativeHandle handle = GenerateHandle();
        if (texResource)
        {
            const wchar_t* usageTag = L"Sampled";
            if (desc.usage == TextureUsage::DEPTH_STENCIL)
            {
                usageTag = (desc.type == TextureType::TEXTURE_2D_ARRAY) ? L"DepthArray" : L"Depth2D";
            }
            else if (desc.usage == TextureUsage::RENDER_TARGET)
            {
                usageTag = L"RenderTarget";
            }
            else if (desc.usage == TextureUsage::STORAGE)
            {
                usageTag = L"Storage";
            }

            wchar_t texDebugName[128]{};
            std::swprintf(
                texDebugName,
                sizeof(texDebugName) / sizeof(texDebugName[0]),
                L"Northstar.Tex.%ls_%ux%ux%u_M%u_H%llu",
                usageTag,
                desc.width,
                desc.height,
                (std::max)(1u, arraySliceCount),
                resolvedMipLevels,
                static_cast<unsigned long long>(handle));
            texResource->SetName(texDebugName);
        }
        m_textures[handle] = std::move(res);
        return handle;
    }

    BOOL D3D12ResourceManager::UpdateTextureSubresource(
        D3D12DeviceContext& ctx,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle handle,
        UINT32 mipSlice,
        UINT32 arraySlice,
        const void* data,
        SIZE_T dataByteSize)
    {
        if (!data || dataByteSize == 0u) return false;

        auto it = m_textures.find(handle);
        if (it == m_textures.end() || !it->second.resource) return false;

        ID3D12Device* device = ctx.GetDevice();
        if (!device) return false;

        D3D12TextureResource& tex = it->second;
        if (IsTextureFormatDepth(tex.desc.format)) return false;

        const UINT32 mipCount   = (std::max)(1u, tex.desc.mipLevels);
        const UINT32 sliceCount = ComputeTextureArraySliceCount(tex.desc.type, tex.desc.arrayLayers);
        if (mipSlice >= mipCount || arraySlice >= sliceCount)
            return false;

        SIZE_T tightRowBytes = 0u, tightSubBytes = 0u;
        UINT32 tightRows = 0u, tightDepth = 0u;
        if (!ComputeTextureSubresourceTightLayout(
                tex.desc.width, tex.desc.height, tex.desc.depth, tex.desc.type, tex.desc.format, mipSlice,
                tightRowBytes, tightRows, tightDepth, tightSubBytes))
        {
            return false;
        }
        if (dataByteSize < tightSubBytes)
            return false;

        D3D12_RESOURCE_DESC texDesc = tex.resource->GetDesc();
        const UINT32 subresourceIndex = mipSlice + arraySlice * mipCount;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT32 numRows = 0u;
        UINT64 rowSizeInBytes = 0ull;
        UINT64 uploadBytes = 0ull;
        device->GetCopyableFootprints(&texDesc, subresourceIndex, 1u, 0ull, &footprint, &numRows, &rowSizeInBytes, &uploadBytes);
        if (uploadBytes == 0ull || numRows == 0u || rowSizeInBytes == 0ull)
            return false;

        D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC upDesc = CreateBufferResourceDesc(uploadBytes);
        ComPtr<ID3D12Resource> uploadBuffer;
        if (FAILED(device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &upDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer))) || !uploadBuffer)
        {
            return false;
        }

        BYTE* mapped = nullptr;
        D3D12_RANGE emptyRead{ 0, 0 };
        if (FAILED(uploadBuffer->Map(0, &emptyRead, reinterpret_cast<void**>(&mapped))) || !mapped)
            return false;

        const BYTE* srcBase       = static_cast<const BYTE*>(data);
        BYTE*       dstBase       = mapped + footprint.Offset;
        const SIZE_T rowBytes     = static_cast<SIZE_T>(rowSizeInBytes);
        const SIZE_T dstRowPitch  = static_cast<SIZE_T>(footprint.Footprint.RowPitch);
        const UINT32 depthSlices  = (std::max)(1u, static_cast<UINT32>(footprint.Footprint.Depth));
        const SIZE_T srcSliceStep = rowBytes * static_cast<SIZE_T>(numRows);
        const SIZE_T dstSliceStep = dstRowPitch * static_cast<SIZE_T>(numRows);

        for (UINT32 z = 0u; z < depthSlices; ++z)
        {
            const BYTE* srcSlice = srcBase + static_cast<SIZE_T>(z) * srcSliceStep;
            BYTE*       dstSlice = dstBase + static_cast<SIZE_T>(z) * dstSliceStep;
            for (UINT32 r = 0u; r < numRows; ++r)
            {
                std::memcpy(dstSlice + static_cast<SIZE_T>(r) * dstRowPitch,
                            srcSlice + static_cast<SIZE_T>(r) * rowBytes,
                            rowBytes);
            }
        }
        uploadBuffer->Unmap(0, nullptr);

        const GPUResourceAccess restoreAccess =
            (tex.desc.usage == TextureUsage::STORAGE)
                ? GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE
                : GPUResourceAccess::ALL_SHADER_RESOURCE;

        ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
            const BOOL isTracked = stateTracker.IsResourceTracked(tex.resource.Get());
            if (isTracked)
            {
                stateTracker.TransitionSubresource(tex.resource.Get(), subresourceIndex, GPUResourceAccess::COPY_DEST);
                stateTracker.FlushBarriers(cmdList);
            }
            else if (tex.currentState != D3D12_RESOURCE_STATE_COPY_DEST)
            {
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(tex.resource.Get(), tex.currentState, D3D12_RESOURCE_STATE_COPY_DEST);
                cmdList->ResourceBarrier(1, &b);
            }

            D3D12_TEXTURE_COPY_LOCATION dstLoc{};
            dstLoc.pResource        = tex.resource.Get();
            dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex = subresourceIndex;

            D3D12_TEXTURE_COPY_LOCATION srcLoc{};
            srcLoc.pResource       = uploadBuffer.Get();
            srcLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            srcLoc.PlacedFootprint = footprint;
            cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

            if (isTracked)
            {
                stateTracker.TransitionSubresource(tex.resource.Get(), subresourceIndex, restoreAccess);
                stateTracker.FlushBarriers(cmdList);
                tex.currentState = stateTracker.GetResourceState(tex.resource.Get());
            }
            else
            {
                const D3D12_RESOURCE_STATES afterState =
                    (tex.desc.usage == TextureUsage::STORAGE)
                        ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                        : (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(tex.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, afterState);
                cmdList->ResourceBarrier(1, &b);
                tex.currentState = afterState;
            }
        });

        return true;
    }

    BOOL D3D12ResourceManager::GenerateTextureMips(
        D3D12DeviceContext& ctx,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle handle)
    {
        auto it = m_textures.find(handle);
        if (it == m_textures.end() || !it->second.resource) return false;

        D3D12TextureResource& tex = it->second;
        if (!IsTextureFormatCPUFilterable(tex.desc.format))
            return false;

        const UINT32 mipCount = (std::max)(1u, tex.desc.mipLevels);
        if (mipCount <= 1u)
            return true;

        const UINT32 sliceCount = ComputeTextureArraySliceCount(tex.desc.type, tex.desc.arrayLayers);
        SIZE_T mip0Row = 0u, mip0Bytes = 0u;
        UINT32 mip0Rows = 0u, mip0Depth = 0u;
        if (!ComputeTextureSubresourceTightLayout(
                tex.desc.width, tex.desc.height, tex.desc.depth, tex.desc.type, tex.desc.format, 0u,
                mip0Row, mip0Rows, mip0Depth, mip0Bytes) || mip0Bytes > 0x7FFFFFFFu)
        {
            return false;
        }

        std::vector<BYTE> mip0Buf(mip0Bytes);
        std::vector<BYTE> prevBuf;
        std::vector<BYTE> nextBuf;

        for (UINT32 slice = 0u; slice < sliceCount; ++slice)
        {
            if (!ReadbackTextureSubresourceSync(
                    ctx, stateTracker, handle, 0u, slice, mip0Buf.data(), static_cast<UINT32>(mip0Bytes)))
            {
                return false;
            }

            prevBuf = mip0Buf;
            UINT32 prevW = tex.desc.width;
            UINT32 prevH = tex.desc.height;
            UINT32 prevD = (tex.desc.type == TextureType::TEXTURE_3D) ? tex.desc.depth : 1u;

            for (UINT32 mip = 1u; mip < mipCount; ++mip)
            {
                const UINT32 curW = (std::max)(1u, tex.desc.width >> mip);
                const UINT32 curH = (std::max)(1u, tex.desc.height >> mip);
                const UINT32 curD = (tex.desc.type == TextureType::TEXTURE_3D) ? (std::max)(1u, tex.desc.depth >> mip) : 1u;

                SIZE_T curRow = 0u, curBytes = 0u;
                UINT32 curRows = 0u, curDepth = 0u;
                if (!ComputeTextureSubresourceTightLayout(
                        tex.desc.width, tex.desc.height, tex.desc.depth, tex.desc.type, tex.desc.format, mip,
                        curRow, curRows, curDepth, curBytes))
                {
                    return false;
                }

                nextBuf.resize(curBytes);
                DownsampleMipSlicePolyphase(
                    tex.desc.format,
                    prevBuf.data(), prevW, prevH, prevD,
                    nextBuf.data(), curW,  curH,  curD);

                if (!UpdateTextureSubresource(ctx, stateTracker, handle, mip, slice, nextBuf.data(), curBytes))
                {
                    return false;
                }

                prevBuf.swap(nextBuf);
                prevW = curW;
                prevH = curH;
                prevD = curD;
            }
        }

        return true;
    }

    void D3D12ResourceManager::DestroyTexture(
        NativeHandle handle,
        D3D12DescriptorPool& srvPool,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        UINT64 fenceValue)
    {
        (void)srvPool;
        (void)rtvPool;
        (void)dsvPool;
        auto it = m_textures.find(handle);
        if (it != m_textures.end())
        {
            D3D12RetiredResource retired{};
            retired.fenceValue                  = fenceValue;
            retired.resource                    = std::move(it->second.resource);
            retired.sharedHandle                = it->second.sharedHandle;
            it->second.sharedHandle             = nullptr;
            retired.srvDescriptorIndex          = it->second.srvDescriptorIndex;
            retired.uavDescriptorIndex          = it->second.uavDescriptorIndex;
            for (UINT32 mSrv : it->second.mipSrvDescriptors)
            {
                if (mSrv != UINT32_MAX) retired.extraSrvUavDescriptors.push_back(mSrv);
            }
            for (SIZE_T m = 1; m < it->second.mipUavDescriptors.size(); ++m)
            {
                if (it->second.mipUavDescriptors[m] != UINT32_MAX)
                    retired.extraSrvUavDescriptors.push_back(it->second.mipUavDescriptors[m]);
            }
            retired.rtvDescriptorIndex          = it->second.rtvDescriptorIndex;
            retired.dsvDescriptorIndex          = it->second.dsvDescriptorIndex;
            retired.readOnlyDsvDescriptorIndex  = it->second.readOnlyDsvDescriptorIndex;
            retired.fullArrayDsvDescriptorIndex = it->second.fullArrayDsvDescriptorIndex;
            retired.dsvLayerDescriptors         = it->second.dsvLayerDescriptors;
            m_retiredResources.push_back(std::move(retired));

            m_textures.erase(it);
        }
    }

    NativeHandle D3D12ResourceManager::CreateSampler(
        D3D12DeviceContext& ctx,
        D3D12DescriptorPool& samplerPool,
        const SamplerDesc& desc)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device) return 0;

        UINT32 samplerIdx = samplerPool.Allocate();
        if (samplerIdx == UINT32_MAX) return 0;

        D3D12_SAMPLER_DESC sDesc{};
        sDesc.Filter         = ToD3D12Filter(desc.minFilter, desc.magFilter, desc.mipFilter, desc.comparisonMode != SamplerComparisonMode::NONE);
        sDesc.AddressU       = ToD3D12AddressMode(desc.wrapU);
        sDesc.AddressV       = ToD3D12AddressMode(desc.wrapV);
        sDesc.AddressW       = ToD3D12AddressMode(desc.wrapW);
        sDesc.MipLODBias     = 0.0f;
        sDesc.MaxAnisotropy  = desc.maxAnisotropy > 0 ? desc.maxAnisotropy : 1;
        sDesc.ComparisonFunc = ToD3D12ComparisonFunc(desc.comparisonMode);
        sDesc.MinLOD         = 0.0f;
        sDesc.MaxLOD         = D3D12_FLOAT32_MAX;

        for (INT32 i = 0; i < 4; ++i) sDesc.BorderColor[i] = 0.0f;

        device->CreateSampler(&sDesc, samplerPool.GetCpuStagingHandle(samplerIdx));
        device->CreateSampler(&sDesc, samplerPool.GetCPUHandle(samplerIdx));

        D3D12SamplerResource res{};
        res.desc                   = desc;
        res.samplerDescriptorIndex = samplerIdx;

        NativeHandle handle = GenerateHandle();
        m_samplers[handle] = res;
        return handle;
    }

    void D3D12ResourceManager::DestroySampler(NativeHandle handle, D3D12DescriptorPool& samplerPool, UINT64 fenceValue)
    {
        (void)samplerPool;
        auto it = m_samplers.find(handle);
        if (it != m_samplers.end())
        {
            D3D12RetiredResource retired{};
            retired.fenceValue             = fenceValue;
            retired.samplerDescriptorIndex = it->second.samplerDescriptorIndex;
            m_retiredResources.push_back(std::move(retired));

            m_samplers.erase(it);
        }
    }

    void D3D12ResourceManager::ReclaimCompletedUploadRingRegions(UINT64 completedFenceValue)
    {
        auto it = m_uploadRingInFlight.begin();
        while (it != m_uploadRingInFlight.end())
        {
            if (it->fenceValue != UINT64_MAX && completedFenceValue >= it->fenceValue)
            {
                it = m_uploadRingInFlight.erase(it);
            }
            else
            {
                break;
            }
        }

        if (m_uploadRingInFlight.empty())
        {
            m_uploadRingReadOffset  = 0;
            m_uploadRingWriteOffset = 0;
        }
        else
        {
            m_uploadRingReadOffset = m_uploadRingInFlight.front().offset;
        }
    }

    BOOL D3D12ResourceManager::AllocateFromUploadRing(
        D3D12DeviceContext& ctx,
        NativeHandle targetHandle,
        UINT32 byteSize,
        ComPtr<ID3D12Resource>& outUploadResource,
        UINT64& outSrcOffset,
        void*& outMappedPtr,
        BOOL& outIsStandalone)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || byteSize == 0) return false;

        const UINT64 completedFence = ctx.GetFrameFence() ? ctx.GetFrameFence()->GetCompletedValue() : 0;
        ReclaimCompletedUploadRingRegions(completedFence);

        if (!m_uploadRingBuffer)
        {
            D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
            D3D12_RESOURCE_DESC ringDesc = CreateBufferResourceDesc(kUploadRingCapacity);
            if (SUCCEEDED(device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &ringDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_uploadRingBuffer))))
            {
                D3D12_RANGE emptyRead{ 0, 0 };
                void* mapped = nullptr;
                if (SUCCEEDED(m_uploadRingBuffer->Map(0, &emptyRead, &mapped)))
                {
                    m_uploadRingMappedData = static_cast<BYTE*>(mapped);
                }
                else
                {
                    m_uploadRingBuffer.Reset();
                }
            }
        }

        const UINT64 alignedSize = (static_cast<UINT64>(byteSize) + 255ULL) & ~255ULL;

        if (m_uploadRingBuffer && m_uploadRingMappedData && alignedSize <= kUploadRingCapacity)
        {
            BOOL canSuballocate = false;
            UINT64 allocOffset = 0;

            if (m_uploadRingInFlight.empty())
            {
                allocOffset = 0;
                m_uploadRingWriteOffset = alignedSize;
                canSuballocate = true;
            }
            else if (m_uploadRingWriteOffset > m_uploadRingReadOffset)
            {
                if (m_uploadRingWriteOffset + alignedSize <= kUploadRingCapacity)
                {
                    allocOffset = m_uploadRingWriteOffset;
                    m_uploadRingWriteOffset += alignedSize;
                    canSuballocate = true;
                }
                else if (alignedSize <= m_uploadRingReadOffset)
                {
                    allocOffset = 0;
                    m_uploadRingWriteOffset = alignedSize;
                    canSuballocate = true;
                }
            }
            else if (m_uploadRingWriteOffset < m_uploadRingReadOffset)
            {
                // Write offset has wrapped behind read offset: never overwrite [m_uploadRingReadOffset .. in-flight]
                // Note: when m_uploadRingWriteOffset == m_uploadRingReadOffset and !m_uploadRingInFlight.empty(),
                // the ring is 100% full, so canSuballocate remains false and falls back to standalone upload buffer.
                if (m_uploadRingWriteOffset + alignedSize <= m_uploadRingReadOffset)
                {
                    allocOffset = m_uploadRingWriteOffset;
                    m_uploadRingWriteOffset += alignedSize;
                    canSuballocate = true;
                }
            }

            if (canSuballocate)
            {
                D3D12UploadRingRegion region{};
                region.targetHandle = targetHandle;
                region.fenceValue   = UINT64_MAX;
                region.offset       = allocOffset;
                region.endOffset    = allocOffset + alignedSize;
                m_uploadRingInFlight.push_back(region);

                outUploadResource = m_uploadRingBuffer;
                outSrcOffset      = allocOffset;
                outMappedPtr      = m_uploadRingMappedData + allocOffset;
                outIsStandalone   = false;
                return true;
            }
        }

        // Fallback when ring is full of in-flight GPU work or upload exceeds ring size:
        // Allocate a dedicated transient UPLOAD buffer retired by fence completion.
        D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC tempDesc = CreateBufferResourceDesc(alignedSize);
        ComPtr<ID3D12Resource> tempUpload;
        if (FAILED(device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &tempDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&tempUpload))))
        {
            return false;
        }

        D3D12_RANGE emptyRead{ 0, 0 };
        void* mapped = nullptr;
        if (FAILED(tempUpload->Map(0, &emptyRead, &mapped)) || !mapped)
        {
            return false;
        }

        outUploadResource = tempUpload;
        outSrcOffset      = 0;
        outMappedPtr      = mapped;
        outIsStandalone   = true;
        return true;
    }

    NativeHandle D3D12ResourceManager::CreateGPUBuffer(
        D3D12DeviceContext& ctx,
        D3D12DescriptorPool& srvUavPool,
        D3D12ResourceStateTracker& stateTracker,
        const GPUBufferDesc& desc)
    {
        ID3D12Device* device = ctx.GetDevice();
        if (!device || desc.elementCount == 0 || desc.stride == 0 || desc.usage == GPUBufferUsage::NONE)
        {
            return 0;
        }

        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::RAW_BYTE_ADDRESS) && (desc.stride % 4) != 0)
        {
            return 0;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(desc.elementCount) * static_cast<UINT64>(desc.stride);
        if (totalBytes64 == 0 || totalBytes64 > 0x7FFFFFFFULL)
        {
            return 0;
        }
        const UINT32 byteSize = static_cast<UINT32>(totalBytes64);

        D3D12_HEAP_PROPERTIES defaultHeap = GetDefaultHeapProperties();
        D3D12_RESOURCE_DESC bufDesc = CreateBufferResourceDesc(byteSize);
        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_UAV))
        {
            bufDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        GPUResourceAccess defaultAccess = GPUResourceAccess::ALL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES targetInitialState =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_SRV) ||
            HasGPUBufferUsage(desc.usage, GPUBufferUsage::RAW_BYTE_ADDRESS))
        {
            defaultAccess = GPUResourceAccess::ALL_SHADER_RESOURCE;
            targetInitialState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
        else if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_UAV))
        {
            defaultAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
            targetInitialState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        else if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT))
        {
            defaultAccess = GPUResourceAccess::INDIRECT_ARGUMENT;
            targetInitialState = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        }
        else
        {
            defaultAccess = GPUResourceAccess::COPY_SOURCE;
            targetInitialState = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }

        ComPtr<ID3D12Resource> defaultBuffer;
        if (FAILED(device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &bufDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&defaultBuffer))))
        {
            return 0;
        }

        D3D12_RESOURCE_STATES finalCreatedState = D3D12_RESOURCE_STATE_COMMON;
        GPUResourceAccess finalCreatedAccess = GPUResourceAccess::COMMON;

        if (desc.initialData)
        {
            D3D12_HEAP_PROPERTIES uploadHeap = GetUploadHeapProperties();
            D3D12_RESOURCE_DESC upDesc = CreateBufferResourceDesc(byteSize);
            ComPtr<ID3D12Resource> uploadBuffer;
            if (FAILED(device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &upDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&uploadBuffer))))
            {
                return 0;
            }

            void* mapped = nullptr;
            D3D12_RANGE emptyRead{ 0, 0 };
            if (FAILED(uploadBuffer->Map(0, &emptyRead, &mapped)) || !mapped)
            {
                return 0;
            }
            std::memcpy(mapped, desc.initialData, byteSize);
            uploadBuffer->Unmap(0, nullptr);

            ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
                D3D12_RESOURCE_BARRIER toCopy = CreateTransitionBarrier(
                    defaultBuffer.Get(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST
                );
                cmdList->ResourceBarrier(1, &toCopy);
                cmdList->CopyBufferRegion(defaultBuffer.Get(), 0, uploadBuffer.Get(), 0, byteSize);
                if (targetInitialState != D3D12_RESOURCE_STATE_COPY_DEST)
                {
                    D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(
                        defaultBuffer.Get(),
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        targetInitialState
                    );
                    cmdList->ResourceBarrier(1, &barrier);
                }
            });

            finalCreatedState  = targetInitialState;
            finalCreatedAccess = defaultAccess;
        }

        D3D12GPUBufferResource res{};
        res.resource      = defaultBuffer;
        res.desc          = desc;
        res.byteSize      = byteSize;
        res.currentState  = finalCreatedState;
        res.defaultAccess = defaultAccess;

        // Readback staging buffers are allocated lazily on first RequestGPUBufferReadback / ReadbackGPUBufferSync
        // so large CPU_READBACK-capable buffers (e.g., 100K-object GPUScene) pay 0 MB readback heap unless actually read back.

        const BOOL isRawOnly = HasGPUBufferUsage(desc.usage, GPUBufferUsage::RAW_BYTE_ADDRESS) &&
                               !HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_SRV) &&
                               !HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_UAV);

        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_SRV) ||
            HasGPUBufferUsage(desc.usage, GPUBufferUsage::RAW_BYTE_ADDRESS))
        {
            res.srvDescriptorIndex = srvUavPool.Allocate();
            if (res.srvDescriptorIndex == UINT32_MAX)
            {
                return 0;
            }

            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
            srvDesc.ViewDimension           = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDesc.Buffer.FirstElement     = 0;
            if (isRawOnly)
            {
                srvDesc.Format                     = DXGI_FORMAT_R32_TYPELESS;
                srvDesc.Buffer.NumElements         = byteSize / 4;
                srvDesc.Buffer.StructureByteStride = 0;
                srvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_RAW;
            }
            else
            {
                srvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                srvDesc.Buffer.NumElements         = desc.elementCount;
                srvDesc.Buffer.StructureByteStride = desc.stride;
                srvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
            }

            device->CreateShaderResourceView(
                defaultBuffer.Get(),
                &srvDesc,
                srvUavPool.GetCpuStagingHandle(res.srvDescriptorIndex)
            );
            device->CreateShaderResourceView(
                defaultBuffer.Get(),
                &srvDesc,
                srvUavPool.GetCPUHandle(res.srvDescriptorIndex)
            );
        }

        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::STRUCTURED_UAV))
        {
            res.uavDescriptorIndex = srvUavPool.Allocate();
            if (res.uavDescriptorIndex == UINT32_MAX)
            {
                if (res.srvDescriptorIndex != UINT32_MAX)
                {
                    srvUavPool.Free(res.srvDescriptorIndex);
                }
                return 0;
            }

            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
            uavDesc.ViewDimension              = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.FirstElement        = 0;
            uavDesc.Buffer.CounterOffsetInBytes = 0;
            if (isRawOnly)
            {
                uavDesc.Format                     = DXGI_FORMAT_R32_TYPELESS;
                uavDesc.Buffer.NumElements         = byteSize / 4;
                uavDesc.Buffer.StructureByteStride = 0;
                uavDesc.Buffer.Flags               = D3D12_BUFFER_UAV_FLAG_RAW;
            }
            else
            {
                uavDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                uavDesc.Buffer.NumElements         = desc.elementCount;
                uavDesc.Buffer.StructureByteStride = desc.stride;
                uavDesc.Buffer.Flags               = D3D12_BUFFER_UAV_FLAG_NONE;
            }

            device->CreateUnorderedAccessView(
                defaultBuffer.Get(),
                nullptr,
                &uavDesc,
                srvUavPool.GetCpuStagingHandle(res.uavDescriptorIndex)
            );
            device->CreateUnorderedAccessView(
                defaultBuffer.Get(),
                nullptr,
                &uavDesc,
                srvUavPool.GetCPUHandle(res.uavDescriptorIndex)
            );
        }

        stateTracker.RegisterResource(defaultBuffer.Get(), finalCreatedState, finalCreatedAccess);

        const NativeHandle handle = GenerateHandle();
        m_gpuBuffers[handle] = std::move(res);
        m_gpuBufferAllocationsThisSubmission++;
        return handle;
    }

    BOOL D3D12ResourceManager::UpdateGPUBuffer(
        D3D12DeviceContext& ctx,
        NativeHandle handle,
        const void* data,
        UINT32 byteSize,
        UINT32 dstOffset)
    {
        if (!data || byteSize == 0) return false;

        auto it = m_gpuBuffers.find(handle);
        if (it == m_gpuBuffers.end() || !it->second.resource || it->second.pendingDestroy)
        {
            return false;
        }

        const D3D12GPUBufferResource& buf = it->second;
        if (dstOffset > buf.byteSize || byteSize > (buf.byteSize - dstOffset))
        {
            return false;
        }

        ComPtr<ID3D12Resource> uploadResource;
        UINT64 srcOffset = 0;
        void* mappedPtr = nullptr;
        BOOL isStandalone = false;

        if (!AllocateFromUploadRing(ctx, handle, byteSize, uploadResource, srcOffset, mappedPtr, isStandalone))
        {
            return false;
        }

        std::memcpy(mappedPtr, data, byteSize);
        if (isStandalone)
        {
            uploadResource->Unmap(0, nullptr);
        }

        D3D12PendingBufferUpload pending{};
        pending.targetHandle       = handle;
        pending.uploadResource     = uploadResource;
        pending.isStandaloneUpload = isStandalone;
        pending.srcOffset          = srcOffset;
        pending.dstOffset          = dstOffset;
        pending.byteSize           = byteSize;
        m_pendingBufferUploads.push_back(std::move(pending));

        return true;
    }

    void D3D12ResourceManager::FlushPendingGPUBufferUploads(
        ID3D12GraphicsCommandList* cmdList,
        D3D12ResourceStateTracker& stateTracker,
        UINT64 upcomingFenceValue)
    {
        // Tag all newly allocated upload ring regions with the fence value of this submission
        for (auto& region : m_uploadRingInFlight)
        {
            if (region.fenceValue == UINT64_MAX)
            {
                region.fenceValue = upcomingFenceValue;
            }
        }

        if (m_pendingBufferUploads.empty())
        {
            return;
        }

        if (!cmdList)
        {
            m_pendingBufferUploads.clear();
            return;
        }

        BOOL hasGPUCopy = false;
        for (const auto& up : m_pendingBufferUploads)
        {
            if (up.srcGPUHandle != 0)
            {
                hasGPUCopy = true;
                break;
            }
        }

        if (hasGPUCopy)
        {
            // Ordered FIFO flush preserving exact causality between CPU->GPU uploads and GPU->GPU buffer copies
            for (auto& up : m_pendingBufferUploads)
            {
                auto itDst = m_gpuBuffers.find(up.targetHandle);
                if (itDst != m_gpuBuffers.end() && itDst->second.resource && up.uploadResource)
                {
                    if (up.srcGPUHandle != 0)
                    {
                        ID3D12Resource* srcRes   = up.uploadResource.Get();
                        const BOOL srcTracked    = stateTracker.IsResourceTracked(srcRes);
                        auto itSrc               = m_gpuBuffers.find(up.srcGPUHandle);
                        const GPUResourceAccess srcRestoreAccess = (itSrc != m_gpuBuffers.end())
                            ? itSrc->second.defaultAccess
                            : GPUResourceAccess::ALL_SHADER_RESOURCE;

                        if (srcTracked)
                        {
                            stateTracker.TransitionResource(srcRes, GPUResourceAccess::COPY_SOURCE);
                        }
                        else if (up.srcFallbackState != D3D12_RESOURCE_STATE_COPY_SOURCE)
                        {
                            D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(srcRes, up.srcFallbackState, D3D12_RESOURCE_STATE_COPY_SOURCE);
                            cmdList->ResourceBarrier(1, &b);
                        }
                        stateTracker.TransitionResource(itDst->second.resource.Get(), GPUResourceAccess::COPY_DEST);
                        stateTracker.FlushBarriers(cmdList);

                        cmdList->CopyBufferRegion(
                            itDst->second.resource.Get(),
                            up.dstOffset,
                            srcRes,
                            up.srcOffset,
                            up.byteSize);

                        if (srcTracked)
                        {
                            stateTracker.TransitionResource(srcRes, srcRestoreAccess);
                            if (itSrc != m_gpuBuffers.end())
                            {
                                itSrc->second.currentState = stateTracker.GetResourceState(srcRes);
                            }
                        }
                        else if (up.srcFallbackState != D3D12_RESOURCE_STATE_COPY_SOURCE)
                        {
                            D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(srcRes, D3D12_RESOURCE_STATE_COPY_SOURCE, up.srcFallbackState);
                            cmdList->ResourceBarrier(1, &b);
                        }
                        stateTracker.TransitionResource(itDst->second.resource.Get(), itDst->second.defaultAccess);
                        stateTracker.FlushBarriers(cmdList);
                        itDst->second.currentState = stateTracker.GetResourceState(itDst->second.resource.Get());
                    }
                    else
                    {
                        stateTracker.TransitionResource(itDst->second.resource.Get(), GPUResourceAccess::COPY_DEST);
                        stateTracker.FlushBarriers(cmdList);
                        cmdList->CopyBufferRegion(
                            itDst->second.resource.Get(),
                            up.dstOffset,
                            up.uploadResource.Get(),
                            up.srcOffset,
                            up.byteSize);
                        m_uploadedBytesThisSubmission += up.byteSize;
                        stateTracker.TransitionResource(itDst->second.resource.Get(), itDst->second.defaultAccess);
                        stateTracker.FlushBarriers(cmdList);
                        itDst->second.currentState = stateTracker.GetResourceState(itDst->second.resource.Get());
                    }
                }

                if (up.isStandaloneUpload && up.uploadResource)
                {
                    D3D12RetiredResource retired{};
                    retired.fenceValue = upcomingFenceValue;
                    retired.resource   = std::move(up.uploadResource);
                    m_retiredResources.push_back(std::move(retired));
                }
            }
            m_pendingBufferUploads.clear();
            return;
        }

        // 1. Transition all target buffers to COPY_DEST and flush barriers in one batch
        std::vector<NativeHandle> touchedHandles;
        touchedHandles.reserve(m_pendingBufferUploads.size());

        for (const auto& up : m_pendingBufferUploads)
        {
            auto it = m_gpuBuffers.find(up.targetHandle);
            if (it == m_gpuBuffers.end() || !it->second.resource)
            {
                continue;
            }
            stateTracker.TransitionResource(it->second.resource.Get(), GPUResourceAccess::COPY_DEST);
            it->second.currentState = stateTracker.GetResourceState(it->second.resource.Get());

            if (std::find(touchedHandles.begin(), touchedHandles.end(), up.targetHandle) == touchedHandles.end())
            {
                touchedHandles.push_back(up.targetHandle);
            }
        }
        stateTracker.FlushBarriers(cmdList);

        // 2. Record CopyBufferRegion commands and retire any standalone upload buffers
        for (auto& up : m_pendingBufferUploads)
        {
            auto it = m_gpuBuffers.find(up.targetHandle);
            if (it != m_gpuBuffers.end() && it->second.resource && up.uploadResource)
            {
                cmdList->CopyBufferRegion(
                    it->second.resource.Get(),
                    up.dstOffset,
                    up.uploadResource.Get(),
                    up.srcOffset,
                    up.byteSize
                );
                m_uploadedBytesThisSubmission += up.byteSize;
            }

            if (up.isStandaloneUpload && up.uploadResource)
            {
                D3D12RetiredResource retired{};
                retired.fenceValue = upcomingFenceValue;
                retired.resource   = std::move(up.uploadResource);
                m_retiredResources.push_back(std::move(retired));
            }
        }

        // 3. Transition updated buffers to their default shader-ready state and flush barriers
        for (NativeHandle h : touchedHandles)
        {
            auto it = m_gpuBuffers.find(h);
            if (it != m_gpuBuffers.end() && it->second.resource)
            {
                stateTracker.TransitionResource(it->second.resource.Get(), it->second.defaultAccess);
                it->second.currentState = stateTracker.GetResourceState(it->second.resource.Get());
            }
        }
        stateTracker.FlushBarriers(cmdList);

        m_pendingBufferUploads.clear();
    }

    BOOL D3D12ResourceManager::EnsureGPUBufferReadbackResources(
        D3D12DeviceContext& ctx,
        D3D12GPUBufferResource& buf)
    {
        if (buf.readbackBuffers[0] && buf.readbackBuffers[1] && buf.readbackBuffers[2])
        {
            return true;
        }
        ID3D12Device* device = ctx.GetDevice();
        if (!device || buf.byteSize == 0)
        {
            return false;
        }

        D3D12_HEAP_PROPERTIES rbHeap{};
        rbHeap.Type                 = D3D12_HEAP_TYPE_READBACK;
        rbHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        rbHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        rbHeap.CreationNodeMask     = 1;
        rbHeap.VisibleNodeMask      = 1;

        D3D12_RESOURCE_DESC rbDesc = CreateBufferResourceDesc(buf.byteSize);
        for (UINT32 i = 0; i < 3; ++i)
        {
            if (!buf.readbackBuffers[i])
            {
                if (FAILED(device->CreateCommittedResource(
                    &rbHeap,
                    D3D12_HEAP_FLAG_NONE,
                    &rbDesc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    nullptr,
                    IID_PPV_ARGS(&buf.readbackBuffers[i]))))
                {
                    return false;
                }
            }
        }
        return true;
    }

    BOOL D3D12ResourceManager::RequestGPUBufferReadback(
        D3D12DeviceContext& ctx,
        NativeHandle handle,
        UINT32 srcOffset,
        UINT32 byteSize)
    {
        auto it = m_gpuBuffers.find(handle);
        if (it == m_gpuBuffers.end() || !it->second.resource || it->second.pendingDestroy)
        {
            return false;
        }

        D3D12GPUBufferResource& buf = it->second;
        if (!HasGPUBufferUsage(buf.desc.usage, GPUBufferUsage::CPU_READBACK) ||
            !EnsureGPUBufferReadbackResources(ctx, buf))
        {
            return false;
        }

        if (srcOffset >= buf.byteSize)
        {
            return false;
        }

        const UINT32 effectiveSize = (byteSize == 0) ? (buf.byteSize - srcOffset) : byteSize;
        if (effectiveSize == 0 || effectiveSize > (buf.byteSize - srcOffset))
        {
            return false;
        }

        buf.hasPendingReadbackReq   = true;
        buf.pendingReadbackOffset   = srcOffset;
        buf.pendingReadbackSize     = effectiveSize;
        buf.readbackReadyFenceValue = UINT64_MAX;
        return true;
    }

    void D3D12ResourceManager::FlushPendingGPUBufferReadbacks(
        ID3D12GraphicsCommandList* cmdList,
        D3D12ResourceStateTracker& stateTracker,
        UINT32 bbIndex,
        UINT64 upcomingFenceValue)
    {
        if (!cmdList) return;

        const UINT32 slot = bbIndex % 3;
        std::vector<NativeHandle> readbackHandles;

        for (auto& pair : m_gpuBuffers)
        {
            D3D12GPUBufferResource& buf = pair.second;
            if (buf.hasPendingReadbackReq && buf.resource && buf.readbackBuffers[slot])
            {
                stateTracker.TransitionResource(buf.resource.Get(), GPUResourceAccess::COPY_SOURCE);
                buf.currentState = stateTracker.GetResourceState(buf.resource.Get());
                readbackHandles.push_back(pair.first);
            }
        }

        if (readbackHandles.empty())
        {
            return;
        }

        stateTracker.FlushBarriers(cmdList);

        for (NativeHandle h : readbackHandles)
        {
            D3D12GPUBufferResource& buf = m_gpuBuffers[h];
            cmdList->CopyBufferRegion(
                buf.readbackBuffers[slot].Get(),
                0,
                buf.resource.Get(),
                buf.pendingReadbackOffset,
                buf.pendingReadbackSize
            );
            buf.latestReadbackSlot      = slot;
            buf.latestReadbackSize      = buf.pendingReadbackSize;
            buf.readbackReadyFenceValue = upcomingFenceValue;
            buf.hasPendingReadbackReq   = false;
        }

        for (NativeHandle h : readbackHandles)
        {
            D3D12GPUBufferResource& buf = m_gpuBuffers[h];
            stateTracker.TransitionResource(buf.resource.Get(), buf.defaultAccess);
            buf.currentState = stateTracker.GetResourceState(buf.resource.Get());
        }
        stateTracker.FlushBarriers(cmdList);
    }

    BOOL D3D12ResourceManager::TryGetGPUBufferReadbackData(
        D3D12DeviceContext& ctx,
        NativeHandle handle,
        void* outData,
        UINT32 byteSize)
    {
        if (!outData || byteSize == 0) return false;

        auto it = m_gpuBuffers.find(handle);
        if (it == m_gpuBuffers.end() || !it->second.resource || it->second.pendingDestroy)
        {
            return false;
        }

        const D3D12GPUBufferResource& buf = it->second;
        if (!HasGPUBufferUsage(buf.desc.usage, GPUBufferUsage::CPU_READBACK) || byteSize > buf.byteSize)
        {
            return false;
        }

        // If the readback request has not yet been recorded or its GPU fence has not completed, return false without waiting!
        if (buf.hasPendingReadbackReq || buf.readbackReadyFenceValue == 0 || buf.readbackReadyFenceValue == UINT64_MAX)
        {
            return false;
        }

        ID3D12Fence* fence = ctx.GetFrameFence();
        if (!fence || fence->GetCompletedValue() < buf.readbackReadyFenceValue)
        {
            return false;
        }

        const UINT32 slot = buf.latestReadbackSlot % 3;
        if (!buf.readbackBuffers[slot] || buf.latestReadbackSize == 0)
        {
            return false;
        }

        const UINT32 copyBytes = (std::min)(byteSize, buf.latestReadbackSize);
        D3D12_RANGE readRange{ 0, copyBytes };
        void* mapped = nullptr;
        if (FAILED(buf.readbackBuffers[slot]->Map(0, &readRange, &mapped)) || !mapped)
        {
            return false;
        }

        std::memcpy(outData, mapped, copyBytes);
        D3D12_RANGE emptyWrite{ 0, 0 };
        buf.readbackBuffers[slot]->Unmap(0, &emptyWrite);
        return true;
    }

    BOOL D3D12ResourceManager::ReadbackGPUBufferSync(
        D3D12DeviceContext& ctx,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle handle,
        void* outData,
        UINT32 byteSize)
    {
        if (!outData || byteSize == 0) return false;

        auto it = m_gpuBuffers.find(handle);
        if (it == m_gpuBuffers.end() || !it->second.resource || it->second.pendingDestroy)
        {
            return false;
        }

        D3D12GPUBufferResource& buf = it->second;
        if (!HasGPUBufferUsage(buf.desc.usage, GPUBufferUsage::CPU_READBACK) ||
            byteSize > buf.byteSize ||
            !EnsureGPUBufferReadbackResources(ctx, buf))
        {
            return false;
        }

        const BOOL hadPendingUploads = !m_pendingBufferUploads.empty();
        const UINT64 syncFenceVal = ctx.GetFrameFenceValue() + 1;

        ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
            if (hadPendingUploads)
            {
                FlushPendingGPUBufferUploads(cmdList, stateTracker, syncFenceVal);
            }

            stateTracker.TransitionResource(buf.resource.Get(), GPUResourceAccess::COPY_SOURCE);
            stateTracker.FlushBarriers(cmdList);

            cmdList->CopyBufferRegion(buf.readbackBuffers[0].Get(), 0, buf.resource.Get(), 0, byteSize);

            stateTracker.TransitionResource(buf.resource.Get(), buf.defaultAccess);
            stateTracker.FlushBarriers(cmdList);
            buf.currentState = stateTracker.GetResourceState(buf.resource.Get());
        });

        if ((buf.latestReadbackSlot % 3u) == 0u)
        {
            buf.readbackReadyFenceValue = 0;
        }

        if (hadPendingUploads)
        {
            ctx.WaitForGPU();
            const UINT64 completedFence = ctx.GetFrameFence() ? ctx.GetFrameFence()->GetCompletedValue() : 0;
            ReclaimCompletedUploadRingRegions(completedFence);
        }

        D3D12_RANGE readRange{ 0, byteSize };
        void* mapped = nullptr;
        if (FAILED(buf.readbackBuffers[0]->Map(0, &readRange, &mapped)) || !mapped)
        {
            return false;
        }

        std::memcpy(outData, mapped, byteSize);
        D3D12_RANGE emptyWrite{ 0, 0 };
        buf.readbackBuffers[0]->Unmap(0, &emptyWrite);
        return true;
    }

    BOOL D3D12ResourceManager::ReadbackTextureSubresourceSync(
        D3D12DeviceContext& ctx,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle handle,
        UINT32 mipSlice,
        UINT32 arraySlice,
        void* outData,
        UINT32 byteSize)
    {
        if (!outData || byteSize == 0u) return false;

        auto it = m_textures.find(handle);
        if (it == m_textures.end() || !it->second.resource) return false;

        ID3D12Device* device = ctx.GetDevice();
        if (!device) return false;

        D3D12TextureResource& tex = it->second;
        const UINT32 mipLevels   = (std::max)(1u, tex.desc.mipLevels);
        const UINT32 arraySlices = ComputeTextureArraySliceCount(tex.desc.type, tex.desc.arrayLayers);
        if (mipSlice >= mipLevels || arraySlice >= arraySlices)
            return false;

        D3D12_RESOURCE_DESC texDesc = tex.resource->GetDesc();
        texDesc.Format = ResolveCopyableReadbackFormat(texDesc.Format, tex.desc.format);

        const UINT32 subresIdx = mipSlice + arraySlice * mipLevels;
        UINT64 requiredBytes = 0ull;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT32 numRows = 0u;
        UINT64 rowSizeInBytes = 0ull;
        device->GetCopyableFootprints(&texDesc, subresIdx, 1u, 0ull, &footprint, &numRows, &rowSizeInBytes, &requiredBytes);

        const UINT32 depthSlices = (std::max)(1u, static_cast<UINT32>(footprint.Footprint.Depth));
        const SIZE_T tightBytes  = static_cast<SIZE_T>(rowSizeInBytes) * static_cast<SIZE_T>(numRows) * static_cast<SIZE_T>(depthSlices);
        if (requiredBytes == 0ull || numRows == 0u || rowSizeInBytes == 0ull || byteSize < tightBytes)
            return false;

        D3D12_HEAP_PROPERTIES rbHeap{};
        rbHeap.Type                 = D3D12_HEAP_TYPE_READBACK;
        rbHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        rbHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        rbHeap.CreationNodeMask     = 1;
        rbHeap.VisibleNodeMask      = 1;

        D3D12_RESOURCE_DESC rbDesc = CreateBufferResourceDesc(requiredBytes);
        ComPtr<ID3D12Resource> rbBuffer;
        if (FAILED(device->CreateCommittedResource(
            &rbHeap,
            D3D12_HEAP_FLAG_NONE,
            &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&rbBuffer))) || !rbBuffer)
        {
            return false;
        }

        ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
            const BOOL isTracked = stateTracker.IsResourceTracked(tex.resource.Get());
            const D3D12_RESOURCE_STATES prevState = isTracked
                ? stateTracker.GetSubresourceState(tex.resource.Get(), subresIdx)
                : tex.currentState;
            const GPUResourceAccess prevAccess = isTracked
                ? stateTracker.GetSubresourceAccess(tex.resource.Get(), subresIdx)
                : GPUResourceAccess::ALL_SHADER_RESOURCE;

            if (isTracked)
            {
                stateTracker.TransitionSubresource(tex.resource.Get(), subresIdx, GPUResourceAccess::COPY_SOURCE);
                stateTracker.FlushBarriers(cmdList);
            }
            else if (prevState != D3D12_RESOURCE_STATE_COPY_SOURCE)
            {
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(
                    tex.resource.Get(), prevState, D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmdList->ResourceBarrier(1, &b);
            }

            D3D12_TEXTURE_COPY_LOCATION dstLoc{};
            dstLoc.pResource       = rbBuffer.Get();
            dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dstLoc.PlacedFootprint = footprint;

            D3D12_TEXTURE_COPY_LOCATION srcLoc{};
            srcLoc.pResource        = tex.resource.Get();
            srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            srcLoc.SubresourceIndex = subresIdx;

            cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

            if (isTracked)
            {
                stateTracker.TransitionSubresource(tex.resource.Get(), subresIdx, prevAccess);
                stateTracker.FlushBarriers(cmdList);
                tex.currentState = stateTracker.GetResourceState(tex.resource.Get());
            }
            else if (prevState != D3D12_RESOURCE_STATE_COPY_SOURCE)
            {
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(
                    tex.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, prevState);
                cmdList->ResourceBarrier(1, &b);
            }
        });

        D3D12_RANGE readRange{ 0, static_cast<SIZE_T>(requiredBytes) };
        BYTE* mapped = nullptr;
        if (FAILED(rbBuffer->Map(0, &readRange, reinterpret_cast<void**>(&mapped))) || !mapped)
            return false;

        BYTE*       dstBase      = static_cast<BYTE*>(outData);
        const BYTE* srcBase      = mapped + footprint.Offset;
        const SIZE_T rowBytes    = static_cast<SIZE_T>(rowSizeInBytes);
        const SIZE_T srcRowPitch = static_cast<SIZE_T>(footprint.Footprint.RowPitch);
        const SIZE_T dstSliceStep = rowBytes * static_cast<SIZE_T>(numRows);
        const SIZE_T srcSliceStep = srcRowPitch * static_cast<SIZE_T>(numRows);

        for (UINT32 z = 0u; z < depthSlices; ++z)
        {
            BYTE*       dstSlice = dstBase + static_cast<SIZE_T>(z) * dstSliceStep;
            const BYTE* srcSlice = srcBase + static_cast<SIZE_T>(z) * srcSliceStep;
            for (UINT32 r = 0u; r < numRows; ++r)
            {
                std::memcpy(dstSlice + static_cast<SIZE_T>(r) * rowBytes,
                            srcSlice + static_cast<SIZE_T>(r) * srcRowPitch,
                            rowBytes);
            }
        }

        D3D12_RANGE emptyWrite{ 0, 0 };
        rbBuffer->Unmap(0, &emptyWrite);
        return true;
    }

    BOOL D3D12ResourceManager::ReadbackTextureSync(
        D3D12DeviceContext& ctx,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle handle,
        void* outData,
        UINT32 byteSize)
    {
        if (!outData || byteSize == 0) return false;

        auto it = m_textures.find(handle);
        if (it == m_textures.end() || !it->second.resource)
        {
            return false;
        }

        ID3D12Device* device = ctx.GetDevice();
        if (!device) return false;

        D3D12TextureResource& tex = it->second;
        D3D12_RESOURCE_DESC texDesc = tex.resource->GetDesc();
        texDesc.Format = ResolveCopyableReadbackFormat(texDesc.Format, tex.desc.format);

        UINT64 singleSliceBytes = 0;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint0{};
        UINT32 numRows0 = 0;
        UINT64 rowSizeInBytes0 = 0;
        device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint0, &numRows0, &rowSizeInBytes0, &singleSliceBytes);

        const UINT32 depthSlices0 = (std::max)(1u, static_cast<UINT32>(footprint0.Footprint.Depth));
        const SIZE_T sliceTightBytes = static_cast<SIZE_T>(numRows0) * static_cast<SIZE_T>(rowSizeInBytes0) * static_cast<SIZE_T>(depthSlices0);
        if (numRows0 == 0 || rowSizeInBytes0 == 0 || sliceTightBytes == 0 || byteSize < sliceTightBytes)
        {
            return false;
        }

        const UINT32 totalArraySlices = ComputeTextureArraySliceCount(tex.desc.type, tex.desc.arrayLayers);
        const UINT32 slicesToRead = (std::min)(
            totalArraySlices,
            (std::max)(1u, static_cast<UINT32>(static_cast<SIZE_T>(byteSize) / sliceTightBytes))
        );
        if (slicesToRead == 1u)
        {
            return ReadbackTextureSubresourceSync(ctx, stateTracker, handle, 0u, 0u, outData, byteSize);
        }

        const UINT32 mipLevels = (std::max)(1u, static_cast<UINT32>(texDesc.MipLevels));

        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(slicesToRead);
        std::vector<UINT32> numRowsPerSlice(slicesToRead, 0);
        std::vector<UINT64> rowSizePerSlice(slicesToRead, 0);
        UINT64 totalBytes = 0;
        for (UINT32 s = 0; s < slicesToRead; ++s)
        {
            const UINT64 baseOffset = (totalBytes + 511ULL) & ~511ULL;
            UINT64 sliceRequiredBytes = 0;
            const UINT32 subresIdx = s * mipLevels;
            device->GetCopyableFootprints(
                &texDesc,
                subresIdx,
                1,
                baseOffset,
                &footprints[s],
                &numRowsPerSlice[s],
                &rowSizePerSlice[s],
                &sliceRequiredBytes
            );
            totalBytes = baseOffset + sliceRequiredBytes;
        }

        if (totalBytes == 0)
        {
            return false;
        }

        D3D12_HEAP_PROPERTIES rbHeap{};
        rbHeap.Type                 = D3D12_HEAP_TYPE_READBACK;
        rbHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        rbHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        rbHeap.CreationNodeMask     = 1;
        rbHeap.VisibleNodeMask      = 1;

        D3D12_RESOURCE_DESC rbDesc = CreateBufferResourceDesc(totalBytes);
        ComPtr<ID3D12Resource> rbBuffer;
        if (FAILED(device->CreateCommittedResource(
            &rbHeap,
            D3D12_HEAP_FLAG_NONE,
            &rbDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&rbBuffer))))
        {
            return false;
        }

        ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
            const BOOL isTracked = stateTracker.IsResourceTracked(tex.resource.Get());
            const D3D12_RESOURCE_STATES prevState = isTracked
                ? stateTracker.GetResourceState(tex.resource.Get())
                : tex.currentState;
            const GPUResourceAccess prevAccess = isTracked
                ? stateTracker.GetResourceAccess(tex.resource.Get())
                : GPUResourceAccess::ALL_SHADER_RESOURCE;

            if (isTracked)
            {
                stateTracker.TransitionResource(tex.resource.Get(), GPUResourceAccess::COPY_SOURCE);
                stateTracker.FlushBarriers(cmdList);
            }
            else if (prevState != D3D12_RESOURCE_STATE_COPY_SOURCE)
            {
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(
                    tex.resource.Get(), prevState, D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmdList->ResourceBarrier(1, &b);
            }

            for (UINT32 s = 0; s < slicesToRead; ++s)
            {
                D3D12_TEXTURE_COPY_LOCATION dstLoc{};
                dstLoc.pResource       = rbBuffer.Get();
                dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dstLoc.PlacedFootprint = footprints[s];

                D3D12_TEXTURE_COPY_LOCATION srcLoc{};
                srcLoc.pResource        = tex.resource.Get();
                srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                srcLoc.SubresourceIndex = s * mipLevels;

                cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
            }

            if (isTracked)
            {
                stateTracker.TransitionResource(tex.resource.Get(), prevAccess);
                stateTracker.FlushBarriers(cmdList);
                tex.currentState = stateTracker.GetResourceState(tex.resource.Get());
            }
            else if (prevState != D3D12_RESOURCE_STATE_COPY_SOURCE)
            {
                D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(
                    tex.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, prevState);
                cmdList->ResourceBarrier(1, &b);
            }
        });

        D3D12_RANGE readRange{ 0, static_cast<SIZE_T>(totalBytes) };
        BYTE* mapped = nullptr;
        if (FAILED(rbBuffer->Map(0, &readRange, reinterpret_cast<void**>(&mapped))) || !mapped)
        {
            return false;
        }

        BYTE* dstBytes = static_cast<BYTE*>(outData);
        for (UINT32 s = 0; s < slicesToRead; ++s)
        {
            BYTE* sliceDstBase = dstBytes + static_cast<SIZE_T>(s) * sliceTightBytes;
            const BYTE* srcBase = mapped + footprints[s].Offset;
            const UINT32 rows = numRowsPerSlice[s];
            const SIZE_T rowBytes = static_cast<SIZE_T>(rowSizePerSlice[s]);
            const SIZE_T rowPitch = static_cast<SIZE_T>(footprints[s].Footprint.RowPitch);
            for (UINT32 r = 0; r < rows; ++r)
            {
                std::memcpy(
                    sliceDstBase + static_cast<SIZE_T>(r) * rowBytes,
                    srcBase      + static_cast<SIZE_T>(r) * rowPitch,
                    rowBytes);
            }
        }

        D3D12_RANGE emptyWrite{ 0, 0 };
        rbBuffer->Unmap(0, &emptyWrite);
        return true;
    }

    void D3D12ResourceManager::DestroyGPUBuffer(
        NativeHandle handle,
        D3D12ResourceStateTracker& stateTracker,
        UINT64 fenceValue)
    {
        (void)stateTracker;
        auto it = m_gpuBuffers.find(handle);
        if (it == m_gpuBuffers.end() || it->second.pendingDestroy)
        {
            return;
        }

        // Mark as pending destroy with at least the upcoming submission fence value (fenceValue + 1).
        // The resource and its SRV/UAV descriptors remain valid in m_gpuBuffers until the current
        // frame finishes recording (FinalizePendingDestroyedGPUBuffers) or WaitForGPU/ProcessRetirementQueue
        // confirms GPU completion, preventing premature reclamation when DestroyGPUBuffer is called
        // on a buffer already scheduled for GPU usage in an unsubmitted frame.
        UINT64 safeFenceValue = fenceValue + 1;
        if (it->second.readbackReadyFenceValue != 0 &&
            it->second.readbackReadyFenceValue != UINT64_MAX &&
            it->second.readbackReadyFenceValue > safeFenceValue)
        {
            safeFenceValue = it->second.readbackReadyFenceValue;
        }

        it->second.pendingDestroy         = true;
        it->second.retireFenceValue       = safeFenceValue;
        m_hasPendingDestroyedGPUBuffers   = true;
    }

    void D3D12ResourceManager::FinalizePendingDestroyedGPUBuffers(
        D3D12ResourceStateTracker& stateTracker,
        UINT64 submittedFenceValue,
        const NativeHandle* filterHandles,
        UINT32 filterCount)
    {
        if (!m_hasPendingDestroyedGPUBuffers)
        {
            return;
        }

        auto matchesFilter = [filterHandles, filterCount](NativeHandle h) -> BOOL {
            if (!filterHandles)
            {
                return true;
            }
            for (UINT32 i = 0; i < filterCount; ++i)
            {
                if (filterHandles[i] == h)
                {
                    return true;
                }
            }
            return false;
        };

        BOOL anyRemainingPendingDestroy = false;
        std::vector<NativeHandle> finalizedHandles;

        auto it = m_gpuBuffers.begin();
        while (it != m_gpuBuffers.end())
        {
            if (it->second.pendingDestroy)
            {
                const NativeHandle handle = it->first;
                if (!matchesFilter(handle))
                {
                    anyRemainingPendingDestroy = true;
                    ++it;
                    continue;
                }

                finalizedHandles.push_back(handle);
                m_pendingBufferUploads.erase(
                    std::remove_if(
                        m_pendingBufferUploads.begin(),
                        m_pendingBufferUploads.end(),
                        [handle](const D3D12PendingBufferUpload& up) { return up.targetHandle == handle; }),
                    m_pendingBufferUploads.end());

                if (it->second.resource)
                {
                    stateTracker.UnregisterResource(it->second.resource.Get());
                }

                UINT64 safeFence = (std::max)(submittedFenceValue, it->second.retireFenceValue);
                if (it->second.readbackReadyFenceValue != 0 &&
                    it->second.readbackReadyFenceValue != UINT64_MAX &&
                    it->second.readbackReadyFenceValue > safeFence)
                {
                    safeFence = it->second.readbackReadyFenceValue;
                }

                D3D12RetiredResource retired{};
                retired.fenceValue            = safeFence;
                retired.resource              = std::move(it->second.resource);
                retired.auxiliaryResources[0] = std::move(it->second.readbackBuffers[0]);
                retired.auxiliaryResources[1] = std::move(it->second.readbackBuffers[1]);
                retired.auxiliaryResources[2] = std::move(it->second.readbackBuffers[2]);
                retired.srvDescriptorIndex    = it->second.srvDescriptorIndex;
                retired.uavDescriptorIndex    = it->second.uavDescriptorIndex;

                it->second.srvDescriptorIndex = UINT32_MAX;
                it->second.uavDescriptorIndex = UINT32_MAX;

                m_retiredResources.push_back(std::move(retired));
                it = m_gpuBuffers.erase(it);
            }
            else
            {
                ++it;
            }
        }

        m_hasPendingDestroyedGPUBuffers = anyRemainingPendingDestroy;

        if (!m_uploadRingInFlight.empty() && (!finalizedHandles.empty() || !filterHandles))
        {
            m_uploadRingInFlight.erase(
                std::remove_if(
                    m_uploadRingInFlight.begin(),
                    m_uploadRingInFlight.end(),
                    [&](const D3D12UploadRingRegion& r) {
                        if (r.fenceValue != UINT64_MAX)
                        {
                            return false;
                        }
                        if (!filterHandles && m_pendingBufferUploads.empty())
                        {
                            return true;
                        }
                        return std::find(finalizedHandles.begin(), finalizedHandles.end(), r.targetHandle) != finalizedHandles.end();
                    }),
                m_uploadRingInFlight.end());
            if (m_uploadRingInFlight.empty())
            {
                m_uploadRingReadOffset  = 0;
                m_uploadRingWriteOffset = 0;
            }
            else
            {
                m_uploadRingReadOffset  = m_uploadRingInFlight.front().offset;
                m_uploadRingWriteOffset = m_uploadRingInFlight.back().endOffset;
            }
        }
    }

    void D3D12ResourceManager::ProcessRetirementQueue(
        UINT64 completedFenceValue,
        D3D12DescriptorPool& cbvPool,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        D3D12DescriptorPool& samplerPool,
        D3D12ResourceStateTracker* stateTracker)
    {
        ReclaimCompletedUploadRingRegions(completedFenceValue);

        // During Shutdown or forced retirement flush (completedFenceValue == UINT64_MAX),
        // finalize any remaining pendingDestroy buffers and reclaim cancelled upload ring regions.
        if (completedFenceValue == UINT64_MAX)
        {
            m_hasPendingDestroyedGPUBuffers = false;
            auto bufIt = m_gpuBuffers.begin();
            while (bufIt != m_gpuBuffers.end())
            {
                if (bufIt->second.pendingDestroy)
                {
                    const NativeHandle handle = bufIt->first;
                    m_pendingBufferUploads.erase(
                        std::remove_if(
                            m_pendingBufferUploads.begin(),
                            m_pendingBufferUploads.end(),
                            [handle](const D3D12PendingBufferUpload& up) { return up.targetHandle == handle; }),
                        m_pendingBufferUploads.end());

                    if (stateTracker && bufIt->second.resource)
                    {
                        stateTracker->UnregisterResource(bufIt->second.resource.Get());
                    }

                    D3D12RetiredResource retired{};
                    retired.fenceValue            = bufIt->second.retireFenceValue;
                    retired.resource              = std::move(bufIt->second.resource);
                    retired.auxiliaryResources[0] = std::move(bufIt->second.readbackBuffers[0]);
                    retired.auxiliaryResources[1] = std::move(bufIt->second.readbackBuffers[1]);
                    retired.auxiliaryResources[2] = std::move(bufIt->second.readbackBuffers[2]);
                    retired.srvDescriptorIndex    = bufIt->second.srvDescriptorIndex;
                    retired.uavDescriptorIndex    = bufIt->second.uavDescriptorIndex;

                    bufIt->second.srvDescriptorIndex = UINT32_MAX;
                    bufIt->second.uavDescriptorIndex = UINT32_MAX;

                    m_retiredResources.push_back(std::move(retired));
                    bufIt = m_gpuBuffers.erase(bufIt);
                }
                else
                {
                    ++bufIt;
                }
            }

            if (m_pendingBufferUploads.empty() && !m_uploadRingInFlight.empty())
            {
                m_uploadRingInFlight.clear();
                m_uploadRingReadOffset  = 0;
                m_uploadRingWriteOffset = 0;
            }
        }

        auto it = m_retiredResources.begin();
        while (it != m_retiredResources.end())
        {
            if (completedFenceValue >= it->fenceValue)
            {
                if (it->mappedData && it->resource)
                {
                    it->resource->Unmap(0, nullptr);
                    it->mappedData = nullptr;
                }
                if (it->cbvDescriptorIndex != UINT32_MAX)
                {
                    cbvPool.Free(it->cbvDescriptorIndex);
                    it->cbvDescriptorIndex = UINT32_MAX;
                }
                if (it->srvDescriptorIndex != UINT32_MAX)
                {
                    cbvPool.Free(it->srvDescriptorIndex);
                    it->srvDescriptorIndex = UINT32_MAX;
                }
                if (it->uavDescriptorIndex != UINT32_MAX)
                {
                    cbvPool.Free(it->uavDescriptorIndex);
                    it->uavDescriptorIndex = UINT32_MAX;
                }
                for (UINT32 extraDesc : it->extraSrvUavDescriptors)
                {
                    if (extraDesc != UINT32_MAX) cbvPool.Free(extraDesc);
                }
                it->extraSrvUavDescriptors.clear();
                if (it->rtvDescriptorIndex != UINT32_MAX)
                {
                    rtvPool.Free(it->rtvDescriptorIndex);
                    it->rtvDescriptorIndex = UINT32_MAX;
                }
                if (it->dsvDescriptorIndex != UINT32_MAX && it->dsvLayerDescriptors.empty())
                {
                    dsvPool.Free(it->dsvDescriptorIndex);
                    it->dsvDescriptorIndex = UINT32_MAX;
                }
                if (it->readOnlyDsvDescriptorIndex != UINT32_MAX)
                {
                    dsvPool.Free(it->readOnlyDsvDescriptorIndex);
                    it->readOnlyDsvDescriptorIndex = UINT32_MAX;
                }
                if (it->fullArrayDsvDescriptorIndex != UINT32_MAX)
                {
                    dsvPool.Free(it->fullArrayDsvDescriptorIndex);
                    it->fullArrayDsvDescriptorIndex = UINT32_MAX;
                }
                for (UINT32 sliceDsv : it->dsvLayerDescriptors) dsvPool.Free(sliceDsv);
                it->dsvLayerDescriptors.clear();
                if (it->samplerDescriptorIndex != UINT32_MAX)
                {
                    samplerPool.Free(it->samplerDescriptorIndex);
                    it->samplerDescriptorIndex = UINT32_MAX;
                }

                if (it->sharedHandle)
                {
                    CloseHandle(it->sharedHandle);
                    it->sharedHandle = nullptr;
                }
                if (stateTracker && it->resource)
                {
                    stateTracker->UnregisterResource(it->resource.Get());
                }
                it->auxiliaryResources[0].Reset();
                it->auxiliaryResources[1].Reset();
                it->auxiliaryResources[2].Reset();
                it->resource.Reset();
                it = m_retiredResources.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    namespace
    {
#ifdef _WIN32
        struct DXCRuntimeState
        {
            HMODULE              hDxil       = nullptr;
            HMODULE              hDxCompiler = nullptr;
            DxcCreateInstanceProc createFn   = nullptr;
            BOOL                 available   = false;
        };

        static const DXCRuntimeState& GetDXCRuntimeState() noexcept
        {
            static std::once_flag  s_once;
            static DXCRuntimeState s_state{};
            std::call_once(s_once, []() {
                s_state.hDxil       = ::LoadLibraryW(L"dxil.dll");
                s_state.hDxCompiler = ::LoadLibraryW(L"dxcompiler.dll");
                if (s_state.hDxil != nullptr && s_state.hDxCompiler != nullptr)
                {
                    s_state.createFn = reinterpret_cast<DxcCreateInstanceProc>(
                        ::GetProcAddress(s_state.hDxCompiler, "DxcCreateInstance"));
                    if (s_state.createFn != nullptr)
                    {
                        ComPtr<IDxcUtils>     testUtils;
                        ComPtr<IDxcCompiler3> testCompiler;
                        if (SUCCEEDED(s_state.createFn(CLSID_DxcUtils, IID_PPV_ARGS(&testUtils))) &&
                            SUCCEEDED(s_state.createFn(CLSID_DxcCompiler, IID_PPV_ARGS(&testCompiler))) &&
                            testUtils && testCompiler)
                        {
                            s_state.available = true;
                        }
                    }
                }
            });
            return s_state;
        }

        static BOOL CompileHLSLWithDXC(
            const void*        srcPtr,
            SIZE_T             srcSize,
            const wchar_t*     targetProfile,
            const wchar_t*     entryPoint,
            const wchar_t*     hlslVersion,
            std::vector<BYTE>& outBytecode,
            std::string&       outErrorText)
        {
            outBytecode.clear();
            outErrorText.clear();
            const DXCRuntimeState& dxc = GetDXCRuntimeState();
            if (!dxc.available || !dxc.createFn || !srcPtr || srcSize == 0 || !targetProfile || !entryPoint)
            {
                return false;
            }

            ComPtr<IDxcUtils>     utils;
            ComPtr<IDxcCompiler3> compiler;
            if (FAILED(dxc.createFn(CLSID_DxcUtils, IID_PPV_ARGS(&utils))) ||
                FAILED(dxc.createFn(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))) ||
                !utils || !compiler)
            {
                return false;
            }

            // Trim optional trailing NUL byte if caller passed size including '\0'
            SIZE_T effectiveSize = srcSize;
            while (effectiveSize > 0 && static_cast<const char*>(srcPtr)[effectiveSize - 1] == '\0')
            {
                --effectiveSize;
            }
            if (effectiveSize == 0) return false;

            DxcBuffer srcBuf{};
            srcBuf.Ptr      = srcPtr;
            srcBuf.Size     = effectiveSize;
            srcBuf.Encoding = DXC_CP_UTF8;

            const wchar_t* args[] = {
                L"NorthstarShader.hlsl",
                L"-E",  entryPoint,
                L"-T",  targetProfile,
                L"-HV", hlslVersion,
                L"-O3",
                L"-Qstrip_reflect",
                L"-Qstrip_debug"
            };

            ComPtr<IDxcResult> result;
            if (FAILED(compiler->Compile(&srcBuf, args, _countof(args), nullptr, IID_PPV_ARGS(&result))) || !result)
            {
                return false;
            }

            ComPtr<IDxcBlobUtf8> errors;
            if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr)) &&
                errors && errors->GetStringLength() > 0)
            {
                outErrorText.assign(errors->GetStringPointer(), errors->GetStringLength());
            }

            HRESULT status = E_FAIL;
            if (FAILED(result->GetStatus(&status)) || FAILED(status))
            {
                return false;
            }

            ComPtr<IDxcBlob> objBlob;
            if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&objBlob), nullptr)) ||
                !objBlob || objBlob->GetBufferSize() < 4u)
            {
                return false;
            }

            outBytecode.resize(objBlob->GetBufferSize());
            std::memcpy(outBytecode.data(), objBlob->GetBufferPointer(), objBlob->GetBufferSize());
            return true;
        }
#endif
    }

    BOOL D3D12ResourceManager::IsDXILCompilerAvailable() noexcept
    {
#ifdef _WIN32
        return GetDXCRuntimeState().available;
#else
        return false;
#endif
    }

    void D3D12ResourceManager::InspectDXBCContainerCapabilities(
        const void* dxbcPtr,
        SIZE_T      dxbcSize,
        UINT32&     outRequiredSM,
        BOOL&       outRequiresInt64Ops,
        BOOL&       outRequiresInt64DescHeapAtomics) noexcept
    {
        outRequiredSM                   = 0x51u;
        outRequiresInt64Ops             = false;
        outRequiresInt64DescHeapAtomics = false;

        if (!dxbcPtr || dxbcSize < 32u || std::memcmp(dxbcPtr, "DXBC", 4) != 0)
        {
            return;
        }

        const BYTE* bytes = static_cast<const BYTE*>(dxbcPtr);
        UINT32 partCount = 0u;
        std::memcpy(&partCount, bytes + 28u, sizeof(UINT32));
        if (partCount > 64u || (32ull + static_cast<UINT64>(partCount) * 4ull) > static_cast<UINT64>(dxbcSize))
        {
            return;
        }

        BOOL hasDxilChunk = false;
        for (UINT32 i = 0u; i < partCount; ++i)
        {
            UINT32 partOff = 0u;
            std::memcpy(&partOff, bytes + 32u + i * 4u, sizeof(UINT32));
            if (static_cast<UINT64>(partOff) + 8ull > static_cast<UINT64>(dxbcSize))
            {
                continue;
            }
            UINT32 partSize = 0u;
            std::memcpy(&partSize, bytes + partOff + 4u, sizeof(UINT32));
            if (static_cast<UINT64>(partOff) + 8ull + static_cast<UINT64>(partSize) > static_cast<UINT64>(dxbcSize))
            {
                continue;
            }

            if (std::memcmp(bytes + partOff, "DXIL", 4) == 0 && partSize >= 8u)
            {
                hasDxilChunk = true;
                UINT32 progVer = 0u;
                std::memcpy(&progVer, bytes + partOff + 8u, sizeof(UINT32));
                const UINT32 smByte = progVer & 0xFFu;
                if (smByte >= 0x60u)
                {
                    outRequiredSM = (std::max)(outRequiredSM, smByte);
                }
            }
            else if (std::memcmp(bytes + partOff, "SFI0", 4) == 0 && partSize >= 8u)
            {
                UINT64 sfiFlags = 0ull;
                std::memcpy(&sfiFlags, bytes + partOff + 8u, sizeof(UINT64));
                // D3D_SHADER_FEATURE_INT64_OPS = 0x00008000ULL
                if ((sfiFlags & 0x00008000ULL) != 0ull)
                {
                    outRequiresInt64Ops = true;
                    outRequiredSM       = (std::max)(outRequiredSM, 0x60u);
                }
                // D3D_SHADER_FEATURE_ATOMIC_INT64_ON_TYPED_RESOURCE (1<<22) |
                // D3D_SHADER_FEATURE_ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE (1<<28)
                if ((sfiFlags & 0x10400000ULL) != 0ull)
                {
                    outRequiresInt64Ops             = true;
                    outRequiresInt64DescHeapAtomics = true;
                    outRequiredSM                   = (std::max)(outRequiredSM, 0x66u);
                }
            }
        }

#ifdef _WIN32
        // DXC sets SFI0 D3D_SHADER_FEATURE_INT64_OPS (0x8000) on RWStructuredBuffer<uint64_t> : register(uN, spaceM)
        // without setting bit 28 unless ResourceDescriptorHeap[] is indexed directly. Because NorthstarD3D12 binds
        // compute UAVs through 64-descriptor CBV/SRV/UAV descriptor tables, inspect the DXIL module for 64-bit
        // resource atomic opcodes whenever Int64Ops is present so descriptor-heap atomic support is strictly enforced.
        if (hasDxilChunk && outRequiresInt64Ops && !outRequiresInt64DescHeapAtomics)
        {
            const DXCRuntimeState& dxc = GetDXCRuntimeState();
            if (dxc.available && dxc.createFn)
            {
                ComPtr<IDxcCompiler3> compiler;
                if (SUCCEEDED(dxc.createFn(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))) && compiler)
                {
                    DxcBuffer binBuf{};
                    binBuf.Ptr      = dxbcPtr;
                    binBuf.Size     = dxbcSize;
                    binBuf.Encoding = 0u;
                    ComPtr<IDxcResult> disRes;
                    if (SUCCEEDED(compiler->Disassemble(&binBuf, IID_PPV_ARGS(&disRes))) && disRes)
                    {
                        ComPtr<IDxcBlobUtf8> disText;
                        if (SUCCEEDED(disRes->GetOutput(DXC_OUT_DISASSEMBLY, IID_PPV_ARGS(&disText), nullptr)) &&
                            disText && disText->GetStringLength() > 0)
                        {
                            const char* txt = disText->GetStringPointer();
                            if (txt && (std::strstr(txt, "dx.op.atomicBinOp.i64") != nullptr ||
                                        std::strstr(txt, "dx.op.atomicCompareExchange.i64") != nullptr))
                            {
                                outRequiresInt64DescHeapAtomics = true;
                                outRequiredSM                   = (std::max)(outRequiredSM, 0x66u);
                            }
                        }
                    }
                }
            }
        }
#else
        (void)hasDxilChunk;
#endif
    }

    NativeHandle D3D12ResourceManager::CreateShader(ShaderStage stage, const void* bytecode, SIZE_T bytecodeSize, BOOL allowDXILSM66)
    {
        if (!bytecode || bytecodeSize == 0) return 0;

        D3D12ShaderResource res{};
        res.stage = stage;
        auto populateBinaryCaps = [&]() {
            const void* ptr = !res.bytecode.empty() ? static_cast<const void*>(res.bytecode.data())
                                                    : (res.blob ? res.blob->GetBufferPointer() : nullptr);
            const SIZE_T sz = !res.bytecode.empty() ? res.bytecode.size()
                                                    : (res.blob ? res.blob->GetBufferSize() : 0u);
            InspectDXBCContainerCapabilities(
                ptr, sz,
                res.requiredShaderModel,
                res.requiresInt64Ops,
                res.requiresInt64DescriptorHeapAtomics);
        };

        // Check if precompiled DXBC / DXIL container binary (both start with "DXBC")
        if (bytecodeSize >= 4 && std::memcmp(bytecode, "DXBC", 4) == 0)
        {
            res.bytecode.resize(bytecodeSize);
            std::memcpy(res.bytecode.data(), bytecode, bytecodeSize);
            populateBinaryCaps();
            if (allowDXILSM66 == FALSE && res.requiredShaderModel >= 0x66u)
            {
                return 0;
            }
        }
        else
        {
            const BOOL useDXIL66 = (allowDXILSM66 != FALSE) && IsDXILCompilerAvailable();
            if (useDXIL66)
            {
#ifdef _WIN32
                const wchar_t* wTarget = nullptr;
                const char*    cTarget = nullptr;
                switch (stage)
                {
                    case ShaderStage::VERTEX:   wTarget = L"vs_6_6"; cTarget = "vs_6_6"; break;
                    case ShaderStage::PIXEL:    wTarget = L"ps_6_6"; cTarget = "ps_6_6"; break;
                    case ShaderStage::COMPUTE:  wTarget = L"cs_6_6"; cTarget = "cs_6_6"; break;
                    case ShaderStage::GEOMETRY: wTarget = L"gs_6_6"; cTarget = "gs_6_6"; break;
                    case ShaderStage::HULL:     wTarget = L"hs_6_6"; cTarget = "hs_6_6"; break;
                    case ShaderStage::DOMAIN:   wTarget = L"ds_6_6"; cTarget = "ds_6_6"; break;
                    default: break;
                }
                if (!wTarget || !cTarget)
                {
                    return 0;
                }

                const UINT32 dxcSaltFlags = 0x00662021u;
                const DXBCCacheKey cacheKey = ComputeDXBCCacheKey(cTarget, dxcSaltFlags, bytecode, bytecodeSize);
                if (TryLoadCachedDXBC(cacheKey, res.bytecode))
                {
                    populateBinaryCaps();
                    NativeHandle handle = GenerateHandle();
                    res.generation = handle;
                    m_shaders[handle] = std::move(res);
                    return handle;
                }

                const wchar_t* wEntryPoints[4] = { L"Main", L"main", nullptr, nullptr };
                if (stage == ShaderStage::VERTEX)       wEntryPoints[2] = L"VSMain";
                else if (stage == ShaderStage::PIXEL)   wEntryPoints[2] = L"PSMain";
                else if (stage == ShaderStage::COMPUTE) wEntryPoints[2] = L"CSMain";

                BOOL        compiledOk = false;
                std::string firstRealError;
                for (const wchar_t* ep : wEntryPoints)
                {
                    if (!ep) continue;
                    std::string err2021;
                    if (CompileHLSLWithDXC(bytecode, bytecodeSize, wTarget, ep, L"2021", res.bytecode, err2021))
                    {
                        compiledOk = true;
                        break;
                    }
                    std::string err2018;
                    if (CompileHLSLWithDXC(bytecode, bytecodeSize, wTarget, ep, L"2018", res.bytecode, err2018))
                    {
                        compiledOk = true;
                        break;
                    }
                    const std::string& candidateErr = !err2021.empty() ? err2021 : err2018;
                    if (!candidateErr.empty() && firstRealError.empty() &&
                        candidateErr.find("missing entry point") == std::string::npos)
                    {
                        firstRealError = candidateErr;
                    }
                }

                if (compiledOk && !res.bytecode.empty())
                {
                    StoreCachedDXBC(cacheKey, res.bytecode.data(), res.bytecode.size());
                    populateBinaryCaps();
                    NativeHandle handle = GenerateHandle();
                    res.generation = handle;
                    m_shaders[handle] = std::move(res);
                    return handle;
                }

                if (!firstRealError.empty())
                {
                    std::cerr << "[D3D12ResourceManager] DXC compile error (" << cTarget << "):\n"
                              << firstRealError << "\n";
                }
                return 0;
#endif
            }

            // Fallback: compile via D3DCompile (Shader Model 5.1) when adapter < SM 6.6 or DXC runtime is unavailable
            const char* target = nullptr;
            if (stage == ShaderStage::VERTEX)
                target = "vs_5_1";
            else if (stage == ShaderStage::PIXEL)
                target = "ps_5_1";
            else if (stage == ShaderStage::COMPUTE)
                target = "cs_5_1";
            else if (stage == ShaderStage::GEOMETRY)
                target = "gs_5_1";
            else if (stage == ShaderStage::HULL)
                target = "hs_5_1";
            else if (stage == ShaderStage::DOMAIN)
                target = "ds_5_1";

            if (target)
            {
                const UINT32 compileFlags = D3DCOMPILE_OPTIMIZATION_LEVEL1;
                const DXBCCacheKey cacheKey = ComputeDXBCCacheKey(target, compileFlags, bytecode, bytecodeSize);
                if (TryLoadCachedDXBC(cacheKey, res.bytecode))
                {
                    populateBinaryCaps();
                    NativeHandle handle = GenerateHandle();
                    res.generation = handle;
                    m_shaders[handle] = std::move(res);
                    return handle;
                }

                ComPtr<ID3DBlob> blob;
                ComPtr<ID3DBlob> errorBlob;

                // Try common entry point names in order: "Main", "main", stage-specific ("VSMain", "PSMain", "CSMain")
                const char* entryPoints[4] = { "Main", "main", nullptr, nullptr };
                if (stage == ShaderStage::VERTEX) entryPoints[2] = "VSMain";
                else if (stage == ShaderStage::PIXEL) entryPoints[2] = "PSMain";
                else if (stage == ShaderStage::COMPUTE) entryPoints[2] = "CSMain";

                HRESULT hr = E_FAIL;
                ComPtr<ID3DBlob> firstRealErrorBlob;
                for (const char* ep : entryPoints)
                {
                    if (!ep) continue;
                    blob.Reset();
                    errorBlob.Reset();
                    // Use D3DCOMPILE_OPTIMIZATION_LEVEL1: avoids FXC O3 exponential compile stalls on
                    // complex branching/PCF/PCSS shaders while preserving full DCE and register optimization.
                    hr = D3DCompile(
                        bytecode,
                        bytecodeSize,
                        "NorthstarShader.hlsl",
                        nullptr,
                        nullptr,
                        ep,
                        target,
                        compileFlags,
                        0,
                        &blob,
                        &errorBlob
                    );
                    if (SUCCEEDED(hr) && blob)
                    {
                        break;
                    }
                    if (errorBlob && !firstRealErrorBlob)
                    {
                        const char* errText = static_cast<const char*>(errorBlob->GetBufferPointer());
                        if (errText && std::strstr(errText, "X3501") == nullptr)
                        {
                            firstRealErrorBlob = errorBlob;
                        }
                    }
                }

                if (SUCCEEDED(hr) && blob)
                {
                    StoreCachedDXBC(cacheKey, blob->GetBufferPointer(), blob->GetBufferSize());
                    res.blob = blob;
                    res.bytecode.resize(blob->GetBufferSize());
                    std::memcpy(res.bytecode.data(), blob->GetBufferPointer(), blob->GetBufferSize());
                    populateBinaryCaps();
                }
                else
                {
                    ComPtr<ID3DBlob> reportErr = firstRealErrorBlob ? firstRealErrorBlob : errorBlob;
                    if (reportErr)
                    {
                        std::cerr << "[D3D12ResourceManager] D3DCompile error (" << target << "):\n"
                                  << (const char*)reportErr->GetBufferPointer() << "\n";
                    }
                    return 0;
                }
            }
            else
            {
                return 0;
            }
        }

        NativeHandle handle = GenerateHandle();
        res.generation = handle;
        m_shaders[handle] = std::move(res);
        return handle;
    }

    BOOL D3D12ResourceManager::GetShaderCompiledBytecode(NativeHandle handle, std::vector<BYTE>& outBytecode) const
    {
        outBytecode.clear();
        auto it = m_shaders.find(handle);
        if (it == m_shaders.end()) return false;
        if (!it->second.bytecode.empty() && it->second.bytecode.size() >= 4u &&
            std::memcmp(it->second.bytecode.data(), "DXBC", 4) == 0)
        {
            outBytecode = it->second.bytecode;
            return true;
        }
        if (it->second.blob && it->second.blob->GetBufferSize() >= 4u &&
            std::memcmp(it->second.blob->GetBufferPointer(), "DXBC", 4) == 0)
        {
            outBytecode.resize(it->second.blob->GetBufferSize());
            std::memcpy(outBytecode.data(), it->second.blob->GetBufferPointer(), outBytecode.size());
            return true;
        }
        return false;
    }

    void D3D12ResourceManager::SetShaderInputLayout(NativeHandle handle, const std::vector<NSCStageIOEntry>& inputLayout)
    {
        auto it = m_shaders.find(handle);
        if (it != m_shaders.end())
        {
            it->second.inputLayout = inputLayout;
        }
    }

    void D3D12ResourceManager::SetShaderReflectionSpaces(NativeHandle handle, const ShaderSpaceMask& spaceMask)
    {
        auto it = m_shaders.find(handle);
        if (it != m_shaders.end())
        {
            it->second.spaceMask = spaceMask;
            it->second.generation = (it->second.generation * 1099511628211ULL) ^
                                    (static_cast<UINT64>(spaceMask.cbvSrvMask) << 32) ^
                                    (static_cast<UINT64>(spaceMask.samplerMask) << 16) ^
                                    static_cast<UINT64>(spaceMask.uavMask);
        }
    }

    BOOL D3D12ResourceManager::CopyGPUBufferRegion(
        D3D12DeviceContext&        ctx,
        ID3D12GraphicsCommandList* activeFrameCmdList,
        D3D12ResourceStateTracker& stateTracker,
        NativeHandle               dstHandle,
        UINT64                     dstOffset,
        NativeHandle               srcHandle,
        UINT64                     srcOffset,
        UINT64                     byteSize)
    {
        if (byteSize == 0 || dstHandle == 0 || srcHandle == 0 || dstHandle == srcHandle)
        {
            return false;
        }
        D3D12GPUBufferResource* dstBuf = GetGPUBuffer(dstHandle);
        D3D12GPUBufferResource* srcBuf = GetGPUBuffer(srcHandle);
        if (!dstBuf || !dstBuf->resource || dstBuf->pendingDestroy ||
            !srcBuf || !srcBuf->resource || srcBuf->pendingDestroy)
        {
            return false;
        }
        if (dstOffset > dstBuf->byteSize || byteSize > (dstBuf->byteSize - dstOffset) ||
            srcOffset > srcBuf->byteSize || byteSize > (srcBuf->byteSize - srcOffset))
        {
            return false;
        }

        auto recordCopy = [&](ID3D12GraphicsCommandList* cmdList) {
            // Ensure any queued CPU->GPU uploads targeting srcBuf execute before copying from srcBuf
            if (!m_pendingBufferUploads.empty() && activeFrameCmdList != nullptr)
            {
                FlushPendingGPUBufferUploads(cmdList, stateTracker, ctx.GetFrameFenceValue() + 1ULL);
            }
            stateTracker.TransitionResource(srcBuf->resource.Get(), GPUResourceAccess::COPY_SOURCE);
            stateTracker.TransitionResource(dstBuf->resource.Get(), GPUResourceAccess::COPY_DEST);
            stateTracker.FlushBarriers(cmdList);
            srcBuf->currentState = stateTracker.GetResourceState(srcBuf->resource.Get());
            dstBuf->currentState = stateTracker.GetResourceState(dstBuf->resource.Get());

            cmdList->CopyBufferRegion(
                dstBuf->resource.Get(),
                dstOffset,
                srcBuf->resource.Get(),
                srcOffset,
                byteSize);

            stateTracker.TransitionResource(srcBuf->resource.Get(), srcBuf->defaultAccess);
            stateTracker.TransitionResource(dstBuf->resource.Get(), dstBuf->defaultAccess);
            stateTracker.FlushBarriers(cmdList);
            srcBuf->currentState = stateTracker.GetResourceState(srcBuf->resource.Get());
            dstBuf->currentState = stateTracker.GetResourceState(dstBuf->resource.Get());
        };

        if (activeFrameCmdList != nullptr)
        {
            recordCopy(activeFrameCmdList);
            return true;
        }

        // Queue non-blocking GPU-to-GPU copy in FIFO upload queue (zero WaitForGPU CPU stall; preserves ComPtr lifetime!)
        if (dstOffset <= UINT32_MAX && byteSize <= UINT32_MAX)
        {
            D3D12PendingBufferUpload pending{};
            pending.targetHandle       = dstHandle;
            pending.srcGPUHandle       = srcHandle;
            pending.srcFallbackState   = srcBuf->currentState;
            pending.uploadResource     = srcBuf->resource;
            pending.isStandaloneUpload = true; // Retires ComPtr reference safely at upcomingFenceValue
            pending.srcOffset          = srcOffset;
            pending.dstOffset          = static_cast<UINT32>(dstOffset);
            pending.byteSize           = static_cast<UINT32>(byteSize);
            m_pendingBufferUploads.push_back(std::move(pending));
            return true;
        }

        // Fallback for >4 GiB copies outside frame recording
        ctx.ExecuteUpload([&](ID3D12GraphicsCommandList* uploadList) {
            FlushPendingGPUBufferUploads(uploadList, stateTracker, ctx.GetFrameFenceValue());
            recordCopy(uploadList);
        });
        return true;
    }

    void D3D12ResourceManager::DestroyShader(NativeHandle handle)
    {
        m_shaders.erase(handle);
    }

    BOOL D3D12ResourceManager::HasVertexBuffer(NativeHandle handle) const
    {
        return m_vertexBuffers.find(handle) != m_vertexBuffers.end();
    }

    BOOL D3D12ResourceManager::HasIndexBuffer(NativeHandle handle) const
    {
        return m_indexBuffers.find(handle) != m_indexBuffers.end();
    }

    BOOL D3D12ResourceManager::HasConstantBuffer(NativeHandle handle) const
    {
        return m_constantBuffers.find(handle) != m_constantBuffers.end();
    }

    BOOL D3D12ResourceManager::HasTexture(NativeHandle handle) const
    {
        return m_textures.find(handle) != m_textures.end();
    }

    BOOL D3D12ResourceManager::HasSampler(NativeHandle handle) const
    {
        return m_samplers.find(handle) != m_samplers.end();
    }

    BOOL D3D12ResourceManager::HasShader(NativeHandle handle) const
    {
        return m_shaders.find(handle) != m_shaders.end();
    }

    BOOL D3D12ResourceManager::HasGPUBuffer(NativeHandle handle) const
    {
        auto it = m_gpuBuffers.find(handle);
        return (it != m_gpuBuffers.end()) && !it->second.pendingDestroy;
    }

    UINT32 D3D12ResourceManager::GetActiveResourceCount() const
    {
        UINT32 activeGpuBuffers = 0;
        for (const auto& pair : m_gpuBuffers)
        {
            if (!pair.second.pendingDestroy)
            {
                activeGpuBuffers++;
            }
        }
        return static_cast<UINT32>(
            m_vertexBuffers.size() +
            m_indexBuffers.size() +
            m_constantBuffers.size() +
            m_textures.size() +
            m_samplers.size() +
            m_shaders.size()
        ) + activeGpuBuffers;
    }

    UINT32 D3D12ResourceManager::GetPendingRetiredResourceCount() const
    {
        UINT32 pendingDestroyCount = 0;
        for (const auto& pair : m_gpuBuffers)
        {
            if (pair.second.pendingDestroy)
            {
                pendingDestroyCount++;
            }
        }
        return static_cast<UINT32>(m_retiredResources.size()) + pendingDestroyCount;
    }

    UINT32 D3D12ResourceManager::GetUnsubmittedUploadRingRegionCount() const
    {
        UINT32 count = 0;
        for (const auto& region : m_uploadRingInFlight)
        {
            if (region.fenceValue == UINT64_MAX)
            {
                count++;
            }
        }
        return count;
    }

    const D3D12VertexBufferResource* D3D12ResourceManager::GetVertexBuffer(NativeHandle handle) const
    {
        auto it = m_vertexBuffers.find(handle);
        return (it != m_vertexBuffers.end()) ? &it->second : nullptr;
    }

    const D3D12IndexBufferResource* D3D12ResourceManager::GetIndexBuffer(NativeHandle handle) const
    {
        auto it = m_indexBuffers.find(handle);
        return (it != m_indexBuffers.end()) ? &it->second : nullptr;
    }

    const D3D12ConstantBufferResource* D3D12ResourceManager::GetConstantBuffer(NativeHandle handle) const
    {
        auto it = m_constantBuffers.find(handle);
        return (it != m_constantBuffers.end()) ? &it->second : nullptr;
    }

    const D3D12TextureResource* D3D12ResourceManager::GetTexture(NativeHandle handle) const
    {
        auto it = m_textures.find(handle);
        return (it != m_textures.end()) ? &it->second : nullptr;
    }

    D3D12TextureResource* D3D12ResourceManager::GetTexture(NativeHandle handle)
    {
        auto it = m_textures.find(handle);
        return (it != m_textures.end()) ? &it->second : nullptr;
    }

    BOOL D3D12ResourceManager::HasTextureSharedHandle(NativeHandle handle) const
    {
        const D3D12TextureResource* tex = GetTexture(handle);
        return tex != nullptr && tex->sharedHandle != nullptr;
    }

    HANDLE D3D12ResourceManager::GetTextureSharedHandle(NativeHandle handle) const
    {
        const D3D12TextureResource* tex = GetTexture(handle);
        if (!tex || !tex->sharedHandle)
        {
            return nullptr;
        }
        HANDLE duplicatedHandle = nullptr;
        if (!::DuplicateHandle(
                ::GetCurrentProcess(),
                tex->sharedHandle,
                ::GetCurrentProcess(),
                &duplicatedHandle,
                0,
                FALSE,
                DUPLICATE_SAME_ACCESS))
        {
            return nullptr;
        }
        return duplicatedHandle;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE D3D12ResourceManager::GetTextureGPUDescriptorHandle(
        NativeHandle handle,
        const D3D12DescriptorPool& srvPool) const
    {
        const D3D12TextureResource* tex = GetTexture(handle);
        if (!tex || tex->srvDescriptorIndex == UINT32_MAX)
        {
            return {};
        }
        return srvPool.GetGPUHandle(tex->srvDescriptorIndex);
    }

    const D3D12SamplerResource* D3D12ResourceManager::GetSampler(NativeHandle handle) const
    {
        auto it = m_samplers.find(handle);
        return (it != m_samplers.end()) ? &it->second : nullptr;
    }

    const D3D12ShaderResource* D3D12ResourceManager::GetShader(NativeHandle handle) const
    {
        auto it = m_shaders.find(handle);
        return (it != m_shaders.end()) ? &it->second : nullptr;
    }

    const D3D12GPUBufferResource* D3D12ResourceManager::GetGPUBuffer(NativeHandle handle) const
    {
        auto it = m_gpuBuffers.find(handle);
        return (it != m_gpuBuffers.end()) ? &it->second : nullptr;
    }

    D3D12GPUBufferResource* D3D12ResourceManager::GetGPUBuffer(NativeHandle handle)
    {
        auto it = m_gpuBuffers.find(handle);
        return (it != m_gpuBuffers.end()) ? &it->second : nullptr;
    }

    NativeHandle D3D12ResourceManager::GetOrCreateCachedCpuVertexBuffer(
        D3D12DeviceContext& ctx,
        const void* cpuVertices,
        UINT32 vertexCount,
        UINT32 stride)
    {
        if (!cpuVertices || vertexCount == 0 || stride == 0) return 0;

        auto it = m_cpuMeshToGpuVb.find(cpuVertices);
        if (it != m_cpuMeshToGpuVb.end())
        {
            return it->second;
        }

        NativeHandle handle = CreateVertexBuffer(ctx, cpuVertices, vertexCount * stride, stride);
        if (handle != 0)
        {
            m_cpuMeshToGpuVb[cpuVertices] = handle;
        }
        return handle;
    }

    NativeHandle D3D12ResourceManager::GetOrCreateCachedCpuIndexBuffer(
        D3D12DeviceContext& ctx,
        const void* cpuIndices,
        UINT32 indexCount,
        BOOL is32Bit)
    {
        if (!cpuIndices || indexCount == 0) return 0;

        auto it = m_cpuMeshToGpuIb.find(cpuIndices);
        if (it != m_cpuMeshToGpuIb.end())
        {
            return it->second;
        }

        const UINT32 byteSize = indexCount * (is32Bit ? sizeof(UINT32) : sizeof(UINT16));
        NativeHandle handle = CreateIndexBuffer(ctx, cpuIndices, byteSize, is32Bit);
        if (handle != 0)
        {
            m_cpuMeshToGpuIb[cpuIndices] = handle;
        }
        return handle;
    }

    void D3D12ResourceManager::Shutdown(
        D3D12DescriptorPool& cbvPool,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        D3D12DescriptorPool& samplerPool)
    {
        // Flush all pending retired resources
        ProcessRetirementQueue(UINT64_MAX, cbvPool, rtvPool, dsvPool, samplerPool);
        m_retiredResources.clear();
        // Unmap constant buffers
        for (auto& pair : m_constantBuffers)
        {
            if (pair.second.uploadBuffer && pair.second.mappedData)
            {
                pair.second.uploadBuffer->Unmap(0, nullptr);
                pair.second.mappedData = nullptr;
            }
            cbvPool.Free(pair.second.descriptorIndex);
        }
        m_constantBuffers.clear();

        // Release textures and free their descriptors
        for (auto& pair : m_textures)
        {
            if (pair.second.sharedHandle)
            {
                CloseHandle(pair.second.sharedHandle);
                pair.second.sharedHandle = nullptr;
            }
            if (pair.second.srvDescriptorIndex != UINT32_MAX)
                cbvPool.Free(pair.second.srvDescriptorIndex);
            if (pair.second.uavDescriptorIndex != UINT32_MAX)
                cbvPool.Free(pair.second.uavDescriptorIndex);
            for (UINT32 mSrv : pair.second.mipSrvDescriptors)
            {
                if (mSrv != UINT32_MAX) cbvPool.Free(mSrv);
            }
            for (SIZE_T m = 1u; m < pair.second.mipUavDescriptors.size(); ++m)
            {
                if (pair.second.mipUavDescriptors[m] != UINT32_MAX)
                    cbvPool.Free(pair.second.mipUavDescriptors[m]);
            }
            if (pair.second.rtvDescriptorIndex != UINT32_MAX)
                rtvPool.Free(pair.second.rtvDescriptorIndex);
            if (pair.second.dsvDescriptorIndex != UINT32_MAX && pair.second.dsvLayerDescriptors.empty())
                dsvPool.Free(pair.second.dsvDescriptorIndex);
            if (pair.second.readOnlyDsvDescriptorIndex != UINT32_MAX)
                dsvPool.Free(pair.second.readOnlyDsvDescriptorIndex);
            if (pair.second.fullArrayDsvDescriptorIndex != UINT32_MAX)
                dsvPool.Free(pair.second.fullArrayDsvDescriptorIndex);
            for (UINT32 sliceDsv : pair.second.dsvLayerDescriptors) dsvPool.Free(sliceDsv);
        }
        m_textures.clear();

        // Release samplers and free descriptors
        for (auto& pair : m_samplers)
        {
            samplerPool.Free(pair.second.samplerDescriptorIndex);
        }
        m_samplers.clear();

        // Release GPU buffers and free their SRV/UAV descriptors
        for (auto& pair : m_gpuBuffers)
        {
            if (pair.second.srvDescriptorIndex != UINT32_MAX)
                cbvPool.Free(pair.second.srvDescriptorIndex);
            if (pair.second.uavDescriptorIndex != UINT32_MAX)
                cbvPool.Free(pair.second.uavDescriptorIndex);
        }
        m_gpuBuffers.clear();

        m_pendingBufferUploads.clear();
        m_uploadRingInFlight.clear();
        if (m_uploadRingBuffer && m_uploadRingMappedData)
        {
            m_uploadRingBuffer->Unmap(0, nullptr);
            m_uploadRingMappedData = nullptr;
        }
        m_uploadRingBuffer.Reset();
        m_uploadRingWriteOffset = 0;
        m_uploadRingReadOffset  = 0;
        m_hasPendingDestroyedGPUBuffers = false;
        m_uploadedBytesThisSubmission = 0;
        m_gpuBufferAllocationsThisSubmission = 0;

        m_vertexBuffers.clear();
        m_indexBuffers.clear();
        m_shaders.clear();

        m_cpuMeshToGpuVb.clear();
        m_cpuMeshToGpuIb.clear();
    }
}
