#include "northstar_d3d12.h"
#include "d3d12_common.h"
#include "d3d12_device.h"
#include "d3d12_swapchain.h"
#include "d3d12_descriptor_pool.h"
#include "d3d12_resource_manager.h"
#include "d3d12_resource_state_tracker.h"
#include "d3d12_pipeline.h"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <array>
#include <chrono>
#include <thread>
#include <cassert>

namespace Northstar
{
    using namespace D3D12Util;

    struct NorthstarD3D12::Impl
    {
        D3D12DeviceContext    deviceContext;
        D3D12SwapChain        swapChain;
        D3D12DescriptorPool   cbvSrvUavPool;
        D3D12DescriptorPool   samplerPool;
        D3D12DescriptorPool   rtvPool;
        D3D12DescriptorPool        dsvPool;
        D3D12ResourceManager       resourceMgr;
        D3D12ResourceStateTracker  stateTracker;
        D3D12Pipeline              pipeline;
        std::thread::id            ownerThreadId{};
        std::unordered_map<void*, ComPtr<ID3D12Fence>> importedSharedFences;

        void AssertOwnerThread() const noexcept
        {
#ifndef NDEBUG
            if (isInitialized && ownerThreadId != std::thread::id{})
            {
                assert(std::this_thread::get_id() == ownerThreadId &&
                       "NorthstarD3D12 command/resource methods must be invoked from the initializing owner thread.");
            }
#endif
        }

        static constexpr UINT32 kMaxFramesInFlight         = D3D12Pipeline::kBackBufferCount; // 3

        ComPtr<ID3D12CommandAllocator>    frameCmdAlloc[kMaxFramesInFlight];
        ComPtr<ID3D12GraphicsCommandList> frameCmdList;
        UINT64                            frameFenceValues[kMaxFramesInFlight] = { 0, 0, 0 };
        UINT64                            swapChainFenceValues[D3D12SwapChain::kBackBufferCount] = { 0, 0 };
        UINT32                            maxFramesInFlight = 2;
        UINT64                            submissionSlotCursor = 0;

        static constexpr UINT32 kCBVPerSpace               = D3D12Pipeline::kCBVPerSpace;               // 16
        static constexpr UINT32 kSRVPerSpace               = D3D12Pipeline::kSRVPerSpace;               // 32
        static constexpr UINT32 kUAVPerSpace               = D3D12Pipeline::kUAVPerSpace;               // 16
        static constexpr UINT32 kGraphicsDescPerSpace      = kCBVPerSpace + kSRVPerSpace;               // 48
        static constexpr UINT32 kDescriptorsPerSpaceTable  = D3D12Pipeline::kDescriptorsPerSpaceTable;  // 64
        static constexpr UINT32 kMaxSpaces                 = D3D12Pipeline::kMaxSpaces;                 // 4
        static constexpr UINT32 kGraphicsSamplerParamBase  = D3D12Pipeline::kGraphicsSamplerParamBase;  // 8
        static constexpr UINT32 kComputeSamplerParamBase   = D3D12Pipeline::kComputeSamplerParamBase;   // 4
        static constexpr UINT32 kComputeRootConstantsParam = D3D12Pipeline::kComputeRootConstantsParamIndex; // 8
        static constexpr UINT32 kTablesPerBackBuffer       = 6144;
        static constexpr UINT32 kRingDescriptorsPerBB      = 4608 * kDescriptorsPerSpaceTable; // 294912 (3 * 294912 = 884736 < 1000000)
        UINT32 dynamicTableBase[kMaxFramesInFlight] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
        BOOL   isRecordingCmdList = false;

        UINT64 GetSafeRetireFenceValue() const
        {
            return deviceContext.GetFrameFenceValue() + (isRecordingCmdList ? 1ULL : 0ULL);
        }

        UINT64 GetSafeCompletedFenceValue() const
        {
            ID3D12Fence* fence = deviceContext.GetFrameFence();
            if (!fence || deviceContext.IsDeviceRemoved())
                return 0ULL;
            const UINT64 val = fence->GetCompletedValue();
            if (val == UINT64_MAX)
                return 0ULL; // D3D12 returns UINT64_MAX when the device has been removed/hung
            return val;
        }

        static void ComputeTextureSubresourceTopology(
            const TextureDesc& d,
            UINT32& outMips,
            UINT32& outSlices,
            UINT32& outPlanes) noexcept
        {
            outMips   = (std::max)(1u, d.mipLevels);
            outSlices = ComputeTextureArraySliceCount(d.type, d.arrayLayers);
            outPlanes = IsTextureFormatDepthStencil(d.format) ? 2u : 1u;
        }

        static UINT32 ComputeTextureSubresourceCount(const TextureDesc& d) noexcept
        {
            UINT32 mips = 1u, slices = 1u, planes = 1u;
            ComputeTextureSubresourceTopology(d, mips, slices, planes);
            return mips * slices * planes;
        }

        static void RegisterTrackedTexture(
            D3D12ResourceStateTracker&  tracker,
            const D3D12TextureResource* texRes)
        {
            if (!texRes || !texRes->resource) return;
            UINT32 mips = 1u, slices = 1u, planes = 1u;
            ComputeTextureSubresourceTopology(texRes->desc, mips, slices, planes);

            GPUResourceAccess initAccess = GPUResourceAccess::COMMON;
            if (texRes->currentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                initAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
            }
            else if ((texRes->currentState & (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) != 0)
            {
                initAccess = GPUResourceAccess::ALL_SHADER_RESOURCE;
            }
            tracker.RegisterResource(
                texRes->resource.Get(),
                texRes->currentState,
                initAccess,
                mips * slices * planes,
                mips,
                slices,
                planes);
        }

        static void WriteNullTextureSRV(
            ID3D12Device*               device,
            D3D12_CPU_DESCRIPTOR_HANDLE dst,
            TextureType                 viewType,
            TextureFormat               fmt)
        {
            if (!device || dst.ptr == 0) return;

            DXGI_FORMAT dxgiFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            if (fmt == TextureFormat::R32_FLOAT || fmt == TextureFormat::D32_FLOAT || fmt == TextureFormat::D32_FLOAT_S8)
                dxgiFmt = DXGI_FORMAT_R32_FLOAT;
            else if (fmt == TextureFormat::D16_UNORM)
                dxgiFmt = DXGI_FORMAT_R16_UNORM;
            else if (fmt != TextureFormat::UNKNOWN && !IsTextureFormatDepth(fmt))
                dxgiFmt = ToDxgiFormat(fmt);

            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
            nullSrv.Format                  = dxgiFmt;
            nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            switch (viewType)
            {
                case TextureType::TEXTURE_2D_ARRAY:
                    nullSrv.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                    nullSrv.Texture2DArray.MostDetailedMip = 0;
                    nullSrv.Texture2DArray.MipLevels       = 1;
                    nullSrv.Texture2DArray.FirstArraySlice = 0;
                    nullSrv.Texture2DArray.ArraySize       = 1;
                    break;
                case TextureType::TEXTURE_CUBE:
                    nullSrv.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURECUBE;
                    nullSrv.TextureCube.MostDetailedMip   = 0;
                    nullSrv.TextureCube.MipLevels         = 1;
                    break;
                case TextureType::TEXTURE_CUBE_ARRAY:
                    nullSrv.ViewDimension                      = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                    nullSrv.TextureCubeArray.MostDetailedMip   = 0;
                    nullSrv.TextureCubeArray.MipLevels         = 1;
                    nullSrv.TextureCubeArray.First2DArrayFace  = 0;
                    nullSrv.TextureCubeArray.NumCubes          = 1;
                    break;
                case TextureType::TEXTURE_3D:
                    nullSrv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE3D;
                    nullSrv.Texture3D.MostDetailedMip = 0;
                    nullSrv.Texture3D.MipLevels       = 1;
                    break;
                default:
                    nullSrv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
                    nullSrv.Texture2D.MostDetailedMip = 0;
                    nullSrv.Texture2D.MipLevels       = 1;
                    break;
            }
            device->CreateShaderResourceView(nullptr, &nullSrv, dst);
        }

        void ResetGraphicsSamplerCache(UINT32 outSamplerCache[5][kMaxSpaces]) const noexcept
        {
            for (UINT32 st = 0u; st < 5u; ++st)
            {
                for (UINT32 sp = 0u; sp < kMaxSpaces; ++sp)
                {
                    outSamplerCache[st][sp] = defaultDummySamplerTableIndex;
                }
            }
        }

        struct RetiredSamplerTable
        {
            UINT64 fenceValue = 0;
            UINT32 startIdx   = UINT32_MAX;
        };
        std::vector<RetiredSamplerTable> retiredSamplerTables;

        void ProcessRetiredSamplerTables(UINT64 completedFenceValue)
        {
            auto it = retiredSamplerTables.begin();
            while (it != retiredSamplerTables.end())
            {
                if (completedFenceValue >= it->fenceValue)
                {
                    if (it->startIdx != UINT32_MAX)
                    {
                        samplerPool.FreeRange(it->startIdx, 16u);
                    }
                    it = retiredSamplerTables.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        UINT32 defaultDummySpaceTableIndex = UINT32_MAX;
        UINT32 defaultDummySamplerTableIndex = UINT32_MAX;
        ComPtr<ID3D12Resource> defaultDummyZeroCBuffer;
        NativeHandle defaultSamplerHandle = 0;
        PrePresentCallback prePresentCallback = nullptr;
        PostFrameExecutionCallback postFrameExecutionCallback = nullptr;
        BOOL postFrameWaitForGPU = true;

        // GPU Timestamp Queries & Backend Frame Metrics
        // Dynamic span-based query layout per frame slot (up to 128 queries / 64 spans per backbuffer)
        static constexpr UINT32 kQueriesPerBackBuffer = 128;
        static constexpr UINT32 kMaxSpansPerBackBuffer = 64;

        enum class TimestampSpanKind : UINT8
        {
            UPLOAD              = 0,
            PRE_COMPUTE         = 1,
            PRE_LAYERED_DEPTH   = 2,
            INTER_LAYERED_DEPTH = 3,
            SHADOW_RASTER       = 4,
            DEPTH_PREPASS       = 5,
            NVS_FEEDBACK        = 6,
            NVS_RESIDENCY_CULL  = 7,
            INTER_COMPUTE       = 8,
            POST_COMPUTE        = 9,
            MAIN_SPAN           = 10,
            FULL_FRAME          = 11,
            NVG_HZB             = 12,
            NVG_CULL_LOD        = 13,
            NVG_DRAW_GEN        = 14
        };

        struct TimestampSpanRecord
        {
            UINT16            startIdx = 0;
            UINT16            endIdx   = 0;
            TimestampSpanKind kind     = TimestampSpanKind::FULL_FRAME;
        };

        struct TimestampSlotFlags
        {
            BOOL                valid                = false;
            BOOL                isStandaloneLayered  = false;
            BOOL                isStandaloneCompute  = false;
            BOOL                hadInterleavedShadow = false;
            UINT32              queryCount           = 0;
            UINT32              spanCount            = 0;
            TimestampSpanRecord spans[kMaxSpansPerBackBuffer]{};
        };

        ComPtr<ID3D12QueryHeap> timestampQueryHeap;
        ComPtr<ID3D12Resource>  timestampReadbackBuffer;
        UINT64                  gpuTimestampFrequency = 0;
        TimestampSlotFlags      timestampSlotFlags[kMaxFramesInFlight]{};
        BackendFrameMetrics     lastFrameMetrics{};

        struct CachedSharedSBUpload
        {
            ID3D12Resource* resource     = nullptr;
            UINT64          firstElement = 0;
            UINT32          numElements  = 0;
            UINT32          stride       = 0;
        };
        std::unordered_map<UINT64, CachedSharedSBUpload> cachedSharedSBUploads;

        BOOL AllocateOrReuseStructuredBuffer(
            const NativeStructuredBufferBinding& sb,
            UINT32                               stride,
            CachedSharedSBUpload&                outUpload)
        {
            const UINT32 elemCount = sb.elementCount > 0 ? sb.elementCount : (sb.byteSize / stride);
            if (sb.sharedDataId != 0)
            {
                const UINT64 cacheKey = (sb.sharedDataId * 1099511628211ull) ^
                                        (static_cast<UINT64>(stride) << 32) ^
                                        static_cast<UINT64>(elemCount);
                auto it = cachedSharedSBUploads.find(cacheKey);
                if (it != cachedSharedSBUploads.end() && it->second.resource != nullptr)
                {
                    outUpload = it->second;
                    return true;
                }
                const BYTE* data = sb.GetData();
                if (!data || sb.byteSize == 0)
                {
                    return false;
                }
                const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateStructuredBuffer(data, sb.byteSize, stride);
                ID3D12Resource* res = pipeline.GetSBufferUploadBuffer();
                if (va == 0 || !res)
                {
                    return false;
                }
                outUpload.resource     = res;
                outUpload.firstElement = (va - res->GetGPUVirtualAddress()) / stride;
                outUpload.numElements  = elemCount;
                outUpload.stride       = stride;
                cachedSharedSBUploads.emplace(cacheKey, outUpload);
                return true;
            }

            const BYTE* data = sb.GetData();
            if (!data || sb.byteSize == 0)
            {
                return false;
            }
            const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateStructuredBuffer(data, sb.byteSize, stride);
            ID3D12Resource* res = pipeline.GetSBufferUploadBuffer();
            if (va == 0 || !res)
            {
                return false;
            }
            outUpload.resource     = res;
            outUpload.firstElement = (va - res->GetGPUVirtualAddress()) / stride;
            outUpload.numElements  = elemCount;
            outUpload.stride       = stride;
            return true;
        }

        UINT32 EmitTimestamp(UINT32 queryBase, TimestampSlotFlags& slotFlags)
        {
            if (!timestampQueryHeap || !frameCmdList || slotFlags.queryCount >= kQueriesPerBackBuffer)
            {
                return UINT32_MAX;
            }
            const UINT32 localIdx = slotFlags.queryCount++;
            frameCmdList->EndQuery(timestampQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryBase + localIdx);
            return localIdx;
        }

        static void AddTimestampSpan(
            TimestampSlotFlags& slotFlags,
            UINT32              startIdx,
            UINT32              endIdx,
            TimestampSpanKind   kind) noexcept
        {
            if (startIdx == UINT32_MAX || endIdx == UINT32_MAX || slotFlags.spanCount >= kMaxSpansPerBackBuffer)
            {
                return;
            }
            TimestampSpanRecord rec{};
            rec.startIdx = static_cast<UINT16>(startIdx);
            rec.endIdx   = static_cast<UINT16>(endIdx);
            rec.kind     = kind;
            slotFlags.spans[slotFlags.spanCount++] = rec;
        }

        static TimestampSpanKind MapProfileGroupToSpanKind(
            ComputeDispatchProfileGroup group,
            TimestampSpanKind           genericKind) noexcept
        {
            if (group == ComputeDispatchProfileGroup::FEEDBACK)
            {
                return TimestampSpanKind::NVS_FEEDBACK;
            }
            if (group == ComputeDispatchProfileGroup::RESIDENCY_CULL)
            {
                return TimestampSpanKind::NVS_RESIDENCY_CULL;
            }
            if (group == ComputeDispatchProfileGroup::NVG_HZB)
            {
                return TimestampSpanKind::NVG_HZB;
            }
            if (group == ComputeDispatchProfileGroup::NVG_CULL_LOD)
            {
                return TimestampSpanKind::NVG_CULL_LOD;
            }
            if (group == ComputeDispatchProfileGroup::NVG_DRAW_GEN)
            {
                return TimestampSpanKind::NVG_DRAW_GEN;
            }
            return genericKind;
        }

        void FinalizeTimestampSlot(UINT32 bbIndex, UINT32 queryBase, TimestampSlotFlags& slotFlags)
        {
            if (timestampQueryHeap && timestampReadbackBuffer && frameCmdList && slotFlags.queryCount > 0)
            {
                frameCmdList->ResolveQueryData(
                    timestampQueryHeap.Get(),
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    queryBase,
                    slotFlags.queryCount,
                    timestampReadbackBuffer.Get(),
                    static_cast<UINT64>(queryBase) * sizeof(UINT64)
                );
                slotFlags.valid = true;
                timestampSlotFlags[bbIndex] = slotFlags;
            }
            else
            {
                timestampSlotFlags[bbIndex] = TimestampSlotFlags{};
            }
        }

        void ResolveTimestampSlot(UINT32 slot)
        {
            if (slot >= kMaxFramesInFlight) return;
            const TimestampSlotFlags flags = timestampSlotFlags[slot];
            if (!flags.valid || flags.queryCount == 0 || !timestampReadbackBuffer || gpuTimestampFrequency == 0)
            {
                timestampSlotFlags[slot] = TimestampSlotFlags{};
                return;
            }

            const UINT32 resolvedQueries = (std::min)(flags.queryCount, kQueriesPerBackBuffer);
            const SIZE_T byteOffset      = static_cast<SIZE_T>(slot) * kQueriesPerBackBuffer * sizeof(UINT64);
            D3D12_RANGE readRange{ byteOffset, byteOffset + static_cast<SIZE_T>(resolvedQueries) * sizeof(UINT64) };
            void* mapped = nullptr;
            if (SUCCEEDED(timestampReadbackBuffer->Map(0, &readRange, &mapped)) && mapped)
            {
                const UINT64* ticks = reinterpret_cast<const UINT64*>(static_cast<const BYTE*>(mapped) + byteOffset);
                const double freqMs = static_cast<double>(gpuTimestampFrequency) / 1000.0;
                auto deltaMs = [freqMs, resolvedQueries, ticks](UINT16 sIdx, UINT16 eIdx) -> FLOAT32 {
                    if (sIdx >= resolvedQueries || eIdx >= resolvedQueries)
                        return 0.0f;
                    return (ticks[eIdx] >= ticks[sIdx])
                        ? static_cast<FLOAT32>(static_cast<double>(ticks[eIdx] - ticks[sIdx]) / freqMs)
                        : 0.0f;
                };

                FLOAT32 uploadMs       = 0.0f;
                FLOAT32 preCompMs      = 0.0f;
                FLOAT32 preLayeredMs   = 0.0f;
                FLOAT32 interLayeredMs = 0.0f;
                FLOAT32 shadowRasterMs = 0.0f;
                FLOAT32 depthPreMs     = 0.0f;
                FLOAT32 feedbackMs     = 0.0f;
                FLOAT32 residencyMs    = 0.0f;
                FLOAT32 nvgHzbMs       = 0.0f;
                FLOAT32 nvgCullLodMs   = 0.0f;
                FLOAT32 nvgDrawGenMs   = 0.0f;
                FLOAT32 interCompMs    = 0.0f;
                FLOAT32 postCompMs     = 0.0f;
                FLOAT32 mainSpanMs     = 0.0f;
                FLOAT32 fullFrameMs    = 0.0f;

                const UINT32 spanN = (std::min)(flags.spanCount, kMaxSpansPerBackBuffer);
                for (UINT32 i = 0; i < spanN; ++i)
                {
                    const TimestampSpanRecord& sp = flags.spans[i];
                    const FLOAT32 d = deltaMs(sp.startIdx, sp.endIdx);
                    switch (sp.kind)
                    {
                    case TimestampSpanKind::UPLOAD:              uploadMs       += d; break;
                    case TimestampSpanKind::PRE_COMPUTE:         preCompMs      += d; break;
                    case TimestampSpanKind::PRE_LAYERED_DEPTH:   preLayeredMs   += d; break;
                    case TimestampSpanKind::INTER_LAYERED_DEPTH: interLayeredMs += d; break;
                    case TimestampSpanKind::SHADOW_RASTER:       shadowRasterMs += d; break;
                    case TimestampSpanKind::DEPTH_PREPASS:       depthPreMs     += d; break;
                    case TimestampSpanKind::NVS_FEEDBACK:        feedbackMs     += d; break;
                    case TimestampSpanKind::NVS_RESIDENCY_CULL:  residencyMs    += d; break;
                    case TimestampSpanKind::NVG_HZB:             nvgHzbMs       += d; break;
                    case TimestampSpanKind::NVG_CULL_LOD:        nvgCullLodMs   += d; break;
                    case TimestampSpanKind::NVG_DRAW_GEN:        nvgDrawGenMs   += d; break;
                    case TimestampSpanKind::INTER_COMPUTE:       interCompMs    += d; break;
                    case TimestampSpanKind::POST_COMPUTE:        postCompMs     += d; break;
                    case TimestampSpanKind::MAIN_SPAN:           mainSpanMs     += d; break;
                    case TimestampSpanKind::FULL_FRAME:          fullFrameMs    += d; break;
                    }
                }

                const FLOAT32 layeredMs   = preLayeredMs + interLayeredMs;
                const FLOAT32 totalCompMs = preCompMs + feedbackMs + residencyMs + nvgHzbMs + nvgCullLodMs + nvgDrawGenMs + interCompMs + postCompMs;

                lastFrameMetrics.gpuSceneUploadMs      = uploadMs;
                lastFrameMetrics.gpuComputePassMs      = totalCompMs;
                lastFrameMetrics.gpuNVSFeedbackMs      = feedbackMs;
                lastFrameMetrics.gpuNVSResidencyCullMs = residencyMs;
                lastFrameMetrics.gpuNVGHZBMs           = nvgHzbMs;
                lastFrameMetrics.gpuNVGCullLodMs       = nvgCullLodMs;
                lastFrameMetrics.gpuNVGDrawGenMs       = nvgDrawGenMs;
                lastFrameMetrics.gpuLayeredDepthPassMs = layeredMs;
                lastFrameMetrics.gpuDepthPrepassMs     = depthPreMs;

                if (flags.isStandaloneCompute)
                {
                    lastFrameMetrics.gpuShadowPassMs   = 0.0f;
                    lastFrameMetrics.gpuGeometryPassMs = 0.0f;
                    lastFrameMetrics.gpuTotalFrameMs   = fullFrameMs;
                }
                else if (flags.isStandaloneLayered)
                {
                    lastFrameMetrics.gpuShadowPassMs   = (shadowRasterMs > 0.0f) ? shadowRasterMs : (preCompMs + feedbackMs + residencyMs);
                    lastFrameMetrics.gpuGeometryPassMs = layeredMs + shadowRasterMs;
                    lastFrameMetrics.gpuTotalFrameMs   = fullFrameMs;
                }
                else
                {
                    lastFrameMetrics.gpuShadowPassMs = shadowRasterMs;
                    const FLOAT32 subPasses =
                        depthPreMs + feedbackMs + residencyMs + nvgHzbMs + nvgCullLodMs + nvgDrawGenMs + interCompMs + interLayeredMs + shadowRasterMs;
                    lastFrameMetrics.gpuGeometryPassMs = (mainSpanMs > subPasses) ? (mainSpanMs - subPasses) : 0.0f;
                    lastFrameMetrics.gpuTotalFrameMs   = fullFrameMs;
                }

                D3D12_RANGE emptyWrite{ 0, 0 };
                timestampReadbackBuffer->Unmap(0, &emptyWrite);
            }
            timestampSlotFlags[slot] = TimestampSlotFlags{};
        }

        struct SamplerTableKey
        {
            NativeHandle samplers[16]{};

            bool operator==(const SamplerTableKey& other) const
            {
                return std::memcmp(samplers, other.samplers, sizeof(samplers)) == 0;
            }
        };

        struct SamplerTableKeyHash
        {
            std::size_t operator()(const SamplerTableKey& k) const noexcept
            {
                std::size_t h = 0;
                for (int i = 0; i < 16; ++i)
                {
                    h ^= std::hash<NativeHandle>{}(k.samplers[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                return h;
            }
        };

        std::unordered_map<SamplerTableKey, UINT32, SamplerTableKeyHash> samplerTableCache;

        UINT32 GetOrCreateSamplerTable(
            ID3D12Device* device,
            const std::vector<NativeSamplerBinding>& samplers,
            UINT32 targetSpace = UINT32_MAX,
            ShaderStage targetStage = ShaderStage::NONE)
        {
            if (samplers.empty() || defaultDummySamplerTableIndex == UINT32_MAX)
            {
                return defaultDummySamplerTableIndex;
            }

            // Fast pre-check: avoid initializing 128-byte SamplerTableKey if no sampler targets this (stage, space)
            BOOL hasAnyInSpace = false;
            for (SIZE_T sm = 0; sm < samplers.size(); ++sm)
            {
                const auto& samp = samplers[sm];
                if ((targetStage == ShaderStage::NONE || samp.stage == targetStage) &&
                    (targetSpace == UINT32_MAX || samp.registerSpace == targetSpace) &&
                    samp.slotIndex < 16 && samp.sampler != 0)
                {
                    hasAnyInSpace = true;
                    break;
                }
            }

            if (!hasAnyInSpace)
            {
                return defaultDummySamplerTableIndex;
            }

            SamplerTableKey key{};
            for (UINT32 s = 0; s < 16; ++s)
            {
                key.samplers[s] = defaultSamplerHandle;
            }

            for (SIZE_T sm = 0; sm < samplers.size(); ++sm)
            {
                const auto& samp = samplers[sm];
                if ((targetStage == ShaderStage::NONE || samp.stage == targetStage) &&
                    (targetSpace == UINT32_MAX || samp.registerSpace == targetSpace) &&
                    samp.slotIndex < 16 && samp.sampler != 0)
                {
                    key.samplers[samp.slotIndex] = samp.sampler;
                }
            }

            auto it = samplerTableCache.find(key);
            if (it != samplerTableCache.end())
            {
                return it->second;
            }

            UINT32 tableIndex = samplerPool.AllocateRange(16);
            if (tableIndex == UINT32_MAX)
            {
                ProcessRetiredSamplerTables(GetSafeCompletedFenceValue());
                tableIndex = samplerPool.AllocateRange(16);
            }
            if (tableIndex == UINT32_MAX && !samplerTableCache.empty())
            {
                const UINT64 retireFence = GetSafeRetireFenceValue();
                for (const auto& kv : samplerTableCache)
                {
                    if (kv.second != UINT32_MAX && kv.second != defaultDummySamplerTableIndex)
                    {
                        retiredSamplerTables.push_back({ retireFence, kv.second });
                    }
                }
                samplerTableCache.clear();
                deviceContext.WaitForGPU();
                ProcessRetiredSamplerTables(GetSafeCompletedFenceValue());
                tableIndex = samplerPool.AllocateRange(16);
            }
            if (tableIndex == UINT32_MAX)
            {
                return UINT32_MAX;
            }

            for (UINT32 s = 0; s < 16; ++s)
            {
                const auto* sRes = resourceMgr.GetSampler(key.samplers[s]);
                if (sRes && sRes->samplerDescriptorIndex != UINT32_MAX)
                {
                    device->CopyDescriptorsSimple(
                        1,
                        samplerPool.GetCPUHandle(tableIndex + s),
                        samplerPool.GetCpuStagingHandle(sRes->samplerDescriptorIndex),
                        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
                    );
                }
            }

            samplerTableCache[key] = tableIndex;
            return tableIndex;
        }

        BOOL RecordComputeDispatch(
            const ComputeDispatchCommand& cmd,
            UINT32 bbIndex,
            UINT32& currentTableOffset,
            UINT32& tablesUsedThisFrame,
            BOOL& isComputeRootSigActive,
            ID3D12PipelineState*& currentComputePSO,
            UINT32 currentComputeRootTable[kMaxSpaces],
            UINT32 currentComputeSamplerTable[kMaxSpaces])
        {
            if (cmd.computeShader == 0)
            {
                if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                return false;
            }

            D3D12GPUBufferResource* indirectArgBuf = nullptr;
            if (cmd.indirectArgsBuffer != 0)
            {
                indirectArgBuf = resourceMgr.GetGPUBuffer(cmd.indirectArgsBuffer);
                if (!indirectArgBuf || !indirectArgBuf->resource ||
                    !HasGPUBufferUsage(indirectArgBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                    (cmd.indirectArgsOffset % 4u) != 0u ||
                    static_cast<UINT64>(cmd.indirectArgsOffset) + sizeof(GPUIndirectDispatchArgs) > static_cast<UINT64>(indirectArgBuf->byteSize) ||
                    pipeline.GetDispatchIndirectSignature() == nullptr)
                {
                    lastFrameMetrics.indirectCommandsRejected++;
                    return false;
                }
            }
            else
            {
                if (cmd.threadGroupCountX == 0 || cmd.threadGroupCountY == 0 || cmd.threadGroupCountZ == 0 ||
                    cmd.threadGroupCountX > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
                    cmd.threadGroupCountY > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
                    cmd.threadGroupCountZ > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
                {
                    return false;
                }
            }

            ID3D12Device* device = deviceContext.GetDevice();
            if (!device || !frameCmdList)
            {
                if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                return false;
            }

            ID3D12PipelineState* csPso = pipeline.GetOrCreateComputePSO(device, resourceMgr, cmd.computeShader);
            if (!csPso)
            {
                if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                return false;
            }

            // 1. Strict validation Validation of all bound SRV/UAV buffers and textures
            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
            {
                const auto& sb = cmd.structuredBuffers[s];
                if (sb.stage != ShaderStage::COMPUTE || sb.registerSpace >= kMaxSpaces)
                {
                    if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                    return false;
                }

                const BOOL isUav = (sb.isUAV != FALSE);
                if (isUav)
                {
                    if (sb.slotIndex >= kUAVPerSpace || sb.buffer == 0 || sb.buffer == cmd.indirectArgsBuffer)
                    {
                        if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }
                    const D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                    if (!gpuBuf || !gpuBuf->resource || gpuBuf->uavDescriptorIndex == UINT32_MAX)
                    {
                        if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }
                }
                else
                {
                    if (sb.slotIndex >= kSRVPerSpace)
                    {
                        if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }
                    if (sb.buffer != 0)
                    {
                        const D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                        if (!gpuBuf || !gpuBuf->resource || gpuBuf->srvDescriptorIndex == UINT32_MAX)
                        {
                            if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                            return false;
                        }
                    }
                }

                // Reject simultaneous SRV + UAV binding of the same GPUBuffer in a single dispatch
                if (sb.buffer != 0)
                {
                    for (SIZE_T s2 = s + 1; s2 < cmd.structuredBuffers.size(); ++s2)
                    {
                        const auto& sb2 = cmd.structuredBuffers[s2];
                        if (sb2.buffer != sb.buffer) continue;
                        const BOOL isUav2 = (sb2.isUAV != FALSE);
                        if (isUav != isUav2)
                        {
                            if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                            return false;
                        }
                    }
                }
            }

            for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
            {
                const auto& rwt = cmd.rwTextures[r];
                if (rwt.stage != ShaderStage::COMPUTE || rwt.registerSpace >= kMaxSpaces || rwt.slotIndex >= kUAVPerSpace || rwt.texture == 0)
                {
                    return false;
                }
                const D3D12TextureResource* texRes = resourceMgr.GetTexture(rwt.texture);
                if (!texRes || !texRes->resource || texRes->uavDescriptorIndex == UINT32_MAX)
                {
                    return false;
                }
                for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                {
                    if (cmd.textures[t].texture == rwt.texture)
                    {
                        // Simultaneous SRV + UAV on the SAME subresource (or full-chain SRV) is invalid;
                        // disjoint mip slices (e.g. HZB Mip[k-1] SRV -> Mip[k] UAV) are valid.
                        if (cmd.textures[t].mipSlice == 0xFFFFFFFFu || cmd.textures[t].mipSlice == rwt.mipSlice)
                        {
                            return false;
                        }
                    }
                }
            }

            for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
            {
                const auto& tex = cmd.textures[t];
                if (tex.stage != ShaderStage::COMPUTE || tex.registerSpace >= kMaxSpaces || tex.slotIndex >= kSRVPerSpace)
                {
                    return false;
                }
                if (tex.texture != 0)
                {
                    const D3D12TextureResource* texRes = resourceMgr.GetTexture(tex.texture);
                    if (!texRes || !texRes->resource || texRes->srvDescriptorIndex == UINT32_MAX)
                    {
                        return false;
                    }
                }
            }

            for (SIZE_T sm = 0; sm < cmd.samplers.size(); ++sm)
            {
                const auto& samp = cmd.samplers[sm];
                if (samp.stage != ShaderStage::COMPUTE || samp.registerSpace >= kMaxSpaces || samp.slotIndex >= 16)
                {
                    return false;
                }
                if (samp.sampler != 0)
                {
                    const D3D12SamplerResource* sRes = resourceMgr.GetSampler(samp.sampler);
                    if (!sRes || sRes->samplerDescriptorIndex == UINT32_MAX)
                    {
                        return false;
                    }
                }
            }

            // 2. Dependency-driven Resource State & UAV Barrier Transitions (batched before Dispatch)
            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
            {
                const auto& sb = cmd.structuredBuffers[s];
                if (sb.buffer == 0) continue;

                D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                const BOOL isUav = (sb.isUAV != FALSE);
                const GPUResourceAccess uavMode = (sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE ||
                                                   sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_WRITE ||
                                                   sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ)
                    ? sb.uavAccess
                    : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
                const GPUResourceAccess targetAccess = isUav
                    ? uavMode
                    : GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE;

                stateTracker.TransitionResource(gpuBuf->resource.Get(), targetAccess);
                gpuBuf->currentState = stateTracker.GetResourceState(gpuBuf->resource.Get());
            }

            for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
            {
                const auto& rwt = cmd.rwTextures[r];
                D3D12TextureResource* texRes = resourceMgr.GetTexture(rwt.texture);
                if (!stateTracker.IsResourceTracked(texRes->resource.Get()))
                {
                    RegisterTrackedTexture(stateTracker, texRes);
                }

                const GPUResourceAccess targetAccess = (rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE ||
                                                        rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_WRITE ||
                                                        rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ)
                    ? rwt.uavAccess
                    : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;

                if (texRes->desc.mipLevels > 1u && rwt.mipSlice < texRes->mipUavDescriptors.size())
                {
                    stateTracker.TransitionMipLevel(texRes->resource.Get(), rwt.mipSlice, targetAccess);
                }
                else
                {
                    stateTracker.TransitionResource(texRes->resource.Get(), targetAccess);
                }
                texRes->currentState = stateTracker.GetResourceState(texRes->resource.Get());
            }

            for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
            {
                const auto& tex = cmd.textures[t];
                if (tex.texture == 0) continue;

                D3D12TextureResource* texRes = resourceMgr.GetTexture(tex.texture);
                if (!stateTracker.IsResourceTracked(texRes->resource.Get()))
                {
                    RegisterTrackedTexture(stateTracker, texRes);
                }

                if (stateTracker.IsResourceTracked(texRes->resource.Get()))
                {
                    if (texRes->desc.mipLevels > 1u && tex.mipSlice != 0xFFFFFFFFu && tex.mipSlice < texRes->desc.mipLevels)
                    {
                        const D3D12_RESOURCE_STATES srState = stateTracker.GetSubresourceState(texRes->resource.Get(), tex.mipSlice);
                        if ((srState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                        {
                            stateTracker.TransitionMipLevel(texRes->resource.Get(), tex.mipSlice, GPUResourceAccess::ALL_SHADER_RESOURCE);
                        }
                    }
                    else
                    {
                        const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(texRes->resource.Get());
                        if (!stateTracker.IsResourceUniformState(texRes->resource.Get()) ||
                            (curState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                        {
                            stateTracker.TransitionResource(texRes->resource.Get(), GPUResourceAccess::ALL_SHADER_RESOURCE);
                        }
                    }
                    texRes->currentState = stateTracker.GetResourceState(texRes->resource.Get());
                }
            }

            if (indirectArgBuf != nullptr)
            {
                stateTracker.TransitionResource(indirectArgBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                indirectArgBuf->currentState = stateTracker.GetResourceState(indirectArgBuf->resource.Get());
            }

            stateTracker.FlushBarriers(frameCmdList.Get());

            // 3. Bind Compute Root Signature & Safe Fallback Tables if not already active
            if (!isComputeRootSigActive)
            {
                frameCmdList->SetComputeRootSignature(pipeline.GetComputeRootSignature());
                if (defaultDummySpaceTableIndex != UINT32_MAX)
                {
                    D3D12_GPU_DESCRIPTOR_HANDLE dummySpaceGPU = cbvSrvUavPool.GetGPUHandle(defaultDummySpaceTableIndex);
                    for (UINT32 s = 0; s < kMaxSpaces; ++s)
                    {
                        frameCmdList->SetComputeRootDescriptorTable(s, dummySpaceGPU);
                        currentComputeRootTable[s] = defaultDummySpaceTableIndex;
                    }
                }
                if (defaultDummySamplerTableIndex != UINT32_MAX)
                {
                    D3D12_GPU_DESCRIPTOR_HANDLE dummySampGPU = samplerPool.GetGPUHandle(defaultDummySamplerTableIndex);
                    for (UINT32 s = 0; s < kMaxSpaces; ++s)
                    {
                        frameCmdList->SetComputeRootDescriptorTable(kComputeSamplerParamBase + s, dummySampGPU);
                        currentComputeSamplerTable[s] = defaultDummySamplerTableIndex;
                    }
                }
                isComputeRootSigActive = true;
            }

            if (csPso != currentComputePSO)
            {
                frameCmdList->SetPipelineState(csPso);
                currentComputePSO = csPso;
            }

            // 4. Populate 64-Descriptor Space Tables (CBV b0..b15, SRV t0..t31, UAV u0..u15) for active spaces 0..3
            for (UINT32 space = 0; space < kMaxSpaces; ++space)
            {
                BOOL isActive = false;
                for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                {
                    if (cmd.constantBuffers[b].stage == ShaderStage::COMPUTE && cmd.constantBuffers[b].registerSpace == space)
                    {
                        isActive = true;
                        break;
                    }
                }
                if (!isActive)
                {
                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                    {
                        if (cmd.structuredBuffers[s].stage == ShaderStage::COMPUTE && cmd.structuredBuffers[s].registerSpace == space)
                        {
                            isActive = true;
                            break;
                        }
                    }
                }
                if (!isActive)
                {
                    for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                    {
                        if (cmd.textures[t].stage == ShaderStage::COMPUTE && cmd.textures[t].registerSpace == space)
                        {
                            isActive = true;
                            break;
                        }
                    }
                }
                if (!isActive)
                {
                    for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
                    {
                        if (cmd.rwTextures[r].stage == ShaderStage::COMPUTE && cmd.rwTextures[r].registerSpace == space)
                        {
                            isActive = true;
                            break;
                        }
                    }
                }

                if (!isActive) continue;

                if (tablesUsedThisFrame >= kTablesPerBackBuffer ||
                    currentTableOffset + kDescriptorsPerSpaceTable > kRingDescriptorsPerBB)
                {
                    if (indirectArgBuf) lastFrameMetrics.indirectCommandsRejected++;
                    return false;
                }
                const UINT32 tableStart = dynamicTableBase[bbIndex] + currentTableOffset;
                currentTableOffset += kDescriptorsPerSpaceTable;
                tablesUsedThisFrame++;

                // Copy all 64 fallback descriptors (16 CBV + 32 SRV + 16 UAV)
                if (defaultDummySpaceTableIndex != UINT32_MAX)
                {
                    device->CopyDescriptorsSimple(
                        kDescriptorsPerSpaceTable,
                        cbvSrvUavPool.GetCPUHandle(tableStart),
                        cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex),
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                    );
                }

                // Populate Constant Buffers (b0..b15 -> offset 0..15)
                for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                {
                    const auto& cb = cmd.constantBuffers[b];
                    if (cb.stage == ShaderStage::COMPUTE && cb.registerSpace == space && cb.slotIndex < kCBVPerSpace)
                    {
                        const BYTE* data = cb.GetData();
                        if (!data || cb.byteSize == 0) continue;

                        D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateCBuffer(data, cb.byteSize);
                        if (va == 0) continue;

                        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                        cbvDesc.BufferLocation = va;
                        cbvDesc.SizeInBytes = (cb.byteSize + 255) & ~255;
                        device->CreateConstantBufferView(
                            &cbvDesc,
                            cbvSrvUavPool.GetCPUHandle(tableStart + cb.slotIndex)
                        );
                    }
                }

                // Populate SRV & UAV StructuredBuffers
                for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                {
                    const auto& sb = cmd.structuredBuffers[s];
                    if (sb.stage != ShaderStage::COMPUTE || sb.registerSpace != space) continue;

                    const BOOL isUav = (sb.isUAV != FALSE);
                    if (isUav)
                    {
                        D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                        D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->uavDescriptorIndex);
                        device->CopyDescriptorsSimple(
                            1,
                            cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + kSRVPerSpace + sb.slotIndex),
                            src,
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                        );
                    }
                    else
                    {
                        if (sb.buffer != 0)
                        {
                            D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                            const UINT32 reqStride = (sb.stride > 0u) ? sb.stride : (gpuBuf->desc.stride > 0u ? gpuBuf->desc.stride : 4u);
                            const BOOL   explicitZeroSlice = (sb.elementCount == 0u && sb.stride > 0u);
                            const UINT32 reqCount  = (sb.elementCount > 0u) ? sb.elementCount : gpuBuf->desc.elementCount;
                            if (explicitZeroSlice)
                            {
                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                nullSrvDesc.Buffer.FirstElement        = 0;
                                nullSrvDesc.Buffer.NumElements         = 0;
                                nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                device->CreateShaderResourceView(
                                    nullptr,
                                    &nullSrvDesc,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                );
                            }
                            else if (sb.firstElement > 0u ||
                                     reqStride != gpuBuf->desc.stride ||
                                     reqCount != gpuBuf->desc.elementCount)
                            {
                                const UINT32 maxElems = (reqStride > 0u) ? (gpuBuf->byteSize / reqStride) : 0u;
                                if (sb.firstElement < maxElems)
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC viewSrv{};
                                    viewSrv.Format                     = DXGI_FORMAT_UNKNOWN;
                                    viewSrv.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                    viewSrv.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    viewSrv.Buffer.FirstElement        = sb.firstElement;
                                    viewSrv.Buffer.NumElements         = (std::min)(reqCount, maxElems - sb.firstElement);
                                    viewSrv.Buffer.StructureByteStride = reqStride;
                                    viewSrv.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                    device->CreateShaderResourceView(
                                        gpuBuf->resource.Get(),
                                        &viewSrv,
                                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                    );
                                }
                                else
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                    nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                    nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                    nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    nullSrvDesc.Buffer.FirstElement        = 0;
                                    nullSrvDesc.Buffer.NumElements         = 0;
                                    nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                    nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                    device->CreateShaderResourceView(
                                        nullptr,
                                        &nullSrvDesc,
                                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                    );
                                }
                            }
                            else
                            {
                                D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->srvDescriptorIndex);
                                device->CopyDescriptorsSimple(
                                    1,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex),
                                    src,
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }
                        }
                        else
                        {
                            const UINT32 stride = sb.stride > 0 ? sb.stride : 4;
                            if (sb.elementCount == 0 || sb.byteSize == 0 || !sb.GetData())
                            {
                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                nullSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                nullSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                nullSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                nullSrvDesc.Buffer.FirstElement = 0;
                                nullSrvDesc.Buffer.NumElements = 0;
                                nullSrvDesc.Buffer.StructureByteStride = stride;
                                nullSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                                device->CreateShaderResourceView(
                                    nullptr,
                                    &nullSrvDesc,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                );
                            }
                            else
                            {
                                CachedSharedSBUpload sbUpload{};
                                if (AllocateOrReuseStructuredBuffer(sb, stride, sbUpload))
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
                                    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    srvDesc.Buffer.FirstElement = sbUpload.firstElement;
                                    srvDesc.Buffer.NumElements = sbUpload.numElements;
                                    srvDesc.Buffer.StructureByteStride = sbUpload.stride;
                                    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                                    device->CreateShaderResourceView(
                                        sbUpload.resource,
                                        &srvDesc,
                                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                    );
                                }
                            }
                        }
                    }
                }

                // Populate SRV Textures (t0..t31 -> offset 16..47)
                for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                {
                    const auto& tex = cmd.textures[t];
                    if (tex.stage == ShaderStage::COMPUTE && tex.registerSpace == space && tex.slotIndex < kSRVPerSpace)
                    {
                        if (tex.texture != 0)
                        {
                            const auto* texRes = resourceMgr.GetTexture(tex.texture);
                            if (texRes && texRes->srvDescriptorIndex != UINT32_MAX)
                            {
                                UINT32 descIdx = texRes->srvDescriptorIndex;
                                if (tex.mipSlice != 0xFFFFFFFFu &&
                                    tex.mipSlice < texRes->mipSrvDescriptors.size() &&
                                    texRes->mipSrvDescriptors[tex.mipSlice] != UINT32_MAX)
                                {
                                    descIdx = texRes->mipSrvDescriptors[tex.mipSlice];
                                }
                                D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(descIdx);
                                device->CopyDescriptorsSimple(
                                    1,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + tex.slotIndex),
                                    src,
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }
                        }
                        else
                        {
                            WriteNullTextureSRV(
                                device,
                                cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + tex.slotIndex),
                                tex.nullViewType,
                                tex.nullFormat);
                        }
                    }
                }

                // Populate UAV RWTextures (u0..u15 -> offset 48..63)
                for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
                {
                    const auto& rwt = cmd.rwTextures[r];
                    if (rwt.stage == ShaderStage::COMPUTE && rwt.registerSpace == space && rwt.slotIndex < kUAVPerSpace && rwt.texture != 0)
                    {
                        const auto* texRes = resourceMgr.GetTexture(rwt.texture);
                        if (texRes && texRes->uavDescriptorIndex != UINT32_MAX)
                        {
                            UINT32 descIdx = texRes->uavDescriptorIndex;
                            if (rwt.mipSlice < texRes->mipUavDescriptors.size() &&
                                texRes->mipUavDescriptors[rwt.mipSlice] != UINT32_MAX)
                            {
                                descIdx = texRes->mipUavDescriptors[rwt.mipSlice];
                            }
                            D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(descIdx);
                            device->CopyDescriptorsSimple(
                                1,
                                cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + kSRVPerSpace + rwt.slotIndex),
                                src,
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                            );
                        }
                    }
                }

                if (currentComputeRootTable[space] != tableStart)
                {
                    frameCmdList->SetComputeRootDescriptorTable(space, cbvSrvUavPool.GetGPUHandle(tableStart));
                    currentComputeRootTable[space] = tableStart;
                }
            }

            // 5. Populate Multi-Space Samplers (Params 4..7 -> s0..s15, space0..space3)
            for (UINT32 space = 0; space < kMaxSpaces; ++space)
            {
                const UINT32 targetSamplerTable = GetOrCreateSamplerTable(device, cmd.samplers, space, ShaderStage::COMPUTE);
                if (targetSamplerTable != UINT32_MAX && targetSamplerTable != currentComputeSamplerTable[space])
                {
                    frameCmdList->SetComputeRootDescriptorTable(
                        kComputeSamplerParamBase + space,
                        samplerPool.GetGPUHandle(targetSamplerTable)
                    );
                    currentComputeSamplerTable[space] = targetSamplerTable;
                }
            }

            // 6. Bind Compute Root Constants (Param 8 -> b0, space4, 4 DWORDs)
            if (cmd.useRootConstants)
            {
                frameCmdList->SetComputeRoot32BitConstants(kComputeRootConstantsParam, 4, cmd.rootConstants, 0);
            }

            // 7. Dispatch Thread Groups (Direct or GPU-Driven DispatchIndirect)
            if (indirectArgBuf != nullptr)
            {
                frameCmdList->ExecuteIndirect(
                    pipeline.GetDispatchIndirectSignature(),
                    1,
                    indirectArgBuf->resource.Get(),
                    cmd.indirectArgsOffset,
                    nullptr,
                    0
                );
                lastFrameMetrics.indirectDispatchCommandsExecuted++;
            }
            else
            {
                frameCmdList->Dispatch(cmd.threadGroupCountX, cmd.threadGroupCountY, cmd.threadGroupCountZ);
            }
            return true;
        }

        BOOL forceLayeredDepthFallback = false;
        BOOL loggedLayeredDepthFallbackOnce = false;

        void BindGraphicsRootSigAndFallbacks(ID3D12RootSignature* targetRootSig = nullptr)
        {
            if (!frameCmdList) return;
            ID3D12RootSignature* rootSig = targetRootSig ? targetRootSig : pipeline.GetRootSignature();
            if (!rootSig) return;
            frameCmdList->SetGraphicsRootSignature(rootSig);
            if (defaultDummySpaceTableIndex != UINT32_MAX)
            {
                D3D12_GPU_DESCRIPTOR_HANDLE dummySpaceGPU = cbvSrvUavPool.GetGPUHandle(defaultDummySpaceTableIndex);
                D3D12_GPU_DESCRIPTOR_HANDLE dummyUavGPU   = cbvSrvUavPool.GetGPUHandle(defaultDummySpaceTableIndex + kCBVPerSpace + kSRVPerSpace);
                for (UINT32 p = 0; p < 8; ++p)
                {
                    frameCmdList->SetGraphicsRootDescriptorTable(p, dummySpaceGPU);
                }
                for (UINT32 s = 0; s < kMaxSpaces; ++s)
                {
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsGeometrySpaceParamBase + s,
                        dummySpaceGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsHullSpaceParamBase + s,
                        dummySpaceGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsDomainSpaceParamBase + s,
                        dummySpaceGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsUAVSpaceParamBase + s,
                        dummyUavGPU
                    );
                }
            }
            if (defaultDummySamplerTableIndex != UINT32_MAX)
            {
                D3D12_GPU_DESCRIPTOR_HANDLE dummySampGPU = samplerPool.GetGPUHandle(defaultDummySamplerTableIndex);
                for (UINT32 s = 0; s < kMaxSpaces; ++s)
                {
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsVertexSamplerParamBase + s,
                        dummySampGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsPixelSamplerParamBase + s,
                        dummySampGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsGeometrySamplerParamBase + s,
                        dummySampGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsHullSamplerParamBase + s,
                        dummySampGPU
                    );
                    frameCmdList->SetGraphicsRootDescriptorTable(
                        D3D12Pipeline::kGraphicsDomainSamplerParamBase + s,
                        dummySampGPU
                    );
                }
            }
            const UINT32 zeroRootConsts[4] = { 0, 0, 0, 0 };
            frameCmdList->SetGraphicsRoot32BitConstants(
                D3D12Pipeline::kGraphicsRootConstantsParamIndex,
                4,
                zeroRootConsts,
                0
            );
            if (rootSig != pipeline.GetRootSignature() && defaultDummyZeroCBuffer)
            {
                const D3D12_GPU_VIRTUAL_ADDRESS zeroVA = defaultDummyZeroCBuffer->GetGPUVirtualAddress();
                for (UINT32 i = 0; i < D3D12Pipeline::kGraphicsIndirectCBVParamCount; ++i)
                {
                    frameCmdList->SetGraphicsRootConstantBufferView(
                        D3D12Pipeline::kGraphicsIndirectCBVParamBase + i,
                        zeroVA
                    );
                }
                frameCmdList->SetGraphicsRootShaderResourceView(
                    D3D12Pipeline::kGraphicsIndirectSRVParamIndex,
                    zeroVA
                );
            }
        }

        BOOL RecordLayeredDepthPass(
            const LayeredDepthPassView& pass,
            const Mat4& defaultViewProjMatrix,
            UINT32 bbIndex,
            UINT32& currentTableOffset,
            UINT32& tablesUsedThisFrame,
            BOOL& isComputeRootSigActive,
            ID3D12PipelineState*& currentPSO,
            ID3D12PipelineState*& currentComputePSO,
            NativeHandle& currentVB,
            NativeHandle& currentIB,
            UINT32 currentRootTable[D3D12Pipeline::kGraphicsRootParamCount],
            UINT32 currentSamplerTableIndex[5][kMaxSpaces])
        {
            ID3D12Device* device = deviceContext.GetDevice();
            if (!device || !frameCmdList || pass.depthArrayTexture == 0)
            {
                return false;
            }

            D3D12TextureResource* depthTex = resourceMgr.GetTexture(pass.depthArrayTexture);
            if (!depthTex || !depthTex->resource ||
                depthTex->desc.type != TextureType::TEXTURE_2D_ARRAY ||
                depthTex->desc.usage != TextureUsage::DEPTH_STENCIL ||
                depthTex->desc.format != TextureFormat::D32_FLOAT ||
                depthTex->fullArrayDsvDescriptorIndex == UINT32_MAX)
            {
                return false;
            }

            const UINT32 effectiveLayers = (pass.arrayLayerCount > 0) ? pass.arrayLayerCount : depthTex->desc.arrayLayers;
            if (effectiveLayers == 0 || effectiveLayers > depthTex->desc.arrayLayers)
            {
                return false;
            }

            const FLOAT32 vpWidth  = (pass.viewportWidth > 0.0f)  ? pass.viewportWidth  : static_cast<FLOAT32>(depthTex->desc.width);
            const FLOAT32 vpHeight = (pass.viewportHeight > 0.0f) ? pass.viewportHeight : static_cast<FLOAT32>(depthTex->desc.height);
            if (pass.viewportX < 0.0f || pass.viewportY < 0.0f ||
                vpWidth <= 0.0f || vpHeight <= 0.0f ||
                (pass.viewportX + vpWidth) > static_cast<FLOAT32>(depthTex->desc.width) ||
                (pass.viewportY + vpHeight) > static_cast<FLOAT32>(depthTex->desc.height))
            {
                return false;
            }

            const INT32 scLeft   = pass.scissorLeft;
            const INT32 scTop    = pass.scissorTop;
            const INT32 scRight  = (pass.scissorRight > 0)  ? pass.scissorRight  : static_cast<INT32>(depthTex->desc.width);
            const INT32 scBottom = (pass.scissorBottom > 0) ? pass.scissorBottom : static_cast<INT32>(depthTex->desc.height);
            if (scLeft < 0 || scTop < 0 || scRight <= scLeft || scBottom <= scTop ||
                scRight > static_cast<INT32>(depthTex->desc.width) ||
                scBottom > static_cast<INT32>(depthTex->desc.height))
            {
                return false;
            }

            const BOOL useGsFallback = (!deviceContext.SupportsVPAndRTArrayIndexWithoutGS() || forceLayeredDepthFallback);
            if (useGsFallback && !loggedLayeredDepthFallbackOnce)
            {
                std::cout << "[NorthstarD3D12] Layered depth rendering using Geometry Shader pass-through fallback "
                          << "(VPAndRTArrayIndexWithoutGS=" << (deviceContext.SupportsVPAndRTArrayIndexWithoutGS() ? 1 : 0)
                          << ", forced=" << (forceLayeredDepthFallback ? 1 : 0) << ").\n";
                loggedLayeredDepthFallbackOnce = true;
            }

            if (!stateTracker.IsResourceTracked(depthTex->resource.Get()))
            {
                RegisterTrackedTexture(stateTracker, depthTex);
            }

            D3D12_VIEWPORT passVp{};
            passVp.TopLeftX = pass.viewportX;
            passVp.TopLeftY = pass.viewportY;
            passVp.Width    = vpWidth;
            passVp.Height   = vpHeight;
            passVp.MinDepth = 0.0f;
            passVp.MaxDepth = 1.0f;

            D3D12_RECT passScissor{};
            passScissor.left   = scLeft;
            passScissor.top    = scTop;
            passScissor.right  = scRight;
            passScissor.bottom = scBottom;

            const D3D12_CPU_DESCRIPTOR_HANDLE fullDsv = dsvPool.GetCPUHandle(depthTex->fullArrayDsvDescriptorIndex);

            // 1. Optional Selective Layered Depth Clear
            if (pass.selectiveClear.enabled)
            {
                if (pass.selectiveClear.useCpuPerSliceClearFallback)
                {
                    // Explicitly separated CPU Per-Slice DSV Clear Fallback / Baseline comparison path
                    stateTracker.TransitionResource(depthTex->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                    stateTracker.FlushBarriers(frameCmdList.Get());
                    depthTex->currentState = stateTracker.GetResourceState(depthTex->resource.Get());

                    std::vector<bool> clearedSlice(effectiveLayers, false);
                    for (SIZE_T i = 0; i < pass.selectiveClear.cpuDirtySlices.size(); ++i)
                    {
                        const UINT32 sliceIdx = pass.selectiveClear.cpuDirtySlices[i];
                        if (sliceIdx >= effectiveLayers || clearedSlice[sliceIdx]) continue;
                        if (sliceIdx >= depthTex->dsvLayerDescriptors.size()) continue;
                        const UINT32 layerDsvIdx = depthTex->dsvLayerDescriptors[sliceIdx];
                        if (layerDsvIdx == UINT32_MAX) continue;

                        clearedSlice[sliceIdx] = true;
                        const D3D12_CPU_DESCRIPTOR_HANDLE layerDsv = dsvPool.GetCPUHandle(layerDsvIdx);
                        frameCmdList->ClearDepthStencilView(
                            layerDsv,
                            D3D12_CLEAR_FLAG_DEPTH,
                            pass.selectiveClear.clearDepth,
                            0,
                            1,
                            &passScissor
                        );
                        lastFrameMetrics.cpuPerSliceDepthClearsExecuted++;
                    }
                }
                else
                {
                    // Primary GPU-Driven Selective Layered Depth Clear Path (ExecuteIndirect + Fullscreen Triangle per Dirty Slice)
                    const UINT32 maxCmdCount = pass.selectiveClear.maxIndirectCommandCount;
                    if (pass.selectiveClear.clearVertexShader == 0 ||
                        pass.selectiveClear.dirtySliceIndicesBuffer == 0 ||
                        pass.selectiveClear.indirectDrawArgsBuffer == 0 ||
                        pass.selectiveClear.dirtySliceIndicesBuffer == pass.selectiveClear.indirectDrawArgsBuffer ||
                        maxCmdCount == 0 ||
                        pipeline.GetDrawIndirectSignature() == nullptr)
                    {
                        lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }

                    if (useGsFallback && pass.selectiveClear.clearGeometryShader == 0)
                    {
                        lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }

                    D3D12GPUBufferResource* dirtyListBuf = resourceMgr.GetGPUBuffer(pass.selectiveClear.dirtySliceIndicesBuffer);
                    D3D12GPUBufferResource* indirectArgsBuf = resourceMgr.GetGPUBuffer(pass.selectiveClear.indirectDrawArgsBuffer);
                    if (!dirtyListBuf || !dirtyListBuf->resource ||
                        !HasGPUBufferUsage(dirtyListBuf->desc.usage, GPUBufferUsage::STRUCTURED_SRV) ||
                        dirtyListBuf->srvDescriptorIndex == UINT32_MAX ||
                        !indirectArgsBuf || !indirectArgsBuf->resource ||
                        !HasGPUBufferUsage(indirectArgsBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                        (pass.selectiveClear.indirectDrawArgsOffset % 4u) != 0u ||
                        static_cast<UINT64>(pass.selectiveClear.indirectDrawArgsOffset) + static_cast<UINT64>(maxCmdCount) * sizeof(GPUIndirectDrawArgs) > static_cast<UINT64>(indirectArgsBuf->byteSize))
                    {
                        lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }

                    D3D12GPUBufferResource* indirectCountBuf = nullptr;
                    if (pass.selectiveClear.indirectCountBuffer != 0)
                    {
                        if (pass.selectiveClear.indirectCountBuffer == pass.selectiveClear.dirtySliceIndicesBuffer)
                        {
                            lastFrameMetrics.indirectCommandsRejected++;
                            return false;
                        }
                        indirectCountBuf = resourceMgr.GetGPUBuffer(pass.selectiveClear.indirectCountBuffer);
                        if (!indirectCountBuf || !indirectCountBuf->resource ||
                            !HasGPUBufferUsage(indirectCountBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                            (pass.selectiveClear.indirectCountBufferOffset % 4u) != 0u ||
                            static_cast<UINT64>(pass.selectiveClear.indirectCountBufferOffset) + sizeof(UINT32) > static_cast<UINT64>(indirectCountBuf->byteSize))
                        {
                            lastFrameMetrics.indirectCommandsRejected++;
                            return false;
                        }

                        if (pass.selectiveClear.indirectCountBuffer == pass.selectiveClear.indirectDrawArgsBuffer)
                        {
                            const UINT64 argBegin   = static_cast<UINT64>(pass.selectiveClear.indirectDrawArgsOffset);
                            const UINT64 argEnd     = argBegin + static_cast<UINT64>(maxCmdCount) * sizeof(GPUIndirectDrawArgs);
                            const UINT64 countBegin = static_cast<UINT64>(pass.selectiveClear.indirectCountBufferOffset);
                            const UINT64 countEnd   = countBegin + sizeof(UINT32);
                            if (countBegin < argEnd && argBegin < countEnd)
                            {
                                lastFrameMetrics.indirectCommandsRejected++;
                                return false;
                            }
                        }
                    }

                    DepthState clearDepthState{};
                    clearDepthState.test    = true;
                    clearDepthState.write   = true;
                    clearDepthState.compare = DepthCompare::ALWAYS;

                    const NativeHandle clearGsHandle = useGsFallback ? pass.selectiveClear.clearGeometryShader : 0;
                    ID3D12PipelineState* clearPso = pipeline.GetOrCreatePSO(
                        device,
                        resourceMgr,
                        pass.selectiveClear.clearVertexShader,
                        0, // Depth-only pass: NumRenderTargets = 0, RTVFormats[0] = UNKNOWN
                        CullMode::NONE,
                        BlendMode::OPAQUE,
                        FillMode::SOLID,
                        clearDepthState,
                        false,
                        0,
                        0.0f,
                        0.0f,
                        clearGsHandle,
                        true,
                        0,
                        0,
                        IndirectRootBindingKey{},
                        DXGI_FORMAT_UNKNOWN,
                        depthTex->dsvFormat
                    );
                    if (!clearPso)
                    {
                        lastFrameMetrics.indirectCommandsRejected++;
                        return false;
                    }

                    if (tablesUsedThisFrame >= kTablesPerBackBuffer ||
                        currentTableOffset + kGraphicsDescPerSpace > kRingDescriptorsPerBB)
                    {
                        return false;
                    }

                    // Batch resource transitions: DirtyList -> NON_PIXEL_SHADER_RESOURCE, IndirectArgs/Count -> INDIRECT_ARGUMENT, DepthArray -> DEPTH_WRITE
                    stateTracker.TransitionResource(dirtyListBuf->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                    dirtyListBuf->currentState = stateTracker.GetResourceState(dirtyListBuf->resource.Get());

                    stateTracker.TransitionResource(indirectArgsBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                    indirectArgsBuf->currentState = stateTracker.GetResourceState(indirectArgsBuf->resource.Get());

                    if (indirectCountBuf != nullptr)
                    {
                        stateTracker.TransitionResource(indirectCountBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                        indirectCountBuf->currentState = stateTracker.GetResourceState(indirectCountBuf->resource.Get());
                    }

                    stateTracker.TransitionResource(depthTex->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                    stateTracker.FlushBarriers(frameCmdList.Get());
                    depthTex->currentState = stateTracker.GetResourceState(depthTex->resource.Get());

                    if (isComputeRootSigActive)
                    {
                        BindGraphicsRootSigAndFallbacks();
                        for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                        ResetGraphicsSamplerCache(currentSamplerTableIndex);
                        isComputeRootSigActive = false;
                        currentPSO = nullptr;
                    }

                    frameCmdList->RSSetViewports(1, &passVp);
                    frameCmdList->RSSetScissorRects(1, &passScissor);
                    frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, &fullDsv);
                    frameCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                    if (clearPso != currentPSO)
                    {
                        frameCmdList->SetPipelineState(clearPso);
                        currentPSO        = clearPso;
                        currentComputePSO = nullptr;
                    }

                    // Allocate 48-descriptor Graphics table for Vertex space0 (b0 = Config, t0 = DirtySlices)
                    const UINT32 tableStart = dynamicTableBase[bbIndex] + currentTableOffset;
                    currentTableOffset += kGraphicsDescPerSpace;
                    tablesUsedThisFrame++;

                    if (defaultDummySpaceTableIndex != UINT32_MAX)
                    {
                        device->CopyDescriptorsSimple(
                            kGraphicsDescPerSpace,
                            cbvSrvUavPool.GetCPUHandle(tableStart),
                            cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex),
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                        );
                    }

                    struct LayeredClearConfigCB
                    {
                        FLOAT32 clearDepth;
                        UINT32  maxArrayLayers;
                        UINT32  maxDirtySlices;
                        UINT32  clearFlags;
                    } clearCfg{};
                    clearCfg.clearDepth     = pass.selectiveClear.clearDepth;
                    clearCfg.maxArrayLayers = effectiveLayers;
                    clearCfg.maxDirtySlices = pass.selectiveClear.maxDirtySlices > 0 ? pass.selectiveClear.maxDirtySlices : effectiveLayers;
                    clearCfg.clearFlags     = 0;

                    const D3D12_GPU_VIRTUAL_ADDRESS cbVa = pipeline.AllocateCBuffer(&clearCfg, sizeof(clearCfg));
                    if (cbVa != 0)
                    {
                        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                        cbvDesc.BufferLocation = cbVa;
                        cbvDesc.SizeInBytes    = (sizeof(clearCfg) + 255) & ~255;
                        device->CreateConstantBufferView(&cbvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + 0));
                    }

                    const D3D12_CPU_DESCRIPTOR_HANDLE dirtySrvSrc = cbvSrvUavPool.GetCpuStagingHandle(dirtyListBuf->srvDescriptorIndex);
                    device->CopyDescriptorsSimple(
                        1,
                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + 0), // t0, space0
                        dirtySrvSrc,
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                    );

                    frameCmdList->SetGraphicsRootDescriptorTable(0, cbvSrvUavPool.GetGPUHandle(tableStart));
                    currentRootTable[0] = tableStart;

                    frameCmdList->ExecuteIndirect(
                        pipeline.GetDrawIndirectSignature(),
                        maxCmdCount,
                        indirectArgsBuf->resource.Get(),
                        pass.selectiveClear.indirectDrawArgsOffset,
                        indirectCountBuf ? indirectCountBuf->resource.Get() : nullptr,
                        indirectCountBuf ? pass.selectiveClear.indirectCountBufferOffset : 0
                    );
                    lastFrameMetrics.indirectDrawCommandsExecuted++;
                    if (useGsFallback)
                    {
                        lastFrameMetrics.layeredDepthFallbackPassesExecuted++;
                    }
                }
            }

            // 2. Depth-Only Layered Rasterization Draw Commands
            if (!pass.drawCommands.empty())
            {
                if (pass.indirectPerDrawCBVBuffer != 0)
                {
                    D3D12GPUBufferResource* cbvPoolBuf = resourceMgr.GetGPUBuffer(pass.indirectPerDrawCBVBuffer);
                    if (cbvPoolBuf && cbvPoolBuf->resource)
                    {
                        stateTracker.TransitionResource(cbvPoolBuf->resource.Get(), GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                        cbvPoolBuf->currentState = stateTracker.GetResourceState(cbvPoolBuf->resource.Get());
                    }
                }
                if (pass.indirectPerBatchInstanceBuffer != 0)
                {
                    D3D12GPUBufferResource* instPoolBuf = resourceMgr.GetGPUBuffer(pass.indirectPerBatchInstanceBuffer);
                    if (instPoolBuf && instPoolBuf->resource)
                    {
                        stateTracker.TransitionResource(instPoolBuf->resource.Get(), GPUResourceAccess::ALL_SHADER_RESOURCE);
                        instPoolBuf->currentState = stateTracker.GetResourceState(instPoolBuf->resource.Get());
                    }
                }
                if (pass.indirectBonePaletteBuffer != 0)
                {
                    D3D12GPUBufferResource* bonePoolBuf = resourceMgr.GetGPUBuffer(pass.indirectBonePaletteBuffer);
                    if (bonePoolBuf && bonePoolBuf->resource)
                    {
                        stateTracker.TransitionResource(bonePoolBuf->resource.Get(), GPUResourceAccess::ALL_SHADER_RESOURCE);
                        bonePoolBuf->currentState = stateTracker.GetResourceState(bonePoolBuf->resource.Get());
                    }
                }
                if (pass.indirectVertexPoolBuffer != 0)
                {
                    D3D12GPUBufferResource* vbPoolBuf = resourceMgr.GetGPUBuffer(pass.indirectVertexPoolBuffer);
                    if (vbPoolBuf && vbPoolBuf->resource)
                    {
                        stateTracker.TransitionResource(vbPoolBuf->resource.Get(), GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                        vbPoolBuf->currentState = stateTracker.GetResourceState(vbPoolBuf->resource.Get());
                    }
                }
                if (pass.indirectIndexPoolBuffer != 0)
                {
                    D3D12GPUBufferResource* ibPoolBuf = resourceMgr.GetGPUBuffer(pass.indirectIndexPoolBuffer);
                    if (ibPoolBuf && ibPoolBuf->resource)
                    {
                        stateTracker.TransitionResource(ibPoolBuf->resource.Get(), GPUResourceAccess::INDEX_BUFFER);
                        ibPoolBuf->currentState = stateTracker.GetResourceState(ibPoolBuf->resource.Get());
                    }
                }
                for (const auto& cmd : pass.drawCommands)
                {
                    auto trPool = [&](NativeHandle h, GPUResourceAccess access) {
                        if (h == 0) return;
                        D3D12GPUBufferResource* b = resourceMgr.GetGPUBuffer(h);
                        if (b && b->resource)
                        {
                            stateTracker.TransitionResource(b->resource.Get(), access);
                            b->currentState = stateTracker.GetResourceState(b->resource.Get());
                        }
                    };
                    trPool(cmd.indirectPerDrawCBVBuffer,       GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                    trPool(cmd.indirectPerBatchInstanceBuffer, GPUResourceAccess::ALL_SHADER_RESOURCE);
                    trPool(cmd.indirectBonePaletteBuffer,      GPUResourceAccess::ALL_SHADER_RESOURCE);
                    trPool(cmd.indirectVertexPoolBuffer,       GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                    trPool(cmd.indirectIndexPoolBuffer,        GPUResourceAccess::INDEX_BUFFER);
                }

                stateTracker.TransitionResource(depthTex->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                stateTracker.FlushBarriers(frameCmdList.Get());
                depthTex->currentState = stateTracker.GetResourceState(depthTex->resource.Get());

                ID3D12RootSignature* currentGraphicsRootSig = pipeline.GetRootSignature();
                if (isComputeRootSigActive)
                {
                    BindGraphicsRootSigAndFallbacks(currentGraphicsRootSig);
                    for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                    ResetGraphicsSamplerCache(currentSamplerTableIndex);
                    isComputeRootSigActive = false;
                    currentPSO = nullptr;
                }

                frameCmdList->RSSetViewports(1, &passVp);
                frameCmdList->RSSetScissorRects(1, &passScissor);
                frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, &fullDsv);
                D3D_PRIMITIVE_TOPOLOGY currentTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
                frameCmdList->IASetPrimitiveTopology(currentTopology);

                for (SIZE_T i = 0; i < pass.drawCommands.size(); ++i)
                {
                    const auto& cmd = pass.drawCommands[i];
                    const BOOL isNvsIndirect        = (cmd.useNVSIndirectCommandSignature != FALSE) && (cmd.indirectArgsBuffer != 0);
                    const BOOL isVertexPullIndirect = (cmd.useVertexPullingIndirect != FALSE) && (cmd.indirectArgsBuffer != 0) && !isNvsIndirect;
                    const BOOL skipsIaBuffers       = isNvsIndirect || isVertexPullIndirect;

                    NativeHandle vbHandle = cmd.vertexBuffer;
                    if (!skipsIaBuffers && vbHandle == 0 && cmd.cpuVertices != nullptr && cmd.vertexCount > 0)
                    {
                        vbHandle = resourceMgr.GetOrCreateCachedCpuVertexBuffer(
                            deviceContext, cmd.cpuVertices, cmd.vertexCount, sizeof(Vertex));
                    }

                    const auto* pVb = skipsIaBuffers ? nullptr : resourceMgr.GetVertexBuffer(vbHandle);
                    if (!skipsIaBuffers && !pVb)
                    {
                        if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }

                    NativeHandle ibHandle = cmd.indexBuffer;
                    if (!skipsIaBuffers && ibHandle == 0 && cmd.cpuIndices != nullptr && cmd.indexCount > 0)
                    {
                        ibHandle = resourceMgr.GetOrCreateCachedCpuIndexBuffer(
                            deviceContext, cmd.cpuIndices, cmd.indexCount, true);
                    }

                    const auto* pIb = skipsIaBuffers ? nullptr : resourceMgr.GetIndexBuffer(ibHandle);
                    const BOOL hasIndices = isNvsIndirect ? true : (isVertexPullIndirect ? false : (pIb != nullptr));

                    D3D12GPUBufferResource* indirectArgBuf   = nullptr;
                    D3D12GPUBufferResource* indirectCountBuf = nullptr;
                    ID3D12CommandSignature* indirectCmdSig   = nullptr;

                    if (cmd.indirectArgsBuffer != 0)
                    {
                        const UINT32 argStride = isNvsIndirect
                            ? D3D12Pipeline::kNVSIndirectCommandByteStride
                            : (hasIndices ? sizeof(GPUIndirectDrawIndexedArgs) : sizeof(GPUIndirectDrawArgs));
                        const UINT32 reqAlign = isNvsIndirect ? 8u : 4u;
                        indirectCmdSig = isNvsIndirect
                            ? pipeline.GetOrCreateIndirectCommandSignature(device, cmd.indirectRootKey)
                            : (hasIndices ? pipeline.GetDrawIndexedIndirectSignature() : pipeline.GetDrawIndirectSignature());
                        indirectArgBuf = resourceMgr.GetGPUBuffer(cmd.indirectArgsBuffer);

                        if (cmd.maxIndirectCommandCount == 0 ||
                            !indirectCmdSig ||
                            !indirectArgBuf || !indirectArgBuf->resource ||
                            !HasGPUBufferUsage(indirectArgBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                            (cmd.indirectArgsOffset % reqAlign) != 0u ||
                            static_cast<UINT64>(cmd.indirectArgsOffset) + static_cast<UINT64>(cmd.maxIndirectCommandCount) * static_cast<UINT64>(argStride) > static_cast<UINT64>(indirectArgBuf->byteSize))
                        {
                            lastFrameMetrics.indirectCommandsRejected++;
                            continue;
                        }

                        if (cmd.indirectCountBuffer != 0)
                        {
                            indirectCountBuf = resourceMgr.GetGPUBuffer(cmd.indirectCountBuffer);
                            if (!indirectCountBuf || !indirectCountBuf->resource ||
                                !HasGPUBufferUsage(indirectCountBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                                (cmd.indirectCountBufferOffset % 4u) != 0u ||
                                static_cast<UINT64>(cmd.indirectCountBufferOffset) + sizeof(UINT32) > static_cast<UINT64>(indirectCountBuf->byteSize))
                            {
                                lastFrameMetrics.indirectCommandsRejected++;
                                continue;
                            }

                            if (cmd.indirectCountBuffer == cmd.indirectArgsBuffer)
                            {
                                const UINT64 argBegin   = static_cast<UINT64>(cmd.indirectArgsOffset);
                                const UINT64 argEnd     = argBegin + static_cast<UINT64>(cmd.maxIndirectCommandCount) * static_cast<UINT64>(argStride);
                                const UINT64 countBegin = static_cast<UINT64>(cmd.indirectCountBufferOffset);
                                const UINT64 countEnd   = countBegin + sizeof(UINT32);
                                if (countBegin < argEnd && argBegin < countEnd)
                                {
                                    lastFrameMetrics.indirectCommandsRejected++;
                                    continue;
                                }
                            }
                        }
                    }

                    D3D12GPUBufferResource* gpuInstanceBuf = nullptr;
                    if (cmd.instanceBuffer != 0)
                    {
                        if (cmd.instanceBuffer == cmd.indirectArgsBuffer || cmd.instanceBuffer == cmd.indirectCountBuffer)
                        {
                            if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                            continue;
                        }
                        gpuInstanceBuf = resourceMgr.GetGPUBuffer(cmd.instanceBuffer);
                        if (!gpuInstanceBuf || !gpuInstanceBuf->resource ||
                            !HasGPUBufferUsage(gpuInstanceBuf->desc.usage, GPUBufferUsage::STRUCTURED_SRV) ||
                            gpuInstanceBuf->srvDescriptorIndex == UINT32_MAX)
                        {
                            if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                            continue;
                        }
                    }

                    if (isNvsIndirect && cmd.geometryShader == 0 && useGsFallback)
                    {
                        lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }

                    const NativeHandle gsHandle = isNvsIndirect
                        ? cmd.geometryShader
                        : ((cmd.geometryShader != 0)
                            ? cmd.geometryShader
                            : (useGsFallback ? pass.fallbackGeometryShader : 0));

                    ID3D12RootSignature* reqRootSig = cmd.indirectRootKey.enabled
                        ? pipeline.GetOrCreateIndirectRootSignature(device, cmd.indirectRootKey)
                        : pipeline.GetRootSignature();
                    if (!reqRootSig)
                    {
                        if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }
                    if (reqRootSig != currentGraphicsRootSig)
                    {
                        BindGraphicsRootSigAndFallbacks(reqRootSig);
                        for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                        ResetGraphicsSamplerCache(currentSamplerTableIndex);
                        currentGraphicsRootSig = reqRootSig;
                        currentPSO             = nullptr;
                    }

                    // Depth-only target PSO (NumRenderTargets = 0, DSVFormat = depthTex->dsvFormat; supports optional alpha-test PS and user GS/HS/DS)
                    ID3D12PipelineState* pso = pipeline.GetOrCreatePSO(
                        device,
                        resourceMgr,
                        cmd.vertexShader,
                        cmd.pixelShader,
                        cmd.cullMode,
                        cmd.blendMode,
                        cmd.fillMode,
                        cmd.depthState,
                        false,
                        pass.depthBias,
                        pass.slopeScaledDepthBias,
                        pass.depthBiasClamp,
                        gsHandle,
                        true,
                        cmd.hullShader,
                        cmd.domainShader,
                        cmd.indirectRootKey,
                        DXGI_FORMAT_UNKNOWN,
                        depthTex->dsvFormat
                    );
                    if (!pso)
                    {
                        if (cmd.indirectArgsBuffer != 0) lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }

                    if (pso != currentPSO)
                    {
                        frameCmdList->SetPipelineState(pso);
                        currentPSO        = pso;
                        currentComputePSO = nullptr;
                    }

                    const D3D_PRIMITIVE_TOPOLOGY reqTopology = (cmd.hullShader != 0 && cmd.domainShader != 0)
                        ? D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST
                        : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
                    if (reqTopology != currentTopology)
                    {
                        frameCmdList->IASetPrimitiveTopology(reqTopology);
                        currentTopology = reqTopology;
                    }

                    if (!skipsIaBuffers)
                    {
                        if (vbHandle != currentVB)
                        {
                            frameCmdList->IASetVertexBuffers(0, 1, &pVb->view);
                            currentVB = vbHandle;
                        }

                        if (hasIndices && ibHandle != currentIB)
                        {
                            frameCmdList->IASetIndexBuffer(&pIb->view);
                            currentIB = ibHandle;
                        }
                    }

                    if (gpuInstanceBuf != nullptr)
                    {
                        stateTracker.TransitionResource(gpuInstanceBuf->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                        gpuInstanceBuf->currentState = stateTracker.GetResourceState(gpuInstanceBuf->resource.Get());
                    }
                    if (indirectArgBuf != nullptr)
                    {
                        stateTracker.TransitionResource(indirectArgBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                        indirectArgBuf->currentState = stateTracker.GetResourceState(indirectArgBuf->resource.Get());
                    }
                    if (indirectCountBuf != nullptr)
                    {
                        stateTracker.TransitionResource(indirectCountBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                        indirectCountBuf->currentState = stateTracker.GetResourceState(indirectCountBuf->resource.Get());
                    }
                    if (gpuInstanceBuf != nullptr || indirectArgBuf != nullptr || indirectCountBuf != nullptr)
                    {
                        stateTracker.FlushBarriers(frameCmdList.Get());
                    }

                    BOOL hasVertexSpace0CB0 = false;
                    for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                    {
                        if (cmd.constantBuffers[b].stage == ShaderStage::VERTEX &&
                            cmd.constantBuffers[b].registerSpace == 0 &&
                            cmd.constantBuffers[b].slotIndex == 0)
                        {
                            hasVertexSpace0CB0 = true;
                            break;
                        }
                    }

                    UINT32 instFirstElement = 0;
                    const BOOL isInstanced = (gpuInstanceBuf == nullptr) &&
                        ((cmd.instanceCount > 1 || cmd.bufferCategory == BufferCategory::INSTANCE) && !cmd.instances.empty());
                    if (isInstanced)
                    {
                        instFirstElement = pipeline.AllocateInstances(
                            reinterpret_cast<const D3D12InstancePayload*>(cmd.instances.data()),
                            static_cast<UINT32>(cmd.instances.size()));
                    }

                    BOOL layeredRingExhausted = false;
                    for (UINT32 space = 0; space < kMaxSpaces; ++space)
                    {
                        BOOL spaceActive = (space == 0);
                        if (!spaceActive &&
                            cmd.hasReflectedInstanceBinding &&
                            (isInstanced || gpuInstanceBuf != nullptr) &&
                            cmd.instanceBindingSpace == space &&
                            cmd.instanceBindingSlot < kSRVPerSpace)
                        {
                            spaceActive = true;
                        }
                        if (!spaceActive)
                        {
                            for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                            {
                                if (cmd.constantBuffers[b].stage == ShaderStage::VERTEX && cmd.constantBuffers[b].registerSpace == space)
                                { spaceActive = true; break; }
                            }
                        }
                        if (!spaceActive)
                        {
                            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                            {
                                if (cmd.structuredBuffers[s].stage == ShaderStage::VERTEX && cmd.structuredBuffers[s].registerSpace == space)
                                { spaceActive = true; break; }
                            }
                        }
                        if (!spaceActive)
                        {
                            for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                            {
                                if (cmd.textures[t].stage == ShaderStage::VERTEX && cmd.textures[t].registerSpace == space)
                                { spaceActive = true; break; }
                            }
                        }
                        if (!spaceActive) continue;

                        if (tablesUsedThisFrame >= kTablesPerBackBuffer ||
                            currentTableOffset + kGraphicsDescPerSpace > kRingDescriptorsPerBB)
                        {
                            layeredRingExhausted = true;
                            break;
                        }
                        const UINT32 tableStart = dynamicTableBase[bbIndex] + currentTableOffset;
                        currentTableOffset += kGraphicsDescPerSpace;
                        tablesUsedThisFrame++;

                        if (defaultDummySpaceTableIndex != UINT32_MAX)
                        {
                            device->CopyDescriptorsSimple(
                                kGraphicsDescPerSpace,
                                cbvSrvUavPool.GetCPUHandle(tableStart),
                                cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                            );
                        }

                        if (space == 0)
                        {
                            if (!hasVertexSpace0CB0 && !isNvsIndirect)
                            {
                                struct TransformsFallback {
                                    Mat4  world;
                                    float normalMat[12];
                                    Mat4  viewProj;
                                } fallback{};
                                fallback.world        = cmd.worldMatrix;
                                {
                                    const Vec3 c0(cmd.worldMatrix.m[0], cmd.worldMatrix.m[1], cmd.worldMatrix.m[2]);
                                    const Vec3 c1(cmd.worldMatrix.m[4], cmd.worldMatrix.m[5], cmd.worldMatrix.m[6]);
                                    const Vec3 c2(cmd.worldMatrix.m[8], cmd.worldMatrix.m[9], cmd.worldMatrix.m[10]);
                                    const Vec3 cof0 = c1.Cross(c2);
                                    const Vec3 cof1 = c2.Cross(c0);
                                    const Vec3 cof2 = c0.Cross(c1);
                                    const FLOAT32 det = c0.Dot(cof0);
                                    if (std::isfinite(det) && std::fabs(det) > 1e-12f)
                                    {
                                        const FLOAT32 invDet = 1.0f / det;
                                        fallback.normalMat[0]  = cof0.x * invDet; fallback.normalMat[1]  = cof0.y * invDet; fallback.normalMat[2]  = cof0.z * invDet;
                                        fallback.normalMat[4]  = cof1.x * invDet; fallback.normalMat[5]  = cof1.y * invDet; fallback.normalMat[6]  = cof1.z * invDet;
                                        fallback.normalMat[8]  = cof2.x * invDet; fallback.normalMat[9]  = cof2.y * invDet; fallback.normalMat[10] = cof2.z * invDet;
                                    }
                                    else
                                    {
                                        fallback.normalMat[0] = 1.0f; fallback.normalMat[5] = 1.0f; fallback.normalMat[10] = 1.0f;
                                    }
                                }
                                fallback.viewProj     = defaultViewProjMatrix;

                                const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateCBuffer(&fallback, sizeof(fallback));
                                if (va != 0)
                                {
                                    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                    cbvDesc.BufferLocation = va;
                                    cbvDesc.SizeInBytes    = (sizeof(fallback) + 255) & ~255;
                                    device->CreateConstantBufferView(&cbvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + 0));
                                }
                            }

                            if (!cmd.hasReflectedInstanceBinding)
                            {
                                auto hasVertexSpace0SBAtSlot = [&](UINT16 slotIdx) -> BOOL {
                                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                                    {
                                        const auto& sb = cmd.structuredBuffers[s];
                                        if (sb.stage == ShaderStage::VERTEX && sb.registerSpace == 0 && sb.slotIndex == slotIdx)
                                            return true;
                                    }
                                    return false;
                                };
                                if (gpuInstanceBuf != nullptr)
                                {
                                    D3D12_CPU_DESCRIPTOR_HANDLE instSrc = cbvSrvUavPool.GetCpuStagingHandle(gpuInstanceBuf->srvDescriptorIndex);
                                    if (!hasVertexSpace0SBAtSlot(0))
                                    {
                                        device->CopyDescriptorsSimple(1, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + 0), instSrc, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                    }
                                    if (!hasVertexSpace0SBAtSlot(1))
                                    {
                                        device->CopyDescriptorsSimple(1, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + 1), instSrc, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                    }
                                }
                                else if (isInstanced && pipeline.GetInstanceUploadBuffer())
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC instSrvDesc{};
                                    instSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                    instSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                    instSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    instSrvDesc.Buffer.FirstElement        = instFirstElement;
                                    instSrvDesc.Buffer.NumElements         = static_cast<UINT>(cmd.instances.size());
                                    instSrvDesc.Buffer.StructureByteStride = sizeof(D3D12InstancePayload);
                                    instSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;

                                    if (!hasVertexSpace0SBAtSlot(0))
                                    {
                                        device->CreateShaderResourceView(pipeline.GetInstanceUploadBuffer(), &instSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + 0));
                                    }
                                    if (!hasVertexSpace0SBAtSlot(1))
                                    {
                                        device->CreateShaderResourceView(pipeline.GetInstanceUploadBuffer(), &instSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + 1));
                                    }
                                }
                            }
                        }

                        if (cmd.hasReflectedInstanceBinding &&
                            cmd.instanceBindingSpace == space &&
                            cmd.instanceBindingSlot < kSRVPerSpace)
                        {
                            if (gpuInstanceBuf != nullptr)
                            {
                                D3D12_CPU_DESCRIPTOR_HANDLE instSrc = cbvSrvUavPool.GetCpuStagingHandle(gpuInstanceBuf->srvDescriptorIndex);
                                device->CopyDescriptorsSimple(
                                    1,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + cmd.instanceBindingSlot),
                                    instSrc,
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }
                            else if (isInstanced && pipeline.GetInstanceUploadBuffer())
                            {
                                D3D12_SHADER_RESOURCE_VIEW_DESC instSrvDesc{};
                                instSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                instSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                instSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                instSrvDesc.Buffer.FirstElement        = instFirstElement;
                                instSrvDesc.Buffer.NumElements         = static_cast<UINT>(cmd.instances.size());
                                instSrvDesc.Buffer.StructureByteStride = sizeof(D3D12InstancePayload);
                                instSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                device->CreateShaderResourceView(
                                    pipeline.GetInstanceUploadBuffer(),
                                    &instSrvDesc,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + cmd.instanceBindingSlot)
                                );
                            }
                        }

                        for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                        {
                            const auto& cb = cmd.constantBuffers[b];
                            if (cb.stage == ShaderStage::VERTEX && cb.registerSpace == space && cb.slotIndex < kCBVPerSpace)
                            {
                                const BYTE* data = cb.GetData();
                                if (!data || cb.byteSize == 0) continue;
                                const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateCBuffer(data, cb.byteSize);
                                if (va == 0) continue;

                                D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                cbvDesc.BufferLocation = va;
                                cbvDesc.SizeInBytes    = (cb.byteSize + 255) & ~255;
                                device->CreateConstantBufferView(&cbvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + cb.slotIndex));
                            }
                        }

                        for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                        {
                            const auto& sb = cmd.structuredBuffers[s];
                            if (sb.isUAV) continue;
                            if (sb.stage == ShaderStage::VERTEX && sb.registerSpace == space && sb.slotIndex < kSRVPerSpace)
                            {
                                if (sb.buffer != 0)
                                {
                                    D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                                    if (gpuBuf && gpuBuf->resource && gpuBuf->srvDescriptorIndex != UINT32_MAX)
                                    {
                                        const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                        if ((curState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                                        {
                                            stateTracker.TransitionResource(gpuBuf->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                                            stateTracker.FlushBarriers(frameCmdList.Get());
                                            gpuBuf->currentState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                        }

                                        const UINT32 reqStride = (sb.stride > 0u) ? sb.stride : (gpuBuf->desc.stride > 0u ? gpuBuf->desc.stride : 4u);
                                        const BOOL   explicitZeroSlice = (sb.elementCount == 0u && sb.stride > 0u);
                                        const UINT32 reqCount  = (sb.elementCount > 0u) ? sb.elementCount : gpuBuf->desc.elementCount;
                                        if (explicitZeroSlice)
                                        {
                                            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                            nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                            nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                            nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                            nullSrvDesc.Buffer.FirstElement        = 0;
                                            nullSrvDesc.Buffer.NumElements         = 0;
                                            nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                            nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                            device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                        }
                                        else if (sb.firstElement > 0u ||
                                                 reqStride != gpuBuf->desc.stride ||
                                                 reqCount != gpuBuf->desc.elementCount)
                                        {
                                            const UINT32 maxElems = (reqStride > 0u) ? (gpuBuf->byteSize / reqStride) : 0u;
                                            if (sb.firstElement < maxElems)
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC sliceSrv{};
                                                sliceSrv.Format                     = DXGI_FORMAT_UNKNOWN;
                                                sliceSrv.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                sliceSrv.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                sliceSrv.Buffer.FirstElement        = sb.firstElement;
                                                sliceSrv.Buffer.NumElements         = (std::min)(reqCount, maxElems - sb.firstElement);
                                                sliceSrv.Buffer.StructureByteStride = reqStride;
                                                sliceSrv.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(
                                                    gpuBuf->resource.Get(),
                                                    &sliceSrv,
                                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                            }
                                            else
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                                nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                                nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                nullSrvDesc.Buffer.FirstElement        = 0;
                                                nullSrvDesc.Buffer.NumElements         = 0;
                                                nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                                nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                            }
                                        }
                                        else
                                        {
                                            D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->srvDescriptorIndex);
                                            device->CopyDescriptorsSimple(
                                                1,
                                                cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex),
                                                src,
                                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                            );
                                        }
                                    }
                                    continue;
                                }

                                const UINT32 stride = sb.stride > 0 ? sb.stride : 4;
                                if (sb.elementCount == 0 || sb.byteSize == 0 || !sb.GetData())
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                    nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                    nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                    nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    nullSrvDesc.Buffer.FirstElement        = 0;
                                    nullSrvDesc.Buffer.NumElements         = 0;
                                    nullSrvDesc.Buffer.StructureByteStride = stride;
                                    nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                    device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                    continue;
                                }

                                CachedSharedSBUpload sbUpload{};
                                if (!AllocateOrReuseStructuredBuffer(sb, stride, sbUpload)) continue;

                                D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
                                srvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                srvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                srvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                srvDesc.Buffer.FirstElement        = sbUpload.firstElement;
                                srvDesc.Buffer.NumElements         = sbUpload.numElements;
                                srvDesc.Buffer.StructureByteStride = sbUpload.stride;
                                srvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;

                                device->CreateShaderResourceView(
                                    sbUpload.resource,
                                    &srvDesc,
                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                );
                            }
                        }

                        for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                        {
                            const auto& tex = cmd.textures[t];
                            if (tex.stage == ShaderStage::VERTEX && tex.registerSpace == space && tex.slotIndex < kSRVPerSpace && tex.texture != 0)
                            {
                                D3D12TextureResource* texRes = resourceMgr.GetTexture(tex.texture);
                                if (texRes && texRes->resource && texRes->srvDescriptorIndex != UINT32_MAX)
                                {
                                    if (tex.texture != pass.depthArrayTexture &&
                                        (texRes->desc.usage == TextureUsage::STORAGE ||
                                         texRes->desc.usage == TextureUsage::DEPTH_STENCIL ||
                                         texRes->desc.usage == TextureUsage::RENDER_TARGET))
                                    {
                                        if (!stateTracker.IsResourceTracked(texRes->resource.Get()))
                                        {
                                            RegisterTrackedTexture(stateTracker, texRes);
                                        }
                                        const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(texRes->resource.Get());
                                        if ((curState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                                        {
                                            stateTracker.TransitionResource(texRes->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                                            stateTracker.FlushBarriers(frameCmdList.Get());
                                            texRes->currentState = stateTracker.GetResourceState(texRes->resource.Get());
                                        }
                                    }
                                    D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(texRes->srvDescriptorIndex);
                                    device->CopyDescriptorsSimple(
                                        1,
                                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + tex.slotIndex),
                                        src,
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                    );
                                }
                            }
                        }

                        if (currentRootTable[space] != tableStart)
                        {
                            frameCmdList->SetGraphicsRootDescriptorTable(space, cbvSrvUavPool.GetGPUHandle(tableStart));
                            currentRootTable[space] = tableStart;
                        }
                    }

                    // Bind Pixel Space Tables (Params 4..7) when an alpha-tested or custom shadow PS is bound
                    if (cmd.pixelShader != 0)
                    {
                        for (UINT32 space = 0; space < kMaxSpaces; ++space)
                        {
                            BOOL psSpaceActive = false;
                            for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                            {
                                if (cmd.constantBuffers[b].stage == ShaderStage::PIXEL && cmd.constantBuffers[b].registerSpace == space)
                                { psSpaceActive = true; break; }
                            }
                            if (!psSpaceActive)
                            {
                                for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                                {
                                    if (cmd.structuredBuffers[s].stage == ShaderStage::PIXEL && cmd.structuredBuffers[s].registerSpace == space)
                                    { psSpaceActive = true; break; }
                                }
                            }
                            if (!psSpaceActive)
                            {
                                for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                                {
                                    if (cmd.textures[t].stage == ShaderStage::PIXEL && cmd.textures[t].registerSpace == space)
                                    { psSpaceActive = true; break; }
                                }
                            }
                            if (!psSpaceActive) continue;
                            if (tablesUsedThisFrame >= kTablesPerBackBuffer ||
                                currentTableOffset + kGraphicsDescPerSpace > kRingDescriptorsPerBB)
                            {
                                layeredRingExhausted = true;
                                break;
                            }
                            const UINT32 tableStart = dynamicTableBase[bbIndex] + currentTableOffset;
                            currentTableOffset += kGraphicsDescPerSpace;
                            tablesUsedThisFrame++;

                            if (defaultDummySpaceTableIndex != UINT32_MAX)
                            {
                                device->CopyDescriptorsSimple(
                                    kGraphicsDescPerSpace,
                                    cbvSrvUavPool.GetCPUHandle(tableStart),
                                    cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex),
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }

                            for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                            {
                                const auto& cb = cmd.constantBuffers[b];
                                if (cb.stage == ShaderStage::PIXEL && cb.registerSpace == space && cb.slotIndex < kCBVPerSpace)
                                {
                                    const BYTE* data = cb.GetData();
                                    if (!data || cb.byteSize == 0) continue;
                                    const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateCBuffer(data, cb.byteSize);
                                    if (va == 0) continue;

                                    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                    cbvDesc.BufferLocation = va;
                                    cbvDesc.SizeInBytes    = (cb.byteSize + 255) & ~255;
                                    device->CreateConstantBufferView(&cbvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + cb.slotIndex));
                                }
                            }

                            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                            {
                                const auto& sb = cmd.structuredBuffers[s];
                                if (sb.isUAV) continue;
                                if (sb.stage == ShaderStage::PIXEL && sb.registerSpace == space && sb.slotIndex < kSRVPerSpace)
                                {
                                    if (sb.buffer != 0)
                                    {
                                        D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                                        if (gpuBuf && gpuBuf->resource && gpuBuf->srvDescriptorIndex != UINT32_MAX)
                                        {
                                            const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                            if ((curState & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) == 0)
                                            {
                                                stateTracker.TransitionResource(gpuBuf->resource.Get(), GPUResourceAccess::ALL_SHADER_RESOURCE);
                                                stateTracker.FlushBarriers(frameCmdList.Get());
                                                gpuBuf->currentState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                            }
                                            const UINT32 reqStride = (sb.stride > 0u) ? sb.stride : (gpuBuf->desc.stride > 0u ? gpuBuf->desc.stride : 4u);
                                            const BOOL   explicitZeroSlice = (sb.elementCount == 0u && sb.stride > 0u);
                                            const UINT32 reqCount  = (sb.elementCount > 0u) ? sb.elementCount : gpuBuf->desc.elementCount;
                                            if (explicitZeroSlice)
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                                nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                                nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                nullSrvDesc.Buffer.FirstElement        = 0;
                                                nullSrvDesc.Buffer.NumElements         = 0;
                                                nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                                nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                            }
                                            else if (sb.firstElement > 0u || reqStride != gpuBuf->desc.stride || reqCount != gpuBuf->desc.elementCount)
                                            {
                                                const UINT32 maxElems = (reqStride > 0u) ? (gpuBuf->byteSize / reqStride) : 0u;
                                                D3D12_SHADER_RESOURCE_VIEW_DESC sliceSrv{};
                                                sliceSrv.Format                     = DXGI_FORMAT_UNKNOWN;
                                                sliceSrv.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                sliceSrv.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                sliceSrv.Buffer.StructureByteStride = reqStride;
                                                sliceSrv.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                if (sb.firstElement < maxElems)
                                                {
                                                    sliceSrv.Buffer.FirstElement = sb.firstElement;
                                                    sliceSrv.Buffer.NumElements  = (std::min)(reqCount, maxElems - sb.firstElement);
                                                    device->CreateShaderResourceView(gpuBuf->resource.Get(), &sliceSrv, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                                }
                                                else
                                                {
                                                    sliceSrv.Buffer.FirstElement = 0;
                                                    sliceSrv.Buffer.NumElements  = 0;
                                                    device->CreateShaderResourceView(nullptr, &sliceSrv, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                                }
                                            }
                                            else
                                            {
                                                D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->srvDescriptorIndex);
                                                device->CopyDescriptorsSimple(
                                                    1,
                                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex),
                                                    src,
                                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                                );
                                            }
                                        }
                                    }
                                    else
                                    {
                                        const UINT32 stride = sb.stride > 0 ? sb.stride : 4;
                                        if (sb.elementCount == 0 || sb.byteSize == 0 || !sb.GetData())
                                        {
                                            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                            nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                            nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                            nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                            nullSrvDesc.Buffer.FirstElement        = 0;
                                            nullSrvDesc.Buffer.NumElements         = 0;
                                            nullSrvDesc.Buffer.StructureByteStride = stride;
                                            nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                            device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                        }
                                        else
                                        {
                                            CachedSharedSBUpload sbUpload{};
                                            if (AllocateOrReuseStructuredBuffer(sb, stride, sbUpload))
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
                                                srvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                                srvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                srvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                srvDesc.Buffer.FirstElement        = sbUpload.firstElement;
                                                srvDesc.Buffer.NumElements         = sbUpload.numElements;
                                                srvDesc.Buffer.StructureByteStride = sbUpload.stride;
                                                srvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(sbUpload.resource, &srvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                            }
                                        }
                                    }
                                }
                            }

                            for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                            {
                                const auto& tex = cmd.textures[t];
                                if (tex.stage == ShaderStage::PIXEL && tex.registerSpace == space && tex.slotIndex < kSRVPerSpace && tex.texture != 0)
                                {
                                    D3D12TextureResource* texRes = resourceMgr.GetTexture(tex.texture);
                                    if (texRes && texRes->resource && texRes->srvDescriptorIndex != UINT32_MAX)
                                    {
                                        if (tex.texture != pass.depthArrayTexture &&
                                            (texRes->desc.usage == TextureUsage::STORAGE ||
                                             texRes->desc.usage == TextureUsage::DEPTH_STENCIL ||
                                             texRes->desc.usage == TextureUsage::RENDER_TARGET))
                                        {
                                            if (!stateTracker.IsResourceTracked(texRes->resource.Get()))
                                            {
                                                RegisterTrackedTexture(stateTracker, texRes);
                                            }
                                            const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(texRes->resource.Get());
                                            if ((curState & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) == 0)
                                            {
                                                stateTracker.TransitionResource(texRes->resource.Get(), GPUResourceAccess::ALL_SHADER_RESOURCE);
                                                stateTracker.FlushBarriers(frameCmdList.Get());
                                                texRes->currentState = stateTracker.GetResourceState(texRes->resource.Get());
                                            }
                                        }
                                        D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(texRes->srvDescriptorIndex);
                                        device->CopyDescriptorsSimple(
                                            1,
                                            cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + tex.slotIndex),
                                            src,
                                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                        );
                                    }
                                }
                            }

                            if (currentRootTable[4 + space] != tableStart)
                            {
                                frameCmdList->SetGraphicsRootDescriptorTable(4 + space, cbvSrvUavPool.GetGPUHandle(tableStart));
                                currentRootTable[4 + space] = tableStart;
                            }
                        }
                    }

                    // Bind Geometry / Hull / Domain Space Tables when the corresponding stage is bound
                    struct ExtraStageBindingSpec
                    {
                        NativeHandle handle;
                        ShaderStage  stage;
                        UINT32       rootParamBase;
                    };
                    const ExtraStageBindingSpec extraStages[3] = {
                        { gsHandle,         ShaderStage::GEOMETRY, D3D12Pipeline::kGraphicsGeometrySpaceParamBase },
                        { cmd.hullShader,   ShaderStage::HULL,     D3D12Pipeline::kGraphicsHullSpaceParamBase     },
                        { cmd.domainShader, ShaderStage::DOMAIN,   D3D12Pipeline::kGraphicsDomainSpaceParamBase   }
                    };

                    for (UINT32 esIdx = 0; esIdx < 3; ++esIdx)
                    {
                        const NativeHandle stageHandle   = extraStages[esIdx].handle;
                        const ShaderStage  curStage      = extraStages[esIdx].stage;
                        const UINT32       rootParamBase = extraStages[esIdx].rootParamBase;
                        if (stageHandle == 0) continue;

                        for (UINT32 space = 0; space < kMaxSpaces; ++space)
                        {
                            BOOL stageSpaceActive = false;
                            for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                            {
                                if (cmd.constantBuffers[b].stage == curStage && cmd.constantBuffers[b].registerSpace == space)
                                { stageSpaceActive = true; break; }
                            }
                            if (!stageSpaceActive)
                            {
                                for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                                {
                                    if (cmd.structuredBuffers[s].stage == curStage && cmd.structuredBuffers[s].registerSpace == space)
                                    { stageSpaceActive = true; break; }
                                }
                            }
                            if (!stageSpaceActive)
                            {
                                for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                                {
                                    if (cmd.textures[t].stage == curStage && cmd.textures[t].registerSpace == space)
                                    { stageSpaceActive = true; break; }
                                }
                            }
                            if (!stageSpaceActive) continue;
                            if (tablesUsedThisFrame >= kTablesPerBackBuffer ||
                                currentTableOffset + kGraphicsDescPerSpace > kRingDescriptorsPerBB)
                            {
                                layeredRingExhausted = true;
                                break;
                            }
                            const UINT32 tableStart = dynamicTableBase[bbIndex] + currentTableOffset;
                            currentTableOffset += kGraphicsDescPerSpace;
                            tablesUsedThisFrame++;

                            if (defaultDummySpaceTableIndex != UINT32_MAX)
                            {
                                device->CopyDescriptorsSimple(
                                    kGraphicsDescPerSpace,
                                    cbvSrvUavPool.GetCPUHandle(tableStart),
                                    cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex),
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }

                            for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                            {
                                const auto& cb = cmd.constantBuffers[b];
                                if (cb.stage == curStage && cb.registerSpace == space && cb.slotIndex < kCBVPerSpace)
                                {
                                    const BYTE* data = cb.GetData();
                                    if (!data || cb.byteSize == 0) continue;
                                    const D3D12_GPU_VIRTUAL_ADDRESS va = pipeline.AllocateCBuffer(data, cb.byteSize);
                                    if (va == 0) continue;

                                    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                    cbvDesc.BufferLocation = va;
                                    cbvDesc.SizeInBytes    = (cb.byteSize + 255) & ~255;
                                    device->CreateConstantBufferView(&cbvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + cb.slotIndex));
                                }
                            }

                            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                            {
                                const auto& sb = cmd.structuredBuffers[s];
                                if (sb.isUAV) continue;
                                if (sb.stage == curStage && sb.registerSpace == space && sb.slotIndex < kSRVPerSpace)
                                {
                                    if (sb.buffer != 0)
                                    {
                                        D3D12GPUBufferResource* gpuBuf = resourceMgr.GetGPUBuffer(sb.buffer);
                                        if (gpuBuf && gpuBuf->resource && gpuBuf->srvDescriptorIndex != UINT32_MAX)
                                        {
                                            const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                            if ((curState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                                            {
                                                stateTracker.TransitionResource(gpuBuf->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                                                stateTracker.FlushBarriers(frameCmdList.Get());
                                                gpuBuf->currentState = stateTracker.GetResourceState(gpuBuf->resource.Get());
                                            }
                                            const UINT32 reqStride = (sb.stride > 0u) ? sb.stride : (gpuBuf->desc.stride > 0u ? gpuBuf->desc.stride : 4u);
                                            const BOOL   explicitZeroSlice = (sb.elementCount == 0u && sb.stride > 0u);
                                            const UINT32 reqCount  = (sb.elementCount > 0u) ? sb.elementCount : gpuBuf->desc.elementCount;
                                            if (explicitZeroSlice)
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                                nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                                nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                nullSrvDesc.Buffer.FirstElement        = 0;
                                                nullSrvDesc.Buffer.NumElements         = 0;
                                                nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                                nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                            }
                                            else if (sb.firstElement > 0u || reqStride != gpuBuf->desc.stride || reqCount != gpuBuf->desc.elementCount)
                                            {
                                                const UINT32 maxElems = (reqStride > 0u) ? (gpuBuf->byteSize / reqStride) : 0u;
                                                D3D12_SHADER_RESOURCE_VIEW_DESC sliceSrv{};
                                                sliceSrv.Format                     = DXGI_FORMAT_UNKNOWN;
                                                sliceSrv.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                sliceSrv.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                sliceSrv.Buffer.StructureByteStride = reqStride;
                                                sliceSrv.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                if (sb.firstElement < maxElems)
                                                {
                                                    sliceSrv.Buffer.FirstElement = sb.firstElement;
                                                    sliceSrv.Buffer.NumElements  = (std::min)(reqCount, maxElems - sb.firstElement);
                                                    device->CreateShaderResourceView(gpuBuf->resource.Get(), &sliceSrv, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                                }
                                                else
                                                {
                                                    sliceSrv.Buffer.FirstElement = 0;
                                                    sliceSrv.Buffer.NumElements  = 0;
                                                    device->CreateShaderResourceView(nullptr, &sliceSrv, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                                }
                                            }
                                            else
                                            {
                                                D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->srvDescriptorIndex);
                                                device->CopyDescriptorsSimple(
                                                    1,
                                                    cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex),
                                                    src,
                                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                                );
                                            }
                                        }
                                        continue;
                                    }

                                    const UINT32 stride = sb.stride > 0 ? sb.stride : 4;
                                    if (sb.elementCount == 0 || sb.byteSize == 0 || !sb.GetData())
                                    {
                                        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                        nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                        nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                        nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                        nullSrvDesc.Buffer.FirstElement        = 0;
                                        nullSrvDesc.Buffer.NumElements         = 0;
                                        nullSrvDesc.Buffer.StructureByteStride = stride;
                                        nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                        device->CreateShaderResourceView(nullptr, &nullSrvDesc, cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex));
                                        continue;
                                    }

                                    CachedSharedSBUpload sbUpload{};
                                    if (!AllocateOrReuseStructuredBuffer(sb, stride, sbUpload)) continue;

                                    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
                                    srvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                    srvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                    srvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    srvDesc.Buffer.FirstElement        = sbUpload.firstElement;
                                    srvDesc.Buffer.NumElements         = sbUpload.numElements;
                                    srvDesc.Buffer.StructureByteStride = sbUpload.stride;
                                    srvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;

                                    device->CreateShaderResourceView(
                                        sbUpload.resource,
                                        &srvDesc,
                                        cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + sb.slotIndex)
                                    );
                                }
                            }

                            for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                            {
                                const auto& tex = cmd.textures[t];
                                if (tex.stage == curStage && tex.registerSpace == space && tex.slotIndex < kSRVPerSpace && tex.texture != 0)
                                {
                                    D3D12TextureResource* texRes = resourceMgr.GetTexture(tex.texture);
                                    if (texRes && texRes->resource && texRes->srvDescriptorIndex != UINT32_MAX)
                                    {
                                        if (tex.texture != pass.depthArrayTexture &&
                                            (texRes->desc.usage == TextureUsage::STORAGE ||
                                             texRes->desc.usage == TextureUsage::DEPTH_STENCIL ||
                                             texRes->desc.usage == TextureUsage::RENDER_TARGET))
                                        {
                                            if (!stateTracker.IsResourceTracked(texRes->resource.Get()))
                                            {
                                                RegisterTrackedTexture(stateTracker, texRes);
                                            }
                                            const D3D12_RESOURCE_STATES curState = stateTracker.GetResourceState(texRes->resource.Get());
                                            if ((curState & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) == 0)
                                            {
                                                stateTracker.TransitionResource(texRes->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                                                stateTracker.FlushBarriers(frameCmdList.Get());
                                                texRes->currentState = stateTracker.GetResourceState(texRes->resource.Get());
                                            }
                                        }
                                        D3D12_CPU_DESCRIPTOR_HANDLE src = cbvSrvUavPool.GetCpuStagingHandle(texRes->srvDescriptorIndex);
                                        device->CopyDescriptorsSimple(
                                            1,
                                            cbvSrvUavPool.GetCPUHandle(tableStart + kCBVPerSpace + tex.slotIndex),
                                            src,
                                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                        );
                                    }
                                }
                            }

                            const UINT32 stageRootParam = rootParamBase + space;
                            if (currentRootTable[stageRootParam] != tableStart)
                            {
                                frameCmdList->SetGraphicsRootDescriptorTable(stageRootParam, cbvSrvUavPool.GetGPUHandle(tableStart));
                                currentRootTable[stageRootParam] = tableStart;
                            }
                        }
                    }

                    if (layeredRingExhausted)
                    {
                        if (indirectArgBuf != nullptr) lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }

                    // Bind per-stage multi-space samplers (VS: 8..11, PS: 12..15, GS: 29..32, HS: 33..36, DS: 37..40) for user shadow shaders
                    if (cmd.pixelShader != 0 || !cmd.samplers.empty())
                    {
                        const ShaderStage sampStages[5] = {
                            ShaderStage::VERTEX,
                            ShaderStage::PIXEL,
                            ShaderStage::GEOMETRY,
                            ShaderStage::HULL,
                            ShaderStage::DOMAIN
                        };
                        const BOOL stageActive[5] = {
                            true,
                            cmd.pixelShader != 0,
                            gsHandle != 0,
                            cmd.hullShader != 0,
                            cmd.domainShader != 0
                        };
                        for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
                        {
                            if (!stageActive[stIdx]) continue;
                            const ShaderStage st = sampStages[stIdx];
                            const UINT32 paramBase = D3D12Pipeline::GetGraphicsStageSamplerParamBase(st);
                            for (UINT32 space = 0; space < kMaxSpaces; ++space)
                            {
                                const UINT32 targetSamplerTable = cmd.samplers.empty()
                                    ? defaultDummySamplerTableIndex
                                    : GetOrCreateSamplerTable(device, cmd.samplers, space, st);
                                if (targetSamplerTable != UINT32_MAX && targetSamplerTable != currentSamplerTableIndex[stIdx][space])
                                {
                                    frameCmdList->SetGraphicsRootDescriptorTable(
                                        paramBase + space,
                                        samplerPool.GetGPUHandle(targetSamplerTable)
                                    );
                                    currentSamplerTableIndex[stIdx][space] = targetSamplerTable;
                                }
                            }
                        }
                    }

                    if (cmd.useGraphicsRootConstants)
                    {
                        frameCmdList->SetGraphicsRoot32BitConstants(
                            D3D12Pipeline::kGraphicsRootConstantsParamIndex,
                            4,
                            cmd.graphicsRootConstants,
                            0
                        );
                    }

                    if (indirectArgBuf != nullptr)
                    {
                        frameCmdList->ExecuteIndirect(
                            indirectCmdSig,
                            cmd.maxIndirectCommandCount,
                            indirectArgBuf->resource.Get(),
                            cmd.indirectArgsOffset,
                            indirectCountBuf ? indirectCountBuf->resource.Get() : nullptr,
                            indirectCountBuf ? cmd.indirectCountBufferOffset : 0
                        );
                        if (isNvsIndirect)
                        {
                            currentVB = 0;
                            currentIB = 0;
                            const UINT32 zeroRootConsts[4] = { 0, 0, 0, 0 };
                            frameCmdList->SetGraphicsRoot32BitConstants(
                                D3D12Pipeline::kGraphicsRootConstantsParamIndex,
                                4,
                                zeroRootConsts,
                                0
                            );
                        }
                        lastFrameMetrics.indirectDrawCommandsExecuted++;
                    }
                    else if (isInstanced || gpuInstanceBuf != nullptr)
                    {
                        if (hasIndices)
                        {
                            frameCmdList->DrawIndexedInstanced(
                                cmd.indexCount, cmd.instanceCount, cmd.startIndex, cmd.baseVertex, cmd.startInstance);
                        }
                        else
                        {
                            frameCmdList->DrawInstanced(
                                cmd.vertexCount, cmd.instanceCount, 0, cmd.startInstance);
                        }
                    }
                    else
                    {
                        if (hasIndices)
                        {
                            frameCmdList->DrawIndexedInstanced(
                                cmd.indexCount, 1, cmd.startIndex, cmd.baseVertex, 0);
                        }
                        else
                        {
                            frameCmdList->DrawInstanced(
                                cmd.vertexCount, 1, 0, 0);
                        }
                    }

                    if (gsHandle != 0 && cmd.geometryShader == 0 && useGsFallback)
                    {
                        lastFrameMetrics.layeredDepthFallbackPassesExecuted++;
                    }
                }

                if (currentGraphicsRootSig != pipeline.GetRootSignature())
                {
                    BindGraphicsRootSigAndFallbacks(pipeline.GetRootSignature());
                    for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                    ResetGraphicsSamplerCache(currentSamplerTableIndex);
                    currentPSO = nullptr;
                }
            }

            // Unbind DSV so subsequent compute/graphics passes can transition depthArrayTexture to SRV safely
            frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);

            lastFrameMetrics.layeredDepthPassesExecuted++;
            return true;
        }

        BOOL isInitialized = false;

        void CleanupResources()
        {
            deviceContext.WaitForGPU();

            timestampReadbackBuffer.Reset();
            timestampQueryHeap.Reset();
            for (UINT32 i = 0; i < kMaxFramesInFlight; ++i)
            {
                timestampSlotFlags[i] = TimestampSlotFlags{};
            }
            pipeline.Shutdown();
            cachedSharedSBUploads.clear();
            samplerTableCache.clear();
            retiredSamplerTables.clear();
            importedSharedFences.clear();
            ownerThreadId = std::thread::id{};
            isRecordingCmdList = false;
            stateTracker.Reset();
            resourceMgr.Shutdown(cbvSrvUavPool, rtvPool, dsvPool, samplerPool);

            frameCmdList.Reset();
            for (UINT32 i = 0; i < kMaxFramesInFlight; ++i)
            {
                frameCmdAlloc[i].Reset();
                frameFenceValues[i] = 0;
                dynamicTableBase[i] = UINT32_MAX;
            }
            for (UINT32 i = 0; i < D3D12SwapChain::kBackBufferCount; ++i)
            {
                swapChainFenceValues[i] = 0;
            }
            submissionSlotCursor = 0;
            defaultDummyZeroCBuffer.Reset();
            defaultDummySpaceTableIndex   = UINT32_MAX;
            defaultDummySamplerTableIndex = UINT32_MAX;
            defaultSamplerHandle          = 0;

            swapChain.Shutdown(rtvPool, dsvPool);
            cbvSrvUavPool.Reset();
            samplerPool.Reset();
            rtvPool.Reset();
            dsvPool.Reset();

            deviceContext.Shutdown();
            isInitialized = false;
        }

        BOOL Initialize(const NativeRenderTarget& renderTarget)
        {
            if (isInitialized) return true;

            // 1. Initialize Device Context (discrete GPU adapter, device, command queue, upload sync)
            if (!deviceContext.Initialize())
                return false;

            ID3D12Device* device = deviceContext.GetDevice();
            ID3D12CommandQueue* queue = deviceContext.GetCommandQueue();

            // 2. Initialize Descriptor Pools (1000000 CBV/SRV/UAV for universal space tables)
            if (!cbvSrvUavPool.Initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1000000, true))
            {
                CleanupResources();
                return false;
            }

            if (!samplerPool.Initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048, true))
            {
                CleanupResources();
                return false;
            }

            if (!rtvPool.Initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 256, false))
            {
                CleanupResources();
                return false;
            }

            if (!dsvPool.Initialize(device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 8192, false))
            {
                CleanupResources();
                return false;
            }

            // 3. Initialize SwapChain if window handle provided
            HWND hwnd = reinterpret_cast<HWND>(renderTarget.surface);
            if (hwnd)
            {
                if (!swapChain.Initialize(
                    deviceContext.GetFactory(),
                    queue,
                    device,
                    rtvPool,
                    dsvPool,
                    hwnd,
                    1280,
                    720))
                {
                    CleanupResources();
                    return false;
                }
            }

            // 4. Initialize Command Allocators and Command List
            for (UINT32 i = 0; i < kMaxFramesInFlight; ++i)
            {
                if (FAILED(device->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&frameCmdAlloc[i]))))
                {
                    CleanupResources();
                    return false;
                }
            }

            if (FAILED(device->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                frameCmdAlloc[0].Get(),
                nullptr,
                IID_PPV_ARGS(&frameCmdList))))
            {
                CleanupResources();
                return false;
            }
            frameCmdList->Close();

            // 5. Initialize Pipeline (Root Signature + NSL Lit Default PSO + Instance Upload Buffer)
            if (!pipeline.Initialize(deviceContext))
            {
                CleanupResources();
                return false;
            }

            // 6. Pre-allocate dynamic descriptor table rings and default dummy space table
            for (UINT32 i = 0; i < kMaxFramesInFlight; ++i)
            {
                dynamicTableBase[i] = cbvSrvUavPool.AllocateRange(kRingDescriptorsPerBB);
            }

            defaultDummySpaceTableIndex = cbvSrvUavPool.AllocateRange(kDescriptorsPerSpaceTable);
            if (defaultDummySpaceTableIndex != UINT32_MAX)
            {
                // Dedicated persistent 256-byte zero constant buffer for 16 dummy CBVs (b0..b15)
                // Must NOT be allocated from pipeline's per-frame dynamic CBuffer ring (which resets on BeginFrameBuffers(0)).
                D3D12_HEAP_PROPERTIES zeroCbHeap{};
                zeroCbHeap.Type                 = D3D12_HEAP_TYPE_UPLOAD;
                zeroCbHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
                zeroCbHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
                zeroCbHeap.CreationNodeMask     = 1;
                zeroCbHeap.VisibleNodeMask      = 1;

                D3D12_RESOURCE_DESC zeroCbResDesc = CreateBufferResourceDesc(256);
                if (SUCCEEDED(device->CreateCommittedResource(
                    &zeroCbHeap,
                    D3D12_HEAP_FLAG_NONE,
                    &zeroCbResDesc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    nullptr,
                    IID_PPV_ARGS(&defaultDummyZeroCBuffer))))
                {
                    void* mappedZero = nullptr;
                    D3D12_RANGE noRead{ 0, 0 };
                    if (SUCCEEDED(defaultDummyZeroCBuffer->Map(0, &noRead, &mappedZero)) && mappedZero)
                    {
                        std::memset(mappedZero, 0, 256);
                        defaultDummyZeroCBuffer->Unmap(0, nullptr);
                    }

                    D3D12_CONSTANT_BUFFER_VIEW_DESC dummyCbvDesc{};
                    dummyCbvDesc.BufferLocation = defaultDummyZeroCBuffer->GetGPUVirtualAddress();
                    dummyCbvDesc.SizeInBytes    = 256;
                    for (UINT32 s = 0; s < kCBVPerSpace; ++s)
                    {
                        device->CreateConstantBufferView(&dummyCbvDesc, cbvSrvUavPool.GetCpuStagingHandle(defaultDummySpaceTableIndex + s));
                        device->CreateConstantBufferView(&dummyCbvDesc, cbvSrvUavPool.GetCPUHandle(defaultDummySpaceTableIndex + s));
                    }
                }

                // Hardware Null Texture2D SRVs for 32 fallback SRVs (t0..t31): reads (0,0,0,0) with zero VRAM allocation
                D3D12_SHADER_RESOURCE_VIEW_DESC dummyNullSrvDesc{};
                dummyNullSrvDesc.Format                        = DXGI_FORMAT_R8G8B8A8_UNORM;
                dummyNullSrvDesc.ViewDimension                 = D3D12_SRV_DIMENSION_TEXTURE2D;
                dummyNullSrvDesc.Shader4ComponentMapping       = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                dummyNullSrvDesc.Texture2D.MostDetailedMip     = 0;
                dummyNullSrvDesc.Texture2D.MipLevels           = 1;
                dummyNullSrvDesc.Texture2D.PlaneSlice          = 0;
                dummyNullSrvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
                for (UINT32 s = 0; s < kSRVPerSpace; ++s)
                {
                    const UINT32 srvSlot = defaultDummySpaceTableIndex + kCBVPerSpace + s;
                    device->CreateShaderResourceView(
                        nullptr,
                        &dummyNullSrvDesc,
                        cbvSrvUavPool.GetCpuStagingHandle(srvSlot)
                    );
                    device->CreateShaderResourceView(
                        nullptr,
                        &dummyNullSrvDesc,
                        cbvSrvUavPool.GetCPUHandle(srvSlot)
                    );
                }

                // Dummy Null Buffer UAVs for 16 dummy UAVs (u0..u15 at offset 48..63)
                D3D12_UNORDERED_ACCESS_VIEW_DESC dummyUavDesc{};
                dummyUavDesc.Format                      = DXGI_FORMAT_UNKNOWN;
                dummyUavDesc.ViewDimension               = D3D12_UAV_DIMENSION_BUFFER;
                dummyUavDesc.Buffer.FirstElement         = 0;
                dummyUavDesc.Buffer.NumElements          = 0;
                dummyUavDesc.Buffer.StructureByteStride  = 4;
                dummyUavDesc.Buffer.CounterOffsetInBytes = 0;
                dummyUavDesc.Buffer.Flags                = D3D12_BUFFER_UAV_FLAG_NONE;
                for (UINT32 u = 0; u < kUAVPerSpace; ++u)
                {
                    const UINT32 uavSlot = defaultDummySpaceTableIndex + kCBVPerSpace + kSRVPerSpace + u;
                    device->CreateUnorderedAccessView(
                        nullptr,
                        nullptr,
                        &dummyUavDesc,
                        cbvSrvUavPool.GetCpuStagingHandle(uavSlot)
                    );
                    device->CreateUnorderedAccessView(
                        nullptr,
                        nullptr,
                        &dummyUavDesc,
                        cbvSrvUavPool.GetCPUHandle(uavSlot)
                    );
                }
            }

            // 7. Pre-allocate default dummy sampler table
            defaultDummySamplerTableIndex = samplerPool.AllocateRange(16);
            SamplerDesc defaultSampDesc{};
            defaultSampDesc.minFilter = SamplerFilter::LINEAR;
            defaultSampDesc.magFilter = SamplerFilter::LINEAR;
            defaultSampDesc.mipFilter = SamplerFilter::LINEAR;
            defaultSampDesc.wrapU = SamplerWrap::REPEAT;
            defaultSampDesc.wrapV = SamplerWrap::REPEAT;
            defaultSampDesc.wrapW = SamplerWrap::REPEAT;
            defaultSamplerHandle = resourceMgr.CreateSampler(deviceContext, samplerPool, defaultSampDesc);
            const auto* sampRes = resourceMgr.GetSampler(defaultSamplerHandle);
            if (sampRes && sampRes->samplerDescriptorIndex != UINT32_MAX && defaultDummySamplerTableIndex != UINT32_MAX)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE sampSrc = samplerPool.GetCpuStagingHandle(sampRes->samplerDescriptorIndex);
                for (UINT32 s = 0; s < 16; ++s)
                {
                    D3D12_CPU_DESCRIPTOR_HANDLE dst = samplerPool.GetCPUHandle(defaultDummySamplerTableIndex + s);
                    device->CopyDescriptorsSimple(1, dst, sampSrc, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
                }
            }

            // 8. Initialize GPU Timestamp Query Heap & Readback Buffer
            if (queue && SUCCEEDED(queue->GetTimestampFrequency(&gpuTimestampFrequency)) && gpuTimestampFrequency > 0)
            {
                D3D12_QUERY_HEAP_DESC qhDesc{};
                qhDesc.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
                qhDesc.Count    = kQueriesPerBackBuffer * kMaxFramesInFlight;
                qhDesc.NodeMask = 0;
                if (SUCCEEDED(device->CreateQueryHeap(&qhDesc, IID_PPV_ARGS(&timestampQueryHeap))))
                {
                    D3D12_HEAP_PROPERTIES rbHeap{};
                    rbHeap.Type                 = D3D12_HEAP_TYPE_READBACK;
                    rbHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
                    rbHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
                    rbHeap.CreationNodeMask     = 1;
                    rbHeap.VisibleNodeMask      = 1;

                    D3D12_RESOURCE_DESC rbDesc = CreateBufferResourceDesc(
                        sizeof(UINT64) * kQueriesPerBackBuffer * kMaxFramesInFlight);
                    device->CreateCommittedResource(
                        &rbHeap,
                        D3D12_HEAP_FLAG_NONE,
                        &rbDesc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        nullptr,
                        IID_PPV_ARGS(&timestampReadbackBuffer));
                }
            }

            ownerThreadId = std::this_thread::get_id();
            isInitialized = true;
            return true;
        }

        void Shutdown()
        {
            if (!isInitialized) return;
            CleanupResources();
        }
    };

    // ==================== CONSTRUCTOR & DESTRUCTOR ====================
    NorthstarD3D12::NorthstarD3D12()
        : m_impl(new Impl())
    {
    }

    NorthstarD3D12::~NorthstarD3D12()
    {
        Shutdown();
        delete m_impl;
        m_impl = nullptr;
    }

    // ==================== LIFECYCLE ====================
    BOOL NorthstarD3D12::Initialize(const NativeRenderTarget& renderTarget)
    {
        return m_impl ? m_impl->Initialize(renderTarget) : FALSE;
    }

    void NorthstarD3D12::Shutdown()
    {
        if (m_impl) m_impl->Shutdown();
    }

    BOOL NorthstarD3D12::IsInitialized() const
    {
        return (m_impl && m_impl->isInitialized) ? true : false;
    }

    // ==================== FRAME PIPELINE ====================
    void NorthstarD3D12::ExecuteFrame(const FrameView& view)
    {
        if (!m_impl || !m_impl->isInitialized || m_impl->deviceContext.IsDeviceRemoved()) return;
        m_impl->AssertOwnerThread();

        const BOOL isTextureTarget = (view.camera.renderTarget.type == RenderTargetType::TEXTURE);
        D3D12SwapChain& sc = m_impl->swapChain;

        D3D12TextureResource* targetTexture = nullptr;
        D3D12TextureResource* targetDepthTexture = nullptr;
        const NativeHandle activeDepthHandle = (view.camera.renderTarget.depthStencilTexture != 0)
            ? view.camera.renderTarget.depthStencilTexture
            : view.sceneDepthTexture;

        if (isTextureTarget)
        {
            targetTexture = m_impl->resourceMgr.GetTexture(view.camera.renderTarget.colorTexture);
            if (!targetTexture || !targetTexture->resource || targetTexture->rtvDescriptorIndex == UINT32_MAX)
            {
                // target texture is invalid or not an RTV -> safe skip (0 Draw Calls)
                return;
            }
        }
        else
        {
            if (!sc.IsValid()) return;
            if (sc.GetHwnd() && !::IsWindow(sc.GetHwnd())) return;
            if (!sc.GetCurrentBackBuffer()) return;
        }

        if (activeDepthHandle != 0)
        {
            targetDepthTexture = m_impl->resourceMgr.GetTexture(activeDepthHandle);
        }

        const UINT32 bbIndex = static_cast<UINT32>(m_impl->submissionSlotCursor % m_impl->maxFramesInFlight);
        m_impl->submissionSlotCursor++;
        ID3D12Fence* fence = m_impl->deviceContext.GetFrameFence();
        HANDLE fenceEvent = m_impl->deviceContext.GetFrameFenceEvent();

        // 1. Wait for previous execution of this frame-in-flight slot (and swapchain backbuffer if window target)
        const auto fenceWaitStart = std::chrono::high_resolution_clock::now();
        if (m_impl->frameFenceValues[bbIndex] > 0 && m_impl->GetSafeCompletedFenceValue() < m_impl->frameFenceValues[bbIndex])
        {
            if (!m_impl->deviceContext.WaitForFenceValue(fence, fenceEvent, m_impl->frameFenceValues[bbIndex]))
                return;
        }
        if (!isTextureTarget)
        {
            const UINT32 scIdx = sc.GetCurrentBackBufferIndex() % D3D12SwapChain::kBackBufferCount;
            if (m_impl->swapChainFenceValues[scIdx] > 0 && m_impl->GetSafeCompletedFenceValue() < m_impl->swapChainFenceValues[scIdx])
            {
                if (!m_impl->deviceContext.WaitForFenceValue(fence, fenceEvent, m_impl->swapChainFenceValues[scIdx]))
                    return;
            }
        }
        const auto fenceWaitEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.fenceWaitCpuMs = std::chrono::duration<float, std::milli>(fenceWaitEnd - fenceWaitStart).count();

        // Read back GPU timestamps from previous completed frame on this frame-in-flight slot
        m_impl->ResolveTimestampSlot(bbIndex);

        const auto cmdRecordStart = std::chrono::high_resolution_clock::now();
        UINT32 tablesUsedThisFrame = 0;
        m_impl->lastFrameMetrics.indirectDrawCommandsExecuted       = 0;
        m_impl->lastFrameMetrics.indirectDispatchCommandsExecuted   = 0;
        m_impl->lastFrameMetrics.indirectCommandsRejected           = 0;
        m_impl->lastFrameMetrics.layeredDepthPassesExecuted         = 0;
        m_impl->lastFrameMetrics.layeredDepthFallbackPassesExecuted = 0;
        m_impl->lastFrameMetrics.cpuPerSliceDepthClearsExecuted     = 0;

        // Process retired resources that GPU has finished using
        const UINT64 completedFence = m_impl->GetSafeCompletedFenceValue();
        m_impl->resourceMgr.ProcessRetirementQueue(
            completedFence,
            m_impl->cbvSrvUavPool,
            m_impl->rtvPool,
            m_impl->dsvPool,
            m_impl->samplerPool,
            &m_impl->stateTracker
        );
        m_impl->ProcessRetiredSamplerTables(completedFence);

        // 2. Reset allocator and command list
        if (FAILED(m_impl->frameCmdAlloc[bbIndex]->Reset())) return;
        if (FAILED(m_impl->frameCmdList->Reset(m_impl->frameCmdAlloc[bbIndex].Get(), nullptr))) return;
        m_impl->isRecordingCmdList = true;

        const UINT64 upcomingFenceValue = m_impl->deviceContext.GetFrameFenceValue() + 1;
        const UINT32 queryBase = bbIndex * NorthstarD3D12::Impl::kQueriesPerBackBuffer;
        NorthstarD3D12::Impl::TimestampSlotFlags slotFlags{};
        const BOOL hadUpload = m_impl->resourceMgr.HasPendingBufferUploads();

        const UINT32 tFrameStart = m_impl->EmitTimestamp(queryBase, slotFlags);

        // 2.1 Flush queued asynchronous GPUBuffer uploads before any render/compute passes execute
        const auto uploadRecordStart = std::chrono::high_resolution_clock::now();
        m_impl->resourceMgr.FlushPendingGPUBufferUploads(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            upcomingFenceValue
        );
        const auto uploadRecordEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.gpuBufferUploadRecordCpuMs =
            std::chrono::duration<float, std::milli>(uploadRecordEnd - uploadRecordStart).count();

        if (hadUpload)
        {
            const UINT32 tUploadEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
            NorthstarD3D12::Impl::AddTimestampSpan(
                slotFlags,
                tFrameStart,
                tUploadEnd,
                NorthstarD3D12::Impl::TimestampSpanKind::UPLOAD);
        }

        // 3. Viewport & Scissor
        D3D12_VIEWPORT viewport{};
        D3D12_RECT scissor{};

        if (view.camera.viewportWidth > 0.0f && view.camera.viewportHeight > 0.0f)
        {
            viewport.TopLeftX = view.camera.viewportX;
            viewport.TopLeftY = view.camera.viewportY;
            viewport.Width    = view.camera.viewportWidth;
            viewport.Height   = view.camera.viewportHeight;
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;

            scissor.left   = static_cast<LONG>(view.camera.viewportX);
            scissor.top    = static_cast<LONG>(view.camera.viewportY);
            scissor.right  = static_cast<LONG>(view.camera.viewportX + view.camera.viewportWidth);
            scissor.bottom = static_cast<LONG>(view.camera.viewportY + view.camera.viewportHeight);
        }
        else if (isTextureTarget && targetTexture)
        {
            viewport.TopLeftX = 0.0f;
            viewport.TopLeftY = 0.0f;
            viewport.Width    = static_cast<float>(targetTexture->desc.width);
            viewport.Height   = static_cast<float>(targetTexture->desc.height);
            viewport.MinDepth = 0.0f;
            viewport.MaxDepth = 1.0f;

            scissor.left   = 0;
            scissor.top    = 0;
            scissor.right  = static_cast<LONG>(targetTexture->desc.width);
            scissor.bottom = static_cast<LONG>(targetTexture->desc.height);
        }
        else
        {
            viewport = sc.GetViewport();
            scissor  = sc.GetScissorRect();
        }

        m_impl->frameCmdList->RSSetViewports(1, &viewport);
        m_impl->frameCmdList->RSSetScissorRects(1, &scissor);

        // 4. Bind Heaps & Reset Dynamic Upload Rings
        ID3D12DescriptorHeap* heaps[] = {
            m_impl->cbvSrvUavPool.GetHeap(),
            m_impl->samplerPool.GetHeap()
        };
        m_impl->frameCmdList->SetDescriptorHeaps(2, heaps);

        // Reset dynamic upload rings for the active backbuffer partition
        m_impl->pipeline.BeginFrameBuffers(bbIndex, completedFence, upcomingFenceValue);
        m_impl->cachedSharedSBUploads.clear();
        m_impl->pipeline.ResetFramePSOMetrics();

        UINT32 currentTableOffset = 0;
        BOOL isComputeRootSigActive = false;
        ID3D12PipelineState* currentComputePSO = nullptr;
        UINT32 currentComputeRootTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };
        UINT32 currentComputeSamplerTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };

        BOOL sceneDepthCleared = false;
        if (!view.prePassComputeDispatches.empty() &&
            targetDepthTexture && targetDepthTexture->resource && targetDepthTexture->dsvDescriptorIndex != UINT32_MAX)
        {
            if (!m_impl->stateTracker.IsResourceTracked(targetDepthTexture->resource.Get()))
            {
                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, targetDepthTexture);
            }
            if (m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get()) != D3D12_RESOURCE_STATE_DEPTH_WRITE)
            {
                m_impl->stateTracker.TransitionResource(targetDepthTexture->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
            D3D12_CPU_DESCRIPTOR_HANDLE preClearDsv = m_impl->dsvPool.GetCPUHandle(targetDepthTexture->dsvDescriptorIndex);
            m_impl->frameCmdList->ClearDepthStencilView(preClearDsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            sceneDepthCleared = true;
        }

        // 4.1 Execute Pre-Pass Compute Dispatches (grouped by profileGroup)
        {
            UINT32 preBurstStart = UINT32_MAX;
            BOOL   preBurstExec  = false;
            for (SIZE_T c = 0; c < view.prePassComputeDispatches.size(); ++c)
            {
                if (preBurstStart == UINT32_MAX)
                {
                    preBurstStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                    preBurstExec  = false;
                }
                if (m_impl->RecordComputeDispatch(
                    view.prePassComputeDispatches[c],
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentComputePSO,
                    currentComputeRootTable,
                    currentComputeSamplerTable
                ))
                {
                    preBurstExec = true;
                }
                const BOOL sameNextGroup =
                    (c + 1 < view.prePassComputeDispatches.size()) &&
                    (view.prePassComputeDispatches[c + 1].profileGroup == view.prePassComputeDispatches[c].profileGroup);
                if (!sameNextGroup)
                {
                    const UINT32 preBurstEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                    if (preBurstExec)
                    {
                        NorthstarD3D12::Impl::AddTimestampSpan(
                            slotFlags,
                            preBurstStart,
                            preBurstEnd,
                            NorthstarD3D12::Impl::MapProfileGroupToSpanKind(
                                view.prePassComputeDispatches[c].profileGroup,
                                NorthstarD3D12::Impl::TimestampSpanKind::PRE_COMPUTE));
                    }
                    preBurstStart = UINT32_MAX;
                }
            }
        }

        // Bind Graphics Root Signature & default dummy fallback tables
        ID3D12RootSignature* currentGraphicsRootSig = nullptr;
        auto bindGraphicsRootSignatureAndFallbacks = [&]() {
            m_impl->BindGraphicsRootSigAndFallbacks();
            currentGraphicsRootSig = m_impl->pipeline.GetRootSignature();
            isComputeRootSigActive = false;
        };

        bindGraphicsRootSignatureAndFallbacks();

        // 5. Transition Target to RENDER_TARGET state & SceneDepth to DEPTH_WRITE state
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle{};
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle{};
        D3D12_CPU_DESCRIPTOR_HANDLE readOnlyDsvHandle{};
        BOOL hasDepth = false;
        BOOL hasReadOnlyDsv = false;
        BOOL isReadOnlyDepthBound = false;

        if (isTextureTarget)
        {
            if (!m_impl->stateTracker.IsResourceTracked(targetTexture->resource.Get()))
            {
                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, targetTexture);
            }
            if (m_impl->stateTracker.GetResourceState(targetTexture->resource.Get()) != D3D12_RESOURCE_STATE_RENDER_TARGET)
            {
                m_impl->stateTracker.TransitionResource(targetTexture->resource.Get(), GPUResourceAccess::RENDER_TARGET);
                m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            targetTexture->currentState = m_impl->stateTracker.GetResourceState(targetTexture->resource.Get());

            rtvHandle = m_impl->rtvPool.GetCPUHandle(targetTexture->rtvDescriptorIndex);
        }
        else
        {
            D3D12_RESOURCE_BARRIER toRenderTarget = CreateTransitionBarrier(
                sc.GetCurrentBackBuffer(),
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_RENDER_TARGET
            );
            m_impl->frameCmdList->ResourceBarrier(1, &toRenderTarget);

            rtvHandle = sc.GetCurrentRtvHandle(m_impl->rtvPool);
        }

        if (targetDepthTexture && targetDepthTexture->resource && targetDepthTexture->dsvDescriptorIndex != UINT32_MAX)
        {
            if (!m_impl->stateTracker.IsResourceTracked(targetDepthTexture->resource.Get()))
            {
                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, targetDepthTexture);
            }
            if (m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get()) != D3D12_RESOURCE_STATE_DEPTH_WRITE)
            {
                m_impl->stateTracker.TransitionResource(targetDepthTexture->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
            hasDepth = true;
            dsvHandle = m_impl->dsvPool.GetCPUHandle(targetDepthTexture->dsvDescriptorIndex);
            if (targetDepthTexture->readOnlyDsvDescriptorIndex != UINT32_MAX)
            {
                hasReadOnlyDsv = true;
                readOnlyDsvHandle = m_impl->dsvPool.GetCPUHandle(targetDepthTexture->readOnlyDsvDescriptorIndex);
            }
        }
        else if (!isTextureTarget && sc.HasDepthBuffer())
        {
            hasDepth = true;
            dsvHandle = sc.GetDsvHandle(m_impl->dsvPool);
        }

        // 6. Clear Render Target with Scene Ambient Color & Depth Stencil
        const float clearColor[4] = {
            view.environment.ambientColor.r,
            view.environment.ambientColor.g,
            view.environment.ambientColor.b,
            view.environment.ambientColor.a > 0.0f ? view.environment.ambientColor.a : 1.0f
        };
        D3D12_RECT clearRect{};
        const D3D12_RECT* pClearRect = nullptr;
        UINT numClearRects = 0;
        if (view.camera.viewportWidth > 0.0f && view.camera.viewportHeight > 0.0f)
        {
            const LONG maxRtvW = (isTextureTarget && targetTexture)
                ? static_cast<LONG>(targetTexture->desc.width)
                : static_cast<LONG>(sc.GetViewport().Width);
            const LONG maxRtvH = (isTextureTarget && targetTexture)
                ? static_cast<LONG>(targetTexture->desc.height)
                : static_cast<LONG>(sc.GetViewport().Height);
            clearRect.left   = std::max<LONG>(0, scissor.left);
            clearRect.top    = std::max<LONG>(0, scissor.top);
            clearRect.right  = std::min<LONG>(maxRtvW, scissor.right);
            clearRect.bottom = std::min<LONG>(maxRtvH, scissor.bottom);
            if (clearRect.right > clearRect.left && clearRect.bottom > clearRect.top)
            {
                pClearRect = &clearRect;
                numClearRects = 1;
            }
        }
        m_impl->frameCmdList->ClearRenderTargetView(rtvHandle, clearColor, numClearRects, pClearRect);

        if (hasDepth)
        {
            if (targetDepthTexture)
            {
                if (!sceneDepthCleared)
                {
                    // Clear entire camera-owned SceneDepth texture to 1.0f
                    m_impl->frameCmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
                    sceneDepthCleared = true;
                }
            }
            else
            {
                m_impl->frameCmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, numClearRects, pClearRect);
            }
            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
        }
        else
        {
            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        }

        // 7. Render Commands
        m_impl->frameCmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        ID3D12PipelineState* currentPSO = nullptr;
        NativeHandle currentVB = 0;
        NativeHandle currentIB = 0;
        UINT32 currentRootTable[D3D12Pipeline::kGraphicsRootParamCount];
        for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
        UINT32 currentSamplerTableIndex[5][NorthstarD3D12::Impl::kMaxSpaces]{};
        m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
        const UINT32 defaultDummyUavTableIndex = (m_impl->defaultDummySpaceTableIndex != UINT32_MAX)
            ? (m_impl->defaultDummySpaceTableIndex + NorthstarD3D12::Impl::kCBVPerSpace + NorthstarD3D12::Impl::kSRVPerSpace)
            : UINT32_MAX;
        auto resetGraphicsUavCache = [&](UINT32 outUavCache[NorthstarD3D12::Impl::kMaxSpaces]) {
            for (UINT32 sp = 0u; sp < NorthstarD3D12::Impl::kMaxSpaces; ++sp)
            {
                outUavCache[sp] = defaultDummyUavTableIndex;
            }
        };
        UINT32 currentGraphicsUavTable[NorthstarD3D12::Impl::kMaxSpaces]{};
        resetGraphicsUavCache(currentGraphicsUavTable);

        struct CachedSharedInstAlloc
        {
            ID3D12Resource* resource     = nullptr;
            UINT32          firstElement = 0;
        };
        std::unordered_map<UINT64, CachedSharedInstAlloc>     cachedSharedInstFirstElem;
        std::unordered_map<UINT64, D3D12_GPU_VIRTUAL_ADDRESS> cachedSharedCbVA;
        std::unordered_map<UINT64, UINT32>                    cachedSharedSpaceTable;

        auto recordDrawList = [&](const std::vector<DrawCommand>& cmdList, BOOL isShadowPass = false, BOOL isDepthPrepass = false) {
            if (cmdList.empty()) return;
            D3D_PRIMITIVE_TOPOLOGY currentTopology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            m_impl->frameCmdList->IASetPrimitiveTopology(currentTopology);
            for (SIZE_T i = 0; i < cmdList.size(); ++i)
            {
                const auto& cmd = cmdList[i];
                const BOOL isPromotedIndirect   = (cmd.useNVSIndirectCommandSignature != FALSE) && (cmd.indirectArgsBuffer != 0);
                const BOOL isVertexPullIndirect = (cmd.useVertexPullingIndirect != FALSE) && (cmd.indirectArgsBuffer != 0) && !isPromotedIndirect;
                const BOOL skipsIaBuffers       = isPromotedIndirect || isVertexPullIndirect;

                // Resolve Vertex Buffer (Cached or Direct)
                NativeHandle vbHandle = cmd.vertexBuffer;
                if (!skipsIaBuffers && vbHandle == 0 && cmd.cpuVertices != nullptr && cmd.vertexCount > 0)
                {
                    vbHandle = m_impl->resourceMgr.GetOrCreateCachedCpuVertexBuffer(
                        m_impl->deviceContext, cmd.cpuVertices, cmd.vertexCount, sizeof(Vertex));
                }

                const auto* pVb = skipsIaBuffers ? nullptr : m_impl->resourceMgr.GetVertexBuffer(vbHandle);
                if (!skipsIaBuffers && !pVb)
                {
                    if (cmd.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                    continue;
                }

                // Resolve Index Buffer (Cached or Direct)
                NativeHandle ibHandle = cmd.indexBuffer;
                if (!skipsIaBuffers && ibHandle == 0 && cmd.cpuIndices != nullptr && cmd.indexCount > 0)
                {
                    ibHandle = m_impl->resourceMgr.GetOrCreateCachedCpuIndexBuffer(
                        m_impl->deviceContext, cmd.cpuIndices, cmd.indexCount, true);
                }

                const auto* pIb = skipsIaBuffers ? nullptr : m_impl->resourceMgr.GetIndexBuffer(ibHandle);
                const BOOL hasIndices = isPromotedIndirect ? true : (isVertexPullIndirect ? false : (pIb != nullptr));

                // Validate GPU-Driven Indirect Arguments & Count Buffer
                D3D12GPUBufferResource* indirectArgBuf   = nullptr;
                D3D12GPUBufferResource* indirectCountBuf = nullptr;
                ID3D12CommandSignature* indirectCmdSig   = nullptr;

                if (cmd.indirectArgsBuffer != 0)
                {
                    const UINT32 argStride = isPromotedIndirect
                        ? D3D12Pipeline::kNVSIndirectCommandByteStride
                        : (hasIndices ? sizeof(GPUIndirectDrawIndexedArgs) : sizeof(GPUIndirectDrawArgs));
                    const UINT32 reqAlign = isPromotedIndirect ? 8u : 4u;
                    indirectCmdSig = isPromotedIndirect
                        ? m_impl->pipeline.GetOrCreateIndirectCommandSignature(m_impl->deviceContext.GetDevice(), cmd.indirectRootKey)
                        : (hasIndices ? m_impl->pipeline.GetDrawIndexedIndirectSignature() : m_impl->pipeline.GetDrawIndirectSignature());
                    indirectArgBuf = m_impl->resourceMgr.GetGPUBuffer(cmd.indirectArgsBuffer);

                    if (cmd.maxIndirectCommandCount == 0 ||
                        !indirectCmdSig ||
                        !indirectArgBuf || !indirectArgBuf->resource ||
                        !HasGPUBufferUsage(indirectArgBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                        (cmd.indirectArgsOffset % reqAlign) != 0u ||
                        static_cast<UINT64>(cmd.indirectArgsOffset) + static_cast<UINT64>(cmd.maxIndirectCommandCount) * static_cast<UINT64>(argStride) > static_cast<UINT64>(indirectArgBuf->byteSize))
                    {
                        m_impl->lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }

                    if (cmd.indirectCountBuffer != 0)
                    {
                        indirectCountBuf = m_impl->resourceMgr.GetGPUBuffer(cmd.indirectCountBuffer);
                        if (!indirectCountBuf || !indirectCountBuf->resource ||
                            !HasGPUBufferUsage(indirectCountBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                            (cmd.indirectCountBufferOffset % 4u) != 0u ||
                            static_cast<UINT64>(cmd.indirectCountBufferOffset) + sizeof(UINT32) > static_cast<UINT64>(indirectCountBuf->byteSize))
                        {
                            m_impl->lastFrameMetrics.indirectCommandsRejected++;
                            continue;
                        }

                        if (cmd.indirectCountBuffer == cmd.indirectArgsBuffer)
                        {
                            const UINT64 argBegin   = static_cast<UINT64>(cmd.indirectArgsOffset);
                            const UINT64 argEnd     = argBegin + static_cast<UINT64>(cmd.maxIndirectCommandCount) * static_cast<UINT64>(argStride);
                            const UINT64 countBegin = static_cast<UINT64>(cmd.indirectCountBufferOffset);
                            const UINT64 countEnd   = countBegin + sizeof(UINT32);
                            if (countBegin < argEnd && argBegin < countEnd)
                            {
                                m_impl->lastFrameMetrics.indirectCommandsRejected++;
                                continue;
                            }
                        }
                    }
                }

                // Validate optional GPU-resident Instance Buffer (cmd.instanceBuffer)
                D3D12GPUBufferResource* gpuInstanceBuf = nullptr;
                if (cmd.instanceBuffer != 0)
                {
                    if (cmd.instanceBuffer == cmd.indirectArgsBuffer || cmd.instanceBuffer == cmd.indirectCountBuffer)
                    {
                        if (cmd.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }
                    gpuInstanceBuf = m_impl->resourceMgr.GetGPUBuffer(cmd.instanceBuffer);
                    if (!gpuInstanceBuf || !gpuInstanceBuf->resource ||
                        !HasGPUBufferUsage(gpuInstanceBuf->desc.usage, GPUBufferUsage::STRUCTURED_SRV) ||
                        gpuInstanceBuf->srvDescriptorIndex == UINT32_MAX)
                    {
                        if (cmd.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                        continue;
                    }
                }

                DepthState effectiveDepthState = cmd.depthState;
                if (!isShadowPass && !isDepthPrepass && hasDepth && targetDepthTexture)
                {
                    BOOL cmdSamplesActiveSceneDepth = false;
                    if (activeDepthHandle != 0)
                    {
                        for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                        {
                            if (cmd.textures[t].texture == activeDepthHandle)
                            {
                                cmdSamplesActiveSceneDepth = true;
                                break;
                            }
                        }
                    }
                    if (cmdSamplesActiveSceneDepth)
                    {
                        effectiveDepthState.write = false;
                    }

                    const D3D12_RESOURCE_STATES curDepthState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                    if (!effectiveDepthState.write && hasReadOnlyDsv)
                    {
                        const D3D12_RESOURCE_STATES reqRoState =
                            D3D12_RESOURCE_STATE_DEPTH_READ |
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        if (!isReadOnlyDepthBound || (curDepthState & reqRoState) != reqRoState)
                        {
                            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
                            m_impl->stateTracker.TransitionResource(
                                targetDepthTexture->resource.Get(),
                                GPUResourceAccess::DEPTH_READ | GPUResourceAccess::ALL_SHADER_RESOURCE
                            );
                            m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &readOnlyDsvHandle);
                            isReadOnlyDepthBound = true;
                        }
                    }
                    else
                    {
                        if (isReadOnlyDepthBound || curDepthState != D3D12_RESOURCE_STATE_DEPTH_WRITE)
                        {
                            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
                            m_impl->stateTracker.TransitionResource(
                                targetDepthTexture->resource.Get(),
                                GPUResourceAccess::DEPTH_WRITE
                            );
                            m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
                            isReadOnlyDepthBound = false;
                        }
                    }
                }

                // Switch Root Signature dynamically when entering/exiting promoted indirect draw commands
                ID3D12RootSignature* reqRootSig = cmd.indirectRootKey.enabled
                    ? m_impl->pipeline.GetOrCreateIndirectRootSignature(m_impl->deviceContext.GetDevice(), cmd.indirectRootKey)
                    : m_impl->pipeline.GetRootSignature();
                if (!reqRootSig)
                {
                    if (cmd.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                    continue;
                }
                if (reqRootSig != currentGraphicsRootSig || isComputeRootSigActive)
                {
                    m_impl->BindGraphicsRootSigAndFallbacks(reqRootSig);
                    for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                    m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
                    resetGraphicsUavCache(currentGraphicsUavTable);
                    currentGraphicsRootSig = reqRootSig;
                    isComputeRootSigActive = false;
                    currentPSO             = nullptr;
                }

                // Bind PSO dynamically from DrawCommand shader handles & active RTV / DSV formats
                const DXGI_FORMAT activeRtvFormat = (isTextureTarget && targetTexture)
                    ? targetTexture->rtvFormat
                    : DXGI_FORMAT_R8G8B8A8_UNORM;
                const DXGI_FORMAT activeDsvFormat = !hasDepth
                    ? DXGI_FORMAT_UNKNOWN
                    : (targetDepthTexture ? targetDepthTexture->dsvFormat : DXGI_FORMAT_D32_FLOAT);
                ID3D12PipelineState* pso = m_impl->pipeline.GetOrCreatePSO(
                    m_impl->deviceContext.GetDevice(),
                    m_impl->resourceMgr,
                    cmd.vertexShader,
                    cmd.pixelShader,
                    cmd.cullMode,
                    cmd.blendMode,
                    cmd.fillMode,
                    effectiveDepthState,
                    isShadowPass,
                    0,
                    0.0f,
                    0.0f,
                    cmd.geometryShader,
                    isShadowPass || isDepthPrepass,
                    cmd.hullShader,
                    cmd.domainShader,
                    cmd.indirectRootKey,
                    activeRtvFormat,
                    activeDsvFormat
                );

                if (!pso)
                {
                    if (cmd.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                    continue;
                }

                if (pso != currentPSO)
                {
                    m_impl->frameCmdList->SetPipelineState(pso);
                    currentPSO        = pso;
                    currentComputePSO = nullptr;
                }

                const D3D_PRIMITIVE_TOPOLOGY reqTopology = (cmd.hullShader != 0 && cmd.domainShader != 0)
                    ? D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST
                    : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
                if (reqTopology != currentTopology)
                {
                    m_impl->frameCmdList->IASetPrimitiveTopology(reqTopology);
                    currentTopology = reqTopology;
                }

                if (!skipsIaBuffers)
                {
                    if (vbHandle != currentVB)
                    {
                        m_impl->frameCmdList->IASetVertexBuffers(0, 1, &pVb->view);
                        currentVB = vbHandle;
                    }

                    if (hasIndices && ibHandle != currentIB)
                    {
                        m_impl->frameCmdList->IASetIndexBuffer(&pIb->view);
                        currentIB = ibHandle;
                    }
                }

                // Transition GPU-resident indirect pools, instance buffer, and indirect argument/count buffers
                BOOL anyIndirectTransition = false;
                auto transitionPoolBuf = [&](NativeHandle hBuf, GPUResourceAccess acc) {
                    if (hBuf == 0) return;
                    D3D12GPUBufferResource* res = m_impl->resourceMgr.GetGPUBuffer(hBuf);
                    if (res && res->resource)
                    {
                        m_impl->stateTracker.TransitionResource(res->resource.Get(), acc);
                        res->currentState = m_impl->stateTracker.GetResourceState(res->resource.Get());
                        anyIndirectTransition = true;
                    }
                };
                transitionPoolBuf(cmd.indirectPerDrawCBVBuffer,       GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                transitionPoolBuf(cmd.indirectPerBatchInstanceBuffer, GPUResourceAccess::ALL_SHADER_RESOURCE);
                transitionPoolBuf(cmd.indirectBonePaletteBuffer,      GPUResourceAccess::ALL_SHADER_RESOURCE);
                transitionPoolBuf(cmd.indirectVertexPoolBuffer,       GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER);
                transitionPoolBuf(cmd.indirectIndexPoolBuffer,        GPUResourceAccess::INDEX_BUFFER);

                if (gpuInstanceBuf != nullptr)
                {
                    m_impl->stateTracker.TransitionResource(gpuInstanceBuf->resource.Get(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                    gpuInstanceBuf->currentState = m_impl->stateTracker.GetResourceState(gpuInstanceBuf->resource.Get());
                    anyIndirectTransition = true;
                }
                if (indirectArgBuf != nullptr)
                {
                    m_impl->stateTracker.TransitionResource(indirectArgBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                    indirectArgBuf->currentState = m_impl->stateTracker.GetResourceState(indirectArgBuf->resource.Get());
                    anyIndirectTransition = true;
                }
                if (indirectCountBuf != nullptr)
                {
                    m_impl->stateTracker.TransitionResource(indirectCountBuf->resource.Get(), GPUResourceAccess::INDIRECT_ARGUMENT);
                    indirectCountBuf->currentState = m_impl->stateTracker.GetResourceState(indirectCountBuf->resource.Get());
                    anyIndirectTransition = true;
                }
                if (anyIndirectTransition)
                {
                    m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                }

                // Check if Vertex Space 0 CB0 (Transforms) is provided by DrawCommand
                BOOL hasVertexSpace0CB0 = false;
                for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                {
                    if (cmd.constantBuffers[b].stage == ShaderStage::VERTEX &&
                        cmd.constantBuffers[b].registerSpace == 0 &&
                        cmd.constantBuffers[b].slotIndex == 0)
                    {
                        hasVertexSpace0CB0 = true;
                        break;
                    }
                }

                UINT32 instFirstElement = 0;
                ID3D12Resource* instBufferResource = nullptr;
                const BOOL isInstanced = (gpuInstanceBuf == nullptr) &&
                    ((cmd.instanceCount > 1 || cmd.bufferCategory == BufferCategory::INSTANCE) && !cmd.instances.empty());
                if (isInstanced)
                {
                    if (cmd.sharedInstanceBatchId != 0)
                    {
                        auto it = cachedSharedInstFirstElem.find(cmd.sharedInstanceBatchId);
                        if (it != cachedSharedInstFirstElem.end() &&
                            it->second.resource != nullptr &&
                            it->second.resource == m_impl->pipeline.GetInstanceUploadBuffer())
                        {
                            instFirstElement   = it->second.firstElement;
                            instBufferResource = it->second.resource;
                        }
                        else
                        {
                            instFirstElement = m_impl->pipeline.AllocateInstances(
                                reinterpret_cast<const D3D12InstancePayload*>(cmd.instances.data()),
                                static_cast<UINT32>(cmd.instances.size()));
                            instBufferResource = m_impl->pipeline.GetInstanceUploadBuffer();
                            if (instBufferResource)
                            {
                                cachedSharedInstFirstElem[cmd.sharedInstanceBatchId] = { instBufferResource, instFirstElement };
                            }
                        }
                    }
                    else
                    {
                        instFirstElement = m_impl->pipeline.AllocateInstances(
                            reinterpret_cast<const D3D12InstancePayload*>(cmd.instances.data()),
                            static_cast<UINT32>(cmd.instances.size()));
                        instBufferResource = m_impl->pipeline.GetInstanceUploadBuffer();
                    }
                }

                // Universal Space-Based Descriptor Table Binding (NSL Reflection-Driven)
                // Vertex Spaces 0..3   -> Root Params 0..3
                // Pixel Spaces 0..3    -> Root Params 4..7
                // Geometry Spaces 0..3 -> Root Params 13..16
                // Hull Spaces 0..3     -> Root Params 22..25
                // Domain Spaces 0..3   -> Root Params 26..29
                ID3D12Device* device = m_impl->deviceContext.GetDevice();

                static constexpr ShaderStage kGraphicsStageOrder[5] = {
                    ShaderStage::VERTEX,
                    ShaderStage::PIXEL,
                    ShaderStage::GEOMETRY,
                    ShaderStage::HULL,
                    ShaderStage::DOMAIN
                };

                BOOL drawDescriptorRingExhausted = false;
                BOOL drawQueuedResourceBarriers  = false;
                for (UINT32 stageIdx = 0; stageIdx < 5 && !drawDescriptorRingExhausted; ++stageIdx)
                {
                    const ShaderStage stage = kGraphicsStageOrder[stageIdx];
                    if (stage == ShaderStage::PIXEL && cmd.pixelShader == 0)
                    {
                        continue; // Depth-only pass (e.g. Shadow Pass) has no Pixel Shader bound
                    }
                    if (stage == ShaderStage::GEOMETRY && cmd.geometryShader == 0)
                    {
                        continue; // No Geometry Shader bound
                    }
                    if (stage == ShaderStage::HULL && cmd.hullShader == 0)
                    {
                        continue; // No Hull Shader bound
                    }
                    if (stage == ShaderStage::DOMAIN && cmd.domainShader == 0)
                    {
                        continue; // No Domain Shader bound
                    }

                    // Precompute active spaces and shared CB cache eligibility in a single pass over bindings for this stage
                    BOOL   spaceActive[NorthstarD3D12::Impl::kMaxSpaces]        = { false, false, false, false };
                    UINT32 cbCountInSpace[NorthstarD3D12::Impl::kMaxSpaces]     = { 0, 0, 0, 0 };
                    UINT64 singleSharedCbId[NorthstarD3D12::Impl::kMaxSpaces]   = { 0, 0, 0, 0 };
                    BOOL   allCbSharedInSpace[NorthstarD3D12::Impl::kMaxSpaces] = { true, true, true, true };
                    BOOL   hasSbOrTexInSpace[NorthstarD3D12::Impl::kMaxSpaces]  = { false, false, false, false };

                    if (stage == ShaderStage::VERTEX)
                    {
                        spaceActive[0]        = true; // Always active for Transforms fallback or Instancing
                        allCbSharedInSpace[0] = false;
                        if (cmd.hasReflectedInstanceBinding &&
                            (isInstanced || gpuInstanceBuf != nullptr) &&
                            cmd.instanceBindingSpace < NorthstarD3D12::Impl::kMaxSpaces &&
                            cmd.instanceBindingSlot < NorthstarD3D12::Impl::kSRVPerSpace)
                        {
                            spaceActive[cmd.instanceBindingSpace]       = true;
                            hasSbOrTexInSpace[cmd.instanceBindingSpace] = true;
                        }
                    }

                    for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                    {
                        const auto& cb = cmd.constantBuffers[b];
                        if (cb.stage == stage && cb.registerSpace < NorthstarD3D12::Impl::kMaxSpaces)
                        {
                            const UINT32 sp = cb.registerSpace;
                            spaceActive[sp] = true;
                            cbCountInSpace[sp]++;
                            if (cb.sharedDataId == 0)
                            {
                                allCbSharedInSpace[sp] = false;
                            }
                            else
                            {
                                singleSharedCbId[sp] = cb.sharedDataId;
                            }
                        }
                    }

                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                    {
                        const auto& sb = cmd.structuredBuffers[s];
                        if (sb.isUAV) continue;
                        if (sb.stage == stage && sb.registerSpace < NorthstarD3D12::Impl::kMaxSpaces)
                        {
                            const UINT32 sp = sb.registerSpace;
                            spaceActive[sp]       = true;
                            hasSbOrTexInSpace[sp] = true;
                        }
                    }

                    for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                    {
                        const auto& tex = cmd.textures[t];
                        if (tex.stage == stage && tex.registerSpace < NorthstarD3D12::Impl::kMaxSpaces)
                        {
                            const UINT32 sp = tex.registerSpace;
                            spaceActive[sp]       = true;
                            hasSbOrTexInSpace[sp] = true;
                        }
                    }

                    for (UINT32 space = 0; space < NorthstarD3D12::Impl::kMaxSpaces; ++space)
                    {
                        if (!spaceActive[space]) continue;

                        const UINT32 rootParam = (stage == ShaderStage::VERTEX)
                            ? (D3D12Pipeline::kGraphicsVertexSpaceParamBase + space)
                            : ((stage == ShaderStage::PIXEL)
                                ? (D3D12Pipeline::kGraphicsPixelSpaceParamBase + space)
                                : ((stage == ShaderStage::GEOMETRY)
                                    ? (D3D12Pipeline::kGraphicsGeometrySpaceParamBase + space)
                                    : ((stage == ShaderStage::HULL)
                                        ? (D3D12Pipeline::kGraphicsHullSpaceParamBase + space)
                                        : (D3D12Pipeline::kGraphicsDomainSpaceParamBase + space))));

                        // Fast path: Reuse descriptor table across tiles when a non-0 space only binds a shared ConstantBuffer
                        const BOOL canCacheSpaceTable = !(stage == ShaderStage::VERTEX && space == 0) &&
                                                        !hasSbOrTexInSpace[space] &&
                                                        (cbCountInSpace[space] == 1) &&
                                                        allCbSharedInSpace[space] &&
                                                        (singleSharedCbId[space] != 0);
                        const UINT64 spaceCacheKey = (static_cast<UINT64>(stageIdx) << 60) |
                                                     (static_cast<UINT64>(space) << 56) |
                                                     singleSharedCbId[space];
                        if (canCacheSpaceTable)
                        {
                            auto itTable = cachedSharedSpaceTable.find(spaceCacheKey);
                            if (itTable != cachedSharedSpaceTable.end())
                            {
                                const UINT32 cachedTableStart = itTable->second;
                                if (currentRootTable[rootParam] != cachedTableStart)
                                {
                                    m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                        rootParam,
                                        m_impl->cbvSrvUavPool.GetGPUHandle(cachedTableStart)
                                    );
                                    currentRootTable[rootParam] = cachedTableStart;
                                }
                                continue;
                            }
                        }

                        if (tablesUsedThisFrame >= NorthstarD3D12::Impl::kTablesPerBackBuffer ||
                            currentTableOffset + NorthstarD3D12::Impl::kGraphicsDescPerSpace > NorthstarD3D12::Impl::kRingDescriptorsPerBB)
                        {
                            drawDescriptorRingExhausted = true;
                            break;
                        }
                        const UINT32 tableStart = m_impl->dynamicTableBase[bbIndex] + currentTableOffset;
                        currentTableOffset += NorthstarD3D12::Impl::kGraphicsDescPerSpace;
                        tablesUsedThisFrame++;

                        // 1. Initialize graphics table with safe dummy fallbacks (16 dummy CBVs + 32 dummy SRVs) from CPU staging heap
                        if (m_impl->defaultDummySpaceTableIndex != UINT32_MAX)
                        {
                            device->CopyDescriptorsSimple(
                                NorthstarD3D12::Impl::kGraphicsDescPerSpace,
                                m_impl->cbvSrvUavPool.GetCPUHandle(tableStart),
                                m_impl->cbvSrvUavPool.GetCpuStagingHandle(m_impl->defaultDummySpaceTableIndex),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                            );
                        }

                        // 2. Vertex Space 0 Default Fallbacks (Transforms & Instancing)
                        if (stage == ShaderStage::VERTEX && space == 0)
                        {
                            if (!hasVertexSpace0CB0)
                            {
                                struct TransformsFallback {
                                    Mat4  world;
                                    float normalMat[12];
                                    Mat4  viewProj;
                                } fallback{};
                                fallback.world = cmd.worldMatrix;
                                {
                                    const Vec3 c0(cmd.worldMatrix.m[0], cmd.worldMatrix.m[1], cmd.worldMatrix.m[2]);
                                    const Vec3 c1(cmd.worldMatrix.m[4], cmd.worldMatrix.m[5], cmd.worldMatrix.m[6]);
                                    const Vec3 c2(cmd.worldMatrix.m[8], cmd.worldMatrix.m[9], cmd.worldMatrix.m[10]);
                                    const Vec3 cof0 = c1.Cross(c2);
                                    const Vec3 cof1 = c2.Cross(c0);
                                    const Vec3 cof2 = c0.Cross(c1);
                                    const FLOAT32 det = c0.Dot(cof0);
                                    if (std::isfinite(det) && std::fabs(det) > 1e-12f)
                                    {
                                        const FLOAT32 invDet = 1.0f / det;
                                        fallback.normalMat[0]  = cof0.x * invDet; fallback.normalMat[1]  = cof0.y * invDet; fallback.normalMat[2]  = cof0.z * invDet; fallback.normalMat[3]  = 0.0f;
                                        fallback.normalMat[4]  = cof1.x * invDet; fallback.normalMat[5]  = cof1.y * invDet; fallback.normalMat[6]  = cof1.z * invDet; fallback.normalMat[7]  = 0.0f;
                                        fallback.normalMat[8]  = cof2.x * invDet; fallback.normalMat[9]  = cof2.y * invDet; fallback.normalMat[10] = cof2.z * invDet; fallback.normalMat[11] = 0.0f;
                                    }
                                    else
                                    {
                                        fallback.normalMat[0] = 1.0f; fallback.normalMat[5] = 1.0f; fallback.normalMat[10] = 1.0f;
                                    }
                                }
                                fallback.viewProj = view.camera.viewProjectionMatrix;

                                D3D12_GPU_VIRTUAL_ADDRESS va = m_impl->pipeline.AllocateCBuffer(&fallback, sizeof(fallback));
                                if (va != 0)
                                {
                                    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                    cbvDesc.BufferLocation = va;
                                    cbvDesc.SizeInBytes = (sizeof(fallback) + 255) & ~255;
                                    device->CreateConstantBufferView(
                                        &cbvDesc,
                                        m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + 0)
                                    );
                                }
                            }

                            if (!cmd.hasReflectedInstanceBinding)
                            {
                                auto hasVertexSpace0SBAtSlot = [&](UINT16 slotIdx) -> BOOL {
                                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                                    {
                                        const auto& sb = cmd.structuredBuffers[s];
                                        if (sb.stage == ShaderStage::VERTEX && sb.registerSpace == 0 && sb.slotIndex == slotIdx)
                                            return true;
                                    }
                                    return false;
                                };
                                if (gpuInstanceBuf != nullptr)
                                {
                                    D3D12_CPU_DESCRIPTOR_HANDLE instSrc = m_impl->cbvSrvUavPool.GetCpuStagingHandle(gpuInstanceBuf->srvDescriptorIndex);
                                    if (!hasVertexSpace0SBAtSlot(0))
                                    {
                                        device->CopyDescriptorsSimple(1, m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + 0), instSrc, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                    }
                                    if (!hasVertexSpace0SBAtSlot(1))
                                    {
                                        device->CopyDescriptorsSimple(1, m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + 1), instSrc, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                    }
                                }
                                else if (isInstanced && instBufferResource)
                                {
                                    D3D12_SHADER_RESOURCE_VIEW_DESC instSrvDesc{};
                                    instSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                    instSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                    instSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    instSrvDesc.Buffer.FirstElement = instFirstElement;
                                    instSrvDesc.Buffer.NumElements = static_cast<UINT>(cmd.instances.size());
                                    instSrvDesc.Buffer.StructureByteStride = sizeof(D3D12InstancePayload);
                                    instSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

                                    if (!hasVertexSpace0SBAtSlot(0))
                                    {
                                        device->CreateShaderResourceView(instBufferResource, &instSrvDesc, m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + 0));
                                    }
                                    if (!hasVertexSpace0SBAtSlot(1))
                                    {
                                        device->CreateShaderResourceView(instBufferResource, &instSrvDesc, m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + 1));
                                    }
                                }
                            }
                        }

                        // 2b. Reflected Tier-1 GPU Instancing SRV bound strictly at its reflected (space, slot)
                        if (stage == ShaderStage::VERTEX &&
                            cmd.hasReflectedInstanceBinding &&
                            cmd.instanceBindingSpace == space &&
                            cmd.instanceBindingSlot < NorthstarD3D12::Impl::kSRVPerSpace)
                        {
                            if (gpuInstanceBuf != nullptr)
                            {
                                D3D12_CPU_DESCRIPTOR_HANDLE instSrc = m_impl->cbvSrvUavPool.GetCpuStagingHandle(gpuInstanceBuf->srvDescriptorIndex);
                                device->CopyDescriptorsSimple(
                                    1,
                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + cmd.instanceBindingSlot),
                                    instSrc,
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                );
                            }
                            else if (isInstanced && instBufferResource)
                            {
                                D3D12_SHADER_RESOURCE_VIEW_DESC instSrvDesc{};
                                instSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                instSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                instSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                instSrvDesc.Buffer.FirstElement = instFirstElement;
                                instSrvDesc.Buffer.NumElements = static_cast<UINT>(cmd.instances.size());
                                instSrvDesc.Buffer.StructureByteStride = sizeof(D3D12InstancePayload);
                                instSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
                                device->CreateShaderResourceView(
                                    instBufferResource,
                                    &instSrvDesc,
                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + cmd.instanceBindingSlot)
                                );
                            }
                        }

                        // 3. Populate Constant Buffers (b0..b15 -> offset 0..15)
                        for (SIZE_T b = 0; b < cmd.constantBuffers.size(); ++b)
                        {
                            const auto& cb = cmd.constantBuffers[b];
                            if (cb.stage == stage && cb.registerSpace == space && cb.slotIndex < NorthstarD3D12::Impl::kCBVPerSpace)
                            {
                                const BYTE* data = cb.GetData();
                                if (!data || cb.byteSize == 0) continue;

                                D3D12_GPU_VIRTUAL_ADDRESS va = 0;
                                if (cb.sharedDataId != 0)
                                {
                                    auto itVa = cachedSharedCbVA.find(cb.sharedDataId);
                                    if (itVa != cachedSharedCbVA.end())
                                    {
                                        va = itVa->second;
                                    }
                                    else
                                    {
                                        va = m_impl->pipeline.AllocateCBuffer(data, cb.byteSize);
                                        if (va != 0) cachedSharedCbVA[cb.sharedDataId] = va;
                                    }
                                }
                                else
                                {
                                    va = m_impl->pipeline.AllocateCBuffer(data, cb.byteSize);
                                }
                                if (va == 0) continue;

                                D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc{};
                                cbvDesc.BufferLocation = va;
                                cbvDesc.SizeInBytes = (cb.byteSize + 255) & ~255;
                                device->CreateConstantBufferView(
                                    &cbvDesc,
                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + cb.slotIndex)
                                );
                            }
                        }

                        // 4. Populate Structured Buffers (t0..t31 -> offset 16..47)
                        for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                        {
                            const auto& sb = cmd.structuredBuffers[s];
                            if (sb.isUAV) continue;
                            if (sb.stage == stage && sb.registerSpace == space && sb.slotIndex < NorthstarD3D12::Impl::kSRVPerSpace)
                            {
                                // Persistent GPU-resident buffer path (zero CPU upload ring allocation)
                                if (sb.buffer != 0)
                                {
                                    D3D12GPUBufferResource* gpuBuf = m_impl->resourceMgr.GetGPUBuffer(sb.buffer);
                                    if (gpuBuf && gpuBuf->resource && gpuBuf->srvDescriptorIndex != UINT32_MAX)
                                    {
                                        const D3D12_RESOURCE_STATES reqState = (stage == ShaderStage::PIXEL)
                                            ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                            : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                                        const D3D12_RESOURCE_STATES curState = m_impl->stateTracker.GetResourceState(gpuBuf->resource.Get());
                                        if ((curState & reqState) == 0)
                                        {
                                            m_impl->stateTracker.TransitionResource(
                                                gpuBuf->resource.Get(),
                                                GPUResourceAccess::ALL_SHADER_RESOURCE
                                            );
                                            gpuBuf->currentState = m_impl->stateTracker.GetResourceState(gpuBuf->resource.Get());
                                            drawQueuedResourceBarriers = true;
                                        }

                                        const UINT32 reqStride = (sb.stride > 0u) ? sb.stride : (gpuBuf->desc.stride > 0u ? gpuBuf->desc.stride : 4u);
                                        const BOOL   explicitZeroSlice = (sb.elementCount == 0u && sb.stride > 0u);
                                        const UINT32 reqCount  = (sb.elementCount > 0u) ? sb.elementCount : gpuBuf->desc.elementCount;
                                        if (explicitZeroSlice)
                                        {
                                            D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                            nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                            nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                            nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                            nullSrvDesc.Buffer.FirstElement        = 0;
                                            nullSrvDesc.Buffer.NumElements         = 0;
                                            nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                            nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                            device->CreateShaderResourceView(
                                                nullptr,
                                                &nullSrvDesc,
                                                m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex));
                                        }
                                        else if (sb.firstElement > 0u ||
                                                 reqStride != gpuBuf->desc.stride ||
                                                 reqCount != gpuBuf->desc.elementCount)
                                        {
                                            const UINT32 maxElems = (reqStride > 0u) ? (gpuBuf->byteSize / reqStride) : 0u;
                                            if (sb.firstElement < maxElems)
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC sliceSrv{};
                                                sliceSrv.Format                     = DXGI_FORMAT_UNKNOWN;
                                                sliceSrv.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                sliceSrv.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                sliceSrv.Buffer.FirstElement        = sb.firstElement;
                                                sliceSrv.Buffer.NumElements         = (std::min)(reqCount, maxElems - sb.firstElement);
                                                sliceSrv.Buffer.StructureByteStride = reqStride;
                                                sliceSrv.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(
                                                    gpuBuf->resource.Get(),
                                                    &sliceSrv,
                                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex));
                                            }
                                            else
                                            {
                                                D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                                nullSrvDesc.Format                     = DXGI_FORMAT_UNKNOWN;
                                                nullSrvDesc.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                                                nullSrvDesc.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                                nullSrvDesc.Buffer.FirstElement        = 0;
                                                nullSrvDesc.Buffer.NumElements         = 0;
                                                nullSrvDesc.Buffer.StructureByteStride = reqStride;
                                                nullSrvDesc.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
                                                device->CreateShaderResourceView(
                                                    nullptr,
                                                    &nullSrvDesc,
                                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex));
                                            }
                                        }
                                        else
                                        {
                                            D3D12_CPU_DESCRIPTOR_HANDLE src = m_impl->cbvSrvUavPool.GetCpuStagingHandle(gpuBuf->srvDescriptorIndex);
                                            device->CopyDescriptorsSimple(
                                                1,
                                                m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex),
                                                src,
                                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                            );
                                        }
                                    }
                                    continue;
                                }

                                const UINT32 stride = sb.stride > 0 ? sb.stride : 4;

                                if (sb.elementCount == 0 || sb.byteSize == 0 || !sb.GetData())
                                {
                                    // Zero-element StructuredBuffer -> Bind DirectX 12 Null Buffer SRV with NumElements = 0!
                                    // Hardware GetDimensions(sb, count, stride) returns count = 0 and stride = stride.
                                    D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc{};
                                    nullSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                    nullSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                    nullSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                    nullSrvDesc.Buffer.FirstElement = 0;
                                    nullSrvDesc.Buffer.NumElements = 0;
                                    nullSrvDesc.Buffer.StructureByteStride = stride;
                                    nullSrvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

                                    device->CreateShaderResourceView(
                                        nullptr,
                                        &nullSrvDesc,
                                        m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex)
                                    );
                                    continue;
                                }

                                NorthstarD3D12::Impl::CachedSharedSBUpload sbUpload{};
                                if (!m_impl->AllocateOrReuseStructuredBuffer(sb, stride, sbUpload)) continue;

                                D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
                                srvDesc.Format = DXGI_FORMAT_UNKNOWN;
                                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                                srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                                srvDesc.Buffer.FirstElement = sbUpload.firstElement;
                                srvDesc.Buffer.NumElements = sbUpload.numElements;
                                srvDesc.Buffer.StructureByteStride = sbUpload.stride;
                                srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

                                device->CreateShaderResourceView(
                                    sbUpload.resource,
                                    &srvDesc,
                                    m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + sb.slotIndex)
                                );
                            }
                        }

                        // 5. Populate Textures (t0..t31 -> offset 16..47)
                        for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                        {
                            const auto& tex = cmd.textures[t];
                            if (tex.stage == stage && tex.registerSpace == space && tex.slotIndex < NorthstarD3D12::Impl::kSRVPerSpace)
                            {
                                if (tex.texture != 0)
                                {
                                    D3D12TextureResource* texRes = m_impl->resourceMgr.GetTexture(tex.texture);
                                    if (texRes && texRes->resource && texRes->srvDescriptorIndex != UINT32_MAX)
                                    {
                                        const BOOL isActiveDepth = (hasDepth && tex.texture == activeDepthHandle);
                                        const BOOL isActiveColor = (isTextureTarget && tex.texture == view.camera.renderTarget.colorTexture);
                                        if (!isActiveDepth && !isActiveColor)
                                        {
                                            if (!m_impl->stateTracker.IsResourceTracked(texRes->resource.Get()))
                                            {
                                                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, texRes);
                                            }
                                            const D3D12_RESOURCE_STATES reqState = (stage == ShaderStage::PIXEL)
                                                ? D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                                            if (texRes->desc.mipLevels > 1u && tex.mipSlice != 0xFFFFFFFFu && tex.mipSlice < texRes->desc.mipLevels)
                                            {
                                                const D3D12_RESOURCE_STATES srState = m_impl->stateTracker.GetSubresourceState(texRes->resource.Get(), tex.mipSlice);
                                                if ((srState & reqState) == 0)
                                                {
                                                    m_impl->stateTracker.TransitionMipLevel(
                                                        texRes->resource.Get(),
                                                        tex.mipSlice,
                                                        GPUResourceAccess::ALL_SHADER_RESOURCE
                                                    );
                                                    texRes->currentState = m_impl->stateTracker.GetResourceState(texRes->resource.Get());
                                                    drawQueuedResourceBarriers = true;
                                                }
                                            }
                                            else
                                            {
                                                const D3D12_RESOURCE_STATES curState = m_impl->stateTracker.GetResourceState(texRes->resource.Get());
                                                if (!m_impl->stateTracker.IsResourceUniformState(texRes->resource.Get()) || (curState & reqState) == 0)
                                                {
                                                    m_impl->stateTracker.TransitionResource(
                                                        texRes->resource.Get(),
                                                        GPUResourceAccess::ALL_SHADER_RESOURCE
                                                    );
                                                    texRes->currentState = m_impl->stateTracker.GetResourceState(texRes->resource.Get());
                                                    drawQueuedResourceBarriers = true;
                                                }
                                            }
                                        }

                                        UINT32 srcSrvIdx = texRes->srvDescriptorIndex;
                                        if (tex.mipSlice != 0xFFFFFFFFu &&
                                            tex.mipSlice < texRes->mipSrvDescriptors.size() &&
                                            texRes->mipSrvDescriptors[tex.mipSlice] != UINT32_MAX)
                                        {
                                            srcSrvIdx = texRes->mipSrvDescriptors[tex.mipSlice];
                                        }
                                        D3D12_CPU_DESCRIPTOR_HANDLE src = m_impl->cbvSrvUavPool.GetCpuStagingHandle(srcSrvIdx);
                                        device->CopyDescriptorsSimple(
                                            1,
                                            m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + tex.slotIndex),
                                            src,
                                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
                                        );
                                    }
                                }
                                else
                                {
                                    NorthstarD3D12::Impl::WriteNullTextureSRV(
                                        device,
                                        m_impl->cbvSrvUavPool.GetCPUHandle(tableStart + NorthstarD3D12::Impl::kCBVPerSpace + tex.slotIndex),
                                        tex.nullViewType,
                                        tex.nullFormat);
                                }
                            }
                        }

                        if (canCacheSpaceTable)
                        {
                            cachedSharedSpaceTable[spaceCacheKey] = tableStart;
                        }

                        // 6. Bind Root Parameter
                        if (currentRootTable[rootParam] != tableStart)
                        {
                            m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                rootParam,
                                m_impl->cbvSrvUavPool.GetGPUHandle(tableStart)
                            );
                            currentRootTable[rootParam] = tableStart;
                        }
                    }
                }

                // 6b. Graphics UAV Table Binding (Params 41..44 -> u0..u15, space0..space3, VISIBILITY_ALL)
                BOOL hasGraphicsUAV = !cmd.rwTextures.empty();
                if (!hasGraphicsUAV)
                {
                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                    {
                        if (cmd.structuredBuffers[s].isUAV != FALSE)
                        {
                            hasGraphicsUAV = true;
                            break;
                        }
                    }
                }

                BOOL graphicsUavRejected = false;
                if (hasGraphicsUAV)
                {
                    // Strict validation UAV hazard validation (RTV/DSV/Indirect/SRV overlap check)
                    for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                    {
                        const auto& sb = cmd.structuredBuffers[s];
                        if (!sb.isUAV) continue;
                        if (sb.registerSpace >= NorthstarD3D12::Impl::kMaxSpaces ||
                            sb.slotIndex >= NorthstarD3D12::Impl::kUAVPerSpace ||
                            sb.buffer == 0 ||
                            sb.buffer == cmd.indirectArgsBuffer ||
                            sb.buffer == cmd.indirectCountBuffer ||
                            sb.buffer == cmd.instanceBuffer)
                        {
                            graphicsUavRejected = true;
                            break;
                        }
                        D3D12GPUBufferResource* uavBuf = m_impl->resourceMgr.GetGPUBuffer(sb.buffer);
                        if (!uavBuf || !uavBuf->resource || uavBuf->uavDescriptorIndex == UINT32_MAX)
                        {
                            graphicsUavRejected = true;
                            break;
                        }
                        for (SIZE_T s2 = 0; s2 < cmd.structuredBuffers.size(); ++s2)
                        {
                            if (!cmd.structuredBuffers[s2].isUAV && cmd.structuredBuffers[s2].buffer == sb.buffer)
                            {
                                graphicsUavRejected = true;
                                break;
                            }
                        }
                        if (graphicsUavRejected) break;
                    }

                    for (SIZE_T r = 0; !graphicsUavRejected && r < cmd.rwTextures.size(); ++r)
                    {
                        const auto& rwt = cmd.rwTextures[r];
                        if (rwt.registerSpace >= NorthstarD3D12::Impl::kMaxSpaces ||
                            rwt.slotIndex >= NorthstarD3D12::Impl::kUAVPerSpace ||
                            rwt.texture == 0 ||
                            (isTextureTarget && rwt.texture == view.camera.renderTarget.colorTexture) ||
                            (hasDepth && rwt.texture == activeDepthHandle))
                        {
                            graphicsUavRejected = true;
                            break;
                        }
                        D3D12TextureResource* uavTex = m_impl->resourceMgr.GetTexture(rwt.texture);
                        if (!uavTex || !uavTex->resource || uavTex->uavDescriptorIndex == UINT32_MAX)
                        {
                            graphicsUavRejected = true;
                            break;
                        }
                        for (SIZE_T t = 0; t < cmd.textures.size(); ++t)
                        {
                            if (cmd.textures[t].texture == rwt.texture &&
                                (cmd.textures[t].mipSlice == 0xFFFFFFFFu || cmd.textures[t].mipSlice == rwt.mipSlice))
                            {
                                graphicsUavRejected = true;
                                break;
                            }
                        }
                    }

                    if (!graphicsUavRejected)
                    {
                        for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                        {
                            const auto& sb = cmd.structuredBuffers[s];
                            if (!sb.isUAV) continue;
                            D3D12GPUBufferResource* uavBuf = m_impl->resourceMgr.GetGPUBuffer(sb.buffer);
                            const GPUResourceAccess targetAcc =
                                (sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE ||
                                 sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_WRITE ||
                                 sb.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ)
                                    ? sb.uavAccess
                                    : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
                            m_impl->stateTracker.TransitionResource(uavBuf->resource.Get(), targetAcc);
                            uavBuf->currentState = m_impl->stateTracker.GetResourceState(uavBuf->resource.Get());
                            drawQueuedResourceBarriers = true;
                        }

                        for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
                        {
                            const auto& rwt = cmd.rwTextures[r];
                            D3D12TextureResource* uavTex = m_impl->resourceMgr.GetTexture(rwt.texture);
                            if (!m_impl->stateTracker.IsResourceTracked(uavTex->resource.Get()))
                            {
                                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, uavTex);
                            }
                            const GPUResourceAccess targetAcc =
                                (rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE ||
                                 rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_WRITE ||
                                 rwt.uavAccess == GPUResourceAccess::UNORDERED_ACCESS_READ)
                                    ? rwt.uavAccess
                                    : GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
                            if (uavTex->desc.mipLevels > 1u && rwt.mipSlice < uavTex->mipUavDescriptors.size())
                            {
                                m_impl->stateTracker.TransitionMipLevel(uavTex->resource.Get(), rwt.mipSlice, targetAcc);
                            }
                            else
                            {
                                m_impl->stateTracker.TransitionResource(uavTex->resource.Get(), targetAcc);
                            }
                            uavTex->currentState = m_impl->stateTracker.GetResourceState(uavTex->resource.Get());
                            drawQueuedResourceBarriers = true;
                        }

                        for (UINT32 space = 0; space < NorthstarD3D12::Impl::kMaxSpaces; ++space)
                        {
                            BOOL uavSpaceActive = false;
                            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                            {
                                if (cmd.structuredBuffers[s].isUAV && cmd.structuredBuffers[s].registerSpace == space)
                                { uavSpaceActive = true; break; }
                            }
                            for (SIZE_T r = 0; !uavSpaceActive && r < cmd.rwTextures.size(); ++r)
                            {
                                if (cmd.rwTextures[r].registerSpace == space)
                                { uavSpaceActive = true; break; }
                            }

                            if (!uavSpaceActive)
                            {
                                if (defaultDummyUavTableIndex != UINT32_MAX && currentGraphicsUavTable[space] != defaultDummyUavTableIndex)
                                {
                                    m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                        D3D12Pipeline::kGraphicsUAVSpaceParamBase + space,
                                        m_impl->cbvSrvUavPool.GetGPUHandle(defaultDummyUavTableIndex));
                                    currentGraphicsUavTable[space] = defaultDummyUavTableIndex;
                                }
                                continue;
                            }

                            if (tablesUsedThisFrame >= NorthstarD3D12::Impl::kTablesPerBackBuffer ||
                                currentTableOffset + NorthstarD3D12::Impl::kUAVPerSpace > NorthstarD3D12::Impl::kRingDescriptorsPerBB)
                            {
                                drawDescriptorRingExhausted = true;
                                break;
                            }

                            const UINT32 uavTableStart = m_impl->dynamicTableBase[bbIndex] + currentTableOffset;
                            currentTableOffset += NorthstarD3D12::Impl::kUAVPerSpace;
                            tablesUsedThisFrame++;

                            if (defaultDummyUavTableIndex != UINT32_MAX)
                            {
                                device->CopyDescriptorsSimple(
                                    NorthstarD3D12::Impl::kUAVPerSpace,
                                    m_impl->cbvSrvUavPool.GetCPUHandle(uavTableStart),
                                    m_impl->cbvSrvUavPool.GetCpuStagingHandle(defaultDummyUavTableIndex),
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                            }

                            for (SIZE_T s = 0; s < cmd.structuredBuffers.size(); ++s)
                            {
                                const auto& sb = cmd.structuredBuffers[s];
                                if (!sb.isUAV || sb.registerSpace != space || sb.slotIndex >= NorthstarD3D12::Impl::kUAVPerSpace) continue;
                                const D3D12GPUBufferResource* uavBuf = m_impl->resourceMgr.GetGPUBuffer(sb.buffer);
                                if (uavBuf && uavBuf->uavDescriptorIndex != UINT32_MAX)
                                {
                                    device->CopyDescriptorsSimple(
                                        1,
                                        m_impl->cbvSrvUavPool.GetCPUHandle(uavTableStart + sb.slotIndex),
                                        m_impl->cbvSrvUavPool.GetCpuStagingHandle(uavBuf->uavDescriptorIndex),
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                }
                            }

                            for (SIZE_T r = 0; r < cmd.rwTextures.size(); ++r)
                            {
                                const auto& rwt = cmd.rwTextures[r];
                                if (rwt.registerSpace != space || rwt.slotIndex >= NorthstarD3D12::Impl::kUAVPerSpace) continue;
                                const D3D12TextureResource* uavTex = m_impl->resourceMgr.GetTexture(rwt.texture);
                                if (uavTex && uavTex->uavDescriptorIndex != UINT32_MAX)
                                {
                                    UINT32 uavDescIdx = uavTex->uavDescriptorIndex;
                                    if (rwt.mipSlice < uavTex->mipUavDescriptors.size() &&
                                        uavTex->mipUavDescriptors[rwt.mipSlice] != UINT32_MAX)
                                    {
                                        uavDescIdx = uavTex->mipUavDescriptors[rwt.mipSlice];
                                    }
                                    device->CopyDescriptorsSimple(
                                        1,
                                        m_impl->cbvSrvUavPool.GetCPUHandle(uavTableStart + rwt.slotIndex),
                                        m_impl->cbvSrvUavPool.GetCpuStagingHandle(uavDescIdx),
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                                }
                            }

                            if (currentGraphicsUavTable[space] != uavTableStart)
                            {
                                m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                    D3D12Pipeline::kGraphicsUAVSpaceParamBase + space,
                                    m_impl->cbvSrvUavPool.GetGPUHandle(uavTableStart));
                                currentGraphicsUavTable[space] = uavTableStart;
                            }
                        }
                    }
                }
                else if (defaultDummyUavTableIndex != UINT32_MAX)
                {
                    for (UINT32 space = 0; space < NorthstarD3D12::Impl::kMaxSpaces; ++space)
                    {
                        if (currentGraphicsUavTable[space] != defaultDummyUavTableIndex)
                        {
                            m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                D3D12Pipeline::kGraphicsUAVSpaceParamBase + space,
                                m_impl->cbvSrvUavPool.GetGPUHandle(defaultDummyUavTableIndex));
                            currentGraphicsUavTable[space] = defaultDummyUavTableIndex;
                        }
                    }
                }

                if (drawQueuedResourceBarriers)
                {
                    m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                }
                if (drawDescriptorRingExhausted || graphicsUavRejected)
                {
                    if (indirectArgBuf != nullptr) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                    continue;
                }

                // 7. Dynamic Per-Stage Multi-Space Samplers Binding (VS: 8..11, PS: 12..15, GS: 29..32, HS: 33..36, DS: 37..40)
                if (cmd.pixelShader != 0 || !cmd.samplers.empty())
                {
                    const BOOL stageBound[5] = {
                        true,
                        cmd.pixelShader != 0,
                        cmd.geometryShader != 0,
                        cmd.hullShader != 0,
                        cmd.domainShader != 0
                    };
                    for (UINT32 stIdx = 0u; stIdx < 5u; ++stIdx)
                    {
                        if (!stageBound[stIdx]) continue;
                        const ShaderStage st   = kGraphicsStageOrder[stIdx];
                        const UINT32 paramBase = D3D12Pipeline::GetGraphicsStageSamplerParamBase(st);
                        for (UINT32 space = 0; space < NorthstarD3D12::Impl::kMaxSpaces; ++space)
                        {
                            const UINT32 targetSamplerTable = cmd.samplers.empty()
                                ? m_impl->defaultDummySamplerTableIndex
                                : m_impl->GetOrCreateSamplerTable(device, cmd.samplers, space, st);
                            if (targetSamplerTable != UINT32_MAX && targetSamplerTable != currentSamplerTableIndex[stIdx][space])
                            {
                                m_impl->frameCmdList->SetGraphicsRootDescriptorTable(
                                    paramBase + space,
                                    m_impl->samplerPool.GetGPUHandle(targetSamplerTable)
                                );
                                currentSamplerTableIndex[stIdx][space] = targetSamplerTable;
                            }
                        }
                    }
                }

                // Optional Graphics Root Constants (Param 12 -> b1, space3, 4 DWORDs)
                if (cmd.useGraphicsRootConstants)
                {
                    m_impl->frameCmdList->SetGraphicsRoot32BitConstants(
                        D3D12Pipeline::kGraphicsRootConstantsParamIndex,
                        4,
                        cmd.graphicsRootConstants,
                        0);
                }

                // Draw Call (Direct or GPU-Driven ExecuteIndirect)
                if (indirectArgBuf != nullptr)
                {
                    m_impl->frameCmdList->ExecuteIndirect(
                        indirectCmdSig,
                        cmd.maxIndirectCommandCount,
                        indirectArgBuf->resource.Get(),
                        cmd.indirectArgsOffset,
                        indirectCountBuf ? indirectCountBuf->resource.Get() : nullptr,
                        indirectCountBuf ? cmd.indirectCountBufferOffset : 0
                    );
                    if (isPromotedIndirect)
                    {
                        // Promoted ExecuteIndirect overwrites IA VB/IB and root constants on the command list.
                        // Invalidate cached handles so subsequent direct draws rebind their own VB/IB cleanly.
                        currentVB = 0;
                        currentIB = 0;
                        const UINT32 zeroRootConsts[4] = { 0, 0, 0, 0 };
                        m_impl->frameCmdList->SetGraphicsRoot32BitConstants(
                            D3D12Pipeline::kGraphicsRootConstantsParamIndex,
                            4,
                            zeroRootConsts,
                            0
                        );
                    }
                    m_impl->lastFrameMetrics.indirectDrawCommandsExecuted++;
                }
                else if (isInstanced || gpuInstanceBuf != nullptr)
                {
                    if (hasIndices)
                    {
                        m_impl->frameCmdList->DrawIndexedInstanced(
                            cmd.indexCount, cmd.instanceCount, cmd.startIndex, cmd.baseVertex, cmd.startInstance);
                    }
                    else
                    {
                        m_impl->frameCmdList->DrawInstanced(
                            cmd.vertexCount, cmd.instanceCount, 0, cmd.startInstance);
                    }
                }
                else
                {
                    if (hasIndices)
                    {
                        m_impl->frameCmdList->DrawIndexedInstanced(
                            cmd.indexCount, 1, cmd.startIndex, cmd.baseVertex, 0);
                    }
                    else
                    {
                        m_impl->frameCmdList->DrawInstanced(
                            cmd.vertexCount, 1, 0, 0);
                    }
                }
            }
        };

        // 7.1 Execute Scheduled Layered Depth Passes
        if (!view.layeredDepthPasses.empty())
        {
            const UINT32 tPreLayeredStart = m_impl->EmitTimestamp(queryBase, slotFlags);
            BOOL anyPreLayeredExec        = false;
            for (SIZE_T p = 0; p < view.layeredDepthPasses.size(); ++p)
            {
                if (m_impl->RecordLayeredDepthPass(
                    view.layeredDepthPasses[p],
                    view.camera.viewProjectionMatrix,
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentPSO,
                    currentComputePSO,
                    currentVB,
                    currentIB,
                    currentRootTable,
                    currentSamplerTableIndex
                ))
                {
                    anyPreLayeredExec = true;
                }
            }
            const UINT32 tPreLayeredEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
            if (anyPreLayeredExec)
            {
                NorthstarD3D12::Impl::AddTimestampSpan(
                    slotFlags,
                    tPreLayeredStart,
                    tPreLayeredEnd,
                    NorthstarD3D12::Impl::TimestampSpanKind::PRE_LAYERED_DEPTH);
            }
            if (isComputeRootSigActive)
            {
                bindGraphicsRootSignatureAndFallbacks();
                for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
                resetGraphicsUavCache(currentGraphicsUavTable);
            }
            else
            {
                currentGraphicsRootSig = nullptr;
            }
        }

        const UINT32 tMainStart = m_impl->EmitTimestamp(queryBase, slotFlags);

        // Restore Camera Viewport & Scissor for Geometry Pass
        m_impl->frameCmdList->RSSetViewports(1, &viewport);
        m_impl->frameCmdList->RSSetScissorRects(1, &scissor);
        m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, hasDepth ? &dsvHandle : nullptr);
        currentPSO = nullptr;
        currentVB  = 0;
        currentIB  = 0;

        recordDrawList(view.geometryPass.opaqueCommands);
        recordDrawList(view.geometryPass.transparentCommands);

        auto restoreRenderTargetStates = [&](BOOL bindRtv) {
            if (isTextureTarget && targetTexture && targetTexture->resource)
            {
                if (!m_impl->stateTracker.IsResourceTracked(targetTexture->resource.Get()))
                {
                    NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, targetTexture);
                }
                if (m_impl->stateTracker.GetResourceState(targetTexture->resource.Get()) != D3D12_RESOURCE_STATE_RENDER_TARGET)
                {
                    m_impl->stateTracker.TransitionResource(targetTexture->resource.Get(), GPUResourceAccess::RENDER_TARGET);
                    m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                }
                targetTexture->currentState = m_impl->stateTracker.GetResourceState(targetTexture->resource.Get());
            }

            if (!bindRtv)
            {
                // Depth-only pass (DEPTH_PREPASS) requires writable DSV in DEPTH_WRITE state
                if (hasDepth && targetDepthTexture && targetDepthTexture->resource)
                {
                    if (m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get()) != D3D12_RESOURCE_STATE_DEPTH_WRITE)
                    {
                        m_impl->stateTracker.TransitionResource(targetDepthTexture->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                        m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                    }
                    targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                }
                isReadOnlyDepthBound = false;
                if (hasDepth)
                {
                    m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, &dsvHandle);
                }
                else
                {
                    m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
                }
                return;
            }

            if (hasDepth && targetDepthTexture && targetDepthTexture->resource)
            {
                const D3D12_RESOURCE_STATES curDepthState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                if (isReadOnlyDepthBound && hasReadOnlyDsv)
                {
                    const D3D12_RESOURCE_STATES reqRoState =
                        D3D12_RESOURCE_STATE_DEPTH_READ |
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    if ((curDepthState & reqRoState) != reqRoState)
                    {
                        m_impl->stateTracker.TransitionResource(
                            targetDepthTexture->resource.Get(),
                            GPUResourceAccess::DEPTH_READ | GPUResourceAccess::ALL_SHADER_RESOURCE
                        );
                        m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                    }
                    targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                    m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &readOnlyDsvHandle);
                }
                else
                {
                    if (curDepthState != D3D12_RESOURCE_STATE_DEPTH_WRITE)
                    {
                        m_impl->stateTracker.TransitionResource(targetDepthTexture->resource.Get(), GPUResourceAccess::DEPTH_WRITE);
                        m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                    }
                    targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                    isReadOnlyDepthBound = false;
                    m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
                }
            }
            else
            {
                m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, hasDepth ? &dsvHandle : nullptr);
            }
        };

        UINT32 activeComputeBurstStart = UINT32_MAX;
        BOOL   activeComputeBurstExec  = false;

        // 7.2 Execute Interleaved Compute + Graphics Pass Items in exact registration order
        for (SIZE_T itemIdx = 0; itemIdx < view.interleavedPassItems.size(); ++itemIdx)
        {
            const auto& item = view.interleavedPassItems[itemIdx];
            if (item.type == FramePassItemType::COMPUTE_DISPATCH)
            {
                if (activeComputeBurstStart == UINT32_MAX)
                {
                    activeComputeBurstStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                    activeComputeBurstExec  = false;
                }

                // Unbind RTV/DSV before compute in case the compute kernel transitions/samples the active depth or color target
                m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
                if (m_impl->RecordComputeDispatch(
                    item.computeDispatch,
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentComputePSO,
                    currentComputeRootTable,
                    currentComputeSamplerTable
                ))
                {
                    activeComputeBurstExec = true;
                }
                // SetPipelineState is shared between Compute and Graphics on ID3D12GraphicsCommandList;
                // reset currentPSO so any subsequent Graphics draw re-binds its Graphics PSO.
                currentPSO = nullptr;

                const BOOL nextIsSameComputeBurst =
                    (itemIdx + 1 < view.interleavedPassItems.size()) &&
                    (view.interleavedPassItems[itemIdx + 1].type == FramePassItemType::COMPUTE_DISPATCH) &&
                    (view.interleavedPassItems[itemIdx + 1].computeDispatch.profileGroup == item.computeDispatch.profileGroup);
                if (!nextIsSameComputeBurst)
                {
                    const UINT32 burstEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                    if (activeComputeBurstExec)
                    {
                        NorthstarD3D12::Impl::AddTimestampSpan(
                            slotFlags,
                            activeComputeBurstStart,
                            burstEnd,
                            NorthstarD3D12::Impl::MapProfileGroupToSpanKind(
                                item.computeDispatch.profileGroup,
                                NorthstarD3D12::Impl::TimestampSpanKind::INTER_COMPUTE));
                    }
                    activeComputeBurstStart = UINT32_MAX;
                }
            }
            else if (item.type == FramePassItemType::DEPTH_PREPASS)
            {
                if (hasDepth)
                {
                    if (isComputeRootSigActive)
                    {
                        bindGraphicsRootSignatureAndFallbacks();
                        for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                        m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
                        resetGraphicsUavCache(currentGraphicsUavTable);
                    }
                    const UINT32 tDpStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                    // Bind depth-only target (NULL RTV, NumRenderTargets = 0)
                    restoreRenderTargetStates(false);
                    currentPSO = nullptr;
                    recordDrawList(item.drawCommands, false, true);
                    m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
                    if (targetDepthTexture && targetDepthTexture->resource)
                    {
                        if (hasReadOnlyDsv)
                        {
                            m_impl->stateTracker.TransitionResource(
                                targetDepthTexture->resource.Get(),
                                GPUResourceAccess::DEPTH_READ | GPUResourceAccess::ALL_SHADER_RESOURCE
                            );
                            m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                            isReadOnlyDepthBound = true;
                        }
                        else
                        {
                            m_impl->stateTracker.TransitionResource(
                                targetDepthTexture->resource.Get(),
                                GPUResourceAccess::ALL_SHADER_RESOURCE
                            );
                            m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
                            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
                            isReadOnlyDepthBound = false;
                        }
                    }
                    const UINT32 tDpEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                    NorthstarD3D12::Impl::AddTimestampSpan(
                        slotFlags,
                        tDpStart,
                        tDpEnd,
                        NorthstarD3D12::Impl::TimestampSpanKind::DEPTH_PREPASS);
                    // Restore color + depth render targets for subsequent graphics passes
                    restoreRenderTargetStates(true);
                    currentPSO        = nullptr;
                    currentComputePSO = nullptr;
                }
            }
            else if (item.type == FramePassItemType::GRAPHICS_DRAWS)
            {
                if (isComputeRootSigActive)
                {
                    bindGraphicsRootSignatureAndFallbacks();
                    for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
                    m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
                    resetGraphicsUavCache(currentGraphicsUavTable);
                    currentPSO = nullptr;
                }
                restoreRenderTargetStates(true);
                recordDrawList(item.drawCommands, false, false);
                currentComputePSO = nullptr;
            }
            else if (item.type == FramePassItemType::LAYERED_DEPTH_PASS)
            {
                const BOOL isShadowCasterPass =
                    !item.layeredDepthPass.drawCommands.empty() && !item.layeredDepthPass.selectiveClear.enabled;
                const UINT32 tLdStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                const BOOL ldExecuted = m_impl->RecordLayeredDepthPass(
                    item.layeredDepthPass,
                    view.camera.viewProjectionMatrix,
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentPSO,
                    currentComputePSO,
                    currentVB,
                    currentIB,
                    currentRootTable,
                    currentSamplerTableIndex
                );
                const UINT32 tLdEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                if (ldExecuted)
                {
                    if (isShadowCasterPass)
                    {
                        slotFlags.hadInterleavedShadow = true;
                        NorthstarD3D12::Impl::AddTimestampSpan(
                            slotFlags,
                            tLdStart,
                            tLdEnd,
                            NorthstarD3D12::Impl::TimestampSpanKind::SHADOW_RASTER);
                    }
                    else
                    {
                        NorthstarD3D12::Impl::AddTimestampSpan(
                            slotFlags,
                            tLdStart,
                            tLdEnd,
                            NorthstarD3D12::Impl::TimestampSpanKind::INTER_LAYERED_DEPTH);
                    }
                }
                // Restore camera viewport, scissor, and primary render targets after layered depth pass
                m_impl->frameCmdList->RSSetViewports(1, &viewport);
                m_impl->frameCmdList->RSSetScissorRects(1, &scissor);
                restoreRenderTargetStates(true);
                currentGraphicsRootSig = nullptr;
                currentPSO             = nullptr;
                currentComputePSO      = nullptr;
            }
        }

        const UINT32 tMainEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
        NorthstarD3D12::Impl::AddTimestampSpan(
            slotFlags,
            tMainStart,
            tMainEnd,
            NorthstarD3D12::Impl::TimestampSpanKind::MAIN_SPAN);

        // 7.3 Execute Post-Pass Compute Dispatches
        if (!view.postPassComputeDispatches.empty())
        {
            m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        }
        {
            UINT32 postBurstStart = UINT32_MAX;
            BOOL   postBurstExec  = false;
            for (SIZE_T c = 0; c < view.postPassComputeDispatches.size(); ++c)
            {
                if (postBurstStart == UINT32_MAX)
                {
                    postBurstStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                    postBurstExec  = false;
                }
                if (m_impl->RecordComputeDispatch(
                    view.postPassComputeDispatches[c],
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentComputePSO,
                    currentComputeRootTable,
                    currentComputeSamplerTable
                ))
                {
                    postBurstExec = true;
                }
                const BOOL sameNextGroup =
                    (c + 1 < view.postPassComputeDispatches.size()) &&
                    (view.postPassComputeDispatches[c + 1].profileGroup == view.postPassComputeDispatches[c].profileGroup);
                if (!sameNextGroup)
                {
                    const UINT32 postBurstEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                    if (postBurstExec)
                    {
                        NorthstarD3D12::Impl::AddTimestampSpan(
                            slotFlags,
                            postBurstStart,
                            postBurstEnd,
                            NorthstarD3D12::Impl::MapProfileGroupToSpanKind(
                                view.postPassComputeDispatches[c].profileGroup,
                                NorthstarD3D12::Impl::TimestampSpanKind::POST_COMPUTE));
                    }
                    postBurstStart = UINT32_MAX;
                }
            }
        }

        const UINT32 tFrameEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
        NorthstarD3D12::Impl::AddTimestampSpan(
            slotFlags,
            tFrameStart,
            tFrameEnd,
            NorthstarD3D12::Impl::TimestampSpanKind::FULL_FRAME);
        m_impl->FinalizeTimestampSlot(bbIndex, queryBase, slotFlags);

        // Pre-Present Callback (e.g. ImGui UI overlay) - only for window target
        if (!isTextureTarget && m_impl->prePresentCallback)
        {
            m_impl->frameCmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
            PrePresentCallback prePresentCopy = m_impl->prePresentCallback;
            prePresentCopy(m_impl->frameCmdList.Get());
        }

        // 7.9 Record queued asynchronous GPUBuffer readback copies before command list close
        m_impl->resourceMgr.FlushPendingGPUBufferReadbacks(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            bbIndex,
            upcomingFenceValue
        );

        // 8. Unbind render targets and transition SceneDepth & color target out of render states
        m_impl->frameCmdList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
        if (targetDepthTexture && targetDepthTexture->resource)
        {
            const D3D12_RESOURCE_STATES reqSrvState =
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if (m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get()) != reqSrvState)
            {
                m_impl->stateTracker.TransitionResource(
                    targetDepthTexture->resource.Get(),
                    GPUResourceAccess::ALL_SHADER_RESOURCE
                );
                m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            targetDepthTexture->currentState = m_impl->stateTracker.GetResourceState(targetDepthTexture->resource.Get());
        }

        if (isTextureTarget)
        {
            if (!m_impl->stateTracker.IsResourceTracked(targetTexture->resource.Get()))
            {
                NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, targetTexture);
            }
            if (m_impl->stateTracker.GetResourceState(targetTexture->resource.Get()) != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
            {
                m_impl->stateTracker.TransitionResource(targetTexture->resource.Get(), GPUResourceAccess::PIXEL_SHADER_RESOURCE);
                m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            targetTexture->currentState = m_impl->stateTracker.GetResourceState(targetTexture->resource.Get());
        }
        else
        {
            // Transition Back Buffer: RENDER_TARGET -> PRESENT
            D3D12_RESOURCE_BARRIER toPresent = CreateTransitionBarrier(
                sc.GetCurrentBackBuffer(),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT
            );
            m_impl->frameCmdList->ResourceBarrier(1, &toPresent);
        }

        // 9. Close & Execute
        m_impl->frameCmdList->Close();
        m_impl->isRecordingCmdList = false;
        const auto cmdRecordEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.cmdRecordCpuMs       = std::chrono::duration<float, std::milli>(cmdRecordEnd - cmdRecordStart).count();
        m_impl->lastFrameMetrics.descriptorTablesUsed = tablesUsedThisFrame;

        const auto cmdSubmitStart = std::chrono::high_resolution_clock::now();
        ID3D12CommandList* lists[] = { m_impl->frameCmdList.Get() };
        m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, lists);

        // 10. Signal Fence & Finalize Pending Destroyed GPUBuffers
        m_impl->deviceContext.IncrementFrameFenceValue();
        UINT64 fValue = m_impl->deviceContext.GetFrameFenceValue();
        m_impl->deviceContext.GetCommandQueue()->Signal(fence, fValue);
        m_impl->frameFenceValues[bbIndex] = fValue;
        if (!isTextureTarget)
        {
            const UINT32 scIdx = sc.GetCurrentBackBufferIndex() % D3D12SwapChain::kBackBufferCount;
            m_impl->swapChainFenceValues[scIdx] = fValue;
        }
        m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fValue);
        const auto cmdSubmitEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.cmdSubmitCpuMs = std::chrono::duration<float, std::milli>(cmdSubmitEnd - cmdSubmitStart).count();

        const UINT64 uploadedBytes64 = m_impl->resourceMgr.ConsumeUploadedBytesThisSubmission();
        m_impl->lastFrameMetrics.uploadedBytesThisFrame        = static_cast<UINT32>((std::min)(uploadedBytes64, static_cast<UINT64>(0xFFFFFFFFull)));
        m_impl->lastFrameMetrics.gpuBufferAllocationsThisFrame = m_impl->resourceMgr.ConsumeGPUBufferAllocationsThisSubmission();
        m_impl->lastFrameMetrics.pendingRetiredResources       = m_impl->resourceMgr.GetPendingRetiredResourceCount();
        m_impl->lastFrameMetrics.psoCacheHits                  = m_impl->pipeline.GetPSOCacheHits();
        m_impl->lastFrameMetrics.psoCacheMisses                = m_impl->pipeline.GetPSOCacheMisses();
        m_impl->lastFrameMetrics.activeCachedPSOCount          = m_impl->pipeline.GetActiveCachedPSOCount();
        m_impl->lastFrameMetrics.retiredPSOCount               = m_impl->pipeline.GetRetiredPSOCount();

        if (m_impl->postFrameExecutionCallback && m_impl->postFrameWaitForGPU)
        {
            if (fence && fenceEvent && m_impl->GetSafeCompletedFenceValue() < fValue)
            {
                const auto postWaitStart = std::chrono::high_resolution_clock::now();
                m_impl->deviceContext.WaitForFenceValue(fence, fenceEvent, fValue);
                const auto postWaitEnd = std::chrono::high_resolution_clock::now();
                m_impl->lastFrameMetrics.fenceWaitCpuMs +=
                    std::chrono::duration<float, std::milli>(postWaitEnd - postWaitStart).count();
            }
        }

        const UINT64 completedNow = fence ? m_impl->GetSafeCompletedFenceValue() : fValue;
        UINT32 activeInFlight = 0;
        for (UINT32 s = 0; s < NorthstarD3D12::Impl::kMaxFramesInFlight; ++s)
        {
            if (m_impl->frameFenceValues[s] > completedNow)
            {
                activeInFlight++;
            }
        }
        m_impl->lastFrameMetrics.framesInFlightCount = (std::max)(1u, activeInFlight);

        if (m_impl->postFrameExecutionCallback)
        {
            PostFrameExecutionInfo info{};
            info.renderTargetType    = isTextureTarget ? RenderTargetType::TEXTURE : RenderTargetType::WINDOW;
            info.colorTexture        = isTextureTarget ? view.camera.renderTarget.colorTexture : 0;
            info.depthStencilTexture = activeDepthHandle;
            if (isTextureTarget && targetTexture && targetTexture->sharedHandle)
            {
                info.colorSharedHandle     = targetTexture->sharedHandle;
                info.colorSharedHandleType = SharedHandleType::WIN32_NT_HANDLE;
            }
            info.signaledFenceValue = fValue;
            info.nativeFence        = fence;
            HANDLE sharedFenceH     = m_impl->deviceContext.GetFrameFenceSharedHandle();
            if (sharedFenceH)
            {
                info.sharedFenceHandle     = sharedFenceH;
                info.sharedFenceHandleType = SharedHandleType::WIN32_NT_HANDLE;
            }
            info.gpuCompletedOnEntry = (!fence) || (m_impl->GetSafeCompletedFenceValue() >= fValue);

            PostFrameExecutionCallback postCbCopy = m_impl->postFrameExecutionCallback;
            postCbCopy(info);
        }
    }

    BOOL NorthstarD3D12::ExecuteComputeDispatches(const std::vector<ComputeDispatchCommand>& dispatches)
    {
        if (!m_impl || !m_impl->isInitialized || m_impl->deviceContext.IsDeviceRemoved() || dispatches.empty())
        {
            return false;
        }
        m_impl->AssertOwnerThread();
        m_impl->lastFrameMetrics.indirectDispatchCommandsExecuted = 0;
        m_impl->lastFrameMetrics.indirectCommandsRejected         = 0;

        // Pre-validate basic compute dispatch contracts BEFORE touching command lists or flushing uploads
        for (const auto& d : dispatches)
        {
            if (d.computeShader == 0)
            {
                if (d.indirectArgsBuffer != 0) m_impl->lastFrameMetrics.indirectCommandsRejected++;
                return false;
            }
            if (d.indirectArgsBuffer == 0)
            {
                if (d.threadGroupCountX == 0 || d.threadGroupCountY == 0 || d.threadGroupCountZ == 0 ||
                    d.threadGroupCountX > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
                    d.threadGroupCountY > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
                    d.threadGroupCountZ > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
                {
                    return false;
                }
            }
            else
            {
                const D3D12GPUBufferResource* iBuf = m_impl->resourceMgr.GetGPUBuffer(d.indirectArgsBuffer);
                if (!iBuf || !iBuf->resource ||
                    !HasGPUBufferUsage(iBuf->desc.usage, GPUBufferUsage::INDIRECT_ARGUMENT) ||
                    (d.indirectArgsOffset % 4u) != 0u ||
                    static_cast<UINT64>(d.indirectArgsOffset) + sizeof(GPUIndirectDispatchArgs) > static_cast<UINT64>(iBuf->byteSize))
                {
                    m_impl->lastFrameMetrics.indirectCommandsRejected++;
                    return false;
                }
            }
        }

        const UINT32 bbIndex = static_cast<UINT32>(m_impl->submissionSlotCursor % m_impl->maxFramesInFlight);
        m_impl->submissionSlotCursor++;
        ID3D12Fence* fence   = m_impl->deviceContext.GetFrameFence();
        HANDLE fenceEvent    = m_impl->deviceContext.GetFrameFenceEvent();

        const UINT64 requiredFence = m_impl->frameFenceValues[bbIndex];
        if (requiredFence > 0 && m_impl->GetSafeCompletedFenceValue() < requiredFence)
        {
            if (!m_impl->deviceContext.WaitForFenceValue(fence, fenceEvent, requiredFence))
                return false;
        }
        m_impl->ResolveTimestampSlot(bbIndex);

        const UINT64 completedFence = m_impl->GetSafeCompletedFenceValue();
        m_impl->resourceMgr.ProcessRetirementQueue(
            completedFence,
            m_impl->cbvSrvUavPool,
            m_impl->rtvPool,
            m_impl->dsvPool,
            m_impl->samplerPool,
            &m_impl->stateTracker
        );
        m_impl->ProcessRetiredSamplerTables(completedFence);

        if (FAILED(m_impl->frameCmdAlloc[bbIndex]->Reset())) return false;
        if (FAILED(m_impl->frameCmdList->Reset(m_impl->frameCmdAlloc[bbIndex].Get(), nullptr))) return false;
        m_impl->isRecordingCmdList = true;

        const UINT64 upcomingFenceValue = m_impl->deviceContext.GetFrameFenceValue() + 1;
        const UINT32 queryBase = bbIndex * NorthstarD3D12::Impl::kQueriesPerBackBuffer;
        NorthstarD3D12::Impl::TimestampSlotFlags slotFlags{};
        slotFlags.isStandaloneCompute = true;
        const BOOL hadUpload          = m_impl->resourceMgr.HasPendingBufferUploads();

        const UINT32 tStart = m_impl->EmitTimestamp(queryBase, slotFlags);

        const auto uploadRecordStart = std::chrono::high_resolution_clock::now();
        m_impl->resourceMgr.FlushPendingGPUBufferUploads(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            upcomingFenceValue
        );
        const auto uploadRecordEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.gpuBufferUploadRecordCpuMs =
            std::chrono::duration<float, std::milli>(uploadRecordEnd - uploadRecordStart).count();

        if (hadUpload)
        {
            const UINT32 tUploadEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
            NorthstarD3D12::Impl::AddTimestampSpan(
                slotFlags,
                tStart,
                tUploadEnd,
                NorthstarD3D12::Impl::TimestampSpanKind::UPLOAD);
        }

        ID3D12DescriptorHeap* heaps[] = {
            m_impl->cbvSrvUavPool.GetHeap(),
            m_impl->samplerPool.GetHeap()
        };
        m_impl->frameCmdList->SetDescriptorHeaps(2, heaps);
        m_impl->pipeline.BeginFrameBuffers(bbIndex, completedFence, upcomingFenceValue);
        m_impl->cachedSharedSBUploads.clear();

        UINT32 currentTableOffset = 0;
        UINT32 tablesUsedThisFrame = 0;
        BOOL isComputeRootSigActive = false;
        ID3D12PipelineState* currentComputePSO = nullptr;
        UINT32 currentComputeRootTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };
        UINT32 currentComputeSamplerTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };

        UINT32 burstStart = UINT32_MAX;
        for (SIZE_T i = 0; i < dispatches.size(); ++i)
        {
            if (burstStart == UINT32_MAX)
            {
                burstStart = m_impl->EmitTimestamp(queryBase, slotFlags);
            }
            if (!m_impl->RecordComputeDispatch(
                dispatches[i],
                bbIndex,
                currentTableOffset,
                tablesUsedThisFrame,
                isComputeRootSigActive,
                currentComputePSO,
                currentComputeRootTable,
                currentComputeSamplerTable))
            {
                m_impl->frameCmdList->Close();
                m_impl->isRecordingCmdList = false;
                if (hadUpload || i > 0)
                {
                    // Preserve already-recorded buffer uploads / state transitions so stateTracker never desyncs
                    ID3D12CommandList* abortLists[] = { m_impl->frameCmdList.Get() };
                    m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, abortLists);
                    m_impl->deviceContext.IncrementFrameFenceValue();
                    const UINT64 fAbort = m_impl->deviceContext.GetFrameFenceValue();
                    m_impl->deviceContext.GetCommandQueue()->Signal(fence, fAbort);
                    m_impl->frameFenceValues[bbIndex] = fAbort;
                    m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fAbort);
                }
                return false;
            }
            const BOOL sameNextGroup =
                (i + 1 < dispatches.size()) &&
                (dispatches[i + 1].profileGroup == dispatches[i].profileGroup);
            if (!sameNextGroup)
            {
                const UINT32 burstEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                NorthstarD3D12::Impl::AddTimestampSpan(
                    slotFlags,
                    burstStart,
                    burstEnd,
                    NorthstarD3D12::Impl::MapProfileGroupToSpanKind(
                        dispatches[i].profileGroup,
                        NorthstarD3D12::Impl::TimestampSpanKind::PRE_COMPUTE));
                burstStart = UINT32_MAX;
            }
        }

        const UINT32 tEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
        NorthstarD3D12::Impl::AddTimestampSpan(
            slotFlags,
            tStart,
            tEnd,
            NorthstarD3D12::Impl::TimestampSpanKind::FULL_FRAME);
        m_impl->FinalizeTimestampSlot(bbIndex, queryBase, slotFlags);

        m_impl->resourceMgr.FlushPendingGPUBufferReadbacks(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            bbIndex,
            upcomingFenceValue
        );

        m_impl->frameCmdList->Close();
        m_impl->isRecordingCmdList = false;
        ID3D12CommandList* lists[] = { m_impl->frameCmdList.Get() };
        m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, lists);

        m_impl->deviceContext.IncrementFrameFenceValue();
        const UINT64 fValue = m_impl->deviceContext.GetFrameFenceValue();
        m_impl->deviceContext.GetCommandQueue()->Signal(fence, fValue);
        m_impl->frameFenceValues[bbIndex] = fValue;
        m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fValue);
        const UINT64 uploadedBytes64 = m_impl->resourceMgr.ConsumeUploadedBytesThisSubmission();
        m_impl->lastFrameMetrics.uploadedBytesThisFrame        = static_cast<UINT32>((std::min)(uploadedBytes64, static_cast<UINT64>(0xFFFFFFFFull)));
        m_impl->lastFrameMetrics.gpuBufferAllocationsThisFrame = m_impl->resourceMgr.ConsumeGPUBufferAllocationsThisSubmission();
        m_impl->lastFrameMetrics.pendingRetiredResources       = m_impl->resourceMgr.GetPendingRetiredResourceCount();
        return true;
    }

    BOOL NorthstarD3D12::ExecuteLayeredDepthPasses(
        const std::vector<LayeredDepthPassView>& passes,
        const std::vector<ComputeDispatchCommand>* preDispatches)
    {
        if (!m_impl || !m_impl->isInitialized || m_impl->deviceContext.IsDeviceRemoved() || passes.empty())
        {
            return false;
        }
        m_impl->AssertOwnerThread();

        const UINT32 bbIndex = static_cast<UINT32>(m_impl->submissionSlotCursor % m_impl->maxFramesInFlight);
        m_impl->submissionSlotCursor++;
        ID3D12Fence* fence   = m_impl->deviceContext.GetFrameFence();
        HANDLE fenceEvent    = m_impl->deviceContext.GetFrameFenceEvent();

        const auto fenceWaitStart = std::chrono::high_resolution_clock::now();
        const UINT64 requiredFence = m_impl->frameFenceValues[bbIndex];
        if (requiredFence > 0 && m_impl->GetSafeCompletedFenceValue() < requiredFence)
        {
            if (!m_impl->deviceContext.WaitForFenceValue(fence, fenceEvent, requiredFence))
                return false;
        }
        const auto fenceWaitEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.fenceWaitCpuMs = std::chrono::duration<float, std::milli>(fenceWaitEnd - fenceWaitStart).count();

        m_impl->ResolveTimestampSlot(bbIndex);

        const auto cmdRecordStart = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.indirectDrawCommandsExecuted       = 0;
        m_impl->lastFrameMetrics.indirectDispatchCommandsExecuted   = 0;
        m_impl->lastFrameMetrics.indirectCommandsRejected           = 0;
        m_impl->lastFrameMetrics.layeredDepthPassesExecuted         = 0;
        m_impl->lastFrameMetrics.layeredDepthFallbackPassesExecuted = 0;
        m_impl->lastFrameMetrics.cpuPerSliceDepthClearsExecuted     = 0;

        const UINT64 completedFence = m_impl->GetSafeCompletedFenceValue();
        m_impl->resourceMgr.ProcessRetirementQueue(
            completedFence,
            m_impl->cbvSrvUavPool,
            m_impl->rtvPool,
            m_impl->dsvPool,
            m_impl->samplerPool,
            &m_impl->stateTracker
        );
        m_impl->ProcessRetiredSamplerTables(completedFence);

        if (FAILED(m_impl->frameCmdAlloc[bbIndex]->Reset())) return false;
        if (FAILED(m_impl->frameCmdList->Reset(m_impl->frameCmdAlloc[bbIndex].Get(), nullptr))) return false;
        m_impl->isRecordingCmdList = true;

        const UINT64 upcomingFenceValue = m_impl->deviceContext.GetFrameFenceValue() + 1;
        const UINT32 queryBase = bbIndex * NorthstarD3D12::Impl::kQueriesPerBackBuffer;
        NorthstarD3D12::Impl::TimestampSlotFlags slotFlags{};
        slotFlags.isStandaloneLayered = true;
        const BOOL hadUpload          = m_impl->resourceMgr.HasPendingBufferUploads();

        const UINT32 tStart = m_impl->EmitTimestamp(queryBase, slotFlags);

        const auto uploadRecordStart = std::chrono::high_resolution_clock::now();
        m_impl->resourceMgr.FlushPendingGPUBufferUploads(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            upcomingFenceValue
        );
        const auto uploadRecordEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.gpuBufferUploadRecordCpuMs =
            std::chrono::duration<float, std::milli>(uploadRecordEnd - uploadRecordStart).count();

        if (hadUpload)
        {
            const UINT32 tUploadEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
            NorthstarD3D12::Impl::AddTimestampSpan(
                slotFlags,
                tStart,
                tUploadEnd,
                NorthstarD3D12::Impl::TimestampSpanKind::UPLOAD);
        }

        ID3D12DescriptorHeap* heaps[] = {
            m_impl->cbvSrvUavPool.GetHeap(),
            m_impl->samplerPool.GetHeap()
        };
        m_impl->frameCmdList->SetDescriptorHeaps(2, heaps);
        m_impl->pipeline.BeginFrameBuffers(bbIndex, completedFence, upcomingFenceValue);
        m_impl->cachedSharedSBUploads.clear();
        m_impl->pipeline.ResetFramePSOMetrics();

        UINT32 currentTableOffset = 0;
        UINT32 tablesUsedThisFrame = 0;
        BOOL isComputeRootSigActive = false;
        ID3D12PipelineState* currentPSO = nullptr;
        ID3D12PipelineState* currentComputePSO = nullptr;
        NativeHandle currentVB = 0;
        NativeHandle currentIB = 0;
        UINT32 currentRootTable[D3D12Pipeline::kGraphicsRootParamCount];
        for (UINT32 p = 0; p < D3D12Pipeline::kGraphicsRootParamCount; ++p) currentRootTable[p] = UINT32_MAX;
        UINT32 currentSamplerTableIndex[5][NorthstarD3D12::Impl::kMaxSpaces]{};
        m_impl->ResetGraphicsSamplerCache(currentSamplerTableIndex);
        UINT32 currentComputeRootTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };
        UINT32 currentComputeSamplerTable[NorthstarD3D12::Impl::kMaxSpaces] = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
        };

        if (preDispatches != nullptr && !preDispatches->empty())
        {
            UINT32 burstStart = UINT32_MAX;
            for (SIZE_T c = 0; c < preDispatches->size(); ++c)
            {
                if (burstStart == UINT32_MAX)
                {
                    burstStart = m_impl->EmitTimestamp(queryBase, slotFlags);
                }
                if (!m_impl->RecordComputeDispatch(
                    (*preDispatches)[c],
                    bbIndex,
                    currentTableOffset,
                    tablesUsedThisFrame,
                    isComputeRootSigActive,
                    currentComputePSO,
                    currentComputeRootTable,
                    currentComputeSamplerTable))
                {
                    m_impl->frameCmdList->Close();
                    m_impl->isRecordingCmdList = false;
                    if (hadUpload || c > 0)
                    {
                        ID3D12CommandList* abortLists[] = { m_impl->frameCmdList.Get() };
                        m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, abortLists);
                        m_impl->deviceContext.IncrementFrameFenceValue();
                        const UINT64 fAbort = m_impl->deviceContext.GetFrameFenceValue();
                        m_impl->deviceContext.GetCommandQueue()->Signal(fence, fAbort);
                        m_impl->frameFenceValues[bbIndex] = fAbort;
                        m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fAbort);
                    }
                    return false;
                }
                const BOOL sameNextGroup =
                    (c + 1 < preDispatches->size()) &&
                    ((*preDispatches)[c + 1].profileGroup == (*preDispatches)[c].profileGroup);
                if (!sameNextGroup)
                {
                    const UINT32 burstEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
                    NorthstarD3D12::Impl::AddTimestampSpan(
                        slotFlags,
                        burstStart,
                        burstEnd,
                        NorthstarD3D12::Impl::MapProfileGroupToSpanKind(
                            (*preDispatches)[c].profileGroup,
                            NorthstarD3D12::Impl::TimestampSpanKind::PRE_COMPUTE));
                    burstStart = UINT32_MAX;
                }
            }
        }

        m_impl->BindGraphicsRootSigAndFallbacks();
        isComputeRootSigActive = false;

        const Mat4 identityMat{};
        for (SIZE_T p = 0; p < passes.size(); ++p)
        {
            const BOOL isShadowCasterPass =
                !passes[p].drawCommands.empty() && !passes[p].selectiveClear.enabled;
            const UINT32 tPassStart = m_impl->EmitTimestamp(queryBase, slotFlags);
            if (!m_impl->RecordLayeredDepthPass(
                passes[p],
                identityMat,
                bbIndex,
                currentTableOffset,
                tablesUsedThisFrame,
                isComputeRootSigActive,
                currentPSO,
                currentComputePSO,
                currentVB,
                currentIB,
                currentRootTable,
                currentSamplerTableIndex))
            {
                m_impl->frameCmdList->Close();
                m_impl->isRecordingCmdList = false;
                if (hadUpload || p > 0 || (preDispatches != nullptr && !preDispatches->empty()))
                {
                    ID3D12CommandList* abortLists[] = { m_impl->frameCmdList.Get() };
                    m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, abortLists);
                    m_impl->deviceContext.IncrementFrameFenceValue();
                    const UINT64 fAbort = m_impl->deviceContext.GetFrameFenceValue();
                    m_impl->deviceContext.GetCommandQueue()->Signal(fence, fAbort);
                    m_impl->frameFenceValues[bbIndex] = fAbort;
                    m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fAbort);
                }
                return false;
            }
            const UINT32 tPassEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
            NorthstarD3D12::Impl::AddTimestampSpan(
                slotFlags,
                tPassStart,
                tPassEnd,
                isShadowCasterPass
                    ? NorthstarD3D12::Impl::TimestampSpanKind::SHADOW_RASTER
                    : NorthstarD3D12::Impl::TimestampSpanKind::PRE_LAYERED_DEPTH);
        }

        const UINT32 tEnd = m_impl->EmitTimestamp(queryBase, slotFlags);
        NorthstarD3D12::Impl::AddTimestampSpan(
            slotFlags,
            tStart,
            tEnd,
            NorthstarD3D12::Impl::TimestampSpanKind::FULL_FRAME);
        m_impl->FinalizeTimestampSlot(bbIndex, queryBase, slotFlags);

        m_impl->resourceMgr.FlushPendingGPUBufferReadbacks(
            m_impl->frameCmdList.Get(),
            m_impl->stateTracker,
            bbIndex,
            upcomingFenceValue
        );

        m_impl->frameCmdList->Close();
        m_impl->isRecordingCmdList = false;
        const auto cmdRecordEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.cmdRecordCpuMs       = std::chrono::duration<float, std::milli>(cmdRecordEnd - cmdRecordStart).count();
        m_impl->lastFrameMetrics.descriptorTablesUsed = tablesUsedThisFrame;

        const auto cmdSubmitStart = std::chrono::high_resolution_clock::now();
        ID3D12CommandList* lists[] = { m_impl->frameCmdList.Get() };
        m_impl->deviceContext.GetCommandQueue()->ExecuteCommandLists(1, lists);

        m_impl->deviceContext.IncrementFrameFenceValue();
        const UINT64 fValue = m_impl->deviceContext.GetFrameFenceValue();
        m_impl->deviceContext.GetCommandQueue()->Signal(fence, fValue);
        m_impl->frameFenceValues[bbIndex] = fValue;
        m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(m_impl->stateTracker, fValue);
        const auto cmdSubmitEnd = std::chrono::high_resolution_clock::now();
        m_impl->lastFrameMetrics.cmdSubmitCpuMs = std::chrono::duration<float, std::milli>(cmdSubmitEnd - cmdSubmitStart).count();
        const UINT64 uploadedBytes64 = m_impl->resourceMgr.ConsumeUploadedBytesThisSubmission();
        m_impl->lastFrameMetrics.uploadedBytesThisFrame        = static_cast<UINT32>((std::min)(uploadedBytes64, static_cast<UINT64>(0xFFFFFFFFull)));
        m_impl->lastFrameMetrics.gpuBufferAllocationsThisFrame = m_impl->resourceMgr.ConsumeGPUBufferAllocationsThisSubmission();
        m_impl->lastFrameMetrics.pendingRetiredResources       = m_impl->resourceMgr.GetPendingRetiredResourceCount();
        m_impl->lastFrameMetrics.psoCacheHits                  = m_impl->pipeline.GetPSOCacheHits();
        m_impl->lastFrameMetrics.psoCacheMisses                = m_impl->pipeline.GetPSOCacheMisses();
        m_impl->lastFrameMetrics.activeCachedPSOCount          = m_impl->pipeline.GetActiveCachedPSOCount();
        m_impl->lastFrameMetrics.retiredPSOCount               = m_impl->pipeline.GetRetiredPSOCount();
        return true;
    }

    BOOL NorthstarD3D12::SupportsVPAndRTArrayIndexWithoutGS() const
    {
        return (m_impl && m_impl->isInitialized) ? m_impl->deviceContext.SupportsVPAndRTArrayIndexWithoutGS() : false;
    }

    void NorthstarD3D12::SetForceLayeredDepthFallback(BOOL forceFallback)
    {
        if (m_impl)
        {
            m_impl->forceLayeredDepthFallback = (forceFallback != FALSE);
            if (!m_impl->forceLayeredDepthFallback)
            {
                m_impl->loggedLayeredDepthFallbackOnce = false;
            }
        }
    }

    BOOL NorthstarD3D12::IsForceLayeredDepthFallback() const
    {
        return m_impl ? m_impl->forceLayeredDepthFallback : false;
    }

    BackendFrameMetrics NorthstarD3D12::GetLastFrameMetrics() const
    {
        if (!m_impl) return BackendFrameMetrics{};
        BackendFrameMetrics m = m_impl->lastFrameMetrics;
        m.psoCacheHits         = m_impl->pipeline.GetPSOCacheHits();
        m.psoCacheMisses       = m_impl->pipeline.GetPSOCacheMisses();
        m.activeCachedPSOCount = m_impl->pipeline.GetActiveCachedPSOCount();
        m.retiredPSOCount      = m_impl->pipeline.GetRetiredPSOCount();
        return m;
    }

    void NorthstarD3D12::Present()
    {
        if (m_impl && m_impl->swapChain.IsValid())
        {
            if (m_impl->swapChain.GetHwnd() && !::IsWindow(m_impl->swapChain.GetHwnd())) return;
            m_impl->swapChain.Present(m_impl->deviceContext.GetDevice());
        }
    }

    void NorthstarD3D12::SetVSync(BOOL enabled)
    {
        if (m_impl)
        {
            m_impl->swapChain.SetVSync(enabled);
        }
    }

    BOOL NorthstarD3D12::IsVSyncEnabled() const
    {
        return m_impl ? m_impl->swapChain.IsVSyncEnabled() : true;
    }

    void NorthstarD3D12::SetMaxFramesInFlight(UINT32 maxFramesInFlight)
    {
        if (!m_impl) return;
        const UINT32 clamped = (std::max)(1u, (std::min)(maxFramesInFlight, NorthstarD3D12::Impl::kMaxFramesInFlight));
        if (m_impl->maxFramesInFlight != clamped)
        {
            if (m_impl->isInitialized)
            {
                WaitForGPU();
            }
            m_impl->maxFramesInFlight = clamped;
        }
    }

    UINT32 NorthstarD3D12::GetMaxFramesInFlight() const
    {
        return m_impl ? m_impl->maxFramesInFlight : 2u;
    }

    void NorthstarD3D12::OnResize(UINT32 width, UINT32 height)
    {
        if (!m_impl || !m_impl->isInitialized || m_impl->deviceContext.IsDeviceRemoved()) return;
        if (width == 0 || height == 0) return;
        if (m_impl->swapChain.GetHwnd() && !::IsWindow(m_impl->swapChain.GetHwnd())) return;

        WaitForGPU();

        m_impl->swapChain.Resize(
            m_impl->deviceContext.GetDevice(),
            m_impl->rtvPool,
            m_impl->dsvPool,
            width,
            height
        );
    }

    void NorthstarD3D12::WaitForGPU()
    {
        if (m_impl && m_impl->isInitialized)
        {
            m_impl->deviceContext.WaitForGPU();
            const UINT32 slotCount = (m_impl->maxFramesInFlight > 0u)
                ? m_impl->maxFramesInFlight
                : NorthstarD3D12::Impl::kMaxFramesInFlight;
            for (UINT32 b = 0; b < slotCount; ++b)
            {
                const UINT32 orderedSlot = static_cast<UINT32>((m_impl->submissionSlotCursor + b) % slotCount);
                m_impl->ResolveTimestampSlot(orderedSlot);
            }
            if (m_impl->deviceContext.GetFrameFence())
            {
                const UINT64 completed = m_impl->GetSafeCompletedFenceValue();
                m_impl->resourceMgr.ProcessRetirementQueue(
                    completed,
                    m_impl->cbvSrvUavPool,
                    m_impl->rtvPool,
                    m_impl->dsvPool,
                    m_impl->samplerPool,
                    &m_impl->stateTracker
                );
                m_impl->ProcessRetiredSamplerTables(completed);
            }
            m_impl->pipeline.FlushAllRetiredResources();
        }
    }

    BOOL NorthstarD3D12::EnqueueExternalFenceWait(void* sharedFenceHandle, UINT64 fenceValue)
    {
        if (!m_impl || !m_impl->isInitialized || !sharedFenceHandle || m_impl->deviceContext.IsDeviceRemoved())
        {
            return false;
        }
        ID3D12Device* device      = m_impl->deviceContext.GetDevice();
        ID3D12CommandQueue* queue = m_impl->deviceContext.GetCommandQueue();
        if (!device || !queue)
        {
            return false;
        }

        ComPtr<ID3D12Fence> extFence;
        auto it = m_impl->importedSharedFences.find(sharedFenceHandle);
        if (it != m_impl->importedSharedFences.end() && it->second)
        {
            extFence = it->second;
        }
        else
        {
            if (FAILED(device->OpenSharedHandle(static_cast<HANDLE>(sharedFenceHandle), IID_PPV_ARGS(&extFence))) || !extFence)
            {
                return false;
            }
            m_impl->importedSharedFences[sharedFenceHandle] = extFence;
        }

        return SUCCEEDED(queue->Wait(extFence.Get(), fenceValue)) ? true : false;
    }

    void NorthstarD3D12::FlushDeferredRetirements()
    {
        if (m_impl && m_impl->isInitialized)
        {
            m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(
                m_impl->stateTracker,
                m_impl->deviceContext.GetFrameFenceValue()
            );
            m_impl->deviceContext.WaitForGPU();
            if (m_impl->deviceContext.GetFrameFence())
            {
                const UINT64 completed = m_impl->GetSafeCompletedFenceValue();
                m_impl->resourceMgr.ProcessRetirementQueue(
                    completed,
                    m_impl->cbvSrvUavPool,
                    m_impl->rtvPool,
                    m_impl->dsvPool,
                    m_impl->samplerPool,
                    &m_impl->stateTracker
                );
                m_impl->ProcessRetiredSamplerTables(completed);
            }
            m_impl->pipeline.FlushAllRetiredResources();
        }
    }

    void NorthstarD3D12::FlushSpecificBufferRetirements(const NativeHandle* bufferHandles, UINT32 handleCount)
    {
        if (m_impl && m_impl->isInitialized)
        {
            if (bufferHandles != nullptr && handleCount > 0)
            {
                m_impl->resourceMgr.FinalizePendingDestroyedGPUBuffers(
                    m_impl->stateTracker,
                    m_impl->deviceContext.GetFrameFenceValue(),
                    bufferHandles,
                    handleCount
                );
            }
            m_impl->deviceContext.WaitForGPU();
            if (m_impl->deviceContext.GetFrameFence())
            {
                const UINT64 completed = m_impl->GetSafeCompletedFenceValue();
                m_impl->resourceMgr.ProcessRetirementQueue(
                    completed,
                    m_impl->cbvSrvUavPool,
                    m_impl->rtvPool,
                    m_impl->dsvPool,
                    m_impl->samplerPool,
                    &m_impl->stateTracker
                );
                m_impl->ProcessRetiredSamplerTables(completed);
            }
        }
    }

    NativeContext NorthstarD3D12::GetNativeContext() const
    {
        NativeContext ctx{};
        if (m_impl && m_impl->isInitialized)
        {
            ctx.nativeDevice = m_impl->deviceContext.GetDevice();
            ctx.nativeCommandQueue = m_impl->deviceContext.GetCommandQueue();
            ctx.nativeDescriptorHeap = m_impl->cbvSrvUavPool.GetHeap();
            if (m_impl->deviceContext.GetDevice())
            {
                ctx.descriptorSize = m_impl->deviceContext.GetDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
            ctx.backBufferCount = D3D12SwapChain::kBackBufferCount;
            ctx.renderTargetFormat = 28; // DXGI_FORMAT_R8G8B8A8_UNORM
        }
        return ctx;
    }

    void NorthstarD3D12::SetPrePresentCallback(PrePresentCallback cb)
    {
        if (m_impl)
        {
            m_impl->prePresentCallback = std::move(cb);
        }
    }

    void NorthstarD3D12::SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU)
    {
        if (m_impl)
        {
            m_impl->postFrameExecutionCallback = std::move(cb);
            m_impl->postFrameWaitForGPU        = waitForGPU;
        }
    }

    // ==================== RESOURCE MANAGEMENT DELEGATION ====================
    NativeHandle NorthstarD3D12::CreateVertexBuffer(const void* data, UINT32 byteSize, UINT32 stride)
    {
        return m_impl ? m_impl->resourceMgr.CreateVertexBuffer(m_impl->deviceContext, data, byteSize, stride) : 0;
    }

    void NorthstarD3D12::DestroyVertexBuffer(NativeHandle handle)
    {
        if (m_impl) m_impl->resourceMgr.DestroyVertexBuffer(handle, m_impl->GetSafeRetireFenceValue());
    }

    NativeHandle NorthstarD3D12::CreateIndexBuffer(const void* data, UINT32 byteSize, BOOL is32Bit)
    {
        return m_impl ? m_impl->resourceMgr.CreateIndexBuffer(m_impl->deviceContext, data, byteSize, is32Bit) : 0;
    }

    void NorthstarD3D12::DestroyIndexBuffer(NativeHandle handle)
    {
        if (m_impl) m_impl->resourceMgr.DestroyIndexBuffer(handle, m_impl->GetSafeRetireFenceValue());
    }

    BOOL NorthstarD3D12::GetGeometryBufferGPUViews(
        NativeHandle            vertexBufferHandle,
        NativeHandle            indexBufferHandle,
        GeometryBufferGPUViews& outViews) const
    {
        outViews = {};
        if (!m_impl)
        {
            return false;
        }

        const auto* pVb = m_impl->resourceMgr.GetVertexBuffer(vertexBufferHandle);
        const auto* pIb = m_impl->resourceMgr.GetIndexBuffer(indexBufferHandle);
        if (!pVb || !pVb->resource || !pIb || !pIb->resource)
        {
            return false;
        }

        outViews.vbGPUVirtualAddress = static_cast<UINT64>(pVb->view.BufferLocation);
        outViews.vbSizeInBytes       = static_cast<UINT32>(pVb->view.SizeInBytes);
        outViews.vbStrideInBytes     = static_cast<UINT32>(pVb->view.StrideInBytes);
        outViews.ibGPUVirtualAddress = static_cast<UINT64>(pIb->view.BufferLocation);
        outViews.ibSizeInBytes       = static_cast<UINT32>(pIb->view.SizeInBytes);
        outViews.ibFormat            = static_cast<UINT32>(pIb->view.Format);
        outViews.valid               = (outViews.vbGPUVirtualAddress != 0 &&
                                        outViews.vbSizeInBytes > 0 &&
                                        outViews.vbStrideInBytes > 0 &&
                                        outViews.ibGPUVirtualAddress != 0 &&
                                        outViews.ibSizeInBytes > 0);
        return outViews.valid;
    }

    NativeHandle NorthstarD3D12::CreateConstantBuffer(UINT32 byteSize)
    {
        return m_impl ? m_impl->resourceMgr.CreateConstantBuffer(m_impl->deviceContext, m_impl->cbvSrvUavPool, byteSize) : 0;
    }

    void NorthstarD3D12::UpdateConstantBuffer(NativeHandle handle, const void* data, UINT32 byteSize)
    {
        if (m_impl) m_impl->resourceMgr.UpdateConstantBuffer(handle, data, byteSize);
    }

    void NorthstarD3D12::DestroyConstantBuffer(NativeHandle handle)
    {
        if (m_impl) m_impl->resourceMgr.DestroyConstantBuffer(handle, m_impl->cbvSrvUavPool, m_impl->GetSafeRetireFenceValue());
    }

    NativeHandle NorthstarD3D12::CreateTexture(const TextureDesc& desc, const void* initialData)
    {
        if (!m_impl) return 0;
        const NativeHandle h = m_impl->resourceMgr.CreateTexture(
            m_impl->deviceContext, m_impl->cbvSrvUavPool, m_impl->rtvPool, m_impl->dsvPool, desc, initialData);
        if (h != 0)
        {
            if (const D3D12TextureResource* tex = m_impl->resourceMgr.GetTexture(h))
            {
                if (tex->resource)
                {
                    NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, tex);
                }
            }
        }
        return h;
    }

    BOOL NorthstarD3D12::UpdateTextureSubresource(
        NativeHandle handle,
        UINT32       mipSlice,
        UINT32       arraySlice,
        const void*  data,
        SIZE_T       dataByteSize)
    {
        if (!m_impl || !m_impl->isInitialized) return false;
        return m_impl->resourceMgr.UpdateTextureSubresource(
            m_impl->deviceContext, m_impl->stateTracker, handle, mipSlice, arraySlice, data, dataByteSize);
    }

    BOOL NorthstarD3D12::GenerateTextureMips(NativeHandle handle)
    {
        if (!m_impl || !m_impl->isInitialized) return false;
        return m_impl->resourceMgr.GenerateTextureMips(m_impl->deviceContext, m_impl->stateTracker, handle);
    }

    void NorthstarD3D12::DestroyTexture(NativeHandle handle)
    {
        if (m_impl)
        {
            m_impl->resourceMgr.DestroyTexture(
                handle, m_impl->cbvSrvUavPool, m_impl->rtvPool, m_impl->dsvPool, m_impl->GetSafeRetireFenceValue());
        }
    }

    void* NorthstarD3D12::GetTextureGPUDescriptor(NativeHandle handle) const
    {
        if (!m_impl) return nullptr;
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_impl->resourceMgr.GetTextureGPUDescriptorHandle(handle, m_impl->cbvSrvUavPool);
        return reinterpret_cast<void*>(gpuHandle.ptr);
    }

    void* NorthstarD3D12::GetTextureSharedHandle(NativeHandle handle) const
    {
        if (!m_impl) return nullptr;
        return m_impl->resourceMgr.GetTextureSharedHandle(handle);
    }

    SharedHandleType NorthstarD3D12::GetTextureSharedHandleType(NativeHandle handle) const
    {
        if (!m_impl) return SharedHandleType::NONE;
        return m_impl->resourceMgr.GetTextureSharedHandle(handle) ? SharedHandleType::WIN32_NT_HANDLE
                                                                  : SharedHandleType::NONE;
    }

    NativeHandle NorthstarD3D12::CreateSampler(const SamplerDesc& desc)
    {
        return m_impl ? m_impl->resourceMgr.CreateSampler(m_impl->deviceContext, m_impl->samplerPool, desc) : 0;
    }

    void NorthstarD3D12::DestroySampler(NativeHandle handle)
    {
        if (m_impl)
        {
            const UINT64 retireFence = m_impl->GetSafeRetireFenceValue();
            m_impl->resourceMgr.DestroySampler(handle, m_impl->samplerPool, retireFence);
            for (auto it = m_impl->samplerTableCache.begin(); it != m_impl->samplerTableCache.end();)
            {
                bool usesHandle = false;
                for (UINT32 s = 0; s < 16; ++s)
                {
                    if (it->first.samplers[s] == handle)
                    {
                        usesHandle = true;
                        break;
                    }
                }
                if (usesHandle)
                {
                    if (it->second != UINT32_MAX && it->second != m_impl->defaultDummySamplerTableIndex)
                    {
                        m_impl->retiredSamplerTables.push_back({ retireFence, it->second });
                    }
                    it = m_impl->samplerTableCache.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
    }

    NativeHandle NorthstarD3D12::CreateShader(ShaderStage stage, const void* bytecode, SIZE_T bytecodeSize)
    {
        return m_impl ? m_impl->resourceMgr.CreateShader(stage, bytecode, bytecodeSize) : 0;
    }

    BOOL NorthstarD3D12::GetShaderCompiledBytecode(NativeHandle handle, std::vector<BYTE>& outBytecode) const
    {
        return m_impl ? m_impl->resourceMgr.GetShaderCompiledBytecode(handle, outBytecode) : false;
    }

    void NorthstarD3D12::SetShaderInputLayout(NativeHandle handle, const std::vector<NSCStageIOEntry>& inputLayout)
    {
        if (m_impl) m_impl->resourceMgr.SetShaderInputLayout(handle, inputLayout);
    }

    void NorthstarD3D12::DestroyShader(NativeHandle handle)
    {
        if (m_impl)
        {
            m_impl->pipeline.InvalidatePSOsForShader(handle, m_impl->GetSafeRetireFenceValue());
            m_impl->resourceMgr.DestroyShader(handle);
        }
    }

    // ==================== PERSISTENT GPU BUFFER DELEGATION ====================
    NativeHandle NorthstarD3D12::CreateGPUBuffer(const GPUBufferDesc& desc)
    {
        return m_impl ? m_impl->resourceMgr.CreateGPUBuffer(
            m_impl->deviceContext, m_impl->cbvSrvUavPool, m_impl->stateTracker, desc) : 0;
    }

    BOOL NorthstarD3D12::UpdateGPUBuffer(NativeHandle handle, const void* data, UINT32 byteSize, UINT32 dstOffset)
    {
        return m_impl ? m_impl->resourceMgr.UpdateGPUBuffer(
            m_impl->deviceContext, handle, data, byteSize, dstOffset) : false;
    }

    UINT64 NorthstarD3D12::GetGPUBufferGPUVirtualAddress(NativeHandle handle) const
    {
        if (!m_impl || handle == 0) return 0;
        const D3D12GPUBufferResource* buf = m_impl->resourceMgr.GetGPUBuffer(handle);
        if (!buf || !buf->resource || buf->pendingDestroy) return 0;
        return static_cast<UINT64>(buf->resource->GetGPUVirtualAddress());
    }

    BOOL NorthstarD3D12::RequestGPUBufferReadback(NativeHandle handle, UINT32 srcOffset, UINT32 byteSize)
    {
        return m_impl ? m_impl->resourceMgr.RequestGPUBufferReadback(m_impl->deviceContext, handle, srcOffset, byteSize) : false;
    }

    BOOL NorthstarD3D12::TryGetGPUBufferReadbackData(NativeHandle handle, void* outData, UINT32 byteSize)
    {
        return m_impl ? m_impl->resourceMgr.TryGetGPUBufferReadbackData(
            m_impl->deviceContext, handle, outData, byteSize) : false;
    }

    BOOL NorthstarD3D12::ReadbackGPUBufferSync(NativeHandle handle, void* outData, UINT32 byteSize)
    {
        return m_impl ? m_impl->resourceMgr.ReadbackGPUBufferSync(
            m_impl->deviceContext, m_impl->stateTracker, handle, outData, byteSize) : false;
    }

    BOOL NorthstarD3D12::ReadbackTextureSync(NativeHandle handle, void* outData, UINT32 byteSize)
    {
        return m_impl ? m_impl->resourceMgr.ReadbackTextureSync(
            m_impl->deviceContext, m_impl->stateTracker, handle, outData, byteSize) : false;
    }

    BOOL NorthstarD3D12::ReadbackTextureSubresourceSync(
        NativeHandle handle,
        UINT32       mipSlice,
        UINT32       arraySlice,
        void*        outData,
        UINT32       byteSize)
    {
        return m_impl ? m_impl->resourceMgr.ReadbackTextureSubresourceSync(
            m_impl->deviceContext, m_impl->stateTracker, handle, mipSlice, arraySlice, outData, byteSize) : false;
    }

    void NorthstarD3D12::DestroyGPUBuffer(NativeHandle handle)
    {
        if (m_impl)
        {
            m_impl->resourceMgr.DestroyGPUBuffer(
                handle, m_impl->stateTracker, m_impl->GetSafeRetireFenceValue());
        }
    }

    // ==================== GPU VALIDATION & UPLOAD RING DIAGNOSTICS ====================
    void NorthstarD3D12::SetValidationConfig(BOOL enableDebugLayer, BOOL enableGPUBasedValidation)
    {
        if (m_impl)
        {
            m_impl->deviceContext.SetValidationConfig(enableDebugLayer, enableGPUBasedValidation);
        }
    }

    BOOL NorthstarD3D12::IsDebugLayerActive() const
    {
        return m_impl ? m_impl->deviceContext.IsDebugLayerActive() : false;
    }

    BOOL NorthstarD3D12::IsGPUBasedValidationActive() const
    {
        return m_impl ? m_impl->deviceContext.IsGPUBasedValidationActive() : false;
    }

    UINT32 NorthstarD3D12::GetDebugErrorCount() const
    {
        return m_impl ? m_impl->deviceContext.GetDebugErrorCount() : 0;
    }

    void NorthstarD3D12::ClearDebugMessages()
    {
        if (m_impl)
        {
            m_impl->deviceContext.ClearDebugMessages();
        }
    }

    UINT32 NorthstarD3D12::GetInFlightUploadRingRegionCount() const
    {
        return m_impl ? m_impl->resourceMgr.GetInFlightUploadRingRegionCount() : 0;
    }

    UINT32 NorthstarD3D12::GetUnsubmittedUploadRingRegionCount() const
    {
        return m_impl ? m_impl->resourceMgr.GetUnsubmittedUploadRingRegionCount() : 0;
    }

    // ==================== RESOURCE STATE & BARRIER DIAGNOSTICS ====================
    BOOL NorthstarD3D12::TransitionGPUBuffer(NativeHandle handle, GPUResourceAccess targetAccess)
    {
        if (!m_impl) return false;
        D3D12GPUBufferResource* buf = m_impl->resourceMgr.GetGPUBuffer(handle);
        if (!buf || !buf->resource) return false;

        const BOOL ok = m_impl->stateTracker.TransitionResource(buf->resource.Get(), targetAccess);
        buf->currentState = m_impl->stateTracker.GetResourceState(buf->resource.Get());
        return ok;
    }

    BOOL NorthstarD3D12::TransitionTexture(NativeHandle handle, GPUResourceAccess targetAccess)
    {
        if (!m_impl) return false;
        D3D12TextureResource* tex = m_impl->resourceMgr.GetTexture(handle);
        if (!tex || !tex->resource) return false;

        if (!m_impl->stateTracker.IsResourceTracked(tex->resource.Get()))
        {
            NorthstarD3D12::Impl::RegisterTrackedTexture(m_impl->stateTracker, tex);
        }

        const BOOL ok = m_impl->stateTracker.TransitionResource(tex->resource.Get(), targetAccess);
        tex->currentState = m_impl->stateTracker.GetResourceState(tex->resource.Get());
        return ok;
    }

    UINT32 NorthstarD3D12::FlushResourceBarriers(BOOL executeOnGPU)
    {
        if (!m_impl) return 0;
        const UINT32 pending = m_impl->stateTracker.GetPendingBarrierCount();
        if (pending == 0) return 0;

        if (executeOnGPU)
        {
            if (m_impl->isRecordingCmdList && m_impl->frameCmdList)
            {
                return m_impl->stateTracker.FlushBarriers(m_impl->frameCmdList.Get());
            }
            UINT32 flushed = 0;
            m_impl->deviceContext.ExecuteUpload([&](ID3D12GraphicsCommandList* cmdList) {
                flushed = m_impl->stateTracker.FlushBarriers(cmdList);
            });
            return flushed;
        }

        // Non-destructive inspection when executeOnGPU == false: never drop queued GPU barriers without recording them!
        return pending;
    }

    GPUResourceAccess NorthstarD3D12::GetGPUBufferAccessState(NativeHandle handle) const
    {
        if (!m_impl) return GPUResourceAccess::COMMON;
        const D3D12GPUBufferResource* buf = m_impl->resourceMgr.GetGPUBuffer(handle);
        if (!buf || !buf->resource) return GPUResourceAccess::COMMON;
        return m_impl->stateTracker.GetResourceAccess(buf->resource.Get());
    }

    ResourceBarrierStats NorthstarD3D12::GetResourceBarrierStats() const
    {
        return m_impl ? m_impl->stateTracker.GetStats() : ResourceBarrierStats{};
    }

    void NorthstarD3D12::ResetResourceBarrierStats()
    {
        if (m_impl) m_impl->stateTracker.ResetStats();
    }

    // ==================== INSPECTION HELPERS ====================
    BOOL NorthstarD3D12::HasVertexBuffer(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasVertexBuffer(handle);
    }

    BOOL NorthstarD3D12::HasIndexBuffer(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasIndexBuffer(handle);
    }

    BOOL NorthstarD3D12::HasConstantBuffer(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasConstantBuffer(handle);
    }

    BOOL NorthstarD3D12::HasTexture(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasTexture(handle);
    }

    BOOL NorthstarD3D12::HasSampler(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasSampler(handle);
    }

    BOOL NorthstarD3D12::HasShader(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasShader(handle);
    }

    BOOL NorthstarD3D12::HasGPUBuffer(NativeHandle handle) const
    {
        return m_impl && m_impl->resourceMgr.HasGPUBuffer(handle);
    }

    UINT32 NorthstarD3D12::GetPendingRetiredResourceCount() const
    {
        return m_impl ? m_impl->resourceMgr.GetPendingRetiredResourceCount() : 0;
    }

    UINT32 NorthstarD3D12::GetActiveResourceCount() const
    {
        return m_impl ? m_impl->resourceMgr.GetActiveResourceCount() : 0;
    }
}

#include "api/northstar_engine.h"

// ==================== DYNAMIC MODULE BOUNDARY ====================
Northstar::UINT32 Northstar_GetAPIVersion()
{
    return Northstar::NORTHSTAR_ENGINE_API_VERSION;
}

Northstar::NorthstarBackend* CreateNorthstarD3D12Backend()
{
    return new Northstar::NorthstarD3D12();
}

void DestroyNorthstarD3D12Backend(Northstar::NorthstarBackend* backend)
{
    if (backend)
    {
        delete backend;
    }
}
