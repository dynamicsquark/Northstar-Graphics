#pragma once

#include "d3d12_common.h"
#include "d3d12_device.h"
#include "types/northstar_types.h"
#include "types/northstar_math.h"
#include "types/pipeline_state_types.h"
#include "buffer/gpu_indirect_args.h"
#include "platform/frame_view.h"
#include "d3d12_resource_manager.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace Northstar
{

    struct D3D12InstancePayload
    {
        Mat4   worldMatrix;
        UINT32 instanceId = 0;
        UINT32 customData = 0;
    };

    struct IndirectRootBindingKeyHash
    {
        SIZE_T operator()(const IndirectRootBindingKey& k) const noexcept
        {
            const UINT8 effSrv = (k.srvCount > 0u) ? k.srvCount : (k.hasPromotedSRV ? 1u : 0u);
            SIZE_T h = std::hash<bool>{}(k.enabled != FALSE);
            h ^= std::hash<bool>{}(k.extended != FALSE) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT32>{}(k.cbvCount) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT32>{}(effSrv) + 0x9e3779b9 + (h << 6) + (h >> 2);
            if (k.enabled)
            {
                const UINT8 maxCbv = k.extended ? 8u : 3u;
                for (UINT8 i = 0; i < k.cbvCount && i < maxCbv; ++i)
                {
                    h ^= std::hash<UINT16>{}(k.cbvSpace[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<UINT16>{}(k.cbvSlot[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int>{}(static_cast<int>(k.cbvStage[i])) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                if (effSrv >= 1u)
                {
                    h ^= std::hash<UINT16>{}(k.srvSpace) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<UINT16>{}(k.srvSlot) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int>{}(static_cast<int>(k.srvStage)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                if (effSrv >= 2u)
                {
                    h ^= std::hash<UINT16>{}(k.srv2Space) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<UINT16>{}(k.srv2Slot) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int>{}(static_cast<int>(k.srv2Stage)) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
            }
            return h;
        }
    };

    struct PSOCacheKey
    {
        NativeHandle                   vsHandle              = 0;
        NativeHandle                   psHandle              = 0;
        NativeHandle                   gsHandle              = 0;
        NativeHandle                   hsHandle              = 0;
        NativeHandle                   dsHandle              = 0;
        UINT64                         shaderGenerationHash  = 0;
        UINT64                         inputLayoutHash       = 0;
        CullMode                       cullMode              = CullMode::BACK;
        BlendMode                      blendMode             = BlendMode::OPAQUE;
        FillMode                       fillMode              = FillMode::SOLID;
        BOOL                           depthTest             = true;
        BOOL                           depthWrite            = true;
        DepthCompare                   depthCompare          = DepthCompare::LESS_EQUAL;
        BOOL                           shadowDepthBias       = false;
        BOOL                           depthOnlyTarget       = false;
        INT32                          depthBias             = 0;
        FLOAT32                        slopeScaledDepthBias  = 0.0f;
        FLOAT32                        depthBiasClamp        = 0.0f;
        D3D12_PRIMITIVE_TOPOLOGY_TYPE  primitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        DXGI_FORMAT                    rtvFormat0            = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT                    dsvFormat             = DXGI_FORMAT_D32_FLOAT;
        IndirectRootBindingKey         indirectRootKey{};

        BOOL operator==(const PSOCacheKey& o) const
        {
            return vsHandle              == o.vsHandle &&
                   psHandle              == o.psHandle &&
                   gsHandle              == o.gsHandle &&
                   hsHandle              == o.hsHandle &&
                   dsHandle              == o.dsHandle &&
                   shaderGenerationHash  == o.shaderGenerationHash &&
                   inputLayoutHash       == o.inputLayoutHash &&
                   cullMode              == o.cullMode &&
                   blendMode             == o.blendMode &&
                   fillMode              == o.fillMode &&
                   depthTest             == o.depthTest &&
                   depthWrite            == o.depthWrite &&
                   depthCompare          == o.depthCompare &&
                   shadowDepthBias       == o.shadowDepthBias &&
                   depthOnlyTarget       == o.depthOnlyTarget &&
                   depthBias             == o.depthBias &&
                   slopeScaledDepthBias  == o.slopeScaledDepthBias &&
                   depthBiasClamp        == o.depthBiasClamp &&
                   primitiveTopologyType == o.primitiveTopologyType &&
                   rtvFormat0            == o.rtvFormat0 &&
                   dsvFormat             == o.dsvFormat &&
                   indirectRootKey       == o.indirectRootKey;
        }
    };

    struct PSOCacheKeyHash
    {
        SIZE_T operator()(const PSOCacheKey& k) const
        {
            SIZE_T h = std::hash<UINT64>{}(k.vsHandle);
            h ^= std::hash<UINT64>{}(k.psHandle) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT64>{}(k.gsHandle) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT64>{}(k.hsHandle) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT64>{}(k.dsHandle) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT64>{}(k.shaderGenerationHash) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<UINT64>{}(k.inputLayoutHash) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.cullMode)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.blendMode)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.fillMode)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>{}(k.depthTest != FALSE) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>{}(k.depthWrite != FALSE) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.depthCompare)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>{}(k.shadowDepthBias != FALSE) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>{}(k.depthOnlyTarget != FALSE) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.depthBias) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<float>{}(k.slopeScaledDepthBias) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<float>{}(k.depthBiasClamp) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.primitiveTopologyType)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.rtvFormat0)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(static_cast<int>(k.dsvFormat)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= IndirectRootBindingKeyHash{}(k.indirectRootKey) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct GraphicsRootLayoutKey
    {
        UINT16                 cbvSrvMask[5]  = { 0x000Fu, 0x000Fu, 0x000Fu, 0x000Fu, 0x000Fu };
        UINT16                 samplerMask[5] = { 0x000Fu, 0x000Fu, 0x000Fu, 0x000Fu, 0x000Fu };
        UINT16                 uavMask        = 0x000Fu;
        IndirectRootBindingKey indirectKey{};

        bool operator==(const GraphicsRootLayoutKey& o) const noexcept
        {
            for (UINT32 i = 0; i < 5u; ++i)
            {
                if (cbvSrvMask[i] != o.cbvSrvMask[i] || samplerMask[i] != o.samplerMask[i])
                    return false;
            }
            return uavMask == o.uavMask && (indirectKey == o.indirectKey) != FALSE;
        }
    };

    struct GraphicsRootLayoutKeyHash
    {
        SIZE_T operator()(const GraphicsRootLayoutKey& k) const noexcept
        {
            SIZE_T h = std::hash<UINT16>{}(k.uavMask);
            for (UINT32 i = 0; i < 5u; ++i)
            {
                h ^= std::hash<UINT16>{}(k.cbvSrvMask[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<UINT16>{}(k.samplerMask[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            h ^= IndirectRootBindingKeyHash{}(k.indirectKey) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct GraphicsRootLayout
    {
        ComPtr<ID3D12RootSignature>    rootSignature;
        ComPtr<ID3D12CommandSignature> indirectCommandSignature;
        INT16                          cbvSrvParam[5][16]{};
        INT16                          samplerParam[5][16]{};
        INT16                          uavParam[16]{};
        INT16                          rootConstantsParam  = -1;
        INT16                          indirectCbvParam[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
        INT16                          indirectSrvParam    = -1;
        INT16                          indirectSrv2Param   = -1;
        BOOL                           isExtendedIndirect  = false;
        UINT32                         totalParamCount     = 0;
        UINT32                         totalDwordCost      = 0;
    };

    struct ComputeRootLayoutKey
    {
        UINT16 cbvSrvUavMask = 0x000Fu;
        UINT16 samplerMask   = 0x000Fu;

        bool operator==(const ComputeRootLayoutKey& o) const noexcept
        {
            return cbvSrvUavMask == o.cbvSrvUavMask && samplerMask == o.samplerMask;
        }
    };

    struct ComputeRootLayoutKeyHash
    {
        SIZE_T operator()(const ComputeRootLayoutKey& k) const noexcept
        {
            return (static_cast<SIZE_T>(k.cbvSrvUavMask) << 16) ^ static_cast<SIZE_T>(k.samplerMask);
        }
    };

    struct ComputeRootLayout
    {
        ComPtr<ID3D12RootSignature> rootSignature;
        INT16                       spaceParam[16]{};
        INT16                       samplerParam[16]{};
        INT16                       rootConstantsParam = -1;
        UINT32                      totalParamCount    = 0;
        UINT32                      totalDwordCost     = 0;
    };

    class D3D12Pipeline
    {
    public:
        D3D12Pipeline() = default;
        ~D3D12Pipeline() { Shutdown(); }

        BOOL Initialize(D3D12DeviceContext& ctx);
        void Shutdown();

        ID3D12RootSignature*      GetRootSignature()               const { return m_defaultGraphicsLayout.rootSignature.Get(); }
        ID3D12RootSignature*      GetComputeRootSignature()        const { return m_defaultComputeLayout.rootSignature.Get(); }
        const GraphicsRootLayout* GetDefaultGraphicsRootLayout()   const { return &m_defaultGraphicsLayout; }
        const ComputeRootLayout*  GetDefaultComputeRootLayout()    const { return &m_defaultComputeLayout; }
        ID3D12CommandSignature*   GetDrawIndirectSignature()           const { return m_drawIndirectSignature.Get(); }
        ID3D12CommandSignature*   GetDrawIndexedIndirectSignature()    const { return m_drawIndexedIndirectSignature.Get(); }
        ID3D12CommandSignature*   GetNVSDrawIndexedIndirectSignature() const { return m_nvsDrawIndexedIndirectSignature.Get(); }
        ID3D12CommandSignature*   GetDispatchIndirectSignature()       const { return m_dispatchIndirectSignature.Get(); }
        const GraphicsRootLayout* GetOrCreateGraphicsRootLayout(
            ID3D12Device*                 device,
            const D3D12ResourceManager&   resMgr,
            NativeHandle                  vsHandle,
            NativeHandle                  psHandle,
            NativeHandle                  gsHandle,
            NativeHandle                  hsHandle,
            NativeHandle                  dsHandle,
            const IndirectRootBindingKey& indirectRootKey);
        const ComputeRootLayout*  GetOrCreateComputeRootLayout(
            ID3D12Device*               device,
            const D3D12ResourceManager& resMgr,
            NativeHandle                csHandle);
        ID3D12CommandSignature*   GetOrCreateIndirectCommandSignatureForLayout(
            ID3D12Device*     device,
            GraphicsRootLayout& layout);
        ID3D12RootSignature*      GetOrCreateIndirectRootSignature(ID3D12Device* device, const IndirectRootBindingKey& key);
        ID3D12CommandSignature*   GetOrCreateIndirectCommandSignature(ID3D12Device* device, const IndirectRootBindingKey& key);
        ID3D12CommandSignature* GetCommandSignature(GPUIndirectCommandType type) const
        {
            switch (type)
            {
            case GPUIndirectCommandType::DRAW_INSTANCED:         return m_drawIndirectSignature.Get();
            case GPUIndirectCommandType::DRAW_INDEXED_INSTANCED: return m_drawIndexedIndirectSignature.Get();
            case GPUIndirectCommandType::DISPATCH:               return m_dispatchIndirectSignature.Get();
            default:                                             return nullptr;
            }
        }

        ID3D12PipelineState* GetOrCreatePSO(
            ID3D12Device* device,
            const D3D12ResourceManager& resMgr,
            NativeHandle vsHandle,
            NativeHandle psHandle,
            CullMode cullMode,
            BlendMode blendMode,
            FillMode fillMode,
            const DepthState& depthState,
            BOOL shadowDepthBias = false,
            INT32 customDepthBias = 0,
            FLOAT32 customSlopeScaledDepthBias = 0.0f,
            FLOAT32 customDepthBiasClamp = 0.0f,
            NativeHandle gsHandle = 0,
            BOOL depthOnlyTarget = false,
            NativeHandle hsHandle = 0,
            NativeHandle dsHandle = 0,
            const IndirectRootBindingKey& indirectRootKey = IndirectRootBindingKey{},
            DXGI_FORMAT rtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT dsvFormat = DXGI_FORMAT_D32_FLOAT);

        ID3D12PipelineState* GetOrCreateComputePSO(
            ID3D12Device*               device,
            const D3D12ResourceManager& resMgr,
            NativeHandle                csHandle,
            const BackendCapabilities*  deviceCaps = nullptr);

        void   InvalidatePSOsForShader(NativeHandle shaderHandle, UINT64 retireFenceValue = 0);
        UINT32 GetPSOCacheHits() const { return m_psoCacheHits; }
        UINT32 GetPSOCacheMisses() const { return m_psoCacheMisses; }
        UINT32 GetTotalPSOCacheHits() const { return m_totalPsoCacheHits; }
        UINT32 GetTotalPSOCacheMisses() const { return m_totalPsoCacheMisses; }
        UINT32 GetActiveCachedPSOCount() const { return static_cast<UINT32>(m_psoCache.size() + m_computePsoCache.size()); }
        UINT32 GetRetiredPSOCount() const;
        void   ResetFramePSOMetrics() { m_psoCacheHits = 0; m_psoCacheMisses = 0; }

        ID3D12Resource*      GetInstanceUploadBuffer() const { return m_instanceUploadBuffer.Get(); }
        void*                GetInstanceMappedData()   const { return m_instanceMappedData; }
        ID3D12Resource*      GetCBufferUploadBuffer()  const { return m_cbufferUploadBuffer.Get(); }
        ID3D12Resource*      GetSBufferUploadBuffer()  const { return m_sbufferUploadBuffer.Get(); }

        static constexpr UINT32 kCBVPerSpace                        = 16;  // b0 .. b15 (offset 0..15)
        static constexpr UINT32 kSRVPerSpace                        = 32;  // t0 .. t31 (offset 16..47)
        static constexpr UINT32 kUAVPerSpace                        = 16;  // u0 .. u15 (offset 48..63 in CS; 0..15 in GFX UAV table)
        static constexpr UINT32 kGraphicsDescriptorsPerSpaceTable   = kCBVPerSpace + kSRVPerSpace;                // 48
        static constexpr UINT32 kDescriptorsPerSpaceTable           = kCBVPerSpace + kSRVPerSpace + kUAVPerSpace; // 64
        static_assert(kGraphicsDescriptorsPerSpaceTable == 48, "Graphics RootSignature CBV+SRV space table stride must be 48 descriptors.");
        static_assert(kDescriptorsPerSpaceTable == 64, "Compute RootSignature space table stride must be 64 descriptors (16 CBV + 32 SRV + 16 UAV).");

        static constexpr UINT32 kMaxSpaces                         = 16;  // space0 .. space15
        static constexpr UINT32 kDefaultSpaces                     = 4;   // Fast default layout covers space0 .. space3
        static constexpr UINT32 kInternalRootConstantsSpace        = 16;  // Isolated from NSL space0..space15
        static constexpr UINT32 kInternalPlaceholderSpace          = 17;  // Isolated placeholder space for unused indirect root slots
        static constexpr UINT32 kMaxRootSignatureDwords            = 64;  // D3D12 hardware root signature DWORD ceiling
        static constexpr UINT32 kMaxRootParameters                 = 64;
        static constexpr UINT32 kGraphicsIndirectCBVParamCount           = 3;
        static constexpr UINT32 kExtendedGraphicsIndirectCBVParamCount   = 8;
        static constexpr UINT32 kNVSIndirectCommandByteStride            = 96;  // sizeof(NVSIndirectDrawCommand)
        static constexpr UINT32 kNVGExtendedIndirectCommandByteStride    = 144; // 36 DWORDs (VB+IB+8xCBV+2xSRV+DrawIndexed+3xCtx)

        static constexpr UINT32 kBackBufferCount          = 3;
        static constexpr UINT32 kMaxFramesInFlight        = 3;

        static constexpr UINT64 kInitialInstanceRingCount = 24576;
        static constexpr UINT64 kInstanceRingCount        = 24576;

        UINT64                  GetInstanceRingCount() const { return m_instanceRingCount; }
        UINT64                  GetInstanceRingSize()  const { return m_instanceRingSize; }
        UINT64                  GetCBufferRingSize()   const { return m_cbufferRingSize; }
        UINT64                  GetSBufferRingSize()   const { return m_sbufferRingSize; }

        UINT32                    AllocateInstances(const D3D12InstancePayload* data, UINT32 count);
        void                      ResetFrameInstanceOffset(UINT32 bbIndex = 0)
        {
            m_instanceOffset = (bbIndex % kBackBufferCount) * (m_instanceRingSize / kBackBufferCount);
        }

        static constexpr UINT64   kInitialCBufferRingSize = 3 * 1024 * 1024;
        static constexpr UINT64   kCBufferRingSize        = 3 * 1024 * 1024;

        D3D12_GPU_VIRTUAL_ADDRESS AllocateCBuffer(const void* data, UINT32 byteSize);
        void                      ResetFrameCBufferOffset(UINT32 bbIndex = 0)
        {
            m_cbufferOffset = (bbIndex % kBackBufferCount) * (m_cbufferRingSize / kBackBufferCount);
        }

        static constexpr UINT64   kInitialSBufferRingSize = 6 * 1024 * 1024;
        static constexpr UINT64   kSBufferRingSize        = 6 * 1024 * 1024;

        D3D12_GPU_VIRTUAL_ADDRESS AllocateStructuredBuffer(const void* data, UINT32 byteSize, UINT32 stride = 0);
        void                      ResetFrameSBufferOffset(UINT32 bbIndex = 0)
        {
            m_sbufferOffset = (bbIndex % kBackBufferCount) * (m_sbufferRingSize / kBackBufferCount);
        }

        void                      BeginFrameBuffers(UINT32 bbIndex, UINT64 completedFenceValue = 0, UINT64 upcomingFenceValue = 0);
        void                      ReclaimRetiredResources(UINT64 completedFenceValue);
        void                      SetCurrentRetireFenceValue(UINT64 retireFenceValue) { m_currentRetireFenceValue = retireFenceValue; }
        void                      FlushAllRetiredResources();

    private:
        BOOL BuildGraphicsRootLayout(ID3D12Device* device, const GraphicsRootLayoutKey& key, GraphicsRootLayout& outLayout);
        BOOL BuildComputeRootLayout(ID3D12Device* device, const ComputeRootLayoutKey& key, ComputeRootLayout& outLayout);
        BOOL CreateRootSignature(ID3D12Device* device);
        BOOL CreateComputeRootSignature(ID3D12Device* device);
        BOOL CreateCommandSignatures(ID3D12Device* device);
        BOOL CreateInstanceBuffer(ID3D12Device* device);
        BOOL CreateCBufferRing(ID3D12Device* device);
        BOOL CreateSBufferRing(ID3D12Device* device);

        BOOL GrowInstanceBuffer(UINT64 minRequiredCount);
        BOOL GrowCBuffer(UINT64 minRequiredBytes);
        BOOL GrowSBuffer(UINT64 minRequiredBytes);

    private:
        struct RetiredBufferEntry
        {
            UINT64                 fenceValue = 0;
            ComPtr<ID3D12Resource> resource;
        };

        struct RetiredPSOEntry
        {
            UINT64                      fenceValue = 0;
            ComPtr<ID3D12PipelineState> pso;
        };

        ID3D12Device*                    m_device = nullptr;
        GraphicsRootLayout               m_defaultGraphicsLayout{};
        ComputeRootLayout                m_defaultComputeLayout{};
        ComPtr<ID3D12CommandSignature>   m_drawIndirectSignature;
        ComPtr<ID3D12CommandSignature>   m_drawIndexedIndirectSignature;
        ComPtr<ID3D12CommandSignature>   m_nvsDrawIndexedIndirectSignature;
        ComPtr<ID3D12CommandSignature>   m_dispatchIndirectSignature;

        std::unordered_map<PSOCacheKey, ComPtr<ID3D12PipelineState>, PSOCacheKeyHash>                   m_psoCache;
        std::unordered_map<NativeHandle, ComPtr<ID3D12PipelineState>>                                   m_computePsoCache;
        std::unordered_map<GraphicsRootLayoutKey, GraphicsRootLayout, GraphicsRootLayoutKeyHash>        m_graphicsRootLayouts;
        std::unordered_map<ComputeRootLayoutKey, ComputeRootLayout, ComputeRootLayoutKeyHash>           m_computeRootLayouts;

        UINT32                      m_activeBB                = 0;
        UINT64                      m_currentRetireFenceValue = 0;

        UINT64                      m_instanceRingCount  = kInitialInstanceRingCount;
        UINT64                      m_instanceRingSize   = sizeof(D3D12InstancePayload) * kInitialInstanceRingCount;
        ComPtr<ID3D12Resource>      m_instanceUploadBuffer;
        void*                       m_instanceMappedData = nullptr;
        UINT64                      m_instanceOffset     = 0;

        UINT64                      m_cbufferRingSize   = kInitialCBufferRingSize;
        ComPtr<ID3D12Resource>      m_cbufferUploadBuffer;
        void*                       m_cbufferMappedData = nullptr;
        UINT64                      m_cbufferOffset     = 0;

        UINT64                      m_sbufferRingSize   = kInitialSBufferRingSize;
        ComPtr<ID3D12Resource>      m_sbufferUploadBuffer;
        void*                       m_sbufferMappedData = nullptr;
        UINT64                      m_sbufferOffset     = 0;

        // Fence-tagged retirement queues: resources/PSOs are only freed once GPU completedFence >= fenceValue
        std::vector<RetiredBufferEntry> m_retiredBuffers;
        std::vector<RetiredPSOEntry>    m_retiredPSOs;
        UINT32                          m_psoCacheHits        = 0;
        UINT32                          m_psoCacheMisses      = 0;
        UINT32                          m_totalPsoCacheHits   = 0;
        UINT32                          m_totalPsoCacheMisses = 0;
    };
}
