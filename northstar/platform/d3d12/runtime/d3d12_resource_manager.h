#pragma once

#include "d3d12_common.h"
#include "d3d12_device.h"
#include "d3d12_descriptor_pool.h"
#include "d3d12_resource_state_tracker.h"
#include "texture/texture_desc.h"
#include "sampler/sampler_desc.h"
#include "shader/shader_types.h"
#include "buffer/gpu_buffer_desc.h"

#include <unordered_map>
#include <vector>

namespace Northstar
{
    struct D3D12VertexBufferResource
    {
        ComPtr<ID3D12Resource>   resource;
        UINT32                   byteSize    = 0;
        UINT32                   stride      = 0;
        UINT32                   vertexCount = 0;
        D3D12_VERTEX_BUFFER_VIEW view{};
    };

    struct D3D12IndexBufferResource
    {
        ComPtr<ID3D12Resource>  resource;
        UINT32                  byteSize   = 0;
        DXGI_FORMAT             format     = DXGI_FORMAT_R32_UINT;
        UINT32                  indexCount = 0;
        D3D12_INDEX_BUFFER_VIEW view{};
    };

    struct D3D12ConstantBufferResource
    {
        ComPtr<ID3D12Resource>    uploadBuffer;
        void*                     mappedData      = nullptr;
        UINT32                    byteSize        = 0;
        UINT32                    alignedSize     = 0;
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress      = 0;
        UINT32                    descriptorIndex = UINT32_MAX;
    };

    struct D3D12TextureResource
    {
        ComPtr<ID3D12Resource> resource;
        TextureDesc            desc{};
        DXGI_FORMAT            format                      = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT            srvFormat                   = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT            uavFormat                   = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT            rtvFormat                   = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT            dsvFormat                   = DXGI_FORMAT_UNKNOWN;
        D3D12_RESOURCE_STATES  currentState                = D3D12_RESOURCE_STATE_COMMON;
        UINT32                 srvDescriptorIndex          = UINT32_MAX;
        UINT32                 uavDescriptorIndex          = UINT32_MAX;
        std::vector<UINT32>    mipSrvDescriptors;
        std::vector<UINT32>    mipUavDescriptors;
        UINT32                 rtvDescriptorIndex          = UINT32_MAX;
        UINT32                 dsvDescriptorIndex          = UINT32_MAX;
        UINT32                 readOnlyDsvDescriptorIndex  = UINT32_MAX;
        UINT32                 fullArrayDsvDescriptorIndex = UINT32_MAX;
        std::vector<UINT32>    dsvLayerDescriptors;
        std::vector<bool>      dsvLayerNeedsClear;
        HANDLE                 sharedHandle                = nullptr;
    };

    struct D3D12SamplerResource
    {
        SamplerDesc desc{};
        UINT32      samplerDescriptorIndex = UINT32_MAX;
    };

    struct D3D12ShaderResource
    {
        ShaderStage                   stage      = ShaderStage::NONE;
        ComPtr<ID3DBlob>              blob;
        std::vector<BYTE>             bytecode;
        std::vector<NSCStageIOEntry>  inputLayout;
        UINT64                        generation = 0;
    };

    struct D3D12GPUBufferResource
    {
        ComPtr<ID3D12Resource> resource;
        GPUBufferDesc          desc{};
        UINT32                 byteSize                = 0;
        D3D12_RESOURCE_STATES  currentState            = D3D12_RESOURCE_STATE_COMMON;
        GPUResourceAccess      defaultAccess           = GPUResourceAccess::ALL_SHADER_RESOURCE;
        UINT32                 srvDescriptorIndex      = UINT32_MAX;
        UINT32                 uavDescriptorIndex      = UINT32_MAX;
        ComPtr<ID3D12Resource> readbackBuffers[3];
        BOOL                   hasPendingReadbackReq   = false;
        UINT32                 pendingReadbackOffset   = 0;
        UINT32                 pendingReadbackSize     = 0;
        UINT32                 latestReadbackSlot      = 0;
        UINT32                 latestReadbackSize      = 0;
        UINT64                 readbackReadyFenceValue = 0;
        BOOL                   pendingDestroy          = false;
        UINT64                 retireFenceValue        = 0;
    };

    struct D3D12RetiredResource
    {
        UINT64                 fenceValue                  = 0;
        ComPtr<ID3D12Resource> resource;
        ComPtr<ID3D12Resource> auxiliaryResources[3];
        void*                  mappedData                  = nullptr;
        UINT32                 cbvDescriptorIndex          = UINT32_MAX;
        UINT32                 srvDescriptorIndex          = UINT32_MAX;
        UINT32                 uavDescriptorIndex          = UINT32_MAX;
        std::vector<UINT32>    extraSrvUavDescriptors;
        UINT32                 rtvDescriptorIndex          = UINT32_MAX;
        UINT32                 dsvDescriptorIndex          = UINT32_MAX;
        UINT32                 readOnlyDsvDescriptorIndex  = UINT32_MAX;
        UINT32                 fullArrayDsvDescriptorIndex = UINT32_MAX;
        std::vector<UINT32>    dsvLayerDescriptors;
        UINT32                 samplerDescriptorIndex      = UINT32_MAX;
        HANDLE                 sharedHandle                = nullptr;
    };

    struct D3D12UploadRingRegion
    {
        NativeHandle targetHandle = 0;
        UINT64       fenceValue   = UINT64_MAX;
        UINT64       offset       = 0;
        UINT64       endOffset    = 0;
    };

    struct D3D12PendingBufferUpload
    {
        NativeHandle           targetHandle       = 0;
        ComPtr<ID3D12Resource> uploadResource;
        BOOL                   isStandaloneUpload = false;
        UINT64                 srcOffset          = 0;
        UINT32                 dstOffset          = 0;
        UINT32                 byteSize           = 0;
    };

    class D3D12ResourceManager
    {
    public:
        static constexpr UINT64 kUploadRingCapacity = 16ULL * 1024ULL * 1024ULL; // 16 MB persistent upload ring

        D3D12ResourceManager()  = default;
        ~D3D12ResourceManager() = default;

        NativeHandle CreateVertexBuffer(
            D3D12DeviceContext& ctx,
            const void* data,
            UINT32 byteSize,
            UINT32 stride);
        void DestroyVertexBuffer(NativeHandle handle, UINT64 fenceValue);

        NativeHandle CreateIndexBuffer(
            D3D12DeviceContext& ctx,
            const void* data,
            UINT32 byteSize,
            BOOL is32Bit = true);
        void DestroyIndexBuffer(NativeHandle handle, UINT64 fenceValue);

        NativeHandle CreateConstantBuffer(
            D3D12DeviceContext& ctx,
            D3D12DescriptorPool& cbvPool,
            UINT32 byteSize);
        void UpdateConstantBuffer(NativeHandle handle, const void* data, UINT32 byteSize);
        void DestroyConstantBuffer(NativeHandle handle, D3D12DescriptorPool& cbvPool, UINT64 fenceValue);

        NativeHandle CreateTexture(
            D3D12DeviceContext& ctx,
            D3D12DescriptorPool& srvPool,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            const TextureDesc& desc,
            const void* initialData = nullptr);
        BOOL UpdateTextureSubresource(
            D3D12DeviceContext& ctx,
            D3D12ResourceStateTracker& stateTracker,
            NativeHandle handle,
            UINT32 mipSlice,
            UINT32 arraySlice,
            const void* data,
            SIZE_T dataByteSize);
        BOOL GenerateTextureMips(
            D3D12DeviceContext& ctx,
            D3D12ResourceStateTracker& stateTracker,
            NativeHandle handle);
        void DestroyTexture(
            NativeHandle handle,
            D3D12DescriptorPool& srvPool,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            UINT64 fenceValue);

        NativeHandle CreateSampler(
            D3D12DeviceContext& ctx,
            D3D12DescriptorPool& samplerPool,
            const SamplerDesc& desc);
        void DestroySampler(NativeHandle handle, D3D12DescriptorPool& samplerPool, UINT64 fenceValue);

        NativeHandle CreateGPUBuffer(
            D3D12DeviceContext& ctx,
            D3D12DescriptorPool& srvUavPool,
            D3D12ResourceStateTracker& stateTracker,
            const GPUBufferDesc& desc);
        BOOL UpdateGPUBuffer(
            D3D12DeviceContext& ctx,
            NativeHandle handle,
            const void* data,
            UINT32 byteSize,
            UINT32 dstOffset = 0);
        void FlushPendingGPUBufferUploads(
            ID3D12GraphicsCommandList* cmdList,
            D3D12ResourceStateTracker& stateTracker,
            UINT64 upcomingFenceValue);
        BOOL RequestGPUBufferReadback(
            D3D12DeviceContext& ctx,
            NativeHandle handle,
            UINT32 srcOffset = 0,
            UINT32 byteSize = 0);
        void FlushPendingGPUBufferReadbacks(
            ID3D12GraphicsCommandList* cmdList,
            D3D12ResourceStateTracker& stateTracker,
            UINT32 bbIndex,
            UINT64 upcomingFenceValue);
        BOOL TryGetGPUBufferReadbackData(
            D3D12DeviceContext& ctx,
            NativeHandle handle,
            void* outData,
            UINT32 byteSize);
        BOOL ReadbackGPUBufferSync(
            D3D12DeviceContext& ctx,
            D3D12ResourceStateTracker& stateTracker,
            NativeHandle handle,
            void* outData,
            UINT32 byteSize);
        BOOL ReadbackTextureSync(
            D3D12DeviceContext& ctx,
            D3D12ResourceStateTracker& stateTracker,
            NativeHandle handle,
            void* outData,
            UINT32 byteSize);
        BOOL ReadbackTextureSubresourceSync(
            D3D12DeviceContext& ctx,
            D3D12ResourceStateTracker& stateTracker,
            NativeHandle handle,
            UINT32 mipSlice,
            UINT32 arraySlice,
            void* outData,
            UINT32 byteSize);
        void DestroyGPUBuffer(
            NativeHandle handle,
            D3D12ResourceStateTracker& stateTracker,
            UINT64 fenceValue);
        void FinalizePendingDestroyedGPUBuffers(
            D3D12ResourceStateTracker& stateTracker,
            UINT64 submittedFenceValue,
            const NativeHandle* filterHandles = nullptr,
            UINT32 filterCount = 0);

        void ProcessRetirementQueue(
            UINT64 completedFenceValue,
            D3D12DescriptorPool& cbvPool,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            D3D12DescriptorPool& samplerPool,
            D3D12ResourceStateTracker* stateTracker = nullptr);

        NativeHandle CreateShader(ShaderStage stage, const void* bytecode, SIZE_T bytecodeSize);
        BOOL         GetShaderCompiledBytecode(NativeHandle handle, std::vector<BYTE>& outBytecode) const;
        void         SetShaderInputLayout(NativeHandle handle, const std::vector<NSCStageIOEntry>& inputLayout);
        void         DestroyShader(NativeHandle handle);

        BOOL HasVertexBuffer(NativeHandle handle)   const;
        BOOL HasIndexBuffer(NativeHandle handle)    const;
        BOOL HasConstantBuffer(NativeHandle handle) const;
        BOOL HasTexture(NativeHandle handle)        const;
        BOOL HasSampler(NativeHandle handle)        const;
        BOOL HasShader(NativeHandle handle)         const;
        BOOL HasGPUBuffer(NativeHandle handle)      const;

        UINT32 GetActiveResourceCount()              const;
        UINT32 GetPendingRetiredResourceCount()      const;
        UINT32 GetInFlightUploadRingRegionCount()    const { return static_cast<UINT32>(m_uploadRingInFlight.size()); }
        UINT32 GetUnsubmittedUploadRingRegionCount() const;
        BOOL   HasPendingBufferUploads()             const { return !m_pendingBufferUploads.empty(); }

        UINT64 ConsumeUploadedBytesThisSubmission()
        {
            const UINT64 bytes = m_uploadedBytesThisSubmission;
            m_uploadedBytesThisSubmission = 0;
            return bytes;
        }

        UINT32 ConsumeGPUBufferAllocationsThisSubmission()
        {
            const UINT32 allocs = m_gpuBufferAllocationsThisSubmission;
            m_gpuBufferAllocationsThisSubmission = 0;
            return allocs;
        }

        const D3D12VertexBufferResource*   GetVertexBuffer(NativeHandle handle)   const;
        const D3D12IndexBufferResource*    GetIndexBuffer(NativeHandle handle)    const;
        const D3D12ConstantBufferResource* GetConstantBuffer(NativeHandle handle) const;
        const D3D12TextureResource*        GetTexture(NativeHandle handle)        const;
        D3D12TextureResource*              GetTexture(NativeHandle handle);
        HANDLE                             GetTextureSharedHandle(NativeHandle handle) const;
        D3D12_GPU_DESCRIPTOR_HANDLE        GetTextureGPUDescriptorHandle(NativeHandle handle, const D3D12DescriptorPool& srvPool) const;
        const D3D12SamplerResource*        GetSampler(NativeHandle handle)        const;
        const D3D12ShaderResource*         GetShader(NativeHandle handle)         const;
        const D3D12GPUBufferResource*      GetGPUBuffer(NativeHandle handle)      const;
        D3D12GPUBufferResource*            GetGPUBuffer(NativeHandle handle);

        NativeHandle GetOrCreateCachedCpuVertexBuffer(
            D3D12DeviceContext& ctx,
            const void* cpuVertices,
            UINT32 vertexCount,
            UINT32 stride);
        NativeHandle GetOrCreateCachedCpuIndexBuffer(
            D3D12DeviceContext& ctx,
            const void* cpuIndices,
            UINT32 indexCount,
            BOOL is32Bit = true);

        void Shutdown(
            D3D12DescriptorPool& cbvPool,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            D3D12DescriptorPool& samplerPool);

    private:
        NativeHandle GenerateHandle() { return m_nextHandle++; }
        BOOL         EnsureGPUBufferReadbackResources(D3D12DeviceContext& ctx, D3D12GPUBufferResource& buf);
        void         ReclaimCompletedUploadRingRegions(UINT64 completedFenceValue);
        BOOL         AllocateFromUploadRing(
            D3D12DeviceContext& ctx,
            NativeHandle targetHandle,
            UINT32 byteSize,
            ComPtr<ID3D12Resource>& outUploadResource,
            UINT64& outSrcOffset,
            void*& outMappedPtr,
            BOOL& outIsStandalone);

    private:
        NativeHandle m_nextHandle = 1;

        std::unordered_map<NativeHandle, D3D12VertexBufferResource>   m_vertexBuffers;
        std::unordered_map<NativeHandle, D3D12IndexBufferResource>    m_indexBuffers;
        std::unordered_map<NativeHandle, D3D12ConstantBufferResource> m_constantBuffers;
        std::unordered_map<NativeHandle, D3D12TextureResource>        m_textures;
        std::unordered_map<NativeHandle, D3D12SamplerResource>        m_samplers;
        std::unordered_map<NativeHandle, D3D12ShaderResource>         m_shaders;
        std::unordered_map<NativeHandle, D3D12GPUBufferResource>      m_gpuBuffers;

        std::unordered_map<const void*, NativeHandle> m_cpuMeshToGpuVb;
        std::unordered_map<const void*, NativeHandle> m_cpuMeshToGpuIb;

        std::vector<D3D12RetiredResource>     m_retiredResources;
        std::vector<D3D12PendingBufferUpload> m_pendingBufferUploads;
        std::vector<D3D12UploadRingRegion>    m_uploadRingInFlight;

        ComPtr<ID3D12Resource> m_uploadRingBuffer;
        BYTE*                  m_uploadRingMappedData               = nullptr;
        UINT64                 m_uploadRingWriteOffset              = 0;
        UINT64                 m_uploadRingReadOffset               = 0;
        BOOL                   m_hasPendingDestroyedGPUBuffers      = false;
        UINT64                 m_uploadedBytesThisSubmission        = 0;
        UINT32                 m_gpuBufferAllocationsThisSubmission = 0;
    };
}
