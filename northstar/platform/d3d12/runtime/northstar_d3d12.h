#pragma once

#include "platform/northstar_backend.h"
#include "platform/native_render_target.h"
#include "platform/frame_view.h"
#include "texture/texture_desc.h"
#include "sampler/sampler_desc.h"
#include "types/northstar_types.h"

#ifdef _WIN32
    #ifdef NORTHSTAR_D3D12_EXPORTS
        #define NORTHSTAR_D3D12_API __declspec(dllexport)
    #else
        #define NORTHSTAR_D3D12_API __declspec(dllimport)
    #endif
#else
    #define NORTHSTAR_D3D12_API
#endif

namespace Northstar
{
    // Direct3D 12 GPU Backend Implementation.
    // Handles GPU device initialization, resource creation/management,
    // descriptor allocation, command recording, and execution.
    class NorthstarD3D12 final : public NorthstarBackend
    {
    public:
        NorthstarD3D12();
        ~NorthstarD3D12() override;

        // ==================== LIFECYCLE ====================
        TargetAPI GetTargetAPI() const                          override { return TargetAPI::HLSL12; }
        BOOL IsHardwareBackend() const                          override { return true; }
        BOOL Initialize(const NativeRenderTarget& renderTarget) override;
        void Shutdown()                                         override;
        BOOL IsInitialized() const                              override;

        // ==================== FRAME PIPELINE ====================
        void ExecuteFrame(const FrameView& view) override;
        void Present()                           override;
        BackendFrameMetrics GetLastFrameMetrics() const override;

        void SetVSync(BOOL enabled) override;
        BOOL IsVSyncEnabled() const override;

        void   SetMaxFramesInFlight(UINT32 maxFramesInFlight) override;
        UINT32 GetMaxFramesInFlight() const override;

        void OnResize(UINT32 width, UINT32 height) override;

        void WaitForGPU() override;
        BOOL EnqueueExternalFenceWait(void* sharedFenceHandle, UINT64 fenceValue) override;
        void FlushDeferredRetirements() override;
        void FlushSpecificBufferRetirements(const NativeHandle* bufferHandles, UINT32 handleCount) override;

        NativeContext GetNativeContext() const override;
        void SetPrePresentCallback(PrePresentCallback cb) override;
        void SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU = true) override;

        // ==================== RESOURCE MANAGEMENT ====================
        NativeHandle CreateVertexBuffer(const void* data, UINT32 byteSize, UINT32 stride) override;
        void         DestroyVertexBuffer(NativeHandle handle)                              override;

        NativeHandle CreateIndexBuffer(const void* data, UINT32 byteSize, BOOL is32Bit = true) override;
        void         DestroyIndexBuffer(NativeHandle handle)                                    override;
        BOOL         GetGeometryBufferGPUViews(
            NativeHandle            vertexBufferHandle,
            NativeHandle            indexBufferHandle,
            GeometryBufferGPUViews& outViews) const override;

        NativeHandle CreateConstantBuffer(UINT32 byteSize)                                         override;
        void         UpdateConstantBuffer(NativeHandle handle, const void* data, UINT32 byteSize) override;
        void         DestroyConstantBuffer(NativeHandle handle)                                    override;

        NativeHandle CreateTexture(const TextureDesc& desc, const void* initialData = nullptr) override;
        BOOL         UpdateTextureSubresource(
            NativeHandle handle,
            UINT32       mipSlice,
            UINT32       arraySlice,
            const void*  data,
            SIZE_T       dataByteSize) override;
        BOOL         GenerateTextureMips(NativeHandle handle)                                  override;
        void         DestroyTexture(NativeHandle handle)                                       override;
        void*        GetTextureGPUDescriptor(NativeHandle handle)                        const override;
        void*        GetTextureSharedHandle(NativeHandle handle)                         const override;
        SharedHandleType GetTextureSharedHandleType(NativeHandle handle)                 const override;

        NativeHandle CreateSampler(const SamplerDesc& desc) override;
        void         DestroySampler(NativeHandle handle)    override;

        NativeHandle CreateShader(ShaderStage stage, const void* bytecode, SIZE_T bytecodeSize) override;
        BOOL         GetShaderCompiledBytecode(NativeHandle handle, std::vector<BYTE>& outBytecode) const override;
        void         SetShaderInputLayout(NativeHandle handle, const std::vector<NSCStageIOEntry>& inputLayout) override;
        void         DestroyShader(NativeHandle handle)                                         override;

        // ==================== PERSISTENT GPU BUFFER MANAGEMENT ====================
        NativeHandle CreateGPUBuffer(const GPUBufferDesc& desc) override;
        BOOL         UpdateGPUBuffer(NativeHandle handle, const void* data, UINT32 byteSize, UINT32 dstOffset = 0) override;
        UINT64       GetGPUBufferGPUVirtualAddress(NativeHandle handle) const override;
        BOOL         RequestGPUBufferReadback(NativeHandle handle, UINT32 srcOffset = 0, UINT32 byteSize = 0) override;
        BOOL         TryGetGPUBufferReadbackData(NativeHandle handle, void* outData, UINT32 byteSize) override;
        BOOL         ReadbackGPUBufferSync(NativeHandle handle, void* outData, UINT32 byteSize) override;
        void         DestroyGPUBuffer(NativeHandle handle) override;

        // ==================== COMPUTE DISPATCH, LAYERED DEPTH & TEXTURE READBACK ====================
        BOOL ExecuteComputeDispatches(const std::vector<ComputeDispatchCommand>& dispatches) override;
        BOOL ExecuteLayeredDepthPasses(
            const std::vector<LayeredDepthPassView>& passes,
            const std::vector<ComputeDispatchCommand>* preDispatches = nullptr) override;
        BOOL ReadbackTextureSync(NativeHandle handle, void* outData, UINT32 byteSize) override;
        BOOL ReadbackTextureSubresourceSync(
            NativeHandle handle,
            UINT32       mipSlice,
            UINT32       arraySlice,
            void*        outData,
            UINT32       byteSize) override;

        // ==================== HARDWARE CAPABILITIES & FALLBACK DIAGNOSTICS ====================
        BOOL SupportsVPAndRTArrayIndexWithoutGS() const override;
        void SetForceLayeredDepthFallback(BOOL forceFallback) override;
        BOOL IsForceLayeredDepthFallback() const override;

        // ==================== VALIDATION & DIAGNOSTICS ====================
        void   SetValidationConfig(BOOL enableDebugLayer, BOOL enableGPUBasedValidation) override;
        BOOL   IsDebugLayerActive() const override;
        BOOL   IsGPUBasedValidationActive() const override;
        UINT32 GetDebugErrorCount() const override;
        void   ClearDebugMessages() override;

        // ==================== RESOURCE STATE & BARRIER DIAGNOSTICS ====================
        BOOL                 TransitionGPUBuffer(NativeHandle handle, GPUResourceAccess targetAccess) override;
        BOOL                 TransitionTexture(NativeHandle handle, GPUResourceAccess targetAccess) override;
        UINT32               FlushResourceBarriers(BOOL executeOnGPU = true) override;
        GPUResourceAccess    GetGPUBufferAccessState(NativeHandle handle) const override;
        ResourceBarrierStats GetResourceBarrierStats() const override;
        void                 ResetResourceBarrierStats() override;

        // ==================== RESOURCE INSPECTION (DIAGNOSTICS) ====================
        BOOL HasVertexBuffer(NativeHandle handle)   const;
        BOOL HasIndexBuffer(NativeHandle handle)    const;
        BOOL HasConstantBuffer(NativeHandle handle) const;
        BOOL HasTexture(NativeHandle handle)        const;
        BOOL HasSampler(NativeHandle handle)        const;
        BOOL HasShader(NativeHandle handle)         const;
        BOOL HasGPUBuffer(NativeHandle handle)      const override;

        UINT32 GetPendingRetiredResourceCount() const override;
        UINT32 GetActiveResourceCount() const override;
        UINT32 GetInFlightUploadRingRegionCount() const override;
        UINT32 GetUnsubmittedUploadRingRegionCount() const override;

    private:
        struct Impl;
        Impl* m_impl = nullptr;
    };
}

// ==================== DYNAMIC MODULE BOUNDARY ====================
extern "C" {
    NORTHSTAR_D3D12_API Northstar::UINT32            Northstar_GetAPIVersion();
    NORTHSTAR_D3D12_API Northstar::NorthstarBackend* CreateNorthstarD3D12Backend();
    NORTHSTAR_D3D12_API void                         DestroyNorthstarD3D12Backend(Northstar::NorthstarBackend* backend);
}
