#pragma once

#include "native_render_target.h"
#include "frame_view.h"
#include "../texture/texture_desc.h"
#include "../sampler/sampler_desc.h"
#include "../shader/shader_types.h"
#include "../buffer/gpu_buffer_types.h"
#include "../buffer/gpu_buffer_desc.h"
#include "../types/northstar_types.h"
#include "native_context.h"

namespace Northstar
{
    struct BackendFrameMetrics
    {
        FLOAT32 fenceWaitCpuMs                     = 0.0f;
        FLOAT32 cmdRecordCpuMs                     = 0.0f;
        FLOAT32 cmdSubmitCpuMs                     = 0.0f;
        FLOAT32 gpuBufferUploadRecordCpuMs         = 0.0f;
        FLOAT32 gpuSceneUploadMs                   = 0.0f;
        FLOAT32 gpuComputePassMs                   = 0.0f;
        FLOAT32 gpuNVSFeedbackMs                   = 0.0f;
        FLOAT32 gpuNVSResidencyCullMs              = 0.0f;
        FLOAT32 gpuNVGHZBMs                        = 0.0f;
        FLOAT32 gpuNVGCullLodMs                    = 0.0f;
        FLOAT32 gpuNVGDrawGenMs                    = 0.0f;
        FLOAT32 gpuLayeredDepthPassMs              = 0.0f;
        FLOAT32 gpuShadowPassMs                    = 0.0f;
        FLOAT32 gpuDepthPrepassMs                  = 0.0f;
        FLOAT32 gpuGeometryPassMs                  = 0.0f;
        FLOAT32 gpuTotalFrameMs                    = 0.0f;
        UINT32  descriptorTablesUsed               = 0;
        UINT32  indirectDrawCommandsExecuted       = 0;
        UINT32  indirectDispatchCommandsExecuted   = 0;
        UINT32  indirectCommandsRejected           = 0;
        UINT32  layeredDepthPassesExecuted         = 0;
        UINT32  layeredDepthFallbackPassesExecuted = 0;
        UINT32  cpuPerSliceDepthClearsExecuted     = 0;
        UINT32  uploadedBytesThisFrame             = 0;
        UINT32  gpuBufferAllocationsThisFrame      = 0;
        UINT32  pendingRetiredResources            = 0;
        UINT32  framesInFlightCount                = 0;
        UINT32  psoCacheHits                       = 0;
        UINT32  psoCacheMisses                     = 0;
        UINT32  activeCachedPSOCount               = 0;
        UINT32  retiredPSOCount                    = 0;
    };

    struct GeometryBufferGPUViews
    {
        UINT64 vbGPUVirtualAddress = 0;
        UINT32 vbSizeInBytes       = 0;
        UINT32 vbStrideInBytes     = 0;
        UINT64 ibGPUVirtualAddress = 0;
        UINT32 ibSizeInBytes       = 0;
        UINT32 ibFormat            = 42u; // DXGI_FORMAT_R32_UINT = 42u, DXGI_FORMAT_R16_UINT = 57u
        BOOL   valid               = false;
    };

    // Low-level GPU plugin.
    // Frame path: Core calls ExecuteFrame(view), then Present.
    // No BeginFrame/EndFrame here — those are Core (user) API.
    class NorthstarBackend
    {
    public:
        virtual ~NorthstarBackend() = default;

        virtual TargetAPI GetTargetAPI() const                          = 0;
        virtual BOOL IsHardwareBackend() const                          { return false; }
        virtual BOOL Initialize(const NativeRenderTarget& renderTarget) = 0;
        virtual void Shutdown()                                         = 0;
        virtual BOOL IsInitialized() const                              { return true; }

        virtual void ExecuteFrame(const FrameView& view) = 0;
        virtual void Present()                           = 0;
        virtual BackendFrameMetrics GetLastFrameMetrics() const { return {}; }

        virtual void SetVSync(BOOL enabled)              { (void)enabled; }
        virtual BOOL IsVSyncEnabled() const              { return true; }

        virtual void   SetMaxFramesInFlight(UINT32 count) { (void)count; }
        virtual UINT32 GetMaxFramesInFlight() const       { return 2; }

        virtual void OnResize(UINT32 width, UINT32 height) { (void)width; (void)height; }

        virtual void WaitForGPU() {}
        virtual BOOL EnqueueExternalFenceWait(void* sharedFenceHandle, UINT64 fenceValue)
        { (void)sharedFenceHandle; (void)fenceValue; return false; }
        virtual void FlushDeferredRetirements() { WaitForGPU(); }
        virtual void FlushSpecificBufferRetirements(const NativeHandle* bufferHandles, UINT32 handleCount)
        { (void)bufferHandles; (void)handleCount; WaitForGPU(); }

        virtual NativeContext GetNativeContext() const { return {}; }
        virtual void SetPrePresentCallback(PrePresentCallback cb) { (void)cb; }
        virtual void SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU = true)
        { (void)cb; (void)waitForGPU; }

        // ==================== RESOURCE MANAGEMENT API ====================
        virtual NativeHandle CreateVertexBuffer(const void* data, UINT32 byteSize, UINT32 stride)
        { (void)data; (void)byteSize; (void)stride; return 0; }
        virtual void         DestroyVertexBuffer(NativeHandle handle)
        { (void)handle; }

        virtual NativeHandle CreateIndexBuffer(const void* data, UINT32 byteSize, BOOL is32Bit = true)
        { (void)data; (void)byteSize; (void)is32Bit; return 0; }
        virtual void         DestroyIndexBuffer(NativeHandle handle)
        { (void)handle; }
        virtual BOOL         GetGeometryBufferGPUViews(
            NativeHandle            vertexBufferHandle,
            NativeHandle            indexBufferHandle,
            GeometryBufferGPUViews& outViews) const
        { (void)vertexBufferHandle; (void)indexBufferHandle; outViews = {}; return false; }

        virtual NativeHandle CreateConstantBuffer(UINT32 byteSize)
        { (void)byteSize; return 0; }
        virtual void         UpdateConstantBuffer(NativeHandle handle, const void* data, UINT32 byteSize)
        { (void)handle; (void)data; (void)byteSize; }
        virtual void         DestroyConstantBuffer(NativeHandle handle)
        { (void)handle; }

        virtual NativeHandle CreateTexture(const TextureDesc& desc, const void* initialData = nullptr)
        { (void)desc; (void)initialData; return 0; }
        virtual BOOL         UpdateTextureSubresource(
            NativeHandle handle,
            UINT32       mipSlice,
            UINT32       arraySlice,
            const void*  data,
            SIZE_T       dataByteSize)
        { (void)handle; (void)mipSlice; (void)arraySlice; (void)data; (void)dataByteSize; return false; }
        virtual BOOL         GenerateTextureMips(NativeHandle handle)
        { (void)handle; return false; }
        virtual void         DestroyTexture(NativeHandle handle)
        { (void)handle; }
        virtual void*        GetTextureGPUDescriptor(NativeHandle handle) const
        { (void)handle; return nullptr; }
        virtual void*        GetTextureSharedHandle(NativeHandle handle) const
        { (void)handle; return nullptr; }
        virtual SharedHandleType GetTextureSharedHandleType(NativeHandle handle) const
        { (void)handle; return SharedHandleType::NONE; }

        virtual NativeHandle CreateSampler(const SamplerDesc& desc)
        { (void)desc; return 0; }
        virtual void         DestroySampler(NativeHandle handle)
        { (void)handle; }

        virtual NativeHandle CreateShader(ShaderStage stage, const void* bytecode, SIZE_T bytecodeSize)
        { (void)stage; (void)bytecode; (void)bytecodeSize; return 0; }
        virtual BOOL         GetShaderCompiledBytecode(NativeHandle handle, std::vector<BYTE>& outBytecode) const
        { (void)handle; (void)outBytecode; return false; }
        virtual void         SetShaderInputLayout(NativeHandle handle, const std::vector<NSCStageIOEntry>& inputLayout)
        { (void)handle; (void)inputLayout; }
        virtual void         DestroyShader(NativeHandle handle)
        { (void)handle; }

        // ==================== PERSISTENT GPU BUFFER API ====================
        virtual NativeHandle CreateGPUBuffer(const GPUBufferDesc& desc)
        { (void)desc; return 0; }
        virtual BOOL         UpdateGPUBuffer(NativeHandle handle, const void* data, UINT32 byteSize, UINT32 dstOffset = 0)
        { (void)handle; (void)data; (void)byteSize; (void)dstOffset; return false; }
        virtual BOOL         RequestGPUBufferReadback(NativeHandle handle, UINT32 srcOffset = 0, UINT32 byteSize = 0)
        { (void)handle; (void)srcOffset; (void)byteSize; return false; }
        virtual BOOL         TryGetGPUBufferReadbackData(NativeHandle handle, void* outData, UINT32 byteSize)
        { (void)handle; (void)outData; (void)byteSize; return false; }
        virtual BOOL         ReadbackGPUBufferSync(NativeHandle handle, void* outData, UINT32 byteSize)
        { (void)handle; (void)outData; (void)byteSize; return false; }
        virtual UINT64       GetGPUBufferGPUVirtualAddress(NativeHandle handle) const
        { (void)handle; return 0ull; }
        virtual void         DestroyGPUBuffer(NativeHandle handle)
        { (void)handle; }

        // ==================== COMPUTE DISPATCH, LAYERED DEPTH & TEXTURE READBACK API ====================
        virtual BOOL ExecuteComputeDispatches(const std::vector<ComputeDispatchCommand>& dispatches)
        { (void)dispatches; return false; }
        virtual BOOL ExecuteLayeredDepthPasses(
            const std::vector<LayeredDepthPassView>& passes,
            const std::vector<ComputeDispatchCommand>* preDispatches = nullptr)
        { (void)passes; (void)preDispatches; return false; }
        virtual BOOL ReadbackTextureSync(NativeHandle handle, void* outData, UINT32 byteSize)
        { (void)handle; (void)outData; (void)byteSize; return false; }
        virtual BOOL ReadbackTextureSubresourceSync(
            NativeHandle handle,
            UINT32       mipSlice,
            UINT32       arraySlice,
            void*        outData,
            UINT32       byteSize)
        {
            if (mipSlice == 0u && arraySlice == 0u)
                return ReadbackTextureSync(handle, outData, byteSize);
            (void)handle; (void)mipSlice; (void)arraySlice; (void)outData; (void)byteSize;
            return false;
        }

        // ==================== HARDWARE CAPABILITIES & FALLBACK DIAGNOSTICS ====================
        virtual BOOL SupportsVPAndRTArrayIndexWithoutGS() const { return false; }
        virtual void SetForceLayeredDepthFallback(BOOL forceFallback) { (void)forceFallback; }
        virtual BOOL IsForceLayeredDepthFallback() const { return false; }

        // ==================== VALIDATION & DIAGNOSTICS API ====================
        virtual void   SetValidationConfig(BOOL enableDebugLayer, BOOL enableGPUBasedValidation)
        { (void)enableDebugLayer; (void)enableGPUBasedValidation; }
        virtual BOOL   IsDebugLayerActive() const { return false; }
        virtual BOOL   IsGPUBasedValidationActive() const { return false; }
        virtual UINT32 GetDebugErrorCount() const { return 0; }
        virtual void   ClearDebugMessages() {}

        // ==================== RESOURCE STATE & BARRIER DIAGNOSTICS API ====================
        virtual BOOL                TransitionGPUBuffer(NativeHandle handle, GPUResourceAccess targetAccess)
        { (void)handle; (void)targetAccess; return false; }
        virtual BOOL                TransitionTexture(NativeHandle handle, GPUResourceAccess targetAccess)
        { (void)handle; (void)targetAccess; return false; }
        virtual UINT32              FlushResourceBarriers(BOOL executeOnGPU = true)
        { (void)executeOnGPU; return 0; }
        virtual GPUResourceAccess   GetGPUBufferAccessState(NativeHandle handle) const
        { (void)handle; return GPUResourceAccess::COMMON; }
        virtual ResourceBarrierStats GetResourceBarrierStats() const
        { return {}; }
        virtual void                ResetResourceBarrierStats() {}
        virtual BOOL                HasGPUBuffer(NativeHandle handle) const
        { (void)handle; return false; }
        virtual UINT32              GetPendingRetiredResourceCount() const
        { return 0; }
        virtual UINT32              GetActiveResourceCount() const
        { return 0; }
        virtual UINT32              GetInFlightUploadRingRegionCount() const
        { return 0; }
        virtual UINT32              GetUnsubmittedUploadRingRegionCount() const
        { return 0; }
    };
}
