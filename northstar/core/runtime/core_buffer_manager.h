#pragma once

#include "buffer/gpu_buffer.h"
#include "buffer/gpu_buffer_desc.h"
#include "buffer/gpu_buffer_types.h"
#include "platform/northstar_backend.h"
#include <vector>

namespace Northstar
{
    class CoreBufferManager
    {
    public:
        CoreBufferManager() = default;
        ~CoreBufferManager();

        GPUBuffer* CreateGPUBuffer(const GPUBufferDesc& desc);
        BOOL       ResizeGPUBuffer(GPUBuffer* buffer, UINT32 newElementCount, const void* initialData = nullptr);
        BOOL       UpdateGPUBuffer(GPUBuffer* buffer, const void* data, UINT32 byteSize, UINT32 dstOffset = 0);
        BOOL       RequestGPUBufferReadback(GPUBuffer* buffer, UINT32 srcOffset = 0, UINT32 byteSize = 0);
        BOOL       TryGetGPUBufferReadbackData(GPUBuffer* buffer, void* outData, UINT32 byteSize);
        BOOL       ReadbackGPUBufferSync(GPUBuffer* buffer, void* outData, UINT32 byteSize);
        void       DestroyGPUBuffer(GPUBuffer* buffer);
        BOOL       IsManagedBuffer(const GPUBuffer* buffer) const;

        void       RegisterWithBackend(NorthstarBackend* backend);
        void       UnregisterFromBackend(BOOL destroyNativeHandles = false);
        void       PruneRetiredNativeHandles(const NorthstarBackend* backend);
        void       FlushRetiredNativeHandles(NorthstarBackend* backend);
        void       Shutdown();

    private:
        std::vector<GPUBuffer*>   m_buffers;
        std::vector<NativeHandle> m_retiredNativeHandles;
        NorthstarBackend*         m_backend = nullptr;
    };
}

