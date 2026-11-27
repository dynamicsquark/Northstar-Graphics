#include "core_buffer_manager.h"
#include <algorithm>

namespace Northstar
{
    CoreBufferManager::~CoreBufferManager()
    {
        Shutdown();
    }

    BOOL CoreBufferManager::IsManagedBuffer(const GPUBuffer* buffer) const
    {
        if (!buffer) return false;
        return std::find(m_buffers.begin(), m_buffers.end(), buffer) != m_buffers.end();
    }

    GPUBuffer* CoreBufferManager::CreateGPUBuffer(const GPUBufferDesc& desc)
    {
        if (desc.elementCount == 0 || desc.stride == 0 || desc.usage == GPUBufferUsage::NONE)
        {
            return nullptr;
        }

        if (HasGPUBufferUsage(desc.usage, GPUBufferUsage::RAW_BYTE_ADDRESS) && (desc.stride % 4) != 0)
        {
            return nullptr;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(desc.elementCount) * static_cast<UINT64>(desc.stride);
        if (totalBytes64 == 0 || totalBytes64 > 0x7FFFFFFFULL)
        {
            return nullptr;
        }

        NativeHandle handle = 0;
        if (m_backend)
        {
            handle = m_backend->CreateGPUBuffer(desc);
            if (handle == 0)
            {
                return nullptr;
            }
        }

        GPUBuffer* buffer = new GPUBuffer();
        buffer->Initialize(desc, handle);
        m_buffers.push_back(buffer);
        return buffer;
    }

    BOOL CoreBufferManager::ResizeGPUBuffer(GPUBuffer* buffer, UINT32 newElementCount, const void* initialData)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || newElementCount == 0 || buffer->GetStride() == 0)
        {
            return false;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(newElementCount) * static_cast<UINT64>(buffer->GetStride());
        if (totalBytes64 == 0 || totalBytes64 > 0x7FFFFFFFULL)
        {
            return false;
        }

        GPUBufferDesc newDesc{};
        newDesc.elementCount = newElementCount;
        newDesc.stride       = buffer->GetStride();
        newDesc.usage        = buffer->GetUsage();
        newDesc.initialData  = initialData;
        newDesc.debugName    = buffer->GetDebugName();

        const NativeHandle newHandle = m_backend->CreateGPUBuffer(newDesc);
        if (newHandle == 0)
        {
            return false;
        }

        const NativeHandle oldHandle = buffer->GetNativeHandle();
        if (oldHandle != 0)
        {
            m_backend->DestroyGPUBuffer(oldHandle);
            m_retiredNativeHandles.push_back(oldHandle);
        }

        buffer->Initialize(newDesc, newHandle);
        return true;
    }

    BOOL CoreBufferManager::ResizeGPUBufferPreservingData(GPUBuffer* buffer, UINT32 newElementCount)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || newElementCount == 0 || buffer->GetStride() == 0)
        {
            return false;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(newElementCount) * static_cast<UINT64>(buffer->GetStride());
        if (totalBytes64 == 0 || totalBytes64 > 0x7FFFFFFFULL)
        {
            return false;
        }

        const UINT64 copyBytes = (std::min<UINT64>)(buffer->GetByteSize(), totalBytes64);
        const NativeHandle oldHandle = buffer->GetNativeHandle();

        GPUBufferDesc newDesc{};
        newDesc.elementCount = newElementCount;
        newDesc.stride       = buffer->GetStride();
        newDesc.usage        = buffer->GetUsage();
        newDesc.initialData  = nullptr;
        newDesc.debugName    = buffer->GetDebugName();

        const NativeHandle newHandle = m_backend->CreateGPUBuffer(newDesc);
        if (newHandle == 0)
        {
            return false;
        }

        if (oldHandle != 0 && copyBytes > 0ull)
        {
            if (!m_backend->CopyGPUBufferRegion(newHandle, 0ull, oldHandle, 0ull, copyBytes))
            {
                m_backend->DestroyGPUBuffer(newHandle);
                return false;
            }
        }

        if (oldHandle != 0)
        {
            m_backend->DestroyGPUBuffer(oldHandle);
            m_retiredNativeHandles.push_back(oldHandle);
        }

        buffer->Initialize(newDesc, newHandle);
        return true;
    }

    BOOL CoreBufferManager::ResizeGPUBufferWithSpans(
        GPUBuffer*               buffer,
        UINT32                   newElementCount,
        const void*              fillInitialData,
        const GPUBufferCopySpan* spans,
        UINT32                   spanCount)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || newElementCount == 0 || buffer->GetStride() == 0)
        {
            return false;
        }
        if (spanCount > 0u && spans == nullptr)
        {
            return false;
        }

        const UINT64 totalBytes64 = static_cast<UINT64>(newElementCount) * static_cast<UINT64>(buffer->GetStride());
        if (totalBytes64 == 0ull || totalBytes64 > 0x7FFFFFFFULL)
        {
            return false;
        }

        const UINT64 srcTotalBytes  = buffer->GetByteSize();
        const NativeHandle oldHandle = buffer->GetNativeHandle();
        for (UINT32 i = 0u; i < spanCount; ++i)
        {
            const GPUBufferCopySpan& sp = spans[i];
            if (sp.byteSize == 0ull) continue;
            if (oldHandle == 0 ||
                sp.srcOffset > srcTotalBytes || sp.byteSize > (srcTotalBytes - sp.srcOffset) ||
                sp.dstOffset > totalBytes64  || sp.byteSize > (totalBytes64 - sp.dstOffset))
            {
                return false;
            }
        }

        GPUBufferDesc newDesc{};
        newDesc.elementCount = newElementCount;
        newDesc.stride       = buffer->GetStride();
        newDesc.usage        = buffer->GetUsage();
        newDesc.initialData  = fillInitialData;
        newDesc.debugName    = buffer->GetDebugName();

        const NativeHandle newHandle = m_backend->CreateGPUBuffer(newDesc);
        if (newHandle == 0)
        {
            return false;
        }

        for (UINT32 i = 0u; i < spanCount; ++i)
        {
            const GPUBufferCopySpan& sp = spans[i];
            if (sp.byteSize == 0ull) continue;
            if (!m_backend->CopyGPUBufferRegion(newHandle, sp.dstOffset, oldHandle, sp.srcOffset, sp.byteSize))
            {
                m_backend->DestroyGPUBuffer(newHandle);
                return false;
            }
        }

        if (oldHandle != 0)
        {
            m_backend->DestroyGPUBuffer(oldHandle);
            m_retiredNativeHandles.push_back(oldHandle);
        }

        buffer->Initialize(newDesc, newHandle);
        return true;
    }

    BOOL CoreBufferManager::CopyGPUBufferRegion(
        GPUBuffer*       dstBuffer,
        UINT64           dstOffset,
        const GPUBuffer* srcBuffer,
        UINT64           srcOffset,
        UINT64           byteSize)
    {
        if (!m_backend || !IsManagedBuffer(dstBuffer) || !IsManagedBuffer(srcBuffer) ||
            !dstBuffer->IsValid() || !srcBuffer->IsValid() || dstBuffer == srcBuffer || byteSize == 0ull)
        {
            return false;
        }
        const UINT64 dstTotal = dstBuffer->GetByteSize();
        const UINT64 srcTotal = srcBuffer->GetByteSize();
        if (dstOffset > dstTotal || byteSize > (dstTotal - dstOffset) ||
            srcOffset > srcTotal || byteSize > (srcTotal - srcOffset))
        {
            return false;
        }
        return m_backend->CopyGPUBufferRegion(dstBuffer->GetNativeHandle(), dstOffset, srcBuffer->GetNativeHandle(), srcOffset, byteSize);
    }

    BOOL CoreBufferManager::UpdateGPUBuffer(GPUBuffer* buffer, const void* data, UINT32 byteSize, UINT32 dstOffset)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || !buffer->IsValid() || !data || byteSize == 0)
        {
            return false;
        }

        const UINT32 totalBytes = buffer->GetByteSize();
        if (dstOffset > totalBytes || byteSize > (totalBytes - dstOffset))
        {
            return false;
        }

        return m_backend->UpdateGPUBuffer(buffer->GetNativeHandle(), data, byteSize, dstOffset);
    }

    BOOL CoreBufferManager::RequestGPUBufferReadback(GPUBuffer* buffer, UINT32 srcOffset, UINT32 byteSize)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || !buffer->IsValid())
        {
            return false;
        }

        if (!buffer->HasUsage(GPUBufferUsage::CPU_READBACK))
        {
            return false;
        }

        const UINT32 totalBytes = buffer->GetByteSize();
        if (srcOffset >= totalBytes)
        {
            return false;
        }

        const UINT32 effectiveSize = (byteSize == 0) ? (totalBytes - srcOffset) : byteSize;
        if (effectiveSize == 0 || effectiveSize > (totalBytes - srcOffset))
        {
            return false;
        }

        return m_backend->RequestGPUBufferReadback(buffer->GetNativeHandle(), srcOffset, effectiveSize);
    }

    BOOL CoreBufferManager::TryGetGPUBufferReadbackData(GPUBuffer* buffer, void* outData, UINT32 byteSize)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || !buffer->IsValid() || !outData || byteSize == 0)
        {
            return false;
        }

        if (!buffer->HasUsage(GPUBufferUsage::CPU_READBACK))
        {
            return false;
        }

        if (byteSize > buffer->GetByteSize())
        {
            return false;
        }

        return m_backend->TryGetGPUBufferReadbackData(buffer->GetNativeHandle(), outData, byteSize);
    }

    BOOL CoreBufferManager::ReadbackGPUBufferSync(GPUBuffer* buffer, void* outData, UINT32 byteSize)
    {
        if (!m_backend || !IsManagedBuffer(buffer) || !buffer->IsValid() || !outData || byteSize == 0)
        {
            return false;
        }

        if (!buffer->HasUsage(GPUBufferUsage::CPU_READBACK))
        {
            return false;
        }

        if (byteSize > buffer->GetByteSize())
        {
            return false;
        }

        return m_backend->ReadbackGPUBufferSync(buffer->GetNativeHandle(), outData, byteSize);
    }

    void CoreBufferManager::DestroyGPUBuffer(GPUBuffer* buffer)
    {
        if (!buffer) return;

        auto it = std::find(m_buffers.begin(), m_buffers.end(), buffer);
        if (it == m_buffers.end())
        {
            return;
        }

        if (m_backend && buffer->GetNativeHandle() != 0)
        {
            const NativeHandle h = buffer->GetNativeHandle();
            m_backend->DestroyGPUBuffer(h);
            m_retiredNativeHandles.push_back(h);
            buffer->SetNativeHandle(0);
        }

        m_buffers.erase(it);
        delete buffer;
    }

    void CoreBufferManager::RegisterWithBackend(NorthstarBackend* backend)
    {
        m_backend = backend;
        if (!m_backend) return;

        for (GPUBuffer* buf : m_buffers)
        {
            if (buf && buf->GetNativeHandle() == 0 && buf->GetElementCount() > 0 && buf->GetStride() > 0)
            {
                GPUBufferDesc desc{};
                desc.elementCount = buf->GetElementCount();
                desc.stride       = buf->GetStride();
                desc.usage        = buf->GetUsage();
                desc.initialData  = buf->GetPendingInitialData();
                desc.debugName    = buf->GetDebugName();
                const NativeHandle h = m_backend->CreateGPUBuffer(desc);
                buf->SetNativeHandle(h);
            }
        }
    }

    void CoreBufferManager::PruneRetiredNativeHandles(const NorthstarBackend* backend)
    {
        if (!backend || m_retiredNativeHandles.empty()) return;
        m_retiredNativeHandles.erase(
            std::remove_if(m_retiredNativeHandles.begin(), m_retiredNativeHandles.end(),
                [backend](NativeHandle h) { return !backend->HasGPUBuffer(h); }),
            m_retiredNativeHandles.end());
    }

    void CoreBufferManager::UnregisterFromBackend(BOOL destroyNativeHandles)
    {
        if (destroyNativeHandles && m_backend)
        {
            for (GPUBuffer* buf : m_buffers)
            {
                if (buf && buf->GetNativeHandle() != 0)
                {
                    const NativeHandle h = buf->GetNativeHandle();
                    m_backend->DestroyGPUBuffer(h);
                    m_retiredNativeHandles.push_back(h);
                    buf->SetNativeHandle(0);
                }
            }
        }
        m_backend = nullptr;
    }

    void CoreBufferManager::FlushRetiredNativeHandles(NorthstarBackend* backend)
    {
        if (backend)
        {
            if (!m_retiredNativeHandles.empty())
            {
                backend->FlushSpecificBufferRetirements(
                    m_retiredNativeHandles.data(),
                    static_cast<UINT32>(m_retiredNativeHandles.size())
                );
            }
            else
            {
                backend->FlushSpecificBufferRetirements(nullptr, 0u);
            }
        }
        m_retiredNativeHandles.clear();
    }

    void CoreBufferManager::Shutdown()
    {
        for (GPUBuffer* buf : m_buffers)
        {
            if (buf)
            {
                if (m_backend && buf->GetNativeHandle() != 0)
                {
                    const NativeHandle h = buf->GetNativeHandle();
                    m_backend->DestroyGPUBuffer(h);
                    m_retiredNativeHandles.push_back(h);
                    buf->SetNativeHandle(0);
                }
                delete buf;
            }
        }
        m_buffers.clear();
        m_backend = nullptr;
    }
}

