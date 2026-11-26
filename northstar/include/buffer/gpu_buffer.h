#pragma once

#include "gpu_buffer_types.h"
#include "gpu_buffer_desc.h"
#include "../types/northstar_types.h"
#include <cstring>
#include <string>
#include <vector>

namespace Northstar
{
    class GPUBuffer
    {
    public:
        ~GPUBuffer() = default;

        NativeHandle       GetNativeHandle() const { return m_nativeHandle; }
        UINT32             GetElementCount() const { return m_elementCount; }
        UINT32             GetStride()       const { return m_stride;       }
        UINT32             GetByteSize()     const { return m_elementCount * m_stride; }
        GPUBufferUsage     GetUsage()        const { return m_usage;        }
        BOOL               HasUsage(GPUBufferUsage flag) const { return HasGPUBufferUsage(m_usage, flag); }
        const std::string& GetDebugName()    const { return m_debugName;    }
        BOOL               IsValid()         const { return m_nativeHandle != 0 && m_elementCount > 0 && m_stride > 0; }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreBufferManager;

        GPUBuffer() = default;

        void Initialize(const GPUBufferDesc& desc, NativeHandle handle)
        {
            m_elementCount = desc.elementCount;
            m_stride       = desc.stride;
            m_usage        = desc.usage;
            m_debugName    = desc.debugName;
            m_nativeHandle = handle;
            if (handle == 0 && desc.initialData != nullptr && desc.elementCount > 0 && desc.stride > 0)
            {
                const SIZE_T bytes = static_cast<SIZE_T>(desc.elementCount) * static_cast<SIZE_T>(desc.stride);
                m_pendingInitialData.resize(bytes);
                std::memcpy(m_pendingInitialData.data(), desc.initialData, bytes);
            }
            else if (handle != 0)
            {
                m_pendingInitialData.clear();
                m_pendingInitialData.shrink_to_fit();
            }
        }

        void SetNativeHandle(NativeHandle handle)
        {
            m_nativeHandle = handle;
            if (handle != 0)
            {
                m_pendingInitialData.clear();
                m_pendingInitialData.shrink_to_fit();
            }
        }

        const BYTE* GetPendingInitialData() const
        {
            return m_pendingInitialData.empty() ? nullptr : m_pendingInitialData.data();
        }

        NativeHandle      m_nativeHandle = 0;
        UINT32            m_elementCount = 0;
        UINT32            m_stride       = 0;
        GPUBufferUsage    m_usage        = GPUBufferUsage::NONE;
        std::string       m_debugName;
        std::vector<BYTE> m_pendingInitialData;
    };
}
