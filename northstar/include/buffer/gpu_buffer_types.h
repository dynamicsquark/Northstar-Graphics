#pragma once

#include "../types/northstar_types.h"

namespace Northstar
{
    enum class GPUBufferUsage : UINT32
    {
        NONE              = 0,
        STRUCTURED_SRV    = 1 << 0,
        STRUCTURED_UAV    = 1 << 1,
        RAW_BYTE_ADDRESS  = 1 << 2,
        INDIRECT_ARGUMENT = 1 << 3,
        CPU_READBACK      = 1 << 4
    };

    inline GPUBufferUsage operator|(GPUBufferUsage a, GPUBufferUsage b)
    {
        return static_cast<GPUBufferUsage>(static_cast<UINT32>(a) | static_cast<UINT32>(b));
    }

    inline GPUBufferUsage operator&(GPUBufferUsage a, GPUBufferUsage b)
    {
        return static_cast<GPUBufferUsage>(static_cast<UINT32>(a) & static_cast<UINT32>(b));
    }

    inline GPUBufferUsage& operator|=(GPUBufferUsage& a, GPUBufferUsage b)
    {
        a = a | b;
        return a;
    }

    inline BOOL HasGPUBufferUsage(GPUBufferUsage flags, GPUBufferUsage test)
    {
        return (static_cast<UINT32>(flags) & static_cast<UINT32>(test)) != 0;
    }

    enum class GPUResourceAccess : UINT32
    {
        COMMON                     = 0,
        VERTEX_AND_CONSTANT_BUFFER = 1 << 0,
        INDEX_BUFFER               = 1 << 1,
        RENDER_TARGET              = 1 << 2,
        UNORDERED_ACCESS_READ      = 1 << 3,
        UNORDERED_ACCESS_WRITE     = 1 << 4,
        UNORDERED_ACCESS_READ_WRITE = (UNORDERED_ACCESS_READ | UNORDERED_ACCESS_WRITE),
        DEPTH_WRITE                = 1 << 5,
        DEPTH_READ                 = 1 << 6,
        NON_PIXEL_SHADER_RESOURCE  = 1 << 7,
        PIXEL_SHADER_RESOURCE      = 1 << 8,
        ALL_SHADER_RESOURCE        = (NON_PIXEL_SHADER_RESOURCE | PIXEL_SHADER_RESOURCE),
        INDIRECT_ARGUMENT          = 1 << 9,
        COPY_DEST                  = 1 << 10,
        COPY_SOURCE                = 1 << 11,
        PRESENT                    = 1 << 12
    };

    inline GPUResourceAccess operator|(GPUResourceAccess a, GPUResourceAccess b)
    {
        return static_cast<GPUResourceAccess>(static_cast<UINT32>(a) | static_cast<UINT32>(b));
    }

    inline GPUResourceAccess operator&(GPUResourceAccess a, GPUResourceAccess b)
    {
        return static_cast<GPUResourceAccess>(static_cast<UINT32>(a) & static_cast<UINT32>(b));
    }

    inline GPUResourceAccess& operator|=(GPUResourceAccess& a, GPUResourceAccess b)
    {
        a = a | b;
        return a;
    }

    inline BOOL HasGPUResourceAccess(GPUResourceAccess flags, GPUResourceAccess test)
    {
        return (static_cast<UINT32>(flags) & static_cast<UINT32>(test)) != 0;
    }

    struct ResourceBarrierStats
    {
        UINT32 pendingBarriers         = 0;
        UINT32 totalTransitionBarriers = 0;
        UINT32 totalUavBarriers        = 0;
        UINT32 flushedBatches          = 0;
    };
}

