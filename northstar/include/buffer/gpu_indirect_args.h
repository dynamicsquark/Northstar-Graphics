#pragma once

#include "../types/northstar_types.h"

namespace Northstar
{
    enum class GPUIndirectCommandType : UINT8
    {
        DRAW_INSTANCED         = 0,
        DRAW_INDEXED_INSTANCED = 1,
        DISPATCH               = 2
    };

    // 16 bytes — Bit-compatible with D3D12_DRAW_ARGUMENTS and VkDrawIndirectCommand
    struct GPUIndirectDrawArgs
    {
        UINT32 vertexCountPerInstance = 0;
        UINT32 instanceCount          = 0;
        UINT32 startVertexLocation    = 0;
        UINT32 startInstanceLocation  = 0;
    };
    static_assert(sizeof(GPUIndirectDrawArgs) == 16, "GPUIndirectDrawArgs must be 16 bytes (D3D12_DRAW_ARGUMENTS / VkDrawIndirectCommand)");

    // 20 bytes — Bit-compatible with D3D12_DRAW_INDEXED_ARGUMENTS and VkDrawIndexedIndirectCommand
    struct GPUIndirectDrawIndexedArgs
    {
        UINT32 indexCountPerInstance = 0;
        UINT32 instanceCount         = 0;
        UINT32 startIndexLocation    = 0;
        INT32  baseVertexLocation    = 0;
        UINT32 startInstanceLocation = 0;
    };
    static_assert(sizeof(GPUIndirectDrawIndexedArgs) == 20, "GPUIndirectDrawIndexedArgs must be 20 bytes (D3D12_DRAW_INDEXED_ARGUMENTS / VkDrawIndexedIndirectCommand)");

    // 12 bytes — Bit-compatible with D3D12_DISPATCH_ARGUMENTS and VkDispatchIndirectCommand
    struct GPUIndirectDispatchArgs
    {
        UINT32 threadGroupCountX = 1;
        UINT32 threadGroupCountY = 1;
        UINT32 threadGroupCountZ = 1;
    };
    static_assert(sizeof(GPUIndirectDispatchArgs) == 12, "GPUIndirectDispatchArgs must be 12 bytes (D3D12_DISPATCH_ARGUMENTS / VkDispatchIndirectCommand)");
}
