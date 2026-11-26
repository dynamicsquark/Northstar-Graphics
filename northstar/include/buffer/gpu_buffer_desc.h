#pragma once

#include "gpu_buffer_types.h"
#include "../types/northstar_types.h"
#include <string>

namespace Northstar
{
    struct GPUBufferDesc
    {
        UINT32         elementCount = 0;
        UINT32         stride       = 4;
        GPUBufferUsage usage        = GPUBufferUsage::STRUCTURED_SRV;
        const void*    initialData  = nullptr;
        std::string    debugName;
    };
}

