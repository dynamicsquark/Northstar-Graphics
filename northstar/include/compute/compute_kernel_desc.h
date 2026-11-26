#pragma once

#include "../shader/shader.h"
#include "../types/northstar_types.h"
#include "../types/pipeline_state_types.h"
#include <string>

namespace Northstar
{
    struct ComputeKernelDesc
    {
        Shader*               computeShader     = nullptr;
        std::string           name;
        UINT32                threadGroupCountX = 1;
        UINT32                threadGroupCountY = 1;
        UINT32                threadGroupCountZ = 1;
        ComputeExecutionScope executionScope    = ComputeExecutionScope::PER_VIEW;
    };
}

