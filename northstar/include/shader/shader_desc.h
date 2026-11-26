#pragma once

#include "shader_types.h"
#include "../types/northstar_types.h"

#include <string>

namespace Northstar
{
    struct ShaderDesc
    {
        // Data source (MEMORY or REFLECTION)
        ShaderDataSource dataSource        = ShaderDataSource::MEMORY;

        // Raw memory payload: .nsc binary or compiled bytecode (for MEMORY mode)
        const BYTE*      initialData       = nullptr;
        SIZE_T           initialDataSize   = 0;

        // Preloaded reflection data (for REFLECTION mode, if provided directly)
        const ShaderReflection* preloadedReflection = nullptr;

        // Optional identification / tracking name
        std::string      name;
    };
}
