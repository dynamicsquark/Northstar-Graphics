#pragma once

#include "material_types.h"
#include "../shader_set/shader_set.h"
#include "../compute/compute_kernel.h"

namespace Northstar
{
    struct MaterialDesc
    {
        // Primary geometry pass shader set
        ShaderSet*     geometryShaderSet = nullptr;

        // Optional shadow pass shader set (if null, object does not cast shadows)
        ShaderSet*     shadowShaderSet   = nullptr;

        // Optional explicit depth prepass shader set (PassType::DEPTH).
        // When null, Depth Prepass reuses geometryShaderSet.
        ShaderSet*     depthShaderSet    = nullptr;

        // Optional Material-driven pre-pass ComputeKernel (e.g., procedural texture/buffer generation)
        ComputeKernel* computeKernel     = nullptr;

        CullMode       cullMode          = CullMode::BACK;
        BlendMode      blendMode         = BlendMode::OPAQUE;
        FillMode       fillMode          = FillMode::SOLID;
        DepthState     depthState;
    };
}
