#pragma once

#include <vector>
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "../mesh/mesh.h"

namespace Northstar
{
    // ==================== LOD LEVEL ====================
    struct LODLevel
    {
        Mesh*   mesh                           = nullptr;
        FLOAT32 screenRelativeTransitionHeight = 0.5f;
    };

    // ==================== LOD GROUP DESC ====================
    struct LODGroupDesc
    {
        std::vector<LODLevel> levels;

        Vec3    localCenter = Vec3::Zero();
        FLOAT32 localRadius = 1.0f;
    };
}
