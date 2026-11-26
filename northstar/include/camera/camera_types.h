#pragma once

#include "../types/northstar_types.h"

namespace Northstar
{
    enum class ProjectionType : UINT8
    {
        PERSPECTIVE  = 0,
        ORTHOGRAPHIC = 1,

        Perspective  = PERSPECTIVE,
        Orthographic = ORTHOGRAPHIC
    };

    enum class CameraRenderType : UINT8
    {
        TO_TEXTURE = 0,
        TO_WINDOW  = 1,

        ToTexture  = TO_TEXTURE,
        ToWindow   = TO_WINDOW
    };
}
