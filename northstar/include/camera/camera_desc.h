#pragma once

#include "camera_types.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "../texture/texture.h"

namespace Northstar
{
    // ==================== CAMERA DESC ====================
    struct CameraDesc
    {
        // Render Destination Mode (can be dynamically toggled via camera->SetRenderType)
        CameraRenderType renderType            = CameraRenderType::TO_WINDOW;

        // Transform
        Vec3           position                = Vec3::Zero();
        Quat           rotation                = Quat::Identity();

        // Projection
        ProjectionType projectionType          = ProjectionType::PERSPECTIVE;
        FLOAT32        fov                     = 60.0f;       // degrees (converted to radians internally)
        FLOAT32        aspectRatio             = 16.0f / 9.0f;
        FLOAT32        nearPlane               = 0.1f;
        FLOAT32        farPlane                = 1000.0f;

        // Orthographic (used only when projectionType == Orthographic)
        FLOAT32        orthoHeight             = 10.0f;

        // Rendering
        FLOAT32        depth                   = 0.0f;        // camera render order (lower = earlier)
        UINT32         cullingMask             = 0xFFFFFFFF;  // layer bitmask for culling
        
        // Render Target & Viewport (Optional)
        Texture*       renderTarget            = nullptr;     // bound texture when renderType == TO_TEXTURE
        Texture*       depthStencilTarget      = nullptr;     // optional dedicated depth-stencil buffer
        // Viewport (in pixels, 0.0f means full target texture / window swapchain)
        FLOAT32        viewportX               = 0.0f;
        FLOAT32        viewportY               = 0.0f;
        FLOAT32        viewportWidth           = 0.0f;
        FLOAT32        viewportHeight          = 0.0f;
    };
}