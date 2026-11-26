#pragma once

#include <vector>
#include "../render_object/render_object.h"
#include "../camera/camera.h"
#include "../light/light.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "scene_types.h"

namespace Northstar
{
    // ==================== SCENE DESC ====================
    struct SceneDesc
    {
        // Scene Objects & Active Elements
        std::vector<RenderObject*> renderObjects;
        std::vector<Camera*>       cameras;
        std::vector<Light*>        lights;

        // Main Camera & Directional Light
        Camera*                    mainCamera       = nullptr;
        Light*                     directionalLight = nullptr;

        // Ambient Lighting
        Color                      ambientColor     = Color(0.1f, 0.1f, 0.15f, 1.0f);
        FLOAT32                    ambientIntensity = 1.0f;
    };
}