#pragma once

#include "render_object_types.h"
#include "render_object_bitwise.h"
#include "../mesh/mesh.h"
#include "../material/material.h"
#include "../lod_group/lod_group.h"
#include "../types/northstar_math.h"

namespace Northstar
{
    // ==================== RENDER OBJECT DESC ====================
    struct RenderObjectDesc
    {
        Mesh*              mesh                  = nullptr;
        Material*          material              = nullptr;
        LODGroup*          lodGroup              = nullptr;
        Mat4               worldMatrix           = Mat4::Identity();
        AABB               worldAABB             = {};
        UINT32             layerMask             = 0x00000001u;
        RenderObjectFlags  flags                 = RenderObjectFlags::VISIBLE        |
                                                   RenderObjectFlags::FRUSTUM_CULL   |
                                                   RenderObjectFlags::CAST_SHADOW    |
                                                   RenderObjectFlags::RECEIVE_SHADOW |
                                                   RenderObjectFlags::DISTANCE_CULL;
    };
}