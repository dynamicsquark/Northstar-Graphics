#pragma once

#include "../types/northstar_math.h"
#include "../shader/shader_types.h"
#include "../types/northstar_types.h"

#include <cstddef>
#include <string_view>

namespace Northstar
{
    struct Vertex
    {
        Vec3   position;                                      // VERTEX_POSITION     (offset 0,  12 bytes)
        Vec3   normal;                                        // VERTEX_NORMAL       (offset 12, 12 bytes)
        Vec3   tangent;                                       // VERTEX_TANGENT      (offset 24, 12 bytes)
        Vec3   bitangent;                                     // VERTEX_BITANGENT    (offset 36, 12 bytes)
        Vec2   texCoord0;                                     // VERTEX_UV0          (offset 48,  8 bytes)
        Vec2   texCoord1;                                     // VERTEX_UV1          (offset 56,  8 bytes)
        Vec4   color;                                         // VERTEX_COLOR        (offset 64, 16 bytes)
        UINT32 boneIndices[4] = { 0u, 0u, 0u, 0u };           // VERTEX_BONE_INDICES (offset 80, 16 bytes)
        Vec4   boneWeights    = Vec4(0.0f, 0.0f, 0.0f, 0.0f); // VERTEX_BONE_WEIGHTS (offset 96, 16 bytes)
    };

    static_assert(sizeof(Vertex) == 112, "Northstar::Vertex must be 112 bytes.");
    static_assert(offsetof(Vertex, boneIndices) == 80, "Vertex::boneIndices must be at byte offset 80.");
    static_assert(offsetof(Vertex, boneWeights) == 96, "Vertex::boneWeights must be at byte offset 96.");
}
