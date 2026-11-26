#pragma once

#include "vertex.h"
#include "../nvg/nvg_types.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"

namespace Northstar
{
    struct MeshDesc
    {
        const Vertex*              vertices                      = nullptr;
        UINT32                     vertexCount                   = 0;
        const UINT32*              indices                       = nullptr;
        UINT32                     indexCount                    = 0;
        AABB                       boundingBox                   = {};
        GeometryMode               geometryMode                  = GeometryMode::STANDARD;
        NVGBuildConfig             nvgBuildConfig                = {};
        const BYTE*                serializedVirtualGeometryData = nullptr;
        SIZE_T                     serializedVirtualGeometrySize = 0;
        const INVGPageDataSource*  customPageDataSource          = nullptr;
    };
}