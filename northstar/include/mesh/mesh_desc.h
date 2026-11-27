#pragma once

#include "vertex.h"
#include "../nvg/nvg_types.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include <memory>

namespace Northstar
{
    struct MeshDesc
    {
        const Vertex*                           vertices                      = nullptr;
        UINT32                                  vertexCount                   = 0;
        const UINT32*                           indices                       = nullptr;
        UINT32                                  indexCount                    = 0;
        const NVGSubmeshRange*                  submeshes                     = nullptr;
        UINT32                                  submeshCount                  = 0;
        AABB                                    boundingBox                   = {};
        GeometryMode                            geometryMode                  = GeometryMode::STANDARD;
        NVGBuildConfig                          nvgBuildConfig                = {};
        const BYTE*                             serializedVirtualGeometryData = nullptr;
        SIZE_T                                  serializedVirtualGeometrySize = 0;
        const char*                             virtualGeometryFilePath       = nullptr;
        const INVGPageDataSource*               customPageDataSource          = nullptr;
        std::shared_ptr<INVGPageStreamProvider> pageStreamProvider            = nullptr;
    };
}