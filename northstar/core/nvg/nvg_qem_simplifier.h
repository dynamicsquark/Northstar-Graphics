#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../../include/mesh/vertex.h"
#include <vector>

namespace Northstar
{
    struct NVGQEMTriangle
    {
        UINT32 v[3]         = { 0u, 0u, 0u };
        UINT16 materialSlot = 0u;
    };

    struct NVGQEMSimplifierInput
    {
        const std::vector<Vertex>*        vertices                = nullptr;
        const std::vector<UINT32>*        spatialIds              = nullptr;
        const std::vector<NVGQEMTriangle>* triangles              = nullptr;
        const std::vector<UINT32>*        sortedLockedSpatialIds  = nullptr;
        UINT32                            targetTriangleCount     = 0u;
        NVGBuildConfig                    config{};
        BOOL                              hasSkinning             = false;
        BOOL                              relaxAttributeSeams     = false;
        BOOL                              allowCoarseRidgeFlatten = false;
        BOOL                              allowRootNonPlanarFold  = false;
    };

    // Clean-room double-precision Coupled Position + Wedge-Attribute QEM Simplifier:
    //   - Lossless 2^E IEEE-754 exponent normalization + local centroid rebasing
    //   - Area-weighted Garland-Heckbert planes + coupled in-plane wedge attribute gradients (Schur complement)
    //   - 4-stage solver cascade (constrained/unconstrained 3x3 LUP + 1-step residual refinement -> 1D segment -> endpoints/midpoint)
    //   - Strict Z2 homology exterior boundary lock, multi-material seam lock, mirrored-UV & tangent-handedness domain separation,
    //     non-manifold (>2 incidence) & open-boundary ear-collapse protection
    //   - 100% deterministic total ordering (zero hash-iteration order dependence)
    class NVGQEMSimplifier
    {
    public:
        static FLOAT32 SimplifyGroup(
            const NVGQEMSimplifierInput& input,
            std::vector<Vertex>&         outVertices,
            std::vector<UINT32>&         outSpatialIds,
            std::vector<NVGQEMTriangle>& outTriangles);
    };
}
