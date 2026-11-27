#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../../include/types/pipeline_state_types.h"
#include "nvg_builder.h"

#include <vector>

namespace Northstar
{
    struct NVGSWRasterInputCluster
    {
        UINT32              visibleClusterIndex = 0u;
        CullMode            cullMode            = CullMode::BACK;
        Mat4                worldMatrix         = Mat4::Identity();
        std::vector<Vec3>   localPositions;
        std::vector<UINT32> indices; // 3 * triangleCount local vertex indices in [0 .. 255]
    };

    struct NVGSWRasterConfigCPU
    {
        UINT32  viewportWidth       = 1u;
        UINT32  viewportHeight      = 1u;
        FLOAT32 nearPlane           = 0.1f;
        BOOL    isOrthographic      = false;
        BOOL    isReversedZ         = false;
        BOOL    enableNearClip      = true;
        UINT32  maxSWTriangleSpanPx = 2048u; // Pre-commit span cap matching GPU maxSWTriangleSpanPx (64 in hybrid, 2048 in full VisBuffer)
    };

    // Clean-Room Reference Software Micropoly Rasterizer (16.8 sub-pixel snap,
    // homogeneous near-plane polygon fan clipper, pre-commit span/guardband HW fallback,
    // MinPixel local rebasing, branchless D3D Top-Left fill rule, exact affine DeviceZ plane,
    // and 64-bit VisPixel64 deterministic tie-breaking).
    class NVGSoftwareRasterizerCPU
    {
    public:
        static BOOL ExtractClusterFromAsset(
            const NVGVirtualMeshAsset& asset,
            UINT32                     clusterIndex,
            UINT32                     visibleClusterIndex,
            CullMode                   cullMode,
            const Mat4&                worldMatrix,
            const Mat4*                skinningPalette,
            UINT32                     boneCount,
            NVGSWRasterInputCluster&   outCluster);

        static void RasterizeClustersToVisBuffer64(
            const NVGSWRasterInputCluster* clusters,
            SIZE_T                         clusterCount,
            const Mat4&                    vpMatrix,
            const NVGSWRasterConfigCPU&    config,
            std::vector<UINT64>&           outVisBuffer64,
            std::vector<UINT8>*            outClusterAbortedToHW = nullptr);

        static void RasterizeClusterToShadowPageR32(
            const NVGSWRasterInputCluster& cluster,
            const Mat4&                    worldToPageClip,
            FLOAT32                        depthBiasNdc,
            BOOL                           isReversedZ,
            UINT32*                        ioShadowPage128x128,
            BOOL*                          outAbortedToHW = nullptr);
    };
}
