#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../../include/mesh/vertex.h"
#include <string>
#include <vector>

namespace Northstar
{
    struct NVGVirtualPagePayload
    {
        UINT32              localPageIndex = 0u;
        UINT32              lodLevel       = 0u;
        BOOL                isRootPage     = false;
        UINT32              clusterCount   = 0u;
        UINT32              usedWords      = 0u;
        std::vector<UINT32> words; // Exactly kNVGPageWordCount (8192) words = 32 KiB
    };

    struct NVGVirtualMeshAsset
    {
        UINT32                              virtualMeshId      = 0u;
        AABB                                localBounds        = {};
        BOOL                                hasSkinning        = false;
        UINT32                              lodLevelCount      = 0u;
        UINT32                              rootPageCount      = 0u;
        UINT32                              rootNodeIndex      = 0u;
        NVGBuildReport                      buildReport{};
        std::vector<NVGClusterMetaGPUData>  clusters;
        std::vector<NVGClusterGroupGPUData> clusterGroups;
        std::vector<UINT32>                 childGroupIndices;
        std::vector<NVGHierarchyNodeGPUData> bvhNodes;
        std::vector<NVGVirtualPagePayload>  pages;
    };

    // Default in-memory page provider backed by an NVGVirtualMeshAsset's compressed pages
    class NVGMemoryPageDataSource : public INVGPageDataSource
    {
    public:
        void RegisterAsset(UINT32 virtualMeshId, const std::vector<NVGVirtualPagePayload>* pages);
        void UnregisterAsset(UINT32 virtualMeshId);
        void Clear();

        BOOL ReadVirtualPageData(
            UINT32 virtualMeshId,
            UINT32 localPageIndex,
            BYTE*  outPageBytes,
            UINT32 maxBytes) const override;

    private:
        struct Entry
        {
            UINT32                                    virtualMeshId = 0u;
            const std::vector<NVGVirtualPagePayload>* pages         = nullptr;
        };
        std::vector<Entry> m_entries;
    };

    class NVGBuilder
    {
    public:
        static BOOL BuildVirtualMesh(
            const Vertex*         vertices,
            UINT32                vertexCount,
            const UINT32*         indices,
            UINT32                indexCount,
            const NVGBuildConfig& config,
            NVGVirtualMeshAsset&  outAsset);

        // Decodes a compressed cluster from a page payload back into standard Northstar::Vertex + UINT32 indices
        // (Used for exact CPU-side round-trip verification & error auditing)
        static BOOL DecodeClusterCPU(
            const UINT32*        pageWords,
            UINT32               pageWordCount,
            UINT32               clusterWordOffset,
            std::vector<Vertex>& outVertices,
            std::vector<UINT32>& outIndices);

        // Standalone versioned binary serialization ("NVG1") with FNV-1a 64-bit integrity checksum
        static BOOL Serialize(
            const NVGVirtualMeshAsset& asset,
            std::vector<BYTE>&         outBlob);

        static BOOL Deserialize(
            const BYTE*          data,
            SIZE_T               size,
            NVGVirtualMeshAsset& outAsset,
            std::string&         outError);

        // IEEE-754 float32 <-> float16 helpers (matches HLSL f32tof16 / f16tof32 bitwise)
        static UINT16  FloatToHalf(FLOAT32 value) noexcept;
        static FLOAT32 HalfToFloat(UINT16 bits) noexcept;

        // Octahedral unit normal encode / decode (16-bit x 2 SNORM)
        static UINT32 PackOctahedralNormal(const Vec3& unitNormal) noexcept;
        static Vec3   UnpackOctahedralNormal(UINT32 packed) noexcept;

        // Orthonormal tangent frame angle + bitangent handedness encode / decode
        static UINT16 PackTangentFrame(const Vec3& unitNormal, const Vec3& unitTangent, const Vec3& unitBitangent) noexcept;
        static void   UnpackTangentFrame(const Vec3& unitNormal, UINT16 packed, Vec3& outTangent, Vec3& outBitangent) noexcept;
    };
}
