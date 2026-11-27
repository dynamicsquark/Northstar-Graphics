#pragma once

#include "../../include/nvg/nvg_types.h"
#include "../../include/mesh/vertex.h"
#include <atomic>
#include <functional>
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
        UINT32                               virtualMeshId      = 0u;
        AABB                                 localBounds        = {};
        BOOL                                 hasSkinning        = false;
        UINT32                               lodLevelCount      = 0u;
        UINT32                               rootPageCount      = 0u;
        UINT32                               rootNodeIndex      = 0u;
        NVGBuildReport                       buildReport{};
        std::vector<NVGClusterMetaGPUData>   clusters;
        std::vector<NVGClusterGroupGPUData>  clusterGroups;
        std::vector<UINT32>                  childGroupIndices;
        std::vector<NVGHierarchyNodeGPUData> bvhNodes;
        std::vector<NVGVirtualPagePayload>   pages;
    };

    // Binary NVG2 on-disk header & page directory entry (shared between NVGBuilder and NVGFilePageStreamProvider)
    #pragma pack(push, 1)
    struct NVG2BinaryHeader
    {
        UINT32  magic                      = kNVG2BinaryMagic;
        UINT32  version                    = kNVG2BinaryVersion;
        UINT32  flags                      = 0u; // bit 0 = hasSkinning, bit 1 = lz4PageCompressionEnabled
        UINT32  lodLevelCount              = 0u;
        UINT32  rootPageCount              = 0u;
        UINT32  rootNodeIndex              = 0u;
        UINT32  clusterCount               = 0u;
        UINT32  groupCount                 = 0u;
        UINT32  childGroupIdCount          = 0u;
        UINT32  bvhNodeCount               = 0u;
        UINT32  pageCount                  = 0u;
        UINT32  pageSizeBytes              = kNVGPageSizeBytes;
        Vec3    boundsMin                  = Vec3::Zero();
        FLOAT32 maxQuantError              = 0.0f;
        Vec3    boundsMax                  = Vec3::Zero();
        FLOAT32 compressionRatio           = 1.0f;

        UINT32  sourceVertexCount          = 0u;
        UINT32  sourceTriangleCount        = 0u;
        UINT32  validTriangleCount         = 0u;
        UINT32  degenerateTrianglesRemoved = 0u;
        UINT32  sanitizedVertexCount       = 0u;
        UINT32  nonManifoldEdgeCount       = 0u;
        UINT32  boundaryEdgeCount          = 0u;
        UINT32  seamEdgeCount              = 0u;
        UINT32  disconnectedIslandCount    = 0u;
        UINT32  lod0ClusterCount           = 0u;
        UINT32  rootClusterCount           = 0u;
        FLOAT32 rootGeometricError         = 0.0f;

        UINT64  uncompressedSourceBytes    = 0ull;
        UINT64  compressedPagesBytes       = 0ull;
        UINT64  tablesByteOffset           = 0ull;
        UINT64  tablesByteSize             = 0ull;
        UINT64  pageDirectoryByteOffset    = 0ull;
        UINT64  pageDataByteOffset         = 0ull;
        UINT64  totalFileByteSize          = 0ull;
        UINT64  headerAndTablesChecksum    = 0ull; // FNV-1a 64-bit over header (with this field = 0) + tables + page directory
    };

    struct NVG2PageDirectoryEntry
    {
        UINT64 byteOffsetInFile     = 0ull;
        UINT32 compressedByteSize   = 0u;
        UINT32 uncompressedByteSize = 0u; // usedWords * 4 (<= 32768)
        UINT32 localPageIndex       = 0u;
        UINT16 lodLevel             = 0u;
        UINT16 codecAndFlags        = 0u; // bit 0 = isRootPage, bit 1 = isLZ4Compressed
        UINT32 clusterCount         = 0u;
        UINT32 crc32c               = 0u; // Castagnoli CRC32C of uncompressedByteSize bytes
    };
    #pragma pack(pop)
    static_assert(sizeof(NVG2BinaryHeader) == 192, "NVG2BinaryHeader must be 192 bytes.");
    static_assert(sizeof(NVG2PageDirectoryEntry) == 32, "NVG2PageDirectoryEntry must be 32 bytes.");

    using NVGBuildProgressCallback = std::function<void(FLOAT32 normalizedProgress, const char* phaseName)>;

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
            const Vertex*                  vertices,
            UINT32                         vertexCount,
            const UINT32*                  indices,
            UINT32                         indexCount,
            const NVGBuildConfig&          config,
            NVGVirtualMeshAsset&           outAsset);

        static BOOL BuildVirtualMesh(
            const Vertex*                  vertices,
            UINT32                         vertexCount,
            const UINT32*                  indices,
            UINT32                         indexCount,
            const NVGSubmeshRange*         submeshes,
            UINT32                         submeshCount,
            const NVGBuildConfig&          config,
            NVGVirtualMeshAsset&           outAsset,
            const std::atomic<bool>*       cancelFlag       = nullptr,
            const NVGBuildProgressCallback& progressCallback = nullptr);

        // Decodes a compressed cluster from a page payload back into standard Northstar::Vertex + UINT32 indices
        // (Supports both NVG2 2^P integer lattice and legacy NVG1 float origin/extent cluster headers)
        static BOOL DecodeClusterCPU(
            const UINT32*        pageWords,
            UINT32               pageWordCount,
            UINT32               clusterWordOffset,
            std::vector<Vertex>& outVertices,
            std::vector<UINT32>& outIndices,
            UINT32*              outMaterialSlot = nullptr);

        // Extracts a clean coarse proxy / fallback mesh from the asset's pinned Root LOD clusters
        static BOOL ExtractFallbackMesh(
            const NVGVirtualMeshAsset&    asset,
            std::vector<Vertex>&          outVertices,
            std::vector<UINT32>&          outIndices,
            std::vector<NVGSubmeshRange>* outSubmeshes = nullptr);

        // Standalone versioned binary serialization:
        //   Serialize   -> writes NVG2 (LZ4 pages + FNV-1a 64 header/table checksum + per-page CRC32C)
        //   Deserialize -> loads both NVG2 and legacy NVG1 with full deterministic corruption rejection
        static BOOL Serialize(
            const NVGVirtualMeshAsset& asset,
            std::vector<BYTE>&         outBlob);

        static BOOL Deserialize(
            const BYTE*          data,
            SIZE_T               size,
            NVGVirtualMeshAsset& outAsset,
            std::string&         outError);

        // Parses only header + metadata tables + page directory from an NVG2 blob without retaining streaming page bytes in RAM
        static BOOL ParseNVG2MetadataAndDirectory(
            const BYTE*                          headerAndTablesData,
            SIZE_T                               availableBytes,
            NVG2BinaryHeader&                    outHeader,
            NVGVirtualMeshAsset&                 outMetadataOnlyAsset,
            std::vector<NVG2PageDirectoryEntry>& outPageDirectory,
            std::string&                         outError);

        // IEEE-754 float32 <-> float16 helpers (round-to-nearest-even with arithmetic exponent carry)
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
