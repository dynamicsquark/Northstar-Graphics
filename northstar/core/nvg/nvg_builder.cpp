#include "nvg_builder.h"
#include "nvg_codec_lz4.h"
#include "nvg_qem_simplifier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace Northstar
{
    namespace
    {
        inline constexpr FLOAT32 kNVGPi           = 3.14159265358979323846f;
        inline constexpr FLOAT32 kNVGTwoPi        = 6.28318530717958647692f;
        inline constexpr FLOAT32 kNVGRootErrorInf = 1.0e30f;

        inline BOOL IsFiniteFloat(FLOAT32 v) noexcept
        {
            return std::isfinite(v) ? true : false;
        }

        inline BOOL IsFiniteVec3(const Vec3& v) noexcept
        {
            return (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) ? true : false;
        }

        inline FLOAT32 ClampF(FLOAT32 v, FLOAT32 lo, FLOAT32 hi) noexcept
        {
            return (v < lo) ? lo : ((v > hi) ? hi : v);
        }

        inline UINT32 BitCastFloatToUInt(FLOAT32 f) noexcept
        {
            UINT32 u = 0u;
            std::memcpy(&u, &f, sizeof(UINT32));
            return u;
        }

        inline FLOAT32 BitCastUIntToFloat(UINT32 u) noexcept
        {
            FLOAT32 f = 0.0f;
            std::memcpy(&f, &u, sizeof(FLOAT32));
            return f;
        }

        inline UINT32 CanonicalFloatBits(FLOAT32 f) noexcept
        {
            return (f == 0.0f) ? 0u : BitCastFloatToUInt(f);
        }

        inline UINT32 ExpandBits10(UINT32 v) noexcept
        {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        inline UINT32 Morton3D(FLOAT32 nx, FLOAT32 ny, FLOAT32 nz) noexcept
        {
            const UINT32 ix = static_cast<UINT32>(ClampF(nx * 1023.0f, 0.0f, 1023.0f));
            const UINT32 iy = static_cast<UINT32>(ClampF(ny * 1023.0f, 0.0f, 1023.0f));
            const UINT32 iz = static_cast<UINT32>(ClampF(nz * 1023.0f, 0.0f, 1023.0f));
            return (ExpandBits10(ix) << 2u) | (ExpandBits10(iy) << 1u) | ExpandBits10(iz);
        }

        inline UINT64 MakeUndirectedEdgeKey(UINT32 a, UINT32 b) noexcept
        {
            const UINT32 lo = (a < b) ? a : b;
            const UINT32 hi = (a < b) ? b : a;
            return (static_cast<UINT64>(lo) << 32u) | static_cast<UINT64>(hi);
        }

        struct Vec4Sphere
        {
            Vec3    center = Vec3::Zero();
            FLOAT32 radius = 0.0f;
        };

        inline Vec4Sphere MergeSpheres(const Vec4Sphere& a, const Vec4Sphere& b) noexcept
        {
            if (a.radius <= 0.0f) return b;
            if (b.radius <= 0.0f) return a;
            const Vec3    diff = b.center - a.center;
            const FLOAT32 dist = diff.Length();
            if (dist + b.radius <= a.radius) return a;
            if (dist + a.radius <= b.radius) return b;
            const FLOAT32 newR = 0.5f * (dist + a.radius + b.radius);
            const FLOAT32 t    = (dist > 1e-8f) ? ((newR - a.radius) / dist) : 0.0f;
            Vec4Sphere res{};
            res.center = a.center + diff * t;
            res.radius = newR;
            return res;
        }

        struct CanonicalVertexKey
        {
            std::array<UINT32, 28> words{};

            bool operator==(const CanonicalVertexKey& o) const noexcept
            {
                return words == o.words;
            }
        };

        struct CanonicalVertexKeyHash
        {
            SIZE_T operator()(const CanonicalVertexKey& k) const noexcept
            {
                return static_cast<SIZE_T>(NVGCodecLZ4::ComputeFNV1a64(
                    reinterpret_cast<const BYTE*>(k.words.data()),
                    k.words.size() * sizeof(UINT32)));
            }
        };

        inline CanonicalVertexKey MakeCanonicalVertexKey(const Vertex& v, BOOL hasSkinning) noexcept
        {
            CanonicalVertexKey k{};
            k.words[0]  = CanonicalFloatBits(v.position.x);
            k.words[1]  = CanonicalFloatBits(v.position.y);
            k.words[2]  = CanonicalFloatBits(v.position.z);
            k.words[3]  = CanonicalFloatBits(v.normal.x);
            k.words[4]  = CanonicalFloatBits(v.normal.y);
            k.words[5]  = CanonicalFloatBits(v.normal.z);
            k.words[6]  = CanonicalFloatBits(v.tangent.x);
            k.words[7]  = CanonicalFloatBits(v.tangent.y);
            k.words[8]  = CanonicalFloatBits(v.tangent.z);
            k.words[9]  = CanonicalFloatBits(v.bitangent.x);
            k.words[10] = CanonicalFloatBits(v.bitangent.y);
            k.words[11] = CanonicalFloatBits(v.bitangent.z);
            k.words[12] = CanonicalFloatBits(v.texCoord0.x);
            k.words[13] = CanonicalFloatBits(v.texCoord0.y);
            k.words[14] = CanonicalFloatBits(v.texCoord1.x);
            k.words[15] = CanonicalFloatBits(v.texCoord1.y);
            k.words[16] = CanonicalFloatBits(v.color.x);
            k.words[17] = CanonicalFloatBits(v.color.y);
            k.words[18] = CanonicalFloatBits(v.color.z);
            k.words[19] = CanonicalFloatBits(v.color.w);
            if (hasSkinning)
            {
                k.words[20] = v.boneIndices[0];
                k.words[21] = v.boneIndices[1];
                k.words[22] = v.boneIndices[2];
                k.words[23] = v.boneIndices[3];
                k.words[24] = CanonicalFloatBits(v.boneWeights.x);
                k.words[25] = CanonicalFloatBits(v.boneWeights.y);
                k.words[26] = CanonicalFloatBits(v.boneWeights.z);
                k.words[27] = CanonicalFloatBits(v.boneWeights.w);
            }
            return k;
        }

        struct BuildCluster
        {
            UINT32                  lodLevel          = 0u;
            UINT16                  materialSlot      = 0u;
            UINT16                  reservedPad       = 0u;
            UINT32                  globalIndex       = 0u;
            UINT32                  selfGroupIndex    = 0u;
            UINT32                  parentGroupIndex  = kNVGInvalidIndex;
            UINT32                  childGroupStart   = 0u;
            UINT32                  childGroupCount   = 0u;
            std::vector<Vertex>     vertices;
            std::vector<UINT32>     spatialIds;       // Parallel to vertices: welded 3D spatial ID
            std::vector<UINT32>     indices;          // Local indices [0 .. vertices.size()-1], 3 per triangle
            AABB                    aabb              = {};
            Vec4Sphere              sphere            = {};
            Vec4Sphere              selfGroupSphere   = {};
            Vec4Sphere              parentGroupSphere = {};
            FLOAT32                 selfGroupError    = 0.0f;
            FLOAT32                 parentGroupError  = kNVGRootErrorInf;
            Vec4                    normalConeAxisSin = Vec4(0.0f, 1.0f, 0.0f, -1.0f);
            FLOAT32                 thinFeatureMetric = 0.0f;
            FLOAT32                 selfGroupThin     = 0.0f;
            FLOAT32                 parentGroupThin   = 0.0f;
            FLOAT32                 maxEdgeLength     = 0.0f;
            std::vector<UINT64>     boundaryEdges;    // Odd-multiplicity (Z2) perimeter spatial edges
            std::vector<UINT64>     allSpatialEdges;  // Sorted unique spatial edges (catches 2+2 non-manifold seams)
            UINT32                  virtualPageId     = kNVGInvalidIndex;
            UINT32                  wordOffsetInPage  = 0u;
            std::vector<UINT32>     encodedWords;
        };

        struct BuildGroup
        {
            UINT32              lodLevel         = 0u;
            UINT32              globalIndex      = 0u;
            UINT32              parentGroupIndex = kNVGInvalidIndex;
            std::vector<UINT32> clusterIndices;
            std::vector<UINT32> childGroupIndices;
            Vec4Sphere          sphere           = {};
            FLOAT32             error            = 0.0f;
        };

        static inline UINT32 QuantizeThinMetric14(FLOAT32 metric) noexcept
        {
            const FLOAT32 clamped = ClampF(metric, 0.0f, 4.0f);
            return static_cast<UINT32>(std::lround((clamped / 4.0f) * 16383.0f)) & 0x3FFFu;
        }

        // Deterministic O(N log N) spatial weld ID assignment (zero unordered_map iteration dependence)
        static void BuildSpatialWeldIdsDeterministic(
            const std::vector<Vertex>& vertices,
            double                     weldEpsilon,
            std::vector<UINT32>&       outSpatialIds)
        {
            outSpatialIds.assign(vertices.size(), 0u);
            if (vertices.empty()) return;

            const double invEps = (weldEpsilon > 1.0e-12) ? (1.0 / weldEpsilon) : 1.0e8;
            struct KeyedVertex
            {
                INT64  kx  = 0;
                INT64  ky  = 0;
                INT64  kz  = 0;
                UINT32 idx = 0u;
            };
            std::vector<KeyedVertex> items(vertices.size());
            for (SIZE_T i = 0; i < vertices.size(); ++i)
            {
                const Vec3& p = vertices[i].position;
                items[i].kx  = static_cast<INT64>(std::llround(static_cast<double>(p.x) * invEps));
                items[i].ky  = static_cast<INT64>(std::llround(static_cast<double>(p.y) * invEps));
                items[i].kz  = static_cast<INT64>(std::llround(static_cast<double>(p.z) * invEps));
                items[i].idx = static_cast<UINT32>(i);
            }

            std::sort(items.begin(), items.end(), [](const KeyedVertex& a, const KeyedVertex& b) {
                if (a.kx != b.kx) return a.kx < b.kx;
                if (a.ky != b.ky) return a.ky < b.ky;
                if (a.kz != b.kz) return a.kz < b.kz;
                return a.idx < b.idx;
            });

            UINT32 curId = 0u;
            outSpatialIds[items[0].idx] = 0u;
            for (SIZE_T i = 1; i < items.size(); ++i)
            {
                if (items[i].kx != items[i - 1].kx ||
                    items[i].ky != items[i - 1].ky ||
                    items[i].kz != items[i - 1].kz)
                {
                    ++curId;
                }
                outSpatialIds[items[i].idx] = curId;
            }
        }

        // Optimizes local triangle order for a 32-entry FIFO post-transform vertex cache and reorders
        // cluster vertices in first-reference order.
        static void OptimizeClusterVertexCacheAndOrder(BuildCluster& c)
        {
            const UINT32 numVerts = static_cast<UINT32>(c.vertices.size());
            const UINT32 numTris  = static_cast<UINT32>(c.indices.size() / 3u);
            if (numVerts == 0u || numTris <= 1u) return;

            std::vector<UINT32> vertValence(numVerts, 0u);
            std::vector<UINT32> vertTriOffsets(numVerts + 1u, 0u);
            for (UINT32 idx : c.indices)
            {
                if (idx < numVerts) ++vertValence[idx];
            }
            for (UINT32 v = 0u; v < numVerts; ++v)
            {
                vertTriOffsets[v + 1u] = vertTriOffsets[v] + vertValence[v];
            }
            std::vector<UINT32> vertTris(vertTriOffsets[numVerts], 0u);
            std::vector<UINT32> fillCur = vertTriOffsets;
            for (UINT32 t = 0u; t < numTris; ++t)
            {
                for (UINT32 k = 0u; k < 3u; ++k)
                {
                    const UINT32 v = c.indices[t * 3u + k];
                    vertTris[fillCur[v]++] = t;
                }
            }

            std::vector<UINT8>  triEmitted(numTris, 0u);
            std::vector<UINT32> liveValence = vertValence;
            std::array<UINT32, 32> fifo{};
            fifo.fill(kNVGInvalidIndex);
            UINT32 fifoCount = 0u;

            auto cachePosition = [&](UINT32 v) noexcept -> INT32 {
                for (UINT32 i = 0u; i < fifoCount; ++i)
                {
                    if (fifo[i] == v) return static_cast<INT32>(i);
                }
                return -1;
            };

            auto vertexScore = [&](UINT32 v) noexcept -> FLOAT32 {
                const UINT32 rem = liveValence[v];
                if (rem == 0u) return -1.0f;
                const INT32 pos = cachePosition(v);
                FLOAT32 score = 0.0f;
                if (pos >= 0)
                {
                    if (pos < 3) score = 0.75f;
                    else score = std::pow(1.0f - static_cast<FLOAT32>(pos - 3) / 29.0f, 1.5f);
                }
                score += 2.0f * std::pow(static_cast<FLOAT32>(rem), -0.5f);
                return score;
            };

            std::vector<UINT32> newIndices;
            newIndices.reserve(c.indices.size());

            UINT32 scanSeed = 0u;
            for (UINT32 emitted = 0u; emitted < numTris; ++emitted)
            {
                UINT32  bestTri   = kNVGInvalidIndex;
                FLOAT32 bestScore = -1.0f;

                for (UINT32 ci = 0u; ci < fifoCount; ++ci)
                {
                    const UINT32 cv = fifo[ci];
                    for (UINT32 tiIdx = vertTriOffsets[cv]; tiIdx < vertTriOffsets[cv + 1u]; ++tiIdx)
                    {
                        const UINT32 cand = vertTris[tiIdx];
                        if (triEmitted[cand]) continue;
                        const FLOAT32 sc =
                            vertexScore(c.indices[cand * 3u + 0u]) +
                            vertexScore(c.indices[cand * 3u + 1u]) +
                            vertexScore(c.indices[cand * 3u + 2u]);
                        if (sc > bestScore || (sc == bestScore && cand < bestTri))
                        {
                            bestScore = sc;
                            bestTri   = cand;
                        }
                    }
                }

                if (bestTri == kNVGInvalidIndex)
                {
                    while (scanSeed < numTris && triEmitted[scanSeed]) ++scanSeed;
                    if (scanSeed >= numTris) break;
                    bestTri = scanSeed;
                }

                triEmitted[bestTri] = 1u;
                for (UINT32 k = 0u; k < 3u; ++k)
                {
                    const UINT32 v = c.indices[bestTri * 3u + k];
                    newIndices.push_back(v);
                    if (liveValence[v] > 0u) --liveValence[v];

                    const INT32 existing = cachePosition(v);
                    if (existing >= 0)
                    {
                        for (INT32 s = existing; s > 0; --s)
                        {
                            fifo[static_cast<SIZE_T>(s)] = fifo[static_cast<SIZE_T>(s - 1)];
                        }
                        fifo[0] = v;
                    }
                    else
                    {
                        const UINT32 limit = (fifoCount < 32u) ? fifoCount : 31u;
                        for (UINT32 s = limit; s > 0u; --s)
                        {
                            fifo[s] = fifo[s - 1u];
                        }
                        fifo[0] = v;
                        if (fifoCount < 32u) ++fifoCount;
                    }
                }
            }

            // Reorder vertices & spatialIds in first-reference order of newIndices
            std::vector<UINT32> remap(numVerts, kNVGInvalidIndex);
            std::vector<Vertex> orderedVerts;
            std::vector<UINT32> orderedSpatial;
            orderedVerts.reserve(numVerts);
            orderedSpatial.reserve(numVerts);

            for (UINT32& idx : newIndices)
            {
                if (remap[idx] == kNVGInvalidIndex)
                {
                    remap[idx] = static_cast<UINT32>(orderedVerts.size());
                    orderedVerts.push_back(c.vertices[idx]);
                    orderedSpatial.push_back(c.spatialIds[idx]);
                }
                idx = remap[idx];
            }

            c.vertices   = std::move(orderedVerts);
            c.spatialIds = std::move(orderedSpatial);
            c.indices    = std::move(newIndices);
        }

        static void ComputeClusterBoundsAndMetrics(BuildCluster& c)
        {
            if (c.vertices.empty() || c.indices.size() < 3u)
            {
                c.aabb   = AABB();
                c.sphere = {};
                return;
            }

            Vec3 mn = c.vertices[0].position;
            Vec3 mx = c.vertices[0].position;
            for (SIZE_T i = 1; i < c.vertices.size(); ++i)
            {
                const Vec3& p = c.vertices[i].position;
                if (p.x < mn.x) mn.x = p.x;
                if (p.y < mn.y) mn.y = p.y;
                if (p.z < mn.z) mn.z = p.z;
                if (p.x > mx.x) mx.x = p.x;
                if (p.y > mx.y) mx.y = p.y;
                if (p.z > mx.z) mx.z = p.z;
            }
            c.aabb = AABB(mn, mx);

            const Vec3 center = c.aabb.Center();
            FLOAT32 maxR2 = 0.0f;
            for (const Vertex& v : c.vertices)
            {
                const FLOAT32 d2 = (v.position - center).LengthSq();
                if (d2 > maxR2) maxR2 = d2;
            }
            c.sphere.center = center;
            c.sphere.radius = std::sqrt(maxR2);

            const SIZE_T triCount = c.indices.size() / 3u;
            std::vector<Vec3> faceNormals;
            faceNormals.reserve(triCount);
            Vec3    avgNormal  = Vec3::Zero();
            FLOAT32 totalArea  = 0.0f;
            FLOAT32 maxEdge    = 0.0f;
            FLOAT32 sumEdge    = 0.0f;
            FLOAT32 thinAspect = 0.0f;
            UINT32  validFaces = 0u;

            std::vector<UINT64> rawSpatialEdges;
            rawSpatialEdges.reserve(triCount * 3u);

            for (SIZE_T t = 0; t < triCount; ++t)
            {
                const UINT32 i0 = c.indices[t * 3u + 0u];
                const UINT32 i1 = c.indices[t * 3u + 1u];
                const UINT32 i2 = c.indices[t * 3u + 2u];
                const Vec3& p0  = c.vertices[i0].position;
                const Vec3& p1  = c.vertices[i1].position;
                const Vec3& p2  = c.vertices[i2].position;

                const Vec3 e01 = p1 - p0;
                const Vec3 e12 = p2 - p1;
                const Vec3 e20 = p0 - p2;
                const FLOAT32 l0 = e01.Length();
                const FLOAT32 l1 = e12.Length();
                const FLOAT32 l2 = e20.Length();
                const FLOAT32 triMaxE = (std::max)(l0, (std::max)(l1, l2));
                const FLOAT32 triMinE = (std::min)(l0, (std::min)(l1, l2));
                if (triMaxE > maxEdge) maxEdge = triMaxE;
                sumEdge += (l0 + l1 + l2);

                const Vec3    cr  = e01.Cross(p2 - p0);
                const FLOAT32 crL = cr.Length();
                if (crL > 1e-12f)
                {
                    const FLOAT32 area = 0.5f * crL;
                    const Vec3    n    = cr * (1.0f / crL);
                    faceNormals.push_back(n);
                    avgNormal += n * area;
                    totalArea += area;
                    ++validFaces;

                    const FLOAT32 altitude = (triMaxE > 1e-8f) ? (2.0f * area / triMaxE) : 0.0f;
                    if (triMaxE > 1e-6f && (altitude < triMaxE * 0.25f || (triMinE > 1e-8f && triMaxE / triMinE > 4.0f)))
                    {
                        thinAspect += 1.0f;
                    }
                }

                const UINT32 s0 = c.spatialIds[i0];
                const UINT32 s1 = c.spatialIds[i1];
                const UINT32 s2 = c.spatialIds[i2];
                if (s0 != s1) rawSpatialEdges.push_back(MakeUndirectedEdgeKey(s0, s1));
                if (s1 != s2) rawSpatialEdges.push_back(MakeUndirectedEdgeKey(s1, s2));
                if (s2 != s0) rawSpatialEdges.push_back(MakeUndirectedEdgeKey(s2, s0));
            }

            // Deterministic Z2 boundary & unique spatial edge extraction via sorted runs
            std::sort(rawSpatialEdges.begin(), rawSpatialEdges.end());
            c.boundaryEdges.clear();
            c.allSpatialEdges.clear();
            SIZE_T runStart = 0u;
            while (runStart < rawSpatialEdges.size())
            {
                const UINT64 key = rawSpatialEdges[runStart];
                SIZE_T runEnd = runStart + 1u;
                while (runEnd < rawSpatialEdges.size() && rawSpatialEdges[runEnd] == key)
                {
                    ++runEnd;
                }
                const SIZE_T mult = runEnd - runStart;
                c.allSpatialEdges.push_back(key);
                if ((mult & 1u) != 0u) // Z2 odd multiplicity (1, 3, ...) = topological boundary
                {
                    c.boundaryEdges.push_back(key);
                }
                runStart = runEnd;
            }

            c.maxEdgeLength = maxEdge;

            const FLOAT32 avgLen = avgNormal.Length();
            if (validFaces > 0u && avgLen > 1e-6f)
            {
                const Vec3 axis = avgNormal * (1.0f / avgLen);
                FLOAT32 minDot = 1.0f;
                for (const Vec3& fn : faceNormals)
                {
                    const FLOAT32 d = axis.Dot(fn);
                    if (d < minDot) minDot = d;
                }
                if (minDot > 0.175f)
                {
                    const FLOAT32 halfAngle = std::acos(ClampF(minDot, -1.0f, 1.0f));
                    const FLOAT32 sinAlpha  = std::sin(halfAngle);
                    c.normalConeAxisSin     = Vec4(axis.x, axis.y, axis.z, ClampF(sinAlpha + 0.005f, 0.0f, 0.99f));
                }
                else
                {
                    c.normalConeAxisSin = Vec4(axis.x, axis.y, axis.z, -1.0f);
                }
            }
            else
            {
                c.normalConeAxisSin = Vec4(0.0f, 1.0f, 0.0f, -1.0f);
            }

            const FLOAT32 avgEdge  = (triCount > 0u) ? (sumEdge / static_cast<FLOAT32>(triCount * 3u)) : 0.0f;
            const FLOAT32 thinFrac = (validFaces > 0u) ? (thinAspect / static_cast<FLOAT32>(validFaces)) : 0.0f;
            const Vec3    extents  = c.aabb.Size();
            const FLOAT32 maxExt   = (std::max)(extents.x, (std::max)(extents.y, extents.z));
            const FLOAT32 minExt   = (std::min)(extents.x, (std::min)(extents.y, extents.z));
            FLOAT32 flatness = 0.0f;
            if (maxExt > 1e-5f)
            {
                const FLOAT32 thicknessRatio = minExt / maxExt;
                if (thicknessRatio < 0.08f)
                {
                    flatness = ClampF((0.08f - thicknessRatio) / 0.08f, 0.0f, 1.0f) * 1.5f;
                }
            }
            const FLOAT32 boundaryRatio = (triCount > 0u && totalArea > 1e-8f)
                ? ClampF((static_cast<FLOAT32>(c.boundaryEdges.size()) * avgEdge * avgEdge) / (totalArea * 4.0f), 0.0f, 1.5f)
                : 0.0f;
            c.thinFeatureMetric = ClampF(thinFrac * 2.0f + flatness + boundaryRatio, 0.0f, 4.0f);
        }

        // Partition triangles into homogeneous-material clusters (<= 128 tris, <= 256 unique vertices)
        static void PartitionIntoClusters(
            const std::vector<Vertex>&         verts,
            const std::vector<UINT32>&         spatialIds,
            const std::vector<NVGQEMTriangle>& tris,
            UINT32                             targetTris,
            UINT32                             lodLevel,
            FLOAT32                            maxClusterWorldSpan,
            std::vector<BuildCluster>&         outClusters)
        {
            if (tris.empty()) return;

            const UINT32 maxTris  = (std::min)(kNVGMaxClusterTriangles, (std::max)(16u, targetTris));
            const UINT32 maxVerts = kNVGMaxClusterVertices;
            const SIZE_T numTris  = tris.size();

            Vec3 minB = verts[tris[0].v[0]].position;
            Vec3 maxB = minB;
            std::vector<Vec3> triCentroids(numTris);
            for (SIZE_T t = 0; t < numTris; ++t)
            {
                const Vec3 c = (verts[tris[t].v[0]].position +
                                verts[tris[t].v[1]].position +
                                verts[tris[t].v[2]].position) * (1.0f / 3.0f);
                triCentroids[t] = c;
                if (c.x < minB.x) minB.x = c.x;
                if (c.y < minB.y) minB.y = c.y;
                if (c.z < minB.z) minB.z = c.z;
                if (c.x > maxB.x) maxB.x = c.x;
                if (c.y > maxB.y) maxB.y = c.y;
                if (c.z > maxB.z) maxB.z = c.z;
            }
            const Vec3 invSpan(
                (maxB.x - minB.x > 1e-6f) ? (1.0f / (maxB.x - minB.x)) : 0.0f,
                (maxB.y - minB.y > 1e-6f) ? (1.0f / (maxB.y - minB.y)) : 0.0f,
                (maxB.z - minB.z > 1e-6f) ? (1.0f / (maxB.z - minB.z)) : 0.0f);

            // Deterministic edge-to-triangle adjacency (sorted flat array instead of unordered_map)
            struct EdgeTriPair
            {
                UINT64 edgeKey = 0ull;
                UINT32 triIdx  = 0u;
            };
            std::vector<EdgeTriPair> edgePairs;
            edgePairs.reserve(numTris * 3u);
            for (UINT32 t = 0u; t < static_cast<UINT32>(numTris); ++t)
            {
                const UINT32 s[3] = {
                    spatialIds[tris[t].v[0]],
                    spatialIds[tris[t].v[1]],
                    spatialIds[tris[t].v[2]]
                };
                for (UINT32 e = 0u; e < 3u; ++e)
                {
                    const UINT32 a = s[e];
                    const UINT32 b = s[(e + 1u) % 3u];
                    if (a != b)
                    {
                        edgePairs.push_back({ MakeUndirectedEdgeKey(a, b), t });
                    }
                }
            }
            std::sort(edgePairs.begin(), edgePairs.end(), [](const EdgeTriPair& a, const EdgeTriPair& b) {
                if (a.edgeKey != b.edgeKey) return a.edgeKey < b.edgeKey;
                return a.triIdx < b.triIdx;
            });

            std::vector<std::vector<UINT32>> triAdj(numTris);
            SIZE_T epStart = 0u;
            while (epStart < edgePairs.size())
            {
                const UINT64 key = edgePairs[epStart].edgeKey;
                SIZE_T epEnd = epStart + 1u;
                while (epEnd < edgePairs.size() && edgePairs[epEnd].edgeKey == key)
                {
                    ++epEnd;
                }
                const SIZE_T runLen = epEnd - epStart;
                if (runLen >= 2u && runLen <= 8u)
                {
                    for (SIZE_T i = epStart; i < epEnd; ++i)
                    {
                        const UINT32 ta = edgePairs[i].triIdx;
                        for (SIZE_T j = i + 1u; j < epEnd; ++j)
                        {
                            const UINT32 tb = edgePairs[j].triIdx;
                            if (ta != tb && tris[ta].materialSlot == tris[tb].materialSlot)
                            {
                                triAdj[ta].push_back(tb);
                                triAdj[tb].push_back(ta);
                            }
                        }
                    }
                }
                epStart = epEnd;
            }

            // Order seed candidates by (materialSlot, MortonCode, triIndex) for homogeneous material locality
            std::vector<UINT32> mortonOrder(numTris);
            std::vector<UINT32> mortonCodes(numTris);
            for (UINT32 t = 0u; t < static_cast<UINT32>(numTris); ++t)
            {
                mortonOrder[t] = t;
                const Vec3 n   = (triCentroids[t] - minB);
                mortonCodes[t] = Morton3D(n.x * invSpan.x, n.y * invSpan.y, n.z * invSpan.z);
            }
            std::sort(mortonOrder.begin(), mortonOrder.end(), [&](UINT32 a, UINT32 b) {
                if (tris[a].materialSlot != tris[b].materialSlot)
                    return tris[a].materialSlot < tris[b].materialSlot;
                if (mortonCodes[a] != mortonCodes[b])
                    return mortonCodes[a] < mortonCodes[b];
                return a < b;
            });

            std::vector<UINT8>  assigned(numTris, 0u);
            std::vector<UINT32> vertexMap(verts.size(), kNVGInvalidIndex);
            std::vector<UINT32> touchedVerts;
            touchedVerts.reserve(maxVerts);
            std::vector<UINT32> frontier;
            std::vector<UINT8>  inFrontier(numTris, 0u);

            SIZE_T        seedCursor = 0u;
            const FLOAT32 diag       = (maxB - minB).Length();
            const FLOAT32 invDiag    = (diag > 1e-6f) ? (1.0f / diag) : 1.0f;

            while (seedCursor < numTris)
            {
                while (seedCursor < numTris && assigned[mortonOrder[seedCursor]] != 0u)
                {
                    ++seedCursor;
                }
                if (seedCursor >= numTris) break;

                const UINT32 seedTri = mortonOrder[seedCursor];
                const UINT16 matSlot = tris[seedTri].materialSlot;

                BuildCluster cluster{};
                cluster.lodLevel     = lodLevel;
                cluster.materialSlot = matSlot;
                cluster.vertices.reserve(maxVerts);
                cluster.spatialIds.reserve(maxVerts);
                cluster.indices.reserve(maxTris * 3u);

                touchedVerts.clear();
                frontier.clear();

                Vec3   clusterCentroidSum = Vec3::Zero();
                Vec3   curMinPos(1.0e30f, 1.0e30f, 1.0e30f);
                Vec3   curMaxPos(-1.0e30f, -1.0e30f, -1.0e30f);
                UINT32 clusterTriCount = 0u;

                auto fitsSpanCap = [&](const NVGQEMTriangle& bt) noexcept -> BOOL {
                    if (maxClusterWorldSpan <= 0.0f || clusterTriCount == 0u) return true;
                    Vec3 nMin = curMinPos;
                    Vec3 nMax = curMaxPos;
                    for (UINT32 k = 0u; k < 3u; ++k)
                    {
                        const Vec3& p = verts[bt.v[k]].position;
                        if (p.x < nMin.x) nMin.x = p.x;
                        if (p.y < nMin.y) nMin.y = p.y;
                        if (p.z < nMin.z) nMin.z = p.z;
                        if (p.x > nMax.x) nMax.x = p.x;
                        if (p.y > nMax.y) nMax.y = p.y;
                        if (p.z > nMax.z) nMax.z = p.z;
                    }
                    return ((nMax.x - nMin.x) <= maxClusterWorldSpan &&
                            (nMax.y - nMin.y) <= maxClusterWorldSpan &&
                            (nMax.z - nMin.z) <= maxClusterWorldSpan);
                };

                auto addTriToCluster = [&](UINT32 tIdx) {
                    assigned[tIdx] = 1u;
                    const NVGQEMTriangle& bt = tris[tIdx];
                    for (UINT32 k = 0u; k < 3u; ++k)
                    {
                        const UINT32 gv = bt.v[k];
                        const Vec3&  vp = verts[gv].position;
                        if (vp.x < curMinPos.x) curMinPos.x = vp.x;
                        if (vp.y < curMinPos.y) curMinPos.y = vp.y;
                        if (vp.z < curMinPos.z) curMinPos.z = vp.z;
                        if (vp.x > curMaxPos.x) curMaxPos.x = vp.x;
                        if (vp.y > curMaxPos.y) curMaxPos.y = vp.y;
                        if (vp.z > curMaxPos.z) curMaxPos.z = vp.z;

                        UINT32 lv = vertexMap[gv];
                        if (lv == kNVGInvalidIndex)
                        {
                            lv = static_cast<UINT32>(cluster.vertices.size());
                            vertexMap[gv] = lv;
                            touchedVerts.push_back(gv);
                            cluster.vertices.push_back(verts[gv]);
                            cluster.spatialIds.push_back(spatialIds[gv]);
                        }
                        cluster.indices.push_back(lv);
                    }
                    clusterCentroidSum += triCentroids[tIdx];
                    ++clusterTriCount;

                    for (UINT32 nb : triAdj[tIdx])
                    {
                        if (assigned[nb] == 0u && inFrontier[nb] == 0u && tris[nb].materialSlot == matSlot)
                        {
                            inFrontier[nb] = 1u;
                            frontier.push_back(nb);
                        }
                    }
                };

                addTriToCluster(seedTri);

                while (clusterTriCount < maxTris)
                {
                    const Vec3 clusterCenter = clusterCentroidSum * (1.0f / static_cast<FLOAT32>(clusterTriCount));
                    UINT32  bestTri      = kNVGInvalidIndex;
                    SIZE_T  bestFrontIdx = SIZE_MAX;
                    FLOAT32 bestScore    = 1.0e30f;

                    for (SIZE_T fi = 0; fi < frontier.size(); )
                    {
                        const UINT32 cand = frontier[fi];
                        if (assigned[cand] != 0u)
                        {
                            inFrontier[cand] = 0u;
                            frontier[fi]     = frontier.back();
                            frontier.pop_back();
                            continue;
                        }

                        const NVGQEMTriangle& bt = tris[cand];
                        if (bt.materialSlot == matSlot && fitsSpanCap(bt))
                        {
                            UINT32 newVerts = 0u;
                            for (UINT32 k = 0u; k < 3u; ++k)
                            {
                                if (vertexMap[bt.v[k]] == kNVGInvalidIndex) ++newVerts;
                            }

                            if (cluster.vertices.size() + newVerts <= maxVerts)
                            {
                                UINT32 sharedClusterNeighbors = 0u;
                                for (UINT32 nb : triAdj[cand])
                                {
                                    if (assigned[nb] != 0u) ++sharedClusterNeighbors;
                                }
                                const FLOAT32 distNorm = (triCentroids[cand] - clusterCenter).Length() * invDiag;
                                const FLOAT32 score =
                                    static_cast<FLOAT32>(newVerts) * 4.0f -
                                    static_cast<FLOAT32>(sharedClusterNeighbors) * 3.0f +
                                    distNorm * 2.0f;
                                if (score < bestScore || (score == bestScore && cand < bestTri))
                                {
                                    bestScore    = score;
                                    bestTri      = cand;
                                    bestFrontIdx = fi;
                                }
                            }
                        }
                        ++fi;
                    }

                    if (bestTri == kNVGInvalidIndex)
                    {
                        if (lodLevel == 0u)
                        {
                            if (clusterTriCount < maxTris / 2u)
                            {
                                for (SIZE_T probe = seedCursor + 1u; probe < numTris && probe < seedCursor + 64u; ++probe)
                                {
                                    const UINT32 cand = mortonOrder[probe];
                                    if (assigned[cand] != 0u || tris[cand].materialSlot != matSlot) continue;
                                    if (!fitsSpanCap(tris[cand])) continue;
                                    if ((triCentroids[cand] - clusterCenter).Length() * invDiag > 0.25f) continue;
                                    UINT32 newVerts = 0u;
                                    for (UINT32 k = 0u; k < 3u; ++k)
                                    {
                                        if (vertexMap[tris[cand].v[k]] == kNVGInvalidIndex) ++newVerts;
                                    }
                                    if (cluster.vertices.size() + newVerts <= maxVerts)
                                    {
                                        bestTri = cand;
                                        break;
                                    }
                                }
                            }
                        }
                        else if (clusterTriCount < maxTris)
                        {
                            FLOAT32 bestIslandDistSq = 1.0e30f;
                            for (SIZE_T probe = seedCursor + 1u; probe < numTris; ++probe)
                            {
                                const UINT32 cand = mortonOrder[probe];
                                if (assigned[cand] != 0u || tris[cand].materialSlot != matSlot) continue;
                                if (!fitsSpanCap(tris[cand])) continue;
                                UINT32 newVerts = 0u;
                                for (UINT32 k = 0u; k < 3u; ++k)
                                {
                                    if (vertexMap[tris[cand].v[k]] == kNVGInvalidIndex) ++newVerts;
                                }
                                if (cluster.vertices.size() + newVerts <= maxVerts)
                                {
                                    const FLOAT32 d2 = (triCentroids[cand] - clusterCenter).LengthSq();
                                    if (d2 < bestIslandDistSq || (d2 == bestIslandDistSq && cand < bestTri))
                                    {
                                        bestIslandDistSq = d2;
                                        bestTri          = cand;
                                    }
                                }
                            }
                        }
                        if (bestTri == kNVGInvalidIndex) break;
                    }
                    else if (bestFrontIdx < frontier.size())
                    {
                        inFrontier[bestTri]    = 0u;
                        frontier[bestFrontIdx] = frontier.back();
                        frontier.pop_back();
                    }

                    addTriToCluster(bestTri);
                }

                for (UINT32 f : frontier) inFrontier[f] = 0u;
                for (UINT32 gv : touchedVerts) vertexMap[gv] = kNVGInvalidIndex;

                OptimizeClusterVertexCacheAndOrder(cluster);
                ComputeClusterBoundsAndMetrics(cluster);
                outClusters.push_back(std::move(cluster));
            }
        }

        // Encodes a cluster into 32-bit page words using NVG2's global 2^P integer lattice + INT32 intOrigin + lodShift
        // and updates c.aabb / c.sphere from the exact quantized decoded positions.
        static FLOAT32 EncodeClusterToWords(
            BuildCluster& c,
            double        baseLatticeStep,
            BOOL          hasSkinning)
        {
            const UINT32 vertCount = static_cast<UINT32>(c.vertices.size());
            const UINT32 triCount  = static_cast<UINT32>(c.indices.size() / 3u);

            const double invStep   = 1.0 / baseLatticeStep;
            const FLOAT32 posScale = static_cast<FLOAT32>(baseLatticeStep);

            std::vector<std::array<INT32, 3>> latticeCoords(vertCount);
            INT32 minI[3] = {  std::numeric_limits<INT32>::max(),  std::numeric_limits<INT32>::max(),  std::numeric_limits<INT32>::max() };
            INT32 maxI[3] = { -std::numeric_limits<INT32>::max(), -std::numeric_limits<INT32>::max(), -std::numeric_limits<INT32>::max() };

            Vec2 uv0Min = c.vertices[0].texCoord0;
            Vec2 uv0Max = uv0Min;
            Vec2 uv1Min = c.vertices[0].texCoord1;
            Vec2 uv1Max = uv1Min;

            for (UINT32 vi = 0u; vi < vertCount; ++vi)
            {
                const Vertex& v = c.vertices[vi];
                const INT32 ix = static_cast<INT32>(std::llround(static_cast<double>(v.position.x) * invStep));
                const INT32 iy = static_cast<INT32>(std::llround(static_cast<double>(v.position.y) * invStep));
                const INT32 iz = static_cast<INT32>(std::llround(static_cast<double>(v.position.z) * invStep));
                latticeCoords[vi] = { ix, iy, iz };

                if (ix < minI[0]) minI[0] = ix;
                if (iy < minI[1]) minI[1] = iy;
                if (iz < minI[2]) minI[2] = iz;
                if (ix > maxI[0]) maxI[0] = ix;
                if (iy > maxI[1]) maxI[1] = iy;
                if (iz > maxI[2]) maxI[2] = iz;

                if (v.texCoord0.x < uv0Min.x) uv0Min.x = v.texCoord0.x;
                if (v.texCoord0.y < uv0Min.y) uv0Min.y = v.texCoord0.y;
                if (v.texCoord0.x > uv0Max.x) uv0Max.x = v.texCoord0.x;
                if (v.texCoord0.y > uv0Max.y) uv0Max.y = v.texCoord0.y;

                if (v.texCoord1.x < uv1Min.x) uv1Min.x = v.texCoord1.x;
                if (v.texCoord1.y < uv1Min.y) uv1Min.y = v.texCoord1.y;
                if (v.texCoord1.x > uv1Max.x) uv1Max.x = v.texCoord1.x;
                if (v.texCoord1.y > uv1Max.y) uv1Max.y = v.texCoord1.y;
            }

            const UINT64 spanX = static_cast<UINT64>(static_cast<INT64>(maxI[0]) - static_cast<INT64>(minI[0]));
            const UINT64 spanY = static_cast<UINT64>(static_cast<INT64>(maxI[1]) - static_cast<INT64>(minI[1]));
            const UINT64 spanZ = static_cast<UINT64>(static_cast<INT64>(maxI[2]) - static_cast<INT64>(minI[2]));
            const UINT64 maxSpanI = (std::max)(spanX, (std::max)(spanY, spanZ));

            UINT32 lodShift = 0u;
            while ((maxSpanI >> lodShift) > 65535ull && lodShift < 16u)
            {
                ++lodShift;
            }

            Vec2 uv0Ext = uv0Max - uv0Min;
            if (uv0Ext.x < 1e-6f) uv0Ext.x = 1e-6f;
            if (uv0Ext.y < 1e-6f) uv0Ext.y = 1e-6f;

            Vec2 uv1Ext = uv1Max - uv1Min;
            if (uv1Ext.x < 1e-6f) uv1Ext.x = 1e-6f;
            if (uv1Ext.y < 1e-6f) uv1Ext.y = 1e-6f;

            const UINT32 vertStrideWords = hasSkinning ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;
            const UINT32 vertRelOffset   = kNVGClusterHeaderWords; // 16 words (fits in 8 bits)
            const UINT32 idxRelOffset    = vertRelOffset + vertCount * vertStrideWords;
            const UINT32 totalWords      = idxRelOffset + triCount;

            c.encodedWords.assign(totalWords, 0u);
            UINT32* w = c.encodedWords.data();

            // NVG2 In-Page Cluster Header (16 UINT32 words = 64 bytes)
            w[0]  = static_cast<UINT32>(minI[0]);
            w[1]  = static_cast<UINT32>(minI[1]);
            w[2]  = static_cast<UINT32>(minI[2]);
            w[3]  = BitCastFloatToUInt(posScale);
            w[4]  = lodShift;
            w[5]  = 0u;
            w[6]  = BitCastFloatToUInt(uv0Min.x);
            w[7]  = BitCastFloatToUInt(uv0Min.y);
            w[8]  = BitCastFloatToUInt(uv0Ext.x);
            w[9]  = BitCastFloatToUInt(uv0Ext.y);
            w[10] = BitCastFloatToUInt(uv1Min.x);
            w[11] = BitCastFloatToUInt(uv1Min.y);
            w[12] = BitCastFloatToUInt(uv1Ext.x);
            w[13] = BitCastFloatToUInt(uv1Ext.y);

            const UINT32 attrFlags =
                NVG_CLUSTER_ATTR_INT_LATTICE |
                NVG_CLUSTER_ATTR_TANGENT |
                NVG_CLUSTER_ATTR_UV1 |
                NVG_CLUSTER_ATTR_COLOR |
                (hasSkinning ? NVG_CLUSTER_ATTR_SKINNING : 0u);

            w[14] = PackNVGClusterCountsAndLod(vertCount, triCount, c.lodLevel, attrFlags);
            w[15] = (vertRelOffset & 0xFFu) |
                    ((static_cast<UINT32>(c.materialSlot) & 0xFFu) << 8u) |
                    ((idxRelOffset & 0xFFFFu) << 16u);

            FLOAT32 maxPosErr = 0.0f;
            Vec3 qMin(1.0e30f, 1.0e30f, 1.0e30f);
            Vec3 qMax(-1.0e30f, -1.0e30f, -1.0e30f);
            std::vector<Vec3> decodedPositions(vertCount);

            const UINT64 roundHalf = (lodShift > 0u) ? (1ull << (lodShift - 1u)) : 0ull;

            for (UINT32 vi = 0u; vi < vertCount; ++vi)
            {
                const Vertex& v = c.vertices[vi];
                const UINT64 dx = static_cast<UINT64>(static_cast<INT64>(latticeCoords[vi][0]) - static_cast<INT64>(minI[0]));
                const UINT64 dy = static_cast<UINT64>(static_cast<INT64>(latticeCoords[vi][1]) - static_cast<INT64>(minI[1]));
                const UINT64 dz = static_cast<UINT64>(static_cast<INT64>(latticeCoords[vi][2]) - static_cast<INT64>(minI[2]));

                const UINT32 qx = static_cast<UINT32>((std::min<UINT64>)((dx + roundHalf) >> lodShift, 65535ull)) & 0xFFFFu;
                const UINT32 qy = static_cast<UINT32>((std::min<UINT64>)((dy + roundHalf) >> lodShift, 65535ull)) & 0xFFFFu;
                const UINT32 qz = static_cast<UINT32>((std::min<UINT64>)((dz + roundHalf) >> lodShift, 65535ull)) & 0xFFFFu;

                const INT32 recIx = static_cast<INT32>(static_cast<UINT32>(minI[0]) + (qx << lodShift));
                const INT32 recIy = static_cast<INT32>(static_cast<UINT32>(minI[1]) + (qy << lodShift));
                const INT32 recIz = static_cast<INT32>(static_cast<UINT32>(minI[2]) + (qz << lodShift));

                const Vec3 decPos(
                    static_cast<FLOAT32>(recIx) * posScale,
                    static_cast<FLOAT32>(recIy) * posScale,
                    static_cast<FLOAT32>(recIz) * posScale);
                decodedPositions[vi] = decPos;

                if (decPos.x < qMin.x) qMin.x = decPos.x;
                if (decPos.y < qMin.y) qMin.y = decPos.y;
                if (decPos.z < qMin.z) qMin.z = decPos.z;
                if (decPos.x > qMax.x) qMax.x = decPos.x;
                if (decPos.y > qMax.y) qMax.y = decPos.y;
                if (decPos.z > qMax.z) qMax.z = decPos.z;

                const FLOAT32 err = (decPos - v.position).Length();
                if (err > maxPosErr) maxPosErr = err;

                const Vec3 unitN = (v.normal.LengthSq() > 1e-8f) ? v.normal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
                const Vec3 unitT = (v.tangent.LengthSq() > 1e-8f) ? v.tangent.Normalized() : Vec3(1.0f, 0.0f, 0.0f);
                const Vec3 unitB = (v.bitangent.LengthSq() > 1e-8f) ? v.bitangent.Normalized() : unitN.Cross(unitT);

                const UINT32 octN    = NVGBuilder::PackOctahedralNormal(unitN);
                const Vec3   decN    = NVGBuilder::UnpackOctahedralNormal(octN);
                const UINT16 tanPack = NVGBuilder::PackTangentFrame(decN, unitT, unitB);

                const UINT32 qu0 = static_cast<UINT32>(std::lround(ClampF((v.texCoord0.x - uv0Min.x) / uv0Ext.x, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                const UINT32 qv0 = static_cast<UINT32>(std::lround(ClampF((v.texCoord0.y - uv0Min.y) / uv0Ext.y, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                const UINT32 qu1 = static_cast<UINT32>(std::lround(ClampF((v.texCoord1.x - uv1Min.x) / uv1Ext.x, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                const UINT32 qv1 = static_cast<UINT32>(std::lround(ClampF((v.texCoord1.y - uv1Min.y) / uv1Ext.y, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;

                const UINT32 colRG = static_cast<UINT32>(NVGBuilder::FloatToHalf(v.color.x)) |
                                     (static_cast<UINT32>(NVGBuilder::FloatToHalf(v.color.y)) << 16u);
                const UINT32 colBA = static_cast<UINT32>(NVGBuilder::FloatToHalf(v.color.z)) |
                                     (static_cast<UINT32>(NVGBuilder::FloatToHalf(v.color.w)) << 16u);

                UINT32* dst = w + vertRelOffset + vi * vertStrideWords;
                dst[0] = qx | (qy << 16u);
                dst[1] = qz | (static_cast<UINT32>(tanPack) << 16u);
                dst[2] = octN;
                dst[3] = qu0 | (qv0 << 16u);
                dst[4] = qu1 | (qv1 << 16u);
                dst[5] = colRG;
                dst[6] = colBA;

                if (hasSkinning)
                {
                    const UINT32 bi0 = (std::min)(v.boneIndices[0], 65535u);
                    const UINT32 bi1 = (std::min)(v.boneIndices[1], 65535u);
                    const UINT32 bi2 = (std::min)(v.boneIndices[2], 65535u);
                    const UINT32 bi3 = (std::min)(v.boneIndices[3], 65535u);
                    const UINT32 bw0 = static_cast<UINT32>(std::lround(ClampF(v.boneWeights.x, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                    const UINT32 bw1 = static_cast<UINT32>(std::lround(ClampF(v.boneWeights.y, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                    const UINT32 bw2 = static_cast<UINT32>(std::lround(ClampF(v.boneWeights.z, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;
                    const UINT32 bw3 = static_cast<UINT32>(std::lround(ClampF(v.boneWeights.w, 0.0f, 1.0f) * 65535.0f)) & 0xFFFFu;

                    dst[7]  = bi0 | (bi1 << 16u);
                    dst[8]  = bi2 | (bi3 << 16u);
                    dst[9]  = bw0 | (bw1 << 16u);
                    dst[10] = bw2 | (bw3 << 16u);
                }
            }

            for (UINT32 ti = 0u; ti < triCount; ++ti)
            {
                const UINT32 i0 = c.indices[ti * 3u + 0u] & 0xFFu;
                const UINT32 i1 = c.indices[ti * 3u + 1u] & 0xFFu;
                const UINT32 i2 = c.indices[ti * 3u + 2u] & 0xFFu;
                w[idxRelOffset + ti] = i0 | (i1 << 8u) | (i2 << 16u);
            }

            // Update cluster AABB and bounding sphere from actual post-quantization decoded positions (NVG-012)
            c.aabb = AABB(qMin, qMax);
            const Vec3 qCenter = c.aabb.Center();
            FLOAT32 qMaxR2 = 0.0f;
            for (const Vec3& dp : decodedPositions)
            {
                const FLOAT32 d2 = (dp - qCenter).LengthSq();
                if (d2 > qMaxR2) qMaxR2 = d2;
            }
            c.sphere.center = qCenter;
            c.sphere.radius = std::sqrt(qMaxR2);

            return maxPosErr;
        }

        inline FLOAT32 HalfBoxArea(const Vec3& mn, const Vec3& mx) noexcept
        {
            const FLOAT32 dx = (std::max)(0.0f, mx.x - mn.x);
            const FLOAT32 dy = (std::max)(0.0f, mx.y - mn.y);
            const FLOAT32 dz = (std::max)(0.0f, mx.z - mn.z);
            return dx * dy + dy * dz + dz * dx;
        }

        // Recursively builds internal BVH4 nodes (<= 4 children) over a contiguous cluster span [clBegin, clEnd)
        static void BuildClusterSpanBVH4(
            UINT32                                  dstNodeIdx,
            UINT32                                  clBegin,
            UINT32                                  clEnd,
            UINT32                                  depth,
            const std::vector<NVGClusterMetaGPUData>& orderedClusters,
            std::vector<NVGHierarchyNodeGPUData>&   outNodes)
        {
            const UINT32 count = clEnd - clBegin;
            if (count <= 8u || depth >= kNVGMaxBVHDepth - 1u)
            {
                NVGHierarchyNodeGPUData leaf{};
                leaf.boundsMin      = orderedClusters[clBegin].aabbMin;
                leaf.boundsMax      = orderedClusters[clBegin].aabbMax;
                leaf.childStart     = clBegin;
                leaf.maxParentError = 0.0f;
                leaf.minSelfError   = 1.0e30f;
                leaf.meshId         = orderedClusters[clBegin].meshId;

                Vec4Sphere s{};
                UINT32 minLod         = 255u;
                UINT32 maxLod         = 0u;
                UINT32 maxParentThinQ = 0u;
                for (UINT32 i = clBegin; i < clEnd; ++i)
                {
                    const NVGClusterMetaGPUData& cm = orderedClusters[i];
                    if (cm.aabbMin.x < leaf.boundsMin.x) leaf.boundsMin.x = cm.aabbMin.x;
                    if (cm.aabbMin.y < leaf.boundsMin.y) leaf.boundsMin.y = cm.aabbMin.y;
                    if (cm.aabbMin.z < leaf.boundsMin.z) leaf.boundsMin.z = cm.aabbMin.z;
                    if (cm.aabbMax.x > leaf.boundsMax.x) leaf.boundsMax.x = cm.aabbMax.x;
                    if (cm.aabbMax.y > leaf.boundsMax.y) leaf.boundsMax.y = cm.aabbMax.y;
                    if (cm.aabbMax.z > leaf.boundsMax.z) leaf.boundsMax.z = cm.aabbMax.z;

                    Vec4Sphere cs{ Vec3(cm.parentGroupSphere.x, cm.parentGroupSphere.y, cm.parentGroupSphere.z), cm.parentGroupSphere.w };
                    s = MergeSpheres(s, cs);

                    if (cm.parentGroupError > leaf.maxParentError) leaf.maxParentError = cm.parentGroupError;
                    if (cm.selfGroupError   < leaf.minSelfError)   leaf.minSelfError   = cm.selfGroupError;

                    const UINT32 pThinQ = ((cm.clusterFlags & 2u) != 0u)
                        ? ((cm.clusterFlags >> 16u) & 0x3FFFu)
                        : QuantizeThinMetric14(cm.thinFeatureMetric);
                    if (pThinQ > maxParentThinQ) maxParentThinQ = pThinQ;

                    const UINT32 lod = (cm.countsAndLod >> 16u) & 0xFFu;
                    if (lod < minLod) minLod = lod;
                    if (lod > maxLod) maxLod = lod;
                }
                leaf.lodSphere          = Vec4(s.center.x, s.center.y, s.center.z, s.radius);
                leaf.childCountAndFlags = PackNVGNodeChildCountAndFlags(count, true, minLod, maxLod);
                leaf.reserved           = (1u << 31u) | (maxParentThinQ & 0x3FFFu);
                outNodes[dstNodeIdx]    = leaf;
                return;
            }

            const UINT32 fanout        = (count <= 16u) ? 2u : ((count <= 24u) ? 3u : 4u);
            const UINT32 firstChildIdx = static_cast<UINT32>(outNodes.size());
            for (UINT32 c = 0u; c < fanout; ++c)
            {
                outNodes.push_back(NVGHierarchyNodeGPUData{});
            }

            NVGHierarchyNodeGPUData node{};
            node.boundsMin      = Vec3(1.0e30f, 1.0e30f, 1.0e30f);
            node.boundsMax      = Vec3(-1.0e30f, -1.0e30f, -1.0e30f);
            node.childStart     = firstChildIdx;
            node.maxParentError = 0.0f;
            node.minSelfError   = 1.0e30f;
            node.meshId         = orderedClusters[clBegin].meshId;

            Vec4Sphere mergedS{};
            UINT32 minLod         = 255u;
            UINT32 maxLod         = 0u;
            UINT32 maxParentThinQ = 0u;

            for (UINT32 c = 0u; c < fanout; ++c)
            {
                const UINT32 subB = clBegin + (count * c) / fanout;
                const UINT32 subE = clBegin + (count * (c + 1u)) / fanout;
                BuildClusterSpanBVH4(firstChildIdx + c, subB, subE, depth + 1u, orderedClusters, outNodes);
                const NVGHierarchyNodeGPUData& ch = outNodes[firstChildIdx + c];

                if (ch.boundsMin.x < node.boundsMin.x) node.boundsMin.x = ch.boundsMin.x;
                if (ch.boundsMin.y < node.boundsMin.y) node.boundsMin.y = ch.boundsMin.y;
                if (ch.boundsMin.z < node.boundsMin.z) node.boundsMin.z = ch.boundsMin.z;
                if (ch.boundsMax.x > node.boundsMax.x) node.boundsMax.x = ch.boundsMax.x;
                if (ch.boundsMax.y > node.boundsMax.y) node.boundsMax.y = ch.boundsMax.y;
                if (ch.boundsMax.z > node.boundsMax.z) node.boundsMax.z = ch.boundsMax.z;

                Vec4Sphere cs{ Vec3(ch.lodSphere.x, ch.lodSphere.y, ch.lodSphere.z), ch.lodSphere.w };
                mergedS = MergeSpheres(mergedS, cs);
                if (ch.maxParentError > node.maxParentError) node.maxParentError = ch.maxParentError;
                if (ch.minSelfError   < node.minSelfError)   node.minSelfError   = ch.minSelfError;
                if ((ch.reserved & (1u << 31u)) != 0u)
                {
                    const UINT32 cq = ch.reserved & 0x3FFFu;
                    if (cq > maxParentThinQ) maxParentThinQ = cq;
                }

                const UINT32 cMinLod = (ch.childCountAndFlags >> 17u) & 0x7Fu;
                const UINT32 cMaxLod = (ch.childCountAndFlags >> 24u) & 0xFFu;
                if (cMinLod < minLod) minLod = cMinLod;
                if (cMaxLod > maxLod) maxLod = cMaxLod;
            }

            node.lodSphere          = Vec4(mergedS.center.x, mergedS.center.y, mergedS.center.z, mergedS.radius);
            node.childCountAndFlags = PackNVGNodeChildCountAndFlags(fanout, false, minLod, maxLod);
            node.reserved           = (1u << 31u) | (maxParentThinQ & 0x3FFFu);
            outNodes[dstNodeIdx]    = node;
        }

        // 12-Bin Surface Area Heuristic (SAH) binary split over groupOrder[rangeBegin .. rangeEnd)
        static UINT32 PartitionGroupsSAHBinary(
            std::vector<UINT32>&               groupOrder,
            UINT32                             rangeBegin,
            UINT32                             rangeEnd,
            const std::vector<AABB>&           groupAABBs,
            const std::vector<NVGClusterGroupGPUData>& groups)
        {
            const UINT32 count = rangeEnd - rangeBegin;
            if (count <= 2u) return rangeBegin + 1u;

            Vec3 cMin(1.0e30f, 1.0e30f, 1.0e30f);
            Vec3 cMax(-1.0e30f, -1.0e30f, -1.0e30f);
            for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
            {
                const Vec3 c = groupAABBs[groupOrder[i]].Center();
                if (c.x < cMin.x) cMin.x = c.x;
                if (c.y < cMin.y) cMin.y = c.y;
                if (c.z < cMin.z) cMin.z = c.z;
                if (c.x > cMax.x) cMax.x = c.x;
                if (c.y > cMax.y) cMax.y = c.y;
                if (c.z > cMax.z) cMax.z = c.z;
            }

            constexpr UINT32 kNumBins = 12u;
            int     bestAxis  = -1;
            UINT32  bestSplit = 0u;
            FLOAT32 bestCost  = 1.0e30f;

            for (int axis = 0; axis < 3; ++axis)
            {
                const FLOAT32 minC = (axis == 0) ? cMin.x : ((axis == 1) ? cMin.y : cMin.z);
                const FLOAT32 maxC = (axis == 0) ? cMax.x : ((axis == 1) ? cMax.y : cMax.z);
                const FLOAT32 span = maxC - minC;
                if (span <= 1e-6f) continue;

                struct SAHBin
                {
                    Vec3   bMin = Vec3(1.0e30f, 1.0e30f, 1.0e30f);
                    Vec3   bMax = Vec3(-1.0e30f, -1.0e30f, -1.0e30f);
                    UINT32 weight = 0u;
                };
                std::array<SAHBin, kNumBins> bins{};
                const FLOAT32 scale = static_cast<FLOAT32>(kNumBins) / span;

                for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
                {
                    const UINT32 gIdx = groupOrder[i];
                    const AABB&  gb   = groupAABBs[gIdx];
                    const Vec3   cen  = gb.Center();
                    const FLOAT32 coord = (axis == 0) ? cen.x : ((axis == 1) ? cen.y : cen.z);
                    UINT32 binIdx = static_cast<UINT32>((std::max)(0.0f, (coord - minC) * scale));
                    if (binIdx >= kNumBins) binIdx = kNumBins - 1u;

                    SAHBin& b = bins[binIdx];
                    if (gb.m_Min.x < b.bMin.x) b.bMin.x = gb.m_Min.x;
                    if (gb.m_Min.y < b.bMin.y) b.bMin.y = gb.m_Min.y;
                    if (gb.m_Min.z < b.bMin.z) b.bMin.z = gb.m_Min.z;
                    if (gb.m_Max.x > b.bMax.x) b.bMax.x = gb.m_Max.x;
                    if (gb.m_Max.y > b.bMax.y) b.bMax.y = gb.m_Max.y;
                    if (gb.m_Max.z > b.bMax.z) b.bMax.z = gb.m_Max.z;
                    b.weight += (std::max)(1u, groups[gIdx].clusterCount);
                }

                std::array<FLOAT32, kNumBins - 1u> leftArea{};
                std::array<UINT32,  kNumBins - 1u> leftWeight{};
                Vec3   curMin(1.0e30f, 1.0e30f, 1.0e30f);
                Vec3   curMax(-1.0e30f, -1.0e30f, -1.0e30f);
                UINT32 curW = 0u;
                for (UINT32 b = 0u; b < kNumBins - 1u; ++b)
                {
                    if (bins[b].weight > 0u)
                    {
                        if (bins[b].bMin.x < curMin.x) curMin.x = bins[b].bMin.x;
                        if (bins[b].bMin.y < curMin.y) curMin.y = bins[b].bMin.y;
                        if (bins[b].bMin.z < curMin.z) curMin.z = bins[b].bMin.z;
                        if (bins[b].bMax.x > curMax.x) curMax.x = bins[b].bMax.x;
                        if (bins[b].bMax.y > curMax.y) curMax.y = bins[b].bMax.y;
                        if (bins[b].bMax.z > curMax.z) curMax.z = bins[b].bMax.z;
                        curW += bins[b].weight;
                    }
                    leftWeight[b] = curW;
                    leftArea[b]   = (curW > 0u) ? HalfBoxArea(curMin, curMax) : 0.0f;
                }

                curMin = Vec3(1.0e30f, 1.0e30f, 1.0e30f);
                curMax = Vec3(-1.0e30f, -1.0e30f, -1.0e30f);
                curW   = 0u;
                for (INT32 b = static_cast<INT32>(kNumBins) - 1; b >= 1; --b)
                {
                    const UINT32 ub = static_cast<UINT32>(b);
                    if (bins[ub].weight > 0u)
                    {
                        if (bins[ub].bMin.x < curMin.x) curMin.x = bins[ub].bMin.x;
                        if (bins[ub].bMin.y < curMin.y) curMin.y = bins[ub].bMin.y;
                        if (bins[ub].bMin.z < curMin.z) curMin.z = bins[ub].bMin.z;
                        if (bins[ub].bMax.x > curMax.x) curMax.x = bins[ub].bMax.x;
                        if (bins[ub].bMax.y > curMax.y) curMax.y = bins[ub].bMax.y;
                        if (bins[ub].bMax.z > curMax.z) curMax.z = bins[ub].bMax.z;
                        curW += bins[ub].weight;
                    }
                    const UINT32 lw = leftWeight[ub - 1u];
                    if (lw > 0u && curW > 0u)
                    {
                        const FLOAT32 cost = leftArea[ub - 1u] * static_cast<FLOAT32>(lw) +
                                             HalfBoxArea(curMin, curMax) * static_cast<FLOAT32>(curW);
                        if (cost < bestCost)
                        {
                            bestCost  = cost;
                            bestAxis  = axis;
                            bestSplit = ub;
                        }
                    }
                }
            }

            if (bestAxis >= 0)
            {
                const FLOAT32 minC  = (bestAxis == 0) ? cMin.x : ((bestAxis == 1) ? cMin.y : cMin.z);
                const FLOAT32 maxC  = (bestAxis == 0) ? cMax.x : ((bestAxis == 1) ? cMax.y : cMax.z);
                const FLOAT32 scale = static_cast<FLOAT32>(kNumBins) / (maxC - minC);
                auto midIt = std::stable_partition(
                    groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeBegin),
                    groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeEnd),
                    [&](UINT32 gIdx) {
                        const Vec3 cen = groupAABBs[gIdx].Center();
                        const FLOAT32 coord = (bestAxis == 0) ? cen.x : ((bestAxis == 1) ? cen.y : cen.z);
                        UINT32 binIdx = static_cast<UINT32>((std::max)(0.0f, (coord - minC) * scale));
                        if (binIdx >= kNumBins) binIdx = kNumBins - 1u;
                        return binIdx < bestSplit;
                    });
                const UINT32 mid = static_cast<UINT32>(midIt - groupOrder.begin());
                if (mid > rangeBegin && mid < rangeEnd)
                {
                    return mid;
                }
            }

            // Deterministic longest-axis median fallback
            const Vec3 span = cMax - cMin;
            int axis = 0;
            if (span.y > span.x && span.y >= span.z) axis = 1;
            else if (span.z > span.x && span.z >= span.y) axis = 2;

            std::sort(
                groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeBegin),
                groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeEnd),
                [&](UINT32 a, UINT32 b) {
                    const Vec3 ca = groupAABBs[a].Center();
                    const Vec3 cb = groupAABBs[b].Center();
                    const FLOAT32 va = (axis == 0) ? ca.x : ((axis == 1) ? ca.y : ca.z);
                    const FLOAT32 vb = (axis == 0) ? cb.x : ((axis == 1) ? cb.y : cb.z);
                    if (va != vb) return va < vb;
                    return a < b;
                });
            return rangeBegin + count / 2u;
        }

        // LOD-Band + 12-Bin SAH 4-ary BVH4 builder (internal childCount <= 4, leaf childCount <= 8)
        static void BuildGroupBVH4Recursive(
            UINT32                                     nodeIdx,
            std::vector<UINT32>&                       groupOrder,
            UINT32                                     rangeBegin,
            UINT32                                     rangeEnd,
            UINT32                                     depth,
            const std::vector<AABB>&                   groupAABBs,
            const std::vector<NVGClusterGroupGPUData>& groups,
            const std::vector<NVGClusterMetaGPUData>&  rawClusters,
            std::vector<NVGClusterMetaGPUData>&        outOrderedClusters,
            std::vector<UINT32>&                       outNewGroupClusterStarts,
            std::vector<NVGHierarchyNodeGPUData>&      outNodes)
        {
            const UINT32 grpCount = rangeEnd - rangeBegin;
            UINT32 totalClustersInSpan = 0u;
            UINT32 minGrpLod = 255u;
            UINT32 maxGrpLod = 0u;
            for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
            {
                const NVGClusterGroupGPUData& g = groups[groupOrder[i]];
                totalClustersInSpan += g.clusterCount;
                const UINT32 lod = UnpackNVGGroupLodLevel(g.lodLevel);
                if (lod < minGrpLod) minGrpLod = lod;
                if (lod > maxGrpLod) maxGrpLod = lod;
            }

            if (grpCount <= 1u || totalClustersInSpan <= 8u || depth >= kNVGMaxBVHDepth - 1u)
            {
                const UINT32 clSpanStart = static_cast<UINT32>(outOrderedClusters.size());
                for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
                {
                    const UINT32 gIdx = groupOrder[i];
                    const NVGClusterGroupGPUData& g = groups[gIdx];
                    outNewGroupClusterStarts[gIdx] = static_cast<UINT32>(outOrderedClusters.size());
                    for (UINT32 k = 0u; k < g.clusterCount; ++k)
                    {
                        NVGClusterMetaGPUData cm = rawClusters[g.clusterStart + k];
                        cm.clusterLocalIndex     = static_cast<UINT32>(outOrderedClusters.size());
                        outOrderedClusters.push_back(cm);
                    }
                }
                const UINT32 clSpanEnd = static_cast<UINT32>(outOrderedClusters.size());
                BuildClusterSpanBVH4(nodeIdx, clSpanStart, clSpanEnd, depth, outOrderedClusters, outNodes);
                return;
            }

            std::vector<std::pair<UINT32, UINT32>> subRanges;
            subRanges.reserve(4u);

            // Separate distinct LOD bands near the top of the hierarchy so coarse LOD queries prune entire fine LOD subtrees
            if (minGrpLod < maxGrpLod && depth <= 2u && grpCount >= 2u)
            {
                std::sort(
                    groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeBegin),
                    groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeEnd),
                    [&](UINT32 a, UINT32 b) {
                        const UINT32 la = UnpackNVGGroupLodLevel(groups[a].lodLevel);
                        const UINT32 lb = UnpackNVGGroupLodLevel(groups[b].lodLevel);
                        if (la != lb) return la > lb; // Coarser LOD bands first
                        return a < b;
                    });

                const UINT32 lodSpan  = maxGrpLod - minGrpLod + 1u;
                const UINT32 numBands = (std::min)(4u, lodSpan);
                UINT32 curPos = rangeBegin;
                for (UINT32 b = 0u; b < numBands && curPos < rangeEnd; ++b)
                {
                    if (b + 1u == numBands)
                    {
                        subRanges.push_back({ curPos, rangeEnd });
                        curPos = rangeEnd;
                        break;
                    }
                    const UINT32 bandMinLod = maxGrpLod - ((lodSpan * (b + 1u)) / numBands) + 1u;
                    UINT32 nextPos = curPos;
                    while (nextPos < rangeEnd && UnpackNVGGroupLodLevel(groups[groupOrder[nextPos]].lodLevel) >= bandMinLod)
                    {
                        ++nextPos;
                    }
                    if (nextPos > curPos && nextPos < rangeEnd)
                    {
                        subRanges.push_back({ curPos, nextPos });
                        curPos = nextPos;
                    }
                }
                if (curPos < rangeEnd)
                {
                    subRanges.push_back({ curPos, rangeEnd });
                }
                if (subRanges.size() == 1u)
                {
                    subRanges.clear();
                }
            }

            // Within a homogeneous LOD band (or if LOD-band split collapsed), apply 2-level 12-bin SAH -> up to 4 children
            if (subRanges.empty())
            {
                const UINT32 mid = PartitionGroupsSAHBinary(groupOrder, rangeBegin, rangeEnd, groupAABBs, groups);
                if (grpCount >= 4u)
                {
                    if (mid - rangeBegin >= 2u)
                    {
                        const UINT32 m0 = PartitionGroupsSAHBinary(groupOrder, rangeBegin, mid, groupAABBs, groups);
                        subRanges.push_back({ rangeBegin, m0 });
                        subRanges.push_back({ m0, mid });
                    }
                    else
                    {
                        subRanges.push_back({ rangeBegin, mid });
                    }

                    if (rangeEnd - mid >= 2u)
                    {
                        const UINT32 m1 = PartitionGroupsSAHBinary(groupOrder, mid, rangeEnd, groupAABBs, groups);
                        subRanges.push_back({ mid, m1 });
                        subRanges.push_back({ m1, rangeEnd });
                    }
                    else
                    {
                        subRanges.push_back({ mid, rangeEnd });
                    }
                }
                else
                {
                    subRanges.push_back({ rangeBegin, mid });
                    subRanges.push_back({ mid, rangeEnd });
                }
            }

            const UINT32 firstChildIdx = static_cast<UINT32>(outNodes.size());
            const UINT32 numChildren   = static_cast<UINT32>(subRanges.size());
            for (UINT32 c = 0u; c < numChildren; ++c)
            {
                outNodes.push_back(NVGHierarchyNodeGPUData{});
            }

            NVGHierarchyNodeGPUData node{};
            node.boundsMin      = Vec3(1.0e30f, 1.0e30f, 1.0e30f);
            node.boundsMax      = Vec3(-1.0e30f, -1.0e30f, -1.0e30f);
            node.childStart     = firstChildIdx;
            node.maxParentError = 0.0f;
            node.minSelfError   = 1.0e30f;
            node.meshId         = groups[groupOrder[rangeBegin]].meshId;

            Vec4Sphere mergedS{};
            UINT32 minLod         = 255u;
            UINT32 maxLod         = 0u;
            UINT32 maxParentThinQ = 0u;

            for (UINT32 c = 0u; c < numChildren; ++c)
            {
                BuildGroupBVH4Recursive(
                    firstChildIdx + c, groupOrder, subRanges[c].first, subRanges[c].second,
                    depth + 1u, groupAABBs, groups, rawClusters, outOrderedClusters, outNewGroupClusterStarts, outNodes);
                const NVGHierarchyNodeGPUData& ch = outNodes[firstChildIdx + c];

                if (ch.boundsMin.x < node.boundsMin.x) node.boundsMin.x = ch.boundsMin.x;
                if (ch.boundsMin.y < node.boundsMin.y) node.boundsMin.y = ch.boundsMin.y;
                if (ch.boundsMin.z < node.boundsMin.z) node.boundsMin.z = ch.boundsMin.z;
                if (ch.boundsMax.x > node.boundsMax.x) node.boundsMax.x = ch.boundsMax.x;
                if (ch.boundsMax.y > node.boundsMax.y) node.boundsMax.y = ch.boundsMax.y;
                if (ch.boundsMax.z > node.boundsMax.z) node.boundsMax.z = ch.boundsMax.z;

                Vec4Sphere cs{ Vec3(ch.lodSphere.x, ch.lodSphere.y, ch.lodSphere.z), ch.lodSphere.w };
                mergedS = MergeSpheres(mergedS, cs);
                if (ch.maxParentError > node.maxParentError) node.maxParentError = ch.maxParentError;
                if (ch.minSelfError   < node.minSelfError)   node.minSelfError   = ch.minSelfError;
                if ((ch.reserved & (1u << 31u)) != 0u)
                {
                    const UINT32 cq = ch.reserved & 0x3FFFu;
                    if (cq > maxParentThinQ) maxParentThinQ = cq;
                }

                const UINT32 cMinLod = (ch.childCountAndFlags >> 17u) & 0x7Fu;
                const UINT32 cMaxLod = (ch.childCountAndFlags >> 24u) & 0xFFu;
                if (cMinLod < minLod) minLod = cMinLod;
                if (cMaxLod > maxLod) maxLod = cMaxLod;
            }

            node.lodSphere          = Vec4(mergedS.center.x, mergedS.center.y, mergedS.center.z, mergedS.radius);
            node.childCountAndFlags = PackNVGNodeChildCountAndFlags(numChildren, false, minLod, maxLod);
            node.reserved           = (1u << 31u) | (maxParentThinQ & 0x3FFFu);
            outNodes[nodeIdx]       = node;
        }

        // Legacy NVG1 on-disk structures preserved strictly for backward-compatible Deserialize(NVG1)
        struct NVG1BinaryHeader
        {
            UINT32  magic             = kNVG1BinaryMagic;
            UINT32  version           = kNVGBinaryVersion;
            UINT32  flags             = 0u;
            UINT32  lodLevelCount     = 0u;
            UINT32  rootPageCount     = 0u;
            UINT32  rootNodeIndex     = 0u;
            UINT32  clusterCount      = 0u;
            UINT32  groupCount        = 0u;
            UINT32  childGroupIdCount = 0u;
            UINT32  bvhNodeCount      = 0u;
            UINT32  pageCount         = 0u;
            UINT32  pageSizeBytes     = kNVGPageSizeBytes;
            Vec3    boundsMin         = Vec3::Zero();
            FLOAT32 maxQuantError     = 0.0f;
            Vec3    boundsMax         = Vec3::Zero();
            FLOAT32 compressionRatio  = 1.0f;
            UINT64  payloadChecksum   = 0ull;
            UINT64  payloadByteSize   = 0ull;
        };

        struct NVG1BinaryPageMeta
        {
            UINT32 localPageIndex = 0u;
            UINT32 lodLevel       = 0u;
            UINT32 isRootPage     = 0u;
            UINT32 clusterCount   = 0u;
            UINT32 usedWords      = 0u;
            UINT32 reserved       = 0u;
        };

        // Full deterministic structural & topological verifier for deserialized NVG1/NVG2 assets (NVG-008, NVG-009)
        static BOOL ValidateDeserializedTablesAndBVH(
            const NVGVirtualMeshAsset& asset,
            UINT32                     expectedPageCount,
            const std::vector<UINT32>& pageUsedWords,
            BOOL                       verifyPageClusterPayloads,
            std::string&               outError)
        {
            const UINT32 clusterCount = static_cast<UINT32>(asset.clusters.size());
            const UINT32 groupCount   = static_cast<UINT32>(asset.clusterGroups.size());
            const UINT32 bvhCount     = static_cast<UINT32>(asset.bvhNodes.size());
            const UINT32 childIdCount = static_cast<UINT32>(asset.childGroupIndices.size());

            if (!IsFiniteVec3(asset.localBounds.m_Min) || !IsFiniteVec3(asset.localBounds.m_Max) ||
                asset.localBounds.m_Min.x > asset.localBounds.m_Max.x ||
                asset.localBounds.m_Min.y > asset.localBounds.m_Max.y ||
                asset.localBounds.m_Min.z > asset.localBounds.m_Max.z)
            {
                outError = "NVG binary contains NaN/Inf or inverted local AABB bounds.";
                return false;
            }

            // 1. Validate Cluster metadata & optional in-page payload bounds
            for (UINT32 ci = 0u; ci < clusterCount; ++ci)
            {
                const NVGClusterMetaGPUData& cm = asset.clusters[ci];
                if (cm.virtualPageId >= expectedPageCount ||
                    cm.clusterWordOffsetInPage + kNVGClusterHeaderWords > kNVGPageWordCount ||
                    cm.clusterWordOffsetInPage + kNVGClusterHeaderWords > pageUsedWords[cm.virtualPageId] ||
                    cm.selfGroupIndex >= groupCount ||
                    (cm.parentGroupIndex != kNVGInvalidIndex && cm.parentGroupIndex >= groupCount))
                {
                    outError = "NVG binary contains out-of-bounds cluster page, offset, or group index.";
                    return false;
                }

                const UINT32 childGrpCnt = UnpackNVGRefinedChildGroupCount(cm.refinedByChildGroupCount);
                if (childGrpCnt > 0u)
                {
                    const BOOL validDirect = (childGrpCnt == 1u && cm.refinedByChildGroupStart < groupCount);
                    const BOOL validList   = (static_cast<UINT64>(cm.refinedByChildGroupStart) + static_cast<UINT64>(childGrpCnt) <= childIdCount);
                    if (!validDirect && !validList)
                    {
                        outError = "NVG binary cluster references out-of-bounds refined child group index.";
                        return false;
                    }
                }

                if (verifyPageClusterPayloads && cm.virtualPageId < asset.pages.size())
                {
                    const NVGVirtualPagePayload& pg = asset.pages[cm.virtualPageId];
                    if (pg.words.size() < pg.usedWords)
                    {
                        outError = "NVG binary page word buffer smaller than usedWords.";
                        return false;
                    }
                    const UINT32* h            = pg.words.data() + cm.clusterWordOffsetInPage;
                    const UINT32  countsAndLod = h[14];
                    const UINT32  vCount       = UnpackNVGClusterVertexCount(countsAndLod);
                    const UINT32  tCount       = UnpackNVGClusterTriangleCount(countsAndLod);
                    const UINT32  attrFlags    = (countsAndLod >> 24u) & 0xFFu;
                    const BOOL    skinned      = ((attrFlags & NVG_CLUSTER_ATTR_SKINNING) != 0u);
                    const UINT32  vStride      = skinned ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;
                    const UINT32  vRelOff      = ((attrFlags & NVG_CLUSTER_ATTR_INT_LATTICE) != 0u)
                        ? (h[15] & 0xFFu)
                        : (h[15] & 0xFFFFu);
                    const UINT32  iRelOff      = (h[15] >> 16u) & 0xFFFFu;

                    if (vCount == 0u || vCount > kNVGMaxClusterVertices ||
                        tCount == 0u || tCount > kNVGMaxClusterTriangles ||
                        static_cast<UINT64>(cm.clusterWordOffsetInPage) + vRelOff + static_cast<UINT64>(vCount) * vStride > pg.usedWords ||
                        static_cast<UINT64>(cm.clusterWordOffsetInPage) + iRelOff + tCount > pg.usedWords)
                    {
                        outError = "NVG binary in-page cluster header vertex/index offsets exceed page bounds.";
                        return false;
                    }

                    for (UINT32 ti = 0u; ti < tCount; ++ti)
                    {
                        const UINT32 pk = h[iRelOff + ti];
                        if ((pk & 0xFFu) >= vCount || ((pk >> 8u) & 0xFFu) >= vCount || ((pk >> 16u) & 0xFFu) >= vCount)
                        {
                            outError = "NVG binary cluster contains out-of-bounds local vertex index.";
                            return false;
                        }
                    }
                }
            }

            // 2. Validate ClusterGroups & DAG acyclicity
            for (UINT32 cid : asset.childGroupIndices)
            {
                if (cid >= groupCount)
                {
                    outError = "NVG binary childGroupIndices table contains out-of-bounds group ID.";
                    return false;
                }
            }

            for (UINT32 gi = 0u; gi < groupCount; ++gi)
            {
                const NVGClusterGroupGPUData& grp = asset.clusterGroups[gi];
                const UINT32 grpPageCnt = UnpackNVGGroupPageCount(grp.lodLevel);
                if (static_cast<UINT64>(grp.clusterStart) + static_cast<UINT64>(grp.clusterCount) > clusterCount ||
                    (grp.parentGroupIndex != kNVGInvalidIndex && grp.parentGroupIndex >= groupCount) ||
                    static_cast<UINT64>(grp.childGroupStart) + static_cast<UINT64>(grp.childGroupCount) > childIdCount ||
                    (grp.firstVirtualPageId != kNVGInvalidIndex &&
                     static_cast<UINT64>(grp.firstVirtualPageId) + static_cast<UINT64>(grpPageCnt) > expectedPageCount))
                {
                    outError = "NVG binary contains out-of-bounds cluster group range or page index.";
                    return false;
                }

                // Walk parentGroupIndex chain to reject group cycles
                UINT32 walk  = grp.parentGroupIndex;
                UINT32 steps = 0u;
                while (walk != kNVGInvalidIndex)
                {
                    if (walk >= groupCount || ++steps > kNVGMaxLODLevels + 2u)
                    {
                        outError = "NVG binary cluster group hierarchy contains a cycle or invalid depth.";
                        return false;
                    }
                    walk = asset.clusterGroups[walk].parentGroupIndex;
                }
            }

            // 3. Validate all BVH nodes & run iterative DFS from rootNodeIndex to reject cycles or excessive depth (NVG-009)
            for (UINT32 ni = 0u; ni < bvhCount; ++ni)
            {
                const NVGHierarchyNodeGPUData& node = asset.bvhNodes[ni];
                const UINT32 childCnt = node.childCountAndFlags & 0xFFFFu;
                const BOOL   isLeaf   = ((node.childCountAndFlags & 65536u) != 0u);
                const UINT64 childEnd = static_cast<UINT64>(node.childStart) + static_cast<UINT64>(childCnt);
                if (childCnt == 0u || (isLeaf ? (childEnd > clusterCount) : (childEnd > bvhCount)))
                {
                    outError = "NVG binary BVH node references empty or out-of-bounds child range.";
                    return false;
                }
            }

            // 0 = unvisited, 1 = active on recursion stack, 2 = completed
            std::vector<UINT8> nodeState(bvhCount, 0u);
            struct DFSFrame
            {
                UINT32 nodeIdx   = 0u;
                UINT32 nextChild = 0u;
                UINT32 depth     = 1u;
            };
            std::vector<DFSFrame> stack;
            stack.reserve(kNVGMaxBVHDepth + 4u);
            stack.push_back({ asset.rootNodeIndex, 0u, 1u });
            nodeState[asset.rootNodeIndex] = 1u;

            while (!stack.empty())
            {
                DFSFrame& top = stack.back();
                if (top.depth > kNVGMaxBVHDepth + 4u)
                {
                    outError = "NVG binary BVH exceeds maximum safe traversal depth.";
                    return false;
                }

                const NVGHierarchyNodeGPUData& nd = asset.bvhNodes[top.nodeIdx];
                const BOOL   isLeaf   = ((nd.childCountAndFlags & 65536u) != 0u);
                const UINT32 childCnt = nd.childCountAndFlags & 0xFFFFu;

                if (isLeaf || top.nextChild >= childCnt)
                {
                    nodeState[top.nodeIdx] = 2u;
                    stack.pop_back();
                    continue;
                }

                const UINT32 childIdx = nd.childStart + top.nextChild++;
                if (nodeState[childIdx] == 1u)
                {
                    outError = "NVG binary BVH contains a directed cycle.";
                    return false;
                }
                if (nodeState[childIdx] == 0u)
                {
                    nodeState[childIdx] = 1u;
                    stack.push_back({ childIdx, 0u, top.depth + 1u });
                }
            }

            return true;
        }
    } // anonymous namespace

    // =====================================================================
    //  NVGMemoryPageDataSource
    // =====================================================================

    void NVGMemoryPageDataSource::RegisterAsset(UINT32 virtualMeshId, const std::vector<NVGVirtualPagePayload>* pages)
    {
        for (auto& e : m_entries)
        {
            if (e.virtualMeshId == virtualMeshId)
            {
                e.pages = pages;
                return;
            }
        }
        m_entries.push_back({ virtualMeshId, pages });
    }

    void NVGMemoryPageDataSource::UnregisterAsset(UINT32 virtualMeshId)
    {
        m_entries.erase(
            std::remove_if(m_entries.begin(), m_entries.end(),
                [virtualMeshId](const Entry& e) { return e.virtualMeshId == virtualMeshId; }),
            m_entries.end());
    }

    void NVGMemoryPageDataSource::Clear()
    {
        m_entries.clear();
    }

    BOOL NVGMemoryPageDataSource::ReadVirtualPageData(
        UINT32 virtualMeshId,
        UINT32 localPageIndex,
        BYTE*  outPageBytes,
        UINT32 maxBytes) const
    {
        if (!outPageBytes || maxBytes < kNVGPageSizeBytes)
            return false;

        for (const auto& e : m_entries)
        {
            if (e.virtualMeshId == virtualMeshId && e.pages != nullptr)
            {
                if (localPageIndex >= e.pages->size())
                    return false;
                const NVGVirtualPagePayload& p = (*e.pages)[localPageIndex];
                if (p.words.size() * sizeof(UINT32) != kNVGPageSizeBytes)
                    return false;
                std::memcpy(outPageBytes, p.words.data(), kNVGPageSizeBytes);
                return true;
            }
        }
        return false;
    }

    // =====================================================================
    //  IEEE-754 Float16, Octahedral Normal & Tangent Frame Helpers
    // =====================================================================

    UINT16 NVGBuilder::FloatToHalf(FLOAT32 value) noexcept
    {
        const UINT32 f      = BitCastFloatToUInt(value);
        const UINT32 sign   = (f >> 16u) & 0x8000u;
        const UINT32 exp32  = (f >> 23u) & 0xFFu;
        const UINT32 mant32 = f & 0x007FFFFFu;

        if (exp32 == 0xFFu)
        {
            return static_cast<UINT16>(sign | (mant32 ? 0x7E00u : 0x7C00u));
        }

        const INT32 exp16 = static_cast<INT32>(exp32) - 127 + 15;
        if (exp16 >= 31)
        {
            return static_cast<UINT16>(sign | 0x7C00u);
        }

        if (exp16 <= 0)
        {
            if (exp16 < -10)
            {
                return static_cast<UINT16>(sign);
            }
            const UINT32 fullMant = mant32 | 0x00800000u;
            const UINT32 shift    = static_cast<UINT32>(14 - exp16);
            const UINT32 halfMant = fullMant >> shift;
            const UINT32 rem      = fullMant & ((1u << shift) - 1u);
            const UINT32 halfway  = 1u << (shift - 1u);
            const UINT32 roundUp  = (rem > halfway || (rem == halfway && (halfMant & 1u) != 0u)) ? 1u : 0u;
            // Arithmetic '+' propagates carry into exponent if 0x03FF rounds up to 0x0400
            return static_cast<UINT16>(sign | (halfMant + roundUp));
        }

        const UINT32 halfMant = mant32 >> 13u;
        const UINT32 rem      = mant32 & 0x1FFFu;
        const UINT32 roundUp  = (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1u) != 0u)) ? 1u : 0u;
        // Arithmetic '+' carries mantissa overflow cleanly into exponent (saturating to Inf at exp16 == 30)
        const UINT32 expAndMant = (static_cast<UINT32>(exp16) << 10u) + halfMant + roundUp;
        return static_cast<UINT16>(sign | expAndMant);
    }

    FLOAT32 NVGBuilder::HalfToFloat(UINT16 bits) noexcept
    {
        const UINT32 sign = (static_cast<UINT32>(bits) & 0x8000u) << 16u;
        UINT32       exp  = (static_cast<UINT32>(bits) >> 10u) & 0x1Fu;
        UINT32       mant = static_cast<UINT32>(bits) & 0x03FFu;

        if (exp == 0u)
        {
            if (mant == 0u) return BitCastUIntToFloat(sign);
            exp = 127u - 14u;
            while ((mant & 0x0400u) == 0u)
            {
                mant <<= 1u;
                --exp;
            }
            mant &= 0x03FFu;
            return BitCastUIntToFloat(sign | (exp << 23u) | (mant << 13u));
        }
        if (exp == 31u)
        {
            return BitCastUIntToFloat(sign | 0x7F800000u | (mant << 13u));
        }
        exp = exp + (127u - 15u);
        return BitCastUIntToFloat(sign | (exp << 23u) | (mant << 13u));
    }

    UINT32 NVGBuilder::PackOctahedralNormal(const Vec3& unitNormal) noexcept
    {
        const Vec3    n  = (unitNormal.LengthSq() > 1e-8f) ? unitNormal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
        const FLOAT32 l1 = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
        FLOAT32 ox = (l1 > 1e-8f) ? (n.x / l1) : 0.0f;
        FLOAT32 oy = (l1 > 1e-8f) ? (n.y / l1) : 0.0f;
        if (n.z < 0.0f)
        {
            const FLOAT32 nx = (1.0f - std::fabs(oy)) * ((ox >= 0.0f) ? 1.0f : -1.0f);
            const FLOAT32 ny = (1.0f - std::fabs(ox)) * ((oy >= 0.0f) ? 1.0f : -1.0f);
            ox = nx;
            oy = ny;
        }
        const INT16  ix = static_cast<INT16>(std::lround(ClampF(ox, -1.0f, 1.0f) * 32767.0f));
        const INT16  iy = static_cast<INT16>(std::lround(ClampF(oy, -1.0f, 1.0f) * 32767.0f));
        const UINT16 ux = static_cast<UINT16>(ix);
        const UINT16 uy = static_cast<UINT16>(iy);
        return static_cast<UINT32>(ux) | (static_cast<UINT32>(uy) << 16u);
    }

    Vec3 NVGBuilder::UnpackOctahedralNormal(UINT32 packed) noexcept
    {
        const INT16   ix = static_cast<INT16>(packed & 0xFFFFu);
        const INT16   iy = static_cast<INT16>((packed >> 16u) & 0xFFFFu);
        const FLOAT32 ox = ClampF(static_cast<FLOAT32>(ix) / 32767.0f, -1.0f, 1.0f);
        const FLOAT32 oy = ClampF(static_cast<FLOAT32>(iy) / 32767.0f, -1.0f, 1.0f);
        Vec3 n(ox, oy, 1.0f - std::fabs(ox) - std::fabs(oy));
        const FLOAT32 t = (n.z < 0.0f) ? (-n.z) : 0.0f;
        n.x += (n.x >= 0.0f) ? -t : t;
        n.y += (n.y >= 0.0f) ? -t : t;
        return (n.LengthSq() > 1e-12f) ? n.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
    }

    UINT16 NVGBuilder::PackTangentFrame(const Vec3& unitNormal, const Vec3& unitTangent, const Vec3& unitBitangent) noexcept
    {
        const Vec3 N  = (unitNormal.LengthSq() > 1e-8f) ? unitNormal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 up = (std::fabs(N.y) < 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        const Vec3 B0 = up.Cross(N).Normalized();
        const Vec3 B1 = N.Cross(B0);

        Vec3 T = unitTangent - N * N.Dot(unitTangent);
        if (T.LengthSq() < 1e-8f) T = B0;
        else T = T.Normalized();

        const FLOAT32 angle  = std::atan2(T.Dot(B1), T.Dot(B0));
        const FLOAT32 normA  = ClampF((angle + kNVGPi) / kNVGTwoPi, 0.0f, 1.0f);
        const UINT16  qAngle = static_cast<UINT16>(std::lround(normA * 32767.0f)) & 0x7FFFu;
        const UINT16  hand   = (N.Cross(T).Dot(unitBitangent) < 0.0f) ? 0x8000u : 0x0000u;
        return static_cast<UINT16>(qAngle | hand);
    }

    void NVGBuilder::UnpackTangentFrame(const Vec3& unitNormal, UINT16 packed, Vec3& outTangent, Vec3& outBitangent) noexcept
    {
        const Vec3 N  = (unitNormal.LengthSq() > 1e-8f) ? unitNormal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 up = (std::fabs(N.y) < 0.999f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        const Vec3 B0 = up.Cross(N).Normalized();
        const Vec3 B1 = N.Cross(B0);

        const FLOAT32 normA = static_cast<FLOAT32>(packed & 0x7FFFu) / 32767.0f;
        const FLOAT32 angle = normA * kNVGTwoPi - kNVGPi;
        outTangent          = (B0 * std::cos(angle) + B1 * std::sin(angle)).Normalized();
        const FLOAT32 sign  = ((packed & 0x8000u) != 0u) ? -1.0f : 1.0f;
        outBitangent        = N.Cross(outTangent) * sign;
    }

    // =====================================================================
    //  BuildVirtualMesh — Crack-Free End-to-End NVG2 Builder
    // =====================================================================

    BOOL NVGBuilder::BuildVirtualMesh(
        const Vertex*         vertices,
        UINT32                vertexCount,
        const UINT32*         indices,
        UINT32                indexCount,
        const NVGBuildConfig& config,
        NVGVirtualMeshAsset&  outAsset)
    {
        return BuildVirtualMesh(
            vertices, vertexCount, indices, indexCount,
            nullptr, 0u, config, outAsset, nullptr, nullptr);
    }

    BOOL NVGBuilder::BuildVirtualMesh(
        const Vertex*                   vertices,
        UINT32                          vertexCount,
        const UINT32*                   indices,
        UINT32                          indexCount,
        const NVGSubmeshRange*          submeshes,
        UINT32                          submeshCount,
        const NVGBuildConfig&           config,
        NVGVirtualMeshAsset&            outAsset,
        const std::atomic<bool>*        cancelFlag,
        const NVGBuildProgressCallback& progressCallback)
    {
        const auto tBuildStart = std::chrono::high_resolution_clock::now();
        outAsset = NVGVirtualMeshAsset{};
        NVGBuildReport& rep   = outAsset.buildReport;
        rep.sourceVertexCount = vertexCount;

        auto checkCancelled = [&]() noexcept -> BOOL {
            return (cancelFlag != nullptr && cancelFlag->load(std::memory_order_relaxed)) ? true : false;
        };
        auto emitProgress = [&](FLOAT32 p, const char* phase) {
            if (progressCallback) progressCallback(ClampF(p, 0.0f, 1.0f), phase);
        };
        auto failWith = [&](const char* msg) -> BOOL {
            outAsset = NVGVirtualMeshAsset{};
            outAsset.buildReport.succeeded         = false;
            outAsset.buildReport.sourceVertexCount = vertexCount;
            outAsset.buildReport.diagnosticMessage = msg;
            return false;
        };

        if (checkCancelled())
            return failWith("NVGBuilder cancelled.");

        if (!vertices || vertexCount < 3u)
            return failWith("NVGBuilder rejected mesh: vertexCount < 3 or null vertex data.");

        const BOOL   hasIndexArray       = (indices != nullptr && indexCount > 0u);
        const UINT32 effectiveIndexCount = hasIndexArray ? (indexCount - (indexCount % 3u)) : (vertexCount - (vertexCount % 3u));
        if (effectiveIndexCount < 3u)
            return failWith("NVGBuilder rejected mesh: fewer than 3 triangle indices.");

        // 1. Validate submeshes & collect raw (i0, i1, i2, materialSlot) triangles
        emitProgress(0.02f, "ValidateAndSanitize");
        struct RawTriRef
        {
            UINT32 i0           = 0u;
            UINT32 i1           = 0u;
            UINT32 i2           = 0u;
            UINT16 materialSlot = 0u;
        };
        std::vector<RawTriRef> rawTris;

        if (submeshCount > 0u)
        {
            if (!submeshes)
                return failWith("NVGBuilder rejected mesh: submeshCount > 0 with null submeshes pointer.");

            UINT64 totalSubIndices = 0ull;
            for (UINT32 si = 0u; si < submeshCount; ++si)
            {
                const NVGSubmeshRange& sr = submeshes[si];
                if (sr.materialSlot >= kNVGMaxMaterialSlots)
                    return failWith("NVGBuilder rejected mesh: submesh materialSlot exceeds 255.");
                if (sr.indexCount == 0u || (sr.indexCount % 3u) != 0u)
                    return failWith("NVGBuilder rejected mesh: submesh indexCount must be non-zero and a multiple of 3.");
                if (static_cast<UINT64>(sr.startIndex) + static_cast<UINT64>(sr.indexCount) > static_cast<UINT64>(effectiveIndexCount))
                    return failWith("NVGBuilder rejected mesh: submesh index range exceeds index buffer.");
                totalSubIndices += sr.indexCount;
            }

            rawTris.reserve(static_cast<SIZE_T>(totalSubIndices / 3ull));
            for (UINT32 si = 0u; si < submeshCount; ++si)
            {
                const NVGSubmeshRange& sr = submeshes[si];
                const UINT32 triCnt = sr.indexCount / 3u;
                for (UINT32 t = 0u; t < triCnt; ++t)
                {
                    const UINT32 base = sr.startIndex + t * 3u;
                    const UINT32 i0   = hasIndexArray ? indices[base + 0u] : (base + 0u);
                    const UINT32 i1   = hasIndexArray ? indices[base + 1u] : (base + 1u);
                    const UINT32 i2   = hasIndexArray ? indices[base + 2u] : (base + 2u);
                    if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                        return failWith("NVGBuilder rejected mesh: index out of vertexCount bounds.");
                    rawTris.push_back({ i0, i1, i2, static_cast<UINT16>(sr.materialSlot) });
                }
            }
        }
        else
        {
            const UINT32 rawTriCount = effectiveIndexCount / 3u;
            rawTris.reserve(rawTriCount);
            for (UINT32 t = 0u; t < rawTriCount; ++t)
            {
                const UINT32 i0 = hasIndexArray ? indices[t * 3u + 0u] : (t * 3u + 0u);
                const UINT32 i1 = hasIndexArray ? indices[t * 3u + 1u] : (t * 3u + 1u);
                const UINT32 i2 = hasIndexArray ? indices[t * 3u + 2u] : (t * 3u + 2u);
                if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                    return failWith("NVGBuilder rejected mesh: index out of vertexCount bounds.");
                rawTris.push_back({ i0, i1, i2, 0u });
            }
        }

        rep.sourceTriangleCount = static_cast<UINT32>(rawTris.size());

        // Mark referenced vertices so unreferenced outlier padding vertices NEVER distort AABB or quantization (NVG-027)
        std::vector<UINT8> vertReferenced(vertexCount, 0u);
        for (const RawTriRef& rt : rawTris)
        {
            vertReferenced[rt.i0] = 1u;
            vertReferenced[rt.i1] = 1u;
            vertReferenced[rt.i2] = 1u;
        }

        // Sanitize referenced vertices & compute exact referenced-only AABB
        std::vector<Vertex> sanitizedSourceVerts(vertexCount);
        BOOL hasSkinning = false;
        Vec3 meshMin(1.0e30f, 1.0e30f, 1.0e30f);
        Vec3 meshMax(-1.0e30f, -1.0e30f, -1.0e30f);
        BOOL anyReferenced = false;

        for (UINT32 i = 0u; i < vertexCount; ++i)
        {
            if (!vertReferenced[i]) continue;
            Vertex v = vertices[i];
            if (!IsFiniteVec3(v.position))
                return failWith("NVGBuilder rejected mesh: referenced vertex position contains NaN or Inf.");

            BOOL sanitized = false;
            if (!IsFiniteVec3(v.normal) || v.normal.LengthSq() < 1e-10f)
            {
                v.normal  = Vec3(0.0f, 1.0f, 0.0f);
                sanitized = true;
            }
            else
            {
                v.normal = v.normal.Normalized();
            }

            if (!IsFiniteVec3(v.tangent) || v.tangent.LengthSq() < 1e-10f)
            {
                const Vec3 up = (std::fabs(v.normal.y) < 0.99f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
                v.tangent = up.Cross(v.normal).Normalized();
                sanitized = true;
            }
            else
            {
                v.tangent = v.tangent.Normalized();
            }

            if (!IsFiniteVec3(v.bitangent) || v.bitangent.LengthSq() < 1e-10f)
            {
                v.bitangent = v.normal.Cross(v.tangent);
                sanitized   = true;
            }
            else
            {
                v.bitangent = v.bitangent.Normalized();
            }

            if (!IsFiniteFloat(v.texCoord0.x) || !IsFiniteFloat(v.texCoord0.y))
            {
                v.texCoord0 = Vec2(0.0f, 0.0f);
                sanitized   = true;
            }
            if (!IsFiniteFloat(v.texCoord1.x) || !IsFiniteFloat(v.texCoord1.y))
            {
                v.texCoord1 = Vec2(0.0f, 0.0f);
                sanitized   = true;
            }
            if (!IsFiniteFloat(v.color.x) || !IsFiniteFloat(v.color.y) ||
                !IsFiniteFloat(v.color.z) || !IsFiniteFloat(v.color.w))
            {
                v.color   = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
                sanitized = true;
            }
            if (!IsFiniteFloat(v.boneWeights.x) || !IsFiniteFloat(v.boneWeights.y) ||
                !IsFiniteFloat(v.boneWeights.z) || !IsFiniteFloat(v.boneWeights.w))
            {
                v.boneWeights = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
                sanitized     = true;
            }

            const FLOAT32 bwSum = v.boneWeights.x + v.boneWeights.y + v.boneWeights.z + v.boneWeights.w;
            if (config.preserveSkinningAttributes && bwSum > 1e-5f)
            {
                hasSkinning   = true;
                v.boneWeights = v.boneWeights * (1.0f / bwSum);
            }
            else if (!config.preserveSkinningAttributes)
            {
                v.boneWeights = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if (sanitized) ++rep.sanitizedVertexCount;

            if (v.position.x < meshMin.x) meshMin.x = v.position.x;
            if (v.position.y < meshMin.y) meshMin.y = v.position.y;
            if (v.position.z < meshMin.z) meshMin.z = v.position.z;
            if (v.position.x > meshMax.x) meshMax.x = v.position.x;
            if (v.position.y > meshMax.y) meshMax.y = v.position.y;
            if (v.position.z > meshMax.z) meshMax.z = v.position.z;
            anyReferenced = true;

            sanitizedSourceVerts[i] = v;
        }

        if (!anyReferenced)
            return failWith("NVGBuilder rejected mesh: no referenced vertices.");

        const double meshSpanX = static_cast<double>(meshMax.x) - static_cast<double>(meshMin.x);
        const double meshSpanY = static_cast<double>(meshMax.y) - static_cast<double>(meshMin.y);
        const double meshSpanZ = static_cast<double>(meshMax.z) - static_cast<double>(meshMin.z);
        const double meshDiagD = (std::max)(std::sqrt(meshSpanX * meshSpanX + meshSpanY * meshSpanY + meshSpanZ * meshSpanZ), 1.0e-12);
        const double minCrossSq = (meshDiagD * meshDiagD) * (meshDiagD * meshDiagD) * 1.0e-32;

        // Canonical exact vertex deduplication + double-precision relative degenerate filter (NVG-010)
        std::vector<Vertex> cleanVerts;
        cleanVerts.reserve(vertexCount);
        std::vector<UINT32> srcToDedup(vertexCount, kNVGInvalidIndex);
        std::unordered_map<CanonicalVertexKey, UINT32, CanonicalVertexKeyHash> dedupMap;
        dedupMap.reserve(vertexCount);

        auto getOrInsertCanonicalVertex = [&](UINT32 srcIdx) -> UINT32 {
            if (srcToDedup[srcIdx] != kNVGInvalidIndex)
                return srcToDedup[srcIdx];
            const Vertex& v = sanitizedSourceVerts[srcIdx];
            const CanonicalVertexKey key = MakeCanonicalVertexKey(v, hasSkinning);
            auto [it, inserted] = dedupMap.emplace(key, static_cast<UINT32>(cleanVerts.size()));
            if (inserted)
            {
                cleanVerts.push_back(v);
            }
            srcToDedup[srcIdx] = it->second;
            return it->second;
        };

        std::vector<NVGQEMTriangle> validTris;
        validTris.reserve(rawTris.size());
        Vec3 validMin(1.0e30f, 1.0e30f, 1.0e30f);
        Vec3 validMax(-1.0e30f, -1.0e30f, -1.0e30f);

        for (const RawTriRef& rt : rawTris)
        {
            if (rt.i0 == rt.i1 || rt.i1 == rt.i2 || rt.i2 == rt.i0)
            {
                ++rep.degenerateTrianglesRemoved;
                continue;
            }
            const Vec3& p0 = sanitizedSourceVerts[rt.i0].position;
            const Vec3& p1 = sanitizedSourceVerts[rt.i1].position;
            const Vec3& p2 = sanitizedSourceVerts[rt.i2].position;

            const double e1x = static_cast<double>(p1.x) - static_cast<double>(p0.x);
            const double e1y = static_cast<double>(p1.y) - static_cast<double>(p0.y);
            const double e1z = static_cast<double>(p1.z) - static_cast<double>(p0.z);
            const double e2x = static_cast<double>(p2.x) - static_cast<double>(p0.x);
            const double e2y = static_cast<double>(p2.y) - static_cast<double>(p0.y);
            const double e2z = static_cast<double>(p2.z) - static_cast<double>(p0.z);
            const double cx  = e1y * e2z - e1z * e2y;
            const double cy  = e1z * e2x - e1x * e2z;
            const double cz  = e1x * e2y - e1y * e2x;
            if ((cx * cx + cy * cy + cz * cz) <= minCrossSq)
            {
                ++rep.degenerateTrianglesRemoved;
                continue;
            }

            const UINT32 d0 = getOrInsertCanonicalVertex(rt.i0);
            const UINT32 d1 = getOrInsertCanonicalVertex(rt.i1);
            const UINT32 d2 = getOrInsertCanonicalVertex(rt.i2);
            if (d0 == d1 || d1 == d2 || d2 == d0)
            {
                ++rep.degenerateTrianglesRemoved;
                continue;
            }

            for (const Vec3* vp : { &cleanVerts[d0].position, &cleanVerts[d1].position, &cleanVerts[d2].position })
            {
                if (vp->x < validMin.x) validMin.x = vp->x;
                if (vp->y < validMin.y) validMin.y = vp->y;
                if (vp->z < validMin.z) validMin.z = vp->z;
                if (vp->x > validMax.x) validMax.x = vp->x;
                if (vp->y > validMax.y) validMax.y = vp->y;
                if (vp->z > validMax.z) validMax.z = vp->z;
            }

            validTris.push_back({ { d0, d1, d2 }, rt.materialSlot });
        }

        rep.validTriangleCount = static_cast<UINT32>(validTris.size());
        if (validTris.empty())
        {
            outAsset = NVGVirtualMeshAsset{};
            outAsset.buildReport = rep;
            outAsset.buildReport.succeeded         = false;
            outAsset.buildReport.diagnosticMessage = "NVGBuilder rejected mesh: 0 valid non-degenerate triangles remain.";
            return false;
        }

        meshMin = validMin;
        meshMax = validMax;
        outAsset.localBounds = AABB(meshMin, meshMax);
        outAsset.hasSkinning = hasSkinning;
        rep.hasSkinning      = hasSkinning;

        const FLOAT32 meshDiag = (std::max)((meshMax - meshMin).Length(), 1e-6f);
        const double  weldEps  = static_cast<double>(meshDiag) * 1.0e-6;

        // 2. Deterministic spatial weld IDs & topological analysis
        emitProgress(0.10f, "TopologyAnalysis");
        if (checkCancelled()) return failWith("NVGBuilder cancelled.");

        std::vector<UINT32> spatialIds;
        BuildSpatialWeldIdsDeterministic(cleanVerts, weldEps, spatialIds);

        {
            std::vector<UINT64> sortedSpatialEdges;
            sortedSpatialEdges.reserve(validTris.size() * 3u);
            std::vector<UINT64> sortedRawEdges;
            sortedRawEdges.reserve(validTris.size() * 3u);

            for (const NVGQEMTriangle& t : validTris)
            {
                const UINT32 s0 = spatialIds[t.v[0]];
                const UINT32 s1 = spatialIds[t.v[1]];
                const UINT32 s2 = spatialIds[t.v[2]];
                if (s0 != s1) sortedSpatialEdges.push_back(MakeUndirectedEdgeKey(s0, s1));
                if (s1 != s2) sortedSpatialEdges.push_back(MakeUndirectedEdgeKey(s1, s2));
                if (s2 != s0) sortedSpatialEdges.push_back(MakeUndirectedEdgeKey(s2, s0));

                sortedRawEdges.push_back(MakeUndirectedEdgeKey(t.v[0], t.v[1]));
                sortedRawEdges.push_back(MakeUndirectedEdgeKey(t.v[1], t.v[2]));
                sortedRawEdges.push_back(MakeUndirectedEdgeKey(t.v[2], t.v[0]));
            }

            std::sort(sortedSpatialEdges.begin(), sortedSpatialEdges.end());
            SIZE_T pos = 0u;
            while (pos < sortedSpatialEdges.size())
            {
                const UINT64 k = sortedSpatialEdges[pos];
                SIZE_T end = pos + 1u;
                while (end < sortedSpatialEdges.size() && sortedSpatialEdges[end] == k) ++end;
                const SIZE_T mult = end - pos;
                if (mult == 1u) ++rep.boundaryEdgeCount;
                else if (mult > 2u) ++rep.nonManifoldEdgeCount;
                pos = end;
            }

            if (rep.nonManifoldEdgeCount > 0u && !config.allowNonManifoldGeometry)
            {
                outAsset = NVGVirtualMeshAsset{};
                outAsset.buildReport = rep;
                outAsset.buildReport.succeeded         = false;
                outAsset.buildReport.diagnosticMessage = "NVGBuilder rejected non-manifold mesh (allowNonManifoldGeometry == false).";
                return false;
            }

            std::sort(sortedRawEdges.begin(), sortedRawEdges.end());
            UINT32 rawOpen = 0u;
            pos = 0u;
            while (pos < sortedRawEdges.size())
            {
                const UINT64 k = sortedRawEdges[pos];
                SIZE_T end = pos + 1u;
                while (end < sortedRawEdges.size() && sortedRawEdges[end] == k) ++end;
                if (end - pos == 1u) ++rawOpen;
                pos = end;
            }
            rep.seamEdgeCount = (rawOpen > rep.boundaryEdgeCount) ? ((rawOpen - rep.boundaryEdgeCount) / 2u) : 0u;

            UINT32 maxSpatial = 0u;
            for (UINT32 sid : spatialIds) if (sid > maxSpatial) maxSpatial = sid;
            std::vector<UINT32> dsu(maxSpatial + 1u);
            std::iota(dsu.begin(), dsu.end(), 0u);
            auto findRoot = [&](UINT32 x) {
                while (dsu[x] != x)
                {
                    dsu[x] = dsu[dsu[x]];
                    x      = dsu[x];
                }
                return x;
            };
            for (const NVGQEMTriangle& t : validTris)
            {
                const UINT32 r0 = findRoot(spatialIds[t.v[0]]);
                const UINT32 r1 = findRoot(spatialIds[t.v[1]]);
                const UINT32 r2 = findRoot(spatialIds[t.v[2]]);
                dsu[r1] = r0;
                dsu[r2] = r0;
            }
            std::vector<UINT32> roots;
            roots.reserve(validTris.size());
            for (const NVGQEMTriangle& t : validTris)
            {
                roots.push_back(findRoot(spatialIds[t.v[0]]));
            }
            std::sort(roots.begin(), roots.end());
            roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
            rep.disconnectedIslandCount = static_cast<UINT32>(roots.size());
        }

        // 3. Level 0 Cluster Partitioning (Exact Deduplicated Source Geometry)
        emitProgress(0.20f, "LOD0Clustering");
        if (checkCancelled()) return failWith("NVGBuilder cancelled.");

        const UINT32 targetClusterTris = (std::min)(kNVGMaxClusterTriangles, (std::max)(32u, config.targetTrianglesPerCluster));
        std::vector<BuildCluster> allClusters;
        std::vector<BuildGroup>   allGroups;

        std::vector<BuildCluster> levelClusters;
        PartitionIntoClusters(cleanVerts, spatialIds, validTris, targetClusterTris, 0u, 0.0f, levelClusters);
        rep.lod0ClusterCount = static_cast<UINT32>(levelClusters.size());

        FLOAT32 maxLod0AxisSpan = 1e-6f;
        std::vector<UINT32> currentLevelClusterIds;
        currentLevelClusterIds.reserve(levelClusters.size());
        for (BuildCluster& c : levelClusters)
        {
            const Vec3 ext = c.aabb.Size();
            maxLod0AxisSpan = (std::max)(maxLod0AxisSpan, (std::max)(ext.x, (std::max)(ext.y, ext.z)));

            c.globalIndex     = static_cast<UINT32>(allClusters.size());
            c.selfGroupError  = 0.0f;
            c.selfGroupThin   = 0.0f;
            c.parentGroupThin = 0.0f;
            c.selfGroupSphere = c.sphere;
            currentLevelClusterIds.push_back(c.globalIndex);
            allClusters.push_back(std::move(c));
        }

        const Vec3    meshSpanVec = meshMax - meshMin;
        const FLOAT32 maxMeshAxis = (std::max)(meshSpanVec.x, (std::max)(meshSpanVec.y, meshSpanVec.z));
        const FLOAT32 nonRootMaxClusterSpan = (std::max)(maxLod0AxisSpan * 8.0f, maxMeshAxis * 0.25f);

        // 4. Bottom-Up Hierarchical Cluster Group DAG & Deterministic Parallel QEM Simplification
        UINT32        currentLod          = 0u;
        const FLOAT32 ratio               = ClampF(config.simplificationRatioPerLevel, 0.25f, 0.75f);
        BOOL          forceFinalRootMerge = false;

        const UINT32 hwThreads = (std::max)(1u, std::thread::hardware_concurrency());
        const UINT32 configuredThreads = (config.workerThreadCount == 1u)
            ? 1u
            : ((config.workerThreadCount == 0u) ? hwThreads : config.workerThreadCount);

        while (currentLevelClusterIds.size() > 1u && currentLod + 1u < kNVGMaxLODLevels)
        {
            if (checkCancelled()) return failWith("NVGBuilder cancelled.");
            emitProgress(0.25f + 0.50f * (static_cast<FLOAT32>(currentLod) / static_cast<FLOAT32>(kNVGMaxLODLevels)), "DAGSimplification");

            {
                const Vec3 mSpan = meshMax - meshMin;
                const Vec3 mInvSpan(
                    (mSpan.x > 1e-6f) ? (1.0f / mSpan.x) : 0.0f,
                    (mSpan.y > 1e-6f) ? (1.0f / mSpan.y) : 0.0f,
                    (mSpan.z > 1e-6f) ? (1.0f / mSpan.z) : 0.0f);
                std::sort(currentLevelClusterIds.begin(), currentLevelClusterIds.end(), [&](UINT32 ca, UINT32 cb) {
                    const Vec3 na = allClusters[ca].sphere.center - meshMin;
                    const Vec3 nb = allClusters[cb].sphere.center - meshMin;
                    const UINT32 ma = Morton3D(na.x * mInvSpan.x, na.y * mInvSpan.y, na.z * mInvSpan.z);
                    const UINT32 mb = Morton3D(nb.x * mInvSpan.x, nb.y * mInvSpan.y, nb.z * mInvSpan.z);
                    if (ma != mb) return ma < mb;
                    return ca < cb;
                });
            }

            const UINT32 levelCount     = static_cast<UINT32>(currentLevelClusterIds.size());
            const UINT32 vtxStrideWords = hasSkinning ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;
            std::vector<UINT32> localClusterWords(levelCount, 0u);
            for (UINT32 li = 0u; li < levelCount; ++li)
            {
                const BuildCluster& c = allClusters[currentLevelClusterIds[li]];
                localClusterWords[li] = kNVGClusterHeaderWords +
                                        static_cast<UINT32>(c.vertices.size()) * vtxStrideWords +
                                        static_cast<UINT32>(c.indices.size() / 3u);
            }

            const UINT32 collapseClusterCap = hasSkinning ? 6u : 8u;
            const BOOL   collapseAllToRoot  = (levelCount <= collapseClusterCap) ||
                                              (forceFinalRootMerge && levelCount <= 512u) ||
                                              ((currentLod + 2u >= kNVGMaxLODLevels) && (levelCount <= 1024u));
            const BOOL   enforcePageWordCap = !collapseAllToRoot;
            const UINT32 groupTarget        = collapseAllToRoot ? levelCount : (forceFinalRootMerge ? (hasSkinning ? 12u : 16u) : 4u);
            const UINT32 groupCap           = collapseAllToRoot ? levelCount : (hasSkinning ? 6u : 8u);
            const UINT32 pairMergeCap       = collapseAllToRoot ? levelCount : (forceFinalRootMerge ? 32u : (hasSkinning ? 12u : 14u));
            const UINT32 pairMergeWords     = forceFinalRootMerge ? (kNVGPageWordCount * 4u) : (kNVGPageWordCount * 2u);

            // Build deterministic sorted edge-to-localCluster maps:
            //   1) Z2 boundary edges -> topological adjacency weighting
            //   2) ALL spatial edges -> 100% airtight inter-group boundary vertex lock (including 2+2 non-manifold seams)
            struct EdgeClusterPair
            {
                UINT64 edgeKey = 0ull;
                UINT32 lc      = 0u;
            };
            std::vector<EdgeClusterPair> boundaryPairs;
            boundaryPairs.reserve(static_cast<SIZE_T>(levelCount) * 24u);
            for (UINT32 li = 0u; li < levelCount; ++li)
            {
                const BuildCluster& c = allClusters[currentLevelClusterIds[li]];
                for (UINT64 ek : c.boundaryEdges)
                {
                    boundaryPairs.push_back({ ek, li });
                }
            }
            std::sort(boundaryPairs.begin(), boundaryPairs.end(), [](const EdgeClusterPair& a, const EdgeClusterPair& b) {
                if (a.edgeKey != b.edgeKey) return a.edgeKey < b.edgeKey;
                return a.lc < b.lc;
            });

            // Deterministic sorted neighbor lists per local cluster
            std::vector<std::vector<std::pair<UINT32, UINT32>>> adjWeight(levelCount);
            {
                SIZE_T bpStart = 0u;
                while (bpStart < boundaryPairs.size())
                {
                    const UINT64 key = boundaryPairs[bpStart].edgeKey;
                    SIZE_T bpEnd = bpStart + 1u;
                    while (bpEnd < boundaryPairs.size() && boundaryPairs[bpEnd].edgeKey == key) ++bpEnd;
                    const SIZE_T runLen = bpEnd - bpStart;
                    if (runLen >= 2u && runLen <= 8u)
                    {
                        for (SIZE_T a = bpStart; a < bpEnd; ++a)
                        {
                            for (SIZE_T b = a + 1u; b < bpEnd; ++b)
                            {
                                const UINT32 la = boundaryPairs[a].lc;
                                const UINT32 lb = boundaryPairs[b].lc;
                                if (la != lb)
                                {
                                    adjWeight[la].push_back({ lb, 1u });
                                    adjWeight[lb].push_back({ la, 1u });
                                }
                            }
                        }
                    }
                    bpStart = bpEnd;
                }
                for (UINT32 li = 0u; li < levelCount; ++li)
                {
                    auto& list = adjWeight[li];
                    if (list.empty()) continue;
                    std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                    SIZE_T w = 0u;
                    for (SIZE_T i = 0u; i < list.size(); ++i)
                    {
                        if (w > 0u && list[w - 1u].first == list[i].first)
                            list[w - 1u].second += list[i].second;
                        else
                            list[w++] = list[i];
                    }
                    list.resize(w);
                }
            }

            auto findAdjWeight = [&](UINT32 a, UINT32 b) noexcept -> UINT32 {
                const auto& list = adjWeight[a];
                auto it = std::lower_bound(list.begin(), list.end(), b, [](const auto& p, UINT32 val) { return p.first < val; });
                return (it != list.end() && it->first == b) ? it->second : 0u;
            };

            std::vector<std::vector<UINT32>> localGroups;
            if (collapseAllToRoot)
            {
                std::vector<UINT32> rootGrp(levelCount);
                std::iota(rootGrp.begin(), rootGrp.end(), 0u);
                localGroups.push_back(std::move(rootGrp));
            }
            else
            {
                std::vector<UINT8> grouped(levelCount, 0u);
                std::unordered_map<UINT32, std::vector<UINT32>> siblingBuckets;
                siblingBuckets.reserve(levelCount);
                for (UINT32 li = 0u; li < levelCount; ++li)
                {
                    const BuildCluster& c = allClusters[currentLevelClusterIds[li]];
                    if (c.childGroupCount == 1u)
                    {
                        siblingBuckets[c.childGroupStart].push_back(li);
                    }
                }

                auto computeGroupWords = [&](const std::vector<UINT32>& grp) -> UINT32 {
                    UINT32 sum = 0u;
                    for (UINT32 m : grp) sum += localClusterWords[m];
                    return sum;
                };

                auto tryAddWithSiblings = [&](std::vector<UINT32>& grp, UINT32 cand, UINT32 capClusters, UINT32 capWords) -> BOOL {
                    const UINT32 curWords = computeGroupWords(grp);
                    const BuildCluster& cc = allClusters[currentLevelClusterIds[cand]];
                    if (cc.childGroupCount == 1u)
                    {
                        const auto itS = siblingBuckets.find(cc.childGroupStart);
                        if (itS != siblingBuckets.end())
                        {
                            UINT32 needed    = 0u;
                            UINT32 needWords = 0u;
                            for (UINT32 sib : itS->second)
                            {
                                if (!grouped[sib])
                                {
                                    ++needed;
                                    needWords += localClusterWords[sib];
                                }
                            }
                            if (needed == 0u) return false;
                            if (!grp.empty() && (grp.size() + needed > capClusters || (enforcePageWordCap && curWords + needWords > capWords)))
                                return false;
                            for (UINT32 sib : itS->second)
                            {
                                if (!grouped[sib])
                                {
                                    grouped[sib] = 1u;
                                    grp.push_back(sib);
                                }
                            }
                            return true;
                        }
                    }
                    if (!grp.empty() && (grp.size() + 1u > capClusters || (enforcePageWordCap && curWords + localClusterWords[cand] > capWords)))
                        return false;
                    grouped[cand] = 1u;
                    grp.push_back(cand);
                    return true;
                };

                std::vector<UINT32> groupPackCounts;
                for (UINT32 li = 0u; li < levelCount; ++li)
                {
                    if (grouped[li]) continue;
                    std::vector<UINT32> grp;
                    UINT32 distinctPacks = 0u;
                    if (tryAddWithSiblings(grp, li, pairMergeCap, pairMergeWords))
                    {
                        ++distinctPacks;
                    }

                    while (distinctPacks < 2u || grp.size() < groupTarget)
                    {
                        const UINT32 activeCap   = (distinctPacks < 2u || forceFinalRootMerge) ? pairMergeCap   : groupCap;
                        const UINT32 activeWords = (distinctPacks < 2u || forceFinalRootMerge) ? pairMergeWords : kNVGPageWordCount;
                        const UINT32 curWords    = computeGroupWords(grp);
                        Vec4Sphere curSphere{};
                        for (UINT32 m : grp) curSphere = MergeSpheres(curSphere, allClusters[currentLevelClusterIds[m]].sphere);

                        UINT32  bestNb    = kNVGInvalidIndex;
                        FLOAT32 bestScore = -1.0e30f;
                        for (UINT32 member : grp)
                        {
                            for (const auto& kv : adjWeight[member])
                            {
                                const UINT32 candIdx = kv.first;
                                if (grouped[candIdx]) continue;
                                const BuildCluster& cc = allClusters[currentLevelClusterIds[candIdx]];
                                UINT32 sibNeed  = 1u;
                                UINT32 sibWords = localClusterWords[candIdx];
                                if (cc.childGroupCount == 1u)
                                {
                                    const auto itS = siblingBuckets.find(cc.childGroupStart);
                                    if (itS != siblingBuckets.end())
                                    {
                                        sibNeed  = 0u;
                                        sibWords = 0u;
                                        for (UINT32 sib : itS->second)
                                        {
                                            if (!grouped[sib])
                                            {
                                                ++sibNeed;
                                                sibWords += localClusterWords[sib];
                                            }
                                        }
                                    }
                                }
                                if (grp.size() + sibNeed > activeCap || (enforcePageWordCap && curWords + sibWords > activeWords))
                                    continue;

                                UINT32 wSum = 0u;
                                for (UINT32 m2 : grp) wSum += findAdjWeight(candIdx, m2);
                                if (wSum == 0u) continue;

                                const FLOAT32 mergedR = MergeSpheres(curSphere, cc.sphere).radius;
                                const FLOAT32 score   = static_cast<FLOAT32>(wSum) * 16.0f - (mergedR / meshDiag) * 4.0f;
                                if (score > bestScore || (score == bestScore && candIdx < bestNb))
                                {
                                    bestScore = score;
                                    bestNb    = candIdx;
                                }
                            }
                        }

                        if (bestNb == kNVGInvalidIndex) break;
                        if (!tryAddWithSiblings(grp, bestNb, activeCap, activeWords)) break;
                        ++distinctPacks;
                    }
                    localGroups.push_back(std::move(grp));
                    groupPackCounts.push_back(distinctPacks);
                }

                if (localGroups.size() > 1u)
                {
                    const UINT32 numRawGroups = static_cast<UINT32>(localGroups.size());
                    std::vector<UINT32>     grpWords(numRawGroups, 0u);
                    std::vector<Vec4Sphere> grpSpheres(numRawGroups);
                    std::vector<UINT32>     clusterOwnerGrp(levelCount, 0u);
                    for (UINT32 gi = 0u; gi < numRawGroups; ++gi)
                    {
                        for (UINT32 lc : localGroups[gi])
                        {
                            grpWords[gi]       += localClusterWords[lc];
                            grpSpheres[gi]      = MergeSpheres(grpSpheres[gi], allClusters[currentLevelClusterIds[lc]].sphere);
                            clusterOwnerGrp[lc] = gi;
                        }
                    }

                    for (UINT32 gi = 0u; gi < numRawGroups; ++gi)
                    {
                        if (localGroups[gi].empty()) continue;
                        const BOOL isStraggler = (groupPackCounts[gi] < 2u) || (localGroups[gi].size() <= 2u);
                        if (!isStraggler) continue;

                        std::vector<std::pair<UINT32, UINT32>> sharedWithGroup;
                        for (UINT32 lc : localGroups[gi])
                        {
                            for (const auto& kv : adjWeight[lc])
                            {
                                const UINT32 otherGi = clusterOwnerGrp[kv.first];
                                if (otherGi != gi && !localGroups[otherGi].empty())
                                {
                                    sharedWithGroup.push_back({ otherGi, kv.second });
                                }
                            }
                        }
                        std::sort(sharedWithGroup.begin(), sharedWithGroup.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                        SIZE_T sw = 0u;
                        for (SIZE_T i = 0u; i < sharedWithGroup.size(); ++i)
                        {
                            if (sw > 0u && sharedWithGroup[sw - 1u].first == sharedWithGroup[i].first)
                                sharedWithGroup[sw - 1u].second += sharedWithGroup[i].second;
                            else
                                sharedWithGroup[sw++] = sharedWithGroup[i];
                        }
                        sharedWithGroup.resize(sw);

                        UINT32  bestTargetGi   = kNVGInvalidIndex;
                        FLOAT32 bestMergeScore = -1.0e30f;
                        for (const auto& kv : sharedWithGroup)
                        {
                            const UINT32 candGi = kv.first;
                            if (localGroups[candGi].size() + localGroups[gi].size() > pairMergeCap) continue;
                            if (enforcePageWordCap && grpWords[candGi] + grpWords[gi] > pairMergeWords) continue;
                            const FLOAT32 mergedR = MergeSpheres(grpSpheres[candGi], grpSpheres[gi]).radius;
                            const FLOAT32 score   = static_cast<FLOAT32>(kv.second) * 16.0f - (mergedR / meshDiag) * 4.0f;
                            if (score > bestMergeScore || (score == bestMergeScore && candGi < bestTargetGi))
                            {
                                bestMergeScore = score;
                                bestTargetGi   = candGi;
                            }
                        }

                        if (bestTargetGi != kNVGInvalidIndex)
                        {
                            for (UINT32 lc : localGroups[gi])
                            {
                                localGroups[bestTargetGi].push_back(lc);
                                clusterOwnerGrp[lc] = bestTargetGi;
                            }
                            grpWords[bestTargetGi]        += grpWords[gi];
                            grpSpheres[bestTargetGi]       = MergeSpheres(grpSpheres[bestTargetGi], grpSpheres[gi]);
                            groupPackCounts[bestTargetGi] += groupPackCounts[gi];
                            localGroups[gi].clear();
                            grpWords[gi] = 0u;
                        }
                    }

                    for (UINT32 gi = 0u; gi < numRawGroups; ++gi)
                    {
                        if (localGroups[gi].empty()) continue;
                        const BOOL needsIslandMerge = (groupPackCounts[gi] < 2u) || (localGroups[gi].size() <= 2u);
                        if (!needsIslandMerge) continue;

                        while (groupPackCounts[gi] < 2u || localGroups[gi].size() < groupTarget)
                        {
                            const UINT32 activeCap   = (groupPackCounts[gi] < 2u) ? pairMergeCap   : groupCap;
                            const UINT32 activeWords = (groupPackCounts[gi] < 2u) ? pairMergeWords : kNVGPageWordCount;
                            UINT32  bestOtherGi = kNVGInvalidIndex;
                            FLOAT32 bestRadius  = 1.0e30f;
                            UINT32  probes      = 0u;
                            for (UINT32 candGi = gi + 1u; candGi < numRawGroups && probes < 64u; ++candGi)
                            {
                                if (localGroups[candGi].empty()) continue;
                                ++probes;
                                if (localGroups[gi].size() + localGroups[candGi].size() > activeCap) continue;
                                if (enforcePageWordCap && grpWords[gi] + grpWords[candGi] > activeWords) continue;
                                const FLOAT32 mergedR = MergeSpheres(grpSpheres[gi], grpSpheres[candGi]).radius;
                                if (mergedR < bestRadius)
                                {
                                    bestRadius  = mergedR;
                                    bestOtherGi = candGi;
                                }
                            }
                            if (bestOtherGi == kNVGInvalidIndex) break;

                            for (UINT32 lc : localGroups[bestOtherGi])
                            {
                                localGroups[gi].push_back(lc);
                                clusterOwnerGrp[lc] = gi;
                            }
                            grpWords[gi]        += grpWords[bestOtherGi];
                            grpSpheres[gi]       = MergeSpheres(grpSpheres[gi], grpSpheres[bestOtherGi]);
                            groupPackCounts[gi] += groupPackCounts[bestOtherGi];
                            localGroups[bestOtherGi].clear();
                            grpWords[bestOtherGi] = 0u;
                        }
                    }

                    localGroups.erase(
                        std::remove_if(localGroups.begin(), localGroups.end(),
                            [](const std::vector<UINT32>& g) { return g.empty(); }),
                        localGroups.end());
                }
            }

            const UINT32 numLocalGroups    = static_cast<UINT32>(localGroups.size());
            const BOOL   isFinalMergeToRoot = (numLocalGroups == 1u);

            std::vector<UINT32> localClusterToGroup(levelCount, 0u);
            for (UINT32 gi = 0u; gi < numLocalGroups; ++gi)
            {
                for (UINT32 lc : localGroups[gi]) localClusterToGroup[lc] = gi;
            }

            // Build exact per-group sorted locked spatial ID lists across ALL inter-group shared spatial edges
            std::vector<std::vector<UINT32>> perGroupLockedSids(numLocalGroups);
            if (!isFinalMergeToRoot)
            {
                std::vector<EdgeClusterPair> allEdgeGroupPairs;
                allEdgeGroupPairs.reserve(static_cast<SIZE_T>(levelCount) * 64u);
                for (UINT32 li = 0u; li < levelCount; ++li)
                {
                    const UINT32 gi = localClusterToGroup[li];
                    for (UINT64 ek : allClusters[currentLevelClusterIds[li]].allSpatialEdges)
                    {
                        allEdgeGroupPairs.push_back({ ek, gi });
                    }
                }
                std::sort(allEdgeGroupPairs.begin(), allEdgeGroupPairs.end(), [](const EdgeClusterPair& a, const EdgeClusterPair& b) {
                    if (a.edgeKey != b.edgeKey) return a.edgeKey < b.edgeKey;
                    return a.lc < b.lc;
                });

                SIZE_T p = 0u;
                while (p < allEdgeGroupPairs.size())
                {
                    const UINT64 ek      = allEdgeGroupPairs[p].edgeKey;
                    const UINT32 firstGi = allEdgeGroupPairs[p].lc;
                    SIZE_T end           = p + 1u;
                    BOOL   multiGroup    = false;
                    while (end < allEdgeGroupPairs.size() && allEdgeGroupPairs[end].edgeKey == ek)
                    {
                        if (allEdgeGroupPairs[end].lc != firstGi) multiGroup = true;
                        ++end;
                    }
                    if (multiGroup)
                    {
                        const UINT32 sa = static_cast<UINT32>(ek >> 32u);
                        const UINT32 sb = static_cast<UINT32>(ek & 0xFFFFFFFFu);
                        UINT32 prevGi = kNVGInvalidIndex;
                        for (SIZE_T k = p; k < end; ++k)
                        {
                            const UINT32 g = allEdgeGroupPairs[k].lc;
                            if (g != prevGi)
                            {
                                perGroupLockedSids[g].push_back(sa);
                                perGroupLockedSids[g].push_back(sb);
                                prevGi = g;
                            }
                        }
                    }
                    p = end;
                }

                for (UINT32 gi = 0u; gi < numLocalGroups; ++gi)
                {
                    auto& ls = perGroupLockedSids[gi];
                    std::sort(ls.begin(), ls.end());
                    ls.erase(std::unique(ls.begin(), ls.end()), ls.end());
                }
            }

            // Independent per-group output slots (1 thread == N threads bit-identical!)
            struct GroupSimpOutput
            {
                std::vector<BuildCluster> parentClusters;
                FLOAT32                   stepError = 0.0f;
            };
            std::vector<GroupSimpOutput> groupOutputs(numLocalGroups);

            auto simplifyOneGroup = [&](UINT32 gi) {
                if (checkCancelled()) return;
                const auto& grpLocal = localGroups[gi];
                const auto& lockedSids = perGroupLockedSids[gi];

                std::vector<Vertex>         mergedVerts;
                std::vector<UINT32>         mergedSpatial;
                std::vector<NVGQEMTriangle> mergedTris;

                std::unordered_map<CanonicalVertexKey, UINT32, CanonicalVertexKeyHash> wedgeDedup;
                for (UINT32 lc : grpLocal)
                {
                    const BuildCluster& c = allClusters[currentLevelClusterIds[lc]];
                    std::vector<UINT32> localToMerged(c.vertices.size(), 0u);
                    for (SIZE_T vi = 0; vi < c.vertices.size(); ++vi)
                    {
                        const Vertex& v = c.vertices[vi];
                        CanonicalVertexKey wk = MakeCanonicalVertexKey(v, hasSkinning);
                        wk.words[0] = c.spatialIds[vi]; // Tie canonical wedge to exact welded spatialId
                        auto [it, inserted] = wedgeDedup.emplace(wk, static_cast<UINT32>(mergedVerts.size()));
                        if (inserted)
                        {
                            mergedVerts.push_back(v);
                            mergedSpatial.push_back(c.spatialIds[vi]);
                        }
                        localToMerged[vi] = it->second;
                    }
                    const SIZE_T tc = c.indices.size() / 3u;
                    for (SIZE_T t = 0; t < tc; ++t)
                    {
                        const UINT32 i0 = localToMerged[c.indices[t * 3u + 0u]];
                        const UINT32 i1 = localToMerged[c.indices[t * 3u + 1u]];
                        const UINT32 i2 = localToMerged[c.indices[t * 3u + 2u]];
                        if (i0 != i1 && i1 != i2 && i2 != i0)
                        {
                            mergedTris.push_back({ { i0, i1, i2 }, c.materialSlot });
                        }
                    }
                }

                // Count distinct material slots in this group (each distinct material requires >= 1 parent cluster)
                UINT32 distinctMaterialCount = 1u;
                if (!mergedTris.empty())
                {
                    std::array<UINT8, kNVGMaxMaterialSlots> seenMats{};
                    distinctMaterialCount = 0u;
                    for (const NVGQEMTriangle& mt : mergedTris)
                    {
                        if (mt.materialSlot < kNVGMaxMaterialSlots && !seenMats[mt.materialSlot])
                        {
                            seenMats[mt.materialSlot] = 1u;
                            ++distinctMaterialCount;
                        }
                    }
                    if (distinctMaterialCount == 0u) distinctMaterialCount = 1u;
                }

                const BOOL   relaxSeamsInitial  = forceFinalRootMerge || isFinalMergeToRoot || (currentLod >= 2u);
                const BOOL   coarseRidgeFlatten = forceFinalRootMerge || isFinalMergeToRoot || (currentLod >= 3u);
                const UINT32 inTriCount         = static_cast<UINT32>(mergedTris.size());
                const UINT32 rawTargetTris      = (std::max)(4u, static_cast<UINT32>(std::ceil(static_cast<FLOAT32>(inTriCount) * ratio)));

                UINT32 maxParents = distinctMaterialCount;
                if (!isFinalMergeToRoot && grpLocal.size() >= 2u)
                {
                    const UINT32 byClusterHalving = (std::max)(distinctMaterialCount, static_cast<UINT32>(grpLocal.size()) / 2u);
                    const UINT32 byTriCount       = (std::max)(distinctMaterialCount, (rawTargetTris + targetClusterTris - 1u) / targetClusterTris);
                    maxParents = (std::min)(byClusterHalving, byTriCount);
                }

                UINT32 perParentTriBudget = (targetClusterTris > 8u) ? (targetClusterTris - 4u) : targetClusterTris;
                if (mergedVerts.size() > mergedTris.size() && inTriCount > 0u)
                {
                    const UINT32 safeWedgeBudget = (std::max)(
                        28u,
                        ((kNVGMaxClusterVertices - 8u) * inTriCount) / static_cast<UINT32>(mergedVerts.size()));
                    perParentTriBudget = (std::min)(perParentTriBudget, safeWedgeBudget);
                }
                const UINT32 targetSimpTris = (std::max)(4u, (std::min)(rawTargetTris, maxParents * perParentTriBudget));

                auto invokeSimplifier = [&](const std::vector<Vertex>&         srcV,
                                            const std::vector<UINT32>&         srcS,
                                            const std::vector<NVGQEMTriangle>& srcT,
                                            const std::vector<UINT32>&         locks,
                                            UINT32                             tgtTris,
                                            BOOL                               relaxSeams,
                                            BOOL                               ridgeFlat,
                                            BOOL                               rootFold,
                                            std::vector<Vertex>&               dstV,
                                            std::vector<UINT32>&               dstS,
                                            std::vector<NVGQEMTriangle>&       dstT) -> FLOAT32
                {
                    NVGQEMSimplifierInput qemIn{};
                    qemIn.vertices                = &srcV;
                    qemIn.spatialIds              = &srcS;
                    qemIn.triangles               = &srcT;
                    qemIn.sortedLockedSpatialIds  = &locks;
                    qemIn.targetTriangleCount     = tgtTris;
                    qemIn.config                  = config;
                    qemIn.hasSkinning             = hasSkinning;
                    qemIn.relaxAttributeSeams     = relaxSeams;
                    qemIn.allowCoarseRidgeFlatten = ridgeFlat;
                    qemIn.allowRootNonPlanarFold  = rootFold;
                    return NVGQEMSimplifier::SimplifyGroup(qemIn, dstV, dstS, dstT);
                };

                std::vector<Vertex>         simpVerts;
                std::vector<UINT32>         simpSpatial;
                std::vector<NVGQEMTriangle> simpTris;
                FLOAT32 stepErr = invokeSimplifier(
                    mergedVerts, mergedSpatial, mergedTris, lockedSids,
                    targetSimpTris, relaxSeamsInitial, coarseRidgeFlatten, false,
                    simpVerts, simpSpatial, simpTris);

                if (simpTris.size() > targetSimpTris && simpTris.size() * 4u >= inTriCount * 3u)
                {
                    std::vector<Vertex>         svRelax;
                    std::vector<UINT32>         ssRelax;
                    std::vector<NVGQEMTriangle> stRelax;
                    const FLOAT32 errRelax = invokeSimplifier(
                        mergedVerts, mergedSpatial, mergedTris, lockedSids,
                        targetSimpTris, true, true, false,
                        svRelax, ssRelax, stRelax);
                    if (!stRelax.empty() && stRelax.size() < simpTris.size())
                    {
                        simpVerts   = std::move(svRelax);
                        simpSpatial = std::move(ssRelax);
                        simpTris    = std::move(stRelax);
                        stepErr     = (std::max)(stepErr, errRelax);
                    }
                }

                if (isFinalMergeToRoot && simpTris.size() > distinctMaterialCount * perParentTriBudget)
                {
                    const std::vector<UINT32> noLocks;
                    std::vector<Vertex>         sv2;
                    std::vector<UINT32>         ss2;
                    std::vector<NVGQEMTriangle> st2;
                    const FLOAT32 err2 = invokeSimplifier(
                        simpVerts, simpSpatial, simpTris, noLocks,
                        distinctMaterialCount * perParentTriBudget, true, true, true,
                        sv2, ss2, st2);
                    if (!st2.empty())
                    {
                        simpVerts   = std::move(sv2);
                        simpSpatial = std::move(ss2);
                        simpTris    = std::move(st2);
                        stepErr     = (std::max)(stepErr, err2);
                    }
                }

                if (simpTris.empty())
                {
                    simpVerts   = std::move(mergedVerts);
                    simpSpatial = std::move(mergedSpatial);
                    simpTris    = std::move(mergedTris);
                }

                const FLOAT32 activeSpanCap = isFinalMergeToRoot ? 0.0f : nonRootMaxClusterSpan;
                std::vector<BuildCluster> parentClusters;
                PartitionIntoClusters(simpVerts, simpSpatial, simpTris, targetClusterTris, currentLod + 1u, activeSpanCap, parentClusters);

                if (parentClusters.size() > maxParents && simpTris.size() > 8u)
                {
                    const UINT32 scaledTarget = static_cast<UINT32>((simpTris.size() * maxParents) / parentClusters.size());
                    const UINT32 tightTargetTris = (scaledTarget > 8u) ? (scaledTarget - 4u) : 4u;
                    const std::vector<UINT32> emptyLocks;
                    const std::vector<UINT32>& activeLocks = isFinalMergeToRoot ? emptyLocks : lockedSids;

                    std::vector<Vertex>         svTight;
                    std::vector<UINT32>         ssTight;
                    std::vector<NVGQEMTriangle> stTight;
                    const FLOAT32 errTight = invokeSimplifier(
                        simpVerts, simpSpatial, simpTris, activeLocks,
                        tightTargetTris, true, true, isFinalMergeToRoot,
                        svTight, ssTight, stTight);
                    if (!stTight.empty())
                    {
                        std::vector<BuildCluster> retryClusters;
                        PartitionIntoClusters(svTight, ssTight, stTight, targetClusterTris, currentLod + 1u, activeSpanCap, retryClusters);
                        if (!retryClusters.empty() && retryClusters.size() < parentClusters.size())
                        {
                            parentClusters = std::move(retryClusters);
                            stepErr        = (std::max)(stepErr, errTight);
                        }
                    }
                }

                groupOutputs[gi].parentClusters = std::move(parentClusters);
                groupOutputs[gi].stepError      = stepErr;
            };

            const UINT32 activeWorkers = (std::min)(configuredThreads, numLocalGroups);
            if (activeWorkers <= 1u || numLocalGroups < 4u)
            {
                for (UINT32 gi = 0u; gi < numLocalGroups; ++gi)
                {
                    simplifyOneGroup(gi);
                }
            }
            else
            {
                std::atomic<UINT32> nextGroupIdx{ 0u };
                std::vector<std::thread> workers;
                workers.reserve(activeWorkers - 1u);
                for (UINT32 w = 1u; w < activeWorkers; ++w)
                {
                    workers.emplace_back([&]() {
                        for (;;)
                        {
                            const UINT32 gi = nextGroupIdx.fetch_add(1u, std::memory_order_relaxed);
                            if (gi >= numLocalGroups) break;
                            simplifyOneGroup(gi);
                        }
                    });
                }
                for (;;)
                {
                    const UINT32 gi = nextGroupIdx.fetch_add(1u, std::memory_order_relaxed);
                    if (gi >= numLocalGroups) break;
                    simplifyOneGroup(gi);
                }
                for (std::thread& th : workers)
                {
                    if (th.joinable()) th.join();
                }
            }

            if (checkCancelled()) return failWith("NVGBuilder cancelled.");

            // Deterministic sequential merge of groupOutputs[0 .. numLocalGroups - 1]
            std::vector<UINT32> nextLevelClusterIds;
            for (UINT32 gi = 0u; gi < numLocalGroups; ++gi)
            {
                const auto& grpLocal = localGroups[gi];
                auto&       gOut     = groupOutputs[gi];

                BuildGroup childGroup{};
                childGroup.lodLevel    = currentLod;
                childGroup.globalIndex = static_cast<UINT32>(allGroups.size());
                FLOAT32    maxChildSelfErr = 0.0f;
                Vec4Sphere groupSphere{};

                for (UINT32 lc : grpLocal)
                {
                    const UINT32 gc = currentLevelClusterIds[lc];
                    childGroup.clusterIndices.push_back(gc);
                    allClusters[gc].selfGroupIndex = childGroup.globalIndex;
                    if (allClusters[gc].selfGroupError > maxChildSelfErr)
                    {
                        maxChildSelfErr = allClusters[gc].selfGroupError;
                    }
                    groupSphere = MergeSpheres(groupSphere, allClusters[gc].sphere);
                }

                for (const BuildCluster& pc : gOut.parentClusters)
                {
                    groupSphere = MergeSpheres(groupSphere, pc.sphere);
                }

                const FLOAT32 minStepEpsilon = (std::max)(1e-5f * meshDiag, 1e-6f);
                const FLOAT32 parentError    = maxChildSelfErr + (std::max)(gOut.stepError, minStepEpsilon);

                FLOAT32 groupThin = 0.0f;
                for (UINT32 lc : grpLocal)
                {
                    const BuildCluster& cc = allClusters[currentLevelClusterIds[lc]];
                    groupThin = (std::max)(groupThin, (std::max)(cc.selfGroupThin, cc.thinFeatureMetric));
                }
                for (const BuildCluster& pc : gOut.parentClusters)
                {
                    groupThin = (std::max)(groupThin, pc.thinFeatureMetric);
                }

                childGroup.sphere         = groupSphere;
                childGroup.error          = parentError;
                const UINT32 childGroupId = childGroup.globalIndex;
                allGroups.push_back(std::move(childGroup));

                for (UINT32 lc : grpLocal)
                {
                    BuildCluster& cc     = allClusters[currentLevelClusterIds[lc]];
                    cc.parentGroupError  = parentError;
                    cc.parentGroupSphere = groupSphere;
                    cc.parentGroupThin   = groupThin;
                }

                for (BuildCluster& pc : gOut.parentClusters)
                {
                    pc.globalIndex       = static_cast<UINT32>(allClusters.size());
                    pc.selfGroupError    = parentError;
                    pc.selfGroupSphere   = groupSphere;
                    pc.selfGroupThin     = groupThin;
                    pc.parentGroupError  = kNVGRootErrorInf;
                    pc.parentGroupSphere = groupSphere;
                    pc.parentGroupThin   = groupThin;
                    pc.childGroupStart   = childGroupId;
                    pc.childGroupCount   = 1u;
                    nextLevelClusterIds.push_back(pc.globalIndex);
                    allClusters.push_back(std::move(pc));
                }
            }

            if (nextLevelClusterIds.size() >= currentLevelClusterIds.size() && currentLevelClusterIds.size() > 1u)
            {
                if (collapseAllToRoot || forceFinalRootMerge)
                {
                    currentLevelClusterIds = std::move(nextLevelClusterIds);
                    ++currentLod;
                    break;
                }
                forceFinalRootMerge = true;
            }
            else
            {
                forceFinalRootMerge = false;
            }

            currentLevelClusterIds = std::move(nextLevelClusterIds);
            ++currentLod;
        }

        // Create final root group record for topmost clusters
        {
            BuildGroup rootGroup{};
            rootGroup.lodLevel    = currentLod;
            rootGroup.globalIndex = static_cast<UINT32>(allGroups.size());
            Vec4Sphere rootSphere{};
            FLOAT32    rootErr = 0.0f;
            for (UINT32 rc : currentLevelClusterIds)
            {
                rootGroup.clusterIndices.push_back(rc);
                allClusters[rc].selfGroupIndex   = rootGroup.globalIndex;
                allClusters[rc].parentGroupError = kNVGRootErrorInf;
                rootSphere = MergeSpheres(rootSphere, allClusters[rc].sphere);
                if (allClusters[rc].selfGroupError > rootErr) rootErr = allClusters[rc].selfGroupError;
            }
            rootGroup.sphere = rootSphere;
            rootGroup.error  = rootErr;
            for (UINT32 rc : currentLevelClusterIds)
            {
                allClusters[rc].parentGroupSphere = rootSphere;
                allClusters[rc].parentGroupThin   = allClusters[rc].selfGroupThin;
            }
            allGroups.push_back(std::move(rootGroup));
            rep.rootClusterCount   = static_cast<UINT32>(currentLevelClusterIds.size());
            rep.rootGeometricError = rootErr;
        }

        outAsset.childGroupIndices.clear();
        for (BuildCluster& c : allClusters)
        {
            if (c.childGroupCount == 1u && c.childGroupStart < allGroups.size())
            {
                const UINT32 childG  = c.childGroupStart;
                const UINT32 parentG = c.selfGroupIndex;
                allGroups[childG].parentGroupIndex = parentG;
                auto& chList = allGroups[parentG].childGroupIndices;
                if (std::find(chList.begin(), chList.end(), childG) == chList.end())
                {
                    chList.push_back(childG);
                }
            }
        }
        for (BuildCluster& c : allClusters)
        {
            if (c.selfGroupIndex < allGroups.size())
            {
                c.parentGroupIndex = allGroups[c.selfGroupIndex].parentGroupIndex;
            }
        }

        outAsset.lodLevelCount = currentLod + 1u;
        rep.lodLevelCount      = outAsset.lodLevelCount;
        rep.totalClusterCount  = static_cast<UINT32>(allClusters.size());
        rep.clusterGroupCount  = static_cast<UINT32>(allGroups.size());

        // 5. Global 2^P Integer Lattice Quantization (0 ULP seam identity across all non-root LOD cuts!)
        emitProgress(0.80f, "QuantizeAndPackPages");
        if (checkCancelled()) return failWith("NVGBuilder cancelled.");

        double maxNonRootAxisSpan = 1.0e-12;
        double maxAbsCoord        = 1.0e-12;
        for (const BuildCluster& c : allClusters)
        {
            const BOOL isNonRoot = (c.lodLevel < currentLod) || (currentLod == 0u);
            if (isNonRoot)
            {
                const Vec3 ext = c.aabb.m_Max - c.aabb.m_Min;
                maxNonRootAxisSpan = (std::max)(maxNonRootAxisSpan, static_cast<double>(ext.x));
                maxNonRootAxisSpan = (std::max)(maxNonRootAxisSpan, static_cast<double>(ext.y));
                maxNonRootAxisSpan = (std::max)(maxNonRootAxisSpan, static_cast<double>(ext.z));
            }
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.x)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.y)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.z)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.x)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.y)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.z)));
        }

        // Exact IEEE-754 power-of-two lattice step s_0 = 2^P:
        //   - M_abs / 268435456.0 guarantees |llround(v / s_0)| <= 2^28 << INT32_MAX (zero INT32 overflow).
        //   - maxNonRootAxisSpan / 65534.0 guarantees lodShift == 0 on 100% of non-root clusters (0 ULP seam identity).
        const double minLatticeStep = (std::max)(
            maxNonRootAxisSpan / 65534.0,
            (std::max)(maxAbsCoord / 268435456.0, std::ldexp(1.0, -40)));
        double baseLatticeStep = std::ldexp(1.0, static_cast<int>(std::ceil(std::log2(minLatticeStep))));

        // Verify rounding edge cases so 100% of non-root clusters strictly satisfy maxI - minI <= 65535 (lodShift == 0)
        if (currentLod > 0u)
        {
            const double testInv = 1.0 / baseLatticeStep;
            BOOL needDouble = false;
            for (const BuildCluster& c : allClusters)
            {
                if (c.lodLevel >= currentLod) continue;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double lo = (axis == 0) ? c.aabb.m_Min.x : ((axis == 1) ? c.aabb.m_Min.y : c.aabb.m_Min.z);
                    const double hi = (axis == 0) ? c.aabb.m_Max.x : ((axis == 1) ? c.aabb.m_Max.y : c.aabb.m_Max.z);
                    const INT64 ilo = std::llround(lo * testInv);
                    const INT64 ihi = std::llround(hi * testInv);
                    if (ihi - ilo > 65535ll)
                    {
                        needDouble = true;
                        break;
                    }
                }
                if (needDouble) break;
            }
            if (needDouble) baseLatticeStep *= 2.0;
        }

        FLOAT32 globalMaxQuantErr = 0.0f;
        Vec3 postQuantMeshMin(1.0e30f, 1.0e30f, 1.0e30f);
        Vec3 postQuantMeshMax(-1.0e30f, -1.0e30f, -1.0e30f);

        for (BuildCluster& c : allClusters)
        {
            const FLOAT32 qErr = EncodeClusterToWords(c, baseLatticeStep, hasSkinning);
            if (qErr > globalMaxQuantErr) globalMaxQuantErr = qErr;

            if (c.aabb.m_Min.x < postQuantMeshMin.x) postQuantMeshMin.x = c.aabb.m_Min.x;
            if (c.aabb.m_Min.y < postQuantMeshMin.y) postQuantMeshMin.y = c.aabb.m_Min.y;
            if (c.aabb.m_Min.z < postQuantMeshMin.z) postQuantMeshMin.z = c.aabb.m_Min.z;
            if (c.aabb.m_Max.x > postQuantMeshMax.x) postQuantMeshMax.x = c.aabb.m_Max.x;
            if (c.aabb.m_Max.y > postQuantMeshMax.y) postQuantMeshMax.y = c.aabb.m_Max.y;
            if (c.aabb.m_Max.z > postQuantMeshMax.z) postQuantMeshMax.z = c.aabb.m_Max.z;
        }
        rep.maxQuantizationError = globalMaxQuantErr;
        outAsset.localBounds     = AABB(postQuantMeshMin, postQuantMeshMax);

        // Recompute conservative post-quantization group spheres bottom-up (allGroups is ordered LOD 0 -> Root)
        // while preserving exact bit-for-bit parent/child DAG cut sphere identity:
        //   Parent.selfGroupSphere == Child.parentGroupSphere == allGroups[Child.selfGroupIndex].sphere (NVG-012)
        for (BuildGroup& bg : allGroups)
        {
            Vec4Sphere postSphere = bg.sphere;
            for (UINT32 ci : bg.clusterIndices)
            {
                postSphere = MergeSpheres(postSphere, allClusters[ci].sphere);
            }
            bg.sphere = postSphere;
        }
        for (const BuildCluster& c : allClusters)
        {
            if (c.childGroupCount == 1u && c.childGroupStart < allGroups.size())
            {
                allGroups[c.childGroupStart].sphere = MergeSpheres(allGroups[c.childGroupStart].sphere, c.sphere);
            }
        }
        for (BuildGroup& bg : allGroups)
        {
            for (UINT32 childGi : bg.childGroupIndices)
            {
                if (childGi < allGroups.size())
                {
                    bg.sphere = MergeSpheres(bg.sphere, allGroups[childGi].sphere);
                }
            }
        }
        for (BuildCluster& c : allClusters)
        {
            if (c.childGroupCount == 1u && c.childGroupStart < allGroups.size())
            {
                c.selfGroupSphere = allGroups[c.childGroupStart].sphere;
            }
            else
            {
                c.selfGroupSphere = c.sphere;
            }

            if (c.selfGroupIndex < allGroups.size())
            {
                c.parentGroupSphere = allGroups[c.selfGroupIndex].sphere;
            }
            else
            {
                c.parentGroupSphere = c.selfGroupSphere;
            }
        }

        // Pack root clusters into pinned Root Virtual Pages, then streaming pages by (LOD desc, group asc)
        std::vector<UINT32> rootClusterList;
        std::vector<UINT32> streamingClusterList;
        for (const BuildCluster& c : allClusters)
        {
            if (c.lodLevel == currentLod)
                rootClusterList.push_back(c.globalIndex);
            else
                streamingClusterList.push_back(c.globalIndex);
        }

        std::sort(rootClusterList.begin(), rootClusterList.end(), [&](UINT32 a, UINT32 b) {
            if (allClusters[a].selfGroupIndex != allClusters[b].selfGroupIndex)
                return allClusters[a].selfGroupIndex < allClusters[b].selfGroupIndex;
            return a < b;
        });

        std::sort(streamingClusterList.begin(), streamingClusterList.end(), [&](UINT32 a, UINT32 b) {
            if (allClusters[a].lodLevel != allClusters[b].lodLevel)
                return allClusters[a].lodLevel > allClusters[b].lodLevel;
            if (allClusters[a].selfGroupIndex != allClusters[b].selfGroupIndex)
                return allClusters[a].selfGroupIndex < allClusters[b].selfGroupIndex;
            return a < b;
        });

        auto packClusterListIntoPages = [&](const std::vector<UINT32>& list, BOOL isRoot) {
            if (list.empty()) return;
            NVGVirtualPagePayload curPage{};
            curPage.localPageIndex = static_cast<UINT32>(outAsset.pages.size());
            curPage.lodLevel       = allClusters[list[0]].lodLevel;
            curPage.isRootPage     = isRoot;
            curPage.words.assign(kNVGPageWordCount, 0u);
            curPage.usedWords      = 0u;
            curPage.clusterCount   = 0u;

            SIZE_T idx = 0u;
            while (idx < list.size())
            {
                SIZE_T runEnd   = idx + 1u;
                UINT32 runWords = static_cast<UINT32>(allClusters[list[idx]].encodedWords.size());
                BOOL   groupSpansMultiplePages = false;
                if (!isRoot)
                {
                    const UINT32 grpId = allClusters[list[idx]].selfGroupIndex;
                    while (runEnd < list.size() && allClusters[list[runEnd]].selfGroupIndex == grpId)
                    {
                        const UINT32 w = static_cast<UINT32>(allClusters[list[runEnd]].encodedWords.size());
                        if (runWords + w > kNVGPageWordCount)
                        {
                            groupSpansMultiplePages = true;
                            break;
                        }
                        runWords += w;
                        ++runEnd;
                    }
                }

                const BOOL lodChanged = (!isRoot && allClusters[list[idx]].lodLevel != curPage.lodLevel);
                if ((curPage.usedWords + runWords > kNVGPageWordCount || lodChanged || groupSpansMultiplePages) && curPage.clusterCount > 0u)
                {
                    outAsset.pages.push_back(std::move(curPage));
                    curPage                = NVGVirtualPagePayload{};
                    curPage.localPageIndex = static_cast<UINT32>(outAsset.pages.size());
                    curPage.lodLevel       = allClusters[list[idx]].lodLevel;
                    curPage.isRootPage     = isRoot;
                    curPage.words.assign(kNVGPageWordCount, 0u);
                    curPage.usedWords      = 0u;
                    curPage.clusterCount   = 0u;
                }

                for (SIZE_T k = idx; k < runEnd; ++k)
                {
                    BuildCluster& c       = allClusters[list[k]];
                    const UINT32 needWords = static_cast<UINT32>(c.encodedWords.size());
                    c.virtualPageId       = curPage.localPageIndex;
                    c.wordOffsetInPage    = curPage.usedWords;
                    std::memcpy(curPage.words.data() + curPage.usedWords, c.encodedWords.data(), needWords * sizeof(UINT32));
                    curPage.usedWords += needWords;
                    ++curPage.clusterCount;
                }
                idx = runEnd;
            }

            if (curPage.clusterCount > 0u)
            {
                outAsset.pages.push_back(std::move(curPage));
            }
        };

        packClusterListIntoPages(rootClusterList, true);
        outAsset.rootPageCount = static_cast<UINT32>(outAsset.pages.size());
        packClusterListIntoPages(streamingClusterList, false);

        rep.totalVirtualPageCount     = static_cast<UINT32>(outAsset.pages.size());
        rep.rootVirtualPageCount      = outAsset.rootPageCount;
        rep.streamingVirtualPageCount = rep.totalVirtualPageCount - rep.rootVirtualPageCount;

        // 6. Populate GPU-Aligned Cluster & ClusterGroup Metadata Arrays
        emitProgress(0.92f, "BuildBVH4");
        if (checkCancelled()) return failWith("NVGBuilder cancelled.");

        std::vector<NVGClusterMetaGPUData> rawMeta;
        rawMeta.reserve(allClusters.size());
        outAsset.clusterGroups.resize(allGroups.size());
        std::vector<AABB> groupAABBs(allGroups.size());

        for (SIZE_T gi = 0; gi < allGroups.size(); ++gi)
        {
            const BuildGroup& bg          = allGroups[gi];
            const UINT32 grpClusterStart  = static_cast<UINT32>(rawMeta.size());
            UINT32 minPageId              = kNVGInvalidIndex;
            UINT32 maxPageId              = 0u;
            Vec3   gMin(1.0e30f, 1.0e30f, 1.0e30f);
            Vec3   gMax(-1.0e30f, -1.0e30f, -1.0e30f);

            for (UINT32 origCi : bg.clusterIndices)
            {
                const BuildCluster& bc = allClusters[origCi];
                if (bc.virtualPageId != kNVGInvalidIndex)
                {
                    if (minPageId == kNVGInvalidIndex || bc.virtualPageId < minPageId) minPageId = bc.virtualPageId;
                    if (bc.virtualPageId > maxPageId) maxPageId = bc.virtualPageId;
                }
                if (bc.aabb.m_Min.x < gMin.x) gMin.x = bc.aabb.m_Min.x;
                if (bc.aabb.m_Min.y < gMin.y) gMin.y = bc.aabb.m_Min.y;
                if (bc.aabb.m_Min.z < gMin.z) gMin.z = bc.aabb.m_Min.z;
                if (bc.aabb.m_Max.x > gMax.x) gMax.x = bc.aabb.m_Max.x;
                if (bc.aabb.m_Max.y > gMax.y) gMax.y = bc.aabb.m_Max.y;
                if (bc.aabb.m_Max.z > gMax.z) gMax.z = bc.aabb.m_Max.z;

                NVGClusterMetaGPUData m{};
                m.meshId                   = 0u;
                m.clusterLocalIndex        = static_cast<UINT32>(rawMeta.size());
                m.virtualPageId            = bc.virtualPageId;
                m.clusterWordOffsetInPage  = bc.wordOffsetInPage;
                m.countsAndLod             = bc.encodedWords[14];
                m.selfGroupIndex           = bc.selfGroupIndex;
                m.parentGroupIndex         = bc.parentGroupIndex;
                m.refinedByChildGroupStart = bc.childGroupStart;
                m.refinedByChildGroupCount = PackNVGRefinedCountAndMaterialSlot(bc.childGroupCount, bc.materialSlot);
                m.aabbMin                  = bc.aabb.m_Min;
                m.selfGroupError           = bc.selfGroupError;
                m.aabbMax                  = bc.aabb.m_Max;
                m.parentGroupError         = bc.parentGroupError;
                m.selfGroupSphere          = Vec4(bc.selfGroupSphere.center.x, bc.selfGroupSphere.center.y, bc.selfGroupSphere.center.z, bc.selfGroupSphere.radius);
                m.parentGroupSphere        = Vec4(bc.parentGroupSphere.center.x, bc.parentGroupSphere.center.y, bc.parentGroupSphere.center.z, bc.parentGroupSphere.radius);
                m.normalConeAxisAndSin     = bc.normalConeAxisSin;
                m.thinFeatureMetric        = bc.thinFeatureMetric;
                m.maxEdgeLength            = bc.maxEdgeLength;
                const UINT32 selfThinQ     = QuantizeThinMetric14(bc.selfGroupThin);
                const UINT32 parentThinQ   = QuantizeThinMetric14(bc.parentGroupThin);
                m.clusterFlags             = ((bc.lodLevel == currentLod) ? 1u : 0u) |
                                             2u |
                                             (selfThinQ << 2u) |
                                             (parentThinQ << 16u);
                rawMeta.push_back(m);
            }

            groupAABBs[gi] = AABB(gMin, gMax);
            const UINT32 groupPageCount = (minPageId != kNVGInvalidIndex && maxPageId >= minPageId)
                ? (maxPageId - minPageId + 1u)
                : 1u;

            NVGClusterGroupGPUData gd{};
            gd.meshId             = 0u;
            gd.lodLevel           = PackNVGGroupLodAndPageCount(bg.lodLevel, groupPageCount);
            gd.clusterStart       = grpClusterStart;
            gd.clusterCount       = static_cast<UINT32>(bg.clusterIndices.size());
            gd.parentGroupIndex   = bg.parentGroupIndex;
            gd.childGroupStart    = static_cast<UINT32>(outAsset.childGroupIndices.size());
            gd.childGroupCount    = static_cast<UINT32>(bg.childGroupIndices.size());
            gd.firstVirtualPageId = minPageId;
            gd.groupSphere        = Vec4(bg.sphere.center.x, bg.sphere.center.y, bg.sphere.center.z, bg.sphere.radius);
            for (UINT32 cg : bg.childGroupIndices)
            {
                outAsset.childGroupIndices.push_back(cg);
            }
            outAsset.clusterGroups[gi] = gd;
        }

        // 7. Build LOD-Layered 12-Bin SAH 4-ary BVH4 (internal childCount <= 4, leaf childCount <= 8)
        std::vector<UINT32> groupOrder(outAsset.clusterGroups.size());
        std::iota(groupOrder.begin(), groupOrder.end(), 0u);
        std::vector<UINT32> remappedGroupClusterStarts(outAsset.clusterGroups.size(), 0u);

        outAsset.clusters.clear();
        outAsset.clusters.reserve(rawMeta.size());
        outAsset.bvhNodes.clear();
        outAsset.bvhNodes.push_back(NVGHierarchyNodeGPUData{});
        outAsset.rootNodeIndex = 0u;
        BuildGroupBVH4Recursive(
            0u,
            groupOrder,
            0u,
            static_cast<UINT32>(groupOrder.size()),
            0u,
            groupAABBs,
            outAsset.clusterGroups,
            rawMeta,
            outAsset.clusters,
            remappedGroupClusterStarts,
            outAsset.bvhNodes);

        for (SIZE_T gi = 0; gi < outAsset.clusterGroups.size(); ++gi)
        {
            outAsset.clusterGroups[gi].clusterStart = remappedGroupClusterStarts[gi];
        }

        rep.bvhNodeCount = static_cast<UINT32>(outAsset.bvhNodes.size());

        UINT64 totalEncodedPayloadBytes = 0ull;
        for (const NVGVirtualPagePayload& p : outAsset.pages)
        {
            totalEncodedPayloadBytes += static_cast<UINT64>(p.usedWords) * sizeof(UINT32);
        }
        const UINT64 srcBytes = static_cast<UINT64>(vertexCount) * sizeof(Vertex) +
                                static_cast<UINT64>(rep.validTriangleCount) * 3ull * sizeof(UINT32);
        rep.uncompressedSourceBytes = srcBytes;
        rep.compressedPagesBytes    = totalEncodedPayloadBytes;
        rep.compressionRatio        = (totalEncodedPayloadBytes > 0ull)
            ? static_cast<FLOAT32>(static_cast<double>(srcBytes) / static_cast<double>(totalEncodedPayloadBytes))
            : 1.0f;

        const auto tBuildEnd  = std::chrono::high_resolution_clock::now();
        rep.buildTimeMs       = std::chrono::duration<FLOAT32, std::milli>(tBuildEnd - tBuildStart).count();
        rep.succeeded         = true;
        rep.diagnosticMessage = "OK";
        emitProgress(1.0f, "Completed");
        return true;
    }

    // =====================================================================
    //  DecodeClusterCPU & ExtractFallbackMesh
    // =====================================================================

    BOOL NVGBuilder::DecodeClusterCPU(
        const UINT32*        pageWords,
        UINT32               pageWordCount,
        UINT32               clusterWordOffset,
        std::vector<Vertex>& outVertices,
        std::vector<UINT32>& outIndices,
        UINT32*              outMaterialSlot)
    {
        outVertices.clear();
        outIndices.clear();
        if (outMaterialSlot) *outMaterialSlot = 0u;
        if (!pageWords || clusterWordOffset + kNVGClusterHeaderWords > pageWordCount)
            return false;

        const UINT32* h = pageWords + clusterWordOffset;
        const Vec2 uv0Origin(BitCastUIntToFloat(h[6]), BitCastUIntToFloat(h[7]));
        const Vec2 uv0Extent(BitCastUIntToFloat(h[8]), BitCastUIntToFloat(h[9]));
        const Vec2 uv1Origin(BitCastUIntToFloat(h[10]), BitCastUIntToFloat(h[11]));
        const Vec2 uv1Extent(BitCastUIntToFloat(h[12]), BitCastUIntToFloat(h[13]));

        const UINT32 countsAndFlags = h[14];
        const UINT32 vertCount      = UnpackNVGClusterVertexCount(countsAndFlags);
        const UINT32 triCount       = UnpackNVGClusterTriangleCount(countsAndFlags);
        const UINT32 attrFlags      = (countsAndFlags >> 24u) & 0xFFu;
        const BOOL   hasSkinning    = ((attrFlags & NVG_CLUSTER_ATTR_SKINNING) != 0u);
        const BOOL   useIntLattice  = ((attrFlags & NVG_CLUSTER_ATTR_INT_LATTICE) != 0u);

        const UINT32 offsetsPacked   = h[15];
        const UINT32 vertRelOffset   = useIntLattice ? (offsetsPacked & 0xFFu) : (offsetsPacked & 0xFFFFu);
        const UINT32 materialSlot    = useIntLattice ? ((offsetsPacked >> 8u) & 0xFFu) : 0u;
        const UINT32 idxRelOffset    = (offsetsPacked >> 16u) & 0xFFFFu;
        const UINT32 vertStrideWords = hasSkinning ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;

        if (outMaterialSlot) *outMaterialSlot = materialSlot;

        if (vertCount == 0u || vertCount > kNVGMaxClusterVertices ||
            triCount == 0u || triCount > kNVGMaxClusterTriangles ||
            static_cast<UINT64>(clusterWordOffset) + vertRelOffset + static_cast<UINT64>(vertCount) * vertStrideWords > pageWordCount ||
            static_cast<UINT64>(clusterWordOffset) + idxRelOffset + triCount > pageWordCount)
        {
            return false;
        }

        const INT32   intOriginX = static_cast<INT32>(h[0]);
        const INT32   intOriginY = static_cast<INT32>(h[1]);
        const INT32   intOriginZ = static_cast<INT32>(h[2]);
        const FLOAT32 posScale   = BitCastUIntToFloat(h[3]);
        const UINT32  lodShift   = h[4] & 31u;

        const Vec3 posOrigin(BitCastUIntToFloat(h[0]), BitCastUIntToFloat(h[1]), BitCastUIntToFloat(h[2]));
        const Vec3 posExtent(BitCastUIntToFloat(h[3]), BitCastUIntToFloat(h[4]), BitCastUIntToFloat(h[5]));

        outVertices.resize(vertCount);
        for (UINT32 vi = 0u; vi < vertCount; ++vi)
        {
            const UINT32* src = h + vertRelOffset + vi * vertStrideWords;
            Vertex v{};

            const UINT32 qx      = src[0] & 0xFFFFu;
            const UINT32 qy      = (src[0] >> 16u) & 0xFFFFu;
            const UINT32 qz      = src[1] & 0xFFFFu;
            const UINT16 tanPack = static_cast<UINT16>((src[1] >> 16u) & 0xFFFFu);

            if (useIntLattice)
            {
                const INT32 ix = static_cast<INT32>(static_cast<UINT32>(intOriginX) + (qx << lodShift));
                const INT32 iy = static_cast<INT32>(static_cast<UINT32>(intOriginY) + (qy << lodShift));
                const INT32 iz = static_cast<INT32>(static_cast<UINT32>(intOriginZ) + (qz << lodShift));
                v.position = Vec3(
                    static_cast<FLOAT32>(ix) * posScale,
                    static_cast<FLOAT32>(iy) * posScale,
                    static_cast<FLOAT32>(iz) * posScale);
            }
            else
            {
                v.position = Vec3(
                    posOrigin.x + (static_cast<FLOAT32>(qx) / 65535.0f) * posExtent.x,
                    posOrigin.y + (static_cast<FLOAT32>(qy) / 65535.0f) * posExtent.y,
                    posOrigin.z + (static_cast<FLOAT32>(qz) / 65535.0f) * posExtent.z);
            }

            v.normal = UnpackOctahedralNormal(src[2]);
            UnpackTangentFrame(v.normal, tanPack, v.tangent, v.bitangent);

            const UINT32 qu0 = src[3] & 0xFFFFu;
            const UINT32 qv0 = (src[3] >> 16u) & 0xFFFFu;
            const UINT32 qu1 = src[4] & 0xFFFFu;
            const UINT32 qv1 = (src[4] >> 16u) & 0xFFFFu;
            v.texCoord0 = Vec2(
                uv0Origin.x + (static_cast<FLOAT32>(qu0) / 65535.0f) * uv0Extent.x,
                uv0Origin.y + (static_cast<FLOAT32>(qv0) / 65535.0f) * uv0Extent.y);
            v.texCoord1 = Vec2(
                uv1Origin.x + (static_cast<FLOAT32>(qu1) / 65535.0f) * uv1Extent.x,
                uv1Origin.y + (static_cast<FLOAT32>(qv1) / 65535.0f) * uv1Extent.y);

            v.color = Vec4(
                HalfToFloat(static_cast<UINT16>(src[5] & 0xFFFFu)),
                HalfToFloat(static_cast<UINT16>((src[5] >> 16u) & 0xFFFFu)),
                HalfToFloat(static_cast<UINT16>(src[6] & 0xFFFFu)),
                HalfToFloat(static_cast<UINT16>((src[6] >> 16u) & 0xFFFFu)));

            if (hasSkinning)
            {
                v.boneIndices[0] = src[7] & 0xFFFFu;
                v.boneIndices[1] = (src[7] >> 16u) & 0xFFFFu;
                v.boneIndices[2] = src[8] & 0xFFFFu;
                v.boneIndices[3] = (src[8] >> 16u) & 0xFFFFu;
                v.boneWeights = Vec4(
                    static_cast<FLOAT32>(src[9] & 0xFFFFu) / 65535.0f,
                    static_cast<FLOAT32>((src[9] >> 16u) & 0xFFFFu) / 65535.0f,
                    static_cast<FLOAT32>(src[10] & 0xFFFFu) / 65535.0f,
                    static_cast<FLOAT32>((src[10] >> 16u) & 0xFFFFu) / 65535.0f);
            }

            outVertices[vi] = v;
        }

        outIndices.resize(triCount * 3u);
        for (UINT32 ti = 0u; ti < triCount; ++ti)
        {
            const UINT32 packedTri = h[idxRelOffset + ti];
            outIndices[ti * 3u + 0u] = packedTri & 0xFFu;
            outIndices[ti * 3u + 1u] = (packedTri >> 8u) & 0xFFu;
            outIndices[ti * 3u + 2u] = (packedTri >> 16u) & 0xFFu;
            if (outIndices[ti * 3u + 0u] >= vertCount ||
                outIndices[ti * 3u + 1u] >= vertCount ||
                outIndices[ti * 3u + 2u] >= vertCount)
            {
                return false;
            }
        }
        return true;
    }

    BOOL NVGBuilder::ExtractFallbackMesh(
        const NVGVirtualMeshAsset&    asset,
        std::vector<Vertex>&          outVertices,
        std::vector<UINT32>&          outIndices,
        std::vector<NVGSubmeshRange>* outSubmeshes)
    {
        outVertices.clear();
        outIndices.clear();
        if (outSubmeshes) outSubmeshes->clear();
        if (asset.clusters.empty() || asset.pages.empty() || asset.lodLevelCount == 0u)
            return false;

        const UINT32 rootLod = asset.lodLevelCount - 1u;
        struct RootEntry
        {
            UINT32 clusterIdx   = 0u;
            UINT32 materialSlot = 0u;
        };
        std::vector<RootEntry> rootClusters;
        for (UINT32 ci = 0u; ci < static_cast<UINT32>(asset.clusters.size()); ++ci)
        {
            const NVGClusterMetaGPUData& cm = asset.clusters[ci];
            const UINT32 lod = (cm.countsAndLod >> 16u) & 0xFFu;
            if (((cm.clusterFlags & 1u) != 0u) || lod == rootLod)
            {
                rootClusters.push_back({ ci, UnpackNVGClusterMaterialSlot(cm.refinedByChildGroupCount) });
            }
        }
        if (rootClusters.empty())
            return false;

        std::sort(rootClusters.begin(), rootClusters.end(), [](const RootEntry& a, const RootEntry& b) {
            if (a.materialSlot != b.materialSlot) return a.materialSlot < b.materialSlot;
            return a.clusterIdx < b.clusterIdx;
        });

        std::vector<Vertex> clVerts;
        std::vector<UINT32> clIdxs;
        SIZE_T cursor = 0u;
        while (cursor < rootClusters.size())
        {
            const UINT32 matSlot    = rootClusters[cursor].materialSlot;
            const UINT32 rangeStart = static_cast<UINT32>(outIndices.size());

            while (cursor < rootClusters.size() && rootClusters[cursor].materialSlot == matSlot)
            {
                const NVGClusterMetaGPUData& cm = asset.clusters[rootClusters[cursor].clusterIdx];
                if (cm.virtualPageId >= asset.pages.size())
                    return false;
                const NVGVirtualPagePayload& pg = asset.pages[cm.virtualPageId];
                UINT32 decodedMat = 0u;
                if (!DecodeClusterCPU(pg.words.data(), static_cast<UINT32>(pg.words.size()), cm.clusterWordOffsetInPage, clVerts, clIdxs, &decodedMat))
                    return false;

                const UINT32 baseVert = static_cast<UINT32>(outVertices.size());
                outVertices.insert(outVertices.end(), clVerts.begin(), clVerts.end());
                for (UINT32 li : clIdxs)
                {
                    outIndices.push_back(baseVert + li);
                }
                ++cursor;
            }

            const UINT32 rangeCount = static_cast<UINT32>(outIndices.size()) - rangeStart;
            if (outSubmeshes && rangeCount > 0u)
            {
                outSubmeshes->push_back({ rangeStart, rangeCount, matSlot });
            }
        }

        return !outIndices.empty();
    }

    // =====================================================================
    //  NVG2 Serialization & Dual NVG1/NVG2 Deserialization (Decision 2A)
    // =====================================================================

    BOOL NVGBuilder::Serialize(
        const NVGVirtualMeshAsset& asset,
        std::vector<BYTE>&         outBlob)
    {
        outBlob.clear();
        if (!asset.buildReport.succeeded || asset.clusters.empty() || asset.clusterGroups.empty() ||
            asset.bvhNodes.empty() || asset.pages.empty() || asset.rootPageCount == 0u ||
            asset.rootPageCount > asset.pages.size())
        {
            return false;
        }

        const UINT32 pageCount = static_cast<UINT32>(asset.pages.size());
        std::vector<NVG2PageDirectoryEntry> pageDir(pageCount);
        std::vector<std::vector<BYTE>>      encodedPages(pageCount);
        UINT64 totalCompressedPageBytes = 0ull;

        for (UINT32 pi = 0u; pi < pageCount; ++pi)
        {
            const NVGVirtualPagePayload& p = asset.pages[pi];
            if (p.usedWords > kNVGPageWordCount || p.words.size() < p.usedWords)
                return false;

            const UINT32 rawBytes = p.usedWords * static_cast<UINT32>(sizeof(UINT32));
            const BYTE*  srcPtr   = reinterpret_cast<const BYTE*>(p.words.data());
            const UINT32 crc      = NVGCodecLZ4::ComputeCRC32C(srcPtr, rawBytes);

            std::vector<BYTE> lz4Buf;
            BOOL storedLZ4 = false;
            if (rawBytes > 0u && NVGCodecLZ4::CompressToVector(srcPtr, rawBytes, lz4Buf) &&
                !lz4Buf.empty() && lz4Buf.size() < rawBytes)
            {
                encodedPages[pi] = std::move(lz4Buf);
                storedLZ4        = true;
            }
            else
            {
                encodedPages[pi].assign(srcPtr, srcPtr + rawBytes);
            }

            NVG2PageDirectoryEntry de{};
            de.byteOffsetInFile     = 0ull; // Patched once directory offset is known
            de.compressedByteSize   = static_cast<UINT32>(encodedPages[pi].size());
            de.uncompressedByteSize = rawBytes;
            de.localPageIndex       = pi;
            de.lodLevel             = static_cast<UINT16>(p.lodLevel & 0xFFFFu);
            de.codecAndFlags        = static_cast<UINT16>((p.isRootPage ? 1u : 0u) | (storedLZ4 ? 2u : 0u));
            de.clusterCount         = p.clusterCount;
            de.crc32c               = crc;
            pageDir[pi]             = de;

            totalCompressedPageBytes += de.compressedByteSize;
        }

        const UINT64 clustersBytes = static_cast<UINT64>(asset.clusters.size())          * sizeof(NVGClusterMetaGPUData);
        const UINT64 groupsBytes   = static_cast<UINT64>(asset.clusterGroups.size())     * sizeof(NVGClusterGroupGPUData);
        const UINT64 childBytes    = static_cast<UINT64>(asset.childGroupIndices.size()) * sizeof(UINT32);
        const UINT64 nodesBytes    = static_cast<UINT64>(asset.bvhNodes.size())          * sizeof(NVGHierarchyNodeGPUData);
        const UINT64 tablesSize    = clustersBytes + groupsBytes + childBytes + nodesBytes;
        const UINT64 dirSize       = static_cast<UINT64>(pageCount) * sizeof(NVG2PageDirectoryEntry);

        const UINT64 tablesOffset  = sizeof(NVG2BinaryHeader);
        const UINT64 dirOffset     = tablesOffset + tablesSize;
        const UINT64 pageDataOff   = dirOffset + dirSize;
        const UINT64 totalFileSize = pageDataOff + totalCompressedPageBytes;

        UINT64 curPageOff = pageDataOff;
        for (UINT32 pi = 0u; pi < pageCount; ++pi)
        {
            pageDir[pi].byteOffsetInFile = curPageOff;
            curPageOff += pageDir[pi].compressedByteSize;
        }

        const NVGBuildReport& rep = asset.buildReport;
        NVG2BinaryHeader hdr{};
        hdr.magic                      = kNVG2BinaryMagic;
        hdr.version                    = kNVG2BinaryVersion;
        hdr.flags                      = (asset.hasSkinning ? 1u : 0u) | 2u;
        hdr.lodLevelCount              = asset.lodLevelCount;
        hdr.rootPageCount              = asset.rootPageCount;
        hdr.rootNodeIndex              = asset.rootNodeIndex;
        hdr.clusterCount               = static_cast<UINT32>(asset.clusters.size());
        hdr.groupCount                 = static_cast<UINT32>(asset.clusterGroups.size());
        hdr.childGroupIdCount          = static_cast<UINT32>(asset.childGroupIndices.size());
        hdr.bvhNodeCount               = static_cast<UINT32>(asset.bvhNodes.size());
        hdr.pageCount                  = pageCount;
        hdr.pageSizeBytes              = kNVGPageSizeBytes;
        hdr.boundsMin                  = asset.localBounds.m_Min;
        hdr.maxQuantError              = rep.maxQuantizationError;
        hdr.boundsMax                  = asset.localBounds.m_Max;
        hdr.compressionRatio           = (totalCompressedPageBytes > 0ull && rep.uncompressedSourceBytes > 0ull)
            ? static_cast<FLOAT32>(static_cast<double>(rep.uncompressedSourceBytes) / static_cast<double>(totalCompressedPageBytes))
            : rep.compressionRatio;
        hdr.sourceVertexCount          = rep.sourceVertexCount;
        hdr.sourceTriangleCount        = rep.sourceTriangleCount;
        hdr.validTriangleCount         = rep.validTriangleCount;
        hdr.degenerateTrianglesRemoved = rep.degenerateTrianglesRemoved;
        hdr.sanitizedVertexCount       = rep.sanitizedVertexCount;
        hdr.nonManifoldEdgeCount       = rep.nonManifoldEdgeCount;
        hdr.boundaryEdgeCount          = rep.boundaryEdgeCount;
        hdr.seamEdgeCount              = rep.seamEdgeCount;
        hdr.disconnectedIslandCount    = rep.disconnectedIslandCount;
        hdr.lod0ClusterCount           = rep.lod0ClusterCount;
        hdr.rootClusterCount           = rep.rootClusterCount;
        hdr.rootGeometricError         = rep.rootGeometricError;
        hdr.uncompressedSourceBytes    = rep.uncompressedSourceBytes;
        hdr.compressedPagesBytes       = totalCompressedPageBytes;
        hdr.tablesByteOffset           = tablesOffset;
        hdr.tablesByteSize             = tablesSize;
        hdr.pageDirectoryByteOffset    = dirOffset;
        hdr.pageDataByteOffset         = pageDataOff;
        hdr.totalFileByteSize          = totalFileSize;
        hdr.headerAndTablesChecksum    = 0ull;

        outBlob.resize(static_cast<SIZE_T>(totalFileSize));
        BYTE* dst = outBlob.data();

        std::memcpy(dst, &hdr, sizeof(NVG2BinaryHeader));
        SIZE_T writeCur = sizeof(NVG2BinaryHeader);
        auto writeBytes = [&](const void* src, UINT64 count) {
            if (count > 0ull)
            {
                std::memcpy(dst + writeCur, src, static_cast<SIZE_T>(count));
                writeCur += static_cast<SIZE_T>(count);
            }
        };

        writeBytes(asset.clusters.data(),          clustersBytes);
        writeBytes(asset.clusterGroups.data(),     groupsBytes);
        writeBytes(asset.childGroupIndices.data(), childBytes);
        writeBytes(asset.bvhNodes.data(),          nodesBytes);
        writeBytes(pageDir.data(),                 dirSize);

        // Compute FNV-1a 64-bit over [0 .. pageDataByteOffset) with headerAndTablesChecksum == 0
        hdr.headerAndTablesChecksum = NVGCodecLZ4::ComputeFNV1a64(dst, static_cast<SIZE_T>(pageDataOff));
        std::memcpy(dst, &hdr, sizeof(NVG2BinaryHeader));

        for (UINT32 pi = 0u; pi < pageCount; ++pi)
        {
            if (!encodedPages[pi].empty())
            {
                writeBytes(encodedPages[pi].data(), encodedPages[pi].size());
            }
        }

        return true;
    }

    BOOL NVGBuilder::ParseNVG2MetadataAndDirectory(
        const BYTE*                          headerAndTablesData,
        SIZE_T                               availableBytes,
        NVG2BinaryHeader&                    outHeader,
        NVGVirtualMeshAsset&                 outMetadataOnlyAsset,
        std::vector<NVG2PageDirectoryEntry>& outPageDirectory,
        std::string&                         outError)
    {
        outHeader            = NVG2BinaryHeader{};
        outMetadataOnlyAsset = NVGVirtualMeshAsset{};
        outPageDirectory.clear();
        outError.clear();

        if (!headerAndTablesData || availableBytes < sizeof(NVG2BinaryHeader))
        {
            outError = "NVG2 binary truncated before 192-byte header.";
            return false;
        }

        std::memcpy(&outHeader, headerAndTablesData, sizeof(NVG2BinaryHeader));
        if (outHeader.magic != kNVG2BinaryMagic)
        {
            outError = "NVG2 magic mismatch.";
            return false;
        }
        if (outHeader.version != kNVG2BinaryVersion)
        {
            outError = "NVG2 binary version unsupported.";
            return false;
        }
        if (outHeader.pageSizeBytes != kNVGPageSizeBytes ||
            outHeader.clusterCount == 0u || outHeader.groupCount == 0u ||
            outHeader.bvhNodeCount == 0u || outHeader.pageCount == 0u ||
            outHeader.rootPageCount == 0u || outHeader.rootPageCount > outHeader.pageCount ||
            outHeader.lodLevelCount == 0u || outHeader.lodLevelCount > kNVGMaxLODLevels ||
            outHeader.rootNodeIndex >= outHeader.bvhNodeCount)
        {
            outError = "NVG2 header contains invalid page, LOD, group, BVH, or cluster counts.";
            return false;
        }

        const UINT64 clustersBytes = static_cast<UINT64>(outHeader.clusterCount)      * sizeof(NVGClusterMetaGPUData);
        const UINT64 groupsBytes   = static_cast<UINT64>(outHeader.groupCount)        * sizeof(NVGClusterGroupGPUData);
        const UINT64 childBytes    = static_cast<UINT64>(outHeader.childGroupIdCount) * sizeof(UINT32);
        const UINT64 nodesBytes    = static_cast<UINT64>(outHeader.bvhNodeCount)      * sizeof(NVGHierarchyNodeGPUData);
        const UINT64 dirBytes      = static_cast<UINT64>(outHeader.pageCount)         * sizeof(NVG2PageDirectoryEntry);

        if (clustersBytes > 0x40000000ull || groupsBytes > 0x40000000ull ||
            childBytes > 0x40000000ull || nodesBytes > 0x40000000ull || dirBytes > 0x40000000ull)
        {
            outError = "NVG2 header table dimensions exceed safe allocation limits.";
            return false;
        }

        const UINT64 expectedTablesOff = sizeof(NVG2BinaryHeader);
        const UINT64 expectedTablesSz  = clustersBytes + groupsBytes + childBytes + nodesBytes;
        const UINT64 expectedDirOff    = expectedTablesOff + expectedTablesSz;
        const UINT64 expectedDataOff   = expectedDirOff + dirBytes;

        if (outHeader.tablesByteOffset != expectedTablesOff ||
            outHeader.tablesByteSize != expectedTablesSz ||
            outHeader.pageDirectoryByteOffset != expectedDirOff ||
            outHeader.pageDataByteOffset != expectedDataOff ||
            outHeader.totalFileByteSize < expectedDataOff ||
            availableBytes < expectedDataOff)
        {
            outError = "NVG2 header section offsets mismatch or buffer truncated before page directory end.";
            return false;
        }

        // Verify FNV-1a 64-bit checksum over [header (with checksum zeroed) + tables + page directory]
        NVG2BinaryHeader zeroedHdr = outHeader;
        zeroedHdr.headerAndTablesChecksum = 0ull;
        UINT64 rollingHash = NVGCodecLZ4::ComputeFNV1a64(
            reinterpret_cast<const BYTE*>(&zeroedHdr), sizeof(NVG2BinaryHeader));
        rollingHash = NVGCodecLZ4::ComputeFNV1a64(
            headerAndTablesData + sizeof(NVG2BinaryHeader),
            static_cast<SIZE_T>(expectedDataOff - sizeof(NVG2BinaryHeader)),
            rollingHash);
        if (rollingHash != outHeader.headerAndTablesChecksum)
        {
            outError = "NVG2 header and metadata checksum mismatch (corrupted data).";
            return false;
        }

        outMetadataOnlyAsset.localBounds   = AABB(outHeader.boundsMin, outHeader.boundsMax);
        outMetadataOnlyAsset.hasSkinning   = (outHeader.flags & 1u) != 0u;
        outMetadataOnlyAsset.lodLevelCount = outHeader.lodLevelCount;
        outMetadataOnlyAsset.rootPageCount = outHeader.rootPageCount;
        outMetadataOnlyAsset.rootNodeIndex = outHeader.rootNodeIndex;

        outMetadataOnlyAsset.clusters.resize(outHeader.clusterCount);
        outMetadataOnlyAsset.clusterGroups.resize(outHeader.groupCount);
        outMetadataOnlyAsset.childGroupIndices.resize(outHeader.childGroupIdCount);
        outMetadataOnlyAsset.bvhNodes.resize(outHeader.bvhNodeCount);
        outPageDirectory.resize(outHeader.pageCount);

        SIZE_T cur = sizeof(NVG2BinaryHeader);
        auto copySection = [&](void* dst, UINT64 bytes) {
            if (bytes > 0ull)
            {
                std::memcpy(dst, headerAndTablesData + cur, static_cast<SIZE_T>(bytes));
                cur += static_cast<SIZE_T>(bytes);
            }
        };
        copySection(outMetadataOnlyAsset.clusters.data(),          clustersBytes);
        copySection(outMetadataOnlyAsset.clusterGroups.data(),     groupsBytes);
        copySection(outMetadataOnlyAsset.childGroupIndices.data(), childBytes);
        copySection(outMetadataOnlyAsset.bvhNodes.data(),          nodesBytes);
        copySection(outPageDirectory.data(),                       dirBytes);

        std::vector<UINT32> pageUsedWords(outHeader.pageCount, 0u);
        for (UINT32 pi = 0u; pi < outHeader.pageCount; ++pi)
        {
            const NVG2PageDirectoryEntry& de = outPageDirectory[pi];
            if (de.localPageIndex != pi ||
                de.uncompressedByteSize > kNVGPageSizeBytes ||
                (de.uncompressedByteSize % sizeof(UINT32)) != 0u ||
                de.byteOffsetInFile < outHeader.pageDataByteOffset ||
                de.compressedByteSize > outHeader.totalFileByteSize ||
                de.byteOffsetInFile > outHeader.totalFileByteSize - de.compressedByteSize)
            {
                outMetadataOnlyAsset = NVGVirtualMeshAsset{};
                outPageDirectory.clear();
                outError = "NVG2 page directory entry contains out-of-bounds offset or byte size.";
                return false;
            }
            const BOOL isLZ4 = ((de.codecAndFlags & 2u) != 0u);
            if (!isLZ4 && de.compressedByteSize != de.uncompressedByteSize)
            {
                outMetadataOnlyAsset = NVGVirtualMeshAsset{};
                outPageDirectory.clear();
                outError = "NVG2 uncompressed page directory entry has mismatched byte sizes.";
                return false;
            }
            pageUsedWords[pi] = de.uncompressedByteSize / static_cast<UINT32>(sizeof(UINT32));
        }

        if (!ValidateDeserializedTablesAndBVH(outMetadataOnlyAsset, outHeader.pageCount, pageUsedWords, false, outError))
        {
            outMetadataOnlyAsset = NVGVirtualMeshAsset{};
            outPageDirectory.clear();
            return false;
        }

        NVGBuildReport& rep = outMetadataOnlyAsset.buildReport;
        rep.succeeded                  = true;
        rep.sourceVertexCount          = outHeader.sourceVertexCount;
        rep.sourceTriangleCount        = outHeader.sourceTriangleCount;
        rep.validTriangleCount         = outHeader.validTriangleCount;
        rep.degenerateTrianglesRemoved = outHeader.degenerateTrianglesRemoved;
        rep.sanitizedVertexCount       = outHeader.sanitizedVertexCount;
        rep.nonManifoldEdgeCount       = outHeader.nonManifoldEdgeCount;
        rep.boundaryEdgeCount          = outHeader.boundaryEdgeCount;
        rep.seamEdgeCount              = outHeader.seamEdgeCount;
        rep.disconnectedIslandCount    = outHeader.disconnectedIslandCount;
        rep.totalClusterCount          = outHeader.clusterCount;
        rep.lod0ClusterCount           = outHeader.lod0ClusterCount;
        rep.rootClusterCount           = outHeader.rootClusterCount;
        rep.lodLevelCount              = outHeader.lodLevelCount;
        rep.clusterGroupCount          = outHeader.groupCount;
        rep.bvhNodeCount               = outHeader.bvhNodeCount;
        rep.totalVirtualPageCount      = outHeader.pageCount;
        rep.rootVirtualPageCount       = outHeader.rootPageCount;
        rep.streamingVirtualPageCount  = outHeader.pageCount - outHeader.rootPageCount;
        rep.uncompressedSourceBytes    = outHeader.uncompressedSourceBytes;
        rep.compressedPagesBytes       = outHeader.compressedPagesBytes;
        rep.compressionRatio           = outHeader.compressionRatio;
        rep.maxQuantizationError       = outHeader.maxQuantError;
        rep.rootGeometricError         = outHeader.rootGeometricError;
        rep.hasSkinning                = outMetadataOnlyAsset.hasSkinning;
        rep.diagnosticMessage          = "Deserialized NVG2";
        return true;
    }

    BOOL NVGBuilder::Deserialize(
        const BYTE*          data,
        SIZE_T               size,
        NVGVirtualMeshAsset& outAsset,
        std::string&         outError)
    {
        outAsset = NVGVirtualMeshAsset{};
        outError.clear();

        if (!data || size < sizeof(UINT32) * 2u)
        {
            outError = "NVG binary blob is null or truncated before magic.";
            return false;
        }

        UINT32 magic = 0u;
        std::memcpy(&magic, data, sizeof(UINT32));

        // Path 1: Modern NVG2 container (LZ4 pages + FNV-1a metadata + per-page CRC32C)
        if (magic == kNVG2BinaryMagic)
        {
            NVG2BinaryHeader hdr{};
            std::vector<NVG2PageDirectoryEntry> pageDir;
            if (!ParseNVG2MetadataAndDirectory(data, size, hdr, outAsset, pageDir, outError))
            {
                outAsset = NVGVirtualMeshAsset{};
                return false;
            }
            if (size != hdr.totalFileByteSize)
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG2 binary blob size does not match header totalFileByteSize.";
                return false;
            }

            outAsset.pages.resize(hdr.pageCount);
            std::vector<UINT32> pageUsedWords(hdr.pageCount, 0u);
            for (UINT32 pi = 0u; pi < hdr.pageCount; ++pi)
            {
                const NVG2PageDirectoryEntry& de = pageDir[pi];
                NVGVirtualPagePayload& p         = outAsset.pages[pi];
                p.localPageIndex = pi;
                p.lodLevel       = de.lodLevel;
                p.isRootPage     = ((de.codecAndFlags & 1u) != 0u);
                p.clusterCount   = de.clusterCount;
                p.usedWords      = de.uncompressedByteSize / static_cast<UINT32>(sizeof(UINT32));
                p.words.assign(kNVGPageWordCount, 0u);
                pageUsedWords[pi] = p.usedWords;

                const BYTE* srcBytes = data + static_cast<SIZE_T>(de.byteOffsetInFile);
                BYTE*       dstBytes = reinterpret_cast<BYTE*>(p.words.data());
                if ((de.codecAndFlags & 2u) != 0u)
                {
                    if (!NVGCodecLZ4::DecompressBlockExact(srcBytes, de.compressedByteSize, dstBytes, de.uncompressedByteSize))
                    {
                        outAsset = NVGVirtualMeshAsset{};
                        outError = "NVG2 LZ4 page decompression failed (corrupted or malformed block).";
                        return false;
                    }
                }
                else if (de.uncompressedByteSize > 0u)
                {
                    std::memcpy(dstBytes, srcBytes, de.uncompressedByteSize);
                }

                const UINT32 actualCrc = NVGCodecLZ4::ComputeCRC32C(dstBytes, de.uncompressedByteSize);
                if (actualCrc != de.crc32c)
                {
                    outAsset = NVGVirtualMeshAsset{};
                    outError = "NVG2 page CRC32C checksum mismatch (corrupted page payload).";
                    return false;
                }
            }

            // Re-run payload-level cluster offset & index validation now that page words are decompressed
            if (!ValidateDeserializedTablesAndBVH(outAsset, hdr.pageCount, pageUsedWords, true, outError))
            {
                outAsset = NVGVirtualMeshAsset{};
                return false;
            }
            return true;
        }

        // Path 2: Backward-compatible legacy NVG1 container
        if (magic != kNVG1BinaryMagic)
        {
            outError = "NVG binary magic mismatch (expected 'NVG2' or 'NVG1').";
            return false;
        }
        if (size < sizeof(NVG1BinaryHeader))
        {
            outError = "NVG1 binary blob truncated before header.";
            return false;
        }

        NVG1BinaryHeader hdr{};
        std::memcpy(&hdr, data, sizeof(NVG1BinaryHeader));
        if (hdr.version != kNVGBinaryVersion)
        {
            outError = "NVG1 binary version unsupported.";
            return false;
        }
        if (hdr.pageSizeBytes != kNVGPageSizeBytes || hdr.clusterCount == 0u || hdr.pageCount == 0u ||
            hdr.rootPageCount == 0u || hdr.rootPageCount > hdr.pageCount ||
            hdr.lodLevelCount == 0u || hdr.lodLevelCount > kNVGMaxLODLevels ||
            hdr.groupCount == 0u || hdr.bvhNodeCount == 0u || hdr.rootNodeIndex >= hdr.bvhNodeCount)
        {
            outError = "NVG1 binary header contains invalid geometry page, LOD, group, BVH, or cluster counts.";
            return false;
        }
        if (hdr.payloadByteSize > size || sizeof(NVG1BinaryHeader) + hdr.payloadByteSize != size)
        {
            outError = "NVG1 binary payload byte size mismatch.";
            return false;
        }

        const UINT64 clustersBytes = static_cast<UINT64>(hdr.clusterCount)      * sizeof(NVGClusterMetaGPUData);
        const UINT64 groupsBytes   = static_cast<UINT64>(hdr.groupCount)        * sizeof(NVGClusterGroupGPUData);
        const UINT64 childBytes    = static_cast<UINT64>(hdr.childGroupIdCount) * sizeof(UINT32);
        const UINT64 nodesBytes    = static_cast<UINT64>(hdr.bvhNodeCount)      * sizeof(NVGHierarchyNodeGPUData);
        const UINT64 pageHdrBytes  = static_cast<UINT64>(hdr.pageCount)         * sizeof(NVG1BinaryPageMeta);
        if (clustersBytes > hdr.payloadByteSize ||
            groupsBytes   > hdr.payloadByteSize - clustersBytes ||
            childBytes    > hdr.payloadByteSize - clustersBytes - groupsBytes ||
            nodesBytes    > hdr.payloadByteSize - clustersBytes - groupsBytes - childBytes ||
            pageHdrBytes  > hdr.payloadByteSize - clustersBytes - groupsBytes - childBytes - nodesBytes)
        {
            outError = "NVG1 binary header table sizes exceed payload byte size.";
            return false;
        }

        const BYTE*  payloadPtr     = data + sizeof(NVG1BinaryHeader);
        const UINT64 actualChecksum = NVGCodecLZ4::ComputeFNV1a64(payloadPtr, static_cast<SIZE_T>(hdr.payloadByteSize));
        if (actualChecksum != hdr.payloadChecksum)
        {
            outError = "NVG1 binary checksum mismatch (corrupted data).";
            return false;
        }

        SIZE_T cursor = 0u;
        auto readRaw = [&](void* dst, SIZE_T bytes) -> BOOL {
            if (bytes > static_cast<SIZE_T>(hdr.payloadByteSize) ||
                cursor > static_cast<SIZE_T>(hdr.payloadByteSize) - bytes)
                return false;
            if (bytes > 0u)
                std::memcpy(dst, payloadPtr + cursor, bytes);
            cursor += bytes;
            return true;
        };

        outAsset.localBounds   = AABB(hdr.boundsMin, hdr.boundsMax);
        outAsset.hasSkinning   = (hdr.flags & 1u) != 0u;
        outAsset.lodLevelCount = hdr.lodLevelCount;
        outAsset.rootPageCount = hdr.rootPageCount;
        outAsset.rootNodeIndex = hdr.rootNodeIndex;

        outAsset.clusters.resize(hdr.clusterCount);
        outAsset.clusterGroups.resize(hdr.groupCount);
        outAsset.childGroupIndices.resize(hdr.childGroupIdCount);
        outAsset.bvhNodes.resize(hdr.bvhNodeCount);

        if (!readRaw(outAsset.clusters.data(),          static_cast<SIZE_T>(clustersBytes)) ||
            !readRaw(outAsset.clusterGroups.data(),     static_cast<SIZE_T>(groupsBytes)) ||
            !readRaw(outAsset.childGroupIndices.data(), static_cast<SIZE_T>(childBytes)) ||
            !readRaw(outAsset.bvhNodes.data(),          static_cast<SIZE_T>(nodesBytes)))
        {
            outAsset = NVGVirtualMeshAsset{};
            outError = "NVG1 binary payload truncated in metadata tables.";
            return false;
        }

        outAsset.pages.resize(hdr.pageCount);
        std::vector<UINT32> pageUsedWords(hdr.pageCount, 0u);
        for (UINT32 pi = 0u; pi < hdr.pageCount; ++pi)
        {
            NVG1BinaryPageMeta pm{};
            if (!readRaw(&pm, sizeof(NVG1BinaryPageMeta)) || pm.usedWords > kNVGPageWordCount)
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG1 binary payload truncated or invalid page metadata.";
                return false;
            }
            NVGVirtualPagePayload& p = outAsset.pages[pi];
            p.localPageIndex  = pm.localPageIndex;
            p.lodLevel        = pm.lodLevel;
            p.isRootPage      = (pm.isRootPage != 0u);
            p.clusterCount    = pm.clusterCount;
            p.usedWords       = pm.usedWords;
            p.words.assign(kNVGPageWordCount, 0u);
            pageUsedWords[pi] = pm.usedWords;
            if (!readRaw(p.words.data(), static_cast<SIZE_T>(pm.usedWords) * sizeof(UINT32)))
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG1 binary payload truncated in compressed page words.";
                return false;
            }
        }

        if (cursor != static_cast<SIZE_T>(hdr.payloadByteSize))
        {
            outAsset = NVGVirtualMeshAsset{};
            outError = "NVG1 binary contains trailing unparsed bytes.";
            return false;
        }

        if (!ValidateDeserializedTablesAndBVH(outAsset, hdr.pageCount, pageUsedWords, true, outError))
        {
            outAsset = NVGVirtualMeshAsset{};
            return false;
        }

        NVGBuildReport& rep = outAsset.buildReport;
        rep.succeeded                 = true;
        rep.totalClusterCount         = hdr.clusterCount;
        rep.lodLevelCount             = hdr.lodLevelCount;
        rep.clusterGroupCount         = hdr.groupCount;
        rep.bvhNodeCount              = hdr.bvhNodeCount;
        rep.totalVirtualPageCount     = hdr.pageCount;
        rep.rootVirtualPageCount      = hdr.rootPageCount;
        rep.streamingVirtualPageCount = hdr.pageCount - hdr.rootPageCount;
        rep.maxQuantizationError      = hdr.maxQuantError;
        rep.compressionRatio          = hdr.compressionRatio;
        rep.hasSkinning               = outAsset.hasSkinning;
        rep.diagnosticMessage         = "Deserialized NVG1";
        return true;
    }
}
