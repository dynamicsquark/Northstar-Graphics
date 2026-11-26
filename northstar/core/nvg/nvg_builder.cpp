#include "nvg_builder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
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

        inline UINT64 ComputeFNV1a64(const BYTE* data, SIZE_T size) noexcept
        {
            UINT64 h = 14695981039346656037ULL;
            for (SIZE_T i = 0; i < size; ++i)
            {
                h ^= static_cast<UINT64>(data[i]);
                h *= 1099511628211ULL;
            }
            return h;
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

        struct SpatialKey
        {
            INT64 x = 0;
            INT64 y = 0;
            INT64 z = 0;

            bool operator==(const SpatialKey& o) const noexcept
            {
                return x == o.x && y == o.y && z == o.z;
            }
        };

        struct SpatialKeyHash
        {
            SIZE_T operator()(const SpatialKey& k) const noexcept
            {
                UINT64 h = 14695981039346656037ULL;
                h ^= static_cast<UINT64>(k.x); h *= 1099511628211ULL;
                h ^= static_cast<UINT64>(k.y); h *= 1099511628211ULL;
                h ^= static_cast<UINT64>(k.z); h *= 1099511628211ULL;
                return static_cast<SIZE_T>(h);
            }
        };

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

        // Symmetric 4x4 Garland-Heckbert Quadric Error Metric
        struct SymmetricQuadric
        {
            double a00 = 0.0, a01 = 0.0, a02 = 0.0, a03 = 0.0;
            double            a11 = 0.0, a12 = 0.0, a13 = 0.0;
            double                       a22 = 0.0, a23 = 0.0;
            double                                  a33 = 0.0;

            void AddPlane(double nx, double ny, double nz, double d, double weight = 1.0) noexcept
            {
                a00 += nx * nx * weight; a01 += nx * ny * weight; a02 += nx * nz * weight; a03 += nx * d * weight;
                a11 += ny * ny * weight; a12 += ny * nz * weight; a13 += ny * d * weight;
                a22 += nz * nz * weight; a23 += nz * d * weight;
                a33 += d  * d  * weight;
            }

            SymmetricQuadric operator+(const SymmetricQuadric& o) const noexcept
            {
                SymmetricQuadric q{};
                q.a00 = a00 + o.a00; q.a01 = a01 + o.a01; q.a02 = a02 + o.a02; q.a03 = a03 + o.a03;
                q.a11 = a11 + o.a11; q.a12 = a12 + o.a12; q.a13 = a13 + o.a13;
                q.a22 = a22 + o.a22; q.a23 = a23 + o.a23;
                q.a33 = a33 + o.a33;
                return q;
            }

            double Evaluate(const Vec3& p) const noexcept
            {
                const double x = p.x;
                const double y = p.y;
                const double z = p.z;
                const double val =
                    a00 * x * x + 2.0 * a01 * x * y + 2.0 * a02 * x * z + 2.0 * a03 * x +
                    a11 * y * y + 2.0 * a12 * y * z + 2.0 * a13 * y +
                    a22 * z * z + 2.0 * a23 * z +
                    a33;
                return (val > 0.0) ? val : 0.0;
            }

            BOOL SolveOptimalPosition(Vec3& outPos) const noexcept
            {
                const double det =
                    a00 * (a11 * a22 - a12 * a12) -
                    a01 * (a01 * a22 - a12 * a02) +
                    a02 * (a01 * a12 - a11 * a02);
                if (std::fabs(det) < 1e-10)
                    return false;

                const double invDet = 1.0 / det;
                const double c00 =  (a11 * a22 - a12 * a12) * invDet;
                const double c01 = -(a01 * a22 - a12 * a02) * invDet;
                const double c02 =  (a01 * a12 - a11 * a02) * invDet;
                const double c11 =  (a00 * a22 - a02 * a02) * invDet;
                const double c12 = -(a00 * a12 - a01 * a02) * invDet;
                const double c22 =  (a00 * a11 - a01 * a01) * invDet;

                const double rx = -(c00 * a03 + c01 * a13 + c02 * a23);
                const double ry = -(c01 * a03 + c11 * a13 + c12 * a23);
                const double rz = -(c02 * a03 + c12 * a13 + c22 * a23);

                if (!std::isfinite(rx) || !std::isfinite(ry) || !std::isfinite(rz))
                    return false;

                outPos = Vec3(static_cast<FLOAT32>(rx), static_cast<FLOAT32>(ry), static_cast<FLOAT32>(rz));
                return true;
            }
        };

        struct BuildTriangle
        {
            UINT32 v[3] = { 0u, 0u, 0u };
        };

        struct BuildCluster
        {
            UINT32                  lodLevel          = 0u;
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
            FLOAT32                 selfGroupThin     = 0.0f; // Synchronized thin metric matching child group transition
            FLOAT32                 parentGroupThin   = 0.0f; // Synchronized thin metric matching parent group transition
            FLOAT32                 maxEdgeLength     = 0.0f;
            std::vector<UINT64>     boundaryEdges;    // Spatial-welded edges on perimeter of cluster
            UINT32                  virtualPageId     = kNVGInvalidIndex;
            UINT32                  wordOffsetInPage  = 0u;
            std::vector<UINT32>     encodedWords;
        };

        static inline UINT32 QuantizeThinMetric14(FLOAT32 metric) noexcept
        {
            const FLOAT32 clamped = ClampF(metric, 0.0f, 4.0f);
            return static_cast<UINT32>(std::lround((clamped / 4.0f) * 16383.0f)) & 0x3FFFu;
        }

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

        static void BuildSpatialWeldIds(
            const std::vector<Vertex>& vertices,
            FLOAT32                    weldEpsilon,
            std::vector<UINT32>&       outSpatialIds)
        {
            outSpatialIds.resize(vertices.size(), 0u);
            const FLOAT32 invEps = (weldEpsilon > 1e-8f) ? (1.0f / weldEpsilon) : 1.0e5f;
            std::unordered_map<SpatialKey, UINT32, SpatialKeyHash> map;
            map.reserve(vertices.size());
            UINT32 nextId = 0u;
            for (SIZE_T i = 0; i < vertices.size(); ++i)
            {
                const Vec3& p = vertices[i].position;
                SpatialKey k{};
                k.x = static_cast<INT64>(std::llround(static_cast<double>(p.x) * static_cast<double>(invEps)));
                k.y = static_cast<INT64>(std::llround(static_cast<double>(p.y) * static_cast<double>(invEps)));
                k.z = static_cast<INT64>(std::llround(static_cast<double>(p.z) * static_cast<double>(invEps)));
                auto [it, inserted] = map.emplace(k, nextId);
                if (inserted)
                {
                    ++nextId;
                }
                outSpatialIds[i] = it->second;
            }
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

            // Ritter-tight bounding sphere anchored on AABB center
            const Vec3 center = c.aabb.Center();
            FLOAT32 maxR2 = 0.0f;
            for (const Vertex& v : c.vertices)
            {
                const FLOAT32 d2 = (v.position - center).LengthSq();
                if (d2 > maxR2) maxR2 = d2;
            }
            c.sphere.center = center;
            c.sphere.radius = std::sqrt(maxR2);

            // Face normal cone, boundary edges, max edge length, and thin-feature metric
            const SIZE_T triCount = c.indices.size() / 3u;
            std::vector<Vec3> faceNormals;
            faceNormals.reserve(triCount);
            Vec3    avgNormal   = Vec3::Zero();
            FLOAT32 totalArea   = 0.0f;
            FLOAT32 maxEdge     = 0.0f;
            FLOAT32 sumEdge     = 0.0f;
            FLOAT32 thinAspect  = 0.0f;
            UINT32  validFaces  = 0u;

            std::unordered_map<UINT64, UINT32> edgeCounts;
            edgeCounts.reserve(triCount * 3u);

            for (SIZE_T t = 0; t < triCount; ++t)
            {
                const UINT32 i0 = c.indices[t * 3u + 0u];
                const UINT32 i1 = c.indices[t * 3u + 1u];
                const UINT32 i2 = c.indices[t * 3u + 2u];
                const Vec3& p0 = c.vertices[i0].position;
                const Vec3& p1 = c.vertices[i1].position;
                const Vec3& p2 = c.vertices[i2].position;

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

                const Vec3 cr    = e01.Cross(p2 - p0);
                const FLOAT32 crL = cr.Length();
                if (crL > 1e-10f)
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
                if (s0 != s1) edgeCounts[MakeUndirectedEdgeKey(s0, s1)]++;
                if (s1 != s2) edgeCounts[MakeUndirectedEdgeKey(s1, s2)]++;
                if (s2 != s0) edgeCounts[MakeUndirectedEdgeKey(s2, s0)]++;
            }

            c.boundaryEdges.clear();
            for (const auto& kv : edgeCounts)
            {
                if (kv.second == 1u)
                {
                    c.boundaryEdges.push_back(kv.first);
                }
            }
            std::sort(c.boundaryEdges.begin(), c.boundaryEdges.end());

            c.maxEdgeLength = maxEdge;

            // Compute conservative bounding normal cone
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
                // Only enable cone culling when all triangle normals lie strictly within 80 degrees of cone axis
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

            // Compute thin-feature silhouette prominence metric [0 .. 4]
            const FLOAT32 avgEdge = (triCount > 0u) ? (sumEdge / static_cast<FLOAT32>(triCount * 3u)) : 0.0f;
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

        // Partition a triangle soup into compact clusters (<= targetTris, <= kNVGMaxClusterVertices)
        static void PartitionIntoClusters(
            const std::vector<Vertex>&        verts,
            const std::vector<UINT32>&        spatialIds,
            const std::vector<BuildTriangle>& tris,
            UINT32                            targetTris,
            UINT32                            lodLevel,
            std::vector<BuildCluster>&        outClusters)
        {
            if (tris.empty()) return;

            const UINT32 maxTris  = (std::min)(kNVGMaxClusterTriangles, (std::max)(16u, targetTris));
            const UINT32 maxVerts = kNVGMaxClusterVertices;
            const SIZE_T numTris  = tris.size();

            // Compute bounds for Morton sorting
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

            // Build triangle adjacency across spatial-welded edges
            std::unordered_map<UINT64, std::vector<UINT32>> edgeToTris;
            edgeToTris.reserve(numTris * 2u);
            for (UINT32 t = 0; t < static_cast<UINT32>(numTris); ++t)
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
                        edgeToTris[MakeUndirectedEdgeKey(a, b)].push_back(t);
                    }
                }
            }

            std::vector<std::vector<UINT32>> triAdj(numTris);
            for (const auto& kv : edgeToTris)
            {
                const auto& inc = kv.second;
                if (inc.size() < 2u || inc.size() > 8u) continue;
                for (SIZE_T i = 0; i < inc.size(); ++i)
                {
                    for (SIZE_T j = i + 1; j < inc.size(); ++j)
                    {
                        triAdj[inc[i]].push_back(inc[j]);
                        triAdj[inc[j]].push_back(inc[i]);
                    }
                }
            }

            // Order seed candidates by Morton code for tight spatial locality
            std::vector<UINT32> mortonOrder(numTris);
            std::vector<UINT32> mortonCodes(numTris);
            for (UINT32 t = 0; t < static_cast<UINT32>(numTris); ++t)
            {
                mortonOrder[t] = t;
                const Vec3 n = (triCentroids[t] - minB);
                mortonCodes[t] = Morton3D(n.x * invSpan.x, n.y * invSpan.y, n.z * invSpan.z);
            }
            std::sort(mortonOrder.begin(), mortonOrder.end(), [&](UINT32 a, UINT32 b) {
                if (mortonCodes[a] != mortonCodes[b]) return mortonCodes[a] < mortonCodes[b];
                return a < b;
            });

            std::vector<UINT8> assigned(numTris, 0u);
            std::vector<UINT32> vertexMap(verts.size(), kNVGInvalidIndex);
            std::vector<UINT32> touchedVerts;
            touchedVerts.reserve(maxVerts);
            std::vector<UINT32> frontier;
            std::vector<UINT8>  inFrontier(numTris, 0u);

            SIZE_T seedCursor = 0u;
            const FLOAT32 diag = (maxB - minB).Length();
            const FLOAT32 invDiag = (diag > 1e-6f) ? (1.0f / diag) : 1.0f;

            while (seedCursor < numTris)
            {
                while (seedCursor < numTris && assigned[mortonOrder[seedCursor]] != 0u)
                {
                    ++seedCursor;
                }
                if (seedCursor >= numTris) break;

                const UINT32 seedTri = mortonOrder[seedCursor];
                BuildCluster cluster{};
                cluster.lodLevel = lodLevel;
                cluster.vertices.reserve(maxVerts);
                cluster.spatialIds.reserve(maxVerts);
                cluster.indices.reserve(maxTris * 3u);

                touchedVerts.clear();
                frontier.clear();

                Vec3   clusterCentroidSum = Vec3::Zero();
                UINT32 clusterTriCount    = 0u;

                auto addTriToCluster = [&](UINT32 tIdx) {
                    assigned[tIdx] = 1u;
                    const BuildTriangle& bt = tris[tIdx];
                    for (UINT32 k = 0u; k < 3u; ++k)
                    {
                        const UINT32 gv = bt.v[k];
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
                        if (assigned[nb] == 0u && inFrontier[nb] == 0u)
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

                    // Evaluate frontier triangles (compact out already-assigned)
                    for (SIZE_T fi = 0; fi < frontier.size(); )
                    {
                        const UINT32 cand = frontier[fi];
                        if (assigned[cand] != 0u)
                        {
                            inFrontier[cand] = 0u;
                            frontier[fi] = frontier.back();
                            frontier.pop_back();
                            continue;
                        }

                        const BuildTriangle& bt = tris[cand];
                        UINT32 newVerts = 0u;
                        for (UINT32 k = 0u; k < 3u; ++k)
                        {
                            if (vertexMap[bt.v[k]] == kNVGInvalidIndex)
                            {
                                ++newVerts;
                            }
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
                            if (score < bestScore)
                            {
                                bestScore    = score;
                                bestTri      = cand;
                                bestFrontIdx = fi;
                            }
                        }
                        ++fi;
                    }

                    if (bestTri == kNVGInvalidIndex)
                    {
                        if (lodLevel == 0u)
                        {
                            // At LOD 0, only pull a spatially close Morton-ordered disconnected triangle into a tiny cluster
                            if (clusterTriCount < maxTris / 2u)
                            {
                                for (SIZE_T probe = seedCursor + 1u; probe < numTris && probe < seedCursor + 64u; ++probe)
                                {
                                    const UINT32 cand = mortonOrder[probe];
                                    if (assigned[cand] != 0u) continue;
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
                            // At coarse LODs, all input triangles belong to a single ClusterGroup; pack disconnected
                            // sub-islands tightly by nearest centroid so groups do not fragment into runt clusters.
                            FLOAT32 bestIslandDistSq = 1.0e30f;
                            for (SIZE_T probe = seedCursor + 1u; probe < numTris; ++probe)
                            {
                                const UINT32 cand = mortonOrder[probe];
                                if (assigned[cand] != 0u) continue;
                                UINT32 newVerts = 0u;
                                for (UINT32 k = 0u; k < 3u; ++k)
                                {
                                    if (vertexMap[tris[cand].v[k]] == kNVGInvalidIndex) ++newVerts;
                                }
                                if (cluster.vertices.size() + newVerts <= maxVerts)
                                {
                                    const FLOAT32 d2 = (triCentroids[cand] - clusterCenter).LengthSq();
                                    if (d2 < bestIslandDistSq)
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
                        inFrontier[bestTri] = 0u;
                        frontier[bestFrontIdx] = frontier.back();
                        frontier.pop_back();
                    }

                    addTriToCluster(bestTri);
                }

                for (UINT32 f : frontier) inFrontier[f] = 0u;
                for (UINT32 gv : touchedVerts) vertexMap[gv] = kNVGInvalidIndex;

                ComputeClusterBoundsAndMetrics(cluster);
                outClusters.push_back(std::move(cluster));
            }
        }

        // Garland-Heckbert Coupled-Wedge Position-Pair QEM Simplifier (1:1 Wedge Architecture):
        //   - Collapses 3D spatial edges (sFrom -> sTo) across all co-located attribute wedges in lockstep
        //     so UV/normal seams NEVER stall root convergence and NEVER open T-junction cracks.
        //   - Preserves distinct per-chart UV/normal/bone wedges at every shared spatialId (never mixes UVs
        //     across disjoint UV islands or wrap borders during LOD transitions).
        static FLOAT32 SimplifyClusterGroupQEM(
            const std::vector<Vertex>&        inVerts,
            const std::vector<UINT32>&        inSpatialIds,
            const std::vector<BuildTriangle>& inTris,
            const std::unordered_set<UINT32>& lockedSpatialIds,
            UINT32                            targetTriCount,
            const NVGBuildConfig&             config,
            BOOL                              hasSkinning,
            std::vector<Vertex>&              outVerts,
            std::vector<UINT32>&              outSpatialIds,
            std::vector<BuildTriangle>&       outTris,
            BOOL                              relaxAttributeSeams     = false,
            BOOL                              allowCoarseRidgeFlatten = false,
            BOOL                              allowRootNonPlanarFold  = false)
        {
            outVerts      = inVerts;
            outSpatialIds = inSpatialIds;
            outTris       = inTris;

            if (inTris.size() <= targetTriCount || inVerts.size() < 4u)
            {
                return 0.0f;
            }

            const UINT32 numVerts = static_cast<UINT32>(inVerts.size());
            const UINT32 numTris  = static_cast<UINT32>(inTris.size());

            // 1. Build compact local spatial-position table & per-position wedge lists
            std::unordered_map<UINT32, UINT32> globalToLocalSid;
            globalToLocalSid.reserve(numVerts);
            std::vector<UINT32>              localToGlobalSid;
            std::vector<Vec3>                spatialPos;
            std::vector<std::vector<UINT32>> spatialWedges;
            std::vector<UINT32>              vertLocalSid(numVerts, 0u);
            std::vector<UINT8>               vertAlive(numVerts, 1u);

            localToGlobalSid.reserve(numVerts);
            spatialPos.reserve(numVerts);
            spatialWedges.reserve(numVerts);

            for (UINT32 vi = 0u; vi < numVerts; ++vi)
            {
                const UINT32 gSid = inSpatialIds[vi];
                auto [it, inserted] = globalToLocalSid.emplace(gSid, static_cast<UINT32>(spatialPos.size()));
                const UINT32 lSid = it->second;
                if (inserted)
                {
                    localToGlobalSid.push_back(gSid);
                    spatialPos.push_back(inVerts[vi].position);
                    spatialWedges.emplace_back();
                }
                vertLocalSid[vi] = lSid;
                spatialWedges[lSid].push_back(vi);
            }

            const UINT32 numSpatial = static_cast<UINT32>(spatialPos.size());
            if (numSpatial < 3u)
            {
                return 0.0f;
            }

            // 2. Detect open-boundary spatial edges & exterior group locks
            std::unordered_map<UINT64, UINT32> spatialEdgeCount;
            spatialEdgeCount.reserve(numTris * 2u);
            for (const BuildTriangle& t : inTris)
            {
                const UINT32 s0 = vertLocalSid[t.v[0]];
                const UINT32 s1 = vertLocalSid[t.v[1]];
                const UINT32 s2 = vertLocalSid[t.v[2]];
                if (s0 != s1) spatialEdgeCount[MakeUndirectedEdgeKey(s0, s1)]++;
                if (s1 != s2) spatialEdgeCount[MakeUndirectedEdgeKey(s1, s2)]++;
                if (s2 != s0) spatialEdgeCount[MakeUndirectedEdgeKey(s2, s0)]++;
            }

            std::vector<UINT8>  spatialLocked(numSpatial, 0u);
            std::vector<UINT8>  spatialOpenBoundary(numSpatial, 0u);
            std::vector<UINT8>  spatialAlive(numSpatial, 1u);
            std::vector<UINT32> spatialVersion(numSpatial, 1u);

            for (UINT32 s = 0u; s < numSpatial; ++s)
            {
                if (lockedSpatialIds.find(localToGlobalSid[s]) != lockedSpatialIds.end())
                {
                    spatialLocked[s] = 1u;
                }
            }

            // Recenter coordinates in double precision around group centroid
            double cenX = 0.0, cenY = 0.0, cenZ = 0.0;
            for (UINT32 s = 0u; s < numSpatial; ++s)
            {
                cenX += static_cast<double>(spatialPos[s].x);
                cenY += static_cast<double>(spatialPos[s].y);
                cenZ += static_cast<double>(spatialPos[s].z);
            }
            const double invNumS = 1.0 / static_cast<double>(numSpatial);
            const Vec3 qemOrigin(
                static_cast<FLOAT32>(cenX * invNumS),
                static_cast<FLOAT32>(cenY * invNumS),
                static_cast<FLOAT32>(cenZ * invNumS));

            // 3. Accumulate geometric + open-boundary quadrics & weight sums per 3D spatial vertex
            std::vector<SymmetricQuadric>    spatialQuadrics(numSpatial);
            std::vector<double>              spatialWeightSum(numSpatial, 0.0);
            std::vector<std::vector<UINT32>> spatialTris(numSpatial);
            std::vector<UINT8>               triAlive(numTris, 1u);

            for (UINT32 ti = 0u; ti < numTris; ++ti)
            {
                const BuildTriangle& t = outTris[ti];
                const UINT32 s0 = vertLocalSid[t.v[0]];
                const UINT32 s1 = vertLocalSid[t.v[1]];
                const UINT32 s2 = vertLocalSid[t.v[2]];
                if (s0 == s1 || s1 == s2 || s2 == s0)
                {
                    triAlive[ti] = 0u;
                    continue;
                }

                spatialTris[s0].push_back(ti);
                spatialTris[s1].push_back(ti);
                spatialTris[s2].push_back(ti);

                const Vec3 p0 = spatialPos[s0] - qemOrigin;
                const Vec3 p1 = spatialPos[s1] - qemOrigin;
                const Vec3 p2 = spatialPos[s2] - qemOrigin;

                const Vec3    cr    = (p1 - p0).Cross(p2 - p0);
                const FLOAT32 crLen = cr.Length();
                if (crLen > 1e-10f)
                {
                    const Vec3   n = cr * (1.0f / crLen);
                    const double d = -(static_cast<double>(n.x) * p0.x +
                                       static_cast<double>(n.y) * p0.y +
                                       static_cast<double>(n.z) * p0.z);
                    const double areaWeight = (std::max)(1e-5, 0.5 * static_cast<double>(crLen));
                    spatialQuadrics[s0].AddPlane(n.x, n.y, n.z, d, areaWeight);
                    spatialQuadrics[s1].AddPlane(n.x, n.y, n.z, d, areaWeight);
                    spatialQuadrics[s2].AddPlane(n.x, n.y, n.z, d, areaWeight);
                    spatialWeightSum[s0] += areaWeight;
                    spatialWeightSum[s1] += areaWeight;
                    spatialWeightSum[s2] += areaWeight;

                    const UINT32 sTri[3] = { s0, s1, s2 };
                    for (UINT32 e = 0u; e < 3u; ++e)
                    {
                        const UINT32 sa  = sTri[e];
                        const UINT32 sb  = sTri[(e + 1u) % 3u];
                        const auto   itE = spatialEdgeCount.find(MakeUndirectedEdgeKey(sa, sb));
                        if (itE != spatialEdgeCount.end() && itE->second == 1u)
                        {
                            spatialOpenBoundary[sa] = 1u;
                            spatialOpenBoundary[sb] = 1u;

                            const Vec3    vaRel   = spatialPos[sa] - qemOrigin;
                            const Vec3    vbRel   = spatialPos[sb] - qemOrigin;
                            const Vec3    edgeVec = vbRel - vaRel;
                            const FLOAT32 edgeLen = edgeVec.Length();
                            if (edgeLen > 1e-8f)
                            {
                                const Vec3    bp    = edgeVec.Cross(n);
                                const FLOAT32 bpLen = bp.Length();
                                if (bpLen > 1e-8f)
                                {
                                    const Vec3   bn = bp * (1.0f / bpLen);
                                    const double bd = -(static_cast<double>(bn.x) * vaRel.x +
                                                        static_cast<double>(bn.y) * vaRel.y +
                                                        static_cast<double>(bn.z) * vaRel.z);
                                    const double boundaryScale = allowRootNonPlanarFold ? 2.0 : 8.0;
                                    const double bw = static_cast<double>(edgeLen * edgeLen) *
                                                      static_cast<double>((std::max)(1.0f, config.thinFeatureWeight)) * boundaryScale;
                                    spatialQuadrics[sa].AddPlane(bn.x, bn.y, bn.z, bd, bw);
                                    spatialQuadrics[sb].AddPlane(bn.x, bn.y, bn.z, bd, bw);
                                    spatialWeightSum[sa] += bw;
                                    spatialWeightSum[sb] += bw;
                                }
                            }
                        }
                    }
                }
            }

            // Helper: project a small 3D position shift onto a wedge's own incident UV triangle plane
            // so unpaired UV-chart wedges never swim or sample outside their own UV island.
            auto projectUnpairedWedgeUV = [&](UINT32 wedgeVi, const Vec3& oldPos, const Vec3& newPos) {
                const Vec3 dp = newPos - oldPos;
                if (dp.LengthSq() <= 1e-12f) return;

                const UINT32 lSid = vertLocalSid[wedgeVi];
                UINT32  bestTri  = kNVGInvalidIndex;
                FLOAT32 bestArea = 0.0f;
                for (UINT32 ti : spatialTris[lSid])
                {
                    if (!triAlive[ti]) continue;
                    const BuildTriangle& tr = outTris[ti];
                    if (tr.v[0] != wedgeVi && tr.v[1] != wedgeVi && tr.v[2] != wedgeVi) continue;
                    const Vec3 a = outVerts[tr.v[1]].position - outVerts[tr.v[0]].position;
                    const Vec3 b = outVerts[tr.v[2]].position - outVerts[tr.v[0]].position;
                    const FLOAT32 area2 = a.Cross(b).LengthSq();
                    if (area2 > bestArea)
                    {
                        bestArea = area2;
                        bestTri  = ti;
                    }
                }
                if (bestTri == kNVGInvalidIndex || bestArea <= 1e-12f) return;

                const BuildTriangle& tr = outTris[bestTri];
                UINT32 i0 = tr.v[0], i1 = tr.v[1], i2 = tr.v[2];
                if (i1 == wedgeVi)      { i0 = tr.v[1]; i1 = tr.v[2]; i2 = tr.v[0]; }
                else if (i2 == wedgeVi) { i0 = tr.v[2]; i1 = tr.v[0]; i2 = tr.v[1]; }

                const Vec3 e1 = outVerts[i1].position - oldPos;
                const Vec3 e2 = outVerts[i2].position - oldPos;
                const double g11 = static_cast<double>(e1.Dot(e1));
                const double g12 = static_cast<double>(e1.Dot(e2));
                const double g22 = static_cast<double>(e2.Dot(e2));
                const double det = g11 * g22 - g12 * g12;
                if (det <= 1e-10 * g11 * g22 || det <= 1e-18) return;

                const double b1 = static_cast<double>(dp.Dot(e1));
                const double b2 = static_cast<double>(dp.Dot(e2));
                FLOAT32 alpha = (std::max)(0.0f, static_cast<FLOAT32>((b1 * g22 - b2 * g12) / det));
                FLOAT32 beta  = (std::max)(0.0f, static_cast<FLOAT32>((g11 * b2 - g12 * b1) / det));
                const FLOAT32 abSum = alpha + beta;
                if (abSum > 1.0f)
                {
                    const FLOAT32 invSum = 1.0f / abSum;
                    alpha *= invSum;
                    beta  *= invSum;
                }

                const Vec2 duv0_1 = outVerts[i1].texCoord0 - outVerts[i0].texCoord0;
                const Vec2 duv0_2 = outVerts[i2].texCoord0 - outVerts[i0].texCoord0;
                outVerts[wedgeVi].texCoord0 = outVerts[i0].texCoord0 + duv0_1 * alpha + duv0_2 * beta;

                const Vec2 duv1_1 = outVerts[i1].texCoord1 - outVerts[i0].texCoord1;
                const Vec2 duv1_2 = outVerts[i2].texCoord1 - outVerts[i0].texCoord1;
                outVerts[wedgeVi].texCoord1 = outVerts[i0].texCoord1 + duv1_1 * alpha + duv1_2 * beta;
            };

            struct SpatialCollapseCandidate
            {
                double  cost    = 0.0;
                UINT32  sFrom   = 0u;
                UINT32  sTo     = 0u;
                UINT32  verFrom = 0u;
                UINT32  verTo   = 0u;
                Vec3    newPos  = Vec3::Zero();
                FLOAT32 interpT = 1.0f;

                bool operator>(const SpatialCollapseCandidate& o) const noexcept
                {
                    return cost > o.cost;
                }
            };

            std::priority_queue<
                SpatialCollapseCandidate,
                std::vector<SpatialCollapseCandidate>,
                std::greater<SpatialCollapseCandidate>> heap;

            auto evaluateSpatialEdge = [&](UINT32 sA, UINT32 sB, SpatialCollapseCandidate& outCand) -> BOOL {
                if (sA == sB || !spatialAlive[sA] || !spatialAlive[sB]) return false;
                if (spatialLocked[sA] && spatialLocked[sB]) return false;

                UINT32 sFrom = sA;
                UINT32 sTo   = sB;
                BOOL   pinTargetPos = false;

                const BOOL seamA = (spatialWedges[sA].size() > 1u);
                const BOOL seamB = (spatialWedges[sB].size() > 1u);
                const BOOL openA = (spatialOpenBoundary[sA] != 0u);
                const BOOL openB = (spatialOpenBoundary[sB] != 0u);

                if (spatialLocked[sA])
                {
                    sFrom        = sB;
                    sTo          = sA;
                    pinTargetPos = true;
                }
                else if (spatialLocked[sB])
                {
                    sFrom        = sA;
                    sTo          = sB;
                    pinTargetPos = true;
                }
                else if (openA != openB)
                {
                    // Collapse interior vertex onto open boundary without shrinking the silhouette border
                    sFrom        = openA ? sB : sA;
                    sTo          = openA ? sA : sB;
                    pinTargetPos = !allowCoarseRidgeFlatten;
                }
                else if (seamA != seamB)
                {
                    // Collapse single-chart interior vertex onto multi-chart UV/normal seam so the seam curve stays stationary
                    sFrom        = seamA ? sB : sA;
                    sTo          = seamA ? sA : sB;
                    pinTargetPos = !relaxAttributeSeams;
                }

                // Inspect wing triangles sharing (sFrom, sTo) to measure within-chart attribute continuity & unpaired wedges
                FLOAT32 maxWingNormalDiff = 0.0f;
                FLOAT32 maxWingUvDiff     = 0.0f;
                BOOL    skinMismatch      = false;
                UINT32  pairedFromCount   = 0u;

                for (UINT32 ti : spatialTris[sFrom])
                {
                    if (!triAlive[ti]) continue;
                    const BuildTriangle& tr = outTris[ti];
                    UINT32 vF = kNVGInvalidIndex;
                    UINT32 vT = kNVGInvalidIndex;
                    for (UINT32 k = 0u; k < 3u; ++k)
                    {
                        const UINT32 ls = vertLocalSid[tr.v[k]];
                        if (ls == sFrom) vF = tr.v[k];
                        else if (ls == sTo) vT = tr.v[k];
                    }
                    if (vF != kNVGInvalidIndex && vT != kNVGInvalidIndex)
                    {
                        ++pairedFromCount;
                        const FLOAT32 nDot = ClampF(outVerts[vF].normal.Dot(outVerts[vT].normal), -1.0f, 1.0f);
                        maxWingNormalDiff  = (std::max)(maxWingNormalDiff, 1.0f - nDot);
                        maxWingUvDiff      = (std::max)(maxWingUvDiff, (outVerts[vF].texCoord0 - outVerts[vT].texCoord0).Length());
                        if (hasSkinning && outVerts[vF].boneIndices[0] != outVerts[vT].boneIndices[0])
                        {
                            skinMismatch = true;
                        }
                    }
                }

                if (pairedFromCount == 0u)
                {
                    return false;
                }

                const SymmetricQuadric qSum = spatialQuadrics[sFrom] + spatialQuadrics[sTo];
                const Vec3& pFrom = spatialPos[sFrom];
                const Vec3& pTo   = spatialPos[sTo];
                Vec3    bestPos   = pTo;
                FLOAT32 bestT     = 1.0f;
                double  bestCost  = qSum.Evaluate(bestPos - qemOrigin);

                if (!pinTargetPos && !spatialLocked[sTo])
                {
                    const Vec3   pMid  = (pFrom + pTo) * 0.5f;
                    const double cFrom = qSum.Evaluate(pFrom - qemOrigin);
                    const double cMid  = qSum.Evaluate(pMid - qemOrigin);
                    if (cFrom < bestCost) { bestCost = cFrom; bestPos = pFrom; bestT = 0.0f; }
                    if (cMid  < bestCost) { bestCost = cMid;  bestPos = pMid;  bestT = 0.5f; }

                    // Only use unconstrained 3D quadric optimum when neither endpoint is a multi-chart UV seam
                    // (along a UV seam, restricting bestPos to the segment [pFrom, pTo] keeps both UV charts linear!)
                    if (!seamA && !seamB)
                    {
                        Vec3 pOptRel{};
                        if (qSum.SolveOptimalPosition(pOptRel))
                        {
                            const Vec3    pOpt    = pOptRel + qemOrigin;
                            const FLOAT32 edgeLen = (pTo - pFrom).Length();
                            if ((pOpt - pMid).Length() <= edgeLen * 1.25f + 1e-4f)
                            {
                                const double cOpt = qSum.Evaluate(pOptRel);
                                if (cOpt < bestCost)
                                {
                                    bestCost = cOpt;
                                    bestPos  = pOpt;
                                    const Vec3    seg      = pTo - pFrom;
                                    const FLOAT32 segLenSq = seg.LengthSq();
                                    bestT = (segLenSq > 1e-10f) ? ClampF((pOpt - pFrom).Dot(seg) / segLenSq, 0.0f, 1.0f) : 0.5f;
                                }
                            }
                        }
                    }
                }

                const double attrPenaltyScale = allowCoarseRidgeFlatten ? 0.35 : 1.0;
                double penalty = 1.0;
                if (maxWingNormalDiff > 0.15f)
                {
                    penalty += static_cast<double>(maxWingNormalDiff) * static_cast<double>(config.normalPreservationWeight) * 4.0 * attrPenaltyScale;
                }
                if (openA || openB)
                {
                    penalty += static_cast<double>(config.thinFeatureWeight) * 2.0 * attrPenaltyScale;
                }
                if (seamA || seamB)
                {
                    penalty += static_cast<double>(config.uvSeamWeight) * (0.5 + static_cast<double>(maxWingUvDiff)) * attrPenaltyScale;
                    // Extra preference against growing wedge fan count unless interior edges are exhausted
                    if (spatialWedges[sFrom].size() > pairedFromCount)
                    {
                        penalty += 2.0 * attrPenaltyScale;
                    }
                }
                if (skinMismatch)
                {
                    penalty += 4.0;
                }

                const FLOAT32 edgeLen = (pTo - pFrom).Length();
                const double  geomErr2 = (std::max)(0.0, bestCost) + static_cast<double>(edgeLen * edgeLen) * 1e-5;

                outCand.cost    = geomErr2 * penalty;
                outCand.sFrom   = sFrom;
                outCand.sTo     = sTo;
                outCand.verFrom = spatialVersion[sFrom];
                outCand.verTo   = spatialVersion[sTo];
                outCand.newPos  = bestPos;
                outCand.interpT = bestT;
                return true;
            };

            // 4. Seed initial 3D spatial edge candidates
            std::unordered_set<UINT64> seededSpatialEdges;
            seededSpatialEdges.reserve(numTris * 2u);
            for (UINT32 ti = 0u; ti < numTris; ++ti)
            {
                if (!triAlive[ti]) continue;
                const BuildTriangle& t = outTris[ti];
                const UINT32 sTri[3] = {
                    vertLocalSid[t.v[0]],
                    vertLocalSid[t.v[1]],
                    vertLocalSid[t.v[2]]
                };
                for (UINT32 e = 0u; e < 3u; ++e)
                {
                    const UINT32 a = sTri[e];
                    const UINT32 b = sTri[(e + 1u) % 3u];
                    if (a == b) continue;
                    if (seededSpatialEdges.insert(MakeUndirectedEdgeKey(a, b)).second)
                    {
                        SpatialCollapseCandidate cand{};
                        if (evaluateSpatialEdge(a, b, cand))
                        {
                            heap.push(cand);
                        }
                    }
                }
            }

            UINT32 aliveTriCount = 0u;
            for (UINT32 ti = 0u; ti < numTris; ++ti)
            {
                if (triAlive[ti]) ++aliveTriCount;
            }
            FLOAT32 maxGeomError   = 0.0f;
            UINT32  collapseBudget = numSpatial * 3u;

            // Strict triangle orientation preservation (cosAngle >= 0 on normal/coarse passes)
            const FLOAT32 minNormalCos = allowRootNonPlanarFold
                ? -0.35f
                : (allowCoarseRidgeFlatten ? 0.0f : 0.05f);

            while (aliveTriCount > targetTriCount && !heap.empty() && collapseBudget > 0u)
            {
                const SpatialCollapseCandidate cand = heap.top();
                heap.pop();

                if (!spatialAlive[cand.sFrom] || !spatialAlive[cand.sTo]) continue;
                if (spatialVersion[cand.sFrom] != cand.verFrom || spatialVersion[cand.sTo] != cand.verTo) continue;
                if (spatialLocked[cand.sFrom]) continue;

                // Prevent local wedge fan explosion past cluster vertex capacity
                if (spatialWedges[cand.sFrom].size() + spatialWedges[cand.sTo].size() > 24u) continue;

                // Check triangle normal inversion across all non-collapsing incident faces
                BOOL flipsNormal = false;
                auto checkSpatialFlips = [&](UINT32 checkS) {
                    for (UINT32 ti : spatialTris[checkS])
                    {
                        if (!triAlive[ti]) continue;
                        const BuildTriangle& t = outTris[ti];
                        const UINT32 ls0 = vertLocalSid[t.v[0]];
                        const UINT32 ls1 = vertLocalSid[t.v[1]];
                        const UINT32 ls2 = vertLocalSid[t.v[2]];
                        const BOOL hasFrom = (ls0 == cand.sFrom || ls1 == cand.sFrom || ls2 == cand.sFrom);
                        const BOOL hasTo   = (ls0 == cand.sTo   || ls1 == cand.sTo   || ls2 == cand.sTo);
                        if (hasFrom && hasTo) continue; // Wing triangle collapses away cleanly

                        const Vec3 oldP[3] = { spatialPos[ls0], spatialPos[ls1], spatialPos[ls2] };
                        Vec3 newP[3] = { oldP[0], oldP[1], oldP[2] };
                        if (ls0 == cand.sFrom || ls0 == cand.sTo) newP[0] = cand.newPos;
                        if (ls1 == cand.sFrom || ls1 == cand.sTo) newP[1] = cand.newPos;
                        if (ls2 == cand.sFrom || ls2 == cand.sTo) newP[2] = cand.newPos;

                        const Vec3    oldN   = (oldP[1] - oldP[0]).Cross(oldP[2] - oldP[0]);
                        const Vec3    newN   = (newP[1] - newP[0]).Cross(newP[2] - newP[0]);
                        const FLOAT32 oldLen = oldN.Length();
                        const FLOAT32 newLen = newN.Length();
                        if (oldLen > 1e-10f)
                        {
                            if (newLen <= 1e-10f || (oldN.Dot(newN) / (oldLen * newLen)) < minNormalCos)
                            {
                                flipsNormal = true;
                                return;
                            }
                        }
                    }
                };
                checkSpatialFlips(cand.sFrom);
                if (!flipsNormal) checkSpatialFlips(cand.sTo);
                if (flipsNormal) continue;

                // Apply coupled-wedge spatial collapse: sFrom -> sTo at cand.newPos.
                // Measure exact scale-consistent RMS perpendicular distance in meters to BOTH endpoints' accumulated
                // surface & open-boundary planes (zero flat-plane path-length inflation, zero small-spike area dilution!).
                --collapseBudget;
                const Vec3   oldPosFrom = spatialPos[cand.sFrom];
                const Vec3   oldPosTo   = spatialPos[cand.sTo];
                const Vec3   relNew     = cand.newPos - qemOrigin;
                const double errFromM   = (spatialWeightSum[cand.sFrom] > 1e-12)
                    ? std::sqrt((std::max)(0.0, spatialQuadrics[cand.sFrom].Evaluate(relNew) / spatialWeightSum[cand.sFrom]))
                    : 0.0;
                const double errToM     = (spatialWeightSum[cand.sTo] > 1e-12)
                    ? std::sqrt((std::max)(0.0, spatialQuadrics[cand.sTo].Evaluate(relNew) / spatialWeightSum[cand.sTo]))
                    : 0.0;
                const FLOAT32 minHop    = (std::min)((cand.newPos - oldPosFrom).Length(), (cand.newPos - oldPosTo).Length());
                const FLOAT32 stepErr   = (std::max)(
                    static_cast<FLOAT32>((std::max)(errFromM, errToM) * 1.35),
                    minHop * 0.25f);
                if (stepErr > maxGeomError) maxGeomError = stepErr;

                // Bijectively pair wedges (1-to-1) strictly within identical UV/attribute charts via wing triangles sharing (sFrom, sTo).
                // Enforcing 1-to-1 pairing guarantees UV darts (where 2 seam wedges at sFrom touch 1 tip wedge at sTo) never merge disjoint UVs!
                const std::vector<UINT32>& wedgesFrom = spatialWedges[cand.sFrom];
                const std::vector<UINT32>& wedgesTo   = spatialWedges[cand.sTo];
                std::vector<UINT32> partnerAtTo(wedgesFrom.size(), kNVGInvalidIndex);

                for (UINT32 vT : wedgesTo)
                {
                    if (!vertAlive[vT]) continue;
                    SIZE_T  bestFi   = wedgesFrom.size();
                    FLOAT32 bestDist = 1e30f;
                    for (SIZE_T fi = 0u; fi < wedgesFrom.size(); ++fi)
                    {
                        if (partnerAtTo[fi] != kNVGInvalidIndex) continue;
                        const UINT32 vF = wedgesFrom[fi];
                        if (!vertAlive[vF]) continue;
                        if (hasSkinning && outVerts[vF].boneIndices[0] != outVerts[vT].boneIndices[0]) continue;

                        BOOL sharesWingTri = false;
                        for (UINT32 ti : spatialTris[cand.sFrom])
                        {
                            if (!triAlive[ti]) continue;
                            const BuildTriangle& tr = outTris[ti];
                            const BOOL hasVF = (tr.v[0] == vF || tr.v[1] == vF || tr.v[2] == vF);
                            const BOOL hasVT = (tr.v[0] == vT || tr.v[1] == vT || tr.v[2] == vT);
                            if (hasVF && hasVT)
                            {
                                sharesWingTri = true;
                                break;
                            }
                        }
                        if (sharesWingTri)
                        {
                            const FLOAT32 uvDist = (outVerts[vF].texCoord0 - outVerts[vT].texCoord0).LengthSq() +
                                                   (outVerts[vF].texCoord1 - outVerts[vT].texCoord1).LengthSq() +
                                                   (1.0f - ClampF(outVerts[vF].normal.Dot(outVerts[vT].normal), -1.0f, 1.0f)) * 0.05f;
                            if (uvDist < bestDist)
                            {
                                bestDist = uvDist;
                                bestFi   = fi;
                            }
                        }
                    }
                    if (bestFi < wedgesFrom.size())
                    {
                        partnerAtTo[bestFi] = vT;
                    }
                }

                const FLOAT32 tLerp = cand.interpT;
                const FLOAT32 invT  = 1.0f - tLerp;
                const UINT32  newGlobalSid = localToGlobalSid[cand.sTo];

                // Update existing wedges at sTo (paired wedges interpolate within their own UV chart; unpaired project locally)
                for (UINT32 vT : wedgesTo)
                {
                    if (!vertAlive[vT]) continue;
                    UINT32 matchedVF = kNVGInvalidIndex;
                    for (SIZE_T fi = 0u; fi < wedgesFrom.size(); ++fi)
                    {
                        if (partnerAtTo[fi] == vT && vertAlive[wedgesFrom[fi]])
                        {
                            matchedVF = wedgesFrom[fi];
                            break;
                        }
                    }

                    if (matchedVF != kNVGInvalidIndex)
                    {
                        const Vertex vA = outVerts[matchedVF];
                        const Vertex vB = outVerts[vT];
                        Vertex merged   = vB;
                        merged.position  = cand.newPos;
                        merged.normal    = (vA.normal * invT + vB.normal * tLerp).Normalized();
                        merged.tangent   = (vA.tangent * invT + vB.tangent * tLerp).Normalized();
                        merged.bitangent = (vA.bitangent * invT + vB.bitangent * tLerp).Normalized();
                        if (merged.normal.LengthSq() < 1e-6f)    merged.normal    = vB.normal;
                        if (merged.tangent.LengthSq() < 1e-6f)   merged.tangent   = vB.tangent;
                        if (merged.bitangent.LengthSq() < 1e-6f) merged.bitangent = vB.bitangent;
                        merged.texCoord0 = vA.texCoord0 * invT + vB.texCoord0 * tLerp;
                        merged.texCoord1 = vA.texCoord1 * invT + vB.texCoord1 * tLerp;
                        merged.color     = vA.color * invT + vB.color * tLerp;
                        if (hasSkinning)
                        {
                            if (tLerp < 0.5f)
                            {
                                for (UINT32 b = 0u; b < 4u; ++b) merged.boneIndices[b] = vA.boneIndices[b];
                                merged.boneWeights = vA.boneWeights;
                            }
                            const FLOAT32 wSum = merged.boneWeights.x + merged.boneWeights.y + merged.boneWeights.z + merged.boneWeights.w;
                            if (wSum > 1e-5f) merged.boneWeights = merged.boneWeights * (1.0f / wSum);
                        }
                        outVerts[vT] = merged;
                    }
                    else
                    {
                        projectUnpairedWedgeUV(vT, oldPosTo, cand.newPos);
                        outVerts[vT].position = cand.newPos;
                    }
                    outSpatialIds[vT] = newGlobalSid;
                }

                // Move or merge wedges from sFrom into sTo (NEVER merging across different UV charts!)
                std::unordered_map<UINT32, UINT32> wedgeRemap;
                for (SIZE_T fi = 0u; fi < wedgesFrom.size(); ++fi)
                {
                    const UINT32 vF = wedgesFrom[fi];
                    if (!vertAlive[vF]) continue;
                    const UINT32 vT = partnerAtTo[fi];
                    if (vT != kNVGInvalidIndex && vertAlive[vT])
                    {
                        wedgeRemap[vF] = vT;
                        vertAlive[vF]  = 0u;
                    }
                    else
                    {
                        // Keep distinct UV-island wedge alive at sTo with its own chart UVs!
                        projectUnpairedWedgeUV(vF, oldPosFrom, cand.newPos);
                        outVerts[vF].position = cand.newPos;
                        outSpatialIds[vF]     = newGlobalSid;
                        vertLocalSid[vF]      = cand.sTo;
                        spatialWedges[cand.sTo].push_back(vF);
                    }
                }
                spatialWedges[cand.sFrom].clear();

                // Update spatial vertex state & incident triangles
                spatialPos[cand.sTo]       = cand.newPos;
                spatialQuadrics[cand.sTo]  = spatialQuadrics[cand.sFrom] + spatialQuadrics[cand.sTo];
                spatialWeightSum[cand.sTo] += spatialWeightSum[cand.sFrom];
                spatialAlive[cand.sFrom]   = 0u;
                spatialVersion[cand.sTo]++;
                if (spatialOpenBoundary[cand.sFrom]) spatialOpenBoundary[cand.sTo] = 1u;

                for (UINT32 ti : spatialTris[cand.sFrom])
                {
                    if (!triAlive[ti]) continue;
                    BuildTriangle& tr = outTris[ti];
                    for (UINT32 k = 0u; k < 3u; ++k)
                    {
                        auto itR = wedgeRemap.find(tr.v[k]);
                        if (itR != wedgeRemap.end()) tr.v[k] = itR->second;
                    }
                    const UINT32 ls0 = vertLocalSid[tr.v[0]];
                    const UINT32 ls1 = vertLocalSid[tr.v[1]];
                    const UINT32 ls2 = vertLocalSid[tr.v[2]];
                    if (ls0 == ls1 || ls1 == ls2 || ls2 == ls0)
                    {
                        triAlive[ti] = 0u;
                        if (aliveTriCount > 0u) --aliveTriCount;
                    }
                    else
                    {
                        spatialTris[cand.sTo].push_back(ti);
                    }
                }
                spatialTris[cand.sFrom].clear();

                // Deduplicate spatialTris[cand.sTo] & re-queue neighbor spatial edges
                std::unordered_set<UINT32> uniqueTris;
                uniqueTris.reserve(spatialTris[cand.sTo].size());
                std::vector<UINT32> cleanToTris;
                cleanToTris.reserve(spatialTris[cand.sTo].size());
                std::unordered_set<UINT32> spatialNeighbors;

                for (UINT32 ti : spatialTris[cand.sTo])
                {
                    if (!triAlive[ti]) continue;
                    const BuildTriangle& tr = outTris[ti];
                    const UINT32 ls0 = vertLocalSid[tr.v[0]];
                    const UINT32 ls1 = vertLocalSid[tr.v[1]];
                    const UINT32 ls2 = vertLocalSid[tr.v[2]];
                    if (ls0 == ls1 || ls1 == ls2 || ls2 == ls0)
                    {
                        triAlive[ti] = 0u;
                        if (aliveTriCount > 0u) --aliveTriCount;
                        continue;
                    }
                    if (uniqueTris.insert(ti).second)
                    {
                        cleanToTris.push_back(ti);
                        if (ls0 != cand.sTo && spatialAlive[ls0]) spatialNeighbors.insert(ls0);
                        if (ls1 != cand.sTo && spatialAlive[ls1]) spatialNeighbors.insert(ls1);
                        if (ls2 != cand.sTo && spatialAlive[ls2]) spatialNeighbors.insert(ls2);
                    }
                }
                spatialTris[cand.sTo] = std::move(cleanToTris);

                for (UINT32 nb : spatialNeighbors)
                {
                    SpatialCollapseCandidate newCand{};
                    if (evaluateSpatialEdge(cand.sTo, nb, newCand))
                    {
                        heap.push(newCand);
                    }
                }
            }

            // 5. Compact surviving vertices & triangles
            std::vector<UINT32>        remap(numVerts, kNVGInvalidIndex);
            std::vector<Vertex>        compactedVerts;
            std::vector<UINT32>        compactedSpatial;
            std::vector<BuildTriangle> compactedTris;
            compactedVerts.reserve(numVerts);
            compactedSpatial.reserve(numVerts);
            compactedTris.reserve(aliveTriCount);

            for (UINT32 ti = 0u; ti < numTris; ++ti)
            {
                if (!triAlive[ti]) continue;
                BuildTriangle tr = outTris[ti];
                if (vertLocalSid[tr.v[0]] == vertLocalSid[tr.v[1]] ||
                    vertLocalSid[tr.v[1]] == vertLocalSid[tr.v[2]] ||
                    vertLocalSid[tr.v[2]] == vertLocalSid[tr.v[0]])
                {
                    continue;
                }

                for (UINT32 k = 0u; k < 3u; ++k)
                {
                    const UINT32 oldV = tr.v[k];
                    if (remap[oldV] == kNVGInvalidIndex)
                    {
                        remap[oldV] = static_cast<UINT32>(compactedVerts.size());
                        compactedVerts.push_back(outVerts[oldV]);
                        compactedSpatial.push_back(outSpatialIds[oldV]);
                    }
                    tr.v[k] = remap[oldV];
                }
                compactedTris.push_back(tr);
            }

            outVerts      = std::move(compactedVerts);
            outSpatialIds = std::move(compactedSpatial);
            outTris       = std::move(compactedTris);
            return maxGeomError;
        }

        // Quantized compression encoder for a single cluster on a mesh-wide power-of-two grid.
        // Snapping all clusters across all LODs onto a shared power-of-two step (quantStep = 2^k)
        // guarantees that posOrigin + q * posStep = G * 2^k is bit-for-bit identical in IEEE-754
        // float32 across every shared cluster and LOD-cut boundary edge (zero sub-texel cracks).
        static FLOAT32 EncodeClusterToWords(
            BuildCluster& c,
            double        quantStep,
            BOOL          hasSkinning)
        {
            const UINT32 vertCount = static_cast<UINT32>(c.vertices.size());
            const UINT32 triCount  = static_cast<UINT32>(c.indices.size() / 3u);

            std::vector<INT64> gx(vertCount, 0);
            std::vector<INT64> gy(vertCount, 0);
            std::vector<INT64> gz(vertCount, 0);
            INT64 gMinX = (std::numeric_limits<INT64>::max)();
            INT64 gMinY = (std::numeric_limits<INT64>::max)();
            INT64 gMinZ = (std::numeric_limits<INT64>::max)();

            for (UINT32 vi = 0u; vi < vertCount; ++vi)
            {
                const Vec3& p = c.vertices[vi].position;
                gx[vi] = static_cast<INT64>(std::llround(static_cast<double>(p.x) / quantStep));
                gy[vi] = static_cast<INT64>(std::llround(static_cast<double>(p.y) / quantStep));
                gz[vi] = static_cast<INT64>(std::llround(static_cast<double>(p.z) / quantStep));
                if (gx[vi] < gMinX) gMinX = gx[vi];
                if (gy[vi] < gMinY) gMinY = gy[vi];
                if (gz[vi] < gMinZ) gMinZ = gz[vi];
            }

            const FLOAT32 stepF = static_cast<FLOAT32>(quantStep);
            const Vec3 posMin(
                static_cast<FLOAT32>(static_cast<double>(gMinX) * quantStep),
                static_cast<FLOAT32>(static_cast<double>(gMinY) * quantStep),
                static_cast<FLOAT32>(static_cast<double>(gMinZ) * quantStep));
            const Vec3 posExt(
                stepF * 65535.0f,
                stepF * 65535.0f,
                stepF * 65535.0f);

            Vec2 uv0Min = c.vertices[0].texCoord0;
            Vec2 uv0Max = uv0Min;
            Vec2 uv1Min = c.vertices[0].texCoord1;
            Vec2 uv1Max = uv1Min;
            BOOL hasUv1   = false;
            BOOL hasColor = false;

            for (const Vertex& v : c.vertices)
            {
                if (v.texCoord0.x < uv0Min.x) uv0Min.x = v.texCoord0.x;
                if (v.texCoord0.y < uv0Min.y) uv0Min.y = v.texCoord0.y;
                if (v.texCoord0.x > uv0Max.x) uv0Max.x = v.texCoord0.x;
                if (v.texCoord0.y > uv0Max.y) uv0Max.y = v.texCoord0.y;

                if (v.texCoord1.x < uv1Min.x) uv1Min.x = v.texCoord1.x;
                if (v.texCoord1.y < uv1Min.y) uv1Min.y = v.texCoord1.y;
                if (v.texCoord1.x > uv1Max.x) uv1Max.x = v.texCoord1.x;
                if (v.texCoord1.y > uv1Max.y) uv1Max.y = v.texCoord1.y;

                if (std::fabs(v.texCoord1.x) > 1e-5f || std::fabs(v.texCoord1.y) > 1e-5f) hasUv1 = true;
                if (std::fabs(v.color.x - 1.0f) > 1e-4f || std::fabs(v.color.y - 1.0f) > 1e-4f ||
                    std::fabs(v.color.z - 1.0f) > 1e-4f || std::fabs(v.color.w - 1.0f) > 1e-4f)
                {
                    hasColor = true;
                }
            }

            Vec2 uv0Ext = uv0Max - uv0Min;
            if (uv0Ext.x < 1e-6f) uv0Ext.x = 1.0f;
            if (uv0Ext.y < 1e-6f) uv0Ext.y = 1.0f;
            Vec2 uv1Ext = uv1Max - uv1Min;
            if (uv1Ext.x < 1e-6f) uv1Ext.x = 1.0f;
            if (uv1Ext.y < 1e-6f) uv1Ext.y = 1.0f;

            UINT32 attrFlags = NVG_CLUSTER_ATTR_TANGENT;
            if (hasUv1)      attrFlags |= NVG_CLUSTER_ATTR_UV1;
            if (hasColor)    attrFlags |= NVG_CLUSTER_ATTR_COLOR;
            if (hasSkinning) attrFlags |= NVG_CLUSTER_ATTR_SKINNING;

            const UINT32 vertStrideWords = hasSkinning ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;
            const UINT32 vertRelOffset   = kNVGClusterHeaderWords;
            const UINT32 idxRelOffset    = vertRelOffset + vertCount * vertStrideWords;
            const UINT32 totalWords      = idxRelOffset + triCount;

            c.encodedWords.assign(totalWords, 0u);
            UINT32* w = c.encodedWords.data();

            // 16-word Header (64 bytes)
            w[0]  = BitCastFloatToUInt(posMin.x);
            w[1]  = BitCastFloatToUInt(posMin.y);
            w[2]  = BitCastFloatToUInt(posMin.z);
            w[3]  = BitCastFloatToUInt(posExt.x);
            w[4]  = BitCastFloatToUInt(posExt.y);
            w[5]  = BitCastFloatToUInt(posExt.z);
            w[6]  = BitCastFloatToUInt(uv0Min.x);
            w[7]  = BitCastFloatToUInt(uv0Min.y);
            w[8]  = BitCastFloatToUInt(uv0Ext.x);
            w[9]  = BitCastFloatToUInt(uv0Ext.y);
            w[10] = BitCastFloatToUInt(uv1Min.x);
            w[11] = BitCastFloatToUInt(uv1Min.y);
            w[12] = BitCastFloatToUInt(uv1Ext.x);
            w[13] = BitCastFloatToUInt(uv1Ext.y);
            w[14] = PackNVGClusterCountsAndLod(vertCount, triCount, c.lodLevel, attrFlags);
            w[15] = (vertRelOffset & 0xFFFFu) | ((idxRelOffset & 0xFFFFu) << 16u);

            FLOAT32 maxPosErr = 0.0f;
            for (UINT32 vi = 0u; vi < vertCount; ++vi)
            {
                const Vertex& v = c.vertices[vi];
                const UINT32 qx = static_cast<UINT32>((std::clamp<INT64>)(gx[vi] - gMinX, 0, 65535)) & 0xFFFFu;
                const UINT32 qy = static_cast<UINT32>((std::clamp<INT64>)(gy[vi] - gMinY, 0, 65535)) & 0xFFFFu;
                const UINT32 qz = static_cast<UINT32>((std::clamp<INT64>)(gz[vi] - gMinZ, 0, 65535)) & 0xFFFFu;

                const Vec3 reconPos(
                    posMin.x + static_cast<FLOAT32>(qx) * stepF,
                    posMin.y + static_cast<FLOAT32>(qy) * stepF,
                    posMin.z + static_cast<FLOAT32>(qz) * stepF);
                if (reconPos.x < c.aabb.m_Min.x) c.aabb.m_Min.x = reconPos.x;
                if (reconPos.y < c.aabb.m_Min.y) c.aabb.m_Min.y = reconPos.y;
                if (reconPos.z < c.aabb.m_Min.z) c.aabb.m_Min.z = reconPos.z;
                if (reconPos.x > c.aabb.m_Max.x) c.aabb.m_Max.x = reconPos.x;
                if (reconPos.y > c.aabb.m_Max.y) c.aabb.m_Max.y = reconPos.y;
                if (reconPos.z > c.aabb.m_Max.z) c.aabb.m_Max.z = reconPos.z;
                const FLOAT32 err = (reconPos - v.position).Length();
                if (err > maxPosErr) maxPosErr = err;

                Vec3 unitN = v.normal.LengthSq() > 1e-8f ? v.normal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
                Vec3 unitT = v.tangent.LengthSq() > 1e-8f ? v.tangent.Normalized() : Vec3(1.0f, 0.0f, 0.0f);
                Vec3 unitB = v.bitangent.LengthSq() > 1e-8f ? v.bitangent.Normalized() : unitN.Cross(unitT);
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

            return maxPosErr;
        }

        // Top-down compact BVH over ClusterGroups (preserves contiguous per-group cluster ranges while keeping leaf size <= 8)
        static void BuildGroupBVHRecursive(
            UINT32                                    nodeIdx,
            std::vector<UINT32>&                      groupOrder,
            UINT32                                    rangeBegin,
            UINT32                                    rangeEnd,
            UINT32                                    depth,
            const std::vector<NVGClusterGroupGPUData>& groups,
            const std::vector<NVGClusterMetaGPUData>&  rawClusters,
            std::vector<NVGClusterMetaGPUData>&        outOrderedClusters,
            std::vector<UINT32>&                       outNewGroupClusterStarts,
            std::vector<NVGHierarchyNodeGPUData>&      outNodes)
        {
            auto buildLeafForClusterSpan = [&](UINT32 dstNodeIdx, UINT32 clBegin, UINT32 clEnd) {
                NVGHierarchyNodeGPUData leaf{};
                leaf.boundsMin      = outOrderedClusters[clBegin].aabbMin;
                leaf.boundsMax      = outOrderedClusters[clBegin].aabbMax;
                leaf.childStart     = clBegin;
                leaf.maxParentError = 0.0f;
                leaf.minSelfError   = 1.0e30f;
                leaf.meshId         = outOrderedClusters[clBegin].meshId;

                Vec4Sphere s{};
                UINT32 minLod         = 255u;
                UINT32 maxLod         = 0u;
                UINT32 maxParentThinQ = 0u;
                for (UINT32 i = clBegin; i < clEnd; ++i)
                {
                    const NVGClusterMetaGPUData& cm = outOrderedClusters[i];
                    if (cm.aabbMin.x < leaf.boundsMin.x) leaf.boundsMin.x = cm.aabbMin.x;
                    if (cm.aabbMin.y < leaf.boundsMin.y) leaf.boundsMin.y = cm.aabbMin.y;
                    if (cm.aabbMin.z < leaf.boundsMin.z) leaf.boundsMin.z = cm.aabbMin.z;
                    if (cm.aabbMax.x > leaf.boundsMax.x) leaf.boundsMax.x = cm.aabbMax.x;
                    if (cm.aabbMax.y > leaf.boundsMax.y) leaf.boundsMax.y = cm.aabbMax.y;
                    if (cm.aabbMax.z > leaf.boundsMax.z) leaf.boundsMax.z = cm.aabbMax.z;

                    Vec4Sphere cs{};
                    cs.center = Vec3(cm.parentGroupSphere.x, cm.parentGroupSphere.y, cm.parentGroupSphere.z);
                    cs.radius = cm.parentGroupSphere.w;
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
                leaf.childCountAndFlags = PackNVGNodeChildCountAndFlags(clEnd - clBegin, true, minLod, maxLod);
                leaf.reserved           = (1u << 31u) | (maxParentThinQ & 0x3FFFu);
                outNodes[dstNodeIdx]    = leaf;
            };

            const UINT32 grpCount = rangeEnd - rangeBegin;
            UINT32 totalClustersInSpan = 0u;
            for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
            {
                totalClustersInSpan += groups[groupOrder[i]].clusterCount;
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
                const UINT32 clSpanEnd   = static_cast<UINT32>(outOrderedClusters.size());
                const UINT32 emittedCnt  = clSpanEnd - clSpanStart;

                if (emittedCnt <= 8u || depth >= kNVGMaxBVHDepth - 1u)
                {
                    buildLeafForClusterSpan(nodeIdx, clSpanStart, clSpanEnd);
                    return;
                }

                // Single oversized group (>8 clusters) with depth budget remaining: split contiguous cluster chunks <= 8
                const UINT32 numLeaves     = (emittedCnt + 7u) / 8u;
                const UINT32 firstChildIdx = static_cast<UINT32>(outNodes.size());
                for (UINT32 c = 0u; c < numLeaves; ++c)
                {
                    outNodes.push_back(NVGHierarchyNodeGPUData{});
                }

                NVGHierarchyNodeGPUData node{};
                node.boundsMin      = Vec3(1.0e30f, 1.0e30f, 1.0e30f);
                node.boundsMax      = Vec3(-1.0e30f, -1.0e30f, -1.0e30f);
                node.childStart     = firstChildIdx;
                node.maxParentError = 0.0f;
                node.minSelfError   = 1.0e30f;
                node.meshId         = outOrderedClusters[clSpanStart].meshId;

                Vec4Sphere mergedS{};
                UINT32 minLod         = 255u;
                UINT32 maxLod         = 0u;
                UINT32 maxParentThinQ = 0u;
                for (UINT32 c = 0u; c < numLeaves; ++c)
                {
                    const UINT32 lb = clSpanStart + c * 8u;
                    const UINT32 le = (std::min)(lb + 8u, clSpanEnd);
                    buildLeafForClusterSpan(firstChildIdx + c, lb, le);
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
                node.childCountAndFlags = PackNVGNodeChildCountAndFlags(numLeaves, false, minLod, maxLod);
                node.reserved           = (1u << 31u) | (maxParentThinQ & 0x3FFFu);
                outNodes[nodeIdx]       = node;
                return;
            }

            // Split group range into up to 4 spatial chunks along longest group-sphere centroid axis
            Vec3 cMin(1.0e30f, 1.0e30f, 1.0e30f);
            Vec3 cMax(-1.0e30f, -1.0e30f, -1.0e30f);
            for (UINT32 i = rangeBegin; i < rangeEnd; ++i)
            {
                const Vec4& gs = groups[groupOrder[i]].groupSphere;
                if (gs.x < cMin.x) cMin.x = gs.x;
                if (gs.y < cMin.y) cMin.y = gs.y;
                if (gs.z < cMin.z) cMin.z = gs.z;
                if (gs.x > cMax.x) cMax.x = gs.x;
                if (gs.y > cMax.y) cMax.y = gs.y;
                if (gs.z > cMax.z) cMax.z = gs.z;
            }
            const Vec3 span = cMax - cMin;
            int axis = 0;
            if (span.y > span.x && span.y >= span.z) axis = 1;
            else if (span.z > span.x && span.z >= span.y) axis = 2;

            std::sort(
                groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeBegin),
                groupOrder.begin() + static_cast<std::ptrdiff_t>(rangeEnd),
                [&](UINT32 a, UINT32 b) {
                    const Vec4& ca = groups[a].groupSphere;
                    const Vec4& cb = groups[b].groupSphere;
                    if (axis == 0) return ca.x < cb.x;
                    if (axis == 1) return ca.y < cb.y;
                    return ca.z < cb.z;
                });

            const UINT32 childPartitions = (grpCount >= 8u) ? 4u : 2u;
            std::vector<std::pair<UINT32, UINT32>> subRanges;
            for (UINT32 p = 0u; p < childPartitions; ++p)
            {
                const UINT32 sb = rangeBegin + (grpCount * p) / childPartitions;
                const UINT32 se = rangeBegin + (grpCount * (p + 1u)) / childPartitions;
                if (se > sb) subRanges.push_back({ sb, se });
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
                BuildGroupBVHRecursive(
                    firstChildIdx + c, groupOrder, subRanges[c].first, subRanges[c].second,
                    depth + 1u, groups, rawClusters, outOrderedClusters, outNewGroupClusterStarts, outNodes);
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
    //  Bit-Exact Encoding / Decoding Helpers
    // =====================================================================

    UINT16 NVGBuilder::FloatToHalf(FLOAT32 value) noexcept
    {
        const UINT32 f = BitCastFloatToUInt(value);
        const UINT32 sign = (f >> 16u) & 0x8000u;
        const INT32  exp  = static_cast<INT32>((f >> 23u) & 0xFFu) - 127 + 15;
        UINT32       mant = f & 0x007FFFFFu;

        if (exp <= 0)
        {
            if (exp < -10) return static_cast<UINT16>(sign);
            mant = (mant | 0x00800000u) >> static_cast<UINT32>(1 - exp);
            return static_cast<UINT16>(sign | ((mant + 0x00001000u) >> 13u));
        }
        if (exp >= 31)
        {
            return static_cast<UINT16>(sign | 0x7C00u);
        }
        return static_cast<UINT16>(sign | (static_cast<UINT32>(exp) << 10u) | ((mant + 0x00001000u) >> 13u));
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
        const Vec3 n = (unitNormal.LengthSq() > 1e-8f) ? unitNormal.Normalized() : Vec3(0.0f, 1.0f, 0.0f);
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
        const INT16 ix = static_cast<INT16>(std::lround(ClampF(ox, -1.0f, 1.0f) * 32767.0f));
        const INT16 iy = static_cast<INT16>(std::lround(ClampF(oy, -1.0f, 1.0f) * 32767.0f));
        const UINT16 ux = static_cast<UINT16>(ix);
        const UINT16 uy = static_cast<UINT16>(iy);
        return static_cast<UINT32>(ux) | (static_cast<UINT32>(uy) << 16u);
    }

    Vec3 NVGBuilder::UnpackOctahedralNormal(UINT32 packed) noexcept
    {
        const INT16 ix = static_cast<INT16>(packed & 0xFFFFu);
        const INT16 iy = static_cast<INT16>((packed >> 16u) & 0xFFFFu);
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

        // Orthogonalize T against N
        Vec3 T = unitTangent - N * N.Dot(unitTangent);
        if (T.LengthSq() < 1e-8f) T = B0;
        else T = T.Normalized();

        const FLOAT32 angle  = std::atan2(T.Dot(B1), T.Dot(B0)); // [-pi, +pi]
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
    //  BuildVirtualMesh — Complete End-to-End Virtual Mesh Pipeline
    // =====================================================================

    BOOL NVGBuilder::BuildVirtualMesh(
        const Vertex*         vertices,
        UINT32                vertexCount,
        const UINT32*         indices,
        UINT32                indexCount,
        const NVGBuildConfig& config,
        NVGVirtualMeshAsset&  outAsset)
    {
        const auto tBuildStart = std::chrono::high_resolution_clock::now();
        outAsset = NVGVirtualMeshAsset{};
        NVGBuildReport& rep = outAsset.buildReport;
        rep.sourceVertexCount = vertexCount;

        if (!vertices || vertexCount < 3u)
        {
            rep.succeeded         = false;
            rep.diagnosticMessage = "NVGBuilder rejected mesh: vertexCount < 3 or null vertex data.";
            return false;
        }

        // 1. Geometry validation & sanitization
        std::vector<Vertex> cleanVerts(vertexCount);
        BOOL hasSkinning = false;
        Vec3 meshMin(1.0e30f, 1.0e30f, 1.0e30f);
        Vec3 meshMax(-1.0e30f, -1.0e30f, -1.0e30f);

        for (UINT32 i = 0u; i < vertexCount; ++i)
        {
            Vertex v = vertices[i];
            if (!IsFiniteVec3(v.position))
            {
                rep.succeeded         = false;
                rep.diagnosticMessage = "NVGBuilder rejected mesh: vertex position contains NaN or Inf.";
                return false;
            }
            BOOL sanitized = false;
            if (!IsFiniteVec3(v.normal) || v.normal.LengthSq() < 1e-10f)
            {
                v.normal = Vec3(0.0f, 1.0f, 0.0f);
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
                sanitized = true;
            }
            else
            {
                v.bitangent = v.bitangent.Normalized();
            }
            if (!IsFiniteFloat(v.texCoord0.x) || !IsFiniteFloat(v.texCoord0.y))
            {
                v.texCoord0 = Vec2(0.0f, 0.0f);
                sanitized = true;
            }
            if (!IsFiniteFloat(v.texCoord1.x) || !IsFiniteFloat(v.texCoord1.y))
            {
                v.texCoord1 = Vec2(0.0f, 0.0f);
                sanitized = true;
            }
            if (!IsFiniteFloat(v.color.x) || !IsFiniteFloat(v.color.y) ||
                !IsFiniteFloat(v.color.z) || !IsFiniteFloat(v.color.w))
            {
                v.color = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
                sanitized = true;
            }
            if (!IsFiniteFloat(v.boneWeights.x) || !IsFiniteFloat(v.boneWeights.y) ||
                !IsFiniteFloat(v.boneWeights.z) || !IsFiniteFloat(v.boneWeights.w))
            {
                v.boneWeights = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
                sanitized = true;
            }
            const FLOAT32 bwSum = v.boneWeights.x + v.boneWeights.y + v.boneWeights.z + v.boneWeights.w;
            if (config.preserveSkinningAttributes && bwSum > 1e-5f)
            {
                hasSkinning = true;
                v.boneWeights = v.boneWeights * (1.0f / bwSum);
            }
            if (sanitized) rep.sanitizedVertexCount++;

            if (v.position.x < meshMin.x) meshMin.x = v.position.x;
            if (v.position.y < meshMin.y) meshMin.y = v.position.y;
            if (v.position.z < meshMin.z) meshMin.z = v.position.z;
            if (v.position.x > meshMax.x) meshMax.x = v.position.x;
            if (v.position.y > meshMax.y) meshMax.y = v.position.y;
            if (v.position.z > meshMax.z) meshMax.z = v.position.z;

            cleanVerts[i] = v;
        }

        outAsset.localBounds = AABB(meshMin, meshMax);
        outAsset.hasSkinning = hasSkinning;
        rep.hasSkinning      = hasSkinning;

        const FLOAT32 meshDiag = (std::max)((meshMax - meshMin).Length(), 1e-4f);
        const FLOAT32 weldEps  = meshDiag * 1.0e-5f;

        // Build triangle list & validate indices / degeneracy
        const UINT32 rawTriCount = (indices != nullptr && indexCount > 0u) ? (indexCount / 3u) : (vertexCount / 3u);
        rep.sourceTriangleCount = rawTriCount;

        std::vector<BuildTriangle> validTris;
        validTris.reserve(rawTriCount);
        for (UINT32 t = 0u; t < rawTriCount; ++t)
        {
            const UINT32 i0 = (indices != nullptr && indexCount > 0u) ? indices[t * 3u + 0u] : (t * 3u + 0u);
            const UINT32 i1 = (indices != nullptr && indexCount > 0u) ? indices[t * 3u + 1u] : (t * 3u + 1u);
            const UINT32 i2 = (indices != nullptr && indexCount > 0u) ? indices[t * 3u + 2u] : (t * 3u + 2u);
            if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
            {
                rep.succeeded         = false;
                rep.diagnosticMessage = "NVGBuilder rejected mesh: index out of vertexCount bounds.";
                return false;
            }
            if (i0 == i1 || i1 == i2 || i2 == i0)
            {
                rep.degenerateTrianglesRemoved++;
                continue;
            }
            const Vec3& p0 = cleanVerts[i0].position;
            const Vec3& p1 = cleanVerts[i1].position;
            const Vec3& p2 = cleanVerts[i2].position;
            if ((p1 - p0).Cross(p2 - p0).LengthSq() <= 1e-16f)
            {
                rep.degenerateTrianglesRemoved++;
                continue;
            }
            validTris.push_back({ { i0, i1, i2 } });
        }

        rep.validTriangleCount = static_cast<UINT32>(validTris.size());
        if (validTris.empty())
        {
            rep.succeeded         = false;
            rep.diagnosticMessage = "NVGBuilder rejected mesh: 0 valid non-degenerate triangles remain.";
            return false;
        }

        // 2. Topology, seam, non-manifold & island analysis
        std::vector<UINT32> spatialIds;
        BuildSpatialWeldIds(cleanVerts, weldEps, spatialIds);

        {
            std::unordered_map<UINT64, UINT32> edgeCounts;
            edgeCounts.reserve(validTris.size() * 2u);
            for (const BuildTriangle& t : validTris)
            {
                const UINT32 s0 = spatialIds[t.v[0]];
                const UINT32 s1 = spatialIds[t.v[1]];
                const UINT32 s2 = spatialIds[t.v[2]];
                if (s0 != s1) edgeCounts[MakeUndirectedEdgeKey(s0, s1)]++;
                if (s1 != s2) edgeCounts[MakeUndirectedEdgeKey(s1, s2)]++;
                if (s2 != s0) edgeCounts[MakeUndirectedEdgeKey(s2, s0)]++;
            }
            for (const auto& kv : edgeCounts)
            {
                if (kv.second == 1u) rep.boundaryEdgeCount++;
                else if (kv.second > 2u) rep.nonManifoldEdgeCount++;
            }
            if (rep.nonManifoldEdgeCount > 0u && !config.allowNonManifoldGeometry)
            {
                rep.succeeded         = false;
                rep.diagnosticMessage = "NVGBuilder rejected non-manifold mesh (allowNonManifoldGeometry == false).";
                return false;
            }

            // Count attribute seam edges
            std::unordered_map<UINT64, UINT32> rawEdgeCounts;
            rawEdgeCounts.reserve(validTris.size() * 2u);
            for (const BuildTriangle& t : validTris)
            {
                rawEdgeCounts[MakeUndirectedEdgeKey(t.v[0], t.v[1])]++;
                rawEdgeCounts[MakeUndirectedEdgeKey(t.v[1], t.v[2])]++;
                rawEdgeCounts[MakeUndirectedEdgeKey(t.v[2], t.v[0])]++;
            }
            UINT32 rawOpen = 0u;
            for (const auto& kv : rawEdgeCounts)
            {
                if (kv.second == 1u) ++rawOpen;
            }
            rep.seamEdgeCount = (rawOpen > rep.boundaryEdgeCount) ? ((rawOpen - rep.boundaryEdgeCount) / 2u) : 0u;

            // Count connected geometry islands using disjoint-set union over spatial IDs
            UINT32 maxSpatial = 0u;
            for (UINT32 sid : spatialIds) if (sid > maxSpatial) maxSpatial = sid;
            std::vector<UINT32> dsu(maxSpatial + 1u);
            std::iota(dsu.begin(), dsu.end(), 0u);
            auto findRoot = [&](UINT32 x) {
                while (dsu[x] != x)
                {
                    dsu[x] = dsu[dsu[x]];
                    x = dsu[x];
                }
                return x;
            };
            for (const BuildTriangle& t : validTris)
            {
                const UINT32 r0 = findRoot(spatialIds[t.v[0]]);
                const UINT32 r1 = findRoot(spatialIds[t.v[1]]);
                const UINT32 r2 = findRoot(spatialIds[t.v[2]]);
                dsu[r1] = r0;
                dsu[r2] = r0;
            }
            std::unordered_set<UINT32> islandRoots;
            for (const BuildTriangle& t : validTris)
            {
                islandRoots.insert(findRoot(spatialIds[t.v[0]]));
            }
            rep.disconnectedIslandCount = static_cast<UINT32>(islandRoots.size());
        }

        // 3. Level 0 Cluster Partitioning (Exact Source Geometry)
        const UINT32 targetClusterTris = (std::min)(kNVGMaxClusterTriangles, (std::max)(32u, config.targetTrianglesPerCluster));
        std::vector<BuildCluster> allClusters;
        std::vector<BuildGroup>   allGroups;

        std::vector<BuildCluster> levelClusters;
        PartitionIntoClusters(cleanVerts, spatialIds, validTris, targetClusterTris, 0u, levelClusters);
        rep.lod0ClusterCount = static_cast<UINT32>(levelClusters.size());

        std::vector<UINT32> currentLevelClusterIds;
        currentLevelClusterIds.reserve(levelClusters.size());
        for (BuildCluster& c : levelClusters)
        {
            c.globalIndex     = static_cast<UINT32>(allClusters.size());
            c.selfGroupError  = 0.0f;
            c.selfGroupThin   = 0.0f;
            c.parentGroupThin = 0.0f;
            c.selfGroupSphere = c.sphere;
            currentLevelClusterIds.push_back(c.globalIndex);
            allClusters.push_back(std::move(c));
        }

        // 4. Bottom-Up Hierarchical Cluster Group DAG & Crack-Free QEM Simplification
        UINT32 currentLod = 0u;
        const FLOAT32 ratio = ClampF(config.simplificationRatioPerLevel, 0.25f, 0.75f);
        BOOL forceFinalRootMerge = false;

        while (currentLevelClusterIds.size() > 1u && currentLod + 1u < kNVGMaxLODLevels)
        {
            // Spatially order current-level clusters along a 3D Morton curve so seed selection and
            // disconnected-island pairing remain strictly local at every hierarchy level.
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

            const UINT32 levelCount = static_cast<UINT32>(currentLevelClusterIds.size());
            // Compute exact encoded word footprint per cluster so no ClusterGroup ever exceeds
            // one 32 KiB streaming page (kNVGPageWordCount = 8192 words), preserving atomic single-page residency.
            const UINT32 vtxStrideWords = hasSkinning ? 11u : 7u;
            std::vector<UINT32> localClusterWords(levelCount, 0u);
            for (UINT32 li = 0u; li < levelCount; ++li)
            {
                const BuildCluster& c = allClusters[currentLevelClusterIds[li]];
                localClusterWords[li] = 16u + static_cast<UINT32>(c.vertices.size()) * vtxStrideWords +
                                        static_cast<UINT32>(c.indices.size() / 3u);
            }
            const UINT32 collapseClusterCap = hasSkinning ? 5u : 7u;
            const BOOL   collapseAllToRoot  = (levelCount <= collapseClusterCap) ||
                                              (forceFinalRootMerge && levelCount <= 512u) ||
                                              ((currentLod + 2u >= kNVGMaxLODLevels) && (levelCount <= 1024u));
            const BOOL   enforcePageWordCap = !collapseAllToRoot;
            const UINT32 groupTarget   = collapseAllToRoot ? levelCount : (forceFinalRootMerge ? (hasSkinning ? 12u : 16u) : 4u);
            const UINT32 groupCap      = collapseAllToRoot ? levelCount : (hasSkinning ? 5u : 7u);
            const UINT32 pairMergeCap  = collapseAllToRoot ? levelCount : (forceFinalRootMerge ? 32u : (hasSkinning ? 10u : 12u));
            const UINT32 pairMergeWords= forceFinalRootMerge ? (kNVGPageWordCount * 4u) : (kNVGPageWordCount * 2u);

            // Build cluster adjacency map via shared spatial-welded boundary edges
            std::unordered_map<UINT64, std::vector<UINT32>> boundaryEdgeToLocalCluster;
            boundaryEdgeToLocalCluster.reserve(static_cast<SIZE_T>(levelCount) * 24u);
            for (UINT32 li = 0u; li < levelCount; ++li)
            {
                const BuildCluster& c = allClusters[currentLevelClusterIds[li]];
                for (UINT64 ek : c.boundaryEdges)
                {
                    boundaryEdgeToLocalCluster[ek].push_back(li);
                }
            }

            std::vector<std::unordered_map<UINT32, UINT32>> adjWeight(levelCount);
            for (const auto& kv : boundaryEdgeToLocalCluster)
            {
                const auto& list = kv.second;
                if (list.size() < 2u || list.size() > 8u) continue;
                for (SIZE_T a = 0; a < list.size(); ++a)
                {
                    for (SIZE_T b = a + 1; b < list.size(); ++b)
                    {
                        if (list[a] == list[b]) continue;
                        adjWeight[list[a]][list[b]]++;
                        adjWeight[list[b]][list[a]]++;
                    }
                }
            }

            // Partition current level clusters into spatially compact, mutually-adjacent Cluster Groups.
            // Keep sibling clusters originating from the same child group together in the same parent
            // group so every child group has a single unique parentGroupIndex (strict residency tree).
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

                // Map childGroupId -> all local cluster indices created from that child group
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
                            UINT32 needed = 0u;
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

                // Phase A: Pure shared-edge topological grouping (never merges non-adjacent clusters across the mesh)
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
                                if (grouped[kv.first]) continue;
                                const BuildCluster& cc = allClusters[currentLevelClusterIds[kv.first]];
                                UINT32 sibNeed  = 1u;
                                UINT32 sibWords = localClusterWords[kv.first];
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
                                for (UINT32 m2 : grp)
                                {
                                    auto itW = adjWeight[kv.first].find(m2);
                                    if (itW != adjWeight[kv.first].end()) wSum += itW->second;
                                }
                                if (wSum == 0u) continue;

                                const FLOAT32 mergedR = MergeSpheres(curSphere, cc.sphere).radius;
                                const FLOAT32 score   = static_cast<FLOAT32>(wSum) * 16.0f - (mergedR / meshDiag) * 4.0f;
                                if (score > bestScore)
                                {
                                    bestScore = score;
                                    bestNb    = kv.first;
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

                // Phase B: Absorb single-pack / undersized topological stragglers into adjacent groups first,
                // then pair genuinely disconnected islands locally in Morton order.
                if (localGroups.size() > 1u)
                {
                    const UINT32 numRawGroups = static_cast<UINT32>(localGroups.size());
                    std::vector<UINT32> grpWords(numRawGroups, 0u);
                    std::vector<Vec4Sphere> grpSpheres(numRawGroups);
                    std::vector<UINT32> clusterOwnerGrp(levelCount, 0u);
                    for (UINT32 gi = 0u; gi < numRawGroups; ++gi)
                    {
                        for (UINT32 lc : localGroups[gi])
                        {
                            grpWords[gi]       += localClusterWords[lc];
                            grpSpheres[gi]      = MergeSpheres(grpSpheres[gi], allClusters[currentLevelClusterIds[lc]].sphere);
                            clusterOwnerGrp[lc] = gi;
                        }
                    }

                    // Pass B1: Merge straggler groups into shared-edge adjacent groups that have remaining capacity
                    for (UINT32 gi = 0u; gi < numRawGroups; ++gi)
                    {
                        if (localGroups[gi].empty()) continue;
                        const BOOL isStraggler = (groupPackCounts[gi] < 2u) || (localGroups[gi].size() <= 2u);
                        if (!isStraggler) continue;

                        std::unordered_map<UINT32, UINT32> sharedWithGroup;
                        for (UINT32 lc : localGroups[gi])
                        {
                            for (const auto& kv : adjWeight[lc])
                            {
                                const UINT32 otherGi = clusterOwnerGrp[kv.first];
                                if (otherGi != gi && !localGroups[otherGi].empty())
                                {
                                    sharedWithGroup[otherGi] += kv.second;
                                }
                            }
                        }

                        UINT32  bestTargetGi = kNVGInvalidIndex;
                        FLOAT32 bestMergeScore = -1.0e30f;
                        for (const auto& kv : sharedWithGroup)
                        {
                            const UINT32 candGi = kv.first;
                            if (localGroups[candGi].size() + localGroups[gi].size() > pairMergeCap) continue;
                            if (enforcePageWordCap && grpWords[candGi] + grpWords[gi] > pairMergeWords) continue;
                            const FLOAT32 mergedR = MergeSpheres(grpSpheres[candGi], grpSpheres[gi]).radius;
                            const FLOAT32 score   = static_cast<FLOAT32>(kv.second) * 16.0f - (mergedR / meshDiag) * 4.0f;
                            if (score > bestMergeScore)
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
                            grpWords[bestTargetGi]  += grpWords[gi];
                            grpSpheres[bestTargetGi] = MergeSpheres(grpSpheres[bestTargetGi], grpSpheres[gi]);
                            groupPackCounts[bestTargetGi] += groupPackCounts[gi];
                            localGroups[gi].clear();
                            grpWords[gi] = 0u;
                        }
                    }

                    // Pass B2: Pair remaining disconnected-island stragglers locally along the 3D Morton curve
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
                            grpWords[gi]       += grpWords[bestOtherGi];
                            grpSpheres[gi]      = MergeSpheres(grpSpheres[gi], grpSpheres[bestOtherGi]);
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

            // Map each local cluster to its localGroup index for exterior boundary detection
            std::vector<UINT32> localClusterToGroup(levelCount, 0u);
            for (UINT32 gi = 0u; gi < static_cast<UINT32>(localGroups.size()); ++gi)
            {
                for (UINT32 lc : localGroups[gi]) localClusterToGroup[lc] = gi;
            }

            const BOOL isFinalMergeToRoot = (localGroups.size() == 1u);
            std::vector<UINT32> nextLevelClusterIds;

            for (UINT32 gi = 0u; gi < static_cast<UINT32>(localGroups.size()); ++gi)
            {
                const auto& grpLocal = localGroups[gi];

                // Create a BuildGroup record for these child clusters at currentLod
                BuildGroup childGroup{};
                childGroup.lodLevel    = currentLod;
                childGroup.globalIndex = static_cast<UINT32>(allGroups.size());
                FLOAT32 maxChildSelfErr = 0.0f;
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

                // Determine strictly locked exterior boundary spatial IDs (edges shared with OTHER groups!)
                std::unordered_set<UINT32> lockedSpatialIds;
                if (!isFinalMergeToRoot)
                {
                    for (UINT32 lc : grpLocal)
                    {
                        const BuildCluster& c = allClusters[currentLevelClusterIds[lc]];
                        for (UINT64 ek : c.boundaryEdges)
                        {
                            auto itE = boundaryEdgeToLocalCluster.find(ek);
                            if (itE == boundaryEdgeToLocalCluster.end()) continue;
                            BOOL touchesExternalGroup = false;
                            for (UINT32 otherLc : itE->second)
                            {
                                if (localClusterToGroup[otherLc] != gi)
                                {
                                    touchesExternalGroup = true;
                                    break;
                                }
                            }
                            if (touchesExternalGroup)
                            {
                                lockedSpatialIds.insert(static_cast<UINT32>(ek >> 32u));
                                lockedSpatialIds.insert(static_cast<UINT32>(ek & 0xFFFFFFFFu));
                            }
                        }
                    }
                }

                // Merge all triangles of this child group into a temporary mesh while strictly preserving
                // per-chart (spatialId, normal, texCoord0, texCoord1, bone) wedges across ALL LOD levels.
                // Coupled-Wedge Position-Pair QEM collapses 3D spatial edges in lockstep across seams without
                // ever welding disparate UV islands or cylindrical wrap borders!
                auto buildMergedGroupMesh = [&](std::vector<Vertex>&        outMVerts,
                                                std::vector<UINT32>&        outMSpatial,
                                                std::vector<BuildTriangle>& outMTris)
                {
                    outMVerts.clear();
                    outMSpatial.clear();
                    outMTris.clear();

                    std::unordered_map<UINT64, std::vector<UINT32>> vertDedup;
                    for (UINT32 lc : grpLocal)
                    {
                        const BuildCluster& c = allClusters[currentLevelClusterIds[lc]];
                        std::vector<UINT32> localToMerged(c.vertices.size(), 0u);
                        for (SIZE_T vi = 0; vi < c.vertices.size(); ++vi)
                        {
                            const Vertex& v   = c.vertices[vi];
                            const UINT32  sid = c.spatialIds[vi];

                            UINT64 vHash = static_cast<UINT64>(sid) * 1099511628211ULL;
                            const UINT32 u0q = static_cast<UINT32>(std::lround(v.texCoord0.x * 512.0f));
                            const UINT32 v0q = static_cast<UINT32>(std::lround(v.texCoord0.y * 512.0f));
                            vHash ^= (static_cast<UINT64>(u0q) << 16u) ^ static_cast<UINT64>(v0q);

                            UINT32 matchedIdx = kNVGInvalidIndex;
                            auto itD = vertDedup.find(vHash);
                            if (itD != vertDedup.end())
                            {
                                for (UINT32 candIdx : itD->second)
                                {
                                    if (outMSpatial[candIdx] == sid &&
                                        (outMVerts[candIdx].texCoord0 - v.texCoord0).LengthSq() < 1e-6f &&
                                        (outMVerts[candIdx].texCoord1 - v.texCoord1).LengthSq() < 1e-6f &&
                                        outMVerts[candIdx].normal.Dot(v.normal) > 0.995f &&
                                        (!hasSkinning || outMVerts[candIdx].boneIndices[0] == v.boneIndices[0]))
                                    {
                                        matchedIdx = candIdx;
                                        break;
                                    }
                                }
                            }

                            if (matchedIdx != kNVGInvalidIndex)
                            {
                                localToMerged[vi] = matchedIdx;
                            }
                            else
                            {
                                const UINT32 idx = static_cast<UINT32>(outMVerts.size());
                                outMVerts.push_back(v);
                                outMSpatial.push_back(sid);
                                vertDedup[vHash].push_back(idx);
                                localToMerged[vi] = idx;
                            }
                        }
                        const SIZE_T tc = c.indices.size() / 3u;
                        for (SIZE_T t = 0; t < tc; ++t)
                        {
                            const UINT32 i0 = localToMerged[c.indices[t * 3u + 0u]];
                            const UINT32 i1 = localToMerged[c.indices[t * 3u + 1u]];
                            const UINT32 i2 = localToMerged[c.indices[t * 3u + 2u]];
                            if (i0 != i1 && i1 != i2 && i2 != i0)
                            {
                                outMTris.push_back({ { i0, i1, i2 } });
                            }
                        }
                    }
                };

                const BOOL relaxSeamsInitial  = forceFinalRootMerge || isFinalMergeToRoot || (currentLod >= 2u);
                const BOOL coarseRidgeFlatten = forceFinalRootMerge || isFinalMergeToRoot || (currentLod >= 3u);
                std::vector<Vertex>        mergedVerts;
                std::vector<UINT32>        mergedSpatial;
                std::vector<BuildTriangle> mergedTris;
                buildMergedGroupMesh(mergedVerts, mergedSpatial, mergedTris);

                const UINT32 inTriCount    = static_cast<UINT32>(mergedTris.size());
                const UINT32 rawTargetTris = (std::max)(4u, static_cast<UINT32>(std::ceil(static_cast<FLOAT32>(inTriCount) * ratio)));
                UINT32 maxParents = 1u;
                if (isFinalMergeToRoot)
                {
                    maxParents = 1u;
                }
                else if (grpLocal.size() >= 2u)
                {
                    const UINT32 byClusterHalving = (std::max)(1u, static_cast<UINT32>(grpLocal.size()) / 2u);
                    const UINT32 byTriCount       = (std::max)(1u, (rawTargetTris + targetClusterTris - 1u) / targetClusterTris);
                    maxParents = (std::min)(byClusterHalving, byTriCount);
                }
                UINT32 perParentTriBudget = (targetClusterTris > 8u) ? (targetClusterTris - 6u) : targetClusterTris;
                if (mergedVerts.size() > mergedTris.size() && inTriCount > 0u)
                {
                    // Reserve cluster vertex headroom for preserved multi-chart UV-seam wedges (<= 128 vertices/cluster)
                    const UINT32 safeWedgeBudget = (std::max)(
                        28u,
                        ((kNVGMaxClusterVertices - 8u) * inTriCount) / static_cast<UINT32>(mergedVerts.size()));
                    perParentTriBudget = (std::min)(perParentTriBudget, safeWedgeBudget);
                }
                UINT32 targetSimpTris = (std::min)(rawTargetTris, maxParents * perParentTriBudget);
                targetSimpTris = (std::max)(4u, targetSimpTris);

                std::vector<Vertex>        simpVerts;
                std::vector<UINT32>        simpSpatial;
                std::vector<BuildTriangle> simpTris;
                FLOAT32 stepErr = SimplifyClusterGroupQEM(
                    mergedVerts, mergedSpatial, mergedTris, lockedSpatialIds,
                    targetSimpTris, config, hasSkinning,
                    simpVerts, simpSpatial, simpTris,
                    relaxSeamsInitial,
                    coarseRidgeFlatten,
                    false);

                // If strict ridge/seam weighting stalled reduction (<25% drop), relax penalty weights while STILL preserving UV wedges!
                if (simpTris.size() > targetSimpTris && simpTris.size() * 4u >= inTriCount * 3u)
                {
                    std::vector<Vertex>        svRelax;
                    std::vector<UINT32>        ssRelax;
                    std::vector<BuildTriangle> stRelax;
                    const FLOAT32 errRelax = SimplifyClusterGroupQEM(
                        mergedVerts, mergedSpatial, mergedTris, lockedSpatialIds,
                        targetSimpTris, config, hasSkinning,
                        svRelax, ssRelax, stRelax,
                        true,
                        true,
                        false);
                    if (!stRelax.empty() && stRelax.size() < simpTris.size())
                    {
                        simpVerts   = std::move(svRelax);
                        simpSpatial = std::move(ssRelax);
                        simpTris    = std::move(stRelax);
                        stepErr     = (std::max)(stepErr, errRelax);
                    }
                }

                // Final single-group root pass: zero exterior group neighbors exist, allow coarse root convergence
                if (isFinalMergeToRoot && simpTris.size() > perParentTriBudget)
                {
                    std::unordered_set<UINT32> noLocks;
                    std::vector<Vertex>        sv2;
                    std::vector<UINT32>        ss2;
                    std::vector<BuildTriangle> st2;
                    const FLOAT32 err2 = SimplifyClusterGroupQEM(
                        simpVerts, simpSpatial, simpTris, noLocks,
                        perParentTriBudget, config, hasSkinning,
                        sv2, ss2, st2,
                        true,
                        true,
                        true);
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

                // Partition simplified group into parent cluster(s); if multi-chart UV-wedge splits caused
                // PartitionIntoClusters to exceed maxParents (128-vertex cap), re-simplify with tighter triangle headroom
                // while keeping all per-chart UV wedges 100% unmerged!
                std::vector<BuildCluster> parentClusters;
                PartitionIntoClusters(simpVerts, simpSpatial, simpTris, targetClusterTris, currentLod + 1u, parentClusters);
                if (parentClusters.size() > maxParents && simpTris.size() > 8u)
                {
                    const UINT32 tightTargetTris = (std::max)(
                        4u,
                        static_cast<UINT32>((simpTris.size() * maxParents) / parentClusters.size()) > 4u
                            ? static_cast<UINT32>((simpTris.size() * maxParents) / parentClusters.size()) - 4u
                            : 4u);
                    const std::unordered_set<UINT32> emptyLocks;
                    const std::unordered_set<UINT32>& activeLocks = isFinalMergeToRoot ? emptyLocks : lockedSpatialIds;

                    std::vector<Vertex>        svTight;
                    std::vector<UINT32>        ssTight;
                    std::vector<BuildTriangle> stTight;
                    const FLOAT32 errTight = SimplifyClusterGroupQEM(
                        simpVerts, simpSpatial, simpTris, activeLocks,
                        tightTargetTris, config, hasSkinning,
                        svTight, ssTight, stTight,
                        true,
                        true,
                        isFinalMergeToRoot);
                    if (!stTight.empty())
                    {
                        std::vector<BuildCluster> retryClusters;
                        PartitionIntoClusters(svTight, ssTight, stTight, targetClusterTris, currentLod + 1u, retryClusters);
                        if (!retryClusters.empty() && retryClusters.size() < parentClusters.size())
                        {
                            parentClusters = std::move(retryClusters);
                            stepErr        = (std::max)(stepErr, errTight);
                        }
                    }
                }

                // Compute conservative group sphere enclosing both children and parents
                for (const BuildCluster& pc : parentClusters)
                {
                    groupSphere = MergeSpheres(groupSphere, pc.sphere);
                }

                // Strictly monotonic geometric error & thin-feature shadow metric from children to parents.
                // Synchronizing BOTH error and thin metric across every (childGroup -> parentClusters) edge
                // guarantees bit-identical shadow LOD cuts with zero holes and zero overlapping LODs!
                const FLOAT32 minStepEpsilon = (std::max)(1e-5f * meshDiag, 1e-6f);
                const FLOAT32 parentError    = maxChildSelfErr + (std::max)(stepErr, minStepEpsilon);

                FLOAT32 groupThin = 0.0f;
                for (UINT32 lc : grpLocal)
                {
                    const BuildCluster& cc = allClusters[currentLevelClusterIds[lc]];
                    groupThin = (std::max)(groupThin, (std::max)(cc.selfGroupThin, cc.thinFeatureMetric));
                }
                for (const BuildCluster& pc : parentClusters)
                {
                    groupThin = (std::max)(groupThin, pc.thinFeatureMetric);
                }

                childGroup.sphere = groupSphere;
                childGroup.error  = parentError;
                const UINT32 childGroupId = childGroup.globalIndex;
                allGroups.push_back(std::move(childGroup));

                // Synchronize parent test metrics across ALL children in this group (guarantees crack-free DAG cuts!)
                for (UINT32 lc : grpLocal)
                {
                    BuildCluster& cc = allClusters[currentLevelClusterIds[lc]];
                    cc.parentGroupError  = parentError;
                    cc.parentGroupSphere = groupSphere;
                    cc.parentGroupThin   = groupThin;
                }

                // Record newly created parent clusters
                for (BuildCluster& pc : parentClusters)
                {
                    pc.globalIndex       = static_cast<UINT32>(allClusters.size());
                    pc.selfGroupError    = parentError;
                    pc.selfGroupSphere   = groupSphere;
                    pc.selfGroupThin     = groupThin;
                    pc.parentGroupError  = kNVGRootErrorInf;
                    pc.parentGroupSphere = groupSphere;
                    pc.parentGroupThin   = groupThin;
                    pc.childGroupStart   = childGroupId; // Direct child group index refining this parent cluster
                    pc.childGroupCount   = 1u;
                    nextLevelClusterIds.push_back(pc.globalIndex);
                    allClusters.push_back(std::move(pc));
                }
            }

            // Guard against non-progressing degenerate cases: if cluster count failed to drop,
            // enable seam-relaxed QEM on the next pass, or terminate cleanly if already stalled twice.
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
            FLOAT32 rootErr = 0.0f;
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

        // Link parentGroupIndex and childGroupIndices across BuildGroups
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
        // Populate cluster parentGroupIndex from its selfGroup's parentGroupIndex
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

        // 5. Quantized Cluster Compression & 32 KiB Virtual Geometry Page Packing
        double maxClusterAxisSpan = 1e-6;
        double maxAbsCoord        = 1e-6;
        for (const BuildCluster& c : allClusters)
        {
            const Vec3 ext = c.aabb.m_Max - c.aabb.m_Min;
            maxClusterAxisSpan = (std::max)(maxClusterAxisSpan, static_cast<double>(ext.x));
            maxClusterAxisSpan = (std::max)(maxClusterAxisSpan, static_cast<double>(ext.y));
            maxClusterAxisSpan = (std::max)(maxClusterAxisSpan, static_cast<double>(ext.z));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.x)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.y)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Min.z)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.x)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.y)));
            maxAbsCoord = (std::max)(maxAbsCoord, std::fabs(static_cast<double>(c.aabb.m_Max.z)));
        }
        const double minQuantStep = (std::max)(maxClusterAxisSpan / 65534.0, maxAbsCoord / 8388600.0);
        const double quantStep    = std::ldexp(1.0, static_cast<int>(std::ceil(std::log2((std::max)(minQuantStep, 1e-12)))));

        FLOAT32 globalMaxQuantErr = 0.0f;
        for (BuildCluster& c : allClusters)
        {
            const FLOAT32 qErr = EncodeClusterToWords(c, quantStep, hasSkinning);
            if (qErr > globalMaxQuantErr) globalMaxQuantErr = qErr;
        }
        rep.maxQuantizationError = globalMaxQuantErr;

        // Pack root clusters into pinned Root Virtual Pages (pages 0 .. rootPageCount-1),
        // then pack finer clusters ordered by (LOD descending, groupIndex ascending) into Streaming Virtual Pages.
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
                return allClusters[a].lodLevel > allClusters[b].lodLevel; // Coarser LODs first
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
                // For streaming pages, pack every ClusterGroup atomically into a single 32 KiB page
                // so a group never straddles two streaming pages (guarantees single-page group residency).
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
                    curPage = NVGVirtualPagePayload{};
                    curPage.localPageIndex = static_cast<UINT32>(outAsset.pages.size());
                    curPage.lodLevel       = allClusters[list[idx]].lodLevel;
                    curPage.isRootPage     = isRoot;
                    curPage.words.assign(kNVGPageWordCount, 0u);
                    curPage.usedWords      = 0u;
                    curPage.clusterCount   = 0u;
                }

                for (SIZE_T k = idx; k < runEnd; ++k)
                {
                    BuildCluster& c = allClusters[list[k]];
                    const UINT32 needWords = static_cast<UINT32>(c.encodedWords.size());
                    c.virtualPageId    = curPage.localPageIndex;
                    c.wordOffsetInPage = curPage.usedWords;
                    std::memcpy(curPage.words.data() + curPage.usedWords, c.encodedWords.data(), needWords * sizeof(UINT32));
                    curPage.usedWords += needWords;
                    curPage.clusterCount++;
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

        // 6. Populate GPU-Aligned Cluster & ClusterGroup Metadata Arrays (preserving contiguous per-group cluster spans!)
        std::vector<NVGClusterMetaGPUData> rawMeta;
        rawMeta.reserve(allClusters.size());
        outAsset.clusterGroups.resize(allGroups.size());

        for (SIZE_T gi = 0; gi < allGroups.size(); ++gi)
        {
            const BuildGroup& bg = allGroups[gi];
            const UINT32 grpClusterStart = static_cast<UINT32>(rawMeta.size());
            UINT32 minPageId = kNVGInvalidIndex;
            UINT32 maxPageId = 0u;

            for (UINT32 origCi : bg.clusterIndices)
            {
                const BuildCluster& bc = allClusters[origCi];
                if (bc.virtualPageId != kNVGInvalidIndex)
                {
                    if (minPageId == kNVGInvalidIndex || bc.virtualPageId < minPageId) minPageId = bc.virtualPageId;
                    if (bc.virtualPageId > maxPageId) maxPageId = bc.virtualPageId;
                }

                NVGClusterMetaGPUData m{};
                m.meshId                   = 0u;
                m.clusterLocalIndex        = static_cast<UINT32>(rawMeta.size());
                m.virtualPageId            = bc.virtualPageId;
                m.clusterWordOffsetInPage  = bc.wordOffsetInPage;
                m.countsAndLod             = bc.encodedWords[14];
                m.selfGroupIndex           = bc.selfGroupIndex;
                m.parentGroupIndex         = bc.parentGroupIndex;
                m.refinedByChildGroupStart = bc.childGroupStart;
                m.refinedByChildGroupCount = bc.childGroupCount;
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

            const UINT32 groupPageCount = (minPageId != kNVGInvalidIndex && maxPageId >= minPageId)
                ? (maxPageId - minPageId + 1u)
                : 1u;

            NVGClusterGroupGPUData gd{};
            gd.meshId             = 0u; // Assigned on runtime registration
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

        // 7. Build Compact BVH Over ClusterGroups so BOTH BVH leaf spans [childStart .. +childCount)
        //    AND ClusterGroup spans [clusterStart .. +clusterCount) remain 100% contiguous and exact!
        std::vector<UINT32> groupOrder(outAsset.clusterGroups.size());
        std::iota(groupOrder.begin(), groupOrder.end(), 0u);
        std::vector<UINT32> remappedGroupClusterStarts(outAsset.clusterGroups.size(), 0u);

        outAsset.clusters.clear();
        outAsset.clusters.reserve(rawMeta.size());
        outAsset.bvhNodes.clear();
        outAsset.bvhNodes.push_back(NVGHierarchyNodeGPUData{});
        outAsset.rootNodeIndex = 0u;
        BuildGroupBVHRecursive(
            0u,
            groupOrder,
            0u,
            static_cast<UINT32>(groupOrder.size()),
            0u,
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

        // Compute compression telemetry
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

        const auto tBuildEnd = std::chrono::high_resolution_clock::now();
        rep.buildTimeMs      = std::chrono::duration<FLOAT32, std::milli>(tBuildEnd - tBuildStart).count();
        rep.succeeded        = true;
        rep.diagnosticMessage = "OK";
        return true;
    }

    // =====================================================================
    //  DecodeClusterCPU — Exact Reference Decoder Matching GPU Shader Decode
    // =====================================================================

    BOOL NVGBuilder::DecodeClusterCPU(
        const UINT32*        pageWords,
        UINT32               pageWordCount,
        UINT32               clusterWordOffset,
        std::vector<Vertex>& outVertices,
        std::vector<UINT32>& outIndices)
    {
        outVertices.clear();
        outIndices.clear();
        if (!pageWords || clusterWordOffset + kNVGClusterHeaderWords > pageWordCount)
            return false;

        const UINT32* h = pageWords + clusterWordOffset;
        const Vec3 posOrigin(BitCastUIntToFloat(h[0]), BitCastUIntToFloat(h[1]), BitCastUIntToFloat(h[2]));
        const Vec3 posExtent(BitCastUIntToFloat(h[3]), BitCastUIntToFloat(h[4]), BitCastUIntToFloat(h[5]));
        const Vec2 uv0Origin(BitCastUIntToFloat(h[6]), BitCastUIntToFloat(h[7]));
        const Vec2 uv0Extent(BitCastUIntToFloat(h[8]), BitCastUIntToFloat(h[9]));
        const Vec2 uv1Origin(BitCastUIntToFloat(h[10]), BitCastUIntToFloat(h[11]));
        const Vec2 uv1Extent(BitCastUIntToFloat(h[12]), BitCastUIntToFloat(h[13]));

        const UINT32 countsAndFlags = h[14];
        const UINT32 vertCount      = countsAndFlags & 0xFFu;
        const UINT32 triCount       = (countsAndFlags >> 8u) & 0xFFu;
        const UINT32 attrFlags      = (countsAndFlags >> 24u) & 0xFFu;
        const BOOL   hasSkinning    = ((attrFlags & NVG_CLUSTER_ATTR_SKINNING) != 0u);

        const UINT32 offsetsPacked   = h[15];
        const UINT32 vertRelOffset   = offsetsPacked & 0xFFFFu;
        const UINT32 idxRelOffset    = (offsetsPacked >> 16u) & 0xFFFFu;
        const UINT32 vertStrideWords = hasSkinning ? kNVGSkinnedVertexWords : kNVGRigidVertexWords;

        if (vertCount == 0u || vertCount > kNVGMaxClusterVertices ||
            triCount == 0u || triCount > kNVGMaxClusterTriangles ||
            clusterWordOffset + vertRelOffset + vertCount * vertStrideWords > pageWordCount ||
            clusterWordOffset + idxRelOffset + triCount > pageWordCount)
        {
            return false;
        }

        outVertices.resize(vertCount);
        for (UINT32 vi = 0u; vi < vertCount; ++vi)
        {
            const UINT32* src = h + vertRelOffset + vi * vertStrideWords;
            Vertex v{};

            const UINT32 qx = src[0] & 0xFFFFu;
            const UINT32 qy = (src[0] >> 16u) & 0xFFFFu;
            const UINT32 qz = src[1] & 0xFFFFu;
            const UINT16 tanPack = static_cast<UINT16>((src[1] >> 16u) & 0xFFFFu);

            v.position = Vec3(
                posOrigin.x + (static_cast<FLOAT32>(qx) / 65535.0f) * posExtent.x,
                posOrigin.y + (static_cast<FLOAT32>(qy) / 65535.0f) * posExtent.y,
                posOrigin.z + (static_cast<FLOAT32>(qz) / 65535.0f) * posExtent.z);

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

    // =====================================================================
    //  Standalone Versioned Binary Serialization ("NVG1")
    // =====================================================================

    namespace
    {
        struct NVGBinaryHeader
        {
            UINT32 magic             = kNVGBinaryMagic;
            UINT32 version           = kNVGBinaryVersion;
            UINT32 flags             = 0u; // bit 0 = hasSkinning
            UINT32 lodLevelCount     = 0u;
            UINT32 rootPageCount     = 0u;
            UINT32 rootNodeIndex     = 0u;
            UINT32 clusterCount      = 0u;
            UINT32 groupCount        = 0u;
            UINT32 childGroupIdCount = 0u;
            UINT32 bvhNodeCount      = 0u;
            UINT32 pageCount         = 0u;
            UINT32 pageSizeBytes     = kNVGPageSizeBytes;
            Vec3   boundsMin         = Vec3::Zero();
            FLOAT32 maxQuantError    = 0.0f;
            Vec3   boundsMax         = Vec3::Zero();
            FLOAT32 compressionRatio = 1.0f;
            UINT64 payloadChecksum   = 0ull;
            UINT64 payloadByteSize   = 0ull;
        };

        struct NVGBinaryPageMeta
        {
            UINT32 localPageIndex = 0u;
            UINT32 lodLevel       = 0u;
            UINT32 isRootPage     = 0u;
            UINT32 clusterCount   = 0u;
            UINT32 usedWords      = 0u;
            UINT32 reserved       = 0u;
        };
    }

    BOOL NVGBuilder::Serialize(
        const NVGVirtualMeshAsset& asset,
        std::vector<BYTE>&         outBlob)
    {
        outBlob.clear();
        if (!asset.buildReport.succeeded || asset.clusters.empty() || asset.pages.empty())
            return false;

        std::vector<BYTE> payload;
        auto appendRaw = [&](const void* ptr, SIZE_T bytes) {
            if (bytes == 0u) return;
            const SIZE_T off = payload.size();
            payload.resize(off + bytes);
            std::memcpy(payload.data() + off, ptr, bytes);
        };

        appendRaw(asset.clusters.data(),          asset.clusters.size()          * sizeof(NVGClusterMetaGPUData));
        appendRaw(asset.clusterGroups.data(),     asset.clusterGroups.size()     * sizeof(NVGClusterGroupGPUData));
        appendRaw(asset.childGroupIndices.data(), asset.childGroupIndices.size() * sizeof(UINT32));
        appendRaw(asset.bvhNodes.data(),          asset.bvhNodes.size()          * sizeof(NVGHierarchyNodeGPUData));

        for (const NVGVirtualPagePayload& p : asset.pages)
        {
            NVGBinaryPageMeta pm{};
            pm.localPageIndex = p.localPageIndex;
            pm.lodLevel       = p.lodLevel;
            pm.isRootPage     = p.isRootPage ? 1u : 0u;
            pm.clusterCount   = p.clusterCount;
            pm.usedWords      = p.usedWords;
            appendRaw(&pm, sizeof(NVGBinaryPageMeta));
            appendRaw(p.words.data(), static_cast<SIZE_T>(p.usedWords) * sizeof(UINT32));
        }

        NVGBinaryHeader hdr{};
        hdr.magic             = kNVGBinaryMagic;
        hdr.version           = kNVGBinaryVersion;
        hdr.flags             = asset.hasSkinning ? 1u : 0u;
        hdr.lodLevelCount     = asset.lodLevelCount;
        hdr.rootPageCount     = asset.rootPageCount;
        hdr.rootNodeIndex     = asset.rootNodeIndex;
        hdr.clusterCount      = static_cast<UINT32>(asset.clusters.size());
        hdr.groupCount        = static_cast<UINT32>(asset.clusterGroups.size());
        hdr.childGroupIdCount = static_cast<UINT32>(asset.childGroupIndices.size());
        hdr.bvhNodeCount      = static_cast<UINT32>(asset.bvhNodes.size());
        hdr.pageCount         = static_cast<UINT32>(asset.pages.size());
        hdr.pageSizeBytes     = kNVGPageSizeBytes;
        hdr.boundsMin         = asset.localBounds.m_Min;
        hdr.boundsMax         = asset.localBounds.m_Max;
        hdr.maxQuantError     = asset.buildReport.maxQuantizationError;
        hdr.compressionRatio  = asset.buildReport.compressionRatio;
        hdr.payloadByteSize   = static_cast<UINT64>(payload.size());
        hdr.payloadChecksum   = ComputeFNV1a64(payload.data(), payload.size());

        outBlob.resize(sizeof(NVGBinaryHeader) + payload.size());
        std::memcpy(outBlob.data(), &hdr, sizeof(NVGBinaryHeader));
        if (!payload.empty())
        {
            std::memcpy(outBlob.data() + sizeof(NVGBinaryHeader), payload.data(), payload.size());
        }
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

        if (!data || size < sizeof(NVGBinaryHeader))
        {
            outError = "NVG binary blob truncated before header.";
            return false;
        }

        NVGBinaryHeader hdr{};
        std::memcpy(&hdr, data, sizeof(NVGBinaryHeader));

        if (hdr.magic != kNVGBinaryMagic)
        {
            outError = "NVG binary magic mismatch (expected 'NVG1').";
            return false;
        }
        if (hdr.version != kNVGBinaryVersion)
        {
            outError = "NVG binary version unsupported.";
            return false;
        }
        if (hdr.pageSizeBytes != kNVGPageSizeBytes || hdr.clusterCount == 0u || hdr.pageCount == 0u ||
            hdr.rootPageCount == 0u || hdr.rootPageCount > hdr.pageCount ||
            hdr.lodLevelCount == 0u || hdr.lodLevelCount > kNVGMaxLODLevels ||
            hdr.groupCount == 0u || hdr.bvhNodeCount == 0u || hdr.rootNodeIndex >= hdr.bvhNodeCount)
        {
            outError = "NVG binary header contains invalid geometry page, LOD, group, BVH, or cluster counts.";
            return false;
        }
        if (hdr.payloadByteSize > size || sizeof(NVGBinaryHeader) + hdr.payloadByteSize != size)
        {
            outError = "NVG binary payload byte size mismatch.";
            return false;
        }

        // Pre-validate required minimum payload bytes in 64-bit arithmetic BEFORE any vector allocation
        // so corrupt headers cannot trigger OOM or integer overflow.
        const UINT64 clustersBytes = static_cast<UINT64>(hdr.clusterCount)      * sizeof(NVGClusterMetaGPUData);
        const UINT64 groupsBytes   = static_cast<UINT64>(hdr.groupCount)        * sizeof(NVGClusterGroupGPUData);
        const UINT64 childBytes    = static_cast<UINT64>(hdr.childGroupIdCount) * sizeof(UINT32);
        const UINT64 nodesBytes    = static_cast<UINT64>(hdr.bvhNodeCount)      * sizeof(NVGHierarchyNodeGPUData);
        const UINT64 pageHdrBytes  = static_cast<UINT64>(hdr.pageCount)         * sizeof(NVGBinaryPageMeta);
        if (clustersBytes > hdr.payloadByteSize ||
            groupsBytes   > hdr.payloadByteSize - clustersBytes ||
            childBytes    > hdr.payloadByteSize - clustersBytes - groupsBytes ||
            nodesBytes    > hdr.payloadByteSize - clustersBytes - groupsBytes - childBytes ||
            pageHdrBytes  > hdr.payloadByteSize - clustersBytes - groupsBytes - childBytes - nodesBytes)
        {
            outError = "NVG binary header table sizes exceed payload byte size.";
            return false;
        }

        const BYTE* payloadPtr = data + sizeof(NVGBinaryHeader);
        const UINT64 actualChecksum = ComputeFNV1a64(payloadPtr, static_cast<SIZE_T>(hdr.payloadByteSize));
        if (actualChecksum != hdr.payloadChecksum)
        {
            outError = "NVG binary checksum mismatch (corrupted data).";
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
            outError = "NVG binary payload truncated in metadata tables.";
            return false;
        }

        outAsset.pages.resize(hdr.pageCount);
        for (UINT32 pi = 0u; pi < hdr.pageCount; ++pi)
        {
            NVGBinaryPageMeta pm{};
            if (!readRaw(&pm, sizeof(NVGBinaryPageMeta)) || pm.usedWords > kNVGPageWordCount)
            {
                outError = "NVG binary payload truncated or invalid page metadata.";
                return false;
            }
            NVGVirtualPagePayload& p = outAsset.pages[pi];
            p.localPageIndex = pm.localPageIndex;
            p.lodLevel       = pm.lodLevel;
            p.isRootPage     = (pm.isRootPage != 0u);
            p.clusterCount   = pm.clusterCount;
            p.usedWords      = pm.usedWords;
            p.words.assign(kNVGPageWordCount, 0u);
            if (!readRaw(p.words.data(), static_cast<SIZE_T>(pm.usedWords) * sizeof(UINT32)))
            {
                outError = "NVG binary payload truncated in compressed page words.";
                return false;
            }
        }

        // Post-load structural integrity validation (prevent out-of-bounds GPU page/node/group indexing)
        for (const NVGClusterMetaGPUData& cm : outAsset.clusters)
        {
            if (cm.virtualPageId >= hdr.pageCount ||
                cm.clusterWordOffsetInPage + 16u > kNVGPageWordCount ||
                cm.clusterWordOffsetInPage + 16u > outAsset.pages[cm.virtualPageId].usedWords ||
                cm.selfGroupIndex >= hdr.groupCount ||
                (cm.parentGroupIndex != kNVGInvalidIndex && cm.parentGroupIndex >= hdr.groupCount))
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG binary contains out-of-bounds cluster page, offset, or group index.";
                return false;
            }
        }
        for (const NVGClusterGroupGPUData& grp : outAsset.clusterGroups)
        {
            const UINT32 grpPageCnt = UnpackNVGGroupPageCount(grp.lodLevel);
            if (static_cast<UINT64>(grp.clusterStart) + static_cast<UINT64>(grp.clusterCount) > hdr.clusterCount ||
                (grp.parentGroupIndex != kNVGInvalidIndex && grp.parentGroupIndex >= hdr.groupCount) ||
                (grp.firstVirtualPageId != kNVGInvalidIndex &&
                 (static_cast<UINT64>(grp.firstVirtualPageId) + static_cast<UINT64>(grpPageCnt) > hdr.pageCount)))
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG binary contains out-of-bounds cluster group range or page index.";
                return false;
            }
        }
        for (const NVGHierarchyNodeGPUData& node : outAsset.bvhNodes)
        {
            const UINT32 childCnt = node.childCountAndFlags & 0xFFFFu;
            const BOOL   isLeaf   = ((node.childCountAndFlags & 65536u) != 0u);
            const UINT64 childEnd = static_cast<UINT64>(node.childStart) + static_cast<UINT64>(childCnt);
            if (isLeaf ? (childEnd > hdr.clusterCount) : (childEnd > hdr.bvhNodeCount))
            {
                outAsset = NVGVirtualMeshAsset{};
                outError = "NVG binary BVH node references out-of-bounds child range.";
                return false;
            }
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
