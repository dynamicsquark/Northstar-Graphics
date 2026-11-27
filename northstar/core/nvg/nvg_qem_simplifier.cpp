#include "nvg_qem_simplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace Northstar
{
    namespace
    {
        inline constexpr UINT32 kNumScalarWedgeAttrs = 14u; // nx,ny,nz, tx,ty,tz, hand, u0,v0, u1,v1, r,g,b

        inline FLOAT32 ClampFloat(FLOAT32 v, FLOAT32 lo, FLOAT32 hi) noexcept
        {
            return (v < lo) ? lo : ((v > hi) ? hi : v);
        }

        inline UINT64 MakeEdgeKeyU64(UINT32 a, UINT32 b) noexcept
        {
            const UINT32 lo = (a < b) ? a : b;
            const UINT32 hi = (a < b) ? b : a;
            return (static_cast<UINT64>(lo) << 32u) | static_cast<UINT64>(hi);
        }

        inline FLOAT32 ComputeTangentHandSign(const Vertex& v) noexcept
        {
            return (v.normal.Cross(v.tangent).Dot(v.bitangent) < 0.0f) ? -1.0f : 1.0f;
        }

        inline INT8 ComputeTriangleUVOrientationSign(const Vertex& v0, const Vertex& v1, const Vertex& v2) noexcept
        {
            const double du1 = static_cast<double>(v1.texCoord0.x) - static_cast<double>(v0.texCoord0.x);
            const double dv1 = static_cast<double>(v1.texCoord0.y) - static_cast<double>(v0.texCoord0.y);
            const double du2 = static_cast<double>(v2.texCoord0.x) - static_cast<double>(v0.texCoord0.x);
            const double dv2 = static_cast<double>(v2.texCoord0.y) - static_cast<double>(v0.texCoord0.y);
            const double area2 = du1 * dv2 - dv1 * du2;
            if (std::fabs(area2) <= 1e-12)
            {
                return 0;
            }
            return (area2 > 0.0) ? static_cast<INT8>(1) : static_cast<INT8>(-1);
        }

        // Pivoted 3x3 LUP solver with 1-step Newton residual refinement in double precision
        inline BOOL Solve3x3LUPWithRefinement(
            const double A[3][3],
            const double b[3],
            double       outX[3]) noexcept
        {
            double M[3][3];
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    M[r][c] = A[r][c];
                }
            }
            int piv[3] = { 0, 1, 2 };

            for (int k = 0; k < 3; ++k)
            {
                int    bestRow = k;
                double bestVal = std::fabs(M[k][k]);
                for (int r = k + 1; r < 3; ++r)
                {
                    const double v = std::fabs(M[r][k]);
                    if (v > bestVal)
                    {
                        bestVal = v;
                        bestRow = r;
                    }
                }
                if (bestVal < 1e-13)
                {
                    return false;
                }
                if (bestRow != k)
                {
                    std::swap(piv[k], piv[bestRow]);
                    for (int c = 0; c < 3; ++c)
                    {
                        std::swap(M[k][c], M[bestRow][c]);
                    }
                }
                const double invDiag = 1.0 / M[k][k];
                for (int r = k + 1; r < 3; ++r)
                {
                    M[r][k] *= invDiag;
                    for (int c = k + 1; c < 3; ++c)
                    {
                        M[r][c] -= M[r][k] * M[k][c];
                    }
                }
            }

            auto backSubstitute = [&](const double rhs[3], double sol[3]) noexcept {
                double y[3];
                for (int i = 0; i < 3; ++i)
                {
                    double sum = rhs[piv[i]];
                    for (int j = 0; j < i; ++j)
                    {
                        sum -= M[i][j] * y[j];
                    }
                    y[i] = sum;
                }
                for (int i = 2; i >= 0; --i)
                {
                    double sum = y[i];
                    for (int j = i + 1; j < 3; ++j)
                    {
                        sum -= M[i][j] * sol[j];
                    }
                    sol[i] = sum / M[i][i];
                }
            };

            backSubstitute(b, outX);
            if (!std::isfinite(outX[0]) || !std::isfinite(outX[1]) || !std::isfinite(outX[2]))
            {
                return false;
            }

            // 1-step iterative residual refinement: dx solves A * dx = b - A * x
            double res[3];
            for (int r = 0; r < 3; ++r)
            {
                res[r] = b[r] - (A[r][0] * outX[0] + A[r][1] * outX[1] + A[r][2] * outX[2]);
            }
            double dx[3]{};
            backSubstitute(res, dx);
            if (std::isfinite(dx[0]) && std::isfinite(dx[1]) && std::isfinite(dx[2]))
            {
                outX[0] += dx[0];
                outX[1] += dx[1];
                outX[2] += dx[2];
            }
            return std::isfinite(outX[0]) && std::isfinite(outX[1]) && std::isfinite(outX[2]);
        }

        // Rebased Symmetric 4x4 Quadric (stores coupled geometric + Schur-complement wedge attribute terms)
        struct CoupledSpatialQuadric
        {
            double a00 = 0.0, a01 = 0.0, a02 = 0.0, b0 = 0.0;
            double            a11 = 0.0, a12 = 0.0, b1 = 0.0;
            double                       a22 = 0.0, b2 = 0.0;
            double                                  c  = 0.0;
            // Volumetric mean normal constraint for volume-preserving 4x4 Lagrange stage
            double vn0 = 0.0, vn1 = 0.0, vn2 = 0.0, vd = 0.0;

            void AddSymmetricPlane(double nx, double ny, double nz, double d, double w) noexcept
            {
                a00 += w * nx * nx; a01 += w * nx * ny; a02 += w * nx * nz; b0 += w * nx * d;
                a11 += w * ny * ny; a12 += w * ny * nz; b1 += w * ny * d;
                a22 += w * nz * nz; b2 += w * nz * d;
                c   += w * d  * d;
            }

            void AddVolumePlane(double nx, double ny, double nz, double d, double area) noexcept
            {
                vn0 += area * nx;
                vn1 += area * ny;
                vn2 += area * nz;
                vd  += area * d;
            }

            // Schur-complement projection of an in-plane attribute gradient g over a single triangle of area A:
            // Minimizing A * (g . (x - p_centroid))^2 adds A * (g g^T) around p_centroid (zero mean shift at centroid)
            // so curved attribute fields resist tangential sliding without biasing flat planar positions!
            void AddAttributeGradientAroundPoint(
                const double g[3],
                const double refPt[3],
                double       weightSqArea) noexcept
            {
                const double d = -(g[0] * refPt[0] + g[1] * refPt[1] + g[2] * refPt[2]);
                AddSymmetricPlane(g[0], g[1], g[2], d, weightSqArea);
            }

            CoupledSpatialQuadric operator+(const CoupledSpatialQuadric& o) const noexcept
            {
                CoupledSpatialQuadric r{};
                r.a00 = a00 + o.a00; r.a01 = a01 + o.a01; r.a02 = a02 + o.a02; r.b0 = b0 + o.b0;
                r.a11 = a11 + o.a11; r.a12 = a12 + o.a12; r.b1 = b1 + o.b1;
                r.a22 = a22 + o.a22; r.b2 = b2 + o.b2;
                r.c   = c   + o.c;
                r.vn0 = vn0 + o.vn0; r.vn1 = vn1 + o.vn1; r.vn2 = vn2 + o.vn2; r.vd = vd + o.vd;
                return r;
            }

            double Evaluate(double x, double y, double z) const noexcept
            {
                const double val =
                    a00 * x * x + 2.0 * a01 * x * y + 2.0 * a02 * x * z + 2.0 * b0 * x +
                    a11 * y * y + 2.0 * a12 * y * z + 2.0 * b1 * y +
                    a22 * z * z + 2.0 * b2  * z +
                    c;
                return (val > 0.0) ? val : 0.0;
            }

            // 4-stage solver cascade:
            //   Stage 1: Volume-preserving Lagrange-multiplier constrained 3x3 Schur reduction
            //   Stage 2: Unconstrained 3x3 LUP + 1-step Newton refinement
            //   Stage 3: Exact 1D analytic quadratic minimum along segment [pFrom, pTo]
            //   Stage 4: Endpoints / midpoint fallback
            void SolveCascade(
                const double pFrom[3],
                const double pTo[3],
                double       maxRadiusFromMid,
                BOOL         restrictToSegment,
                double       outBest[3],
                double&      outCost,
                FLOAT32&     outSegT) const noexcept
            {
                outBest[0] = pTo[0];
                outBest[1] = pTo[1];
                outBest[2] = pTo[2];
                outCost    = Evaluate(pTo[0], pTo[1], pTo[2]);
                outSegT    = 1.0f;

                const double cFrom = Evaluate(pFrom[0], pFrom[1], pFrom[2]);
                if (cFrom < outCost)
                {
                    outCost    = cFrom;
                    outBest[0] = pFrom[0];
                    outBest[1] = pFrom[1];
                    outBest[2] = pFrom[2];
                    outSegT    = 0.0f;
                }

                const double pMid[3] = {
                    0.5 * (pFrom[0] + pTo[0]),
                    0.5 * (pFrom[1] + pTo[1]),
                    0.5 * (pFrom[2] + pTo[2])
                };
                const double cMid = Evaluate(pMid[0], pMid[1], pMid[2]);
                if (cMid < outCost)
                {
                    outCost    = cMid;
                    outBest[0] = pMid[0];
                    outBest[1] = pMid[1];
                    outBest[2] = pMid[2];
                    outSegT    = 0.5f;
                }

                // Stage 3: Analytic 1D minimum on [pFrom, pTo]: f(t) = q(pFrom + t * d)
                const double dx = pTo[0] - pFrom[0];
                const double dy = pTo[1] - pFrom[1];
                const double dz = pTo[2] - pFrom[2];
                const double segLenSq = dx * dx + dy * dy + dz * dz;
                const double Ad0 = a00 * dx + a01 * dy + a02 * dz;
                const double Ad1 = a01 * dx + a11 * dy + a12 * dz;
                const double Ad2 = a02 * dx + a12 * dy + a22 * dz;
                const double denom1D = dx * Ad0 + dy * Ad1 + dz * Ad2;
                if (denom1D > 1e-14)
                {
                    const double gFrom0 = a00 * pFrom[0] + a01 * pFrom[1] + a02 * pFrom[2] + b0;
                    const double gFrom1 = a01 * pFrom[0] + a11 * pFrom[1] + a12 * pFrom[2] + b1;
                    const double gFrom2 = a02 * pFrom[0] + a12 * pFrom[1] + a22 * pFrom[2] + b2;
                    const double tOpt1D = -((dx * gFrom0 + dy * gFrom1 + dz * gFrom2) / denom1D);
                    if (tOpt1D > 0.0 && tOpt1D < 1.0)
                    {
                        const double cand1D[3] = {
                            pFrom[0] + tOpt1D * dx,
                            pFrom[1] + tOpt1D * dy,
                            pFrom[2] + tOpt1D * dz
                        };
                        const double c1D = Evaluate(cand1D[0], cand1D[1], cand1D[2]);
                        if (c1D < outCost)
                        {
                            outCost    = c1D;
                            outBest[0] = cand1D[0];
                            outBest[1] = cand1D[1];
                            outBest[2] = cand1D[2];
                            outSegT    = static_cast<FLOAT32>(tOpt1D);
                        }
                    }
                }

                if (restrictToSegment)
                {
                    return;
                }

                const double A[3][3] = {
                    { a00, a01, a02 },
                    { a01, a11, a12 },
                    { a02, a12, a22 }
                };
                const double rhs[3] = { -b0, -b1, -b2 };

                auto tryAccept3D = [&](const double sol[3]) noexcept -> BOOL {
                    const double rx = sol[0] - pMid[0];
                    const double ry = sol[1] - pMid[1];
                    const double rz = sol[2] - pMid[2];
                    if ((rx * rx + ry * ry + rz * rz) > maxRadiusFromMid * maxRadiusFromMid)
                    {
                        return false;
                    }
                    const double cSol = Evaluate(sol[0], sol[1], sol[2]);
                    if (cSol < outCost)
                    {
                        outCost    = cSol;
                        outBest[0] = sol[0];
                        outBest[1] = sol[1];
                        outBest[2] = sol[2];
                        if (segLenSq > 1e-16)
                        {
                            const double proj = ((sol[0] - pFrom[0]) * dx +
                                                 (sol[1] - pFrom[1]) * dy +
                                                 (sol[2] - pFrom[2]) * dz) / segLenSq;
                            outSegT = ClampFloat(static_cast<FLOAT32>(proj), 0.0f, 1.0f);
                        }
                        return true;
                    }
                    return false;
                };

                // Stage 2: Unconstrained 3x3 LUP + residual refinement
                double xUnc[3]{};
                if (Solve3x3LUPWithRefinement(A, rhs, xUnc))
                {
                    // Stage 1: Volume-preserving Lagrange correction along A^{-1} * vn when vn . x + vd != 0
                    const double vnNormSq = vn0 * vn0 + vn1 * vn1 + vn2 * vn2;
                    if (vnNormSq > 1e-16)
                    {
                        const double vnVec[3] = { vn0, vn1, vn2 };
                        double invAvn[3]{};
                        if (Solve3x3LUPWithRefinement(A, vnVec, invAvn))
                        {
                            const double s = vn0 * invAvn[0] + vn1 * invAvn[1] + vn2 * invAvn[2];
                            if (s > 1e-14)
                            {
                                const double viol = (vn0 * xUnc[0] + vn1 * xUnc[1] + vn2 * xUnc[2] + vd) / s;
                                const double xVol[3] = {
                                    xUnc[0] - invAvn[0] * viol,
                                    xUnc[1] - invAvn[1] * viol,
                                    xUnc[2] - invAvn[2] * viol
                                };
                                tryAccept3D(xVol);
                            }
                        }
                    }
                    tryAccept3D(xUnc);
                }
            }
        };

        struct WedgeAccumGradient
        {
            double g[kNumScalarWedgeAttrs][3]{};
            double areaSum = 0.0;
        };

        inline void ExtractVertexScalarAttrs(const Vertex& v, FLOAT32 handSign, double outA[kNumScalarWedgeAttrs]) noexcept
        {
            outA[0]  = v.normal.x;    outA[1]  = v.normal.y;    outA[2]  = v.normal.z;
            outA[3]  = v.tangent.x;   outA[4]  = v.tangent.y;   outA[5]  = v.tangent.z;
            outA[6]  = handSign;
            outA[7]  = v.texCoord0.x; outA[8]  = v.texCoord0.y;
            outA[9]  = v.texCoord1.x; outA[10] = v.texCoord1.y;
            outA[11] = v.color.x;     outA[12] = v.color.y;     outA[13] = v.color.z;
        }
    }

    FLOAT32 NVGQEMSimplifier::SimplifyGroup(
        const NVGQEMSimplifierInput& input,
        std::vector<Vertex>&         outVertices,
        std::vector<UINT32>&         outSpatialIds,
        std::vector<NVGQEMTriangle>& outTriangles)
    {
        if (!input.vertices || !input.spatialIds || !input.triangles)
        {
            outVertices.clear();
            outSpatialIds.clear();
            outTriangles.clear();
            return 0.0f;
        }

        outVertices   = *input.vertices;
        outSpatialIds = *input.spatialIds;
        outTriangles  = *input.triangles;

        if (outTriangles.size() <= input.targetTriangleCount || outVertices.size() < 4u)
        {
            return 0.0f;
        }

        const UINT32 numVerts = static_cast<UINT32>(outVertices.size());
        const UINT32 numTris  = static_cast<UINT32>(outTriangles.size());

        // 1. Deterministic compact local spatial ID table (preserving first-seen order without hash-iteration order)
        std::vector<std::pair<UINT32, UINT32>> globalSidVertPairs(numVerts);
        for (UINT32 vi = 0u; vi < numVerts; ++vi)
        {
            globalSidVertPairs[vi] = { outSpatialIds[vi], vi };
        }
        std::sort(globalSidVertPairs.begin(), globalSidVertPairs.end());

        std::vector<UINT32>              localToGlobalSid;
        std::vector<std::array<double, 3>> spatialPosD;
        std::vector<std::vector<UINT32>> spatialWedges;
        std::vector<UINT32>              vertLocalSid(numVerts, 0u);
        std::vector<UINT8>               vertAlive(numVerts, 1u);
        std::vector<FLOAT32>             vertHandSign(numVerts, 1.0f);
        std::vector<INT8>                vertUvOrient(numVerts, 0);

        localToGlobalSid.reserve(numVerts);
        spatialPosD.reserve(numVerts);
        spatialWedges.reserve(numVerts);

        for (const auto& [gSid, vi] : globalSidVertPairs)
        {
            UINT32 lSid = 0u;
            if (localToGlobalSid.empty() || localToGlobalSid.back() != gSid)
            {
                lSid = static_cast<UINT32>(localToGlobalSid.size());
                localToGlobalSid.push_back(gSid);
                const Vec3& p = outVertices[vi].position;
                spatialPosD.push_back({ static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z) });
                spatialWedges.emplace_back();
            }
            else
            {
                lSid = static_cast<UINT32>(localToGlobalSid.size() - 1u);
            }
            vertLocalSid[vi] = lSid;
            vertHandSign[vi] = ComputeTangentHandSign(outVertices[vi]);
            spatialWedges[lSid].push_back(vi);
        }

        const UINT32 numSpatial = static_cast<UINT32>(localToGlobalSid.size());
        if (numSpatial < 3u)
        {
            return 0.0f;
        }

        // 2. Lossless IEEE-754 2^E exponent normalization + local centroid rebasing (0 ULP mantissa preservation!)
        double cen[3] = { 0.0, 0.0, 0.0 };
        for (UINT32 s = 0u; s < numSpatial; ++s)
        {
            cen[0] += spatialPosD[s][0];
            cen[1] += spatialPosD[s][1];
            cen[2] += spatialPosD[s][2];
        }
        cen[0] /= static_cast<double>(numSpatial);
        cen[1] /= static_cast<double>(numSpatial);
        cen[2] /= static_cast<double>(numSpatial);
        // Snap centroid to power-of-two friendly float32 so subtracting cen is exact on float32 inputs
        const double qemOrigin[3] = {
            static_cast<double>(static_cast<FLOAT32>(cen[0])),
            static_cast<double>(static_cast<FLOAT32>(cen[1])),
            static_cast<double>(static_cast<FLOAT32>(cen[2]))
        };

        double edgeLenSum   = 0.0;
        double uvEdgeLenSum = 0.0;
        UINT32 edgeSamples  = 0u;
        for (UINT32 ti = 0u; ti < numTris; ++ti)
        {
            const NVGQEMTriangle& tr = outTriangles[ti];
            const INT8 uvSign = ComputeTriangleUVOrientationSign(outVertices[tr.v[0]], outVertices[tr.v[1]], outVertices[tr.v[2]]);
            if (uvSign != 0)
            {
                for (int k = 0; k < 3; ++k)
                {
                    if (vertUvOrient[tr.v[k]] == 0)
                        vertUvOrient[tr.v[k]] = uvSign;
                }
            }
            const UINT32 s0 = vertLocalSid[tr.v[0]];
            const UINT32 s1 = vertLocalSid[tr.v[1]];
            const UINT32 s2 = vertLocalSid[tr.v[2]];
            if (s0 == s1 || s1 == s2 || s2 == s0) continue;

            for (int e = 0; e < 3; ++e)
            {
                const UINT32 va = tr.v[e];
                const UINT32 vb = tr.v[(e + 1) % 3];
                const UINT32 sa = vertLocalSid[va];
                const UINT32 sb = vertLocalSid[vb];
                const double dx = spatialPosD[sb][0] - spatialPosD[sa][0];
                const double dy = spatialPosD[sb][1] - spatialPosD[sa][1];
                const double dz = spatialPosD[sb][2] - spatialPosD[sa][2];
                edgeLenSum   += std::sqrt(dx * dx + dy * dy + dz * dz);
                uvEdgeLenSum += static_cast<double>((outVertices[vb].texCoord0 - outVertices[va].texCoord0).Length());
                ++edgeSamples;
            }
        }

        const double avgEdgeWorld = (edgeSamples > 0u && edgeLenSum > 1e-12)
            ? (edgeLenSum / static_cast<double>(edgeSamples))
            : 1.0;
        int normExp = 0;
        std::frexp(0.25 / avgEdgeWorld, &normExp);
        normExp = std::clamp(normExp - 1, -40, 40);
        const double scaleToNorm   = std::ldexp(1.0, normExp);   // Exact 2^normExp
        const double scaleToWorld  = std::ldexp(1.0, -normExp);  // Exact 2^-normExp

        for (UINT32 s = 0u; s < numSpatial; ++s)
        {
            spatialPosD[s][0] = (spatialPosD[s][0] - qemOrigin[0]) * scaleToNorm;
            spatialPosD[s][1] = (spatialPosD[s][1] - qemOrigin[1]) * scaleToNorm;
            spatialPosD[s][2] = (spatialPosD[s][2] - qemOrigin[2]) * scaleToNorm;
        }

        const double avgTriUvSpan = (edgeSamples > 0u)
            ? (std::max)(1e-4, uvEdgeLenSum / static_cast<double>(edgeSamples))
            : 0.05;

        // 3. Deterministic edge topology analysis: open boundaries, >2 non-manifold junctions, and multi-material seams
        struct EdgeTopologyRecord
        {
            UINT64 key          = 0ull;
            UINT32 incidence    = 0u;
            UINT16 firstMatSlot = 0u;
            BOOL   multiMatSeam = false;
        };
        std::vector<EdgeTopologyRecord> edgeRecords;
        edgeRecords.reserve(static_cast<SIZE_T>(numTris) * 3u);
        for (UINT32 ti = 0u; ti < numTris; ++ti)
        {
            const NVGQEMTriangle& t = outTriangles[ti];
            const UINT32 s[3] = { vertLocalSid[t.v[0]], vertLocalSid[t.v[1]], vertLocalSid[t.v[2]] };
            if (s[0] == s[1] || s[1] == s[2] || s[2] == s[0]) continue;
            for (int e = 0; e < 3; ++e)
            {
                EdgeTopologyRecord rec{};
                rec.key          = MakeEdgeKeyU64(s[e], s[(e + 1) % 3]);
                rec.incidence    = 1u;
                rec.firstMatSlot = t.materialSlot;
                edgeRecords.push_back(rec);
            }
        }
        std::sort(edgeRecords.begin(), edgeRecords.end(), [](const EdgeTopologyRecord& a, const EdgeTopologyRecord& b) {
            if (a.key != b.key) return a.key < b.key;
            return a.firstMatSlot < b.firstMatSlot;
        });

        std::vector<EdgeTopologyRecord> uniqueEdges;
        uniqueEdges.reserve(edgeRecords.size());
        for (const EdgeTopologyRecord& r : edgeRecords)
        {
            if (uniqueEdges.empty() || uniqueEdges.back().key != r.key)
            {
                uniqueEdges.push_back(r);
            }
            else
            {
                EdgeTopologyRecord& back = uniqueEdges.back();
                back.incidence++;
                if (back.firstMatSlot != r.firstMatSlot)
                {
                    back.multiMatSeam = true;
                }
            }
        }

        auto findUniqueEdge = [&](UINT64 key) noexcept -> const EdgeTopologyRecord* {
            auto it = std::lower_bound(
                uniqueEdges.begin(), uniqueEdges.end(), key,
                [](const EdgeTopologyRecord& e, UINT64 k) { return e.key < k; });
            return (it != uniqueEdges.end() && it->key == key) ? &(*it) : nullptr;
        };

        std::vector<UINT8>  spatialLocked(numSpatial, 0u);
        std::vector<UINT8>  spatialOpenBoundary(numSpatial, 0u);
        std::vector<UINT8>  spatialNonManifold(numSpatial, 0u);
        std::vector<UINT8>  spatialMatSeam(numSpatial, 0u);
        std::vector<UINT8>  spatialAlive(numSpatial, 1u);
        std::vector<UINT32> spatialVersion(numSpatial, 1u);

        if (input.sortedLockedSpatialIds != nullptr && !input.sortedLockedSpatialIds->empty())
        {
            const auto& lockedVec = *input.sortedLockedSpatialIds;
            for (UINT32 s = 0u; s < numSpatial; ++s)
            {
                if (std::binary_search(lockedVec.begin(), lockedVec.end(), localToGlobalSid[s]))
                {
                    spatialLocked[s] = 1u;
                }
            }
        }

        for (const EdgeTopologyRecord& ue : uniqueEdges)
        {
            const UINT32 sa = static_cast<UINT32>(ue.key >> 32u);
            const UINT32 sb = static_cast<UINT32>(ue.key & 0xFFFFFFFFu);
            if (ue.incidence == 1u)
            {
                spatialOpenBoundary[sa] = 1u;
                spatialOpenBoundary[sb] = 1u;
            }
            else if (ue.incidence > 2u)
            {
                spatialNonManifold[sa] = 1u;
                spatialNonManifold[sb] = 1u;
            }
            if (ue.multiMatSeam)
            {
                spatialMatSeam[sa] = 1u;
                spatialMatSeam[sb] = 1u;
            }
        }

        // 4. Accumulate Coupled Geometric + Schur-Complement Wedge Attribute Quadrics
        const double attrScale = input.allowCoarseRidgeFlatten ? 0.25 : (input.relaxAttributeSeams ? 0.5 : 1.0);
        const double wNormal   = 0.04 * static_cast<double>((std::max)(0.25f, input.config.normalPreservationWeight)) * attrScale;
        const double wTangent  = 0.015 * attrScale;
        const double wHand     = 0.08 * attrScale;
        const double wUv       = (0.015 / avgTriUvSpan) * static_cast<double>((std::max)(0.25f, input.config.uvSeamWeight) * 0.125f) * attrScale;
        const double wColor    = 0.02 * attrScale;
        const double attrWeights[kNumScalarWedgeAttrs] = {
            wNormal, wNormal, wNormal,
            wTangent, wTangent, wTangent,
            wHand,
            wUv, wUv,
            wUv * 0.5, wUv * 0.5,
            wColor, wColor, wColor
        };

        std::vector<CoupledSpatialQuadric> spatialQuadrics(numSpatial);
        std::vector<CoupledSpatialQuadric> spatialPureGeomQuadrics(numSpatial);
        std::vector<double>                spatialGeomWeightSum(numSpatial, 0.0);
        std::vector<WedgeAccumGradient>    wedgeGrads(numVerts);
        std::vector<std::vector<UINT32>>   spatialTris(numSpatial);
        std::vector<UINT8>                 triAlive(numTris, 1u);
        std::vector<std::array<double, 3>> origTriNormals(numTris, { 0.0, 0.0, 0.0 });
        std::vector<UINT8>                 triHasOrigNormal(numTris, 0u);
        std::vector<UINT8>                 triHasConsistentVertN(numTris, 0u);

        for (UINT32 ti = 0u; ti < numTris; ++ti)
        {
            const NVGQEMTriangle& t = outTriangles[ti];
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

            const double* p0 = spatialPosD[s0].data();
            const double* p1 = spatialPosD[s1].data();
            const double* p2 = spatialPosD[s2].data();

            const double e01[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
            const double e02[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
            const double cr[3]  = {
                e01[1] * e02[2] - e01[2] * e02[1],
                e01[2] * e02[0] - e01[0] * e02[2],
                e01[0] * e02[1] - e01[1] * e02[0]
            };
            const double crLen = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
            if (crLen <= 1e-13)
            {
                continue;
            }

            const double invCr = 1.0 / crLen;
            const double n[3]  = { cr[0] * invCr, cr[1] * invCr, cr[2] * invCr };
            origTriNormals[ti]   = { n[0], n[1], n[2] };
            triHasOrigNormal[ti] = 1u;
            {
                const Vec3 avgVn = outVertices[t.v[0]].normal + outVertices[t.v[1]].normal + outVertices[t.v[2]].normal;
                const double vnLen = std::sqrt(
                    static_cast<double>(avgVn.x) * avgVn.x +
                    static_cast<double>(avgVn.y) * avgVn.y +
                    static_cast<double>(avgVn.z) * avgVn.z);
                if (vnLen > 1e-6)
                {
                    const double vDot = (avgVn.x * n[0] + avgVn.y * n[1] + avgVn.z * n[2]) / vnLen;
                    if (vDot >= 0.20)
                    {
                        triHasConsistentVertN[ti] = 1u;
                    }
                }
            }
            const double area  = 0.5 * crLen;
            const double d     = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);

            spatialQuadrics[s0].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialQuadrics[s1].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialQuadrics[s2].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialQuadrics[s0].AddVolumePlane(n[0], n[1], n[2], d, area);
            spatialQuadrics[s1].AddVolumePlane(n[0], n[1], n[2], d, area);
            spatialQuadrics[s2].AddVolumePlane(n[0], n[1], n[2], d, area);

            spatialPureGeomQuadrics[s0].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialPureGeomQuadrics[s1].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialPureGeomQuadrics[s2].AddSymmetricPlane(n[0], n[1], n[2], d, area);
            spatialGeomWeightSum[s0] += area;
            spatialGeomWeightSum[s1] += area;
            spatialGeomWeightSum[s2] += area;

            // Solve in-plane attribute gradients [e01^T; e02^T; n^T] * g_m = [a1 - a0; a2 - a0; 0] via 3x3 LUP + refinement
            const double J[3][3] = {
                { e01[0], e01[1], e01[2] },
                { e02[0], e02[1], e02[2] },
                { n[0],   n[1],   n[2]   }
            };
            double a0[kNumScalarWedgeAttrs], a1[kNumScalarWedgeAttrs], a2[kNumScalarWedgeAttrs];
            ExtractVertexScalarAttrs(outVertices[t.v[0]], vertHandSign[t.v[0]], a0);
            ExtractVertexScalarAttrs(outVertices[t.v[1]], vertHandSign[t.v[1]], a1);
            ExtractVertexScalarAttrs(outVertices[t.v[2]], vertHandSign[t.v[2]], a2);

            const double triCentroid[3] = {
                (p0[0] + p1[0] + p2[0]) * (1.0 / 3.0),
                (p0[1] + p1[1] + p2[1]) * (1.0 / 3.0),
                (p0[2] + p1[2] + p2[2]) * (1.0 / 3.0)
            };

            for (UINT32 m = 0u; m < kNumScalarWedgeAttrs; ++m)
            {
                const double da1 = a1[m] - a0[m];
                const double da2 = a2[m] - a0[m];
                if (std::fabs(da1) <= 1e-10 && std::fabs(da2) <= 1e-10)
                {
                    continue;
                }
                const double rhsAttr[3] = { da1, da2, 0.0 };
                double gm[3]{};
                if (Solve3x3LUPWithRefinement(J, rhsAttr, gm))
                {
                    const double wSqArea = (attrWeights[m] * attrWeights[m]) * area * (1.0 / 3.0);
                    spatialQuadrics[s0].AddAttributeGradientAroundPoint(gm, triCentroid, wSqArea);
                    spatialQuadrics[s1].AddAttributeGradientAroundPoint(gm, triCentroid, wSqArea);
                    spatialQuadrics[s2].AddAttributeGradientAroundPoint(gm, triCentroid, wSqArea);

                    for (int k = 0; k < 3; ++k)
                    {
                        WedgeAccumGradient& wg = wedgeGrads[t.v[k]];
                        wg.g[m][0] += gm[0] * area;
                        wg.g[m][1] += gm[1] * area;
                        wg.g[m][2] += gm[2] * area;
                    }
                }
            }
            for (int k = 0; k < 3; ++k)
            {
                wedgeGrads[t.v[k]].areaSum += area;
            }

            // Add perpendicular boundary/seam stabilization planes for open edges, multi-material seams, and non-manifold edges
            const UINT32 sTri[3] = { s0, s1, s2 };
            for (int e = 0; e < 3; ++e)
            {
                const UINT32 sa = sTri[e];
                const UINT32 sb = sTri[(e + 1) % 3];
                const EdgeTopologyRecord* ue = findUniqueEdge(MakeEdgeKeyU64(sa, sb));
                if (!ue) continue;
                if (ue->incidence == 1u || ue->incidence > 2u || ue->multiMatSeam)
                {
                    const double ex = spatialPosD[sb][0] - spatialPosD[sa][0];
                    const double ey = spatialPosD[sb][1] - spatialPosD[sa][1];
                    const double ez = spatialPosD[sb][2] - spatialPosD[sa][2];
                    const double eLenSq = ex * ex + ey * ey + ez * ez;
                    if (eLenSq > 1e-16)
                    {
                        const double bx = ey * n[2] - ez * n[1];
                        const double by = ez * n[0] - ex * n[2];
                        const double bz = ex * n[1] - ey * n[0];
                        const double bLen = std::sqrt(bx * bx + by * by + bz * bz);
                        if (bLen > 1e-13)
                        {
                            const double invB = 1.0 / bLen;
                            const double bn[3] = { bx * invB, by * invB, bz * invB };
                            const double bd    = -(bn[0] * spatialPosD[sa][0] + bn[1] * spatialPosD[sa][1] + bn[2] * spatialPosD[sa][2]);
                            const double lockMult = ue->multiMatSeam ? 32.0 : 24.0;
                            const double bw = eLenSq * static_cast<double>((std::max)(1.0f, input.config.thinFeatureWeight)) * lockMult;
                            spatialQuadrics[sa].AddSymmetricPlane(bn[0], bn[1], bn[2], bd, bw);
                            spatialQuadrics[sb].AddSymmetricPlane(bn[0], bn[1], bn[2], bd, bw);
                            spatialPureGeomQuadrics[sa].AddSymmetricPlane(bn[0], bn[1], bn[2], bd, bw);
                            spatialPureGeomQuadrics[sb].AddSymmetricPlane(bn[0], bn[1], bn[2], bd, bw);
                            spatialGeomWeightSum[sa] += bw;
                            spatialGeomWeightSum[sb] += bw;
                        }
                    }
                }
            }
        }

        // Helper: update unpaired wedge attributes along its own UV/tangent gradient field
        auto advanceUnpairedWedgeByGradient = [&](UINT32 wedgeVi, const double oldP[3], const double newP[3]) noexcept {
            const double dx = newP[0] - oldP[0];
            const double dy = newP[1] - oldP[1];
            const double dz = newP[2] - oldP[2];
            if ((dx * dx + dy * dy + dz * dz) <= 1e-18)
            {
                return;
            }
            const WedgeAccumGradient& wg = wedgeGrads[wedgeVi];
            if (wg.areaSum <= 1e-14)
            {
                return;
            }
            const double invA = 1.0 / wg.areaSum;
            auto evalShift = [&](UINT32 m) noexcept -> FLOAT32 {
                return static_cast<FLOAT32>((wg.g[m][0] * dx + wg.g[m][1] * dy + wg.g[m][2] * dz) * invA);
            };
            Vertex& v = outVertices[wedgeVi];
            v.texCoord0.x += evalShift(7u);
            v.texCoord0.y += evalShift(8u);
            v.texCoord1.x += evalShift(9u);
            v.texCoord1.y += evalShift(10u);
        };

        struct CollapseCandidate
        {
            double  cost       = 0.0;
            UINT32  sFrom      = 0u;
            UINT32  sTo        = 0u;
            UINT32  verFrom    = 0u;
            UINT32  verTo      = 0u;
            double  newPos[3]  = { 0.0, 0.0, 0.0 };
            FLOAT32 interpT    = 1.0f;

            // Strict deterministic total ordering (NVG-NANITE-AUDIT-032)
            bool operator>(const CollapseCandidate& o) const noexcept
            {
                if (cost != o.cost)       return cost > o.cost;
                if (sFrom != o.sFrom)     return sFrom > o.sFrom;
                if (sTo != o.sTo)         return sTo > o.sTo;
                if (verFrom != o.verFrom) return verFrom > o.verFrom;
                return verTo > o.verTo;
            }
        };

        std::priority_queue<
            CollapseCandidate,
            std::vector<CollapseCandidate>,
            std::greater<CollapseCandidate>> heap;

        auto evaluateSpatialEdge = [&](UINT32 sA, UINT32 sB, CollapseCandidate& outCand) -> BOOL {
            if (sA == sB || !spatialAlive[sA] || !spatialAlive[sB])
                return false;
            if (spatialLocked[sA] && spatialLocked[sB])
                return false;

            // Protect non-manifold (>2 incidence) vertices unless collapsing final root
            if (!input.allowRootNonPlanarFold && (spatialNonManifold[sA] || spatialNonManifold[sB]))
            {
                if (spatialNonManifold[sA] && spatialNonManifold[sB])
                    return false;
            }

            UINT32 sFrom        = (sA > sB) ? sA : sB;
            UINT32 sTo          = (sA > sB) ? sB : sA;
            BOOL   pinTargetPos = false;

            const BOOL seamA = (spatialWedges[sA].size() > 1u);
            const BOOL seamB = (spatialWedges[sB].size() > 1u);
            const BOOL openA = (spatialOpenBoundary[sA] != 0u);
            const BOOL openB = (spatialOpenBoundary[sB] != 0u);
            const BOOL matA  = (spatialMatSeam[sA] != 0u);
            const BOOL matB  = (spatialMatSeam[sB] != 0u);
            const BOOL nmA   = (spatialNonManifold[sA] != 0u);
            const BOOL nmB   = (spatialNonManifold[sB] != 0u);

            if (spatialLocked[sA])
            {
                sFrom        = sB;
                sTo          = sA;
                pinTargetPos = true; // Zero-ULP exterior group boundary lock!
            }
            else if (spatialLocked[sB])
            {
                sFrom        = sA;
                sTo          = sB;
                pinTargetPos = true; // Zero-ULP exterior group boundary lock!
            }
            else if (nmA != nmB)
            {
                sFrom        = nmA ? sB : sA;
                sTo          = nmA ? sA : sB;
                pinTargetPos = !input.allowRootNonPlanarFold;
            }
            else if (matA != matB)
            {
                // Multi-material seam lock: interior collapses onto material seam without shifting submesh border!
                sFrom        = matA ? sB : sA;
                sTo          = matA ? sA : sB;
                pinTargetPos = !input.allowRootNonPlanarFold;
            }
            else if (openA != openB)
            {
                sFrom        = openA ? sB : sA;
                sTo          = openA ? sA : sB;
                pinTargetPos = true;
            }
            else if (seamA != seamB)
            {
                sFrom        = seamA ? sB : sA;
                sTo          = seamA ? sA : sB;
                pinTargetPos = !input.relaxAttributeSeams;
            }

            FLOAT32 maxWingNormalDiff = 0.0f;
            FLOAT32 maxWingUvDiff     = 0.0f;
            BOOL    skinMismatch      = false;
            BOOL    mirrorSeamEdge    = false;
            UINT32  pairedFromCount   = 0u;
            BOOL    edgeOnMatSeam     = false;

            if (const EdgeTopologyRecord* ue = findUniqueEdge(MakeEdgeKeyU64(sFrom, sTo)))
            {
                edgeOnMatSeam = ue->multiMatSeam;
            }
            if (matA && matB && !edgeOnMatSeam && !input.allowRootNonPlanarFold)
            {
                // Do not bridge two distinct multi-material seam vertices across an interior chord
                return false;
            }

            for (UINT32 ti : spatialTris[sFrom])
            {
                if (!triAlive[ti]) continue;
                const NVGQEMTriangle& tr = outTriangles[ti];
                UINT32 vF = kNVGInvalidIndex;
                UINT32 vT = kNVGInvalidIndex;
                for (int k = 0; k < 3; ++k)
                {
                    const UINT32 ls = vertLocalSid[tr.v[k]];
                    if (ls == sFrom) vF = tr.v[k];
                    else if (ls == sTo) vT = tr.v[k];
                }
                if (vF != kNVGInvalidIndex && vT != kNVGInvalidIndex)
                {
                    ++pairedFromCount;
                    const FLOAT32 nDot = ClampFloat(outVertices[vF].normal.Dot(outVertices[vT].normal), -1.0f, 1.0f);
                    maxWingNormalDiff  = (std::max)(maxWingNormalDiff, 1.0f - nDot);
                    maxWingUvDiff      = (std::max)(maxWingUvDiff, (outVertices[vF].texCoord0 - outVertices[vT].texCoord0).Length());
                    if (vertHandSign[vF] * vertHandSign[vT] < 0.0f ||
                        (vertUvOrient[vF] != 0 && vertUvOrient[vT] != 0 && vertUvOrient[vF] != vertUvOrient[vT]))
                    {
                        mirrorSeamEdge = true;
                    }
                    if (input.hasSkinning && outVertices[vF].boneIndices[0] != outVertices[vT].boneIndices[0])
                    {
                        skinMismatch = true;
                    }
                }
            }

            if (pairedFromCount == 0u)
                return false;
            if (openA && openB && pairedFromCount > 1u)
            {
                // Do not collapse an interior chord bridging two open-boundary silhouette vertices
                return false;
            }

            const CoupledSpatialQuadric qSum = spatialQuadrics[sFrom] + spatialQuadrics[sTo];
            const double* pFrom = spatialPosD[sFrom].data();
            const double* pTo   = spatialPosD[sTo].data();
            const double dx = pTo[0] - pFrom[0];
            const double dy = pTo[1] - pFrom[1];
            const double dz = pTo[2] - pFrom[2];
            const double edgeLen = std::sqrt(dx * dx + dy * dy + dz * dz);

            double  bestPos[3] = { pTo[0], pTo[1], pTo[2] };
            double  bestCost   = qSum.Evaluate(pTo[0], pTo[1], pTo[2]);
            FLOAT32 bestT      = 1.0f;

            if (!pinTargetPos && !spatialLocked[sTo])
            {
                const BOOL restrictSeg = openA || openB || seamA || seamB || matA || matB || mirrorSeamEdge;
                qSum.SolveCascade(pFrom, pTo, edgeLen * 1.5 + 1e-4, restrictSeg, bestPos, bestCost, bestT);
            }

            double penalty = 1.0;
            if (maxWingNormalDiff > 0.12f)
            {
                penalty += static_cast<double>(maxWingNormalDiff) * static_cast<double>(input.config.normalPreservationWeight) * 3.5 * attrScale;
            }
            if (openA || openB)
            {
                penalty += static_cast<double>(input.config.thinFeatureWeight) * 2.0 * attrScale;
            }
            if (seamA || seamB || mirrorSeamEdge)
            {
                penalty += static_cast<double>(input.config.uvSeamWeight) * (0.5 + static_cast<double>(maxWingUvDiff)) * attrScale;
                if (spatialWedges[sFrom].size() > pairedFromCount)
                {
                    penalty += 2.0 * attrScale;
                }
            }
            if (matA || matB)
            {
                penalty += 8.0 * attrScale;
            }
            if (skinMismatch)
            {
                penalty += 4.0;
            }

            outCand.cost      = ((std::max)(0.0, bestCost) + (edgeLen * edgeLen) * 1e-6) * penalty;
            outCand.sFrom     = sFrom;
            outCand.sTo       = sTo;
            outCand.verFrom   = spatialVersion[sFrom];
            outCand.verTo     = spatialVersion[sTo];
            outCand.newPos[0] = bestPos[0];
            outCand.newPos[1] = bestPos[1];
            outCand.newPos[2] = bestPos[2];
            outCand.interpT   = bestT;
            return true;
        };

        // 5. Seed initial spatial edge candidates in deterministic sorted order
        for (const EdgeTopologyRecord& ue : uniqueEdges)
        {
            const UINT32 sa = static_cast<UINT32>(ue.key >> 32u);
            const UINT32 sb = static_cast<UINT32>(ue.key & 0xFFFFFFFFu);
            CollapseCandidate cand{};
            if (evaluateSpatialEdge(sa, sb, cand))
            {
                heap.push(cand);
            }
        }

        UINT32 aliveTriCount = 0u;
        for (UINT32 ti = 0u; ti < numTris; ++ti)
        {
            if (triAlive[ti]) ++aliveTriCount;
        }

        double maxGeomErrorNorm = 0.0;
        UINT32 collapseBudget   = numSpatial * 4u;
        const double minNormalCos = input.allowCoarseRidgeFlatten ? 0.02 : 0.05;

        std::vector<UINT32> neighborsFrom;
        std::vector<UINT32> neighborsTo;
        std::vector<UINT32> wingCommonVerts;

        while (aliveTriCount > input.targetTriangleCount && !heap.empty() && collapseBudget > 0u)
        {
            const CollapseCandidate cand = heap.top();
            heap.pop();

            if (!spatialAlive[cand.sFrom] || !spatialAlive[cand.sTo]) continue;
            if (spatialVersion[cand.sFrom] != cand.verFrom || spatialVersion[cand.sTo] != cand.verTo) continue;
            if (spatialLocked[cand.sFrom]) continue;
            if (spatialWedges[cand.sFrom].size() + spatialWedges[cand.sTo].size() > 32u) continue;

            // Ear-collapse / topological link condition check (prevents open-boundary ear deletion & non-manifold fin creation!)
            {
                neighborsFrom.clear();
                neighborsTo.clear();
                wingCommonVerts.clear();
                UINT32 aliveTrisFrom = 0u;
                UINT32 aliveTrisTo   = 0u;

                for (UINT32 ti : spatialTris[cand.sFrom])
                {
                    if (!triAlive[ti]) continue;
                    ++aliveTrisFrom;
                    const NVGQEMTriangle& t = outTriangles[ti];
                    const UINT32 ls[3] = { vertLocalSid[t.v[0]], vertLocalSid[t.v[1]], vertLocalSid[t.v[2]] };
                    BOOL hasTo = false;
                    for (int k = 0; k < 3; ++k)
                    {
                        if (ls[k] != cand.sFrom && spatialAlive[ls[k]]) neighborsFrom.push_back(ls[k]);
                        if (ls[k] == cand.sTo) hasTo = true;
                    }
                    if (hasTo)
                    {
                        for (int k = 0; k < 3; ++k)
                        {
                            if (ls[k] != cand.sFrom && ls[k] != cand.sTo) wingCommonVerts.push_back(ls[k]);
                        }
                    }
                }
                for (UINT32 ti : spatialTris[cand.sTo])
                {
                    if (!triAlive[ti]) continue;
                    ++aliveTrisTo;
                    const NVGQEMTriangle& t = outTriangles[ti];
                    for (int k = 0; k < 3; ++k)
                    {
                        const UINT32 ls = vertLocalSid[t.v[k]];
                        if (ls != cand.sTo && spatialAlive[ls]) neighborsTo.push_back(ls);
                    }
                }

                // Protect valence-1 open-boundary corner ears so silhouette corners are never deleted into empty sky
                if (aliveTriCount > 2u &&
                    ((spatialOpenBoundary[cand.sFrom] && aliveTrisFrom <= 1u) ||
                     (spatialOpenBoundary[cand.sTo]   && aliveTrisTo   <= 1u)))
                {
                    continue;
                }

                std::sort(neighborsFrom.begin(), neighborsFrom.end());
                neighborsFrom.erase(std::unique(neighborsFrom.begin(), neighborsFrom.end()), neighborsFrom.end());
                std::sort(neighborsTo.begin(), neighborsTo.end());
                neighborsTo.erase(std::unique(neighborsTo.begin(), neighborsTo.end()), neighborsTo.end());
                std::sort(wingCommonVerts.begin(), wingCommonVerts.end());
                wingCommonVerts.erase(std::unique(wingCommonVerts.begin(), wingCommonVerts.end()), wingCommonVerts.end());

                BOOL violatesLinkCondition = false;
                SIZE_T iF = 0u, iT = 0u;
                while (iF < neighborsFrom.size() && iT < neighborsTo.size())
                {
                    if (neighborsFrom[iF] < neighborsTo[iT]) ++iF;
                    else if (neighborsTo[iT] < neighborsFrom[iF]) ++iT;
                    else
                    {
                        const UINT32 sharedNb = neighborsFrom[iF];
                        if (!std::binary_search(wingCommonVerts.begin(), wingCommonVerts.end(), sharedNb))
                        {
                            violatesLinkCondition = true;
                            break;
                        }
                        if ((spatialLocked[cand.sTo] && spatialLocked[sharedNb] && wingCommonVerts.size() == 1u) ||
                            (spatialOpenBoundary[cand.sFrom] && spatialOpenBoundary[cand.sTo] &&
                             spatialOpenBoundary[sharedNb] && wingCommonVerts.size() == 1u))
                        {
                            violatesLinkCondition = true;
                            break;
                        }
                        ++iF;
                        ++iT;
                    }
                }
                if (violatesLinkCondition) continue;
            }

            // Triangle normal inversion check across all surviving incident faces
            BOOL flipsNormal = false;
            auto checkSpatialFlips = [&](UINT32 checkS) {
                for (UINT32 ti : spatialTris[checkS])
                {
                    if (!triAlive[ti]) continue;
                    const NVGQEMTriangle& t = outTriangles[ti];
                    const UINT32 ls[3] = { vertLocalSid[t.v[0]], vertLocalSid[t.v[1]], vertLocalSid[t.v[2]] };
                    const BOOL hasFrom = (ls[0] == cand.sFrom || ls[1] == cand.sFrom || ls[2] == cand.sFrom);
                    const BOOL hasTo   = (ls[0] == cand.sTo   || ls[1] == cand.sTo   || ls[2] == cand.sTo);
                    if (hasFrom && hasTo) continue;

                    double oldP[3][3], newP[3][3];
                    for (int k = 0; k < 3; ++k)
                    {
                        oldP[k][0] = spatialPosD[ls[k]][0];
                        oldP[k][1] = spatialPosD[ls[k]][1];
                        oldP[k][2] = spatialPosD[ls[k]][2];
                        if (ls[k] == cand.sFrom || ls[k] == cand.sTo)
                        {
                            newP[k][0] = cand.newPos[0];
                            newP[k][1] = cand.newPos[1];
                            newP[k][2] = cand.newPos[2];
                        }
                        else
                        {
                            newP[k][0] = oldP[k][0];
                            newP[k][1] = oldP[k][1];
                            newP[k][2] = oldP[k][2];
                        }
                    }

                    const double oe1[3] = { oldP[1][0] - oldP[0][0], oldP[1][1] - oldP[0][1], oldP[1][2] - oldP[0][2] };
                    const double oe2[3] = { oldP[2][0] - oldP[0][0], oldP[2][1] - oldP[0][1], oldP[2][2] - oldP[0][2] };
                    const double ne1[3] = { newP[1][0] - newP[0][0], newP[1][1] - newP[0][1], newP[1][2] - newP[0][2] };
                    const double ne2[3] = { newP[2][0] - newP[0][0], newP[2][1] - newP[0][1], newP[2][2] - newP[0][2] };

                    const double on[3] = {
                        oe1[1] * oe2[2] - oe1[2] * oe2[1],
                        oe1[2] * oe2[0] - oe1[0] * oe2[2],
                        oe1[0] * oe2[1] - oe1[1] * oe2[0]
                    };
                    const double nn[3] = {
                        ne1[1] * ne2[2] - ne1[2] * ne2[1],
                        ne1[2] * ne2[0] - ne1[0] * ne2[2],
                        ne1[0] * ne2[1] - ne1[1] * ne2[0]
                    };
                    const double oLen = std::sqrt(on[0] * on[0] + on[1] * on[1] + on[2] * on[2]);
                    const double nLen = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
                    if (oLen > 1e-13)
                    {
                        if (nLen <= 1e-13)
                        {
                            flipsNormal = true;
                            return;
                        }
                        const double invNLen = 1.0 / nLen;
                        const double nnUnit[3] = { nn[0] * invNLen, nn[1] * invNLen, nn[2] * invNLen };
                        const double dotCos = (on[0] * nnUnit[0] + on[1] * nnUnit[1] + on[2] * nnUnit[2]) / oLen;
                        if (dotCos < minNormalCos)
                        {
                            flipsNormal = true;
                            return;
                        }
                        // Prevent cumulative multi-step foldover within this group simplification pass
                        if (triHasOrigNormal[ti])
                        {
                            const double minOrigCos = input.allowCoarseRidgeFlatten ? 0.15 : 0.25;
                            const double origDot = origTriNormals[ti][0] * nnUnit[0] +
                                                   origTriNormals[ti][1] * nnUnit[1] +
                                                   origTriNormals[ti][2] * nnUnit[2];
                            if (origDot < minOrigCos)
                            {
                                flipsNormal = true;
                                return;
                            }
                        }
                        // Prevent cumulative multi-level foldover across sequential DAG levels relative to LOD 0 surface hemisphere
                        if (triHasConsistentVertN[ti])
                        {
                            const Vec3 avgVn = outVertices[t.v[0]].normal + outVertices[t.v[1]].normal + outVertices[t.v[2]].normal;
                            const double vnLen = std::sqrt(
                                static_cast<double>(avgVn.x) * avgVn.x +
                                static_cast<double>(avgVn.y) * avgVn.y +
                                static_cast<double>(avgVn.z) * avgVn.z);
                            if (vnLen > 1e-6)
                            {
                                const double minVertCos = input.allowRootNonPlanarFold ? 0.08 : 0.15;
                                const double vertDot = (avgVn.x * nnUnit[0] + avgVn.y * nnUnit[1] + avgVn.z * nnUnit[2]) / vnLen;
                                if (vertDot < minVertCos)
                                {
                                    flipsNormal = true;
                                    return;
                                }
                            }
                        }
                    }
                }
            };
            checkSpatialFlips(cand.sFrom);
            if (!flipsNormal) checkSpatialFlips(cand.sTo);
            if (flipsNormal) continue;

            // Commit collapse sFrom -> sTo
            --collapseBudget;
            const double oldPosFrom[3] = { spatialPosD[cand.sFrom][0], spatialPosD[cand.sFrom][1], spatialPosD[cand.sFrom][2] };
            const double oldPosTo[3]   = { spatialPosD[cand.sTo][0],   spatialPosD[cand.sTo][1],   spatialPosD[cand.sTo][2] };

            const double errFrom = (spatialGeomWeightSum[cand.sFrom] > 1e-14)
                ? std::sqrt(spatialPureGeomQuadrics[cand.sFrom].Evaluate(cand.newPos[0], cand.newPos[1], cand.newPos[2]) / spatialGeomWeightSum[cand.sFrom])
                : 0.0;
            const double errTo = (spatialGeomWeightSum[cand.sTo] > 1e-14)
                ? std::sqrt(spatialPureGeomQuadrics[cand.sTo].Evaluate(cand.newPos[0], cand.newPos[1], cand.newPos[2]) / spatialGeomWeightSum[cand.sTo])
                : 0.0;
            const double hopFrom = std::sqrt(
                (cand.newPos[0] - oldPosFrom[0]) * (cand.newPos[0] - oldPosFrom[0]) +
                (cand.newPos[1] - oldPosFrom[1]) * (cand.newPos[1] - oldPosFrom[1]) +
                (cand.newPos[2] - oldPosFrom[2]) * (cand.newPos[2] - oldPosFrom[2]));
            const double hopTo = std::sqrt(
                (cand.newPos[0] - oldPosTo[0]) * (cand.newPos[0] - oldPosTo[0]) +
                (cand.newPos[1] - oldPosTo[1]) * (cand.newPos[1] - oldPosTo[1]) +
                (cand.newPos[2] - oldPosTo[2]) * (cand.newPos[2] - oldPosTo[2]));
            const double stepErrNorm = (std::max)((std::max)(errFrom, errTo) * 1.35, (std::min)(hopFrom, hopTo) * 0.25);
            if (stepErrNorm > maxGeomErrorNorm) maxGeomErrorNorm = stepErrNorm;

            // Exact float32 world-space position (preserving 0 ULP identity when pinned to a locked boundary vertex!)
            Vec3 newWorldPos{};
            if (spatialLocked[cand.sTo] &&
                cand.newPos[0] == oldPosTo[0] &&
                cand.newPos[1] == oldPosTo[1] &&
                cand.newPos[2] == oldPosTo[2])
            {
                newWorldPos = outVertices[spatialWedges[cand.sTo][0]].position;
            }
            else
            {
                newWorldPos = Vec3(
                    static_cast<FLOAT32>(cand.newPos[0] * scaleToWorld + qemOrigin[0]),
                    static_cast<FLOAT32>(cand.newPos[1] * scaleToWorld + qemOrigin[1]),
                    static_cast<FLOAT32>(cand.newPos[2] * scaleToWorld + qemOrigin[2]));
            }

            // Bijective 1-to-1 wedge pairing strictly within matching (materialSlot, tangentHandedness, uvOrient, chart) domains!
            const std::vector<UINT32>& wedgesFrom = spatialWedges[cand.sFrom];
            const std::vector<UINT32>& wedgesTo   = spatialWedges[cand.sTo];
            std::vector<UINT32> partnerAtTo(wedgesFrom.size(), kNVGInvalidIndex);

            for (UINT32 vT : wedgesTo)
            {
                if (!vertAlive[vT]) continue;
                SIZE_T  bestFi   = wedgesFrom.size();
                FLOAT32 bestDist = 1.0e30f;

                for (SIZE_T fi = 0u; fi < wedgesFrom.size(); ++fi)
                {
                    if (partnerAtTo[fi] != kNVGInvalidIndex) continue;
                    const UINT32 vF = wedgesFrom[fi];
                    if (!vertAlive[vF]) continue;
                    // Never merge opposite tangent handedness or mirrored UV domains (NVG-NANITE-AUDIT-039)!
                    if (vertHandSign[vF] * vertHandSign[vT] < 0.0f) continue;
                    if (vertUvOrient[vF] != 0 && vertUvOrient[vT] != 0 && vertUvOrient[vF] != vertUvOrient[vT]) continue;
                    if (input.hasSkinning && outVertices[vF].boneIndices[0] != outVertices[vT].boneIndices[0]) continue;

                    BOOL sharesSameMatWingTri = false;
                    for (UINT32 ti : spatialTris[cand.sFrom])
                    {
                        if (!triAlive[ti]) continue;
                        const NVGQEMTriangle& tr = outTriangles[ti];
                        const BOOL hasVF = (tr.v[0] == vF || tr.v[1] == vF || tr.v[2] == vF);
                        const BOOL hasVT = (tr.v[0] == vT || tr.v[1] == vT || tr.v[2] == vT);
                        if (hasVF && hasVT)
                        {
                            sharesSameMatWingTri = true;
                            break;
                        }
                    }
                    if (sharesSameMatWingTri)
                    {
                        const FLOAT32 d = (outVertices[vF].texCoord0 - outVertices[vT].texCoord0).LengthSq() +
                                          (outVertices[vF].texCoord1 - outVertices[vT].texCoord1).LengthSq() +
                                          (1.0f - ClampFloat(outVertices[vF].normal.Dot(outVertices[vT].normal), -1.0f, 1.0f)) * 0.05f;
                        if (d < bestDist)
                        {
                            bestDist = d;
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
                    const Vertex vA = outVertices[matchedVF];
                    const Vertex vB = outVertices[vT];
                    Vertex merged   = vB;
                    merged.position = newWorldPos;

                    Vec3 nMix = vA.normal * invT + vB.normal * tLerp;
                    merged.normal = (nMix.LengthSq() > 1e-8f) ? nMix.Normalized() : vB.normal;

                    Vec3 tMix = vA.tangent * invT + vB.tangent * tLerp;
                    tMix = tMix - merged.normal * merged.normal.Dot(tMix);
                    merged.tangent = (tMix.LengthSq() > 1e-8f) ? tMix.Normalized() : vB.tangent;

                    const FLOAT32 hand = vertHandSign[vT];
                    merged.bitangent   = merged.normal.Cross(merged.tangent) * hand;
                    merged.texCoord0   = vA.texCoord0 * invT + vB.texCoord0 * tLerp;
                    merged.texCoord1   = vA.texCoord1 * invT + vB.texCoord1 * tLerp;
                    merged.color       = vA.color * invT + vB.color * tLerp;

                    if (input.hasSkinning)
                    {
                        if (tLerp < 0.5f)
                        {
                            for (UINT32 b = 0u; b < 4u; ++b) merged.boneIndices[b] = vA.boneIndices[b];
                            merged.boneWeights = vA.boneWeights;
                        }
                        const FLOAT32 wSum = merged.boneWeights.x + merged.boneWeights.y + merged.boneWeights.z + merged.boneWeights.w;
                        if (wSum > 1e-5f) merged.boneWeights = merged.boneWeights * (1.0f / wSum);
                    }
                    outVertices[vT] = merged;

                    // Merge accumulated attribute gradients
                    WedgeAccumGradient& wgT = wedgeGrads[vT];
                    const WedgeAccumGradient& wgF = wedgeGrads[matchedVF];
                    for (UINT32 m = 0u; m < kNumScalarWedgeAttrs; ++m)
                    {
                        wgT.g[m][0] += wgF.g[m][0];
                        wgT.g[m][1] += wgF.g[m][1];
                        wgT.g[m][2] += wgF.g[m][2];
                    }
                    wgT.areaSum += wgF.areaSum;
                }
                else
                {
                    advanceUnpairedWedgeByGradient(vT, oldPosTo, cand.newPos);
                    outVertices[vT].position = newWorldPos;
                }
                outSpatialIds[vT] = newGlobalSid;
            }

            std::vector<std::pair<UINT32, UINT32>> wedgeRemap;
            wedgeRemap.reserve(wedgesFrom.size());
            for (SIZE_T fi = 0u; fi < wedgesFrom.size(); ++fi)
            {
                const UINT32 vF = wedgesFrom[fi];
                if (!vertAlive[vF]) continue;
                const UINT32 vT = partnerAtTo[fi];
                if (vT != kNVGInvalidIndex && vertAlive[vT])
                {
                    wedgeRemap.push_back({ vF, vT });
                    vertAlive[vF] = 0u;
                }
                else
                {
                    advanceUnpairedWedgeByGradient(vF, oldPosFrom, cand.newPos);
                    outVertices[vF].position = newWorldPos;
                    outSpatialIds[vF]        = newGlobalSid;
                    vertLocalSid[vF]         = cand.sTo;
                    spatialWedges[cand.sTo].push_back(vF);
                }
            }
            std::sort(wedgeRemap.begin(), wedgeRemap.end());
            spatialWedges[cand.sFrom].clear();

            spatialPosD[cand.sTo][0]           = cand.newPos[0];
            spatialPosD[cand.sTo][1]           = cand.newPos[1];
            spatialPosD[cand.sTo][2]           = cand.newPos[2];
            spatialQuadrics[cand.sTo]          = spatialQuadrics[cand.sFrom] + spatialQuadrics[cand.sTo];
            spatialPureGeomQuadrics[cand.sTo]  = spatialPureGeomQuadrics[cand.sFrom] + spatialPureGeomQuadrics[cand.sTo];
            spatialGeomWeightSum[cand.sTo]    += spatialGeomWeightSum[cand.sFrom];
            spatialAlive[cand.sFrom]           = 0u;
            spatialVersion[cand.sTo]++;
            if (spatialOpenBoundary[cand.sFrom]) spatialOpenBoundary[cand.sTo] = 1u;
            if (spatialNonManifold[cand.sFrom])  spatialNonManifold[cand.sTo]  = 1u;
            if (spatialMatSeam[cand.sFrom])      spatialMatSeam[cand.sTo]      = 1u;

            for (UINT32 ti : spatialTris[cand.sFrom])
            {
                if (!triAlive[ti]) continue;
                NVGQEMTriangle& tr = outTriangles[ti];
                for (int k = 0; k < 3; ++k)
                {
                    auto itR = std::lower_bound(
                        wedgeRemap.begin(), wedgeRemap.end(), tr.v[k],
                        [](const std::pair<UINT32, UINT32>& p, UINT32 key) { return p.first < key; });
                    if (itR != wedgeRemap.end() && itR->first == tr.v[k])
                    {
                        tr.v[k] = itR->second;
                    }
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

            // Deterministically deduplicate spatialTris[cand.sTo] and collect neighbor spatial IDs
            auto& toTris = spatialTris[cand.sTo];
            std::sort(toTris.begin(), toTris.end());
            toTris.erase(std::unique(toTris.begin(), toTris.end()), toTris.end());

            std::vector<UINT32> cleanToTris;
            cleanToTris.reserve(toTris.size());
            std::vector<UINT32> spatialNeighbors;
            spatialNeighbors.reserve(toTris.size() * 2u);

            for (UINT32 ti : toTris)
            {
                if (!triAlive[ti]) continue;
                const NVGQEMTriangle& tr = outTriangles[ti];
                const UINT32 ls0 = vertLocalSid[tr.v[0]];
                const UINT32 ls1 = vertLocalSid[tr.v[1]];
                const UINT32 ls2 = vertLocalSid[tr.v[2]];
                if (ls0 == ls1 || ls1 == ls2 || ls2 == ls0)
                {
                    triAlive[ti] = 0u;
                    if (aliveTriCount > 0u) --aliveTriCount;
                    continue;
                }
                cleanToTris.push_back(ti);
                if (ls0 != cand.sTo && spatialAlive[ls0]) spatialNeighbors.push_back(ls0);
                if (ls1 != cand.sTo && spatialAlive[ls1]) spatialNeighbors.push_back(ls1);
                if (ls2 != cand.sTo && spatialAlive[ls2]) spatialNeighbors.push_back(ls2);
            }
            toTris = std::move(cleanToTris);

            std::sort(spatialNeighbors.begin(), spatialNeighbors.end());
            spatialNeighbors.erase(std::unique(spatialNeighbors.begin(), spatialNeighbors.end()), spatialNeighbors.end());

            for (UINT32 nb : spatialNeighbors)
            {
                CollapseCandidate newCand{};
                if (evaluateSpatialEdge(cand.sTo, nb, newCand))
                {
                    heap.push(newCand);
                }
            }
        }

        // 6. Compact surviving vertices & triangles deterministically
        std::vector<UINT32>         remap(numVerts, kNVGInvalidIndex);
        std::vector<Vertex>         compactedVerts;
        std::vector<UINT32>         compactedSpatial;
        std::vector<NVGQEMTriangle> compactedTris;
        compactedVerts.reserve(numVerts);
        compactedSpatial.reserve(numVerts);
        compactedTris.reserve(aliveTriCount);

        for (UINT32 ti = 0u; ti < numTris; ++ti)
        {
            if (!triAlive[ti]) continue;
            NVGQEMTriangle tr = outTriangles[ti];
            if (vertLocalSid[tr.v[0]] == vertLocalSid[tr.v[1]] ||
                vertLocalSid[tr.v[1]] == vertLocalSid[tr.v[2]] ||
                vertLocalSid[tr.v[2]] == vertLocalSid[tr.v[0]])
            {
                continue;
            }
            for (int k = 0; k < 3; ++k)
            {
                const UINT32 oldV = tr.v[k];
                if (remap[oldV] == kNVGInvalidIndex)
                {
                    remap[oldV] = static_cast<UINT32>(compactedVerts.size());
                    compactedVerts.push_back(outVertices[oldV]);
                    compactedSpatial.push_back(outSpatialIds[oldV]);
                }
                tr.v[k] = remap[oldV];
            }
            compactedTris.push_back(tr);
        }

        outVertices   = std::move(compactedVerts);
        outSpatialIds = std::move(compactedSpatial);
        outTriangles  = std::move(compactedTris);
        return static_cast<FLOAT32>(maxGeomErrorNorm * scaleToWorld);
    }
}
