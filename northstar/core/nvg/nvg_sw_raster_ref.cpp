#include "nvg_sw_raster_ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Northstar
{
    namespace
    {
        inline UINT32 FloatToBits(FLOAT32 f) noexcept
        {
            UINT32 u = 0u;
            std::memcpy(&u, &f, sizeof(UINT32));
            return u;
        }

        inline FLOAT32 SaturateF32(FLOAT32 x) noexcept
        {
            return (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
        }

        inline Vec4 MulMat4Vec4(const Mat4& m, const Vec4& v) noexcept
        {
            return Vec4(
                m.m[0] * v.x + m.m[4] * v.y + m.m[8]  * v.z + m.m[12] * v.w,
                m.m[1] * v.x + m.m[5] * v.y + m.m[9]  * v.z + m.m[13] * v.w,
                m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14] * v.w,
                m.m[3] * v.x + m.m[7] * v.y + m.m[11] * v.z + m.m[15] * v.w);
        }

        static INT32 ComputeSubpixelAreaSign16_8(INT32 e01x, INT32 e01y, INT32 e12x, INT32 e12y) noexcept
        {
            const INT32 q01x = e01x >> 8; const INT32 r01x = e01x & 255;
            const INT32 q01y = e01y >> 8; const INT32 r01y = e01y & 255;
            const INT32 q12x = e12x >> 8; const INT32 r12x = e12x & 255;
            const INT32 q12y = e12y >> 8; const INT32 r12y = e12y & 255;

            const INT32 rTerm = r01x * r12y - r01y * r12x;
            const INT32 mTerm = (q01x * r12y + r01x * q12y - q01y * r12x - r01y * q12x) + (rTerm >> 8);
            const INT32 qTerm = (q01x * q12y - q01y * q12x) + (mTerm >> 8);
            if (qTerm > 0) return 1;
            if (qTerm < 0) return -1;
            const INT32 rem16 = ((mTerm & 255) << 8) | (rTerm & 255);
            return (rem16 > 0) ? 1 : 0;
        }

        // Exact CPU mirror of NVGClipPolygonHomogeneous in NVGRasterMath.nsl:
        // Clips against w >= max(wNear, minNearEps) + tight 8-pixel homogeneous X/Y guard-band.
        static UINT32 ClipTriangleHomogeneous(
            const Vec4& c0,
            const Vec4& c1,
            const Vec4& c2,
            UINT32      vpW,
            UINT32      vpH,
            FLOAT32     wNear,
            FLOAT32     minNearEps,
            BOOL        isOrtho,
            BOOL        enableNearClip,
            Vec4        outPoly[8]) noexcept
        {
            outPoly[0] = c0; outPoly[1] = c1; outPoly[2] = c2;
            for (UINT32 k = 3u; k < 8u; ++k) outPoly[k] = c2;

            const FLOAT32 fW      = (std::max)(static_cast<FLOAT32>(vpW), 1.0f);
            const FLOAT32 fH      = (std::max)(static_cast<FLOAT32>(vpH), 1.0f);
            const FLOAT32 gX      = 1.0f + (16.0f / fW);
            const FLOAT32 gY      = 1.0f + (16.0f / fH);
            const FLOAT32 effNear = (std::max)(wNear, minNearEps);

            const bool needNear = (!isOrtho && enableNearClip) ||
                                  (!isOrtho && (c0.w <= 1.0e-5f || c1.w <= 1.0e-5f || c2.w <= 1.0e-5f));
            if (needNear)
            {
                if (c0.w < effNear && c1.w < effNear && c2.w < effNear)
                {
                    return 0u;
                }
            }

            const bool wOk = (!needNear) || (c0.w >= effNear && c1.w >= effNear && c2.w >= effNear);
            if (wOk && c0.w > 0.0f && c1.w > 0.0f && c2.w > 0.0f)
            {
                if ((c0.x < -gX * c0.w && c1.x < -gX * c1.w && c2.x < -gX * c2.w) ||
                    (c0.x >  gX * c0.w && c1.x >  gX * c1.w && c2.x >  gX * c2.w) ||
                    (c0.y < -gY * c0.w && c1.y < -gY * c1.w && c2.y < -gY * c2.w) ||
                    (c0.y >  gY * c0.w && c1.y >  gY * c1.w && c2.y >  gY * c2.w))
                {
                    return 0u;
                }
                if (std::fabs(c0.x) <= gX * c0.w && std::fabs(c1.x) <= gX * c1.w && std::fabs(c2.x) <= gX * c2.w &&
                    std::fabs(c0.y) <= gY * c0.w && std::fabs(c1.y) <= gY * c1.w && std::fabs(c2.y) <= gY * c2.w)
                {
                    return 3u;
                }
            }

            Vec4 vA[8]{};
            Vec4 vB[8]{};
            vA[0] = c0; vA[1] = c1; vA[2] = c2;
            UINT32 cnt = 3u;

            if (needNear)
            {
                UINT32 outCnt = 0u;
                for (UINT32 i = 0u; i < cnt; ++i)
                {
                    const UINT32 j  = (i + 1u == cnt) ? 0u : (i + 1u);
                    const Vec4&  va = vA[i];
                    const Vec4&  vb = vA[j];
                    const FLOAT32 da = va.w - effNear;
                    const FLOAT32 db = vb.w - effNear;
                    const bool    ia = (da >= 0.0f);
                    const bool    ib = (db >= 0.0f);
                    if (ia && outCnt < 8u)
                    {
                        vB[outCnt++] = va;
                    }
                    if (ia != ib && outCnt < 8u)
                    {
                        const FLOAT32 denom = da - db;
                        const FLOAT32 t     = (std::fabs(denom) > 1.0e-12f) ? SaturateF32(da / denom) : 0.0f;
                        vB[outCnt++] = Vec4(
                            va.x + (vb.x - va.x) * t,
                            va.y + (vb.y - va.y) * t,
                            va.z + (vb.z - va.z) * t,
                            effNear);
                    }
                }
                cnt = outCnt;
                if (cnt < 3u) return 0u;
                for (UINT32 c = 0u; c < cnt; ++c) vA[c] = vB[c];
            }

            for (UINT32 plane = 0u; plane < 4u; ++plane)
            {
                UINT32 outCnt = 0u;
                for (UINT32 i = 0u; i < cnt; ++i)
                {
                    const UINT32 j  = (i + 1u == cnt) ? 0u : (i + 1u);
                    const Vec4&  va = vA[i];
                    const Vec4&  vb = vA[j];
                    FLOAT32 da = 0.0f;
                    FLOAT32 db = 0.0f;
                    if (plane == 0u)      { da = va.x + gX * va.w; db = vb.x + gX * vb.w; }
                    else if (plane == 1u) { da = gX * va.w - va.x; db = gX * vb.w - vb.x; }
                    else if (plane == 2u) { da = va.y + gY * va.w; db = vb.y + gY * vb.w; }
                    else                  { da = gY * va.w - va.y; db = gY * vb.w - vb.y; }

                    const bool ia = (da >= 0.0f);
                    const bool ib = (db >= 0.0f);
                    if (ia && outCnt < 8u)
                    {
                        vB[outCnt++] = va;
                    }
                    if (ia != ib && outCnt < 8u)
                    {
                        const FLOAT32 denom = da - db;
                        const FLOAT32 t     = (std::fabs(denom) > 1.0e-12f) ? SaturateF32(da / denom) : 0.0f;
                        vB[outCnt++] = Vec4(
                            va.x + (vb.x - va.x) * t,
                            va.y + (vb.y - va.y) * t,
                            va.z + (vb.z - va.z) * t,
                            va.w + (vb.w - va.w) * t);
                    }
                }
                cnt = outCnt;
                if (cnt < 3u) return 0u;
                for (UINT32 c = 0u; c < cnt; ++c) vA[c] = vB[c];
            }

            for (UINT32 c = 0u; c < cnt; ++c) outPoly[c] = vA[c];
            return cnt;
        }

        // Exact CPU mirror of NVGSubTriangleRequiresHWFallback16_8 in NVGRasterMath.nsl:
        // Checks whether a clipped sub-triangle would overflow the 16.8 guard-band or exceed
        // maxAllowedSpan pixels while overlapping the viewport.
        static BOOL SubTriangleRequiresHWFallback16_8(
            const Vec4& p0,
            const Vec4& p1,
            const Vec4& p2,
            UINT32      vpW,
            UINT32      vpH,
            CullMode    cullMode,
            UINT32      maxAllowedSpan) noexcept
        {
            if (p0.w <= 0.0f || p1.w <= 0.0f || p2.w <= 0.0f || vpW == 0u || vpH == 0u)
            {
                return false;
            }

            const FLOAT32 invW0 = 1.0f / p0.w;
            const FLOAT32 invW1 = 1.0f / p1.w;
            const FLOAT32 invW2 = 1.0f / p2.w;

            const FLOAT32 ndc0X = p0.x * invW0;
            const FLOAT32 ndc0Y = p0.y * invW0;
            const FLOAT32 ndc0Z = p0.z * invW0;

            const FLOAT32 ndc1X = p1.x * invW1;
            const FLOAT32 ndc1Y = p1.y * invW1;
            const FLOAT32 ndc1Z = p1.z * invW1;

            const FLOAT32 ndc2X = p2.x * invW2;
            const FLOAT32 ndc2Y = p2.y * invW2;
            const FLOAT32 ndc2Z = p2.z * invW2;

            if (!std::isfinite(ndc0X) || !std::isfinite(ndc0Y) || !std::isfinite(ndc0Z) ||
                !std::isfinite(ndc1X) || !std::isfinite(ndc1Y) || !std::isfinite(ndc1Z) ||
                !std::isfinite(ndc2X) || !std::isfinite(ndc2Y) || !std::isfinite(ndc2Z))
            {
                return true;
            }

            // Trivial 6-plane NDC frustum reject: completely off-screen sub-triangles write 0 pixels in both SW and HW
            if ((ndc0X < -1.0f && ndc1X < -1.0f && ndc2X < -1.0f) ||
                (ndc0X >  1.0f && ndc1X >  1.0f && ndc2X >  1.0f) ||
                (ndc0Y < -1.0f && ndc1Y < -1.0f && ndc2Y < -1.0f) ||
                (ndc0Y >  1.0f && ndc1Y >  1.0f && ndc2Y >  1.0f) ||
                (ndc0Z <  0.0f && ndc1Z <  0.0f && ndc2Z <  0.0f) ||
                (ndc0Z >  1.0f && ndc1Z >  1.0f && ndc2Z >  1.0f))
            {
                return false;
            }

            const FLOAT32 fW     = static_cast<FLOAT32>(vpW);
            const FLOAT32 fH     = static_cast<FLOAT32>(vpH);
            const FLOAT32 scaleX =  0.5f * fW * 256.0f;
            const FLOAT32 scaleY = -0.5f * fH * 256.0f;
            const FLOAT32 biasX  =  0.5f * fW * 256.0f + 0.5f;
            const FLOAT32 biasY  =  0.5f * fH * 256.0f + 0.5f;

            const FLOAT32 fv0x = std::floor(ndc0X * scaleX + biasX);
            const FLOAT32 fv0y = std::floor(ndc0Y * scaleY + biasY);
            const FLOAT32 fv1x = std::floor(ndc1X * scaleX + biasX);
            const FLOAT32 fv1y = std::floor(ndc1Y * scaleY + biasY);
            const FLOAT32 fv2x = std::floor(ndc2X * scaleX + biasX);
            const FLOAT32 fv2y = std::floor(ndc2Y * scaleY + biasY);

            constexpr FLOAT32 kGuardLimit = 1048576.0f; // +/- 4,096 px in 16.8 fixed-point
            if (!(std::fabs(fv0x) <= kGuardLimit && std::fabs(fv0y) <= kGuardLimit &&
                  std::fabs(fv1x) <= kGuardLimit && std::fabs(fv1y) <= kGuardLimit &&
                  std::fabs(fv2x) <= kGuardLimit && std::fabs(fv2y) <= kGuardLimit))
            {
                return true;
            }

            const INT32 v0x = static_cast<INT32>(fv0x);
            const INT32 v0y = static_cast<INT32>(fv0y);
            const INT32 v1x = static_cast<INT32>(fv1x);
            const INT32 v1y = static_cast<INT32>(fv1y);
            const INT32 v2x = static_cast<INT32>(fv2x);
            const INT32 v2y = static_cast<INT32>(fv2y);

            const INT32 e01x = v1x - v0x;
            const INT32 e01y = v1y - v0y;
            const INT32 e12x = v2x - v1x;
            const INT32 e12y = v2y - v1y;
            const INT32 e20x = v0x - v2x;
            const INT32 e20y = v0y - v2y;
            if (std::abs(e01x) > 360000 || std::abs(e01y) > 360000 ||
                std::abs(e12x) > 360000 || std::abs(e12y) > 360000 ||
                std::abs(e20x) > 360000 || std::abs(e20y) > 360000)
            {
                return true;
            }

            const INT32 areaSign = ComputeSubpixelAreaSign16_8(e01x, e01y, e12x, e12y);
            if (areaSign == 0)
            {
                return false;
            }
            const bool areaNeg = (areaSign < 0);
            if ((cullMode == CullMode::BACK && areaNeg) || (cullMode == CullMode::FRONT && !areaNeg))
            {
                return false;
            }

            const INT32 minVx  = (std::min)({ v0x, v1x, v2x });
            const INT32 minVy  = (std::min)({ v0y, v1y, v2y });
            const INT32 maxVx  = (std::max)({ v0x, v1x, v2x });
            const INT32 maxVy  = (std::max)({ v0y, v1y, v2y });
            const INT32 minPxX = (std::max)((minVx + 127) >> 8, 0);
            const INT32 minPxY = (std::max)((minVy + 127) >> 8, 0);
            const INT32 maxPxX = (std::min)((maxVx - 128) >> 8, static_cast<INT32>(vpW) - 1);
            const INT32 maxPxY = (std::min)((maxVy - 128) >> 8, static_cast<INT32>(vpH) - 1);
            if (minPxX > maxPxX || minPxY > maxPxY)
            {
                return false;
            }

            const INT32 spanCap = static_cast<INT32>((std::max)(maxAllowedSpan, 1u));
            if ((maxPxX - minPxX + 1) > spanCap || (maxPxY - minPxY + 1) > spanCap)
            {
                return true;
            }
            return false;
        }

        template <typename PixelCommitFn>
        static void RasterizeClippedSubTriangle16_8(
            const Vec4&    p0,
            const Vec4&    p1,
            const Vec4&    p2,
            UINT32         vpW,
            UINT32         vpH,
            CullMode       cullMode,
            BOOL           clampDepthPancake,
            PixelCommitFn& commitFn) noexcept
        {
            if (p0.w <= 0.0f || p1.w <= 0.0f || p2.w <= 0.0f || vpW == 0u || vpH == 0u)
            {
                return;
            }

            const FLOAT32 invW0 = 1.0f / p0.w;
            const FLOAT32 invW1 = 1.0f / p1.w;
            const FLOAT32 invW2 = 1.0f / p2.w;

            const FLOAT32 ndc0X = p0.x * invW0;
            const FLOAT32 ndc0Y = p0.y * invW0;
            FLOAT32       z0    = p0.z * invW0;

            const FLOAT32 ndc1X = p1.x * invW1;
            const FLOAT32 ndc1Y = p1.y * invW1;
            FLOAT32       z1    = p1.z * invW1;

            const FLOAT32 ndc2X = p2.x * invW2;
            const FLOAT32 ndc2Y = p2.y * invW2;
            FLOAT32       z2    = p2.z * invW2;

            const FLOAT32 fW     = static_cast<FLOAT32>(vpW);
            const FLOAT32 fH     = static_cast<FLOAT32>(vpH);
            const FLOAT32 scaleX =  0.5f * fW * 256.0f;
            const FLOAT32 scaleY = -0.5f * fH * 256.0f;
            const FLOAT32 biasX  =  0.5f * fW * 256.0f + 0.5f;
            const FLOAT32 biasY  =  0.5f * fH * 256.0f + 0.5f;

            const FLOAT32 fv0x = std::floor(ndc0X * scaleX + biasX);
            const FLOAT32 fv0y = std::floor(ndc0Y * scaleY + biasY);
            const FLOAT32 fv1x = std::floor(ndc1X * scaleX + biasX);
            const FLOAT32 fv1y = std::floor(ndc1Y * scaleY + biasY);
            const FLOAT32 fv2x = std::floor(ndc2X * scaleX + biasX);
            const FLOAT32 fv2y = std::floor(ndc2Y * scaleY + biasY);

            constexpr FLOAT32 kGuardLimit = 1048576.0f; // +/- 4,096 px in 16.8 fixed-point
            if (!(std::fabs(fv0x) <= kGuardLimit && std::fabs(fv0y) <= kGuardLimit &&
                  std::fabs(fv1x) <= kGuardLimit && std::fabs(fv1y) <= kGuardLimit &&
                  std::fabs(fv2x) <= kGuardLimit && std::fabs(fv2y) <= kGuardLimit))
            {
                return;
            }

            INT32 v0x = static_cast<INT32>(fv0x);
            INT32 v0y = static_cast<INT32>(fv0y);
            INT32 v1x = static_cast<INT32>(fv1x);
            INT32 v1y = static_cast<INT32>(fv1y);
            INT32 v2x = static_cast<INT32>(fv2x);
            INT32 v2y = static_cast<INT32>(fv2y);

            // Directed edges around 0 -> 1 -> 2 -> 0: Area2 > 0 for D3D12 CW front-facing triangles
            INT32 e01x = v1x - v0x;
            INT32 e01y = v1y - v0y;
            INT32 e12x = v2x - v1x;
            INT32 e12y = v2y - v1y;
            if (std::abs(e01x) > 360000 || std::abs(e01y) > 360000 ||
                std::abs(e12x) > 360000 || std::abs(e12y) > 360000 ||
                std::abs(v0x - v2x) > 360000 || std::abs(v0y - v2y) > 360000)
            {
                return;
            }

            const INT32 areaSign = ComputeSubpixelAreaSign16_8(e01x, e01y, e12x, e12y);
            if (areaSign == 0)
            {
                return;
            }
            const bool areaNeg = (areaSign < 0);
            if (cullMode == CullMode::BACK && areaNeg)
            {
                return;
            }
            if (cullMode == CullMode::FRONT && !areaNeg)
            {
                return;
            }
            if (areaNeg)
            {
                std::swap(v1x, v2x);
                std::swap(v1y, v2y);
                std::swap(z1, z2);
                e01x = v1x - v0x;
                e01y = v1y - v0y;
                e12x = v2x - v1x;
                e12y = v2y - v1y;
            }
            const INT32 e20x = v0x - v2x;
            const INT32 e20y = v0y - v2y;

            const INT32 minVx = (std::min)({ v0x, v1x, v2x });
            const INT32 minVy = (std::min)({ v0y, v1y, v2y });
            const INT32 maxVx = (std::max)({ v0x, v1x, v2x });
            const INT32 maxVy = (std::max)({ v0y, v1y, v2y });

            const INT32 minPxX = (std::max)((minVx + 127) >> 8, 0);
            const INT32 minPxY = (std::max)((minVy + 127) >> 8, 0);
            const INT32 maxPxX = (std::min)((maxVx - 128) >> 8, static_cast<INT32>(vpW) - 1);
            const INT32 maxPxY = (std::min)((maxVy - 128) >> 8, static_cast<INT32>(vpH) - 1);

            if (minPxX > maxPxX || minPxY > maxPxY)
            {
                return;
            }

            // Exact integer rebasing around MinPixel sample center ((minPx << 8) + 128)
            const INT32 baseSubX = (minPxX << 8) + 128;
            const INT32 baseSubY = (minPxY << 8) + 128;

            const INT32 dv1x = v1x - baseSubX;
            const INT32 dv1y = v1y - baseSubY;
            const INT32 dv2x = v2x - baseSubX;
            const INT32 dv2y = v2y - baseSubY;
            const INT32 dv0x = v0x - baseSubX;
            const INT32 dv0y = v0y - baseSubY;

            // Exact D3D12 Top-Left tie-breaking bias in 16.8 subpixels:
            //   Top-Left edge: (Ey < 0) || (Ey == 0 && Ex > 0) -> bias = 0
            //   Other edges:   bias = 1
            const INT32 tlBias0 = ((e12y < 0) || (e12y == 0 && e12x > 0)) ? 0 : 1;
            const INT32 tlBias1 = ((e20y < 0) || (e20y == 0 && e20x > 0)) ? 0 : 1;
            const INT32 tlBias2 = ((e01y < 0) || (e01y == 0 && e01x > 0)) ? 0 : 1;

            const INT32 rem0 = e12y * (dv1x & 255) - e12x * (dv1y & 255) - tlBias0;
            const INT32 rem1 = e20y * (dv2x & 255) - e20x * (dv2y & 255) - tlBias1;
            const INT32 rem2 = e01y * (dv0x & 255) - e01x * (dv0y & 255) - tlBias2;

            INT32 cy0 = (e12y * (dv1x >> 8) - e12x * (dv1y >> 8)) + (rem0 >> 8);
            INT32 cy1 = (e20y * (dv2x >> 8) - e20x * (dv2y >> 8)) + (rem1 >> 8);
            INT32 cy2 = (e01y * (dv0x >> 8) - e01x * (dv0y >> 8)) + (rem2 >> 8);

            const FLOAT32 fSub0 = static_cast<FLOAT32>((rem0 & 255) + tlBias0) * (1.0f / 256.0f);
            const FLOAT32 fSub1 = static_cast<FLOAT32>((rem1 & 255) + tlBias1) * (1.0f / 256.0f);
            const FLOAT32 fSub2 = static_cast<FLOAT32>((rem2 & 255) + tlBias2) * (1.0f / 256.0f);

            const FLOAT32 sumC = (static_cast<FLOAT32>(cy0) + fSub0) +
                                 (static_cast<FLOAT32>(cy1) + fSub1) +
                                 (static_cast<FLOAT32>(cy2) + fSub2);
            if (sumC <= 0.0f)
            {
                return;
            }
            const FLOAT32 invAreaScaled = 1.0f / sumC;
            const FLOAT32 dZ1           = (z1 - z0) * invAreaScaled;
            const FLOAT32 dZ2           = (z2 - z0) * invAreaScaled;

            const INT32 spanX = maxPxX - minPxX;
            if (spanX <= 4)
            {
                for (INT32 py = minPxY; py <= maxPxY; ++py)
                {
                    INT32 cx0 = cy0;
                    INT32 cx1 = cy1;
                    INT32 cx2 = cy2;
                    for (INT32 px = minPxX; px <= maxPxX; ++px)
                    {
                        if ((cx0 | cx1 | cx2) >= 0)
                        {
                            FLOAT32 devZ = z0 + dZ1 * (static_cast<FLOAT32>(cx1) + fSub1) +
                                                dZ2 * (static_cast<FLOAT32>(cx2) + fSub2);
                            if (clampDepthPancake)
                            {
                                devZ = SaturateF32(devZ);
                            }
                            if (devZ >= 0.0f && devZ <= 1.0f)
                            {
                                commitFn(static_cast<UINT32>(px), static_cast<UINT32>(py), devZ);
                            }
                        }
                        cx0 -= e12y;
                        cx1 -= e20y;
                        cx2 -= e01y;
                    }
                    cy0 += e12x;
                    cy1 += e20x;
                    cy2 += e01x;
                }
            }
            else
            {
                auto clipScanlineEdge = [](INT32 cy, INT32 ey, INT32& ioStart, INT32& ioEnd) noexcept {
                    if (ey == 0)
                    {
                        if (cy < 0) ioEnd = -1;
                    }
                    else if (ey > 0)
                    {
                        if (cy < 0) ioEnd = -1;
                        else        ioEnd = (std::min)(ioEnd, cy / ey);
                    }
                    else
                    {
                        if (cy < 0)
                        {
                            const INT32 posEy = -ey;
                            const INT32 need  = -cy;
                            ioStart = (std::max)(ioStart, (need - 1) / posEy + 1);
                        }
                    }
                };

                for (INT32 py = minPxY; py <= maxPxY; ++py)
                {
                    INT32 dxStart = 0;
                    INT32 dxEnd   = spanX;
                    clipScanlineEdge(cy0, e12y, dxStart, dxEnd);
                    clipScanlineEdge(cy1, e20y, dxStart, dxEnd);
                    clipScanlineEdge(cy2, e01y, dxStart, dxEnd);

                    if (dxStart <= dxEnd)
                    {
                        INT32 cx0 = cy0 - dxStart * e12y;
                        INT32 cx1 = cy1 - dxStart * e20y;
                        INT32 cx2 = cy2 - dxStart * e01y;
                        for (INT32 dx = dxStart; dx <= dxEnd; ++dx)
                        {
                            if ((cx0 | cx1 | cx2) >= 0)
                            {
                                FLOAT32 devZ = z0 + dZ1 * (static_cast<FLOAT32>(cx1) + fSub1) +
                                                    dZ2 * (static_cast<FLOAT32>(cx2) + fSub2);
                                if (clampDepthPancake)
                                {
                                    devZ = SaturateF32(devZ);
                                }
                                if (devZ >= 0.0f && devZ <= 1.0f)
                                {
                                    commitFn(static_cast<UINT32>(minPxX + dx), static_cast<UINT32>(py), devZ);
                                }
                            }
                            cx0 -= e12y;
                            cx1 -= e20y;
                            cx2 -= e01y;
                        }
                    }

                    cy0 += e12x;
                    cy1 += e20x;
                    cy2 += e01x;
                }
            }
        }
    }

    BOOL NVGSoftwareRasterizerCPU::ExtractClusterFromAsset(
        const NVGVirtualMeshAsset& asset,
        UINT32                     clusterIndex,
        UINT32                     visibleClusterIndex,
        CullMode                   cullMode,
        const Mat4&                worldMatrix,
        const Mat4*                skinningPalette,
        UINT32                     boneCount,
        NVGSWRasterInputCluster&   outCluster)
    {
        outCluster = NVGSWRasterInputCluster{};
        if (clusterIndex >= asset.clusters.size())
        {
            return false;
        }
        const NVGClusterMetaGPUData& clMeta  = asset.clusters[clusterIndex];
        const UINT32                 pageIdx = clMeta.virtualPageId;
        if (pageIdx >= asset.pages.size() || asset.pages[pageIdx].words.empty())
        {
            return false;
        }
        const NVGVirtualPagePayload& page = asset.pages[pageIdx];

        std::vector<Vertex> decodedVerts;
        std::vector<UINT32> decodedIdx;
        if (!NVGBuilder::DecodeClusterCPU(
                page.words.data(),
                static_cast<UINT32>(page.words.size()),
                clMeta.clusterWordOffsetInPage,
                decodedVerts,
                decodedIdx))
        {
            return false;
        }

        outCluster.visibleClusterIndex = visibleClusterIndex;
        outCluster.cullMode            = cullMode;
        outCluster.worldMatrix         = worldMatrix;
        outCluster.localPositions.resize(decodedVerts.size());
        outCluster.indices = std::move(decodedIdx);

        for (SIZE_T i = 0; i < decodedVerts.size(); ++i)
        {
            Vec3 pos = decodedVerts[i].position;
            if (skinningPalette != nullptr && boneCount > 0u)
            {
                const Vertex& v = decodedVerts[i];
                const FLOAT32 wSum = v.boneWeights[0] + v.boneWeights[1] + v.boneWeights[2] + v.boneWeights[3];
                if (wSum > 1.0e-5f)
                {
                    const FLOAT32 invW = 1.0f / wSum;
                    const UINT32  maxB = boneCount - 1u;
                    const Vec4    p4(pos.x, pos.y, pos.z, 1.0f);
                    Vec3          sk(0.0f, 0.0f, 0.0f);
                    for (int k = 0; k < 4; ++k)
                    {
                        const UINT32  bi = (std::min)(v.boneIndices[k], maxB);
                        const FLOAT32 wk = v.boneWeights[k] * invW;
                        const Vec4    bp = MulMat4Vec4(skinningPalette[bi], p4);
                        sk = sk + Vec3(bp.x, bp.y, bp.z) * wk;
                    }
                    pos = sk;
                }
            }
            outCluster.localPositions[i] = pos;
        }
        return true;
    }

    void NVGSoftwareRasterizerCPU::RasterizeClustersToVisBuffer64(
        const NVGSWRasterInputCluster* clusters,
        SIZE_T                         clusterCount,
        const Mat4&                    vpMatrix,
        const NVGSWRasterConfigCPU&    config,
        std::vector<UINT64>&           outVisBuffer64,
        std::vector<UINT8>*            outClusterAbortedToHW)
    {
        const SIZE_T totalPixels = static_cast<SIZE_T>(config.viewportWidth) * static_cast<SIZE_T>(config.viewportHeight);
        outVisBuffer64.assign(totalPixels, 0ull);
        if (outClusterAbortedToHW)
        {
            outClusterAbortedToHW->assign(clusterCount, 0u);
        }
        if (!clusters || clusterCount == 0u || totalPixels == 0u)
        {
            return;
        }

        std::vector<Vec4> clipVerts;
        for (SIZE_T ci = 0; ci < clusterCount; ++ci)
        {
            const NVGSWRasterInputCluster& cl = clusters[ci];
            const UINT32 vCnt = static_cast<UINT32>(cl.localPositions.size());
            const UINT32 tCnt = static_cast<UINT32>(cl.indices.size() / 3u);
            if (vCnt == 0u || tCnt == 0u)
            {
                continue;
            }

            clipVerts.resize(vCnt);
            for (UINT32 v = 0u; v < vCnt; ++v)
            {
                const Vec3& lp = cl.localPositions[v];
                const Vec4  wp = MulMat4Vec4(cl.worldMatrix, Vec4(lp.x, lp.y, lp.z, 1.0f));
                clipVerts[v]   = MulMat4Vec4(vpMatrix, Vec4(wp.x, wp.y, wp.z, 1.0f));
            }

            // Cooperative pre-commit validation matching GPU Mode 2 (zero partial pixel writes on fallback)
            BOOL abortToHW = false;
            for (UINT32 ti = 0u; ti < tCnt && ti < 128u; ++ti)
            {
                const UINT32 i0 = cl.indices[ti * 3u + 0u];
                const UINT32 i1 = cl.indices[ti * 3u + 1u];
                const UINT32 i2 = cl.indices[ti * 3u + 2u];
                if (i0 >= vCnt || i1 >= vCnt || i2 >= vCnt)
                {
                    continue;
                }
                Vec4 poly[8];
                const UINT32 polyCnt = ClipTriangleHomogeneous(
                    clipVerts[i0], clipVerts[i1], clipVerts[i2],
                    config.viewportWidth, config.viewportHeight,
                    config.nearPlane, 1.0e-5f, config.isOrthographic, config.enableNearClip, poly);
                for (UINT32 sub = 1u; sub + 1u < polyCnt; ++sub)
                {
                    if (SubTriangleRequiresHWFallback16_8(
                            poly[0], poly[sub], poly[sub + 1u],
                            config.viewportWidth, config.viewportHeight,
                            cl.cullMode, config.maxSWTriangleSpanPx))
                    {
                        abortToHW = true;
                        break;
                    }
                }
                if (abortToHW)
                {
                    break;
                }
            }
            if (abortToHW)
            {
                if (outClusterAbortedToHW)
                {
                    (*outClusterAbortedToHW)[ci] = 1u;
                }
                continue;
            }

            for (UINT32 ti = 0u; ti < tCnt && ti < 128u; ++ti)
            {
                const UINT32 i0 = cl.indices[ti * 3u + 0u];
                const UINT32 i1 = cl.indices[ti * 3u + 1u];
                const UINT32 i2 = cl.indices[ti * 3u + 2u];
                if (i0 >= vCnt || i1 >= vCnt || i2 >= vCnt)
                {
                    continue;
                }

                Vec4 poly[8];
                const UINT32 polyCnt = ClipTriangleHomogeneous(
                    clipVerts[i0], clipVerts[i1], clipVerts[i2],
                    config.viewportWidth, config.viewportHeight,
                    config.nearPlane, 1.0e-5f, config.isOrthographic, config.enableNearClip, poly);
                if (polyCnt < 3u)
                {
                    continue;
                }

                const UINT32 payload32 = ((cl.visibleClusterIndex + 1u) << 8u) | (ti & 255u);
                auto commitPixel = [&](UINT32 px, UINT32 py, FLOAT32 devZ) noexcept {
                    const FLOAT32 clampedZ   = SaturateF32(devZ);
                    const UINT32  zBits      = FloatToBits(clampedZ);
                    const UINT32  depthKey32 = config.isReversedZ ? zBits : (~zBits);
                    const UINT64  vis64      = (static_cast<UINT64>(depthKey32) << 32u) | static_cast<UINT64>(payload32);
                    UINT64&       dst        = outVisBuffer64[static_cast<SIZE_T>(py) * config.viewportWidth + px];
                    if (vis64 > dst)
                    {
                        dst = vis64;
                    }
                };

                for (UINT32 sub = 1u; sub + 1u < polyCnt; ++sub)
                {
                    RasterizeClippedSubTriangle16_8(
                        poly[0], poly[sub], poly[sub + 1u],
                        config.viewportWidth, config.viewportHeight,
                        cl.cullMode, false, commitPixel);
                }
            }
        }
    }

    void NVGSoftwareRasterizerCPU::RasterizeClusterToShadowPageR32(
        const NVGSWRasterInputCluster& cluster,
        const Mat4&                    worldToPageClip,
        FLOAT32                        depthBiasNdc,
        BOOL                           isReversedZ,
        UINT32*                        ioShadowPage128x128,
        BOOL*                          outAbortedToHW)
    {
        if (outAbortedToHW)
        {
            *outAbortedToHW = false;
        }
        if (!ioShadowPage128x128)
        {
            return;
        }
        const UINT32 vCnt = static_cast<UINT32>(cluster.localPositions.size());
        const UINT32 tCnt = static_cast<UINT32>(cluster.indices.size() / 3u);
        if (vCnt == 0u || tCnt == 0u)
        {
            return;
        }

        std::vector<Vec4> clipVerts(vCnt);
        for (UINT32 v = 0u; v < vCnt; ++v)
        {
            const Vec3& lp = cluster.localPositions[v];
            const Vec4  wp = MulMat4Vec4(cluster.worldMatrix, Vec4(lp.x, lp.y, lp.z, 1.0f));
            Vec4        cp = MulMat4Vec4(worldToPageClip, Vec4(wp.x, wp.y, wp.z, 1.0f));
            cp.z += depthBiasNdc * cp.w;
            clipVerts[v] = cp;
        }

        const BOOL isShadowOrtho =
            (std::fabs(worldToPageClip.m[3]) + std::fabs(worldToPageClip.m[7]) + std::fabs(worldToPageClip.m[11])) <= 1.0e-5f;

        for (UINT32 ti = 0u; ti < tCnt && ti < 128u; ++ti)
        {
            const UINT32 i0 = cluster.indices[ti * 3u + 0u];
            const UINT32 i1 = cluster.indices[ti * 3u + 1u];
            const UINT32 i2 = cluster.indices[ti * 3u + 2u];
            if (i0 >= vCnt || i1 >= vCnt || i2 >= vCnt)
            {
                continue;
            }
            Vec4 poly[8];
            const UINT32 polyCnt = ClipTriangleHomogeneous(
                clipVerts[i0], clipVerts[i1], clipVerts[i2],
                128u, 128u, 1.0e-3f, 1.0e-4f, isShadowOrtho, !isShadowOrtho, poly);
            for (UINT32 sub = 1u; sub + 1u < polyCnt; ++sub)
            {
                if (SubTriangleRequiresHWFallback16_8(poly[0], poly[sub], poly[sub + 1u], 128u, 128u, cluster.cullMode, 128u))
                {
                    if (outAbortedToHW)
                    {
                        *outAbortedToHW = true;
                    }
                    return;
                }
            }
        }

        for (UINT32 ti = 0u; ti < tCnt && ti < 128u; ++ti)
        {
            const UINT32 i0 = cluster.indices[ti * 3u + 0u];
            const UINT32 i1 = cluster.indices[ti * 3u + 1u];
            const UINT32 i2 = cluster.indices[ti * 3u + 2u];
            if (i0 >= vCnt || i1 >= vCnt || i2 >= vCnt)
            {
                continue;
            }

            Vec4 poly[8];
            const UINT32 polyCnt = ClipTriangleHomogeneous(
                clipVerts[i0], clipVerts[i1], clipVerts[i2],
                128u, 128u, 1.0e-3f, 1.0e-4f, isShadowOrtho, !isShadowOrtho, poly);
            if (polyCnt < 3u)
            {
                continue;
            }

            auto commitShadowTexel = [&](UINT32 px, UINT32 py, FLOAT32 devZ) noexcept {
                const UINT32 zBits = FloatToBits(SaturateF32(devZ));
                UINT32&      dst   = ioShadowPage128x128[py * 128u + px];
                if (isReversedZ)
                {
                    if (zBits > dst) dst = zBits;
                }
                else
                {
                    if (zBits < dst) dst = zBits;
                }
            };

            for (UINT32 sub = 1u; sub + 1u < polyCnt; ++sub)
            {
                RasterizeClippedSubTriangle16_8(
                    poly[0], poly[sub], poly[sub + 1u],
                    128u, 128u, cluster.cullMode, true, commitShadowTexel);
            }
        }
    }
}
