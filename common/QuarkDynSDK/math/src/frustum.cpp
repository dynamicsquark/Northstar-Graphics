// ============================================================================
// QuarkDyn SDK — Math Module
// frustum.cpp — View Frustum Implementation
// ============================================================================

#include "../include/frustum.h"
#include "../include/mat4.h"
#include "../include/aabb.h"
#include "../include/sphere.h"

namespace QuarkDyn
{
    Frustum::Frustum() {}

    Frustum Frustum::FromViewProjection(const Mat4& vp)
    {
        Frustum f;

        // Extract planes from the View-Projection matrix (column-major).
        // Each plane is: (row3 ± rowN) of the VP matrix.
        // In column-major storage, row i values across columns are:
        //   vp.m[i], vp.m[4+i], vp.m[8+i], vp.m[12+i]

        // Left:   row3 + row0
        f.m_Planes[PLANE_LEFT].m_Normal = Vec3(vp.m[3] + vp.m[0], vp.m[7] + vp.m[4], vp.m[11] + vp.m[8]);
        f.m_Planes[PLANE_LEFT].m_Distance = -(vp.m[15] + vp.m[12]);

        // Right:  row3 - row0
        f.m_Planes[PLANE_RIGHT].m_Normal = Vec3(vp.m[3] - vp.m[0], vp.m[7] - vp.m[4], vp.m[11] - vp.m[8]);
        f.m_Planes[PLANE_RIGHT].m_Distance = -(vp.m[15] - vp.m[12]);

        // Bottom: row3 + row1
        f.m_Planes[PLANE_BOTTOM].m_Normal = Vec3(vp.m[3] + vp.m[1], vp.m[7] + vp.m[5], vp.m[11] + vp.m[9]);
        f.m_Planes[PLANE_BOTTOM].m_Distance = -(vp.m[15] + vp.m[13]);

        // Top:    row3 - row1
        f.m_Planes[PLANE_TOP].m_Normal = Vec3(vp.m[3] - vp.m[1], vp.m[7] - vp.m[5], vp.m[11] - vp.m[9]);
        f.m_Planes[PLANE_TOP].m_Distance = -(vp.m[15] - vp.m[13]);

        // Clip-space depth inequalities for [0, 1] NDC:
        //   z_clip >= 0          -> row2
        //   w_clip - z_clip >= 0 -> row3 - row2
        // In standard [0, 1] depth, row2 is Near and (row3 - row2) is Far.
        // In Reverse-Z [1, 0] depth, (row3 - row2) is Near and row2 is Far (degenerate 0 normal for infinite far).
        Vec3 zMinNormal(vp.m[2], vp.m[6], vp.m[10]);
        float zMinDist = -vp.m[14];
        Vec3 zMaxNormal(vp.m[3] - vp.m[2], vp.m[7] - vp.m[6], vp.m[11] - vp.m[10]);
        float zMaxDist = -(vp.m[15] - vp.m[14]);

        const float zMinLenSq = zMinNormal.LengthSq();
        const float zMaxLenSq = zMaxNormal.LengthSq();

        for (int i = PLANE_LEFT; i < PLANE_COUNT; ++i)
            f.m_Planes[i].Normalize();

        if (zMinLenSq < SMALL_NUMBER * SMALL_NUMBER && zMaxLenSq >= SMALL_NUMBER * SMALL_NUMBER)
        {
            // Infinite Reverse-Z: (row3 - row2) is the Near plane; Far plane is at infinity.
            f.m_Planes[PLANE_NEAR].m_Normal   = zMaxNormal;
            f.m_Planes[PLANE_NEAR].m_Distance = zMaxDist;
            f.m_Planes[PLANE_NEAR].Normalize();

            f.m_Planes[PLANE_FAR].m_Normal   = -f.m_Planes[PLANE_NEAR].m_Normal;
            f.m_Planes[PLANE_FAR].m_Distance = (vp.m[14] >= 0.0f) ? -1e30f : 1e30f;
        }
        else if (zMaxLenSq < SMALL_NUMBER * SMALL_NUMBER && zMinLenSq >= SMALL_NUMBER * SMALL_NUMBER)
        {
            // Infinite standard-Z: row2 is the Near plane; Far plane is at infinity.
            f.m_Planes[PLANE_NEAR].m_Normal   = zMinNormal;
            f.m_Planes[PLANE_NEAR].m_Distance = zMinDist;
            f.m_Planes[PLANE_NEAR].Normalize();

            f.m_Planes[PLANE_FAR].m_Normal   = -f.m_Planes[PLANE_NEAR].m_Normal;
            f.m_Planes[PLANE_FAR].m_Distance = ((vp.m[15] - vp.m[14]) >= 0.0f) ? -1e30f : 1e30f;
        }
        else
        {
            Vec3 row0(vp.m[0], vp.m[4], vp.m[8]);
            Vec3 row1(vp.m[1], vp.m[5], vp.m[9]);
            Vec3 row3(vp.m[3], vp.m[7], vp.m[11]);
            Vec3 viewFwd = (row3.LengthSq() > SMALL_NUMBER * SMALL_NUMBER) ? row3 : row0.Cross(row1);

            const bool isReverseZ = (zMinNormal.Dot(viewFwd) < 0.0f && zMaxNormal.Dot(viewFwd) > 0.0f);
            if (isReverseZ)
            {
                f.m_Planes[PLANE_NEAR].m_Normal   = zMaxNormal;
                f.m_Planes[PLANE_NEAR].m_Distance = zMaxDist;
                f.m_Planes[PLANE_FAR].m_Normal    = zMinNormal;
                f.m_Planes[PLANE_FAR].m_Distance  = zMinDist;
            }
            else
            {
                f.m_Planes[PLANE_NEAR].m_Normal   = zMinNormal;
                f.m_Planes[PLANE_NEAR].m_Distance = zMinDist;
                f.m_Planes[PLANE_FAR].m_Normal    = zMaxNormal;
                f.m_Planes[PLANE_FAR].m_Distance  = zMaxDist;
            }
            f.m_Planes[PLANE_NEAR].Normalize();
            f.m_Planes[PLANE_FAR].Normalize();
        }

        return f;
    }

    bool Frustum::ContainsPoint(const Vec3& point) const
    {
        for (int i = 0; i < PLANE_COUNT; i++)
        {
            if (m_Planes[i].DistanceToPoint(point) < 0.0f)
                return false;
        }
        return true;
    }

    bool Frustum::IntersectsSphere(const Sphere& sphere) const
    {
        for (int i = 0; i < PLANE_COUNT; i++)
        {
            if (m_Planes[i].DistanceToPoint(sphere.m_Center) < -sphere.m_Radius)
                return false;
        }
        return true;
    }

    bool Frustum::IntersectsAABB(const AABB& aabb) const
    {
        for (int i = 0; i < PLANE_COUNT; i++)
        {
            const Vec3& n = m_Planes[i].m_Normal;

            Vec3 pVertex(
                (n.x >= 0.0f) ? aabb.m_Max.x : aabb.m_Min.x,
                (n.y >= 0.0f) ? aabb.m_Max.y : aabb.m_Min.y,
                (n.z >= 0.0f) ? aabb.m_Max.z : aabb.m_Min.z
            );

            if (m_Planes[i].DistanceToPoint(pVertex) < 0.0f)
                return false;
        }
        return true;
    }

} // namespace QuarkDyn
