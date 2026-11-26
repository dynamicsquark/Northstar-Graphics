// ============================================================================
// QuarkDyn SDK — Math Module
// frustum.h — View Frustum (6 planes)
// ============================================================================
#pragma once

#include "plane.h"
#include "mat4.h"
#include "aabb.h"
#include "sphere.h"

namespace QuarkDyn
{
    struct Frustum
    {
        // Plane order: Near, Far, Left, Right, Top, Bottom
        enum PlaneIndex
        {
            PLANE_NEAR   = 0,
            PLANE_FAR    = 1,
            PLANE_LEFT   = 2,
            PLANE_RIGHT  = 3,
            PLANE_TOP    = 4,
            PLANE_BOTTOM = 5,
            PLANE_COUNT  = 6
        };

        Plane m_Planes[PLANE_COUNT];

        // ==================== CONSTRUCTORS ====================
        Frustum();

        // ==================== STATIC BUILDERS ====================
        static Frustum FromViewProjection(const Mat4& vp);

        // ==================== METHODS ====================
        bool ContainsPoint(const Vec3& point) const;
        bool IntersectsSphere(const Sphere& sphere) const;
        bool IntersectsAABB(const AABB& aabb) const;
    };

} // namespace QuarkDyn
