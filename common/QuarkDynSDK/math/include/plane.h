// ============================================================================
// QuarkDyn SDK — Math Module
// plane.h — 3D Plane
// ============================================================================
#pragma once

#include "vec3.h"

namespace QuarkDyn
{
    struct Plane
    {
        Vec3  m_Normal;
        float m_Distance;

        // ==================== CONSTRUCTORS ====================
        Plane();
        Plane(const Vec3& normal, float distance);
        Plane(const Vec3& normal, const Vec3& point);
        Plane(const Vec3& p1, const Vec3& p2, const Vec3& p3);

        // ==================== METHODS ====================
        float DistanceToPoint(const Vec3& point) const;
        Vec3  ClosestPoint(const Vec3& point) const;
        bool  IsOnPlane(const Vec3& point, float tolerance = EPSILON) const;
        Plane Normalized() const;
        void  Normalize();
        int   Side(const Vec3& point) const;  // -1, 0, 1
    };

} // namespace QuarkDyn
