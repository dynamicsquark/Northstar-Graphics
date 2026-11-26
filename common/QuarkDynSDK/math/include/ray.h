// ============================================================================
// QuarkDyn SDK — Math Module
// ray.h — 3D Ray
// ============================================================================
#pragma once

#include "vec3.h"
#include "aabb.h"

namespace QuarkDyn
{
    struct Ray
    {
        Vec3 m_Origin;
        Vec3 m_Direction;

        // ==================== CONSTRUCTORS ====================
        Ray();
        Ray(const Vec3& origin, const Vec3& direction);

        // ==================== METHODS ====================
        Vec3 PointAt(float t) const;
        Vec3 ClosestPoint(const Vec3& point) const;

        bool IntersectAABB(const AABB& aabb, float& tMin, float& tMax) const;
        bool IntersectSphere(const Vec3& center, float radius, float& t) const;
        bool IntersectPlane(const Vec3& planeNormal, const Vec3& planePoint, float& t) const;
        bool IntersectTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2,
                               float& t, float& u, float& v) const;
    };

} // namespace QuarkDyn
