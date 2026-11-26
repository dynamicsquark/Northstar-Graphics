// ============================================================================
// QuarkDyn SDK — Math Module
// sphere.h — 3D Sphere
// ============================================================================
#pragma once

#include "vec3.h"
#include "aabb.h"

namespace QuarkDyn
{
    struct Sphere
    {
        Vec3  m_Center;
        float m_Radius;

        // ==================== CONSTRUCTORS ====================
        Sphere();
        Sphere(const Vec3& center, float radius);

        // ==================== METHODS ====================
        bool Contains(const Vec3& point) const;
        bool Intersects(const Sphere& other) const;
        bool Intersects(const AABB& aabb) const;
        Vec3 ClosestPoint(const Vec3& point) const;
        float SurfaceArea() const;
        float Volume() const;

        AABB ToAABB() const;

        // ==================== STATIC METHODS ====================
        static Sphere Merge(const Sphere& a, const Sphere& b);
    };

} // namespace QuarkDyn
