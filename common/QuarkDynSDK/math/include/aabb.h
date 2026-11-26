// ============================================================================
// QuarkDyn SDK — Math Module
// aabb.h — Axis-Aligned Bounding Box
// ============================================================================
#pragma once

#include "vec3.h"
#include "mat4.h"

namespace QuarkDyn
{
    struct AABB
    {
        Vec3 m_Min;
        Vec3 m_Max;

        // ==================== CONSTRUCTORS ====================
        AABB();
        AABB(const Vec3& min, const Vec3& max);

        // ==================== METHODS ====================
        Vec3 Center() const;
        Vec3 Size() const;
        Vec3 Extents() const;

        bool Contains(const Vec3& point) const;
        bool Intersects(const AABB& other) const;
        AABB Merge(const AABB& other) const;
        void Expand(const Vec3& point);
        Vec3 ClosestPoint(const Vec3& point) const;
        float SurfaceArea() const;
        float Volume() const;
        bool IsValid() const;

        AABB Transformed(const Mat4& matrix) const;

        // ==================== STATIC METHODS ====================
        static AABB FromPoints(const Vec3* points, int count);
    };

} // namespace QuarkDyn
