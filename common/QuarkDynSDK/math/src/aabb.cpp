// ============================================================================
// QuarkDyn SDK — Math Module
// aabb.cpp — Axis-Aligned Bounding Box Implementation
// ============================================================================

#include "../include/aabb.h"
#include "../include/mat4.h"

namespace QuarkDyn
{
    AABB::AABB() : m_Min(Vec3::Zero()), m_Max(Vec3::Zero()) {}
    AABB::AABB(const Vec3& min, const Vec3& max) : m_Min(min), m_Max(max) {}

    Vec3 AABB::Center() const { return (m_Min + m_Max) * 0.5f; }
    Vec3 AABB::Size() const { return m_Max - m_Min; }
    Vec3 AABB::Extents() const { return (m_Max - m_Min) * 0.5f; }

    bool AABB::Contains(const Vec3& point) const
    {
        return point.x >= m_Min.x && point.x <= m_Max.x &&
               point.y >= m_Min.y && point.y <= m_Max.y &&
               point.z >= m_Min.z && point.z <= m_Max.z;
    }

    bool AABB::Intersects(const AABB& other) const
    {
        return m_Min.x <= other.m_Max.x && m_Max.x >= other.m_Min.x &&
               m_Min.y <= other.m_Max.y && m_Max.y >= other.m_Min.y &&
               m_Min.z <= other.m_Max.z && m_Max.z >= other.m_Min.z;
    }

    AABB AABB::Merge(const AABB& other) const
    {
        return AABB(
            Vec3(Min(m_Min.x, other.m_Min.x), Min(m_Min.y, other.m_Min.y), Min(m_Min.z, other.m_Min.z)),
            Vec3(Max(m_Max.x, other.m_Max.x), Max(m_Max.y, other.m_Max.y), Max(m_Max.z, other.m_Max.z))
        );
    }

    void AABB::Expand(const Vec3& point)
    {
        m_Min.x = Min(m_Min.x, point.x);
        m_Min.y = Min(m_Min.y, point.y);
        m_Min.z = Min(m_Min.z, point.z);
        m_Max.x = Max(m_Max.x, point.x);
        m_Max.y = Max(m_Max.y, point.y);
        m_Max.z = Max(m_Max.z, point.z);
    }

    Vec3 AABB::ClosestPoint(const Vec3& point) const
    {
        return Vec3(
            Clamp(point.x, m_Min.x, m_Max.x),
            Clamp(point.y, m_Min.y, m_Max.y),
            Clamp(point.z, m_Min.z, m_Max.z)
        );
    }

    AABB AABB::Transformed(const Mat4& matrix) const
    {
        Vec3 corners[8] = {
            Vec3(m_Min.x, m_Min.y, m_Min.z),
            Vec3(m_Max.x, m_Min.y, m_Min.z),
            Vec3(m_Min.x, m_Max.y, m_Min.z),
            Vec3(m_Max.x, m_Max.y, m_Min.z),
            Vec3(m_Min.x, m_Min.y, m_Max.z),
            Vec3(m_Max.x, m_Min.y, m_Max.z),
            Vec3(m_Min.x, m_Max.y, m_Max.z),
            Vec3(m_Max.x, m_Max.y, m_Max.z)
        };

        Vec3 newMin = matrix.TransformPoint(corners[0]);
        Vec3 newMax = newMin;

        for (int i = 1; i < 8; i++)
        {
            Vec3 p = matrix.TransformPoint(corners[i]);
            newMin.x = Min(newMin.x, p.x);
            newMin.y = Min(newMin.y, p.y);
            newMin.z = Min(newMin.z, p.z);
            newMax.x = Max(newMax.x, p.x);
            newMax.y = Max(newMax.y, p.y);
            newMax.z = Max(newMax.z, p.z);
        }

        return AABB(newMin, newMax);
    }

    float AABB::SurfaceArea() const
    {
        Vec3 s = Size();
        return 2.0f * (s.x * s.y + s.y * s.z + s.z * s.x);
    }

    float AABB::Volume() const
    {
        Vec3 s = Size();
        return s.x * s.y * s.z;
    }

    bool AABB::IsValid() const
    {
        return m_Min.x <= m_Max.x && m_Min.y <= m_Max.y && m_Min.z <= m_Max.z;
    }

    AABB AABB::FromPoints(const Vec3* points, int count)
    {
        assert(points != nullptr && count > 0);
        AABB result(points[0], points[0]);
        for (int i = 1; i < count; i++)
            result.Expand(points[i]);
        return result;
    }

} // namespace QuarkDyn
