// ============================================================================
// QuarkDyn SDK — Math Module
// plane.cpp — 3D Plane Implementation
// ============================================================================

#include "../include/plane.h"

namespace QuarkDyn
{
    Plane::Plane() : m_Normal(Vec3::Up()), m_Distance(0.0f) {}
    Plane::Plane(const Vec3& normal, float distance) : m_Normal(normal.Normalized()), m_Distance(distance) {}

    Plane::Plane(const Vec3& normal, const Vec3& point)
        : m_Normal(normal.Normalized())
        , m_Distance(m_Normal.Dot(point))
    {}

    Plane::Plane(const Vec3& p1, const Vec3& p2, const Vec3& p3)
    {
        Vec3 edge1 = p2 - p1;
        Vec3 edge2 = p3 - p1;
        m_Normal = edge1.Cross(edge2).Normalized();
        m_Distance = m_Normal.Dot(p1);
    }

    float Plane::DistanceToPoint(const Vec3& point) const
    {
        return m_Normal.Dot(point) - m_Distance;
    }

    Vec3 Plane::ClosestPoint(const Vec3& point) const
    {
        return point - m_Normal * DistanceToPoint(point);
    }

    bool Plane::IsOnPlane(const Vec3& point, float tolerance) const
    {
        return std::abs(DistanceToPoint(point)) <= tolerance;
    }

    Plane Plane::Normalized() const
    {
        float len = m_Normal.Length();
        if (len < EPSILON) return *this;
        float invLen = 1.0f / len;
        Plane result;
        result.m_Normal = m_Normal * invLen;
        result.m_Distance = m_Distance * invLen;
        return result;
    }

    void Plane::Normalize()
    {
        float len = m_Normal.Length();
        if (len < EPSILON) return;
        float invLen = 1.0f / len;
        m_Normal = m_Normal * invLen;
        m_Distance *= invLen;
    }

    int Plane::Side(const Vec3& point) const
    {
        float d = DistanceToPoint(point);
        if (d > EPSILON) return 1;
        if (d < -EPSILON) return -1;
        return 0;
    }

} // namespace QuarkDyn
