// ============================================================================
// QuarkDyn SDK — Math Module
// sphere.cpp — 3D Sphere Implementation
// ============================================================================

#include "../include/sphere.h"
#include "../include/aabb.h"

namespace QuarkDyn
{
    Sphere::Sphere() : m_Center(Vec3::Zero()), m_Radius(1.0f) {}
    Sphere::Sphere(const Vec3& center, float radius) : m_Center(center), m_Radius(radius) {}

    bool Sphere::Contains(const Vec3& point) const
    {
        return m_Center.DistanceSq(point) <= m_Radius * m_Radius;
    }

    bool Sphere::Intersects(const Sphere& other) const
    {
        float r = m_Radius + other.m_Radius;
        return m_Center.DistanceSq(other.m_Center) <= r * r;
    }

    bool Sphere::Intersects(const AABB& aabb) const
    {
        Vec3 closest = aabb.ClosestPoint(m_Center);
        return m_Center.DistanceSq(closest) <= m_Radius * m_Radius;
    }

    Vec3 Sphere::ClosestPoint(const Vec3& point) const
    {
        Vec3 dir = (point - m_Center).Normalized();
        return m_Center + dir * m_Radius;
    }

    AABB Sphere::ToAABB() const
    {
        Vec3 r(m_Radius);
        return AABB(m_Center - r, m_Center + r);
    }

    float Sphere::SurfaceArea() const
    {
        return 4.0f * PI * m_Radius * m_Radius;
    }

    float Sphere::Volume() const
    {
        return (4.0f / 3.0f) * PI * m_Radius * m_Radius * m_Radius;
    }

    Sphere Sphere::Merge(const Sphere& a, const Sphere& b)
    {
        Vec3 diff = b.m_Center - a.m_Center;
        float dist = diff.Length();

        if (dist + b.m_Radius <= a.m_Radius) return a;
        if (dist + a.m_Radius <= b.m_Radius) return b;

        float newRadius = (dist + a.m_Radius + b.m_Radius) * 0.5f;
        Vec3 newCenter = a.m_Center;
        if (dist > EPSILON)
            newCenter = a.m_Center + diff * ((newRadius - a.m_Radius) / dist);

        return Sphere(newCenter, newRadius);
    }

} // namespace QuarkDyn
