// ============================================================================
// QuarkDyn SDK — Math Module
// ray.cpp — 3D Ray Implementation
// ============================================================================

#include "../include/ray.h"
#include "../include/aabb.h"

namespace QuarkDyn
{
    Ray::Ray() : m_Origin(Vec3::Zero()), m_Direction(Vec3::Forward()) {}
    Ray::Ray(const Vec3& origin, const Vec3& direction)
        : m_Origin(origin), m_Direction(direction.Normalized()) {}

    Vec3 Ray::PointAt(float t) const { return m_Origin + m_Direction * t; }

    Vec3 Ray::ClosestPoint(const Vec3& point) const
    {
        Vec3 diff = point - m_Origin;
        float t = Max(0.0f, diff.Dot(m_Direction));
        return PointAt(t);
    }

    bool Ray::IntersectAABB(const AABB& aabb, float& tMin, float& tMax) const
    {
        float t1, t2;
        tMin = 0.0f;
        tMax = INFINITY_;

        for (int i = 0; i < 3; i++)
        {
            if (std::abs(m_Direction[i]) < EPSILON)
            {
                if (m_Origin[i] < aabb.m_Min[i] || m_Origin[i] > aabb.m_Max[i])
                    return false;
            }
            else
            {
                float invD = 1.0f / m_Direction[i];
                t1 = (aabb.m_Min[i] - m_Origin[i]) * invD;
                t2 = (aabb.m_Max[i] - m_Origin[i]) * invD;

                if (t1 > t2) { float temp = t1; t1 = t2; t2 = temp; }

                tMin = Max(tMin, t1);
                tMax = Min(tMax, t2);

                if (tMin > tMax) return false;
            }
        }

        return true;
    }

    bool Ray::IntersectSphere(const Vec3& center, float radius, float& t) const
    {
        Vec3 oc = m_Origin - center;
        float a = m_Direction.Dot(m_Direction);
        float b = 2.0f * oc.Dot(m_Direction);
        float c = oc.Dot(oc) - radius * radius;
        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0.0f) return false;

        float sqrtD = std::sqrt(discriminant);
        float t0 = (-b - sqrtD) / (2.0f * a);
        float t1 = (-b + sqrtD) / (2.0f * a);

        if (t0 > 0.0f) { t = t0; return true; }
        if (t1 > 0.0f) { t = t1; return true; }
        return false;
    }

    bool Ray::IntersectPlane(const Vec3& planeNormal, const Vec3& planePoint, float& t) const
    {
        float denom = planeNormal.Dot(m_Direction);
        if (std::abs(denom) < EPSILON) return false;

        Vec3 diff = planePoint - m_Origin;
        t = diff.Dot(planeNormal) / denom;
        return t >= 0.0f;
    }

    // Möller-Trumbore algorithm
    bool Ray::IntersectTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2,
                                float& t, float& u, float& v) const
    {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 pvec = m_Direction.Cross(edge2);
        float det = edge1.Dot(pvec);

        if (std::abs(det) < EPSILON) return false;

        float invDet = 1.0f / det;
        Vec3 tvec = m_Origin - v0;
        u = tvec.Dot(pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;

        Vec3 qvec = tvec.Cross(edge1);
        v = m_Direction.Dot(qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;

        t = edge2.Dot(qvec) * invDet;
        return t >= 0.0f;
    }

} // namespace QuarkDyn
