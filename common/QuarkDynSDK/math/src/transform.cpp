// ============================================================================
// QuarkDyn SDK — Math Module
// transform.cpp — TRS Transform Implementation
// ============================================================================

#include "../include/transform.h"
#include "../include/mat4.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Transform::Transform()
        : m_Position(Vec3::Zero())
        , m_Rotation(Quat::Identity())
        , m_Scale(Vec3::One())
    {}

    Transform::Transform(const Vec3& position, const Quat& rotation, const Vec3& scale)
        : m_Position(position)
        , m_Rotation(rotation)
        , m_Scale(scale)
    {}

    // ==================== METHODS ====================
    Mat4 Transform::ToMatrix() const
    {
        return Mat4::TRS(m_Position, m_Rotation, m_Scale);
    }

    Vec3 Transform::TransformPoint(const Vec3& point) const
    {
        return m_Rotation * (m_Scale * point) + m_Position;
    }

    Vec3 Transform::TransformDirection(const Vec3& dir) const
    {
        return m_Rotation * dir;
    }

    Vec3 Transform::InverseTransformPoint(const Vec3& point) const
    {
        Vec3 local = m_Rotation.Inverse() * (point - m_Position);
        float invSx = (std::abs(m_Scale.x) > SMALL_NUMBER) ? (1.0f / m_Scale.x) : 0.0f;
        float invSy = (std::abs(m_Scale.y) > SMALL_NUMBER) ? (1.0f / m_Scale.y) : 0.0f;
        float invSz = (std::abs(m_Scale.z) > SMALL_NUMBER) ? (1.0f / m_Scale.z) : 0.0f;
        return Vec3(local.x * invSx, local.y * invSy, local.z * invSz);
    }

    Vec3 Transform::InverseTransformDirection(const Vec3& dir) const
    {
        return m_Rotation.Inverse() * dir;
    }

    Vec3 Transform::Forward() const { return m_Rotation.GetForward(); }
    Vec3 Transform::Right() const   { return m_Rotation.GetRight(); }
    Vec3 Transform::Up() const      { return m_Rotation.GetUp(); }

    Transform Transform::Inverse() const
    {
        Quat invRot = m_Rotation.Inverse();
        Vec3 invScale(
            (std::abs(m_Scale.x) > SMALL_NUMBER) ? (1.0f / m_Scale.x) : 0.0f,
            (std::abs(m_Scale.y) > SMALL_NUMBER) ? (1.0f / m_Scale.y) : 0.0f,
            (std::abs(m_Scale.z) > SMALL_NUMBER) ? (1.0f / m_Scale.z) : 0.0f
        );
        // Exact TRS inverse for rigid and uniform-scale transforms; for non-uniformly scaled
        // rotated transforms (which induce shear outside TRS), maps m_Position back to origin.
        Vec3 invPos = (invRot * (m_Position * -1.0f)) * invScale;
        return Transform(invPos, invRot, invScale);
    }

    // ==================== STATIC METHODS ====================
    Transform Transform::Identity()
    {
        return Transform();
    }

    Transform Transform::Lerp(const Transform& a, const Transform& b, float t)
    {
        return Transform(
            Vec3::Lerp(a.m_Position, b.m_Position, t),
            Quat::Slerp(a.m_Rotation, b.m_Rotation, t),
            Vec3::Lerp(a.m_Scale, b.m_Scale, t)
        );
    }

    Transform Transform::Multiply(const Transform& parent, const Transform& child)
    {
        return Transform(
            parent.TransformPoint(child.m_Position),
            parent.m_Rotation * child.m_Rotation,
            parent.m_Scale * child.m_Scale
        );
    }

} // namespace QuarkDyn
