// ============================================================================
// QuarkDyn SDK — Math Module
// transform.h — TRS Transform (Position, Rotation, Scale)
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"
#include "quat.h"
#include "mat4.h"

namespace QuarkDyn
{
    struct Transform
    {
        Vec3 m_Position;
        Quat m_Rotation;
        Vec3 m_Scale;

        // ==================== CONSTRUCTORS ====================
        Transform();
        Transform(const Vec3& position, const Quat& rotation, const Vec3& scale);

        // ==================== METHODS ====================
        Mat4 ToMatrix() const;
        Vec3 TransformPoint(const Vec3& point) const;
        Vec3 TransformDirection(const Vec3& dir) const;
        Vec3 InverseTransformPoint(const Vec3& point) const;
        Vec3 InverseTransformDirection(const Vec3& dir) const;

        Vec3 Forward() const;
        Vec3 Right() const;
        Vec3 Up() const;

        Transform Inverse() const;

        // ==================== STATIC METHODS ====================
        static Transform Identity();
        static Transform Lerp(const Transform& a, const Transform& b, float t);
        static Transform Multiply(const Transform& parent, const Transform& child);
    };

} // namespace QuarkDyn
