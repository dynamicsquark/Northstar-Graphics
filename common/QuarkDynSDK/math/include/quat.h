// ============================================================================
// QuarkDyn SDK — Math Module
// quat.h — Quaternion (Left-handed)
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"
#include "mat3.h"
#include "mat4.h"

namespace QuarkDyn
{
    struct Quat
    {
        float x, y, z, w;

        // ==================== CONSTRUCTORS ====================
        Quat();                                         // Identity (0,0,0,1)
        Quat(float x, float y, float z, float w);
        Quat(const Vec3& axis, float angle);

        // ==================== OPERATORS ====================
        Quat operator*(const Quat& q) const;
        Vec3 operator*(const Vec3& v) const;
        Quat operator*(float s) const;
        Quat operator+(const Quat& q) const;
        Quat operator-() const;

        bool operator==(const Quat& q) const;
        bool operator!=(const Quat& q) const;

        // ==================== METHODS ====================
        float Dot(const Quat& q) const;
        float Length() const;
        float LengthSq() const;
        Quat  Normalized() const;
        void  Normalize();
        Quat  Conjugate() const;
        Quat  Inverse() const;

        Mat4 ToMatrix() const;
        Mat3 ToMat3() const;
        Vec3 ToEuler() const;

        Vec3 GetForward() const;
        Vec3 GetRight() const;
        Vec3 GetUp() const;

        // ==================== STATIC METHODS ====================
        static Quat Identity();
        static Quat FromEuler(const Vec3& euler);
        static Quat FromEuler(float pitch, float yaw, float roll);
        static Quat Slerp(const Quat& a, const Quat& b, float t);
        static Quat Nlerp(const Quat& a, const Quat& b, float t);
        static Quat FromLookRotation(const Vec3& forward, const Vec3& up);
        static Quat FromMatrix(const Mat3& m);
        static Quat FromAxisAngle(const Vec3& axis, float angle);
        static Quat RotateTowards(const Quat& from, const Quat& to, float maxAngle);
        static float AngleBetween(const Quat& a, const Quat& b);
    };

} // namespace QuarkDyn
