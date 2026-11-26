// ============================================================================
// QuarkDyn SDK — Math Module
// vec3.h — 3D Vector
// Left-handed: X+ = Right, Y+ = Up, Z+ = Forward
// ============================================================================
#pragma once

#include "math_types.h"

namespace QuarkDyn
{
    struct Vec3
    {
        float x, y, z;

        // ==================== CONSTRUCTORS ====================
        Vec3();
        Vec3(float scalar);
        Vec3(float x, float y, float z);

        // ==================== OPERATORS ====================
        Vec3 operator+(const Vec3& v) const;
        Vec3 operator-(const Vec3& v) const;
        Vec3 operator*(float s) const;
        Vec3 operator*(const Vec3& v) const;
        Vec3 operator/(float s) const;
        Vec3 operator/(const Vec3& v) const;
        Vec3 operator-() const;

        Vec3& operator+=(const Vec3& v);
        Vec3& operator-=(const Vec3& v);
        Vec3& operator*=(float s);
        Vec3& operator*=(const Vec3& v);
        Vec3& operator/=(float s);

        bool operator==(const Vec3& v) const;
        bool operator!=(const Vec3& v) const;

        float& operator[](int index);
        const float& operator[](int index) const;

        // ==================== METHODS ====================
        float Dot(const Vec3& v) const;
        Vec3  Cross(const Vec3& v) const;
        float Length() const;
        float LengthSq() const;
        Vec3  Normalized() const;
        void  Normalize();
        float Distance(const Vec3& v) const;
        float DistanceSq(const Vec3& v) const;
        Vec3  Reflect(const Vec3& normal) const;
        Vec3  Project(const Vec3& onto) const;
        Vec3  Reject(const Vec3& from) const;
        float Angle(const Vec3& v) const;
        float SignedAngle(const Vec3& v, const Vec3& axis) const;

        // ==================== STATIC METHODS ====================
        static Vec3 Lerp(const Vec3& a, const Vec3& b, float t);
        static Vec3 Slerp(const Vec3& a, const Vec3& b, float t);
        static Vec3 MoveTowards(const Vec3& current, const Vec3& target, float maxDelta);
        static Vec3 ClampMagnitude(const Vec3& v, float maxLength);
        static Vec3 SmoothDamp(const Vec3& current, const Vec3& target, Vec3& velocity,
                               float smoothTime, float maxSpeed, float deltaTime);
        static void OrthoNormalize(Vec3& normal, Vec3& tangent);

        static Vec3 Zero();
        static Vec3 One();
        static Vec3 UnitX();
        static Vec3 UnitY();
        static Vec3 UnitZ();

        // Direction constants (Left-handed: Z+ = Forward)
        static Vec3 Right();     // ( 1,  0,  0)
        static Vec3 Left();      // (-1,  0,  0)
        static Vec3 Up();        // ( 0,  1,  0)
        static Vec3 Down();      // ( 0, -1,  0)
        static Vec3 Forward();   // ( 0,  0,  1)
        static Vec3 Back();      // ( 0,  0, -1)
    };

    // ==================== GLOBAL OPERATORS ====================
    inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

} // namespace QuarkDyn
