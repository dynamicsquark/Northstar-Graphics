// ============================================================================
// QuarkDyn SDK — Math Module
// vec4.h — 4D Vector
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"

namespace QuarkDyn
{
    struct Vec4
    {
        float x, y, z, w;

        // ==================== CONSTRUCTORS ====================
        Vec4();
        Vec4(float scalar);
        Vec4(float x, float y, float z, float w);
        Vec4(const Vec3& v, float w);

        // ==================== OPERATORS ====================
        Vec4 operator+(const Vec4& v) const;
        Vec4 operator-(const Vec4& v) const;
        Vec4 operator*(float s) const;
        Vec4 operator*(const Vec4& v) const;
        Vec4 operator/(float s) const;
        Vec4 operator-() const;

        Vec4& operator+=(const Vec4& v);
        Vec4& operator-=(const Vec4& v);
        Vec4& operator*=(float s);
        Vec4& operator/=(float s);

        bool operator==(const Vec4& v) const;
        bool operator!=(const Vec4& v) const;

        float& operator[](int index);
        const float& operator[](int index) const;

        // ==================== METHODS ====================
        float Dot(const Vec4& v) const;
        float Length() const;
        float LengthSq() const;
        Vec4  Normalized() const;
        void  Normalize();

        // ==================== SWIZZLE ====================
        Vec3 XYZ() const;

        // ==================== STATIC METHODS ====================
        static Vec4 Lerp(const Vec4& a, const Vec4& b, float t);
        static Vec4 Zero();
        static Vec4 One();
    };

    // ==================== GLOBAL OPERATORS ====================
    inline Vec4 operator*(float s, const Vec4& v) { return v * s; }

} // namespace QuarkDyn
