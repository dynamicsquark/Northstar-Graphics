// ============================================================================
// QuarkDyn SDK — Math Module
// vec2.h — 2D Vector
// ============================================================================
#pragma once

#include "math_types.h"

namespace QuarkDyn
{
    struct Vec2
    {
        float x, y;

        // ==================== CONSTRUCTORS ====================
        Vec2();
        Vec2(float scalar);
        Vec2(float x, float y);

        // ==================== OPERATORS ====================
        Vec2 operator+(const Vec2& v) const;
        Vec2 operator-(const Vec2& v) const;
        Vec2 operator*(float s) const;
        Vec2 operator*(const Vec2& v) const;
        Vec2 operator/(float s) const;
        Vec2 operator/(const Vec2& v) const;
        Vec2 operator-() const;

        Vec2& operator+=(const Vec2& v);
        Vec2& operator-=(const Vec2& v);
        Vec2& operator*=(float s);
        Vec2& operator*=(const Vec2& v);
        Vec2& operator/=(float s);

        bool operator==(const Vec2& v) const;
        bool operator!=(const Vec2& v) const;

        float& operator[](int index);
        const float& operator[](int index) const;

        // ==================== METHODS ====================
        float Dot(const Vec2& v) const;
        float Cross(const Vec2& v) const;  // 2D cross product (returns scalar)
        float Length() const;
        float LengthSq() const;
        Vec2  Normalized() const;
        void  Normalize();
        float Distance(const Vec2& v) const;
        float DistanceSq(const Vec2& v) const;
        Vec2  Reflect(const Vec2& normal) const;
        Vec2  Perpendicular() const;
        float Angle(const Vec2& v) const;

        // ==================== STATIC METHODS ====================
        static Vec2 Lerp(const Vec2& a, const Vec2& b, float t);
        static Vec2 MoveTowards(const Vec2& current, const Vec2& target, float maxDelta);
        static Vec2 Zero();
        static Vec2 One();
        static Vec2 UnitX();
        static Vec2 UnitY();
    };

    // ==================== GLOBAL OPERATORS ====================
    inline Vec2 operator*(float s, const Vec2& v) { return v * s; }

} // namespace QuarkDyn
