// ============================================================================
// QuarkDyn SDK — Math Module
// vec2.cpp — 2D Vector Implementation
// ============================================================================

#include "../include/vec2.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Vec2::Vec2() : x(0.0f), y(0.0f) {}
    Vec2::Vec2(float scalar) : x(scalar), y(scalar) {}
    Vec2::Vec2(float x, float y) : x(x), y(y) {}

    // ==================== OPERATORS ====================
    Vec2 Vec2::operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 Vec2::operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 Vec2::operator*(float s) const { return Vec2(x * s, y * s); }
    Vec2 Vec2::operator*(const Vec2& v) const { return Vec2(x * v.x, y * v.y); }
    Vec2 Vec2::operator/(float s) const { float inv = 1.0f / s; return Vec2(x * inv, y * inv); }
    Vec2 Vec2::operator/(const Vec2& v) const { return Vec2(x / v.x, y / v.y); }
    Vec2 Vec2::operator-() const { return Vec2(-x, -y); }

    Vec2& Vec2::operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    Vec2& Vec2::operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
    Vec2& Vec2::operator*=(float s) { x *= s; y *= s; return *this; }
    Vec2& Vec2::operator*=(const Vec2& v) { x *= v.x; y *= v.y; return *this; }
    Vec2& Vec2::operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; return *this; }

    bool Vec2::operator==(const Vec2& v) const { return ApproxEqual(x, v.x) && ApproxEqual(y, v.y); }
    bool Vec2::operator!=(const Vec2& v) const { return !(*this == v); }

    float& Vec2::operator[](int index)
    {
        assert(index >= 0 && index < 2);
        return (index == 0) ? x : y;
    }

    const float& Vec2::operator[](int index) const
    {
        assert(index >= 0 && index < 2);
        return (index == 0) ? x : y;
    }

    // ==================== METHODS ====================
    float Vec2::Dot(const Vec2& v) const { return x * v.x + y * v.y; }
    float Vec2::Cross(const Vec2& v) const { return x * v.y - y * v.x; }
    float Vec2::Length() const { return std::sqrt(x * x + y * y); }
    float Vec2::LengthSq() const { return x * x + y * y; }

    Vec2 Vec2::Normalized() const
    {
        float len = Length();
        if (len < EPSILON) return Vec2::Zero();
        float inv = 1.0f / len;
        return Vec2(x * inv, y * inv);
    }

    void Vec2::Normalize()
    {
        float len = Length();
        if (len < EPSILON) { x = 0.0f; y = 0.0f; return; }
        float inv = 1.0f / len;
        x *= inv;
        y *= inv;
    }

    float Vec2::Distance(const Vec2& v) const { return (*this - v).Length(); }
    float Vec2::DistanceSq(const Vec2& v) const { return (*this - v).LengthSq(); }

    Vec2 Vec2::Reflect(const Vec2& normal) const
    {
        return *this - normal * (2.0f * Dot(normal));
    }

    Vec2 Vec2::Perpendicular() const { return Vec2(-y, x); }

    float Vec2::Angle(const Vec2& v) const
    {
        float d = Dot(v);
        float lenProduct = Length() * v.Length();
        if (lenProduct < EPSILON) return 0.0f;
        return std::acos(Clamp(d / lenProduct, -1.0f, 1.0f));
    }

    // ==================== STATIC METHODS ====================
    Vec2 Vec2::Lerp(const Vec2& a, const Vec2& b, float t)
    {
        return Vec2(
            QuarkDyn::Lerp(a.x, b.x, t),
            QuarkDyn::Lerp(a.y, b.y, t)
        );
    }

    Vec2 Vec2::MoveTowards(const Vec2& current, const Vec2& target, float maxDelta)
    {
        Vec2 diff = target - current;
        float dist = diff.Length();
        if (dist <= maxDelta || dist < EPSILON) return target;
        return current + diff * (maxDelta / dist);
    }

    Vec2 Vec2::Zero()  { return Vec2(0.0f, 0.0f); }
    Vec2 Vec2::One()   { return Vec2(1.0f, 1.0f); }
    Vec2 Vec2::UnitX() { return Vec2(1.0f, 0.0f); }
    Vec2 Vec2::UnitY() { return Vec2(0.0f, 1.0f); }

} // namespace QuarkDyn
