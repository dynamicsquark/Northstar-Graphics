// ============================================================================
// QuarkDyn SDK — Math Module
// vec4.cpp — 4D Vector Implementation
// ============================================================================

#include "../include/vec4.h"
#include "../include/vec3.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Vec4::Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vec4::Vec4(float scalar) : x(scalar), y(scalar), z(scalar), w(scalar) {}
    Vec4::Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    Vec4::Vec4(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    // ==================== OPERATORS ====================
    Vec4 Vec4::operator+(const Vec4& v) const { return Vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
    Vec4 Vec4::operator-(const Vec4& v) const { return Vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
    Vec4 Vec4::operator*(float s) const { return Vec4(x * s, y * s, z * s, w * s); }
    Vec4 Vec4::operator*(const Vec4& v) const { return Vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
    Vec4 Vec4::operator/(float s) const { float inv = 1.0f / s; return Vec4(x * inv, y * inv, z * inv, w * inv); }
    Vec4 Vec4::operator-() const { return Vec4(-x, -y, -z, -w); }

    Vec4& Vec4::operator+=(const Vec4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
    Vec4& Vec4::operator-=(const Vec4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
    Vec4& Vec4::operator*=(float s) { x *= s; y *= s; z *= s; w *= s; return *this; }
    Vec4& Vec4::operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; w *= inv; return *this; }

    bool Vec4::operator==(const Vec4& v) const
    {
        return ApproxEqual(x, v.x) && ApproxEqual(y, v.y) && ApproxEqual(z, v.z) && ApproxEqual(w, v.w);
    }
    bool Vec4::operator!=(const Vec4& v) const { return !(*this == v); }

    float& Vec4::operator[](int index)
    {
        assert(index >= 0 && index < 4);
        switch (index)
        {
            case 0:  return x;
            case 1:  return y;
            case 2:  return z;
            default: return w;
        }
    }

    const float& Vec4::operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        switch (index)
        {
            case 0:  return x;
            case 1:  return y;
            case 2:  return z;
            default: return w;
        }
    }

    // ==================== METHODS ====================
    float Vec4::Dot(const Vec4& v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
    float Vec4::Length() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    float Vec4::LengthSq() const { return x * x + y * y + z * z + w * w; }

    Vec4 Vec4::Normalized() const
    {
        float len = Length();
        if (len < EPSILON) return Vec4::Zero();
        float inv = 1.0f / len;
        return Vec4(x * inv, y * inv, z * inv, w * inv);
    }

    void Vec4::Normalize()
    {
        float len = Length();
        if (len < EPSILON) { x = 0.0f; y = 0.0f; z = 0.0f; w = 0.0f; return; }
        float inv = 1.0f / len;
        x *= inv; y *= inv; z *= inv; w *= inv;
    }

    // ==================== SWIZZLE ====================
    Vec3 Vec4::XYZ() const { return Vec3(x, y, z); }

    // ==================== STATIC METHODS ====================
    Vec4 Vec4::Lerp(const Vec4& a, const Vec4& b, float t)
    {
        return Vec4(
            QuarkDyn::Lerp(a.x, b.x, t),
            QuarkDyn::Lerp(a.y, b.y, t),
            QuarkDyn::Lerp(a.z, b.z, t),
            QuarkDyn::Lerp(a.w, b.w, t)
        );
    }

    Vec4 Vec4::Zero() { return Vec4(0.0f, 0.0f, 0.0f, 0.0f); }
    Vec4 Vec4::One()  { return Vec4(1.0f, 1.0f, 1.0f, 1.0f); }

} // namespace QuarkDyn
