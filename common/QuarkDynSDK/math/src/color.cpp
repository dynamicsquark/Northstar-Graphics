// ============================================================================
// QuarkDyn SDK — Math Module
// color.cpp — RGBA Color Implementation
// ============================================================================

#include "../include/color.h"
#include "../include/vec3.h"
#include "../include/vec4.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Color::Color() : r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}
    Color::Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
    Color::Color(const Vec3& v, float a) : r(v.x), g(v.y), b(v.z), a(a) {}
    Color::Color(const Vec4& v) : r(v.x), g(v.y), b(v.z), a(v.w) {}

    Color Color::FromUint32(uint32_t rgba)
    {
        return Color(
            ((rgba >> 24) & 0xFF) / 255.0f,
            ((rgba >> 16) & 0xFF) / 255.0f,
            ((rgba >> 8) & 0xFF) / 255.0f,
            (rgba & 0xFF) / 255.0f
        );
    }

    Color Color::FromHSV(float h, float s, float v, float a)
    {
        if (!std::isfinite(h)) h = 0.0f;
        h = std::fmod(h, 360.0f);
        if (h < 0.0f) h += 360.0f;
        s = Clamp01(s);

        float c = v * s;
        float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = v - c;

        float r, g, b;
        if (h < 60.0f)       { r = c; g = x; b = 0; }
        else if (h < 120.0f) { r = x; g = c; b = 0; }
        else if (h < 180.0f) { r = 0; g = c; b = x; }
        else if (h < 240.0f) { r = 0; g = x; b = c; }
        else if (h < 300.0f) { r = x; g = 0; b = c; }
        else                 { r = c; g = 0; b = x; }

        return Color(r + m, g + m, b + m, a);
    }

    // ==================== OPERATORS ====================
    Color Color::operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b, a + c.a); }
    Color Color::operator-(const Color& c) const { return Color(r - c.r, g - c.g, b - c.b, a - c.a); }
    Color Color::operator*(float s) const { return Color(r * s, g * s, b * s, a * s); }
    Color Color::operator*(const Color& c) const { return Color(r * c.r, g * c.g, b * c.b, a * c.a); }

    Color& Color::operator+=(const Color& c) { r += c.r; g += c.g; b += c.b; a += c.a; return *this; }
    Color& Color::operator-=(const Color& c) { r -= c.r; g -= c.g; b -= c.b; a -= c.a; return *this; }
    Color& Color::operator*=(float s) { r *= s; g *= s; b *= s; a *= s; return *this; }

    bool Color::operator==(const Color& c) const
    {
        return ApproxEqual(r, c.r) && ApproxEqual(g, c.g) && ApproxEqual(b, c.b) && ApproxEqual(a, c.a);
    }
    bool Color::operator!=(const Color& c) const { return !(*this == c); }

    float& Color::operator[](int index)
    {
        assert(index >= 0 && index < 4);
        switch (index)
        {
            case 0:  return r;
            case 1:  return g;
            case 2:  return b;
            default: return a;
        }
    }

    const float& Color::operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        switch (index)
        {
            case 0:  return r;
            case 1:  return g;
            case 2:  return b;
            default: return a;
        }
    }

    // ==================== METHODS ====================
    Color Color::Clamped() const
    {
        return Color(
            Clamp01(r), Clamp01(g), Clamp01(b), Clamp01(a)
        );
    }

    Vec3 Color::ToVec3() const { return Vec3(r, g, b); }
    Vec4 Color::ToVec4() const { return Vec4(r, g, b, a); }

    uint32_t Color::ToUint32() const
    {
        Color c = Clamped();
        uint32_t ri = static_cast<uint32_t>(c.r * 255.0f + 0.5f);
        uint32_t gi = static_cast<uint32_t>(c.g * 255.0f + 0.5f);
        uint32_t bi = static_cast<uint32_t>(c.b * 255.0f + 0.5f);
        uint32_t ai = static_cast<uint32_t>(c.a * 255.0f + 0.5f);
        return (ri << 24) | (gi << 16) | (bi << 8) | ai;
    }

    void Color::ToHSV(float& outH, float& outS, float& outV) const
    {
        float cMax = Max(r, Max(g, b));
        float cMin = Min(r, Min(g, b));
        float delta = cMax - cMin;

        outV = cMax;
        outS = (cMax > EPSILON) ? (delta / cMax) : 0.0f;

        if (delta < EPSILON)
        {
            outH = 0.0f;
        }
        else if (ApproxEqual(cMax, r))
        {
            outH = 60.0f * std::fmod((g - b) / delta, 6.0f);
        }
        else if (ApproxEqual(cMax, g))
        {
            outH = 60.0f * ((b - r) / delta + 2.0f);
        }
        else
        {
            outH = 60.0f * ((r - g) / delta + 4.0f);
        }

        if (outH < 0.0f) outH += 360.0f;
    }

    Color Color::GammaToLinear() const
    {
        auto toLinear = [](float v) -> float {
            return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
        };
        return Color(toLinear(r), toLinear(g), toLinear(b), a);
    }

    Color Color::LinearToGamma() const
    {
        auto toGamma = [](float v) -> float {
            return (v <= 0.0031308f) ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
        };
        return Color(toGamma(r), toGamma(g), toGamma(b), a);
    }

    float Color::Luminance() const
    {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }

    // ==================== STATIC METHODS ====================
    Color Color::Lerp(const Color& a, const Color& b, float t)
    {
        return Color(
            QuarkDyn::Lerp(a.r, b.r, t),
            QuarkDyn::Lerp(a.g, b.g, t),
            QuarkDyn::Lerp(a.b, b.b, t),
            QuarkDyn::Lerp(a.a, b.a, t)
        );
    }

    Color Color::White()       { return Color(1.0f, 1.0f, 1.0f, 1.0f); }
    Color Color::Black()       { return Color(0.0f, 0.0f, 0.0f, 1.0f); }
    Color Color::Red()         { return Color(1.0f, 0.0f, 0.0f, 1.0f); }
    Color Color::Green()       { return Color(0.0f, 1.0f, 0.0f, 1.0f); }
    Color Color::Blue()        { return Color(0.0f, 0.0f, 1.0f, 1.0f); }
    Color Color::Yellow()      { return Color(1.0f, 1.0f, 0.0f, 1.0f); }
    Color Color::Cyan()        { return Color(0.0f, 1.0f, 1.0f, 1.0f); }
    Color Color::Magenta()     { return Color(1.0f, 0.0f, 1.0f, 1.0f); }
    Color Color::Transparent() { return Color(0.0f, 0.0f, 0.0f, 0.0f); }
    Color Color::Gray(float brightness) { return Color(brightness, brightness, brightness, 1.0f); }

} // namespace QuarkDyn
