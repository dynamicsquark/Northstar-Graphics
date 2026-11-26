// ============================================================================
// QuarkDyn SDK — Math Module
// color.h — RGBA Color
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"
#include "vec4.h"

namespace QuarkDyn
{
    struct Color
    {
        float r, g, b, a;

        // ==================== CONSTRUCTORS ====================
        Color();
        Color(float r, float g, float b, float a = 1.0f);
        Color(const Vec3& v, float a = 1.0f);
        Color(const Vec4& v);

        static Color FromUint32(uint32_t rgba);
        static Color FromHSV(float h, float s, float v, float a = 1.0f);

        // ==================== OPERATORS ====================
        Color operator+(const Color& c) const;
        Color operator-(const Color& c) const;
        Color operator*(float s) const;
        Color operator*(const Color& c) const;

        Color& operator+=(const Color& c);
        Color& operator-=(const Color& c);
        Color& operator*=(float s);

        bool operator==(const Color& c) const;
        bool operator!=(const Color& c) const;

        float& operator[](int index);
        const float& operator[](int index) const;

        // ==================== METHODS ====================
        Color   Clamped() const;
        Vec3    ToVec3() const;
        Vec4    ToVec4() const;
        uint32_t ToUint32() const;
        void    ToHSV(float& outH, float& outS, float& outV) const;
        Color   GammaToLinear() const;
        Color   LinearToGamma() const;
        float   Luminance() const;

        // ==================== STATIC METHODS ====================
        static Color Lerp(const Color& a, const Color& b, float t);
        static Color White();
        static Color Black();
        static Color Red();
        static Color Green();
        static Color Blue();
        static Color Yellow();
        static Color Cyan();
        static Color Magenta();
        static Color Transparent();
        static Color Gray(float brightness = 0.5f);
    };

    // ==================== GLOBAL OPERATORS ====================
    inline Color operator*(float s, const Color& c) { return c * s; }

} // namespace QuarkDyn
