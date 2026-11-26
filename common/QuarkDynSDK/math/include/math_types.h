// ============================================================================
// QuarkDyn SDK — Math Module
// math_types.h — Constants, utility functions, and forward declarations
//
// Coordinate System: Left-handed
//   X+ = Right, Y+ = Up, Z+ = Forward
// Matrix Layout: Column-major
// ============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cfloat>
#include <cassert>
#include <algorithm>

namespace QuarkDyn
{
    // ==================== CONSTANTS ====================
    constexpr float PI        = 3.14159265358979323846f;
    constexpr float TAU       = 6.28318530717958647692f;
    constexpr float HALF_PI   = 1.57079632679489661923f;
    constexpr float DEG2RAD   = PI / 180.0f;
    constexpr float RAD2DEG   = 180.0f / PI;
    constexpr float EPSILON      = 1e-6f;
    constexpr float SMALL_NUMBER = 1e-8f;
    constexpr float INFINITY_    = FLT_MAX;

    // ==================== UTILITY FUNCTIONS ====================
    template<typename T>
    inline T Min(const T& a, const T& b) { return (a < b) ? a : b; }

    template<typename T>
    inline T Max(const T& a, const T& b) { return (a > b) ? a : b; }

    inline float Clamp(float v, float minVal, float maxVal)
    {
        return Max(minVal, Min(v, maxVal));
    }

    inline float Clamp01(float v)
    {
        return Clamp(v, 0.0f, 1.0f);
    }

    inline float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    inline float InverseLerp(float a, float b, float value)
    {
        float denom = b - a;
        if (std::abs(denom) < EPSILON) return 0.0f;
        float safeDenom = (std::abs(denom) < EPSILON) ? 1.0f : denom;
        return (value - a) / safeDenom;
    }

    inline float Radians(float degrees) { return degrees * DEG2RAD; }
    inline float Degrees(float radians) { return radians * RAD2DEG; }

    inline float Abs(float v) { return std::abs(v); }

    inline float SmoothStep(float edge0, float edge1, float x)
    {
        float denom = edge1 - edge0;
        if (std::abs(denom) < SMALL_NUMBER) return (x >= edge1) ? 1.0f : 0.0f;
        float safeDenom = (std::abs(denom) < SMALL_NUMBER) ? 1.0f : denom;
        float t = Clamp01((x - edge0) / safeDenom);
        return t * t * (3.0f - 2.0f * t);
    }

    inline float SmootherStep(float edge0, float edge1, float x)
    {
        float denom = edge1 - edge0;
        if (std::abs(denom) < SMALL_NUMBER) return (x >= edge1) ? 1.0f : 0.0f;
        float safeDenom = (std::abs(denom) < SMALL_NUMBER) ? 1.0f : denom;
        float t = Clamp01((x - edge0) / safeDenom);
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    inline float Sign(float v)
    {
        if (v > EPSILON) return 1.0f;
        if (v < -EPSILON) return -1.0f;
        return 0.0f;
    }

    inline float Remap(float inMin, float inMax, float outMin, float outMax, float value)
    {
        float t = InverseLerp(inMin, inMax, value);
        return Lerp(outMin, outMax, t);
    }

    inline bool ApproxEqual(float a, float b, float epsilon = EPSILON)
    {
        return std::abs(a - b) < epsilon;
    }

    inline float MoveTowards(float current, float target, float maxDelta)
    {
        if (maxDelta == 0.0f) return current;
        float diff = target - current;
        if (std::abs(diff) <= maxDelta) return target;
        return current + ((diff > 0.0f) ? maxDelta : -maxDelta);
    }

    inline float Repeat(float t, float length)
    {
        if (length <= SMALL_NUMBER) return 0.0f;
        float safeLength = (length <= SMALL_NUMBER) ? 1.0f : length;
        return Clamp(t - std::floor(t / safeLength) * safeLength, 0.0f, safeLength);
    }

    inline float PingPong(float t, float length)
    {
        if (length <= SMALL_NUMBER) return 0.0f;
        t = Repeat(t, length * 2.0f);
        return length - std::abs(t - length);
    }

} // namespace QuarkDyn
