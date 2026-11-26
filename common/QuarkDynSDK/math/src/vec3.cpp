// ============================================================================
// QuarkDyn SDK — Math Module
// vec3.cpp — 3D Vector Implementation
// Left-handed: X+ = Right, Y+ = Up, Z+ = Forward
// ============================================================================

#include "../include/vec3.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Vec3::Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vec3::Vec3(float scalar) : x(scalar), y(scalar), z(scalar) {}
    Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    // ==================== OPERATORS ====================
    Vec3 Vec3::operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 Vec3::operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 Vec3::operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 Vec3::operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 Vec3::operator/(float s) const { float inv = 1.0f / s; return Vec3(x * inv, y * inv, z * inv); }
    Vec3 Vec3::operator/(const Vec3& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }
    Vec3 Vec3::operator-() const { return Vec3(-x, -y, -z); }

    Vec3& Vec3::operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& Vec3::operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& Vec3::operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Vec3& Vec3::operator*=(const Vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    Vec3& Vec3::operator/=(float s) { float inv = 1.0f / s; x *= inv; y *= inv; z *= inv; return *this; }

    bool Vec3::operator==(const Vec3& v) const { return ApproxEqual(x, v.x) && ApproxEqual(y, v.y) && ApproxEqual(z, v.z); }
    bool Vec3::operator!=(const Vec3& v) const { return !(*this == v); }

    float& Vec3::operator[](int index)
    {
        assert(index >= 0 && index < 3);
        switch (index)
        {
            case 0:  return x;
            case 1:  return y;
            default: return z;
        }
    }

    const float& Vec3::operator[](int index) const
    {
        assert(index >= 0 && index < 3);
        switch (index)
        {
            case 0:  return x;
            case 1:  return y;
            default: return z;
        }
    }

    // ==================== METHODS ====================
    float Vec3::Dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    // Cross product — same mathematical formula for both LH and RH.
    // In LH convention: Right × Up = Forward  (positive Z)
    Vec3 Vec3::Cross(const Vec3& v) const
    {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    float Vec3::Length() const { return std::sqrt(x * x + y * y + z * z); }
    float Vec3::LengthSq() const { return x * x + y * y + z * z; }

    Vec3 Vec3::Normalized() const
    {
        float len = Length();
        if (len < EPSILON) return Vec3::Zero();
        float inv = 1.0f / len;
        return Vec3(x * inv, y * inv, z * inv);
    }

    void Vec3::Normalize()
    {
        float len = Length();
        if (len < EPSILON) { x = 0.0f; y = 0.0f; z = 0.0f; return; }
        float inv = 1.0f / len;
        x *= inv; y *= inv; z *= inv;
    }

    float Vec3::Distance(const Vec3& v) const { return (*this - v).Length(); }
    float Vec3::DistanceSq(const Vec3& v) const { return (*this - v).LengthSq(); }

    Vec3 Vec3::Reflect(const Vec3& normal) const
    {
        return *this - normal * (2.0f * Dot(normal));
    }

    Vec3 Vec3::Project(const Vec3& onto) const
    {
        float d = onto.Dot(onto);
        if (d < EPSILON) return Vec3::Zero();
        return onto * (Dot(onto) / d);
    }

    Vec3 Vec3::Reject(const Vec3& from) const
    {
        return *this - Project(from);
    }

    float Vec3::Angle(const Vec3& v) const
    {
        float d = Dot(v);
        float lenProduct = Length() * v.Length();
        if (lenProduct < EPSILON) return 0.0f;
        return std::acos(Clamp(d / lenProduct, -1.0f, 1.0f));
    }

    float Vec3::SignedAngle(const Vec3& v, const Vec3& axis) const
    {
        float unsignedAngle = Angle(v);
        Vec3 cross = Cross(v);
        float sign = cross.Dot(axis);
        return (sign < 0.0f) ? -unsignedAngle : unsignedAngle;
    }

    // ==================== STATIC METHODS ====================
    Vec3 Vec3::Lerp(const Vec3& a, const Vec3& b, float t)
    {
        return Vec3(
            QuarkDyn::Lerp(a.x, b.x, t),
            QuarkDyn::Lerp(a.y, b.y, t),
            QuarkDyn::Lerp(a.z, b.z, t)
        );
    }

    Vec3 Vec3::Slerp(const Vec3& a, const Vec3& b, float t)
    {
        float lenA = a.Length();
        float lenB = b.Length();
        if (lenA < EPSILON || lenB < EPSILON)
            return Lerp(a, b, t);

        Vec3 na = a * (1.0f / lenA);
        Vec3 nb = b * (1.0f / lenB);
        float targetLen = QuarkDyn::Lerp(lenA, lenB, t);

        float dot = Clamp(na.Dot(nb), -1.0f, 1.0f);
        if (dot > 0.9995f)
            return Lerp(na, nb, t).Normalized() * targetLen;

        if (dot < -0.9995f)
        {
            // Antipodal directions: choose a deterministic unit axis orthogonal to na
            Vec3 refAxis = (std::abs(na.x) < 0.9f) ? Vec3::UnitX() : Vec3::UnitY();
            Vec3 ortho = (refAxis - na * refAxis.Dot(na)).Normalized();
            float angle = PI * t;
            return (na * std::cos(angle) + ortho * std::sin(angle)) * targetLen;
        }

        float theta = std::acos(dot);
        float sinTheta = std::sin(theta);
        float wa = std::sin((1.0f - t) * theta) / sinTheta;
        float wb = std::sin(t * theta) / sinTheta;
        return (na * wa + nb * wb) * targetLen;
    }

    Vec3 Vec3::MoveTowards(const Vec3& current, const Vec3& target, float maxDelta)
    {
        Vec3 diff = target - current;
        float dist = diff.Length();
        if (dist <= maxDelta || dist < EPSILON) return target;
        return current + diff * (maxDelta / dist);
    }

    Vec3 Vec3::ClampMagnitude(const Vec3& v, float maxLength)
    {
        float lenSq = v.LengthSq();
        if (lenSq <= maxLength * maxLength) return v;
        float len = std::sqrt(lenSq);
        return v * (maxLength / len);
    }

    Vec3 Vec3::Zero()    { return Vec3(0.0f, 0.0f, 0.0f); }
    Vec3 Vec3::One()     { return Vec3(1.0f, 1.0f, 1.0f); }
    Vec3 Vec3::UnitX()   { return Vec3(1.0f, 0.0f, 0.0f); }
    Vec3 Vec3::UnitY()   { return Vec3(0.0f, 1.0f, 0.0f); }
    Vec3 Vec3::UnitZ()   { return Vec3(0.0f, 0.0f, 1.0f); }

    // Left-handed directions: X+ = Right, Y+ = Up, Z+ = Forward
    Vec3 Vec3::Right()   { return Vec3( 1.0f,  0.0f,  0.0f); }
    Vec3 Vec3::Left()    { return Vec3(-1.0f,  0.0f,  0.0f); }
    Vec3 Vec3::Up()      { return Vec3( 0.0f,  1.0f,  0.0f); }
    Vec3 Vec3::Down()    { return Vec3( 0.0f, -1.0f,  0.0f); }
    Vec3 Vec3::Forward() { return Vec3( 0.0f,  0.0f,  1.0f); }
    Vec3 Vec3::Back()    { return Vec3( 0.0f,  0.0f, -1.0f); }

    Vec3 Vec3::SmoothDamp(const Vec3& current, const Vec3& target, Vec3& velocity,
                          float smoothTime, float maxSpeed, float deltaTime)
    {
        if (deltaTime <= SMALL_NUMBER)
            return current;

        // Critically damped spring-damper interpolation
        smoothTime = Max(0.0001f, smoothTime);
        float omega = 2.0f / smoothTime;
        float x = omega * deltaTime;
        // Padé approximant for exp(-x)
        float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);

        Vec3 diff = current - target;

        // Clamp magnitude
        float maxDist = maxSpeed * smoothTime;
        diff = ClampMagnitude(diff, maxDist);

        Vec3 clampedTarget = current - diff;
        Vec3 temp = (velocity + diff * omega) * deltaTime;
        velocity = (velocity - temp * omega) * exp;

        Vec3 result = clampedTarget + (diff + temp) * exp;

        // Prevent overshoot
        Vec3 origMinusCurrent = target - current;
        Vec3 resultMinusTarget = result - target;
        if (origMinusCurrent.Dot(resultMinusTarget) > 0.0f)
        {
            result = target;
            velocity = Vec3::Zero();
        }

        return result;
    }

    void Vec3::OrthoNormalize(Vec3& normal, Vec3& tangent)
    {
        // Gram-Schmidt orthogonalization
        normal.Normalize();
        tangent = tangent - normal * tangent.Dot(normal);
        tangent.Normalize();
    }

} // namespace QuarkDyn
