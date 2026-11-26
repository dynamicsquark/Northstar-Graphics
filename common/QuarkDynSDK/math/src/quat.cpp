// ============================================================================
// QuarkDyn SDK — Math Module
// quat.cpp — Quaternion Implementation (Left-handed)
// ============================================================================

#include "../include/quat.h"
#include "../include/vec3.h"
#include "../include/mat3.h"
#include "../include/mat4.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Quat::Quat() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    Quat::Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    Quat::Quat(const Vec3& axis, float angle)
    {
        float lenSq = axis.LengthSq();
        if (lenSq < SMALL_NUMBER * SMALL_NUMBER)
        {
            x = 0.0f; y = 0.0f; z = 0.0f; w = 1.0f;
            return;
        }
        float halfAngle = angle * 0.5f;
        float s = std::sin(halfAngle);
        Vec3 n = axis * (1.0f / std::sqrt(lenSq));
        x = n.x * s;
        y = n.y * s;
        z = n.z * s;
        w = std::cos(halfAngle);
    }

    // ==================== OPERATORS ====================
    Quat Quat::operator*(const Quat& q) const
    {
        return Quat(
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w,
            w * q.w - x * q.x - y * q.y - z * q.z
        );
    }

    Vec3 Quat::operator*(const Vec3& v) const
    {
        Vec3 qv(x, y, z);
        Vec3 uv = qv.Cross(v);
        Vec3 uuv = qv.Cross(uv);
        return v + (uv * w + uuv) * 2.0f;
    }

    Quat Quat::operator*(float s) const { return Quat(x * s, y * s, z * s, w * s); }
    Quat Quat::operator+(const Quat& q) const { return Quat(x + q.x, y + q.y, z + q.z, w + q.w); }
    Quat Quat::operator-() const { return Quat(-x, -y, -z, -w); }

    bool Quat::operator==(const Quat& q) const
    {
        return ApproxEqual(x, q.x) && ApproxEqual(y, q.y) && ApproxEqual(z, q.z) && ApproxEqual(w, q.w);
    }
    bool Quat::operator!=(const Quat& q) const { return !(*this == q); }

    // ==================== METHODS ====================
    float Quat::Dot(const Quat& q) const { return x * q.x + y * q.y + z * q.z + w * q.w; }
    float Quat::Length() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    float Quat::LengthSq() const { return x * x + y * y + z * z + w * w; }

    Quat Quat::Normalized() const
    {
        float len = Length();
        if (len < EPSILON) return Quat::Identity();
        float inv = 1.0f / len;
        return Quat(x * inv, y * inv, z * inv, w * inv);
    }

    void Quat::Normalize()
    {
        float len = Length();
        if (len < EPSILON) { *this = Quat::Identity(); return; }
        float inv = 1.0f / len;
        x *= inv; y *= inv; z *= inv; w *= inv;
    }

    Quat Quat::Conjugate() const { return Quat(-x, -y, -z, w); }

    Quat Quat::Inverse() const
    {
        float lenSq = LengthSq();
        if (lenSq < EPSILON) return Quat::Identity();
        float inv = 1.0f / lenSq;
        return Quat(-x * inv, -y * inv, -z * inv, w * inv);
    }

    Mat4 Quat::ToMatrix() const { return Mat4::FromQuat(*this); }
    Mat3 Quat::ToMat3() const { return Mat3::FromQuat(*this); }

    // Left-handed Euler extraction (ZXY intrinsic order: Q = Qy(yaw) * Qx(pitch) * Qz(roll))
    // euler.x = pitch (X), euler.y = yaw (Y), euler.z = roll (Z)
    Vec3 Quat::ToEuler() const
    {
        Quat q = Normalized();
        Vec3 euler;

        // R12 = 2(yz - wx) = -sin(pitch)  =>  sin(pitch) = 2(wx - yz)
        float sinP = 2.0f * (q.w * q.x - q.y * q.z);
        if (std::abs(sinP) >= 0.99999f)
        {
            // Gimbal lock at pitch = +-90 deg: set roll = 0 and extract yaw from R00 & -R20
            euler.x = std::copysign(HALF_PI, sinP);
            euler.y = std::atan2(2.0f * (q.w * q.y - q.x * q.z),
                                 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
            euler.z = 0.0f;
        }
        else
        {
            euler.x = std::asin(Clamp(sinP, -1.0f, 1.0f));
            // R02 = sin(yaw)cos(pitch), R22 = cos(yaw)cos(pitch)
            euler.y = std::atan2(2.0f * (q.x * q.z + q.w * q.y),
                                 1.0f - 2.0f * (q.x * q.x + q.y * q.y));
            // R10 = cos(pitch)sin(roll), R11 = cos(pitch)cos(roll)
            euler.z = std::atan2(2.0f * (q.x * q.y + q.w * q.z),
                                 1.0f - 2.0f * (q.x * q.x + q.z * q.z));
        }

        return euler;
    }

    // Direction queries using Z+ = Forward convention (LH)
    Vec3 Quat::GetForward() const { return *this * Vec3::Forward(); }
    Vec3 Quat::GetRight() const   { return *this * Vec3::Right(); }
    Vec3 Quat::GetUp() const      { return *this * Vec3::Up(); }

    // ==================== STATIC METHODS ====================
    Quat Quat::Identity() { return Quat(0.0f, 0.0f, 0.0f, 1.0f); }

    Quat Quat::FromEuler(const Vec3& euler)
    {
        return FromEuler(euler.x, euler.y, euler.z);
    }

    // Left-handed Euler to Quaternion (ZXY intrinsic order)
    // pitch = X rotation, yaw = Y rotation, roll = Z rotation
    Quat Quat::FromEuler(float pitch, float yaw, float roll)
    {
        float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
        float cy = std::cos(yaw * 0.5f),   sy = std::sin(yaw * 0.5f);
        float cr = std::cos(roll * 0.5f),   sr = std::sin(roll * 0.5f);

        // ZXY order: Q = Qy * Qx * Qz
        return Quat(
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * cp * cr + sy * sp * sr
        );
    }

    Quat Quat::Slerp(const Quat& a, const Quat& b, float t)
    {
        float dot = a.Dot(b);

        // Ensure shortest path
        Quat b2 = b;
        if (dot < 0.0f)
        {
            b2 = -b;
            dot = -dot;
        }

        // If very close, use linear interpolation to avoid division by zero
        if (dot > 0.9995f)
            return Nlerp(a, b2, t);

        float theta = std::acos(Clamp(dot, -1.0f, 1.0f));
        float sinTheta = std::sin(theta);
        float wa = std::sin((1.0f - t) * theta) / sinTheta;
        float wb = std::sin(t * theta) / sinTheta;

        return Quat(
            a.x * wa + b2.x * wb,
            a.y * wa + b2.y * wb,
            a.z * wa + b2.z * wb,
            a.w * wa + b2.w * wb
        );
    }

    Quat Quat::Nlerp(const Quat& a, const Quat& b, float t)
    {
        Quat b2 = b;
        if (a.Dot(b) < 0.0f) b2 = -b;

        return Quat(
            QuarkDyn::Lerp(a.x, b2.x, t),
            QuarkDyn::Lerp(a.y, b2.y, t),
            QuarkDyn::Lerp(a.z, b2.z, t),
            QuarkDyn::Lerp(a.w, b2.w, t)
        ).Normalized();
    }

    // Left-handed LookRotation: builds rotation from forward and up
    Quat Quat::FromLookRotation(const Vec3& forward, const Vec3& up)
    {
        if (forward.LengthSq() < SMALL_NUMBER * SMALL_NUMBER)
            return Quat::Identity();

        Vec3 f = forward.Normalized();
        Vec3 upRef = (up.LengthSq() > SMALL_NUMBER * SMALL_NUMBER) ? up : Vec3::Up();
        Vec3 rRaw = upRef.Cross(f);            // LH: up × forward = right
        if (rRaw.LengthSq() <= SMALL_NUMBER * SMALL_NUMBER)
        {
            Vec3 fallbackUp = (std::abs(f.y) < 0.999f) ? Vec3::Up() : Vec3::Right();
            rRaw = fallbackUp.Cross(f);
        }
        Vec3 r = rRaw.Normalized();
        Vec3 u = f.Cross(r);                   // LH: forward × right = up

        // Build rotation matrix and extract quaternion
        Mat3 rot;
        // Column 0 = right
        rot.m[0] = r.x;  rot.m[1] = r.y;  rot.m[2] = r.z;
        // Column 1 = up
        rot.m[3] = u.x;  rot.m[4] = u.y;  rot.m[5] = u.z;
        // Column 2 = forward
        rot.m[6] = f.x;  rot.m[7] = f.y;  rot.m[8] = f.z;

        return FromMatrix(rot);
    }

    Quat Quat::FromMatrix(const Mat3& m)
    {
        // Shepperd's method — numerically stable quaternion extraction
        float m00 = m.m[0], m11 = m.m[4], m22 = m.m[8];
        float m01 = m.m[3], m10 = m.m[1];
        float m02 = m.m[6], m20 = m.m[2];
        float m12 = m.m[7], m21 = m.m[5];

        float trace = m00 + m11 + m22;
        Quat q;

        if (trace > 0.0f)
        {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (m21 - m12) * s;
            q.y = (m02 - m20) * s;
            q.z = (m10 - m01) * s;
        }
        else if (m00 > m11 && m00 > m22)
        {
            float s = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
            q.w = (m21 - m12) / s;
            q.x = 0.25f * s;
            q.y = (m01 + m10) / s;
            q.z = (m02 + m20) / s;
        }
        else if (m11 > m22)
        {
            float s = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
            q.w = (m02 - m20) / s;
            q.x = (m01 + m10) / s;
            q.y = 0.25f * s;
            q.z = (m12 + m21) / s;
        }
        else
        {
            float s = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
            q.w = (m10 - m01) / s;
            q.x = (m02 + m20) / s;
            q.y = (m12 + m21) / s;
            q.z = 0.25f * s;
        }

        return q.Normalized();
    }

    Quat Quat::FromAxisAngle(const Vec3& axis, float angle)
    {
        return Quat(axis, angle);
    }

    Quat Quat::RotateTowards(const Quat& from, const Quat& to, float maxAngle)
    {
        float angle = AngleBetween(from, to);
        if (angle < EPSILON) return to;

        float t = Min(1.0f, maxAngle / angle);
        return Slerp(from, to, t);
    }

    float Quat::AngleBetween(const Quat& a, const Quat& b)
    {
        float dot = Abs(a.Dot(b));
        dot = Clamp(dot, 0.0f, 1.0f);
        return 2.0f * std::acos(dot);
    }

} // namespace QuarkDyn
