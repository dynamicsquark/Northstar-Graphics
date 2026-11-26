// ============================================================================
// QuarkDyn SDK — Math Module
// mat4.cpp — 4x4 Matrix Implementation (Column-major, Left-handed)
// Left-handed: X+ = Right, Y+ = Up, Z+ = Forward
// ============================================================================

#include "../include/mat4.h"
#include "../include/vec3.h"
#include "../include/vec4.h"
#include "../include/mat3.h"
#include "../include/quat.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Mat4::Mat4()
    {
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    Mat4::Mat4(float diagonal)
    {
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = diagonal;
    }

    // ==================== OPERATORS ====================
    // Column-major: result[col*4+row] = sum over k of this[k*4+row] * other[col*4+k]
    Mat4 Mat4::operator*(const Mat4& other) const
    {
        Mat4 result;
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                result.m[col * 4 + row] =
                    m[0 * 4 + row] * other.m[col * 4 + 0] +
                    m[1 * 4 + row] * other.m[col * 4 + 1] +
                    m[2 * 4 + row] * other.m[col * 4 + 2] +
                    m[3 * 4 + row] * other.m[col * 4 + 3];
            }
        }
        return result;
    }

    Vec4 Mat4::operator*(const Vec4& v) const
    {
        return Vec4(
            m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12] * v.w,
            m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13] * v.w,
            m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
            m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w
        );
    }

    Mat4 Mat4::operator*(float s) const
    {
        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = m[i] * s;
        return result;
    }

    Mat4 Mat4::operator+(const Mat4& other) const
    {
        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = m[i] + other.m[i];
        return result;
    }

    Mat4 Mat4::operator-(const Mat4& other) const
    {
        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = m[i] - other.m[i];
        return result;
    }

    Mat4& Mat4::operator*=(const Mat4& other) { *this = *this * other; return *this; }
    Mat4& Mat4::operator*=(float s) { for (int i = 0; i < 16; i++) m[i] *= s; return *this; }

    bool Mat4::operator==(const Mat4& other) const
    {
        for (int i = 0; i < 16; i++)
            if (!ApproxEqual(m[i], other.m[i])) return false;
        return true;
    }

    bool Mat4::operator!=(const Mat4& other) const { return !(*this == other); }

    float& Mat4::operator()(int row, int col) { assert(row >= 0 && row < 4 && col >= 0 && col < 4); return m[col * 4 + row]; }
    const float& Mat4::operator()(int row, int col) const { assert(row >= 0 && row < 4 && col >= 0 && col < 4); return m[col * 4 + row]; }

    // ==================== TRANSFORM METHODS ====================
    Vec3 Mat4::TransformPoint(const Vec3& v) const
    {
        float w = m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15];
        float invW = (std::abs(w) > EPSILON) ? 1.0f / w : 1.0f;
        return Vec3(
            (m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12]) * invW,
            (m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13]) * invW,
            (m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14]) * invW
        );
    }

    Vec3 Mat4::TransformDirection(const Vec3& v) const
    {
        return Vec3(
            m[0] * v.x + m[4] * v.y + m[8]  * v.z,
            m[1] * v.x + m[5] * v.y + m[9]  * v.z,
            m[2] * v.x + m[6] * v.y + m[10] * v.z
        );
    }

    // ==================== METHODS ====================
    Mat4 Mat4::Transposed() const
    {
        Mat4 result;
        for (int col = 0; col < 4; col++)
            for (int row = 0; row < 4; row++)
                result.m[col * 4 + row] = m[row * 4 + col];
        return result;
    }

    void Mat4::Transpose() { *this = Transposed(); }

    float Mat4::Determinant() const
    {
        float a00 = m[0],  a01 = m[4],  a02 = m[8],  a03 = m[12];
        float a10 = m[1],  a11 = m[5],  a12 = m[9],  a13 = m[13];
        float a20 = m[2],  a21 = m[6],  a22 = m[10], a23 = m[14];
        float a30 = m[3],  a31 = m[7],  a32 = m[11], a33 = m[15];

        float f0 = a22 * a33 - a32 * a23;
        float f1 = a21 * a33 - a31 * a23;
        float f2 = a21 * a32 - a31 * a22;
        float f3 = a20 * a33 - a30 * a23;
        float f4 = a20 * a32 - a30 * a22;
        float f5 = a20 * a31 - a30 * a21;

        float c0 = a11 * f0 - a12 * f1 + a13 * f2;
        float c1 = -(a10 * f0 - a12 * f3 + a13 * f4);
        float c2 = a10 * f1 - a11 * f3 + a13 * f5;
        float c3 = -(a10 * f2 - a11 * f4 + a12 * f5);

        return a00 * c0 + a01 * c1 + a02 * c2 + a03 * c3;
    }

    Mat4 Mat4::Inverted() const
    {
        const double a00 = m[0],  a01 = m[4],  a02 = m[8],  a03 = m[12];
        const double a10 = m[1],  a11 = m[5],  a12 = m[9],  a13 = m[13];
        const double a20 = m[2],  a21 = m[6],  a22 = m[10], a23 = m[14];
        const double a30 = m[3],  a31 = m[7],  a32 = m[11], a33 = m[15];

        const double s0 = a00 * a11 - a10 * a01;
        const double s1 = a00 * a12 - a10 * a02;
        const double s2 = a00 * a13 - a10 * a03;
        const double s3 = a01 * a12 - a11 * a02;
        const double s4 = a01 * a13 - a11 * a03;
        const double s5 = a02 * a13 - a12 * a03;

        const double c5 = a22 * a33 - a32 * a23;
        const double c4 = a21 * a33 - a31 * a23;
        const double c3 = a21 * a32 - a31 * a22;
        const double c2 = a20 * a33 - a30 * a23;
        const double c1 = a20 * a32 - a30 * a22;
        const double c0 = a20 * a31 - a30 * a21;

        const double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if (!(std::abs(det) > 1e-30)) return Mat4::Identity();

        const double invDet = 1.0 / det;

        Mat4 result;
        result.m[0]  = static_cast<float>(( a11 * c5 - a12 * c4 + a13 * c3) * invDet);
        result.m[1]  = static_cast<float>((-a10 * c5 + a12 * c2 - a13 * c1) * invDet);
        result.m[2]  = static_cast<float>(( a10 * c4 - a11 * c2 + a13 * c0) * invDet);
        result.m[3]  = static_cast<float>((-a10 * c3 + a11 * c1 - a12 * c0) * invDet);

        result.m[4]  = static_cast<float>((-a01 * c5 + a02 * c4 - a03 * c3) * invDet);
        result.m[5]  = static_cast<float>(( a00 * c5 - a02 * c2 + a03 * c1) * invDet);
        result.m[6]  = static_cast<float>((-a00 * c4 + a01 * c2 - a03 * c0) * invDet);
        result.m[7]  = static_cast<float>(( a00 * c3 - a01 * c1 + a02 * c0) * invDet);

        result.m[8]  = static_cast<float>(( a31 * s5 - a32 * s4 + a33 * s3) * invDet);
        result.m[9]  = static_cast<float>((-a30 * s5 + a32 * s2 - a33 * s1) * invDet);
        result.m[10] = static_cast<float>(( a30 * s4 - a31 * s2 + a33 * s0) * invDet);
        result.m[11] = static_cast<float>((-a30 * s3 + a31 * s1 - a32 * s0) * invDet);

        result.m[12] = static_cast<float>((-a21 * s5 + a22 * s4 - a23 * s3) * invDet);
        result.m[13] = static_cast<float>(( a20 * s5 - a22 * s2 + a23 * s1) * invDet);
        result.m[14] = static_cast<float>((-a20 * s4 + a21 * s2 - a23 * s0) * invDet);
        result.m[15] = static_cast<float>(( a20 * s3 - a21 * s1 + a22 * s0) * invDet);

        for (int i = 0; i < 16; ++i)
        {
            if (!std::isfinite(result.m[i])) return Mat4::Identity();
        }
        return result;
    }

    Mat4 Mat4::InverseAffine() const
    {
        Mat3 upper;
        upper.m[0] = m[0]; upper.m[1] = m[1]; upper.m[2] = m[2];
        upper.m[3] = m[4]; upper.m[4] = m[5]; upper.m[5] = m[6];
        upper.m[6] = m[8]; upper.m[7] = m[9]; upper.m[8] = m[10];

        Mat3 invUpper = upper.Inverted();
        Vec3 t(m[12], m[13], m[14]);
        Vec3 invT = invUpper * t * -1.0f;

        Mat4 result;
        result.m[0]  = invUpper.m[0]; result.m[1]  = invUpper.m[1]; result.m[2]  = invUpper.m[2]; result.m[3]  = 0.0f;
        result.m[4]  = invUpper.m[3]; result.m[5]  = invUpper.m[4]; result.m[6]  = invUpper.m[5]; result.m[7]  = 0.0f;
        result.m[8]  = invUpper.m[6]; result.m[9]  = invUpper.m[7]; result.m[10] = invUpper.m[8]; result.m[11] = 0.0f;
        result.m[12] = invT.x;        result.m[13] = invT.y;        result.m[14] = invT.z;        result.m[15] = 1.0f;

        return result;
    }

    void Mat4::Inverse() { *this = Inverted(); }

    // ==================== ACCESSORS ====================
    Vec4 Mat4::GetColumn(int col) const
    {
        return Vec4(m[col * 4], m[col * 4 + 1], m[col * 4 + 2], m[col * 4 + 3]);
    }

    void Mat4::SetColumn(int col, const Vec4& v)
    {
        m[col * 4 + 0] = v.x;
        m[col * 4 + 1] = v.y;
        m[col * 4 + 2] = v.z;
        m[col * 4 + 3] = v.w;
    }

    Vec4 Mat4::GetRow(int row) const
    {
        return Vec4(m[0 * 4 + row], m[1 * 4 + row], m[2 * 4 + row], m[3 * 4 + row]);
    }

    Vec3 Mat4::GetTranslation() const { return Vec3(m[12], m[13], m[14]); }

    void Mat4::SetTranslation(const Vec3& v)
    {
        m[12] = v.x;
        m[13] = v.y;
        m[14] = v.z;
    }

    Vec3 Mat4::GetScale() const
    {
        float sx = Vec3(m[0], m[1], m[2]).Length();
        float sy = Vec3(m[4], m[5], m[6]).Length();
        float sz = Vec3(m[8], m[9], m[10]).Length();
        return Vec3(sx, sy, sz);
    }

    Quat Mat4::GetRotation() const
    {
        Vec3 scale = GetScale();
        if (scale.x < SMALL_NUMBER || scale.y < SMALL_NUMBER || scale.z < SMALL_NUMBER)
            return Quat::Identity();

        Mat3 rot;
        rot.m[0] = m[0] / scale.x; rot.m[1] = m[1] / scale.x; rot.m[2] = m[2] / scale.x;
        rot.m[3] = m[4] / scale.y; rot.m[4] = m[5] / scale.y; rot.m[5] = m[6] / scale.y;
        rot.m[6] = m[8] / scale.z; rot.m[7] = m[9] / scale.z; rot.m[8] = m[10] / scale.z;

        if (rot.Determinant() < 0.0f)
        {
            rot.m[0] = -rot.m[0];
            rot.m[1] = -rot.m[1];
            rot.m[2] = -rot.m[2];
        }

        return Quat::FromMatrix(rot);
    }

    Mat3 Mat4::ToMat3() const
    {
        Mat3 result;
        result.m[0] = m[0]; result.m[1] = m[1]; result.m[2] = m[2];
        result.m[3] = m[4]; result.m[4] = m[5]; result.m[5] = m[6];
        result.m[6] = m[8]; result.m[7] = m[9]; result.m[8] = m[10];
        return result;
    }

    // ==================== STATIC BUILDERS ====================
    Mat4 Mat4::Identity() { return Mat4(); }

    Mat4 Mat4::Translation(const Vec3& v)
    {
        Mat4 result;
        result.m[12] = v.x;
        result.m[13] = v.y;
        result.m[14] = v.z;
        return result;
    }

    Mat4 Mat4::Scaling(const Vec3& v)
    {
        Mat4 result;
        result.m[0]  = v.x;
        result.m[5]  = v.y;
        result.m[10] = v.z;
        return result;
    }

    // Left-handed (X+ Right, Y+ Up, Z+ Forward), column-major rotation builders.
    // Identical to Mat4::FromQuat(Quat(axis, angle)): left-hand rule around +axis.
    Mat4 Mat4::RotationX(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat4 result;
        result.m[5]  =  c;  result.m[9]  = -s;
        result.m[6]  =  s;  result.m[10] =  c;
        return result;
    }

    Mat4 Mat4::RotationY(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat4 result;
        result.m[0]  =  c;  result.m[8]  =  s;
        result.m[2]  = -s;  result.m[10] =  c;
        return result;
    }

    Mat4 Mat4::RotationZ(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat4 result;
        result.m[0] =  c;  result.m[4] = -s;
        result.m[1] =  s;  result.m[5] =  c;
        return result;
    }

    Mat4 Mat4::Rotation(const Vec3& axis, float angle)
    {
        float lenSq = axis.LengthSq();
        if (lenSq < SMALL_NUMBER * SMALL_NUMBER) return Mat4::Identity();

        Vec3 a = axis * (1.0f / std::sqrt(lenSq));
        float c = std::cos(angle), s = std::sin(angle);
        float t = 1.0f - c;

        Mat4 result;
        // Column 0
        result.m[0] = t * a.x * a.x + c;
        result.m[1] = t * a.x * a.y + s * a.z;
        result.m[2] = t * a.x * a.z - s * a.y;
        // Column 1
        result.m[4] = t * a.x * a.y - s * a.z;
        result.m[5] = t * a.y * a.y + c;
        result.m[6] = t * a.y * a.z + s * a.x;
        // Column 2
        result.m[8]  = t * a.x * a.z + s * a.y;
        result.m[9]  = t * a.y * a.z - s * a.x;
        result.m[10] = t * a.z * a.z + c;

        return result;
    }

    Mat4 Mat4::FromQuat(const Quat& q)
    {
        float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        Mat4 result;
        // Column 0
        result.m[0] = 1.0f - 2.0f * (yy + zz);
        result.m[1] = 2.0f * (xy + wz);
        result.m[2] = 2.0f * (xz - wy);
        // Column 1
        result.m[4] = 2.0f * (xy - wz);
        result.m[5] = 1.0f - 2.0f * (xx + zz);
        result.m[6] = 2.0f * (yz + wx);
        // Column 2
        result.m[8]  = 2.0f * (xz + wy);
        result.m[9]  = 2.0f * (yz - wx);
        result.m[10] = 1.0f - 2.0f * (xx + yy);

        return result;
    }

    Mat4 Mat4::TRS(const Vec3& translation, const Quat& rotation, const Vec3& scale)
    {
        Mat4 result = FromQuat(rotation);

        // Apply scale to rotation columns
        result.m[0]  *= scale.x; result.m[1]  *= scale.x; result.m[2]  *= scale.x;
        result.m[4]  *= scale.y; result.m[5]  *= scale.y; result.m[6]  *= scale.y;
        result.m[8]  *= scale.z; result.m[9]  *= scale.z; result.m[10] *= scale.z;

        // Set translation
        result.m[12] = translation.x;
        result.m[13] = translation.y;
        result.m[14] = translation.z;

        return result;
    }

    // ==================== VIEW & PROJECTION (Left-handed) ====================

    // Left-handed LookAt: camera looks along +Z in view space
    Mat4 Mat4::LookAt(const Vec3& eye, const Vec3& target, const Vec3& up)
    {
        Vec3 delta = target - eye;
        float fwdLenSq = delta.LengthSq();
        Vec3 forward = (fwdLenSq > SMALL_NUMBER * SMALL_NUMBER)
            ? (delta * (1.0f / std::sqrt(fwdLenSq)))
            : Vec3::Forward();

        Vec3 upRef = (up.LengthSq() > SMALL_NUMBER * SMALL_NUMBER) ? up : Vec3::Up();
        Vec3 rightRaw = upRef.Cross(forward);
        if (rightRaw.LengthSq() <= SMALL_NUMBER * SMALL_NUMBER)
        {
            Vec3 fallbackUp = (std::abs(forward.y) < 0.999f) ? Vec3::Up() : Vec3::Right();
            rightRaw = fallbackUp.Cross(forward);
        }
        Vec3 right = rightRaw.Normalized();             // LH: up × forward = right
        Vec3 camUp = forward.Cross(right);              // LH: forward × right = up

        Mat4 result;
        // Row 0 — right axis
        result.m[0]  = right.x;
        result.m[4]  = right.y;
        result.m[8]  = right.z;
        result.m[12] = -right.Dot(eye);
        // Row 1 — up axis
        result.m[1]  = camUp.x;
        result.m[5]  = camUp.y;
        result.m[9]  = camUp.z;
        result.m[13] = -camUp.Dot(eye);
        // Row 2 — forward axis (LH: NOT negated, camera looks along +Z)
        result.m[2]  = forward.x;
        result.m[6]  = forward.y;
        result.m[10] = forward.z;
        result.m[14] = -forward.Dot(eye);
        // Row 3
        result.m[3]  = 0.0f;
        result.m[7]  = 0.0f;
        result.m[11] = 0.0f;
        result.m[15] = 1.0f;

        return result;
    }

    // Left-handed perspective projection (Z mapped to [0, 1])
    Mat4 Mat4::Perspective(float fovY, float aspect, float nearPlane, float farPlane)
    {
        float safeFov = Clamp(fovY, 1e-4f, PI - 1e-4f);
        float safeAspect = (std::abs(aspect) > SMALL_NUMBER) ? aspect : 1.0f;
        float safeNear = Max(1e-5f, nearPlane);
        float depthSpan = farPlane - safeNear;
        if (std::abs(depthSpan) < SMALL_NUMBER) depthSpan = (depthSpan >= 0.0f) ? SMALL_NUMBER : -SMALL_NUMBER;

        float tanHalfFov = std::tan(safeFov * 0.5f);

        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = 0.0f;

        result.m[0]  = 1.0f / (safeAspect * tanHalfFov);
        result.m[5]  = 1.0f / tanHalfFov;
        result.m[10] = farPlane / depthSpan;              // LH
        result.m[11] = 1.0f;                              // LH: +1
        result.m[14] = -(safeNear * farPlane) / depthSpan; // LH

        return result;
    }

    // Left-handed Reverse-Z perspective (infinite far plane, better depth precision)
    Mat4 Mat4::PerspectiveReverseZ(float fovY, float aspect, float nearPlane)
    {
        float safeFov = Clamp(fovY, 1e-4f, PI - 1e-4f);
        float safeAspect = (std::abs(aspect) > SMALL_NUMBER) ? aspect : 1.0f;
        float safeNear = Max(1e-5f, nearPlane);
        float tanHalfFov = std::tan(safeFov * 0.5f);

        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = 0.0f;

        result.m[0]  = 1.0f / (safeAspect * tanHalfFov);
        result.m[5]  = 1.0f / tanHalfFov;
        result.m[10] = 0.0f;
        result.m[11] = 1.0f;          // LH: +1
        result.m[14] = safeNear;

        return result;
    }

    // Left-handed orthographic projection
    Mat4 Mat4::Orthographic(float left, float right, float bottom, float top, float nearPlane, float farPlane)
    {
        float w = right - left;
        float h = top - bottom;
        float d = farPlane - nearPlane;
        if (std::abs(w) < SMALL_NUMBER) w = (w >= 0.0f) ? SMALL_NUMBER : -SMALL_NUMBER;
        if (std::abs(h) < SMALL_NUMBER) h = (h >= 0.0f) ? SMALL_NUMBER : -SMALL_NUMBER;
        if (std::abs(d) < SMALL_NUMBER) d = (d >= 0.0f) ? SMALL_NUMBER : -SMALL_NUMBER;

        Mat4 result;
        for (int i = 0; i < 16; i++) result.m[i] = 0.0f;

        result.m[0]  = 2.0f / w;
        result.m[5]  = 2.0f / h;
        result.m[10] = 1.0f / d;                        // LH
        result.m[12] = -(right + left) / w;
        result.m[13] = -(top + bottom) / h;
        result.m[14] = -nearPlane / d;                  // LH
        result.m[15] = 1.0f;

        return result;
    }

} // namespace QuarkDyn
