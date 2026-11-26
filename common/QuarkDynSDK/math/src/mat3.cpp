// ============================================================================
// QuarkDyn SDK — Math Module
// mat3.cpp — 3x3 Matrix Implementation (Column-major, Left-handed)
// ============================================================================

#include "../include/mat3.h"
#include "../include/vec3.h"
#include "../include/quat.h"

namespace QuarkDyn
{
    // ==================== CONSTRUCTORS ====================
    Mat3::Mat3()
    {
        for (int i = 0; i < 9; i++) m[i] = 0.0f;
        m[0] = m[4] = m[8] = 1.0f;
    }

    Mat3::Mat3(float diagonal)
    {
        for (int i = 0; i < 9; i++) m[i] = 0.0f;
        m[0] = m[4] = m[8] = diagonal;
    }

    // ==================== OPERATORS ====================
    // Column-major: result[col*3+row] = sum over k of this[k*3+row] * other[col*3+k]
    Mat3 Mat3::operator*(const Mat3& other) const
    {
        Mat3 result;
        for (int col = 0; col < 3; col++)
        {
            for (int row = 0; row < 3; row++)
            {
                result.m[col * 3 + row] =
                    m[0 * 3 + row] * other.m[col * 3 + 0] +
                    m[1 * 3 + row] * other.m[col * 3 + 1] +
                    m[2 * 3 + row] * other.m[col * 3 + 2];
            }
        }
        return result;
    }

    Vec3 Mat3::operator*(const Vec3& v) const
    {
        return Vec3(
            m[0] * v.x + m[3] * v.y + m[6] * v.z,
            m[1] * v.x + m[4] * v.y + m[7] * v.z,
            m[2] * v.x + m[5] * v.y + m[8] * v.z
        );
    }

    Mat3 Mat3::operator*(float s) const
    {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = m[i] * s;
        return result;
    }

    Mat3 Mat3::operator+(const Mat3& other) const
    {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = m[i] + other.m[i];
        return result;
    }

    Mat3 Mat3::operator-(const Mat3& other) const
    {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = m[i] - other.m[i];
        return result;
    }

    Mat3& Mat3::operator*=(const Mat3& other)
    {
        *this = *this * other;
        return *this;
    }

    Mat3& Mat3::operator*=(float s)
    {
        for (int i = 0; i < 9; i++) m[i] *= s;
        return *this;
    }

    bool Mat3::operator==(const Mat3& other) const
    {
        for (int i = 0; i < 9; i++)
            if (!ApproxEqual(m[i], other.m[i])) return false;
        return true;
    }

    bool Mat3::operator!=(const Mat3& other) const { return !(*this == other); }

    float& Mat3::operator()(int row, int col) { assert(row >= 0 && row < 3 && col >= 0 && col < 3); return m[col * 3 + row]; }
    const float& Mat3::operator()(int row, int col) const { assert(row >= 0 && row < 3 && col >= 0 && col < 3); return m[col * 3 + row]; }

    // ==================== METHODS ====================
    Mat3 Mat3::Transposed() const
    {
        Mat3 result;
        for (int col = 0; col < 3; col++)
            for (int row = 0; row < 3; row++)
                result.m[col * 3 + row] = m[row * 3 + col];
        return result;
    }

    void Mat3::Transpose() { *this = Transposed(); }

    float Mat3::Determinant() const
    {
        return m[0] * (m[4] * m[8] - m[7] * m[5])
             - m[3] * (m[1] * m[8] - m[7] * m[2])
             + m[6] * (m[1] * m[5] - m[4] * m[2]);
    }

    Mat3 Mat3::Inverted() const
    {
        const double m0 = m[0], m1 = m[1], m2 = m[2];
        const double m3 = m[3], m4 = m[4], m5 = m[5];
        const double m6 = m[6], m7 = m[7], m8 = m[8];

        const double c0 = m4 * m8 - m7 * m5;
        const double c1 = m7 * m2 - m1 * m8;
        const double c2 = m1 * m5 - m4 * m2;

        const double det = m0 * c0 + m3 * c1 + m6 * c2;
        if (!(std::abs(det) > 1e-30)) return Mat3::Identity();

        const double invDet = 1.0 / det;
        Mat3 result;

        // Cofactor matrix transposed, scaled by 1/det (column-major)
        result.m[0] = static_cast<float>(c0 * invDet);
        result.m[1] = static_cast<float>(c1 * invDet);
        result.m[2] = static_cast<float>(c2 * invDet);

        result.m[3] = static_cast<float>((m6 * m5 - m3 * m8) * invDet);
        result.m[4] = static_cast<float>((m0 * m8 - m6 * m2) * invDet);
        result.m[5] = static_cast<float>((m3 * m2 - m0 * m5) * invDet);

        result.m[6] = static_cast<float>((m3 * m7 - m6 * m4) * invDet);
        result.m[7] = static_cast<float>((m6 * m1 - m0 * m7) * invDet);
        result.m[8] = static_cast<float>((m0 * m4 - m3 * m1) * invDet);

        for (int i = 0; i < 9; ++i)
        {
            if (!std::isfinite(result.m[i])) return Mat3::Identity();
        }
        return result;
    }

    void Mat3::Inverse() { *this = Inverted(); }

    Vec3 Mat3::GetColumn(int col) const
    {
        return Vec3(m[col * 3 + 0], m[col * 3 + 1], m[col * 3 + 2]);
    }

    void Mat3::SetColumn(int col, const Vec3& v)
    {
        m[col * 3 + 0] = v.x;
        m[col * 3 + 1] = v.y;
        m[col * 3 + 2] = v.z;
    }

    Vec3 Mat3::GetRow(int row) const
    {
        return Vec3(m[0 * 3 + row], m[1 * 3 + row], m[2 * 3 + row]);
    }

    // ==================== STATIC METHODS ====================
    Mat3 Mat3::Identity() { return Mat3(); }

    Mat3 Mat3::Zero()
    {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = 0.0f;
        return result;
    }

    Mat3 Mat3::FromDiagonal(float xx, float yy, float zz)
    {
        Mat3 result;
        for (int i = 0; i < 9; i++) result.m[i] = 0.0f;
        result.m[0] = xx;
        result.m[4] = yy;
        result.m[8] = zz;
        return result;
    }

    Mat3 Mat3::FromQuat(const Quat& q)
    {
        float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        Mat3 result;
        // Column 0
        result.m[0] = 1.0f - 2.0f * (yy + zz);
        result.m[1] = 2.0f * (xy + wz);
        result.m[2] = 2.0f * (xz - wy);
        // Column 1
        result.m[3] = 2.0f * (xy - wz);
        result.m[4] = 1.0f - 2.0f * (xx + zz);
        result.m[5] = 2.0f * (yz + wx);
        // Column 2
        result.m[6] = 2.0f * (xz + wy);
        result.m[7] = 2.0f * (yz - wx);
        result.m[8] = 1.0f - 2.0f * (xx + yy);

        return result;
    }

    // Left-handed (X+ Right, Y+ Up, Z+ Forward), column-major rotation matrices.
    // Identical to Mat3::FromQuat(Quat(axis, angle)): left-hand rule around +axis.

    Mat3 Mat3::RotationX(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat3 result;
        result.m[0] = 1.0f; result.m[3] = 0.0f; result.m[6] = 0.0f;
        result.m[1] = 0.0f; result.m[4] = c;    result.m[7] = -s;
        result.m[2] = 0.0f; result.m[5] = s;    result.m[8] =  c;
        return result;
    }

    Mat3 Mat3::RotationY(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat3 result;
        result.m[0] =  c;   result.m[3] = 0.0f; result.m[6] = s;
        result.m[1] = 0.0f; result.m[4] = 1.0f; result.m[7] = 0.0f;
        result.m[2] = -s;   result.m[5] = 0.0f; result.m[8] = c;
        return result;
    }

    Mat3 Mat3::RotationZ(float angle)
    {
        float c = std::cos(angle), s = std::sin(angle);
        Mat3 result;
        result.m[0] =  c;   result.m[3] = -s;   result.m[6] = 0.0f;
        result.m[1] =  s;   result.m[4] =  c;   result.m[7] = 0.0f;
        result.m[2] = 0.0f; result.m[5] = 0.0f; result.m[8] = 1.0f;
        return result;
    }

    Mat3 Mat3::Rotation(const Vec3& axis, float angle)
    {
        float lenSq = axis.LengthSq();
        if (lenSq < SMALL_NUMBER * SMALL_NUMBER) return Mat3::Identity();

        Vec3 a = axis * (1.0f / std::sqrt(lenSq));
        float c = std::cos(angle), s = std::sin(angle);
        float t = 1.0f - c;

        Mat3 result;
        // Column 0
        result.m[0] = t * a.x * a.x + c;
        result.m[1] = t * a.x * a.y + s * a.z;
        result.m[2] = t * a.x * a.z - s * a.y;
        // Column 1
        result.m[3] = t * a.x * a.y - s * a.z;
        result.m[4] = t * a.y * a.y + c;
        result.m[5] = t * a.y * a.z + s * a.x;
        // Column 2
        result.m[6] = t * a.x * a.z + s * a.y;
        result.m[7] = t * a.y * a.z - s * a.x;
        result.m[8] = t * a.z * a.z + c;

        return result;
    }

    Mat3 Mat3::Scaling(const Vec3& s)
    {
        return FromDiagonal(s.x, s.y, s.z);
    }

    // ==================== PHYSICS INERTIA TENSORS ====================
    Mat3 Mat3::SphereInertia(float mass, float radius)
    {
        float i = (2.0f / 5.0f) * mass * radius * radius;
        return FromDiagonal(i, i, i);
    }

    Mat3 Mat3::BoxInertia(float mass, const Vec3& halfExtents)
    {
        float x2 = halfExtents.x * halfExtents.x;
        float y2 = halfExtents.y * halfExtents.y;
        float z2 = halfExtents.z * halfExtents.z;
        float factor = mass / 3.0f;
        return FromDiagonal(
            factor * (y2 + z2),
            factor * (x2 + z2),
            factor * (x2 + y2)
        );
    }

    Mat3 Mat3::CapsuleInertia(float mass, float radius, float halfHeight)
    {
        float h = halfHeight * 2.0f;
        float r2 = radius * radius;
        float h2 = h * h;

        float cylinderVolume = PI * r2 * h;
        float hemiVolume = (4.0f / 3.0f) * PI * r2 * radius;
        float totalVolume = cylinderVolume + hemiVolume;

        float cylinderMass = mass * (cylinderVolume / totalVolume);
        float hemiMass = mass * (hemiVolume / totalVolume);

        // Cylinder inertia
        float cylIxx = cylinderMass * (3.0f * r2 + h2) / 12.0f;
        float cylIyy = cylIxx;
        float cylIzz = cylinderMass * r2 / 2.0f;

        // Hemisphere inertia (shifted via parallel axis theorem)
        float hemiIxx = hemiMass * (83.0f * r2 / 320.0f);
        float offset = halfHeight + (3.0f * radius / 8.0f);
        hemiIxx += hemiMass * offset * offset;

        float hemiIyy = hemiIxx;
        float hemiIzz = hemiMass * (2.0f * r2 / 5.0f);

        return FromDiagonal(
            cylIxx + 2.0f * hemiIxx,
            cylIyy + 2.0f * hemiIyy,
            cylIzz + 2.0f * hemiIzz
        );
    }

} // namespace QuarkDyn
