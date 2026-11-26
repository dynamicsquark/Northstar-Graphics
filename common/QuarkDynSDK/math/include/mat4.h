// ============================================================================
// QuarkDyn SDK — Math Module
// mat4.h — 4x4 Matrix (Column-major, Left-handed)
// Left-handed: X+ = Right, Y+ = Up, Z+ = Forward
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"
#include "vec4.h"
#include "mat3.h"

namespace QuarkDyn
{
    struct Quat;

    // Column-major 4x4 matrix
    // Layout: m[col * 4 + row]
    //
    //   col0  col1  col2  col3
    // [ m[0]  m[4]  m[8]  m[12] ]  row0
    // [ m[1]  m[5]  m[9]  m[13] ]  row1
    // [ m[2]  m[6]  m[10] m[14] ]  row2
    // [ m[3]  m[7]  m[11] m[15] ]  row3
    //
    // m[12], m[13], m[14] = translation
    struct Mat4
    {
        float m[16];

        // ==================== CONSTRUCTORS ====================
        Mat4();                     // Identity
        Mat4(float diagonal);

        // ==================== OPERATORS ====================
        Mat4 operator*(const Mat4& other) const;
        Vec4 operator*(const Vec4& v) const;
        Mat4 operator*(float s) const;
        Mat4 operator+(const Mat4& other) const;
        Mat4 operator-(const Mat4& other) const;

        Mat4& operator*=(const Mat4& other);
        Mat4& operator*=(float s);

        bool operator==(const Mat4& other) const;
        bool operator!=(const Mat4& other) const;

        float& operator()(int row, int col);
        const float& operator()(int row, int col) const;

        // ==================== TRANSFORM METHODS ====================
        Vec3 TransformPoint(const Vec3& v) const;
        Vec3 TransformDirection(const Vec3& v) const;

        // ==================== METHODS ====================
        Mat4 Transposed() const;
        void Transpose();
        float Determinant() const;
        Mat4 Inverted() const;
        Mat4 InverseAffine() const;
        void Inverse();

        // ==================== ACCESSORS ====================
        Vec4 GetColumn(int col) const;
        void SetColumn(int col, const Vec4& v);
        Vec4 GetRow(int row) const;
        Vec3 GetTranslation() const;
        void SetTranslation(const Vec3& v);
        Vec3 GetScale() const;
        Quat GetRotation() const;
        Mat3 ToMat3() const;

        // ==================== STATIC BUILDERS ====================
        static Mat4 Identity();
        static Mat4 Translation(const Vec3& v);
        static Mat4 Scaling(const Vec3& v);
        static Mat4 RotationX(float angle);
        static Mat4 RotationY(float angle);
        static Mat4 RotationZ(float angle);
        static Mat4 Rotation(const Vec3& axis, float angle);
        static Mat4 FromQuat(const Quat& q);
        static Mat4 TRS(const Vec3& translation, const Quat& rotation, const Vec3& scale);

        // ==================== VIEW & PROJECTION (Left-handed) ====================
        static Mat4 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up);
        static Mat4 Perspective(float fovY, float aspect, float nearPlane, float farPlane);
        static Mat4 PerspectiveReverseZ(float fovY, float aspect, float nearPlane);
        static Mat4 Orthographic(float left, float right, float bottom, float top, float nearPlane, float farPlane);
    };

} // namespace QuarkDyn
