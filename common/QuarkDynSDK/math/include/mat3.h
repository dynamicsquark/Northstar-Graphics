// ============================================================================
// QuarkDyn SDK — Math Module
// mat3.h — 3x3 Matrix (Column-major, Left-handed)
// ============================================================================
#pragma once

#include "math_types.h"
#include "vec3.h"

namespace QuarkDyn
{
    struct Quat;

    // Column-major 3x3 matrix
    // Layout: m[col * 3 + row]
    //
    //   col0  col1  col2
    // [ m[0]  m[3]  m[6] ]  row0
    // [ m[1]  m[4]  m[7] ]  row1
    // [ m[2]  m[5]  m[8] ]  row2
    struct Mat3
    {
        float m[9];

        // ==================== CONSTRUCTORS ====================
        Mat3();                     // Identity
        Mat3(float diagonal);

        // ==================== OPERATORS ====================
        Mat3 operator*(const Mat3& other) const;
        Vec3 operator*(const Vec3& v) const;
        Mat3 operator*(float s) const;
        Mat3 operator+(const Mat3& other) const;
        Mat3 operator-(const Mat3& other) const;

        Mat3& operator*=(const Mat3& other);
        Mat3& operator*=(float s);

        bool operator==(const Mat3& other) const;
        bool operator!=(const Mat3& other) const;

        float& operator()(int row, int col);
        const float& operator()(int row, int col) const;

        // ==================== METHODS ====================
        Mat3 Transposed() const;
        void Transpose();
        float Determinant() const;
        Mat3 Inverted() const;
        void Inverse();

        Vec3 GetColumn(int col) const;
        void SetColumn(int col, const Vec3& v);
        Vec3 GetRow(int row) const;

        // ==================== STATIC METHODS ====================
        static Mat3 Identity();
        static Mat3 Zero();
        static Mat3 FromDiagonal(float xx, float yy, float zz);
        static Mat3 FromQuat(const Quat& q);
        static Mat3 RotationX(float angle);
        static Mat3 RotationY(float angle);
        static Mat3 RotationZ(float angle);
        static Mat3 Rotation(const Vec3& axis, float angle);
        static Mat3 Scaling(const Vec3& s);

        // Physics inertia tensors
        static Mat3 SphereInertia(float mass, float radius);
        static Mat3 BoxInertia(float mass, const Vec3& halfExtents);
        static Mat3 CapsuleInertia(float mass, float radius, float halfHeight);
    };

} // namespace QuarkDyn
