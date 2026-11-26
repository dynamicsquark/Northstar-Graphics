#pragma once

#include "../../core/nsl_error.h"
#include "NSLCommon/nsl_common_types.h"
#include <string>
#include <vector>
#include <memory>

// =====================================================================
//  NSL AST — Expressions
// =====================================================================

namespace NSL
{
    struct Expression;
    using ExprPtr = std::unique_ptr<Expression>;

    // ========================= BASE EXPRESSION =======================

    struct Expression
    {
        virtual ~Expression() = default;
        SourceLocation location;
    };

    // ========================= LITERAL ===============================

    struct LiteralExpr : Expression
    {
        enum class Kind { Int, UInt, Float, Bool, String } litKind = Kind::Int;
        int64_t     intValue    = 0;
        uint64_t    uintValue   = 0;
        double      floatValue  = 0.0;
        bool        boolValue   = false;
        std::string strValue;
    };

    // ========================= IDENTIFIER ============================

    struct IdentifierExpr : Expression
    {
        std::string name;
    };

    // ========================= MEMBER ACCESS =========================
    // a.b  — works for PROPERTY group access (MaterialParams.baseColor)
    // and regular struct field access

    struct MemberAccessExpr : Expression
    {
        ExprPtr     object;         // Left-hand side  (may be another MemberAccessExpr)
        std::string member;         // Field name
        bool        isArrow = false; // true for ->
    };

    // ========================= INDEX =================================
    // a[i]

    struct IndexExpr : Expression
    {
        ExprPtr object;
        ExprPtr index;
    };

    // ========================= CALL ==================================
    // f(args...)  or  Sample(tex, samp, uv)

    struct CallExpr : Expression
    {
        ExprPtr                  callee;    // Usually IdentifierExpr
        std::vector<ExprPtr>     args;
    };

    // ========================= INITIALIZER LIST ======================
    // { a, b, c }

    struct InitializerListExpr : Expression
    {
        std::vector<ExprPtr> elements;
    };

    // ========================= PARENTHESIZED =========================
    // (expr)  — preserves grouping and operator precedence

    struct ParenExpr : Expression
    {
        ExprPtr inner;
    };

    // ========================= CAST ==================================
    // Vec3(x, y, z)  —  constructor-style cast / vector construction

    struct CastExpr : Expression
    {
        std::string          typeName;
        std::vector<ExprPtr> args;
    };

    // ========================= UNARY =================================

    struct UnaryExpr : Expression
    {
        enum class Op {
            Neg, Not, BitNot, PreInc, PreDec, PostInc, PostDec
        } op;
        ExprPtr  operand;
        bool     isPostfix = false;
    };

    // ========================= BINARY ================================

    struct BinaryExpr : Expression
    {
        enum class Op {
            Add, Sub, Mul, Div, Mod,
            Eq, Neq, Lt, Gt, Le, Ge,
            And, Or,
            BitAnd, BitOr, BitXor, Shl, Shr,
            Assign,
            AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
            AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign,
        } op;
        ExprPtr left;
        ExprPtr right;
    };

    // ========================= TERNARY ===============================

    struct TernaryExpr : Expression
    {
        ExprPtr condition;
        ExprPtr thenExpr;
        ExprPtr elseExpr;
    };

    // ========================= SWIZZLE ===============================
    // a.xyz  — distinct from MemberAccess for code generation

    struct SwizzleExpr : Expression
    {
        ExprPtr     object;
        std::string swizzle;    // "xyz", "rg", "xxxx", etc.
    };

    // ========================= LENGTH ================================
    // ParticleData.Length  — built-in StructuredBuffer property

    struct LengthExpr : Expression
    {
        ExprPtr object;
    };

} // namespace NSL
