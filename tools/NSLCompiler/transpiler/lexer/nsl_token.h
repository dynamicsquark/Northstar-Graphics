#pragma once

#include "../../core/nsl_error.h"
#include <string>
#include <cstdint>

// =====================================================================
//  NSL Token Types
// =====================================================================

namespace NSL
{
    // ========================= TOKEN KIND ============================

    enum class TokenKind : uint16_t
    {
        // ==================== Literals ====================
        LIT_INT,            // 42
        LIT_UINT,           // 42u, 0xFFu
        LIT_UINT64,         // 42ull, 0x100000000ull
        LIT_FLOAT,          // 3.14f, 3.14, 1e-5
        LIT_DOUBLE,         // 3.141592653589793238L, 3.14d
        LIT_BOOL,           // true, false
        LIT_STRING,         // "text"

        // ==================== Identifiers ====================
        IDENTIFIER,         // user-defined names

        // ==================== NSL Scalar/Vector Types ====================
        KW_FLOAT,   KW_INT,   KW_UINT,   KW_UINT64, KW_BOOL,   KW_DOUBLE,
        KW_VEC2,    KW_VEC3,  KW_VEC4,
        KW_INT2,    KW_INT3,  KW_INT4,
        KW_UINT2,   KW_UINT3, KW_UINT4,
        KW_MAT3,    KW_MAT4,
        KW_COLOR,   KW_QUAT,

        // ==================== NSL Resource Types ====================
        KW_BUFFER,
        KW_TEXTURE2D,
        KW_TEXTURECUBE,
        KW_TEXTURE2D_ARRAY,
        KW_TEXTURECUBE_ARRAY,
        KW_SAMPLER_STATE,
        KW_SAMPLER_COMPARISON_STATE,
        KW_STRUCTURED_BUFFER,
        KW_RW_STRUCTURED_BUFFER,
        KW_RW_TEXTURE2D,

        // ==================== Flow Control ====================
        KW_IF,      KW_ELSE,
        KW_FOR,     KW_WHILE,    KW_DO,
        KW_SWITCH,  KW_CASE,     KW_DEFAULT,
        KW_BREAK,   KW_CONTINUE, KW_RETURN,
        KW_DISCARD,

        // ==================== Storage / Qualifiers ====================
        KW_STATIC,  KW_CONST,  KW_TYPEDEF,
        KW_STRUCT,
        KW_GROUPSHARED,
        KW_IN,      KW_OUT,    KW_INOUT,
        KW_INLINE,  KW_EXTERN,

        // ==================== NSL Built-in Functions ====================
        KW_DOT,         KW_CROSS,       KW_NORMALIZE,
        KW_SAMPLE,      KW_SAMPLE_LEVEL,KW_SAMPLE_CMP,
        KW_POW,         KW_SQRT,        KW_RSQRT,
        KW_ABS,         KW_SIGN,        KW_FLOOR,   KW_CEIL,   KW_ROUND,
        KW_SATURATE,    KW_CLAMP,
        KW_LERP,        KW_SMOOTHSTEP,
        KW_MAX,         KW_MIN,
        KW_LENGTH,      KW_DISTANCE,
        KW_MUL,
        KW_REFLECT,     KW_REFRACT,
        KW_TRANSPOSE,   KW_DETERMINANT,
        KW_DDX,         KW_DDY,
        KW_FRAC,        KW_TRUNC,       KW_MODF,
        KW_EXP,         KW_EXP2,        KW_LOG,     KW_LOG2,
        KW_SIN,         KW_COS,         KW_TAN,
        KW_ASIN,        KW_ACOS,        KW_ATAN,    KW_ATAN2,
        KW_ALL,         KW_ANY,
        KW_INTERLOCKED_ADD,
        KW_SAMPLE_CMP_LEVEL_ZERO,
        KW_COUNT_BITS,  KW_FIRST_BIT_HIGH, KW_FIRST_BIT_LOW,
        KW_APPEND,      KW_RESTART_STRIP,

        // ==================== Preprocessor ====================
        PP_DEFINE,      // #define
        PP_IF,          // #if
        PP_IFDEF,       // #ifdef
        PP_IFNDEF,      // #ifndef
        PP_ELSE,        // #else
        PP_ELIF,        // #elif
        PP_ENDIF,       // #endif
        PP_UNDEF,       // #undef

        // ==================== Operators ====================
        // Arithmetic
        OP_PLUS,        // +
        OP_MINUS,       // -
        OP_STAR,        // *
        OP_SLASH,       // /
        OP_PERCENT,     // %

        // Assignment
        OP_ASSIGN,          // =
        OP_PLUS_ASSIGN,     // +=
        OP_MINUS_ASSIGN,    // -=
        OP_STAR_ASSIGN,     // *=
        OP_SLASH_ASSIGN,    // /=
        OP_PERCENT_ASSIGN,  // %=
        OP_AND_ASSIGN,      // &=
        OP_OR_ASSIGN,       // |=
        OP_XOR_ASSIGN,      // ^=
        OP_SHL_ASSIGN,      // <<=
        OP_SHR_ASSIGN,      // >>=

        // Comparison
        OP_EQ,          // ==
        OP_NEQ,         // !=
        OP_LT,          // <
        OP_GT,          // >
        OP_LE,          // <=
        OP_GE,          // >=

        // Logical
        OP_AND,         // &&
        OP_OR,          // ||
        OP_NOT,         // !

        // Bitwise
        OP_BIT_AND,     // &
        OP_BIT_OR,      // |
        OP_BIT_XOR,     // ^
        OP_BIT_NOT,     // ~
        OP_SHL,         // <<
        OP_SHR,         // >>

        // Increment / Decrement
        OP_INC,         // ++
        OP_DEC,         // --

        // Member access / Arrow
        OP_DOT,         // .      (member access)
        OP_ARROW,       // ->

        // Ternary
        OP_QUESTION,    // ?
        OP_COLON,       // :  (also used for semantics)

        // ==================== Punctuation ====================
        PUNC_LPAREN,    // (
        PUNC_RPAREN,    // )
        PUNC_LBRACE,    // {
        PUNC_RBRACE,    // }
        PUNC_LBRACKET,  // [
        PUNC_RBRACKET,  // ]
        PUNC_SEMICOLON, // ;
        PUNC_COMMA,     // ,

        // ==================== Special ====================
        END_OF_FILE,
        UNKNOWN,
    };

    // ========================= TOKEN ================================

    struct Token
    {
        TokenKind      kind     = TokenKind::UNKNOWN;
        std::string    lexeme;          // Raw source text
        SourceLocation location;

        // Typed value accessors (valid only for the respective literal kinds)
        int64_t  intValue       = 0;
        uint64_t uintValue      = 0;
        double   floatValue     = 0.0;
        bool     hasFloatSuffix = false;

        bool IsLiteral()    const { return kind == TokenKind::LIT_INT     ||
                                           kind == TokenKind::LIT_UINT    ||
                                           kind == TokenKind::LIT_UINT64  ||
                                           kind == TokenKind::LIT_FLOAT   ||
                                           kind == TokenKind::LIT_DOUBLE  ||
                                           kind == TokenKind::LIT_BOOL    ||
                                           kind == TokenKind::LIT_STRING;  }
        bool IsKeyword()    const { return kind >= TokenKind::KW_FLOAT && kind <= TokenKind::KW_RESTART_STRIP; }
        bool IsOperator()   const { return kind >= TokenKind::OP_PLUS  && kind <= TokenKind::OP_QUESTION; }
        bool IsEOF()        const { return kind == TokenKind::END_OF_FILE; }
        bool IsType()       const;  // true for all type keywords (KW_FLOAT … KW_QUAT)
        bool IsResourceType() const; // true for KW_BUFFER … KW_RW_TEXTURE2D

        std::string KindName() const; // Human-readable name for error messages
    };

} // namespace NSL
