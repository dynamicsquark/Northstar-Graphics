#include "nsl_lexer.h"
#include <cctype>
#include <unordered_map>

namespace NSL
{

// =====================================================================
//  Keyword lookup table
// =====================================================================

TokenKind Lexer::LookupKeyword(const std::string& id)
{
    static const std::unordered_map<std::string, TokenKind> kKeywords =
    {
        // ==================== Scalar / Vector Types ====================
        { "Float",   TokenKind::KW_FLOAT  }, { "Int",     TokenKind::KW_INT    },
        { "UInt",    TokenKind::KW_UINT   }, { "Bool",    TokenKind::KW_BOOL   },
        { "Double",  TokenKind::KW_DOUBLE },
        { "Vec2",    TokenKind::KW_VEC2   }, { "Vec3",    TokenKind::KW_VEC3   },
        { "Vec4",    TokenKind::KW_VEC4   },
        { "Int2",    TokenKind::KW_INT2   }, { "Int3",    TokenKind::KW_INT3   },
        { "Int4",    TokenKind::KW_INT4   },
        { "UInt2",   TokenKind::KW_UINT2  }, { "UInt3",   TokenKind::KW_UINT3  },
        { "UInt4",   TokenKind::KW_UINT4  },
        { "Mat3",    TokenKind::KW_MAT3   }, { "Mat4",    TokenKind::KW_MAT4   },
        { "Color",   TokenKind::KW_COLOR  }, { "Quat",    TokenKind::KW_QUAT   },

        // ==================== Resource Types ====================
        { "Buffer",                  TokenKind::KW_BUFFER                },
        { "Texture2D",               TokenKind::KW_TEXTURE2D             },
        { "TextureCube",             TokenKind::KW_TEXTURECUBE           },
        { "Texture2DArray",          TokenKind::KW_TEXTURE2D_ARRAY       },
        { "TextureCubeArray",        TokenKind::KW_TEXTURECUBE_ARRAY     },
        { "SamplerState",            TokenKind::KW_SAMPLER_STATE         },
        { "SamplerComparisonState",  TokenKind::KW_SAMPLER_COMPARISON_STATE },
        { "StructuredBuffer",        TokenKind::KW_STRUCTURED_BUFFER     },
        { "RWStructuredBuffer",      TokenKind::KW_RW_STRUCTURED_BUFFER  },
        { "RWTexture2D",             TokenKind::KW_RW_TEXTURE2D          },

        // ==================== Flow Control ====================
        { "if",       TokenKind::KW_IF      }, { "else",     TokenKind::KW_ELSE    },
        { "for",      TokenKind::KW_FOR     }, { "while",    TokenKind::KW_WHILE   },
        { "do",       TokenKind::KW_DO      },
        { "switch",   TokenKind::KW_SWITCH  }, { "case",     TokenKind::KW_CASE    },
        { "default",  TokenKind::KW_DEFAULT },
        { "break",    TokenKind::KW_BREAK   }, { "continue", TokenKind::KW_CONTINUE},
        { "return",   TokenKind::KW_RETURN  }, { "discard",  TokenKind::KW_DISCARD },

        // ==================== Storage / Qualifiers ====================
        { "static",      TokenKind::KW_STATIC      }, { "const",   TokenKind::KW_CONST  },
        { "typedef",     TokenKind::KW_TYPEDEF      }, { "struct",  TokenKind::KW_STRUCT },
        { "groupshared", TokenKind::KW_GROUPSHARED  },
        { "in",          TokenKind::KW_IN           }, { "out",     TokenKind::KW_OUT    },
        { "inout",       TokenKind::KW_INOUT        },
        { "inline",      TokenKind::KW_INLINE       }, { "extern",  TokenKind::KW_EXTERN },

        // ==================== Built-in Functions ====================
        { "Dot",         TokenKind::KW_DOT         }, { "Cross",      TokenKind::KW_CROSS      },
        { "Normalize",   TokenKind::KW_NORMALIZE    }, { "Sample",     TokenKind::KW_SAMPLE     },
        { "SampleLevel", TokenKind::KW_SAMPLE_LEVEL }, { "SampleCmp",  TokenKind::KW_SAMPLE_CMP },
        { "Pow",         TokenKind::KW_POW          }, { "Sqrt",       TokenKind::KW_SQRT       },
        { "Rsqrt",       TokenKind::KW_RSQRT        },
        { "Abs",         TokenKind::KW_ABS          }, { "Sign",       TokenKind::KW_SIGN       },
        { "Floor",       TokenKind::KW_FLOOR        }, { "Ceil",       TokenKind::KW_CEIL       },
        { "Round",       TokenKind::KW_ROUND        },
        { "Saturate",    TokenKind::KW_SATURATE     }, { "Clamp",      TokenKind::KW_CLAMP      },
        { "Lerp",        TokenKind::KW_LERP         }, { "SmoothStep", TokenKind::KW_SMOOTHSTEP },
        { "Max",         TokenKind::KW_MAX          }, { "Min",        TokenKind::KW_MIN        },
        { "Length",      TokenKind::KW_LENGTH       }, { "Distance",   TokenKind::KW_DISTANCE   },
        { "Mul",         TokenKind::KW_MUL          },
        { "Reflect",     TokenKind::KW_REFLECT      }, { "Refract",    TokenKind::KW_REFRACT    },
        { "Transpose",   TokenKind::KW_TRANSPOSE    }, { "Determinant",TokenKind::KW_DETERMINANT},
        { "Ddx",         TokenKind::KW_DDX          }, { "Ddy",        TokenKind::KW_DDY        },
        { "Frac",        TokenKind::KW_FRAC         }, { "Trunc",      TokenKind::KW_TRUNC      },
        { "Modf",        TokenKind::KW_MODF         },
        { "Exp",         TokenKind::KW_EXP          }, { "Exp2",       TokenKind::KW_EXP2       },
        { "Log",         TokenKind::KW_LOG          }, { "Log2",       TokenKind::KW_LOG2       },
        { "Sin",         TokenKind::KW_SIN          }, { "Cos",        TokenKind::KW_COS        },
        { "Tan",         TokenKind::KW_TAN          },
        { "Asin",        TokenKind::KW_ASIN         }, { "Acos",       TokenKind::KW_ACOS       },
        { "Atan",        TokenKind::KW_ATAN         }, { "Atan2",      TokenKind::KW_ATAN2      },
        { "All",         TokenKind::KW_ALL          }, { "Any",        TokenKind::KW_ANY        },
        { "InterlockedAdd", TokenKind::KW_INTERLOCKED_ADD },
        { "SampleCmpLevelZero", TokenKind::KW_SAMPLE_CMP_LEVEL_ZERO },
        { "CountBits",     TokenKind::KW_COUNT_BITS      },
        { "FirstBitHigh",  TokenKind::KW_FIRST_BIT_HIGH  },
        { "FirstBitLow",   TokenKind::KW_FIRST_BIT_LOW   },
        { "Append",        TokenKind::KW_APPEND           },
        { "RestartStrip",  TokenKind::KW_RESTART_STRIP    },

        // ==================== Literals ====================
        { "true",   TokenKind::LIT_BOOL },
        { "false",  TokenKind::LIT_BOOL },
    };

    auto it = kKeywords.find(id);
    return (it != kKeywords.end()) ? it->second : TokenKind::IDENTIFIER;
}

// =====================================================================
//  Token helpers
// =====================================================================

bool Token::IsType() const
{
    return kind >= TokenKind::KW_FLOAT && kind <= TokenKind::KW_QUAT;
}

bool Token::IsResourceType() const
{
    return kind >= TokenKind::KW_BUFFER && kind <= TokenKind::KW_RW_TEXTURE2D;
}

std::string Token::KindName() const
{
    if (!lexeme.empty()) return "'" + lexeme + "'";
    switch (kind)
    {
        case TokenKind::END_OF_FILE:    return "end-of-file";
        case TokenKind::IDENTIFIER:     return "identifier";
        case TokenKind::LIT_INT:        return "integer literal";
        case TokenKind::LIT_UINT:       return "unsigned integer literal";
        case TokenKind::LIT_FLOAT:      return "float literal";
        case TokenKind::LIT_BOOL:       return "bool literal";
        case TokenKind::LIT_STRING:     return "string literal";
        case TokenKind::OP_DOT:         return "'.'";
        case TokenKind::PUNC_SEMICOLON: return "';'";
        case TokenKind::PUNC_LBRACE:    return "'{'";
        case TokenKind::PUNC_RBRACE:    return "'}'";
        case TokenKind::PUNC_LPAREN:    return "'('";
        case TokenKind::PUNC_RPAREN:    return "')'";
        case TokenKind::PUNC_COMMA:     return "','";
        case TokenKind::OP_ASSIGN:      return "'='";
        case TokenKind::OP_COLON:       return "':'";
        default:                         return "token";
    }
}

// =====================================================================
//  Constructor
// =====================================================================

Lexer::Lexer(const std::string& source, const std::string& file, uint32_t lineOffset)
    : m_source(source)
    , m_pos(0)
    , m_line(1)
    , m_col(1)
    , m_file(file)
    , m_lineOffset(lineOffset)
{
}

// =====================================================================
//  Public — Tokenize
// =====================================================================

std::vector<Token> Lexer::Tokenize()
{
    std::vector<Token> tokens;

    while (!IsAtEnd())
    {
        SkipWhitespace();
        if (IsAtEnd()) break;

        char c = Current();

        // ==================== Comments ====================
        if (c == '/' && Peek() == '/')
        {
            SkipLineComment();
            continue;
        }
        if (c == '/' && Peek() == '*')
        {
            Advance(); Advance();
            SkipBlockComment();
            continue;
        }

        // ==================== Preprocessor ====================
        if (c == '#')
        {
            Token ppTok = ReadPreprocessorDirective();
            // Internal #line <N> ["<file>"] directives update source location tracking
            // without emitting tokens into the AST / HLSL stream.
            if (ppTok.kind == TokenKind::UNKNOWN)
            {
                const std::string& lex = ppTok.lexeme;
                size_t idx = 1;
                while (idx < lex.size() && (lex[idx] == ' ' || lex[idx] == '\t')) ++idx;
                if (lex.compare(idx, 4, "line") == 0 &&
                    (idx + 4 == lex.size() || lex[idx + 4] == ' ' || lex[idx + 4] == '\t'))
                {
                    idx += 4;
                    while (idx < lex.size() && (lex[idx] == ' ' || lex[idx] == '\t')) ++idx;
                    uint32_t newLine = 0;
                    bool hasDigits = false;
                    while (idx < lex.size() && std::isdigit(static_cast<unsigned char>(lex[idx])))
                    {
                        hasDigits = true;
                        newLine = newLine * 10u + static_cast<uint32_t>(lex[idx] - '0');
                        ++idx;
                    }
                    if (hasDigits)
                    {
                        while (idx < lex.size() && (lex[idx] == ' ' || lex[idx] == '\t')) ++idx;
                        if (idx < lex.size() && lex[idx] == '"')
                        {
                            const size_t closeQ = lex.find('"', idx + 1);
                            if (closeQ != std::string::npos)
                            {
                                m_file = lex.substr(idx + 1, closeQ - idx - 1);
                            }
                        }
                        if (!IsAtEnd() && Current() == '\r') Advance();
                        if (!IsAtEnd() && Current() == '\n') Advance();
                        m_line = (newLine > 0u) ? newLine : 1u;
                        m_col  = 1u;
                        m_lineOffset = 0u;
                        continue;
                    }
                }
            }
            tokens.push_back(std::move(ppTok));
            continue;
        }

        // ==================== Numbers ====================
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && std::isdigit(static_cast<unsigned char>(Peek()))))
        {
            tokens.push_back(ReadNumber());
            continue;
        }

        // ==================== Strings ====================
        if (c == '"')
        {
            tokens.push_back(ReadString());
            continue;
        }

        // ==================== Identifiers / Keywords ====================
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
        {
            tokens.push_back(ReadIdentifierOrKeyword());
            continue;
        }

        // ==================== Operators / Punctuation ====================
        tokens.push_back(ReadOperatorOrPunct());
    }

    // Append EOF
    Token eof;
    eof.kind     = TokenKind::END_OF_FILE;
    eof.lexeme   = "";
    eof.location = { m_file, m_line + m_lineOffset, m_col };
    tokens.push_back(eof);

    return tokens;
}

// =====================================================================
//  ReadNumber
// =====================================================================

Token Lexer::ReadNumber()
{
    Token t = MakeToken(TokenKind::LIT_INT, "");
    std::string lexeme;

    // 1. Hexadecimal integer literal: 0x... / 0X...
    if (Current() == '0' && (Peek() == 'x' || Peek() == 'X'))
    {
        lexeme += Advance(); // '0'
        lexeme += Advance(); // 'x' / 'X'

        size_t hexDigits = 0;
        while (!IsAtEnd() && std::isxdigit(static_cast<unsigned char>(Current())))
        {
            lexeme += Advance();
            ++hexDigits;
        }

        bool isUnsigned = false;
        if (!IsAtEnd() && (Current() == 'u' || Current() == 'U'))
        {
            lexeme += Advance();
            isUnsigned = true;
        }

        // Reject invalid trailing identifier characters (e.g. 0x12g, 0xFFuu)
        if (hexDigits == 0 ||
            (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_')))
        {
            while (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
                lexeme += Advance();
            t.lexeme = lexeme;
            t.kind   = TokenKind::UNKNOWN;
            return t;
        }

        t.lexeme = lexeme;
        try
        {
            const std::string numBody = isUnsigned ? lexeme.substr(0, lexeme.size() - 1) : lexeme;
            size_t parsedPos = 0;
            const uint64_t val = std::stoull(numBody, &parsedPos, 16);
            if (parsedPos != numBody.size())
            {
                t.kind = TokenKind::UNKNOWN;
                return t;
            }
            if (isUnsigned || val > static_cast<uint64_t>(INT64_MAX))
            {
                t.kind      = TokenKind::LIT_UINT;
                t.uintValue = val;
                t.intValue  = static_cast<int64_t>(val);
            }
            else
            {
                t.kind      = TokenKind::LIT_INT;
                t.intValue  = static_cast<int64_t>(val);
                t.uintValue = val;
            }
        }
        catch (...)
        {
            t.kind = TokenKind::UNKNOWN;
        }
        return t;
    }

    // 2. Decimal integer or floating-point literal (including scientific notation e/E[+-]digits)
    bool isFloat = false;
    while (!IsAtEnd() && std::isdigit(static_cast<unsigned char>(Current())))
        lexeme += Advance();

    if (!IsAtEnd() && Current() == '.')
    {
        isFloat = true;
        lexeme += Advance(); // '.'
        while (!IsAtEnd() && std::isdigit(static_cast<unsigned char>(Current())))
            lexeme += Advance();
    }

    if (!IsAtEnd() && (Current() == 'e' || Current() == 'E'))
    {
        isFloat = true;
        lexeme += Advance(); // 'e' / 'E'
        if (!IsAtEnd() && (Current() == '+' || Current() == '-'))
            lexeme += Advance();

        size_t expDigits = 0;
        while (!IsAtEnd() && std::isdigit(static_cast<unsigned char>(Current())))
        {
            lexeme += Advance();
            ++expDigits;
        }

        if (expDigits == 0)
        {
            while (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
                lexeme += Advance();
            t.lexeme = lexeme;
            t.kind   = TokenKind::UNKNOWN;
            return t;
        }
    }

    bool hasFloatSuffix = false;
    bool isUnsigned     = false;
    if (!IsAtEnd() && (Current() == 'f' || Current() == 'F'))
    {
        lexeme += Advance();
        isFloat        = true;
        hasFloatSuffix = true;
    }
    else if (!isFloat && !IsAtEnd() && (Current() == 'u' || Current() == 'U'))
    {
        lexeme += Advance();
        isUnsigned = true;
    }

    // Reject trailing garbage identifier characters (e.g. 123abc, 1.0fu, 1.2.3)
    if (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
    {
        while (!IsAtEnd() && (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
            lexeme += Advance();
        t.lexeme = lexeme;
        t.kind   = TokenKind::UNKNOWN;
        return t;
    }

    t.lexeme = lexeme;
    try
    {
        if (isFloat)
        {
            const std::string numBody = hasFloatSuffix ? lexeme.substr(0, lexeme.size() - 1) : lexeme;
            size_t parsedPos = 0;
            t.floatValue = std::stod(numBody, &parsedPos);
            if (parsedPos != numBody.size())
            {
                t.kind = TokenKind::UNKNOWN;
                return t;
            }
            t.kind = TokenKind::LIT_FLOAT;
        }
        else if (isUnsigned)
        {
            const std::string numBody = lexeme.substr(0, lexeme.size() - 1);
            size_t parsedPos = 0;
            const uint64_t val = std::stoull(numBody, &parsedPos, 10);
            if (parsedPos != numBody.size())
            {
                t.kind = TokenKind::UNKNOWN;
                return t;
            }
            t.kind      = TokenKind::LIT_UINT;
            t.uintValue = val;
            t.intValue  = static_cast<int64_t>(val);
        }
        else
        {
            size_t parsedPos = 0;
            const uint64_t val = std::stoull(lexeme, &parsedPos, 10);
            if (parsedPos != lexeme.size())
            {
                t.kind = TokenKind::UNKNOWN;
                return t;
            }
            if (val > static_cast<uint64_t>(INT64_MAX))
            {
                t.kind      = TokenKind::LIT_UINT;
                t.uintValue = val;
                t.intValue  = static_cast<int64_t>(val);
            }
            else
            {
                t.kind      = TokenKind::LIT_INT;
                t.intValue  = static_cast<int64_t>(val);
                t.uintValue = val;
            }
        }
    }
    catch (...)
    {
        t.kind = TokenKind::UNKNOWN;
    }
    return t;
}

// =====================================================================
//  ReadString
// =====================================================================

Token Lexer::ReadString()
{
    Token t = MakeToken(TokenKind::LIT_STRING, "");
    std::string lexeme;
    Advance(); // opening '"'
    while (!IsAtEnd() && Current() != '"')
    {
        if (Current() == '\\')
        {
            lexeme += Advance(); // backslash
            if (IsAtEnd())
                break; // trailing '\' — unterminated; do not Advance past EOF
            lexeme += Advance(); // escaped char
            continue;
        }
        lexeme += Advance();
    }
    if (!IsAtEnd()) Advance(); // closing '"'
    t.lexeme = lexeme;
    return t;
}

// =====================================================================
//  ReadIdentifierOrKeyword
// =====================================================================

Token Lexer::ReadIdentifierOrKeyword()
{
    Token t = MakeToken(TokenKind::IDENTIFIER, "");
    std::string id;
    while (!IsAtEnd() &&
           (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
        id += Advance();

    TokenKind kw = LookupKeyword(id);

    t.kind   = kw;
    t.lexeme = id;

    // For bool literals, set intValue
    if (kw == TokenKind::LIT_BOOL)
        t.intValue = (id == "true") ? 1 : 0;

    return t;
}

// =====================================================================
//  ReadPreprocessorDirective
// =====================================================================

Token Lexer::ReadPreprocessorDirective()
{
    Token t = MakeToken(TokenKind::UNKNOWN, "");
    std::string lexeme;
    lexeme += Advance(); // '#'

    while (!IsAtEnd() && Current() != '\n')
        lexeme += Advance();

    t.lexeme = lexeme;

    // Classify by exact directive name after '#' (optional whitespace).
    // Allowlist only: define, ifdef, endif, undef.
    // Rejects #include, #pragma, and any unknown directive.
    size_t i = 1;
    while (i < lexeme.size() && (lexeme[i] == ' ' || lexeme[i] == '\t'))
        ++i;

    size_t nameStart = i;
    while (i < lexeme.size() &&
           (std::isalnum(static_cast<unsigned char>(lexeme[i])) || lexeme[i] == '_'))
    {
        ++i;
    }

    const std::string name = lexeme.substr(nameStart, i - nameStart);

    if (name == "define")      t.kind = TokenKind::PP_DEFINE;
    else if (name == "if")     t.kind = TokenKind::PP_IF;
    else if (name == "ifdef")  t.kind = TokenKind::PP_IFDEF;
    else if (name == "ifndef") t.kind = TokenKind::PP_IFNDEF;
    else if (name == "else")   t.kind = TokenKind::PP_ELSE;
    else if (name == "elif")   t.kind = TokenKind::PP_ELIF;
    else if (name == "endif")  t.kind = TokenKind::PP_ENDIF;
    else if (name == "undef")  t.kind = TokenKind::PP_UNDEF;
    else                       t.kind = TokenKind::UNKNOWN;

    return t;
}

// =====================================================================
//  ReadOperatorOrPunct
// =====================================================================

Token Lexer::ReadOperatorOrPunct()
{
    Token t = MakeToken(TokenKind::UNKNOWN, "");
    char  c = Advance();
    char  n = Current(); // peek at next without advancing

    std::string lex(1, c);

    auto make = [&](TokenKind k, bool consumeNext = false) -> Token&
    {
        if (consumeNext) { lex += Advance(); }
        t.kind   = k;
        t.lexeme = lex;
        return t;
    };

    switch (c)
    {
        case '(': return make(TokenKind::PUNC_LPAREN);
        case ')': return make(TokenKind::PUNC_RPAREN);
        case '{': return make(TokenKind::PUNC_LBRACE);
        case '}': return make(TokenKind::PUNC_RBRACE);
        case '[': return make(TokenKind::PUNC_LBRACKET);
        case ']': return make(TokenKind::PUNC_RBRACKET);
        case ';': return make(TokenKind::PUNC_SEMICOLON);
        case ',': return make(TokenKind::PUNC_COMMA);
        case '?': return make(TokenKind::OP_QUESTION);
        case '~': return make(TokenKind::OP_BIT_NOT);

        case '.':
            if (std::isdigit(static_cast<unsigned char>(n)))
            {
                // Part of a float — shouldn't reach here normally
                t.kind = TokenKind::LIT_FLOAT; t.lexeme = lex; return t;
            }
            return make(TokenKind::OP_DOT);

        case ':':
            if (n == ':') return make(TokenKind::UNKNOWN, true); // :: scope (not used in NSL)
            return make(TokenKind::OP_COLON);

        case '+':
            if (n == '+') return make(TokenKind::OP_INC, true);
            if (n == '=') return make(TokenKind::OP_PLUS_ASSIGN, true);
            return make(TokenKind::OP_PLUS);

        case '-':
            if (n == '-') return make(TokenKind::OP_DEC, true);
            if (n == '=') return make(TokenKind::OP_MINUS_ASSIGN, true);
            if (n == '>') return make(TokenKind::OP_ARROW, true);
            return make(TokenKind::OP_MINUS);

        case '*':
            if (n == '=') return make(TokenKind::OP_STAR_ASSIGN, true);
            return make(TokenKind::OP_STAR);

        case '/':
            if (n == '=') return make(TokenKind::OP_SLASH_ASSIGN, true);
            return make(TokenKind::OP_SLASH);

        case '%':
            if (n == '=') return make(TokenKind::OP_PERCENT_ASSIGN, true);
            return make(TokenKind::OP_PERCENT);

        case '=':
            if (n == '=') return make(TokenKind::OP_EQ, true);
            return make(TokenKind::OP_ASSIGN);

        case '!':
            if (n == '=') return make(TokenKind::OP_NEQ, true);
            return make(TokenKind::OP_NOT);

        case '<':
            if (n == '=') return make(TokenKind::OP_LE, true);
            if (n == '<')
            {
                Advance(); // second '<'
                if (!IsAtEnd() && Current() == '=') return make(TokenKind::OP_SHL_ASSIGN, true);
                t.kind = TokenKind::OP_SHL; t.lexeme = "<<"; return t;
            }
            return make(TokenKind::OP_LT);

        case '>':
            if (n == '=') return make(TokenKind::OP_GE, true);
            if (n == '>')
            {
                Advance();
                if (!IsAtEnd() && Current() == '=') return make(TokenKind::OP_SHR_ASSIGN, true);
                t.kind = TokenKind::OP_SHR; t.lexeme = ">>"; return t;
            }
            return make(TokenKind::OP_GT);

        case '&':
            if (n == '&') return make(TokenKind::OP_AND, true);
            if (n == '=') return make(TokenKind::OP_AND_ASSIGN, true);
            return make(TokenKind::OP_BIT_AND);

        case '|':
            if (n == '|') return make(TokenKind::OP_OR, true);
            if (n == '=') return make(TokenKind::OP_OR_ASSIGN, true);
            return make(TokenKind::OP_BIT_OR);

        case '^':
            if (n == '=') return make(TokenKind::OP_XOR_ASSIGN, true);
            return make(TokenKind::OP_BIT_XOR);

        default:
            t.kind = TokenKind::UNKNOWN;
            t.lexeme = lex;
            return t;
    }
}

// =====================================================================
//  Char stream helpers
// =====================================================================

bool Lexer::IsAtEnd() const { return m_pos >= m_source.size(); }

char Lexer::Current() const { return IsAtEnd() ? '\0' : m_source[m_pos]; }

char Lexer::Peek(int n) const
{
    size_t idx = m_pos + static_cast<size_t>(n);
    return (idx < m_source.size()) ? m_source[idx] : '\0';
}

char Lexer::Advance()
{
    char c = m_source[m_pos++];
    if (c == '\n') { ++m_line; m_col = 1; }
    else            { ++m_col; }
    return c;
}

void Lexer::SkipWhitespace()
{
    while (!IsAtEnd() && std::isspace(static_cast<unsigned char>(Current()))) Advance();
}

void Lexer::SkipLineComment()
{
    while (!IsAtEnd() && Current() != '\n') Advance();
}

void Lexer::SkipBlockComment()
{
    while (!IsAtEnd())
    {
        if (Current() == '*' && Peek() == '/') { Advance(); Advance(); return; }
        Advance();
    }
}

Token Lexer::MakeToken(TokenKind kind, const std::string& lexeme) const
{
    Token t;
    t.kind     = kind;
    t.lexeme   = lexeme;
    t.location = { m_file, m_line + m_lineOffset, m_col };
    return t;
}

} // namespace NSL
