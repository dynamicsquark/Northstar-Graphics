#pragma once

#include "nsl_token.h"
#include <vector>

// =====================================================================
//  NSL Lexer
//  Tokenizes the inner SHADER { ... } source (after preprocessing and
//  file parsing).  Does not process PROPERTY blocks.
//
//  SHADER preprocessor directives are allowlisted to #define, #ifdef,
//  #ifndef, #else, #elif, #endif, and #undef (exact directive-name match).
//  #include, #pragma,
//  and any other # directive lex as UNKNOWN and must be rejected later.
// =====================================================================

namespace NSL
{
    class Lexer
    {
    public:
        explicit Lexer(const std::string& source,
                       const std::string& file    = "",
                       uint32_t           lineOffset = 0);

        // Tokenize the full source. Always appends a synthetic END_OF_FILE token.
        std::vector<Token> Tokenize();

    private:
        // ==================== Char stream ====================
        bool   IsAtEnd()      const;
        char   Current()      const;
        char   Peek(int n=1)  const;
        char   Advance();
        void   SkipWhitespace();
        void   SkipLineComment();
        void   SkipBlockComment();

        // ==================== Token builders ====================
        Token  MakeToken(TokenKind kind, const std::string& lexeme) const;
        Token  ReadNumber();
        Token  ReadString();
        Token  ReadIdentifierOrKeyword();
        Token  ReadPreprocessorDirective();
        Token  ReadOperatorOrPunct();

        // ==================== Keyword table ====================
        static TokenKind LookupKeyword(const std::string& id);

        // Source tracking
        std::string m_source;
        size_t      m_pos        = 0;
        uint32_t    m_line       = 1;
        uint32_t    m_col        = 1;
        std::string m_file;
        uint32_t    m_lineOffset = 0; // Added to m_line for correct source locations
    };

} // namespace NSL
