#pragma once

#include "../ast/nsl_ast.h"
#include "../lexer/nsl_token.h"
#include "../../core/nsl_error.h"
#include "../../property/nsl_property_parser.h"
#include <vector>
#include <string>
#include <unordered_set>

// =====================================================================
//  NSL Parser  (v4 — Named PROPERTY Blocks)
//  Converts a flat Token stream into a Program AST.
//
//  Handles:
//    - AliasDecl: "Texture2D albedo = Materials.UserTextures.AlbedoMap;"
//      PROPERTY access is always fully qualified: PROPERTY.Group.Field.
//    - struct declarations
//    - function declarations (with attributes, semantics)
//    - global variables (static const, groupshared, etc.)
//    - preprocessor directives (#define, #ifdef, etc.)
//    - all expression forms including member access, index, call, etc.
//
//  PROPERTY block and group names are passed in for qualification checks
//  and clear diagnostics for forbidden short access/bindings.
// =====================================================================

namespace NSL
{
    class Parser
    {
    public:
        Parser(
            const std::vector<Token>&    tokens,
            const ParsedProperties&      properties,
            const std::string&           sourceFile
        );

        // Returns parsed Program. Check diag.HasErrors() afterwards.
        Program Parse(DiagnosticEngine& diag);

    private:
        // ==================== Token stream ====================
        bool          IsAtEnd()          const;
        const Token&  Current()          const;
        const Token&  Peek(int offset=1) const;
        const Token&  Advance();
        bool          Check(TokenKind k)  const;
        bool          Match(TokenKind k);
        bool          MatchAny(std::initializer_list<TokenKind> kinds);
        bool          Expect(TokenKind k, const std::string& context, DiagnosticEngine& diag);

        // ==================== Top-level parsers ====================
        bool TryParsePreprocessor(Program& prog);
        bool TryParseStructDecl(Program& prog, DiagnosticEngine& diag);
        bool TryParseFunctionOrGlobal(Program& prog, DiagnosticEngine& diag);

        // ==================== Statement parsers ====================
        StmtPtr ParseStatement(DiagnosticEngine& diag);
        StmtPtr ParseBlock(DiagnosticEngine& diag);
        StmtPtr ParseIfStmt(DiagnosticEngine& diag);
        StmtPtr ParseForStmt(DiagnosticEngine& diag);
        StmtPtr ParseWhileStmt(DiagnosticEngine& diag);
        StmtPtr ParseDoWhileStmt(DiagnosticEngine& diag);
        StmtPtr ParseSwitchStmt(DiagnosticEngine& diag);
        StmtPtr ParseReturnStmt(DiagnosticEngine& diag);
        StmtPtr ParseVarDeclStmt(const std::string& typeName, DiagnosticEngine& diag);
        StmtPtr ParseExprStmt(DiagnosticEngine& diag);

        // ==================== Expression parsers ====================
        ExprPtr ParseExpression(DiagnosticEngine& diag);
        ExprPtr ParseAssignment(DiagnosticEngine& diag);
        ExprPtr ParseTernary(DiagnosticEngine& diag);
        ExprPtr ParseOr(DiagnosticEngine& diag);
        ExprPtr ParseAnd(DiagnosticEngine& diag);
        ExprPtr ParseBitOr(DiagnosticEngine& diag);
        ExprPtr ParseBitXor(DiagnosticEngine& diag);
        ExprPtr ParseBitAnd(DiagnosticEngine& diag);
        ExprPtr ParseEquality(DiagnosticEngine& diag);
        ExprPtr ParseComparison(DiagnosticEngine& diag);
        ExprPtr ParseShift(DiagnosticEngine& diag);
        ExprPtr ParseAddSub(DiagnosticEngine& diag);
        ExprPtr ParseMulDiv(DiagnosticEngine& diag);
        ExprPtr ParseUnary(DiagnosticEngine& diag);
        ExprPtr ParsePostfix(DiagnosticEngine& diag);
        ExprPtr ParsePrimary(DiagnosticEngine& diag);
        ExprPtr ParseCallArgs(ExprPtr callee, DiagnosticEngine& diag);

        // ==================== Helpers ====================
        bool        IsTypeName(const Token& tok)    const;
        bool        IsPropertyGroupName(const std::string& name)  const;
        bool        IsPropertyBlockName(const std::string& name)  const; // v4 NEW
        // Returns the qualifiedName for (blockName, groupName) if it exists, else "".
        std::string ResolveQualifiedGroup(const std::string& blockName,
                                          const std::string& groupName) const; // v4 NEW
        std::string ReadTypeName(DiagnosticEngine& diag); // handles template args: StructuredBuffer<T>
        bool        IsSwizzle(const std::string& member)  const;

        // ==================== State ====================
        const std::vector<Token>*   m_tokens     = nullptr;
        size_t                      m_pos        = 0;
        std::string                 m_file;
        const ParsedProperties*     m_properties = nullptr; // v4 NEW: full property data

        // Short names retained only for clear legacy-syntax diagnostics.
        std::unordered_set<std::string> m_propertyGroupNames;
        // PROPERTY block names (e.g. "Materials") — for 3-level access
        std::unordered_set<std::string> m_propertyBlockNames;  // v4 NEW
    };

} // namespace NSL
