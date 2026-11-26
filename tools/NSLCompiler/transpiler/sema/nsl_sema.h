#pragma once

#include "../ast/nsl_ast.h"
#include "../../core/nsl_error.h"
#include "../../property/nsl_property_parser.h"
#include <unordered_map>
#include <unordered_set>
#include <string>

// =====================================================================
//  NSL Semantic Analyzer
//
//  Validates the Program AST in context of the parsed PROPERTY data.
//  Checks:
//    1. PROPERTY binding declarations are rejected
//    2. AliasDecl requires Property.Resource.Data
//    3. Direct PROPERTY access requires Property.Resource.Data
//    4. Function declarations have correct stage assignment for Main()
//    5. Duplicate function names
//    6. Undefined variable references (best-effort)
//    7. PROPERTY groups declared but never accessed → WARNING
//    8. Alias name collisions
// =====================================================================

namespace NSL
{
    class SemanticAnalyzer
    {
    public:
        // Analyze the program against parsed properties.
        // Returns false if there are errors.
        bool Analyze(
            const Program&          program,
            const ParsedProperties& properties,
            DiagnosticEngine&       diag
        );

    private:
        // ==================== Validation passes ====================
        void ValidatePropertyBlockNames(
            const ParsedProperties& properties,
            DiagnosticEngine&       diag
        );

        void ValidateAliases(
            const Program&          program,
            const ParsedProperties& properties,
            DiagnosticEngine&       diag
        );

        void ValidatePropertyAccessSyntax(
            const Program&          program,
            const ParsedProperties& properties,
            DiagnosticEngine&       diag
        ) const;

        void ValidateAccessInStmt(
            const Statement*         stmt,
            const ParsedProperties&  properties,
            const Program&           program,
            std::unordered_set<std::string>& localScope,
            DiagnosticEngine&        diag
        ) const;

        void ValidateAccessInExpr(
            const Expression*        expr,
            const ParsedProperties&  properties,
            const Program&           program,
            const std::unordered_set<std::string>& localScope,
            DiagnosticEngine&        diag,
            bool                     isIndexTarget = false
        ) const;

        void CheckReadOnlyMutationTarget(
            const Expression*        target,
            const ParsedProperties&  properties,
            const std::unordered_set<std::string>& localScope,
            DiagnosticEngine&        diag,
            const SourceLocation&    loc
        ) const;

        void ValidateFunctions(
            const Program&          program,
            DiagnosticEngine&       diag
        );

        void ValidatePropertyUsage(
            const Program&          program,
            const ParsedProperties& properties,
            DiagnosticEngine&       diag
        );

        // ==================== Expression walk ====================
        // Visits all expressions in a function body to collect property accesses.
        void CollectPropertyAccesses(
            const std::vector<StmtPtr>& stmts,
            std::unordered_set<std::string>& outAccessedGroups
        ) const;

        void CollectInStmt(
            const Statement* stmt,
            std::unordered_set<std::string>& outAccessedGroups
        ) const;

        void CollectInExpr(
            const Expression* expr,
            std::unordered_set<std::string>& outAccessedGroups
        ) const;

        // ==================== Lookup helpers ====================
        const PropertyGroup* FindGroup(
            const ParsedProperties& props,
            const std::string&      name
        ) const;

        bool GroupHasField(
            const PropertyGroup& group,
            const std::string&   fieldName
        ) const;

        // ==================== Scope tracking ====================
        // PROPERTY blocks, qualified groups, and aliases known to sema.
        std::unordered_set<std::string> m_knownNames;
        std::unordered_set<std::string> m_structuredAliases;
        std::unordered_set<std::string> m_getAliases;
        std::unordered_set<std::string> m_readOnlyResourceAliases;
    };

} // namespace NSL
