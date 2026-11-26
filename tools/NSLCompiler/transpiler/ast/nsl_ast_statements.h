#pragma once

#include "nsl_ast_expressions.h"

// =====================================================================
//  NSL AST — Statements
// =====================================================================

namespace NSL
{
    struct Statement;
    using StmtPtr = std::unique_ptr<Statement>;

    // ========================= BASE STATEMENT ========================

    struct Statement
    {
        virtual ~Statement() = default;
        SourceLocation location;
    };

    // ========================= BLOCK =================================

    struct BlockStmt : Statement
    {
        std::vector<StmtPtr> statements;
    };

    // ========================= EXPRESSION STATEMENT ==================

    struct ExprStmt : Statement
    {
        ExprPtr expr;
    };

    // ========================= VARIABLE DECLARATION ==================

    struct VarDeclStmt : Statement
    {
        std::string typeName;
        std::string name;
        std::string arraySizeExpr;      // "" if not array
        ExprPtr     initializer;        // may be nullptr
        bool        isConst      = false;
        bool        isStatic     = false;
        bool        isGroupShared = false;
    };

    // ========================= RETURN ================================

    struct ReturnStmt : Statement
    {
        ExprPtr value;  // nullptr for bare return;
    };

    // ========================= IF / ELSE =============================

    struct IfStmt : Statement
    {
        ExprPtr  condition;
        StmtPtr  thenBranch;
        StmtPtr  elseBranch;    // nullptr if no else
    };

    // ========================= FOR ===================================

    struct ForStmt : Statement
    {
        StmtPtr  init;          // VarDeclStmt or ExprStmt or nullptr
        ExprPtr  condition;     // nullptr = infinite loop
        ExprPtr  increment;     // nullptr if omitted
        StmtPtr  body;
    };

    // ========================= WHILE =================================

    struct WhileStmt : Statement
    {
        ExprPtr  condition;
        StmtPtr  body;
    };

    // ========================= DO-WHILE ==============================

    struct DoWhileStmt : Statement
    {
        StmtPtr  body;
        ExprPtr  condition;
    };

    // ========================= SWITCH ================================

    struct SwitchCase
    {
        ExprPtr              value;         // nullptr = default:
        std::vector<StmtPtr> statements;
    };

    struct SwitchStmt : Statement
    {
        ExprPtr                  condition;
        std::vector<SwitchCase>  cases;
    };

    // ========================= BREAK / CONTINUE ======================

    struct BreakStmt    : Statement {};
    struct ContinueStmt : Statement {};

    // ========================= DISCARD ===============================

    struct DiscardStmt : Statement {};

    // ========================= ATTRIBUTE =============================
    // [numthreads(8, 8, 1)], [unroll], etc.

    struct Attribute
    {
        std::string name;
        std::string args; // raw argument string
    };

} // namespace NSL
