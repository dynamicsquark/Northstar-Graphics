#include "nsl_parser.h"
#include <cassert>

namespace NSL
{

// =====================================================================
//  Constructor
// =====================================================================

Parser::Parser(
    const std::vector<Token>&   tokens,
    const ParsedProperties&     properties,
    const std::string&          sourceFile)
    : m_tokens(&tokens)
    , m_pos(0)
    , m_file(sourceFile)
    , m_properties(&properties)
{
    // Collect short names only to diagnose forbidden legacy bindings/aliases.
    for (const auto& g : properties.groups)
        m_propertyGroupNames.insert(g.name);

    // Collect PROPERTY block names (for 3-level access: Property.Group.Field)
    for (const auto& block : properties.blocks)
        m_propertyBlockNames.insert(block.name);
}

// =====================================================================
//  Parse — Top-level entry point
// =====================================================================

Program Parser::Parse(DiagnosticEngine& diag)
{
    diag.SetFile(m_file);
    Program prog;
    prog.stage = ShaderStage::NONE; // set by pipeline orchestrator

    while (!IsAtEnd())
    {
        if (diag.HasErrors()) break;

        // ==================== Preprocessor directives ====================
        if (Current().kind == TokenKind::PP_DEFINE ||
            Current().kind == TokenKind::PP_IF     ||
            Current().kind == TokenKind::PP_IFDEF  ||
            Current().kind == TokenKind::PP_IFNDEF ||
            Current().kind == TokenKind::PP_ELSE   ||
            Current().kind == TokenKind::PP_ELIF   ||
            Current().kind == TokenKind::PP_ENDIF  ||
            Current().kind == TokenKind::PP_UNDEF)
        {
            TryParsePreprocessor(prog);
            continue;
        }

        // ==================== Struct declaration ====================
        if (Current().kind == TokenKind::KW_STRUCT)
        {
            TryParseStructDecl(prog, diag);
            continue;
        }

        // ==================== Typedef ====================
        if (Current().kind == TokenKind::KW_TYPEDEF)
        {
            Advance(); // consume 'typedef'
            std::string origType = ReadTypeName(diag);
            if (diag.HasErrors()) break;

            std::string alias;
            if (!Check(TokenKind::IDENTIFIER))
            {
                diag.ReportError(Current().location, "Expected alias name in typedef.");
                break;
            }
            alias = Advance().lexeme;

            if (!Expect(TokenKind::PUNC_SEMICOLON, "after typedef", diag)) break;

            TypedefDecl td;
            td.originalType = origType;
            td.aliasName    = alias;
            prog.typedefs.push_back(std::move(td));
            continue;
        }

        // ==================== Legacy PROPERTY bindings are forbidden ====================
        bool isBinding = false;
        if (Current().kind == TokenKind::IDENTIFIER)
        {
            if (IsPropertyBlockName(Current().lexeme) && Peek().kind == TokenKind::OP_DOT)
                isBinding = true;
            else if (IsPropertyGroupName(Current().lexeme) && Peek().kind == TokenKind::IDENTIFIER)
                isBinding = true;
        }

        if (isBinding)
        {
            diag.ReportError(Current().location,
                "PROPERTY binding variables are not allowed. Use the fully-qualified "
                "'Property.Resource.Data' path or declare an alias from that path.");
            break;
        }

        // ==================== Type name → could be AliasDecl, VarDecl, or FunctionDecl ====================
        if (IsTypeName(Current()) || 
            (Current().kind == TokenKind::IDENTIFIER && Peek().kind == TokenKind::IDENTIFIER))
        {
            // Look ahead: TypeName Name = Property.Group.Field; → AliasDecl
            //             TypeName Name = <anything>;         → GlobalVarDecl
            //             TypeName Name ( ...                 → FunctionDecl
            //             TypeName Name ;                     → GlobalVarDecl

            // Check if this looks like an alias: TypeName alias = something.something;
            if (Peek().kind == TokenKind::IDENTIFIER)
            {
                // Save position and try alias parse first
                size_t savedPos = m_pos;
                std::string typeName = Advance().lexeme;  // consume type
                std::string name     = Advance().lexeme;  // consume name

                if (Check(TokenKind::OP_ASSIGN))
                {
                    Advance(); // consume '='
                    // PROPERTY aliases must use: Property.Resource.Data
                    if (Check(TokenKind::IDENTIFIER) && Peek(1).kind == TokenKind::OP_DOT)
                    {
                        const SourceLocation aliasLocation = Current().location;
                        std::string first = Advance().lexeme;
                        Advance();                             // '.'

                        std::string srcBinding;
                        std::string srcField;

                        if (IsPropertyBlockName(first))
                        {
                            if (!Check(TokenKind::IDENTIFIER))
                            {
                                diag.ReportError(Current().location,
                                    "Expected PROPERTY resource group after '" + first + ".'.");
                                break;
                            }
                            std::string second = Advance().lexeme;
                            if (!Check(TokenKind::OP_DOT))
                            {
                                diag.ReportError(Current().location,
                                    "PROPERTY aliases require the fully-qualified "
                                    "'Property.Resource.Data' path.");
                                break;
                            }
                            Advance();                             // '.'
                            if (!Check(TokenKind::IDENTIFIER))
                            {
                                diag.ReportError(Current().location, "Expected field name after '.'.");
                                break;
                            }
                            srcField = Advance().lexeme;
                            srcBinding = ResolveQualifiedGroup(first, second);
                            if (srcBinding.empty())
                            {
                                diag.ReportError(aliasLocation,
                                    "Unknown PROPERTY resource group '" + first + "." +
                                    second + "'.");
                                break;
                            }
                        }
                        else if (IsPropertyGroupName(first))
                        {
                            diag.ReportError(aliasLocation,
                                "Short PROPERTY alias source '" + first +
                                ".*' is not allowed. Use 'Property.Resource.Data'.");
                            break;
                        }
                        else
                        {
                            // A normal non-PROPERTY initializer; parse it as a global.
                            m_pos = savedPos;
                            TryParseFunctionOrGlobal(prog, diag);
                            continue;
                        }

                        if (!Expect(TokenKind::PUNC_SEMICOLON, "after alias declaration", diag)) break;

                        AliasDecl alias;
                        alias.typeName      = typeName;
                        alias.aliasName     = name;
                        alias.sourceBinding = srcBinding;
                        alias.sourceField   = srcField;
                        alias.location      = aliasLocation;
                        prog.aliases.push_back(std::move(alias));
                        continue;
                    }
                    else
                    {
                        // Not an alias — roll back and parse as global var / function
                        m_pos = savedPos;
                    }
                }
                else
                {
                    // Not an alias — roll back
                    m_pos = savedPos;
                }
            }

            TryParseFunctionOrGlobal(prog, diag);
            continue;
        }

        // ==================== Storage qualifiers & Attributes ====================
        if (Current().kind == TokenKind::KW_STATIC ||
            Current().kind == TokenKind::KW_CONST  ||
            Current().kind == TokenKind::KW_GROUPSHARED ||
            Current().kind == TokenKind::KW_INLINE ||
            Current().kind == TokenKind::KW_EXTERN ||
            Current().kind == TokenKind::PUNC_LBRACKET)
        {
            TryParseFunctionOrGlobal(prog, diag);
            continue;
        }

        // ==================== Unknown token ====================
        diag.ReportError(Current().location,
            "Unexpected token '" + Current().lexeme + "' at top level. Expected global variable, struct, function, or property binding.");
        break;
    }

    return prog;
}

// =====================================================================
//  TryParsePreprocessor
// =====================================================================

bool Parser::TryParsePreprocessor(Program& prog)
{
    PreprocessorDirective pp;
    pp.line     = Current().lexeme;
    pp.location = Current().location;

    // Track #define names
    if (Current().kind == TokenKind::PP_DEFINE)
    {
        // Extract macro name from "#define NAME ..."
        const std::string& line = Current().lexeme;
        size_t start = line.find("define");
        if (start != std::string::npos)
        {
            start += 6; // skip "define"
            while (start < line.size() && line[start] == ' ') ++start;
            size_t end = start;
            while (end < line.size() && (std::isalnum(static_cast<unsigned char>(line[end])) || line[end] == '_'))
                ++end;
            prog.definedMacros.insert(line.substr(start, end - start));
        }
    }

    prog.preprocessors.push_back(std::move(pp));
    Advance();
    return true;
}

// =====================================================================
//  TryParseStructDecl
// =====================================================================

bool Parser::TryParseStructDecl(Program& prog, DiagnosticEngine& diag)
{
    Advance(); // consume 'struct'

    if (!Check(TokenKind::IDENTIFIER))
    {
        diag.ReportError(Current().location, "Expected struct name.");
        return false;
    }

    StructDecl sd;
    sd.location = Current().location;
    sd.name     = Advance().lexeme;

    if (!Expect(TokenKind::PUNC_LBRACE, "in struct declaration", diag)) return false;

    while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACE))
    {
        if (diag.HasErrors()) return false;

        // Optional interpolation modifier
        std::string interp;
        if (Current().kind == TokenKind::IDENTIFIER &&
            (Current().lexeme == "nointerpolation" || Current().lexeme == "linear" ||
             Current().lexeme == "centroid"        || Current().lexeme == "noperspective"))
        {
            interp = Advance().lexeme;
        }

        // Type name
        std::string typeName = ReadTypeName(diag);
        if (diag.HasErrors()) return false;

        // Field name
        if (!Check(TokenKind::IDENTIFIER))
        {
            diag.ReportError(Current().location, "Expected field name in struct.");
            return false;
        }

        StructField field;
        field.typeName      = typeName;
        field.interpolation = interp;
        field.name          = Advance().lexeme;

        // Optional array: [size]
        if (Check(TokenKind::PUNC_LBRACKET))
        {
            Advance(); // '['
            while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACKET))
                field.arraySizeExpr += Advance().lexeme;
            Expect(TokenKind::PUNC_RBRACKET, "in array size", diag);
        }

        // Optional semantic: : CLIP_POSITION / DISPATCH_THREAD_ID / …
        if (Check(TokenKind::OP_COLON))
        {
            Advance(); // ':'
            while (!IsAtEnd() && !Check(TokenKind::PUNC_SEMICOLON) &&
                   !Check(TokenKind::PUNC_RBRACE))
                field.semantic += Advance().lexeme;
        }

        if (!Expect(TokenKind::PUNC_SEMICOLON, "after struct field", diag)) return false;

        sd.fields.push_back(std::move(field));
    }

    Expect(TokenKind::PUNC_RBRACE, "to close struct", diag);
    // Optional semicolon after struct
    if (Check(TokenKind::PUNC_SEMICOLON)) Advance();

    prog.structs.push_back(std::move(sd));
    return true;
}

// =====================================================================
//  TryParseFunctionOrGlobal
// =====================================================================

bool Parser::TryParseFunctionOrGlobal(Program& prog, DiagnosticEngine& diag)
{
    SourceLocation loc = Current().location;

    // ==================== Collect storage qualifiers ====================
    bool isStatic      = false;
    bool isConst       = false;
    bool isGroupShared = false;
    bool isInline      = false;

    while (!IsAtEnd())
    {
        if      (Check(TokenKind::KW_STATIC))      { isStatic = true;      Advance(); }
        else if (Check(TokenKind::KW_CONST))        { isConst  = true;      Advance(); }
        else if (Check(TokenKind::KW_GROUPSHARED))  { isGroupShared = true; Advance(); }
        else if (Check(TokenKind::KW_INLINE))       { isInline = true;      Advance(); }
        else if (Check(TokenKind::KW_EXTERN))       {                       Advance(); }
        else break;
    }

    // ==================== Attributes ====================
    std::vector<Attribute> attributes;
    while (Check(TokenKind::PUNC_LBRACKET))
    {
        Advance(); // '['
        Attribute attr;
        while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACKET))
        {
            // String-valued attribute args (e.g. domain("tri")) lose their quotes
            // in the lexer; re-add them so attribute text stays a valid string literal.
            if (Check(TokenKind::LIT_STRING))
                attr.args += "\"" + Advance().lexeme + "\"";
            else
                attr.args += Advance().lexeme;
        }
        // attr.name = first word in args
        size_t paren = attr.args.find('(');
        attr.name = (paren != std::string::npos) ? attr.args.substr(0, paren) : attr.args;
        Expect(TokenKind::PUNC_RBRACKET, "to close attribute", diag);
        attributes.push_back(std::move(attr));
    }

    // ==================== Return / variable type ====================
    std::string typeName = ReadTypeName(diag);
    if (diag.HasErrors() || typeName.empty()) return false;

    // ==================== Name ====================
    if (!Check(TokenKind::IDENTIFIER))
    {
        diag.ReportError(Current().location, "Expected identifier after type '" + typeName + "'.");
        return false;
    }
    std::string name = Advance().lexeme;

    // ==================== Function declaration ====================
    if (Check(TokenKind::PUNC_LPAREN))
    {
        FunctionDecl fd;
        fd.location     = loc;
        fd.returnType   = typeName;
        fd.name         = name;
        fd.attributes   = std::move(attributes);

        Advance(); // '('

        // ==================== Parameters ====================
        while (!IsAtEnd() && !Check(TokenKind::PUNC_RPAREN))
        {
            if (diag.HasErrors()) return false;

            FunctionParam param;

            // Optional qualifier
            if      (Check(TokenKind::KW_IN))    { param.qualifier = ParamQualifier::IN;    Advance(); }
            else if (Check(TokenKind::KW_OUT))   { param.qualifier = ParamQualifier::OUT;   Advance(); }
            else if (Check(TokenKind::KW_INOUT)) { param.qualifier = ParamQualifier::INOUT; Advance(); }

            param.typeName = ReadTypeName(diag);
            if (diag.HasErrors()) return false;

            if (!Check(TokenKind::IDENTIFIER))
            {
                diag.ReportError(Current().location, "Expected parameter name.");
                return false;
            }
            param.name = Advance().lexeme;

            // Array param
            if (Check(TokenKind::PUNC_LBRACKET))
            {
                Advance();
                while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACKET))
                    param.arraySizeExpr += Advance().lexeme;
                Expect(TokenKind::PUNC_RBRACKET, "in parameter array size", diag);
            }

            // Semantic
            if (Check(TokenKind::OP_COLON))
            {
                Advance();
                if (Check(TokenKind::IDENTIFIER)) param.semantic = Advance().lexeme;
            }

            fd.parameters.push_back(std::move(param));
            if (!Check(TokenKind::PUNC_RPAREN)) Expect(TokenKind::PUNC_COMMA, "between parameters", diag);
        }

        Expect(TokenKind::PUNC_RPAREN, "to close parameter list", diag);

        // Return semantic: : COLOR_TARGET
        if (Check(TokenKind::OP_COLON))
        {
            Advance();
            if (Check(TokenKind::IDENTIFIER)) fd.returnSemantic = Advance().lexeme;
        }

        // Function body
        StmtPtr body = ParseBlock(diag);
        if (diag.HasErrors()) return false;

        if (auto* blk = dynamic_cast<BlockStmt*>(body.get()))
            fd.body = std::move(blk->statements);

        prog.functions.push_back(std::move(fd));
        return true;
    }

    // ==================== Global variable ====================
    GlobalVarDecl gv;
    gv.location     = loc;
    gv.typeName     = typeName;
    gv.name         = name;
    gv.isStatic     = isStatic;
    gv.isConst      = isConst;
    gv.isGroupShared = isGroupShared;

    // Optional array
    if (Check(TokenKind::PUNC_LBRACKET))
    {
        Advance();
        while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACKET))
            gv.arraySizeExpr += Advance().lexeme;
        Expect(TokenKind::PUNC_RBRACKET, "in global array size", diag);
    }

    // Optional initializer
    if (Check(TokenKind::OP_ASSIGN))
    {
        Advance();
        gv.initializer = ParseExpression(diag);
    }

    Expect(TokenKind::PUNC_SEMICOLON, "after global variable declaration", diag);

    prog.globals.push_back(std::move(gv));
    return true;
}

// =====================================================================
//  Statement parsers
// =====================================================================

StmtPtr Parser::ParseStatement(DiagnosticEngine& diag)
{
    if (diag.HasErrors()) return nullptr;

    // Block
    if (Check(TokenKind::PUNC_LBRACE)) return ParseBlock(diag);

    // Control flow
    if (Check(TokenKind::KW_IF))       return ParseIfStmt(diag);
    if (Check(TokenKind::KW_FOR))      return ParseForStmt(diag);
    if (Check(TokenKind::KW_WHILE))    return ParseWhileStmt(diag);
    if (Check(TokenKind::KW_DO))       return ParseDoWhileStmt(diag);
    if (Check(TokenKind::KW_SWITCH))   return ParseSwitchStmt(diag);
    if (Check(TokenKind::KW_RETURN))   return ParseReturnStmt(diag);

    if (Check(TokenKind::KW_BREAK))
    {
        auto s = std::make_unique<BreakStmt>();
        s->location = Advance().location;
        Expect(TokenKind::PUNC_SEMICOLON, "after break", diag);
        return s;
    }
    if (Check(TokenKind::KW_CONTINUE))
    {
        auto s = std::make_unique<ContinueStmt>();
        s->location = Advance().location;
        Expect(TokenKind::PUNC_SEMICOLON, "after continue", diag);
        return s;
    }
    if (Check(TokenKind::KW_DISCARD))
    {
        auto s = std::make_unique<DiscardStmt>();
        s->location = Advance().location;
        Expect(TokenKind::PUNC_SEMICOLON, "after discard", diag);
        return s;
    }

    // Variable declarations — storage qualifiers or type names
    bool hasQual = Check(TokenKind::KW_STATIC) || Check(TokenKind::KW_CONST) ||
                   Check(TokenKind::KW_GROUPSHARED);

    if (hasQual || IsTypeName(Current()) || 
        (Current().kind == TokenKind::IDENTIFIER && Peek().kind == TokenKind::IDENTIFIER))
    {
        // Peek-ahead: if TypeName Identifier then declaration
        size_t savedPos = m_pos;

        bool isStatic = false, isConst = false, isGroupShared = false;
        while (Check(TokenKind::KW_STATIC) || Check(TokenKind::KW_CONST) || Check(TokenKind::KW_GROUPSHARED))
        {
            if (Check(TokenKind::KW_STATIC))      { isStatic = true;      Advance(); }
            else if (Check(TokenKind::KW_CONST))  { isConst  = true;      Advance(); }
            else                                   { isGroupShared = true; Advance(); }
        }

        if ((IsTypeName(Current()) || Current().kind == TokenKind::IDENTIFIER) &&
            (Peek().kind == TokenKind::IDENTIFIER || Peek().kind == TokenKind::OP_LT))
        {
            std::string typeName = ReadTypeName(diag);
            if (!diag.HasErrors() && !typeName.empty() && Check(TokenKind::IDENTIFIER))
            {
                StmtPtr stmt = ParseVarDeclStmt(typeName, diag);
                if (auto* vd = dynamic_cast<VarDeclStmt*>(stmt.get()))
                {
                    vd->isStatic      = isStatic;
                    vd->isConst       = isConst;
                    vd->isGroupShared = isGroupShared;
                }
                return stmt;
            }
        }

        // Not a declaration — roll back and parse as expression
        m_pos = savedPos;
    }

    return ParseExprStmt(diag);
}

StmtPtr Parser::ParseBlock(DiagnosticEngine& diag)
{
    auto blk = std::make_unique<BlockStmt>();
    blk->location = Current().location;

    Expect(TokenKind::PUNC_LBRACE, "to open block", diag);

    while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACE))
    {
        if (diag.HasErrors()) break;
        auto s = ParseStatement(diag);
        if (s) blk->statements.push_back(std::move(s));
    }

    Expect(TokenKind::PUNC_RBRACE, "to close block", diag);
    return blk;
}

StmtPtr Parser::ParseIfStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<IfStmt>();
    s->location = Advance().location; // 'if'
    Expect(TokenKind::PUNC_LPAREN, "after 'if'", diag);
    s->condition = ParseExpression(diag);
    Expect(TokenKind::PUNC_RPAREN, "after if condition", diag);
    s->thenBranch = ParseStatement(diag);
    if (Check(TokenKind::KW_ELSE))
    {
        Advance();
        s->elseBranch = ParseStatement(diag);
    }
    return s;
}

StmtPtr Parser::ParseForStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<ForStmt>();
    s->location = Advance().location; // 'for'
    Expect(TokenKind::PUNC_LPAREN, "after 'for'", diag);

    // Init
    if (!Check(TokenKind::PUNC_SEMICOLON))
        s->init = ParseStatement(diag);
    else
        Advance();

    // Condition
    if (!Check(TokenKind::PUNC_SEMICOLON))
        s->condition = ParseExpression(diag);
    Expect(TokenKind::PUNC_SEMICOLON, "in for statement", diag);

    // Increment
    if (!Check(TokenKind::PUNC_RPAREN))
        s->increment = ParseExpression(diag);
    Expect(TokenKind::PUNC_RPAREN, "to close for header", diag);

    s->body = ParseStatement(diag);
    return s;
}

StmtPtr Parser::ParseWhileStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<WhileStmt>();
    s->location = Advance().location; // 'while'
    Expect(TokenKind::PUNC_LPAREN, "after 'while'", diag);
    s->condition = ParseExpression(diag);
    Expect(TokenKind::PUNC_RPAREN, "after while condition", diag);
    s->body = ParseStatement(diag);
    return s;
}

StmtPtr Parser::ParseDoWhileStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<DoWhileStmt>();
    s->location = Advance().location; // 'do'
    s->body = ParseStatement(diag);
    Expect(TokenKind::KW_WHILE, "after do body", diag);
    Expect(TokenKind::PUNC_LPAREN, "after 'while' in do-while", diag);
    s->condition = ParseExpression(diag);
    Expect(TokenKind::PUNC_RPAREN, "after do-while condition", diag);
    Expect(TokenKind::PUNC_SEMICOLON, "after do-while", diag);
    return s;
}

StmtPtr Parser::ParseSwitchStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<SwitchStmt>();
    s->location = Advance().location; // 'switch'
    Expect(TokenKind::PUNC_LPAREN, "after 'switch'", diag);
    s->condition = ParseExpression(diag);
    Expect(TokenKind::PUNC_RPAREN, "after switch condition", diag);
    Expect(TokenKind::PUNC_LBRACE, "to open switch body", diag);

    while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACE))
    {
        if (diag.HasErrors()) break;

        SwitchCase sc;
        if (Check(TokenKind::KW_CASE))
        {
            Advance();
            sc.value = ParseExpression(diag);
            Expect(TokenKind::OP_COLON, "after case value", diag);
        }
        else if (Check(TokenKind::KW_DEFAULT))
        {
            Advance();
            Expect(TokenKind::OP_COLON, "after 'default'", diag);
        }
        else
        {
            sc.statements.push_back(ParseStatement(diag));
            s->cases.push_back(std::move(sc));
            continue;
        }

        // Parse statements until next case/default/}
        while (!IsAtEnd() && !Check(TokenKind::KW_CASE) &&
               !Check(TokenKind::KW_DEFAULT) && !Check(TokenKind::PUNC_RBRACE))
        {
            if (diag.HasErrors()) break;
            sc.statements.push_back(ParseStatement(diag));
        }
        s->cases.push_back(std::move(sc));
    }

    Expect(TokenKind::PUNC_RBRACE, "to close switch", diag);
    return s;
}

StmtPtr Parser::ParseReturnStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<ReturnStmt>();
    s->location = Advance().location; // 'return'
    if (!Check(TokenKind::PUNC_SEMICOLON))
        s->value = ParseExpression(diag);
    Expect(TokenKind::PUNC_SEMICOLON, "after return", diag);
    return s;
}

StmtPtr Parser::ParseVarDeclStmt(const std::string& typeName, DiagnosticEngine& diag)
{
    auto s = std::make_unique<VarDeclStmt>();
    s->typeName = typeName;

    if (!Check(TokenKind::IDENTIFIER))
    {
        diag.ReportError(Current().location, "Expected variable name.");
        return nullptr;
    }
    s->name     = Advance().lexeme;
    s->location = Current().location;

    // Array
    if (Check(TokenKind::PUNC_LBRACKET))
    {
        Advance();
        while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACKET))
            s->arraySizeExpr += Advance().lexeme;
        Expect(TokenKind::PUNC_RBRACKET, "in array size", diag);
    }

    // Initializer
    if (Check(TokenKind::OP_ASSIGN))
    {
        Advance();
        s->initializer = ParseExpression(diag);
    }

    Expect(TokenKind::PUNC_SEMICOLON, "after variable declaration", diag);
    return s;
}

StmtPtr Parser::ParseExprStmt(DiagnosticEngine& diag)
{
    auto s = std::make_unique<ExprStmt>();
    s->location = Current().location;
    s->expr     = ParseExpression(diag);
    Expect(TokenKind::PUNC_SEMICOLON, "after expression statement", diag);
    return s;
}

// =====================================================================
//  Expression parsers — Pratt-style recursive descent
// =====================================================================

ExprPtr Parser::ParseExpression(DiagnosticEngine& diag)
{
    return ParseAssignment(diag);
}

ExprPtr Parser::ParseAssignment(DiagnosticEngine& diag)
{
    ExprPtr left = ParseTernary(diag);
    if (diag.HasErrors() || !left) return nullptr;

    static const std::pair<TokenKind, BinaryExpr::Op> kAssignOps[] = {
        { TokenKind::OP_ASSIGN,         BinaryExpr::Op::Assign     },
        { TokenKind::OP_PLUS_ASSIGN,    BinaryExpr::Op::AddAssign  },
        { TokenKind::OP_MINUS_ASSIGN,   BinaryExpr::Op::SubAssign  },
        { TokenKind::OP_STAR_ASSIGN,    BinaryExpr::Op::MulAssign  },
        { TokenKind::OP_SLASH_ASSIGN,   BinaryExpr::Op::DivAssign  },
        { TokenKind::OP_PERCENT_ASSIGN, BinaryExpr::Op::ModAssign  },
        { TokenKind::OP_AND_ASSIGN,     BinaryExpr::Op::AndAssign  },
        { TokenKind::OP_OR_ASSIGN,      BinaryExpr::Op::OrAssign   },
        { TokenKind::OP_XOR_ASSIGN,     BinaryExpr::Op::XorAssign  },
        { TokenKind::OP_SHL_ASSIGN,     BinaryExpr::Op::ShlAssign  },
        { TokenKind::OP_SHR_ASSIGN,     BinaryExpr::Op::ShrAssign  },
    };

    for (auto& [kind, op] : kAssignOps)
    {
        if (Check(kind))
        {
            SourceLocation loc = Advance().location;
            ExprPtr right = ParseAssignment(diag);
            auto bin = std::make_unique<BinaryExpr>();
            bin->location = loc;
            bin->op       = op;
            bin->left     = std::move(left);
            bin->right    = std::move(right);
            return bin;
        }
    }

    return left;
}

ExprPtr Parser::ParseTernary(DiagnosticEngine& diag)
{
    ExprPtr cond = ParseOr(diag);
    if (!Check(TokenKind::OP_QUESTION)) return cond;

    SourceLocation loc = Advance().location; // '?'
    ExprPtr then_ = ParseExpression(diag);
    Expect(TokenKind::OP_COLON, "in ternary expression", diag);
    ExprPtr else_ = ParseExpression(diag);

    auto tern = std::make_unique<TernaryExpr>();
    tern->location  = loc;
    tern->condition = std::move(cond);
    tern->thenExpr  = std::move(then_);
    tern->elseExpr  = std::move(else_);
    return tern;
}

// Macro helper for binary left-associative levels
#define PARSE_BINARY_LEVEL(FnName, NextFn, ...)                         \
ExprPtr Parser::FnName(DiagnosticEngine& diag)                          \
{                                                                        \
    ExprPtr left = NextFn(diag);                                         \
    static const std::pair<TokenKind, BinaryExpr::Op> kOps[] = {        \
        __VA_ARGS__                                                       \
    };                                                                   \
    while (!IsAtEnd())                                                   \
    {                                                                    \
        bool matched = false;                                            \
        for (auto& [k, op] : kOps)                                      \
        {                                                                \
            if (Check(k))                                                \
            {                                                            \
                SourceLocation loc = Advance().location;                 \
                ExprPtr right = NextFn(diag);                            \
                auto bin = std::make_unique<BinaryExpr>();               \
                bin->location = loc;                                     \
                bin->op       = op;                                      \
                bin->left     = std::move(left);                         \
                bin->right    = std::move(right);                        \
                left          = std::move(bin);                          \
                matched = true;                                          \
                break;                                                   \
            }                                                            \
        }                                                                \
        if (!matched) break;                                             \
    }                                                                    \
    return left;                                                         \
}

PARSE_BINARY_LEVEL(ParseOr,      ParseAnd,    { TokenKind::OP_OR,      BinaryExpr::Op::Or      })
PARSE_BINARY_LEVEL(ParseAnd,     ParseBitOr,  { TokenKind::OP_AND,     BinaryExpr::Op::And     })
PARSE_BINARY_LEVEL(ParseBitOr,   ParseBitXor, { TokenKind::OP_BIT_OR,  BinaryExpr::Op::BitOr   })
PARSE_BINARY_LEVEL(ParseBitXor,  ParseBitAnd, { TokenKind::OP_BIT_XOR, BinaryExpr::Op::BitXor  })
PARSE_BINARY_LEVEL(ParseBitAnd,  ParseEquality,{ TokenKind::OP_BIT_AND,BinaryExpr::Op::BitAnd  })
PARSE_BINARY_LEVEL(ParseEquality, ParseComparison,
    { TokenKind::OP_EQ,  BinaryExpr::Op::Eq },
    { TokenKind::OP_NEQ, BinaryExpr::Op::Neq })
PARSE_BINARY_LEVEL(ParseComparison, ParseShift,
    { TokenKind::OP_LT, BinaryExpr::Op::Lt },
    { TokenKind::OP_GT, BinaryExpr::Op::Gt },
    { TokenKind::OP_LE, BinaryExpr::Op::Le },
    { TokenKind::OP_GE, BinaryExpr::Op::Ge })
PARSE_BINARY_LEVEL(ParseShift, ParseAddSub,
    { TokenKind::OP_SHL, BinaryExpr::Op::Shl },
    { TokenKind::OP_SHR, BinaryExpr::Op::Shr })
PARSE_BINARY_LEVEL(ParseAddSub, ParseMulDiv,
    { TokenKind::OP_PLUS,  BinaryExpr::Op::Add },
    { TokenKind::OP_MINUS, BinaryExpr::Op::Sub })
PARSE_BINARY_LEVEL(ParseMulDiv, ParseUnary,
    { TokenKind::OP_STAR,    BinaryExpr::Op::Mul },
    { TokenKind::OP_SLASH,   BinaryExpr::Op::Div },
    { TokenKind::OP_PERCENT, BinaryExpr::Op::Mod })

#undef PARSE_BINARY_LEVEL

ExprPtr Parser::ParseUnary(DiagnosticEngine& diag)
{
    if (Check(TokenKind::OP_MINUS) || Check(TokenKind::OP_NOT) || Check(TokenKind::OP_BIT_NOT))
    {
        SourceLocation loc = Current().location;
        TokenKind k = Advance().kind;
        auto u = std::make_unique<UnaryExpr>();
        u->location = loc;
        u->operand  = ParseUnary(diag);
        u->op = (k == TokenKind::OP_MINUS)   ? UnaryExpr::Op::Neg
              : (k == TokenKind::OP_NOT)      ? UnaryExpr::Op::Not
                                               : UnaryExpr::Op::BitNot;
        return u;
    }
    if (Check(TokenKind::OP_INC) || Check(TokenKind::OP_DEC))
    {
        SourceLocation loc = Current().location;
        bool isInc = Check(TokenKind::OP_INC);
        Advance();
        auto u = std::make_unique<UnaryExpr>();
        u->location = loc;
        u->operand  = ParseUnary(diag);
        u->op       = isInc ? UnaryExpr::Op::PreInc : UnaryExpr::Op::PreDec;
        return u;
    }
    return ParsePostfix(diag);
}

ExprPtr Parser::ParsePostfix(DiagnosticEngine& diag)
{
    ExprPtr expr = ParsePrimary(diag);

    while (true)
    {
        if (diag.HasErrors() || !expr) break;

        // ==================== Post-increment / decrement ====================
        if (Check(TokenKind::OP_INC) || Check(TokenKind::OP_DEC))
        {
            auto u = std::make_unique<UnaryExpr>();
            u->location  = Current().location;
            u->op        = Check(TokenKind::OP_INC) ? UnaryExpr::Op::PostInc : UnaryExpr::Op::PostDec;
            u->isPostfix = true;
            u->operand   = std::move(expr);
            Advance();
            expr = std::move(u);
            continue;
        }

        // ==================== Member access (. or ->) ====================
        if (Check(TokenKind::OP_DOT) || Check(TokenKind::OP_ARROW))
        {
            bool isArrow = Check(TokenKind::OP_ARROW);
            Advance(); // '.' or '->'

            if (!Check(TokenKind::IDENTIFIER) && !IsTypeName(Current()) && !Current().IsKeyword())
            {
                diag.ReportError(Current().location, "Expected member name after '.'.");
                break;
            }

            std::string member = Advance().lexeme;

            // Never classify PROPERTY block or PROPERTY.Group field access as a vector swizzle
            // or .Length property (e.g. Params.ColorGroup.r or Params.BufferGroup.Length).
            bool isPropertyChain = false;
            if (const auto* baseId = dynamic_cast<const IdentifierExpr*>(expr.get()))
            {
                if (IsPropertyBlockName(baseId->name))
                    isPropertyChain = true;
            }
            else if (const auto* innerMa = dynamic_cast<const MemberAccessExpr*>(expr.get()))
            {
                if (const auto* blockId = dynamic_cast<const IdentifierExpr*>(innerMa->object.get()))
                {
                    if (IsPropertyBlockName(blockId->name))
                        isPropertyChain = true;
                }
            }

            // Distinguish swizzle vs member access
            if (!isArrow && !isPropertyChain && IsSwizzle(member))
            {
                auto sw = std::make_unique<SwizzleExpr>();
                sw->location = Current().location;
                sw->object   = std::move(expr);
                sw->swizzle  = member;
                expr = std::move(sw);
            }
            else if (!isArrow && !isPropertyChain && member == "Length")
            {
                auto le = std::make_unique<LengthExpr>();
                le->location = Current().location;
                le->object   = std::move(expr);
                expr = std::move(le);
            }
            else
            {
                auto ma = std::make_unique<MemberAccessExpr>();
                ma->location = Current().location;
                ma->object   = std::move(expr);
                ma->member   = member;
                ma->isArrow  = isArrow;

                // Chain further access: a.b(...)
                if (Check(TokenKind::PUNC_LPAREN))
                    expr = ParseCallArgs(std::move(ma), diag);
                else
                    expr = std::move(ma);
            }
            continue;
        }

        // ==================== Index ====================
        if (Check(TokenKind::PUNC_LBRACKET))
        {
            Advance(); // '['
            auto ie = std::make_unique<IndexExpr>();
            ie->location = Current().location;
            ie->object   = std::move(expr);
            ie->index    = ParseExpression(diag);
            Expect(TokenKind::PUNC_RBRACKET, "in index expression", diag);
            expr = std::move(ie);
            continue;
        }

        // ==================== Call ====================
        if (Check(TokenKind::PUNC_LPAREN))
        {
            expr = ParseCallArgs(std::move(expr), diag);
            continue;
        }

        break;
    }

    return expr;
}

ExprPtr Parser::ParseCallArgs(ExprPtr callee, DiagnosticEngine& diag)
{
    SourceLocation loc = Advance().location; // '('
    auto call = std::make_unique<CallExpr>();
    call->location = loc;
    call->callee   = std::move(callee);

    while (!IsAtEnd() && !Check(TokenKind::PUNC_RPAREN))
    {
        if (diag.HasErrors()) return nullptr;
        call->args.push_back(ParseExpression(diag));
        if (!Check(TokenKind::PUNC_RPAREN))
            Expect(TokenKind::PUNC_COMMA, "between function arguments", diag);
    }

    Expect(TokenKind::PUNC_RPAREN, "to close function call", diag);
    return call;
}

ExprPtr Parser::ParsePrimary(DiagnosticEngine& diag)
{
    SourceLocation loc = Current().location;

    // ==================== Parenthesized expression ====================
    if (Check(TokenKind::PUNC_LPAREN))
    {
        Advance();
        ExprPtr inner = ParseExpression(diag);
        Expect(TokenKind::PUNC_RPAREN, "to close parenthesized expression", diag);
        auto paren = std::make_unique<ParenExpr>();
        paren->location = loc;
        paren->inner = std::move(inner);
        return paren;
    }

    // ==================== Initializer List ====================
    if (Check(TokenKind::PUNC_LBRACE))
    {
        Advance(); // '{'
        auto initList = std::make_unique<InitializerListExpr>();
        initList->location = loc;

        while (!IsAtEnd() && !Check(TokenKind::PUNC_RBRACE))
        {
            if (diag.HasErrors()) return nullptr;
            initList->elements.push_back(ParseExpression(diag));
            if (!Check(TokenKind::PUNC_RBRACE))
                Expect(TokenKind::PUNC_COMMA, "between initializer list elements", diag);
        }
        Expect(TokenKind::PUNC_RBRACE, "to close initializer list", diag);
        return initList;
    }

    // ==================== Type constructor / Cast: Vec3(1, 0, 0) ====================
    if (IsTypeName(Current()))
    {
        std::string typeName = Advance().lexeme;

        if (Check(TokenKind::PUNC_LPAREN))
        {
            Advance(); // '('
            auto cast = std::make_unique<CastExpr>();
            cast->location = loc;
            cast->typeName = typeName;

            while (!IsAtEnd() && !Check(TokenKind::PUNC_RPAREN))
            {
                cast->args.push_back(ParseExpression(diag));
                if (!Check(TokenKind::PUNC_RPAREN))
                    Expect(TokenKind::PUNC_COMMA, "between cast arguments", diag);
            }
            Expect(TokenKind::PUNC_RPAREN, "to close cast expression", diag);
            return cast;
        }

        // Just a type name as expression (unusual but handle gracefully)
        auto id = std::make_unique<IdentifierExpr>();
        id->location = loc;
        id->name     = typeName;
        return id;
    }

    // ==================== Identifier ====================
    if (Check(TokenKind::IDENTIFIER))
    {
        auto id = std::make_unique<IdentifierExpr>();
        id->location = loc;
        id->name     = Advance().lexeme;
        return id;
    }

    // ==================== Literals ====================
    if (Check(TokenKind::LIT_INT))
    {
        auto lit = std::make_unique<LiteralExpr>();
        lit->litKind   = LiteralExpr::Kind::Int;
        lit->intValue  = Current().intValue;
        lit->uintValue = Current().uintValue;
        lit->location  = Current().location;
        Advance();
        return lit;
    }
    if (Check(TokenKind::LIT_UINT))
    {
        auto lit = std::make_unique<LiteralExpr>();
        lit->litKind   = LiteralExpr::Kind::UInt;
        lit->uintValue = Current().uintValue;
        lit->intValue  = Current().intValue;
        lit->location  = Current().location;
        Advance();
        return lit;
    }
    if (Check(TokenKind::LIT_FLOAT))
    {
        auto lit = std::make_unique<LiteralExpr>();
        lit->location   = loc;
        lit->litKind    = LiteralExpr::Kind::Float;
        lit->floatValue = Current().floatValue;
        Advance();
        return lit;
    }
    if (Check(TokenKind::LIT_BOOL))
    {
        auto lit = std::make_unique<LiteralExpr>();
        lit->location  = loc;
        lit->litKind   = LiteralExpr::Kind::Bool;
        lit->boolValue = (Current().intValue != 0);
        Advance();
        return lit;
    }
    if (Check(TokenKind::LIT_STRING))
    {
        auto lit = std::make_unique<LiteralExpr>();
        lit->location = loc;
        lit->litKind  = LiteralExpr::Kind::String;
        lit->strValue = Current().lexeme;
        Advance();
        return lit;
    }

    // ==================== Built-in functions used as identifiers ====================
    if (Current().IsKeyword() && !IsTypeName(Current()))
    {
        // Treat built-in function keywords as identifiers in expression context
        auto id = std::make_unique<IdentifierExpr>();
        id->location = loc;
        id->name     = Advance().lexeme;
        return id;
    }

    diag.ReportError(loc,
        "Unexpected token " + Current().KindName() + " in expression.");
    Advance(); // recover
    return nullptr;
}

// =====================================================================
//  Helpers
// =====================================================================

bool Parser::IsTypeName(const Token& tok) const
{
    return tok.IsType() || tok.IsResourceType();
}

bool Parser::IsPropertyGroupName(const std::string& name) const
{
    return m_propertyGroupNames.count(name) > 0;
}

bool Parser::IsPropertyBlockName(const std::string& name) const
{
    return m_propertyBlockNames.count(name) > 0;
}

std::string Parser::ResolveQualifiedGroup(const std::string& blockName, const std::string& groupName) const
{
    if (!m_properties) return "";
    for (const auto& group : m_properties->groups)
    {
        if (group.ownerPropertyName == blockName && group.name == groupName)
            return group.qualifiedName;
    }
    return "";
}

// Handles "Texture2D", "StructuredBuffer<T>", "Mat4", etc.
std::string Parser::ReadTypeName(DiagnosticEngine& diag)
{
    if (!IsTypeName(Current()) && Current().kind != TokenKind::IDENTIFIER)
    {
        diag.ReportError(Current().location,
            "Expected a type name, got " + Current().KindName() + ".");
        return {};
    }

    std::string typeName = Advance().lexeme;

    // Template arg: StructuredBuffer<T>
    if (Check(TokenKind::OP_LT))
    {
        typeName += Advance().lexeme; // '<'
        while (!IsAtEnd() && !Check(TokenKind::OP_GT))
            typeName += Advance().lexeme;
        if (Check(TokenKind::OP_GT))
            typeName += Advance().lexeme; // '>'
    }

    return typeName;
}

// Returns true for known xyzw/rgba swizzle patterns
bool Parser::IsSwizzle(const std::string& m) const
{
    // All characters must be from {x,y,z,w} or {r,g,b,a}
    if (m.empty() || m.size() > 4) return false;
    for (char c : m)
        if (c != 'x' && c != 'y' && c != 'z' && c != 'w' &&
            c != 'r' && c != 'g' && c != 'b' && c != 'a')
            return false;
    return true;
}

// =====================================================================
//  Token stream helpers
// =====================================================================

bool Parser::IsAtEnd() const
{
    return m_pos >= m_tokens->size() ||
           (*m_tokens)[m_pos].kind == TokenKind::END_OF_FILE;
}

const Token& Parser::Current() const { return (*m_tokens)[m_pos]; }

const Token& Parser::Peek(int offset) const
{
    size_t idx = m_pos + static_cast<size_t>(offset);
    if (idx >= m_tokens->size()) return m_tokens->back();
    return (*m_tokens)[idx];
}

const Token& Parser::Advance()
{
    const Token& t = (*m_tokens)[m_pos];
    if (!IsAtEnd()) ++m_pos;
    return t;
}

bool Parser::Check(TokenKind k) const
{
    return !IsAtEnd() && Current().kind == k;
}

bool Parser::Match(TokenKind k)
{
    if (Check(k)) { Advance(); return true; }
    return false;
}

bool Parser::MatchAny(std::initializer_list<TokenKind> kinds)
{
    for (auto k : kinds) if (Match(k)) return true;
    return false;
}

bool Parser::Expect(TokenKind k, const std::string& context, DiagnosticEngine& diag)
{
    if (Check(k)) { Advance(); return true; }
    diag.ReportError(Current().location,
        "Expected " + Token{ k, "", {} }.KindName() + " " + context +
        ", got " + Current().KindName() + ".");
    return false;
}

} // namespace NSL
