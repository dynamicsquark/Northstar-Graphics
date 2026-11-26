#pragma once

#include "nsl_ast_statements.h"
#include <unordered_set>

// =====================================================================
//  NSL AST — Declarations
//  Top-level declarations: structs, functions, global vars, typedefs.
//  AliasDecl stores mandatory fully-qualified PROPERTY aliases.
// =====================================================================

namespace NSL
{
    // ========================= STRUCT FIELDS =========================

    struct StructField
    {
        std::string typeName;
        std::string name;
        std::string semantic;           // NSL: "CLIP_POSITION", "DISPATCH_THREAD_ID", …
        std::string interpolation;      // "nointerpolation", "linear", etc.
        std::string arraySizeExpr;      // "" if not an array, e.g. "256"
    };

    // ========================= TOP-LEVEL DECLARATIONS ================

    struct StructDecl
    {
        std::string             name;
        std::vector<StructField> fields;
        SourceLocation          location;
    };

    // ========================= FUNCTION DECLARATION ==================

    struct FunctionParam
    {
        std::string    typeName;
        std::string    name;
        std::string    arraySizeExpr;
        std::string    semantic;        // NSL: "DISPATCH_THREAD_ID", "DOMAIN_LOCATION", …
        ParamQualifier qualifier        = ParamQualifier::NONE;
    };

    struct FunctionDecl
    {
        ShaderStage               stage = ShaderStage::NONE; // Set for Main()
        std::string               returnType;
        std::string               name;
        std::string               returnSemantic;    // NSL: "COLOR_TARGET", "CLIP_POSITION", …
        std::vector<FunctionParam> parameters;
        std::vector<StmtPtr>      body;
        std::vector<Attribute>    attributes;        // [numthreads(8,8,1)]
        SourceLocation            location;
    };

    // ========================= PREPROCESSOR DIRECTIVE ================

    struct PreprocessorDirective
    {
        std::string line;   // Full text e.g. "#define MAX_LIGHTS 4"
        SourceLocation location;
    };

    // ========================= GLOBAL VARIABLE =======================

    struct GlobalVarDecl
    {
        std::string    typeName;
        std::string    name;
        ExprPtr        initializer;
        std::string    arraySizeExpr;
        bool           isConst       = false;
        bool           isStatic      = false;
        bool           isGroupShared = false;
        SourceLocation location;
    };

    // ========================= TYPEDEF ===============================

    struct TypedefDecl
    {
        std::string    originalType;
        std::string    aliasName;
        SourceLocation location;
    };

    // ========================= ALIAS DECLARATION =====================
    // Represents a fully-qualified alias inside SHADER { }:
    //   Texture2D albedo = Materials.UserTextures.AlbedoMap;
    //
    // The right-hand side is stored as sourceBinding.sourceField.

    struct AliasDecl
    {
        std::string    typeName;        // "Texture2D", "Vec3"
        std::string    aliasName;       // "albedo", "baseCol"
        std::string    sourceBinding;   // qualified group: "Materials__UserTextures"
        std::string    sourceField;     // "AlbedoMap", "baseColor"
        SourceLocation location;
    };

    // ========================= PROGRAM (ROOT NODE) ===================

    struct Program
    {
        ShaderStage stage = ShaderStage::NONE;      // from #shader directive

        std::vector<PreprocessorDirective>  preprocessors;
        std::vector<StructDecl>             structs;
        std::vector<GlobalVarDecl>          globals;
        std::vector<TypedefDecl>            typedefs;
        std::vector<FunctionDecl>           functions;
        std::vector<AliasDecl>              aliases;
        std::unordered_set<std::string>     definedMacros;
    };

} // namespace NSL
