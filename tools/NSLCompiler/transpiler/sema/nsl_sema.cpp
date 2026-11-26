#include "nsl_sema.h"
#include <cstdint>
#include <initializer_list>

namespace NSL
{

// =====================================================================
//  Analyze — entry point
// =====================================================================

bool SemanticAnalyzer::Analyze(
    const Program&          program,
    const ParsedProperties& properties,
    DiagnosticEngine&       diag)
{
    // Only PROPERTY block and qualified group names are valid. Short group
    // names are deliberately excluded.
    m_knownNames.clear();
    m_structuredAliases.clear();
    m_getAliases.clear();
    m_readOnlyResourceAliases.clear();
    for (const auto& block : properties.blocks)
        m_knownNames.insert(block.name);

    for (const auto& g : properties.groups)
        m_knownNames.insert(g.qualifiedName);

    // Pass 0: validate property block names
    ValidatePropertyBlockNames(properties, diag);
    if (diag.HasErrors()) return false;

    // Pass 1: validate alias declarations
    ValidateAliases(program, properties, diag);
    if (diag.HasErrors()) return false;

    // Pass 2: enforce Property.Resource.Data (or alias) access.
    ValidatePropertyAccessSyntax(program, properties, diag);
    if (diag.HasErrors()) return false;

    // Pass 3: validate function declarations
    ValidateFunctions(program, diag);
    if (diag.HasErrors()) return false;

    // Pass 4: warn about unused PROPERTY groups
    ValidatePropertyUsage(program, properties, diag);

    return !diag.HasErrors();
}

// =====================================================================
//  Pass 1 — Alias Declarations
//  "Texture2D albedo = Materials.UserTextures.AlbedoMap;"
// =====================================================================

void SemanticAnalyzer::ValidateAliases(
    const Program&          program,
    const ParsedProperties& properties,
    DiagnosticEngine&       diag)
{
    std::unordered_set<std::string> aliasNames;

    for (const auto& alias : program.aliases)
    {
        // Alias name must not conflict
        if (m_knownNames.count(alias.aliasName))
        {
            diag.ReportError(alias.location,
                "Alias name '" + alias.aliasName +
                "' conflicts with an existing PROPERTY or alias name.");
            continue;
        }
        if (aliasNames.count(alias.aliasName))
        {
            diag.ReportError(alias.location,
                "Duplicate alias name '" + alias.aliasName + "'.");
            continue;
        }

        // Alias sources must already be qualified as Property__Resource.
        auto sourceIt = properties.byQualifiedName.find(alias.sourceBinding);
        if (sourceIt == properties.byQualifiedName.end())
        {
            diag.ReportError(alias.location,
                "Alias source '" + alias.sourceBinding +
                "' is not fully qualified. Use 'Property.Resource.Data'.");
            continue;
        }

        const PropertyGroup* grp = sourceIt->second;

        // sourceField must exist in that group
        if (!GroupHasField(*grp, alias.sourceField))
        {
            diag.ReportError(alias.location,
                "PROPERTY group '" + alias.sourceBinding +
                "' has no field named '" + alias.sourceField + "'.");
            continue;
        }

        if (grp->resourceKind == ResourceKind::STRUCTURED_BUFFER ||
            grp->resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
        {
            m_structuredAliases.insert(alias.aliasName);
        }

        for (const auto& entry : grp->entries)
        {
            if (entry.name == alias.sourceField)
            {
                if (entry.kind == PropertyEntryKind::GET)
                    m_getAliases.insert(alias.aliasName);
                break;
            }
        }
        if (grp->resourceKind != ResourceKind::RW_STRUCTURED_BUFFER &&
            grp->resourceKind != ResourceKind::RW_TEXTURE_2D)
        {
            m_readOnlyResourceAliases.insert(alias.aliasName);
        }

        aliasNames.insert(alias.aliasName);
        m_knownNames.insert(alias.aliasName);
    }
}

// =====================================================================
//  Pass 3 — Mandatory qualified PROPERTY access & read-only enforcement
// =====================================================================

void SemanticAnalyzer::CheckReadOnlyMutationTarget(
    const Expression*                      target,
    const ParsedProperties&                properties,
    const std::unordered_set<std::string>& localScope,
    DiagnosticEngine&                      diag,
    const SourceLocation&                  loc) const
{
    const Expression* cur = target;
    while (cur)
    {
        if (const auto* pe = dynamic_cast<const ParenExpr*>(cur))
        {
            cur = pe->inner.get();
            continue;
        }
        if (const auto* sw = dynamic_cast<const SwizzleExpr*>(cur))
        {
            cur = sw->object.get();
            continue;
        }
        if (const auto* ie = dynamic_cast<const IndexExpr*>(cur))
        {
            cur = ie->object.get();
            continue;
        }
        if (const auto* ma = dynamic_cast<const MemberAccessExpr*>(cur))
        {
            if (const auto* groupAccess = dynamic_cast<const MemberAccessExpr*>(ma->object.get()))
            {
                if (const auto* propertyId = dynamic_cast<const IdentifierExpr*>(groupAccess->object.get()))
                {
                    const std::string qualified = propertyId->name + "__" + groupAccess->member;
                    auto groupIt = properties.byQualifiedName.find(qualified);
                    if (groupIt != properties.byQualifiedName.end())
                    {
                        const PropertyGroup* grp = groupIt->second;
                        for (const auto& entry : grp->entries)
                        {
                            if (entry.name == ma->member && entry.kind == PropertyEntryKind::GET)
                            {
                                diag.ReportError(loc,
                                    "Cannot modify read-only engine Get() property '" +
                                    propertyId->name + "." + groupAccess->member + "." + ma->member + "'.");
                                return;
                            }
                        }
                        if (grp->resourceKind != ResourceKind::RW_STRUCTURED_BUFFER &&
                            grp->resourceKind != ResourceKind::RW_TEXTURE_2D)
                        {
                            diag.ReportError(loc,
                                "Cannot modify read-only PROPERTY resource '" +
                                propertyId->name + "." + groupAccess->member + "." + ma->member + "'.");
                        }
                        return;
                    }
                }
            }
            if (const auto* blockId = dynamic_cast<const IdentifierExpr*>(ma->object.get()))
            {
                const std::string qualified = blockId->name + "__" + ma->member;
                auto groupIt = properties.byQualifiedName.find(qualified);
                if (groupIt != properties.byQualifiedName.end())
                {
                    const PropertyGroup* grp = groupIt->second;
                    if (grp->resourceKind != ResourceKind::RW_STRUCTURED_BUFFER &&
                        grp->resourceKind != ResourceKind::RW_TEXTURE_2D)
                    {
                        diag.ReportError(loc,
                            "Cannot modify read-only PROPERTY resource '" +
                            blockId->name + "." + ma->member + "'.");
                    }
                    return;
                }
            }
            cur = ma->object.get();
            continue;
        }
        if (const auto* id = dynamic_cast<const IdentifierExpr*>(cur))
        {
            if (!localScope.count(id->name))
            {
                if (m_getAliases.count(id->name))
                {
                    diag.ReportError(loc,
                        "Cannot modify read-only engine Get() property alias '" + id->name + "'.");
                }
                else if (m_readOnlyResourceAliases.count(id->name))
                {
                    diag.ReportError(loc,
                        "Cannot modify read-only PROPERTY resource alias '" + id->name + "'.");
                }
            }
            return;
        }
        break;
    }
}

void SemanticAnalyzer::ValidatePropertyAccessSyntax(
    const Program&          program,
    const ParsedProperties& properties,
    DiagnosticEngine&       diag) const
{
    std::unordered_set<std::string> emptyScope;
    for (const auto& global : program.globals)
        ValidateAccessInExpr(global.initializer.get(), properties, program, emptyScope, diag);

    for (const auto& fn : program.functions)
    {
        std::unordered_set<std::string> localScope;
        for (const auto& g : program.globals)
            localScope.insert(g.name);
        for (const auto& p : fn.parameters)
            localScope.insert(p.name);
        for (const auto& stmt : fn.body)
            ValidateAccessInStmt(stmt.get(), properties, program, localScope, diag);
    }
}

void SemanticAnalyzer::ValidateAccessInStmt(
    const Statement*                 stmt,
    const ParsedProperties&          properties,
    const Program&                   program,
    std::unordered_set<std::string>& localScope,
    DiagnosticEngine&                diag) const
{
    if (!stmt) return;

    if (auto* es = dynamic_cast<const ExprStmt*>(stmt))
        ValidateAccessInExpr(es->expr.get(), properties, program, localScope, diag);
    else if (auto* rs = dynamic_cast<const ReturnStmt*>(stmt))
        ValidateAccessInExpr(rs->value.get(), properties, program, localScope, diag);
    else if (auto* vd = dynamic_cast<const VarDeclStmt*>(stmt))
    {
        ValidateAccessInExpr(vd->initializer.get(), properties, program, localScope, diag);
        localScope.insert(vd->name);
    }
    else if (auto* block = dynamic_cast<const BlockStmt*>(stmt))
    {
        std::unordered_set<std::string> innerScope = localScope;
        for (const auto& child : block->statements)
            ValidateAccessInStmt(child.get(), properties, program, innerScope, diag);
    }
    else if (auto* ifs = dynamic_cast<const IfStmt*>(stmt))
    {
        ValidateAccessInExpr(ifs->condition.get(), properties, program, localScope, diag);
        std::unordered_set<std::string> thenScope = localScope;
        ValidateAccessInStmt(ifs->thenBranch.get(), properties, program, thenScope, diag);
        std::unordered_set<std::string> elseScope = localScope;
        ValidateAccessInStmt(ifs->elseBranch.get(), properties, program, elseScope, diag);
    }
    else if (auto* fs = dynamic_cast<const ForStmt*>(stmt))
    {
        std::unordered_set<std::string> forScope = localScope;
        ValidateAccessInStmt(fs->init.get(), properties, program, forScope, diag);
        ValidateAccessInExpr(fs->condition.get(), properties, program, forScope, diag);
        ValidateAccessInExpr(fs->increment.get(), properties, program, forScope, diag);
        ValidateAccessInStmt(fs->body.get(), properties, program, forScope, diag);
    }
    else if (auto* ws = dynamic_cast<const WhileStmt*>(stmt))
    {
        ValidateAccessInExpr(ws->condition.get(), properties, program, localScope, diag);
        std::unordered_set<std::string> bodyScope = localScope;
        ValidateAccessInStmt(ws->body.get(), properties, program, bodyScope, diag);
    }
    else if (auto* dws = dynamic_cast<const DoWhileStmt*>(stmt))
    {
        std::unordered_set<std::string> bodyScope = localScope;
        ValidateAccessInStmt(dws->body.get(), properties, program, bodyScope, diag);
        ValidateAccessInExpr(dws->condition.get(), properties, program, localScope, diag);
    }
    else if (auto* sw = dynamic_cast<const SwitchStmt*>(stmt))
    {
        ValidateAccessInExpr(sw->condition.get(), properties, program, localScope, diag);
        for (const auto& sc : sw->cases)
        {
            ValidateAccessInExpr(sc.value.get(), properties, program, localScope, diag);
            std::unordered_set<std::string> caseScope = localScope;
            for (const auto& child : sc.statements)
                ValidateAccessInStmt(child.get(), properties, program, caseScope, diag);
        }
    }
}

void SemanticAnalyzer::ValidateAccessInExpr(
    const Expression*                      expr,
    const ParsedProperties&                properties,
    const Program&                         program,
    const std::unordered_set<std::string>& localScope,
    DiagnosticEngine&                      diag,
    bool                                   isIndexTarget) const
{
    if (!expr) return;

    if (auto* id = dynamic_cast<const IdentifierExpr*>(expr))
    {
        if (!isIndexTarget && !localScope.count(id->name) && m_structuredAliases.count(id->name))
        {
            diag.ReportError(id->location,
                "StructuredBuffer alias '" + id->name +
                "' requires an index: use '" + id->name + "[index]'.");
        }
        return;
    }

    if (auto* ie = dynamic_cast<const IndexExpr*>(expr))
    {
        ValidateAccessInExpr(ie->object.get(), properties, program, localScope, diag, true);
        ValidateAccessInExpr(ie->index.get(), properties, program, localScope, diag);
        return;
    }

    if (auto* ma = dynamic_cast<const MemberAccessExpr*>(expr))
    {
        // Property.Resource.Data
        if (auto* groupAccess =
                dynamic_cast<const MemberAccessExpr*>(ma->object.get()))
        {
            if (auto* propertyId =
                    dynamic_cast<const IdentifierExpr*>(groupAccess->object.get()))
            {
                const std::string qualified =
                    propertyId->name + "__" + groupAccess->member;
                auto groupIt = properties.byQualifiedName.find(qualified);
                if (groupIt != properties.byQualifiedName.end())
                {
                    const PropertyGroup* group = groupIt->second;
                    if (!GroupHasField(*group, ma->member))
                    {
                        diag.ReportError(ma->location,
                            "PROPERTY resource '" + propertyId->name + "." +
                            groupAccess->member + "' has no data named '" +
                            ma->member + "'.");
                    }
                    if ((group->resourceKind ==
                             ResourceKind::STRUCTURED_BUFFER ||
                         group->resourceKind ==
                             ResourceKind::RW_STRUCTURED_BUFFER) &&
                        !isIndexTarget)
                    {
                        diag.ReportError(ma->location,
                            "StructuredBuffer field '" + propertyId->name + "." +
                            groupAccess->member + "." + ma->member +
                            "' requires an index: use '...Data[index]'.");
                    }
                    return;
                }
            }
        }

        // Reject legacy Group.Data direct access.
        if (auto* id = dynamic_cast<const IdentifierExpr*>(ma->object.get()))
        {
            if (!localScope.count(id->name) && properties.FindByName(id->name))
            {
                diag.ReportError(ma->location,
                    "Short PROPERTY access '" + id->name + "." + ma->member +
                    "' is not allowed. Use 'Property.Resource.Data' or an alias.");
                return;
            }
        }

        ValidateAccessInExpr(ma->object.get(), properties, program, localScope, diag);
        return;
    }

    if (auto* be = dynamic_cast<const BinaryExpr*>(expr))
    {
        if (be->op >= BinaryExpr::Op::Assign && be->op <= BinaryExpr::Op::ShrAssign)
        {
            CheckReadOnlyMutationTarget(be->left.get(), properties, localScope, diag, be->location);
        }
        ValidateAccessInExpr(be->left.get(), properties, program, localScope, diag);
        ValidateAccessInExpr(be->right.get(), properties, program, localScope, diag);
    }
    else if (auto* ce = dynamic_cast<const CallExpr*>(expr))
    {
        if (const auto* calleeId = dynamic_cast<const IdentifierExpr*>(ce->callee.get()))
        {
            const std::string& fnName = calleeId->name;
            if (fnName == "Sample" && (ce->args.size() < 3 || ce->args.size() > 5))
            {
                diag.ReportError(ce->location,
                    "Sample() expects 3 to 5 arguments (tex, sampler, uv [, offset [, clamp]]), got " +
                    std::to_string(ce->args.size()) + ".");
            }
            else if (fnName == "SampleLevel" && (ce->args.size() < 4 || ce->args.size() > 5))
            {
                diag.ReportError(ce->location,
                    "SampleLevel() expects 4 or 5 arguments (tex, sampler, uv, lod [, offset]), got " +
                    std::to_string(ce->args.size()) + ".");
            }
            else if (fnName == "SampleCmp" && (ce->args.size() < 4 || ce->args.size() > 6))
            {
                diag.ReportError(ce->location,
                    "SampleCmp() expects 4 to 6 arguments (tex, cmpSampler, uv, cmpValue [, offset [, clamp]]), got " +
                    std::to_string(ce->args.size()) + ".");
            }
            else if (fnName == "SampleCmpLevelZero" && (ce->args.size() < 4 || ce->args.size() > 5))
            {
                diag.ReportError(ce->location,
                    "SampleCmpLevelZero() expects 4 or 5 arguments (tex, cmpSampler, uv, cmpValue [, offset]), got " +
                    std::to_string(ce->args.size()) + ".");
            }
            else if (fnName.rfind("Interlocked", 0) == 0 && !ce->args.empty())
            {
                CheckReadOnlyMutationTarget(ce->args[0].get(), properties, localScope, diag, ce->location);
            }
            else
            {
                for (const auto& decl : program.functions)
                {
                    if (decl.name == fnName && decl.parameters.size() == ce->args.size())
                    {
                        for (size_t i = 0; i < decl.parameters.size(); ++i)
                        {
                            if (decl.parameters[i].qualifier == ParamQualifier::OUT ||
                                decl.parameters[i].qualifier == ParamQualifier::INOUT)
                            {
                                CheckReadOnlyMutationTarget(ce->args[i].get(), properties, localScope, diag, ce->location);
                            }
                        }
                    }
                }
            }
        }
        ValidateAccessInExpr(ce->callee.get(), properties, program, localScope, diag);
        for (const auto& arg : ce->args)
            ValidateAccessInExpr(arg.get(), properties, program, localScope, diag);
    }
    else if (auto* sw = dynamic_cast<const SwizzleExpr*>(expr))
        ValidateAccessInExpr(sw->object.get(), properties, program, localScope, diag);
    else if (auto* le = dynamic_cast<const LengthExpr*>(expr))
        ValidateAccessInExpr(le->object.get(), properties, program, localScope, diag);
    else if (auto* ue = dynamic_cast<const UnaryExpr*>(expr))
    {
        if (ue->op == UnaryExpr::Op::PreInc || ue->op == UnaryExpr::Op::PreDec ||
            ue->op == UnaryExpr::Op::PostInc || ue->op == UnaryExpr::Op::PostDec)
        {
            CheckReadOnlyMutationTarget(ue->operand.get(), properties, localScope, diag, ue->location);
        }
        ValidateAccessInExpr(ue->operand.get(), properties, program, localScope, diag);
    }
    else if (auto* pe = dynamic_cast<const ParenExpr*>(expr))
        ValidateAccessInExpr(pe->inner.get(), properties, program, localScope, diag);
    else if (auto* te = dynamic_cast<const TernaryExpr*>(expr))
    {
        ValidateAccessInExpr(te->condition.get(), properties, program, localScope, diag);
        ValidateAccessInExpr(te->thenExpr.get(), properties, program, localScope, diag);
        ValidateAccessInExpr(te->elseExpr.get(), properties, program, localScope, diag);
    }
    else if (auto* cast = dynamic_cast<const CastExpr*>(expr))
    {
        for (const auto& arg : cast->args)
            ValidateAccessInExpr(arg.get(), properties, program, localScope, diag);
    }
    else if (auto* init = dynamic_cast<const InitializerListExpr*>(expr))
    {
        for (const auto& element : init->elements)
            ValidateAccessInExpr(element.get(), properties, program, localScope, diag);
    }
}

// =====================================================================
//  Pass 4 — Function Declarations
// =====================================================================

static const Attribute* FindAttribute(const FunctionDecl& fn, const std::string& name)
{
    for (const auto& attr : fn.attributes)
        if (attr.name == name) return &attr;
    return nullptr;
}

static std::string AttributeValue(const Attribute& attr)
{
    const size_t open = attr.args.find('(');
    const size_t close = attr.args.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1)
        return {};

    std::string value = attr.args.substr(open + 1, close - open - 1);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);
    return value;
}

static bool HasQuotedAttributeValue(const Attribute& attr)
{
    const size_t open = attr.args.find('(');
    const size_t close = attr.args.rfind(')');
    if (open == std::string::npos || close == std::string::npos ||
        close <= open + 2)
        return false;
    return attr.args[open + 1] == '"' && attr.args[close - 1] == '"';
}

static bool ParsePositiveUInt(const std::string& text, uint32_t& value)
{
    if (text.empty()) return false;
    uint64_t parsed = 0;
    for (char c : text)
    {
        if (c < '0' || c > '9') return false;
        parsed = parsed * 10u + static_cast<uint32_t>(c - '0');
        if (parsed > UINT32_MAX) return false;
    }
    if (parsed == 0) return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

struct GenericType
{
    std::string base;
    std::vector<std::string> args;
    bool valid = false;
};

static GenericType ParseGenericType(const std::string& typeName)
{
    GenericType result;
    const size_t open = typeName.find('<');
    const size_t close = typeName.rfind('>');
    if (open == std::string::npos || close == std::string::npos ||
        close != typeName.size() - 1 || close <= open + 1)
        return result;

    result.base = typeName.substr(0, open);
    const std::string body = typeName.substr(open + 1, close - open - 1);
    size_t start = 0;
    while (start <= body.size())
    {
        const size_t comma = body.find(',', start);
        const size_t end = comma == std::string::npos ? body.size() : comma;
        if (end == start) return GenericType{};
        result.args.push_back(body.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    result.valid = true;
    return result;
}

static bool IsOneOf(
    const std::string& value,
    std::initializer_list<const char*> allowed)
{
    for (const char* candidate : allowed)
        if (value == candidate) return true;
    return false;
}

static bool ValidateRequiredAttribute(
    const FunctionDecl& fn,
    const char* name,
    DiagnosticEngine& diag,
    const Attribute*& out)
{
    out = FindAttribute(fn, name);
    if (out) return true;
    diag.ReportError(fn.location,
        "Shader entry '" + fn.name + "' requires [" + name + "(...)].");
    return false;
}

static bool ValidatePatchType(
    const FunctionParam& param,
    const char* expectedBase,
    uint32_t& pointCount,
    DiagnosticEngine& diag,
    const SourceLocation& location)
{
    const GenericType patch = ParseGenericType(param.typeName);
    if (!patch.valid || patch.base != expectedBase || patch.args.size() != 2 ||
        patch.args[0] != "STAGE_INPUT" ||
        !ParsePositiveUInt(patch.args[1], pointCount))
    {
        diag.ReportError(location,
            "Parameter '" + param.name + "' must be " + expectedBase +
            "<STAGE_INPUT,N> with a positive integer N.");
        return false;
    }
    return true;
}

static bool IsAllowedEntryAttribute(ShaderStage stage, const std::string& name)
{
    switch (stage)
    {
        case ShaderStage::VERTEX:
        case ShaderStage::PIXEL:
            return false;
        case ShaderStage::COMPUTE:
            return name == "numthreads";
        case ShaderStage::GEOMETRY:
            return name == "maxvertexcount" ||
                   name == "inputtopology" ||
                   name == "outputtopology";
        case ShaderStage::HULL:
            return name == "domain" ||
                   name == "partitioning" ||
                   name == "outputtopology" ||
                   name == "outputcontrolpoints" ||
                   name == "patchconstantfunc";
        case ShaderStage::DOMAIN:
            return name == "domain";
        default:
            return false;
    }
}

static void ValidateEntryAttributes(
    ShaderStage stage,
    const FunctionDecl& main,
    DiagnosticEngine& diag)
{
    std::unordered_set<std::string> seen;
    for (const auto& attr : main.attributes)
    {
        if (!seen.insert(attr.name).second)
            diag.ReportError(main.location,
                "Duplicate entry-point attribute [" + attr.name + "].");
        if (!IsAllowedEntryAttribute(stage, attr.name))
            diag.ReportError(main.location,
                "Attribute [" + attr.name + "] is not valid for a " +
                std::string(ShaderStageToString(stage)) + " entry point.");
    }
}

void SemanticAnalyzer::ValidateFunctions(
    const Program&    program,
    DiagnosticEngine& diag)
{
    std::unordered_map<std::string, SourceLocation> funcSignatures;
    const FunctionDecl* mainFn = nullptr;
    const FunctionDecl* patchConstantFn = nullptr;

    for (const auto& fn : program.functions)
    {
        // Entry points Main and PatchConstant cannot be overloaded; helper functions may overload by parameter types.
        std::string sigKey = fn.name;
        if (fn.name != "Main" && fn.name != "PatchConstant")
        {
            sigKey += "(";
            for (size_t pIdx = 0; pIdx < fn.parameters.size(); ++pIdx)
            {
                if (pIdx > 0) sigKey += ",";
                sigKey += fn.parameters[pIdx].typeName;
                if (!fn.parameters[pIdx].arraySizeExpr.empty())
                    sigKey += "[" + fn.parameters[pIdx].arraySizeExpr + "]";
            }
            sigKey += ")";
        }

        auto it = funcSignatures.find(sigKey);
        if (it != funcSignatures.end())
        {
            diag.ReportError(fn.location,
                "Duplicate function definition '" + sigKey + "'. "
                "Previously defined at " + it->second.ToString() + ".");
            continue;
        }
        funcSignatures[sigKey] = fn.location;

        if (fn.name == "Main") mainFn = &fn;
        if (fn.name == "PatchConstant") patchConstantFn = &fn;
    }

    if (!mainFn)
    {
        diag.ReportError(0, 0,
            "Shader requires exactly one entry point named 'Main'.");
        return;
    }

    const FunctionDecl& main = *mainFn;
    ValidateEntryAttributes(program.stage, main, diag);

    // The NSL layout model deliberately standardizes V/P entry points.
    if (program.stage == ShaderStage::VERTEX ||
        program.stage == ShaderStage::PIXEL)
    {
        bool isEmptyPixelOutput = false;
        if (program.stage == ShaderStage::PIXEL)
        {
            for (const auto& s : program.structs)
            {
                if (s.name == "STAGE_OUTPUT" && s.fields.empty())
                {
                    isEmptyPixelOutput = true;
                    break;
                }
            }
        }
        const bool validReturn =
            (main.returnType == "STAGE_OUTPUT") ||
            (isEmptyPixelOutput && (main.returnType == "void" || main.returnType == "Void"));
        if (!validReturn)
            diag.ReportError(main.location,
                std::string(ShaderStageToString(program.stage)) +
                " Main() must return STAGE_OUTPUT.");
        if (main.parameters.size() != 1 ||
            main.parameters[0].typeName != "STAGE_INPUT" ||
            !main.parameters[0].arraySizeExpr.empty())
            diag.ReportError(main.location,
                std::string(ShaderStageToString(program.stage)) +
                " Main() must take exactly one STAGE_INPUT parameter.");
        if (!main.returnSemantic.empty())
            diag.ReportError(main.location,
                "STAGE_OUTPUT already owns output semantics; Main() must not "
                "declare a return semantic.");
        return;
    }

    if (program.stage == ShaderStage::COMPUTE)
    {
        if (main.returnType != "void" && main.returnType != "Void")
            diag.ReportError(main.location, "Compute Main() must return void.");
        if (!main.returnSemantic.empty())
            diag.ReportError(main.location,
                "Compute Main() must not declare a return semantic.");

        const Attribute* numthreads = FindAttribute(main, "numthreads");
        if (!numthreads)
        {
            diag.ReportError(main.location,
                "Compute Main() requires [numthreads(x,y,z)].");
        }
        else
        {
            const std::string value = AttributeValue(*numthreads);
            std::vector<std::string> dimensions;
            size_t start = 0;
            while (start <= value.size())
            {
                const size_t comma = value.find(',', start);
                const size_t end =
                    comma == std::string::npos ? value.size() : comma;
                dimensions.push_back(value.substr(start, end - start));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (dimensions.size() != 3 ||
                dimensions[0].empty() ||
                dimensions[1].empty() ||
                dimensions[2].empty())
            {
                diag.ReportError(main.location,
                    "[numthreads] requires exactly three arguments.");
            }
            else
            {
                uint32_t x = 0, y = 0, z = 0;
                const bool allLiteral =
                    ParsePositiveUInt(dimensions[0], x) &&
                    ParsePositiveUInt(dimensions[1], y) &&
                    ParsePositiveUInt(dimensions[2], z);
                if (allLiteral && (x == 0 || y == 0 || z == 0))
                    diag.ReportError(main.location,
                        "[numthreads] dimensions must be positive integers.");
                // Target capability limits (e.g. SM5 product caps) belong to
                // the backend / offline verifier — not the agnostic frontend.
            }
        }

        std::unordered_set<std::string> semantics;
        for (const auto& p : main.parameters)
        {
            const bool uint3Semantic =
                p.semantic == "DISPATCH_THREAD_ID" ||
                p.semantic == "GROUP_ID" ||
                p.semantic == "GROUP_THREAD_ID";
            const bool uintSemantic = p.semantic == "GROUP_INDEX";
            if ((!uint3Semantic && !uintSemantic) ||
                (uint3Semantic && p.typeName != "UInt3") ||
                (uintSemantic && p.typeName != "UInt"))
                diag.ReportError(main.location,
                    "Compute parameter '" + p.name +
                    "' has an invalid type/semantic pair.");
            if (!semantics.insert(p.semantic).second)
                diag.ReportError(main.location,
                    "Duplicate compute system-value semantic '" +
                    p.semantic + "'.");
        }
        return;
    }

    if (program.stage == ShaderStage::GEOMETRY)
    {
        const Attribute* maxVertices = nullptr;
        const Attribute* inputTopology = nullptr;
        const Attribute* outputTopology = nullptr;
        ValidateRequiredAttribute(main, "maxvertexcount", diag, maxVertices);
        ValidateRequiredAttribute(main, "inputtopology", diag, inputTopology);
        ValidateRequiredAttribute(main, "outputtopology", diag, outputTopology);

        if (main.returnType != "void" && main.returnType != "Void")
            diag.ReportError(main.location, "Geometry Main() must return void.");
        if (!main.returnSemantic.empty())
            diag.ReportError(main.location,
                "Geometry Main() must not declare a return semantic.");
        if (main.parameters.size() < 2 || main.parameters.size() > 4)
            diag.ReportError(main.location,
                "Geometry Main() must take a primitive array, an output stream, "
                "and at most the optional PRIMITIVE_ID/GS_INSTANCE_ID parameters.");

        uint32_t maxVertexCount = 0;
        if (maxVertices &&
            !ParsePositiveUInt(AttributeValue(*maxVertices), maxVertexCount))
            diag.ReportError(main.location,
                "[maxvertexcount] requires a positive integer literal.");

        std::string inputTopo;
        uint32_t expectedInputCount = 0;
        if (inputTopology)
        {
            if (!HasQuotedAttributeValue(*inputTopology))
                diag.ReportError(main.location,
                    "Geometry inputtopology value must be a string literal.");
            inputTopo = AttributeValue(*inputTopology);
            if (inputTopo == "point") expectedInputCount = 1;
            else if (inputTopo == "line") expectedInputCount = 2;
            else if (inputTopo == "triangle") expectedInputCount = 3;
            else if (inputTopo == "lineadj") expectedInputCount = 4;
            else if (inputTopo == "triangleadj") expectedInputCount = 6;
            else
                diag.ReportError(main.location,
                    "Geometry inputtopology must be point, line, triangle, "
                    "lineadj, or triangleadj.");
        }

        std::string outputTopo;
        std::string expectedStream;
        if (outputTopology)
        {
            if (!HasQuotedAttributeValue(*outputTopology))
                diag.ReportError(main.location,
                    "Geometry outputtopology value must be a string literal.");
            outputTopo = AttributeValue(*outputTopology);
            if (outputTopo == "point") expectedStream = "PointStream";
            else if (outputTopo == "line") expectedStream = "LineStream";
            else if (outputTopo == "triangle") expectedStream = "TriangleStream";
            else
                diag.ReportError(main.location,
                    "Geometry outputtopology must be point, line, or triangle.");
        }

        if (main.parameters.size() >= 1)
        {
            const auto& input = main.parameters[0];
            uint32_t inputCount = 0;
            if (input.typeName != "STAGE_INPUT" ||
                !ParsePositiveUInt(input.arraySizeExpr, inputCount))
                diag.ReportError(main.location,
                    "Geometry first parameter must be STAGE_INPUT input[N].");
            else if (expectedInputCount != 0 && inputCount != expectedInputCount)
                diag.ReportError(main.location,
                    "Geometry input array size does not match inputtopology.");
        }

        if (main.parameters.size() >= 2)
        {
            const auto& stream = main.parameters[1];
            const GenericType streamType = ParseGenericType(stream.typeName);
            if (!streamType.valid || streamType.base != expectedStream ||
                streamType.args.size() != 1 ||
                streamType.args[0] != "STAGE_OUTPUT")
                diag.ReportError(main.location,
                    "Geometry output stream type does not match outputtopology "
                    "or is not parameterized with STAGE_OUTPUT.");
            if (stream.qualifier != ParamQualifier::INOUT)
                diag.ReportError(main.location,
                    "Geometry output stream parameter must be declared inout.");
        }

        std::unordered_set<std::string> geometrySystemValues;
        for (size_t i = 2; i < main.parameters.size(); ++i)
        {
            const auto& p = main.parameters[i];
            if (p.typeName != "UInt" ||
                (p.semantic != "PRIMITIVE_ID" &&
                 p.semantic != "GS_INSTANCE_ID"))
                diag.ReportError(main.location,
                    "Optional Geometry parameters must be UInt with "
                    "PRIMITIVE_ID or GS_INSTANCE_ID semantic.");
            if (!geometrySystemValues.insert(p.semantic).second)
                diag.ReportError(main.location,
                    "Duplicate Geometry system-value semantic '" +
                    p.semantic + "'.");
        }
        return;
    }

    if (program.stage == ShaderStage::HULL)
    {
        const Attribute* domain = nullptr;
        const Attribute* partitioning = nullptr;
        const Attribute* outputTopology = nullptr;
        const Attribute* outputControlPoints = nullptr;
        ValidateRequiredAttribute(main, "domain", diag, domain);
        ValidateRequiredAttribute(main, "partitioning", diag, partitioning);
        ValidateRequiredAttribute(main, "outputtopology", diag, outputTopology);
        ValidateRequiredAttribute(main, "outputcontrolpoints", diag, outputControlPoints);

        if (main.returnType != "STAGE_OUTPUT")
            diag.ReportError(main.location, "Hull Main() must return STAGE_OUTPUT.");
        if (!main.returnSemantic.empty())
            diag.ReportError(main.location,
                "Hull STAGE_OUTPUT Main() must not declare a return semantic.");

        if (domain)
        {
            if (!HasQuotedAttributeValue(*domain))
                diag.ReportError(main.location,
                    "Hull domain value must be a string literal.");
            if (!IsOneOf(AttributeValue(*domain), {"tri", "quad", "isoline"}))
                diag.ReportError(main.location,
                    "Hull domain must be tri, quad, or isoline.");
        }
        if (partitioning)
        {
            if (!HasQuotedAttributeValue(*partitioning))
                diag.ReportError(main.location,
                    "Hull partitioning value must be a string literal.");
            if (!IsOneOf(AttributeValue(*partitioning),
                {"integer", "pow2", "fractional_even", "fractional_odd"}))
                diag.ReportError(main.location,
                    "Hull partitioning has an unsupported value.");
        }
        if (outputTopology)
        {
            if (!HasQuotedAttributeValue(*outputTopology))
                diag.ReportError(main.location,
                    "Hull outputtopology value must be a string literal.");
            if (!IsOneOf(AttributeValue(*outputTopology),
                {"point", "line", "triangle_cw", "triangle_ccw"}))
                diag.ReportError(main.location,
                    "Hull outputtopology has an unsupported value.");
        }

        const Attribute* patchConstantAttr =
            FindAttribute(main, "patchconstantfunc");
        if (patchConstantAttr &&
            (!HasQuotedAttributeValue(*patchConstantAttr) ||
             AttributeValue(*patchConstantAttr) != "PatchConstant"))
            diag.ReportError(main.location,
                "Hull patchconstantfunc must name the fixed string "
                "\"PatchConstant\".");

        uint32_t outputCount = 0;
        if (outputControlPoints &&
            (!ParsePositiveUInt(AttributeValue(*outputControlPoints), outputCount) ||
             outputCount > 32))
            diag.ReportError(main.location,
                "Hull outputcontrolpoints must be an integer from 1 to 32.");

        uint32_t mainPatchCount = 0;
        if (main.parameters.size() != 2)
        {
            diag.ReportError(main.location,
                "Hull Main() must take InputPatch<STAGE_INPUT,N> and "
                "UInt id : CONTROL_POINT_ID.");
        }
        else
        {
            ValidatePatchType(main.parameters[0], "InputPatch",
                mainPatchCount, diag, main.location);
            const auto& id = main.parameters[1];
            if (id.typeName != "UInt" ||
                id.semantic != "CONTROL_POINT_ID")
                diag.ReportError(main.location,
                    "Hull control-point id must be UInt with "
                    "CONTROL_POINT_ID semantic.");
            if (outputCount != 0 && mainPatchCount != 0 &&
                outputCount != mainPatchCount)
                diag.ReportError(main.location,
                    "Hull InputPatch N must match outputcontrolpoints.");
        }

        if (!patchConstantFn)
        {
            diag.ReportError(main.location,
                "Hull shader requires PatchConstant().");
            return;
        }

        if (!patchConstantFn->attributes.empty())
            diag.ReportError(patchConstantFn->location,
                "PatchConstant() does not accept entry-point attributes.");
        if (patchConstantFn->returnType != "PATCH_CONSTANT_DATA")
            diag.ReportError(patchConstantFn->location,
                "PatchConstant() must return PATCH_CONSTANT_DATA.");
        if (patchConstantFn->parameters.size() != 1)
        {
            diag.ReportError(patchConstantFn->location,
                "PatchConstant() must take exactly one InputPatch<STAGE_INPUT,N>.");
        }
        else
        {
            uint32_t constantPatchCount = 0;
            if (ValidatePatchType(patchConstantFn->parameters[0], "InputPatch",
                    constantPatchCount, diag, patchConstantFn->location) &&
                mainPatchCount != 0 && constantPatchCount != mainPatchCount)
                diag.ReportError(patchConstantFn->location,
                    "PatchConstant InputPatch N must match Hull Main InputPatch N.");
        }
        return;
    }

    if (program.stage == ShaderStage::DOMAIN)
    {
        const Attribute* domain = nullptr;
        ValidateRequiredAttribute(main, "domain", diag, domain);
        const std::string domainValue = domain ? AttributeValue(*domain) : "";
        if (domain)
        {
            if (!HasQuotedAttributeValue(*domain))
                diag.ReportError(main.location,
                    "Domain attribute value must be a string literal.");
            if (!IsOneOf(domainValue, {"tri", "quad", "isoline"}))
                diag.ReportError(main.location,
                    "Domain attribute must be tri, quad, or isoline.");
        }

        if (main.returnType != "STAGE_OUTPUT")
            diag.ReportError(main.location, "Domain Main() must return STAGE_OUTPUT.");
        if (!main.returnSemantic.empty())
            diag.ReportError(main.location,
                "Domain STAGE_OUTPUT Main() must not declare a return semantic.");
        if (main.parameters.size() != 3)
        {
            diag.ReportError(main.location,
                "Domain Main() must take PATCH_CONSTANT_DATA, "
                "DOMAIN_LOCATION, and OutputPatch<STAGE_INPUT,N>.");
            return;
        }

        if (main.parameters[0].typeName != "PATCH_CONSTANT_DATA")
            diag.ReportError(main.location,
                "Domain first parameter must be PATCH_CONSTANT_DATA.");

        const auto& location = main.parameters[1];
        const std::string expectedLocationType =
            domainValue == "tri" ? "Vec3" : "Vec2";
        if (location.semantic != "DOMAIN_LOCATION" ||
            (!domainValue.empty() && location.typeName != expectedLocationType))
            diag.ReportError(main.location,
                "Domain location parameter type/semantic does not match domain "
                "(tri: Vec3, quad/isoline: Vec2; semantic DOMAIN_LOCATION).");

        uint32_t patchCount = 0;
        ValidatePatchType(main.parameters[2], "OutputPatch",
            patchCount, diag, main.location);
    }
}

// =====================================================================
//  Pass X — Validate Property Block Names
// =====================================================================

void SemanticAnalyzer::ValidatePropertyBlockNames(
    const ParsedProperties& properties,
    DiagnosticEngine&       diag)
{
    std::unordered_set<std::string> seenBlocks;
    for (const auto& block : properties.blocks)
    {
        if (seenBlocks.count(block.name))
        {
            // Note: Since location isn't stored in NSCPropertyBlock, we just use 0,0
            diag.ReportError(0, 0, "Duplicate PROPERTY block name '" + block.name + "'. Property block names must be unique.");
        }
        seenBlocks.insert(block.name);
    }
}

// =====================================================================
//  Pass 4 — Property Usage Check
// =====================================================================

void SemanticAnalyzer::ValidatePropertyUsage(
    const Program&          program,
    const ParsedProperties& properties,
    DiagnosticEngine&       diag)
{
    std::unordered_set<std::string> accessedGroups;

    for (const auto& alias : program.aliases)
        accessedGroups.insert(alias.sourceBinding);

    // Walk all function bodies
    for (const auto& fn : program.functions)
        CollectPropertyAccesses(fn.body, accessedGroups);

    // Normalize short names to qualified names to check usage
    std::unordered_set<std::string> accessedQualified;
    for (const auto& name : accessedGroups)
    {
        const PropertyGroup* grp = FindGroup(properties, name);
        if (grp) accessedQualified.insert(grp->qualifiedName);
    }

    for (const auto& grp : properties.groups)
    {
        if (!accessedQualified.count(grp.qualifiedName))
        {
            diag.ReportWarning(grp.location,
                "PROPERTY group '" + grp.name + "' in block '" + grp.ownerPropertyName +
                "' is declared but never accessed in the SHADER block.");
        }
    }
}

// =====================================================================
//  CollectPropertyAccesses
// =====================================================================

void SemanticAnalyzer::CollectPropertyAccesses(
    const std::vector<StmtPtr>&      stmts,
    std::unordered_set<std::string>& outAccessedGroups) const
{
    for (const auto& stmt : stmts)
    {
        CollectInStmt(stmt.get(), outAccessedGroups);
    }
}

void SemanticAnalyzer::CollectInStmt(
    const Statement*                 stmt,
    std::unordered_set<std::string>& outAccessedGroups) const
{
    if (!stmt) return;

    if (auto* es = dynamic_cast<const ExprStmt*>(stmt))
    {
        CollectInExpr(es->expr.get(), outAccessedGroups);
    }
    else if (auto* rs = dynamic_cast<const ReturnStmt*>(stmt))
    {
        CollectInExpr(rs->value.get(), outAccessedGroups);
    }
    else if (auto* vd = dynamic_cast<const VarDeclStmt*>(stmt))
    {
        CollectInExpr(vd->initializer.get(), outAccessedGroups);
    }
    else if (auto* blk = dynamic_cast<const BlockStmt*>(stmt))
    {
        CollectPropertyAccesses(blk->statements, outAccessedGroups);
    }
    else if (auto* ifs = dynamic_cast<const IfStmt*>(stmt))
    {
        CollectInExpr(ifs->condition.get(), outAccessedGroups);
        if (ifs->thenBranch) CollectInStmt(ifs->thenBranch.get(), outAccessedGroups);
        if (ifs->elseBranch) CollectInStmt(ifs->elseBranch.get(), outAccessedGroups);
    }
    else if (auto* fs = dynamic_cast<const ForStmt*>(stmt))
    {
        // init can be VarDeclStmt, ExprStmt, or nullptr — delegate to CollectInStmt
        if (fs->init) CollectInStmt(fs->init.get(), outAccessedGroups);
        CollectInExpr(fs->condition.get(), outAccessedGroups);
        CollectInExpr(fs->increment.get(), outAccessedGroups);
        if (fs->body) CollectInStmt(fs->body.get(), outAccessedGroups);
    }
    else if (auto* ws = dynamic_cast<const WhileStmt*>(stmt))
    {
        CollectInExpr(ws->condition.get(), outAccessedGroups);
        if (ws->body) CollectInStmt(ws->body.get(), outAccessedGroups);
    }
    else if (auto* dws = dynamic_cast<const DoWhileStmt*>(stmt))
    {
        if (dws->body) CollectInStmt(dws->body.get(), outAccessedGroups);
        CollectInExpr(dws->condition.get(), outAccessedGroups);
    }
    else if (auto* sw = dynamic_cast<const SwitchStmt*>(stmt))
    {
        CollectInExpr(sw->condition.get(), outAccessedGroups);
        for (const auto& sc : sw->cases)
        {
            CollectInExpr(sc.value.get(), outAccessedGroups);
            CollectPropertyAccesses(sc.statements, outAccessedGroups);
        }
    }
}

void SemanticAnalyzer::CollectInExpr(
    const Expression*                expr,
    std::unordered_set<std::string>& outAccessedGroups) const
{
    if (!expr) return;

    if (auto* ma = dynamic_cast<const MemberAccessExpr*>(expr))
    {
        CollectInExpr(ma->object.get(), outAccessedGroups);

        // Check 3-level access: Property.Group.Field -> inner object is MemberAccessExpr (Property.Group)
        if (auto* innerMa = dynamic_cast<const MemberAccessExpr*>(ma->object.get()))
        {
            if (auto* id = dynamic_cast<const IdentifierExpr*>(innerMa->object.get()))
            {
                std::string qualifiedName = id->name + "__" + innerMa->member;
                if (m_knownNames.count(qualifiedName))
                    outAccessedGroups.insert(qualifiedName);
            }
        }

        // Check 2-level access: Property.Group -> ma is MemberAccessExpr (Property.Group)
        if (auto* id = dynamic_cast<const IdentifierExpr*>(ma->object.get()))
        {
            std::string qualifiedName = id->name + "__" + ma->member;
            if (m_knownNames.count(qualifiedName))
                outAccessedGroups.insert(qualifiedName);
        }
    }
    else if (auto* be = dynamic_cast<const BinaryExpr*>(expr))
    {
        CollectInExpr(be->left.get(), outAccessedGroups);
        CollectInExpr(be->right.get(), outAccessedGroups);
    }
    else if (auto* ce = dynamic_cast<const CallExpr*>(expr))
    {
        CollectInExpr(ce->callee.get(), outAccessedGroups);
        for (const auto& a : ce->args) CollectInExpr(a.get(), outAccessedGroups);
    }
    else if (auto* ie = dynamic_cast<const IndexExpr*>(expr))
    {
        CollectInExpr(ie->object.get(), outAccessedGroups);
        CollectInExpr(ie->index.get(), outAccessedGroups);
    }
    else if (auto* sw = dynamic_cast<const SwizzleExpr*>(expr))
    {
        CollectInExpr(sw->object.get(), outAccessedGroups);
    }
    else if (auto* le = dynamic_cast<const LengthExpr*>(expr))
    {
        CollectInExpr(le->object.get(), outAccessedGroups);
    }
    else if (auto* ue = dynamic_cast<const UnaryExpr*>(expr))
    {
        CollectInExpr(ue->operand.get(), outAccessedGroups);
    }
    else if (auto* pe = dynamic_cast<const ParenExpr*>(expr))
    {
        CollectInExpr(pe->inner.get(), outAccessedGroups);
    }
    else if (auto* te = dynamic_cast<const TernaryExpr*>(expr))
    {
        CollectInExpr(te->condition.get(), outAccessedGroups);
        CollectInExpr(te->thenExpr.get(), outAccessedGroups);
        CollectInExpr(te->elseExpr.get(), outAccessedGroups);
    }
    else if (auto* cast = dynamic_cast<const CastExpr*>(expr))
    {
        for (const auto& a : cast->args) CollectInExpr(a.get(), outAccessedGroups);
    }
    else if (auto* init = dynamic_cast<const InitializerListExpr*>(expr))
    {
        for (const auto& element : init->elements)
            CollectInExpr(element.get(), outAccessedGroups);
    }
}

// =====================================================================
//  Lookup helpers
// =====================================================================

const PropertyGroup* SemanticAnalyzer::FindGroup(
    const ParsedProperties& props,
    const std::string&      name) const
{
    // Try qualified match first
    auto it = props.byQualifiedName.find(name);
    if (it != props.byQualifiedName.end()) return it->second;

    // Fall back to short name match
    for (const auto& g : props.groups)
        if (g.name == name) return &g;

    return nullptr;
}

bool SemanticAnalyzer::GroupHasField(
    const PropertyGroup& group,
    const std::string&   fieldName) const
{
    for (const auto& e : group.entries)
        if (e.name == fieldName) return true;
    return false;
}

} // namespace NSL
