#include "nsl_property_parser.h"
#include <cctype>
#include <algorithm>
#include <cstdint>

namespace NSL
{

// =====================================================================
//  Trim helper (local)
// =====================================================================

static std::string TrimStr(const std::string& s)
{
    size_t st = s.find_first_not_of(" \t\r\n");
    size_t en = s.find_last_not_of(" \t\r\n");
    return (st == std::string::npos) ? std::string{} : s.substr(st, en - st + 1);
}

// =====================================================================
//  Public — ParseBlock
//
//  Parses one named PROPERTY block's rawText into outProps.
//  Per-property-block group name uniqueness is enforced here.
// =====================================================================

bool PropertyParser::ParseBlock(
    const std::string&  ownerPropertyName,
    const std::string&  rawText,
    uint32_t            startLine,
    ParsedProperties&   outProps,
    DiagnosticEngine&   diag)
{
    ParseCursor cur;
    cur.ptr  = rawText.data();
    cur.end  = rawText.data() + rawText.size();
    cur.line = startLine;
    cur.col  = 1;

    // Parse all groups inside this PROPERTY block
    while (!IsAtEnd(cur))
    {
        SkipWhitespaceAndComments(cur);
        if (IsAtEnd(cur)) break;

        if (!ParseGroup(cur, ownerPropertyName, outProps, diag))
            return false;
    }

    // ==================== ENGINE GET WIRING ====================
    // Flat engine properties only: stamp enginePropertyName / type info.
    for (auto& group : outProps.groups)
    {
        if (group.ownerPropertyName != ownerPropertyName) continue;

        for (auto& entry : group.entries)
        {
            if (entry.kind != PropertyEntryKind::GET) continue;

            if (entry.enginePropertyName.empty())
                entry.enginePropertyName = entry.name;

            if (const EngineProperty* ep = FindEngineProperty(entry.name))
            {
                entry.typeID = ep->typeID;
                if (entry.typeName.empty())
                {
                    if (const DataTypeInfo* dti = FindDataTypeByID(ep->typeID))
                        entry.typeName = dti->nslName;
                }
            }
        }
    }

    return !diag.HasErrors();
}

// =====================================================================
//  ParseGroup
//  Parses: ResourceType GroupName { entries }
//  Enforces per-PROPERTY-block group name uniqueness.
// =====================================================================

bool PropertyParser::ParseGroup(
    ParseCursor&        cur,
    const std::string&  ownerPropertyName,
    ParsedProperties&   outProps,
    DiagnosticEngine&   diag)
{
    // ==================== Resource type ====================
    SourceLocation groupLoc { "", cur.line, cur.col };
    std::string resourceTypeName = ReadIdentifier(cur);

    if (resourceTypeName.empty())
    {
        diag.ReportError(groupLoc,
            "Expected resource type keyword inside PROPERTY '" + ownerPropertyName + "' "
            "(Buffer, Texture2D, TextureCube, Texture2DArray, TextureCubeArray, "
            "SamplerState, SamplerComparisonState, StructuredBuffer, RWStructuredBuffer, RWTexture2D).");
        return false;
    }

    ResourceKind kind = ResourceKindFromString(resourceTypeName);
    if (kind == ResourceKind::UNKNOWN)
    {
        diag.ReportError(groupLoc,
            "Unknown resource type '" + resourceTypeName + "' in PROPERTY '" + ownerPropertyName + "'. "
            "Expected: Buffer, Texture2D, TextureCube, Texture2DArray, TextureCubeArray, "
            "SamplerState, SamplerComparisonState, StructuredBuffer, RWStructuredBuffer, or RWTexture2D.");
        return false;
    }

    // ==================== Group name (MANDATORY) ====================
    SkipWhitespaceAndComments(cur);

    SourceLocation nameLoc { "", cur.line, cur.col };
    std::string groupName = ReadIdentifier(cur);

    if (groupName.empty())
    {
        diag.ReportError(nameLoc,
            "PROPERTY group name is mandatory inside PROPERTY '" + ownerPropertyName + "'. "
            "Example: Buffer MaterialParams { ... }");
        return false;
    }

    // ==================== Per-PROPERTY scoped uniqueness check ====================
    // Groups in the SAME PROPERTY block cannot share a name.
    // Groups in DIFFERENT PROPERTY blocks CAN share names — qualifiedName differentiates them.
    auto& namesInBlock = outProps.groupNamesByBlock[ownerPropertyName];
    if (namesInBlock.count(groupName))
    {
        diag.ReportError(nameLoc,
            "Duplicate group name '" + groupName + "' inside PROPERTY '" + ownerPropertyName + "'. "
            "Group names must be unique within the same PROPERTY block. "
            "Tip: you can use the same name in a different PROPERTY block.");
        return false;
    }
    namesInBlock.insert(groupName);

    // ==================== Opening brace ====================
    SkipWhitespaceAndComments(cur);

    if (IsAtEnd(cur) || Current(cur) != '{')
    {
        diag.ReportError(cur.line, cur.col,
            "Expected '{' after group name '" + groupName + "' in PROPERTY '" + ownerPropertyName + "'.");
        return false;
    }
    Advance(cur); // consume '{'

    // ==================== Parse entries ====================
    PropertyGroup group;
    group.name              = groupName;
    group.ownerPropertyName = ownerPropertyName;
    group.qualifiedName     = BuildQualifiedName(ownerPropertyName, groupName);
    group.resourceKind      = kind;
    group.resourceTypeName  = resourceTypeName;
    group.location          = groupLoc;

    std::unordered_set<std::string> entryNames;

    while (!IsAtEnd(cur))
    {
        SkipWhitespaceAndComments(cur);
        if (IsAtEnd(cur)) break;
        if (Current(cur) == '}') { Advance(cur); break; }

        if (Current(cur) == ';')
        {
            diag.ReportError(cur.line, cur.col, "Semicolons (;) are not allowed in PROPERTY blocks. Please remove it.");
            return false;
        }

        size_t entriesBefore = group.entries.size();
        if (!ParseEntry(cur, group, diag)) return false;

        // Duplicate entry name check
        if (group.entries.size() > entriesBefore)
        {
            const auto& newEntry = group.entries.back();
            if (entryNames.count(newEntry.name))
            {
                diag.ReportError(newEntry.location,
                    "Duplicate entry name '" + newEntry.name + "' in group '" + groupName +
                    "' inside PROPERTY '" + ownerPropertyName + "'.");
                return false;
            }
            entryNames.insert(newEntry.name);
        }
    }

    // ==================== Register group in outProps ====================
    outProps.groups.push_back(std::move(group));

    // Build byQualifiedName index (pointer into stable storage — we use index-based approach)
    // Note: re-index on every push is safe because groups vector is the owner.
    outProps.byQualifiedName.clear();
    for (auto& g : outProps.groups)
        outProps.byQualifiedName[g.qualifiedName] = &g;

    // Update PropertyBlock list
    bool blockExists = false;
    for (auto& block : outProps.blocks)
    {
        if (block.name == ownerPropertyName)
        {
            block.groups.push_back(outProps.groups.back());
            blockExists = true;
            break;
        }
    }
    if (!blockExists)
    {
        PropertyBlock newBlock;
        newBlock.name = ownerPropertyName;
        newBlock.groups.push_back(outProps.groups.back());
        outProps.blocks.push_back(std::move(newBlock));
    }

    return true;
}

// =====================================================================
//  ParseEntry
//  Parses one Add() or Get() line inside a group.
//  On success, pushes the new entry into group.entries.
// =====================================================================

bool PropertyParser::ParseEntry(
    ParseCursor&      cur,
    PropertyGroup&    group,
    DiagnosticEngine& diag)
{
    SourceLocation loc { "", cur.line, cur.col };
    std::string    keyword = ReadIdentifier(cur);

    PropertyEntryKind entryKind = PropertyEntryKind::ADD;
    if (keyword == "Add")
        entryKind = PropertyEntryKind::ADD;
    else if (keyword == "Get")
        entryKind = PropertyEntryKind::GET;
    else
    {
        if (keyword.empty() && !IsAtEnd(cur))
        {
            diag.ReportError(loc,
                "Unexpected character '" + std::string(1, Current(cur)) + "' in group '" + group.name + "'. Expected 'Add' or 'Get'.");
            return false;
        }

        diag.ReportError(loc,
            "Expected 'Add' or 'Get' inside group '" + group.name + "', got '" + keyword + "'.");
        return false;
    }

    // ==================== '(' content ')' ====================
    SkipWhitespaceAndComments(cur);
    if (IsAtEnd(cur) || Current(cur) != '(')
    {
        diag.ReportError(cur.line, cur.col,
            "Expected '(' after '" + keyword + "' in group '" + group.name + "'.");
        return false;
    }
    Advance(cur); // consume '('

    // Read paren content with depth tracking
    std::string parenContent;
    int depth = 1;
    while (!IsAtEnd(cur) && depth > 0)
    {
        char c = Current(cur);
        if (c == '(') ++depth;
        if (c == ')') { --depth; if (depth == 0) { Advance(cur); break; } }
        parenContent += c;
        Advance(cur);
    }

    PropertyEntry entry;
    entry.kind     = entryKind;
    entry.location = loc;

    if (entryKind == PropertyEntryKind::ADD)
    {
        // New syntax:  Add(name) : TypeName[ArraySize] = DefaultValue
        std::string name = TrimStr(parenContent);

        if (name.empty())
        {
            diag.ReportError(loc, "Empty name in Add() inside group '" + group.name + "'.");
            return false;
        }

        if (name.find(',') != std::string::npos)
        {
            diag.ReportError(loc, "Old syntax 'Add(name, type)' is no longer supported. Please use 'Add(name) : Type = value'.");
            return false;
        }

        for (char c : name)
        {
            if (!isalnum(c) && c != '_')
            {
                diag.ReportError(loc, "Invalid character '" + std::string(1, c) + "' in property name '" + name + "'. Only alphanumeric characters and underscores are allowed.");
                return false;
            }
        }
        
        entry.name = name;

        // Parse TypeName [ArraySize] = Default
        SkipWhitespaceAndComments(cur);
        if (!IsAtEnd(cur) && Current(cur) == ':')
        {
            Advance(cur); // consume ':'
            SkipWhitespaceAndComments(cur);
            entry.typeName = ReadIdentifier(cur);

            // Optional array size: [N]
            SkipWhitespaceAndComments(cur);
            if (!IsAtEnd(cur) && Current(cur) == '[')
            {
                Advance(cur); // '['
                entry.arraySizeExpr = TrimStr(ReadUntil(cur, ']'));
                if (!IsAtEnd(cur) && Current(cur) == ']') Advance(cur);

                // Array sizes in PROPERTY must be decimal integer literals.
                if (entry.arraySizeExpr.empty())
                {
                    diag.ReportError(entry.location,
                        "Array size for '" + entry.name + "' cannot be empty.");
                    return false;
                }
                for (char c : entry.arraySizeExpr)
                {
                    if (!std::isdigit(static_cast<unsigned char>(c)))
                    {
                        diag.ReportError(entry.location,
                            "Array size for '" + entry.name + "' must be a decimal integer literal, got '"
                            + entry.arraySizeExpr + "'.");
                        return false;
                    }
                }
                if (entry.arraySizeExpr == "0")
                {
                    diag.ReportError(entry.location,
                        "Array size for '" + entry.name + "' must be greater than zero.");
                    return false;
                }
                try
                {
                    constexpr uint64_t kMaxArrayElements = 64u * 1024u;
                    const uint64_t count = std::stoull(entry.arraySizeExpr);
                    if (count > kMaxArrayElements)
                    {
                        diag.ReportError(entry.location,
                            "Array size for '" + entry.name + "' exceeds the maximum of " +
                            std::to_string(kMaxArrayElements) + " elements.");
                        return false;
                    }
                }
                catch (...)
                {
                    diag.ReportError(entry.location,
                        "Array size for '" + entry.name + "' is out of range.");
                    return false;
                }
            }
        }

        // Optional default value: = ...
        SkipWhitespaceAndComments(cur);
        if (!IsAtEnd(cur) && Current(cur) == '=')
        {
            Advance(cur);
            SkipWhitespaceAndComments(cur);
            std::string defVal = ReadDefaultValue(cur);
            entry.defaultValue = TrimStr(defVal);

            if (!entry.defaultValue.empty() && entry.defaultValue.back() == ';')
            {
                diag.ReportError(cur.line, cur.col, "Semicolons (;) are not allowed at the end of default values in PROPERTY blocks.");
                return false;
            }
        }

        // Validate: Add() must not use engine-reserved names
        if (IsEnginePropertyName(entry.name))
        {
            diag.ReportError(entry.location,
                "Add() cannot use engine-reserved property name '" + entry.name + "'. "
                "Use Get('" + entry.name + "') to access engine-provided values.");
            return false;
        }

        // Resolve type ID from type name
        if (!entry.typeName.empty())
        {
            if (entry.typeName == "SamplerComparisonState")
            {
                entry.typeID = DataTypeID::SAMPLER_COMPARISON_STATE;
            }
            else if (entry.typeName == "SamplerState")
            {
                entry.typeID = DataTypeID::SAMPLER_STATE;
            }
            else
            {
                const DataTypeInfo* dti = FindDataType(entry.typeName);
                if (dti)
                {
                    entry.typeID = dti->id;
                    
                    if (entry.defaultValue.empty())
                    {
                        switch (entry.typeID)
                        {
                            case DataTypeID::FLOAT:
                            case DataTypeID::VEC2:
                            case DataTypeID::VEC3:
                            case DataTypeID::VEC4:
                            case DataTypeID::MAT3:
                            case DataTypeID::MAT4:
                            case DataTypeID::COLOR:
                            case DataTypeID::QUAT:
                                entry.defaultValue = "0.0f";
                                break;
                            case DataTypeID::INT:
                            case DataTypeID::INT2:
                            case DataTypeID::INT3:
                            case DataTypeID::INT4:
                            case DataTypeID::UINT:
                            case DataTypeID::UINT2:
                            case DataTypeID::UINT3:
                            case DataTypeID::UINT4:
                                entry.defaultValue = "0";
                                break;
                            case DataTypeID::BOOL:
                                entry.defaultValue = "false";
                                break;
                            case DataTypeID::DOUBLE:
                                entry.defaultValue = "0.0";
                                break;
                            default:
                                break;
                        }
                    }
                }
                else
                {
                    entry.typeID = DataTypeID::UNKNOWN; // custom struct — validated during alignment
                }
            }

            if (group.resourceKind == ResourceKind::RW_TEXTURE_2D)
            {
                const bool validRWTexType =
                    entry.typeID == DataTypeID::FLOAT ||
                    entry.typeID == DataTypeID::VEC2  ||
                    entry.typeID == DataTypeID::VEC3  ||
                    entry.typeID == DataTypeID::VEC4  ||
                    entry.typeID == DataTypeID::INT   ||
                    entry.typeID == DataTypeID::INT2  ||
                    entry.typeID == DataTypeID::INT3  ||
                    entry.typeID == DataTypeID::INT4  ||
                    entry.typeID == DataTypeID::UINT  ||
                    entry.typeID == DataTypeID::UINT2 ||
                    entry.typeID == DataTypeID::UINT3 ||
                    entry.typeID == DataTypeID::UINT4;
                if (!validRWTexType)
                {
                    diag.ReportError(entry.location,
                        "Unsupported element type '" + entry.typeName + "' for RWTexture2D entry '" +
                        entry.name + "'. Expected Float, Vec2..Vec4, Int..Int4, or UInt..UInt4.");
                    return false;
                }
            }
        }
        else
        {
            if (group.resourceKind == ResourceKind::SAMPLER_STATE)
            {
                if (group.resourceTypeName == "SamplerComparisonState")
                {
                    entry.typeName = "SamplerComparisonState";
                    entry.typeID   = DataTypeID::SAMPLER_COMPARISON_STATE;
                }
                else
                {
                    entry.typeName = "SamplerState";
                    entry.typeID   = DataTypeID::SAMPLER_STATE;
                }
            }
            else if (group.resourceKind == ResourceKind::RW_TEXTURE_2D)
            {
                entry.typeID = DataTypeID::VEC4;
            }
            else if (group.resourceKind == ResourceKind::TEXTURE_2D)
            {
                entry.typeName = "Texture2D";
                entry.typeID   = DataTypeID::TEXTURE_2D;
            }
            else if (group.resourceKind == ResourceKind::TEXTURE_CUBE)
            {
                entry.typeName = "TextureCube";
                entry.typeID   = DataTypeID::TEXTURE_CUBE;
            }
            else if (group.resourceKind == ResourceKind::TEXTURE_2D_ARRAY)
            {
                entry.typeName = "Texture2DArray";
                entry.typeID   = DataTypeID::TEXTURE_2D_ARRAY;
            }
            else if (group.resourceKind == ResourceKind::TEXTURE_CUBE_ARRAY)
            {
                entry.typeName = "TextureCubeArray";
                entry.typeID   = DataTypeID::TEXTURE_CUBE_ARRAY;
            }
            // For Buffer / StructuredBuffer / RWStructuredBuffer, explicit type is required
            else if (group.resourceKind == ResourceKind::BUFFER ||
                     group.resourceKind == ResourceKind::STRUCTURED_BUFFER ||
                     group.resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
            {
                diag.ReportError(entry.location,
                    "Add('" + entry.name + "') in a " + std::string(ResourceKindToString(group.resourceKind)) +
                    " group requires a type. Example: Add(" + entry.name + ") : Vec3");
                return false;
            }
        }
    }
    else
    {
        // GET entry
        std::string name = TrimStr(parenContent);
        if (name.empty())
        {
            diag.ReportError(loc, "Empty name in Get() inside group '" + group.name + "'.");
            return false;
        }

        const EngineProperty* ep = FindEngineProperty(name);
        if (!ep)
        {
            diag.ReportError(loc,
                "Unknown engine property '" + name + "' in Get(). "
                "Only engine-registered property names can be used with Get().");
            return false;
        }

        if (ep->resourceKind != group.resourceKind)
        {
            diag.ReportError(loc,
                "Type mismatch: engine property '" + name + "' requires " +
                ResourceKindToString(ep->resourceKind) + ", but group '" + group.name +
                "' is " + ResourceKindToString(group.resourceKind) + ".");
            return false;
        }

        if (group.resourceTypeName == "SamplerComparisonState" &&
            ep->typeID != DataTypeID::SAMPLER_COMPARISON_STATE)
        {
            diag.ReportError(loc,
                "Type mismatch: engine property '" + name + "' is a regular SamplerState, "
                "not a SamplerComparisonState, in group '" + group.name + "'.");
            return false;
        }
        if (group.resourceTypeName == "SamplerState" &&
            ep->typeID == DataTypeID::SAMPLER_COMPARISON_STATE)
        {
            diag.ReportError(loc,
                "Type mismatch: engine property '" + name + "' is a SamplerComparisonState, "
                "not a regular SamplerState, in group '" + group.name + "'.");
            return false;
        }

        entry.name               = name;
        entry.enginePropertyName = name;
        entry.typeID             = ep->typeID;

        // Fill typeName from typeID
        switch (ep->typeID)
        {
            case DataTypeID::FLOAT:                    entry.typeName = "Float";                  break;
            case DataTypeID::DOUBLE:                   entry.typeName = "Double";                 break;
            case DataTypeID::BOOL:                     entry.typeName = "Bool";                   break;
            case DataTypeID::VEC2:                     entry.typeName = "Vec2";                   break;
            case DataTypeID::VEC3:                     entry.typeName = "Vec3";                   break;
            case DataTypeID::VEC4:                     entry.typeName = "Vec4";                   break;
            case DataTypeID::INT:                      entry.typeName = "Int";                    break;
            case DataTypeID::INT2:                     entry.typeName = "Int2";                   break;
            case DataTypeID::INT3:                     entry.typeName = "Int3";                   break;
            case DataTypeID::INT4:                     entry.typeName = "Int4";                   break;
            case DataTypeID::UINT:                     entry.typeName = "UInt";                   break;
            case DataTypeID::UINT2:                    entry.typeName = "UInt2";                  break;
            case DataTypeID::UINT3:                    entry.typeName = "UInt3";                  break;
            case DataTypeID::UINT4:                    entry.typeName = "UInt4";                  break;
            case DataTypeID::MAT3:                     entry.typeName = "Mat3";                   break;
            case DataTypeID::MAT4:                     entry.typeName = "Mat4";                   break;
            case DataTypeID::COLOR:                    entry.typeName = "Color";                  break;
            case DataTypeID::QUAT:                     entry.typeName = "Quat";                   break;
            case DataTypeID::TEXTURE_2D:               entry.typeName = "Texture2D";              break;
            case DataTypeID::TEXTURE_CUBE:             entry.typeName = "TextureCube";            break;
            case DataTypeID::TEXTURE_2D_ARRAY:         entry.typeName = "Texture2DArray";         break;
            case DataTypeID::TEXTURE_CUBE_ARRAY:       entry.typeName = "TextureCubeArray";       break;
            case DataTypeID::SAMPLER_STATE:            entry.typeName = "SamplerState";           break;
            case DataTypeID::SAMPLER_COMPARISON_STATE: entry.typeName = "SamplerComparisonState"; break;
            case DataTypeID::STRUCTURED_BUFFER:        entry.typeName = "StructuredBuffer";       break;
            default:                                   entry.typeName = "";                       break;
        }
    }

    group.entries.push_back(std::move(entry));
    return true;
}

// =====================================================================
//  Cursor helpers — static
// =====================================================================

bool PropertyParser::IsAtEnd(const ParseCursor& cur)
{
    return cur.ptr >= cur.end;
}

char PropertyParser::Current(const ParseCursor& cur)
{
    return IsAtEnd(cur) ? '\0' : *cur.ptr;
}

char PropertyParser::Peek(const ParseCursor& cur, int offset)
{
    const char* p = cur.ptr + offset;
    return (p < cur.end) ? *p : '\0';
}

char PropertyParser::Advance(ParseCursor& cur)
{
    char c = *cur.ptr++;
    if (c == '\n') { ++cur.line; cur.col = 1; }
    else            { ++cur.col; }
    return c;
}

void PropertyParser::SkipWhitespace(ParseCursor& cur)
{
    while (!IsAtEnd(cur) &&
           (Current(cur) == ' ' || Current(cur) == '\t' ||
            Current(cur) == '\r' || Current(cur) == '\n'))
        Advance(cur);
}

void PropertyParser::SkipLineComment(ParseCursor& cur)
{
    while (!IsAtEnd(cur) && Current(cur) != '\n')
        Advance(cur);
}

void PropertyParser::SkipBlockComment(ParseCursor& cur)
{
    while (!IsAtEnd(cur))
    {
        if (Current(cur) == '*' && Peek(cur) == '/') { Advance(cur); Advance(cur); return; }
        Advance(cur);
    }
}

void PropertyParser::SkipWhitespaceAndComments(ParseCursor& cur)
{
    while (!IsAtEnd(cur))
    {
        char c = Current(cur);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { Advance(cur); }
        else if (c == '/' && Peek(cur) == '/') { SkipLineComment(cur); }
        else if (c == '/' && Peek(cur) == '*') { Advance(cur); Advance(cur); SkipBlockComment(cur); }
        else break;
    }
}

bool PropertyParser::MatchWord(ParseCursor& cur, const std::string& word)
{
    size_t len = word.size();
    if (static_cast<size_t>(cur.end - cur.ptr) < len) return false;
    if (std::string(cur.ptr, len) != word) return false;
    const char* next = cur.ptr + len;
    if (next < cur.end && (std::isalnum(static_cast<unsigned char>(*next)) || *next == '_'))
        return false;
    for (size_t i = 0; i < len; ++i) Advance(cur);
    return true;
}

std::string PropertyParser::ReadIdentifier(ParseCursor& cur)
{
    std::string id;
    while (!IsAtEnd(cur) &&
           (std::isalnum(static_cast<unsigned char>(Current(cur))) || Current(cur) == '_'))
        id += Advance(cur);
    return id;
}

std::string PropertyParser::ReadUntil(ParseCursor& cur, char delim)
{
    std::string result;
    while (!IsAtEnd(cur) && Current(cur) != delim)
        result += Advance(cur);
    return result;
}

std::string PropertyParser::ReadDefaultValue(ParseCursor& cur)
{
    std::string result;
    int braceDepth = 0;
    int parenDepth = 0;

    while (!IsAtEnd(cur))
    {
        const char c = Current(cur);

        // Strip trailing line comments without swallowing them into defaultValue
        if (c == '/' && Peek(cur) == '/')
        {
            SkipLineComment(cur);
            if (braceDepth == 0 && parenDepth == 0)
                break;
            continue;
        }

        // Strip inline block comments cleanly
        if (c == '/' && Peek(cur) == '*')
        {
            Advance(cur);
            Advance(cur);
            SkipBlockComment(cur);
            if (!result.empty() && result.back() != ' ')
                result += ' ';
            continue;
        }

        if ((c == '\n' || c == '\r') && braceDepth == 0 && parenDepth == 0)
            break;

        // Stop before a subsequent Add(...) or Get(...) entry placed on the same line
        if (braceDepth == 0 && parenDepth == 0 &&
            (result.empty() || std::isspace(static_cast<unsigned char>(result.back()))) &&
            static_cast<size_t>(cur.end - cur.ptr) >= 4 &&
            (std::memcmp(cur.ptr, "Add", 3) == 0 || std::memcmp(cur.ptr, "Get", 3) == 0))
        {
            const char afterKw = cur.ptr[3];
            if (afterKw == '(' || afterKw == ' ' || afterKw == '\t')
                break;
        }

        if (c == '{')
        {
            ++braceDepth;
            result += Advance(cur);
            continue;
        }
        if (c == '}')
        {
            if (braceDepth == 0)
                break; // Group closing brace — leave it for ParseGroup!
            --braceDepth;
            result += Advance(cur);
            continue;
        }
        if (c == '(')
        {
            ++parenDepth;
            result += Advance(cur);
            continue;
        }
        if (c == ')' && parenDepth > 0)
        {
            --parenDepth;
            result += Advance(cur);
            continue;
        }

        result += Advance(cur);
    }

    return result;
}

bool PropertyParser::ExpectChar(
    ParseCursor&       cur,
    char               c,
    const std::string& context,
    DiagnosticEngine&  diag)
{
    if (!IsAtEnd(cur) && Current(cur) == c)
    {
        Advance(cur);
        return true;
    }
    diag.ReportError(cur.line, cur.col,
        std::string("Expected '") + c + "' " + context + ", got '" +
        (IsAtEnd(cur) ? "EOF" : std::string(1, Current(cur))) + "'.");
    return false;
}

} // namespace NSL
