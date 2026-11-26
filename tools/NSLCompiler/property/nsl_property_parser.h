#pragma once

#include "../core/nsl_error.h"
#include "NSLCommon/nsl_common_types.h"
#include "NSLCommon/nsl_engine_data.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// =====================================================================
//  NSL Property Parser  (v4 — Per-PROPERTY Scoped Group Name Uniqueness)
//
//  Changes from v3:
//    • PropertyGroup gains two new fields:
//        ownerPropertyName  — name of the PROPERTY block this group lives in
//        qualifiedName      — "PropertyName__GroupName"  (used in all codegen)
//    • Duplicate group name check is now PER PROPERTY BLOCK:
//        - PROPERTY A { Buffer x { } Buffer x { } }  → ERROR  (same property)
//        - PROPERTY A { Buffer x { } } PROPERTY B { Buffer x { } }  → OK
//    • Global uniqueness across PROPERTY blocks is no longer enforced for
//      group names; the qualified name guarantees no backend symbol collision.
//    • The 'name' field of PropertyGroup remains the short/local name for
//      use in NSL source access expressions (A.x.field).
//    • The 'qualifiedName' is used in all binary (NSC) and backend output.
// =====================================================================

namespace NSL
{
    // ========================= PROPERTY ENTRY ========================

    struct PropertyEntry
    {
        std::string         name;               // e.g. "baseColor"
        std::string         typeName;           // e.g. "Vec3", "Texture2D"
        DataTypeID          typeID      = DataTypeID::UNKNOWN;
        PropertyEntryKind   kind        = PropertyEntryKind::ADD;
        std::string         defaultValue;
        std::string         arraySizeExpr;
        std::string         enginePropertyName;
        uint16_t            assignedSet = 0;
        uint16_t            assignedBinding = 0xFFFF;
        SourceLocation      location;
    };

    // ========================= PROPERTY GROUP ========================

    struct PropertyGroup
    {
        std::string                  name;               // Short/local group name (e.g. "MaterialParams")
        std::string                  ownerPropertyName;  // Name of parent PROPERTY block (e.g. "Materials")
        std::string                  qualifiedName;      // "Materials__MaterialParams" — used in all codegen
        ResourceKind                 resourceKind = ResourceKind::BUFFER;
        std::string                  resourceTypeName;   // e.g. "SamplerState", "SamplerComparisonState", "Buffer"
        std::vector<PropertyEntry>   entries;
        uint16_t                     assignedSet = 0;
        uint16_t                     assignedBinding = 0xFFFF;
        SourceLocation               location;
    };

    // ========================= PROPERTY BLOCK ========================
    // Represents one named PROPERTY block (one-to-many groups inside).

    struct PropertyBlock
    {
        std::string                name;       // PROPERTY block name (e.g. "Materials")
        std::vector<PropertyGroup> groups;     // Groups in this block
        SourceLocation             location;
    };

    // ========================= PARSED PROPERTIES =====================

    struct ParsedProperties
    {
        // All groups from all PROPERTY blocks, flattened for easy iteration.
        std::vector<PropertyGroup>  groups;

        // Named PROPERTY blocks (preserves per-block structure for sema).
        std::vector<PropertyBlock>  blocks;

        // Quick lookup: qualifiedName → PropertyGroup*
        std::unordered_map<std::string, const PropertyGroup*> byQualifiedName;

        // Quick lookup: ownerPropertyName → [group short names in that block]
        // Used for per-property uniqueness validation.
        std::unordered_map<std::string, std::unordered_set<std::string>> groupNamesByBlock;

        // Helper: find a group by its short (local) name within a specific PROPERTY block.
        // Returns nullptr if not found.
        const PropertyGroup* FindInBlock(
            const std::string& propertyName,
            const std::string& groupName) const
        {
            for (const auto& g : groups)
            {
                if (g.ownerPropertyName == propertyName && g.name == groupName)
                    return &g;
            }
            return nullptr;
        }

        // Helper: find any group by its short local name (first match across all blocks).
        const PropertyGroup* FindByName(const std::string& name) const
        {
            for (const auto& g : groups)
                if (g.name == name) return &g;
            return nullptr;
        }

        // Helper: find a group by its qualified name ("PropertyName__GroupName").
        const PropertyGroup* FindByQualifiedName(const std::string& qName) const
        {
            auto it = byQualifiedName.find(qName);
            return (it != byQualifiedName.end()) ? it->second : nullptr;
        }
    };

    // ========================= PROPERTY PARSER =======================

    class PropertyParser
    {
    public:
        // Parse one PROPERTY block's rawText.
        //   ownerPropertyName — the name of the enclosing PROPERTY block
        //   rawText           — inner content of PROPERTY Name { ... }
        //   startLine         — line offset for accurate error locations
        bool ParseBlock(
            const std::string&  ownerPropertyName,
            const std::string&  rawText,
            uint32_t            startLine,
            ParsedProperties&   outProps,
            DiagnosticEngine&   diag
        );

    private:
        // ==================== Internal types ====================

        struct ParseCursor
        {
            const char* ptr;
            const char* end;
            uint32_t    line;
            uint32_t    col;
        };

        // ==================== Parsing ====================

        bool ParseGroup(
            ParseCursor&          cur,
            const std::string&    ownerPropertyName,
            ParsedProperties&     outProps,
            DiagnosticEngine&     diag
        );

        bool ParseEntry(
            ParseCursor&          cur,
            PropertyGroup&        group,
            DiagnosticEngine&     diag
        );

        // ==================== Cursor helpers ====================

        static void SkipWhitespace(ParseCursor& cur);
        static void SkipLineComment(ParseCursor& cur);
        static void SkipBlockComment(ParseCursor& cur);
        static void SkipWhitespaceAndComments(ParseCursor& cur);

        static bool IsAtEnd(const ParseCursor& cur);
        static char Current(const ParseCursor& cur);
        static char Peek(const ParseCursor& cur, int offset = 1);
        static char Advance(ParseCursor& cur);

        static bool        MatchWord(ParseCursor& cur, const std::string& word);
        static std::string ReadIdentifier(ParseCursor& cur);
        static std::string ReadUntil(ParseCursor& cur, char delim);
        static std::string ReadDefaultValue(ParseCursor& cur);
        static bool        ExpectChar(ParseCursor& cur, char c, const std::string& context,
                                      DiagnosticEngine& diag);

        // ==================== Name building ====================

        static std::string BuildQualifiedName(
            const std::string& ownerPropertyName,
            const std::string& groupName)
        {
            return ownerPropertyName + "__" + groupName;
        }
    };

} // namespace NSL
