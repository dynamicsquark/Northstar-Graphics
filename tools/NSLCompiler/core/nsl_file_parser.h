#pragma once

#include "nsl_error.h"
#include "NSLCommon/nsl_common_types.h"
#include <string>
#include <vector>
#include <unordered_map>

// =====================================================================
//  NSL File Parser  (v4 — Named PROPERTY Blocks)
//
//  Handles:
//    #shader VERTEX / PIXEL / COMPUTE / GEOMETRY / HULL / DOMAIN
//    PROPERTY Name { ... }   (name is MANDATORY, must be unique)
//    INPUT_LAYOUT  { Get(...) }   (optional, at most one)
//    OUTPUT_LAYOUT { Add(...) }   (optional; produce — catalog Add, not PROPERTY Add)
//    SHADER    { ... }       (exactly one block)
//    // line comments and /* block comments */
//
//  Each extracted PROPERTY block records its name and starting line so
//  that downstream parsers can report accurate source locations.
// =====================================================================

namespace NSL
{
    // =====================================================================
    //  PropertyBlockRaw — raw PROPERTY block text with source location
    // =====================================================================

    struct PropertyBlockRaw
    {
        std::string name;           // PROPERTY block name (e.g. "Materials") — MANDATORY
        std::string rawText;        // Inner content of PROPERTY Name { ... } (without keyword+name+braces)
        uint32_t    startLine = 0;  // Line number of the opening '{' in the source file
        SourceLocation location;    // Location of the PROPERTY keyword
    };

    // =====================================================================
    //  StageIOBlockRaw — raw INPUT_LAYOUT / OUTPUT_LAYOUT block
    // =====================================================================

    enum class StageIOBlockKind : uint8_t
    {
        INPUT  = 0,
        OUTPUT = 1,
    };

    struct StageIOBlockRaw
    {
        StageIOBlockKind kind = StageIOBlockKind::INPUT;
        std::string      rawText;
        uint32_t         startLine = 0;
        SourceLocation   location;
    };

    // =====================================================================
    //  NSLFileLayout — result of file-level parsing
    // =====================================================================

    struct NSLFileLayout
    {
        ShaderStage                    stage = ShaderStage::NONE;
        bool                           isLibrary = false;
        std::vector<std::string>       libraryDeclarations; // paths from LIBRARY_DECLARATION { Add(...) }
        std::vector<PropertyBlockRaw>  propertyBlocks;
        std::vector<StageIOBlockRaw>   stageIOBlocks;
        std::string                    shaderSource;       // Inner content of SHADER { ... }
        uint32_t                       shaderStartLine = 0;
        std::string                    librarySource;      // Inner content of LIBRARY { ... }
        uint32_t                       libraryStartLine = 0;
    };

    // =====================================================================
    //  NSLFileParser
    // =====================================================================

    class NSLFileParser
    {
    public:
        explicit NSLFileParser(const std::string& source);

        // Returns true on success; errors reported through diag.
        bool Parse(NSLFileLayout& outLayout, DiagnosticEngine& diag);

    private:
        // ==================== Internal helpers ====================

        bool   IsAtEnd()                     const;
        char   Current()                     const;
        char   Peek(int offset = 1)          const;
        char   Advance();
        void   SkipWhitespace();
        void   SkipLineComment();
        void   SkipBlockComment();
        void   SkipWhitespaceAndComments();

        bool        MatchWord(const std::string& word);
        std::string ReadIdentifier();

        // Extracts content between matching { } braces (does NOT include braces).
        bool ExtractBracedBlock(std::string& outContent, uint32_t& outStartLine,
                                DiagnosticEngine& diag);

        // Parses the #shader directive and returns the stage.
        ShaderStage ParseShaderDirective(DiagnosticEngine& diag);

        // Current position in source
        std::string m_source;
        size_t      m_pos  = 0;
        uint32_t    m_line = 1;
        uint32_t    m_col  = 1;
    };

} // namespace NSL
