#pragma once

#include "../core/nsl_error.h"
#include "NSLCommon/nsl_common_types.h"
#include "NSLCommon/nsl_stage_io.h"
#include <string>
#include <vector>

// =====================================================================
//  NSL Stage I/O Parser
//
//  Parses INPUT_LAYOUT / OUTPUT_LAYOUT blocks:
//    INPUT_LAYOUT  { Get(VERTEX_POSITION) Get(VERTEX_NORMAL) ... }  // receive
//    OUTPUT_LAYOUT { Add(CLIP_POSITION) Add(WORLD_POSITION) ... }     // produce
//
//  Both resolve names against nsl_stage_io.h for the current stage.
//  Catalog Add() here is NOT PROPERTY Add() — no type/default; type
//  and category come from the stage I/O catalog. Target-API semantics
//  are applied at NSC write / backend emit (not during parse).
// =====================================================================

namespace NSL
{
    struct StageIOEntry
    {
        std::string       name;
        DataTypeID        typeID = DataTypeID::UNKNOWN;
        std::string       semantic;   // Opaque; set by target backend before emit/NSC write
        StageIOCategory   category = StageIOCategory::VARYING;
        SourceLocation    location;
    };

    struct ParsedStageIO
    {
        std::vector<StageIOEntry> inputs;
        std::vector<StageIOEntry> outputs;
        bool hasInputLayout  = false;
        bool hasOutputLayout = false;
    };

    class StageIOParser
    {
    public:
        bool ParseBlock(
            StageIODirection    direction,
            ShaderStage         stage,
            const std::string&  rawText,
            uint32_t            startLine,
            ParsedStageIO&      outIO,
            DiagnosticEngine&   diag
        );

        static bool Validate(
            ShaderStage          stage,
            const ParsedStageIO& io,
            DiagnosticEngine&    diag
        );

    private:
        struct ParseCursor
        {
            const char* ptr;
            const char* end;
            uint32_t    line;
            uint32_t    col;
        };

        bool ParseCatalogEntry(
            ParseCursor&        cur,
            const char*         keyword,
            StageIODirection    direction,
            ShaderStage         stage,
            std::vector<StageIOEntry>& outEntries,
            DiagnosticEngine&   diag
        );

        static void SkipWhitespace(ParseCursor& cur);
        static void SkipLineComment(ParseCursor& cur);
        static void SkipBlockComment(ParseCursor& cur);
        static void SkipWhitespaceAndComments(ParseCursor& cur);
        static bool IsAtEnd(const ParseCursor& cur);
        static char Current(const ParseCursor& cur);
        static char Peek(const ParseCursor& cur, int offset = 1);
        static char Advance(ParseCursor& cur);
        static bool MatchWord(ParseCursor& cur, const std::string& word);
        static std::string ReadIdentifier(ParseCursor& cur);
        static bool ExpectChar(ParseCursor& cur, char c, const std::string& context,
                               DiagnosticEngine& diag);
    };

} // namespace NSL
