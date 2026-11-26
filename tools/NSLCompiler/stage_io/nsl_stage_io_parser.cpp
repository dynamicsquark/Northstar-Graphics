#include "nsl_stage_io_parser.h"
#include <cctype>
#include <unordered_set>

namespace NSL
{

// =====================================================================
//  Cursor helpers
// =====================================================================

void StageIOParser::SkipWhitespace(ParseCursor& cur)
{
    while (!IsAtEnd(cur) && (*cur.ptr == ' ' || *cur.ptr == '\t' ||
                             *cur.ptr == '\r' || *cur.ptr == '\n'))
        Advance(cur);
}

void StageIOParser::SkipLineComment(ParseCursor& cur)
{
    while (!IsAtEnd(cur) && *cur.ptr != '\n')
        Advance(cur);
}

void StageIOParser::SkipBlockComment(ParseCursor& cur)
{
    while (!IsAtEnd(cur))
    {
        if (*cur.ptr == '*' && Peek(cur) == '/')
        {
            Advance(cur);
            Advance(cur);
            return;
        }
        Advance(cur);
    }
}

void StageIOParser::SkipWhitespaceAndComments(ParseCursor& cur)
{
    while (!IsAtEnd(cur))
    {
        if (*cur.ptr == ' ' || *cur.ptr == '\t' || *cur.ptr == '\r' || *cur.ptr == '\n')
            Advance(cur);
        else if (*cur.ptr == '/' && Peek(cur) == '/')
            SkipLineComment(cur);
        else if (*cur.ptr == '/' && Peek(cur) == '*')
        {
            Advance(cur);
            Advance(cur);
            SkipBlockComment(cur);
        }
        else
            break;
    }
}

bool StageIOParser::IsAtEnd(const ParseCursor& cur)
{
    return cur.ptr >= cur.end;
}

char StageIOParser::Current(const ParseCursor& cur)
{
    return IsAtEnd(cur) ? '\0' : *cur.ptr;
}

char StageIOParser::Peek(const ParseCursor& cur, int offset)
{
    const char* p = cur.ptr + offset;
    return (p < cur.end) ? *p : '\0';
}

char StageIOParser::Advance(ParseCursor& cur)
{
    char c = *cur.ptr++;
    if (c == '\n') { ++cur.line; cur.col = 1; }
    else           { ++cur.col; }
    return c;
}

bool StageIOParser::MatchWord(ParseCursor& cur, const std::string& word)
{
    size_t len = word.size();
    if (static_cast<size_t>(cur.end - cur.ptr) < len) return false;
    if (std::string(cur.ptr, len) != word) return false;

    const char* next = cur.ptr + len;
    if (next < cur.end)
    {
        char c = *next;
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
            return false;
    }

    for (size_t i = 0; i < len; ++i) Advance(cur);
    return true;
}

std::string StageIOParser::ReadIdentifier(ParseCursor& cur)
{
    std::string id;
    while (!IsAtEnd(cur) &&
           (std::isalnum(static_cast<unsigned char>(*cur.ptr)) || *cur.ptr == '_'))
    {
        id += Advance(cur);
    }
    return id;
}

bool StageIOParser::ExpectChar(
    ParseCursor& cur, char c, const std::string& context, DiagnosticEngine& diag)
{
    SkipWhitespaceAndComments(cur);
    if (IsAtEnd(cur) || *cur.ptr != c)
    {
        diag.ReportError(cur.line, cur.col,
            std::string("Expected '") + c + "' " + context + ".");
        return false;
    }
    Advance(cur);
    return true;
}

// =====================================================================
//  ParseCatalogEntry
// =====================================================================

bool StageIOParser::ParseCatalogEntry(
    ParseCursor&               cur,
    const char*                keyword,
    StageIODirection           direction,
    ShaderStage                stage,
    std::vector<StageIOEntry>& outEntries,
    DiagnosticEngine&          diag)
{
    SourceLocation loc{ "", cur.line, cur.col };
    const std::string kw = keyword;

    if (!ExpectChar(cur, '(', "after " + kw, diag)) return false;

    SkipWhitespaceAndComments(cur);
    std::string name = ReadIdentifier(cur);
    if (name.empty())
    {
        diag.ReportError(cur.line, cur.col,
            "Expected stage I/O name inside " + kw + "(...).");
        return false;
    }

    if (!ExpectChar(cur, ')', "after " + kw + " name", diag)) return false;

    for (const auto& e : outEntries)
    {
        if (e.name == name)
        {
            diag.ReportError(loc,
                "Duplicate " + kw + "('" + name + "') in "
                + (direction == StageIODirection::INPUT ? "INPUT_LAYOUT" : "OUTPUT_LAYOUT") + ".");
            return false;
        }
    }

    const StageIOProperty* prop = FindStageIOProperty(name, direction, stage);
    if (!prop)
    {
        const char* dirName = (direction == StageIODirection::INPUT) ? "INPUT_LAYOUT" : "OUTPUT_LAYOUT";
        diag.ReportError(loc,
            std::string("Unknown or invalid stage I/O ") + kw + "(" + name + ") for "
            + dirName + " in " + ShaderStageToString(stage) + " shader. "
            "Check nsl_stage_io.h for valid entries.");
        return false;
    }

    if (direction == StageIODirection::INPUT && stage == ShaderStage::VERTEX &&
        (name == "VERTEX_UV2" || name == "VERTEX_UV3" || name == "VERTEX_COLOR1"))
    {
        diag.ReportError(loc,
            "Vertex attribute '" + name + "' exceeds the 112-byte Northstar Vertex layout contract "
            "(supported vertex UVs/colors: VERTEX_UV0, VERTEX_UV1, VERTEX_COLOR / VERTEX_COLOR0).");
        return false;
    }

    StageIOEntry entry;
    entry.name     = name;
    entry.typeID   = prop->typeID;
    entry.category = prop->category;
    entry.location = loc;
    outEntries.push_back(std::move(entry));
    return true;
}

// =====================================================================
//  ParseBlock
// =====================================================================

bool StageIOParser::ParseBlock(
    StageIODirection   direction,
    ShaderStage        stage,
    const std::string& rawText,
    uint32_t           startLine,
    ParsedStageIO&     outIO,
    DiagnosticEngine&  diag)
{
    if (direction == StageIODirection::INPUT)
    {
        if (outIO.hasInputLayout)
        {
            diag.ReportError(startLine, 1, "Duplicate INPUT_LAYOUT block. Only one is allowed per file.");
            return false;
        }
        outIO.hasInputLayout = true;
    }
    else
    {
        if (outIO.hasOutputLayout)
        {
            diag.ReportError(startLine, 1, "Duplicate OUTPUT_LAYOUT block. Only one is allowed per file.");
            return false;
        }
        outIO.hasOutputLayout = true;
    }

    ParseCursor cur;
    cur.ptr  = rawText.data();
    cur.end  = rawText.data() + rawText.size();
    cur.line = startLine;
    cur.col  = 1;

    std::vector<StageIOEntry>& target =
        (direction == StageIODirection::INPUT) ? outIO.inputs : outIO.outputs;

    // INPUT_LAYOUT → Get(...); OUTPUT_LAYOUT → Add(...)  (catalog-only, not PROPERTY Add)
    const char* expectedKeyword = (direction == StageIODirection::INPUT) ? "Get" : "Add";
    const char* wrongKeyword    = (direction == StageIODirection::INPUT) ? "Add" : "Get";

    while (!IsAtEnd(cur))
    {
        SkipWhitespaceAndComments(cur);
        if (IsAtEnd(cur)) break;

        if (MatchWord(cur, wrongKeyword))
        {
            diag.ReportError(cur.line, cur.col,
                std::string(direction == StageIODirection::INPUT ? "INPUT_LAYOUT" : "OUTPUT_LAYOUT")
                + " uses " + expectedKeyword + "(...), not " + wrongKeyword + "(...).");
            return false;
        }

        if (!MatchWord(cur, expectedKeyword))
        {
            diag.ReportError(cur.line, cur.col,
                std::string("Expected ") + expectedKeyword + "(...) inside "
                + (direction == StageIODirection::INPUT ? "INPUT_LAYOUT" : "OUTPUT_LAYOUT") + ".");
            return false;
        }

        if (!ParseCatalogEntry(cur, expectedKeyword, direction, stage, target, diag))
            return false;
    }

    return true;
}

// =====================================================================
//  Validate
// =====================================================================

bool StageIOParser::Validate(
    ShaderStage          stage,
    const ParsedStageIO& io,
    DiagnosticEngine&    diag)
{
    if (stage == ShaderStage::COMPUTE)
    {
        if (io.hasInputLayout || io.hasOutputLayout)
        {
            diag.ReportError(0, 0,
                "INPUT_LAYOUT / OUTPUT_LAYOUT are not valid for COMPUTE shaders.");
            return false;
        }
        return true;
    }

    // Every graphics stage uses the normalized STAGE_INPUT/STAGE_OUTPUT
    // contract. Requiring both layouts prevents successful NSL compilation
    // from producing backend IR with undefined compiler-generated structs.
    if (!io.hasInputLayout || !io.hasOutputLayout)
    {
        diag.ReportError(0, 0,
            std::string(ShaderStageToString(stage)) +
            " shaders require both INPUT_LAYOUT and OUTPUT_LAYOUT.");
        return false;
    }
    if (io.inputs.empty() || (stage != ShaderStage::PIXEL && io.outputs.empty()))
    {
        diag.ReportError(0, 0,
            std::string(ShaderStageToString(stage)) +
            " INPUT_LAYOUT and OUTPUT_LAYOUT must each contain at least one entry.");
        return false;
    }

    if (stage == ShaderStage::VERTEX && io.hasOutputLayout)
    {
        bool hasPos = false;
        for (const auto& e : io.outputs)
        {
            if (e.name == "CLIP_POSITION")
            {
                hasPos = true;
                break;
            }
        }
        if (!hasPos)
        {
            diag.ReportError(0, 0,
                "VERTEX OUTPUT_LAYOUT must include Add(CLIP_POSITION).");
            return false;
        }
    }

    if (stage == ShaderStage::PIXEL && io.hasOutputLayout && !io.outputs.empty())
    {
        bool hasValidPixelOut = false;
        for (const auto& e : io.outputs)
        {
            if (e.name.rfind("COLOR_TARGET", 0) == 0 || e.name == "FRAG_DEPTH")
            {
                hasValidPixelOut = true;
                break;
            }
        }
        if (!hasValidPixelOut)
        {
            diag.ReportError(0, 0,
                "PIXEL OUTPUT_LAYOUT must be empty (depth-only) or include Add(COLOR_TARGET0..7) / Add(FRAG_DEPTH).");
            return false;
        }

        // Unique COLOR_TARGETn indices
        std::unordered_set<std::string> targets;
        for (const auto& e : io.outputs)
        {
            if (e.name.rfind("COLOR_TARGET", 0) == 0)
            {
                if (!targets.insert(e.name).second)
                {
                    diag.ReportError(e.location,
                        "Duplicate render target '" + e.name + "' in OUTPUT_LAYOUT.");
                    return false;
                }
            }
        }
    }

    // Geometry / Domain both emit a rasterizer-bound vertex: OUTPUT_LAYOUT must
    // carry CLIP_POSITION, mirroring the VERTEX rule.
    if ((stage == ShaderStage::GEOMETRY || stage == ShaderStage::DOMAIN))
    {
        bool hasPos = false;
        for (const auto& e : io.outputs)
            if (e.name == "CLIP_POSITION") { hasPos = true; break; }

        if (!hasPos)
        {
            diag.ReportError(0, 0,
                std::string(ShaderStageToString(stage)) +
                " OUTPUT_LAYOUT must include Add(CLIP_POSITION).");
            return false;
        }
    }

    // Hull emits a control-point payload consumed by the domain stage; both
    // layouts are required but CLIP_POSITION is not (the domain stage produces it).
    return !diag.HasErrors();
}

} // namespace NSL
