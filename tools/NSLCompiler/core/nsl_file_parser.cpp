#include "nsl_file_parser.h"
#include <cctype>

namespace NSL
{

// =====================================================================
//  Constructor
// =====================================================================

NSLFileParser::NSLFileParser(const std::string& source)
    : m_source(source)
    , m_pos(0)
    , m_line(1)
    , m_col(1)
{
}

// =====================================================================
//  Public — Parse
// =====================================================================

bool NSLFileParser::Parse(NSLFileLayout& outLayout, DiagnosticEngine& diag)
{
    bool          shaderBlockFound  = false;
    bool          libraryBlockFound = false;
    NSLFileLayout layout;

    // For duplicate PROPERTY name detection
    std::unordered_map<std::string, SourceLocation> seenPropertyNames;

    while (!IsAtEnd())
    {
        SkipWhitespaceAndComments();
        if (IsAtEnd()) break;

        // ==================== STAGE / STAGE_DEF keyword ====================
        if (MatchWord("STAGE") || MatchWord("STAGE_DEF"))
        {
            if (layout.stage != ShaderStage::NONE)
            {
                diag.ReportError(m_line, m_col,
                    "Duplicate STAGE directive. Only one is allowed per file.");
                return false;
            }
            layout.stage = ParseShaderDirective(diag);
            if (diag.HasErrors()) return false;
            continue;
        }

        // ==================== #shader directive ====================
        if (Current() == '#')
        {
            Advance(); // consume '#'
            std::string kw = ReadIdentifier();
            if (kw == "shader")
            {
                if (layout.stage != ShaderStage::NONE)
                {
                    diag.ReportError(m_line, m_col,
                        "Duplicate #shader directive. Only one is allowed per file.");
                    return false;
                }
                layout.stage = ParseShaderDirective(diag);
                if (diag.HasErrors()) return false;
            }
            else if (kw == "include")
            {
                diag.ReportError(m_line, m_col,
                    "#include directives must be resolved by the Preprocessor before file parsing.");
                return false;
            }
            else
            {
                // Unknown directive — skip to end of line
                while (!IsAtEnd() && Current() != '\n') Advance();
            }
            continue;
        }

        // ==================== LIBRARY_DECLARATION ====================
        if (MatchWord("LIBRARY_DECLARATION"))
        {
            SkipWhitespaceAndComments();
            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col, "Expected '{' after LIBRARY_DECLARATION.");
                return false;
            }
            Advance(); // consume '{'

            while (!IsAtEnd())
            {
                SkipWhitespaceAndComments();
                if (IsAtEnd()) break;
                if (Current() == '}')
                {
                    Advance(); // consume '}'
                    break;
                }

                if (MatchWord("Add"))
                {
                    SkipWhitespaceAndComments();
                    if (IsAtEnd() || Current() != '(')
                    {
                        diag.ReportError(m_line, m_col, "Expected '(' after 'Add' in LIBRARY_DECLARATION.");
                        return false;
                    }
                    Advance(); // consume '('
                    SkipWhitespaceAndComments();

                    if (IsAtEnd() || Current() != '"')
                    {
                        diag.ReportError(m_line, m_col, "Expected quoted string path in Add(\"...\").");
                        return false;
                    }

                    Advance(); // consume opening quote '"'
                    std::string libPath;
                    while (!IsAtEnd() && Current() != '"' && Current() != '\n')
                    {
                        libPath += Advance();
                    }

                    if (IsAtEnd() || Current() != '"')
                    {
                        diag.ReportError(m_line, m_col, "Unterminated string literal in Add(\"...\").");
                        return false;
                    }
                    Advance(); // consume closing quote '"'
                    SkipWhitespaceAndComments();

                    if (IsAtEnd() || Current() != ')')
                    {
                        diag.ReportError(m_line, m_col, "Expected ')' after library path in Add(\"...\").");
                        return false;
                    }
                    Advance(); // consume ')'
                    SkipWhitespaceAndComments();

                    if (!IsAtEnd() && Current() == ';')
                    {
                        Advance(); // optional ';'
                    }

                    layout.libraryDeclarations.push_back(libPath);
                }
                else
                {
                    diag.ReportError(m_line, m_col, "Unexpected token in LIBRARY_DECLARATION. Expected Add(\"path.nsl\").");
                    return false;
                }
            }
            continue;
        }

        // ==================== PROPERTY block ====================
        if (MatchWord("PROPERTY"))
        {
            SourceLocation propLoc { "", m_line, m_col };
            SkipWhitespaceAndComments();

            // ==================== PROPERTY Name (MANDATORY) ====================
            if (IsAtEnd() || Current() == '{')
            {
                diag.ReportError(m_line, m_col,
                    "PROPERTY block requires a name. "
                    "Syntax: PROPERTY <Name> { ... }. "
                    "Example: PROPERTY Materials { ... }");
                return false;
            }

            std::string propName = ReadIdentifier();
            if (propName.empty())
            {
                diag.ReportError(m_line, m_col,
                    "Expected PROPERTY block name (identifier). "
                    "Syntax: PROPERTY <Name> { ... }");
                return false;
            }

            // Check duplicate PROPERTY names
            auto it = seenPropertyNames.find(propName);
            if (it != seenPropertyNames.end())
            {
                diag.ReportError(m_line, m_col,
                    "Duplicate PROPERTY block name '" + propName + "'. "
                    "Previously defined at " + it->second.ToString() + ". "
                    "Each PROPERTY block must have a unique name.");
                return false;
            }
            seenPropertyNames[propName] = { "", m_line, m_col };

            SkipWhitespaceAndComments();

            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col,
                    "Expected '{' after PROPERTY name '" + propName + "'.");
                return false;
            }

            std::string  rawText;
            uint32_t     startLine;
            if (!ExtractBracedBlock(rawText, startLine, diag)) return false;

            PropertyBlockRaw raw;
            raw.name      = propName;
            raw.rawText   = rawText;
            raw.startLine = startLine;
            raw.location  = propLoc;
            layout.propertyBlocks.push_back(std::move(raw));
            continue;
        }

        // ==================== INPUT_LAYOUT ====================
        if (MatchWord("INPUT_LAYOUT"))
        {
            SourceLocation loc{ "", m_line, m_col };
            SkipWhitespaceAndComments();
            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col, "Expected '{' after INPUT_LAYOUT.");
                return false;
            }

            std::string rawText;
            uint32_t    startLine;
            if (!ExtractBracedBlock(rawText, startLine, diag)) return false;

            StageIOBlockRaw raw;
            raw.kind      = StageIOBlockKind::INPUT;
            raw.rawText   = rawText;
            raw.startLine = startLine;
            raw.location  = loc;
            layout.stageIOBlocks.push_back(std::move(raw));
            continue;
        }

        // ==================== OUTPUT_LAYOUT ====================
        if (MatchWord("OUTPUT_LAYOUT"))
        {
            SourceLocation loc{ "", m_line, m_col };
            SkipWhitespaceAndComments();
            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col, "Expected '{' after OUTPUT_LAYOUT.");
                return false;
            }

            std::string rawText;
            uint32_t    startLine;
            if (!ExtractBracedBlock(rawText, startLine, diag)) return false;

            StageIOBlockRaw raw;
            raw.kind      = StageIOBlockKind::OUTPUT;
            raw.rawText   = rawText;
            raw.startLine = startLine;
            raw.location  = loc;
            layout.stageIOBlocks.push_back(std::move(raw));
            continue;
        }

        // ==================== SHADER block ====================
        if (MatchWord("SHADER"))
        {
            if (shaderBlockFound)
            {
                diag.ReportError(m_line, m_col,
                    "Duplicate SHADER block. Only one SHADER block is allowed per file.");
                return false;
            }

            SkipWhitespaceAndComments();

            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col,
                    "Expected '{' after SHADER keyword.");
                return false;
            }

            std::string  shaderContent;
            uint32_t     startLine;
            if (!ExtractBracedBlock(shaderContent, startLine, diag)) return false;

            layout.shaderSource    = shaderContent;
            layout.shaderStartLine = startLine;
            shaderBlockFound       = true;
            continue;
        }

        // ==================== LIBRARY block ====================
        if (MatchWord("LIBRARY"))
        {
            if (libraryBlockFound)
            {
                diag.ReportError(m_line, m_col,
                    "Duplicate LIBRARY block. Only one LIBRARY block is allowed per file.");
                return false;
            }

            SkipWhitespaceAndComments();

            if (IsAtEnd() || Current() != '{')
            {
                diag.ReportError(m_line, m_col,
                    "Expected '{' after LIBRARY keyword.");
                return false;
            }

            std::string  libContent;
            uint32_t     startLine;
            if (!ExtractBracedBlock(libContent, startLine, diag)) return false;

            layout.librarySource    = libContent;
            layout.libraryStartLine = startLine;
            layout.isLibrary        = true;
            libraryBlockFound       = true;
            continue;
        }

        // ==================== Unknown token ====================
        diag.ReportError(m_line, m_col,
            std::string("Unexpected character '") + Current()
            + "' at top level. Only STAGE, STAGE_DEF, #shader, LIBRARY_DECLARATION, PROPERTY, INPUT_LAYOUT, OUTPUT_LAYOUT, SHADER, and LIBRARY are allowed here.");
        return false;
    }

    // ==================== Validation ====================

    if (shaderBlockFound)
    {
        if (libraryBlockFound)
        {
            diag.ReportError(0, 0,
                "A file cannot contain both SHADER and LIBRARY blocks. "
                "Main shaders contain SHADER { ... }, while library shaders contain LIBRARY { ... }.");
            return false;
        }

        if (layout.stage == ShaderStage::NONE)
        {
            diag.ReportError(0, 0,
                "Missing STAGE directive for main shader. "
                "Expected: STAGE VERTEX | PIXEL | COMPUTE | GEOMETRY | HULL | DOMAIN");
            return false;
        }
    }
    else if (libraryBlockFound)
    {
        layout.isLibrary = true;
        // Library shaders do not require a STAGE directive.
    }
    else
    {
        diag.ReportError(0, 0,
            "Missing SHADER or LIBRARY block. "
            "Every .nsl file must contain either a SHADER { ... } block (main shader) "
            "or a LIBRARY { ... } block (library shader).");
        return false;
    }

    outLayout = std::move(layout);
    return true;
}

// =====================================================================
//  ParseShaderDirective
// =====================================================================

ShaderStage NSLFileParser::ParseShaderDirective(DiagnosticEngine& diag)
{
    SkipWhitespace();   // only skip spaces/tabs, not newlines

    if (IsAtEnd() || Current() == '\n')
    {
        diag.ReportError(m_line, m_col,
            "Expected shader stage after #shader. "
            "Use: VERTEX, PIXEL, COMPUTE, GEOMETRY, HULL, or DOMAIN.");
        return ShaderStage::NONE;
    }

    std::string stageName = ReadIdentifier();
    ShaderStage stage     = ShaderStageFromString(stageName);

    if (stage == ShaderStage::NONE)
    {
        diag.ReportError(m_line, m_col,
            "Unknown shader stage '" + stageName + "'. "
            "Expected: VERTEX, PIXEL, COMPUTE, GEOMETRY, HULL, or DOMAIN.");
    }

    return stage;
}

// =====================================================================
//  ExtractBracedBlock
// =====================================================================

bool NSLFileParser::ExtractBracedBlock(
    std::string&      outContent,
    uint32_t&         outStartLine,
    DiagnosticEngine& diag)
{
    if (Current() != '{')
    {
        diag.ReportError(m_line, m_col, "Internal error: ExtractBracedBlock called without '{'.");
        return false;
    }

    outStartLine = m_line;
    Advance(); // consume '{'

    std::string content;
    int         depth = 1;

    while (!IsAtEnd() && depth > 0)
    {
        char c = Current();

        if (c == '\n')
        {
            content += c;
            Advance();
            continue;
        }

        // Line comment — copy as-is
        if (c == '/' && Peek() == '/')
        {
            while (!IsAtEnd() && Current() != '\n')
            {
                content += Current();
                Advance();
            }
            continue;
        }

        // Block comment — copy as-is (preserves newlines for line count)
        if (c == '/' && Peek() == '*')
        {
            content += Advance(); // '/'
            content += Advance(); // '*'
            while (!IsAtEnd())
            {
                char cc = Current();
                if (cc == '*' && Peek() == '/')
                {
                    content += Advance(); // '*'
                    content += Advance(); // '/'
                    break;
                }
                content += cc;
                Advance();
            }
            continue;
        }

        // String literal — copy verbatim (guard trailing '\' / EOF)
        if (c == '"')
        {
            const uint32_t strLine = m_line;
            content += Advance(); // opening '"'
            bool closed = false;
            while (!IsAtEnd())
            {
                if (Current() == '"')
                {
                    content += Advance(); // closing '"'
                    closed = true;
                    break;
                }
                if (Current() == '\\')
                {
                    content += Advance(); // backslash
                    if (IsAtEnd())
                        break; // trailing '\' — unterminated
                    content += Advance(); // escaped char
                    continue;
                }
                content += Advance();
            }
            if (!closed)
            {
                diag.ReportError(strLine, 0, "Unterminated string literal.");
                return false;
            }
            continue;
        }

        // Brace depth tracking
        if (c == '{') ++depth;
        if (c == '}') { --depth; if (depth == 0) { Advance(); break; } }

        content += c;
        Advance();
    }

    if (depth != 0)
    {
        diag.ReportError(outStartLine, 0, "Unmatched '{' — missing closing '}'.");
        return false;
    }

    outContent = std::move(content);
    return true;
}

// =====================================================================
//  Character helpers
// =====================================================================

bool NSLFileParser::IsAtEnd() const
{
    return m_pos >= m_source.size();
}

char NSLFileParser::Current() const
{
    return IsAtEnd() ? '\0' : m_source[m_pos];
}

char NSLFileParser::Peek(int offset) const
{
    size_t idx = m_pos + static_cast<size_t>(offset);
    return (idx < m_source.size()) ? m_source[idx] : '\0';
}

char NSLFileParser::Advance()
{
    char c = m_source[m_pos++];
    if (c == '\n') { ++m_line; m_col = 1; }
    else            { ++m_col; }
    return c;
}

void NSLFileParser::SkipWhitespace()
{
    while (!IsAtEnd() && (Current() == ' ' || Current() == '\t' ||
                          Current() == '\r' || Current() == '\n'))
        Advance();
}

void NSLFileParser::SkipLineComment()
{
    while (!IsAtEnd() && Current() != '\n')
        Advance();
}

void NSLFileParser::SkipBlockComment()
{
    while (!IsAtEnd())
    {
        if (Current() == '*' && Peek() == '/')
        {
            Advance(); // '*'
            Advance(); // '/'
            return;
        }
        Advance();
    }
}

void NSLFileParser::SkipWhitespaceAndComments()
{
    while (!IsAtEnd())
    {
        if (Current() == ' ' || Current() == '\t' ||
            Current() == '\r' || Current() == '\n')
        {
            Advance();
        }
        else if (Current() == '/' && Peek() == '/')
        {
            SkipLineComment();
        }
        else if (Current() == '/' && Peek() == '*')
        {
            Advance(); // '/'
            Advance(); // '*'
            SkipBlockComment();
        }
        else
        {
            break;
        }
    }
}

bool NSLFileParser::MatchWord(const std::string& word)
{
    size_t len = word.size();
    if (m_pos + len > m_source.size()) return false;

    if (m_source.compare(m_pos, len, word) != 0) return false;

    size_t nextPos = m_pos + len;
    if (nextPos < m_source.size())
    {
        char next = m_source[nextPos];
        if (std::isalnum(static_cast<unsigned char>(next)) || next == '_')
            return false;
    }

    for (size_t i = 0; i < len; ++i) Advance();
    return true;
}

std::string NSLFileParser::ReadIdentifier()
{
    std::string id;
    while (!IsAtEnd() &&
           (std::isalnum(static_cast<unsigned char>(Current())) || Current() == '_'))
    {
        id += Advance();
    }
    return id;
}

} // namespace NSL
