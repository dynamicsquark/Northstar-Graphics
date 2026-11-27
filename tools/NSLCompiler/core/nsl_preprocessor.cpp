#include "nsl_preprocessor.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>

namespace NSL
{

namespace
{
    bool EqualsIgnoreCase(std::string_view a, std::string_view b)
    {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
        {
            const unsigned char ca = static_cast<unsigned char>(a[i]);
            const unsigned char cb = static_cast<unsigned char>(b[i]);
            if (std::tolower(ca) != std::tolower(cb))
                return false;
        }
        return true;
    }

    std::string_view StripLineCommentAndTrim(std::string_view sv)
    {
        bool inQuotes = false;
        for (size_t i = 0; i < sv.size(); ++i)
        {
            if (sv[i] == '"' && (i == 0 || sv[i - 1] != '\\'))
                inQuotes = !inQuotes;
            else if (!inQuotes && i + 1 < sv.size() && sv[i] == '/' && sv[i + 1] == '/')
            {
                sv = sv.substr(0, i);
                break;
            }
        }
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r'))
            sv.remove_prefix(1);
        while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r'))
            sv.remove_suffix(1);
        return sv;
    }

    bool IsIdentStart(char c)
    {
        const unsigned char uc = static_cast<unsigned char>(c);
        return std::isalpha(uc) != 0 || c == '_';
    }

    bool IsIdentChar(char c)
    {
        const unsigned char uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) != 0 || c == '_';
    }

    std::string_view ConsumeIdentifier(std::string_view& sv)
    {
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t'))
            sv.remove_prefix(1);
        if (sv.empty() || !IsIdentStart(sv.front()))
            return {};
        size_t len = 1;
        while (len < sv.size() && IsIdentChar(sv[len]))
            ++len;
        std::string_view id = sv.substr(0, len);
        sv.remove_prefix(len);
        return id;
    }

    class PreprocExprParser
    {
    public:
        PreprocExprParser(
            std::string_view                               expr,
            const std::unordered_map<std::string, int64_t>& macros)
            : m_sv(expr)
            , m_macros(macros)
        {
        }

        bool Parse(int64_t& outVal)
        {
            outVal = 0;
            if (!ParseLogicalOr(outVal))
                return false;
            SkipWS();
            return m_sv.empty();
        }

    private:
        std::string_view                                m_sv;
        const std::unordered_map<std::string, int64_t>& m_macros;

        void SkipWS()
        {
            while (!m_sv.empty() && (m_sv.front() == ' ' || m_sv.front() == '\t' || m_sv.front() == '\r'))
                m_sv.remove_prefix(1);
        }

        bool Match(std::string_view op)
        {
            SkipWS();
            if (m_sv.substr(0, op.size()) == op)
            {
                m_sv.remove_prefix(op.size());
                return true;
            }
            return false;
        }

        bool ParseLogicalOr(int64_t& out)
        {
            if (!ParseLogicalAnd(out)) return false;
            for (;;)
            {
                SkipWS();
                if (m_sv.substr(0, 2) == "||")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseLogicalAnd(rhs)) return false;
                    out = ((out != 0) || (rhs != 0)) ? 1 : 0;
                }
                else break;
            }
            return true;
        }

        bool ParseLogicalAnd(int64_t& out)
        {
            if (!ParseEquality(out)) return false;
            for (;;)
            {
                SkipWS();
                if (m_sv.substr(0, 2) == "&&")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseEquality(rhs)) return false;
                    out = ((out != 0) && (rhs != 0)) ? 1 : 0;
                }
                else break;
            }
            return true;
        }

        bool ParseEquality(int64_t& out)
        {
            if (!ParseRelational(out)) return false;
            for (;;)
            {
                SkipWS();
                if (m_sv.substr(0, 2) == "==")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseRelational(rhs)) return false;
                    out = (out == rhs) ? 1 : 0;
                }
                else if (m_sv.substr(0, 2) == "!=")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseRelational(rhs)) return false;
                    out = (out != rhs) ? 1 : 0;
                }
                else break;
            }
            return true;
        }

        bool ParseRelational(int64_t& out)
        {
            if (!ParseAdditive(out)) return false;
            for (;;)
            {
                SkipWS();
                if (m_sv.substr(0, 2) == "<=")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseAdditive(rhs)) return false;
                    out = (out <= rhs) ? 1 : 0;
                }
                else if (m_sv.substr(0, 2) == ">=")
                {
                    m_sv.remove_prefix(2);
                    int64_t rhs = 0;
                    if (!ParseAdditive(rhs)) return false;
                    out = (out >= rhs) ? 1 : 0;
                }
                else if (!m_sv.empty() && m_sv.front() == '<' && m_sv.substr(0, 2) != "<<")
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseAdditive(rhs)) return false;
                    out = (out < rhs) ? 1 : 0;
                }
                else if (!m_sv.empty() && m_sv.front() == '>' && m_sv.substr(0, 2) != ">>")
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseAdditive(rhs)) return false;
                    out = (out > rhs) ? 1 : 0;
                }
                else break;
            }
            return true;
        }

        bool ParseAdditive(int64_t& out)
        {
            if (!ParseMultiplicative(out)) return false;
            for (;;)
            {
                SkipWS();
                if (!m_sv.empty() && m_sv.front() == '+')
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseMultiplicative(rhs)) return false;
                    out += rhs;
                }
                else if (!m_sv.empty() && m_sv.front() == '-')
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseMultiplicative(rhs)) return false;
                    out -= rhs;
                }
                else break;
            }
            return true;
        }

        bool ParseMultiplicative(int64_t& out)
        {
            if (!ParseUnary(out)) return false;
            for (;;)
            {
                SkipWS();
                if (!m_sv.empty() && m_sv.front() == '*')
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseUnary(rhs)) return false;
                    out *= rhs;
                }
                else if (!m_sv.empty() && m_sv.front() == '/')
                {
                    m_sv.remove_prefix(1);
                    int64_t rhs = 0;
                    if (!ParseUnary(rhs) || rhs == 0) return false;
                    out /= rhs;
                }
                else break;
            }
            return true;
        }

        bool ParseUnary(int64_t& out)
        {
            SkipWS();
            if (m_sv.empty()) return false;
            if (m_sv.front() == '!' && m_sv.substr(0, 2) != "!=")
            {
                m_sv.remove_prefix(1);
                int64_t v = 0;
                if (!ParseUnary(v)) return false;
                out = (v == 0) ? 1 : 0;
                return true;
            }
            if (m_sv.front() == '+')
            {
                m_sv.remove_prefix(1);
                return ParseUnary(out);
            }
            if (m_sv.front() == '-')
            {
                m_sv.remove_prefix(1);
                int64_t v = 0;
                if (!ParseUnary(v)) return false;
                out = -v;
                return true;
            }
            return ParsePrimary(out);
        }

        bool ParsePrimary(int64_t& out)
        {
            SkipWS();
            if (m_sv.empty()) return false;

            if (m_sv.front() == '(')
            {
                m_sv.remove_prefix(1);
                if (!ParseLogicalOr(out)) return false;
                if (!Match(")")) return false;
                return true;
            }

            if (std::isdigit(static_cast<unsigned char>(m_sv.front())) != 0)
            {
                size_t len = 0;
                int base   = 10;
                if (m_sv.size() >= 2 && m_sv[0] == '0' && (m_sv[1] == 'x' || m_sv[1] == 'X'))
                {
                    base = 16;
                    len  = 2;
                    while (len < m_sv.size() && std::isxdigit(static_cast<unsigned char>(m_sv[len])) != 0)
                        ++len;
                    if (len == 2) return false;
                }
                else
                {
                    while (len < m_sv.size() && std::isdigit(static_cast<unsigned char>(m_sv[len])) != 0)
                        ++len;
                }
                std::string numStr(m_sv.substr(0, len));
                m_sv.remove_prefix(len);
                while (!m_sv.empty() &&
                       (m_sv.front() == 'u' || m_sv.front() == 'U' || m_sv.front() == 'l' || m_sv.front() == 'L'))
                {
                    m_sv.remove_prefix(1);
                }
                try
                {
                    out = static_cast<int64_t>(std::stoull(numStr, nullptr, base));
                    return true;
                }
                catch (...)
                {
                    return false;
                }
            }

            if (IsIdentStart(m_sv.front()))
            {
                std::string_view id = ConsumeIdentifier(m_sv);
                if (id == "defined")
                {
                    SkipWS();
                    bool paren = false;
                    if (!m_sv.empty() && m_sv.front() == '(')
                    {
                        paren = true;
                        m_sv.remove_prefix(1);
                    }
                    std::string_view macroName = ConsumeIdentifier(m_sv);
                    if (macroName.empty()) return false;
                    if (paren && !Match(")")) return false;
                    out = (m_macros.find(std::string(macroName)) != m_macros.end()) ? 1 : 0;
                    return true;
                }
                if (id == "true")  { out = 1; return true; }
                if (id == "false") { out = 0; return true; }
                auto it = m_macros.find(std::string(id));
                out = (it != m_macros.end()) ? it->second : 0;
                return true;
            }

            return false;
        }
    };
} // namespace

// =====================================================================
//  Helpers
// =====================================================================

bool Preprocessor::HasNslExtension(const std::filesystem::path& path)
{
    return EqualsIgnoreCase(path.extension().string(), ".nsl");
}

bool Preprocessor::IsIncludeDirective(std::string_view trimmed)
{
    if (trimmed.empty() || trimmed[0] != '#') return false;

    size_t i = 1;
    while (i < trimmed.size() && (trimmed[i] == ' ' || trimmed[i] == '\t'))
        ++i;

    constexpr std::string_view kInclude = "include";
    if (trimmed.substr(i, kInclude.size()) != kInclude)
        return false;

    const size_t after = i + kInclude.size();
    if (after < trimmed.size())
    {
        const unsigned char c = static_cast<unsigned char>(trimmed[after]);
        if (std::isalnum(c) || c == '_')
            return false;
    }
    return true;
}

void Preprocessor::AddIncludeDirectory(const std::filesystem::path& dir)
{
    m_includeDirs.push_back(dir);
}

void Preprocessor::DefineMacro(const std::string& name, int64_t value)
{
    if (!name.empty())
    {
        m_cliMacros[name] = value;
    }
}

void Preprocessor::ClearMacroDefines()
{
    m_cliMacros.clear();
}

bool Preprocessor::EvaluateIfExpression(
    std::string_view  expr,
    uint32_t          lineNum,
    DiagnosticEngine& diag,
    int64_t&          outValue) const
{
    std::string_view clean = StripLineCommentAndTrim(expr);
    if (clean.empty())
    {
        diag.ReportError(lineNum, 0, "Expected constant integer expression after #if / #elif.");
        return false;
    }
    PreprocExprParser parser(clean, m_activeMacros);
    if (!parser.Parse(outValue))
    {
        diag.ReportError(lineNum, 0, "Invalid preprocessor conditional expression: '" + std::string(clean) + "'.");
        return false;
    }
    return true;
}

bool Preprocessor::IsUnderRoot(const std::filesystem::path& candidate) const
{
    auto isUnder = [](const std::filesystem::path& root, const std::filesystem::path& cand) {
        auto rootIt = root.begin();
        auto candIt = cand.begin();
        for (; rootIt != root.end(); ++rootIt, ++candIt)
        {
            if (candIt == cand.end())
                return false;
            if (*rootIt != *candIt)
                return false;
        }
        return true;
    };

    if (isUnder(m_rootDir, candidate))
        return true;

    for (const auto& inc : m_canonicalIncludeDirs)
    {
        if (isUnder(inc, candidate))
            return true;
    }

    return false;
}

bool Preprocessor::TryCanonicalize(
    const std::filesystem::path& path,
    std::filesystem::path&       outCanonical,
    DiagnosticEngine&            diag,
    uint32_t                     lineNum,
    const std::string&           what) const
{
    try
    {
        outCanonical = std::filesystem::weakly_canonical(path).make_preferred();
        return true;
    }
    catch (const std::filesystem::filesystem_error&)
    {
        diag.ReportError(lineNum, 0, what);
        return false;
    }
    catch (...)
    {
        diag.ReportError(lineNum, 0, what);
        return false;
    }
}

// =====================================================================
//  Preprocessor::Process
// =====================================================================

std::string Preprocessor::Process(
    const std::string&  source,
    const std::string&  baseDir,
    const std::string&  sourceFile,
    DiagnosticEngine&   diag)
{
    m_activeIncludes.clear();
    m_emittedIncludes.clear();
    m_canonicalIncludeDirs.clear();
    m_activeMacros = m_cliMacros;
    m_totalBytes   = 0;
    m_rootDir.clear();

    if (source.size() > kMaxSingleFileBytes)
    {
        diag.ReportError(0, 0,
            "Source file exceeds maximum size (" +
            std::to_string(kMaxSingleFileBytes / (1024 * 1024)) + " MiB).");
        return {};
    }

    if (!TryCanonicalize(
            std::filesystem::path(baseDir),
            m_rootDir,
            diag,
            0,
            "Cannot resolve source root directory: \"" + baseDir + "\""))
    {
        return {};
    }

    for (const auto& inc : m_includeDirs)
    {
        std::filesystem::path canInc;
        if (TryCanonicalize(inc, canInc, diag, 0, "Cannot resolve include directory: \"" + inc.string() + "\""))
        {
            m_canonicalIncludeDirs.push_back(canInc);
        }
    }

    std::filesystem::path rootFile;
    if (!TryCanonicalize(
            std::filesystem::path(baseDir) / sourceFile,
            rootFile,
            diag,
            0,
            "Cannot resolve source file path: \"" + sourceFile + "\""))
    {
        return {};
    }

    if (!IsUnderRoot(rootFile))
    {
        diag.ReportError(0, 0,
            "Source file escapes the source root directory.");
        return {};
    }

    m_activeIncludes.insert(rootFile.string());
    m_totalBytes = source.size();

    return ResolveIncludes(source, m_rootDir.string(), diag, 0);
}

// =====================================================================
//  Preprocessor::ResolveIncludes  (recursive + line-preserving conditionals)
// =====================================================================

std::string Preprocessor::ResolveIncludes(
    const std::string&  source,
    const std::string&  baseDir,
    DiagnosticEngine&   diag,
    int                 depth)
{
    if (depth > kMaxIncludeDepth)
    {
        diag.ReportError(0, 0,
            "Maximum #include depth (" + std::to_string(kMaxIncludeDepth) +
            ") exceeded. Circular include?");
        return {};
    }

    struct CondState
    {
        bool     parentActive = true;
        bool     branchTaken  = false;
        bool     curActive    = true;
        bool     seenElse     = false;
        uint32_t startLine    = 0;
    };
    std::vector<CondState> condStack;

    auto isCurrentlyActive = [&]() noexcept -> bool {
        return condStack.empty() ? true : condStack.back().curActive;
    };

    std::ostringstream result;
    std::istringstream stream(source);
    std::string        line;
    uint32_t           lineNum = 0;

    while (std::getline(stream, line))
    {
        ++lineNum;

        size_t firstNonSpace = line.find_first_not_of(" \t\r");
        if (firstNonSpace == std::string::npos)
        {
            result << (isCurrentlyActive() ? line : std::string{}) << "\n";
            continue;
        }

        std::string_view trimmed(line.data() + firstNonSpace, line.size() - firstNonSpace);

        // ==================== Conditional & Macro Directives ====================
        if (trimmed.front() == '#' && !IsIncludeDirective(trimmed))
        {
            std::string_view rest = trimmed.substr(1);
            std::string_view kw   = ConsumeIdentifier(rest);

            if (kw == "ifdef" || kw == "ifndef")
            {
                std::string_view restClean = StripLineCommentAndTrim(rest);
                std::string_view macroName = ConsumeIdentifier(restClean);
                restClean = StripLineCommentAndTrim(restClean);
                if (macroName.empty() || !restClean.empty())
                {
                    diag.ReportError(lineNum, 0, "Malformed #" + std::string(kw) + " directive.");
                    return {};
                }
                const bool parentAct = isCurrentlyActive();
                const bool isDef     = (m_activeMacros.find(std::string(macroName)) != m_activeMacros.end());
                const bool condTrue  = (kw == "ifdef") ? isDef : !isDef;
                CondState st{};
                st.parentActive = parentAct;
                st.branchTaken  = parentAct && condTrue;
                st.curActive    = st.branchTaken;
                st.startLine    = lineNum;
                condStack.push_back(st);
                result << "\n";
                continue;
            }
            if (kw == "if")
            {
                const bool parentAct = isCurrentlyActive();
                int64_t val = 0;
                if (parentAct)
                {
                    if (!EvaluateIfExpression(rest, lineNum, diag, val))
                        return {};
                }
                CondState st{};
                st.parentActive = parentAct;
                st.branchTaken  = parentAct && (val != 0);
                st.curActive    = st.branchTaken;
                st.startLine    = lineNum;
                condStack.push_back(st);
                result << "\n";
                continue;
            }
            if (kw == "elif")
            {
                if (condStack.empty() || condStack.back().seenElse)
                {
                    diag.ReportError(lineNum, 0, "Unexpected #elif without matching #if.");
                    return {};
                }
                CondState& st = condStack.back();
                if (!st.parentActive || st.branchTaken)
                {
                    st.curActive = false;
                }
                else
                {
                    int64_t val = 0;
                    if (!EvaluateIfExpression(rest, lineNum, diag, val))
                        return {};
                    st.curActive   = (val != 0);
                    st.branchTaken = st.curActive;
                }
                result << "\n";
                continue;
            }
            if (kw == "else")
            {
                std::string_view extra = StripLineCommentAndTrim(rest);
                if (condStack.empty() || condStack.back().seenElse || !extra.empty())
                {
                    diag.ReportError(lineNum, 0, "Unexpected or malformed #else directive.");
                    return {};
                }
                CondState& st = condStack.back();
                st.seenElse   = true;
                st.curActive  = st.parentActive && !st.branchTaken;
                st.branchTaken = true;
                result << "\n";
                continue;
            }
            if (kw == "endif")
            {
                std::string_view extra = StripLineCommentAndTrim(rest);
                if (condStack.empty() || !extra.empty())
                {
                    diag.ReportError(lineNum, 0, "Unexpected or malformed #endif directive.");
                    return {};
                }
                condStack.pop_back();
                result << "\n";
                continue;
            }
            if (kw == "define")
            {
                if (isCurrentlyActive())
                {
                    std::string_view restClean = StripLineCommentAndTrim(rest);
                    std::string_view macroName = ConsumeIdentifier(restClean);
                    if (macroName.empty())
                    {
                        diag.ReportError(lineNum, 0, "Malformed #define directive: missing macro identifier.");
                        return {};
                    }
                    // CLI -D overrides take precedence if already set? No, in C/HLSL #ifndef NAME / #define NAME VAL
                    // only runs when CLI did not define NAME; if #define runs in active branch, evaluate value cleanly.
                    std::string_view valExpr = StripLineCommentAndTrim(restClean);
                    int64_t val = 1;
                    if (!valExpr.empty())
                    {
                        if (!EvaluateIfExpression(valExpr, lineNum, diag, val))
                            return {};
                    }
                    m_activeMacros[std::string(macroName)] = val;
                }
                result << "\n";
                continue;
            }
            if (kw == "undef")
            {
                if (isCurrentlyActive())
                {
                    std::string_view restClean = StripLineCommentAndTrim(rest);
                    std::string_view macroName = ConsumeIdentifier(restClean);
                    restClean = StripLineCommentAndTrim(restClean);
                    if (macroName.empty() || !restClean.empty())
                    {
                        diag.ReportError(lineNum, 0, "Malformed #undef directive.");
                        return {};
                    }
                    m_activeMacros.erase(std::string(macroName));
                }
                result << "\n";
                continue;
            }
        }

        if (!isCurrentlyActive())
        {
            // Preserve exact 1:1 line numbering across inactive conditional branches
            result << "\n";
            continue;
        }

        // ==================== Check for #include ====================
        if (IsIncludeDirective(trimmed))
        {
            // Extract path between quotes
            size_t openQ  = line.find('"');
            size_t closeQ = (openQ != std::string::npos)
                                ? line.find('"', openQ + 1)
                                : std::string::npos;

            if (openQ == std::string::npos || closeQ == std::string::npos)
            {
                diag.ReportError(lineNum, 0,
                    "Malformed #include directive. Use: #include \"filename.nsl\"");
                return {};
            }

            std::string includePath = line.substr(openQ + 1, closeQ - openQ - 1);
            if (includePath.empty())
            {
                diag.ReportError(lineNum, 0,
                    "Empty #include path. Use: #include \"filename.nsl\"");
                return {};
            }

            std::filesystem::path includeFsPath(includePath);
            if (includeFsPath.is_absolute())
            {
                diag.ReportError(lineNum, 0,
                    "Absolute #include paths are not allowed: \"" + includePath + "\"");
                return {};
            }

            if (!HasNslExtension(includeFsPath))
            {
                diag.ReportError(lineNum, 0,
                    "Only .nsl files may be #included: \"" + includePath + "\"");
                return {};
            }

            // Resolve relative to the current file's directory first;
            // if not found, search the registered include directories.
            std::filesystem::path absPath =
                std::filesystem::path(baseDir) / includeFsPath;

            if (!std::filesystem::exists(absPath))
            {
                for (const auto& incDir : m_canonicalIncludeDirs)
                {
                    std::filesystem::path candidate = incDir / includeFsPath;
                    if (std::filesystem::exists(candidate))
                    {
                        absPath = candidate;
                        break;
                    }
                }
            }

            std::filesystem::path canonicalPath;
            if (!TryCanonicalize(
                    absPath,
                    canonicalPath,
                    diag,
                    lineNum,
                    "Cannot resolve #include path: \"" + includePath + "\""))
            {
                return {};
            }

            if (!IsUnderRoot(canonicalPath))
            {
                diag.ReportError(lineNum, 0,
                    "#include path escapes the source root or registered include directories: \"" +
                    includePath + "\"");
                return {};
            }

            const std::string absPathStr = canonicalPath.string();

            // Active ancestry means a real cycle. A path already emitted
            // through another branch is an include-once no-op.
            if (m_activeIncludes.count(absPathStr))
            {
                diag.ReportError(lineNum, 0,
                    "Circular #include detected: \"" + includePath + "\"");
                return {};
            }
            if (m_emittedIncludes.count(absPathStr))
            {
                result << "\n";
                continue;
            }

            // Read included file with per-file and cumulative budgets
            std::ifstream ifs;
            try
            {
                ifs.open(canonicalPath, std::ios::binary);
            }
            catch (...)
            {
                diag.ReportError(lineNum, 0,
                    "Cannot open #include file: \"" + includePath + "\"");
                return {};
            }

            if (!ifs.is_open())
            {
                diag.ReportError(lineNum, 0,
                    "Cannot open #include file: \"" + includePath + "\"");
                return {};
            }

            std::streamoff fileSize = -1;
            try
            {
                ifs.seekg(0, std::ios::end);
                fileSize = ifs.tellg();
                ifs.seekg(0, std::ios::beg);
            }
            catch (...)
            {
                diag.ReportError(lineNum, 0,
                    "Cannot read #include file: \"" + includePath + "\"");
                return {};
            }

            if (fileSize < 0)
            {
                diag.ReportError(lineNum, 0,
                    "Cannot determine size of #include file: \"" + includePath + "\"");
                return {};
            }

            const size_t byteCount = static_cast<size_t>(fileSize);
            if (byteCount > kMaxSingleFileBytes)
            {
                diag.ReportError(lineNum, 0,
                    "#include file exceeds maximum size (" +
                    std::to_string(kMaxSingleFileBytes / (1024 * 1024)) +
                    " MiB): \"" + includePath + "\"");
                return {};
            }

            if (m_totalBytes > kMaxTotalSourceBytes ||
                byteCount > (kMaxTotalSourceBytes - m_totalBytes))
            {
                diag.ReportError(lineNum, 0,
                    "Include-expanded source exceeds maximum size (" +
                    std::to_string(kMaxTotalSourceBytes / (1024 * 1024)) +
                    " MiB).");
                return {};
            }

            std::string includedSource;
            try
            {
                includedSource.assign(byteCount, '\0');
                if (byteCount > 0)
                {
                    ifs.read(includedSource.data(),
                             static_cast<std::streamsize>(byteCount));
                    if (!ifs)
                    {
                        diag.ReportError(lineNum, 0,
                            "Cannot read #include file: \"" + includePath + "\"");
                        return {};
                    }
                }
            }
            catch (...)
            {
                diag.ReportError(lineNum, 0,
                    "Cannot read #include file: \"" + includePath + "\"");
                return {};
            }

            if (includedSource.size() >= 3 &&
                static_cast<unsigned char>(includedSource[0]) == 0xEF &&
                static_cast<unsigned char>(includedSource[1]) == 0xBB &&
                static_cast<unsigned char>(includedSource[2]) == 0xBF)
            {
                includedSource.erase(0, 3);
            }

            m_totalBytes += byteCount;
            m_activeIncludes.insert(absPathStr);

            const std::string includedBaseDir =
                canonicalPath.parent_path().string();

            std::string expanded = ResolveIncludes(
                includedSource, includedBaseDir, diag, depth + 1);

            m_activeIncludes.erase(absPathStr);
            if (diag.HasErrors())
                return {};
            m_emittedIncludes.insert(absPathStr);

            result << expanded;
            result << "\n";
        }
        else
        {
            result << line << "\n";
        }

        if (diag.HasErrors())
            return {};
    }

    if (!condStack.empty())
    {
        diag.ReportError(condStack.back().startLine, 0,
            "Unterminated preprocessor conditional (#if / #ifdef / #ifndef) missing #endif.");
        return {};
    }

    return result.str();
}

} // namespace NSL
