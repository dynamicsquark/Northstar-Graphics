#include "nsl_preprocessor.h"

#include <cctype>
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
    m_totalBytes = 0;
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
//  Preprocessor::ResolveIncludes  (recursive)
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

    std::ostringstream result;
    std::istringstream stream(source);
    std::string        line;
    uint32_t           lineNum = 0;

    while (std::getline(stream, line))
    {
        ++lineNum;

        // ==================== Check for #include ====================
        // Trim leading whitespace
        size_t firstNonSpace = line.find_first_not_of(" \t");
        if (firstNonSpace == std::string::npos)
        {
            result << line << "\n";
            continue;
        }

        std::string_view trimmed(line.data() + firstNonSpace,
                                  line.size()  - firstNonSpace);

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

            // Nested includes resolve relative to this file, but remain
            // confined to the Process root via IsUnderRoot checks.
            const std::string includedBaseDir =
                canonicalPath.parent_path().string();

            std::string expanded = ResolveIncludes(
                includedSource, includedBaseDir, diag, depth + 1);

            m_activeIncludes.erase(absPathStr);
            if (diag.HasErrors())
                return {};
            m_emittedIncludes.insert(absPathStr);

            result << expanded;

            // After include content, preserve a blank line for the #include line itself
            result << "\n";
        }
        else
        {
            result << line << "\n";
        }

        if (diag.HasErrors())
            return {};
    }

    return result.str();
}

} // namespace NSL
