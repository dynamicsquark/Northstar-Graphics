#include "nsl_library_resolver.h"
#include "nsl_preprocessor.h"
#include <cctype>
#include <fstream>
#include <sstream>

namespace NSL
{

namespace
{
    bool HasNslExtensionCaseInsensitive(const std::filesystem::path& p)
    {
        const std::string ext = p.extension().string();
        if (ext.size() != 4) return false;
        return ext[0] == '.' &&
               std::tolower(static_cast<unsigned char>(ext[1])) == 'n' &&
               std::tolower(static_cast<unsigned char>(ext[2])) == 's' &&
               std::tolower(static_cast<unsigned char>(ext[3])) == 'l';
    }

    bool IsPathUnderAnyRoot(
        const std::filesystem::path&              candidate,
        const std::vector<std::filesystem::path>& allowedRoots)
    {
        auto isUnder = [](const std::filesystem::path& root, const std::filesystem::path& cand) -> bool
        {
            auto rootIt = root.begin();
            auto candIt = cand.begin();
            for (; rootIt != root.end(); ++rootIt, ++candIt)
            {
                if (candIt == cand.end() || *rootIt != *candIt)
                    return false;
            }
            return true;
        };

        for (const auto& r : allowedRoots)
        {
            if (!r.empty() && isUnder(r, candidate))
                return true;
        }
        return false;
    }

    std::string EscapeLineDirectiveFile(const std::string& fileName)
    {
        std::string escaped;
        escaped.reserve(fileName.size());
        for (char c : fileName)
        {
            if (c == '\\') escaped += '/';
            else if (c != '"' && c != '\r' && c != '\n') escaped += c;
        }
        return escaped;
    }

    std::unordered_set<std::string> ExtractStageIOTokens(const std::string& rawText)
    {
        std::unordered_set<std::string> tokens;
        size_t i = 0;
        const size_t n = rawText.size();
        while (i < n)
        {
            if (rawText[i] == '/' && i + 1 < n && rawText[i + 1] == '/')
            {
                while (i < n && rawText[i] != '\n') ++i;
                continue;
            }
            if (rawText[i] == '/' && i + 1 < n && rawText[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < n && !(rawText[i] == '*' && rawText[i + 1] == '/')) ++i;
                i = (i + 1 < n) ? (i + 2) : n;
                continue;
            }
            if (rawText[i] == '(')
            {
                const size_t closePos = rawText.find(')', i + 1);
                if (closePos == std::string::npos) break;
                std::string inner = rawText.substr(i + 1, closePos - i - 1);
                const size_t first = inner.find_first_not_of(" \t\r\n");
                const size_t last  = inner.find_last_not_of(" \t\r\n");
                if (first != std::string::npos && last != std::string::npos)
                {
                    tokens.insert(inner.substr(first, last - first + 1));
                }
                i = closePos + 1;
                continue;
            }
            ++i;
        }
        return tokens;
    }
} // namespace

bool LibraryResolver::Resolve(
    NSLFileLayout&                            mainLayout,
    const std::string&                        baseDir,
    const std::vector<std::filesystem::path>& includeDirs,
    DiagnosticEngine&                         diag)
{
    if (mainLayout.libraryDeclarations.empty())
        return true;

    std::vector<std::filesystem::path> allowedRoots;
    auto tryAddRoot = [&](const std::filesystem::path& p)
    {
        if (p.empty()) return;
        try
        {
            if (std::filesystem::exists(p))
                allowedRoots.push_back(std::filesystem::weakly_canonical(p).make_preferred());
        }
        catch (...) {}
    };

    const std::filesystem::path baseFsPath(baseDir);
    tryAddRoot(baseFsPath);
    tryAddRoot(baseFsPath.parent_path());
    for (const auto& inc : includeDirs)
        tryAddRoot(inc);

    std::unordered_set<std::string> activeChain;
    std::unordered_set<std::string> visited;

    NSLFileLayout libraryAccumulator;
    const std::string mainFileName = diag.GetFile();

    for (const auto& libPath : mainLayout.libraryDeclarations)
    {
        if (!ResolveFile(libPath, baseDir, allowedRoots, libraryAccumulator, activeChain, visited, includeDirs, diag))
            return false;
    }

    // Merge properties: main shader properties come first (preserving primary Material block as set 0),
    // then library property blocks are appended or merged into existing blocks.
    std::vector<PropertyBlockRaw> mergedProps;
    for (const auto& mainProp : mainLayout.propertyBlocks)
    {
        MergePropertyBlock(mergedProps, mainProp);
    }
    for (const auto& libProp : libraryAccumulator.propertyBlocks)
    {
        MergePropertyBlock(mergedProps, libProp);
    }
    mainLayout.propertyBlocks = std::move(mergedProps);

    // Merge IO layout entries
    for (const auto& libIO : libraryAccumulator.stageIOBlocks)
    {
        MergeIOBlock(mainLayout.stageIOBlocks, libIO);
    }

    // Prepend library code to main shader's SHADER block with #line markers so both
    // library diagnostics and main-shader diagnostics report exact file and line numbers.
    if (!libraryAccumulator.librarySource.empty())
    {
        const uint32_t mainLine = (mainLayout.shaderStartLine > 0u) ? mainLayout.shaderStartLine : 1u;
        const std::string mainMarker = "#line " + std::to_string(mainLine) +
            " \"" + EscapeLineDirectiveFile(mainFileName) + "\"\n";
        mainLayout.shaderSource = libraryAccumulator.librarySource + "\n" + mainMarker + mainLayout.shaderSource;
        mainLayout.shaderStartLine = 1u;
    }

    return true;
}

bool LibraryResolver::ResolveFile(
    const std::string&                        libPath,
    const std::string&                        currentBaseDir,
    const std::vector<std::filesystem::path>& allowedRoots,
    NSLFileLayout&                            outMergedLayout,
    std::unordered_set<std::string>&          activeChain,
    std::unordered_set<std::string>&          visited,
    const std::vector<std::filesystem::path>& includeDirs,
    DiagnosticEngine&                         diag)
{
    if (libPath.empty())
    {
        diag.ReportError(0, 0, "Empty library path in LIBRARY_DECLARATION.");
        return false;
    }

    const std::filesystem::path libFsPath(libPath);
    if (libFsPath.is_absolute())
    {
        diag.ReportError(0, 0, "Absolute library paths are not allowed in LIBRARY_DECLARATION: \"" + libPath + "\"");
        return false;
    }

    if (!HasNslExtensionCaseInsensitive(libFsPath))
    {
        diag.ReportError(0, 0, "Only .nsl files may be imported in LIBRARY_DECLARATION: \"" + libPath + "\"");
        return false;
    }

    std::filesystem::path resolvedPath = std::filesystem::path(currentBaseDir) / libFsPath;
    if (!std::filesystem::exists(resolvedPath))
    {
        bool found = false;
        for (const auto& inc : includeDirs)
        {
            std::filesystem::path candidate = inc / libFsPath;
            if (std::filesystem::exists(candidate))
            {
                resolvedPath = candidate;
                found = true;
                break;
            }
        }
        if (!found)
        {
            diag.ReportError(0, 0, "Cannot resolve library path in LIBRARY_DECLARATION: \"" + libPath + "\"");
            return false;
        }
    }

    std::filesystem::path canonicalPath;
    try
    {
        canonicalPath = std::filesystem::weakly_canonical(resolvedPath).make_preferred();
    }
    catch (...)
    {
        diag.ReportError(0, 0, "Failed to canonicalize library path: \"" + libPath + "\"");
        return false;
    }

    if (!IsPathUnderAnyRoot(canonicalPath, allowedRoots))
    {
        diag.ReportError(0, 0,
            "Library path escapes the source root or registered include directories: \"" + libPath + "\"");
        return false;
    }

    std::string canStr = canonicalPath.string();
    if (activeChain.count(canStr))
    {
        diag.ReportError(0, 0, "Circular library dependency detected: \"" + libPath + "\"");
        return false;
    }

    if (visited.count(canStr))
    {
        return true; // Already imported once (diamond import)
    }

    activeChain.insert(canStr);

    std::ifstream inFile(canonicalPath, std::ios::binary | std::ios::ate);
    if (!inFile.is_open())
    {
        diag.ReportError(0, 0, "Cannot open library file: \"" + libPath + "\"");
        return false;
    }

    const std::streamoff fsize = inFile.tellg();
    if (fsize < 0 || fsize > 10 * 1024 * 1024)
    {
        diag.ReportError(0, 0, "Library file size invalid or exceeds maximum (10 MiB): \"" + libPath + "\"");
        return false;
    }
    inFile.seekg(0, std::ios::beg);

    std::ostringstream ss;
    ss << inFile.rdbuf();
    std::string rawLibSource = ss.str();
    if (rawLibSource.size() >= 3 &&
        static_cast<unsigned char>(rawLibSource[0]) == 0xEF &&
        static_cast<unsigned char>(rawLibSource[1]) == 0xBB &&
        static_cast<unsigned char>(rawLibSource[2]) == 0xBF)
    {
        rawLibSource.erase(0, 3);
    }

    const std::string libFileName = canonicalPath.filename().string();
    const std::string libParentDir = canonicalPath.parent_path().string();
    const std::string prevDiagFile = diag.GetFile();
    diag.SetFile(libFileName);

    Preprocessor libPreprocessor;
    for (const auto& root : allowedRoots)
        libPreprocessor.AddIncludeDirectory(root);
    for (const auto& inc : includeDirs)
        libPreprocessor.AddIncludeDirectory(inc);

    std::string libSource = libPreprocessor.Process(rawLibSource, libParentDir, libFileName, diag);
    if (diag.HasErrors())
    {
        diag.SetFile(prevDiagFile);
        return false;
    }

    NSLFileParser parser(libSource);
    NSLFileLayout thisLibLayout;
    if (!parser.Parse(thisLibLayout, diag))
    {
        diag.SetFile(prevDiagFile);
        return false;
    }
    diag.SetFile(prevDiagFile);

    if (!thisLibLayout.isLibrary && thisLibLayout.stage != ShaderStage::NONE)
    {
        diag.ReportError(0, 0, "File \"" + libPath + "\" is a main shader with STAGE " +
            std::string(ShaderStageToString(thisLibLayout.stage)) +
            ", but is imported in LIBRARY_DECLARATION. Only LIBRARY shaders can be imported.");
        return false;
    }

    // Recursively resolve sub-libraries
    for (const auto& subLib : thisLibLayout.libraryDeclarations)
    {
        if (!ResolveFile(subLib, libParentDir, allowedRoots, outMergedLayout, activeChain, visited, includeDirs, diag))
            return false;
    }

    // Merge properties
    for (const auto& prop : thisLibLayout.propertyBlocks)
    {
        MergePropertyBlock(outMergedLayout.propertyBlocks, prop);
    }

    // Merge IO blocks
    for (const auto& io : thisLibLayout.stageIOBlocks)
    {
        MergeIOBlock(outMergedLayout.stageIOBlocks, io);
    }

    // Accumulate library code with exact #line directive
    if (!thisLibLayout.librarySource.empty())
    {
        if (!outMergedLayout.librarySource.empty())
            outMergedLayout.librarySource += "\n";
        const uint32_t libLine = (thisLibLayout.libraryStartLine > 0u) ? thisLibLayout.libraryStartLine : 1u;
        outMergedLayout.librarySource += "#line " + std::to_string(libLine) +
            " \"" + EscapeLineDirectiveFile(libFileName) + "\"\n" + thisLibLayout.librarySource;
    }

    activeChain.erase(canStr);
    visited.insert(canStr);
    return true;
}

void LibraryResolver::MergePropertyBlock(
    std::vector<PropertyBlockRaw>& targetBlocks,
    const PropertyBlockRaw& incomingBlock)
{
    for (auto& existing : targetBlocks)
    {
        if (existing.name == incomingBlock.name)
        {
            existing.rawText += "\n" + incomingBlock.rawText;
            return;
        }
    }
    targetBlocks.push_back(incomingBlock);
}

void LibraryResolver::MergeIOBlock(
    std::vector<StageIOBlockRaw>& targetBlocks,
    const StageIOBlockRaw& incomingBlock)
{
    for (auto& existing : targetBlocks)
    {
        if (existing.kind == incomingBlock.kind)
        {
            std::unordered_set<std::string> existingTokens = ExtractStageIOTokens(existing.rawText);
            std::istringstream stream(incomingBlock.rawText);
            std::string line;
            while (std::getline(stream, line))
            {
                const size_t commentPos = line.find("//");
                const std::string_view codePart = (commentPos != std::string::npos)
                    ? std::string_view(line.data(), commentPos)
                    : std::string_view(line);

                const size_t pOpen  = codePart.find('(');
                const size_t pClose = (pOpen != std::string_view::npos) ? codePart.find(')', pOpen + 1) : std::string_view::npos;
                if (pOpen != std::string_view::npos && pClose != std::string_view::npos && pClose > pOpen + 1)
                {
                    const std::string_view token = codePart.substr(pOpen + 1, pClose - pOpen - 1);
                    const size_t first = token.find_first_not_of(" \t\r\n");
                    const size_t last  = token.find_last_not_of(" \t\r\n");
                    if (first != std::string_view::npos && last != std::string_view::npos)
                    {
                        std::string trimmedToken(token.substr(first, last - first + 1));
                        if (existingTokens.insert(trimmedToken).second)
                        {
                            existing.rawText += "\n    " + line;
                        }
                    }
                }
            }
            return;
        }
    }
    targetBlocks.push_back(incomingBlock);
}

} // namespace NSL

