#pragma once

#include "nsl_error.h"
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// =====================================================================
//  NSL Preprocessor
//  Handles #include "file.nsl" directives via C-style text paste.
//  Also strips block-comment lines to preserve line numbers.
//
//  Rules:
//    - Only relative paths with double quotes: #include "path/to/file.nsl"
//    - Absolute paths are rejected.
//    - Angle-bracket includes (<>) are not supported.
//    - Includes must resolve under the root source directory from Process
//      (nested includes allowed; canonical paths that escape the root
//      are rejected).
//    - Only ".nsl" includes are allowed.
//    - Circular includes are rejected; repeated non-cyclic includes are
//      emitted once.
//    - Max include depth: 32 (prevents infinite recursion).
//    - Cumulative include-expanded source budget: 10 MiB; per-file cap
//      matches that budget.
//    - Included file content is inserted in-place; line numbers in the
//      merged source reflect the included files' own line numbers.
// =====================================================================

namespace NSL
{
    class Preprocessor
    {
    public:
        // Add an include search directory (e.g. from -I / --include-dir).
        void AddIncludeDirectory(const std::filesystem::path& dir);

        // Define or override an integer preprocessor macro (e.g. from -D NAME=VALUE).
        void DefineMacro(const std::string& name, int64_t value = 1);
        void ClearMacroDefines();

        // Process source text rooted at baseDir.
        // Returns the merged source with all #includes and conditional directives resolved.
        // On error, diagnostics are reported and an empty string is returned.
        std::string Process(
            const std::string&  source,
            const std::string&  baseDir,
            const std::string&  sourceFile,
            DiagnosticEngine&   diag
        );

    private:
        static constexpr int    kMaxIncludeDepth      = 32;
        static constexpr size_t kMaxTotalSourceBytes  = 10u * 1024u * 1024u; // 10 MiB
        static constexpr size_t kMaxSingleFileBytes   = 10u * 1024u * 1024u; // 10 MiB

        std::string ResolveIncludes(
            const std::string&  source,
            const std::string&  baseDir,
            DiagnosticEngine&   diag,
            int                 depth
        );

        bool EvaluateIfExpression(
            std::string_view  expr,
            uint32_t          lineNum,
            DiagnosticEngine& diag,
            int64_t&          outValue
        ) const;

        bool TryCanonicalize(
            const std::filesystem::path& path,
            std::filesystem::path&       outCanonical,
            DiagnosticEngine&            diag,
            uint32_t                     lineNum,
            const std::string&           what
        ) const;

        bool IsUnderRoot(const std::filesystem::path& candidate) const;

        static bool HasNslExtension(const std::filesystem::path& path);
        static bool IsIncludeDirective(std::string_view trimmed);

        // Root source directory established by Process (canonical).
        std::filesystem::path m_rootDir;

        // Additional include search directories (canonical).
        std::vector<std::filesystem::path> m_includeDirs;
        std::vector<std::filesystem::path> m_canonicalIncludeDirs;

        // Cumulative bytes of root + all successfully read include files.
        size_t m_totalBytes = 0;

        // Active ancestry detects real cycles; emitted paths implement
        // include-once behavior for valid diamond include graphs.
        std::unordered_set<std::string>          m_activeIncludes;
        std::unordered_set<std::string>          m_emittedIncludes;
        std::unordered_map<std::string, int64_t> m_cliMacros;
        std::unordered_map<std::string, int64_t> m_activeMacros;
    };

} // namespace NSL
