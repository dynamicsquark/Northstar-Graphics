#pragma once

#include "nsl_error.h"
#include "nsl_file_parser.h"
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

namespace NSL
{
    class LibraryResolver
    {
    public:
        // Resolves all library declarations in mainLayout recursively,
        // merging property blocks, IO layout entries, and library code.
        static bool Resolve(
            NSLFileLayout&                            mainLayout,
            const std::string&                        baseDir,
            const std::vector<std::filesystem::path>& includeDirs,
            DiagnosticEngine&                         diag
        );

    private:
        static bool ResolveFile(
            const std::string&                        libPath,
            const std::string&                        currentBaseDir,
            const std::vector<std::filesystem::path>& allowedRoots,
            NSLFileLayout&                            outMergedLayout,
            std::unordered_set<std::string>&          activeChain,
            std::unordered_set<std::string>&          visited,
            const std::vector<std::filesystem::path>& includeDirs,
            DiagnosticEngine&                         diag
        );

        static void MergePropertyBlock(
            std::vector<PropertyBlockRaw>&       targetBlocks,
            const PropertyBlockRaw&              incomingBlock
        );

        static void MergeIOBlock(
            std::vector<StageIOBlockRaw>&        targetBlocks,
            const StageIOBlockRaw&               incomingBlock
        );
    };
} // namespace NSL

