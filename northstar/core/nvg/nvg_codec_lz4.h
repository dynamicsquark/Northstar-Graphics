#pragma once

#include "../../include/types/northstar_types.h"
#include <cstddef>
#include <vector>

namespace Northstar
{
    // Clean-room Castagnoli CRC32C, 64-bit FNV-1a, and raw LZ4 block codec for NVG2 32 KiB geometry pages.
    class NVGCodecLZ4
    {
    public:
        static UINT32 ComputeCRC32C(const BYTE* data, SIZE_T size, UINT32 seed = 0u) noexcept;
        static UINT64 ComputeFNV1a64(const BYTE* data, SIZE_T size, UINT64 seed = 14695981039346656037ULL) noexcept;

        static SIZE_T MaxCompressedBound(SIZE_T inputSize) noexcept;

        // Compresses raw input bytes into standard LZ4 block format.
        // Returns compressed byte count (> 0) on success, or 0 on invalid arguments / insufficient dstCapacity.
        static SIZE_T CompressBlock(
            const BYTE* src,
            SIZE_T      srcSize,
            BYTE*       dst,
            SIZE_T      dstCapacity) noexcept;

        static BOOL CompressToVector(
            const BYTE*        src,
            SIZE_T             srcSize,
            std::vector<BYTE>& outCompressed);

        // Safe bounded LZ4 block decompressor. Rejects truncated streams, invalid zero/OOB match offsets,
        // and output overruns deterministically without out-of-bounds memory access.
        static BOOL DecompressBlockExact(
            const BYTE* src,
            SIZE_T      srcSize,
            BYTE*       dst,
            SIZE_T      expectedUncompressedSize) noexcept;
    };
}
