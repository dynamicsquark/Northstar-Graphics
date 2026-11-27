#include "nvg_codec_lz4.h"

#include <array>
#include <cstring>

namespace Northstar
{
    namespace
    {
        constexpr std::array<UINT32, 256> BuildCRC32CTable() noexcept
        {
            std::array<UINT32, 256> table{};
            for (UINT32 i = 0u; i < 256u; ++i)
            {
                UINT32 crc = i;
                for (UINT32 bit = 0u; bit < 8u; ++bit)
                {
                    crc = ((crc & 1u) != 0u) ? ((crc >> 1u) ^ 0x82F63B78u) : (crc >> 1u);
                }
                table[i] = crc;
            }
            return table;
        }

        inline constexpr std::array<UINT32, 256> kCRC32CTable = BuildCRC32CTable();

        inline UINT32 ReadU32LE(const BYTE* p) noexcept
        {
            UINT32 v = 0u;
            std::memcpy(&v, p, sizeof(UINT32));
            return v;
        }

        inline UINT16 ReadU16LE(const BYTE* p) noexcept
        {
            return static_cast<UINT16>(static_cast<UINT16>(p[0]) | (static_cast<UINT16>(p[1]) << 8u));
        }

        inline void WriteU16LE(BYTE* p, UINT16 v) noexcept
        {
            p[0] = static_cast<BYTE>(v & 0xFFu);
            p[1] = static_cast<BYTE>((v >> 8u) & 0xFFu);
        }

        inline UINT32 HashLZ4Seq4(UINT32 seq) noexcept
        {
            return (seq * 2654435761u) >> 20u; // 12-bit hash table index (4096 entries)
        }
    }

    UINT32 NVGCodecLZ4::ComputeCRC32C(const BYTE* data, SIZE_T size, UINT32 seed) noexcept
    {
        if (!data || size == 0u)
        {
            return seed;
        }
        UINT32 crc = ~seed;
        for (SIZE_T i = 0u; i < size; ++i)
        {
            crc = kCRC32CTable[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
        }
        return ~crc;
    }

    UINT64 NVGCodecLZ4::ComputeFNV1a64(const BYTE* data, SIZE_T size, UINT64 seed) noexcept
    {
        UINT64 h = seed;
        if (!data || size == 0u)
        {
            return h;
        }
        for (SIZE_T i = 0u; i < size; ++i)
        {
            h ^= static_cast<UINT64>(data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    SIZE_T NVGCodecLZ4::MaxCompressedBound(SIZE_T inputSize) noexcept
    {
        if (inputSize > 0x7E000000ull)
        {
            return 0u;
        }
        return inputSize + (inputSize / 255u) + 16u;
    }

    SIZE_T NVGCodecLZ4::CompressBlock(
        const BYTE* src,
        SIZE_T      srcSize,
        BYTE*       dst,
        SIZE_T      dstCapacity) noexcept
    {
        if (!dst || dstCapacity == 0u)
        {
            return 0u;
        }
        if (srcSize == 0u)
        {
            return 0u;
        }
        if (!src || srcSize > 0x7E000000ull)
        {
            return 0u;
        }

        constexpr SIZE_T kMinMatch       = 4u;
        constexpr SIZE_T kLastLiterals   = 5u;
        constexpr SIZE_T kMfLimit        = 12u;
        constexpr SIZE_T kMaxDistance    = 65535u;
        constexpr UINT32 kHashTableSize  = 4096u;

        auto emitLengthBytes = [&](BYTE*& op, BYTE* const opEnd, SIZE_T len) noexcept -> BOOL {
            while (len >= 255u)
            {
                if (op >= opEnd) return false;
                *op++ = 255u;
                len -= 255u;
            }
            if (op >= opEnd) return false;
            *op++ = static_cast<BYTE>(len);
            return true;
        };

        BYTE*       op    = dst;
        BYTE* const opEnd = dst + dstCapacity;
        const BYTE* anchor = src;
        const BYTE* const srcEnd = src + srcSize;

        if (srcSize < kMfLimit + 1u)
        {
            // Emit entire short block as a single literal run
            const SIZE_T litLen = srcSize;
            if (op >= opEnd) return 0u;
            BYTE* tokenPtr = op++;
            if (litLen >= 15u)
            {
                *tokenPtr = static_cast<BYTE>(15u << 4u);
                if (!emitLengthBytes(op, opEnd, litLen - 15u)) return 0u;
            }
            else
            {
                *tokenPtr = static_cast<BYTE>(litLen << 4u);
            }
            if (static_cast<SIZE_T>(opEnd - op) < litLen) return 0u;
            std::memcpy(op, src, litLen);
            op += litLen;
            return static_cast<SIZE_T>(op - dst);
        }

        std::array<UINT32, kHashTableSize> dict{};
        dict.fill(0xFFFFFFFFu);

        const BYTE* ip = src;
        const BYTE* const matchLimit = srcEnd - kLastLiterals;
        const BYTE* const searchLimit = srcEnd - kMfLimit;

        dict[HashLZ4Seq4(ReadU32LE(ip))] = 0u;
        ++ip;

        while (ip <= searchLimit)
        {
            const BYTE* matchPtr = nullptr;
            const BYTE* curIp    = ip;
            SIZE_T step = 1u;
            SIZE_T searchMatchNb = (1u << 6u);

            do
            {
                curIp = ip;
                const UINT32 seq = ReadU32LE(curIp);
                const UINT32 h   = HashLZ4Seq4(seq);
                const UINT32 candIdx = dict[h];
                dict[h] = static_cast<UINT32>(curIp - src);

                if (candIdx != 0xFFFFFFFFu)
                {
                    const BYTE* cand = src + candIdx;
                    const SIZE_T dist = static_cast<SIZE_T>(curIp - cand);
                    if (dist > 0u && dist <= kMaxDistance && ReadU32LE(cand) == seq)
                    {
                        matchPtr = cand;
                        break;
                    }
                }

                ip += step;
                step = (searchMatchNb++ >> 6u);
            } while (ip <= searchLimit);

            if (!matchPtr)
            {
                break;
            }

            ip = curIp;
            while (ip > anchor && matchPtr > src && ip[-1] == matchPtr[-1])
            {
                --ip;
                --matchPtr;
            }

            const SIZE_T litLen = static_cast<SIZE_T>(ip - anchor);
            if (op >= opEnd) return 0u;
            BYTE* tokenPtr = op++;

            if (litLen >= 15u)
            {
                *tokenPtr = static_cast<BYTE>(15u << 4u);
                if (!emitLengthBytes(op, opEnd, litLen - 15u)) return 0u;
            }
            else
            {
                *tokenPtr = static_cast<BYTE>(litLen << 4u);
            }

            if (static_cast<SIZE_T>(opEnd - op) < litLen + 2u) return 0u;
            if (litLen > 0u)
            {
                std::memcpy(op, anchor, litLen);
                op += litLen;
            }

            const UINT16 offset = static_cast<UINT16>(ip - matchPtr);
            WriteU16LE(op, offset);
            op += 2u;

            ip       += kMinMatch;
            matchPtr += kMinMatch;
            const BYTE* matchStart = ip;
            while (ip < matchLimit && *ip == *matchPtr)
            {
                ++ip;
                ++matchPtr;
            }
            const SIZE_T matchExtra = static_cast<SIZE_T>(ip - matchStart);
            if (matchExtra >= 15u)
            {
                *tokenPtr = static_cast<BYTE>(*tokenPtr | 15u);
                if (!emitLengthBytes(op, opEnd, matchExtra - 15u)) return 0u;
            }
            else
            {
                *tokenPtr = static_cast<BYTE>(*tokenPtr | static_cast<BYTE>(matchExtra));
            }

            anchor = ip;
            if (ip > searchLimit)
            {
                break;
            }
            dict[HashLZ4Seq4(ReadU32LE(ip - 2))] = static_cast<UINT32>((ip - 2) - src);
            dict[HashLZ4Seq4(ReadU32LE(ip))]     = static_cast<UINT32>(ip - src);
            ++ip;
        }

        // Emit trailing literals
        const SIZE_T tailLit = static_cast<SIZE_T>(srcEnd - anchor);
        if (op >= opEnd) return 0u;
        BYTE* tokenPtr = op++;
        if (tailLit >= 15u)
        {
            *tokenPtr = static_cast<BYTE>(15u << 4u);
            if (!emitLengthBytes(op, opEnd, tailLit - 15u)) return 0u;
        }
        else
        {
            *tokenPtr = static_cast<BYTE>(tailLit << 4u);
        }
        if (static_cast<SIZE_T>(opEnd - op) < tailLit) return 0u;
        if (tailLit > 0u)
        {
            std::memcpy(op, anchor, tailLit);
            op += tailLit;
        }

        return static_cast<SIZE_T>(op - dst);
    }

    BOOL NVGCodecLZ4::CompressToVector(
        const BYTE*        src,
        SIZE_T             srcSize,
        std::vector<BYTE>& outCompressed)
    {
        outCompressed.clear();
        if (!src || srcSize == 0u)
        {
            return false;
        }
        const SIZE_T bound = MaxCompressedBound(srcSize);
        if (bound == 0u)
        {
            return false;
        }
        outCompressed.resize(bound);
        const SIZE_T written = CompressBlock(src, srcSize, outCompressed.data(), outCompressed.size());
        if (written == 0u)
        {
            outCompressed.clear();
            return false;
        }
        outCompressed.resize(written);
        return true;
    }

    BOOL NVGCodecLZ4::DecompressBlockExact(
        const BYTE* src,
        SIZE_T      srcSize,
        BYTE*       dst,
        SIZE_T      expectedUncompressedSize) noexcept
    {
        if (expectedUncompressedSize == 0u)
        {
            return (srcSize == 0u);
        }
        if (!src || !dst || srcSize == 0u)
        {
            return false;
        }

        const BYTE*       ip    = src;
        const BYTE* const ipEnd = src + srcSize;
        BYTE*             op    = dst;
        BYTE* const       opEnd = dst + expectedUncompressedSize;

        while (ip < ipEnd)
        {
            const BYTE token = *ip++;
            SIZE_T litLen = static_cast<SIZE_T>((token >> 4u) & 0x0Fu);
            if (litLen == 15u)
            {
                for (;;)
                {
                    if (ip >= ipEnd) return false;
                    const BYTE s = *ip++;
                    if (litLen > static_cast<SIZE_T>(opEnd - op) - static_cast<SIZE_T>(s))
                    {
                        return false;
                    }
                    litLen += s;
                    if (s != 255u) break;
                }
            }

            if (litLen > static_cast<SIZE_T>(ipEnd - ip) ||
                litLen > static_cast<SIZE_T>(opEnd - op))
            {
                return false;
            }

            if (litLen > 0u)
            {
                std::memcpy(op, ip, litLen);
                ip += litLen;
                op += litLen;
            }

            // End of block occurs immediately after the final literal sequence
            if (ip == ipEnd)
            {
                break;
            }

            if (static_cast<SIZE_T>(ipEnd - ip) < 2u)
            {
                return false;
            }
            const UINT16 offset = ReadU16LE(ip);
            ip += 2u;
            if (offset == 0u || static_cast<SIZE_T>(offset) > static_cast<SIZE_T>(op - dst))
            {
                return false;
            }

            SIZE_T matchLen = static_cast<SIZE_T>(token & 0x0Fu);
            if (matchLen == 15u)
            {
                for (;;)
                {
                    if (ip >= ipEnd) return false;
                    const BYTE s = *ip++;
                    if (matchLen + 4u > static_cast<SIZE_T>(opEnd - op) - static_cast<SIZE_T>(s))
                    {
                        return false;
                    }
                    matchLen += s;
                    if (s != 255u) break;
                }
            }
            matchLen += 4u;

            if (matchLen > static_cast<SIZE_T>(opEnd - op))
            {
                return false;
            }

            const BYTE* matchSrc = op - offset;
            if (offset >= matchLen)
            {
                std::memcpy(op, matchSrc, matchLen);
                op += matchLen;
            }
            else
            {
                // Overlapping RLE match copy (byte-by-byte is exact by LZ4 specification)
                for (SIZE_T i = 0u; i < matchLen; ++i)
                {
                    op[i] = matchSrc[i];
                }
                op += matchLen;
            }
        }

        return (op == opEnd) && (ip == ipEnd);
    }
}
