#pragma once

#include "NSLCommon/nsl_shader_data.h"

#include <cstdint>
#include <string>

// =====================================================================
//  NSCReader — Binary .nsc v4/v5 loader
//
//  Reflection types live in NSLCommon/nsl_shader_data.h (shared with
//  Northstar). This header only provides the Reader that fills ShaderData.
//
//  v5 adds INPUT_LAYOUT / OUTPUT_LAYOUT sections.
//
//  Usage:
//    NSL::ShaderData data;
//    if (NSL::Reader::LoadFromFile("pbr.nsc", data)) { ... }
// =====================================================================

namespace NSL
{
    class Reader
    {
    public:
        static constexpr size_t   kMaxNSCFileSize      = 64u * 1024u * 1024u; // 64 MiB
        static constexpr uint32_t kMaxBufferTotalSize  = 16u * 1024u * 1024u; // 16 MiB
        static constexpr uint32_t kMaxFieldSize        = 1u  * 1024u * 1024u; // 1 MiB
        static constexpr uint32_t kMaxStructStride     = 1u  * 1024u * 1024u; // 1 MiB
        static constexpr uint32_t kMaxDefaultDataSize  = 64u * 1024u;          // 64 KiB
        static constexpr uint16_t kMaxBlocks           = 1024u;
        static constexpr uint16_t kMaxGroups           = 4096u;
        static constexpr uint16_t kMaxResources        = 4096u;

        static bool LoadFromFile(const std::string& path, ShaderData& outData);
        static bool LoadFromMemory(const uint8_t* data, size_t size, ShaderData& outData);

    private:
        struct Cursor
        {
            const uint8_t* data   = nullptr;
            size_t         pos    = 0;
            size_t         size   = 0;
            bool           failed = false;

            // Overflow-safe remaining-bytes check. Once failed, stays failed.
            bool Ok(size_t n = 1) const
            {
                if (failed) return false;
                return n <= size && pos <= size - n;
            }

            bool Good() const { return !failed; }

            void Fail()
            {
                failed = true;
                pos    = size;
            }

            uint8_t ReadU8()
            {
                if (!Ok(1)) { Fail(); return 0; }
                return data[pos++];
            }
            uint16_t ReadU16()
            {
                if (!Ok(2)) { Fail(); return 0; }
                uint16_t v = static_cast<uint16_t>(data[pos])
                           | static_cast<uint16_t>(static_cast<uint16_t>(data[pos + 1]) << 8);
                pos += 2;
                return v;
            }
            uint32_t ReadU32()
            {
                if (!Ok(4)) { Fail(); return 0; }
                uint32_t v = static_cast<uint32_t>(data[pos])
                           | (static_cast<uint32_t>(data[pos + 1]) << 8)
                           | (static_cast<uint32_t>(data[pos + 2]) << 16)
                           | (static_cast<uint32_t>(data[pos + 3]) << 24);
                pos += 4;
                return v;
            }
            int16_t ReadI16() { return static_cast<int16_t>(ReadU16()); }

            std::string ReadStr()
            {
                if (!Ok(2)) { Fail(); return {}; }
                uint16_t len = ReadU16();
                if (!Good() || !Ok(len))
                {
                    Fail();
                    return {};
                }
                std::string s(reinterpret_cast<const char*>(data + pos), len);
                pos += len;
                return s;
            }
        };

        static bool ParseHeaderV4(Cursor& c, ShaderData& out,
                                  uint16_t& blockCount, uint16_t& groupCount,
                                  uint16_t& bufCount, uint16_t& texCount,
                                  uint16_t& sampCount, uint16_t& sbCount,
                                  uint16_t& uavCount,
                                  uint8_t&  inputCount, uint8_t& outputCount,
                                  uint32_t& srcOffset, uint32_t& srcSize);

        static bool ParsePropertyBlockTable(
            Cursor& c, uint16_t blockCount, ShaderData& out,
            std::vector<uint16_t>& declaredGroupCounts);
        static bool ParsePropertyTable(Cursor& c, uint16_t groupCount, ShaderData& out);
        static bool ParseBufferSection(Cursor& c, ShaderData& out);
        static bool ParseTextureSection(Cursor& c, ShaderData& out);
        static bool ParseSamplerSection(Cursor& c, ShaderData& out);
        static bool ParseStructuredBufferSection(Cursor& c, ShaderData& out);
        static bool ParseUAVSection(Cursor& c, ShaderData& out);
        static bool ParseStageIOSection(Cursor& c, uint8_t count, std::vector<NSCStageIOEntry>& out);
    };

} // namespace NSL
