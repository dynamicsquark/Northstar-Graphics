#include "../include/nsc_reader.h"
#include <fstream>
#include <vector>
#include <cctype>
#include <cstring>

// =====================================================================
//  NSC v4/v5 Magic & Version
// =====================================================================

static constexpr uint32_t kNSCMagic      = 0x4E534300; // "NSC\0"
static constexpr uint32_t kNSCVersionMin = 4;
static constexpr uint32_t kNSCVersionMax = 6;

namespace NSL
{

static bool FitsInRange(uint32_t offset, uint32_t size, uint32_t total)
{
    return size <= total && offset <= total - size;
}

static uint32_t ParsePositiveArrayCount(const std::string& expr)
{
    if (expr.empty()) return 1u;
    uint64_t val = 0;
    for (char c : expr)
    {
        if (!std::isdigit(static_cast<unsigned char>(c)))
            return 1u;
        val = val * 10u + static_cast<uint32_t>(c - '0');
        if (val > 65536u)
            return 1u;
    }
    return (val > 0u) ? static_cast<uint32_t>(val) : 1u;
}

static bool PackScalarTokensToBytes(
    const std::vector<std::string>& tokens,
    size_t                          tokenStart,
    DataTypeID                      typeID,
    uint32_t                        elemSize,
    uint8_t*                        dst)
{
    if (tokens.empty() || elemSize == 0 || !dst) return true;

    const auto tokenAt = [&](size_t relIdx) -> const std::string&
    {
        const size_t idx = tokenStart + relIdx;
        return (idx < tokens.size()) ? tokens[idx] : tokens.back();
    };

    if (typeID == DataTypeID::DOUBLE)
    {
        const uint32_t count = elemSize / 8u;
        for (uint32_t i = 0; i < count; ++i)
        {
            double val = 0.0;
            try { val = std::stod(tokenAt(i)); } catch (...) {}
            std::memcpy(dst + (i * 8u), &val, 8u);
        }
        return true;
    }

    if (typeID == DataTypeID::UINT64)
    {
        const uint32_t count = elemSize / 8u;
        for (uint32_t i = 0; i < count; ++i)
        {
            uint64_t val = 0ull;
            try { val = std::stoull(tokenAt(i), nullptr, 0); } catch (...) {}
            std::memcpy(dst + (i * 8u), &val, 8u);
        }
        return true;
    }

    if (typeID == DataTypeID::MAT3 && elemSize == 48u && (tokens.size() - std::min(tokenStart, tokens.size())) >= 9u)
    {
        for (uint32_t r = 0; r < 3u; ++r)
        {
            for (uint32_t col = 0; col < 3u; ++col)
            {
                float val = 0.0f;
                try { val = std::stof(tokenAt(r * 3u + col)); } catch (...) {}
                std::memcpy(dst + (r * 16u + col * 4u), &val, 4u);
            }
        }
        return true;
    }

    const uint32_t elementCount = elemSize / 4u;
    const bool isFloat = (typeID == DataTypeID::FLOAT || typeID == DataTypeID::VEC2 || typeID == DataTypeID::VEC3 || typeID == DataTypeID::VEC4 ||
                          typeID == DataTypeID::MAT3  || typeID == DataTypeID::MAT4 || typeID == DataTypeID::COLOR || typeID == DataTypeID::QUAT);
    const bool isUint  = (typeID == DataTypeID::UINT  || typeID == DataTypeID::UINT2 || typeID == DataTypeID::UINT3 || typeID == DataTypeID::UINT4);

    for (uint32_t i = 0; i < elementCount; ++i)
    {
        const std::string& t = tokenAt(i);
        if (isFloat)
        {
            float val = 0.0f;
            try { val = std::stof(t); } catch (...) {}
            std::memcpy(dst + (i * 4u), &val, 4u);
        }
        else if (isUint)
        {
            uint32_t val = 0;
            try { val = static_cast<uint32_t>(std::stoull(t)); } catch (...) {}
            std::memcpy(dst + (i * 4u), &val, 4u);
        }
        else if (typeID == DataTypeID::BOOL)
        {
            int32_t val = 0;
            if (t == "true" || t == "1") val = 1;
            else if (t == "false" || t == "0") val = 0;
            else { try { val = (std::stoi(t) != 0) ? 1 : 0; } catch (...) {} }
            std::memcpy(dst + (i * 4u), &val, 4u);
        }
        else
        {
            int32_t val = 0;
            try { val = std::stoi(t); } catch (...) {}
            std::memcpy(dst + (i * 4u), &val, 4u);
        }
    }
    return true;
}

// =====================================================================
//  Default-value parser (used while reading buffer / property sections)
// =====================================================================

static bool ParseDefaultValueToData(
    const std::string&    valStr,
    DataTypeID            typeID,
    uint32_t              size,
    std::vector<uint8_t>& outData,
    uint32_t              arrayCount     = 1u,
    bool                  isCBufferField = false)
{
    outData.clear();
    if (valStr.empty() || size == 0) return true;
    if (size > Reader::kMaxDefaultDataSize) return false;

    std::string cleanStr = valStr;
    while (!cleanStr.empty() && std::isspace(static_cast<unsigned char>(cleanStr.front()))) cleanStr.erase(0, 1);
    while (!cleanStr.empty() && std::isspace(static_cast<unsigned char>(cleanStr.back()))) cleanStr.pop_back();
    if (!cleanStr.empty() && (cleanStr.front() == '{' || cleanStr.front() == '(')) cleanStr.erase(0, 1);
    if (!cleanStr.empty() && (cleanStr.back() == '}' || cleanStr.back() == ')')) cleanStr.pop_back();

    // Tokenize on commas, braces, parens, and whitespace
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (pos < cleanStr.size())
    {
        while (pos < cleanStr.size() &&
               (cleanStr[pos] == ',' || cleanStr[pos] == '{' || cleanStr[pos] == '}' ||
                cleanStr[pos] == '(' || cleanStr[pos] == ')' ||
                std::isspace(static_cast<unsigned char>(cleanStr[pos]))))
        {
            ++pos;
        }
        if (pos >= cleanStr.size()) break;
        const size_t tokStart = pos;
        while (pos < cleanStr.size() &&
               cleanStr[pos] != ',' && cleanStr[pos] != '{' && cleanStr[pos] != '}' &&
               cleanStr[pos] != '(' && cleanStr[pos] != ')' &&
               !std::isspace(static_cast<unsigned char>(cleanStr[pos])))
        {
            ++pos;
        }
        tokens.push_back(cleanStr.substr(tokStart, pos - tokStart));
    }

    if (tokens.empty()) return true;

    outData.resize(size, 0);

    if (isCBufferField && arrayCount > 1u)
    {
        const DataTypeInfo* dti = FindDataTypeByID(typeID);
        if (dti && dti->size > 0u)
        {
            const uint32_t elemCbSize = (typeID == DataTypeID::MAT3) ? 48u
                                      : (typeID == DataTypeID::MAT4) ? 64u
                                                                     : dti->size;
            const uint32_t elemStride = (elemCbSize + 15u) & ~15u;
            const uint32_t scalarsPerElem = (typeID == DataTypeID::DOUBLE || typeID == DataTypeID::UINT64) ? 1u
                                          : (typeID == DataTypeID::MAT3)   ? 9u
                                          : (typeID == DataTypeID::MAT4)   ? 16u
                                                                           : (dti->size / 4u);
            for (uint32_t elemIdx = 0; elemIdx < arrayCount; ++elemIdx)
            {
                const uint64_t byteOff = static_cast<uint64_t>(elemIdx) * elemStride;
                if (byteOff + elemCbSize > size) break;
                PackScalarTokensToBytes(
                    tokens,
                    static_cast<size_t>(elemIdx) * scalarsPerElem,
                    typeID,
                    elemCbSize,
                    outData.data() + byteOff);
            }
            return true;
        }
    }

    return PackScalarTokensToBytes(tokens, 0u, typeID, size, outData.data());
}

// =====================================================================
//  Reader::LoadFromFile
// =====================================================================

bool Reader::LoadFromFile(const std::string& path, ShaderData& outData)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return false;

    const std::streamoff fsize = ifs.tellg();
    if (fsize < 0) return false;
    if (static_cast<uint64_t>(fsize) > kMaxNSCFileSize) return false;

    ifs.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(fsize));
    if (fsize > 0)
        ifs.read(reinterpret_cast<char*>(buf.data()), fsize);
    if (!ifs) return false;

    return LoadFromMemory(buf.data(), buf.size(), outData);
}

// =====================================================================
//  Reader::LoadFromMemory
// =====================================================================

bool Reader::LoadFromMemory(const uint8_t* data, size_t size, ShaderData& outData)
{
    if (!data || size < 36 || size > kMaxNSCFileSize) return false; // v4 header = 36 bytes

    Cursor c{ data, 0, size, false };
    ShaderData result;

    uint16_t blockCount = 0, groupCount = 0;
    uint16_t bufCount = 0, texCount = 0, sampCount = 0, sbCount = 0, uavCount = 0;
    uint8_t  inputCount = 0, outputCount = 0;
    uint32_t srcOffset = 0, srcSize = 0;
    std::vector<uint16_t> declaredBlockGroupCounts;

    if (!ParseHeaderV4(c, result, blockCount, groupCount,
                       bufCount, texCount, sampCount, sbCount,
                       uavCount, inputCount, outputCount, srcOffset, srcSize))
        return false;

    if (!ParsePropertyBlockTable(
            c, blockCount, result, declaredBlockGroupCounts)) return false;
    if (!ParsePropertyTable(c, groupCount, result))         return false;
    if (declaredBlockGroupCounts.size() != result.propertyBlocks.size())
        return false;
    for (size_t i = 0; i < result.propertyBlocks.size(); ++i)
    {
        if (result.propertyBlocks[i].groups.size() !=
            declaredBlockGroupCounts[i])
            return false;
    }
    if (!ParseBufferSection(c, result))                     return false;
    if (!ParseTextureSection(c, result))                    return false;
    if (!ParseSamplerSection(c, result))                    return false;
    if (!ParseStructuredBufferSection(c, result))           return false;
    if (!ParseUAVSection(c, result))                        return false;

    // v5 stage I/O sections (v4 has counts == 0 from former padding)
    if (result.nscVersion >= 5)
    {
        if (!ParseStageIOSection(c, inputCount,  result.inputLayout))  return false;
        if (!ParseStageIOSection(c, outputCount, result.outputLayout)) return false;
    }

    // Header resource counts must match what we actually parsed
    if (result.propertyBlocks.size()    != blockCount) return false;
    if (result.propertyGroups.size()    != groupCount) return false;
    if (result.buffers.size()           != bufCount)   return false;
    if (result.textures.size()          != texCount)   return false;
    if (result.samplers.size()          != sampCount)  return false;
    if (result.structuredBuffers.size() != sbCount)    return false;
    if (result.uavBuffers.size()        != uavCount)   return false;
    if (result.nscVersion >= 5)
    {
        if (result.inputLayout.size()  != inputCount)  return false;
        if (result.outputLayout.size() != outputCount) return false;
    }

    // Compute thread-group metadata section (present when NSC_FLAG_HAS_COMPUTE_NUMTHREADS is set)
    if ((result.flags & NSC_FLAG_HAS_COMPUTE_NUMTHREADS) != 0)
    {
        if (result.stage != ShaderStage::COMPUTE) return false;
        if (!c.Ok(12)) return false;
        result.numThreadsX = c.ReadU32();
        result.numThreadsY = c.ReadU32();
        result.numThreadsZ = c.ReadU32();
        if (!c.Good()) return false;
        if (result.numThreadsX == 0 || result.numThreadsY == 0 || result.numThreadsZ == 0)
            return false;
    }

    // Sections must end exactly where the source blob begins. This rejects
    // unknown resource kinds and malformed section layouts instead of
    // accepting unparsed bytes.
    if (!c.Good() || c.pos != static_cast<size_t>(srcOffset)) return false;

    // Checked srcOffset/srcSize; require a valid null-terminated source blob.
    if (srcSize == 0) return false;
    if (srcOffset > size || srcSize > size - srcOffset) return false;
    if (static_cast<size_t>(srcSize) != size - srcOffset) return false;
    if (data[srcOffset + srcSize - 1] != 0) return false;

    const char* src = reinterpret_cast<const char*>(data + srcOffset);
    result.sourceCode = std::string(src, srcSize - 1);

    // Backward-compatibility fallback for legacy .nsc compute binaries compiled before
    // NSC_FLAG_HAS_COMPUTE_NUMTHREADS was emitted in the binary metadata section.
    if (result.stage == ShaderStage::COMPUTE && !result.HasValidNumThreads() && !result.sourceCode.empty())
    {
        const size_t ntPos = result.sourceCode.find("numthreads");
        if (ntPos != std::string::npos)
        {
            const size_t openPos = result.sourceCode.find('(', ntPos);
            const size_t closePos = (openPos != std::string::npos) ? result.sourceCode.find(')', openPos) : std::string::npos;
            if (openPos != std::string::npos && closePos != std::string::npos && closePos > openPos + 1)
            {
                std::string args = result.sourceCode.substr(openPos + 1, closePos - openPos - 1);
                auto parseUIntToken = [](std::string s) -> uint32_t
                {
                    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
                    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
                    if (!s.empty() && (s.back() == 'u' || s.back() == 'U')) s.pop_back();
                    if (s.empty()) return 0u;
                    for (char ch : s) if (ch < '0' || ch > '9') return 0u;
                    try
                    {
                        uint64_t val = std::stoull(s);
                        return (val > 0 && val <= UINT32_MAX) ? static_cast<uint32_t>(val) : 0u;
                    }
                    catch (...) { return 0u; }
                };
                const size_t c1 = args.find(',');
                const size_t c2 = (c1 != std::string::npos) ? args.find(',', c1 + 1) : std::string::npos;
                if (c1 != std::string::npos && c2 != std::string::npos)
                {
                    uint32_t x = parseUIntToken(args.substr(0, c1));
                    uint32_t y = parseUIntToken(args.substr(c1 + 1, c2 - c1 - 1));
                    uint32_t z = parseUIntToken(args.substr(c2 + 1));
                    if (x > 0 && y > 0 && z > 0)
                    {
                        result.numThreadsX = x;
                        result.numThreadsY = y;
                        result.numThreadsZ = z;
                    }
                }
            }
        }
    }

    outData = std::move(result);
    return true;
}

// =====================================================================
//  ParseHeaderV4 — 36 bytes fixed
//
//  Layout:
//    magic (4) version (4) stage (1) api (1) flags (2)
//    blockCount (2) groupCount (2) bufCount (2) texCount (2)
//    sampCount (2) sbCount (2) uavCount (2)
//    srcOffset (4) srcSize (4)
//    padding (2)
//    = 36 bytes
// =====================================================================

bool Reader::ParseHeaderV4(
    Cursor&    c,
    ShaderData& out,
    uint16_t&  blockCount, uint16_t& groupCount,
    uint16_t&  bufCount,   uint16_t& texCount,
    uint16_t&  sampCount,  uint16_t& sbCount,
    uint16_t&  uavCount,
    uint8_t&   inputCount, uint8_t&  outputCount,
    uint32_t&  srcOffset,  uint32_t& srcSize)
{
    if (!c.Ok(36)) { c.Fail(); return false; }

    uint32_t magic   = c.ReadU32();
    uint32_t version = c.ReadU32();

    if (!c.Good()) return false;
    if (magic != kNSCMagic) return false;
    if (version < kNSCVersionMin || version > kNSCVersionMax) return false;

    out.nscVersion = version;
    out.stage = static_cast<ShaderStage>(c.ReadU8());
    out.api   = static_cast<TargetAPI>  (c.ReadU8());
    if (out.stage < ShaderStage::VERTEX || out.stage > ShaderStage::DOMAIN)
        return false;
    if (out.api != TargetAPI::HLSL11 && out.api != TargetAPI::HLSL12)
        return false;

    out.flags = c.ReadU16();

    blockCount  = c.ReadU16();
    groupCount  = c.ReadU16();
    bufCount    = c.ReadU16();
    texCount    = c.ReadU16();
    sampCount   = c.ReadU16();
    sbCount     = c.ReadU16();
    uavCount    = c.ReadU16();
    srcOffset   = c.ReadU32();
    srcSize     = c.ReadU32();

    if (blockCount > kMaxBlocks || groupCount > kMaxGroups ||
        bufCount > kMaxResources || texCount > kMaxResources ||
        sampCount > kMaxResources || sbCount > kMaxResources ||
        uavCount > kMaxResources)
    {
        return false;
    }

    // v4: padding u16 (== 0). v5: inputCount u8 + outputCount u8.
    inputCount  = c.ReadU8();
    outputCount = c.ReadU8();

    return c.Good();
}

// =====================================================================
//  ParsePropertyBlockTable  (v4 NEW)
//  Reads block names + group counts.
//  Group pointers are populated after ParsePropertyTable fills the groups.
// =====================================================================

bool Reader::ParsePropertyBlockTable(
    Cursor& c,
    uint16_t blockCount,
    ShaderData& out,
    std::vector<uint16_t>& declaredGroupCounts)
{
    // We read block names + counts here; we'll wire up group pointers
    // in a post-processing step after all groups are loaded.
    struct BlockMeta
    {
        std::string name;
        uint16_t    groupCount;
    };
    std::vector<BlockMeta> metas;
    metas.reserve(blockCount);
    declaredGroupCounts.clear();
    declaredGroupCounts.reserve(blockCount);

    for (uint16_t b = 0; b < blockCount; ++b)
    {
        BlockMeta m;
        m.name       = c.ReadStr();
        m.groupCount = c.ReadU16();
        if (!c.Good()) return false;
        declaredGroupCounts.push_back(m.groupCount);
        metas.push_back(std::move(m));
    }

    // Reserve NSCPropertyBlock entries for later population
    for (auto& m : metas)
    {
        NSCPropertyBlock block;
        block.name = m.name;
        out.propertyBlocks.push_back(std::move(block));
    }

    // (Group pointers wired up in ParsePropertyTable after groups are loaded)
    return c.Good();
}

// =====================================================================
//  ParsePropertyTable  (v4 — reads ownerPropertyName + qualifiedName)
// =====================================================================

bool Reader::ParsePropertyTable(Cursor& c, uint16_t groupCount, ShaderData& out)
{
    for (uint16_t g = 0; g < groupCount; ++g)
    {
        NSCPropertyGroup grp;
        grp.resourceKind      = static_cast<ResourceKind>(c.ReadU8());
        if (grp.resourceKind < ResourceKind::BUFFER ||
            grp.resourceKind > ResourceKind::TEXTURE_CUBE_ARRAY)
            return false;
        grp.name              = c.ReadStr();
        grp.ownerPropertyName = c.ReadStr(); // v4 NEW
        grp.qualifiedName     = c.ReadStr(); // v4 NEW
        if (out.nscVersion >= 6) {
            grp.set     = c.ReadU16();
            grp.binding = c.ReadU16();
        } else {
            int16_t slot = c.ReadI16();
            grp.set = 0;
            grp.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
        }
        if (!c.Good()) return false;

        uint16_t entryCount = c.ReadU16();
        if (!c.Good()) return false;

        for (uint16_t e = 0; e < entryCount; ++e)
        {
            NSCPropertyEntry entry;
            entry.kind               = static_cast<PropertyEntryKind>(c.ReadU8());
            if (entry.kind != PropertyEntryKind::ADD &&
                entry.kind != PropertyEntryKind::GET)
                return false;
            entry.name               = c.ReadStr();
            entry.typeName           = c.ReadStr();
            entry.defaultValue       = c.ReadStr();
            entry.arraySizeExpr      = c.ReadStr();
            entry.enginePropertyName = c.ReadStr(); // v4 NEW field
            if (!c.Good()) return false;

            DataTypeID typeID = GetDataTypeID(entry.typeName);
            if (const DataTypeInfo* dti = FindDataType(entry.typeName))
            {
                const uint32_t arrCnt = ParsePositiveArrayCount(entry.arraySizeExpr);
                uint32_t totalRawSz = dti->size;
                if (arrCnt > 1u && static_cast<uint64_t>(dti->size) * arrCnt <= Reader::kMaxDefaultDataSize)
                    totalRawSz = dti->size * arrCnt;
                if (!ParseDefaultValueToData(entry.defaultValue, typeID, totalRawSz, entry.defaultData, arrCnt, false))
                    return false;
            }
            if (out.nscVersion >= 6) {
                entry.set     = c.ReadU16();
                entry.binding = c.ReadU16();
            } else {
                int16_t slot = c.ReadI16();
                entry.set = 0;
                entry.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
            }
            if (!c.Good()) return false;
            grp.entries.push_back(std::move(entry));
        }

        out.propertyGroups.push_back(std::move(grp));
    }

    // Wire up PropertyBlock group pointers
    out.RebuildPropertyBlockGroupPointers();

    return c.Good();
}

// =====================================================================
//  ParseBufferSection  (v4 — uses qualifiedName as key)
// =====================================================================

bool Reader::ParseBufferSection(Cursor& c, ShaderData& out)
{
    for (const auto& pg : out.propertyGroups)
    {
        if (pg.resourceKind != ResourceKind::BUFFER) continue;

        NSCBuffer buf;
        buf.name      = c.ReadStr(); // v4: qualifiedName
        if (out.nscVersion >= 6) {
            buf.set     = c.ReadU16();
            buf.binding = c.ReadU16();
        } else {
            int16_t slot = c.ReadI16();
            buf.set = 0;
            buf.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
        }
        buf.totalSize = c.ReadU32();
        if (!c.Good()) return false;
        if (buf.totalSize > kMaxBufferTotalSize) return false;

        uint16_t fieldCount = c.ReadU16();
        if (!c.Good()) return false;

        for (uint16_t f = 0; f < fieldCount; ++f)
        {
            NSCField field;
            field.typeID       = static_cast<DataTypeID>(c.ReadU8());
            field.name         = c.ReadStr();
            field.offset       = c.ReadU32();
            field.size         = c.ReadU32();
            uint8_t flags      = c.ReadU8();
            field.isGet        = (flags & 0x01) != 0;
            field.propertyName = c.ReadStr();
            field.defaultValue = c.ReadStr();
            if (!c.Good()) return false;

            if (field.size > kMaxFieldSize) return false;
            if (!FitsInRange(field.offset, field.size, buf.totalSize)) return false;

            for (const auto& pe : pg.entries)
            {
                if (pe.name == field.name)
                {
                    field.arrayCount = ParsePositiveArrayCount(pe.arraySizeExpr);
                    break;
                }
            }

            if (!ParseDefaultValueToData(field.defaultValue, field.typeID, field.size, field.defaultData, field.arrayCount, true))
                return false;
            buf.fields.push_back(std::move(field));
        }

        out.buffers.push_back(std::move(buf));
    }
    return c.Good();
}

// =====================================================================
//  ParseTextureSection  (v4 — reads groupQualifiedName)
// =====================================================================

bool Reader::ParseTextureSection(Cursor& c, ShaderData& out)
{
    for (const auto& pg : out.propertyGroups)
    {
        if (pg.resourceKind != ResourceKind::TEXTURE_2D  &&
            pg.resourceKind != ResourceKind::TEXTURE_CUBE &&
            pg.resourceKind != ResourceKind::TEXTURE_2D_ARRAY &&
            pg.resourceKind != ResourceKind::TEXTURE_CUBE_ARRAY) continue;

        for (size_t e = 0; e < pg.entries.size(); ++e)
        {
            NSCTexture tex;
            tex.groupQualifiedName = c.ReadStr(); // v4: qualified name
            tex.name               = c.ReadStr();
            if (out.nscVersion >= 6) {
                tex.set     = c.ReadU16();
                tex.binding = c.ReadU16();
            } else {
                int16_t slot = c.ReadI16();
                tex.set = 0;
                tex.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
            }
            tex.typeID             = static_cast<DataTypeID>(c.ReadU8());
            tex.propertyName       = c.ReadStr();
            tex.arrayCount         = static_cast<uint16_t>(std::min<uint32_t>(ParsePositiveArrayCount(pg.entries[e].arraySizeExpr), 0xFFFFu));
            if (!c.Good()) return false;
            out.textures.push_back(std::move(tex));
        }
    }
    return c.Good();
}

// =====================================================================
//  ParseSamplerSection  (v4)
// =====================================================================

bool Reader::ParseSamplerSection(Cursor& c, ShaderData& out)
{
    for (const auto& pg : out.propertyGroups)
    {
        if (pg.resourceKind != ResourceKind::SAMPLER_STATE) continue;

        for (size_t e = 0; e < pg.entries.size(); ++e)
        {
            NSCSampler samp;
            samp.groupQualifiedName = c.ReadStr(); // v4
            samp.name               = c.ReadStr();
            if (out.nscVersion >= 6) {
                samp.set     = c.ReadU16();
                samp.binding = c.ReadU16();
            } else {
                int16_t slot = c.ReadI16();
                samp.set = 0;
                samp.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
            }
            samp.isComparison       = (c.ReadU8() != 0);
            samp.propertyName       = c.ReadStr();
            samp.arrayCount         = static_cast<uint16_t>(std::min<uint32_t>(ParsePositiveArrayCount(pg.entries[e].arraySizeExpr), 0xFFFFu));
            if (!c.Good()) return false;
            out.samplers.push_back(std::move(samp));
        }
    }
    return c.Good();
}

// =====================================================================
//  ParseStructuredBufferSection  (v4)
// =====================================================================

bool Reader::ParseStructuredBufferSection(Cursor& c, ShaderData& out)
{
    for (const auto& pg : out.propertyGroups)
    {
        if (pg.resourceKind != ResourceKind::STRUCTURED_BUFFER &&
            pg.resourceKind != ResourceKind::RW_STRUCTURED_BUFFER)
            continue;

        NSCStructuredBuffer sb;
        sb.name   = c.ReadStr(); // v4: qualifiedName
        if (out.nscVersion >= 6) {
            sb.set     = c.ReadU16();
            sb.binding = c.ReadU16();
        } else {
            int16_t slot = c.ReadI16();
            sb.set = 0;
            sb.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
        }
        sb.stride = c.ReadU32();
        if (!c.Good()) return false;
        if (sb.stride > kMaxStructStride) return false;

        uint16_t fieldCount = c.ReadU16();
        if (!c.Good()) return false;

        for (uint16_t f = 0; f < fieldCount; ++f)
        {
            NSCStructField field;
            field.typeID       = static_cast<DataTypeID>(c.ReadU8());
            field.name         = c.ReadStr();
            field.offset       = c.ReadU32();
            field.size         = c.ReadU32();
            uint8_t flags      = c.ReadU8();
            field.isGet        = (flags & 0x01) != 0;
            field.propertyName = c.ReadStr();
            if (!c.Good()) return false;

            if (field.size > kMaxFieldSize) return false;
            // Engine-provided SBs have stride==0 and no fields; user SBs validate vs stride
            if (sb.stride > 0)
            {
                if (!FitsInRange(field.offset, field.size, sb.stride)) return false;
            }
            else if (field.offset != 0 || field.size != 0)
            {
                return false;
            }
            sb.fields.push_back(std::move(field));
        }

        // Engine-provided SBs have stride==0; no extra data in v4
        out.structuredBuffers.push_back(std::move(sb));
    }
    return c.Good();
}

// =====================================================================
//  ParseUAVSection  (v4)
// =====================================================================

bool Reader::ParseUAVSection(Cursor& c, ShaderData& out)
{
    for (const auto& pg : out.propertyGroups)
    {
        if (pg.resourceKind != ResourceKind::RW_STRUCTURED_BUFFER &&
            pg.resourceKind != ResourceKind::RW_TEXTURE_2D) continue;

        NSCUAVBuffer uav;
        uav.name         = c.ReadStr(); // v4: qualifiedName
        uav.resourceKind = static_cast<ResourceKind>(c.ReadU8());
        if (out.nscVersion >= 6) {
            uav.set     = c.ReadU16();
            uav.binding = c.ReadU16();
        } else {
            int16_t slot = c.ReadI16();
            uav.set = 0;
            uav.binding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
        }
        if (!c.Good()) return false;
        if (uav.resourceKind != pg.resourceKind) return false;

        if (uav.resourceKind == ResourceKind::RW_TEXTURE_2D)
        {
            uint16_t entryCount = c.ReadU16();
            if (!c.Good()) return false;

            for (uint16_t e = 0; e < entryCount; ++e)
            {
                std::string eName     = c.ReadStr();
                std::string eTypeName = c.ReadStr();
                uint16_t eSet = 0;
                uint16_t eBinding = 0xFFFF;
                if (out.nscVersion >= 6) {
                    eSet     = c.ReadU16();
                    eBinding = c.ReadU16();
                } else {
                    int16_t slot = c.ReadI16();
                    eBinding = (slot < 0) ? 0xFFFF : static_cast<uint16_t>(slot);
                }
                if (!c.Good()) return false;

                NSCStructField field;
                field.name       = eName;
                field.typeID     = eTypeName.empty() ? DataTypeID::VEC4 : GetDataTypeID(eTypeName);
                field.offset     = 0;
                field.size       = 0;
                field.set        = eSet;
                field.binding    = eBinding;
                if (e < pg.entries.size())
                {
                    field.arrayCount = static_cast<uint16_t>(std::min<uint32_t>(ParsePositiveArrayCount(pg.entries[e].arraySizeExpr), 0xFFFFu));
                }
                uav.fields.push_back(std::move(field));
            }
        }
        else
        {
            uint16_t fieldCount = c.ReadU16();
            if (!c.Good()) return false;

            for (uint16_t f = 0; f < fieldCount; ++f)
            {
                NSCStructField field;
                field.typeID       = static_cast<DataTypeID>(c.ReadU8());
                field.name         = c.ReadStr();
                field.offset       = c.ReadU32();
                field.size         = c.ReadU32();
                uint8_t flags      = c.ReadU8();
                field.isGet        = (flags & 0x01) != 0;
                field.propertyName = c.ReadStr();
                if (!c.Good()) return false;

                if (field.size > kMaxFieldSize) return false;
                if (field.offset > UINT32_MAX - field.size) return false;
                uav.fields.push_back(std::move(field));
            }
        }

        out.uavBuffers.push_back(std::move(uav));
    }
    return c.Good();
}

// =====================================================================
//  ParseStageIOSection  (v5)
// =====================================================================

bool Reader::ParseStageIOSection(
    Cursor& c, uint8_t count, std::vector<NSCStageIOEntry>& out)
{
    for (uint8_t i = 0; i < count; ++i)
    {
        NSCStageIOEntry e;
        e.name     = c.ReadStr();
        e.typeID   = static_cast<DataTypeID>(c.ReadU8());
        e.semantic = c.ReadStr();
        e.category = c.ReadU8();
        if (!c.Good()) return false;
        out.push_back(std::move(e));
    }
    return c.Good();
}

} // namespace NSL
