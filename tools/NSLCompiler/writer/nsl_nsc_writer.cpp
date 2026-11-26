#include "nsl_nsc_writer.h"
#include <fstream>
#include <cstring>
#include <filesystem>
#include <algorithm>

namespace NSL
{

// =====================================================================
//  Primitive writers
// =====================================================================

void NSCWriter::WriteU8(std::vector<uint8_t>& buf, uint8_t v)
{
    buf.push_back(v);
}

void NSCWriter::WriteU16(std::vector<uint8_t>& buf, uint16_t v)
{
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

void NSCWriter::WriteU32(std::vector<uint8_t>& buf, uint32_t v)
{
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >>  8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void NSCWriter::WriteI16(std::vector<uint8_t>& buf, int16_t v)
{
    WriteU16(buf, static_cast<uint16_t>(v));
}

void NSCWriter::WriteStr(std::vector<uint8_t>& buf, const std::string& s)
{
    uint16_t len = static_cast<uint16_t>(std::min<size_t>(s.size(), 0xFFFFu));
    WriteU16(buf, len);
    for (size_t i = 0; i < len; ++i) buf.push_back(static_cast<uint8_t>(s[i]));
}

void NSCWriter::PatchU32(std::vector<uint8_t>& buf, size_t pos, uint32_t v)
{
    buf[pos + 0] = static_cast<uint8_t>(v & 0xFF);
    buf[pos + 1] = static_cast<uint8_t>((v >>  8) & 0xFF);
    buf[pos + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[pos + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

// =====================================================================
//  Write — main entry point
// =====================================================================

bool NSCWriter::Write(
    const std::string&                                    outputPath,
    ShaderStage                                           stage,
    TargetAPI                                             api,
    const ParsedProperties&                               properties,
    const ParsedStageIO&                                  stageIO,
    const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers,
    const std::unordered_map<std::string, AlignedStruct>& alignedStructs,
    const std::string&                                    sourceCode,
    uint16_t                                              extraFlags,
    uint32_t                                              numThreadsX,
    uint32_t                                              numThreadsY,
    uint32_t                                              numThreadsZ)
{
    std::vector<uint8_t> buf;
    buf.reserve(64 * 1024);

    // ==================== Count resources ====================
    uint16_t bufferCount      = 0;
    uint16_t textureCount     = 0;
    uint16_t samplerCount     = 0;
    uint16_t sbCount          = 0;
    uint16_t uavCount         = 0;
    uint16_t groupCount       = static_cast<uint16_t>(properties.groups.size());
    uint16_t blockCount       = static_cast<uint16_t>(properties.blocks.size());
    uint8_t  inputCount       = static_cast<uint8_t>(stageIO.inputs.size());
    uint8_t  outputCount      = static_cast<uint8_t>(stageIO.outputs.size());

    for (const auto& g : properties.groups)
    {
        switch (g.resourceKind)
        {
            case ResourceKind::BUFFER:              ++bufferCount;  break;
            case ResourceKind::TEXTURE_2D:
            case ResourceKind::TEXTURE_CUBE:
            case ResourceKind::TEXTURE_2D_ARRAY:
            case ResourceKind::TEXTURE_CUBE_ARRAY:
                textureCount += static_cast<uint16_t>(g.entries.size()); break;
            case ResourceKind::SAMPLER_STATE:
                samplerCount += static_cast<uint16_t>(g.entries.size()); break;
            case ResourceKind::STRUCTURED_BUFFER:
            case ResourceKind::RW_STRUCTURED_BUFFER:
                // Layout/stride live in the StructuredBuffer section for both.
                // RW also appears in the UAV section (bind / UAV metadata).
                ++sbCount;
                if (g.resourceKind == ResourceKind::RW_STRUCTURED_BUFFER)
                    ++uavCount;
                break;
            case ResourceKind::RW_TEXTURE_2D:       ++uavCount;     break;
            default: break;
        }
    }

    bool     hasUAV = (uavCount > 0);
    bool     hasComputeNumThreads = (stage == ShaderStage::COMPUTE &&
                                     numThreadsX > 0 && numThreadsY > 0 && numThreadsZ > 0);
    uint16_t flags  = extraFlags;
    if (hasUAV)
        flags |= NSC_FLAG_HAS_UAV;
    if (hasComputeNumThreads)
        flags |= NSC_FLAG_HAS_COMPUTE_NUMTHREADS;
    for (const auto& outEntry : stageIO.outputs)
    {
        if (outEntry.name == "FRAG_DEPTH")
        {
            flags |= NSC_FLAG_WRITES_DEPTH;
            break;
        }
    }

    // ==================== Header (36 bytes fixed) ====================
    WriteU32(buf, NSC_MAGIC);
    WriteU32(buf, NSC_VERSION);         // Version: 6
    WriteU8 (buf, static_cast<uint8_t>(stage));
    WriteU8 (buf, static_cast<uint8_t>(api));
    WriteU16(buf, flags);
    WriteU16(buf, blockCount);
    WriteU16(buf, groupCount);
    WriteU16(buf, bufferCount);
    WriteU16(buf, textureCount);
    WriteU16(buf, samplerCount);
    WriteU16(buf, sbCount);
    WriteU16(buf, uavCount);

    size_t shaderSourceOffsetPos = buf.size();
    WriteU32(buf, 0);
    size_t shaderSourceSizePos = buf.size();
    WriteU32(buf, 0);

    // v5: former padding becomes layout counts
    WriteU8(buf, inputCount);
    WriteU8(buf, outputCount);

    WritePropertyBlockTable(buf, properties);
    WritePropertyTable(buf, properties);
    WriteBufferSection(buf, properties, alignedBuffers);
    WriteTextureSection(buf, properties);
    WriteSamplerSection(buf, properties);
    WriteStructuredBufferSection(buf, properties, alignedStructs);
    WriteUAVSection(buf, properties, alignedStructs);

    // ==================== Stage I/O Sections (v5) ====================
    WriteStageIOSection(buf, stageIO.inputs);
    WriteStageIOSection(buf, stageIO.outputs);

    // ==================== Compute Thread-Group Metadata ====================
    if (hasComputeNumThreads)
    {
        WriteU32(buf, numThreadsX);
        WriteU32(buf, numThreadsY);
        WriteU32(buf, numThreadsZ);
    }

    uint32_t shaderOffset = static_cast<uint32_t>(buf.size());
    uint32_t shaderSize   = static_cast<uint32_t>(sourceCode.size() + 1);
    WriteShaderSource(buf, sourceCode);
    PatchU32(buf, shaderSourceOffsetPos, shaderOffset);
    PatchU32(buf, shaderSourceSizePos,   shaderSize);

    try
    {
        std::filesystem::path outPath(outputPath);
        if (outPath.has_parent_path())
            std::filesystem::create_directories(outPath.parent_path());
    }
    catch (...) {}

    std::ofstream ofs(outputPath, std::ios::binary);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(buf.data()),
              static_cast<std::streamsize>(buf.size()));
    return ofs.good();
}

// =====================================================================
//  WriteStageIOSection  (v5)
//  Each entry: name(str) typeID(u8) semantic(str) category(u8)
// =====================================================================

void NSCWriter::WriteStageIOSection(
    std::vector<uint8_t>&          buf,
    const std::vector<StageIOEntry>& entries)
{
    for (const auto& e : entries)
    {
        WriteStr(buf, e.name);
        WriteU8 (buf, static_cast<uint8_t>(e.typeID));
        WriteStr(buf, e.semantic);
        WriteU8 (buf, static_cast<uint8_t>(e.category));
    }
}

// =====================================================================
//  WritePropertyBlockTable  (v4 NEW)
//  Describes each named PROPERTY block: name + group count.
//  Allows readers to reconstruct the block→group mapping.
// =====================================================================

void NSCWriter::WritePropertyBlockTable(
    std::vector<uint8_t>&   buf,
    const ParsedProperties& props)
{
    // BlockCount already written in header.
    for (const auto& block : props.blocks)
    {
        // Block name
        WriteStr(buf, block.name);
        // Number of groups in this block (2)
        WriteU16(buf, static_cast<uint16_t>(block.groups.size()));
    }
}

// =====================================================================
//  WritePropertyTable  (v4 — adds ownerPropertyName + qualifiedName)
//  Each group record:
//    ResourceKind (1)
//    ShortName    (str)           ← the local group name
//    OwnerPropertyName (str)      ← v4 NEW
//    QualifiedName (str)          ← v4 NEW ("PropertyName__GroupName")
//    Slot (i16)
//    EntryCount (u16)
//    ... entries ...
// =====================================================================

void NSCWriter::WritePropertyTable(
    std::vector<uint8_t>&   buf,
    const ParsedProperties& props)
{
    for (const auto& group : props.groups)
    {
        WriteU8 (buf, static_cast<uint8_t>(group.resourceKind));
        WriteStr (buf, group.name);
        WriteStr (buf, group.ownerPropertyName);    // v4 NEW
        WriteStr (buf, group.qualifiedName);         // v4 NEW
        WriteU16(buf, group.assignedSet);
        WriteU16(buf, group.assignedBinding);
        WriteU16 (buf, static_cast<uint16_t>(group.entries.size()));

        for (const auto& entry : group.entries)
        {
            WriteU8 (buf, static_cast<uint8_t>(entry.kind));
            WriteStr(buf, entry.name);
            WriteStr(buf, entry.typeName);
            WriteStr(buf, entry.defaultValue);
            WriteStr(buf, entry.arraySizeExpr);
            WriteStr(buf, entry.enginePropertyName);
            WriteU16(buf, entry.assignedSet);
            WriteU16(buf, entry.assignedBinding);
        }
    }
}

// =====================================================================
//  WriteBufferSection
//  Key: qualifiedName (v4) instead of short name
// =====================================================================

void NSCWriter::WriteBufferSection(
    std::vector<uint8_t>&                                 buf,
    const ParsedProperties&                               props,
    const std::unordered_map<std::string, AlignedBuffer>& aligned)
{
    for (const auto& group : props.groups)
    {
        if (group.resourceKind != ResourceKind::BUFFER) continue;

        // Try qualifiedName first (v4), fall back to short name (legacy)
        auto it = aligned.find(group.qualifiedName);
        if (it == aligned.end()) it = aligned.find(group.name);
        if (it == aligned.end()) continue;

        const AlignedBuffer& ab = it->second;

        WriteStr(buf, group.qualifiedName); // v4: use qualified name as key
        WriteU16(buf, ab.set);
        WriteU16(buf, ab.binding);
        WriteU32(buf, ab.totalSize);
        WriteU16(buf, static_cast<uint16_t>(ab.fields.size()));

        for (const auto& f : ab.fields)
        {
            WriteU8(buf, static_cast<uint8_t>(f.typeID));
            WriteStr(buf, f.name);
            WriteU32(buf, f.offset);
            WriteU32(buf, f.size);
            uint8_t fieldFlags = 0;
            if (f.isGet)                 fieldFlags |= 0x01;
            if (!f.defaultValue.empty()) fieldFlags |= 0x02;
            WriteU8(buf, fieldFlags);
            WriteStr(buf, f.propertyName);
            WriteStr(buf, f.defaultValue);
        }
    }
}

// =====================================================================
//  WriteTextureSection
// =====================================================================

void NSCWriter::WriteTextureSection(
    std::vector<uint8_t>&   buf,
    const ParsedProperties& props)
{
    for (const auto& group : props.groups)
    {
        if (group.resourceKind != ResourceKind::TEXTURE_2D &&
            group.resourceKind != ResourceKind::TEXTURE_CUBE &&
            group.resourceKind != ResourceKind::TEXTURE_2D_ARRAY &&
            group.resourceKind != ResourceKind::TEXTURE_CUBE_ARRAY) continue;

        for (const auto& entry : group.entries)
        {
            WriteStr(buf, group.qualifiedName); // v4: qualified name
            WriteStr(buf, entry.name);
            WriteU16(buf, entry.assignedSet);
            WriteU16(buf, entry.assignedBinding);
            WriteU8 (buf, static_cast<uint8_t>(entry.typeID));
            WriteStr(buf, (entry.kind == PropertyEntryKind::GET) ? entry.enginePropertyName : "");
        }
    }
}

// =====================================================================
//  WriteSamplerSection
// =====================================================================

void NSCWriter::WriteSamplerSection(
    std::vector<uint8_t>&   buf,
    const ParsedProperties& props)
{
    for (const auto& group : props.groups)
    {
        if (group.resourceKind != ResourceKind::SAMPLER_STATE) continue;

        for (const auto& entry : group.entries)
        {
            WriteStr(buf, group.qualifiedName); // v4: qualified name
            WriteStr(buf, entry.name);
            WriteU16(buf, entry.assignedSet);
            WriteU16(buf, entry.assignedBinding);
            uint8_t samplerType = 0;
            if (entry.kind == PropertyEntryKind::GET)
            {
                const EngineProperty* ep = FindEngineProperty(entry.name);
                if (ep && ep->typeID == DataTypeID::SAMPLER_COMPARISON_STATE)
                    samplerType = 1;
            }
            else
            {
                if (entry.typeID == DataTypeID::SAMPLER_COMPARISON_STATE ||
                    entry.typeName == "SamplerComparisonState" ||
                    group.resourceTypeName == "SamplerComparisonState")
                {
                    samplerType = 1;
                }
            }
            WriteU8(buf, samplerType);
            WriteStr(buf, (entry.kind == PropertyEntryKind::GET) ? entry.enginePropertyName : "");
        }
    }
}

// =====================================================================
//  WriteStructuredBufferSection
// =====================================================================

void NSCWriter::WriteStructuredBufferSection(
    std::vector<uint8_t>&                                 buf,
    const ParsedProperties&                               props,
    const std::unordered_map<std::string, AlignedStruct>& aligned)
{
    for (const auto& group : props.groups)
    {
        if (group.resourceKind != ResourceKind::STRUCTURED_BUFFER &&
            group.resourceKind != ResourceKind::RW_STRUCTURED_BUFFER)
            continue;

        WriteStr(buf, group.qualifiedName); // v4: qualified name
        WriteU16(buf, group.assignedSet);
        WriteU16(buf, group.assignedBinding);

        auto it = aligned.find(group.qualifiedName);
        if (it == aligned.end()) it = aligned.find(group.name);
        if (it != aligned.end())
        {
            const AlignedStruct& as = it->second;
            WriteU32(buf, as.stride);
            WriteU16(buf, static_cast<uint16_t>(as.fields.size()));
            for (const auto& f : as.fields)
            {
                WriteU8 (buf, static_cast<uint8_t>(f.typeID));
                WriteStr(buf, f.name);
                WriteU32(buf, f.offset);
                WriteU32(buf, f.size);
                uint8_t fieldFlags = 0;
                if (f.isGet) fieldFlags |= 0x01;
                WriteU8 (buf, fieldFlags);
                WriteStr(buf, f.propertyName);
            }
        }
        else
        {
            WriteU32(buf, 0); // stride = 0
            WriteU16(buf, 0); // fieldCount = 0
        }
    }
}

// =====================================================================
//  WriteUAVSection
// =====================================================================

void NSCWriter::WriteUAVSection(
    std::vector<uint8_t>&                                 buf,
    const ParsedProperties&                               props,
    const std::unordered_map<std::string, AlignedStruct>& aligned)
{
    for (const auto& group : props.groups)
    {
        if (group.resourceKind != ResourceKind::RW_STRUCTURED_BUFFER &&
            group.resourceKind != ResourceKind::RW_TEXTURE_2D) continue;

        WriteStr(buf, group.qualifiedName); // v4: qualified name
        WriteU8 (buf, static_cast<uint8_t>(group.resourceKind));
        WriteU16(buf, group.assignedSet);
        WriteU16(buf, group.assignedBinding);

        if (group.resourceKind == ResourceKind::RW_TEXTURE_2D)
        {
            WriteU16(buf, static_cast<uint16_t>(group.entries.size()));
            for (const auto& entry : group.entries)
            {
                WriteStr(buf, entry.name);
                WriteStr(buf, entry.typeName);
                WriteU16(buf, entry.assignedSet);
                WriteU16(buf, entry.assignedBinding);
            }
        }
        else
        {
            auto it = aligned.find(group.qualifiedName);
            if (it == aligned.end()) it = aligned.find(group.name);
            if (it != aligned.end())
            {
                const AlignedStruct& as = it->second;
                WriteU16(buf, static_cast<uint16_t>(as.fields.size()));
                for (const auto& f : as.fields)
                {
                    WriteU8 (buf, static_cast<uint8_t>(f.typeID));
                    WriteStr(buf, f.name);
                    WriteU32(buf, f.offset);
                    WriteU32(buf, f.size);
                    uint8_t fieldFlags = 0;
                    if (f.isGet) fieldFlags |= 0x01;
                    WriteU8 (buf, fieldFlags);
                    WriteStr(buf, f.propertyName);
                }
            }
            else
            {
                WriteU16(buf, 0);
            }
        }
    }
}

// =====================================================================
//  WriteShaderSource
// =====================================================================

void NSCWriter::WriteShaderSource(
    std::vector<uint8_t>& buf,
    const std::string&    sourceCode)
{
    for (char c : sourceCode)
        buf.push_back(static_cast<uint8_t>(c));
    buf.push_back(0); // null terminator
}

} // namespace NSL
