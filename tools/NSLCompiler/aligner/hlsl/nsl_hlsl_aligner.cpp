#include "nsl_hlsl_aligner.h"
#include <algorithm>
#include <cctype>

namespace NSL
{
static constexpr uint32_t kMaxArrayElements = 64u * 1024u;
static constexpr uint32_t kMaxLayoutSize    = 16u * 1024u * 1024u;

static bool CheckedAdd(uint32_t a, uint32_t b, uint32_t& out)
{
    const uint64_t value = static_cast<uint64_t>(a) + b;
    if (value > kMaxLayoutSize) return false;
    out = static_cast<uint32_t>(value);
    return true;
}

static bool CheckedMultiply(uint32_t a, uint32_t b, uint32_t& out)
{
    const uint64_t value = static_cast<uint64_t>(a) * b;
    if (value == 0 || value > kMaxLayoutSize) return false;
    out = static_cast<uint32_t>(value);
    return true;
}

// =====================================================================
//  CBuffer Alignment helpers
// =====================================================================

// Returns the next offset that satisfies HLSL cbuffer packing:
// a field of `fieldSize` bytes may not straddle a 16-byte boundary,
// Doubles require 8-byte alignment, and Matrices start on a 16-byte register.
uint32_t HLSLAligner::CBufferAlign(uint32_t currentOffset, uint32_t fieldSize, DataTypeID id)
{
    if (fieldSize == 0 || currentOffset > kMaxLayoutSize)
        return currentOffset;

    if (id == DataTypeID::DOUBLE)
    {
        if (currentOffset > kMaxLayoutSize - 7u) return currentOffset;
        return (currentOffset + 7u) & ~7u;
    }

    if (id == DataTypeID::MAT3 || id == DataTypeID::MAT4)
    {
        if (currentOffset > kMaxLayoutSize - 15u) return currentOffset;
        return (currentOffset + 15u) & ~15u;
    }

    // Clamp fieldSize to 16 for boundary calculation
    uint32_t effectiveSize = std::min(fieldSize, 16u);

    // Check if the field would straddle a 16-byte boundary
    uint32_t startSlot = currentOffset / 16;
    uint32_t endSlot   = (currentOffset + effectiveSize - 1) / 16;

    if (startSlot != endSlot)
    {
        // Push to the next 16-byte boundary
        return (startSlot + 1) * 16;
    }

    return currentOffset;
}

// Returns the size of a type in a cbuffer context.
// Key difference from raw size: Mat3 rows are padded to float4 in cbuffers.
uint32_t HLSLAligner::CBufferSize(DataTypeID id, uint32_t rawSize)
{
    switch (id)
    {
        case DataTypeID::MAT3:
            // float3x3 in a cbuffer = 3 rows, each row padded to float4 = 3 × 16 = 48
            return 48;
        case DataTypeID::MAT4:
            // float4x4 = 4 × 16 = 64
            return 64;
        default:
            return rawSize;
    }
}

// Resolve raw size + NSL type name for a property entry / engine field.
static bool ResolveRawType(
    const std::string& typeName,
    DataTypeID         typeID,
    uint32_t           engineSizeHint,
    std::string&       outTypeName,
    DataTypeID&        outTypeID,
    uint32_t&          outRawSize)
{
    outTypeName = typeName;
    outTypeID   = typeID;
    outRawSize  = 0;

    if (const DataTypeInfo* dti = FindDataType(typeName))
    {
        outTypeID   = dti->id;
        outTypeName = dti->nslName;
        outRawSize  = dti->size;
        return true;
    }

    if (typeID != DataTypeID::UNKNOWN)
    {
        if (const DataTypeInfo* dti = FindDataTypeByID(typeID))
        {
            if (outTypeName.empty())
                outTypeName = dti->nslName;
            outTypeID  = dti->id;
            outRawSize = dti->size;
            return true;
        }
    }

    // Prefer engine-provided raw size when available (non-zero).
    if (engineSizeHint > 0)
    {
        outRawSize = engineSizeHint;
        return true;
    }

    return false;
}

static bool ParseArrayMultiplier(const std::string& arraySizeExpr, uint32_t& outMultiplier)
{
    outMultiplier = 1;
    if (arraySizeExpr.empty())
        return true;

    // Only accept a plain decimal integer literal.
    for (char c : arraySizeExpr)
    {
        if (!std::isdigit(static_cast<unsigned char>(c)))
            return false;
    }

    try
    {
        unsigned long v = std::stoul(arraySizeExpr);
        if (v == 0 || v > kMaxArrayElements)
            return false;
        outMultiplier = static_cast<uint32_t>(v);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// =====================================================================
//  AlignBuffer
//  Lays out a Buffer PROPERTY group as an HLSL cbuffer.
// =====================================================================

AlignedBuffer HLSLAligner::AlignBuffer(
    const PropertyGroup& group,
    const std::vector<StructDecl>& allStructs)
{
    AlignedBuffer result;
    result.name = group.name;
    result.set = group.assignedSet;
    result.binding = group.assignedBinding;

    uint32_t offset = 0;
    uint32_t padCount = 0;

    auto pushField = [&](const std::string& name, const std::string& typeName, DataTypeID typeID, 
                         uint32_t rawSize, bool isGet, const std::string& propertyName, 
                         const std::string& defaultValue, bool isStructOrArray,
                         bool alreadyCBufferSized = false, bool emitExplicitTailPad = false) 
    {
        if (rawSize == 0) return;

        uint32_t cbufferSize = alreadyCBufferSized ? rawSize : CBufferSize(typeID, rawSize);
        if (cbufferSize > kMaxLayoutSize) return;

        // If it's a struct or array, HLSL requires it to be aligned to a 16-byte boundary
        uint32_t alignReq = isStructOrArray ? 16 : 0;
        
        uint32_t alignedOffset = offset;
        if (offset > kMaxLayoutSize - 15u) return;
        if (alignReq == 16)
        {
            alignedOffset = (offset + 15) & ~15;
        }
        else
        {
            alignedOffset = CBufferAlign(offset, cbufferSize, typeID);
        }

        uint32_t fieldEnd = 0;
        if (!CheckedAdd(alignedOffset, cbufferSize, fieldEnd)) return;
        uint32_t nextOffset = fieldEnd;
        if (isStructOrArray)
        {
            if (nextOffset > kMaxLayoutSize - 15u) return;
            nextOffset = (nextOffset + 15u) & ~15u;
        }

        if (alignedOffset > offset)
        {
            AlignedField pad;
            pad.name       = group.qualifiedName + "__pad" + std::to_string(padCount++);
            pad.typeName   = "float";
            pad.typeID     = DataTypeID::FLOAT;
            pad.offset     = offset;
            pad.size       = alignedOffset - offset;
            pad.padding    = 0;
            pad.isPadding  = true;
            result.paddedFields.push_back(pad);
        }

        AlignedField field;
        field.name         = name;
        field.typeName     = typeName;
        field.typeID       = typeID;
        field.offset       = alignedOffset;
        field.size         = cbufferSize;
        field.padding      = 0;
        field.isPadding    = false;
        field.isGet        = isGet;
        field.defaultValue = defaultValue;
        field.propertyName = propertyName;

        result.fields.push_back(field);
        result.paddedFields.push_back(field);

        // Primitive HLSL cbuffer arrays leave the tail of their final 16-byte register open
        // for subsequent scalar packing unless explicitly closed with padding fields.
        // User structs already force FXC/DXC onto the next 16-byte register automatically.
        if (emitExplicitTailPad && nextOffset > fieldEnd)
        {
            AlignedField tailPad;
            tailPad.name      = group.qualifiedName + "__pad" + std::to_string(padCount++);
            tailPad.typeName  = "float";
            tailPad.typeID    = DataTypeID::FLOAT;
            tailPad.offset    = fieldEnd;
            tailPad.size      = nextOffset - fieldEnd;
            tailPad.padding   = 0;
            tailPad.isPadding = true;
            result.paddedFields.push_back(tailPad);
        }

        // Structs/arrays force the next field onto a new 16-byte boundary.
        offset = nextOffset;
    };

    for (const auto& entry : group.entries)
    {
        if (entry.kind == PropertyEntryKind::GET)
        {
            const EngineProperty* ep = FindEngineProperty(entry.name);
            if (ep)
            {
                std::string typeName;
                DataTypeID  typeID = ep->typeID;
                uint32_t    rawSize = 0;
                ResolveRawType(entry.typeName, ep->typeID, ep->size, typeName, typeID, rawSize);
                pushField(entry.name, typeName, typeID, rawSize, true,
                          entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName,
                          entry.defaultValue, false);
                continue;
            }
            else if (entry.typeID != DataTypeID::UNKNOWN)
            {
                std::string typeName;
                DataTypeID  typeID = entry.typeID;
                uint32_t    rawSize = 0;
                if (ResolveRawType(entry.typeName, entry.typeID, 0, typeName, typeID, rawSize))
                {
                    pushField(entry.name, typeName, typeID, rawSize, true,
                              entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName,
                              entry.defaultValue, false);
                    continue;
                }
            }
        }

        // ADD or fallback GET
        uint32_t size = 0;
        bool isStructOrArray = false;
        std::string typeName = entry.typeName;
        DataTypeID  typeID   = entry.typeID;

        const DataTypeInfo* dti = FindDataType(entry.typeName);
        if (!dti && entry.typeID != DataTypeID::UNKNOWN)
            dti = FindDataTypeByID(entry.typeID);

        if (dti)
        {
            size     = dti->size;
            typeID   = dti->id;
            typeName = dti->nslName;
        }
        else
        {
            // Custom struct or struct array — HLSL cbuffer struct packing rules
            size = ResolveCBufferStructSize(entry.typeName, entry.arraySizeExpr, allStructs);
            isStructOrArray = true;
        }

        // Array multiplier for primitives in HLSL cbuffer:
        // Every element of an HLSL cbuffer array starts on a 16-byte register boundary!
        bool alreadyCBufferSized = (dti == nullptr);
        bool emitExplicitTailPad = (dti == nullptr);
        if (dti && !entry.arraySizeExpr.empty())
        {
            uint32_t mult = 1;
            if (ParseArrayMultiplier(entry.arraySizeExpr, mult))
            {
                const uint32_t elemCbSize  = CBufferSize(typeID, size);
                const uint32_t elemStride  = (elemCbSize + 15u) & ~15u;
                uint32_t       prefixBytes = 0;
                uint32_t       totalArrSz  = 0;
                if (mult == 1u)
                {
                    size                = elemCbSize;
                    isStructOrArray     = true;
                    alreadyCBufferSized = true;
                    emitExplicitTailPad = (typeID != DataTypeID::MAT3);
                }
                else if (CheckedMultiply(elemStride, mult - 1u, prefixBytes) &&
                         CheckedAdd(prefixBytes, elemCbSize, totalArrSz))
                {
                    size                = totalArrSz;
                    isStructOrArray     = true;
                    alreadyCBufferSized = true;
                    emitExplicitTailPad = (typeID != DataTypeID::MAT3);
                }
                else
                {
                    size = 0;
                }
            }
            else
            {
                // Invalid array size — skip field (caller should have validated)
                size = 0;
            }
        }

        if (size > 0)
        {
            const bool isGet = (entry.kind == PropertyEntryKind::GET);
            pushField(entry.name, typeName, typeID, size, isGet,
                      isGet ? (entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName) : "",
                      entry.defaultValue, isStructOrArray, alreadyCBufferSized, emitExplicitTailPad);
        }
    }

    // Round up total size to 16-byte multiple
    if (offset > kMaxLayoutSize - 15u)
        return {};
    result.totalSize = (offset + 15u) & ~15u;
    if (result.totalSize == 0) result.totalSize = 16;

    if (result.totalSize > offset)
    {
        AlignedField pad;
        pad.name       = group.qualifiedName + "__pad" + std::to_string(padCount++);
        pad.typeName   = "float";
        pad.typeID     = DataTypeID::FLOAT;
        pad.offset     = offset;
        pad.size       = result.totalSize - offset;
        pad.padding    = 0;
        pad.isPadding  = true;
        result.paddedFields.push_back(pad);
    }

    return result;
}

// =====================================================================
//  AlignStructuredBuffer
//  Lays out a StructuredBuffer PROPERTY group with tight natural packing.
// =====================================================================

AlignedStruct HLSLAligner::AlignStructuredBuffer(
    const PropertyGroup& group,
    const std::vector<StructDecl>& allStructs)
{
    AlignedStruct result;
    result.name = group.name + "__Struct";

    uint32_t offset = 0;
    uint32_t maxAlign = 1;

    auto pushTight = [&](const std::string& name, const std::string& typeName, DataTypeID typeID,
                         uint32_t rawSize, bool isGet, const std::string& propertyName)
    {
        if (rawSize == 0) return;

        uint32_t align = GetTightFieldAlignment(typeName, typeID, rawSize, allStructs);
        if (align == 0) align = 4;
        maxAlign = std::max(maxAlign, align);
        if (offset > kMaxLayoutSize - (align - 1)) return;
        uint32_t alignedOffset = (offset + align - 1) & ~(align - 1);
        uint32_t nextOffset = 0;
        if (!CheckedAdd(alignedOffset, rawSize, nextOffset)) return;

        AlignedStructField f;
        f.name         = name;
        f.typeName     = typeName;
        f.typeID       = typeID;
        f.offset       = alignedOffset;
        f.size         = rawSize; // tight — never apply cbuffer Mat3 padding here
        f.isGet        = isGet;
        f.propertyName = propertyName;

        result.fields.push_back(f);
        offset = nextOffset;
    };

    for (const auto& entry : group.entries)
    {
        if (entry.kind == PropertyEntryKind::GET)
        {
            const EngineProperty* ep = FindEngineProperty(entry.name);
            if (ep)
            {
                std::string typeName;
                DataTypeID  typeID = ep->typeID;
                uint32_t    rawSize = 0;
                ResolveRawType(entry.typeName, ep->typeID, ep->size, typeName, typeID, rawSize);
                pushTight(entry.name, typeName, typeID, rawSize, true,
                          entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName);
                continue;
            }
            else if (entry.typeID != DataTypeID::UNKNOWN)
            {
                std::string typeName;
                DataTypeID  typeID = entry.typeID;
                uint32_t    rawSize = 0;
                if (ResolveRawType(entry.typeName, entry.typeID, 0, typeName, typeID, rawSize))
                {
                    pushTight(entry.name, typeName, typeID, rawSize, true,
                              entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName);
                    continue;
                }
            }
        }

        // Add() or unknown GET fallback (including custom structs)
        std::string typeName;
        DataTypeID  typeID = entry.typeID;
        uint32_t    rawSize = 0;
        if (ResolveRawType(entry.typeName, entry.typeID, 0, typeName, typeID, rawSize))
        {
            if (!entry.arraySizeExpr.empty())
            {
                uint32_t mult = 1;
                if (!ParseArrayMultiplier(entry.arraySizeExpr, mult))
                    continue;
                if (!CheckedMultiply(rawSize, mult, rawSize))
                    continue;
            }
        }
        else if (!allStructs.empty())
        {
            rawSize  = ResolveStructFieldSize(entry.typeName, entry.arraySizeExpr, allStructs);
            typeName = entry.typeName;
            if (rawSize == 0)
                continue;
        }
        else
        {
            continue;
        }

        const bool isGet = (entry.kind == PropertyEntryKind::GET);
        pushTight(entry.name, typeName, typeID, rawSize, isGet,
                  isGet ? (entry.enginePropertyName.empty() ? entry.name : entry.enginePropertyName) : "");
    }

    if (offset > 0 && maxAlign > 1 && offset <= kMaxLayoutSize - (maxAlign - 1))
    {
        offset = (offset + maxAlign - 1) & ~(maxAlign - 1);
    }
    result.stride = offset;
    return result;
}

// =====================================================================
//  AlignStruct
//  Lays out a user-defined struct from the SHADER AST block.
//  Uses tight natural alignment (same rules as StructuredBuffer).
// =====================================================================

AlignedStruct HLSLAligner::AlignStruct(
    const StructDecl&              decl,
    const std::vector<StructDecl>& allStructs)
{
    AlignedStruct result;
    result.name = decl.name;

    uint32_t offset = 0;
    uint32_t maxAlign = 1;

    for (const auto& f : decl.fields)
    {
        uint32_t size = ResolveStructFieldSize(f.typeName, f.arraySizeExpr, allStructs);
        if (size == 0) continue;

        DataTypeID fid        = GetDataTypeID(f.typeName);
        uint32_t align        = GetTightFieldAlignment(f.typeName, fid, size, allStructs);
        if (align == 0) align = 4;
        maxAlign = std::max(maxAlign, align);
        if (offset > kMaxLayoutSize - (align - 1)) continue;
        uint32_t alignedOffset = (offset + align - 1) & ~(align - 1);
        uint32_t nextOffset = 0;
        if (!CheckedAdd(alignedOffset, size, nextOffset)) continue;

        AlignedStructField af;
        af.name     = f.name;
        af.typeName = f.typeName;
        af.typeID   = fid;
        af.offset   = alignedOffset;
        af.size     = size;

        result.fields.push_back(af);
        offset = nextOffset;
    }

    if (offset > 0 && maxAlign > 1 && offset <= kMaxLayoutSize - (maxAlign - 1))
    {
        offset = (offset + maxAlign - 1) & ~(maxAlign - 1);
    }
    result.stride = offset;
    return result;
}

// =====================================================================
//  GetTightFieldAlignment — natural C/HLSL StructuredBuffer alignment
// =====================================================================

uint32_t HLSLAligner::GetTightFieldAlignment(
    const std::string&             typeName,
    DataTypeID                     typeID,
    uint32_t                       rawSize,
    const std::vector<StructDecl>& allStructs,
    uint32_t                       depth) const
{
    if (depth >= 64) return 4u;

    if (typeID == DataTypeID::DOUBLE || typeName == "Double")
        return 8u;

    if (typeID != DataTypeID::UNKNOWN || FindDataType(typeName) != nullptr)
        return std::min(rawSize, 4u);

    for (const auto& s : allStructs)
    {
        if (s.name == typeName)
        {
            uint32_t maxFieldAlign = 4u;
            for (const auto& f : s.fields)
            {
                const DataTypeID fid = GetDataTypeID(f.typeName);
                const DataTypeInfo* dti = FindDataType(f.typeName);
                const uint32_t fRaw = dti ? dti->size : 4u;
                maxFieldAlign = std::max(
                    maxFieldAlign,
                    GetTightFieldAlignment(f.typeName, fid, fRaw, allStructs, depth + 1));
            }
            return maxFieldAlign;
        }
    }

    return std::min(rawSize, 4u);
}

// =====================================================================
//  ResolveCBufferStructSize — HLSL cbuffer packing for user structs
// =====================================================================

uint32_t HLSLAligner::ResolveCBufferStructSize(
    const std::string&             typeName,
    const std::string&             arraySizeExpr,
    const std::vector<StructDecl>& allStructs,
    uint32_t                       depth) const
{
    if (depth >= 64) return 0;

    const StructDecl* targetStruct = nullptr;
    for (const auto& s : allStructs)
    {
        if (s.name == typeName)
        {
            targetStruct = &s;
            break;
        }
    }
    if (!targetStruct || targetStruct->fields.empty())
        return 0;

    uint32_t offset = 0;
    uint32_t lastOccupiedEnd = 0;
    for (const auto& f : targetStruct->fields)
    {
        const DataTypeInfo* dti = FindDataType(f.typeName);
        if (dti)
        {
            const uint32_t elemOccupied = (dti->id == DataTypeID::MAT3) ? 44u
                                        : (dti->id == DataTypeID::MAT4) ? 64u
                                                                        : dti->size;
            if (f.arraySizeExpr.empty())
            {
                const uint32_t aligned = CBufferAlign(offset, elemOccupied, dti->id);
                if (!CheckedAdd(aligned, elemOccupied, lastOccupiedEnd))
                    return 0;
                offset = lastOccupiedEnd;
            }
            else
            {
                uint32_t mult = 1;
                if (!ParseArrayMultiplier(f.arraySizeExpr, mult))
                    return 0;
                if (offset > kMaxLayoutSize - 15u) return 0;
                const uint32_t aligned    = (offset + 15u) & ~15u;
                const uint32_t elemStride = (elemOccupied + 15u) & ~15u;
                uint32_t arrBytes = elemOccupied;
                if (mult > 1u)
                {
                    uint32_t prefix = 0;
                    if (!CheckedMultiply(elemStride, mult - 1u, prefix) ||
                        !CheckedAdd(prefix, elemOccupied, arrBytes))
                    {
                        return 0;
                    }
                }
                if (!CheckedAdd(aligned, arrBytes, lastOccupiedEnd))
                    return 0;
                // In FXC, trailing bytes in the final 16-byte register of an array or struct
                // remain open for subsequent scalar/vector fields inside the enclosing struct.
                offset = lastOccupiedEnd;
            }
        }
        else
        {
            const uint32_t subOccupied = ResolveCBufferStructSize(
                f.typeName, f.arraySizeExpr, allStructs, depth + 1);
            if (subOccupied == 0) return 0;
            if (offset > kMaxLayoutSize - 15u) return 0;
            const uint32_t aligned = (offset + 15u) & ~15u;
            if (!CheckedAdd(aligned, subOccupied, lastOccupiedEnd))
                return 0;
            offset = lastOccupiedEnd;
        }
    }

    if (lastOccupiedEnd == 0) return 0;

    if (arraySizeExpr.empty())
        return lastOccupiedEnd;

    uint32_t mult = 1;
    if (!ParseArrayMultiplier(arraySizeExpr, mult))
        return 0;
    if (mult == 1u)
        return lastOccupiedEnd;

    const uint32_t elemStride = (lastOccupiedEnd + 15u) & ~15u;
    uint32_t prefix = 0;
    uint32_t total  = 0;
    if (!CheckedMultiply(elemStride, mult - 1u, prefix) ||
        !CheckedAdd(prefix, lastOccupiedEnd, total))
    {
        return 0;
    }
    return total;
}

// =====================================================================
//  ResolveStructFieldSize — handles both primitive and nested struct types
// =====================================================================

uint32_t HLSLAligner::ResolveStructFieldSize(
    const std::string&             typeName,
    const std::string&             arraySizeExpr,
    const std::vector<StructDecl>& allStructs,
    uint32_t                       depth) const
{
    if (depth >= 64) return 0;

    uint32_t baseSize = 0;

    // Try primitive type first
    const DataTypeInfo* dti = FindDataType(typeName);
    if (dti) 
    {
        baseSize = dti->size;
    }
    else
    {
        // Try nested struct
        for (const auto& s : allStructs)
        {
            if (s.name == typeName)
            {
                // Recursively compute its size with natural alignment + tail stride rounding
                uint32_t total = 0;
                uint32_t maxAlign = 1;
                for (const auto& f : s.fields)
                {
                    uint32_t fsize = ResolveStructFieldSize(
                        f.typeName, f.arraySizeExpr, allStructs, depth + 1);
                    if (fsize == 0) return 0;
                    DataTypeID fid = GetDataTypeID(f.typeName);
                    uint32_t align = GetTightFieldAlignment(f.typeName, fid, fsize, allStructs, depth + 1);
                    if (align == 0) align = 4;
                    maxAlign = std::max(maxAlign, align);
                    if (total > kMaxLayoutSize - (align - 1)) return 0;
                    uint32_t aligned = (total + align - 1) & ~(align - 1);
                    if (!CheckedAdd(aligned, fsize, total)) return 0;
                }
                if (total > 0 && maxAlign > 1)
                {
                    if (total > kMaxLayoutSize - (maxAlign - 1)) return 0;
                    total = (total + maxAlign - 1) & ~(maxAlign - 1);
                }
                baseSize = total;
                break;
            }
        }
    }

    if (baseSize == 0) return 0; // Unknown / resource type

    // Multiply by array size if applicable
    uint32_t multiplier = 1;
    if (!arraySizeExpr.empty())
    {
        if (!ParseArrayMultiplier(arraySizeExpr, multiplier))
            return 0; // reject non-literal / invalid sizes instead of silently using 1
    }

    uint32_t result = 0;
    return CheckedMultiply(baseSize, multiplier, result) ? result : 0;
}

} // namespace NSL
