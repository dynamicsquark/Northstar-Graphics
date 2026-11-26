#pragma once

#include "nsl_common_types.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// =====================================================================
//  NSL Shader Data — shared reflection payload
//
//  Single source of truth for .nsc reflection used by:
//    • NSCReader  (fills from binary)
//    • Northstar  (runtime Shader / Material binding)
//
//  Reader / Cursor / parsers stay in NSCReader.
// =====================================================================

namespace NSL
{
    // ========================= FIELD TYPES ===========================

    struct NSCField
    {
        std::string  name;           // Mangled: "Materials__MaterialParams__baseColor"
        DataTypeID   typeID = DataTypeID::UNKNOWN;
        uint32_t     offset = 0;
        uint32_t     size   = 0;
        uint32_t     arrayCount = 1;
        bool         isGet  = false;
        std::string  propertyName;
        std::string  defaultValue;
        std::vector<uint8_t> defaultData;
    };

    struct NSCBuffer
    {
        std::string           name;       // qualifiedName e.g. "Materials__MaterialParams"
        uint16_t              set = 0;
        uint16_t              binding = 0xFFFF;
        uint32_t              totalSize = 0;
        std::vector<NSCField> fields;

        bool IsEngineProvided() const {
            for (const auto& f : fields)
                if (f.isGet) return true;
            return false;
        }
    };

    struct NSCTexture
    {
        std::string  groupQualifiedName; // e.g. "Textures__UserTextures"
        std::string  name;               // entry name "AlbedoMap"
        uint16_t     set = 0;
        uint16_t     binding = 0xFFFF;
        uint16_t     arrayCount = 1;
        DataTypeID   typeID = DataTypeID::TEXTURE_2D;
        std::string  propertyName;       // Non-empty = engine-provided (Get)

        bool IsEngineProvided() const { return !propertyName.empty(); }
    };

    struct NSCSampler
    {
        std::string  groupQualifiedName;
        std::string  name;
        uint16_t     set = 0;
        uint16_t     binding = 0xFFFF;
        uint16_t     arrayCount = 1;
        bool         isComparison = false;
        std::string  propertyName;

        bool IsEngineProvided() const { return !propertyName.empty(); }
    };

    struct NSCStructField
    {
        std::string  name;
        DataTypeID   typeID = DataTypeID::UNKNOWN;
        uint32_t     offset = 0;
        uint32_t     size   = 0;
        uint16_t     set = 0;
        uint16_t     binding = 0xFFFF;
        uint16_t     arrayCount = 1;
        bool         isGet  = false;
        std::string  propertyName;
    };

    struct NSCStructuredBuffer
    {
        std::string                 name;   // qualifiedName
        uint16_t                    set = 0;
        uint16_t                    binding = 0xFFFF;
        uint32_t                    stride = 0;
        std::vector<NSCStructField> fields;
        std::string                 propertyName;

        bool IsEngineProvided() const { return !propertyName.empty(); }
    };

    struct NSCUAVBuffer
    {
        std::string                 name;   // qualifiedName
        ResourceKind                resourceKind = ResourceKind::RW_STRUCTURED_BUFFER;
        uint16_t                    set = 0;
        uint16_t                    binding = 0xFFFF;
        std::vector<NSCStructField> fields;
    };

    // ========================= PROPERTY GROUP / BLOCK =================

    struct NSCPropertyEntry
    {
        PropertyEntryKind        kind = PropertyEntryKind::ADD;
        std::string              name;
        std::string              typeName;
        std::string              defaultValue;
        std::vector<uint8_t>     defaultData;
        std::string              arraySizeExpr;
        std::string              enginePropertyName;
        uint16_t                 set = 0;
        uint16_t                 binding = 0xFFFF;
    };

    struct NSCPropertyGroup
    {
        std::string                   name;              // Short/local group name
        std::string                   ownerPropertyName; // parent PROPERTY block
        std::string                   qualifiedName;     // "PropertyName__GroupName"
        ResourceKind                  resourceKind = ResourceKind::UNKNOWN;
        uint16_t                      set = 0;
        uint16_t                      binding = 0xFFFF;
        std::vector<NSCPropertyEntry> entries;
    };

    struct NSCPropertyBlock
    {
        std::string                    name;
        std::vector<NSCPropertyGroup*> groups; // views into ShaderData::propertyGroups
    };

    // ========================= STAGE IO LAYOUT =======================

    struct NSCStageIOEntry
    {
        std::string name;           // Get()/Add() id: "VERTEX_POSITION"
        DataTypeID  typeID = DataTypeID::UNKNOWN;
        std::string semantic;       // Opaque target-API semantic (filled by compiler backend)
        uint8_t     category = 0;   // StageIOCategory as u8
    };

    // ========================= NSC HEADER FLAGS ======================

    enum NSCFlags : uint16_t
    {
        NSC_FLAG_NONE                   = 0,
        NSC_FLAG_HAS_UAV                = (1 << 0),
        NSC_FLAG_USES_DISCARD           = (1 << 1),
        NSC_FLAG_WRITES_DEPTH           = (1 << 2),
        NSC_FLAG_HAS_COMPUTE_NUMTHREADS = (1 << 3),
    };

    // ========================= SHADER DATA ===========================

    struct ShaderData
    {
        ShaderStage stage      = ShaderStage::NONE;
        TargetAPI   api        = TargetAPI::UNKNOWN;
        uint32_t    nscVersion = 0;
        uint16_t    flags      = NSC_FLAG_NONE;

        // Compute thread-group dimensions from binary reflection ([numthreads(X, Y, Z)]).
        // Zero for non-compute stages or when unknown.
        uint32_t    numThreadsX = 0;
        uint32_t    numThreadsY = 0;
        uint32_t    numThreadsZ = 0;

        std::vector<NSCPropertyGroup>    propertyGroups;
        std::vector<NSCPropertyBlock>    propertyBlocks;
        std::vector<NSCBuffer>           buffers;
        std::vector<NSCTexture>          textures;
        std::vector<NSCSampler>          samplers;
        std::vector<NSCStructuredBuffer> structuredBuffers;
        std::vector<NSCUAVBuffer>        uavBuffers;

        std::vector<NSCStageIOEntry>     inputLayout;   // v5
        std::vector<NSCStageIOEntry>     outputLayout;  // v5

        std::string sourceCode; // Transpiled source for `api`

        // -------------------- Lifecycle --------------------

        ShaderData() = default;
        ~ShaderData() = default;

        ShaderData(const ShaderData& other)
            : stage(other.stage)
            , api(other.api)
            , nscVersion(other.nscVersion)
            , flags(other.flags)
            , numThreadsX(other.numThreadsX)
            , numThreadsY(other.numThreadsY)
            , numThreadsZ(other.numThreadsZ)
            , propertyGroups(other.propertyGroups)
            , propertyBlocks(other.propertyBlocks)
            , buffers(other.buffers)
            , textures(other.textures)
            , samplers(other.samplers)
            , structuredBuffers(other.structuredBuffers)
            , uavBuffers(other.uavBuffers)
            , inputLayout(other.inputLayout)
            , outputLayout(other.outputLayout)
            , sourceCode(other.sourceCode)
        {
            RebuildPropertyBlockGroupPointers();
        }

        ShaderData& operator=(const ShaderData& other)
        {
            if (this == &other) return *this;
            stage              = other.stage;
            api                = other.api;
            nscVersion         = other.nscVersion;
            flags              = other.flags;
            numThreadsX        = other.numThreadsX;
            numThreadsY        = other.numThreadsY;
            numThreadsZ        = other.numThreadsZ;
            propertyGroups     = other.propertyGroups;
            propertyBlocks     = other.propertyBlocks;
            buffers            = other.buffers;
            textures           = other.textures;
            samplers           = other.samplers;
            structuredBuffers  = other.structuredBuffers;
            uavBuffers         = other.uavBuffers;
            inputLayout        = other.inputLayout;
            outputLayout       = other.outputLayout;
            sourceCode         = other.sourceCode;
            RebuildPropertyBlockGroupPointers();
            return *this;
        }

        ShaderData(ShaderData&& other) noexcept
            : stage(other.stage)
            , api(other.api)
            , nscVersion(other.nscVersion)
            , flags(other.flags)
            , numThreadsX(other.numThreadsX)
            , numThreadsY(other.numThreadsY)
            , numThreadsZ(other.numThreadsZ)
            , propertyGroups(std::move(other.propertyGroups))
            , propertyBlocks(std::move(other.propertyBlocks))
            , buffers(std::move(other.buffers))
            , textures(std::move(other.textures))
            , samplers(std::move(other.samplers))
            , structuredBuffers(std::move(other.structuredBuffers))
            , uavBuffers(std::move(other.uavBuffers))
            , inputLayout(std::move(other.inputLayout))
            , outputLayout(std::move(other.outputLayout))
            , sourceCode(std::move(other.sourceCode))
        {
            RebuildPropertyBlockGroupPointers();
            other.Clear();
        }

        ShaderData& operator=(ShaderData&& other) noexcept
        {
            if (this == &other) return *this;
            stage              = other.stage;
            api                = other.api;
            nscVersion         = other.nscVersion;
            flags              = other.flags;
            numThreadsX        = other.numThreadsX;
            numThreadsY        = other.numThreadsY;
            numThreadsZ        = other.numThreadsZ;
            propertyGroups     = std::move(other.propertyGroups);
            propertyBlocks     = std::move(other.propertyBlocks);
            buffers            = std::move(other.buffers);
            textures           = std::move(other.textures);
            samplers           = std::move(other.samplers);
            structuredBuffers  = std::move(other.structuredBuffers);
            uavBuffers         = std::move(other.uavBuffers);
            inputLayout        = std::move(other.inputLayout);
            outputLayout       = std::move(other.outputLayout);
            sourceCode         = std::move(other.sourceCode);
            RebuildPropertyBlockGroupPointers();
            other.Clear();
            return *this;
        }

        void Clear()
        {
            stage       = ShaderStage::NONE;
            api         = TargetAPI::UNKNOWN;
            nscVersion  = 0;
            flags       = NSC_FLAG_NONE;
            numThreadsX = 0;
            numThreadsY = 0;
            numThreadsZ = 0;
            propertyGroups.clear();
            propertyBlocks.clear();
            buffers.clear();
            textures.clear();
            samplers.clear();
            structuredBuffers.clear();
            uavBuffers.clear();
            inputLayout.clear();
            outputLayout.clear();
            sourceCode.clear();
        }

        // -------------------- Flags / Capabilities --------------------

        bool HasUAV()                  const { return (flags & NSC_FLAG_HAS_UAV) != 0 || !uavBuffers.empty(); }
        bool UsesDiscard()             const { return (flags & NSC_FLAG_USES_DISCARD) != 0; }
        bool WritesDepth()             const { return (flags & NSC_FLAG_WRITES_DEPTH) != 0; }
        bool ModifiesCoverageOrDepth() const { return UsesDiscard() || WritesDepth(); }
        bool HasValidNumThreads()      const { return numThreadsX > 0 && numThreadsY > 0 && numThreadsZ > 0; }

        void RebuildPropertyBlockGroupPointers()
        {
            for (NSCPropertyBlock& block : propertyBlocks)
            {
                block.groups.clear();
                for (NSCPropertyGroup& group : propertyGroups)
                {
                    if (group.ownerPropertyName == block.name)
                        block.groups.push_back(&group);
                }
            }
        }

        // -------------------- Counts --------------------

        uint32_t GetBufferCount()           const { return static_cast<uint32_t>(buffers.size()); }
        uint32_t GetTextureCount()          const { return static_cast<uint32_t>(textures.size()); }
        uint32_t GetSamplerCount()          const { return static_cast<uint32_t>(samplers.size()); }
        uint32_t GetStructuredBufferCount() const { return static_cast<uint32_t>(structuredBuffers.size()); }
        uint32_t GetUAVCount()              const { return static_cast<uint32_t>(uavBuffers.size()); }
        uint32_t GetPropertyGroupCount()    const { return static_cast<uint32_t>(propertyGroups.size()); }
        uint32_t GetPropertyBlockCount()    const { return static_cast<uint32_t>(propertyBlocks.size()); }

        // -------------------- Finders --------------------

        const NSCBuffer* FindBuffer(const std::string& qualifiedName) const
        {
            for (const NSCBuffer& b : buffers)
                if (b.name == qualifiedName) return &b;
            return nullptr;
        }

        const NSCBuffer* FindBufferByBinding(uint16_t set, uint16_t binding) const
        {
            for (const NSCBuffer& b : buffers)
                if (b.set == set && b.binding == binding) return &b;
            return nullptr;
        }

        const NSCPropertyGroup* FindPropertyGroup(const std::string& qualifiedOrShortName) const
        {
            for (const NSCPropertyGroup& g : propertyGroups)
                if (g.qualifiedName == qualifiedOrShortName) return &g;
            for (const NSCPropertyGroup& g : propertyGroups)
                if (g.name == qualifiedOrShortName) return &g;
            return nullptr;
        }

        const NSCPropertyGroup* FindPropertyGroupInBlock(
            const std::string& blockName,
            const std::string& groupName) const
        {
            for (const NSCPropertyGroup& g : propertyGroups)
                if (g.ownerPropertyName == blockName && g.name == groupName) return &g;
            return nullptr;
        }

        const NSCPropertyBlock* FindPropertyBlock(const std::string& blockName) const
        {
            for (const NSCPropertyBlock& b : propertyBlocks)
                if (b.name == blockName) return &b;
            return nullptr;
        }

        const NSCTexture* FindTexture(
            const std::string& groupQualifiedName,
            const std::string& entryName) const
        {
            for (const NSCTexture& t : textures)
                if (t.groupQualifiedName == groupQualifiedName && t.name == entryName) return &t;
            return nullptr;
        }

        const NSCTexture* FindTexture(const std::string& name) const
        {
            for (const NSCTexture& t : textures)
                if (t.name == name || t.groupQualifiedName == name) return &t;
            return nullptr;
        }

        const NSCTexture* FindTextureByBinding(uint16_t set, uint16_t binding) const
        {
            for (const NSCTexture& t : textures)
                if (t.set == set && t.binding == binding) return &t;
            return nullptr;
        }

        const NSCSampler* FindSampler(
            const std::string& groupQualifiedName,
            const std::string& entryName) const
        {
            for (const NSCSampler& s : samplers)
                if (s.groupQualifiedName == groupQualifiedName && s.name == entryName) return &s;
            return nullptr;
        }

        const NSCSampler* FindSampler(const std::string& name) const
        {
            for (const NSCSampler& s : samplers)
                if (s.name == name || s.groupQualifiedName == name) return &s;
            return nullptr;
        }

        const NSCSampler* FindSamplerByBinding(uint16_t set, uint16_t binding) const
        {
            for (const NSCSampler& s : samplers)
                if (s.set == set && s.binding == binding) return &s;
            return nullptr;
        }

        const NSCStructuredBuffer* FindStructuredBuffer(const std::string& qualifiedName) const
        {
            for (const NSCStructuredBuffer& sb : structuredBuffers)
                if (sb.name == qualifiedName) return &sb;
            return nullptr;
        }

        const NSCStructuredBuffer* FindStructuredBufferByBinding(uint16_t set, uint16_t binding) const
        {
            for (const NSCStructuredBuffer& sb : structuredBuffers)
                if (sb.set == set && sb.binding == binding) return &sb;
            return nullptr;
        }

        const NSCUAVBuffer* FindUAV(const std::string& qualifiedName) const
        {
            for (const NSCUAVBuffer& u : uavBuffers)
                if (u.name == qualifiedName) return &u;
            return nullptr;
        }

        const NSCUAVBuffer* FindUAVByBinding(uint16_t set, uint16_t binding) const
        {
            for (const NSCUAVBuffer& u : uavBuffers)
                if (u.set == set && u.binding == binding) return &u;
            return nullptr;
        }

        const NSCField* FindBufferField(uint16_t set, uint16_t binding, const std::string& fieldName) const
        {
            const NSCBuffer* buf = FindBufferByBinding(set, binding);
            if (!buf) return nullptr;
            for (const NSCField& f : buf->fields)
                if (f.name == fieldName) return &f;
            return nullptr;
        }

        bool HasEngineProperty(const std::string& propName) const
        {
            for (const NSCBuffer& b : buffers)
                for (const NSCField& f : b.fields)
                    if (f.isGet && f.propertyName == propName)
                        return true;

            for (const NSCTexture& t : textures)
                if (t.IsEngineProvided() && t.propertyName == propName)
                    return true;

            for (const NSCSampler& s : samplers)
                if (s.IsEngineProvided() && s.propertyName == propName)
                    return true;

            for (const NSCStructuredBuffer& sb : structuredBuffers)
            {
                if (sb.IsEngineProvided() && sb.propertyName == propName)
                    return true;
                for (const NSCStructField& f : sb.fields)
                    if (f.isGet && f.propertyName == propName)
                        return true;
            }

            return false;
        }
    };

} // namespace NSL
