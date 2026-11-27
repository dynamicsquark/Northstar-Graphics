#include "nsl_binding_resolver.h"

#include <array>
#include <cctype>
#include <unordered_map>

namespace NSL
{

namespace
{
    inline uint16_t ParseResourceArrayRegisterCount(const std::string& arraySizeExpr) noexcept
    {
        if (arraySizeExpr.empty()) return 1u;
        for (char c : arraySizeExpr)
        {
            if (!std::isdigit(static_cast<unsigned char>(c)))
                return 0u;
        }
        try
        {
            const unsigned long v = std::stoul(arraySizeExpr);
            if (v == 0ul || v > 256ul) return 0u;
            return static_cast<uint16_t>(v);
        }
        catch (...)
        {
            return 0u;
        }
    }

    inline bool AllocateSlotRange(
        uint16_t                                       preferredSpace,
        uint16_t                                       spaceCount,
        uint16_t                                       maxPerSpace,
        uint16_t                                       slotCount,
        std::array<uint16_t, BindingResolver::kMaxSpaces>& cursors,
        uint16_t&                                      outSpace,
        uint16_t&                                      outSlot) noexcept
    {
        if (slotCount == 0u || slotCount > maxPerSpace) return false;
        const uint16_t startSpace = (spaceCount > 0u) ? static_cast<uint16_t>(preferredSpace % spaceCount) : 0u;
        for (uint16_t attempt = 0u; attempt < spaceCount; ++attempt)
        {
            const uint16_t s = static_cast<uint16_t>((startSpace + attempt) % spaceCount);
            if (cursors[s] <= static_cast<uint16_t>(maxPerSpace - slotCount))
            {
                outSpace   = s;
                outSlot    = cursors[s];
                cursors[s] = static_cast<uint16_t>(cursors[s] + slotCount);
                return true;
            }
        }
        return false;
    }
    inline bool AllocateSingleSlot(
        uint16_t                                       preferredSpace,
        uint16_t                                       spaceCount,
        uint16_t                                       maxPerSpace,
        std::array<uint16_t, BindingResolver::kMaxSpaces>& cursors,
        uint16_t&                                      outSpace,
        uint16_t&                                      outSlot) noexcept
    {
        const uint16_t startSpace = (spaceCount > 0u) ? static_cast<uint16_t>(preferredSpace % spaceCount) : 0u;
        for (uint16_t attempt = 0u; attempt < spaceCount; ++attempt)
        {
            const uint16_t s = static_cast<uint16_t>((startSpace + attempt) % spaceCount);
            const uint16_t candidate = cursors[s];
            if (candidate < maxPerSpace)
            {
                outSpace   = s;
                outSlot    = candidate;
                cursors[s] = static_cast<uint16_t>(candidate + 1u);
                return true;
            }
        }
        return false;
    }

    inline uint16_t FindSpaceWithContiguousRoom(
        uint16_t                                           preferredSpace,
        uint16_t                                           spaceCount,
        uint16_t                                           maxPerSpace,
        uint16_t                                           neededCount,
        const std::array<uint16_t, BindingResolver::kMaxSpaces>& cursors) noexcept
    {
        const uint16_t startSpace = (spaceCount > 0u) ? static_cast<uint16_t>(preferredSpace % spaceCount) : 0u;
        for (uint16_t attempt = 0u; attempt < spaceCount; ++attempt)
        {
            const uint16_t s = static_cast<uint16_t>((startSpace + attempt) % spaceCount);
            if (cursors[s] + neededCount <= maxPerSpace)
            {
                return s;
            }
        }
        return 0xFFFFu;
    }
}

// =====================================================================
//  BindingResolver::Resolve
//
//  Decouples logical PROPERTY blocks from physical descriptor spaces.
//  For HLSL12: packs arbitrary logical PROPERTY blocks across space0..space3
//              respecting exact D3D12 root-signature register budgets.
//  For HLSL11: packs arbitrary logical PROPERTY blocks into space0
//              respecting SM 5.0 single-space register limits.
// =====================================================================

bool BindingResolver::Resolve(
    ParsedProperties& props,
    TargetAPI         targetAPI,
    ShaderStage       stage,
    DiagnosticEngine& diag)
{
    (void)stage;
    const bool     useMultiSpace      = (targetAPI == TargetAPI::HLSL12);
    const uint16_t physicalSpaceCount = useMultiSpace ? kMaxSpaces : 1u;

    // Map each logical PROPERTY block name to a deterministic preferred physical space.
    // Blocks 0..15 prefer space0..space15 on HLSL12; blocks 16+ wrap across space0..space15.
    std::unordered_map<std::string, uint16_t> blockToPreferredSpace;
    for (size_t i = 0; i < props.blocks.size(); ++i)
    {
        blockToPreferredSpace[props.blocks[i].name] = useMultiSpace
            ? static_cast<uint16_t>(i % kMaxSpaces)
            : 0u;
    }

    std::array<uint16_t, kMaxSpaces> nextCBV{};
    std::array<uint16_t, kMaxSpaces> nextSRV{};
    std::array<uint16_t, kMaxSpaces> nextSampler{};
    std::array<uint16_t, kMaxSpaces> nextUAV{};

    const uint32_t maxTotalCBVs     = static_cast<uint32_t>(physicalSpaceCount) * kMaxCBVBindings;
    const uint32_t maxTotalSRVs     = static_cast<uint32_t>(physicalSpaceCount) * kMaxSRVBindings;
    const uint32_t maxTotalSamplers = static_cast<uint32_t>(physicalSpaceCount) * kMaxSamplerBindings;
    const uint32_t maxTotalUAVs     = static_cast<uint32_t>(physicalSpaceCount) * kMaxUAVBindings;

    for (auto& group : props.groups)
    {
        uint16_t preferredSpace = 0u;
        auto itPref = blockToPreferredSpace.find(group.ownerPropertyName);
        if (itPref != blockToPreferredSpace.end())
        {
            preferredSpace = itPref->second;
        }

        switch (group.resourceKind)
        {
            // ==================== Buffer -> CBV binding ====================
            case ResourceKind::BUFFER:
            {
                uint16_t assignedSpace = 0u;
                uint16_t assignedSlot  = 0u;
                if (!AllocateSingleSlot(
                        preferredSpace,
                        physicalSpaceCount,
                        kMaxCBVBindings,
                        nextCBV,
                        assignedSpace,
                        assignedSlot))
                {
                    diag.ReportError(group.location,
                        "Constant buffer register limit exceeded (max " +
                        std::to_string(maxTotalCBVs) + " CBVs across " +
                        std::to_string(physicalSpaceCount) + " physical space(s)): group '" +
                        group.name + "' in block '" + group.ownerPropertyName + "'.");
                    return false;
                }
                group.assignedSet     = assignedSpace;
                group.assignedBinding = assignedSlot;
                break;
            }

            // ==================== Texture* -> SRV binding per entry ====================
            case ResourceKind::TEXTURE_2D:
            case ResourceKind::TEXTURE_CUBE:
            case ResourceKind::TEXTURE_2D_ARRAY:
            case ResourceKind::TEXTURE_CUBE_ARRAY:
            {
                uint32_t totalSlotsNeeded = 0u;
                for (const auto& entry : group.entries)
                {
                    const uint16_t cnt = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    if (cnt == 0u)
                    {
                        diag.ReportError(entry.location,
                            "Invalid array size '" + entry.arraySizeExpr + "' on texture '" + entry.name + "'.");
                        return false;
                    }
                    totalSlotsNeeded += cnt;
                }
                const uint16_t contiguousSpace = (totalSlotsNeeded <= kMaxSRVBindings)
                    ? FindSpaceWithContiguousRoom(
                        preferredSpace, physicalSpaceCount, kMaxSRVBindings, static_cast<uint16_t>(totalSlotsNeeded), nextSRV)
                    : 0xFFFFu;
                const uint16_t startSpace = (contiguousSpace != 0xFFFFu) ? contiguousSpace : preferredSpace;

                for (auto& entry : group.entries)
                {
                    const uint16_t regCount = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    uint16_t assignedSpace = 0u;
                    uint16_t assignedSlot  = 0u;
                    if (!AllocateSlotRange(
                            startSpace,
                            physicalSpaceCount,
                            kMaxSRVBindings,
                            regCount,
                            nextSRV,
                            assignedSpace,
                            assignedSlot))
                    {
                        diag.ReportError(entry.location,
                            "SRV register limit exceeded (max " +
                            std::to_string(maxTotalSRVs) + " SRVs across " +
                            std::to_string(physicalSpaceCount) + " physical space(s)): texture '" +
                            entry.name + "' in group '" + group.name + "'.");
                        return false;
                    }
                    entry.assignedSet     = assignedSpace;
                    entry.assignedBinding = assignedSlot;
                }

                group.assignedSet     = group.entries.empty() ? preferredSpace : group.entries.front().assignedSet;
                group.assignedBinding = group.entries.empty()
                                       ? static_cast<uint16_t>(0xFFFF)
                                       : group.entries.front().assignedBinding;
                break;
            }

            // ==================== SamplerState -> Sampler binding per entry ====================
            case ResourceKind::SAMPLER_STATE:
            {
                uint32_t totalSlotsNeeded = 0u;
                for (const auto& entry : group.entries)
                {
                    const uint16_t cnt = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    if (cnt == 0u)
                    {
                        diag.ReportError(entry.location,
                            "Invalid array size '" + entry.arraySizeExpr + "' on sampler '" + entry.name + "'.");
                        return false;
                    }
                    totalSlotsNeeded += cnt;
                }
                const uint16_t contiguousSpace = (totalSlotsNeeded <= kMaxSamplerBindings)
                    ? FindSpaceWithContiguousRoom(
                        preferredSpace, physicalSpaceCount, kMaxSamplerBindings, static_cast<uint16_t>(totalSlotsNeeded), nextSampler)
                    : 0xFFFFu;
                const uint16_t startSpace = (contiguousSpace != 0xFFFFu) ? contiguousSpace : preferredSpace;

                for (auto& entry : group.entries)
                {
                    const uint16_t regCount = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    uint16_t assignedSpace = 0u;
                    uint16_t assignedSlot  = 0u;
                    if (!AllocateSlotRange(
                            startSpace,
                            physicalSpaceCount,
                            kMaxSamplerBindings,
                            regCount,
                            nextSampler,
                            assignedSpace,
                            assignedSlot))
                    {
                        diag.ReportError(entry.location,
                            "Sampler register limit exceeded (max " +
                            std::to_string(maxTotalSamplers) + " samplers across " +
                            std::to_string(physicalSpaceCount) + " physical space(s)): sampler '" +
                            entry.name + "' in group '" + group.name + "'.");
                        return false;
                    }
                    entry.assignedSet     = assignedSpace;
                    entry.assignedBinding = assignedSlot;
                }

                group.assignedSet     = group.entries.empty() ? preferredSpace : group.entries.front().assignedSet;
                group.assignedBinding = group.entries.empty()
                                       ? static_cast<uint16_t>(0xFFFF)
                                       : group.entries.front().assignedBinding;
                break;
            }

            // ==================== StructuredBuffer -> SRV binding ====================
            case ResourceKind::STRUCTURED_BUFFER:
            {
                uint16_t assignedSpace = 0u;
                uint16_t assignedSlot  = 0u;
                if (!AllocateSingleSlot(
                        preferredSpace,
                        physicalSpaceCount,
                        kMaxSRVBindings,
                        nextSRV,
                        assignedSpace,
                        assignedSlot))
                {
                    diag.ReportError(group.location,
                        "SRV register limit exceeded (max " +
                        std::to_string(maxTotalSRVs) + " SRVs across " +
                        std::to_string(physicalSpaceCount) + " physical space(s)): StructuredBuffer '" +
                        group.name + "' in block '" + group.ownerPropertyName + "'.");
                    return false;
                }
                group.assignedSet     = assignedSpace;
                group.assignedBinding = assignedSlot;
                break;
            }

            // ==================== RWStructuredBuffer -> UAV binding ====================
            case ResourceKind::RW_STRUCTURED_BUFFER:
            {
                uint16_t assignedSpace = 0u;
                uint16_t assignedSlot  = 0u;
                if (!AllocateSingleSlot(
                        preferredSpace,
                        physicalSpaceCount,
                        kMaxUAVBindings,
                        nextUAV,
                        assignedSpace,
                        assignedSlot))
                {
                    diag.ReportError(group.location,
                        "UAV register limit exceeded (max " +
                        std::to_string(maxTotalUAVs) + " UAVs across " +
                        std::to_string(physicalSpaceCount) + " physical space(s)): RWStructuredBuffer '" +
                        group.name + "' in block '" + group.ownerPropertyName + "'.");
                    return false;
                }
                group.assignedSet     = assignedSpace;
                group.assignedBinding = assignedSlot;
                break;
            }

            // ==================== RWTexture2D -> UAV binding per entry ====================
            case ResourceKind::RW_TEXTURE_2D:
            {
                uint32_t totalSlotsNeeded = 0u;
                for (const auto& entry : group.entries)
                {
                    const uint16_t cnt = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    if (cnt == 0u)
                    {
                        diag.ReportError(entry.location,
                            "Invalid array size '" + entry.arraySizeExpr + "' on RWTexture2D '" + entry.name + "'.");
                        return false;
                    }
                    totalSlotsNeeded += cnt;
                }
                const uint16_t contiguousSpace = (totalSlotsNeeded <= kMaxUAVBindings)
                    ? FindSpaceWithContiguousRoom(
                        preferredSpace, physicalSpaceCount, kMaxUAVBindings, static_cast<uint16_t>(totalSlotsNeeded), nextUAV)
                    : 0xFFFFu;
                const uint16_t startSpace = (contiguousSpace != 0xFFFFu) ? contiguousSpace : preferredSpace;

                for (auto& entry : group.entries)
                {
                    const uint16_t regCount = ParseResourceArrayRegisterCount(entry.arraySizeExpr);
                    uint16_t assignedSpace = 0u;
                    uint16_t assignedSlot  = 0u;
                    if (!AllocateSlotRange(
                            startSpace,
                            physicalSpaceCount,
                            kMaxUAVBindings,
                            regCount,
                            nextUAV,
                            assignedSpace,
                            assignedSlot))
                    {
                        diag.ReportError(entry.location,
                            "UAV register limit exceeded (max " +
                            std::to_string(maxTotalUAVs) + " UAVs across " +
                            std::to_string(physicalSpaceCount) + " physical space(s)): RWTexture2D '" +
                            entry.name + "' in group '" + group.name + "'.");
                        return false;
                    }
                    entry.assignedSet     = assignedSpace;
                    entry.assignedBinding = assignedSlot;
                }

                group.assignedSet     = group.entries.empty() ? preferredSpace : group.entries.front().assignedSet;
                group.assignedBinding = group.entries.empty()
                                       ? static_cast<uint16_t>(0xFFFF)
                                       : group.entries.front().assignedBinding;
                break;
            }

            default:
            {
                diag.ReportError(group.location,
                    "BindingResolver: Unhandled resource kind for group '" + group.name + "'.");
                return false;
            }
        }
    }

    return true;
}

} // namespace NSL
