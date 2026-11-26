#include "d3d12_resource_state_tracker.h"
#include <algorithm>

namespace Northstar
{
    using namespace D3D12Util;

    D3D12_RESOURCE_STATES D3D12ResourceStateTracker::ToD3D12ResourceState(GPUResourceAccess access)
    {
        if (access == GPUResourceAccess::COMMON)
        {
            return D3D12_RESOURCE_STATE_COMMON;
        }
        if (HasGPUResourceAccess(access, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE))
        {
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        if (HasGPUResourceAccess(access, GPUResourceAccess::RENDER_TARGET))
        {
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        }
        if (HasGPUResourceAccess(access, GPUResourceAccess::DEPTH_WRITE))
        {
            return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }
        if (HasGPUResourceAccess(access, GPUResourceAccess::COPY_DEST))
        {
            return D3D12_RESOURCE_STATE_COPY_DEST;
        }
        if (HasGPUResourceAccess(access, GPUResourceAccess::PRESENT))
        {
            return D3D12_RESOURCE_STATE_PRESENT;
        }

        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        if (HasGPUResourceAccess(access, GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER))
            state |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        if (HasGPUResourceAccess(access, GPUResourceAccess::INDEX_BUFFER))
            state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        if (HasGPUResourceAccess(access, GPUResourceAccess::DEPTH_READ))
            state |= D3D12_RESOURCE_STATE_DEPTH_READ;
        if (HasGPUResourceAccess(access, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE))
            state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (HasGPUResourceAccess(access, GPUResourceAccess::PIXEL_SHADER_RESOURCE))
            state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if (HasGPUResourceAccess(access, GPUResourceAccess::INDIRECT_ARGUMENT))
            state |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        if (HasGPUResourceAccess(access, GPUResourceAccess::COPY_SOURCE))
            state |= D3D12_RESOURCE_STATE_COPY_SOURCE;

        return state;
    }

    GPUResourceAccess D3D12ResourceStateTracker::InferAccessFromD3D12State(D3D12_RESOURCE_STATES state)
    {
        if (state == D3D12_RESOURCE_STATE_COMMON)
        {
            return GPUResourceAccess::COMMON;
        }
        if (state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            return GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
        }
        if (state & D3D12_RESOURCE_STATE_RENDER_TARGET)
        {
            return GPUResourceAccess::RENDER_TARGET;
        }
        if (state & D3D12_RESOURCE_STATE_DEPTH_WRITE)
        {
            return GPUResourceAccess::DEPTH_WRITE;
        }
        if (state & D3D12_RESOURCE_STATE_COPY_DEST)
        {
            return GPUResourceAccess::COPY_DEST;
        }

        GPUResourceAccess access = GPUResourceAccess::COMMON;
        if (state & D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)
            access |= GPUResourceAccess::VERTEX_AND_CONSTANT_BUFFER;
        if (state & D3D12_RESOURCE_STATE_INDEX_BUFFER)
            access |= GPUResourceAccess::INDEX_BUFFER;
        if (state & D3D12_RESOURCE_STATE_DEPTH_READ)
            access |= GPUResourceAccess::DEPTH_READ;
        if (state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            access |= GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE;
        if (state & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
            access |= GPUResourceAccess::PIXEL_SHADER_RESOURCE;
        if (state & D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
            access |= GPUResourceAccess::INDIRECT_ARGUMENT;
        if (state & D3D12_RESOURCE_STATE_COPY_SOURCE)
            access |= GPUResourceAccess::COPY_SOURCE;

        return access;
    }

    void D3D12ResourceStateTracker::RegisterResource(
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES initialState,
        GPUResourceAccess initialAccess,
        UINT32 subresourceCount,
        UINT32 mipLevels,
        UINT32 arraySlices,
        UINT32 planeCount)
    {
        if (!resource) return;

        const UINT32 effMips   = (mipLevels > 0u) ? mipLevels : 1u;
        const UINT32 effSlices = (arraySlices > 0u) ? arraySlices : 1u;
        const UINT32 effPlanes = (planeCount > 0u) ? planeCount : 1u;
        const UINT32 computedSubres = effMips * effSlices * effPlanes;
        const UINT32 effSubres = (std::max)((subresourceCount > 0u) ? subresourceCount : 1u, computedSubres);

        D3D12TrackedResourceEntry entry{};
        entry.currentState  = initialState;
        entry.currentAccess = (initialAccess != GPUResourceAccess::COMMON || initialState == D3D12_RESOURCE_STATE_COMMON)
            ? initialAccess
            : InferAccessFromD3D12State(initialState);
        entry.pendingUavAccessSinceBarrier = GPUResourceAccess::COMMON;
        entry.subresourceCount             = effSubres;
        entry.mipLevels                    = (effMips > 1u || effSlices > 1u || effPlanes > 1u) ? effMips : effSubres;
        entry.arraySlices                  = effSlices;
        entry.planeCount                   = effPlanes;
        entry.isUniformState               = true;

        m_trackedResources[resource] = std::move(entry);
    }

    void D3D12ResourceStateTracker::UnregisterResource(ID3D12Resource* resource)
    {
        if (!resource) return;

        m_trackedResources.erase(resource);

        m_pendingBarriers.erase(
            std::remove_if(
                m_pendingBarriers.begin(),
                m_pendingBarriers.end(),
                [resource](const D3D12_RESOURCE_BARRIER& b) {
                    if (b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
                        return b.Transition.pResource == resource;
                    if (b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV)
                        return b.UAV.pResource == resource;
                    return false;
                }),
            m_pendingBarriers.end());
    }

    void D3D12ResourceStateTracker::ExpandPendingAllSubresourcesBarrier(ID3D12Resource* resource, UINT32 subresourceCount)
    {
        if (!resource || subresourceCount <= 1u) return;

        for (SIZE_T i = 0; i < m_pendingBarriers.size(); ++i)
        {
            if (m_pendingBarriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION &&
                m_pendingBarriers[i].Transition.pResource == resource &&
                m_pendingBarriers[i].Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
            {
                const D3D12_RESOURCE_STATES sBefore = m_pendingBarriers[i].Transition.StateBefore;
                const D3D12_RESOURCE_STATES sAfter  = m_pendingBarriers[i].Transition.StateAfter;
                const auto insertPos = static_cast<std::ptrdiff_t>(i);
                m_pendingBarriers.erase(m_pendingBarriers.begin() + insertPos);
                if (m_stats.totalTransitionBarriers > 0u) m_stats.totalTransitionBarriers--;

                std::vector<D3D12_RESOURCE_BARRIER> expanded;
                expanded.reserve(subresourceCount);
                for (UINT32 sr = 0u; sr < subresourceCount; ++sr)
                {
                    D3D12_RESOURCE_BARRIER b = CreateTransitionBarrier(resource, sBefore, sAfter);
                    b.Transition.Subresource = sr;
                    expanded.push_back(b);
                    m_stats.totalTransitionBarriers++;
                }
                m_pendingBarriers.insert(m_pendingBarriers.begin() + insertPos, expanded.begin(), expanded.end());
                return;
            }
        }
    }

    void D3D12ResourceStateTracker::TryCollapseUniformSubresourceState(D3D12TrackedResourceEntry& entry)
    {
        if (entry.isUniformState || entry.subresourceStates.empty())
            return;

        const D3D12_RESOURCE_STATES s0   = entry.subresourceStates[0];
        const GPUResourceAccess     acc0 = entry.subresourceAccess[0];
        const GPUResourceAccess     uav0 = entry.subresourcePendingUav[0];
        for (SIZE_T i = 1; i < entry.subresourceStates.size(); ++i)
        {
            if (entry.subresourceStates[i] != s0 ||
                entry.subresourceAccess[i] != acc0 ||
                entry.subresourcePendingUav[i] != uav0)
            {
                return;
            }
        }

        entry.isUniformState               = true;
        entry.currentState                 = s0;
        entry.currentAccess                = acc0;
        entry.pendingUavAccessSinceBarrier = uav0;
        entry.subresourceStates.clear();
        entry.subresourceAccess.clear();
        entry.subresourcePendingUav.clear();
    }

    BOOL D3D12ResourceStateTracker::TransitionResource(
        ID3D12Resource* resource,
        GPUResourceAccess targetAccess)
    {
        if (!resource) return false;
        const D3D12_RESOURCE_STATES targetState = ToD3D12ResourceState(targetAccess);
        return TransitionResourceState(resource, targetState, targetAccess);
    }

    BOOL D3D12ResourceStateTracker::TransitionResourceState(
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES targetState,
        GPUResourceAccess targetAccess)
    {
        if (!resource) return false;

        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
        {
            const D3D12_RESOURCE_DESC rDesc = resource->GetDesc();
            const UINT32 mips   = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) ? 1u : (std::max)(1u, static_cast<UINT32>(rDesc.MipLevels));
            const UINT32 slices = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) ? (std::max)(1u, static_cast<UINT32>(rDesc.DepthOrArraySize)) : 1u;
            const UINT32 planes = (rDesc.Format == DXGI_FORMAT_R24G8_TYPELESS || rDesc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT ||
                                   rDesc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || rDesc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) ? 2u : 1u;
            RegisterResource(resource, D3D12_RESOURCE_STATE_COMMON, GPUResourceAccess::COMMON, mips * slices * planes, mips, slices, planes);
            it = m_trackedResources.find(resource);
        }

        D3D12TrackedResourceEntry& entry = it->second;

        // If resource is currently split across distinct per-subresource states, unify all subresources to targetState
        if (!entry.isUniformState && entry.subresourceCount > 1u)
        {
            for (UINT32 sr = 0u; sr < entry.subresourceCount; ++sr)
            {
                TransitionSubresourceState(resource, sr, targetState, targetAccess);
            }
            entry.isUniformState = true;
            entry.currentState   = targetState;
            entry.currentAccess  = targetAccess;
            entry.pendingUavAccessSinceBarrier = (targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                ? (targetAccess & GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
                : GPUResourceAccess::COMMON;
            entry.subresourceStates.clear();
            entry.subresourceAccess.clear();
            entry.subresourcePendingUav.clear();
            return true;
        }

        // Case 1: D3D12 Resource State Change (Uniform ALL_SUBRESOURCES fast path)
        if (entry.currentState != targetState)
        {
            BOOL coalesced = false;
            // Only coalesce with the trailing barrier if no UAV or intervening barrier for this resource follows it
            for (auto bIt = m_pendingBarriers.rbegin(); bIt != m_pendingBarriers.rend(); ++bIt)
            {
                if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && bIt->UAV.pResource == resource)
                {
                    break;
                }
                if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION &&
                    bIt->Transition.pResource == resource)
                {
                    if (bIt->Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES &&
                        bIt->Transition.StateAfter == entry.currentState)
                    {
                        if (bIt->Transition.StateBefore == targetState)
                        {
                            m_pendingBarriers.erase(std::next(bIt).base());
                            if (m_stats.totalTransitionBarriers > 0) m_stats.totalTransitionBarriers--;
                            if (entry.transitionBarrierCount > 0) entry.transitionBarrierCount--;
                        }
                        else
                        {
                            bIt->Transition.StateAfter = targetState;
                        }
                        coalesced = true;
                    }
                    break;
                }
            }

            if (entry.currentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                for (auto bIt = m_pendingBarriers.begin(); bIt != m_pendingBarriers.end();)
                {
                    if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && bIt->UAV.pResource == resource)
                    {
                        bIt = m_pendingBarriers.erase(bIt);
                        if (m_stats.totalUavBarriers > 0) m_stats.totalUavBarriers--;
                        if (entry.uavBarrierCount > 0) entry.uavBarrierCount--;
                    }
                    else
                    {
                        ++bIt;
                    }
                }
            }

            if (!coalesced)
            {
                D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(
                    resource,
                    entry.currentState,
                    targetState
                );
                m_pendingBarriers.push_back(barrier);
                m_stats.totalTransitionBarriers++;
                entry.transitionBarrierCount++;
            }

            entry.currentState  = targetState;
            entry.currentAccess = targetAccess;
            if (targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                entry.pendingUavAccessSinceBarrier = targetAccess & GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
            }
            else
            {
                entry.pendingUavAccessSinceBarrier = GPUResourceAccess::COMMON;
            }
            return true;
        }

        // Case 2: Same State — Check UAV Read/Write Data Hazards (RAW, WAW, WAR)
        if (targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            const GPUResourceAccess priorUav = entry.pendingUavAccessSinceBarrier;
            const GPUResourceAccess nextUav  = targetAccess & GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;

            const BOOL rawHazard = HasGPUResourceAccess(priorUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE) &&
                                   HasGPUResourceAccess(nextUav, GPUResourceAccess::UNORDERED_ACCESS_READ);
            const BOOL wawHazard = HasGPUResourceAccess(priorUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE) &&
                                   HasGPUResourceAccess(nextUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            const BOOL warHazard = HasGPUResourceAccess(priorUav, GPUResourceAccess::UNORDERED_ACCESS_READ) &&
                                   HasGPUResourceAccess(nextUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE);

            if (rawHazard || wawHazard || warHazard)
            {
                BOOL alreadyHasUavBarrierInBatch = false;
                for (auto bIt = m_pendingBarriers.rbegin(); bIt != m_pendingBarriers.rend(); ++bIt)
                {
                    if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && bIt->UAV.pResource == resource)
                    {
                        alreadyHasUavBarrierInBatch = true;
                        break;
                    }
                    if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && bIt->Transition.pResource == resource &&
                        bIt->Transition.Subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
                    {
                        alreadyHasUavBarrierInBatch = true;
                        break;
                    }
                }

                if (!alreadyHasUavBarrierInBatch)
                {
                    D3D12_RESOURCE_BARRIER uavBarrier{};
                    uavBarrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    uavBarrier.Flags         = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                    uavBarrier.UAV.pResource = resource;
                    m_pendingBarriers.push_back(uavBarrier);
                    m_stats.totalUavBarriers++;
                    entry.uavBarrierCount++;
                }

                entry.pendingUavAccessSinceBarrier = nextUav;
            }
            else
            {
                entry.pendingUavAccessSinceBarrier = priorUav | nextUav;
            }

            entry.currentAccess = targetAccess;
            return true;
        }

        // Case 3: Same State, Non-UAV
        entry.currentAccess = targetAccess;
        return true;
    }

    BOOL D3D12ResourceStateTracker::TransitionSubresource(
        ID3D12Resource* resource,
        UINT32 subresource,
        GPUResourceAccess targetAccess)
    {
        if (!resource) return false;
        const D3D12_RESOURCE_STATES targetState = ToD3D12ResourceState(targetAccess);
        return TransitionSubresourceState(resource, subresource, targetState, targetAccess);
    }

    BOOL D3D12ResourceStateTracker::TransitionMipLevel(
        ID3D12Resource* resource,
        UINT32 mipSlice,
        GPUResourceAccess targetAccess)
    {
        if (!resource) return false;

        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
        {
            const D3D12_RESOURCE_DESC rDesc = resource->GetDesc();
            const UINT32 mips   = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) ? 1u : (std::max)(1u, static_cast<UINT32>(rDesc.MipLevels));
            const UINT32 slices = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) ? (std::max)(1u, static_cast<UINT32>(rDesc.DepthOrArraySize)) : 1u;
            const UINT32 planes = (rDesc.Format == DXGI_FORMAT_R24G8_TYPELESS || rDesc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT ||
                                   rDesc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || rDesc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) ? 2u : 1u;
            RegisterResource(resource, D3D12_RESOURCE_STATE_COMMON, GPUResourceAccess::COMMON, mips * slices * planes, mips, slices, planes);
            it = m_trackedResources.find(resource);
        }

        const D3D12TrackedResourceEntry& entry = it->second;
        if (mipSlice >= entry.mipLevels)
        {
            return false;
        }

        if (entry.subresourceCount <= 1u || (entry.mipLevels == 1u && entry.arraySlices == 1u && entry.planeCount == 1u))
        {
            return TransitionResource(resource, targetAccess);
        }

        const D3D12_RESOURCE_STATES targetState = ToD3D12ResourceState(targetAccess);
        BOOL allOk = true;
        for (UINT32 p = 0u; p < entry.planeCount; ++p)
        {
            for (UINT32 a = 0u; a < entry.arraySlices; ++a)
            {
                const UINT32 subres = mipSlice + a * entry.mipLevels + p * (entry.mipLevels * entry.arraySlices);
                if (!TransitionSubresourceState(resource, subres, targetState, targetAccess))
                {
                    allOk = false;
                }
            }
        }
        return allOk;
    }

    BOOL D3D12ResourceStateTracker::TransitionSubresourceState(
        ID3D12Resource* resource,
        UINT32 subresource,
        D3D12_RESOURCE_STATES targetState,
        GPUResourceAccess targetAccess)
    {
        if (!resource) return false;
        if (subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
        {
            return TransitionResourceState(resource, targetState, targetAccess);
        }

        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
        {
            const D3D12_RESOURCE_DESC rDesc = resource->GetDesc();
            const UINT32 mips   = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) ? 1u : (std::max)(1u, static_cast<UINT32>(rDesc.MipLevels));
            const UINT32 slices = (rDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) ? (std::max)(1u, static_cast<UINT32>(rDesc.DepthOrArraySize)) : 1u;
            const UINT32 planes = (rDesc.Format == DXGI_FORMAT_R24G8_TYPELESS || rDesc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT ||
                                   rDesc.Format == DXGI_FORMAT_R32G8X24_TYPELESS || rDesc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) ? 2u : 1u;
            const UINT32 totalSr = (std::max)(subresource + 1u, mips * slices * planes);
            RegisterResource(resource, D3D12_RESOURCE_STATE_COMMON, GPUResourceAccess::COMMON, totalSr, mips, slices, planes);
            it = m_trackedResources.find(resource);
        }

        D3D12TrackedResourceEntry& entry = it->second;
        if (entry.subresourceCount <= 1u && subresource == 0u)
        {
            return TransitionResourceState(resource, targetState, targetAccess);
        }

        if (subresource >= entry.subresourceCount)
        {
            return false;
        }

        if (entry.isUniformState)
        {
            ExpandPendingAllSubresourcesBarrier(resource, entry.subresourceCount);
            entry.subresourceStates.assign(entry.subresourceCount, entry.currentState);
            entry.subresourceAccess.assign(entry.subresourceCount, entry.currentAccess);
            entry.subresourcePendingUav.assign(entry.subresourceCount, entry.pendingUavAccessSinceBarrier);
            entry.isUniformState = false;
        }

        D3D12_RESOURCE_STATES& srState = entry.subresourceStates[subresource];
        GPUResourceAccess&     srAcc   = entry.subresourceAccess[subresource];
        GPUResourceAccess&     srUav   = entry.subresourcePendingUav[subresource];

        if (srState != targetState)
        {
            BOOL coalesced = false;
            for (auto bIt = m_pendingBarriers.rbegin(); bIt != m_pendingBarriers.rend(); ++bIt)
            {
                if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && bIt->UAV.pResource == resource)
                {
                    break;
                }
                if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION &&
                    bIt->Transition.pResource == resource &&
                    bIt->Transition.Subresource == subresource)
                {
                    if (bIt->Transition.StateAfter == srState)
                    {
                        if (bIt->Transition.StateBefore == targetState)
                        {
                            m_pendingBarriers.erase(std::next(bIt).base());
                            if (m_stats.totalTransitionBarriers > 0) m_stats.totalTransitionBarriers--;
                            if (entry.transitionBarrierCount > 0) entry.transitionBarrierCount--;
                        }
                        else
                        {
                            bIt->Transition.StateAfter = targetState;
                        }
                        coalesced = true;
                    }
                    break;
                }
            }

            if (!coalesced)
            {
                D3D12_RESOURCE_BARRIER barrier = CreateTransitionBarrier(resource, srState, targetState);
                barrier.Transition.Subresource = subresource;
                m_pendingBarriers.push_back(barrier);
                m_stats.totalTransitionBarriers++;
                entry.transitionBarrierCount++;
            }

            srState = targetState;
            srAcc   = targetAccess;
            srUav   = (targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                ? (targetAccess & GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
                : GPUResourceAccess::COMMON;
            entry.currentState  = entry.subresourceStates[0];
            entry.currentAccess = entry.subresourceAccess[0];
            TryCollapseUniformSubresourceState(entry);
            return true;
        }

        if (targetState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        {
            const GPUResourceAccess nextUav = targetAccess & GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
            const BOOL hazard =
                (HasGPUResourceAccess(srUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE) &&
                 HasGPUResourceAccess(nextUav, GPUResourceAccess::UNORDERED_ACCESS_READ | GPUResourceAccess::UNORDERED_ACCESS_WRITE)) ||
                (HasGPUResourceAccess(srUav, GPUResourceAccess::UNORDERED_ACCESS_READ) &&
                 HasGPUResourceAccess(nextUav, GPUResourceAccess::UNORDERED_ACCESS_WRITE));
            if (hazard)
            {
                InsertUAVBarrier(resource);
                srUav = nextUav;
            }
            else
            {
                srUav = srUav | nextUav;
            }
        }
        srAcc = targetAccess;
        entry.currentAccess = entry.subresourceAccess[0];
        TryCollapseUniformSubresourceState(entry);
        return true;
    }

    void D3D12ResourceStateTracker::InsertUAVBarrier(ID3D12Resource* resource)
    {
        if (!resource) return;

        // Only skip queuing if the trailing barrier for this resource is already a UAV barrier
        BOOL alreadyQueuedAfterLastTransition = false;
        for (auto bIt = m_pendingBarriers.rbegin(); bIt != m_pendingBarriers.rend(); ++bIt)
        {
            if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && bIt->UAV.pResource == resource)
            {
                alreadyQueuedAfterLastTransition = true;
                break;
            }
            if (bIt->Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && bIt->Transition.pResource == resource)
            {
                break;
            }
        }

        if (!alreadyQueuedAfterLastTransition)
        {
            D3D12_RESOURCE_BARRIER uavBarrier{};
            uavBarrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uavBarrier.Flags         = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            uavBarrier.UAV.pResource = resource;
            m_pendingBarriers.push_back(uavBarrier);
            m_stats.totalUavBarriers++;
        }

        auto it = m_trackedResources.find(resource);
        if (it != m_trackedResources.end())
        {
            it->second.pendingUavAccessSinceBarrier = GPUResourceAccess::COMMON;
            for (auto& u : it->second.subresourcePendingUav)
            {
                u = GPUResourceAccess::COMMON;
            }
            it->second.uavBarrierCount++;
        }
    }

    UINT32 D3D12ResourceStateTracker::FlushBarriers(ID3D12GraphicsCommandList* cmdList)
    {
        const UINT32 count = static_cast<UINT32>(m_pendingBarriers.size());
        if (count == 0 || !cmdList)
        {
            // Never discard pending GPU barriers without recording them onto an active command list;
            // discarding them would desynchronize CPU tracked state from actual GPU resource state.
            return 0;
        }

        // Preserve exact recorded barrier order across mixed TRANSITION and UAV barriers.
        cmdList->ResourceBarrier(count, m_pendingBarriers.data());

        m_pendingBarriers.clear();
        m_stats.flushedBatches++;
        return count;
    }

    D3D12_RESOURCE_STATES D3D12ResourceStateTracker::GetResourceState(ID3D12Resource* resource) const
    {
        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
            return D3D12_RESOURCE_STATE_COMMON;
        if (!it->second.isUniformState && !it->second.subresourceStates.empty())
            return it->second.subresourceStates[0];
        return it->second.currentState;
    }

    D3D12_RESOURCE_STATES D3D12ResourceStateTracker::GetSubresourceState(ID3D12Resource* resource, UINT32 subresource) const
    {
        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
            return D3D12_RESOURCE_STATE_COMMON;
        if (!it->second.isUniformState && subresource < it->second.subresourceStates.size())
            return it->second.subresourceStates[subresource];
        return it->second.currentState;
    }

    BOOL D3D12ResourceStateTracker::IsResourceUniformState(ID3D12Resource* resource) const
    {
        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
            return true;
        return it->second.isUniformState;
    }

    GPUResourceAccess D3D12ResourceStateTracker::GetResourceAccess(ID3D12Resource* resource) const
    {
        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
            return GPUResourceAccess::COMMON;
        if (!it->second.isUniformState && !it->second.subresourceAccess.empty())
            return it->second.subresourceAccess[0];
        return it->second.currentAccess;
    }

    GPUResourceAccess D3D12ResourceStateTracker::GetSubresourceAccess(ID3D12Resource* resource, UINT32 subresource) const
    {
        auto it = m_trackedResources.find(resource);
        if (it == m_trackedResources.end())
            return GPUResourceAccess::COMMON;
        if (!it->second.isUniformState && subresource < it->second.subresourceAccess.size())
            return it->second.subresourceAccess[subresource];
        return it->second.currentAccess;
    }

    BOOL D3D12ResourceStateTracker::IsResourceTracked(ID3D12Resource* resource) const
    {
        return m_trackedResources.find(resource) != m_trackedResources.end();
    }

    ResourceBarrierStats D3D12ResourceStateTracker::GetStats() const
    {
        ResourceBarrierStats s = m_stats;
        s.pendingBarriers = static_cast<UINT32>(m_pendingBarriers.size());
        return s;
    }

    void D3D12ResourceStateTracker::ResetStats()
    {
        m_stats = {};
    }

    void D3D12ResourceStateTracker::Reset()
    {
        m_trackedResources.clear();
        m_pendingBarriers.clear();
        m_stats = {};
    }
}
