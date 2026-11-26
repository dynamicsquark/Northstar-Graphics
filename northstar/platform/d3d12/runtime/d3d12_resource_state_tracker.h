#pragma once

#include "d3d12_common.h"
#include "buffer/gpu_buffer_types.h"
#include <unordered_map>
#include <vector>

namespace Northstar
{
    struct D3D12TrackedResourceEntry
    {
        D3D12_RESOURCE_STATES              currentState                 = D3D12_RESOURCE_STATE_COMMON;
        GPUResourceAccess                  currentAccess                = GPUResourceAccess::COMMON;
        GPUResourceAccess                  pendingUavAccessSinceBarrier = GPUResourceAccess::COMMON;
        UINT32                             subresourceCount             = 1;
        UINT32                             mipLevels                    = 1;
        UINT32                             arraySlices                  = 1;
        UINT32                             planeCount                   = 1;
        BOOL                               isUniformState               = true;
        std::vector<D3D12_RESOURCE_STATES> subresourceStates;
        std::vector<GPUResourceAccess>     subresourceAccess;
        std::vector<GPUResourceAccess>     subresourcePendingUav;
        UINT32                             transitionBarrierCount       = 0;
        UINT32                             uavBarrierCount              = 0;
    };

    class D3D12ResourceStateTracker
    {
    public:
        D3D12ResourceStateTracker() = default;
        ~D3D12ResourceStateTracker() = default;

        static D3D12_RESOURCE_STATES ToD3D12ResourceState(GPUResourceAccess access);
        static GPUResourceAccess     InferAccessFromD3D12State(D3D12_RESOURCE_STATES state);

        void RegisterResource(
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES initialState,
            GPUResourceAccess initialAccess = GPUResourceAccess::COMMON,
            UINT32 subresourceCount = 1,
            UINT32 mipLevels = 1,
            UINT32 arraySlices = 1,
            UINT32 planeCount = 1);

        void UnregisterResource(ID3D12Resource* resource);

        BOOL TransitionResource(
            ID3D12Resource* resource,
            GPUResourceAccess targetAccess);

        BOOL TransitionResourceState(
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES targetState,
            GPUResourceAccess targetAccess);

        BOOL TransitionSubresource(
            ID3D12Resource* resource,
            UINT32 subresource,
            GPUResourceAccess targetAccess);

        BOOL TransitionSubresourceState(
            ID3D12Resource* resource,
            UINT32 subresource,
            D3D12_RESOURCE_STATES targetState,
            GPUResourceAccess targetAccess);

        BOOL TransitionMipLevel(
            ID3D12Resource* resource,
            UINT32 mipSlice,
            GPUResourceAccess targetAccess);

        void InsertUAVBarrier(ID3D12Resource* resource);

        UINT32 FlushBarriers(ID3D12GraphicsCommandList* cmdList);

        D3D12_RESOURCE_STATES GetResourceState(ID3D12Resource* resource) const;
        D3D12_RESOURCE_STATES GetSubresourceState(ID3D12Resource* resource, UINT32 subresource) const;
        BOOL                  IsResourceUniformState(ID3D12Resource* resource) const;
        GPUResourceAccess     GetResourceAccess(ID3D12Resource* resource) const;
        GPUResourceAccess     GetSubresourceAccess(ID3D12Resource* resource, UINT32 subresource) const;
        BOOL                  IsResourceTracked(ID3D12Resource* resource) const;

        UINT32                GetPendingBarrierCount() const { return static_cast<UINT32>(m_pendingBarriers.size()); }
        ResourceBarrierStats  GetStats() const;
        void                  ResetStats();
        void                  Reset();

    private:
        void ExpandPendingAllSubresourcesBarrier(ID3D12Resource* resource, UINT32 subresourceCount);
        void TryCollapseUniformSubresourceState(D3D12TrackedResourceEntry& entry);

        std::unordered_map<ID3D12Resource*, D3D12TrackedResourceEntry> m_trackedResources;
        std::vector<D3D12_RESOURCE_BARRIER>                            m_pendingBarriers;
        ResourceBarrierStats                                           m_stats{};
    };
}
