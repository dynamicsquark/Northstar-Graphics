#pragma once

#include "d3d12_common.h"
#include <vector>

namespace Northstar
{
    class D3D12DescriptorPool
    {
    public:
        D3D12DescriptorPool() = default;
        ~D3D12DescriptorPool() { Reset(); }

        BOOL Initialize(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT32 count, BOOL shaderVisible);
        void Reset();

        UINT32 Allocate();
        UINT32 AllocateRange(UINT32 count);
        void   Free(UINT32 idx);
        void   FreeRange(UINT32 startIdx, UINT32 count);

        D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle(UINT32 idx) const;
        D3D12_CPU_DESCRIPTOR_HANDLE GetCpuStagingHandle(UINT32 idx) const;
        D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(UINT32 idx) const;

        ID3D12DescriptorHeap* GetHeap() const { return m_heap.Get(); }
        UINT32                GetDescriptorSize() const { return m_descriptorSize; }
        UINT32                GetCapacity() const { return m_capacity; }
        UINT32                GetAllocatedCount() const { return m_allocated; }
        UINT32                GetFreeListCount() const { return static_cast<UINT32>(m_freeList.size()); }
        UINT32                GetActiveAllocatedCount() const
        {
            const UINT32 freeCount = static_cast<UINT32>(m_freeList.size());
            return (m_allocated >= freeCount) ? (m_allocated - freeCount) : 0;
        }

    private:
        BOOL GrowCpuOnlyPool();

        ID3D12Device*                              m_device        = nullptr;
        D3D12_DESCRIPTOR_HEAP_TYPE                 m_type          = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        BOOL                                       m_shaderVisible = false;
        UINT32                                     m_pageSize      = 0;
        ComPtr<ID3D12DescriptorHeap>               m_heap;
        ComPtr<ID3D12DescriptorHeap>               m_cpuStagingHeap;
        std::vector<ComPtr<ID3D12DescriptorHeap>>  m_extraCpuHeaps;
        UINT32                                     m_descriptorSize = 0;
        UINT32                                     m_capacity       = 0;
        UINT32                                     m_allocated      = 0;
        std::vector<UINT32>                        m_freeList;
        std::vector<UINT8>                         m_isInFreeList;
        std::vector<std::pair<UINT32, UINT32>>     m_freeRanges;
    };
}
