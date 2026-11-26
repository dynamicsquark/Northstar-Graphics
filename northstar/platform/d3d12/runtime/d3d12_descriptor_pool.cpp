#include "d3d12_descriptor_pool.h"
#include <algorithm>

namespace Northstar
{
    BOOL D3D12DescriptorPool::Initialize(
        ID3D12Device* device,
        D3D12_DESCRIPTOR_HEAP_TYPE type,
        UINT32 count,
        BOOL shaderVisible)
    {
        if (!device || count == 0) return false;

        Reset();

        m_device        = device;
        m_type          = type;
        m_shaderVisible = shaderVisible;
        m_pageSize      = count;
        m_capacity      = count;
        m_allocated     = 0;
        m_freeList.clear();
        m_isInFreeList.assign(count, 0);

        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type           = type;
        desc.NumDescriptors = count;
        desc.Flags          = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask       = 0;

        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap))))
            return false;

        if (shaderVisible)
        {
            D3D12_DESCRIPTOR_HEAP_DESC stagingDesc = desc;
            stagingDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            if (FAILED(device->CreateDescriptorHeap(&stagingDesc, IID_PPV_ARGS(&m_cpuStagingHeap))))
                return false;
        }

        m_descriptorSize = device->GetDescriptorHandleIncrementSize(type);
        return true;
    }

    void D3D12DescriptorPool::Reset()
    {
        m_extraCpuHeaps.clear();
        m_cpuStagingHeap.Reset();
        m_heap.Reset();
        m_device         = nullptr;
        m_shaderVisible  = false;
        m_pageSize       = 0;
        m_descriptorSize = 0;
        m_capacity       = 0;
        m_allocated      = 0;
        m_freeList.clear();
        m_isInFreeList.clear();
        m_freeRanges.clear();
    }

    BOOL D3D12DescriptorPool::GrowCpuOnlyPool()
    {
        // Only non-shader-visible heaps (RTV / DSV) can span multiple D3D12 descriptor heaps transparently.
        if (m_shaderVisible || !m_device || m_pageSize == 0)
            return false;

        if (m_capacity > (0x7FFFFFFFu - m_pageSize))
            return false;

        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type           = m_type;
        desc.NumDescriptors = m_pageSize;
        desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask       = 0;

        ComPtr<ID3D12DescriptorHeap> newPage;
        if (FAILED(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&newPage))) || !newPage)
            return false;

        m_extraCpuHeaps.push_back(std::move(newPage));
        m_capacity += m_pageSize;
        m_isInFreeList.resize(m_capacity, 0);
        return true;
    }

    UINT32 D3D12DescriptorPool::Allocate()
    {
        if (!m_freeList.empty())
        {
            UINT32 idx = m_freeList.back();
            m_freeList.pop_back();
            if (idx < m_isInFreeList.size())
            {
                m_isInFreeList[idx] = 0;
            }
            return idx;
        }
        if (m_allocated < m_capacity)
        {
            return m_allocated++;
        }
        if (!m_shaderVisible && GrowCpuOnlyPool() && m_allocated < m_capacity)
        {
            return m_allocated++;
        }
        return UINT32_MAX;
    }

    UINT32 D3D12DescriptorPool::AllocateRange(UINT32 count)
    {
        if (count == 0) return UINT32_MAX;
        if (m_shaderVisible && count > m_capacity) return UINT32_MAX;
        if (!m_shaderVisible && count > m_pageSize) return UINT32_MAX;

        // First reuse an exact-size recycled range, then any larger range (splitting remainder)
        for (SIZE_T i = 0; i < m_freeRanges.size(); ++i)
        {
            if (m_freeRanges[i].second == count)
            {
                const UINT32 start = m_freeRanges[i].first;
                m_freeRanges[i] = m_freeRanges.back();
                m_freeRanges.pop_back();
                return start;
            }
        }
        for (SIZE_T i = 0; i < m_freeRanges.size(); ++i)
        {
            if (m_freeRanges[i].second > count)
            {
                const UINT32 start = m_freeRanges[i].first;
                m_freeRanges[i].first  += count;
                m_freeRanges[i].second -= count;
                return start;
            }
        }

        if (!m_shaderVisible && m_pageSize > 0)
        {
            // Keep contiguous range inside a single CPU heap page
            const UINT32 pageIdx    = m_allocated / m_pageSize;
            const UINT32 pageOffset = m_allocated % m_pageSize;
            if (pageOffset + count > m_pageSize)
            {
                const UINT32 nextPageStart = (pageIdx + 1u) * m_pageSize;
                if (nextPageStart > m_allocated)
                {
                    FreeRange(m_allocated, nextPageStart - m_allocated);
                    m_allocated = nextPageStart;
                }
            }
            while (m_allocated + count > m_capacity)
            {
                if (!GrowCpuOnlyPool())
                    return UINT32_MAX;
            }
        }

        if (m_allocated <= (m_capacity - count))
        {
            const UINT32 start = m_allocated;
            m_allocated += count;
            return start;
        }
        return UINT32_MAX;
    }

    void D3D12DescriptorPool::FreeRange(UINT32 startIdx, UINT32 count)
    {
        if (count == 0 || startIdx == UINT32_MAX) return;
        if (startIdx >= m_capacity || count > (m_capacity - startIdx)) return;

        // Prevent double-free of identical range and coalesce adjacent free ranges on the same page
        for (const auto& r : m_freeRanges)
        {
            if (r.first == startIdx && r.second == count)
                return;
        }
        m_freeRanges.emplace_back(startIdx, count);
        if (m_freeRanges.size() > 1)
        {
            std::sort(m_freeRanges.begin(), m_freeRanges.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            SIZE_T out = 0;
            for (SIZE_T i = 1; i < m_freeRanges.size(); ++i)
            {
                const UINT32 prevEnd = m_freeRanges[out].first + m_freeRanges[out].second;
                const BOOL samePage  = m_shaderVisible || m_pageSize == 0 ||
                                      ((m_freeRanges[out].first / m_pageSize) == (m_freeRanges[i].first / m_pageSize));
                if (samePage && prevEnd == m_freeRanges[i].first)
                {
                    m_freeRanges[out].second += m_freeRanges[i].second;
                }
                else
                {
                    m_freeRanges[++out] = m_freeRanges[i];
                }
            }
            m_freeRanges.resize(out + 1);
        }
    }

    void D3D12DescriptorPool::Free(UINT32 idx)
    {
        if (idx != UINT32_MAX && idx < m_allocated && idx < m_capacity)
        {
            if (idx < m_isInFreeList.size() && m_isInFreeList[idx] == 0)
            {
                m_isInFreeList[idx] = 1;
                m_freeList.push_back(idx);
            }
        }
    }

    D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorPool::GetCPUHandle(UINT32 idx) const
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h{};
        if (!m_heap || idx >= m_capacity) return h;

        if (!m_shaderVisible && m_pageSize > 0 && idx >= m_pageSize)
        {
            const UINT32 extraPage = (idx / m_pageSize) - 1u;
            const UINT32 offset    = idx % m_pageSize;
            if (extraPage >= m_extraCpuHeaps.size() || !m_extraCpuHeaps[extraPage])
                return h;
            h = m_extraCpuHeaps[extraPage]->GetCPUDescriptorHandleForHeapStart();
            h.ptr += static_cast<SIZE_T>(offset) * m_descriptorSize;
            return h;
        }

        h = m_heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(idx) * m_descriptorSize;
        return h;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorPool::GetCpuStagingHandle(UINT32 idx) const
    {
        if (m_cpuStagingHeap)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE h{};
            if (idx >= m_capacity) return h;
            h = m_cpuStagingHeap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += static_cast<SIZE_T>(idx) * m_descriptorSize;
            return h;
        }
        return GetCPUHandle(idx);
    }

    D3D12_GPU_DESCRIPTOR_HANDLE D3D12DescriptorPool::GetGPUHandle(UINT32 idx) const
    {
        D3D12_GPU_DESCRIPTOR_HANDLE h{};
        if (!m_heap || !m_shaderVisible || idx >= m_capacity) return h;
        h = m_heap->GetGPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<UINT64>(idx) * m_descriptorSize;
        return h;
    }
}
