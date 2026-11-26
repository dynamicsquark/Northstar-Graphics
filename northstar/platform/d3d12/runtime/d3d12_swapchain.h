#pragma once

#include "d3d12_common.h"
#include "d3d12_descriptor_pool.h"

namespace Northstar
{
    class D3D12SwapChain
    {
    public:
        static constexpr UINT32 kBackBufferCount = 2;

        D3D12SwapChain() = default;
        ~D3D12SwapChain() = default;

        BOOL Initialize(
            IDXGIFactory4*       factory,
            ID3D12CommandQueue*  commandQueue,
            ID3D12Device*        device,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            HWND                 hwnd,
            UINT32               width,
            UINT32               height);

        void Shutdown(D3D12DescriptorPool& rtvPool, D3D12DescriptorPool& dsvPool);
        BOOL Resize(
            ID3D12Device*        device,
            D3D12DescriptorPool& rtvPool,
            D3D12DescriptorPool& dsvPool,
            UINT32               width,
            UINT32               height);
        HRESULT Present(ID3D12Device* device);

        void                        SetVSync(BOOL enabled) { m_vsyncEnabled = enabled; }
        BOOL                        IsVSyncEnabled() const { return m_vsyncEnabled; }

        ID3D12Resource*             GetBackBuffer(UINT32 index) const;
        ID3D12Resource*             GetCurrentBackBuffer() const;
        D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentRtvHandle(const D3D12DescriptorPool& rtvPool) const;
        UINT32                      GetCurrentBackBufferIndex() const { return m_currentBackBufferIndex; }

        BOOL                        HasDepthBuffer() const { return m_depthStencilBuffer != nullptr && m_dsvIndex != UINT32_MAX; }
        D3D12_CPU_DESCRIPTOR_HANDLE GetDsvHandle(const D3D12DescriptorPool& dsvPool) const
        {
            return dsvPool.GetCPUHandle(m_dsvIndex);
        }

        D3D12_VIEWPORT              GetViewport() const;
        D3D12_RECT                  GetScissorRect() const;

        UINT32                      GetWidth()  const { return m_width; }
        UINT32                      GetHeight() const { return m_height; }
        HWND                        GetHwnd()   const { return m_targetHwnd; }
        BOOL                        IsValid()   const
        {
            return (m_swapChain != nullptr) &&
                   (m_backBuffers[0] != nullptr) &&
                   (!m_targetHwnd || ::IsWindow(m_targetHwnd));
        }
        BOOL                        IsTearingSupported() const { return m_tearingSupported; }

    private:
        ComPtr<IDXGISwapChain3> m_swapChain;
        ComPtr<ID3D12Resource>  m_backBuffers[kBackBufferCount];
        UINT32                  m_backBufferRtvIndices[kBackBufferCount] = { UINT32_MAX, UINT32_MAX };
        UINT32                  m_currentBackBufferIndex = 0;
        ComPtr<ID3D12Resource>  m_depthStencilBuffer;
        UINT32                  m_dsvIndex = UINT32_MAX;
        HWND                    m_targetHwnd = nullptr;
        UINT32                  m_width  = 1280;
        UINT32                  m_height = 720;
        BOOL                    m_vsyncEnabled = true;
        BOOL                    m_tearingSupported = false;
    };
}
