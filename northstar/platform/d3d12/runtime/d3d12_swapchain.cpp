#include "d3d12_swapchain.h"
#include <iostream>

namespace Northstar
{
    BOOL D3D12SwapChain::Initialize(
        IDXGIFactory4*       factory,
        ID3D12CommandQueue*  commandQueue,
        ID3D12Device*        device,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        HWND                 hwnd,
        UINT32               width,
        UINT32               height)
    {
        if (!factory || !commandQueue || !device || !hwnd) return false;

        m_targetHwnd = hwnd;
        m_width      = width;
        m_height     = height;

        RECT clientRect{};
        if (GetClientRect(hwnd, &clientRect))
        {
            if (clientRect.right > clientRect.left) m_width = clientRect.right - clientRect.left;
            if (clientRect.bottom > clientRect.top) m_height = clientRect.bottom - clientRect.top;
        }

        // Check for Tearing Support (Allows uncapped FPS and variable refresh rate in windowed mode)
        WIN_BOOL allowTearing = FALSE;
        ComPtr<IDXGIFactory5> factory5;
        if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory5))))
        {
            if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing))))
            {
                allowTearing = FALSE;
            }
        }
        m_tearingSupported = (allowTearing == TRUE);

        DXGI_SWAP_CHAIN_DESC1 scDesc{};
        scDesc.Width              = m_width;
        scDesc.Height             = m_height;
        scDesc.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
        scDesc.Stereo             = FALSE;
        scDesc.SampleDesc.Count   = 1;
        scDesc.SampleDesc.Quality = 0;
        scDesc.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scDesc.BufferCount        = kBackBufferCount;
        scDesc.Scaling            = DXGI_SCALING_STRETCH;
        scDesc.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        scDesc.AlphaMode          = DXGI_ALPHA_MODE_UNSPECIFIED;
        scDesc.Flags              = m_tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

        ComPtr<IDXGISwapChain1> swapChain1;
        if (FAILED(factory->CreateSwapChainForHwnd(
            commandQueue,
            hwnd,
            &scDesc,
            nullptr,
            nullptr,
            &swapChain1)))
        {
            return false;
        }

        if (m_tearingSupported)
        {
            factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        }

        if (FAILED(swapChain1.As(&m_swapChain)))
        {
            return false;
        }

        // Create RTVs for each back buffer
        for (UINT32 i = 0; i < kBackBufferCount; ++i)
        {
            if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]))) || !m_backBuffers[i])
            {
                Shutdown(rtvPool, dsvPool);
                return false;
            }
            m_backBufferRtvIndices[i] = rtvPool.Allocate();
            if (m_backBufferRtvIndices[i] == UINT32_MAX)
            {
                Shutdown(rtvPool, dsvPool);
                return false;
            }
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvPool.GetCPUHandle(m_backBufferRtvIndices[i]);
            device->CreateRenderTargetView(m_backBuffers[i].Get(), nullptr, rtvHandle);
        }
        m_currentBackBufferIndex = m_swapChain->GetCurrentBackBufferIndex();

        // Create Depth Stencil buffer (DXGI_FORMAT_D32_FLOAT)
        D3D12_RESOURCE_DESC depthDesc{};
        depthDesc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depthDesc.Alignment          = 0;
        depthDesc.Width              = m_width;
        depthDesc.Height             = m_height;
        depthDesc.DepthOrArraySize   = 1;
        depthDesc.MipLevels          = 1;
        depthDesc.Format             = DXGI_FORMAT_D32_FLOAT;
        depthDesc.SampleDesc.Count   = 1;
        depthDesc.SampleDesc.Quality = 0;
        depthDesc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        depthDesc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        D3D12_CLEAR_VALUE depthClearValue{};
        depthClearValue.Format               = DXGI_FORMAT_D32_FLOAT;
        depthClearValue.DepthStencil.Depth   = 1.0f;
        depthClearValue.DepthStencil.Stencil = 0;

        D3D12_HEAP_PROPERTIES defaultHeap{};
        defaultHeap.Type                 = D3D12_HEAP_TYPE_DEFAULT;
        defaultHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        defaultHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        HRESULT hrDepth = device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &depthDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &depthClearValue,
            IID_PPV_ARGS(&m_depthStencilBuffer));

        if (SUCCEEDED(hrDepth) && m_depthStencilBuffer)
        {
            m_dsvIndex = dsvPool.Allocate();
            if (m_dsvIndex != UINT32_MAX)
            {
                D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
                dsvDesc.Format        = DXGI_FORMAT_D32_FLOAT;
                dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                dsvDesc.Flags         = D3D12_DSV_FLAG_NONE;

                D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvPool.GetCPUHandle(m_dsvIndex);
                device->CreateDepthStencilView(m_depthStencilBuffer.Get(), &dsvDesc, dsvHandle);
            }
        }

        return true;
    }

    void D3D12SwapChain::Shutdown(D3D12DescriptorPool& rtvPool, D3D12DescriptorPool& dsvPool)
    {
        if (m_dsvIndex != UINT32_MAX)
        {
            dsvPool.Free(m_dsvIndex);
            m_dsvIndex = UINT32_MAX;
        }
        m_depthStencilBuffer.Reset();

        for (UINT32 i = 0; i < kBackBufferCount; ++i)
        {
            if (m_backBufferRtvIndices[i] != UINT32_MAX)
            {
                rtvPool.Free(m_backBufferRtvIndices[i]);
                m_backBufferRtvIndices[i] = UINT32_MAX;
            }
            m_backBuffers[i].Reset();
        }
        m_swapChain.Reset();
        m_targetHwnd = nullptr;
    }

    BOOL D3D12SwapChain::Resize(
        ID3D12Device*        device,
        D3D12DescriptorPool& rtvPool,
        D3D12DescriptorPool& dsvPool,
        UINT32               width,
        UINT32               height)
    {
        if (!m_swapChain || !device) return false;
        if (width == 0 || height == 0) return false;
        if (m_targetHwnd && !::IsWindow(m_targetHwnd)) return false;
        if (m_width == width && m_height == height) return true;

        m_width  = width;
        m_height = height;

        // 1. Release existing depth-stencil buffer and free DSV descriptor
        if (m_dsvIndex != UINT32_MAX)
        {
            dsvPool.Free(m_dsvIndex);
            m_dsvIndex = UINT32_MAX;
        }
        m_depthStencilBuffer.Reset();

        // 2. Release backbuffer references and free RTV descriptors
        for (UINT32 i = 0; i < kBackBufferCount; ++i)
        {
            if (m_backBufferRtvIndices[i] != UINT32_MAX)
            {
                rtvPool.Free(m_backBufferRtvIndices[i]);
                m_backBufferRtvIndices[i] = UINT32_MAX;
            }
            m_backBuffers[i].Reset();
        }

        // 3. Resize SwapChain backbuffers
        const UINT swapChainFlags = m_tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
        HRESULT hr = m_swapChain->ResizeBuffers(
            kBackBufferCount,
            m_width,
            m_height,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            swapChainFlags);
        if (FAILED(hr))
        {
            std::cerr << "[NorthstarD3D12] ResizeBuffers failed: 0x" << std::hex << hr << std::dec << "\n";
            return false;
        }

        // 4. Re-create RTV descriptors for each back buffer
        for (UINT32 i = 0; i < kBackBufferCount; ++i)
        {
            if (FAILED(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]))) || !m_backBuffers[i])
            {
                return false;
            }
            m_backBufferRtvIndices[i] = rtvPool.Allocate();
            if (m_backBufferRtvIndices[i] == UINT32_MAX)
            {
                return false;
            }
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvPool.GetCPUHandle(m_backBufferRtvIndices[i]);
            device->CreateRenderTargetView(m_backBuffers[i].Get(), nullptr, rtvHandle);
        }
        m_currentBackBufferIndex = m_swapChain->GetCurrentBackBufferIndex();

        // 5. Re-create Depth Stencil buffer with new dimensions
        D3D12_RESOURCE_DESC depthDesc{};
        depthDesc.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depthDesc.Alignment          = 0;
        depthDesc.Width              = m_width;
        depthDesc.Height             = m_height;
        depthDesc.DepthOrArraySize   = 1;
        depthDesc.MipLevels          = 1;
        depthDesc.Format             = DXGI_FORMAT_D32_FLOAT;
        depthDesc.SampleDesc.Count   = 1;
        depthDesc.SampleDesc.Quality = 0;
        depthDesc.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        depthDesc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

        D3D12_CLEAR_VALUE depthClearValue{};
        depthClearValue.Format               = DXGI_FORMAT_D32_FLOAT;
        depthClearValue.DepthStencil.Depth   = 1.0f;
        depthClearValue.DepthStencil.Stencil = 0;

        D3D12_HEAP_PROPERTIES defaultHeap{};
        defaultHeap.Type                 = D3D12_HEAP_TYPE_DEFAULT;
        defaultHeap.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        defaultHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        HRESULT hrDepth = device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &depthDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &depthClearValue,
            IID_PPV_ARGS(&m_depthStencilBuffer));

        if (SUCCEEDED(hrDepth) && m_depthStencilBuffer)
        {
            m_dsvIndex = dsvPool.Allocate();
            if (m_dsvIndex != UINT32_MAX)
            {
                D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
                dsvDesc.Format        = DXGI_FORMAT_D32_FLOAT;
                dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                dsvDesc.Flags         = D3D12_DSV_FLAG_NONE;

                D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvPool.GetCPUHandle(m_dsvIndex);
                device->CreateDepthStencilView(m_depthStencilBuffer.Get(), &dsvDesc, dsvHandle);
            }
        }

        return true;
    }

    HRESULT D3D12SwapChain::Present(ID3D12Device* device)
    {
        if (!m_swapChain) return S_OK;
        if (m_targetHwnd && !::IsWindow(m_targetHwnd)) return S_OK;

        const UINT syncInterval = m_vsyncEnabled ? 1 : 0;
        const UINT presentFlags = (syncInterval == 0 && m_tearingSupported) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        HRESULT hr = m_swapChain->Present(syncInterval, presentFlags);
        if (FAILED(hr))
        {
            static BOOL loggedPresentErr = false;
            if (!loggedPresentErr)
            {
                loggedPresentErr = true;
                std::cerr << "[NorthstarD3D12] Present failed: 0x" << std::hex << hr << std::dec << "\n";
                if (hr == DXGI_ERROR_DEVICE_REMOVED && device)
                {
                    HRESULT reason = device->GetDeviceRemovedReason();
                    std::cerr << "[NorthstarD3D12] Device Removed Reason: 0x" << std::hex << reason << std::dec << "\n";
                }
            }
        }
        m_currentBackBufferIndex = m_swapChain->GetCurrentBackBufferIndex();
        return hr;
    }

    ID3D12Resource* D3D12SwapChain::GetBackBuffer(UINT32 index) const
    {
        if (index < kBackBufferCount) return m_backBuffers[index].Get();
        return nullptr;
    }

    ID3D12Resource* D3D12SwapChain::GetCurrentBackBuffer() const
    {
        return GetBackBuffer(m_currentBackBufferIndex);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE D3D12SwapChain::GetCurrentRtvHandle(const D3D12DescriptorPool& rtvPool) const
    {
        UINT32 idx = m_backBufferRtvIndices[m_currentBackBufferIndex];
        return rtvPool.GetCPUHandle(idx);
    }

    D3D12_VIEWPORT D3D12SwapChain::GetViewport() const
    {
        D3D12_VIEWPORT vp{};
        vp.TopLeftX = 0.0f;
        vp.TopLeftY = 0.0f;
        vp.Width    = static_cast<float>(m_width);
        vp.Height   = static_cast<float>(m_height);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        return vp;
    }

    D3D12_RECT D3D12SwapChain::GetScissorRect() const
    {
        D3D12_RECT rc{};
        rc.left   = 0;
        rc.top    = 0;
        rc.right  = static_cast<LONG>(m_width);
        rc.bottom = static_cast<LONG>(m_height);
        return rc;
    }
}
