#pragma once

#include "d3d12_common.h"
#include "platform/northstar_backend.h"
#include <string>
#include <functional>

namespace Northstar
{
    class D3D12DeviceContext
    {
    public:
        D3D12DeviceContext() = default;
        ~D3D12DeviceContext() { Shutdown(); }

        void SetValidationConfig(BOOL enableDebugLayer, BOOL enableGPUBasedValidation)
        {
            m_requestDebugLayer = enableDebugLayer;
            m_requestGBV        = enableGPUBasedValidation;
        }

        BOOL Initialize();
        void Shutdown();

        void ExecuteUpload(std::function<void(ID3D12GraphicsCommandList*)> recordFn);
        void WaitForGPU();
        BOOL WaitForFenceValue(ID3D12Fence* fence, HANDLE evt, UINT64 targetValue);
        BOOL IsDeviceRemoved() const { return m_deviceRemoved; }
        HRESULT GetDeviceRemovedReason() const { return m_deviceRemovedReason; }

        BOOL   IsDebugLayerActive()         const { return m_debugLayerActive; }
        BOOL   IsGPUBasedValidationActive() const { return m_gbvActive; }
        UINT32 GetDebugErrorCount()         const;
        void   ClearDebugMessages();

        IDXGIFactory4*      GetFactory()        const { return m_factory.Get(); }
        IDXGIAdapter1*      GetAdapter()        const { return m_adapter.Get(); }
        ID3D12Device*       GetDevice()         const { return m_device.Get(); }
        ID3D12CommandQueue* GetCommandQueue()   const { return m_commandQueue.Get(); }

        ID3D12Fence*        GetFrameFence()             const { return m_frameFence.Get(); }
        HANDLE              GetFrameFenceSharedHandle() const { return m_frameFenceSharedHandle; }
        HANDLE              GetFrameFenceEvent()        const { return m_frameFenceEvent; }
        UINT64              GetFrameFenceValue()        const { return m_frameFenceValue; }
        void                IncrementFrameFenceValue() { m_frameFenceValue++; }

        BOOL                        IsInitialized()     const { return m_device != nullptr; }
        BOOL                        SupportsVPAndRTArrayIndexWithoutGS() const { return m_supportsVPAndRTArrayIndexWithoutGS; }
        const BackendCapabilities&  GetCapabilities()   const { return m_hasRestrictedCaps ? m_restrictedCaps : m_capabilities; }
        void                        SetDXILRuntimeAvailable(BOOL available) noexcept
        {
            m_capabilities.supportsDXILRuntime = available;
            if (m_hasRestrictedCaps)
            {
                m_restrictedCaps.supportsDXILRuntime = m_restrictedCaps.supportsDXILRuntime && available;
            }
        }
        void                        SetRestrictedCapabilitiesOverride(BOOL enableOverride, const BackendCapabilities& reqCaps) noexcept
        {
            if (!enableOverride)
            {
                m_hasRestrictedCaps = false;
                m_restrictedCaps    = BackendCapabilities{};
                return;
            }
            BackendCapabilities c{};
            c.maxFeatureLevel                     = (reqCaps.maxFeatureLevel < m_capabilities.maxFeatureLevel) ? reqCaps.maxFeatureLevel : m_capabilities.maxFeatureLevel;
            c.maxShaderModel                      = (reqCaps.maxShaderModel  < m_capabilities.maxShaderModel)  ? reqCaps.maxShaderModel  : m_capabilities.maxShaderModel;
            c.minWaveLaneCount                    = reqCaps.supportsWaveOps ? m_capabilities.minWaveLaneCount : 0u;
            c.maxWaveLaneCount                    = reqCaps.supportsWaveOps ? m_capabilities.maxWaveLaneCount : 0u;
            c.supportsWaveOps                     = reqCaps.supportsWaveOps                     && m_capabilities.supportsWaveOps;
            c.supportsInt64ShaderOps              = reqCaps.supportsInt64ShaderOps              && m_capabilities.supportsInt64ShaderOps;
            c.supportsAtomicInt64OnTypedResource  = reqCaps.supportsAtomicInt64OnTypedResource  && m_capabilities.supportsAtomicInt64OnTypedResource;
            c.supportsAtomicInt64OnGroupShared    = reqCaps.supportsAtomicInt64OnGroupShared    && m_capabilities.supportsAtomicInt64OnGroupShared;
            c.supportsAtomicInt64OnDescriptorHeap = reqCaps.supportsAtomicInt64OnDescriptorHeap && m_capabilities.supportsAtomicInt64OnDescriptorHeap;
            c.supportsTypedUAVLoadR32Uint         = reqCaps.supportsTypedUAVLoadR32Uint         && m_capabilities.supportsTypedUAVLoadR32Uint;
            c.supportsTypedUAVLoadRG32Uint        = reqCaps.supportsTypedUAVLoadRG32Uint        && m_capabilities.supportsTypedUAVLoadRG32Uint;
            c.supportsVPAndRTArrayIndexWithoutGS  = reqCaps.supportsVPAndRTArrayIndexWithoutGS  && m_capabilities.supportsVPAndRTArrayIndexWithoutGS;
            c.supportsDXILRuntime                 = reqCaps.supportsDXILRuntime                 && m_capabilities.supportsDXILRuntime;
            m_restrictedCaps                      = c;
            m_hasRestrictedCaps                   = true;
        }

    private:
        BOOL SelectAdapter();
        void QueryDeviceCapabilities();

    private:
        ComPtr<ID3D12Debug>               m_debugController;
        ComPtr<ID3D12InfoQueue>           m_infoQueue;
        BOOL                              m_requestDebugLayer = false;
        BOOL                              m_requestGBV        = false;
        BOOL                              m_debugLayerActive  = false;
        BOOL                              m_gbvActive         = false;
        BOOL                              m_supportsVPAndRTArrayIndexWithoutGS = false;
        D3D_FEATURE_LEVEL                 m_activeFeatureLevel                 = D3D_FEATURE_LEVEL_11_0;
        BackendCapabilities               m_capabilities{};
        BackendCapabilities               m_restrictedCaps{};
        BOOL                              m_hasRestrictedCaps                  = false;
        BOOL                              m_deviceRemoved                      = false;
        HRESULT                           m_deviceRemovedReason                = S_OK;
        ComPtr<IDXGIFactory4>             m_factory;
        ComPtr<IDXGIAdapter1>             m_adapter;
        ComPtr<ID3D12Device>              m_device;
        ComPtr<ID3D12CommandQueue>        m_commandQueue;

        // Upload execution context (synchronous GPU copies)
        ComPtr<ID3D12CommandAllocator>    m_uploadCmdAlloc;
        ComPtr<ID3D12GraphicsCommandList> m_uploadCmdList;
        ComPtr<ID3D12Fence>               m_uploadFence;
        UINT64                            m_uploadFenceValue = 0;
        HANDLE                            m_uploadFenceEvent = nullptr;

        // Frame synchronization
        ComPtr<ID3D12Fence>               m_frameFence;
        UINT64                            m_frameFenceValue        = 0;
        HANDLE                            m_frameFenceEvent        = nullptr;
        HANDLE                            m_frameFenceSharedHandle = nullptr;

        std::wstring                      m_adapterDescription;
        SIZE_T                            m_dedicatedVramMB = 0;
    };
}
