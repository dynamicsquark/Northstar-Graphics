#include "d3d12_device.h"
#include <iostream>
#include <cstdlib>
#include <vector>

namespace Northstar
{
    BOOL D3D12DeviceContext::Initialize()
    {
        if (m_device) return true;

        // 1. Enable Debug Layer and optional GPU-Based Validation if requested via API or environment variables
        //    (NORTHSTAR_D3D12_DEBUG=1 / NORTHSTAR_D3D12_GBV=1) before any D3D12 device is created.
        char envDebug[8]{};
        char envGbv[8]{};
        const DWORD lenDebug = ::GetEnvironmentVariableA("NORTHSTAR_D3D12_DEBUG", envDebug, sizeof(envDebug));
        const DWORD lenGbv   = ::GetEnvironmentVariableA("NORTHSTAR_D3D12_GBV", envGbv, sizeof(envGbv));
        const BOOL wantDebug = m_requestDebugLayer ||
                               (lenDebug > 0 && envDebug[0] == '1') ||
                               (lenGbv > 0 && envGbv[0] == '1');
        const BOOL wantGbv   = m_requestGBV || (lenGbv > 0 && envGbv[0] == '1');

        if (wantDebug)
        {
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&m_debugController))) && m_debugController)
            {
                m_debugController->EnableDebugLayer();
                m_debugLayerActive = true;

                ComPtr<ID3D12Debug1> debug1;
                if (SUCCEEDED(m_debugController.As(&debug1)) && debug1)
                {
                    debug1->SetEnableGPUBasedValidation(wantGbv ? TRUE : FALSE);
                    m_gbvActive = wantGbv;
                }
            }
        }
        else if (::GetModuleHandleW(L"D3D12SDKLayers.dll") != nullptr)
        {
            // If an earlier D3D12 session in this process enabled GPU-Based Validation,
            // explicitly clear process-global GBV before creating an unvalidated benchmark/runtime device.
            ComPtr<ID3D12Debug> prevDebug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&prevDebug))) && prevDebug)
            {
                ComPtr<ID3D12Debug1> debug1;
                if (SUCCEEDED(prevDebug.As(&debug1)) && debug1)
                {
                    debug1->SetEnableGPUBasedValidation(FALSE);
                }
            }
        }

        // 2. Create DXGI Factory
        UINT factoryFlags = m_debugLayerActive ? DXGI_CREATE_FACTORY_DEBUG : 0;
#ifdef _DEBUG
        factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
        if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory))))
        {
            if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&m_factory))))
            {
                return false;
            }
        }

        // 3. Select Discrete GPU Adapter & Create D3D12 Device
        if (!SelectAdapter())
        {
            Shutdown();
            return false;
        }

        if (m_debugLayerActive && m_device)
        {
            m_device.As(&m_infoQueue);
        }

        // Query D3D12 hardware capabilities (Feature Level, Shader Model 6.6, WaveOps, Int64 Atomics, Typed UAV Loads)
        QueryDeviceCapabilities();

        // 4. Create Command Queue
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queueDesc.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDesc.NodeMask = 0;
        if (FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue))))
        {
            Shutdown();
            return false;
        }

        // 5. Create Upload Synchronization Context
        if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_uploadCmdAlloc))))
        {
            Shutdown();
            return false;
        }

        if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_uploadCmdAlloc.Get(), nullptr, IID_PPV_ARGS(&m_uploadCmdList))))
        {
            Shutdown();
            return false;
        }

        m_uploadCmdList->Close();

        if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_uploadFence))))
        {
            Shutdown();
            return false;
        }

        m_uploadFenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_uploadFenceEvent)
        {
            Shutdown();
            return false;
        }

        // 6. Create Frame Fence (with D3D12_FENCE_FLAG_SHARED for cross-process / cross-device synchronization)
        m_frameFenceSharedHandle = nullptr;
        if (SUCCEEDED(m_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_frameFence))) && m_frameFence)
        {
            if (FAILED(m_device->CreateSharedHandle(m_frameFence.Get(), nullptr, GENERIC_ALL, nullptr, &m_frameFenceSharedHandle)))
            {
                m_frameFenceSharedHandle = nullptr;
            }
        }
        else
        {
            m_frameFence.Reset();
            if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_frameFence))))
            {
                Shutdown();
                return false;
            }
        }

        m_frameFenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!m_frameFenceEvent)
        {
            Shutdown();
            return false;
        }

        return true;
    }

    BOOL D3D12DeviceContext::SelectAdapter()
    {
        ComPtr<IDXGIAdapter1> bestAdapter;
        SIZE_T maxDedicatedVram = 0;
        DXGI_ADAPTER_DESC1 bestDesc{};

        for (UINT i = 0; ; ++i)
        {
            ComPtr<IDXGIAdapter1> candidateAdapter;
            if (m_factory->EnumAdapters1(i, &candidateAdapter) == DXGI_ERROR_NOT_FOUND)
            {
                break;
            }

            DXGI_ADAPTER_DESC1 desc1{};
            candidateAdapter->GetDesc1(&desc1);

            // Skip software rasterizers
            if (desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                continue;
            }

            // Skip virtual display drivers (Parsec, Virtual Display, Remote, etc.)
            std::wstring descStr(desc1.Description);
            if (descStr.find(L"Parsec") != std::wstring::npos ||
                descStr.find(L"Virtual") != std::wstring::npos ||
                descStr.find(L"Remote") != std::wstring::npos ||
                descStr.find(L"Basic Render") != std::wstring::npos)
            {
                continue;
            }

            // Check if D3D12 device can actually be created
            if (SUCCEEDED(D3D12CreateDevice(candidateAdapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
            {
                if (!bestAdapter || desc1.DedicatedVideoMemory > maxDedicatedVram)
                {
                    bestAdapter = candidateAdapter;
                    maxDedicatedVram = desc1.DedicatedVideoMemory;
                    bestDesc = desc1;
                }
            }
        }

        // Fallback: If no adapter passed strict filtering, take any valid hardware adapter
        if (!bestAdapter)
        {
            for (UINT i = 0; m_factory->EnumAdapters1(i, &bestAdapter) != DXGI_ERROR_NOT_FOUND; ++i)
            {
                DXGI_ADAPTER_DESC1 desc1{};
                bestAdapter->GetDesc1(&desc1);
                if (desc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                if (SUCCEEDED(D3D12CreateDevice(bestAdapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
                {
                    bestDesc = desc1;
                    break;
                }
            }
        }

        auto createHighestFeatureLevelDevice = [&](IUnknown* adapterPtr, ComPtr<ID3D12Device>& outDev, D3D_FEATURE_LEVEL& outLevel) -> BOOL
        {
            static constexpr D3D_FEATURE_LEVEL kLevels[] = {
                D3D_FEATURE_LEVEL_12_2,
                D3D_FEATURE_LEVEL_12_1,
                D3D_FEATURE_LEVEL_12_0,
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0
            };
            for (D3D_FEATURE_LEVEL fl : kLevels)
            {
                if (SUCCEEDED(D3D12CreateDevice(adapterPtr, fl, IID_PPV_ARGS(&outDev))) && outDev)
                {
                    outLevel = fl;
                    return true;
                }
            }
            return false;
        };

        if (bestAdapter && createHighestFeatureLevelDevice(bestAdapter.Get(), m_device, m_activeFeatureLevel))
        {
            m_adapter = bestAdapter;
            m_adapterDescription = bestDesc.Description;
            m_dedicatedVramMB = bestDesc.DedicatedVideoMemory / (1024 * 1024);
            std::wcout << L"[NorthstarD3D12] Selected GPU: " << m_adapterDescription
                       << L" (" << m_dedicatedVramMB << L" MB VRAM, FL=0x"
                       << std::hex << static_cast<UINT32>(m_activeFeatureLevel) << std::dec << L")\n";
            return true;
        }

        // Software WARP fallback if hardware adapter is not available
        ComPtr<IDXGIAdapter> warpAdapter;
        m_factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
        if (warpAdapter && createHighestFeatureLevelDevice(warpAdapter.Get(), m_device, m_activeFeatureLevel))
        {
            m_adapterDescription = L"Microsoft WARP (Software)";
            m_dedicatedVramMB = 0;
            std::cout << "[NorthstarD3D12] Warning: Using WARP software adapter.\n";
            return true;
        }

        return false;
    }

    void D3D12DeviceContext::QueryDeviceCapabilities()
    {
        m_capabilities                       = BackendCapabilities{};
        m_capabilities.maxFeatureLevel       = static_cast<UINT32>(m_activeFeatureLevel);
        m_supportsVPAndRTArrayIndexWithoutGS = false;
        if (!m_device) return;

        // 1. Check highest supported D3D_FEATURE_LEVEL via D3D12_FEATURE_FEATURE_LEVELS
        {
            static constexpr D3D_FEATURE_LEVEL kQueryLevels[] = {
                D3D_FEATURE_LEVEL_12_2,
                D3D_FEATURE_LEVEL_12_1,
                D3D_FEATURE_LEVEL_12_0,
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0
            };
            D3D12_FEATURE_DATA_FEATURE_LEVELS flData{};
            flData.NumFeatureLevels        = _countof(kQueryLevels);
            flData.pFeatureLevelsRequested = kQueryLevels;
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &flData, sizeof(flData))))
            {
                m_capabilities.maxFeatureLevel = static_cast<UINT32>(flData.MaxSupportedFeatureLevel);
            }
        }

        // 2. D3D12_OPTIONS (VP/RTArrayIndex without GS)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS d3d12Options{};
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &d3d12Options, sizeof(d3d12Options))))
            {
                m_supportsVPAndRTArrayIndexWithoutGS =
                    (d3d12Options.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation != FALSE);
                m_capabilities.supportsVPAndRTArrayIndexWithoutGS = m_supportsVPAndRTArrayIndexWithoutGS;
            }
            if (!m_supportsVPAndRTArrayIndexWithoutGS)
            {
                std::cout << "[NorthstarD3D12] Note: VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation is FALSE on this adapter; layered depth passes will use the Geometry Shader fallback path.\n";
            }
        }

        // 3. D3D12_FEATURE_SHADER_MODEL (probe 6.6 -> 6.5 -> 6.2 -> 6.0 -> 5.1)
        {
            static constexpr D3D_SHADER_MODEL kShaderModels[] = {
                static_cast<D3D_SHADER_MODEL>(0x66), // D3D_SHADER_MODEL_6_6
                static_cast<D3D_SHADER_MODEL>(0x65), // D3D_SHADER_MODEL_6_5
                static_cast<D3D_SHADER_MODEL>(0x62), // D3D_SHADER_MODEL_6_2
                static_cast<D3D_SHADER_MODEL>(0x60), // D3D_SHADER_MODEL_6_0
                D3D_SHADER_MODEL_5_1
            };
            for (D3D_SHADER_MODEL sm : kShaderModels)
            {
                D3D12_FEATURE_DATA_SHADER_MODEL smData{};
                smData.HighestShaderModel = sm;
                if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &smData, sizeof(smData))))
                {
                    m_capabilities.maxShaderModel = static_cast<UINT32>(smData.HighestShaderModel);
                    break;
                }
            }
        }

        // 4. D3D12_OPTIONS1 (Wave intrinsics + 64-bit integer shader ops)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 opt1{};
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opt1, sizeof(opt1))))
            {
                m_capabilities.supportsWaveOps         = (opt1.WaveOps != FALSE);
                m_capabilities.minWaveLaneCount        = opt1.WaveLaneCountMin;
                m_capabilities.maxWaveLaneCount        = opt1.WaveLaneCountMax;
                m_capabilities.supportsInt64ShaderOps  = (opt1.Int64ShaderOps != FALSE);
            }
        }

        // 5. D3D12_OPTIONS9 (64-bit atomics on typed resources and groupshared memory)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS9 opt9{};
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &opt9, sizeof(opt9))))
            {
                m_capabilities.supportsAtomicInt64OnTypedResource = (opt9.AtomicInt64OnTypedResourceSupported != FALSE);
                m_capabilities.supportsAtomicInt64OnGroupShared   = (opt9.AtomicInt64OnGroupSharedSupported != FALSE);
            }
        }

        // 6. D3D12_OPTIONS11 (64-bit atomics on descriptor-heap buffers/resources)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS11 opt11{};
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &opt11, sizeof(opt11))))
            {
                m_capabilities.supportsAtomicInt64OnDescriptorHeap = (opt11.AtomicInt64OnDescriptorHeapResourceSupported != FALSE);
            }
        }

        // 7. Typed UAV Load support for R32_UINT and R32G32_UINT
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT fmtR32{};
            fmtR32.Format = DXGI_FORMAT_R32_UINT;
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fmtR32, sizeof(fmtR32))))
            {
                m_capabilities.supportsTypedUAVLoadR32Uint =
                    ((fmtR32.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0);
            }

            D3D12_FEATURE_DATA_FORMAT_SUPPORT fmtRG32{};
            fmtRG32.Format = DXGI_FORMAT_R32G32_UINT;
            if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fmtRG32, sizeof(fmtRG32))))
            {
                m_capabilities.supportsTypedUAVLoadRG32Uint =
                    ((fmtRG32.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0);
            }
        }
    }

    BOOL D3D12DeviceContext::WaitForFenceValue(ID3D12Fence* fence, HANDLE evt, UINT64 targetValue)
    {
        if (!fence || !evt) return false;
        if (m_deviceRemoved) return false;

        UINT64 completed = fence->GetCompletedValue();
        if (completed == UINT64_MAX)
        {
            m_deviceRemoved = true;
            if (m_device) m_deviceRemovedReason = m_device->GetDeviceRemovedReason();
            std::cerr << "[NorthstarD3D12] Device removed detected while polling fence (reason=0x"
                      << std::hex << m_deviceRemovedReason << std::dec << ").\n";
            return false;
        }
        if (completed >= targetValue) return true;

        HRESULT hr = fence->SetEventOnCompletion(targetValue, evt);
        if (FAILED(hr))
        {
            if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_HUNG || hr == DXGI_ERROR_DEVICE_RESET)
            {
                m_deviceRemoved = true;
                if (m_device) m_deviceRemovedReason = m_device->GetDeviceRemovedReason();
            }
            return false;
        }

        // Wait in bounded slices so a GPU TDR / device removal can never deadlock the thread on INFINITE
        for (;;)
        {
            const DWORD waitRes = ::WaitForSingleObject(evt, 5000);
            if (waitRes == WAIT_OBJECT_0)
            {
                return true;
            }
            completed = fence->GetCompletedValue();
            if (completed == UINT64_MAX)
            {
                m_deviceRemoved = true;
                if (m_device) m_deviceRemovedReason = m_device->GetDeviceRemovedReason();
                std::cerr << "[NorthstarD3D12] GPU device removed during fence wait (reason=0x"
                          << std::hex << m_deviceRemovedReason << std::dec << ").\n";
                return false;
            }
            if (completed >= targetValue)
            {
                return true;
            }
            if (m_device)
            {
                const HRESULT reason = m_device->GetDeviceRemovedReason();
                if (FAILED(reason))
                {
                    m_deviceRemoved       = true;
                    m_deviceRemovedReason = reason;
                    std::cerr << "[NorthstarD3D12] GPU device lost during fence wait (reason=0x"
                              << std::hex << reason << std::dec << ").\n";
                    return false;
                }
            }
        }
    }

    void D3D12DeviceContext::ExecuteUpload(std::function<void(ID3D12GraphicsCommandList*)> recordFn)
    {
        if (!m_uploadCmdAlloc || !m_uploadCmdList || !m_commandQueue || m_deviceRemoved) return;

        if (FAILED(m_uploadCmdAlloc->Reset())) return;
        if (FAILED(m_uploadCmdList->Reset(m_uploadCmdAlloc.Get(), nullptr))) return;

        recordFn(m_uploadCmdList.Get());

        if (FAILED(m_uploadCmdList->Close())) return;
        ID3D12CommandList* lists[] = { m_uploadCmdList.Get() };
        m_commandQueue->ExecuteCommandLists(1, lists);

        m_uploadFenceValue++;
        if (FAILED(m_commandQueue->Signal(m_uploadFence.Get(), m_uploadFenceValue)))
        {
            if (m_device && FAILED(m_device->GetDeviceRemovedReason()))
            {
                m_deviceRemoved       = true;
                m_deviceRemovedReason = m_device->GetDeviceRemovedReason();
            }
            return;
        }

        WaitForFenceValue(m_uploadFence.Get(), m_uploadFenceEvent, m_uploadFenceValue);
    }

    void D3D12DeviceContext::WaitForGPU()
    {
        if (!m_commandQueue || !m_frameFence || !m_frameFenceEvent || m_deviceRemoved) return;

        m_frameFenceValue++;
        if (FAILED(m_commandQueue->Signal(m_frameFence.Get(), m_frameFenceValue)))
        {
            if (m_device && FAILED(m_device->GetDeviceRemovedReason()))
            {
                m_deviceRemoved       = true;
                m_deviceRemovedReason = m_device->GetDeviceRemovedReason();
            }
            return;
        }

        WaitForFenceValue(m_frameFence.Get(), m_frameFenceEvent, m_frameFenceValue);
    }

    UINT32 D3D12DeviceContext::GetDebugErrorCount() const
    {
        if (!m_infoQueue) return 0;

        UINT32 errorCount = 0;
        const UINT64 numMessages = m_infoQueue->GetNumStoredMessages();
        for (UINT64 i = 0; i < numMessages; ++i)
        {
            SIZE_T msgLen = 0;
            if (SUCCEEDED(m_infoQueue->GetMessage(i, nullptr, &msgLen)) && msgLen > 0)
            {
                std::vector<BYTE> msgBuf(msgLen);
                D3D12_MESSAGE* msg = reinterpret_cast<D3D12_MESSAGE*>(msgBuf.data());
                if (SUCCEEDED(m_infoQueue->GetMessage(i, msg, &msgLen)))
                {
                    if (msg->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
                        msg->Severity == D3D12_MESSAGE_SEVERITY_ERROR)
                    {
                        errorCount++;
                        std::cerr << "[D3D12 Validation Error] " << (msg->pDescription ? msg->pDescription : "") << "\n";
                    }
                }
            }
        }
        return errorCount;
    }

    void D3D12DeviceContext::ClearDebugMessages()
    {
        if (m_infoQueue)
        {
            m_infoQueue->ClearStoredMessages();
        }
    }

    void D3D12DeviceContext::Shutdown()
    {
        WaitForGPU();

        if (m_uploadFenceEvent)
        {
            CloseHandle(m_uploadFenceEvent);
            m_uploadFenceEvent = nullptr;
        }

        if (m_frameFenceSharedHandle)
        {
            CloseHandle(m_frameFenceSharedHandle);
            m_frameFenceSharedHandle = nullptr;
        }

        if (m_frameFenceEvent)
        {
            CloseHandle(m_frameFenceEvent);
            m_frameFenceEvent = nullptr;
        }

        m_infoQueue.Reset();
        m_uploadFence.Reset();
        m_frameFence.Reset();
        m_uploadFenceValue = 0;
        m_frameFenceValue  = 0;
        m_uploadCmdList.Reset();
        m_uploadCmdAlloc.Reset();
        m_commandQueue.Reset();
        m_device.Reset();
        m_adapter.Reset();
        m_factory.Reset();
        m_debugController.Reset();
        m_debugLayerActive = false;
        m_gbvActive        = false;
        m_supportsVPAndRTArrayIndexWithoutGS = false;
        m_activeFeatureLevel                 = D3D_FEATURE_LEVEL_11_0;
        m_capabilities                       = BackendCapabilities{};
        m_deviceRemoved                      = false;
        m_deviceRemovedReason                = S_OK;
    }
}
