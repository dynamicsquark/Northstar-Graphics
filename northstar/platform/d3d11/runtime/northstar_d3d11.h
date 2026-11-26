#pragma once

#include "platform/northstar_backend.h"

#ifdef _WIN32
    #ifdef NORTHSTAR_D3D11_EXPORTS
        #define NORTHSTAR_D3D11_API __declspec(dllexport)
    #else
        #define NORTHSTAR_D3D11_API __declspec(dllimport)
    #endif
#else
    #define NORTHSTAR_D3D11_API
#endif

namespace Northstar
{
    // Direct3D 11 Backend Stub (Unsupported in the current architecture; use NorthstarD3D12).
    // Fails deterministically during Initialize() with a clear diagnostic message.
    class NorthstarD3D11 final : public NorthstarBackend
    {
    public:
        NorthstarD3D11()           = default;
        ~NorthstarD3D11() override = default;

        TargetAPI GetTargetAPI() const                          override { return TargetAPI::HLSL11; }
        BOOL      IsHardwareBackend() const                     override { return true; }
        BOOL      Initialize(const NativeRenderTarget& target)  override;
        void      Shutdown()                                    override {}
        BOOL      IsInitialized() const                         override { return false; }
        void      ExecuteFrame(const FrameView& view)           override { (void)view; }
        void      Present()                                     override {}
    };
}

extern "C" {
    NORTHSTAR_D3D11_API Northstar::UINT32            Northstar_GetAPIVersion();
    NORTHSTAR_D3D11_API Northstar::NorthstarBackend* CreateNorthstarD3D11Backend();
    NORTHSTAR_D3D11_API void                         DestroyNorthstarD3D11Backend(Northstar::NorthstarBackend* backend);
    NORTHSTAR_D3D11_API const char*                  GetNorthstarD3D11BackendStatusMessage();
}
