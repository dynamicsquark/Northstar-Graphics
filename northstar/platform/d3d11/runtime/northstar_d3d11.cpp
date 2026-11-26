#include "northstar_d3d11.h"
#include "api/northstar_engine.h"
#include <iostream>

namespace
{
    constexpr const char* kNorthstarD3D11UnsupportedMessage =
        "[NorthstarD3D11] Direct3D 11 backend is unsupported in this release; initialize with NorthstarD3D12.";
}

namespace Northstar
{
    BOOL NorthstarD3D11::Initialize(const NativeRenderTarget& target)
    {
        (void)target;
        std::cerr << kNorthstarD3D11UnsupportedMessage << "\n";
        return false;
    }
}

Northstar::UINT32 Northstar_GetAPIVersion()
{
    return Northstar::NORTHSTAR_ENGINE_API_VERSION;
}

const char* GetNorthstarD3D11BackendStatusMessage()
{
    return kNorthstarD3D11UnsupportedMessage;
}

Northstar::NorthstarBackend* CreateNorthstarD3D11Backend()
{
    return nullptr;
}

void DestroyNorthstarD3D11Backend(Northstar::NorthstarBackend* backend)
{
    delete backend;
}
