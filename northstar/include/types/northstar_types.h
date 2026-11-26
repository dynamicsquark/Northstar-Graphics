#pragma once

#include "northstar_platform_guard.h"
#include "QuarkDynSDK/header_only/types/global_types.h"

namespace Northstar
{
    using QuarkDyn::UINT8;
    using QuarkDyn::UINT16;
    using QuarkDyn::UINT32;
    using QuarkDyn::UINT64;

    using QuarkDyn::INT8;
    using QuarkDyn::INT16;
    using QuarkDyn::INT32;
    using QuarkDyn::INT64;

    using QuarkDyn::FLOAT32;
    using QuarkDyn::FLOAT64;

    using QuarkDyn::SIZE_T;
    using QuarkDyn::BYTE;
    using QuarkDyn::BOOL;

    using QuarkDyn::NativeHandle;

    // Engine semantic version
    inline constexpr UINT32      VERSION_MAJOR  = 0;
    inline constexpr UINT32      VERSION_MINOR  = 6;
    inline constexpr UINT32      VERSION_PATCH  = 0;
    inline constexpr const char* VERSION_STRING = "0.6.0";

    // Hardware backend selection for RuntimeSession
    enum class BackendAPI : UINT8
    {
        D3D12  = 0,
        D3D11  = 1,
        Custom = 2
    };
}
