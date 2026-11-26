#pragma once

#include "ns_platform_macros.h"
#include "ns_window_types.h"

// Platform Window Resolution
#if defined(NS_PLATFORM_WINDOWS)
    #include "platform/windows/ns_win32_window.h"
    namespace Northstar::Tools
    {
        // On Windows platform, Window is alias to Win32Window
        using Window = Win32Window;
    }
#elif defined(NS_PLATFORM_LINUX)
    #error "Northstar Tools: Linux window backend is not yet implemented!"
#elif defined(NS_PLATFORM_MACOS)
    #error "Northstar Tools: macOS window backend is not yet implemented!"
#else
    #error "Northstar Tools: Unsupported target window platform!"
#endif
