#pragma once

#include "ns_platform_macros.h"
#include "types/northstar_types.h"
#include "platform/native_render_target.h"

#include <string>
#include <functional>
#include <optional>
#include <cstdint>

namespace Northstar::Tools
{
    // ==================== KEY CODES ====================
    enum class KeyCode : UINT16
    {
        Unknown = 0,
        
        // Alphanumeric
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,

        // Function Keys
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

        // Control / Navigation
        Escape,
        Enter,
        Tab,
        Backspace,
        Space,
        Insert,
        Delete,
        Home,
        End,
        PageUp,
        PageDown,
        Left,
        Right,
        Up,
        Down,

        // Modifiers
        LeftShift,
        RightShift,
        LeftControl,
        RightControl,
        LeftAlt,
        RightAlt,
        LeftSuper,  // Windows key / Command key
        RightSuper,
        CapsLock,
        ScrollLock,
        NumLock,
        PrintScreen,
        Pause
    };

    // ==================== MOUSE BUTTONS ====================
    enum class MouseButton : UINT8
    {
        Left = 0,
        Right,
        Middle,
        Button4,
        Button5
    };

    // ==================== WINDOW STATE ====================
    enum class WindowState : UINT8
    {
        Normal = 0,
        Minimized,
        Maximized,
        Fullscreen
    };

    // ==================== WINDOW DESCRIPTION ====================
    struct WindowDesc
    {
        UINT32       width          = 1280;
        UINT32       height         = 720;
        std::wstring title          = L"Northstar Graphics Window";
        BOOL         resizable      = true;
        BOOL         visible        = true;
        BOOL         fullscreen     = false;
        BOOL         enableDarkMode = true;
        BOOL         centered       = true;
    };

    // ==================== CALLBACK TYPES ====================
    using ResizeCallback         = std::function<void(UINT32 width, UINT32 height)>;
    using CloseCallback          = std::function<BOOL()>;
    using KeyCallback            = std::function<void(KeyCode key, BOOL isDown, BOOL isRepeat)>;
    using CharCallback           = std::function<void(UINT32 codepoint)>;
    using MouseButtonCallback    = std::function<void(MouseButton button, BOOL isDown, INT32 x, INT32 y)>;
    using MouseMoveCallback      = std::function<void(INT32 x, INT32 y)>;
    using MouseWheelCallback     = std::function<void(FLOAT32 delta, INT32 x, INT32 y)>;
    using FocusCallback          = std::function<void(BOOL isFocused)>;
    using NativeMessageHookFn    = std::function<std::optional<std::intptr_t>(void* nativeHwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam)>;
}
