#pragma once

#include "ns_window_types.h"

struct HWND__;
using HWND = HWND__*;
struct HINSTANCE__;
using HINSTANCE = HINSTANCE__*;

namespace Northstar::Tools
{
    // ==================== WIN32 WINDOW IMPLEMENTATION ====================
    class Win32Window
    {
    public:
        Win32Window();
        ~Win32Window();

        Win32Window(const Win32Window&) = delete;
        Win32Window& operator=(const Win32Window&) = delete;

        Win32Window(Win32Window&& other) noexcept;
        Win32Window& operator=(Win32Window&& other) noexcept;

        BOOL Create(const WindowDesc& desc);
        BOOL Create(UINT32 width, UINT32 height, const wchar_t* title);
        void PollMessages();
        void Close();

        void SetTitle(const std::wstring& title);
        void SetTitle(const std::string& title);
        void SetSize(UINT32 width, UINT32 height);
        void SetPosition(INT32 x, INT32 y);
        void SetFullscreen(BOOL fullscreen);
        void SetVisible(BOOL visible);

        BOOL IsRunning()    const { return m_isRunning; }
        BOOL IsFocused()    const { return m_isFocused; }
        BOOL IsMinimized()  const { return m_isMinimized; }
        BOOL IsMaximized()  const { return m_isMaximized; }
        BOOL IsFullscreen() const { return m_isFullscreen; }

        UINT32 GetWidth()  const { return m_width; }
        UINT32 GetHeight() const { return m_height; }
        const WindowDesc& GetDesc() const { return m_desc; }

        // Native Render Target for direct NorthstarEngine initialization
        NativeRenderTarget GetNativeRenderTarget() const;

        HWND      GetHWND()      const { return m_hwnd; }
        HINSTANCE GetHINSTANCE() const { return m_hInstance; }
        HWND      GetHandle()    const { return m_hwnd; }
        HINSTANCE GetInstance()  const { return m_hInstance; }
        void*     GetNativeHandle()   const { return reinterpret_cast<void*>(m_hwnd); }
        void*     GetNativeInstance() const { return reinterpret_cast<void*>(m_hInstance); }

        // Event callbacks
        void SetOnResize(ResizeCallback cb)                   { m_onResize = std::move(cb); }
        void SetOnClose(CloseCallback cb)                     { m_onClose = std::move(cb); }
        void SetOnKey(KeyCallback cb)                         { m_onKey = std::move(cb); }
        void SetOnChar(CharCallback cb)                       { m_onChar = std::move(cb); }
        void SetOnMouseButton(MouseButtonCallback cb)         { m_onMouseButton = std::move(cb); }
        void SetOnMouseMove(MouseMoveCallback cb)             { m_onMouseMove = std::move(cb); }
        void SetOnMouseWheel(MouseWheelCallback cb)           { m_onMouseWheel = std::move(cb); }
        void SetOnFocus(FocusCallback cb)                     { m_onFocus = std::move(cb); }
        void SetNativeMessageHook(NativeMessageHookFn cb)     { m_messageHook = std::move(cb); }

        static KeyCode TranslateVirtualKey(std::uintptr_t wParam, std::intptr_t lParam);

    private:
        HWND               m_hwnd         = nullptr;
        HINSTANCE          m_hInstance    = nullptr;
        WindowDesc         m_desc{};
        UINT32             m_width        = 0;
        UINT32             m_height       = 0;
        BOOL               m_isRunning    = false;
        BOOL               m_isFocused    = false;
        BOOL               m_isMinimized  = false;
        BOOL               m_isMaximized  = false;
        BOOL               m_isFullscreen = false;
        BOOL               m_countedInClass = false;
        UINT16             m_highSurrogate  = 0;

        // Saved windowed bounds for restoring from fullscreen
        INT32              m_savedX       = 0;
        INT32              m_savedY       = 0;
        INT32              m_savedW       = 0;
        INT32              m_savedH       = 0;

        ResizeCallback      m_onResize;
        CloseCallback       m_onClose;
        KeyCallback         m_onKey;
        CharCallback        m_onChar;
        MouseButtonCallback m_onMouseButton;
        MouseMoveCallback   m_onMouseMove;
        MouseWheelCallback  m_onMouseWheel;
        FocusCallback       m_onFocus;
        NativeMessageHookFn m_messageHook;

        static std::intptr_t __stdcall StaticWindowProc(HWND hwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam);
        std::intptr_t HandleMessage(HWND hwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam);

        void ApplyImmersiveDarkMode();
        void ReleaseClassRegistration();
    };
}
