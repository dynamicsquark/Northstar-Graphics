#include "platform/windows/ns_win32_window.h"

#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#endif

namespace Northstar::Tools
{
    namespace
    {
        constexpr const wchar_t* kWindowClassName = L"Northstar_Window_Class";

        std::mutex s_windowClassMutex;
        HBRUSH     s_backgroundBrush = nullptr;
        ATOM       s_windowClassAtom = 0;
        UINT32     s_liveWindowCount = 0;

        void AdjustClientRectForDpi(RECT& rect, DWORD style, HWND hwnd)
        {
            using AdjustWindowRectExForDpiFn = BOOL(WINAPI*)(LPRECT, DWORD, BOOL, DWORD, UINT);
            using GetDpiForWindowFn          = UINT(WINAPI*)(HWND);
            using GetDpiForSystemFn          = UINT(WINAPI*)();

            static HMODULE s_user32 = GetModuleHandleW(L"user32.dll");
            static auto s_adjustForDpi = s_user32
                ? reinterpret_cast<AdjustWindowRectExForDpiFn>(GetProcAddress(s_user32, "AdjustWindowRectExForDpi"))
                : nullptr;
            static auto s_getDpiForWindow = s_user32
                ? reinterpret_cast<GetDpiForWindowFn>(GetProcAddress(s_user32, "GetDpiForWindow"))
                : nullptr;
            static auto s_getDpiForSystem = s_user32
                ? reinterpret_cast<GetDpiForSystemFn>(GetProcAddress(s_user32, "GetDpiForSystem"))
                : nullptr;

            if (s_adjustForDpi)
            {
                UINT dpi = 96u;
                if (hwnd && s_getDpiForWindow)
                {
                    const UINT wndDpi = s_getDpiForWindow(hwnd);
                    if (wndDpi > 0u) dpi = wndDpi;
                }
                else if (s_getDpiForSystem)
                {
                    const UINT sysDpi = s_getDpiForSystem();
                    if (sysDpi > 0u) dpi = sysDpi;
                }

                if (s_adjustForDpi(&rect, style, FALSE, 0u, dpi))
                {
                    return;
                }
            }

            AdjustWindowRect(&rect, style, FALSE);
        }
    }

    KeyCode Win32Window::TranslateVirtualKey(std::uintptr_t wParam, std::intptr_t lParam)
    {
        if (wParam >= 'A' && wParam <= 'Z')
        {
            return static_cast<KeyCode>(static_cast<UINT16>(KeyCode::A) + (wParam - 'A'));
        }
        if (wParam >= '0' && wParam <= '9')
        {
            return static_cast<KeyCode>(static_cast<UINT16>(KeyCode::Num0) + (wParam - '0'));
        }
        if (wParam >= VK_F1 && wParam <= VK_F12)
        {
            return static_cast<KeyCode>(static_cast<UINT16>(KeyCode::F1) + (wParam - VK_F1));
        }

        const bool isExtended = (lParam & (static_cast<std::intptr_t>(1) << 24)) != 0;

        switch (wParam)
        {
        case VK_SHIFT:
        {
            const UINT scancode = static_cast<UINT>((static_cast<std::uintptr_t>(lParam) >> 16u) & 0xFFu);
            if (scancode != 0u)
            {
                const UINT mappedVk = MapVirtualKeyW(scancode, MAPVK_VSC_TO_VK_EX);
                if (mappedVk == VK_RSHIFT)
                {
                    return KeyCode::RightShift;
                }
            }
            return isExtended ? KeyCode::RightShift : KeyCode::LeftShift;
        }
        case VK_CONTROL:
            return isExtended ? KeyCode::RightControl : KeyCode::LeftControl;
        case VK_MENU:
            return isExtended ? KeyCode::RightAlt : KeyCode::LeftAlt;

        case VK_ESCAPE:     return KeyCode::Escape;
        case VK_RETURN:     return KeyCode::Enter;
        case VK_TAB:        return KeyCode::Tab;
        case VK_BACK:       return KeyCode::Backspace;
        case VK_SPACE:      return KeyCode::Space;
        case VK_INSERT:     return KeyCode::Insert;
        case VK_DELETE:     return KeyCode::Delete;
        case VK_HOME:       return KeyCode::Home;
        case VK_END:        return KeyCode::End;
        case VK_PRIOR:      return KeyCode::PageUp;
        case VK_NEXT:       return KeyCode::PageDown;
        case VK_LEFT:       return KeyCode::Left;
        case VK_RIGHT:      return KeyCode::Right;
        case VK_UP:         return KeyCode::Up;
        case VK_DOWN:       return KeyCode::Down;
        case VK_LSHIFT:     return KeyCode::LeftShift;
        case VK_RSHIFT:     return KeyCode::RightShift;
        case VK_LCONTROL:   return KeyCode::LeftControl;
        case VK_RCONTROL:   return KeyCode::RightControl;
        case VK_LMENU:      return KeyCode::LeftAlt;
        case VK_RMENU:      return KeyCode::RightAlt;
        case VK_LWIN:       return KeyCode::LeftSuper;
        case VK_RWIN:       return KeyCode::RightSuper;
        case VK_CAPITAL:    return KeyCode::CapsLock;
        case VK_SCROLL:     return KeyCode::ScrollLock;
        case VK_NUMLOCK:    return KeyCode::NumLock;
        case VK_SNAPSHOT:   return KeyCode::PrintScreen;
        case VK_PAUSE:      return KeyCode::Pause;
        default:            return KeyCode::Unknown;
        }
    }

    Win32Window::Win32Window()
        : m_hwnd(nullptr)
        , m_hInstance(static_cast<HINSTANCE>(GetModuleHandleW(nullptr)))
        , m_width(0)
        , m_height(0)
        , m_isRunning(false)
        , m_isFocused(false)
        , m_isMinimized(false)
        , m_isMaximized(false)
        , m_isFullscreen(false)
        , m_countedInClass(false)
        , m_highSurrogate(0)
    {
    }

    Win32Window::~Win32Window()
    {
        Close();
    }

    Win32Window::Win32Window(Win32Window&& other) noexcept
        : m_hwnd(other.m_hwnd)
        , m_hInstance(other.m_hInstance)
        , m_desc(other.m_desc)
        , m_width(other.m_width)
        , m_height(other.m_height)
        , m_isRunning(other.m_isRunning)
        , m_isFocused(other.m_isFocused)
        , m_isMinimized(other.m_isMinimized)
        , m_isMaximized(other.m_isMaximized)
        , m_isFullscreen(other.m_isFullscreen)
        , m_countedInClass(other.m_countedInClass)
        , m_highSurrogate(other.m_highSurrogate)
        , m_savedX(other.m_savedX)
        , m_savedY(other.m_savedY)
        , m_savedW(other.m_savedW)
        , m_savedH(other.m_savedH)
        , m_onResize(std::move(other.m_onResize))
        , m_onClose(std::move(other.m_onClose))
        , m_onKey(std::move(other.m_onKey))
        , m_onChar(std::move(other.m_onChar))
        , m_onMouseButton(std::move(other.m_onMouseButton))
        , m_onMouseMove(std::move(other.m_onMouseMove))
        , m_onMouseWheel(std::move(other.m_onMouseWheel))
        , m_onFocus(std::move(other.m_onFocus))
        , m_messageHook(std::move(other.m_messageHook))
    {
        other.m_hwnd           = nullptr;
        other.m_isRunning      = false;
        other.m_countedInClass = false;
        other.m_highSurrogate  = 0;
        if (m_hwnd)
        {
            SetWindowLongPtrW(m_hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        }
    }

    Win32Window& Win32Window::operator=(Win32Window&& other) noexcept
    {
        if (this != &other)
        {
            Close();

            m_hwnd           = other.m_hwnd;
            m_hInstance      = other.m_hInstance;
            m_desc           = other.m_desc;
            m_width          = other.m_width;
            m_height         = other.m_height;
            m_isRunning      = other.m_isRunning;
            m_isFocused      = other.m_isFocused;
            m_isMinimized    = other.m_isMinimized;
            m_isMaximized    = other.m_isMaximized;
            m_isFullscreen   = other.m_isFullscreen;
            m_countedInClass = other.m_countedInClass;
            m_highSurrogate  = other.m_highSurrogate;
            m_savedX         = other.m_savedX;
            m_savedY         = other.m_savedY;
            m_savedW         = other.m_savedW;
            m_savedH         = other.m_savedH;
            m_onResize       = std::move(other.m_onResize);
            m_onClose        = std::move(other.m_onClose);
            m_onKey          = std::move(other.m_onKey);
            m_onChar         = std::move(other.m_onChar);
            m_onMouseButton  = std::move(other.m_onMouseButton);
            m_onMouseMove    = std::move(other.m_onMouseMove);
            m_onMouseWheel   = std::move(other.m_onMouseWheel);
            m_onFocus        = std::move(other.m_onFocus);
            m_messageHook    = std::move(other.m_messageHook);

            other.m_hwnd           = nullptr;
            other.m_isRunning      = false;
            other.m_countedInClass = false;
            other.m_highSurrogate  = 0;

            if (m_hwnd)
            {
                SetWindowLongPtrW(m_hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
            }
        }
        return *this;
    }

    void Win32Window::ReleaseClassRegistration()
    {
        if (!m_countedInClass)
        {
            return;
        }
        m_countedInClass = false;

        std::lock_guard<std::mutex> lock(s_windowClassMutex);
        if (s_liveWindowCount > 0u)
        {
            --s_liveWindowCount;
        }
        if (s_liveWindowCount == 0u)
        {
            if (s_windowClassAtom != 0)
            {
                UnregisterClassW(kWindowClassName, m_hInstance);
                s_windowClassAtom = 0;
            }
            if (s_backgroundBrush != nullptr)
            {
                DeleteObject(s_backgroundBrush);
                s_backgroundBrush = nullptr;
            }
        }
    }

    std::intptr_t __stdcall Win32Window::StaticWindowProc(HWND hwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam)
    {
        Win32Window* pWindow = nullptr;

        if (msg == WM_NCCREATE)
        {
            CREATESTRUCTW* pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
            pWindow = reinterpret_cast<Win32Window*>(pCreate->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWindow));
            if (pWindow)
            {
                pWindow->m_hwnd = hwnd;
            }
        }
        else
        {
            pWindow = reinterpret_cast<Win32Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }

        if (pWindow)
        {
            return pWindow->HandleMessage(hwnd, msg, wParam, lParam);
        }

        return DefWindowProcW(hwnd, msg, static_cast<WPARAM>(wParam), static_cast<LPARAM>(lParam));
    }

    std::intptr_t Win32Window::HandleMessage(HWND hwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam)
    {
        if (m_messageHook)
        {
            auto result = m_messageHook(reinterpret_cast<void*>(hwnd), msg, wParam, lParam);
            if (result.has_value())
            {
                return result.value();
            }
        }

        switch (msg)
        {
        case WM_SIZE:
        {
            m_width       = LOWORD(lParam);
            m_height      = HIWORD(lParam);
            m_isMinimized = (wParam == SIZE_MINIMIZED);
            m_isMaximized = (wParam == SIZE_MAXIMIZED);

            if (m_onResize && !m_isMinimized)
            {
                m_onResize(m_width, m_height);
            }
            return 0;
        }

        case WM_SETFOCUS:
            m_isFocused = true;
            if (m_onFocus) m_onFocus(true);
            return 0;

        case WM_KILLFOCUS:
            m_isFocused = false;
            m_highSurrogate = 0;
            if (m_onFocus) m_onFocus(false);
            return 0;

        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
        {
            const BOOL isRepeat = (lParam & (static_cast<std::intptr_t>(1) << 30)) != 0;
            const KeyCode key = TranslateVirtualKey(wParam, lParam);
            if (m_onKey)
            {
                m_onKey(key, true, isRepeat);
            }
            if (msg == WM_SYSKEYDOWN)
            {
                if (wParam == VK_F4 &&
                    (lParam & (static_cast<std::intptr_t>(1) << 29)) != 0 &&
                    (GetWindowLongPtrW(hwnd, GWL_STYLE) & WS_SYSMENU) != 0)
                {
                    SendMessageW(hwnd, WM_SYSCOMMAND, SC_CLOSE, 0);
                    return 0;
                }
                // Forward system keys to DefWindowProcW.
                return DefWindowProcW(hwnd, msg, static_cast<WPARAM>(wParam), static_cast<LPARAM>(lParam));
            }
            return 0;
        }

        case WM_KEYUP:
        case WM_SYSKEYUP:
        {
            const KeyCode key = TranslateVirtualKey(wParam, lParam);
            if (m_onKey)
            {
                m_onKey(key, false, false);
            }
            if (msg == WM_SYSKEYUP)
            {
                return DefWindowProcW(hwnd, msg, static_cast<WPARAM>(wParam), static_cast<LPARAM>(lParam));
            }
            return 0;
        }

        case WM_CHAR:
        {
            const UINT16 ch = static_cast<UINT16>(wParam & 0xFFFFu);
            if (ch >= 0xD800u && ch <= 0xDBFFu)
            {
                m_highSurrogate = ch;
                return 0;
            }
            UINT32 codepoint = ch;
            if (ch >= 0xDC00u && ch <= 0xDFFFu)
            {
                if (m_highSurrogate != 0u)
                {
                    codepoint = 0x10000u +
                        ((static_cast<UINT32>(m_highSurrogate - 0xD800u) << 10u) |
                          static_cast<UINT32>(ch - 0xDC00u));
                    m_highSurrogate = 0u;
                }
                else
                {
                    return 0;
                }
            }
            else
            {
                m_highSurrogate = 0u;
            }

            if (m_onChar)
            {
                m_onChar(codepoint);
            }
            return 0;
        }

        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_XBUTTONDOWN:
        {
            MouseButton btn = MouseButton::Left;
            if (msg == WM_RBUTTONDOWN) btn = MouseButton::Right;
            else if (msg == WM_MBUTTONDOWN) btn = MouseButton::Middle;
            else if (msg == WM_XBUTTONDOWN)
            {
                btn = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? MouseButton::Button4 : MouseButton::Button5;
            }

            if (m_onMouseButton)
            {
                m_onMouseButton(btn, true, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            }
            return (msg == WM_XBUTTONDOWN) ? TRUE : 0;
        }

        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        case WM_XBUTTONUP:
        {
            MouseButton btn = MouseButton::Left;
            if (msg == WM_RBUTTONUP) btn = MouseButton::Right;
            else if (msg == WM_MBUTTONUP) btn = MouseButton::Middle;
            else if (msg == WM_XBUTTONUP)
            {
                btn = (GET_XBUTTON_WPARAM(wParam) == XBUTTON1) ? MouseButton::Button4 : MouseButton::Button5;
            }

            if (m_onMouseButton)
            {
                m_onMouseButton(btn, false, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            }
            return (msg == WM_XBUTTONUP) ? TRUE : 0;
        }

        case WM_MOUSEMOVE:
        {
            if (m_onMouseMove)
            {
                m_onMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            }
            return 0;
        }

        case WM_MOUSEWHEEL:
        {
            if (m_onMouseWheel)
            {
                const FLOAT32 delta = static_cast<FLOAT32>(GET_WHEEL_DELTA_WPARAM(wParam)) / static_cast<FLOAT32>(WHEEL_DELTA);
                POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
                ScreenToClient(hwnd, &pt);
                m_onMouseWheel(delta, pt.x, pt.y);
            }
            return 0;
        }

        case WM_ERASEBKGND:
            return 1;

        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            (void)hdc;
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_CLOSE:
        {
            if (m_onClose)
            {
                if (!m_onClose())
                {
                    return 0;
                }
            }
            DestroyWindow(hwnd);
            return 0;
        }

        case WM_DESTROY:
            // Mark only this window instance as closed without broadcasting WM_QUIT to sibling windows.
            m_isRunning = false;
            return 0;

        case WM_NCDESTROY:
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            m_hwnd = nullptr;
            ReleaseClassRegistration();
            return 0;
        }

        return DefWindowProcW(hwnd, msg, static_cast<WPARAM>(wParam), static_cast<LPARAM>(lParam));
    }

    void Win32Window::ApplyImmersiveDarkMode()
    {
        if (!m_hwnd) return;

        BOOL darkMode = TRUE;
        DwmSetWindowAttribute(m_hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &darkMode, sizeof(darkMode));
        DwmSetWindowAttribute(m_hwnd, 19 /* DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 */, &darkMode, sizeof(darkMode));
    }

    BOOL Win32Window::Create(const WindowDesc& desc)
    {
        if (m_hwnd)
        {
            Close();
        }

        m_desc          = desc;
        m_width         = desc.width;
        m_height        = desc.height;
        m_highSurrogate = 0;

        {
            std::lock_guard<std::mutex> lock(s_windowClassMutex);
            if (s_windowClassAtom == 0)
            {
                if (!s_backgroundBrush)
                {
                    s_backgroundBrush = CreateSolidBrush(RGB(13, 15, 18));
                }

                WNDCLASSEXW wc = {};
                wc.cbSize        = sizeof(WNDCLASSEXW);
                wc.style         = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
                wc.lpfnWndProc   = reinterpret_cast<WNDPROC>(StaticWindowProc);
                wc.hInstance     = m_hInstance;
                wc.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
                wc.hbrBackground = s_backgroundBrush;
                wc.lpszClassName = kWindowClassName;

                s_windowClassAtom = RegisterClassExW(&wc);
                if (s_windowClassAtom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
                {
                    if (s_backgroundBrush)
                    {
                        DeleteObject(s_backgroundBrush);
                        s_backgroundBrush = nullptr;
                    }
                    return false;
                }
            }
            ++s_liveWindowCount;
            m_countedInClass = true;
        }

        DWORD style = WS_OVERLAPPEDWINDOW;
        if (!desc.resizable)
        {
            style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
        }

        RECT rect = { 0, 0, static_cast<LONG>(desc.width), static_cast<LONG>(desc.height) };
        AdjustClientRectForDpi(rect, style, nullptr);

        INT32 posX = CW_USEDEFAULT;
        INT32 posY = CW_USEDEFAULT;
        const INT32 winWidth  = rect.right - rect.left;
        const INT32 winHeight = rect.bottom - rect.top;

        if (desc.centered)
        {
            const INT32 screenW = GetSystemMetrics(SM_CXSCREEN);
            const INT32 screenH = GetSystemMetrics(SM_CYSCREEN);
            posX = (screenW - winWidth) / 2;
            posY = (screenH - winHeight) / 2;
            if (posX < 0) posX = CW_USEDEFAULT;
            if (posY < 0) posY = CW_USEDEFAULT;
        }

        m_hwnd = CreateWindowExW(
            0,
            kWindowClassName,
            desc.title.c_str(),
            style,
            posX, posY,
            winWidth, winHeight,
            nullptr,
            nullptr,
            m_hInstance,
            this
        );

        if (!m_hwnd)
        {
            ReleaseClassRegistration();
            return false;
        }

        if (desc.enableDarkMode)
        {
            ApplyImmersiveDarkMode();
        }

        if (desc.fullscreen)
        {
            SetFullscreen(true);
        }

        if (desc.visible)
        {
            ShowWindow(m_hwnd, SW_SHOW);
            UpdateWindow(m_hwnd);
        }

        m_isRunning = true;
        m_isFocused = desc.visible;
        return true;
    }

    BOOL Win32Window::Create(UINT32 width, UINT32 height, const wchar_t* title)
    {
        WindowDesc desc{};
        desc.width  = width;
        desc.height = height;
        if (title)
        {
            desc.title = title;
        }
        return Create(desc);
    }

    void Win32Window::PollMessages()
    {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                m_isRunning = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    void Win32Window::Close()
    {
        if (m_hwnd)
        {
            HWND hwndToDestroy = m_hwnd;
            m_hwnd = nullptr;
            DestroyWindow(hwndToDestroy);
        }
        ReleaseClassRegistration();
        m_isRunning = false;
    }

    void Win32Window::SetTitle(const std::wstring& title)
    {
        m_desc.title = title;
        if (m_hwnd)
        {
            SetWindowTextW(m_hwnd, title.c_str());
        }
    }

    void Win32Window::SetTitle(const std::string& title)
    {
        if (title.empty())
        {
            SetTitle(std::wstring());
            return;
        }
        const int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, title.c_str(), static_cast<int>(title.size()), nullptr, 0);
        if (sizeNeeded <= 0)
        {
            return;
        }
        std::wstring wstr(static_cast<size_t>(sizeNeeded), 0);
        MultiByteToWideChar(CP_UTF8, 0, title.c_str(), static_cast<int>(title.size()), &wstr[0], sizeNeeded);
        SetTitle(wstr);
    }

    void Win32Window::SetSize(UINT32 width, UINT32 height)
    {
        m_width  = width;
        m_height = height;
        if (m_hwnd)
        {
            RECT rect = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
            const DWORD style = static_cast<DWORD>(GetWindowLongPtrW(m_hwnd, GWL_STYLE));
            AdjustClientRectForDpi(rect, style, m_hwnd);
            SetWindowPos(m_hwnd, nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }

    void Win32Window::SetPosition(INT32 x, INT32 y)
    {
        if (m_hwnd)
        {
            SetWindowPos(m_hwnd, nullptr, x, y, 0, 0,
                SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }

    void Win32Window::SetFullscreen(BOOL fullscreen)
    {
        if (m_isFullscreen == fullscreen || !m_hwnd)
        {
            return;
        }

        m_isFullscreen = fullscreen;

        if (fullscreen)
        {
            RECT rc;
            GetWindowRect(m_hwnd, &rc);
            m_savedX = rc.left;
            m_savedY = rc.top;
            m_savedW = rc.right - rc.left;
            m_savedH = rc.bottom - rc.top;

            SetWindowLongPtrW(m_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
            HMONITOR hMon = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi = { sizeof(mi) };
            if (GetMonitorInfoW(hMon, &mi))
            {
                SetWindowPos(m_hwnd, HWND_TOP,
                    mi.rcMonitor.left, mi.rcMonitor.top,
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            }
        }
        else
        {
            DWORD style = WS_OVERLAPPEDWINDOW;
            if (!m_desc.resizable)
            {
                style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
            }
            SetWindowLongPtrW(m_hwnd, GWL_STYLE, style);
            SetWindowPos(m_hwnd, nullptr,
                m_savedX, m_savedY, m_savedW, m_savedH,
                SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        }
    }

    void Win32Window::SetVisible(BOOL visible)
    {
        m_desc.visible = visible;
        if (m_hwnd)
        {
            ShowWindow(m_hwnd, visible ? SW_SHOW : SW_HIDE);
        }
    }

    NativeRenderTarget Win32Window::GetNativeRenderTarget() const
    {
        NativeRenderTarget target{};
        target.surface = reinterpret_cast<void*>(m_hwnd);
        target.context = reinterpret_cast<void*>(m_hInstance);
        return target;
    }
}
