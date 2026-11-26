#pragma once

#include "../types/northstar_types.h"

namespace Northstar
{
    enum class RenderTargetType : UINT8
    {
        WINDOW  = 0,
        TEXTURE = 1,

        Window  = WINDOW,
        Texture = TEXTURE
    };

    struct NativeRenderTarget
    {
        RenderTargetType type                = RenderTargetType::Window;
        NativeHandle     colorTexture        = 0;       // Texture handle when type == RenderTargetType::Texture
        NativeHandle     depthStencilTexture = 0;       // Optional dedicated DSV texture handle
        void*            surface             = nullptr; // Platform window handle (e.g. HWND)
        void*            context             = nullptr; // Optional platform display/instance handle

        [[nodiscard]] static constexpr NativeRenderTarget FromWindow(
            void* surfaceHandle,
            void* appContext = nullptr) noexcept
        {
            NativeRenderTarget rt{};
            rt.type    = RenderTargetType::Window;
            rt.surface = surfaceHandle;
            rt.context = appContext;
            return rt;
        }

        [[nodiscard]] static constexpr NativeRenderTarget FromTexture(
            NativeHandle colorTex,
            NativeHandle depthStencilTex = 0) noexcept
        {
            NativeRenderTarget rt{};
            rt.type                = RenderTargetType::Texture;
            rt.colorTexture        = colorTex;
            rt.depthStencilTexture = depthStencilTex;
            return rt;
        }
    };
}
