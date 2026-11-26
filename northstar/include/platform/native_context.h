#pragma once

#include "../types/northstar_types.h"
#include "native_render_target.h"
#include <functional>

namespace Northstar
{
    // Describes what the opaque pointer returned by GetTextureSharedHandle() actually is,
    // so external consumers (editor, tooling) can import it without guessing the backend.
    enum class SharedHandleType : UINT8
    {
        NONE             = 0,  // Texture is not shareable / backend does not support sharing
        WIN32_NT_HANDLE  = 1,  // Windows NT HANDLE (D3D12 / D3D11 shared resource)
        OPAQUE_FD        = 2,  // POSIX file descriptor (Vulkan external memory)
        IOSURFACE        = 3,  // Apple IOSurfaceRef (Metal)
        DMA_BUF_FD       = 4   // Linux DMA-BUF file descriptor
    };

    // Native Context structure exposing low-level GPU device/queue/heap handles
    // for UI overlays (e.g. ImGui) or external tooling integration.
    struct NativeContext
    {
        void*  nativeDevice         = nullptr;
        void*  nativeCommandQueue   = nullptr;
        void*  nativeDescriptorHeap = nullptr;
        UINT32 descriptorSize       = 0;
        UINT32 backBufferCount      = 2;
        UINT32 renderTargetFormat   = 0;
    };

    // Execution metadata delivered to PostFrameExecutionCallback at the completion of ExecuteFrame().
    struct PostFrameExecutionInfo
    {
        void*            colorSharedHandle     = nullptr;                // Cross-process shared handle of colorTexture (nullptr if WINDOW or non-shareable)
        void*            nativeFence           = nullptr;                // Underlying GPU fence object (e.g. ID3D12Fence*)
        void*            sharedFenceHandle     = nullptr;                // Cross-process shared fence handle (e.g. NT HANDLE created with D3D12_FENCE_FLAG_SHARED)
        UINT64           signaledFenceValue    = 0;                      // Monotonic GPU fence value signaled for this ExecuteFrame() submission
        NativeHandle     colorTexture          = 0;                      // Target color texture handle when renderTargetType == RenderTargetType::TEXTURE (0 for WINDOW)
        NativeHandle     depthStencilTexture   = 0;                      // Active depth-stencil texture handle for this pass (0 if none)
        RenderTargetType renderTargetType      = RenderTargetType::WINDOW;
        SharedHandleType colorSharedHandleType = SharedHandleType::NONE;
        SharedHandleType sharedFenceHandleType = SharedHandleType::NONE;
        BOOL             gpuCompletedOnEntry   = false;                  // True when GPU execution of this submission is guaranteed complete on callback entry
    };

    // Callback invoked right before presenting the backbuffer (e.g. for ImGui overlay rendering)
    using PrePresentCallback = std::function<void(void* nativeCommandList)>;

    // Callback invoked at the end of ExecuteFrame() (after GPU completion when waitForGPU == true,
    // or immediately after queue submission + fence signal when waitForGPU == false).
    using PostFrameExecutionCallback = std::function<void(const PostFrameExecutionInfo& info)>;
}
