#pragma once

#include "render_object_types.h"

namespace Northstar
{
    // ==================== RENDER OBJECT BITWISE OPERATORS ====================
    inline RenderObjectFlags operator|(RenderObjectFlags a, RenderObjectFlags b)
    {
        return static_cast<RenderObjectFlags>(static_cast<UINT32>(a) | static_cast<UINT32>(b));
    }

    inline RenderObjectFlags operator&(RenderObjectFlags a, RenderObjectFlags b)
    {
        return static_cast<RenderObjectFlags>(static_cast<UINT32>(a) & static_cast<UINT32>(b));
    }

    inline RenderObjectFlags& operator|=(RenderObjectFlags& a, RenderObjectFlags b)
    {
        return a = a | b;
    }

    inline RenderObjectFlags& operator&=(RenderObjectFlags& a, RenderObjectFlags b)
    {
        return a = a & b;
    }

    inline RenderObjectFlags operator~(RenderObjectFlags a)
    {
        return static_cast<RenderObjectFlags>(~static_cast<UINT32>(a));
    }
}