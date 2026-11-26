#pragma once

#include "light_types.h"

namespace Northstar
{
    // ==================== LIGHT FLAGS BITWISE OPERATORS ====================
    inline LightFlags operator|(LightFlags a, LightFlags b)
    {
        return static_cast<LightFlags>(static_cast<UINT32>(a) | static_cast<UINT32>(b));
    }

    inline LightFlags operator&(LightFlags a, LightFlags b)
    {
        return static_cast<LightFlags>(static_cast<UINT32>(a) & static_cast<UINT32>(b));
    }

    inline LightFlags& operator|=(LightFlags& a, LightFlags b)
    {
        return a = a | b;
    }

    inline LightFlags& operator&=(LightFlags& a, LightFlags b)
    {
        return a = a & b;
    }

    inline LightFlags operator~(LightFlags a)
    {
        return static_cast<LightFlags>(~static_cast<UINT32>(a));
    }
}
