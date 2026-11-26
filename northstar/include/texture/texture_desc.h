#pragma once

#include "texture_types.h"
#include "../types/northstar_types.h"

namespace Northstar
{
    // ==================== TEXTURE DESC ====================
    struct TextureDesc
    {
        // Source (MEMORY or EMPTY)
        TextureDataSource dataSource           = TextureDataSource::EMPTY;

        // Data Payloads
        const BYTE*    initialData             = nullptr;
        SIZE_T         initialDataSize         = 0;

        // Dimensions
        UINT32         width                   = 1;
        UINT32         height                  = 1;
        UINT32         depth                   = 1;           // for 3D textures
        UINT32         mipLevels               = 1;           // 0 means full mip chain (auto-generated if Mip 0 initialData is provided)
        UINT32         arrayLayers             = 1;           // for array / cubemaparray (numCubes)

        // Format & Type
        TextureFormat  format                  = TextureFormat::RGBA8_UNORM;
        TextureType    type                    = TextureType::TEXTURE_2D;
        TextureUsage   usage                   = TextureUsage::SAMPLED;

        // Opt-in cross-process / cross-device NT shared handle export (RENDER_TARGET only).
        // Disabled by default so standard render targets retain full hardware Delta Color Compression (DCC).
        BOOL           allowSharedHandle       = false;
    };
}
