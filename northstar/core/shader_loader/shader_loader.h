#pragma once

#include "shader/shader.h"
#include "shader/shader_desc.h"

namespace Northstar
{
    class ShaderLoader
    {
    public:
        // Loads and populates a Shader object from a ShaderDesc specification.
        // Supports MEMORY (.nsc in-memory buffer) and direct REFLECTION copying.
        static BOOL Load(const ShaderDesc& desc, Shader& outShader);

        // Direct helper to load from a .nsc binary buffer in memory.
        static BOOL LoadFromMemory(const BYTE* data, SIZE_T size, Shader& outShader, const std::string& name = "");

        // Persistent DXBC/DXIL disk cache (%LOCALAPPDATA%\NorthstarGraphics\DXBCCache\*.nsdc).
        // Keeps all filesystem and cache I/O strictly encapsulated within NorthstarCore.
        static BOOL TryPopulateCachedDXBC(Shader& ioShader, UINT32 targetShaderModel = 0x66u);
        static void StoreCachedDXBC(const Shader& shader, const void* dxbcData, SIZE_T dxbcSize, UINT32 targetShaderModel = 0x66u);
    };
}
