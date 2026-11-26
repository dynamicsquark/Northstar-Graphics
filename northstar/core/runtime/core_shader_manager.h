#pragma once

#include "shader/shader.h"
#include "shader/shader_desc.h"
#include "shader_set/shader_set.h"
#include "shader_set/shader_set_desc.h"
#include "platform/northstar_backend.h"
#include <string>
#include <vector>

namespace Northstar
{
    class CoreShaderManager
    {
    public:
        CoreShaderManager() = default;
        ~CoreShaderManager();

        const std::string& GetLastError() const { return m_lastError; }

        Shader*    CreateShader(const ShaderDesc& desc);
        void       DestroyShader(Shader* shader);
        BOOL       IsManagedShader(const Shader* shader) const;

        ShaderSet* CreateShaderSet(const ShaderSetDesc& desc);
        void       DestroyShaderSet(ShaderSet* shaderSet);
        BOOL       IsManagedShaderSet(const ShaderSet* shaderSet) const;

        void       RegisterShadersWithBackend(NorthstarBackend* backend);
        void       UnregisterFromBackend(BOOL destroyNativeHandles = false);

        void       Shutdown();

    private:
        BOOL       RegisterSingleShaderWithBackend(Shader& shader);

        std::vector<Shader*>    m_shaders;
        std::vector<ShaderSet*> m_shaderSets;
        NorthstarBackend*       m_backend = nullptr;
        std::string             m_lastError;
    };
}
