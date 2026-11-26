#pragma once

#include "compute/compute_kernel.h"
#include "compute/compute_kernel_desc.h"
#include <vector>

namespace Northstar
{
    class CoreComputeManager
    {
    public:
        CoreComputeManager() = default;
        ~CoreComputeManager();

        ComputeKernel* CreateComputeKernel(const ComputeKernelDesc& desc);
        ComputeKernel* CloneComputeKernel(const ComputeKernel* source);
        void           DestroyComputeKernel(ComputeKernel* kernel);
        BOOL           IsManagedKernel(const ComputeKernel* kernel) const;

        ComputeKernel* CreateInternalComputeKernel(const ComputeKernelDesc& desc);
        void           DestroyInternalComputeKernel(ComputeKernel* kernel);
        BOOL           IsInternalKernel(const ComputeKernel* kernel) const;

        void           OnShaderDestroyed(Shader* destroyedShader);
        void           OnTextureDestroyed(Texture* destroyedTexture);
        void           OnSamplerDestroyed(SamplerState* destroyedSampler);
        void           OnGPUBufferDestroyed(GPUBuffer* destroyedBuffer);

        void           Shutdown();
        const std::string& GetLastError() const { return m_lastError; }

    private:
        std::vector<ComputeKernel*> m_computeKernels;
        std::vector<ComputeKernel*> m_internalKernels;
        std::string                 m_lastError;
    };
}

