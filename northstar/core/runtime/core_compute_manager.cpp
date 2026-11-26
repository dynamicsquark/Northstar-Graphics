#include "core_compute_manager.h"
#include <algorithm>

namespace Northstar
{
    CoreComputeManager::~CoreComputeManager()
    {
        Shutdown();
    }

    ComputeKernel* CoreComputeManager::CreateComputeKernel(const ComputeKernelDesc& desc)
    {
        if (!desc.computeShader || !desc.computeShader->IsValid())
        {
            m_lastError = "CreateComputeKernel failed: computeShader is null or invalid.";
            return nullptr;
        }
        if (desc.computeShader->GetStage() != ShaderStage::COMPUTE)
        {
            m_lastError = "CreateComputeKernel failed: shader stage is not ShaderStage::COMPUTE.";
            return nullptr;
        }
        m_lastError.clear();

        ComputeKernel* kernel = new ComputeKernel();
        if (!desc.name.empty())
        {
            kernel->SetName(desc.name);
        }
        kernel->Init(desc.computeShader);
        kernel->SetThreadGroupCount(desc.threadGroupCountX, desc.threadGroupCountY, desc.threadGroupCountZ);
        kernel->SetExecutionScope(desc.executionScope);

        m_computeKernels.push_back(kernel);
        return kernel;
    }

    ComputeKernel* CoreComputeManager::CloneComputeKernel(const ComputeKernel* source)
    {
        if (!source || !IsManagedKernel(source) || IsInternalKernel(source))
        {
            return nullptr;
        }

        ComputeKernel* cloned = new ComputeKernel(*source);
        m_computeKernels.push_back(cloned);
        return cloned;
    }

    void CoreComputeManager::DestroyComputeKernel(ComputeKernel* kernel)
    {
        if (!kernel || IsInternalKernel(kernel)) return;

        auto it = std::find(m_computeKernels.begin(), m_computeKernels.end(), kernel);
        if (it != m_computeKernels.end())
        {
            delete kernel;
            m_computeKernels.erase(it);
        }
    }

    BOOL CoreComputeManager::IsManagedKernel(const ComputeKernel* kernel) const
    {
        if (!kernel) return false;
        return std::find(m_computeKernels.begin(), m_computeKernels.end(), kernel) != m_computeKernels.end();
    }

    ComputeKernel* CoreComputeManager::CreateInternalComputeKernel(const ComputeKernelDesc& desc)
    {
        ComputeKernel* kernel = CreateComputeKernel(desc);
        if (kernel)
        {
            m_internalKernels.push_back(kernel);
        }
        return kernel;
    }

    void CoreComputeManager::DestroyInternalComputeKernel(ComputeKernel* kernel)
    {
        if (!kernel) return;

        auto itInt = std::find(m_internalKernels.begin(), m_internalKernels.end(), kernel);
        if (itInt != m_internalKernels.end())
        {
            m_internalKernels.erase(itInt);
        }

        auto it = std::find(m_computeKernels.begin(), m_computeKernels.end(), kernel);
        if (it != m_computeKernels.end())
        {
            delete kernel;
            m_computeKernels.erase(it);
        }
    }

    BOOL CoreComputeManager::IsInternalKernel(const ComputeKernel* kernel) const
    {
        if (!kernel) return false;
        return std::find(m_internalKernels.begin(), m_internalKernels.end(), kernel) != m_internalKernels.end();
    }

    void CoreComputeManager::OnShaderDestroyed(Shader* destroyedShader)
    {
        if (!destroyedShader) return;
        for (ComputeKernel* kernel : m_computeKernels)
        {
            if (kernel)
            {
                kernel->OnShaderDestroyed(destroyedShader);
            }
        }
    }

    void CoreComputeManager::OnTextureDestroyed(Texture* destroyedTexture)
    {
        if (!destroyedTexture) return;
        for (ComputeKernel* kernel : m_computeKernels)
        {
            if (kernel)
            {
                kernel->OnTextureDestroyed(destroyedTexture);
            }
        }
    }

    void CoreComputeManager::OnSamplerDestroyed(SamplerState* destroyedSampler)
    {
        if (!destroyedSampler) return;
        for (ComputeKernel* kernel : m_computeKernels)
        {
            if (kernel)
            {
                kernel->OnSamplerDestroyed(destroyedSampler);
            }
        }
    }

    void CoreComputeManager::OnGPUBufferDestroyed(GPUBuffer* destroyedBuffer)
    {
        if (!destroyedBuffer) return;
        for (ComputeKernel* kernel : m_computeKernels)
        {
            if (kernel)
            {
                kernel->OnGPUBufferDestroyed(destroyedBuffer);
            }
        }
    }

    void CoreComputeManager::Shutdown()
    {
        for (ComputeKernel* kernel : m_computeKernels)
        {
            delete kernel;
        }
        m_computeKernels.clear();
        m_internalKernels.clear();
    }
}

