#pragma once

#include "northstar_engine.h"
#include <memory>
#include <type_traits>

namespace Northstar
{
    /**
     * @brief Stateless-compatible custom deleter bound to a NorthstarEngine instance.
     */
    template <typename T>
    struct EngineResourceDeleter
    {
        NorthstarEngine* engine = nullptr;

        constexpr EngineResourceDeleter() noexcept = default;
        constexpr explicit EngineResourceDeleter(NorthstarEngine* eng) noexcept : engine(eng) {}

        void operator()(T* resource) const noexcept
        {
            if (!engine || !resource)
            {
                return;
            }

            if constexpr (std::is_same_v<T, Shader>)
                engine->DestroyShader(resource);
            else if constexpr (std::is_same_v<T, ShaderSet>)
                engine->DestroyShaderSet(resource);
            else if constexpr (std::is_same_v<T, Mesh>)
                engine->DestroyMesh(resource);
            else if constexpr (std::is_same_v<T, Material>)
                engine->DestroyMaterial(resource);
            else if constexpr (std::is_same_v<T, ComputeKernel>)
                engine->DestroyComputeKernel(resource);
            else if constexpr (std::is_same_v<T, Texture>)
                engine->DestroyTexture(resource);
            else if constexpr (std::is_same_v<T, SamplerState>)
                engine->DestroySamplerState(resource);
            else if constexpr (std::is_same_v<T, GPUBuffer>)
                engine->DestroyGPUBuffer(resource);
            else if constexpr (std::is_same_v<T, Camera>)
                engine->DestroyCamera(resource);
            else if constexpr (std::is_same_v<T, RenderObject>)
                engine->DestroyRenderObject(resource);
            else if constexpr (std::is_same_v<T, LODGroup>)
                engine->DestroyLODGroup(resource);
            else if constexpr (std::is_same_v<T, Light>)
                engine->DestroyLight(resource);
            else if constexpr (std::is_same_v<T, Scene>)
                engine->DestroyScene(resource);
        }
    };

    template <typename T>
    using UniqueEngineResource = std::unique_ptr<T, EngineResourceDeleter<T>>;

    using UniqueShader        = UniqueEngineResource<Shader>;
    using UniqueShaderSet     = UniqueEngineResource<ShaderSet>;
    using UniqueMesh          = UniqueEngineResource<Mesh>;
    using UniqueMaterial      = UniqueEngineResource<Material>;
    using UniqueComputeKernel = UniqueEngineResource<ComputeKernel>;
    using UniqueTexture       = UniqueEngineResource<Texture>;
    using UniqueSampler       = UniqueEngineResource<SamplerState>;
    using UniqueGPUBuffer     = UniqueEngineResource<GPUBuffer>;
    using UniqueCamera        = UniqueEngineResource<Camera>;
    using UniqueRenderObject  = UniqueEngineResource<RenderObject>;
    using UniqueLODGroup      = UniqueEngineResource<LODGroup>;
    using UniqueLight         = UniqueEngineResource<Light>;
    using UniqueScene         = UniqueEngineResource<Scene>;

    [[nodiscard]] inline UniqueShader MakeUniqueShader(NorthstarEngine& engine, const ShaderDesc& desc)
    {
        return UniqueShader(engine.CreateShader(desc), EngineResourceDeleter<Shader>(&engine));
    }

    [[nodiscard]] inline UniqueShaderSet MakeUniqueShaderSet(NorthstarEngine& engine, const ShaderSetDesc& desc)
    {
        return UniqueShaderSet(engine.CreateShaderSet(desc), EngineResourceDeleter<ShaderSet>(&engine));
    }

    [[nodiscard]] inline UniqueMesh MakeUniqueMesh(NorthstarEngine& engine, const MeshDesc& desc)
    {
        return UniqueMesh(engine.CreateMesh(desc), EngineResourceDeleter<Mesh>(&engine));
    }

    [[nodiscard]] inline UniqueMaterial MakeUniqueMaterial(NorthstarEngine& engine, const MaterialDesc& desc = {})
    {
        return UniqueMaterial(engine.CreateMaterial(desc), EngineResourceDeleter<Material>(&engine));
    }

    [[nodiscard]] inline UniqueComputeKernel MakeUniqueComputeKernel(NorthstarEngine& engine, const ComputeKernelDesc& desc)
    {
        return UniqueComputeKernel(engine.CreateComputeKernel(desc), EngineResourceDeleter<ComputeKernel>(&engine));
    }

    [[nodiscard]] inline UniqueTexture MakeUniqueTexture(NorthstarEngine& engine, const TextureDesc& desc)
    {
        return UniqueTexture(engine.CreateTexture(desc), EngineResourceDeleter<Texture>(&engine));
    }

    [[nodiscard]] inline UniqueSampler MakeUniqueSampler(NorthstarEngine& engine, const SamplerDesc& desc = {})
    {
        return UniqueSampler(engine.CreateSamplerState(desc), EngineResourceDeleter<SamplerState>(&engine));
    }

    [[nodiscard]] inline UniqueGPUBuffer MakeUniqueGPUBuffer(NorthstarEngine& engine, const GPUBufferDesc& desc)
    {
        return UniqueGPUBuffer(engine.CreateGPUBuffer(desc), EngineResourceDeleter<GPUBuffer>(&engine));
    }

    [[nodiscard]] inline UniqueCamera MakeUniqueCamera(NorthstarEngine& engine, const CameraDesc& desc = {})
    {
        return UniqueCamera(engine.CreateCamera(desc), EngineResourceDeleter<Camera>(&engine));
    }

    [[nodiscard]] inline UniqueRenderObject MakeUniqueRenderObject(NorthstarEngine& engine, const RenderObjectDesc& desc = {})
    {
        return UniqueRenderObject(engine.CreateRenderObject(desc), EngineResourceDeleter<RenderObject>(&engine));
    }

    [[nodiscard]] inline UniqueLODGroup MakeUniqueLODGroup(NorthstarEngine& engine, const LODGroupDesc& desc = {})
    {
        return UniqueLODGroup(engine.CreateLODGroup(desc), EngineResourceDeleter<LODGroup>(&engine));
    }

    [[nodiscard]] inline UniqueLight MakeUniqueLight(NorthstarEngine& engine, const LightDesc& desc = {})
    {
        return UniqueLight(engine.CreateLight(desc), EngineResourceDeleter<Light>(&engine));
    }

    [[nodiscard]] inline UniqueScene MakeUniqueScene(NorthstarEngine& engine, const SceneDesc& desc = {})
    {
        return UniqueScene(engine.CreateScene(desc), EngineResourceDeleter<Scene>(&engine));
    }
}
