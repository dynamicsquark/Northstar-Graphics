#pragma once

#include "../types/northstar_platform_guard.h"
#include "../platform/northstar_backend.h"
#include "../platform/native_render_target.h"
#include "../mesh/mesh.h"
#include "../mesh/mesh_desc.h"
#include "../material/material.h"
#include "../material/material_desc.h"
#include "../compute/compute_kernel.h"
#include "../compute/compute_kernel_desc.h"
#include "../texture/texture.h"
#include "../texture/texture_desc.h"
#include "../sampler/sampler.h"
#include "../sampler/sampler_desc.h"
#include "../lod_group/lod_group.h"
#include "../lod_group/lod_group_desc.h"
#include "../stats/northstar_stats.h"
#include "../camera/camera.h"
#include "../camera/camera_desc.h"
#include "../render_object/render_object.h"
#include "../render_object/render_object_desc.h"
#include "../light/light.h"
#include "../light/light_desc.h"
#include "../scene/scene.h"
#include "../scene/scene_desc.h"
#include "../shader/shader.h"
#include "../shader/shader_desc.h"
#include "../shader_set/shader_set.h"
#include "../shader_set/shader_set_desc.h"
#include "../buffer/gpu_buffer_types.h"
#include "../buffer/gpu_buffer_desc.h"
#include "../buffer/gpu_buffer.h"
#include "../nvg/nvg_types.h"

namespace Northstar
{
    /**
     * @brief Binary ABI contract version shared across NorthstarCore, hardware backends, and host applications.
     */
    inline constexpr UINT32 NORTHSTAR_ENGINE_API_VERSION = 7u;

    /**
     * @brief Primary host-facing interface to the Northstar Graphics runtime.
     *
     * Obtain an instance through @ref RuntimeSession (recommended) or via the
     * dynamically exported @c CreateNorthstarEngine / @c DestroyNorthstarEngine entry points.
     */
    class NorthstarEngine
    {
    public:
        static constexpr UINT32 kInterfaceVersion = NORTHSTAR_ENGINE_API_VERSION;

        virtual ~NorthstarEngine() = default;

        // ==================== Lifecycle & Diagnostics ====================
        virtual BOOL                           Initialize(NorthstarBackend* backend, const NativeRenderTarget& renderTarget) = 0;
        virtual void                           Shutdown()                                                                    = 0;
        [[nodiscard]] virtual BOOL             IsInitialized() const                                                         = 0;
        [[nodiscard]] virtual const std::string& GetLastError() const                                                        = 0;
        [[nodiscard]] virtual const std::string& GetRuntimeDirectory() const                                                 = 0;

        // ==================== Frame Execution & Presentation ====================
        virtual void                           BeginFrame()                                                                  = 0;
        virtual void                           RenderFrame()                                                                 = 0;
        virtual void                           EndFrame()                                                                    = 0;

        virtual void                           SetVSync(BOOL enabled)                                                        = 0;
        [[nodiscard]] virtual BOOL             IsVSyncEnabled() const                                                        = 0;
        virtual void                           OnResize(UINT32 width, UINT32 height)                                         = 0;

        virtual void                           WaitForGPU()                                                                  = 0;
        virtual BOOL                           EnqueueExternalFenceWait(void* sharedFenceNtHandle, UINT64 fenceValue)        = 0;

        // ==================== Shaders & Pipeline Sets ====================
        [[nodiscard]] virtual Shader*          CreateShader(const ShaderDesc& desc)                                          = 0;
        virtual void                           DestroyShader(Shader* shader)                                                 = 0;

        [[nodiscard]] virtual ShaderSet*       CreateShaderSet(const ShaderSetDesc& desc)                                    = 0;
        virtual void                           DestroyShaderSet(ShaderSet* shaderSet)                                        = 0;

        // ==================== Geometry & Virtual Geometry (NVG) ====================
        [[nodiscard]] virtual Mesh*            CreateMesh(const MeshDesc& desc)                                              = 0;
        virtual void                           DestroyMesh(Mesh* mesh)                                                       = 0;
        virtual BOOL                           SerializeVirtualMesh(const Mesh* mesh, std::vector<BYTE>& outBinaryBlob) const = 0;
        virtual BOOL                           SerializeVirtualMeshToBuffer(const Mesh* mesh, BYTE** outBuffer, SIZE_T* outByteSize) const = 0;
        virtual void                           FreeSerializedBuffer(BYTE* buffer) const                                      = 0;

        virtual BOOL                           SetNVGConfig(const NVGConfig& config)                                         = 0;
        [[nodiscard]] virtual const NVGConfig& GetNVGConfig() const                                                          = 0;
        [[nodiscard]] virtual const NVGStats&  GetNVGStats() const                                                           = 0;
        [[nodiscard]] virtual NVGCompatibilityStatus CheckMaterialNVGCompatibility(
            const Material* material,
            PassType        pass      = PassType::Geometry,
            BOOL            isSkinned = false,
            std::string*    outReason = nullptr) const                                                                       = 0;

        // ==================== Materials ====================
        [[nodiscard]] virtual Material*        CreateMaterial(const MaterialDesc& desc = {})                                 = 0;
        [[nodiscard]] virtual Material*        CloneMaterial(const Material* source)                                         = 0;
        virtual void                           DestroyMaterial(Material* material)                                           = 0;

        // ==================== Compute Dispatch ====================
        [[nodiscard]] virtual ComputeKernel*   CreateComputeKernel(const ComputeKernelDesc& desc)                            = 0;
        [[nodiscard]] virtual ComputeKernel*   CloneComputeKernel(const ComputeKernel* source)                               = 0;
        virtual void                           DestroyComputeKernel(ComputeKernel* kernel)                                   = 0;
        virtual BOOL                           BuildComputeDispatchCommand(const ComputeKernel* kernel, ComputeDispatchCommand& outCmd) = 0;
        virtual BOOL                           ExecuteComputeKernel(const ComputeKernel* kernel)                             = 0;
        virtual BOOL                           ExecuteComputeKernels(const std::vector<const ComputeKernel*>& kernels)       = 0;

        // ==================== Textures & Samplers ====================
        [[nodiscard]] virtual Texture*         CreateTexture(const TextureDesc& desc)                                        = 0;
        virtual BOOL                           ResizeTexture(Texture* texture, UINT32 newWidth, UINT32 newHeight)            = 0;
        virtual BOOL                           UpdateTextureSubresource(
            Texture*    texture,
            UINT32      mipSlice,
            UINT32      arraySlice,
            const void* srcData,
            SIZE_T      srcByteSize)                                                                                         = 0;
        virtual BOOL                           GenerateTextureMips(Texture* texture)                                         = 0;
        virtual BOOL                           ReadbackTextureSync(const Texture* texture, void* outData, UINT32 maxByteSize) const = 0;
        virtual BOOL                           ReadbackTextureSubresourceSync(
            const Texture* texture,
            UINT32         mipSlice,
            UINT32         arraySlice,
            void*          outData,
            UINT32         maxByteSize) const                                                                                = 0;
        virtual void                           DestroyTexture(Texture* texture)                                              = 0;
        [[nodiscard]] virtual void*            GetTextureGPUDescriptor(Texture* texture) const                               = 0;
        [[nodiscard]] virtual void*            GetTextureSharedHandle(Texture* texture) const                                = 0;
        [[nodiscard]] virtual SharedHandleType GetTextureSharedHandleType(Texture* texture) const                            = 0;

        [[nodiscard]] virtual SamplerState*    CreateSamplerState(const SamplerDesc& desc = {})                              = 0;
        virtual void                           DestroySamplerState(SamplerState* sampler)                                    = 0;

        // ==================== Persistent GPU Buffers ====================
        [[nodiscard]] virtual GPUBuffer*       CreateGPUBuffer(const GPUBufferDesc& desc)                                    = 0;
        virtual BOOL                           ResizeGPUBuffer(GPUBuffer* buffer, UINT32 newElementCount, const void* initialData = nullptr) = 0;
        virtual BOOL                           UpdateGPUBuffer(GPUBuffer* buffer, const void* data, UINT32 byteSize, UINT32 dstOffset = 0)   = 0;
        virtual BOOL                           RequestGPUBufferReadback(GPUBuffer* buffer, UINT32 srcOffset = 0, UINT32 byteSize = 0)        = 0;
        virtual BOOL                           TryGetGPUBufferReadbackData(GPUBuffer* buffer, void* outData, UINT32 byteSize)                = 0;
        virtual BOOL                           ReadbackGPUBufferSync(GPUBuffer* buffer, void* outData, UINT32 byteSize)                      = 0;
        virtual void                           DestroyGPUBuffer(GPUBuffer* buffer)                                           = 0;

        // ==================== Scene Graph, Cameras & Lighting ====================
        [[nodiscard]] virtual Camera*          CreateCamera(const CameraDesc& desc = {})                                     = 0;
        virtual void                           DestroyCamera(Camera* camera)                                                 = 0;

        [[nodiscard]] virtual RenderObject*    CreateRenderObject(const RenderObjectDesc& desc = {})                         = 0;
        virtual void                           DestroyRenderObject(RenderObject* renderObject)                               = 0;

        [[nodiscard]] virtual LODGroup*        CreateLODGroup(const LODGroupDesc& desc = {})                                 = 0;
        virtual void                           DestroyLODGroup(LODGroup* lodGroup)                                           = 0;

        [[nodiscard]] virtual Light*           CreateLight(const LightDesc& desc = {})                                       = 0;
        virtual void                           DestroyLight(Light* light)                                                    = 0;

        [[nodiscard]] virtual Scene*           CreateScene(const SceneDesc& desc = {})                                       = 0;
        virtual void                           ClearScene(Scene* scene)                                                      = 0;
        virtual void                           DestroyScene(Scene* scene)                                                    = 0;
        virtual void                           SetActiveScene(Scene* scene)                                                  = 0;
        [[nodiscard]] virtual Scene*           GetActiveScene() const                                                        = 0;
        virtual void                           BakeStaticBatching(Scene* scene = nullptr)                                    = 0;

        // ==================== Telemetry & Native Interop ====================
        [[nodiscard]] virtual const RenderStats& GetStats() const                                                            = 0;
        virtual void                           ResetStats()                                                                  = 0;

        [[nodiscard]] virtual NativeContext    GetNativeContext() const                                                       = 0;
        virtual void                           SetPrePresentCallback(PrePresentCallback cb)                                  = 0;
        virtual void                           SetPostFrameExecutionCallback(PostFrameExecutionCallback cb, BOOL waitForGPU = true) = 0;
    };

    // Exported dynamic module C ABI function signatures
    using GetNorthstarAPIVersionFn  = UINT32 (*)();
    using CreateNorthstarEngineFn   = NorthstarEngine* (*)();
    using DestroyNorthstarEngineFn  = void (*)(NorthstarEngine*);
    using CreateNorthstarBackendFn  = NorthstarBackend* (*)();
    using DestroyNorthstarBackendFn = void (*)(NorthstarBackend*);
}
