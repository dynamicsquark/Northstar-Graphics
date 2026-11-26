#pragma once

#include "../../include/types/northstar_types.h"
#include "../../include/platform/northstar_backend.h"
#include "../../include/shader/shader.h"
#include "../../include/compute/compute_kernel.h"
#include "core_compute_manager.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Northstar
{
    // Distinguishes mandatory core startup resources from optional feature-dependent resources.
    enum class InternalResourceRequirement : UINT8
    {
        MANDATORY         = 0, // Must exist and load cleanly during NorthstarCore::Initialize()
        FEATURE_DEPENDENT = 1  // Loaded on-demand when the owning engine feature/subsystem is activated
    };

    // Built-in engine-internal compute kernel identifiers.
    // Future subsystems (NVS page request, page allocation, caster culling, indirect command gen)
    // register into the same catalog without creating a second compute pipeline.
    enum class InternalComputeKernelId : UINT32
    {
        BUFFER_CLEAR_UINT         = 0,
        DIRTY_DEPTH_SLICE_BUILD   = 1,
        GPU_SCENE_DIAGNOSTIC      = 2,
        NVS_PAGE_RESIDENCY        = 3,
        NVS_SCREEN_SPACE_FEEDBACK = 4,
        NVS_PAGE_PROJECTION       = 5,
        NVS_CASTER_CULL           = 6,
        NVS_INDIRECT_DRAW_BUILD   = 7,
        NVG_HZB_BUILD             = 8,
        NVG_RESIDENCY_MASK        = 9,
        NVG_INSTANCE_CULL         = 10,
        NVG_HIERARCHY_CULL        = 11,
        NVG_COMPACTION_DECODE     = 12,
        NVG_INDIRECT_DRAW_BUILD   = 13,
        NVG_SHADOW_CULL           = 14,
        REFLECTION_MATERIALIZE    = 15,
        COUNT
    };

    // Built-in engine-internal non-compute (graphics/depth) shader identifiers.
    // Strictly limited to engine-owned physical slice clear helpers; user Depth Prepass
    // and NVS Shadow Rasterization execute user-owned ShaderSets exclusively.
    enum class InternalShaderId : UINT32
    {
        LAYERED_DEPTH_CLEAR_VS = 0,
        LAYERED_DEPTH_PASS_GS  = 1,
        COUNT
    };

    struct InternalComputeShaderSpec
    {
        std::string                 idName;                    // Unique internal ID, e.g. "Runtime.BufferClearUInt"
        std::string                 relativeCategory;          // Subfolder under NorthstarRuntime, e.g. "Compute"
        std::string                 baseFileName;              // PascalCase base filename without _<api>.nsc, e.g. "RuntimeBufferClear"
        InternalResourceRequirement requirement      = InternalResourceRequirement::MANDATORY;
        UINT32                      expectedThreadsX = 0;      // 0 = any valid (>0) numthreads; >0 = exact match required
        UINT32                      expectedThreadsY = 0;
        UINT32                      expectedThreadsZ = 0;
        std::vector<std::string>    requiredConstantBuffers;   // Required constant buffer names in ShaderReflection
        std::vector<std::string>    requiredSRVTextures;       // Required SRV texture names in ShaderReflection
        std::vector<std::string>    requiredSRVStructuredBufs; // Required SRV structured buffer names in ShaderReflection
        std::vector<std::string>    requiredUAVStructuredBufs; // Required UAV structured buffer names in ShaderReflection
    };

    struct InternalGraphicsShaderSpec
    {
        std::string                 idName;                    // Unique internal ID, e.g. "Runtime.LayeredDepthClearVS"
        std::string                 relativeCategory;          // Subfolder under NorthstarRuntime, e.g. "Depth"
        std::string                 baseFileName;              // PascalCase base filename without _<api>.nsc
        ShaderStage                 expectedStage    = ShaderStage::VERTEX;
        InternalResourceRequirement requirement      = InternalResourceRequirement::MANDATORY;
        std::vector<std::string>    requiredConstantBuffers;
        std::vector<std::string>    requiredSRVStructuredBufs;
    };

    class InternalResourceManager
    {
    public:
        InternalResourceManager();
        ~InternalResourceManager();

        // Resolves NorthstarCore's own module directory (never host executable dir or CWD)
        // and initializes all MANDATORY internal resources.
        BOOL Initialize(NorthstarBackend* backend, CoreComputeManager* computeManager);

        // Releases all internal ComputeKernels first, then internal Shaders (and backend handles).
        void Shutdown();

        BOOL               IsInitialized() const { return m_initialized; }
        const std::string& GetLastError() const { return m_lastError; }
        const std::string& GetRuntimeDirectoryUTF8() const { return m_runtimeDirUTF8; }
        const std::filesystem::path& GetRuntimeDirectoryPath() const { return m_runtimeDirPath; }

        // Registers an internal compute shader specification (MANDATORY or FEATURE_DEPENDENT).
        // Must be called before Initialize() for MANDATORY specs, or at any time for FEATURE_DEPENDENT specs.
        BOOL RegisterComputeShaderSpec(const InternalComputeShaderSpec& spec);
        BOOL RegisterGraphicsShaderSpec(const InternalGraphicsShaderSpec& spec);

        // Resets the specification catalog back to the default engine-internal specs.
        void ResetDefaultSpecs();

        // Engine-internal access to loaded ComputeKernels (ownership remains strictly inside NorthstarCore).
        ComputeKernel* GetComputeKernel(InternalComputeKernelId id) const;
        ComputeKernel* GetComputeKernelByName(const std::string& idName) const;
        Shader*        GetShader(InternalComputeKernelId id) const;
        Shader*        GetShaderByName(const std::string& idName) const;
        Shader*        GetGraphicsShader(InternalShaderId id) const;
        Shader*        GetGraphicsShaderByName(const std::string& idName) const;

        // Loads a FEATURE_DEPENDENT compute kernel on demand if not already loaded.
        ComputeKernel* AcquireFeatureKernel(const std::string& idName);
        Shader*        AcquireFeatureGraphicsShader(const std::string& idName);

        // Static helper that resolves the directory containing the NorthstarCore shared module itself.
        static BOOL ResolveCoreModuleDirectory(std::filesystem::path& outModuleDir, std::string& outError);

    private:
        struct LoadedInternalComputeEntry
        {
            InternalComputeShaderSpec spec;
            Shader*                   shader = nullptr;
            ComputeKernel*            kernel = nullptr;
            BOOL                      loaded = false;
        };

        struct LoadedInternalGraphicsEntry
        {
            InternalGraphicsShaderSpec spec;
            Shader*                    shader = nullptr;
            BOOL                       loaded = false;
        };

        BOOL BuildSafeShaderFilePath(const std::string& idName,
                                     const std::string& relativeCategory,
                                     const std::string& baseFileName,
                                     TargetAPI targetApi,
                                     std::filesystem::path& outFilePath,
                                     std::string& outError) const;

        BOOL LoadEntry(LoadedInternalComputeEntry& entry, std::string& outError);
        BOOL LoadGraphicsEntry(LoadedInternalGraphicsEntry& entry, std::string& outError);
        void ReleaseLoadedResources();

        NorthstarBackend*                        m_backend        = nullptr;
        CoreComputeManager*                      m_computeManager = nullptr;
        BOOL                                     m_initialized    = false;
        std::filesystem::path                    m_moduleDirPath;
        std::filesystem::path                    m_runtimeDirPath;
        std::string                              m_runtimeDirUTF8;
        std::string                              m_lastError;
        std::vector<LoadedInternalComputeEntry>  m_entries;
        std::vector<LoadedInternalGraphicsEntry> m_graphicsEntries;
    };
}

