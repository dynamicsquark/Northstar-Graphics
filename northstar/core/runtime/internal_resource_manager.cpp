#include "internal_resource_manager.h"
#include "../shader_loader/shader_loader.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <fstream>
#include <system_error>

namespace Northstar
{
    namespace
    {
        std::string PathToUTF8String(const std::filesystem::path& p)
        {
            const auto u8 = p.u8string();
            return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
        }

        BOOL IsSafeRelativeSegment(const std::string& segment)
        {
            if (segment.empty())
                return false;
            if (segment.find("..") != std::string::npos ||
                segment.find(':') != std::string::npos ||
                segment.find('/') != std::string::npos ||
                segment.find('\\') != std::string::npos)
            {
                return false;
            }
            return true;
        }

        const char* TargetAPIToSuffix(TargetAPI api)
        {
            switch (api)
            {
                case TargetAPI::HLSL11: return "hlsl11";
                case TargetAPI::HLSL12: return "hlsl12";
                default:                return nullptr;
            }
        }

        BOOL ReflectionHasConstantBuffer(const ShaderReflection& refl, const std::string& name)
        {
            if (refl.FindBuffer(name) != nullptr)
                return true;
            const std::string suffix = "__" + name;
            for (const auto& b : refl.buffers)
            {
                if (b.name == name)
                    return true;
                if (b.name.size() > suffix.size() &&
                    b.name.compare(b.name.size() - suffix.size(), suffix.size(), suffix) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        BOOL ReflectionHasSRVStructuredBuffer(const ShaderReflection& refl, const std::string& name)
        {
            if (refl.FindStructuredBuffer(name) != nullptr)
                return true;
            const std::string suffix = "__" + name;
            for (const auto& s : refl.structuredBuffers)
            {
                if (s.name == name)
                    return true;
                if (s.name.size() > suffix.size() &&
                    s.name.compare(s.name.size() - suffix.size(), suffix.size(), suffix) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        BOOL ReflectionHasUAVStructuredBuffer(const ShaderReflection& refl, const std::string& name)
        {
            if (refl.FindUAV(name) != nullptr)
                return true;
            const std::string suffix = "__" + name;
            for (const auto& u : refl.uavBuffers)
            {
                if (u.name == name)
                    return true;
                if (u.name.size() > suffix.size() &&
                    u.name.compare(u.name.size() - suffix.size(), suffix.size(), suffix) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        BOOL ReflectionHasSRVTexture(const ShaderReflection& refl, const std::string& name)
        {
            const std::string suffix = "__" + name;
            const std::string dotSuffix = "." + name;
            for (const auto& t : refl.textures)
            {
                if (t.name == name || t.groupQualifiedName == name)
                    return true;
                if (t.name.size() > suffix.size() &&
                    t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0)
                {
                    return true;
                }
                if (t.groupQualifiedName.size() > dotSuffix.size() &&
                    t.groupQualifiedName.compare(t.groupQualifiedName.size() - dotSuffix.size(), dotSuffix.size(), dotSuffix) == 0)
                {
                    return true;
                }
            }
            return false;
        }

        // Anchor function inside the NorthstarCore shared module whose address is used to resolve the module's own directory.
        void CoreModuleAddressAnchor() {}

        ShaderSpaceMask BuildInternalShaderSpaceMask(const Shader& shader) noexcept
        {
            const ShaderReflection& refl = shader.GetReflection();
            if (refl.nscVersion == 0) return ShaderSpaceMask{};

            ShaderSpaceMask mask{};
            mask.cbvSrvMask  = (shader.GetStage() == ShaderStage::VERTEX) ? 0x0001u : 0x0000u;
            mask.samplerMask = 0x0000u;
            mask.uavMask     = 0x0000u;
            mask.isExplicit  = true;

            for (const auto& b : refl.buffers)
                if (b.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << b.set));
            for (const auto& sb : refl.structuredBuffers)
            {
                if (refl.FindUAV(sb.name) != nullptr) continue;
                if (sb.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << sb.set));
            }
            for (const auto& t : refl.textures)
                if (t.set < 16u) mask.cbvSrvMask = static_cast<UINT16>(mask.cbvSrvMask | (1u << t.set));
            for (const auto& sm : refl.samplers)
                if (sm.set < 16u) mask.samplerMask = static_cast<UINT16>(mask.samplerMask | (1u << sm.set));
            for (const auto& uav : refl.uavBuffers)
            {
                if (uav.set < 16u) mask.uavMask = static_cast<UINT16>(mask.uavMask | (1u << uav.set));
                for (const auto& f : uav.fields)
                    if (f.set < 16u && f.binding != 0xFFFFu)
                        mask.uavMask = static_cast<UINT16>(mask.uavMask | (1u << f.set));
            }
            return mask;
        }

        UINT32 SelectInternalShaderModel(const NorthstarBackend* backend) noexcept
        {
            if (!backend || backend->GetTargetAPI() != TargetAPI::HLSL12)
                return 0x50u;
            const BackendCapabilities caps = backend->GetCapabilities();
            return (caps.maxShaderModel >= 0x66u && caps.supportsDXILRuntime) ? 0x66u : 0x51u;
        }
    }

    InternalResourceManager::InternalResourceManager()
    {
        ResetDefaultSpecs();
    }

    InternalResourceManager::~InternalResourceManager()
    {
        Shutdown();
    }

    void InternalResourceManager::ResetDefaultSpecs()
    {
        ReleaseLoadedResources();
        m_entries.clear();
        m_graphicsEntries.clear();

        // Slot 0: InternalComputeKernelId::BUFFER_CLEAR_UINT
        InternalComputeShaderSpec clearSpec{};
        clearSpec.idName                    = "Runtime.BufferClearUInt";
        clearSpec.relativeCategory          = "Compute";
        clearSpec.baseFileName              = "RuntimeBufferClear";
        clearSpec.requirement               = InternalResourceRequirement::MANDATORY;
        clearSpec.expectedThreadsX          = 64;
        clearSpec.expectedThreadsY          = 1;
        clearSpec.expectedThreadsZ          = 1;
        clearSpec.requiredConstantBuffers   = { "Config" };
        clearSpec.requiredUAVStructuredBufs = { "OutBuffer" };

        LoadedInternalComputeEntry clearEntry{};
        clearEntry.spec = std::move(clearSpec);
        m_entries.push_back(std::move(clearEntry));

        // Slot 1: InternalComputeKernelId::DIRTY_DEPTH_SLICE_BUILD
        InternalComputeShaderSpec dirtySliceSpec{};
        dirtySliceSpec.idName                    = "Runtime.DirtyDepthSliceBuild";
        dirtySliceSpec.relativeCategory          = "Compute";
        dirtySliceSpec.baseFileName              = "RuntimeDirtyDepthSliceBuild";
        dirtySliceSpec.requirement               = InternalResourceRequirement::MANDATORY;
        dirtySliceSpec.expectedThreadsX          = 64;
        dirtySliceSpec.expectedThreadsY          = 1;
        dirtySliceSpec.expectedThreadsZ          = 1;
        dirtySliceSpec.requiredConstantBuffers   = { "Config" };
        dirtySliceSpec.requiredSRVStructuredBufs = { "CandidateSlices" };
        dirtySliceSpec.requiredUAVStructuredBufs = { "OutDirtySliceIndices", "OutDirtyCount", "OutIndirectDrawArgs" };

        LoadedInternalComputeEntry dirtyEntry{};
        dirtyEntry.spec = std::move(dirtySliceSpec);
        m_entries.push_back(std::move(dirtyEntry));

        // Slot 2: InternalComputeKernelId::GPU_SCENE_DIAGNOSTIC
        InternalComputeShaderSpec gpuSceneDiagSpec{};
        gpuSceneDiagSpec.idName                    = "Runtime.GPUSceneDiagnostic";
        gpuSceneDiagSpec.relativeCategory          = "Compute";
        gpuSceneDiagSpec.baseFileName              = "RuntimeGPUSceneDiagnostic";
        gpuSceneDiagSpec.requirement               = InternalResourceRequirement::MANDATORY;
        gpuSceneDiagSpec.expectedThreadsX          = 64;
        gpuSceneDiagSpec.expectedThreadsY          = 1;
        gpuSceneDiagSpec.expectedThreadsZ          = 1;
        gpuSceneDiagSpec.requiredConstantBuffers   = { "Config" };
        gpuSceneDiagSpec.requiredSRVStructuredBufs = { "SceneObjects", "Queries" };
        gpuSceneDiagSpec.requiredUAVStructuredBufs = { "OutDiagnostics" };

        LoadedInternalComputeEntry gpuSceneDiagEntry{};
        gpuSceneDiagEntry.spec = std::move(gpuSceneDiagSpec);
        m_entries.push_back(std::move(gpuSceneDiagEntry));

        // Slot 3: InternalComputeKernelId::NVS_PAGE_RESIDENCY
        InternalComputeShaderSpec nvsResidencySpec{};
        nvsResidencySpec.idName                    = "Runtime.NVSPageResidency";
        nvsResidencySpec.relativeCategory          = "Compute";
        nvsResidencySpec.baseFileName              = "RuntimeNVSPageResidency";
        nvsResidencySpec.requirement               = InternalResourceRequirement::MANDATORY;
        nvsResidencySpec.expectedThreadsX          = 64;
        nvsResidencySpec.expectedThreadsY          = 1;
        nvsResidencySpec.expectedThreadsZ          = 1;
        nvsResidencySpec.requiredConstantBuffers   = { "Config" };
        nvsResidencySpec.requiredSRVStructuredBufs = { "PageRequests" };
        nvsResidencySpec.requiredUAVStructuredBufs = {
            "VirtualPageTable",
            "PhysicalPageTable",
            "OutDirtyPhysicalSlices",
            "OutDirtySliceCount",
            "OutSelectiveClearIndirectArgs",
            "OutTelemetry"
        };

        LoadedInternalComputeEntry nvsResidencyEntry{};
        nvsResidencyEntry.spec = std::move(nvsResidencySpec);
        m_entries.push_back(std::move(nvsResidencyEntry));

        // Slot 4: InternalComputeKernelId::NVS_SCREEN_SPACE_FEEDBACK
        InternalComputeShaderSpec nvsFeedbackSpec{};
        nvsFeedbackSpec.idName                    = "Runtime.NVSScreenSpaceFeedback";
        nvsFeedbackSpec.relativeCategory          = "Compute";
        nvsFeedbackSpec.baseFileName              = "RuntimeNVSFeedback";
        nvsFeedbackSpec.requirement               = InternalResourceRequirement::MANDATORY;
        nvsFeedbackSpec.expectedThreadsX          = 64;
        nvsFeedbackSpec.expectedThreadsY          = 1;
        nvsFeedbackSpec.expectedThreadsZ          = 1;
        nvsFeedbackSpec.requiredConstantBuffers   = { "Config" };
        nvsFeedbackSpec.requiredSRVTextures       = { "SceneDepth" };
        nvsFeedbackSpec.requiredSRVStructuredBufs = { "InClipmapLevels" };
        nvsFeedbackSpec.requiredUAVStructuredBufs = {
            "OutPageBitmask",
            "OutCompactedRequests",
            "OutFeedbackTelemetry",
            "OutResidencyTelemetry"
        };

        LoadedInternalComputeEntry nvsFeedbackEntry{};
        nvsFeedbackEntry.spec = std::move(nvsFeedbackSpec);
        m_entries.push_back(std::move(nvsFeedbackEntry));

        // Slot 5: InternalComputeKernelId::NVS_PAGE_PROJECTION
        InternalComputeShaderSpec nvsProjectionSpec{};
        nvsProjectionSpec.idName                  = "Runtime.NVSPageProjection";
        nvsProjectionSpec.relativeCategory        = "Compute";
        nvsProjectionSpec.baseFileName            = "RuntimeNVSPageProjection";
        nvsProjectionSpec.requirement             = InternalResourceRequirement::MANDATORY;
        nvsProjectionSpec.expectedThreadsX        = 64;
        nvsProjectionSpec.expectedThreadsY        = 1;
        nvsProjectionSpec.expectedThreadsZ        = 1;
        nvsProjectionSpec.requiredConstantBuffers = { "Config" };
        nvsProjectionSpec.requiredSRVStructuredBufs = {
            "InClipmapLevels",
            "InVirtualPageTable",
            "InPhysicalPageTable"
        };
        nvsProjectionSpec.requiredUAVStructuredBufs = {
            "OutPageRenderData",
            "OutActiveRenderPageList",
            "OutProjectionTelemetry"
        };

        LoadedInternalComputeEntry nvsProjectionEntry{};
        nvsProjectionEntry.spec = std::move(nvsProjectionSpec);
        m_entries.push_back(std::move(nvsProjectionEntry));

        // Slot 6: InternalComputeKernelId::NVS_CASTER_CULL
        InternalComputeShaderSpec nvsCasterCullSpec{};
        nvsCasterCullSpec.idName                  = "Runtime.NVSCasterCull";
        nvsCasterCullSpec.relativeCategory        = "Compute";
        nvsCasterCullSpec.baseFileName            = "RuntimeNVSCasterCull";
        nvsCasterCullSpec.requirement             = InternalResourceRequirement::MANDATORY;
        nvsCasterCullSpec.expectedThreadsX        = 64;
        nvsCasterCullSpec.expectedThreadsY        = 1;
        nvsCasterCullSpec.expectedThreadsZ        = 1;
        nvsCasterCullSpec.requiredConstantBuffers = { "Config" };
        nvsCasterCullSpec.requiredSRVStructuredBufs = {
            "InGPUSceneObjects",
            "InClipmapLevels",
            "InPageRenderData",
            "InActiveRenderPageList",
            "InDynamicInvalidationBoxes"
        };
        nvsCasterCullSpec.requiredUAVStructuredBufs = {
            "OutFilteredCasters",
            "OutCasterValidBitmask",
            "OutCasterPageRanges",
            "OutCasterWorklist",
            "OutWorklistIndirectArgs",
            "OutCasterTelemetry",
            "InOutVirtualPageTable",
            "InOutPhysicalPageTable"
        };

        LoadedInternalComputeEntry nvsCasterCullEntry{};
        nvsCasterCullEntry.spec = std::move(nvsCasterCullSpec);
        m_entries.push_back(std::move(nvsCasterCullEntry));

        // Slot 7: InternalComputeKernelId::NVS_INDIRECT_DRAW_BUILD
        InternalComputeShaderSpec nvsIndirectDrawSpec{};
        nvsIndirectDrawSpec.idName                  = "Runtime.NVSIndirectDraw";
        nvsIndirectDrawSpec.relativeCategory        = "Compute";
        nvsIndirectDrawSpec.baseFileName            = "RuntimeNVSIndirectDraw";
        nvsIndirectDrawSpec.requirement             = InternalResourceRequirement::MANDATORY;
        nvsIndirectDrawSpec.expectedThreadsX        = 64;
        nvsIndirectDrawSpec.expectedThreadsY        = 1;
        nvsIndirectDrawSpec.expectedThreadsZ        = 1;
        nvsIndirectDrawSpec.requiredConstantBuffers = { "Config" };
        nvsIndirectDrawSpec.requiredSRVStructuredBufs = {
            "InProjectionCounter",
            "InCasterPageRanges",
            "InCasterWorklist",
            "InPageRenderData",
            "InGPUSceneObjects",
            "InMeshDescriptors",
            "InMaterialDescriptors",
            "InDrawGroupBindings",
            "InDrawGroupFieldPatches",
            "InTemplateWords"
        };
        nvsIndirectDrawSpec.requiredUAVStructuredBufs = {
            "OutShadowDrawContexts",
            "OutIndirectDrawCommands",
            "OutPerDrawCBVWords",
            "OutPerBatchInstanceWords",
            "OutAllocationCounters",
            "OutBucketCommandCounts",
            "OutPageCommitRecords",
            "OutTelemetry"
        };

        LoadedInternalComputeEntry nvsIndirectDrawEntry{};
        nvsIndirectDrawEntry.spec = std::move(nvsIndirectDrawSpec);
        m_entries.push_back(std::move(nvsIndirectDrawEntry));

        // Slot 8: InternalComputeKernelId::NVG_HZB_BUILD (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                  = "Runtime.NVGHZBBuild";
            spec.relativeCategory        = "Compute";
            spec.baseFileName            = "RuntimeNVGHZBBuild";
            spec.requirement             = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX        = 8;
            spec.expectedThreadsY        = 8;
            spec.expectedThreadsZ        = 1;
            spec.requiredConstantBuffers = { "Config" };
            spec.requiredSRVTextures     = { "SrcDepth" };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 9: InternalComputeKernelId::NVG_RESIDENCY_MASK (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGResidencyMask";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGResidencyMask";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = { "InClusterGroups", "InVirtualPageTable" };
            spec.requiredUAVStructuredBufs = { "OutGroupActiveMask" };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 10: InternalComputeKernelId::NVG_INSTANCE_CULL (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGInstanceCull";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGInstanceCull";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVTextures       = { "HZBTexture" };
            spec.requiredSRVStructuredBufs = { "InGPUSceneObjects", "InMeshTable", "InMaterialTable" };
            spec.requiredUAVStructuredBufs = {
                "OutNodeQueue", "OutOccludedObjects", "OutQueueCounters",
                "OutBucketCounts", "OutDispatchArgs", "OutPageRequests", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 11: InternalComputeKernelId::NVG_HIERARCHY_CULL (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGHierarchyCull";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGHierarchyCull";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVTextures       = { "HZBTexture" };
            spec.requiredSRVStructuredBufs = {
                "InGPUSceneObjects", "InMaterialTable", "InBVHNodes", "InClusterMeta",
                "InClusterGroups", "InVirtualPageTable", "InGroupActiveMask", "InNodeQueue"
            };
            spec.requiredUAVStructuredBufs = {
                "OutNodeQueue", "OutCandidateClusters", "OutOccludedClusters",
                "OutVisibleClusters", "OutQueueCounters", "OutBucketCounts",
                "OutPageRequests", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 12: InternalComputeKernelId::NVG_COMPACTION_DECODE (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGCompactionAndDecode";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGCompactionAndDecode";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 128;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InPagePool", "InBucketCounts", "InBucketFlags", "InRawVisibleClusters"
            };
            spec.requiredUAVStructuredBufs = {
                "OutBucketRanges", "OutBucketWriteCursors", "OutPathAIndirectArgs",
                "OutCompactedVisibleClusters", "OutDecodedClusterTable", "OutUniqueDecodeJobs",
                "OutDecodedVertexWords", "OutDecodedIndexPool", "OutQueueCounters", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 13: InternalComputeKernelId::NVG_INDIRECT_DRAW_BUILD (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGIndirectDraw";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGIndirectDraw";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InCompactedVisibleClusters", "InBucketRanges", "InDecodedClusterTable",
                "InGPUSceneObjects", "InDrawGroupBindings", "InDrawGroupFieldPatches",
                "InDrawGroupTemplateWords", "InShadowPageViews"
            };
            spec.requiredUAVStructuredBufs = {
                "OutIndirectDrawCommands", "OutPerDrawCBVWords", "OutPerBatchInstanceWords",
                "OutBucketCommandCounts", "OutQueueCounters", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 14: InternalComputeKernelId::NVG_SHADOW_CULL (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGShadowCull";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGShadowCull";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InProjectionCounter", "InCasterPageRanges", "InPageRenderData",
                "InGPUSceneObjects", "InMeshTable", "InMaterialTable", "InBVHNodes",
                "InClusterMeta", "InClusterGroups", "InVirtualPageTable",
                "InGroupActiveMask", "InNodeQueue"
            };
            spec.requiredUAVStructuredBufs = {
                "OutNodeQueue", "OutCandidateClusters", "OutShadowPageViews",
                "OutVisibleClusters", "OutQueueCounters", "OutBucketCounts",
                "OutPageRequests", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 15: InternalComputeKernelId::REFLECTION_MATERIALIZE (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.ReflectionMaterialize";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeReflectionMaterialize";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = { "InSrcWords", "InCopySpans", "InUserWords" };
            spec.requiredUAVStructuredBufs = { "OutDstWords" };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 16: InternalComputeKernelId::NVG_MICROPOLY_RASTER_ATOMIC64 (FEATURE_DEPENDENT, Single-Pass 64-bit Atomic)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGMicropolyRasterAtomic64";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGMicropolyRasterAtomic64";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InPagePool", "InCompactedVisibleClusters", "InBucketRanges",
                "InGPUSceneObjects", "InClusterMeta", "InMaterialTable",
                "InBonePalette", "InSkinnedBounds"
            };
            spec.requiredUAVStructuredBufs = {
                "OutVisBuffer64", "OutSWClusterList", "OutSWScheduledMask",
                "OutWinningClusterMask", "OutQueueCounters", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 17: InternalComputeKernelId::NVG_MICROPOLY_RASTER_ATOMIC32 (FEATURE_DEPENDENT, Universal Two-Pass 2x32-bit Atomic)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGMicropolyRasterAtomic32";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGMicropolyRasterAtomic32";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InPagePool", "InCompactedVisibleClusters", "InBucketRanges",
                "InGPUSceneObjects", "InClusterMeta", "InMaterialTable",
                "InBonePalette", "InSkinnedBounds"
            };
            spec.requiredUAVStructuredBufs = {
                "OutAtomicDepthKey32", "OutAtomicPayload32", "OutSWClusterList",
                "OutSWScheduledMask", "OutWinningClusterMask", "OutQueueCounters", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 18: InternalComputeKernelId::NVG_DEPTH_EXPORT (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGDepthExport";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGDepthExport";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 8;
            spec.expectedThreadsY          = 8;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = { "InVisBuffer64", "InAtomicDepthKey32", "InAtomicPayload32" };
            spec.requiredUAVStructuredBufs = { "OutWinningClusterMask" };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Slot 19: InternalComputeKernelId::NVG_SHADOW_MICROPOLY_RASTER (FEATURE_DEPENDENT)
        {
            InternalComputeShaderSpec spec{};
            spec.idName                    = "Runtime.NVGShadowMicropolyRaster";
            spec.relativeCategory          = "Compute";
            spec.baseFileName              = "RuntimeNVGShadowMicropolyRaster";
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.expectedThreadsX          = 64;
            spec.expectedThreadsY          = 1;
            spec.expectedThreadsZ          = 1;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = {
                "InPagePool", "InShadowPageViews", "InGPUSceneObjects", "InClusterMeta",
                "InMaterialTable", "InBonePalette", "InSkinnedBounds"
            };
            spec.requiredUAVStructuredBufs = {
                "InOutCompactedVisibleClusters", "InOutBucketRanges", "OutPathAIndirectArgs",
                "OutShadowSWDepthKey", "OutShadowSWClusterList", "OutShadowSWScheduledMask",
                "OutShadowSWFallbackMask", "OutPageNeedsSWTile", "OutPageTileMap",
                "OutDirtySWTileSlices", "OutMergeIndirectArgs", "OutDispatchArgs",
                "OutQueueCounters", "OutTelemetry"
            };
            LoadedInternalComputeEntry entry{};
            entry.spec = std::move(spec);
            m_entries.push_back(std::move(entry));
        }

        // Graphics Slot 0: InternalShaderId::LAYERED_DEPTH_CLEAR_VS
        InternalGraphicsShaderSpec clearVsSpec{};
        clearVsSpec.idName                    = "Runtime.LayeredDepthClearVS";
        clearVsSpec.relativeCategory          = "Depth";
        clearVsSpec.baseFileName              = "RuntimeLayeredDepthClearVS";
        clearVsSpec.expectedStage             = ShaderStage::VERTEX;
        clearVsSpec.requirement               = InternalResourceRequirement::MANDATORY;
        clearVsSpec.requiredConstantBuffers   = { "Config" };
        clearVsSpec.requiredSRVStructuredBufs = { "DirtySlices" };

        LoadedInternalGraphicsEntry clearVsEntry{};
        clearVsEntry.spec = std::move(clearVsSpec);
        m_graphicsEntries.push_back(std::move(clearVsEntry));

        // Graphics Slot 1: InternalShaderId::LAYERED_DEPTH_PASS_GS
        InternalGraphicsShaderSpec passGsSpec{};
        passGsSpec.idName           = "Runtime.LayeredDepthPassGS";
        passGsSpec.relativeCategory = "Depth";
        passGsSpec.baseFileName     = "RuntimeLayeredDepthPassGS";
        passGsSpec.expectedStage    = ShaderStage::GEOMETRY;
        passGsSpec.requirement      = InternalResourceRequirement::MANDATORY;

        LoadedInternalGraphicsEntry passGsEntry{};
        passGsEntry.spec = std::move(passGsSpec);
        m_graphicsEntries.push_back(std::move(passGsEntry));

        // Graphics Slot 2: InternalShaderId::NVG_SHADOW_SW_MERGE_VS (FEATURE_DEPENDENT)
        {
            InternalGraphicsShaderSpec spec{};
            spec.idName                    = "Runtime.NVGShadowSWMergeVS";
            spec.relativeCategory          = "Depth";
            spec.baseFileName              = "RuntimeNVGShadowSWMergeVS";
            spec.expectedStage             = ShaderStage::VERTEX;
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = { "DirtySWTiles" };
            LoadedInternalGraphicsEntry entry{};
            entry.spec = std::move(spec);
            m_graphicsEntries.push_back(std::move(entry));
        }

        // Graphics Slot 3: InternalShaderId::NVG_SHADOW_SW_MERGE_GS (FEATURE_DEPENDENT)
        {
            InternalGraphicsShaderSpec spec{};
            spec.idName           = "Runtime.NVGShadowSWMergeGS";
            spec.relativeCategory = "Depth";
            spec.baseFileName     = "RuntimeNVGShadowSWMergeGS";
            spec.expectedStage    = ShaderStage::GEOMETRY;
            spec.requirement      = InternalResourceRequirement::FEATURE_DEPENDENT;
            LoadedInternalGraphicsEntry entry{};
            entry.spec = std::move(spec);
            m_graphicsEntries.push_back(std::move(entry));
        }

        // Graphics Slot 4: InternalShaderId::NVG_SHADOW_SW_MERGE_PS (FEATURE_DEPENDENT)
        {
            InternalGraphicsShaderSpec spec{};
            spec.idName                    = "Runtime.NVGShadowSWMergePS";
            spec.relativeCategory          = "Depth";
            spec.baseFileName              = "RuntimeNVGShadowSWMergePS";
            spec.expectedStage             = ShaderStage::PIXEL;
            spec.requirement               = InternalResourceRequirement::FEATURE_DEPENDENT;
            spec.requiredConstantBuffers   = { "Config" };
            spec.requiredSRVStructuredBufs = { "InShadowSWDepthKey" };
            LoadedInternalGraphicsEntry entry{};
            entry.spec = std::move(spec);
            m_graphicsEntries.push_back(std::move(entry));
        }
    }

    BOOL InternalResourceManager::RegisterComputeShaderSpec(const InternalComputeShaderSpec& spec)
    {
        if (spec.idName.empty() ||
            !IsSafeRelativeSegment(spec.relativeCategory) ||
            !IsSafeRelativeSegment(spec.baseFileName))
        {
            m_lastError = "Invalid InternalComputeShaderSpec: idName, relativeCategory, and baseFileName must be non-empty safe identifiers.";
            return false;
        }

        for (auto& existing : m_entries)
        {
            if (existing.spec.idName == spec.idName)
            {
                if (existing.loaded)
                {
                    m_lastError = "Cannot overwrite already-loaded internal compute shader spec '" + spec.idName + "'.";
                    return false;
                }
                existing.spec = spec;
                return true;
            }
        }

        LoadedInternalComputeEntry entry{};
        entry.spec = spec;
        m_entries.push_back(std::move(entry));
        return true;
    }

    BOOL InternalResourceManager::RegisterGraphicsShaderSpec(const InternalGraphicsShaderSpec& spec)
    {
        if (spec.idName.empty() ||
            !IsSafeRelativeSegment(spec.relativeCategory) ||
            !IsSafeRelativeSegment(spec.baseFileName) ||
            spec.expectedStage == ShaderStage::NONE ||
            spec.expectedStage == ShaderStage::COMPUTE)
        {
            m_lastError = "Invalid InternalGraphicsShaderSpec: idName, relativeCategory, baseFileName, and non-compute expectedStage are required.";
            return false;
        }

        for (auto& existing : m_graphicsEntries)
        {
            if (existing.spec.idName == spec.idName)
            {
                if (existing.loaded)
                {
                    m_lastError = "Cannot overwrite already-loaded internal graphics shader spec '" + spec.idName + "'.";
                    return false;
                }
                existing.spec = spec;
                return true;
            }
        }

        LoadedInternalGraphicsEntry entry{};
        entry.spec = spec;
        m_graphicsEntries.push_back(std::move(entry));
        return true;
    }

    BOOL InternalResourceManager::ResolveCoreModuleDirectory(std::filesystem::path& outModuleDir, std::string& outError)
    {
        outModuleDir.clear();
        outError.clear();

#ifdef _WIN32
        HMODULE hModule = nullptr;
        const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&CoreModuleAddressAnchor), &hModule) || hModule == nullptr)
        {
            const DWORD err = ::GetLastError();
            outError = "Failed to resolve NorthstarCore module handle (error " + std::to_string(err) + ").";
            return false;
        }

        std::wstring modulePathBuf(512, L'\0');
        DWORD copiedLen = 0;
        while (modulePathBuf.size() <= 32768)
        {
            copiedLen = GetModuleFileNameW(hModule, modulePathBuf.data(), static_cast<DWORD>(modulePathBuf.size()));
            if (copiedLen == 0)
            {
                const DWORD err = ::GetLastError();
                outError = "Failed to query NorthstarCore module file path (error " + std::to_string(err) + ").";
                return false;
            }
            if (copiedLen < modulePathBuf.size() - 1)
            {
                modulePathBuf.resize(copiedLen);
                break;
            }
            modulePathBuf.resize(modulePathBuf.size() * 2, L'\0');
        }

        if (copiedLen == 0 || modulePathBuf.empty())
        {
            outError = "Resolved NorthstarCore module file path is empty.";
            return false;
        }

        const std::filesystem::path fullModulePath(modulePathBuf);
        if (!fullModulePath.has_parent_path())
        {
            outError = "Resolved NorthstarCore module path '" + PathToUTF8String(fullModulePath) + "' has no parent directory.";
            return false;
        }

        outModuleDir = fullModulePath.parent_path();
        return true;
#elif defined(__unix__) || defined(__APPLE__)
        Dl_info dlInfo{};
        if (dladdr(reinterpret_cast<const void*>(&CoreModuleAddressAnchor), &dlInfo) == 0 || dlInfo.dli_fname == nullptr)
        {
            outError = "Failed to resolve NorthstarCore shared module path via dladdr.";
            return false;
        }

        std::error_code ec;
        std::filesystem::path fullModulePath = std::filesystem::canonical(std::filesystem::path(dlInfo.dli_fname), ec);
        if (ec)
        {
            fullModulePath = std::filesystem::absolute(std::filesystem::path(dlInfo.dli_fname), ec);
        }

        if (fullModulePath.empty() || !fullModulePath.has_parent_path())
        {
            outError = "Resolved NorthstarCore module path has no parent directory.";
            return false;
        }

        outModuleDir = fullModulePath.parent_path();
        return true;
#else
        outError = "NorthstarCore module directory resolution is not supported on this platform.";
        return false;
#endif
    }

    BOOL InternalResourceManager::BuildSafeShaderFilePath(const std::string& idName,
                                                          const std::string& relativeCategory,
                                                          const std::string& baseFileName,
                                                          TargetAPI targetApi,
                                                          std::filesystem::path& outFilePath,
                                                          std::string& outError) const
    {
        outFilePath.clear();
        outError.clear();

        if (m_runtimeDirPath.empty())
        {
            outError = "NorthstarRuntime directory path has not been resolved.";
            return false;
        }

        if (!IsSafeRelativeSegment(relativeCategory) || !IsSafeRelativeSegment(baseFileName))
        {
            outError = "Unsafe relative path component in internal shader spec '" + idName + "'.";
            return false;
        }

        const char* apiSuffix = TargetAPIToSuffix(targetApi);
        if (!apiSuffix)
        {
            outError = "Unsupported backend TargetAPI for internal shader '" + idName + "'.";
            return false;
        }

        const std::filesystem::path categoryDir =
            m_runtimeDirPath / std::filesystem::path(reinterpret_cast<const char8_t*>(relativeCategory.c_str()));
        std::error_code ec;
        if (!std::filesystem::exists(categoryDir, ec) || !std::filesystem::is_directory(categoryDir, ec))
        {
            outError = "Required NorthstarRuntime category directory missing or inaccessible: '" + PathToUTF8String(categoryDir) + "'.";
            return false;
        }

        const std::string fileName = baseFileName + "_" + apiSuffix + ".nsc";
        outFilePath = categoryDir / std::filesystem::path(reinterpret_cast<const char8_t*>(fileName.c_str()));
        return true;
    }

    BOOL InternalResourceManager::LoadEntry(LoadedInternalComputeEntry& entry, std::string& outError, BOOL bypassDiskCache)
    {
        outError.clear();
        if (entry.loaded && entry.kernel && entry.kernel->IsValid())
        {
            return true;
        }

        if (!m_backend || !m_computeManager)
        {
            outError = "InternalResourceManager cannot load shader '" + entry.spec.idName + "' without a valid backend and CoreComputeManager.";
            return false;
        }

        std::filesystem::path filePath;
        if (!BuildSafeShaderFilePath(entry.spec.idName, entry.spec.relativeCategory, entry.spec.baseFileName, m_backend->GetTargetAPI(), filePath, outError))
        {
            return false;
        }

        const std::string filePathUTF8 = PathToUTF8String(filePath);
        std::error_code ec;
        if (!std::filesystem::exists(filePath, ec))
        {
            outError = "Required NorthstarRuntime shader file not found: '" + filePathUTF8 + "' (for '" + entry.spec.idName + "').";
            return false;
        }
        if (!std::filesystem::is_regular_file(filePath, ec))
        {
            outError = "NorthstarRuntime shader path is not a regular file: '" + filePathUTF8 + "'.";
            return false;
        }

        const auto fileSize = std::filesystem::file_size(filePath, ec);
        if (ec || fileSize == 0 || fileSize > 64u * 1024u * 1024u)
        {
            outError = "NorthstarRuntime shader file '" + filePathUTF8 + "' is empty or has an invalid file size.";
            return false;
        }

        std::ifstream fileStream(filePath, std::ios::binary);
        if (!fileStream.is_open())
        {
            outError = "Failed to open NorthstarRuntime shader file for reading: '" + filePathUTF8 + "'.";
            return false;
        }

        std::vector<BYTE> binaryData(static_cast<SIZE_T>(fileSize));
        if (!fileStream.read(reinterpret_cast<char*>(binaryData.data()), static_cast<std::streamsize>(fileSize)))
        {
            outError = "Failed to read bytes from NorthstarRuntime shader file: '" + filePathUTF8 + "'.";
            return false;
        }

        Shader* shader = new Shader();
        if (!ShaderLoader::LoadFromMemory(binaryData.data(), binaryData.size(), *shader, entry.spec.idName) || !shader->IsValid())
        {
            delete shader;
            outError = "Failed to parse NorthstarRuntime shader '" + entry.spec.idName + "' from '" + filePathUTF8 + "': corrupted or unsupported .nsc binary.";
            return false;
        }

        // Validate shader stage contract
        if (shader->GetStage() != ShaderStage::COMPUTE)
        {
            delete shader;
            outError = "NorthstarRuntime shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' has invalid stage (expected COMPUTE).";
            return false;
        }

        // Validate backend TargetAPI compatibility
        if (shader->GetTargetAPI() != m_backend->GetTargetAPI())
        {
            delete shader;
            outError = "NorthstarRuntime shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' TargetAPI does not match active backend TargetAPI.";
            return false;
        }

        // Validate binary [numthreads(X,Y,Z)] reflection metadata
        const ShaderReflection& refl = shader->GetReflection();
        if ((refl.flags & NSL::NSC_FLAG_HAS_COMPUTE_NUMTHREADS) == 0 || !refl.HasValidNumThreads())
        {
            delete shader;
            outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' lacks binary [numthreads(X,Y,Z)] reflection metadata.";
            return false;
        }

        if ((entry.spec.expectedThreadsX > 0 && refl.numThreadsX != entry.spec.expectedThreadsX) ||
            (entry.spec.expectedThreadsY > 0 && refl.numThreadsY != entry.spec.expectedThreadsY) ||
            (entry.spec.expectedThreadsZ > 0 && refl.numThreadsZ != entry.spec.expectedThreadsZ))
        {
            delete shader;
            outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' [numthreads] mismatch.";
            return false;
        }

        // Validate required reflection bindings
        for (const std::string& reqCb : entry.spec.requiredConstantBuffers)
        {
            if (!ReflectionHasConstantBuffer(refl, reqCb))
            {
                delete shader;
                outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' is missing required ConstantBuffer '" + reqCb + "' in reflection.";
                return false;
            }
        }

        for (const std::string& reqTex : entry.spec.requiredSRVTextures)
        {
            if (!ReflectionHasSRVTexture(refl, reqTex))
            {
                delete shader;
                outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' is missing required Texture2D '" + reqTex + "' in reflection.";
                return false;
            }
        }

        for (const std::string& reqSrv : entry.spec.requiredSRVStructuredBufs)
        {
            if (!ReflectionHasSRVStructuredBuffer(refl, reqSrv))
            {
                delete shader;
                outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' is missing required StructuredBuffer '" + reqSrv + "' in reflection.";
                return false;
            }
        }

        for (const std::string& reqUav : entry.spec.requiredUAVStructuredBufs)
        {
            if (!ReflectionHasUAVStructuredBuffer(refl, reqUav))
            {
                delete shader;
                outError = "NorthstarRuntime compute shader '" + entry.spec.idName + "' is missing required RWStructuredBuffer '" + reqUav + "' in reflection.";
                return false;
            }
        }

        // Enforce 64-bit integer and 64-bit descriptor-heap atomic hardware capabilities BEFORE DXBC cache lookup or compilation
        const BackendCapabilities backendCaps = m_backend->GetCapabilities();
        if (refl.UsesInt64Atomic() && !backendCaps.SupportsNVGRasterAtomic64())
        {
            delete shader;
            outError = "NorthstarRuntime compute shader '" + entry.spec.idName +
                       "' requires Shader Model 6.6 DXIL, Int64ShaderOps, and 64-bit descriptor-heap atomics, which are unsupported on the active adapter.";
            return false;
        }
        if (refl.UsesInt64() && !(backendCaps.maxShaderModel >= 0x60u && backendCaps.supportsDXILRuntime && backendCaps.supportsInt64ShaderOps))
        {
            delete shader;
            outError = "NorthstarRuntime compute shader '" + entry.spec.idName +
                       "' requires 64-bit integer shader operations (Int64ShaderOps + DXIL), which are unsupported on the active adapter.";
            return false;
        }

        // Register native shader handle on backend (prefer Core-cached DXBC/DXIL bytecode when available)
        const UINT32 targetSM = SelectInternalShaderModel(m_backend);
        if (!bypassDiskCache && m_backend->GetTargetAPI() == TargetAPI::HLSL12 && !shader->HasBytecode() && shader->HasSourceCode())
        {
            ShaderLoader::TryPopulateCachedDXBC(*shader, targetSM);
        }

        NativeHandle nativeHandle = 0;
        if (shader->HasBytecode())
        {
            nativeHandle = m_backend->CreateShader(
                shader->GetStage(),
                shader->GetBytecodeData(),
                shader->GetBytecodeSize()
            );
        }
        else if (shader->HasSourceCode())
        {
            nativeHandle = m_backend->CreateShader(
                shader->GetStage(),
                shader->GetSourceCode().data(),
                shader->GetSourceCode().size()
            );
            if (nativeHandle != 0)
            {
                std::vector<BYTE> compiledDxbc;
                if (m_backend->GetShaderCompiledBytecode(nativeHandle, compiledDxbc) && !compiledDxbc.empty())
                {
                    if (!bypassDiskCache)
                    {
                        ShaderLoader::StoreCachedDXBC(*shader, compiledDxbc.data(), compiledDxbc.size(), targetSM);
                    }
                    shader->SetBytecode(std::move(compiledDxbc));
                }
            }
        }
        shader->SetNativeHandle(nativeHandle);
        if (nativeHandle != 0)
        {
            const ShaderSpaceMask spaceMask = BuildInternalShaderSpaceMask(*shader);
            if (spaceMask.isExplicit)
            {
                m_backend->SetShaderReflectionSpaces(nativeHandle, spaceMask);
            }
        }

        // On D3D12 hardware backend (TargetAPI::HLSL12), native shader compilation/creation must succeed
        if (m_backend->GetTargetAPI() == TargetAPI::HLSL12 && nativeHandle == 0)
        {
            delete shader;
            outError = "Backend failed to compile/create native compute shader for NorthstarRuntime resource '" + entry.spec.idName + "'.";
            return false;
        }

        // Create engine-internal ComputeKernel through CoreComputeManager (reusing the unified ComputeKernel pipeline)
        ComputeKernelDesc kernelDesc{};
        kernelDesc.computeShader     = shader;
        kernelDesc.name              = entry.spec.idName;
        kernelDesc.threadGroupCountX = 1;
        kernelDesc.threadGroupCountY = 1;
        kernelDesc.threadGroupCountZ = 1;
        kernelDesc.executionScope    = ComputeExecutionScope::PER_FRAME;

        ComputeKernel* kernel = m_computeManager->CreateInternalComputeKernel(kernelDesc);
        if (!kernel || !kernel->IsValid() || !kernel->HasValidThreadGroupSize())
        {
            if (kernel)
            {
                m_computeManager->DestroyInternalComputeKernel(kernel);
            }
            if (nativeHandle != 0)
            {
                m_backend->DestroyShader(nativeHandle);
                shader->SetNativeHandle(0);
            }
            delete shader;
            outError = "Failed to create valid internal ComputeKernel for '" + entry.spec.idName + "'.";
            return false;
        }

        entry.shader = shader;
        entry.kernel = kernel;
        entry.loaded = true;
        return true;
    }

    BOOL InternalResourceManager::LoadGraphicsEntry(LoadedInternalGraphicsEntry& entry, std::string& outError)
    {
        outError.clear();
        if (entry.loaded && entry.shader && entry.shader->IsValid())
        {
            return true;
        }

        if (!m_backend)
        {
            outError = "InternalResourceManager cannot load graphics shader '" + entry.spec.idName + "' without a valid backend.";
            return false;
        }

        std::filesystem::path filePath;
        if (!BuildSafeShaderFilePath(entry.spec.idName, entry.spec.relativeCategory, entry.spec.baseFileName, m_backend->GetTargetAPI(), filePath, outError))
        {
            return false;
        }

        const std::string filePathUTF8 = PathToUTF8String(filePath);
        std::error_code ec;
        if (!std::filesystem::exists(filePath, ec))
        {
            outError = "Required NorthstarRuntime shader file not found: '" + filePathUTF8 + "' (for '" + entry.spec.idName + "').";
            return false;
        }
        if (!std::filesystem::is_regular_file(filePath, ec))
        {
            outError = "NorthstarRuntime shader path is not a regular file: '" + filePathUTF8 + "'.";
            return false;
        }

        const auto fileSize = std::filesystem::file_size(filePath, ec);
        if (ec || fileSize == 0 || fileSize > 64u * 1024u * 1024u)
        {
            outError = "NorthstarRuntime shader file '" + filePathUTF8 + "' is empty or has an invalid file size.";
            return false;
        }

        std::ifstream fileStream(filePath, std::ios::binary);
        if (!fileStream.is_open())
        {
            outError = "Failed to open NorthstarRuntime shader file for reading: '" + filePathUTF8 + "'.";
            return false;
        }

        std::vector<BYTE> binaryData(static_cast<SIZE_T>(fileSize));
        if (!fileStream.read(reinterpret_cast<char*>(binaryData.data()), static_cast<std::streamsize>(fileSize)))
        {
            outError = "Failed to read bytes from NorthstarRuntime shader file: '" + filePathUTF8 + "'.";
            return false;
        }

        Shader* shader = new Shader();
        if (!ShaderLoader::LoadFromMemory(binaryData.data(), binaryData.size(), *shader, entry.spec.idName) || !shader->IsValid())
        {
            delete shader;
            outError = "Failed to parse NorthstarRuntime shader '" + entry.spec.idName + "' from '" + filePathUTF8 + "': corrupted or unsupported .nsc binary.";
            return false;
        }

        if (shader->GetStage() != entry.spec.expectedStage)
        {
            delete shader;
            outError = "NorthstarRuntime shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' has unexpected stage.";
            return false;
        }

        if (shader->GetTargetAPI() != m_backend->GetTargetAPI())
        {
            delete shader;
            outError = "NorthstarRuntime shader '" + entry.spec.idName + "' at '" + filePathUTF8 + "' TargetAPI does not match active backend TargetAPI.";
            return false;
        }

        const ShaderReflection& refl = shader->GetReflection();
        for (const std::string& reqCb : entry.spec.requiredConstantBuffers)
        {
            if (!ReflectionHasConstantBuffer(refl, reqCb))
            {
                delete shader;
                outError = "NorthstarRuntime shader '" + entry.spec.idName + "' is missing required ConstantBuffer '" + reqCb + "' in reflection.";
                return false;
            }
        }

        for (const std::string& reqSrv : entry.spec.requiredSRVStructuredBufs)
        {
            if (!ReflectionHasSRVStructuredBuffer(refl, reqSrv))
            {
                delete shader;
                outError = "NorthstarRuntime shader '" + entry.spec.idName + "' is missing required StructuredBuffer '" + reqSrv + "' in reflection.";
                return false;
            }
        }

        const UINT32 targetSM = SelectInternalShaderModel(m_backend);
        if (m_backend->GetTargetAPI() == TargetAPI::HLSL12 && !shader->HasBytecode() && shader->HasSourceCode())
        {
            ShaderLoader::TryPopulateCachedDXBC(*shader, targetSM);
        }

        NativeHandle nativeHandle = 0;
        if (shader->HasBytecode())
        {
            nativeHandle = m_backend->CreateShader(
                shader->GetStage(),
                shader->GetBytecodeData(),
                shader->GetBytecodeSize()
            );
        }
        else if (shader->HasSourceCode())
        {
            nativeHandle = m_backend->CreateShader(
                shader->GetStage(),
                shader->GetSourceCode().data(),
                shader->GetSourceCode().size()
            );
            if (nativeHandle != 0)
            {
                std::vector<BYTE> compiledDxbc;
                if (m_backend->GetShaderCompiledBytecode(nativeHandle, compiledDxbc) && !compiledDxbc.empty())
                {
                    ShaderLoader::StoreCachedDXBC(*shader, compiledDxbc.data(), compiledDxbc.size(), targetSM);
                    shader->SetBytecode(std::move(compiledDxbc));
                }
            }
        }
        shader->SetNativeHandle(nativeHandle);

        if (m_backend->GetTargetAPI() == TargetAPI::HLSL12 && nativeHandle == 0)
        {
            delete shader;
            outError = "Backend failed to compile/create native shader for NorthstarRuntime resource '" + entry.spec.idName + "'.";
            return false;
        }

        if (nativeHandle != 0)
        {
            if (shader->GetStage() == ShaderStage::VERTEX)
            {
                m_backend->SetShaderInputLayout(nativeHandle, shader->GetInputLayout());
            }
            const ShaderSpaceMask spaceMask = BuildInternalShaderSpaceMask(*shader);
            if (spaceMask.isExplicit)
            {
                m_backend->SetShaderReflectionSpaces(nativeHandle, spaceMask);
            }
        }

        entry.shader = shader;
        entry.loaded = true;
        return true;
    }

    BOOL InternalResourceManager::Initialize(NorthstarBackend* backend, CoreComputeManager* computeManager)
    {
        // Clean up any previous state if Initialize is called again
        ReleaseLoadedResources();
        m_initialized    = false;
        m_lastError.clear();
        m_moduleDirPath.clear();
        m_runtimeDirPath.clear();
        m_runtimeDirUTF8.clear();

        if (!backend || !computeManager)
        {
            m_lastError = "InternalResourceManager::Initialize requires non-null NorthstarBackend and CoreComputeManager.";
            return false;
        }

        m_backend        = backend;
        m_computeManager = computeManager;

        // 1. Resolve NorthstarCore's own module directory (never host executable directory or CWD)
        std::string resolveErr;
        if (!ResolveCoreModuleDirectory(m_moduleDirPath, resolveErr))
        {
            m_lastError = resolveErr;
            m_backend = nullptr;
            m_computeManager = nullptr;
            return false;
        }

        m_runtimeDirPath = m_moduleDirPath / "NorthstarRuntime";
        m_runtimeDirUTF8 = PathToUTF8String(m_runtimeDirPath);

        // 2. Verify NorthstarRuntime root directory exists and is a directory
        std::error_code ec;
        if (!std::filesystem::exists(m_runtimeDirPath, ec) || !std::filesystem::is_directory(m_runtimeDirPath, ec))
        {
            m_lastError = "NorthstarRuntime directory not found or inaccessible at '" + m_runtimeDirUTF8 +
                          "' (resolved relative to NorthstarCore module directory '" + PathToUTF8String(m_moduleDirPath) + "').";
            m_backend = nullptr;
            m_computeManager = nullptr;
            return false;
        }

        // 3. Load all MANDATORY internal compute shaders; FEATURE_DEPENDENT ones remain on-demand
        for (auto& entry : m_entries)
        {
            if (entry.spec.requirement == InternalResourceRequirement::MANDATORY)
            {
                std::string loadErr;
                if (!LoadEntry(entry, loadErr))
                {
                    m_lastError = loadErr;
                    ReleaseLoadedResources();
                    m_backend = nullptr;
                    m_computeManager = nullptr;
                    return false;
                }
            }
        }

        // 4. Load all MANDATORY internal graphics/depth shaders (under NorthstarRuntime/Depth, etc.)
        for (auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.spec.requirement == InternalResourceRequirement::MANDATORY)
            {
                std::string loadErr;
                if (!LoadGraphicsEntry(gEntry, loadErr))
                {
                    m_lastError = loadErr;
                    ReleaseLoadedResources();
                    m_backend = nullptr;
                    m_computeManager = nullptr;
                    return false;
                }
            }
        }

        m_initialized = true;
        return true;
    }

    void InternalResourceManager::ReleaseLoadedResources()
    {
        // Deterministic destruction order:
        // 1. Destroy internal ComputeKernels first while their referenced Shaders are still alive.
        for (auto& entry : m_entries)
        {
            if (entry.kernel)
            {
                if (m_computeManager)
                {
                    m_computeManager->DestroyInternalComputeKernel(entry.kernel);
                }
                else
                {
                    delete entry.kernel;
                }
                entry.kernel = nullptr;
            }
        }

        // 2. Destroy internal Compute Shaders and release their backend native handles.
        for (auto& entry : m_entries)
        {
            if (entry.shader)
            {
                if (m_backend && entry.shader->GetNativeHandle() != 0)
                {
                    m_backend->DestroyShader(entry.shader->GetNativeHandle());
                    entry.shader->SetNativeHandle(0);
                }
                delete entry.shader;
                entry.shader = nullptr;
            }
            entry.loaded = false;
        }

        // 3. Destroy internal Graphics/Depth Shaders and release their backend native handles.
        for (auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.shader)
            {
                if (m_backend && gEntry.shader->GetNativeHandle() != 0)
                {
                    m_backend->DestroyShader(gEntry.shader->GetNativeHandle());
                    gEntry.shader->SetNativeHandle(0);
                }
                delete gEntry.shader;
                gEntry.shader = nullptr;
            }
            gEntry.loaded = false;
        }
    }

    void InternalResourceManager::Shutdown()
    {
        ReleaseLoadedResources();
        m_initialized    = false;
        m_backend        = nullptr;
        m_computeManager = nullptr;
    }

    ComputeKernel* InternalResourceManager::GetComputeKernel(InternalComputeKernelId id) const
    {
        const UINT32 idx = static_cast<UINT32>(id);
        if (!m_initialized || idx >= m_entries.size())
            return nullptr;
        return m_entries[idx].kernel;
    }

    ComputeKernel* InternalResourceManager::GetComputeKernelByName(const std::string& idName) const
    {
        if (!m_initialized || idName.empty())
            return nullptr;
        for (const auto& entry : m_entries)
        {
            if (entry.spec.idName == idName)
                return entry.kernel;
        }
        return nullptr;
    }

    Shader* InternalResourceManager::GetShader(InternalComputeKernelId id) const
    {
        const UINT32 idx = static_cast<UINT32>(id);
        if (!m_initialized || idx >= m_entries.size())
            return nullptr;
        return m_entries[idx].shader;
    }

    Shader* InternalResourceManager::GetShaderByName(const std::string& idName) const
    {
        if (!m_initialized || idName.empty())
            return nullptr;
        for (const auto& entry : m_entries)
        {
            if (entry.spec.idName == idName)
                return entry.shader;
        }
        for (const auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.spec.idName == idName)
                return gEntry.shader;
        }
        return nullptr;
    }

    Shader* InternalResourceManager::GetGraphicsShader(InternalShaderId id) const
    {
        const UINT32 idx = static_cast<UINT32>(id);
        if (!m_initialized || idx >= m_graphicsEntries.size())
            return nullptr;
        return m_graphicsEntries[idx].shader;
    }

    Shader* InternalResourceManager::GetGraphicsShaderByName(const std::string& idName) const
    {
        if (!m_initialized || idName.empty())
            return nullptr;
        for (const auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.spec.idName == idName)
                return gEntry.shader;
        }
        return nullptr;
    }

    ComputeKernel* InternalResourceManager::AcquireFeatureKernel(const std::string& idName)
    {
        if (!m_initialized || idName.empty())
        {
            m_lastError = "Cannot acquire feature kernel '" + idName + "': InternalResourceManager is not initialized.";
            return nullptr;
        }

        for (auto& entry : m_entries)
        {
            if (entry.spec.idName == idName)
            {
                if (entry.loaded && entry.kernel && entry.kernel->IsValid())
                    return entry.kernel;

                std::string loadErr;
                if (!LoadEntry(entry, loadErr))
                {
                    m_lastError = loadErr;
                    return nullptr;
                }
                return entry.kernel;
            }
        }

        m_lastError = "Feature kernel '" + idName + "' is not registered in InternalResourceManager.";
        return nullptr;
    }

    ComputeKernel* InternalResourceManager::ReloadFeatureKernel(const std::string& idName, BOOL bypassDiskCache)
    {
        if (!m_initialized || idName.empty())
        {
            m_lastError = "Cannot reload feature kernel '" + idName + "': InternalResourceManager is not initialized.";
            return nullptr;
        }
        if (m_backend)
        {
            m_backend->WaitForGPU();
        }
        for (auto& entry : m_entries)
        {
            if (entry.spec.idName == idName)
            {
                if (entry.kernel)
                {
                    if (m_computeManager)
                    {
                        m_computeManager->DestroyInternalComputeKernel(entry.kernel);
                    }
                    else
                    {
                        delete entry.kernel;
                    }
                    entry.kernel = nullptr;
                }
                if (entry.shader)
                {
                    if (m_backend && entry.shader->GetNativeHandle() != 0)
                    {
                        m_backend->DestroyShader(entry.shader->GetNativeHandle());
                        entry.shader->SetNativeHandle(0);
                    }
                    delete entry.shader;
                    entry.shader = nullptr;
                }
                entry.loaded = false;

                std::string loadErr;
                if (!LoadEntry(entry, loadErr, bypassDiskCache))
                {
                    m_lastError = loadErr;
                    return nullptr;
                }
                return entry.kernel;
            }
        }
        m_lastError = "Feature kernel '" + idName + "' is not registered in InternalResourceManager.";
        return nullptr;
    }

    BOOL InternalResourceManager::RecompileLoadedNativeShaders(BOOL bypassDiskCache)
    {
        if (!m_initialized || !m_backend)
        {
            return false;
        }
        m_backend->WaitForGPU();
        const BackendCapabilities caps     = m_backend->GetCapabilities();
        const UINT32              targetSM = SelectInternalShaderModel(m_backend);

        auto recompileOne = [&](Shader* sh, const std::string& idName) -> BOOL {
            if (!sh || !sh->IsValid()) return true;
            if (sh->GetNativeHandle() != 0)
            {
                m_backend->DestroyShader(sh->GetNativeHandle());
                sh->SetNativeHandle(0);
            }
            const ShaderReflection& refl = sh->GetReflection();
            if (refl.UsesInt64Atomic() && !caps.SupportsNVGRasterAtomic64())
            {
                return true; // Intentionally stays unloaded (nativeHandle == 0) under restricted capability profile
            }
            if (refl.UsesInt64() && !(caps.maxShaderModel >= 0x60u && caps.supportsDXILRuntime && caps.supportsInt64ShaderOps))
            {
                return true;
            }
            if (!sh->HasSourceCode())
            {
                return false;
            }
            sh->SetBytecode(std::vector<BYTE>{});
            if (!bypassDiskCache && m_backend->GetTargetAPI() == TargetAPI::HLSL12)
            {
                ShaderLoader::TryPopulateCachedDXBC(*sh, targetSM);
            }
            NativeHandle newHandle = 0;
            if (sh->HasBytecode())
            {
                newHandle = m_backend->CreateShader(sh->GetStage(), sh->GetBytecodeData(), sh->GetBytecodeSize());
            }
            else
            {
                newHandle = m_backend->CreateShader(sh->GetStage(), sh->GetSourceCode().data(), sh->GetSourceCode().size());
                if (newHandle != 0)
                {
                    std::vector<BYTE> compiledDxbc;
                    if (m_backend->GetShaderCompiledBytecode(newHandle, compiledDxbc) && !compiledDxbc.empty())
                    {
                        if (!bypassDiskCache)
                        {
                            ShaderLoader::StoreCachedDXBC(*sh, compiledDxbc.data(), compiledDxbc.size(), targetSM);
                        }
                        sh->SetBytecode(std::move(compiledDxbc));
                    }
                }
            }
            sh->SetNativeHandle(newHandle);
            if (m_backend->GetTargetAPI() == TargetAPI::HLSL12 && newHandle == 0)
            {
                m_lastError = "Failed to recompile internal shader '" + idName + "' for active capabilities.";
                return false;
            }
            if (newHandle != 0)
            {
                if (sh->GetStage() == ShaderStage::VERTEX)
                {
                    m_backend->SetShaderInputLayout(newHandle, sh->GetInputLayout());
                }
                const ShaderSpaceMask spaceMask = BuildInternalShaderSpaceMask(*sh);
                if (spaceMask.isExplicit)
                {
                    m_backend->SetShaderReflectionSpaces(newHandle, spaceMask);
                }
            }
            return true;
        };

        for (auto& entry : m_entries)
        {
            if (entry.loaded && entry.shader)
            {
                if (!recompileOne(entry.shader, entry.spec.idName))
                {
                    return false;
                }
            }
        }
        for (auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.loaded && gEntry.shader)
            {
                if (!recompileOne(gEntry.shader, gEntry.spec.idName))
                {
                    return false;
                }
            }
        }
        return true;
    }

    Shader* InternalResourceManager::AcquireFeatureGraphicsShader(const std::string& idName)
    {
        if (!m_initialized || idName.empty())
        {
            m_lastError = "Cannot acquire feature graphics shader '" + idName + "': InternalResourceManager is not initialized.";
            return nullptr;
        }

        for (auto& gEntry : m_graphicsEntries)
        {
            if (gEntry.spec.idName == idName)
            {
                if (gEntry.loaded && gEntry.shader && gEntry.shader->IsValid())
                    return gEntry.shader;

                std::string loadErr;
                if (!LoadGraphicsEntry(gEntry, loadErr))
                {
                    m_lastError = loadErr;
                    return nullptr;
                }
                return gEntry.shader;
            }
        }

        m_lastError = "Feature graphics shader '" + idName + "' is not registered in InternalResourceManager.";
        return nullptr;
    }
}
