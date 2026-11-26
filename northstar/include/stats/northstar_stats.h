#pragma once

#include "../types/northstar_types.h"
#include "../nvg/nvg_types.h"

namespace Northstar
{
    // ==================== RENDER STATS ====================
    struct RenderStats
    {
        UINT64  frameIndex        = 0;
        UINT32  drawCalls          = 0;
        UINT32  geometryDrawCalls  = 0;
        UINT32  instancedDrawCalls = 0;
        UINT32  instanceCount      = 0;
        UINT32  staticBatchedCalls = 0;
        UINT32  triangleCount      = 0;
        UINT32  vertexCount        = 0;
        UINT32  visibleObjects     = 0;
        UINT32  culledObjects      = 0;
        UINT32  shadowCasters      = 0;
        UINT32  shaderSetSwitches  = 0;
        UINT32  passCount          = 0;
        FLOAT32 frameTimeMs        = 0.0f;
        FLOAT32 fps                = 0.0f;
        FLOAT32 deltaTimeSeconds   = 0.0f;
        FLOAT64 elapsedTimeSeconds = 0.0;

        // Shadow & Pipeline Profiling Metrics
        FLOAT32 shadowPrepCpuMs            = 0.0f;
        FLOAT32 drawCmdGenCpuMs            = 0.0f;
        FLOAT32 fenceWaitCpuMs             = 0.0f;
        FLOAT32 cmdRecordCpuMs             = 0.0f;
        FLOAT32 cmdSubmitCpuMs             = 0.0f;
        FLOAT32 presentCpuMs               = 0.0f;
        FLOAT32 gpuShadowPassMs            = 0.0f;
        FLOAT32 gpuDepthPrepassMs          = 0.0f;
        FLOAT32 gpuGeometryPassMs          = 0.0f;
        UINT32  depthPrepassDrawCalls      = 0;
        UINT32  logicalTileCount           = 0;
        UINT32  physicalTileCount          = 0;
        UINT32  renderedTileCount          = 0;
        UINT32  skippedTileCount           = 0;
        UINT32  shadowDrawCalls            = 0;
        UINT32  descriptorTablesUsed       = 0;
        FLOAT32 shadowPoolVramMB           = 0.0f;

        // GPU-Driven Frame Scheduling, Synchronization & Profiling Metrics
        FLOAT32 gpuSceneDirtyCpuMs         = 0.0f;
        FLOAT32 gpuSceneRebuildCpuMs       = 0.0f;
        FLOAT32 gpuSceneUploadCpuMs        = 0.0f;
        FLOAT32 gpuBufferUploadRecordCpuMs = 0.0f;
        FLOAT32 computePrepCpuMs           = 0.0f;
        FLOAT32 indirectPrepCpuMs          = 0.0f;
        FLOAT32 depthPrepassPrepCpuMs      = 0.0f;
        FLOAT32 gpuSceneUploadMs           = 0.0f;
        FLOAT32 gpuComputePassMs           = 0.0f;
        FLOAT32 gpuNVSFeedbackMs           = 0.0f;
        FLOAT32 gpuNVSResidencyCullMs      = 0.0f;
        FLOAT32 gpuLayeredDepthPassMs      = 0.0f;
        FLOAT32 gpuTotalFrameMs            = 0.0f;
        FLOAT32 nvsMetadataVramMB          = 0.0f;
        UINT32  nvsActiveShadowLightCount  = 0;
        UINT32  nvsActiveShadowViewCount   = 0;
        UINT32  gpuSceneUploadedBytes      = 0;
        UINT32  gpuSceneCoalescedRanges    = 0;
        UINT32  gpuSceneCapacitySlots      = 0;
        UINT32  gpuSceneActiveSlots        = 0;
        UINT32  gpuSceneInvalidSlots       = 0;
        UINT32  gpuSceneBufferResizes      = 0;
        UINT32  gpuBufferUploadedBytes     = 0;
        UINT32  gpuBufferAllocations       = 0;
        UINT32  pendingRetiredResources             = 0;
        UINT32  framesInFlightCount                 = 0;
        UINT32  skippedMissingSkinningObjects       = 0;
        UINT32  reflectionMaterializationsThisFrame = 0;
        UINT32  reflectionMaterializationCacheHits  = 0;
        FLOAT32 reflectionMaterializationVramMB     = 0.0f;

        // Northstar Virtual Geometry (NVG) Telemetry
        NVGStats nvgStats{};

        void Reset()
        {
            nvgStats.ResetPerFrame();
            drawCalls                     = 0;
            geometryDrawCalls             = 0;
            skippedMissingSkinningObjects = 0;
            instancedDrawCalls         = 0;
            instanceCount              = 0;
            staticBatchedCalls         = 0;
            triangleCount              = 0;
            vertexCount                = 0;
            visibleObjects             = 0;
            culledObjects              = 0;
            shadowCasters              = 0;
            shaderSetSwitches          = 0;
            passCount                  = 0;
            frameTimeMs                = 0.0f;
            fps                        = 0.0f;
            shadowPrepCpuMs            = 0.0f;
            drawCmdGenCpuMs            = 0.0f;
            fenceWaitCpuMs             = 0.0f;
            cmdRecordCpuMs             = 0.0f;
            cmdSubmitCpuMs             = 0.0f;
            presentCpuMs               = 0.0f;
            gpuShadowPassMs            = 0.0f;
            gpuDepthPrepassMs          = 0.0f;
            gpuGeometryPassMs          = 0.0f;
            depthPrepassDrawCalls      = 0;
            logicalTileCount           = 0;
            physicalTileCount          = 0;
            renderedTileCount          = 0;
            skippedTileCount           = 0;
            shadowDrawCalls            = 0;
            descriptorTablesUsed       = 0;
            shadowPoolVramMB           = 0.0f;
            gpuSceneDirtyCpuMs         = 0.0f;
            gpuSceneRebuildCpuMs       = 0.0f;
            gpuSceneUploadCpuMs        = 0.0f;
            gpuBufferUploadRecordCpuMs = 0.0f;
            computePrepCpuMs           = 0.0f;
            indirectPrepCpuMs          = 0.0f;
            depthPrepassPrepCpuMs      = 0.0f;
            gpuSceneUploadMs           = 0.0f;
            gpuComputePassMs           = 0.0f;
            gpuNVSFeedbackMs           = 0.0f;
            gpuNVSResidencyCullMs      = 0.0f;
            gpuLayeredDepthPassMs      = 0.0f;
            gpuTotalFrameMs            = 0.0f;
            nvsMetadataVramMB          = 0.0f;
            nvsActiveShadowLightCount  = 0;
            nvsActiveShadowViewCount   = 0;
            gpuSceneUploadedBytes      = 0;
            gpuSceneCoalescedRanges    = 0;
            gpuSceneCapacitySlots      = 0;
            gpuSceneActiveSlots        = 0;
            gpuSceneInvalidSlots       = 0;
            gpuSceneBufferResizes      = 0;
            gpuBufferUploadedBytes     = 0;
            gpuBufferAllocations       = 0;
            pendingRetiredResources             = 0;
            framesInFlightCount                 = 0;
            reflectionMaterializationsThisFrame = 0;
            reflectionMaterializationCacheHits  = 0;
            reflectionMaterializationVramMB     = 0.0f;
        }
    };
}
