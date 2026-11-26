#include "core_render_pipeline.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <chrono>

namespace Northstar
{
    namespace
    {
        BOOL DoesOpaqueMaterialRequireDepthPrepass(const Material* mat, const Texture* activeDepthTex) noexcept
        {
            if (!mat || mat->IsTransparent() || !mat->GetDepthState().test)
            {
                return false;
            }
            if (mat->HasPass(PassType::DEPTH) && mat->GetDepthShaderSet() != nullptr)
            {
                return true;
            }
            if (!activeDepthTex)
            {
                return false;
            }
            const NativeHandle activeDepthHandle = activeDepthTex->GetNativeHandle();
            const GraphicsShaderSlot slots[5] = {
                GraphicsShaderSlot::VERTEX,
                GraphicsShaderSlot::PIXEL,
                GraphicsShaderSlot::GEOMETRY,
                GraphicsShaderSlot::HULL,
                GraphicsShaderSlot::DOMAIN_STAGE
            };
            for (UINT32 si = 0u; si < 5u; ++si)
            {
                const ShaderStageProps& props = mat->GetPropertyStore().Get(PassType::GEOMETRY, slots[si]);
                for (const auto& t : props.textures)
                {
                    if (t.hasDestroyedResource)
                    {
                        continue;
                    }
                    if (t.texture == activeDepthTex ||
                        (activeDepthHandle != 0 && t.textureHandle == activeDepthHandle))
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    void CoreRenderPipeline::ApplyNVSSamplingPhase(
        Scene* scene,
        FrameView& ioView,
        const FrameNVSSamplingContext* nvsContext)
    {
        if (!nvsContext)
        {
            return;
        }

        ioView.nvsShadowMapTexture          = nvsContext->shadowMapTexture;
        ioView.nvsComparisonSampler         = nvsContext->comparisonSampler;
        ioView.nvsDepthSampler              = nvsContext->depthSampler;
        ioView.nvsSparsePageHierarchyBuffer = nvsContext->sparsePageHierarchyBuffer;
        ioView.nvsSparsePageHierarchyCount  = nvsContext->sparsePageHierarchyElementCount;
        ioView.nvsSparseRootCapacity        = nvsContext->sparseRootCapacity;
        ioView.nvsSparseDirStride           = nvsContext->sparseDirStride;
        ioView.nvsSparseDirCapacity         = nvsContext->sparseDirCapacity;
        const FLOAT32 sparseEnableFlag      = (nvsContext->sparsePageHierarchyBuffer != 0 && nvsContext->sparseRootCapacity > 0u) ? 1.0f : 0.0f;
        if (nvsContext->screenPixelWorldScale.x != 0.0f ||
            nvsContext->screenPixelWorldScale.y != 0.0f ||
            nvsContext->screenPixelWorldScale.z != 0.0f ||
            nvsContext->screenPixelWorldScale.w != 0.0f)
        {
            ioView.nvsScreenPixelWorldScale   = nvsContext->screenPixelWorldScale;
            ioView.nvsScreenPixelWorldScale.w = sparseEnableFlag;
        }
        else
        {
            const FLOAT32 effH = (ioView.camera.viewportHeight > 0.0f) ? ioView.camera.viewportHeight : 1080.0f;
            if (std::fabs(ioView.camera.projectionMatrix.m[15] - 1.0f) < 1e-4f &&
                std::fabs(ioView.camera.projectionMatrix.m[11]) < 1e-6f)
            {
                const FLOAT32 orthoH = (std::fabs(ioView.camera.projectionMatrix.m[5]) > 1e-6f)
                    ? (2.0f / std::fabs(ioView.camera.projectionMatrix.m[5]))
                    : 10.0f;
                ioView.nvsScreenPixelWorldScale = Vec4(0.0f, orthoH / (std::max)(effH, 1.0f), 0.0f, sparseEnableFlag);
            }
            else
            {
                const FLOAT32 fovRad = ((ioView.camera.fov > 1.0f) ? ioView.camera.fov : 60.0f) * DEG2RAD;
                ioView.nvsScreenPixelWorldScale = Vec4((2.0f * std::tan(0.5f * fovRad)) / (std::max)(effH, 1.0f), 0.0f, 0.0f, sparseEnableFlag);
            }
        }
        ioView.nvsVirtualPageTableBuffer    = nvsContext->virtualPageTableBuffer;
        ioView.nvsVirtualPageTableCount     = nvsContext->virtualPageTableElementCount;
        ioView.nvsTotalVirtualPages         = nvsContext->virtualPageTableElementCount;
        ioView.nvsPhysicalPageTableBuffer   = nvsContext->physicalPageTableBuffer;
        ioView.nvsPhysicalPageTableCount    = nvsContext->physicalPageTableElementCount;
        ioView.nvsPagesPerAxis              = nvsContext->pagesPerAxis;
        ioView.nvsPhysicalPoolCapacity      = nvsContext->physicalPoolCapacity;
        ioView.nvsPageTexelSize             = nvsContext->pageTexelSize;
        ioView.nvsDepthMode                 = nvsContext->depthMode;

        ioView.nvsLightViewTableBuffer = nvsContext->opaqueLightViewTableBuffer;
        ioView.nvsLightViewTableCount  = nvsContext->opaqueLightViewTableElementCount;
        ioView.nvsViewTableCount       = nvsContext->opaqueActiveViewCount;
        ioView.nvsSamplingAvailable    = nvsContext->opaqueSamplingAvailable;
        ioView.nvsSnapshotFrameIndex   = nvsContext->opaqueSnapshotFrameIndex;

        if (nvsContext->opaqueViewTableCpuMirror != nullptr && nvsContext->opaqueViewTableCpuMirrorCount > 0u)
        {
            ioView.nvsLightViewTableCpuMirrorPtr   = reinterpret_cast<const BYTE*>(nvsContext->opaqueViewTableCpuMirror);
            ioView.nvsLightViewTableCpuMirrorCount = nvsContext->opaqueViewTableCpuMirrorCount;
            ioView.nvsLightViewTableCpuMirror.clear();
        }
        else
        {
            ioView.nvsLightViewTableCpuMirrorPtr   = nullptr;
            ioView.nvsLightViewTableCpuMirrorCount = 0u;
            ioView.nvsLightViewTableCpuMirror.clear();
        }

        if (scene)
        {
            const auto* assignPtr = (nvsContext->shadowMapTexture != 0) ? &nvsContext->opaqueLightAssignments : nullptr;
            ioView.lights.clear();
            BuildDirectionalLightData(scene, ioView.hasDirectionalLight, ioView.directionalLight, assignPtr);
            BuildLightsData(scene, ioView.lights, assignPtr);
            ioView.lightsDataRevision = ++m_nextLightsDataRevision;
            m_cachedLightsPayloads.clear();
        }
    }

    void CoreRenderPipeline::BuildFrameView(Scene* scene, Camera* camera, const NativeRenderTarget& defaultRenderTarget,
                                           RenderStats& stats, FrameView& outView,
                                           const FrameNVSSamplingContext* nvsContext,
                                           std::unordered_set<const ComputeKernel*>* executedPerFrameKernels,
                                           const InternalResourceManager* internalResources)
    {
        outView.Clear();
        outView.frameIndex           = stats.frameIndex;
        outView.deltaTime            = stats.deltaTimeSeconds;
        outView.elapsedTime          = static_cast<FLOAT32>(stats.elapsedTimeSeconds);
        outView.hasAuthoritativeTime = true;
        BuildCameraData(camera, defaultRenderTarget, outView.camera);
        outView.renderTarget      = outView.camera.renderTarget;
        outView.sceneDepthTexture = outView.camera.renderTarget.depthStencilTexture;
        if (scene)
        {
            GPUBuffer* gpuSceneBuf           = scene->GetGPUSceneBuffer();
            GPUBuffer* gpuSkinningBuf        = scene->GetGPUSkinningPaletteBuffer();
            const GPUSceneUpdateStats& gStat = scene->GetGPUSceneStats();
            outView.gpuSceneBuffer            = gpuSceneBuf ? gpuSceneBuf->GetNativeHandle() : 0;
            outView.gpuSceneCapacity          = gStat.capacitySlots;
            outView.gpuSceneHighWaterMark     = gStat.highWaterMark;
            outView.gpuSceneActiveObjectCount = gStat.activeObjectCount;
            outView.gpuSceneMutationEpoch     = gStat.mutationEpoch;
            outView.gpuSkinningPaletteBuffer  = gpuSkinningBuf ? gpuSkinningBuf->GetNativeHandle() : 0;
            outView.gpuSkinningPaletteCount   = scene->GetGPUSkinningPaletteCount();
        }

        BuildEnvironmentData(scene, outView.environment);
        if (nvsContext)
        {
            ApplyNVSSamplingPhase(scene, outView, nvsContext);
        }
        else
        {
            BuildDirectionalLightData(scene, outView.hasDirectionalLight, outView.directionalLight, nullptr);
            BuildLightsData(scene, outView.lights, nullptr);
            outView.lightsDataRevision = ++m_nextLightsDataRevision;
            m_cachedLightsPayloads.clear();
        }

        const Vec3 cameraPos = outView.camera.position;

        if (!scene || !camera)
        {
            stats.drawCalls         = 0;
            stats.geometryDrawCalls = 0;
            stats.instancedDrawCalls= 0;
            stats.instanceCount     = 0;
            stats.staticBatchedCalls= 0;
            stats.triangleCount     = 0;
            stats.vertexCount       = 0;
            stats.visibleObjects    = 0;
            stats.culledObjects     = 0;
            stats.shadowCasters     = 0;
            stats.passCount         = 0;
            return;
        }

        const GPUSceneUpdateStats& gpuSceneSt = scene->GetGPUSceneStats();
        const BOOL pureVirtualizedScene =
            (gpuSceneSt.activeObjectCount > 0u) &&
            (gpuSceneSt.activeObjectCount == static_cast<UINT32>(scene->GetRenderObjects().size())) &&
            (gpuSceneSt.virtualActiveObjectCount == gpuSceneSt.activeObjectCount) &&
            (gpuSceneSt.materialKernelObjectCount == 0u);

        std::vector<RenderObject*> visibleObjects;
        if (!pureVirtualizedScene)
        {
            visibleObjects = camera ? scene->GetVisibleObjects(camera) : scene->GetRenderObjects();
            stats.visibleObjects = static_cast<UINT32>(visibleObjects.size());
            stats.culledObjects = (scene->GetRenderObjects().size() >= visibleObjects.size()) ?
                static_cast<UINT32>(scene->GetRenderObjects().size() - visibleObjects.size()) : 0;
        }
        else
        {
            stats.visibleObjects = 0u;
            stats.culledObjects  = 0u;
        }
        stats.shadowCasters = 0;

        const auto& sceneKernelEntries = scene->GetComputeKernelEntries();
        auto isRegisteredInScene = [&](const ComputeKernel* k) -> BOOL {
            if (!k) return false;
            for (const auto& entry : sceneKernelEntries)
            {
                if (entry.kernel == k) return true;
            }
            return false;
        };

        auto canExecuteInThisView = [&](const ComputeKernel* k, ComputeExecutionScope scope) -> BOOL {
            if (!k || !k->IsValid()) return false;
            if (scope == ComputeExecutionScope::PER_FRAME && executedPerFrameKernels != nullptr)
            {
                if (executedPerFrameKernels->find(k) != executedPerFrameKernels->end())
                {
                    return false;
                }
                executedPerFrameKernels->insert(k);
            }
            return true;
        };

        std::vector<const ComputeKernel*> prePassKernels;
        std::vector<const ComputeKernel*> postDepthKernels;
        std::vector<const ComputeKernel*> postOpaqueKernels;
        std::vector<const ComputeKernel*> postPassKernels;

        auto addUniqueKernel = [](std::vector<const ComputeKernel*>& list, const ComputeKernel* k) {
            if (!k) return;
            for (const ComputeKernel* existing : list)
            {
                if (existing == k) return;
            }
            list.push_back(k);
        };

        for (const auto& entry : sceneKernelEntries)
        {
            const ComputeKernel* k = entry.kernel;
            if (!k || !k->IsValid()) continue;
            if (!canExecuteInThisView(k, entry.ResolveScope())) continue;

            switch (entry.phase)
            {
                case ComputePassPhase::PRE_PASS:
                    addUniqueKernel(prePassKernels, k);
                    break;
                case ComputePassPhase::POST_DEPTH:
                    addUniqueKernel(postDepthKernels, k);
                    break;
                case ComputePassPhase::POST_OPAQUE:
                    addUniqueKernel(postOpaqueKernels, k);
                    break;
                case ComputePassPhase::POST_PASS:
                    addUniqueKernel(postPassKernels, k);
                    break;
                default:
                    break;
            }
        }

        std::vector<RenderObject*> opaqueObjects;
        std::vector<RenderObject*> transparentObjects;
        opaqueObjects.reserve(visibleObjects.size());
        for (SIZE_T i = 0; i < visibleObjects.size(); ++i)
        {
            RenderObject* obj = visibleObjects[i];
            if (!obj) continue;

            Material* mat = obj->GetMaterial();
            const BOOL isVirtualizedObj = (obj->GetMesh() != nullptr && obj->GetMesh()->IsVirtualized());
            if (mat)
            {
                if (mat->HasComputeKernels())
                {
                    for (ComputeKernel* k : mat->GetComputeKernels())
                    {
                        if (!k || !k->IsValid()) continue;
                        // Scene registration takes precedence over Material's implicit PRE_PASS scheduling
                        if (isRegisteredInScene(k)) continue;
                        BOOL alreadyQueued = false;
                        for (const ComputeKernel* existing : prePassKernels)
                        {
                            if (existing == k) { alreadyQueued = true; break; }
                        }
                        if (!alreadyQueued && canExecuteInThisView(k, k->GetExecutionScope()))
                        {
                            prePassKernels.push_back(k);
                        }
                    }
                }
                if (isVirtualizedObj)
                {
                    continue; // Handled fully on GPU by Northstar Virtual Geometry (NVG) — 0 CPU draw calls
                }
                if (mat->IsTransparent())
                {
                    transparentObjects.push_back(obj);
                }
                else
                {
                    opaqueObjects.push_back(obj);
                }
            }
            else if (!isVirtualizedObj)
            {
                opaqueObjects.push_back(obj);
            }
        }

        if (!prePassKernels.empty())
        {
            const auto tCompStart = std::chrono::high_resolution_clock::now();
            for (const ComputeKernel* k : prePassKernels)
            {
                const auto tSingleStart = std::chrono::high_resolution_clock::now();
                ComputeDispatchCommand dispatchCmd{};
                std::vector<ComputeDispatchCommand> prepDispatches;
                if (BuildComputeDispatchCommand(k, outView, dispatchCmd, &prepDispatches))
                {
                    const auto tSingleEnd = std::chrono::high_resolution_clock::now();
                    if (dispatchCmd.indirectArgsBuffer != 0)
                    {
                        stats.indirectPrepCpuMs += std::chrono::duration<float, std::milli>(tSingleEnd - tSingleStart).count();
                    }
                    for (auto& pd : prepDispatches)
                    {
                        outView.prePassComputeDispatches.push_back(std::move(pd));
                    }
                    outView.prePassComputeDispatches.push_back(std::move(dispatchCmd));
                }
            }
            const auto tCompEnd = std::chrono::high_resolution_clock::now();
            stats.computePrepCpuMs += std::chrono::duration<float, std::milli>(tCompEnd - tCompStart).count();
        }

        const auto tOpaqueCmdStart = std::chrono::high_resolution_clock::now();
        BatchAndCategorizeDrawCommands(scene, opaqueObjects, PassType::GEOMETRY, outView, cameraPos, stats, outView.geometryPass.opaqueCommands, nullptr, &outView.prePassComputeDispatches);
        const auto tOpaqueCmdEnd = std::chrono::high_resolution_clock::now();
        stats.drawCmdGenCpuMs += std::chrono::duration<float, std::milli>(tOpaqueCmdEnd - tOpaqueCmdStart).count();

        std::sort(outView.geometryPass.opaqueCommands.begin(), outView.geometryPass.opaqueCommands.end(),
                  [](const DrawCommand& a, const DrawCommand& b) {
                      if (a.sortKey != b.sortKey)
                          return a.sortKey < b.sortKey;
                      if (a.bufferCategory != b.bufferCategory)
                          return static_cast<UINT8>(a.bufferCategory) < static_cast<UINT8>(b.bufferCategory);
                      return a.depthKey < b.depthKey;
                  });

        const Texture* activeDepthTex = camera ? camera->GetSceneDepthTexture() : nullptr;
        BOOL opaqueMaterialRequiresPrepass = false;
        {
            std::unordered_set<const Material*> checkedPrepassMats;
            for (RenderObject* obj : opaqueObjects)
            {
                if (!obj) continue;
                const Material* mat = obj->GetMaterial();
                if (mat && checkedPrepassMats.insert(mat).second &&
                    DoesOpaqueMaterialRequireDepthPrepass(mat, activeDepthTex))
                {
                    opaqueMaterialRequiresPrepass = true;
                    break;
                }
            }
            if (!opaqueMaterialRequiresPrepass && gpuSceneSt.virtualActiveObjectCount > 0u)
            {
                for (RenderObject* obj : scene->GetRenderObjects())
                {
                    if (!obj || !obj->HasFlag(RenderObjectFlags::VISIBLE)) continue;
                    const Mesh* m = obj->GetMesh();
                    if (!m || !m->IsVirtualized()) continue;
                    const Material* mat = obj->GetMaterial();
                    if (mat && checkedPrepassMats.insert(mat).second &&
                        DoesOpaqueMaterialRequireDepthPrepass(mat, activeDepthTex))
                    {
                        opaqueMaterialRequiresPrepass = true;
                        break;
                    }
                }
            }
        }

        const BOOL hasValidSceneDepthTarget = (outView.sceneDepthTexture != 0) || (activeDepthTex != nullptr);
        const BOOL depthPrepassActive =
            hasValidSceneDepthTarget &&
            (!postDepthKernels.empty() ||
             (nvsContext != nullptr && nvsContext->requiresDepthPrepass) ||
             opaqueMaterialRequiresPrepass);
        outView.hasDepthPrepass = depthPrepassActive;
        const BOOL useInterleavedPasses = depthPrepassActive || !postOpaqueKernels.empty();

        auto accumulateCmds = [&](const std::vector<DrawCommand>& cmds, BOOL isGeometryPass) {
            for (SIZE_T i = 0; i < cmds.size(); ++i)
            {
                stats.drawCalls++;
                if (isGeometryPass)
                {
                    stats.geometryDrawCalls++;
                }
                stats.vertexCount += cmds[i].vertexCount * cmds[i].instanceCount;
                stats.triangleCount += ((cmds[i].indexCount > 0) ? (cmds[i].indexCount / 3) : (cmds[i].vertexCount / 3)) * cmds[i].instanceCount;
            }
        };

        stats.passCount = 0;
        stats.drawCalls = 0;
        stats.geometryDrawCalls = 0;
        stats.triangleCount = 0;
        stats.vertexCount = 0;

        if (!outView.geometryPass.opaqueCommands.empty())
        {
            stats.passCount++;
            accumulateCmds(outView.geometryPass.opaqueCommands, true);
        }

        if (useInterleavedPasses)
        {
            if (depthPrepassActive)
            {
                const auto tDepthCmdStart = std::chrono::high_resolution_clock::now();
                FramePassItem depthPrepassItem{};
                depthPrepassItem.type = FramePassItemType::DEPTH_PREPASS;

                std::vector<RenderObject*> depthOpaqueObjects;
                depthOpaqueObjects.reserve(opaqueObjects.size());
                for (RenderObject* obj : opaqueObjects)
                {
                    if (!obj) continue;
                    Material* mat = obj->GetMaterial();
                    if (mat && (!mat->GetDepthState().test || !mat->GetDepthState().write))
                        continue;
                    depthOpaqueObjects.push_back(obj);
                }

                RenderStats depthDummyStats{};
                BatchAndCategorizeDrawCommands(
                    scene,
                    depthOpaqueObjects,
                    PassType::GEOMETRY,
                    outView,
                    cameraPos,
                    depthDummyStats,
                    depthPrepassItem.drawCommands,
                    internalResources,
                    &outView.prePassComputeDispatches);

                std::sort(depthPrepassItem.drawCommands.begin(), depthPrepassItem.drawCommands.end(),
                          [](const DrawCommand& a, const DrawCommand& b) {
                              if (a.sortKey != b.sortKey)
                                  return a.sortKey < b.sortKey;
                              if (a.bufferCategory != b.bufferCategory)
                                  return static_cast<UINT8>(a.bufferCategory) < static_cast<UINT8>(b.bufferCategory);
                              return a.depthKey < b.depthKey;
                          });

                const auto tDepthCmdEnd = std::chrono::high_resolution_clock::now();
                const float depthPrepMs = std::chrono::duration<float, std::milli>(tDepthCmdEnd - tDepthCmdStart).count();
                stats.depthPrepassPrepCpuMs += depthPrepMs;
                stats.drawCmdGenCpuMs       += depthPrepMs;

                if (!depthPrepassItem.drawCommands.empty())
                {
                    stats.passCount++;
                    stats.depthPrepassDrawCalls += static_cast<UINT32>(depthPrepassItem.drawCommands.size());
                    accumulateCmds(depthPrepassItem.drawCommands, false);
                    outView.interleavedPassItems.push_back(std::move(depthPrepassItem));
                }
            }

            if (outView.IsSceneDepthValidAtPhase(ComputePassPhase::POST_DEPTH))
            {
                const auto tCompStart = std::chrono::high_resolution_clock::now();
                for (const ComputeKernel* k : postDepthKernels)
                {
                    const auto tSingleStart = std::chrono::high_resolution_clock::now();
                    ComputeDispatchCommand dispatchCmd{};
                    std::vector<ComputeDispatchCommand> prepDispatches;
                    if (BuildComputeDispatchCommand(k, outView, dispatchCmd, &prepDispatches))
                    {
                        const auto tSingleEnd = std::chrono::high_resolution_clock::now();
                        if (dispatchCmd.indirectArgsBuffer != 0)
                        {
                            stats.indirectPrepCpuMs += std::chrono::duration<float, std::milli>(tSingleEnd - tSingleStart).count();
                        }
                        for (auto& pd : prepDispatches)
                        {
                            FramePassItem prepItem{};
                            prepItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                            prepItem.computeDispatch = std::move(pd);
                            outView.interleavedPassItems.push_back(std::move(prepItem));
                        }
                        FramePassItem item{};
                        item.type            = FramePassItemType::COMPUTE_DISPATCH;
                        item.computeDispatch = std::move(dispatchCmd);
                        outView.interleavedPassItems.push_back(std::move(item));
                    }
                }
                const auto tCompEnd = std::chrono::high_resolution_clock::now();
                stats.computePrepCpuMs += std::chrono::duration<float, std::milli>(tCompEnd - tCompStart).count();
            }

            if (!outView.geometryPass.opaqueCommands.empty() || gpuSceneSt.virtualActiveObjectCount > 0u)
            {
                if (depthPrepassActive)
                {
                    for (DrawCommand& opaqueCmd : outView.geometryPass.opaqueCommands)
                    {
                        if (opaqueCmd.depthState.test && opaqueCmd.depthState.write)
                        {
                            opaqueCmd.depthState.write = false;
                            if (opaqueCmd.depthState.compare == DepthCompare::LESS)
                            {
                                opaqueCmd.depthState.compare = DepthCompare::LESS_EQUAL;
                            }
                        }
                    }
                }

                FramePassItem opaqueItem{};
                opaqueItem.type         = FramePassItemType::GRAPHICS_DRAWS;
                opaqueItem.drawCommands = std::move(outView.geometryPass.opaqueCommands);
                outView.geometryPass.opaqueCommands.clear();
                outView.interleavedPassItems.push_back(std::move(opaqueItem));
            }
        }

        const auto tTransCmdStart = std::chrono::high_resolution_clock::now();
        // Transparent objects MUST be rendered individually in strict back-to-front view-space order
        // (never merged into cross-object instanced batches that violate painter's algorithm across depth).
        outView.geometryPass.transparentCommands.reserve(transparentObjects.size());
        for (RenderObject* transObj : transparentObjects)
        {
            if (!transObj) continue;
            Mesh* effMesh = ResolveEffectiveMesh(transObj, outView.camera);
            Material* transMat = transObj->GetMaterial();
            if (!effMesh || !transMat) continue;

            const BOOL isSkinnedObj = transObj->HasFlag(RenderObjectFlags::SKINNED) || transObj->HasSkinningMatrices();
            if (!transObj->HasSkinningMatrices())
            {
                ShaderSet* ss   = transMat->GetEffectiveShaderSet(PassType::GEOMETRY, isSkinnedObj);
                Shader* checkVS = ss ? ss->GetVertexShader(isSkinnedObj ? ShaderVariant::SKINNED : ShaderVariant::STANDARD) : nullptr;
                if (checkVS && checkVS->IsValid() && CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(checkVS->GetReflection()))
                {
                    stats.skippedMissingSkinningObjects++;
                    continue;
                }
            }

            DrawCommand transCmd{};
            if (!BuildDrawCommand(transObj, PassType::GEOMETRY, outView, cameraPos, transCmd, ShaderVariant::COUNT, nullptr, &outView.prePassComputeDispatches, scene))
                continue;

            const UINT32 boneOff    = transObj->GetSkinningPaletteOffset(scene);
            const UINT32 boneCnt    = transObj->GetSkinningMatrixCount();
            const UINT32 packedSkin = (boneCnt > 0u) ? PackGPUSceneSkinnedBoneRange(boneOff, boneCnt) : 0u;
            transCmd.instanceSkinning.clear();
            transCmd.instanceSkinning.push_back({ boneOff, boneCnt, transObj->GetSkinningRevision() });
            transCmd.bufferCategory = BufferCategory::DYNAMIC_OBJECT;
            transCmd.instanceCount  = 1u;
            transCmd.startInstance  = 0u;
            transCmd.instances.clear();

            InstanceData inst{};
            inst.worldMatrix = transObj->GetWorldMatrix();
            inst.instanceId  = 0u;
            inst.customData  = packedSkin;
            transCmd.instances.push_back(inst);

            outView.geometryPass.transparentCommands.push_back(std::move(transCmd));
        }
        const auto tTransCmdEnd = std::chrono::high_resolution_clock::now();
        stats.drawCmdGenCpuMs += std::chrono::duration<float, std::milli>(tTransCmdEnd - tTransCmdStart).count();

        std::stable_sort(outView.geometryPass.transparentCommands.begin(), outView.geometryPass.transparentCommands.end(),
                         [](const DrawCommand& a, const DrawCommand& b) {
                             return a.depthKey > b.depthKey;
                         });

        if (!outView.geometryPass.transparentCommands.empty())
        {
            stats.passCount++;
            accumulateCmds(outView.geometryPass.transparentCommands, true);
        }

        if (useInterleavedPasses)
        {
            if (!postOpaqueKernels.empty())
            {
                const auto tCompStart = std::chrono::high_resolution_clock::now();
                for (const ComputeKernel* k : postOpaqueKernels)
                {
                    const auto tSingleStart = std::chrono::high_resolution_clock::now();
                    ComputeDispatchCommand dispatchCmd{};
                    std::vector<ComputeDispatchCommand> prepDispatches;
                    if (BuildComputeDispatchCommand(k, outView, dispatchCmd, &prepDispatches))
                    {
                        const auto tSingleEnd = std::chrono::high_resolution_clock::now();
                        if (dispatchCmd.indirectArgsBuffer != 0)
                        {
                            stats.indirectPrepCpuMs += std::chrono::duration<float, std::milli>(tSingleEnd - tSingleStart).count();
                        }
                        for (auto& pd : prepDispatches)
                        {
                            FramePassItem prepItem{};
                            prepItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                            prepItem.computeDispatch = std::move(pd);
                            outView.interleavedPassItems.push_back(std::move(prepItem));
                        }
                        FramePassItem item{};
                        item.type            = FramePassItemType::COMPUTE_DISPATCH;
                        item.computeDispatch = std::move(dispatchCmd);
                        outView.interleavedPassItems.push_back(std::move(item));
                    }
                }
                const auto tCompEnd = std::chrono::high_resolution_clock::now();
                stats.computePrepCpuMs += std::chrono::duration<float, std::milli>(tCompEnd - tCompStart).count();
            }

            if (!outView.geometryPass.transparentCommands.empty())
            {
                FramePassItem transItem{};
                transItem.type         = FramePassItemType::GRAPHICS_DRAWS;
                transItem.drawCommands = std::move(outView.geometryPass.transparentCommands);
                outView.geometryPass.transparentCommands.clear();
                outView.interleavedPassItems.push_back(std::move(transItem));
            }
        }

        if (!postPassKernels.empty())
        {
            const auto tCompStart = std::chrono::high_resolution_clock::now();
            for (const ComputeKernel* k : postPassKernels)
            {
                const auto tSingleStart = std::chrono::high_resolution_clock::now();
                ComputeDispatchCommand dispatchCmd{};
                std::vector<ComputeDispatchCommand> prepDispatches;
                if (BuildComputeDispatchCommand(k, outView, dispatchCmd, &prepDispatches))
                {
                    const auto tSingleEnd = std::chrono::high_resolution_clock::now();
                    if (dispatchCmd.indirectArgsBuffer != 0)
                    {
                        stats.indirectPrepCpuMs += std::chrono::duration<float, std::milli>(tSingleEnd - tSingleStart).count();
                    }
                    for (auto& pd : prepDispatches)
                    {
                        outView.postPassComputeDispatches.push_back(std::move(pd));
                    }
                    outView.postPassComputeDispatches.push_back(std::move(dispatchCmd));
                }
            }
            const auto tCompEnd = std::chrono::high_resolution_clock::now();
            stats.computePrepCpuMs += std::chrono::duration<float, std::milli>(tCompEnd - tCompStart).count();
        }
    }

    void CoreRenderPipeline::RefreshFrameViewSceneContext(Scene* scene, Camera* camera,
                                                          const NativeRenderTarget& defaultRenderTarget,
                                                          FrameView& ioView,
                                                          const FrameNVSSamplingContext* nvsContext)
    {
        BuildCameraData(camera, defaultRenderTarget, ioView.camera);
        ioView.renderTarget      = ioView.camera.renderTarget;
        ioView.sceneDepthTexture = ioView.camera.renderTarget.depthStencilTexture;
        if (scene)
        {
            GPUBuffer* gpuSceneBuf           = scene->GetGPUSceneBuffer();
            GPUBuffer* gpuSkinningBuf        = scene->GetGPUSkinningPaletteBuffer();
            const GPUSceneUpdateStats& gStat = scene->GetGPUSceneStats();
            ioView.gpuSceneBuffer            = gpuSceneBuf ? gpuSceneBuf->GetNativeHandle() : 0;
            ioView.gpuSceneCapacity          = gStat.capacitySlots;
            ioView.gpuSceneHighWaterMark     = gStat.highWaterMark;
            ioView.gpuSceneActiveObjectCount = gStat.activeObjectCount;
            ioView.gpuSceneMutationEpoch     = gStat.mutationEpoch;
            ioView.gpuSkinningPaletteBuffer  = gpuSkinningBuf ? gpuSkinningBuf->GetNativeHandle() : 0;
            ioView.gpuSkinningPaletteCount   = scene->GetGPUSkinningPaletteCount();
        }
        BuildEnvironmentData(scene, ioView.environment);
        if (nvsContext)
        {
            ApplyNVSSamplingPhase(scene, ioView, nvsContext);
        }
        else
        {
            ioView.lights.clear();
            BuildDirectionalLightData(scene, ioView.hasDirectionalLight, ioView.directionalLight, nullptr);
            BuildLightsData(scene, ioView.lights, nullptr);
            ioView.lightsDataRevision = ++m_nextLightsDataRevision;
            m_cachedLightsPayloads.clear();
        }
    }

    void CoreRenderPipeline::BuildCameraData(Camera* camera, const NativeRenderTarget& defaultRenderTarget, FrameCameraData& outCamera)
    {
        if (camera)
        {
            outCamera.viewMatrix           = camera->GetViewMatrix();
            outCamera.projectionMatrix     = camera->GetProjectionMatrix();
            outCamera.viewProjectionMatrix = camera->GetViewProjectionMatrix();
            outCamera.position             = camera->GetPosition();
            outCamera.rotation             = camera->GetRotation();
            outCamera.direction            = camera->GetDirection();
            outCamera.nearPlane            = camera->GetNearPlane();
            outCamera.farPlane             = camera->GetFarPlane();
            outCamera.fov                  = camera->GetFOV();
            outCamera.aspectRatio          = camera->GetAspectRatio();
            outCamera.frustum              = camera->GetFrustum();
            outCamera.viewportX            = camera->GetViewportX();
            outCamera.viewportY            = camera->GetViewportY();
            outCamera.viewportWidth        = camera->GetViewportWidth();
            outCamera.viewportHeight       = camera->GetViewportHeight();

            Texture* sceneDepthTex = camera->GetSceneDepthTexture();
            const NativeHandle sceneDepthHandle = sceneDepthTex ? sceneDepthTex->GetNativeHandle() : 0;

            if (camera->GetRenderType() == CameraRenderType::TO_TEXTURE)
            {
                outCamera.renderTarget.type                = RenderTargetType::TEXTURE;
                outCamera.renderTarget.colorTexture        = camera->GetRenderTarget() ? camera->GetRenderTarget()->GetNativeHandle() : 0;
                outCamera.renderTarget.depthStencilTexture = sceneDepthHandle;
                outCamera.renderTarget.surface             = nullptr;
                outCamera.renderTarget.context             = nullptr;
            }
            else
            {
                outCamera.renderTarget = defaultRenderTarget;
                if (sceneDepthHandle != 0)
                {
                    outCamera.renderTarget.depthStencilTexture = sceneDepthHandle;
                }
            }
        }
        else
        {
            outCamera.viewMatrix           = Mat4::Identity();
            outCamera.projectionMatrix     = Mat4::Identity();
            outCamera.viewProjectionMatrix = Mat4::Identity();
            outCamera.position             = Vec3(0.0f, 0.0f, 0.0f);
            outCamera.rotation             = Quat();
            outCamera.direction            = Vec3(0.0f, 0.0f, 1.0f);
            outCamera.renderTarget         = defaultRenderTarget;
            outCamera.viewportX            = 0.0f;
            outCamera.viewportY            = 0.0f;
            outCamera.viewportWidth        = 0.0f;
            outCamera.viewportHeight       = 0.0f;
        }
    }

    void CoreRenderPipeline::BuildEnvironmentData(Scene* scene, FrameEnvironmentData& outEnv)
    {
        if (!scene) return;

        outEnv.ambientColor     = scene->GetAmbientColor();
        outEnv.ambientIntensity = scene->GetAmbientIntensity();
    }

    static void PopulateFrameLightFromLight(
        const Light* light,
        const std::vector<LightNVSViewAssignment>* nvsAssignments,
        const std::unordered_map<const Light*, LightNVSViewAssignment>* nvsAssignmentMap,
        FrameLightData& outLd)
    {
        outLd = FrameLightData{};
        if (!light) return;

        outLd.type                 = light->GetType();
        outLd.color                = light->GetColor();
        outLd.intensity            = light->GetIntensity();
        outLd.position             = light->GetPosition();
        outLd.rotation             = light->GetRotation();
        outLd.direction            = light->GetDirection();
        outLd.range                = light->GetRange();
        outLd.attenuation          = Vec3(light->GetConstantAttenuation(), light->GetLinearAttenuation(), light->GetQuadraticAttenuation());
        outLd.innerConeAngle       = light->GetInnerConeAngle();
        outLd.outerConeAngle       = light->GetOuterConeAngle();
        outLd.viewProjectionMatrix = light->GetViewProjectionMatrix();
        outLd.shadowSoftness       = light->GetShadowSoftness();
        outLd.shadowBias           = light->GetShadowBias();
        outLd.shadowNormalBias     = light->GetShadowNormalBias();
        outLd.shadowNearPlane      = light->GetShadowNearPlane();
        outLd.shadowMaxDistance    = light->GetShadowMaxDistance();

        const BOOL wantsShadow = light->HasFlag(LightFlags::ENABLED) && light->HasFlag(LightFlags::CAST_SHADOW);
        outLd.castsShadow      = 0u;
        outLd.nvsLightId       = 0u;
        outLd.shadowViewBase   = 0u;
        outLd.shadowViewCount  = 0u;

        if (nvsAssignments != nullptr)
        {
            if (wantsShadow)
            {
                if (nvsAssignmentMap != nullptr)
                {
                    const auto it = nvsAssignmentMap->find(light);
                    if (it != nvsAssignmentMap->end() && it->second.shadowViewCount > 0u)
                    {
                        outLd.castsShadow     = 1u;
                        outLd.nvsLightId      = it->second.nvsLightId;
                        outLd.shadowViewBase  = it->second.shadowViewBase;
                        outLd.shadowViewCount = it->second.shadowViewCount;
                    }
                }
                else
                {
                    for (const LightNVSViewAssignment& assign : *nvsAssignments)
                    {
                        if (assign.light == light && assign.shadowViewCount > 0u)
                        {
                            outLd.castsShadow     = 1u;
                            outLd.nvsLightId      = assign.nvsLightId;
                            outLd.shadowViewBase  = assign.shadowViewBase;
                            outLd.shadowViewCount = assign.shadowViewCount;
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            outLd.castsShadow = wantsShadow ? 1u : 0u;
        }
    }

    void CoreRenderPipeline::BuildDirectionalLightData(
        Scene* scene,
        BOOL& outHasDirLight,
        FrameLightData& outDirLight,
        const std::vector<LightNVSViewAssignment>* nvsAssignments)
    {
        outHasDirLight = false;
        outDirLight = FrameLightData{};
        outDirLight.intensity = 0.0f;
        outDirLight.color = Color(0.0f, 0.0f, 0.0f, 0.0f);
        outDirLight.direction = Vec3::Zero();
        outDirLight.position = Vec3::Zero();
        outDirLight.viewProjectionMatrix = Mat4(0.0f);

        if (!scene) return;

        Light* dirLight = scene->GetDirectionalLight();
        if (!dirLight || !dirLight->HasFlag(LightFlags::ENABLED)) return;

        outHasDirLight = true;
        PopulateFrameLightFromLight(dirLight, nvsAssignments, nullptr, outDirLight);
    }

    void CoreRenderPipeline::BuildLightsData(
        Scene* scene,
        std::vector<FrameLightData>& outLights,
        const std::vector<LightNVSViewAssignment>* nvsAssignments)
    {
        if (!scene) return;

        const auto& sceneLights = scene->GetLights();
        outLights.reserve(sceneLights.size() + 1u);

        std::unordered_map<const Light*, LightNVSViewAssignment> assignMap;
        const std::unordered_map<const Light*, LightNVSViewAssignment>* assignMapPtr = nullptr;
        if (nvsAssignments != nullptr && nvsAssignments->size() > 16u)
        {
            assignMap.reserve(nvsAssignments->size());
            for (const LightNVSViewAssignment& assign : *nvsAssignments)
            {
                if (assign.light != nullptr && assign.shadowViewCount > 0u)
                {
                    assignMap.emplace(assign.light, assign);
                }
            }
            assignMapPtr = &assignMap;
        }

        for (SIZE_T i = 0; i < sceneLights.size(); ++i)
        {
            Light* light = sceneLights[i];
            if (!light || !light->HasFlag(LightFlags::ENABLED)) continue;

            FrameLightData ld{};
            PopulateFrameLightFromLight(light, nvsAssignments, assignMapPtr, ld);
            outLights.push_back(ld);
        }

        // Also check if scene->GetDirectionalLight() was set directly without being in m_lights
        Light* dirLight = scene->GetDirectionalLight();
        if (dirLight && dirLight->HasFlag(LightFlags::ENABLED))
        {
            bool alreadyIn = false;
            for (SIZE_T i = 0; i < sceneLights.size(); ++i)
            {
                if (sceneLights[i] == dirLight) { alreadyIn = true; break; }
            }
            if (!alreadyIn)
            {
                FrameLightData ld{};
                PopulateFrameLightFromLight(dirLight, nvsAssignments, assignMapPtr, ld);
                outLights.push_back(ld);
            }
        }
    }

    static NativeHandle ResolveEngineTextureHandle(const std::string& propName, const FrameView& frame)
    {
        if (propName == "NVS_SHADOW_MAP")
        {
            return frame.nvsShadowMapTexture;
        }
        return 0;
    }

    static void PopulateInstancedSkinningBuffersForCommand(
        DrawCommand& cmd,
        const Shader* vs)
    {
        if (!vs || !vs->IsValid()) return;
        const ShaderReflection& refl = vs->GetReflection();
        const UINT32 instCount = static_cast<UINT32>(cmd.instances.size());
        if (instCount == 0u) return;

        for (const auto& nscSb : refl.structuredBuffers)
        {
            if (nscSb.binding == 0xFFFF || nscSb.stride == 0) continue;
            const StructuredBufferLayoutPlan plan = CoreResourceBindingResolver::AnalyzeStructuredBuffer(nscSb, ShaderStage::VERTEX, 0u);
            if (!plan.valid) continue;

            // Check whether an INSTANCES StructuredBuffer matches the exact 72-byte canonical D3D12InstancePayload
            // layout (worldMatrix@0, instanceId@64, customData@68). Any subset (e.g. 64B INSTANCE_WORLD_MATRIX only),
            // reordered layout, or skinning-augmented layout is packed explicitly with its exact nscSb.stride so the
            // D3D12 SRV StructureByteStride always matches the shader's HLSL struct stride (zero GBV stride mismatch).
            BOOL isExactCanonicalInstancePayload = false;
            if (plan.domain == EngineResourceDomain::INSTANCES &&
                !plan.hasInstanceSkinFields &&
                nscSb.stride == sizeof(InstanceData))
            {
                BOOL hasWorldAt0 = false, hasIdAt64 = false, hasCustomAt68 = false, hasUnexpected = false;
                for (const auto& f : nscSb.fields)
                {
                    const std::string& pName = f.isGet ? (f.propertyName.empty() ? f.name : f.propertyName) : f.name;
                    if ((pName == "INSTANCE_WORLD_MATRIX" || (!f.isGet && pName == "worldMatrix")) &&
                        f.offset == 0u && f.size == sizeof(Mat4))
                    {
                        hasWorldAt0 = true;
                    }
                    else if ((pName == "INSTANCE_ID_VALUE" || (!f.isGet && pName == "instanceId")) &&
                             f.offset == sizeof(Mat4) && f.size == sizeof(UINT32))
                    {
                        hasIdAt64 = true;
                    }
                    else if ((pName == "INSTANCE_CUSTOM_DATA" || (!f.isGet && pName == "customData")) &&
                             f.offset == sizeof(Mat4) + sizeof(UINT32) && f.size == sizeof(UINT32))
                    {
                        hasCustomAt68 = true;
                    }
                    else
                    {
                        hasUnexpected = true;
                    }
                }
                isExactCanonicalInstancePayload = (hasWorldAt0 && hasIdAt64 && hasCustomAt68 && !hasUnexpected);
            }

            const BOOL isCustomInstancesSb = (plan.domain == EngineResourceDomain::INSTANCES && !isExactCanonicalInstancePayload);
            if (plan.domain != EngineResourceDomain::INSTANCE_SKINNING && !isCustomInstancesSb)
                continue;

            NativeStructuredBufferBinding* targetSbb = nullptr;
            for (auto& existing : cmd.structuredBuffers)
            {
                if (existing.stage == ShaderStage::VERTEX &&
                    existing.slotIndex == nscSb.binding &&
                    existing.registerSpace == nscSb.set)
                {
                    targetSbb = &existing;
                    break;
                }
            }
            if (!targetSbb)
            {
                NativeStructuredBufferBinding newSbb{};
                newSbb.slotIndex     = nscSb.binding;
                newSbb.registerSpace = nscSb.set;
                newSbb.stage         = ShaderStage::VERTEX;
                newSbb.stride        = nscSb.stride;
                cmd.structuredBuffers.push_back(std::move(newSbb));
                targetSbb = &cmd.structuredBuffers.back();
            }

            targetSbb->buffer       = 0;
            targetSbb->elementCount = instCount;
            targetSbb->stride       = nscSb.stride;
            targetSbb->ownedData.assign(static_cast<SIZE_T>(instCount) * nscSb.stride, 0);

            for (UINT32 i = 0; i < instCount; ++i)
            {
                BYTE* elemPtr = targetSbb->ownedData.data() + static_cast<SIZE_T>(i) * nscSb.stride;
                const InstanceData& instData = cmd.instances[i];
                const InstanceSkinningData skinData = (i < cmd.instanceSkinning.size())
                    ? cmd.instanceSkinning[i]
                    : InstanceSkinningData{};

                for (const auto& f : nscSb.fields)
                {
                    if (f.offset >= nscSb.stride) continue;
                    const SIZE_T maxWrite = nscSb.stride - f.offset;
                    if (!f.isGet)
                    {
                        if (f.name == "worldMatrix" && maxWrite >= sizeof(Mat4))
                            std::memcpy(elemPtr + f.offset, &instData.worldMatrix, sizeof(Mat4));
                        else if (f.name == "instanceId" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &instData.instanceId, sizeof(UINT32));
                        else if (f.name == "customData" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &instData.customData, sizeof(UINT32));
                    }
                    else
                    {
                        const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                        if (pName == "INSTANCE_WORLD_MATRIX" && maxWrite >= sizeof(Mat4))
                            std::memcpy(elemPtr + f.offset, &instData.worldMatrix, sizeof(Mat4));
                        else if (pName == "INSTANCE_ID_VALUE" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &instData.instanceId, sizeof(UINT32));
                        else if (pName == "INSTANCE_CUSTOM_DATA" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &instData.customData, sizeof(UINT32));
                        else if (pName == "INSTANCE_BONE_OFFSET" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &skinData.bonePaletteOffset, sizeof(UINT32));
                        else if (pName == "INSTANCE_BONE_COUNT" && maxWrite >= sizeof(UINT32))
                            std::memcpy(elemPtr + f.offset, &skinData.boneCount, sizeof(UINT32));
                    }
                }
            }

            targetSbb->cpuData  = targetSbb->ownedData.data();
            targetSbb->byteSize = static_cast<UINT32>(targetSbb->ownedData.size());
        }
    }

    static UINT64 HashEngineStructuredBufferLayout(const NSL::NSCStructuredBuffer& sb) noexcept
    {
        UINT64 h = 14695981039346656037ull;
        auto mixU64 = [&](UINT64 v) noexcept {
            h ^= v;
            h *= 1099511628211ull;
        };
        mixU64(sb.stride);
        mixU64(sb.fields.size());
        for (const auto& f : sb.fields)
        {
            mixU64(f.isGet ? 1ull : 0ull);
            mixU64(f.offset);
            mixU64(f.size);
            for (char c : f.propertyName)
            {
                mixU64(static_cast<unsigned char>(c));
            }
        }
        return h;
    }

    void CoreRenderPipeline::PopulateSharedEngineLightsPayload(
        const NSL::NSCStructuredBuffer& nscSb,
        const FrameView&                frameView,
        NativeStructuredBufferBinding&  ioSbb)
    {
        const UINT32 engineCount = static_cast<UINT32>(frameView.lights.size());
        if (engineCount == 0u || nscSb.stride == 0u)
        {
            ioSbb.elementCount = 0u;
            ioSbb.stride       = nscSb.stride;
            ioSbb.byteSize     = 0u;
            ioSbb.cpuData      = nullptr;
            ioSbb.ownedData.clear();
            ioSbb.sharedDataId = 0u;
            return;
        }

        const UINT64 layoutHash = HashEngineStructuredBufferLayout(nscSb);
        const UINT64 rev        = frameView.lightsDataRevision;

        if (rev != 0u)
        {
            for (const auto& entry : m_cachedLightsPayloads)
            {
                if (entry.lightsRevision == rev && entry.layoutHash == layoutHash && entry.payload)
                {
                    ioSbb.elementCount = engineCount;
                    ioSbb.stride       = nscSb.stride;
                    ioSbb.ownedData.clear();
                    ioSbb.cpuData      = entry.payload->data();
                    ioSbb.byteSize     = static_cast<UINT32>(entry.payload->size());
                    ioSbb.sharedDataId = entry.sharedDataId;
                    return;
                }
            }
        }

        std::vector<BYTE> builtVec(static_cast<SIZE_T>(engineCount) * nscSb.stride, 0u);
        BYTE* basePtr = builtVec.data();
        for (UINT32 i = 0u; i < engineCount; ++i)
        {
            BYTE* elemPtr = basePtr + static_cast<SIZE_T>(i) * nscSb.stride;
            for (const auto& f : nscSb.fields)
            {
                if (f.isGet)
                {
                    CoreResourceBindingResolver::FillEngineLightsElementField(frameView.lights[i], f, elemPtr, nscSb.stride);
                }
            }
        }

        const UINT64 sid = ++m_nextSharedSbDataId;
        ioSbb.elementCount = engineCount;
        ioSbb.stride       = nscSb.stride;
        ioSbb.sharedDataId = sid;

        if (rev != 0u)
        {
            auto heapVec         = std::make_unique<std::vector<BYTE>>(std::move(builtVec));
            const BYTE* rawPtr   = heapVec->data();
            const UINT32 rawSize = static_cast<UINT32>(heapVec->size());

            CachedEngineLightsSBEntry entry{};
            entry.lightsRevision = rev;
            entry.layoutHash     = layoutHash;
            entry.sharedDataId   = sid;
            entry.payload        = std::move(heapVec);
            m_cachedLightsPayloads.push_back(std::move(entry));

            ioSbb.ownedData.clear();
            ioSbb.cpuData  = rawPtr;
            ioSbb.byteSize = rawSize;
        }
        else
        {
            ioSbb.byteSize  = static_cast<UINT32>(builtVec.size());
            ioSbb.ownedData = std::move(builtVec);
            ioSbb.cpuData   = ioSbb.ownedData.data();
        }
    }

    BOOL CoreRenderPipeline::BuildDrawCommand(
        RenderObject* obj,
        PassType pass,
        const FrameView& frameView,
        const Vec3& cameraPos,
        DrawCommand& outCmd,
        ShaderVariant overrideVertexVariant,
        const InternalResourceManager* internalResources,
        std::vector<ComputeDispatchCommand>* outPreDrawDispatches,
        const Scene* scene)
    {
        if (!obj) return false;

        Mesh* mesh = ResolveEffectiveMesh(obj, frameView.camera);
        if (!mesh) return false;

        Material* mat = obj->GetMaterial();

        // Geometry
        outCmd.vertexBuffer = mesh->GetVertexBufferHandle();
        outCmd.indexBuffer  = mesh->GetIndexBufferHandle();
        outCmd.vertexCount  = mesh->GetVertexCount();
        outCmd.indexCount   = mesh->GetIndexCount();
        outCmd.cpuVertices  = mesh->GetVertices();
        outCmd.cpuIndices   = mesh->GetIndices();
        outCmd.worldMatrix  = obj->GetWorldMatrix();

        const BOOL isSkinned = obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices();
        const BOOL isDepthPrepass = (internalResources != nullptr);
        const BOOL hasExplicitDepthPass = isDepthPrepass && mat && mat->HasPass(PassType::DEPTH);
        const PassType effectivePropPass = hasExplicitDepthPass ? PassType::DEPTH : pass;

        ShaderSet* shaderSet = mat ? mat->GetEffectiveShaderSet(effectivePropPass, isSkinned) : nullptr;
        if (!shaderSet) return false;

        const ShaderVariant variant = (overrideVertexVariant != ShaderVariant::COUNT)
            ? overrideVertexVariant
            : (isSkinned ? ShaderVariant::SKINNED : ShaderVariant::STANDARD);

        Shader* geomVS = shaderSet->GetVertexShader(variant);
        if (!geomVS || !geomVS->IsValid())
        {
            // Missing required vertex shader variant -> safely skip drawing (0 Draw Call)
            return false;
        }

        // If vertex shader requires engine skinning data (Get(GLOBAL_BONE_MATRIX),
        // Get(BONE_MATRIX), Get(BONE_PALETTE_OFFSET), Get(BONE_COUNT), etc.) and the RenderObject
        // has no skinning matrices bound, safely skip drawing!
        if (CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(geomVS->GetReflection()) && !obj->HasSkinningMatrices())
        {
            return false;
        }

        const Shader* ps = shaderSet->GetPixelShader();
        BOOL psModifiesCoverage = false;
        if (ps && ps->IsValid())
        {
            const ShaderReflection& psRefl = ps->GetReflection();
            psModifiesCoverage = psRefl.ModifiesCoverageOrDepth();
            if (!psModifiesCoverage)
            {
                for (const auto& outElem : psRefl.outputLayout)
                {
                    if (outElem.semantic == "SV_Depth" || outElem.semantic == "SV_DEPTH" ||
                        outElem.semantic == "SV_DepthGreaterEqual" || outElem.semantic == "SV_DepthLessEqual")
                    {
                        psModifiesCoverage = true;
                        break;
                    }
                }
            }
            if (!psModifiesCoverage && mat && mat->IsAlphaTested())
            {
                psModifiesCoverage = true;
            }
        }
        outCmd.pixelShaderModifiesCoverage = psModifiesCoverage;

        Shader* vs = geomVS;
        const Shader* activePS = ps;

        if (isDepthPrepass)
        {
            if (hasExplicitDepthPass || psModifiesCoverage)
            {
                // Explicit PassType::DEPTH ShaderSet or coverage/depth-modifying PS: honor exact VS + PS
                vs       = geomVS;
                activePS = ps;
            }
            else
            {
                // Pure opaque material: always use the user's PassType::GEOMETRY vertex shader
                // (never substitute an internal engine VS) and omit the pixel shader for depth-only rasterization.
                vs       = geomVS;
                activePS = nullptr;
            }
        }

        if ((shaderSet->GetHullShader() != nullptr) != (shaderSet->GetDomainShader() != nullptr))
        {
            return false;
        }

        outCmd.vertexShader = vs->GetNativeHandle();
        if (shaderSet->GetHullShader())     outCmd.hullShader     = shaderSet->GetHullShader()->GetNativeHandle();
        if (shaderSet->GetDomainShader())   outCmd.domainShader   = shaderSet->GetDomainShader()->GetNativeHandle();
        if (shaderSet->GetGeometryShader()) outCmd.geometryShader = shaderSet->GetGeometryShader()->GetNativeHandle();
        outCmd.pixelShader = activePS ? activePS->GetNativeHandle() : 0;

        if (mat)
        {
            outCmd.cullMode   = mat->GetCullMode();
            outCmd.blendMode  = isDepthPrepass ? BlendMode::OPAQUE : mat->GetBlendMode();
            outCmd.fillMode   = mat->GetFillMode();
            outCmd.depthState = mat->GetDepthState();
        }

        if (mat)
        {
            // If any active stage of the material's geometry pass (or effective pass)
            // references a destroyed resource, reject the draw command in both Depth Prepass and Main Pass.
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const GraphicsShaderSlot checkSlot = static_cast<GraphicsShaderSlot>(s);
                const ShaderStage checkStage = SlotToShaderStage(checkSlot);
                const ShaderSet* geomSet = mat->GetGeometryShaderSet();
                const Shader* geomStageShader = (checkStage == ShaderStage::VERTEX)
                    ? (geomSet ? geomSet->GetVertexShader(variant) : nullptr)
                    : (geomSet ? geomSet->GetShader(checkStage) : nullptr);
                if (geomStageShader && mat->GetPropertyStore().Get(PassType::GEOMETRY, checkSlot).HasDestroyedResource())
                {
                    return false;
                }
                if (effectivePropPass != PassType::GEOMETRY)
                {
                    const Shader* effStageShader = (checkStage == ShaderStage::VERTEX)
                        ? vs
                        : (shaderSet ? shaderSet->GetShader(checkStage) : nullptr);
                    if (effStageShader && mat->GetPropertyStore().Get(effectivePropPass, checkSlot).HasDestroyedResource())
                    {
                        return false;
                    }
                }
            }

            const BOOL isInstancedVariant = (variant == ShaderVariant::INSTANCED || variant == ShaderVariant::INSTANCED_SKINNED);
            StageResolveContext resolveCtx{};
            resolveCtx.executionContextMask =
                EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_DRAW |
                (isInstancedVariant ? EXEC_CTX_INSTANCE : EXEC_CTX_RENDER_OBJECT);
            resolveCtx.frameView            = &frameView;
            resolveCtx.scene                = scene;
            resolveCtx.renderObject         = obj;
            resolveCtx.material             = mat;
            resolveCtx.passType             = isDepthPrepass ? PassType::DEPTH : pass;
            resolveCtx.worldMatrix          = outCmd.worldMatrix;
            resolveCtx.gpuSceneObjectSlot   = (!isInstancedVariant && obj) ? obj->GetGPUSceneHandle(scene).slotIndex : kInvalidGPUSceneSlot;
            resolveCtx.geometryMode         = static_cast<UINT32>(GeometryMode::STANDARD);
            resolveCtx.nvgDrawBucketId      = kNVGInvalidIndex;
            resolveCtx.nvgVisibleClusterOff = kNVGInvalidIndex;
            resolveCtx.nvgVisibleClusterCnt = 0u;
            resolveCtx.materializer         = m_resourceMaterializer;
            resolveCtx.outPreDrawDispatches = outPreDrawDispatches;

            const BOOL hasMatTx = (m_resourceMaterializer != nullptr);
            CoreGPUResourceMaterializer::TransactionSnapshot matTx{};
            const SIZE_T preDrawDispSnapshot = outPreDrawDispatches ? outPreDrawDispatches->size() : 0u;
            if (hasMatTx)
            {
                matTx = m_resourceMaterializer->BeginTransaction();
            }

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const GraphicsShaderSlot slot = static_cast<GraphicsShaderSlot>(s);
                const ShaderStage stage = SlotToShaderStage(slot);
                if (isDepthPrepass && stage == ShaderStage::PIXEL && activePS == nullptr)
                {
                    continue;
                }

                const ShaderStageProps& props = mat->GetPropertyStore().Get(effectivePropPass, slot);
                const ShaderStageProps* fallbackProps =
                    (effectivePropPass == PassType::DEPTH || effectivePropPass == PassType::SHADOW)
                        ? &mat->GetPropertyStore().Get(PassType::GEOMETRY, slot)
                        : nullptr;
                const Shader* stageShader = (stage == ShaderStage::VERTEX)
                    ? vs
                    : ((stage == ShaderStage::PIXEL) ? activePS : (shaderSet ? shaderSet->GetShader(stage) : nullptr));

                if (stageShader != nullptr &&
                    (props.HasDestroyedResource() || (fallbackProps && fallbackProps->HasDestroyedResource())))
                {
                    if (hasMatTx)
                    {
                        m_resourceMaterializer->RollbackTransaction(matTx);
                    }
                    if (outPreDrawDispatches && outPreDrawDispatches->size() > preDrawDispSnapshot)
                    {
                        outPreDrawDispatches->resize(preDrawDispSnapshot);
                    }
                    return false;
                }

                if (stageShader && stageShader->IsValid())
                {
                    const SIZE_T sbStartIdx = outCmd.structuredBuffers.size();
                    BOOL   stageHasInst = false;
                    UINT16 stageInstSp  = 0u;
                    UINT16 stageInstSl  = 0xFFFFu;
                    if (!CoreResourceBindingResolver::ResolveGraphicsStageBindings(
                            stageShader,
                            stage,
                            props,
                            fallbackProps,
                            resolveCtx,
                            outCmd.constantBuffers,
                            outCmd.textures,
                            outCmd.samplers,
                            outCmd.structuredBuffers,
                            (stage == ShaderStage::VERTEX) ? &stageHasInst : nullptr,
                            (stage == ShaderStage::VERTEX) ? &stageInstSp  : nullptr,
                            (stage == ShaderStage::VERTEX) ? &stageInstSl  : nullptr,
                            nullptr,
                            &outCmd.rwTextures))
                    {
                        if (hasMatTx)
                        {
                            m_resourceMaterializer->RollbackTransaction(matTx);
                        }
                        if (outPreDrawDispatches && outPreDrawDispatches->size() > preDrawDispSnapshot)
                        {
                            outPreDrawDispatches->resize(preDrawDispSnapshot);
                        }
                        return false;
                    }
                    if (stage == ShaderStage::VERTEX && stageHasInst)
                    {
                        outCmd.hasReflectedInstanceBinding = true;
                        outCmd.instanceBindingSpace        = stageInstSp;
                        outCmd.instanceBindingSlot         = stageInstSl;
                    }

                    // Reuse shared engine lights CPU payload across draws in the same frame for high-object scenes
                    if (!frameView.lights.empty() && frameView.lightsDataRevision != 0u)
                    {
                        const ShaderReflection& refl = stageShader->GetReflection();
                        for (SIZE_T sbIdx = sbStartIdx; sbIdx < outCmd.structuredBuffers.size(); ++sbIdx)
                        {
                            auto& sbb = outCmd.structuredBuffers[sbIdx];
                            if (sbb.buffer != 0 || sbb.elementCount != static_cast<UINT32>(frameView.lights.size()))
                                continue;
                            for (const auto& nscSb : refl.structuredBuffers)
                            {
                                if (nscSb.binding == sbb.slotIndex && nscSb.set == sbb.registerSpace &&
                                    CoreResourceBindingResolver::DetectStructuredBufferDomain(nscSb, stage) == EngineResourceDomain::LIGHTS)
                                {
                                    PopulateSharedEngineLightsPayload(nscSb, frameView, sbb);
                                    break;
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (props.HasDestroyedResource())
                    {
                        return false;
                    }
                    for (SIZE_T b = 0; b < props.buffers.size(); ++b)
                    {
                        const auto& buf = props.buffers[b];
                        if (buf.slotIndex != 0xFFFF && !buf.cpuData.empty())
                        {
                            NativeConstantBufferBinding cb{};
                            cb.slotIndex     = buf.slotIndex;
                            cb.registerSpace = buf.registerSpace;
                            cb.stage         = stage;
                            cb.ownedData     = buf.cpuData;
                            cb.cpuData       = cb.ownedData.data();
                            cb.byteSize      = static_cast<UINT32>(cb.ownedData.size());
                            cb.dirty         = buf.dirty;
                            outCmd.constantBuffers.push_back(std::move(cb));
                        }
                    }
                    for (SIZE_T t = 0; t < props.textures.size(); ++t)
                    {
                        const auto& tex = props.textures[t];
                        if (tex.slotIndex != 0xFFFF && tex.texture)
                        {
                            NativeTextureBinding tb{};
                            tb.slotIndex     = tex.slotIndex;
                            tb.registerSpace = tex.registerSpace;
                            tb.stage         = stage;
                            tb.texture       = tex.texture->GetNativeHandle();
                            outCmd.textures.push_back(tb);
                        }
                    }
                    for (SIZE_T rwt = 0; rwt < props.rwTextures.size(); ++rwt)
                    {
                        const auto& rwTex = props.rwTextures[rwt];
                        if (rwTex.slotIndex != 0xFFFF && rwTex.texture && rwTex.texture->GetNativeHandle() != 0)
                        {
                            NativeRWTextureBinding rwb{};
                            rwb.slotIndex     = rwTex.slotIndex;
                            rwb.registerSpace = rwTex.registerSpace;
                            rwb.stage         = stage;
                            rwb.texture       = rwTex.texture->GetNativeHandle();
                            rwb.mipSlice      = rwTex.mipSlice;
                            rwb.uavAccess     = rwTex.uavAccess;
                            outCmd.rwTextures.push_back(rwb);
                        }
                    }
                    for (SIZE_T smp = 0; smp < props.samplers.size(); ++smp)
                    {
                        const auto& samp = props.samplers[smp];
                        SamplerState* sampler = samp.sampler;
                        if (samp.slotIndex != 0xFFFF && sampler)
                        {
                            NativeSamplerBinding sb{};
                            sb.slotIndex     = samp.slotIndex;
                            sb.registerSpace = samp.registerSpace;
                            sb.stage         = stage;
                            sb.sampler       = sampler->GetNativeHandle();
                            outCmd.samplers.push_back(sb);
                        }
                    }
                    for (SIZE_T sbi = 0; sbi < props.structuredBuffers.size(); ++sbi)
                    {
                        const auto& sb = props.structuredBuffers[sbi];
                        if (sb.slotIndex != 0xFFFF && sb.gpuBuffer && sb.gpuBuffer->IsValid())
                        {
                            NativeStructuredBufferBinding sbb{};
                            sbb.slotIndex     = sb.slotIndex;
                            sbb.registerSpace = sb.registerSpace;
                            sbb.stage         = stage;
                            sbb.buffer        = sb.gpuBuffer->GetNativeHandle();
                            sbb.elementCount  = sb.gpuBuffer->GetElementCount();
                            sbb.stride        = sb.gpuBuffer->GetStride();
                            sbb.byteSize      = sb.gpuBuffer->GetByteSize();
                            sbb.isUAV         = sb.isUAV;
                            outCmd.structuredBuffers.push_back(std::move(sbb));
                        }
                    }
                }
            }
        }

        const Vec3 centerPos = obj->GetWorldAABB().IsValid() ? obj->GetWorldAABB().Center() : obj->GetWorldMatrix().GetTranslation();
        const Vec3 viewDelta = centerPos - cameraPos;
        const Vec3 camFwd    = frameView.camera.direction;
        outCmd.depthKey = (camFwd.LengthSq() > 1e-6f) ? viewDelta.Dot(camFwd.Normalized()) : viewDelta.Length();
        outCmd.sortKey  = (static_cast<UINT64>(outCmd.vertexShader) << 32) ^ static_cast<UINT64>(outCmd.pixelShader);

        return true;
    }

    BOOL CoreRenderPipeline::BuildComputeDispatchCommand(
        const ComputeKernel*                 kernel,
        const FrameView&                     frameView,
        ComputeDispatchCommand&              outCmd,
        std::vector<ComputeDispatchCommand>* outPreDispatches)
    {
        outCmd = ComputeDispatchCommand{};
        if (!kernel || !kernel->IsValid())
            return false;

        Shader* cs = kernel->GetComputeShader();
        if (!cs || !cs->IsValid() || cs->GetStage() != ShaderStage::COMPUTE)
            return false;

        outCmd.computeShader = cs->GetNativeHandle();
        if (outCmd.computeShader == 0)
            return false;

        if (kernel->IsIndirectDispatch())
        {
            GPUBuffer* argsBuf = kernel->GetIndirectArgsBuffer();
            const NativeHandle argsHandle = argsBuf ? argsBuf->GetNativeHandle() : kernel->GetIndirectArgsBufferHandle();
            if (argsBuf && (!argsBuf->IsValid() || !argsBuf->HasUsage(GPUBufferUsage::INDIRECT_ARGUMENT)))
                return false;
            if (argsHandle == 0)
                return false;
            if ((kernel->GetIndirectArgsOffset() & 3u) != 0)
                return false;
            if (argsBuf && (static_cast<UINT64>(kernel->GetIndirectArgsOffset()) + sizeof(GPUIndirectDispatchArgs) > argsBuf->GetByteSize()))
                return false;

            outCmd.indirectArgsBuffer = argsHandle;
            outCmd.indirectArgsOffset = kernel->GetIndirectArgsOffset();
            outCmd.threadGroupCountX  = 0;
            outCmd.threadGroupCountY  = 0;
            outCmd.threadGroupCountZ  = 0;
        }
        else
        {
            if (kernel->GetThreadGroupCountX() == 0 ||
                kernel->GetThreadGroupCountY() == 0 ||
                kernel->GetThreadGroupCountZ() == 0)
            {
                return false;
            }
            outCmd.threadGroupCountX = kernel->GetThreadGroupCountX();
            outCmd.threadGroupCountY = kernel->GetThreadGroupCountY();
            outCmd.threadGroupCountZ = kernel->GetThreadGroupCountZ();
        }

        outCmd.useRootConstants = kernel->UsesRootConstants();
        std::memcpy(outCmd.rootConstants, kernel->GetRootConstants(), sizeof(outCmd.rootConstants));

        const ShaderStage stage = ShaderStage::COMPUTE;
        const ShaderStageProps& props = kernel->GetStageProps();
        const ShaderReflection& refl = cs->GetReflection();

        StageResolveContext computeCtx{};
        computeCtx.executionContextMask   = EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_COMPUTE_DISPATCH;
        computeCtx.frameView              = &frameView;
        computeCtx.worldMatrix            = Mat4::Identity();
        computeCtx.gpuSceneObjectSlot     = kInvalidGPUSceneSlot;
        computeCtx.geometryMode           = kNVGInvalidIndex;
        computeCtx.nvgDrawBucketId        = kNVGInvalidIndex;
        computeCtx.nvgVisibleClusterOff   = kNVGInvalidIndex;
        computeCtx.nvgVisibleClusterCnt   = 0u;
        computeCtx.materializer           = m_resourceMaterializer;
        computeCtx.outPreDrawDispatches   = outPreDispatches;
        computeCtx.allowEmptyUserSBAsNull = true;

        // 1. Constant Buffers
        if (!refl.buffers.empty())
        {
            for (const auto& nscBuf : refl.buffers)
            {
                if (nscBuf.binding == 0xFFFF || nscBuf.totalSize == 0) continue;

                NativeConstantBufferBinding cb{};
                cb.slotIndex     = nscBuf.binding;
                cb.registerSpace = nscBuf.set;
                cb.stage         = stage;
                cb.byteSize      = nscBuf.totalSize;
                cb.ownedData.assign(nscBuf.totalSize, 0);

                const MaterialBufferSlot* matSlot = nullptr;
                for (const auto& mbs : props.buffers)
                {
                    if ((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name ||
                        mbs.resourceName == nscBuf.name)
                    {
                        matSlot = &mbs;
                        break;
                    }
                }
                if (!matSlot)
                {
                    for (const auto& mbs : props.buffers)
                    {
                        if (mbs.slotIndex == nscBuf.binding && mbs.registerSpace == nscBuf.set)
                        {
                            matSlot = &mbs;
                            break;
                        }
                    }
                }

                if (matSlot && !matSlot->cpuData.empty())
                {
                    const SIZE_T copySize = (std::min)(cb.ownedData.size(), matSlot->cpuData.size());
                    std::memcpy(cb.ownedData.data(), matSlot->cpuData.data(), copySize);
                    cb.dirty = matSlot->dirty;
                }
                else
                {
                    for (const auto& field : nscBuf.fields)
                    {
                        if (!field.isGet && !field.defaultData.empty() && field.offset + field.size <= nscBuf.totalSize)
                        {
                            const SIZE_T copyFieldSize = (std::min)(static_cast<SIZE_T>(field.size), field.defaultData.size());
                            std::memcpy(cb.ownedData.data() + field.offset,
                                        field.defaultData.data(),
                                        copyFieldSize);
                        }
                    }
                }

                for (const auto& field : nscBuf.fields)
                {
                    if (field.isGet && field.offset < nscBuf.totalSize)
                    {
                        CoreResourceBindingResolver::FillEngineConstantBufferField(
                            field.propertyName.empty() ? field.name : field.propertyName,
                            computeCtx,
                            cb.ownedData.data(),
                            field.offset,
                            field.size,
                            nscBuf.totalSize);
                    }
                }

                cb.cpuData = cb.ownedData.data();
                outCmd.constantBuffers.push_back(std::move(cb));
            }
        }

        for (const auto& buf : props.buffers)
        {
            if (buf.slotIndex == 0xFFFF || buf.cpuData.empty()) continue;
            BOOL alreadyAdded = false;
            for (const auto& existing : outCmd.constantBuffers)
            {
                if (existing.slotIndex == buf.slotIndex && existing.registerSpace == buf.registerSpace)
                {
                    alreadyAdded = true;
                    break;
                }
            }
            if (alreadyAdded) continue;

            NativeConstantBufferBinding cb{};
            cb.slotIndex     = buf.slotIndex;
            cb.registerSpace = buf.registerSpace;
            cb.stage         = stage;
            cb.byteSize      = static_cast<UINT32>(buf.cpuData.size());
            cb.ownedData     = buf.cpuData;
            cb.cpuData       = cb.ownedData.data();
            cb.dirty         = buf.dirty;
            outCmd.constantBuffers.push_back(std::move(cb));
        }

                    // 2. SRV Textures
                    if (!refl.textures.empty())
                    {
                        for (const auto& nscTex : refl.textures)
                        {
                            if (nscTex.binding == 0xFFFF) continue;

                            NativeTextureBinding tb{};
                            tb.slotIndex     = nscTex.binding;
                            tb.registerSpace = nscTex.set;
                            tb.stage         = stage;
                            tb.texture       = 0;

                            if (nscTex.IsEngineProvided())
                            {
                                tb.texture = ResolveEngineTextureHandle(nscTex.propertyName, frameView);
                                if (tb.texture == 0)
                                {
                                    if (nscTex.propertyName == "NVS_SHADOW_MAP")
                                    {
                                        tb.nullViewType = TextureType::TEXTURE_2D_ARRAY;
                                        tb.nullFormat   = TextureFormat::R32_FLOAT;
                                    }
                                    else
                                    {
                                        // Engine-owned texture unavailable -> safely skip compute dispatch!
                                        return false;
                                    }
                                }
                            }
                            else
                            {
                                BOOL matchedSlot = false;
                                // 1. Exact qualified group + entry match first
                                for (const auto& matTex : props.textures)
                                {
                                    if (!nscTex.groupQualifiedName.empty() &&
                                        ((matTex.propertyName + "__" + matTex.groupName) == nscTex.groupQualifiedName ||
                                         (matTex.propertyName + "__" + matTex.resourceName) == nscTex.groupQualifiedName) &&
                                        matTex.resourceName == nscTex.name)
                                    {
                                        matchedSlot = true;
                                        tb.mipSlice = matTex.mipSlice;
                                        if (matTex.texture != nullptr)
                                            tb.texture = matTex.texture->GetNativeHandle();
                                        else if (matTex.textureHandle != 0)
                                            tb.texture = matTex.textureHandle;
                                        break;
                                    }
                                }
                                // 2. Bare resourceName fallback
                                if (!matchedSlot)
                                {
                                    for (const auto& matTex : props.textures)
                                    {
                                        if (matTex.resourceName == nscTex.name ||
                                            (matTex.propertyName + "__" + matTex.resourceName) == nscTex.groupQualifiedName ||
                                            (matTex.groupName + "__" + matTex.resourceName) == nscTex.groupQualifiedName)
                                        {
                                            matchedSlot = true;
                                            tb.mipSlice = matTex.mipSlice;
                                            if (matTex.texture != nullptr)
                                                tb.texture = matTex.texture->GetNativeHandle();
                                            else if (matTex.textureHandle != 0)
                                                tb.texture = matTex.textureHandle;
                                            break;
                                        }
                                    }
                                }
                                if (!matchedSlot && tb.texture == 0)
                                {
                                    for (const auto& matTex : props.textures)
                                    {
                                        if (matTex.slotIndex == nscTex.binding && matTex.registerSpace == nscTex.set)
                                        {
                                            tb.mipSlice = matTex.mipSlice;
                                            if (matTex.texture != nullptr)
                                                tb.texture = matTex.texture->GetNativeHandle();
                                            else if (matTex.textureHandle != 0)
                                                tb.texture = matTex.textureHandle;
                                            break;
                                        }
                                    }
                                }

                                if (tb.texture == 0)
                                {
                                    // Missing required user SRV texture -> safely skip compute dispatch!
                                    return false;
                                }
                            }

                            outCmd.textures.push_back(tb);
                        }
                    }
                    else
                    {
                        for (const auto& tex : props.textures)
                        {
                            if (tex.slotIndex == 0xFFFF) continue;
                            const NativeHandle handle = (tex.texture != nullptr)
                                ? tex.texture->GetNativeHandle()
                                : tex.textureHandle;
                            if (handle != 0)
                            {
                                NativeTextureBinding tb{};
                                tb.slotIndex     = tex.slotIndex;
                                tb.registerSpace = tex.registerSpace;
                                tb.stage         = stage;
                                tb.texture       = handle;
                                tb.mipSlice      = tex.mipSlice;
                                outCmd.textures.push_back(tb);
                            }
                        }
                    }

                    // 3. UAV RWTextures
                    for (const auto& matRwt : props.rwTextures)
                    {
                        if (matRwt.slotIndex == 0xFFFF) continue;
                        if (matRwt.texture &&
                            (matRwt.texture->GetNativeHandle() == 0 ||
                             matRwt.texture->GetUsage() != TextureUsage::STORAGE))
                        {
                            return false;
                        }

                        const NativeHandle handle = (matRwt.texture != nullptr)
                            ? matRwt.texture->GetNativeHandle()
                            : matRwt.textureHandle;
            if (handle == 0)
                continue;

            // Reject simultaneous SRV + UAV binding of the same subresource (disjoint mip slices such as HZB Mip[k-1] -> Mip[k] are allowed)
            for (const auto& srvTex : outCmd.textures)
            {
                if (srvTex.texture == handle &&
                    (srvTex.mipSlice == 0xFFFFFFFFu || srvTex.mipSlice == matRwt.mipSlice))
                    return false;
            }

            NativeRWTextureBinding rwtb{};
            rwtb.slotIndex     = matRwt.slotIndex;
            rwtb.registerSpace = matRwt.registerSpace;
            rwtb.stage         = stage;
            rwtb.texture       = handle;
            rwtb.mipSlice      = matRwt.mipSlice;
            rwtb.uavAccess     = matRwt.uavAccess;
            outCmd.rwTextures.push_back(rwtb);
        }

        // 4. Samplers
        if (!refl.samplers.empty())
        {
            for (const auto& nscSamp : refl.samplers)
            {
                if (nscSamp.binding == 0xFFFF) continue;
                if (nscSamp.IsEngineProvided())
                {
                    NativeSamplerBinding esb{};
                    esb.slotIndex     = nscSamp.binding;
                    esb.registerSpace = nscSamp.set;
                    esb.stage         = stage;
                    esb.sampler       = 0;

                    if (nscSamp.propertyName == "NVS_COMPARISON_SAMPLER" && nscSamp.isComparison)
                    {
                        esb.sampler = frameView.nvsComparisonSampler;
                    }
                    else if (nscSamp.propertyName == "NVS_DEPTH_SAMPLER" && !nscSamp.isComparison)
                    {
                        esb.sampler = frameView.nvsDepthSampler;
                    }
                    else
                    {
                        return false;
                    }

                    if (esb.sampler == 0)
                    {
                        return false;
                    }

                    outCmd.samplers.push_back(esb);
                    continue;
                }

                NativeSamplerBinding sb{};
                sb.slotIndex     = nscSamp.binding;
                sb.registerSpace = nscSamp.set;
                sb.stage         = stage;
                sb.sampler       = 0;

                const SamplerState* boundSampler = nullptr;
                BOOL matchedSamplerSlot = false;
                for (const auto& matSamp : props.samplers)
                {
                    if (!nscSamp.groupQualifiedName.empty() &&
                        ((matSamp.propertyName + "__" + matSamp.groupName) == nscSamp.groupQualifiedName ||
                         (matSamp.propertyName + "__" + matSamp.resourceName) == nscSamp.groupQualifiedName) &&
                        matSamp.resourceName == nscSamp.name)
                    {
                        matchedSamplerSlot = true;
                        if (matSamp.sampler)
                        {
                            boundSampler = matSamp.sampler;
                            sb.sampler = matSamp.sampler->GetNativeHandle();
                        }
                        else if (matSamp.samplerHandle != 0)
                        {
                            sb.sampler = matSamp.samplerHandle;
                        }
                        break;
                    }
                }
                if (!matchedSamplerSlot)
                {
                    for (const auto& matSamp : props.samplers)
                    {
                        if (matSamp.resourceName == nscSamp.name ||
                            (matSamp.propertyName + "__" + matSamp.resourceName) == nscSamp.groupQualifiedName ||
                            (matSamp.groupName + "__" + matSamp.resourceName) == nscSamp.groupQualifiedName)
                        {
                            if (matSamp.sampler)
                            {
                                boundSampler = matSamp.sampler;
                                sb.sampler = matSamp.sampler->GetNativeHandle();
                            }
                            else if (matSamp.samplerHandle != 0)
                            {
                                sb.sampler = matSamp.samplerHandle;
                            }
                            break;
                        }
                    }
                }
                if (sb.sampler == 0)
                {
                    for (const auto& matSamp : props.samplers)
                    {
                        if (matSamp.slotIndex == nscSamp.binding && matSamp.registerSpace == nscSamp.set)
                        {
                            if (matSamp.sampler)
                            {
                                boundSampler = matSamp.sampler;
                                sb.sampler = matSamp.sampler->GetNativeHandle();
                            }
                            else if (matSamp.samplerHandle != 0)
                            {
                                sb.sampler = matSamp.samplerHandle;
                            }
                            break;
                        }
                    }
                }

                if (sb.sampler == 0)
                    return false;

                if (boundSampler && boundSampler->IsComparison() != nscSamp.isComparison)
                    return false;

                outCmd.samplers.push_back(sb);
            }
        }
        else
        {
            for (const auto& samp : props.samplers)
            {
                if (samp.slotIndex == 0xFFFF) continue;
                const NativeHandle handle = samp.sampler ? samp.sampler->GetNativeHandle() : samp.samplerHandle;
                if (handle != 0)
                {
                    NativeSamplerBinding sb{};
                    sb.slotIndex     = samp.slotIndex;
                    sb.registerSpace = samp.registerSpace;
                    sb.stage         = stage;
                    sb.sampler       = handle;
                    outCmd.samplers.push_back(sb);
                }
            }
        }

        // 5. StructuredBuffers & RWStructuredBuffers
        for (const auto& nscSb : refl.structuredBuffers)
        {
            if (nscSb.binding == 0xFFFF || nscSb.stride == 0) continue;

            const BOOL isReflUav = (refl.FindUAV(nscSb.name) != nullptr);

            const ShaderStructuredBufferSlot* matSb = nullptr;
            for (const auto& ms : props.structuredBuffers)
            {
                if (ms.isUAV == isReflUav &&
                    ((ms.propertyName + "__" + ms.resourceName) == nscSb.name ||
                     ms.resourceName == nscSb.name))
                {
                    matSb = &ms;
                    break;
                }
            }
            if (!matSb)
            {
                for (const auto& ms : props.structuredBuffers)
                {
                    if (ms.isUAV == isReflUav && ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set)
                    {
                        matSb = &ms;
                        break;
                    }
                }
            }

            const UINT32 userCount = matSb ? matSb->userElementCount : 0u;
            const StructuredBufferLayoutPlan plan = isReflUav
                ? StructuredBufferLayoutPlan{}
                : CoreResourceBindingResolver::AnalyzeStructuredBuffer(nscSb, stage, userCount);
            if (!isReflUav && !plan.valid)
            {
                return false;
            }
            if (!isReflUav && plan.domain == EngineResourceDomain::INSTANCES)
                continue;

            NativeStructuredBufferBinding sbb{};
            sbb.slotIndex     = nscSb.binding;
            sbb.registerSpace = nscSb.set;
            sbb.stage         = stage;
            sbb.stride        = nscSb.stride;
            sbb.isUAV         = isReflUav;

            const NativeHandle resolvedSbHandle = (matSb && matSb->gpuBuffer != nullptr)
                ? matSb->gpuBuffer->GetNativeHandle()
                : (matSb ? matSb->gpuBufferHandle : 0);
            const UINT32 resolvedSbStride = (matSb && matSb->gpuBuffer != nullptr)
                ? matSb->gpuBuffer->GetStride()
                : (matSb ? matSb->gpuBufferStride : 0);

            if (matSb && resolvedSbHandle != 0)
            {
                if (resolvedSbStride != nscSb.stride)
                    return false;
                if (matSb->gpuBuffer && (!matSb->gpuBuffer->IsValid() || (isReflUav && !matSb->gpuBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV))))
                    return false;
                if (isReflUav && outCmd.indirectArgsBuffer != 0 && resolvedSbHandle == outCmd.indirectArgsBuffer)
                    return false;

                sbb.buffer       = resolvedSbHandle;
                sbb.elementCount = matSb->gpuBuffer ? matSb->gpuBuffer->GetElementCount() : matSb->gpuBufferElementCount;
                sbb.stride       = nscSb.stride;
                sbb.byteSize     = matSb->gpuBuffer ? matSb->gpuBuffer->GetByteSize() : matSb->gpuBufferByteSize;
                sbb.cpuData      = nullptr;
                sbb.ownedData.clear();
                sbb.isUAV        = isReflUav;
                sbb.uavAccess    = matSb->uavAccess;
                outCmd.structuredBuffers.push_back(std::move(sbb));
                continue;
            }

            if (isReflUav)
            {
                // Unbound RWStructuredBuffer slot (e.g. inactive mode in multi-mode kernel);
                // skipped here so D3D12 Null UAV fallback handles the slot, while the global UAV check enforces at least one UAV is bound.
                continue;
            }

            if (plan.domain != EngineResourceDomain::USER_ONLY)
            {
                if (plan.domain == EngineResourceDomain::LIGHTS && userCount == 0u && !frameView.lights.empty())
                {
                    PopulateSharedEngineLightsPayload(nscSb, frameView, sbb);
                    sbb.isUAV = false;
                    outCmd.structuredBuffers.push_back(std::move(sbb));
                    continue;
                }

                if (!CoreResourceBindingResolver::ResolveEngineStructuredBuffer(
                        nscSb, stage, plan, matSb, computeCtx, sbb, nullptr))
                {
                    return false;
                }
                outCmd.structuredBuffers.push_back(std::move(sbb));
                continue;
            }
            else
            {
                const UINT32 count = matSb ? (matSb->userElementCount > 0 ? matSb->userElementCount : matSb->elementCount) : 0;
                if (count == 0)
                {
                    sbb.elementCount = 0;
                    sbb.stride       = nscSb.stride;
                    sbb.byteSize     = 0;
                    sbb.cpuData      = nullptr;
                    sbb.ownedData.clear();
                    sbb.isUAV        = false;
                    outCmd.structuredBuffers.push_back(std::move(sbb));
                    continue;
                }

                sbb.elementCount = count;
                sbb.ownedData.assign(static_cast<SIZE_T>(count) * nscSb.stride, 0);
                if (matSb && !matSb->userData.empty())
                {
                    std::memcpy(sbb.ownedData.data(), matSb->userData.data(), (std::min)(sbb.ownedData.size(), matSb->userData.size()));
                }
                else if (matSb && !matSb->cpuData.empty())
                {
                    std::memcpy(sbb.ownedData.data(), matSb->cpuData.data(), (std::min)(sbb.ownedData.size(), matSb->cpuData.size()));
                }
            }

            sbb.cpuData  = sbb.ownedData.data();
            sbb.byteSize = static_cast<UINT32>(sbb.ownedData.size());
            sbb.isUAV    = false;
            outCmd.structuredBuffers.push_back(std::move(sbb));
        }

        // Also include any structuredBuffers from props that were not in refl.structuredBuffers
        // (e.g., manual unit test UAVs where only refl.uavBuffers was populated, or raw slot bindings)
        for (const auto& ms : props.structuredBuffers)
        {
            if (ms.slotIndex == 0xFFFF) continue;

            BOOL alreadyAdded = false;
            for (const auto& existing : outCmd.structuredBuffers)
            {
                if (existing.slotIndex == ms.slotIndex &&
                    existing.registerSpace == ms.registerSpace &&
                    existing.isUAV == ms.isUAV)
                {
                    alreadyAdded = true;
                    break;
                }
            }
            if (alreadyAdded) continue;

            const NativeHandle resolvedMsHandle = (ms.gpuBuffer != nullptr)
                ? ms.gpuBuffer->GetNativeHandle()
                : ms.gpuBufferHandle;

            if (resolvedMsHandle != 0)
            {
                if (ms.gpuBuffer && (!ms.gpuBuffer->IsValid() || (ms.isUAV && !ms.gpuBuffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV))))
                    return false;
                if (ms.isUAV && outCmd.indirectArgsBuffer != 0 && resolvedMsHandle == outCmd.indirectArgsBuffer)
                    return false;

                NativeStructuredBufferBinding sbb{};
                sbb.slotIndex     = ms.slotIndex;
                sbb.registerSpace = ms.registerSpace;
                sbb.stage         = stage;
                sbb.buffer        = resolvedMsHandle;
                sbb.elementCount  = ms.gpuBuffer ? ms.gpuBuffer->GetElementCount() : ms.gpuBufferElementCount;
                sbb.stride        = ms.gpuBuffer ? ms.gpuBuffer->GetStride()       : ms.gpuBufferStride;
                sbb.byteSize      = ms.gpuBuffer ? ms.gpuBuffer->GetByteSize()     : ms.gpuBufferByteSize;
                sbb.isUAV         = ms.isUAV;
                sbb.uavAccess     = ms.uavAccess;
                outCmd.structuredBuffers.push_back(std::move(sbb));
            }
            else if (!ms.isUAV)
            {
                const UINT32 count = (ms.userElementCount > 0) ? ms.userElementCount : ms.elementCount;
                if (count > 0 && ms.stride > 0)
                {
                    NativeStructuredBufferBinding sbb{};
                    sbb.slotIndex     = ms.slotIndex;
                    sbb.registerSpace = ms.registerSpace;
                    sbb.stage         = stage;
                    sbb.elementCount  = count;
                    sbb.stride        = ms.stride;
                    sbb.ownedData.assign(static_cast<SIZE_T>(count) * ms.stride, 0);
                    if (!ms.userData.empty())
                        std::memcpy(sbb.ownedData.data(), ms.userData.data(), (std::min)(sbb.ownedData.size(), ms.userData.size()));
                    else if (!ms.cpuData.empty())
                        std::memcpy(sbb.ownedData.data(), ms.cpuData.data(), (std::min)(sbb.ownedData.size(), ms.cpuData.size()));
                    sbb.cpuData  = sbb.ownedData.data();
                    sbb.byteSize = static_cast<UINT32>(sbb.ownedData.size());
                    sbb.isUAV    = false;
                    outCmd.structuredBuffers.push_back(std::move(sbb));
                }
            }
        }

        // Prevent simultaneous SRV + UAV binding of the same GPUBuffer in the same dispatch
        for (const auto& uavSb : outCmd.structuredBuffers)
        {
            if (!uavSb.isUAV || uavSb.buffer == 0) continue;
            for (const auto& srvSb : outCmd.structuredBuffers)
            {
                if (!srvSb.isUAV && srvSb.buffer == uavSb.buffer)
                    return false;
            }
        }

        // If the shader declares UAVs, at least one UAV must be bound
        if (refl.HasUAV())
        {
            BOOL hasAnyBoundUav = !outCmd.rwTextures.empty();
            if (!hasAnyBoundUav)
            {
                for (const auto& sbb : outCmd.structuredBuffers)
                {
                    if (sbb.isUAV && sbb.buffer != 0)
                    {
                        hasAnyBoundUav = true;
                        break;
                    }
                }
            }
            if (!hasAnyBoundUav)
                return false;
        }

        return true;
    }

    Mesh* CoreRenderPipeline::ResolveEffectiveMesh(RenderObject* obj, const FrameCameraData& cameraData) const
    {
        if (!obj) return nullptr;
        LODGroup* lodGroup = obj->GetLODGroup();
        if (lodGroup && !lodGroup->GetLevels().empty())
        {
            const Vec3 objPos   = obj->GetWorldMatrix().GetTranslation();
            const Vec3 objScale = obj->GetWorldMatrix().GetScale();
            if (lodGroup->IsUsingScreenSize() &&
                std::fabs(cameraData.projectionMatrix.m[15] - 1.0f) < 1e-4f &&
                std::fabs(cameraData.projectionMatrix.m[11]) < 1e-6f)
            {
                const FLOAT32 orthoH = (std::fabs(cameraData.projectionMatrix.m[5]) > 1e-6f)
                    ? (2.0f / std::fabs(cameraData.projectionMatrix.m[5]))
                    : 10.0f;
                const FLOAT32 maxScale = std::max({
                    std::fabs(objScale.x),
                    std::fabs(objScale.y),
                    std::fabs(objScale.z)
                });
                const FLOAT32 worldRadius = std::max(0.0f, lodGroup->GetLocalRadius()) * maxScale;
                const FLOAT32 screenH     = (orthoH > 1e-5f) ? std::clamp((worldRadius * 2.0f) / orthoH, 0.0f, 1.0f) : 1.0f;
                const auto&   levels      = lodGroup->GetLevels();
                for (SIZE_T i = 0; i < levels.size(); ++i)
                {
                    if (screenH >= levels[i].screenRelativeTransitionHeight)
                    {
                        return levels[i].mesh;
                    }
                }
                return nullptr;
            }

            FLOAT32 fovYRad = ((cameraData.fov > 0.1f && cameraData.fov < 179.9f) ? cameraData.fov : 60.0f) * DEG2RAD;
            if (std::fabs(cameraData.projectionMatrix.m[5]) > 1e-5f &&
                std::fabs(cameraData.projectionMatrix.m[11]) > 1e-4f)
            {
                fovYRad = 2.0f * std::atan(1.0f / std::fabs(cameraData.projectionMatrix.m[5]));
            }
            const INT32 lodIdx = lodGroup->GetCurrentLODIndex(objPos, objScale, cameraData.position, fovYRad);
            if (lodIdx < 0)
            {
                return nullptr;
            }
            return lodGroup->GetMeshForLOD(lodIdx);
        }
        return obj->GetMesh();
    }

    void CoreRenderPipeline::BatchAndCategorizeDrawCommands(
        Scene* scene,
        const std::vector<RenderObject*>& objects,
        PassType pass,
        const FrameView& frameView,
        const Vec3& cameraPos,
        RenderStats& stats,
        std::vector<DrawCommand>& outCommands,
        const InternalResourceManager* internalResources,
        std::vector<ComputeDispatchCommand>* outPreDrawDispatches)
    {
        struct BatchKey
        {
            Mesh*     mesh      = nullptr;
            Material* material  = nullptr;
            BOOL      isSkinned = false;

            BOOL operator==(const BatchKey& o) const
            {
                return mesh == o.mesh && material == o.material && isSkinned == o.isSkinned;
            }
        };

        struct BatchKeyHash
        {
            SIZE_T operator()(const BatchKey& k) const
            {
                return std::hash<void*>{}(k.mesh) ^ (std::hash<void*>{}(k.material) << 1) ^ (static_cast<SIZE_T>(k.isSkinned) << 2);
            }
        };

        std::unordered_map<BatchKey, std::vector<RenderObject*>, BatchKeyHash> groups;
        std::vector<BatchKey> keyOrder;

        for (RenderObject* obj : objects)
        {
            if (!obj) continue;
            Mesh* mesh = ResolveEffectiveMesh(obj, frameView.camera);
            Material* mat = obj->GetMaterial();
            if (!mesh || !mat) continue;

            const BOOL objSkinned = obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices();
            BatchKey key{ mesh, mat, objSkinned };
            if (groups.find(key) == groups.end())
            {
                keyOrder.push_back(key);
            }
            groups[key].push_back(obj);
        }

        for (const BatchKey& key : keyOrder)
        {
            const auto& group = groups[key];
            if (group.empty()) continue;

            const PassType effectivePass = (internalResources != nullptr && key.material && key.material->HasPass(PassType::DEPTH))
                ? PassType::DEPTH
                : pass;
            ShaderSet* shaderSet = key.material ? key.material->GetEffectiveShaderSet(effectivePass, key.isSkinned) : nullptr;
            const BOOL isSkinned = key.isSkinned;
            const ShaderVariant instVariant = isSkinned ? ShaderVariant::INSTANCED_SKINNED : ShaderVariant::INSTANCED;
            if (group.size() > 1)
            {
                if (!shaderSet || !shaderSet->HasVertexVariant(instVariant))
                {
                    // Multiple objects sharing (Mesh, Material) form an instanced batch.
                    // If the ShaderSet lacks the required INSTANCED vertex shader variant, Northstar does NOT
                    // babysit the user by falling back to individual draw calls or ring buffer.
                    // Safely skip drawing (0 draw calls).
                    continue;
                }

                Shader* ivs = shaderSet->GetVertexShader(instVariant);
                const BOOL reqEngineSkinning = (ivs && ivs->IsValid() && CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(ivs->GetReflection()));

                RenderObject* repObj = nullptr;
                for (RenderObject* candidate : group)
                {
                    if (!reqEngineSkinning || candidate->HasSkinningMatrices())
                    {
                        repObj = candidate;
                        break;
                    }
                }
                if (!repObj)
                    continue;

                DrawCommand baseCmd{};
                if (!BuildDrawCommand(repObj, pass, frameView, cameraPos, baseCmd, instVariant, internalResources, outPreDrawDispatches, scene))
                    continue;

                baseCmd.vertexBuffer = key.mesh->GetVertexBufferHandle();
                baseCmd.indexBuffer  = key.mesh->GetIndexBufferHandle();
                baseCmd.vertexCount  = key.mesh->GetVertexCount();
                baseCmd.indexCount   = key.mesh->GetIndexCount();
                baseCmd.cpuVertices  = key.mesh->GetVertices();
                baseCmd.cpuIndices   = key.mesh->GetIndices();

                if (baseCmd.vertexShader == 0 && ivs)
                {
                    baseCmd.vertexShader = ivs->GetNativeHandle();
                }

                baseCmd.bufferCategory = BufferCategory::INSTANCE;
                baseCmd.startInstance  = 0;
                baseCmd.instances.clear();
                baseCmd.instanceSkinning.clear();

                UINT32 validIdx = 0u;
                for (SIZE_T i = 0; i < group.size(); ++i)
                {
                    if (reqEngineSkinning && !group[i]->HasSkinningMatrices())
                    {
                        stats.skippedMissingSkinningObjects++;
                        continue;
                    }

                    const UINT32 boneOff = group[i]->GetSkinningPaletteOffset(scene);
                    const UINT32 boneCnt = group[i]->GetSkinningMatrixCount();

                    InstanceData inst{};
                    inst.worldMatrix = group[i]->GetWorldMatrix();
                    inst.instanceId  = validIdx++;
                    inst.customData  = (boneCnt > 0u) ? PackGPUSceneSkinnedBoneRange(boneOff, boneCnt) : 0u;
                    baseCmd.instances.push_back(inst);

                    InstanceSkinningData skinInst{};
                    skinInst.bonePaletteOffset = boneOff;
                    skinInst.boneCount         = boneCnt;
                    skinInst.skinningRevision  = group[i]->GetSkinningRevision();
                    baseCmd.instanceSkinning.push_back(skinInst);
                }
                if (baseCmd.instances.empty())
                    continue;

                baseCmd.instanceCount = static_cast<UINT32>(baseCmd.instances.size());
                PopulateInstancedSkinningBuffersForCommand(baseCmd, ivs);

                stats.instancedDrawCalls++;
                stats.instanceCount += baseCmd.instanceCount;

                outCommands.push_back(std::move(baseCmd));
            }
            else
            {
                // Single object (group.size() == 1): Render individually.
                RenderObject* obj = group[0];
                if (obj && key.material && !obj->HasSkinningMatrices())
                {
                    const BOOL isSkinnedObj = obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices();
                    const BOOL isPrepass    = (internalResources != nullptr);
                    const PassType effPass  = (isPrepass && key.material->HasPass(PassType::DEPTH)) ? PassType::DEPTH : pass;
                    ShaderSet* ss           = key.material->GetEffectiveShaderSet(effPass, isSkinnedObj);
                    Shader* checkVS         = ss ? ss->GetVertexShader(isSkinnedObj ? ShaderVariant::SKINNED : ShaderVariant::STANDARD) : nullptr;
                    if (checkVS && checkVS->IsValid() && CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(checkVS->GetReflection()))
                    {
                        stats.skippedMissingSkinningObjects++;
                        continue;
                    }
                }
                DrawCommand singleCmd{};
                if (!BuildDrawCommand(obj, pass, frameView, cameraPos, singleCmd, ShaderVariant::COUNT, internalResources, outPreDrawDispatches, scene))
                    continue;

                    singleCmd.vertexBuffer = key.mesh->GetVertexBufferHandle();
                    singleCmd.indexBuffer  = key.mesh->GetIndexBufferHandle();
                    singleCmd.vertexCount  = key.mesh->GetVertexCount();
                    singleCmd.indexCount   = key.mesh->GetIndexCount();
                    singleCmd.cpuVertices  = key.mesh->GetVertices();
                    singleCmd.cpuIndices   = key.mesh->GetIndices();

                    const UINT32 boneOff = obj->GetSkinningPaletteOffset(scene);
                    const UINT32 boneCnt = obj->GetSkinningMatrixCount();
                    const UINT32 packedSkin = (boneCnt > 0u) ? PackGPUSceneSkinnedBoneRange(boneOff, boneCnt) : 0u;
                    singleCmd.instanceSkinning.clear();
                    singleCmd.instanceSkinning.push_back({ boneOff, boneCnt, obj->GetSkinningRevision() });

                    const BOOL isStatic = obj->HasFlag(RenderObjectFlags::STATIC);
                    if (isStatic)
                    {
                        if (scene && scene->IsStaticBatched() && !obj->HasFlag(RenderObjectFlags::BAKED_STATIC))
                        {
                            singleCmd.bufferCategory = BufferCategory::DYNAMIC_OBJECT;
                            singleCmd.instanceCount  = 1;
                            singleCmd.startInstance  = 0;
                            singleCmd.instances.clear();

                            InstanceData inst{};
                            inst.worldMatrix = obj->GetWorldMatrix();
                            inst.instanceId  = 0;
                            inst.customData  = packedSkin;
                            singleCmd.instances.push_back(inst);
                        }
                        else
                        {
                            singleCmd.bufferCategory = BufferCategory::STATIC_OBJECT;
                            singleCmd.instanceCount  = 1;
                            singleCmd.startInstance  = 0;
                            singleCmd.instances.clear();

                            InstanceData inst{};
                            inst.worldMatrix = obj->GetWorldMatrix();
                            inst.instanceId  = 0;
                            inst.customData  = packedSkin;
                            singleCmd.instances.push_back(inst);

                            stats.staticBatchedCalls++;
                        }
                    }
                    else
                    {
                        singleCmd.bufferCategory = BufferCategory::DYNAMIC_OBJECT;
                        singleCmd.instanceCount  = 1;
                        singleCmd.startInstance  = 0;
                        singleCmd.instances.clear();

                        InstanceData inst{};
                        inst.worldMatrix = obj->GetWorldMatrix();
                        inst.instanceId  = 0;
                        inst.customData  = packedSkin;
                        singleCmd.instances.push_back(inst);
                    }

                outCommands.push_back(std::move(singleCmd));
            }
        }
    }
}
