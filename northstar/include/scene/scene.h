#pragma once

#include "scene_desc.h"
#include "gpu_scene_types.h"
#include "../compute/compute_kernel.h"

#include <vector>
#include <algorithm>

namespace Northstar
{
    // ============ Baked Static Submesh & Batch Record ============
    struct BakedStaticSubmesh
    {
        UINT32 startIndex  = 0;
        UINT32 indexCount  = 0;
        AABB   localAABB   = {};
        Mat4   worldMatrix = Mat4::Identity();
        AABB   worldAABB   = {};
        BOOL   castsShadow = true;
    };

    struct BakedStaticBatch
    {
        Mesh*                           combinedMesh = nullptr;
        Material*                       material     = nullptr;
        AABB                            worldAABB    = {};
        RenderObject*                   batchObject  = nullptr;
        std::vector<RenderObject*>      sourceObjects;
        std::vector<BakedStaticSubmesh> submeshes;
    };

    // ============ Scene Scheduled Compute Kernel Entry ============
    struct SceneComputeKernelEntry
    {
        ComputeKernel*        kernel           = nullptr;
        ComputePassPhase      phase            = ComputePassPhase::PRE_PASS;
        ComputeExecutionScope scope            = ComputeExecutionScope::PER_VIEW;
        BOOL                  hasExplicitScope = false;

        ComputeExecutionScope ResolveScope() const
        {
            return hasExplicitScope ? scope : (kernel ? kernel->GetExecutionScope() : scope);
        }
    };

    // ============ Scene ============
    class Scene
    {
    public:
        ~Scene()
        {
            DetachAllRenderObjects();
        }

        // ==================== SCENE OBJECT MANAGEMENT ====================
        void AddRenderObject(RenderObject* obj)
        {
            if (!obj) return;
            if (obj->FindSceneLink(this) != nullptr)
                return;

            m_renderObjects.push_back(obj);
            if (m_dirtyObjects.capacity() < m_renderObjects.capacity())
            {
                m_dirtyObjects.reserve(m_renderObjects.capacity());
            }

            SceneObjectLink& link = obj->AttachSceneLink(this, &Scene::OnLinkedObjectDirtyThunk);
            const UINT32 prev = link.pendingDirtyFlags;
            link.pendingDirtyFlags |= GPU_SCENE_DIRTY_ALL;
            if (prev == 0u && link.dirtyListIndex == 0xFFFFFFFFu)
            {
                link.dirtyListIndex = static_cast<UINT32>(m_dirtyObjects.size());
                m_dirtyObjects.push_back(obj);
            }
        }

        void RemoveRenderObject(RenderObject* obj)
        {
            if (!obj) return;
            if (SceneObjectLink* link = obj->FindSceneLink(this))
            {
                if (link->dirtyListIndex < m_dirtyObjects.size() &&
                    m_dirtyObjects[link->dirtyListIndex] == obj)
                {
                    m_dirtyObjects[link->dirtyListIndex] = nullptr;
                }
                if (link->gpuHandle.IsValid())
                {
                    m_pendingRemovedSlots.push_back(link->gpuHandle.slotIndex);
                }
                obj->DetachSceneLink(this);
            }
            auto it = std::find(m_renderObjects.begin(), m_renderObjects.end(), obj);
            if (it != m_renderObjects.end())
            {
                m_renderObjects.erase(it);
            }
            for (SIZE_T b = 0; b < m_bakedStaticBatches.size();)
            {
                BakedStaticBatch& batch = m_bakedStaticBatches[b];
                if (batch.batchObject == obj)
                {
                    RetireBakedStaticBatchAt(b);
                    continue;
                }
                BOOL matchedSource = false;
                for (RenderObject* srcObj : batch.sourceObjects)
                {
                    if (srcObj == obj)
                    {
                        matchedSource = true;
                        break;
                    }
                }
                if (matchedSource)
                {
                    // Unbake the affected static batch immediately so removed/destroyed source objects
                    // never leave ghost triangles in combinedMesh while remaining static objects keep rendering.
                    for (auto& srcObj : batch.sourceObjects)
                    {
                        if (srcObj == obj)
                            srcObj = nullptr;
                    }
                    RetireBakedStaticBatchAt(b);
                    continue;
                }
                ++b;
            }
        }

        const std::vector<RenderObject*>& GetRenderObjects()   const { return m_renderObjects; }

        GPUSceneHandle GetObjectGPUSceneHandle(const RenderObject* obj) const
        {
            return obj ? obj->GetGPUSceneHandle(this) : GPUSceneHandle{};
        }

        GPUBuffer* GetGPUSceneBuffer() const
        {
            return m_gpuSceneBuffer;
        }

        GPUBuffer* GetGPUSkinningPaletteBuffer() const
        {
            return m_gpuSkinningPaletteBuffer;
        }

        UINT32 GetGPUSkinningPaletteCount() const
        {
            return m_gpuSkinningPaletteCount;
        }

        GPUBuffer* GetGPUSkinningBoundsBuffer() const
        {
            return m_gpuSkinningBoundsBuffer;
        }

        UINT32 GetGPUSkinningBoundsCount() const
        {
            return m_gpuSkinningBoundsCount;
        }

        GPUBuffer* GetGPUMaterialSlotTableBuffer() const
        {
            return m_gpuMaterialSlotTableBuffer;
        }

        UINT32 GetGPUMaterialSlotTableCount() const
        {
            return m_gpuMaterialSlotTableCount;
        }

        const GPUSceneUpdateStats& GetGPUSceneStats() const
        {
            return m_gpuSceneStats;
        }

        SIZE_T GetPendingDirtyObjectCount() const
        {
            return m_dirtyObjects.size();
        }

        void AddCamera(Camera* camera)
        {
            if (!camera) return;
            if (std::find(m_cameras.begin(), m_cameras.end(), camera) == m_cameras.end())
            {
                m_cameras.push_back(camera);
            }
        }
        void RemoveCamera(Camera* camera)
        {
            if (!camera) return;
            if (m_mainCamera == camera)
                m_mainCamera = nullptr;
            m_cameras.erase(std::remove(m_cameras.begin(), m_cameras.end(), camera), m_cameras.end());
        }
        const std::vector<Camera*>& GetCameras()               const { return m_cameras; }

        void AddLight(Light* light)
        {
            if (!light) return;
            if (std::find(m_lights.begin(), m_lights.end(), light) == m_lights.end())
            {
                m_lights.push_back(light);
            }
        }
        void RemoveLight(Light* light)
        {
            if (!light) return;
            if (m_directionalLight == light)
                m_directionalLight = nullptr;
            m_lights.erase(std::remove(m_lights.begin(), m_lights.end(), light), m_lights.end());
        }
        const std::vector<Light*>& GetLights()                 const { return m_lights; }

        void OnMaterialDestroyed(const Material* destroyedMaterial)
        {
            if (!destroyedMaterial) return;
            for (auto& batch : m_bakedStaticBatches)
            {
                if (batch.material == destroyedMaterial)
                {
                    batch.material = nullptr;
                }
            }
        }

        void OnMeshDestroyed(const Mesh* destroyedMesh)
        {
            if (!destroyedMesh) return;
            for (auto& batch : m_bakedStaticBatches)
            {
                if (batch.combinedMesh == destroyedMesh)
                {
                    batch.combinedMesh = nullptr;
                }
            }
        }

        // ==================== MAIN CAMERA ====================
        void    SetMainCamera(Camera* camera)                         { m_mainCamera = camera; }
        Camera* GetMainCamera()                                 const { return m_mainCamera ? m_mainCamera : (!m_cameras.empty() ? m_cameras[0] : nullptr); }

        // ==================== DIRECTIONAL LIGHT ====================
        void   SetDirectionalLight(Light* light)                      { m_directionalLight = light; }
        Light* GetDirectionalLight()                            const { return m_directionalLight;  }

        // ==================== AMBIENT LIGHTING ====================
        const Color& GetAmbientColor()                          const { return m_ambientColor;     }
        FLOAT32      GetAmbientIntensity()                      const { return m_ambientIntensity; }

        void SetAmbientColor(const Color& color)                      { m_ambientColor = color;         }
        void SetAmbientIntensity(FLOAT32 intensity)                   { m_ambientIntensity = intensity; }

        // ==================== SCENE / ENVIRONMENT COMPUTE KERNELS ====================
        // Scene-level ComputeKernels automatically receive scene camera, lights, environment, and time
        // via NSL Get(...) engine properties and execute in the specified ComputePassPhase
        // (PRE_PASS, POST_DEPTH, POST_OPAQUE, or POST_PASS) and ComputeExecutionScope (PER_VIEW or PER_FRAME).
        void AddComputeKernel(ComputeKernel* kernel, ComputePassPhase phase = ComputePassPhase::PRE_PASS)
        {
            if (!kernel) return;
            AddComputeKernelInternal(kernel, phase, kernel->GetExecutionScope(), false);
        }

        void AddComputeKernel(ComputeKernel* kernel, ComputePassPhase phase, ComputeExecutionScope scope)
        {
            if (!kernel) return;
            AddComputeKernelInternal(kernel, phase, scope, true);
        }

        void RemoveComputeKernel(ComputeKernel* kernel)
        {
            auto removeFromVec = [kernel](std::vector<ComputeKernel*>& vec)
            {
                vec.erase(std::remove(vec.begin(), vec.end(), kernel), vec.end());
            };
            removeFromVec(m_prePassComputeKernels);
            removeFromVec(m_postDepthComputeKernels);
            removeFromVec(m_postOpaqueComputeKernels);
            removeFromVec(m_postPassComputeKernels);

            m_computeKernelEntries.erase(
                std::remove_if(m_computeKernelEntries.begin(), m_computeKernelEntries.end(),
                               [kernel](const SceneComputeKernelEntry& e) { return e.kernel == kernel; }),
                m_computeKernelEntries.end());
        }

        void ClearComputeKernels()
        {
            m_prePassComputeKernels.clear();
            m_postDepthComputeKernels.clear();
            m_postOpaqueComputeKernels.clear();
            m_postPassComputeKernels.clear();
            m_computeKernelEntries.clear();
        }

        const std::vector<ComputeKernel*>&          GetPrePassComputeKernels()    const { return m_prePassComputeKernels;    }
        const std::vector<ComputeKernel*>&          GetPostDepthComputeKernels()  const { return m_postDepthComputeKernels;  }
        const std::vector<ComputeKernel*>&          GetPostOpaqueComputeKernels() const { return m_postOpaqueComputeKernels; }
        const std::vector<ComputeKernel*>&          GetPostPassComputeKernels()   const { return m_postPassComputeKernels;   }
        const std::vector<SceneComputeKernelEntry>& GetComputeKernelEntries()     const { return m_computeKernelEntries;     }
        BOOL                                        HasComputeKernels()           const { return !m_computeKernelEntries.empty(); }

        // ==================== STATIC BATCHING ====================
        BOOL                                 IsStaticBatched() const { return m_isStaticBatched; }
        void                                 SetStaticBatched(BOOL batched) { m_isStaticBatched = batched; }
        const std::vector<BakedStaticBatch>& GetBakedStaticBatches() const { return m_bakedStaticBatches; }

        void AddBakedStaticBatch(const BakedStaticBatch& batch)
        {
            m_bakedStaticBatches.push_back(batch);
            if (m_nextStaticBatchGroupId == 0u)
                m_nextStaticBatchGroupId = 1u;
            const UINT32 batchId = m_nextStaticBatchGroupId++;
            if (m_nextStaticBatchGroupId == 0u)
                m_nextStaticBatchGroupId = 1u;
            for (SIZE_T k = 0; k < batch.sourceObjects.size(); ++k)
            {
                RenderObject* srcObj = batch.sourceObjects[k];
                if (!srcObj) continue;
                if (SceneObjectLink* link = srcObj->FindSceneLink(this))
                {
                    link->batchGroupId      = batchId;
                    link->batchSubmeshIndex = static_cast<UINT32>(k);
                    link->batchStartIndex   = (k < batch.submeshes.size()) ? batch.submeshes[k].startIndex : 0u;
                    link->batchIndexCount   = (k < batch.submeshes.size()) ? batch.submeshes[k].indexCount : 0u;
                }
                srcObj->MarkGPUSceneDirty(GPU_SCENE_DIRTY_BATCHING | GPU_SCENE_DIRTY_FLAGS);
            }
        }

        // ==================== OBJECT QUERIES & CLEAR ====================
        void Clear()
        {
            while (!m_bakedStaticBatches.empty())
            {
                RetireBakedStaticBatchAt(m_bakedStaticBatches.size() - 1u);
            }
            DetachAllRenderObjects();
            m_renderObjects.clear();
            m_dirtyObjects.clear();
            m_pendingRemovedSlots.clear();
            m_gpuSceneCleared = true;
            m_cameras.clear();
            m_lights.clear();
            ClearComputeKernels();
            m_mainCamera = nullptr;
            m_directionalLight = nullptr;
            m_bakedStaticBatches.clear();
            m_isStaticBatched = false;
            m_nextStaticBatchGroupId = 1u;
        }

        BOOL HasRenderObject(RenderObject* obj) const
        {
            if (!obj) return false;
            return obj->FindSceneLink(this) != nullptr;
        }

        BOOL HasCamera(Camera* camera) const
        {
            return std::find(m_cameras.begin(), m_cameras.end(), camera) != m_cameras.end();
        }

        BOOL HasLight(Light* light) const
        {
            return std::find(m_lights.begin(), m_lights.end(), light) != m_lights.end();
        }

        // ==================== CULLING & PASS QUERIES ====================
        std::vector<RenderObject*> GetVisibleObjects(const Camera* camera) const
        {
            std::vector<RenderObject*> visible;
            if (!camera) return visible;

            const UINT32 camMask = camera->GetCullingMask();
            if (camMask == 0u) return visible;

            const Frustum frustum = camera->GetFrustum();

            // 1. Process Baked Static Batches (Pre-transformed world-space combined meshes)
            if (m_isStaticBatched)
            {
                for (SIZE_T b = 0; b < m_bakedStaticBatches.size(); ++b)
                {
                    RenderObject* batchObj = m_bakedStaticBatches[b].batchObject;
                    if (!batchObj || !batchObj->HasFlag(RenderObjectFlags::VISIBLE))
                        continue;
                    if ((camMask & batchObj->GetLayerMask()) == 0u)
                        continue;

                    if (batchObj->HasFlag(RenderObjectFlags::FRUSTUM_CULL))
                    {
                        if (!frustum.IntersectsAABB(m_bakedStaticBatches[b].worldAABB))
                            continue;

                        if (!m_bakedStaticBatches[b].submeshes.empty())
                        {
                            BOOL anySubmeshVisible = false;
                            for (const auto& sub : m_bakedStaticBatches[b].submeshes)
                            {
                                if (!sub.worldAABB.IsValid() || frustum.IntersectsAABB(sub.worldAABB))
                                {
                                    anySubmeshVisible = true;
                                    break;
                                }
                            }
                            if (!anySubmeshVisible)
                                continue;
                        }
                    }

                    visible.push_back(batchObj);
                }
            }

            // 2. Individual Scene Objects
            for (SIZE_T i = 0; i < m_renderObjects.size(); ++i)
            {
                RenderObject* obj = m_renderObjects[i];
                if (!obj) continue;

                // Objects merged into a static batch mesh are drawn through the combined batch
                if (m_isStaticBatched && obj->HasFlag(RenderObjectFlags::BAKED_STATIC))
                    continue;

                if (!obj->HasFlag(RenderObjectFlags::VISIBLE))
                    continue;

                if ((camMask & obj->GetLayerMask()) == 0u)
                    continue;

                if (obj->HasFlag(RenderObjectFlags::FRUSTUM_CULL))
                {
                    if (!frustum.IntersectsAABB(obj->GetWorldAABB()))
                        continue;
                }

                visible.push_back(obj);
            }
            return visible;
        }

        std::vector<RenderObject*> GetShadowCasters() const
        {
            std::vector<RenderObject*> casters;

            if (m_isStaticBatched)
            {
                for (SIZE_T b = 0; b < m_bakedStaticBatches.size(); ++b)
                {
                    RenderObject* batchObj = m_bakedStaticBatches[b].batchObject;
                    if (batchObj && batchObj->HasFlag(RenderObjectFlags::VISIBLE) && batchObj->HasFlag(RenderObjectFlags::CAST_SHADOW))
                    {
                        casters.push_back(batchObj);
                    }
                }
            }

            for (SIZE_T i = 0; i < m_renderObjects.size(); ++i)
            {
                RenderObject* obj = m_renderObjects[i];
                if (!obj) continue;

                if (m_isStaticBatched && obj->HasFlag(RenderObjectFlags::BAKED_STATIC))
                    continue;

                if (obj->HasFlag(RenderObjectFlags::VISIBLE) && obj->HasFlag(RenderObjectFlags::CAST_SHADOW))
                {
                    casters.push_back(obj);
                }
            }
            return casters;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreSceneManager;
        friend class CoreGPUSceneManager;

        Scene() = default;

        static void OnLinkedObjectDirtyThunk(Scene* scene, RenderObject* obj, SceneObjectLink& link)
        {
            if (!scene || !obj) return;
            if (link.dirtyListIndex == 0xFFFFFFFFu)
            {
                link.dirtyListIndex = static_cast<UINT32>(scene->m_dirtyObjects.size());
                scene->m_dirtyObjects.push_back(obj);
            }
        }

        void DetachAllRenderObjects()
        {
            for (RenderObject* obj : m_renderObjects)
            {
                if (obj)
                {
                    obj->DetachSceneLink(this);
                }
            }
        }

        void ClearBakedStaticBatches()
        {
            while (!m_bakedStaticBatches.empty())
            {
                RetireBakedStaticBatchAt(m_bakedStaticBatches.size() - 1u);
            }
            m_isStaticBatched = false;
        }

        void RetireBakedStaticBatchAt(SIZE_T batchIdx)
        {
            if (batchIdx >= m_bakedStaticBatches.size())
                return;

            BakedStaticBatch& batch = m_bakedStaticBatches[batchIdx];
            for (RenderObject* srcObj : batch.sourceObjects)
            {
                if (!srcObj) continue;
                if (SceneObjectLink* link = srcObj->FindSceneLink(this))
                {
                    link->batchGroupId      = 0u;
                    link->batchSubmeshIndex = 0u;
                    link->batchStartIndex   = 0u;
                    link->batchIndexCount   = 0u;
                }
                srcObj->SetBakedIntoStaticBatch(false);
            }

            if (batch.batchObject != nullptr)
            {
                m_retiredStaticBatchObjects.push_back(batch.batchObject);
                batch.batchObject = nullptr;
            }
            if (batch.combinedMesh != nullptr)
            {
                m_retiredStaticBatchMeshes.push_back(batch.combinedMesh);
                batch.combinedMesh = nullptr;
            }

            m_bakedStaticBatches.erase(m_bakedStaticBatches.begin() + static_cast<std::ptrdiff_t>(batchIdx));
            if (m_bakedStaticBatches.empty())
            {
                m_isStaticBatched = false;
            }
        }

        // Scene Objects
        std::vector<RenderObject*>    m_renderObjects;
        std::vector<Camera*>          m_cameras;
        std::vector<Light*>           m_lights;

        // Incremental GPU Scene Tracking (O(1) dirty list + pending slot removals)
        std::vector<RenderObject*>    m_dirtyObjects;
        std::vector<UINT32>           m_pendingRemovedSlots;
        BOOL                          m_gpuSceneCleared          = false;
        GPUBuffer*                    m_gpuSceneBuffer           = nullptr;
        GPUBuffer*                    m_gpuSkinningPaletteBuffer = nullptr;
        UINT32                        m_gpuSkinningPaletteCount  = 0u;
        GPUBuffer*                    m_gpuSkinningBoundsBuffer     = nullptr;
        UINT32                        m_gpuSkinningBoundsCount      = 0u;
        GPUBuffer*                    m_gpuMaterialSlotTableBuffer  = nullptr;
        UINT32                        m_gpuMaterialSlotTableCount   = 0u;
        GPUSceneUpdateStats           m_gpuSceneStats               = {};

        // Main Camera & Directional Light
        Camera*                       m_mainCamera       = nullptr;
        Light*                        m_directionalLight = nullptr;

        // Ambient Lighting
        Color                         m_ambientColor     = Color(0.1f, 0.1f, 0.15f, 1.0f);
        FLOAT32                       m_ambientIntensity = 1.0f;

        // Static Batching
        BOOL                          m_isStaticBatched        = false;
        UINT32                        m_nextStaticBatchGroupId = 1u;
        std::vector<BakedStaticBatch> m_bakedStaticBatches;
        std::vector<RenderObject*>    m_retiredStaticBatchObjects;
        std::vector<Mesh*>            m_retiredStaticBatchMeshes;

        // Scene / Environment Compute Kernels
        std::vector<ComputeKernel*>          m_prePassComputeKernels;
        std::vector<ComputeKernel*>          m_postDepthComputeKernels;
        std::vector<ComputeKernel*>          m_postOpaqueComputeKernels;
        std::vector<ComputeKernel*>          m_postPassComputeKernels;
        std::vector<SceneComputeKernelEntry> m_computeKernelEntries;

        std::vector<ComputeKernel*>& GetPhaseVector(ComputePassPhase phase)
        {
            switch (phase)
            {
                case ComputePassPhase::POST_DEPTH:  return m_postDepthComputeKernels;
                case ComputePassPhase::POST_OPAQUE: return m_postOpaqueComputeKernels;
                case ComputePassPhase::POST_PASS:   return m_postPassComputeKernels;
                case ComputePassPhase::PRE_PASS:
                default:                            return m_prePassComputeKernels;
            }
        }

        void AddComputeKernelInternal(
            ComputeKernel* kernel,
            ComputePassPhase phase,
            ComputeExecutionScope scope,
            BOOL hasExplicitScope)
        {
            if (!kernel) return;

            auto& list = GetPhaseVector(phase);
            if (std::find(list.begin(), list.end(), kernel) == list.end())
            {
                list.push_back(kernel);
            }

            for (auto& entry : m_computeKernelEntries)
            {
                if (entry.kernel == kernel && entry.phase == phase)
                {
                    entry.scope            = scope;
                    entry.hasExplicitScope = hasExplicitScope;
                    return;
                }
            }

            SceneComputeKernelEntry entry{};
            entry.kernel           = kernel;
            entry.phase            = phase;
            entry.scope            = scope;
            entry.hasExplicitScope = hasExplicitScope;
            m_computeKernelEntries.push_back(entry);
        }
    };
}
