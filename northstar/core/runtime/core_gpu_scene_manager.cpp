#include "core_gpu_scene_manager.h"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace Northstar
{
    namespace
    {
        static void MarkSkinningPaletteDirtyAllRings(GPUSceneInstance& inst, UINT32 firstBone, UINT32 boneCount) noexcept
        {
            if (boneCount == 0u) return;
            const UINT32 endBone = firstBone + boneCount;
            for (UINT32 r = 0u; r < GPUSceneInstance::kSkinningRingCount; ++r)
            {
                if (firstBone < inst.skinningRingDirtyMin[r])
                    inst.skinningRingDirtyMin[r] = firstBone;
                if (endBone > inst.skinningRingDirtyMax[r])
                    inst.skinningRingDirtyMax[r] = endBone;
            }
        }

        static void FreeSkinningPaletteRange(GPUSceneInstance& inst, UINT32 offset, UINT32 capacity)
        {
            if (capacity == 0u) return;

            SkinningPaletteFreeRange nr{ offset, capacity };
            auto pos = std::lower_bound(
                inst.skinningFreeRanges.begin(),
                inst.skinningFreeRanges.end(),
                nr,
                [](const SkinningPaletteFreeRange& a, const SkinningPaletteFreeRange& b) {
                    return a.offset < b.offset;
                });
            pos = inst.skinningFreeRanges.insert(pos, nr);

            if (pos + 1 != inst.skinningFreeRanges.end() &&
                pos->offset + pos->count == (pos + 1)->offset)
            {
                pos->count += (pos + 1)->count;
                inst.skinningFreeRanges.erase(pos + 1);
            }
            if (pos != inst.skinningFreeRanges.begin())
            {
                auto prev = pos - 1;
                if (prev->offset + prev->count == pos->offset)
                {
                    prev->count += pos->count;
                    inst.skinningFreeRanges.erase(pos);
                }
            }

            if (!inst.skinningFreeRanges.empty())
            {
                const SkinningPaletteFreeRange& tail = inst.skinningFreeRanges.back();
                if (tail.offset + tail.count == static_cast<UINT32>(inst.cpuSkinningPalette.size()))
                {
                    inst.cpuSkinningPalette.resize(tail.offset);
                    inst.skinningFreeRanges.pop_back();
                }
            }
        }

        static UINT32 AllocateSkinningPaletteRange(GPUSceneInstance& inst, UINT32 count)
        {
            for (SIZE_T i = 0u; i < inst.skinningFreeRanges.size(); ++i)
            {
                if (inst.skinningFreeRanges[i].count >= count)
                {
                    const UINT32 allocOffset = inst.skinningFreeRanges[i].offset;
                    if (inst.skinningFreeRanges[i].count == count)
                    {
                        inst.skinningFreeRanges.erase(inst.skinningFreeRanges.begin() + static_cast<std::ptrdiff_t>(i));
                    }
                    else
                    {
                        inst.skinningFreeRanges[i].offset += count;
                        inst.skinningFreeRanges[i].count  -= count;
                    }
                    return allocOffset;
                }
            }

            const UINT32 appendOffset = static_cast<UINT32>(inst.cpuSkinningPalette.size());
            inst.cpuSkinningPalette.resize(static_cast<SIZE_T>(appendOffset) + static_cast<SIZE_T>(count), Mat4::Identity());
            return appendOffset;
        }
    }

    CoreGPUSceneManager::~CoreGPUSceneManager()
    {
        Shutdown();
    }

    void CoreGPUSceneManager::Initialize(CoreBufferManager* bufferManager)
    {
        m_bufferManager             = bufferManager;
        m_simulateAllocationFailure = false;
        m_simulateUploadFailure     = false;
        m_maxSlotCapacity           = kMaxGPUSceneSlotCapacity;
        m_lastError.clear();
    }

    void CoreGPUSceneManager::Shutdown()
    {
        for (auto& inst : m_instances)
        {
            if (inst.scene)
            {
                for (RenderObject* ro : inst.scene->m_renderObjects)
                {
                    if (!ro) continue;
                    if (SceneObjectLink* link = ro->FindSceneLink(inst.scene))
                    {
                        link->gpuHandle             = GPUSceneHandle{};
                        link->pendingDirtyFlags    |= GPU_SCENE_DIRTY_ALL;
                        link->skinningPaletteOffset = 0u;
                        link->skinningBoneCount     = 0u;
                        if (link->dirtyListIndex == 0xFFFFFFFFu)
                        {
                            link->dirtyListIndex = static_cast<UINT32>(inst.scene->m_dirtyObjects.size());
                            inst.scene->m_dirtyObjects.push_back(ro);
                        }
                    }
                }
                inst.scene->m_pendingRemovedSlots.clear();
                inst.scene->m_gpuSceneCleared          = false;
                inst.scene->m_gpuSceneBuffer           = nullptr;
                inst.scene->m_gpuSkinningPaletteBuffer = nullptr;
                inst.scene->m_gpuSkinningPaletteCount  = 0u;
                inst.scene->m_gpuSceneStats            = {};
            }
            if (inst.gpuBuffer && m_bufferManager)
            {
                m_bufferManager->DestroyGPUBuffer(inst.gpuBuffer);
                inst.gpuBuffer = nullptr;
            }
            for (UINT32 r = 0u; r < GPUSceneInstance::kSkinningRingCount; ++r)
            {
                if (inst.skinningRingBuffers[r] && m_bufferManager)
                {
                    m_bufferManager->DestroyGPUBuffer(inst.skinningRingBuffers[r]);
                    inst.skinningRingBuffers[r]    = nullptr;
                    inst.skinningRingCapacities[r] = 0u;
                }
            }
            inst.activeSkinningBuffer = nullptr;
        }
        m_instances.clear();
        m_bufferManager             = nullptr;
        m_simulateAllocationFailure = false;
        m_simulateUploadFailure     = false;
        m_maxSlotCapacity           = kMaxGPUSceneSlotCapacity;
        m_lastError.clear();
    }

    void CoreGPUSceneManager::OnSceneDestroyed(Scene* scene)
    {
        if (!scene) return;
        for (SIZE_T i = 0; i < m_instances.size(); ++i)
        {
            if (m_instances[i].scene == scene)
            {
                scene->m_gpuSceneBuffer           = nullptr;
                scene->m_gpuSkinningPaletteBuffer = nullptr;
                scene->m_gpuSkinningPaletteCount  = 0u;
                scene->m_gpuSceneStats            = {};
                if (m_instances[i].gpuBuffer && m_bufferManager)
                {
                    m_bufferManager->DestroyGPUBuffer(m_instances[i].gpuBuffer);
                    m_instances[i].gpuBuffer = nullptr;
                }
                for (UINT32 r = 0u; r < GPUSceneInstance::kSkinningRingCount; ++r)
                {
                    if (m_instances[i].skinningRingBuffers[r] && m_bufferManager)
                    {
                        m_bufferManager->DestroyGPUBuffer(m_instances[i].skinningRingBuffers[r]);
                        m_instances[i].skinningRingBuffers[r]    = nullptr;
                        m_instances[i].skinningRingCapacities[r] = 0u;
                    }
                }
                m_instances[i].activeSkinningBuffer = nullptr;
                m_instances.erase(m_instances.begin() + static_cast< std::ptrdiff_t >(i));
                return;
            }
        }
    }

    void CoreGPUSceneManager::OnRenderObjectDestroyed(RenderObject* destroyedObject)
    {
        if (!destroyedObject) return;
        for (auto& inst : m_instances)
        {
            for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
            {
                GPUSceneSlotEntry& se = inst.slots[s];
                if (se.object == destroyedObject)
                {
                    if (se.skinningPaletteCapacity > 0u)
                    {
                        FreeSkinningPaletteRange(inst, se.skinningPaletteOffset, se.skinningPaletteCapacity);
                        se.skinningPaletteOffset   = 0u;
                        se.skinningPaletteCapacity = 0u;
                        se.skinningBoneCount       = 0u;
                    }
                    se.object                = nullptr;
                    se.boundMesh             = nullptr;
                    se.boundMeshRevision     = 0u;
                    se.boundMaterial         = nullptr;
                    se.boundMaterialRevision = 0u;
                    se.boundLODGroup         = nullptr;
                    se.boundLODGroupRevision = 0u;
                }
            }
        }
    }

    void CoreGPUSceneManager::OnMeshDestroyed(Mesh* destroyedMesh)
    {
        if (!destroyedMesh) return;
        for (auto& inst : m_instances)
        {
            inst.watchedMeshes.erase(
                std::remove_if(inst.watchedMeshes.begin(), inst.watchedMeshes.end(),
                               [destroyedMesh](const GPUSceneWatchedMesh& w) { return w.mesh == destroyedMesh; }),
                inst.watchedMeshes.end());
            const BOOL sceneCleared = (inst.scene && inst.scene->m_gpuSceneCleared);
            if (inst.scene)
            {
                for (RenderObject* ro : inst.scene->m_renderObjects)
                {
                    if (ro && ro->GetMesh() == destroyedMesh)
                    {
                        ro->SetMesh(nullptr);
                    }
                }
            }
            for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
            {
                GPUSceneSlotEntry& se = inst.slots[s];
                if (sceneCleared)
                {
                    se.object = nullptr;
                    if (se.boundMesh == destroyedMesh)
                    {
                        se.boundMesh         = nullptr;
                        se.boundMeshRevision = 0u;
                    }
                    continue;
                }
                if (se.boundMesh == destroyedMesh || (se.occupied && se.object && se.object->GetMesh() == destroyedMesh))
                {
                    if (se.occupied && se.object && se.object->GetMesh() == destroyedMesh)
                    {
                        se.object->SetMesh(nullptr);
                    }
                    se.boundMesh         = nullptr;
                    se.boundMeshRevision = 0u;
                    if (se.occupied)
                    {
                        se.dirtyMask |= (GPU_SCENE_DIRTY_MESH | GPU_SCENE_DIRTY_BOUNDS);
                        if (!se.queuedDirty)
                        {
                            se.queuedDirty = true;
                            inst.dirtySlotIndices.push_back(s);
                        }
                    }
                }
            }
        }
    }

    void CoreGPUSceneManager::OnMaterialDestroyed(Material* destroyedMaterial)
    {
        if (!destroyedMaterial) return;
        for (auto& inst : m_instances)
        {
            inst.watchedMaterials.erase(
                std::remove_if(inst.watchedMaterials.begin(), inst.watchedMaterials.end(),
                               [destroyedMaterial](const GPUSceneWatchedMaterial& w) { return w.material == destroyedMaterial; }),
                inst.watchedMaterials.end());
            const BOOL sceneCleared = (inst.scene && inst.scene->m_gpuSceneCleared);
            if (inst.scene)
            {
                for (RenderObject* ro : inst.scene->m_renderObjects)
                {
                    if (ro && ro->GetMaterial() == destroyedMaterial)
                    {
                        ro->SetMaterial(nullptr);
                    }
                }
            }
            for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
            {
                GPUSceneSlotEntry& se = inst.slots[s];
                if (sceneCleared)
                {
                    se.object = nullptr;
                    if (se.boundMaterial == destroyedMaterial)
                    {
                        se.boundMaterial         = nullptr;
                        se.boundMaterialRevision = 0u;
                    }
                    continue;
                }
                if (se.boundMaterial == destroyedMaterial || (se.occupied && se.object && se.object->GetMaterial() == destroyedMaterial))
                {
                    if (se.occupied && se.object && se.object->GetMaterial() == destroyedMaterial)
                    {
                        se.object->SetMaterial(nullptr);
                    }
                    se.boundMaterial         = nullptr;
                    se.boundMaterialRevision = 0u;
                    if (se.occupied)
                    {
                        se.dirtyMask |= GPU_SCENE_DIRTY_MATERIAL;
                        if (!se.queuedDirty)
                        {
                            se.queuedDirty = true;
                            inst.dirtySlotIndices.push_back(s);
                        }
                    }
                }
            }
        }
    }

    void CoreGPUSceneManager::OnLODGroupDestroyed(LODGroup* destroyedLODGroup)
    {
        if (!destroyedLODGroup) return;
        for (auto& inst : m_instances)
        {
            inst.watchedLODGroups.erase(
                std::remove_if(inst.watchedLODGroups.begin(), inst.watchedLODGroups.end(),
                               [destroyedLODGroup](const GPUSceneWatchedLODGroup& w) { return w.lodGroup == destroyedLODGroup; }),
                inst.watchedLODGroups.end());
            const BOOL sceneCleared = (inst.scene && inst.scene->m_gpuSceneCleared);
            if (inst.scene)
            {
                for (RenderObject* ro : inst.scene->m_renderObjects)
                {
                    if (ro && ro->GetLODGroup() == destroyedLODGroup)
                    {
                        ro->SetLODGroup(nullptr);
                    }
                }
            }
            for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
            {
                GPUSceneSlotEntry& se = inst.slots[s];
                if (sceneCleared)
                {
                    se.object = nullptr;
                    if (se.boundLODGroup == destroyedLODGroup)
                    {
                        se.boundLODGroup         = nullptr;
                        se.boundLODGroupRevision = 0u;
                    }
                    continue;
                }
                if (se.boundLODGroup == destroyedLODGroup || (se.occupied && se.object && se.object->GetLODGroup() == destroyedLODGroup))
                {
                    if (se.occupied && se.object && se.object->GetLODGroup() == destroyedLODGroup)
                    {
                        se.object->SetLODGroup(nullptr);
                    }
                    se.boundLODGroup         = nullptr;
                    se.boundLODGroupRevision = 0u;
                    if (se.occupied)
                    {
                        se.dirtyMask |= GPU_SCENE_DIRTY_LOD_GROUP;
                        if (!se.queuedDirty)
                        {
                            se.queuedDirty = true;
                            inst.dirtySlotIndices.push_back(s);
                        }
                    }
                }
            }
        }
    }

    void CoreGPUSceneManager::ForceSlotNextGeneration(Scene* scene, UINT32 slotIndex, UINT32 nextGen)
    {
        if (!scene) return;
        GPUSceneInstance* inst = FindInstance(scene);
        if (!inst || slotIndex >= inst->slots.size()) return;
        inst->slots[slotIndex].nextGeneration = (nextGen == 0u) ? 1u : nextGen;
    }

    GPUSceneInstance* CoreGPUSceneManager::FindInstance(const Scene* scene)
    {
        if (!scene) return nullptr;
        for (auto& inst : m_instances)
        {
            if (inst.scene == scene)
                return &inst;
        }
        return nullptr;
    }

    const GPUSceneInstance* CoreGPUSceneManager::FindInstance(const Scene* scene) const
    {
        if (!scene) return nullptr;
        for (const auto& inst : m_instances)
        {
            if (inst.scene == scene)
                return &inst;
        }
        return nullptr;
    }

    GPUSceneInstance& CoreGPUSceneManager::GetOrCreateInstance(Scene* scene)
    {
        if (GPUSceneInstance* existing = FindInstance(scene))
            return *existing;

        GPUSceneInstance inst{};
        inst.scene = scene;
        m_instances.push_back(std::move(inst));
        return m_instances.back();
    }

    UINT32 CoreGPUSceneManager::ComputeTargetCapacity(UINT32 currentCapacity, UINT32 requiredSlots)
    {
        UINT32 cap = (currentCapacity >= kDefaultGPUSceneMinCapacity)
            ? currentCapacity
            : kDefaultGPUSceneMinCapacity;
        while (cap < requiredSlots)
        {
            if (cap > 0x3FFFFFFFu)
            {
                cap = requiredSlots;
                break;
            }
            cap *= 2u;
        }
        return cap;
    }

    void CoreGPUSceneManager::BuildInvalidSlotRecord(UINT32 slotIndex, GPUSceneObjectData& outRecord)
    {
        outRecord                     = GPUSceneObjectData{};
        outRecord.slotIndex           = slotIndex;
        outRecord.generation          = 0u;
        outRecord.objectFlags         = GPU_SCENE_OBJ_NONE;
        outRecord.renderObjectFlags   = 0u;
        outRecord.meshId              = 0u;
        outRecord.materialId          = 0u;
        outRecord.lodGroupId          = 0u;
        outRecord.batchGroupId        = 0u;
        outRecord.vertexCount         = 0u;
        outRecord.indexCount          = 0u;
        outRecord.startIndex          = 0u;
        outRecord.baseVertex          = 0u;
        outRecord.aabbMin             = Vec3(0.0f, 0.0f, 0.0f);
        outRecord.submeshIndex        = 0u;
        outRecord.aabbMax             = Vec3(0.0f, 0.0f, 0.0f);
        outRecord.lodAndMaterialFlags = 0u;
        outRecord.boundingSphere      = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        outRecord.worldMatrix         = Mat4::Identity();
    }

    void CoreGPUSceneManager::BuildObjectRecord(
        const RenderObject*    obj,
        const SceneObjectLink* link,
        UINT32                 slotIndex,
        UINT32                 generation,
        GPUSceneObjectData&    outRecord)
    {
        if (!obj || generation == 0u)
        {
            BuildInvalidSlotRecord(slotIndex, outRecord);
            return;
        }

        outRecord.slotIndex  = slotIndex;
        outRecord.generation = generation;

        UINT32 objFlags = GPU_SCENE_OBJ_VALID;
        if (obj->HasFlag(RenderObjectFlags::RECEIVE_SHADOW))    objFlags |= GPU_SCENE_OBJ_RECEIVE_SHADOW;
        if (obj->HasFlag(RenderObjectFlags::VISIBLE))           objFlags |= GPU_SCENE_OBJ_VISIBLE;
        if (obj->HasFlag(RenderObjectFlags::STATIC))            objFlags |= GPU_SCENE_OBJ_STATIC;
        if (obj->HasFlag(RenderObjectFlags::CAST_SHADOW))       objFlags |= GPU_SCENE_OBJ_CAST_SHADOW;
        if (obj->HasFlag(RenderObjectFlags::FRUSTUM_CULL))      objFlags |= GPU_SCENE_OBJ_FRUSTUM_CULL;
        if (obj->HasFlag(RenderObjectFlags::DISTANCE_CULL))     objFlags |= GPU_SCENE_OBJ_DISTANCE_CULL;
        if (obj->HasFlag(RenderObjectFlags::SMALL_OBJECT_CULL)) objFlags |= GPU_SCENE_OBJ_SMALL_OBJECT_CULL;
        if (obj->HasFlag(RenderObjectFlags::OCCLUSION_CULL))    objFlags |= GPU_SCENE_OBJ_OCCLUSION_CULL;
        if (obj->HasFlag(RenderObjectFlags::USE_LOD))           objFlags |= GPU_SCENE_OBJ_USE_LOD;
        if (obj->HasFlag(RenderObjectFlags::SKINNED) || obj->HasSkinningMatrices())
        {
            objFlags |= GPU_SCENE_OBJ_SKINNED;
        }
        if (obj->HasFlag(RenderObjectFlags::BAKED_STATIC))      objFlags |= GPU_SCENE_OBJ_BAKED_STATIC;

        const Mesh*     mesh = obj->GetMesh();
        const Material* mat  = obj->GetMaterial();
        const LODGroup* lod  = obj->GetLODGroup();

        outRecord.meshId       = mesh ? mesh->GetGPUSceneMeshId()     : 0u;
        outRecord.materialId   = mat  ? mat->GetGPUSceneMaterialId()  : 0u;
        outRecord.lodGroupId   = lod  ? lod->GetGPUSceneLODGroupId()  : 0u;

        const BOOL isStaticBatchMember =
            (link != nullptr) &&
            (link->batchGroupId != 0u) &&
            !obj->HasSkinningMatrices() &&
            !obj->HasFlag(RenderObjectFlags::SKINNED);

        const UINT32 skinBoneCount = (!isStaticBatchMember && link != nullptr) ? link->skinningBoneCount : 0u;
        outRecord.renderObjectFlags = PackGPUSceneRenderFlagsAndBoneCount(
            static_cast<UINT32>(obj->GetFlags()),
            skinBoneCount);

        if (isStaticBatchMember)
        {
            outRecord.batchGroupId = link->batchGroupId;
            objFlags |= GPU_SCENE_OBJ_STATIC_BATCH_MEMBER;
        }
        else if (skinBoneCount > 0u)
        {
            // Full 32-bit scene-wide bonePaletteOffset (0 .. 4,294,967,295)
            outRecord.batchGroupId = link->skinningPaletteOffset;
        }
        else
        {
            outRecord.batchGroupId = 0u;
        }

        if (mesh != nullptr && mat != nullptr && !obj->HasFlag(RenderObjectFlags::BAKED_STATIC))
        {
            objFlags |= GPU_SCENE_OBJ_INSTANCED_CANDIDATE;
        }

        if (mesh != nullptr)
        {
            if (mesh->IsVirtualized())
            {
                objFlags |= GPU_SCENE_OBJ_VIRTUAL_GEOMETRY;
                const NVGMeshInfo& nvgInfo = mesh->GetVirtualMeshInfo();
                outRecord.vertexCount = (std::max)(nvgInfo.buildReport.sourceVertexCount, 1u);
                outRecord.indexCount  = (std::max)(nvgInfo.buildReport.validTriangleCount * 3u, 3u);
                outRecord.startIndex  = 0u;
                outRecord.baseVertex  = 0u;
            }
            else
            {
                outRecord.vertexCount = mesh->GetVertexCount();
                outRecord.indexCount  = mesh->HasIndices() ? mesh->GetIndexCount() : mesh->GetVertexCount();
                outRecord.startIndex  = isStaticBatchMember ? link->batchStartIndex : 0u;
                outRecord.baseVertex  = 0u;
                if (isStaticBatchMember && link->batchIndexCount > 0u)
                {
                    outRecord.indexCount = link->batchIndexCount;
                }
            }
        }
        else
        {
            outRecord.vertexCount = 0u;
            outRecord.indexCount  = 0u;
            outRecord.startIndex  = 0u;
            outRecord.baseVertex  = 0u;
        }

        outRecord.submeshIndex = link ? link->batchSubmeshIndex : 0u;

        const UINT32 lodCount      = lod ? static_cast<UINT32>(lod->GetLevels().size()) : 0u;
        const UINT32 cullMode      = mat ? static_cast<UINT32>(mat->GetCullMode())      : 1u;
        const UINT32 blendMode     = mat ? static_cast<UINT32>(mat->GetBlendMode())     : 0u;
        const BOOL   depthTest     = mat ? mat->IsDepthTestEnabled()                    : true;
        const BOOL   depthWrite    = mat ? mat->IsDepthWriteEnabled()                   : true;
        const BOOL   wireframe     = mat ? mat->IsWireframe()                           : false;
        const BOOL   hasGeomPass   = mat ? mat->HasPass(PassType::GEOMETRY)             : false;
        const BOOL   hasShadowPass = mat ? mat->HasPass(PassType::SHADOW)               : false;
        const BOOL   hasDepthPass  = mat ? mat->HasPass(PassType::DEPTH)                : false;

        if (mat != nullptr)
        {
            if (mat->IsAlphaTested()) objFlags |= GPU_SCENE_OBJ_ALPHA_TESTED;
            if (mat->IsTransparent()) objFlags |= GPU_SCENE_OBJ_TRANSPARENT;
            if (mat->IsTwoSided())    objFlags |= GPU_SCENE_OBJ_TWO_SIDED;
            if (!mat->IsTransparent() && depthTest && depthWrite && obj->HasFlag(RenderObjectFlags::VISIBLE))
            {
                objFlags |= GPU_SCENE_OBJ_DEPTH_PREPASS;
            }
        }
        if (obj->HasFlag(RenderObjectFlags::VISIBLE) && obj->HasFlag(RenderObjectFlags::CAST_SHADOW))
        {
            objFlags |= GPU_SCENE_OBJ_SHADOW_PASS;
        }

        outRecord.lodAndMaterialFlags = PackGPUSceneLODAndMaterialFlags(
            lodCount, cullMode, blendMode, depthTest, depthWrite,
            wireframe, hasGeomPass, hasShadowPass, hasDepthPass);

        outRecord.worldMatrix = obj->GetWorldMatrix();

        const AABB& aabb = obj->GetWorldAABB();
        outRecord.aabbMin = aabb.m_Min;
        outRecord.aabbMax = aabb.m_Max;

        const Vec3    center  = aabb.Center();
        const Vec3    extents = aabb.Extents();
        const FLOAT32 radius  = extents.Length();
        if (aabb.IsValid())
        {
            objFlags |= GPU_SCENE_OBJ_HAS_VALID_BOUNDS;
        }

        outRecord.boundingSphere = Vec4(center.x, center.y, center.z, radius);
        outRecord.objectFlags    = objFlags;
    }

    void CoreGPUSceneManager::WatchSlotResources(GPUSceneInstance& inst, GPUSceneSlotEntry& se)
    {
        if (!se.occupied || !se.object)
        {
            se.boundMesh             = nullptr;
            se.boundMeshRevision     = 0u;
            se.boundMaterial         = nullptr;
            se.boundMaterialRevision = 0u;
            se.boundLODGroup         = nullptr;
            se.boundLODGroupRevision = 0u;
            return;
        }

        Mesh*     mesh = se.object->GetMesh();
        Material* mat  = se.object->GetMaterial();
        LODGroup* lod  = se.object->GetLODGroup();

        se.boundMesh             = mesh;
        se.boundMeshRevision     = mesh ? mesh->GetStateRevision() : 0u;
        se.boundMaterial         = mat;
        se.boundMaterialRevision = mat  ? mat->GetStateRevision()  : 0u;
        se.boundLODGroup         = lod;
        se.boundLODGroupRevision = lod  ? lod->GetStateRevision()  : 0u;

        if (mesh)
        {
            BOOL found = false;
            for (const auto& w : inst.watchedMeshes)
            {
                if (w.mesh == mesh) { found = true; break; }
            }
            if (!found)
            {
                inst.watchedMeshes.push_back({ mesh, mesh->GetStateRevision() });
            }
        }

        if (mat)
        {
            BOOL found = false;
            for (const auto& w : inst.watchedMaterials)
            {
                if (w.material == mat) { found = true; break; }
            }
            if (!found)
            {
                inst.watchedMaterials.push_back({ mat, mat->GetStateRevision() });
            }
        }

        if (lod)
        {
            BOOL found = false;
            for (const auto& w : inst.watchedLODGroups)
            {
                if (w.lodGroup == lod) { found = true; break; }
            }
            if (!found)
            {
                inst.watchedLODGroups.push_back({ lod, lod->GetStateRevision() });
            }
        }
    }

    void CoreGPUSceneManager::CheckWatchedResourceMutations(GPUSceneInstance& inst)
    {
        BOOL anyMeshChanged = false;
        for (auto& w : inst.watchedMeshes)
        {
            if (w.mesh && w.mesh->GetStateRevision() != w.lastRevision)
            {
                w.lastRevision = w.mesh->GetStateRevision();
                anyMeshChanged = true;
            }
        }

        BOOL anyMaterialChanged = false;
        for (auto& w : inst.watchedMaterials)
        {
            if (w.material && w.material->GetStateRevision() != w.lastRevision)
            {
                w.lastRevision     = w.material->GetStateRevision();
                anyMaterialChanged = true;
            }
        }

        BOOL anyLODGroupChanged = false;
        for (auto& w : inst.watchedLODGroups)
        {
            if (w.lodGroup && w.lodGroup->GetStateRevision() != w.lastRevision)
            {
                w.lastRevision     = w.lodGroup->GetStateRevision();
                anyLODGroupChanged = true;
            }
        }

        if (!anyMeshChanged && !anyMaterialChanged && !anyLODGroupChanged)
            return;

        for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
        {
            GPUSceneSlotEntry& se = inst.slots[s];
            if (!se.occupied || !se.object) continue;

            BOOL slotResourceMutated = false;
            if (anyMeshChanged && se.boundMesh && se.boundMesh->GetStateRevision() != se.boundMeshRevision)
            {
                se.boundMeshRevision = se.boundMesh->GetStateRevision();
                se.object->UpdateWorldAABB();
                se.dirtyMask |= (GPU_SCENE_DIRTY_MESH | GPU_SCENE_DIRTY_BOUNDS);
                slotResourceMutated = true;
            }
            if (anyMaterialChanged && se.boundMaterial && se.boundMaterial->GetStateRevision() != se.boundMaterialRevision)
            {
                se.boundMaterialRevision = se.boundMaterial->GetStateRevision();
                se.dirtyMask |= GPU_SCENE_DIRTY_MATERIAL;
                slotResourceMutated = true;
            }
            if (anyLODGroupChanged && se.boundLODGroup && se.boundLODGroup->GetStateRevision() != se.boundLODGroupRevision)
            {
                se.boundLODGroupRevision = se.boundLODGroup->GetStateRevision();
                se.dirtyMask |= GPU_SCENE_DIRTY_LOD_GROUP;
                slotResourceMutated = true;
            }

            if (slotResourceMutated && !se.queuedDirty)
            {
                se.queuedDirty = true;
                inst.dirtySlotIndices.push_back(s);
            }
        }
    }

    BOOL CoreGPUSceneManager::ReserveSceneCapacity(Scene* scene, UINT32 minCapacitySlots)
    {
        if (!scene || !m_bufferManager)
        {
            m_lastError = "ReserveSceneCapacity: null scene or bufferManager.";
            return false;
        }
        if (m_simulateAllocationFailure)
        {
            m_lastError = "ReserveSceneCapacity: simulated allocation failure.";
            return false;
        }
        if (minCapacitySlots > m_maxSlotCapacity)
        {
            m_lastError = "ReserveSceneCapacity: requested capacity exceeds m_maxSlotCapacity.";
            return false;
        }

        GPUSceneInstance& inst = GetOrCreateInstance(scene);
        UINT32 targetCap = ComputeTargetCapacity(inst.capacitySlots, (std::max)(minCapacitySlots, kDefaultGPUSceneMinCapacity));
        if (targetCap > m_maxSlotCapacity)
        {
            targetCap = m_maxSlotCapacity;
        }
        if (inst.gpuBuffer != nullptr && targetCap <= inst.capacitySlots)
        {
            return true;
        }

        if (inst.gpuBuffer == nullptr)
        {
            GPUBufferDesc desc{};
            desc.elementCount = targetCap;
            desc.stride       = sizeof(GPUSceneObjectData);
            desc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;
            desc.initialData  = nullptr;
            desc.debugName    = "Northstar.GPUScene.ObjectBuffer";

            GPUBuffer* newBuf = m_bufferManager->CreateGPUBuffer(desc);
            if (!newBuf)
            {
                m_lastError = "ReserveSceneCapacity: failed to allocate initial GPU Scene buffer.";
                return false;
            }
            inst.gpuBuffer   = newBuf;
            inst.initialized = true;
        }
        else
        {
            if (!m_bufferManager->ResizeGPUBuffer(inst.gpuBuffer, targetCap, nullptr))
            {
                m_lastError = "ReserveSceneCapacity: failed to resize GPU Scene buffer.";
                return false;
            }
            inst.totalBufferResizes++;
            if (inst.highWaterMark > 0)
            {
                m_bufferManager->UpdateGPUBuffer(
                    inst.gpuBuffer,
                    inst.cpuRecords.data(),
                    inst.highWaterMark * static_cast<UINT32>(sizeof(GPUSceneObjectData)),
                    0u);
            }
        }

        const UINT32 oldCap = inst.capacitySlots;
        inst.capacitySlots  = targetCap;
        inst.cpuRecords.resize(targetCap);
        inst.gpuCommittedRecords.resize(targetCap);
        inst.slots.resize(targetCap);
        for (UINT32 s = oldCap; s < targetCap; ++s)
        {
            BuildInvalidSlotRecord(s, inst.cpuRecords[s]);
            BuildInvalidSlotRecord(s, inst.gpuCommittedRecords[s]);
        }
        inst.dirtySlotIndices.reserve(targetCap);
        inst.lastUploadRanges.reserve(targetCap);

        scene->m_gpuSceneBuffer             = inst.gpuBuffer;
        inst.lastStats.capacitySlots        = inst.capacitySlots;
        inst.lastStats.freeSlotCount        = static_cast<UINT32>(inst.freeSlots.size());
        scene->m_gpuSceneStats              = inst.lastStats;
        return true;
    }

    BOOL CoreGPUSceneManager::UpdateScene(Scene* scene, UINT64 frameNumber)
    {
        if (!scene || !m_bufferManager)
        {
            m_lastError = "UpdateScene: null scene or bufferManager.";
            return false;
        }

        const auto tTotalStart = std::chrono::high_resolution_clock::now();
        GPUSceneInstance& inst = GetOrCreateInstance(scene);

        // Single Per-Frame GPU Scene Snapshot Contract:
        // When a Scene is rendered by multiple Cameras within the same frameNumber (frameNumber != 0),
        // the GPU Scene is updated ONCE at the start of RenderFrame(). All cameras in that frame
        // observe the exact same GPU Scene snapshot. Any mid-frame CPU mutations made after Camera 1
        // renders remain queued in scene->m_dirtyObjects / scene->m_pendingRemovedSlots and are
        // deterministically uploaded on frameNumber + 1 with zero silent data loss.
        if (frameNumber != 0ull &&
            inst.initialized &&
            inst.lastUpdateSucceeded &&
            inst.lastUpdatedFrameNumber == frameNumber &&
            inst.gpuBuffer != nullptr &&
            inst.gpuBuffer->IsValid())
        {
            scene->m_gpuSceneBuffer = inst.gpuBuffer;
            scene->m_gpuSceneStats  = inst.lastStats;
            return true;
        }

        // Check watched resources (O(U) where U is unique meshes/materials/LODGroups in the scene)
        if (inst.initialized && !scene->m_gpuSceneCleared)
        {
            CheckWatchedResourceMutations(inst);
        }

        // Fast O(1) exit when already initialized and 0% dirty!
        if (inst.initialized &&
            inst.gpuBuffer != nullptr &&
            inst.gpuBuffer->IsValid() &&
            !scene->m_gpuSceneCleared &&
            scene->m_pendingRemovedSlots.empty() &&
            scene->m_dirtyObjects.empty() &&
            inst.dirtySlotIndices.empty())
        {
            const auto tFastEnd = std::chrono::high_resolution_clock::now();
            inst.lastUploadRanges.clear();
            inst.lastShadowInvalidationBoxes.clear();
            inst.lastUpdateSucceeded            = true;
            inst.lastStats.updateSucceeded      = true;
            inst.lastStats.activeObjectCount         = inst.activeObjectCount;
            inst.lastStats.virtualActiveObjectCount  = inst.virtualActiveCount;
            inst.lastStats.materialKernelObjectCount = inst.materialKernelActiveCount;
            inst.lastStats.mutationEpoch             = inst.mutationEpoch;
            inst.lastStats.highWaterMark        = inst.highWaterMark;
            inst.lastStats.capacitySlots        = inst.capacitySlots;
            inst.lastStats.freeSlotCount        = static_cast<UINT32>(inst.freeSlots.size());
            inst.lastStats.invalidSlotCount     = (inst.capacitySlots >= inst.activeObjectCount) ? (inst.capacitySlots - inst.activeObjectCount) : 0u;
            inst.lastStats.dirtyObjectCount     = 0u;
            inst.lastStats.dirtySlotCount       = 0u;
            inst.lastStats.coalescedUploadCount = 0u;
            inst.lastStats.uploadedBytes        = 0u;
            inst.lastStats.bufferResizeCount    = 0u;
            inst.lastStats.cpuDirtyCollectMs    = std::chrono::duration<FLOAT64, std::milli>(tFastEnd - tTotalStart).count();
            inst.lastStats.cpuRecordBuildMs     = 0.0;
            inst.lastStats.cpuUploadSubmitMs    = 0.0;
            inst.lastStats.cpuTotalUpdateMs     = inst.lastStats.cpuDirtyCollectMs;
            inst.lastUpdatedFrameNumber         = frameNumber;
            scene->m_gpuSceneBuffer             = inst.gpuBuffer;
            scene->m_gpuSceneStats              = inst.lastStats;
            return true;
        }

        // If first initialization and scene already has objects that somehow weren't in m_dirtyObjects, queue them
        if (!inst.initialized)
        {
            for (RenderObject* obj : scene->m_renderObjects)
            {
                if (!obj) continue;
                SceneObjectLink* link = obj->FindSceneLink(scene);
                if (link && link->dirtyListIndex == 0xFFFFFFFFu && !link->gpuHandle.IsValid())
                {
                    link->pendingDirtyFlags |= GPU_SCENE_DIRTY_ALL;
                    link->dirtyListIndex = static_cast<UINT32>(scene->m_dirtyObjects.size());
                    scene->m_dirtyObjects.push_back(obj);
                }
            }
        }

        // Pre-compute required capacity so GPU buffer creation/growth happens transactionally
        // BEFORE mutating authoritative slot tables.
        UINT32 validRemovedCount = 0u;
        if (!scene->m_gpuSceneCleared)
        {
            for (UINT32 remSlot : scene->m_pendingRemovedSlots)
            {
                if (remSlot < inst.slots.size() && inst.slots[remSlot].occupied)
                {
                    validRemovedCount++;
                }
            }
        }

        UINT32 newObjectAllocCount = 0u;
        for (RenderObject* obj : scene->m_dirtyObjects)
        {
            if (!obj) continue;
            const SceneObjectLink* link = obj->FindSceneLink(scene);
            if (!link) continue;
            const BOOL hasExistingValidSlot =
                !scene->m_gpuSceneCleared &&
                link->gpuHandle.IsValid() &&
                link->gpuHandle.slotIndex < inst.slots.size() &&
                inst.slots[link->gpuHandle.slotIndex].occupied &&
                inst.slots[link->gpuHandle.slotIndex].object == obj;
            if (!hasExistingValidSlot)
            {
                newObjectAllocCount++;
            }
        }

        const UINT32 baseFreeSlots = scene->m_gpuSceneCleared
            ? 0u
            : (static_cast<UINT32>(inst.freeSlots.size()) + validRemovedCount);
        const UINT32 baseHighWater = scene->m_gpuSceneCleared ? 0u : inst.highWaterMark;
        const UINT32 extraSlotsNeeded = (newObjectAllocCount > baseFreeSlots)
            ? (newObjectAllocCount - baseFreeSlots)
            : 0u;
        const UINT32 requiredHighWaterMark = baseHighWater + extraSlotsNeeded;
        if (requiredHighWaterMark > m_maxSlotCapacity)
        {
            m_lastError                    = "UpdateScene: required slot capacity exceeds maximum allowed GPU Scene slot capacity.";
            inst.lastUpdateSucceeded       = false;
            inst.lastStats.updateSucceeded = false;
            scene->m_gpuSceneStats         = inst.lastStats;
            return false;
        }

        BOOL   bufferCreatedOrResized = false;
        UINT32 resizeCountThisUpdate  = 0u;

        if (inst.gpuBuffer == nullptr || requiredHighWaterMark > inst.capacitySlots)
        {
            if (m_simulateAllocationFailure)
            {
                m_lastError                    = "UpdateScene: simulated GPU buffer allocation/resize failure.";
                inst.lastUpdateSucceeded       = false;
                inst.lastStats.updateSucceeded = false;
                scene->m_gpuSceneStats         = inst.lastStats;
                return false;
            }

            UINT32 targetCap = ComputeTargetCapacity(inst.capacitySlots, requiredHighWaterMark);
            if (targetCap > m_maxSlotCapacity)
            {
                targetCap = m_maxSlotCapacity;
            }
            if (inst.gpuBuffer == nullptr)
            {
                GPUBufferDesc desc{};
                desc.elementCount = targetCap;
                desc.stride       = sizeof(GPUSceneObjectData);
                desc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;
                desc.initialData  = nullptr;
                desc.debugName    = "Northstar.GPUScene.ObjectBuffer";

                GPUBuffer* newBuf = m_bufferManager->CreateGPUBuffer(desc);
                if (!newBuf)
                {
                    m_lastError                    = "UpdateScene: failed to create GPU Scene buffer.";
                    inst.lastUpdateSucceeded       = false;
                    inst.lastStats.updateSucceeded = false;
                    scene->m_gpuSceneStats         = inst.lastStats;
                    return false;
                }
                inst.gpuBuffer         = newBuf;
                inst.initialized       = true;
                bufferCreatedOrResized = true;
            }
            else
            {
                if (!m_bufferManager->ResizeGPUBuffer(inst.gpuBuffer, targetCap, nullptr))
                {
                    m_lastError                    = "UpdateScene: failed to resize GPU Scene buffer.";
                    inst.lastUpdateSucceeded       = false;
                    inst.lastStats.updateSucceeded = false;
                    scene->m_gpuSceneStats         = inst.lastStats;
                    return false;
                }
                inst.totalBufferResizes++;
                resizeCountThisUpdate  = 1u;
                bufferCreatedOrResized = true;
            }

            const UINT32 oldCap = inst.capacitySlots;
            inst.capacitySlots  = targetCap;
            inst.cpuRecords.resize(targetCap);
            inst.gpuCommittedRecords.resize(targetCap);
            inst.slots.resize(targetCap);
            for (UINT32 s = oldCap; s < targetCap; ++s)
            {
                BuildInvalidSlotRecord(s, inst.cpuRecords[s]);
                BuildInvalidSlotRecord(s, inst.gpuCommittedRecords[s]);
            }
            inst.dirtySlotIndices.reserve(targetCap);
            inst.lastUploadRanges.reserve(targetCap);
        }

        // Step 1: Handle full Scene::Clear() if it occurred
        UINT32 clearedWatermarkToZeroOnGPU = 0u;
        if (scene->m_gpuSceneCleared)
        {
            clearedWatermarkToZeroOnGPU = inst.highWaterMark;
            for (UINT32 s = 0; s < inst.highWaterMark && s < inst.slots.size(); ++s)
            {
                GPUSceneSlotEntry& se = inst.slots[s];
                if (se.occupied)
                {
                    se.nextGeneration = (se.nextGeneration == 0xFFFFFFFFu) ? 1u : (se.nextGeneration + 1u);
                }
                se.occupied              = false;
                se.object                = nullptr;
                se.currentGeneration     = 0u;
                se.queuedDirty           = true;
                se.dirtyMask             = GPU_SCENE_DIRTY_LIFECYCLE;
                se.boundMesh             = nullptr;
                se.boundMeshRevision     = 0u;
                se.boundMaterial         = nullptr;
                se.boundMaterialRevision = 0u;
                se.boundLODGroup         = nullptr;
                se.boundLODGroupRevision = 0u;
                se.boundSkinningRevision = 0u;
                se.skinningPaletteOffset   = 0u;
                se.skinningPaletteCapacity = 0u;
                se.skinningBoneCount       = 0u;
                se.skinningPoseDirty       = false;
                se.trackedVirtual          = false;
                se.trackedMaterialKernel   = false;
                BuildInvalidSlotRecord(s, inst.cpuRecords[s]);
                inst.dirtySlotIndices.push_back(s);
            }
            inst.freeSlots.clear();
            // Push cleared slots in reverse order so slot 0 is popped first on reuse
            for (UINT32 s = clearedWatermarkToZeroOnGPU; s > 0u; --s)
            {
                inst.freeSlots.push_back(s - 1u);
            }
            inst.activeObjectCount         = 0u;
            inst.virtualActiveCount        = 0u;
            inst.materialKernelActiveCount = 0u;
            inst.watchedMeshes.clear();
            inst.watchedMaterials.clear();
            inst.watchedLODGroups.clear();
            inst.cpuSkinningPalette.clear();
            inst.skinningFreeRanges.clear();
            for (UINT32 r = 0u; r < GPUSceneInstance::kSkinningRingCount; ++r)
            {
                inst.skinningRingDirtyMin[r] = 0xFFFFFFFFu;
                inst.skinningRingDirtyMax[r] = 0u;
            }
            scene->m_gpuSkinningPaletteCount = 0u;
            scene->m_gpuSceneCleared = false;
        }

        // Step 2: Process pending slot removals
        for (UINT32 remSlot : scene->m_pendingRemovedSlots)
        {
            if (remSlot < inst.slots.size() && inst.slots[remSlot].occupied)
            {
                GPUSceneSlotEntry& se    = inst.slots[remSlot];
                if (se.skinningPaletteCapacity > 0u)
                {
                    FreeSkinningPaletteRange(inst, se.skinningPaletteOffset, se.skinningPaletteCapacity);
                    se.skinningPaletteOffset   = 0u;
                    se.skinningPaletteCapacity = 0u;
                }
                se.occupied              = false;
                se.object                = nullptr;
                se.currentGeneration     = 0u;
                se.nextGeneration        = (se.nextGeneration == 0xFFFFFFFFu) ? 1u : (se.nextGeneration + 1u);
                se.boundMesh             = nullptr;
                se.boundMeshRevision     = 0u;
                se.boundMaterial         = nullptr;
                se.boundMaterialRevision = 0u;
                se.boundLODGroup         = nullptr;
                se.boundLODGroupRevision = 0u;
                se.boundSkinningRevision = 0u;
                se.skinningBoneCount     = 0u;
                se.skinningPoseDirty     = false;
                if (se.trackedVirtual)
                {
                    if (inst.virtualActiveCount > 0u) inst.virtualActiveCount--;
                    se.trackedVirtual = false;
                }
                if (se.trackedMaterialKernel)
                {
                    if (inst.materialKernelActiveCount > 0u) inst.materialKernelActiveCount--;
                    se.trackedMaterialKernel = false;
                }
                se.dirtyMask            |= GPU_SCENE_DIRTY_LIFECYCLE;

                BuildInvalidSlotRecord(remSlot, inst.cpuRecords[remSlot]);
                inst.freeSlots.push_back(remSlot);
                if (inst.activeObjectCount > 0u)
                {
                    inst.activeObjectCount--;
                }
                if (!se.queuedDirty)
                {
                    se.queuedDirty = true;
                    inst.dirtySlotIndices.push_back(remSlot);
                }
            }
        }
        scene->m_pendingRemovedSlots.clear();

        // Step 3: Process dirty / newly added objects (O(K))
        UINT32 dirtyObjectCount = 0u;
        for (RenderObject* obj : scene->m_dirtyObjects)
        {
            if (!obj) continue;
            SceneObjectLink* link = obj->FindSceneLink(scene);
            if (!link) continue;

            const UINT32 objDirtyFlags = link->pendingDirtyFlags;
            link->pendingDirtyFlags    = 0u;
            link->dirtyListIndex       = 0xFFFFFFFFu;

            UINT32 slotIndex = link->gpuHandle.slotIndex;
            const BOOL hasValidExistingSlot =
                link->gpuHandle.IsValid() &&
                slotIndex < inst.slots.size() &&
                inst.slots[slotIndex].occupied &&
                inst.slots[slotIndex].object == obj;

            if (!hasValidExistingSlot)
            {
                if (!inst.freeSlots.empty())
                {
                    slotIndex = inst.freeSlots.back();
                    inst.freeSlots.pop_back();
                }
                else
                {
                    slotIndex = inst.highWaterMark++;
                }

                GPUSceneSlotEntry& se = inst.slots[slotIndex];
                if (se.nextGeneration == 0u)
                    se.nextGeneration = 1u;
                se.occupied          = true;
                se.object            = obj;
                se.currentGeneration = se.nextGeneration;
                se.dirtyMask        |= (objDirtyFlags | GPU_SCENE_DIRTY_ALL);

                link->gpuHandle.slotIndex  = slotIndex;
                link->gpuHandle.generation = se.currentGeneration;
                inst.activeObjectCount++;

                if (!se.queuedDirty)
                {
                    se.queuedDirty = true;
                    inst.dirtySlotIndices.push_back(slotIndex);
                }
            }
            else
            {
                GPUSceneSlotEntry& se = inst.slots[slotIndex];
                se.dirtyMask |= objDirtyFlags;
                if (!se.queuedDirty)
                {
                    se.queuedDirty = true;
                    inst.dirtySlotIndices.push_back(slotIndex);
                }
            }
            dirtyObjectCount++;
        }
        scene->m_dirtyObjects.clear();

        const auto tDirtyCollectEnd = std::chrono::high_resolution_clock::now();

        // Step 4: Rebuild CPU records & skinning palette entries only for dirty slots (O(K))
        const UINT32 dirtySlotCount = static_cast<UINT32>(inst.dirtySlotIndices.size());
        BOOL anySkinningPaletteModified = false;

        for (UINT32 slotIdx : inst.dirtySlotIndices)
        {
            GPUSceneSlotEntry& se = inst.slots[slotIdx];

            if (se.occupied && se.object != nullptr)
            {
                SceneObjectLink* link = se.object->FindSceneLink(scene);
                const UINT32 boneCount  = se.object->GetSkinningMatrixCount();
                const UINT32 curSkinRev = se.object->GetSkinningRevision();
                const BOOL   poseMutated =
                    (curSkinRev != se.boundSkinningRevision) ||
                    ((se.dirtyMask & GPU_SCENE_DIRTY_SKINNING) != 0u);
                se.skinningPoseDirty     = poseMutated;
                se.boundSkinningRevision = curSkinRev;

                if (boneCount > 0u)
                {
                    if (se.skinningPaletteCapacity < boneCount)
                    {
                        if (se.skinningPaletteCapacity > 0u)
                        {
                            FreeSkinningPaletteRange(inst, se.skinningPaletteOffset, se.skinningPaletteCapacity);
                        }
                        se.skinningPaletteOffset   = AllocateSkinningPaletteRange(inst, boneCount);
                        se.skinningPaletteCapacity = boneCount;
                    }
                    se.skinningBoneCount = boneCount;
                    const Mat4* srcBones = se.object->GetSkinningMatrixData();
                    if (srcBones != nullptr)
                    {
                        std::memcpy(
                            &inst.cpuSkinningPalette[se.skinningPaletteOffset],
                            srcBones,
                            static_cast<SIZE_T>(boneCount) * sizeof(Mat4));
                    }
                    if (link != nullptr)
                    {
                        link->skinningPaletteOffset = se.skinningPaletteOffset;
                        link->skinningBoneCount     = boneCount;
                    }
                    MarkSkinningPaletteDirtyAllRings(inst, se.skinningPaletteOffset, boneCount);
                    anySkinningPaletteModified = true;
                }
                else
                {
                    if (se.skinningPaletteCapacity > 0u)
                    {
                        FreeSkinningPaletteRange(inst, se.skinningPaletteOffset, se.skinningPaletteCapacity);
                        se.skinningPaletteOffset   = 0u;
                        se.skinningPaletteCapacity = 0u;
                    }
                    if (se.skinningBoneCount != 0u)
                    {
                        se.skinningBoneCount       = 0u;
                        anySkinningPaletteModified = true;
                    }
                    if (link != nullptr)
                    {
                        link->skinningPaletteOffset = 0u;
                        link->skinningBoneCount     = 0u;
                    }
                }

                BuildObjectRecord(se.object, link, slotIdx, se.currentGeneration, inst.cpuRecords[slotIdx]);
                const BOOL nowVirtual = (inst.cpuRecords[slotIdx].objectFlags & GPU_SCENE_OBJ_VIRTUAL_GEOMETRY) != 0u;
                if (nowVirtual != se.trackedVirtual)
                {
                    if (nowVirtual) inst.virtualActiveCount++;
                    else if (inst.virtualActiveCount > 0u) inst.virtualActiveCount--;
                    se.trackedVirtual = nowVirtual;
                }
                const Material* slotMat = se.object->GetMaterial();
                const BOOL nowMatKernel = (slotMat != nullptr && slotMat->HasComputeKernels());
                if (nowMatKernel != se.trackedMaterialKernel)
                {
                    if (nowMatKernel) inst.materialKernelActiveCount++;
                    else if (inst.materialKernelActiveCount > 0u) inst.materialKernelActiveCount--;
                    se.trackedMaterialKernel = nowMatKernel;
                }
                if ((se.dirtyMask & (GPU_SCENE_DIRTY_MESH | GPU_SCENE_DIRTY_MATERIAL | GPU_SCENE_DIRTY_LOD_GROUP | GPU_SCENE_DIRTY_LIFECYCLE)) != 0u)
                {
                    WatchSlotResources(inst, se);
                }
            }
            else
            {
                se.skinningPoseDirty = false;
                if (se.trackedVirtual)
                {
                    if (inst.virtualActiveCount > 0u) inst.virtualActiveCount--;
                    se.trackedVirtual = false;
                }
                if (se.trackedMaterialKernel)
                {
                    if (inst.materialKernelActiveCount > 0u) inst.materialKernelActiveCount--;
                    se.trackedMaterialKernel = false;
                }
                BuildInvalidSlotRecord(slotIdx, inst.cpuRecords[slotIdx]);
            }
            se.dirtyMask = 0u;
        }

        // Upload persistent triple-buffered GPU Skinning Palette Buffer when modified (dirty span only)
        if (anySkinningPaletteModified && !inst.cpuSkinningPalette.empty())
        {
            inst.skinningRingIndex = (inst.skinningRingIndex + 1u) % GPUSceneInstance::kSkinningRingCount;
            const UINT32 ringIdx   = inst.skinningRingIndex;
            const UINT32 palCount  = static_cast<UINT32>(inst.cpuSkinningPalette.size());
            const UINT32 reqCap    = ComputeTargetCapacity(inst.skinningRingCapacities[ringIdx], (std::max)(palCount, 64u));

            if (inst.skinningRingBuffers[ringIdx] == nullptr)
            {
                GPUBufferDesc palDesc{};
                palDesc.elementCount = reqCap;
                palDesc.stride       = sizeof(Mat4);
                palDesc.usage        = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::CPU_READBACK;
                palDesc.initialData  = nullptr;
                palDesc.debugName    = "Northstar.GPUScene.SkinningPaletteBuffer";
                inst.skinningRingBuffers[ringIdx]    = m_bufferManager->CreateGPUBuffer(palDesc);
                inst.skinningRingCapacities[ringIdx] = inst.skinningRingBuffers[ringIdx] ? reqCap : 0u;
                inst.skinningRingDirtyMin[ringIdx]   = 0u;
                inst.skinningRingDirtyMax[ringIdx]   = palCount;
            }
            else if (reqCap > inst.skinningRingCapacities[ringIdx])
            {
                if (m_bufferManager->ResizeGPUBuffer(inst.skinningRingBuffers[ringIdx], reqCap, nullptr))
                {
                    inst.skinningRingCapacities[ringIdx] = reqCap;
                    inst.skinningRingDirtyMin[ringIdx]   = 0u;
                    inst.skinningRingDirtyMax[ringIdx]   = palCount;
                }
            }

            if (inst.skinningRingBuffers[ringIdx] != nullptr)
            {
                const UINT32 dirtyMin = (inst.skinningRingDirtyMin[ringIdx] < palCount)
                    ? inst.skinningRingDirtyMin[ringIdx]
                    : 0u;
                const UINT32 dirtyMax = (std::min)(
                    (inst.skinningRingDirtyMax[ringIdx] > dirtyMin) ? inst.skinningRingDirtyMax[ringIdx] : palCount,
                    palCount);
                if (dirtyMax > dirtyMin)
                {
                    m_bufferManager->UpdateGPUBuffer(
                        inst.skinningRingBuffers[ringIdx],
                        &inst.cpuSkinningPalette[dirtyMin],
                        (dirtyMax - dirtyMin) * static_cast<UINT32>(sizeof(Mat4)),
                        dirtyMin * static_cast<UINT32>(sizeof(Mat4)));
                }
                inst.skinningRingDirtyMin[ringIdx] = 0xFFFFFFFFu;
                inst.skinningRingDirtyMax[ringIdx] = 0u;
                inst.activeSkinningBuffer          = inst.skinningRingBuffers[ringIdx];
            }
        }
        scene->m_gpuSkinningPaletteBuffer = inst.activeSkinningBuffer;
        scene->m_gpuSkinningPaletteCount  = static_cast<UINT32>(inst.cpuSkinningPalette.size());

        const auto tRecordBuildEnd = std::chrono::high_resolution_clock::now();

        // Step 5: Coalesce contiguous dirty slots & upload to GPU buffer
        inst.lastUploadRanges.clear();
        UINT32 totalUploadedBytes = 0u;

        if (bufferCreatedOrResized)
        {
            const UINT32 slotsToUpload = (std::max)(inst.highWaterMark, clearedWatermarkToZeroOnGPU);
            if (slotsToUpload > 0u)
            {
                GPUSceneUploadRange fullActiveRange{};
                fullActiveRange.firstSlot  = 0u;
                fullActiveRange.slotCount  = slotsToUpload;
                fullActiveRange.byteOffset = 0u;
                fullActiveRange.byteSize   = slotsToUpload * static_cast<UINT32>(sizeof(GPUSceneObjectData));
                inst.lastUploadRanges.push_back(fullActiveRange);
            }
        }
        else if (!inst.dirtySlotIndices.empty())
        {
            if (inst.dirtySlotIndices.size() > 1u)
            {
                std::sort(inst.dirtySlotIndices.begin(), inst.dirtySlotIndices.end());
            }

            UINT32 runStart = inst.dirtySlotIndices[0];
            UINT32 runPrev  = runStart;

            for (SIZE_T i = 1; i < inst.dirtySlotIndices.size(); ++i)
            {
                const UINT32 cur = inst.dirtySlotIndices[i];
                if (cur == runPrev)
                {
                    continue;
                }
                if (cur == runPrev + 1u)
                {
                    runPrev = cur;
                }
                else
                {
                    const UINT32 count = (runPrev - runStart) + 1u;
                    GPUSceneUploadRange range{};
                    range.firstSlot  = runStart;
                    range.slotCount  = count;
                    range.byteOffset = runStart * static_cast<UINT32>(sizeof(GPUSceneObjectData));
                    range.byteSize   = count * static_cast<UINT32>(sizeof(GPUSceneObjectData));
                    inst.lastUploadRanges.push_back(range);

                    runStart = cur;
                    runPrev  = cur;
                }
            }

            const UINT32 finalCount = (runPrev - runStart) + 1u;
            GPUSceneUploadRange finalRange{};
            finalRange.firstSlot  = runStart;
            finalRange.slotCount  = finalCount;
            finalRange.byteOffset = runStart * static_cast<UINT32>(sizeof(GPUSceneObjectData));
            finalRange.byteSize   = finalCount * static_cast<UINT32>(sizeof(GPUSceneObjectData));
            inst.lastUploadRanges.push_back(finalRange);
        }

        // Upload coalesced ranges transactionally; if any upload range fails (including partial failure on
        // the 2nd or 3rd upload range), roll back any prior ranges [0 .. rIdx-1] to their last committed GPU
        // snapshot (inst.gpuCommittedRecords), retain ALL dirty slots across ALL upload ranges [0 .. N-1] in
        // inst.dirtySlotIndices with queuedDirty == true, and do NOT advance lastUpdatedFrameNumber so GPU
        // consumers never observe a torn/partially-updated scene snapshot and the next frame recovers 100%.
        auto rollbackAndRetainAllUploadRangesAsDirty = [&](SIZE_T failedRangeIndex) {
            for (SIZE_T prevIdx = 0; prevIdx < failedRangeIndex && prevIdx < inst.lastUploadRanges.size(); ++prevIdx)
            {
                const GPUSceneUploadRange& prevRange = inst.lastUploadRanges[prevIdx];
                if (prevRange.byteSize > 0u &&
                    (prevRange.firstSlot + prevRange.slotCount) <= inst.gpuCommittedRecords.size())
                {
                    m_bufferManager->UpdateGPUBuffer(
                        inst.gpuBuffer,
                        &inst.gpuCommittedRecords[prevRange.firstSlot],
                        prevRange.byteSize,
                        prevRange.byteOffset);
                }
            }

            inst.dirtySlotIndices.clear();
            for (auto& se : inst.slots)
            {
                se.queuedDirty = false;
            }
            for (SIZE_T rIdx = 0; rIdx < inst.lastUploadRanges.size(); ++rIdx)
            {
                const GPUSceneUploadRange& r = inst.lastUploadRanges[rIdx];
                for (UINT32 s = r.firstSlot; s < r.firstSlot + r.slotCount && s < inst.slots.size(); ++s)
                {
                    if (!inst.slots[s].queuedDirty)
                    {
                        inst.slots[s].queuedDirty = true;
                        inst.dirtySlotIndices.push_back(s);
                    }
                }
            }
        };

        if (m_simulateUploadFailure && m_simulateUploadFailureRangeIndex < 0 && !inst.lastUploadRanges.empty())
        {
            rollbackAndRetainAllUploadRangesAsDirty(0);
            m_lastError                         = "UpdateScene: simulated GPU buffer upload failure.";
            inst.lastUpdateSucceeded            = false;
            inst.lastStats.updateSucceeded      = false;
            inst.lastStats.uploadedBytes        = 0u;
            inst.lastStats.coalescedUploadCount = 0u;
            inst.lastStats.bufferResizeCount    = 0u;
            scene->m_gpuSceneStats              = inst.lastStats;
            return false;
        }

        for (SIZE_T rIdx = 0; rIdx < inst.lastUploadRanges.size(); ++rIdx)
        {
            const GPUSceneUploadRange& range = inst.lastUploadRanges[rIdx];
            if (range.byteSize == 0u) continue;

            const BOOL failInjectedAtThisRange =
                m_simulateUploadFailure &&
                m_simulateUploadFailureRangeIndex >= 0 &&
                static_cast<SIZE_T>(m_simulateUploadFailureRangeIndex) == rIdx;

            if (failInjectedAtThisRange ||
                !m_bufferManager->UpdateGPUBuffer(
                    inst.gpuBuffer,
                    &inst.cpuRecords[range.firstSlot],
                    range.byteSize,
                    range.byteOffset))
            {
                rollbackAndRetainAllUploadRangesAsDirty(rIdx);
                m_lastError                         = failInjectedAtThisRange
                    ? "UpdateScene: simulated partial GPU buffer upload failure on range."
                    : "UpdateScene: failed to upload dirty range to GPU Scene buffer.";
                inst.lastUpdateSucceeded            = false;
                inst.lastStats.updateSucceeded      = false;
                inst.lastStats.uploadedBytes        = 0u;
                inst.lastStats.coalescedUploadCount = 0u;
                inst.lastStats.bufferResizeCount    = resizeCountThisUpdate;
                scene->m_gpuSceneStats              = inst.lastStats;
                return false;
            }
            totalUploadedBytes += range.byteSize;
        }

        // All upload ranges succeeded: record dynamic shadow caster invalidation boxes (old + new bounds)
        // and commit records into gpuCommittedRecords snapshot
        if (inst.gpuCommittedRecords.size() < inst.cpuRecords.size())
        {
            inst.gpuCommittedRecords.resize(inst.cpuRecords.size());
        }

        inst.lastShadowInvalidationBoxes.clear();
        auto isShadowCasterRecord = [](const GPUSceneObjectData& rec, UINT32 expectedSlot) noexcept -> BOOL
        {
            constexpr UINT32 kReq = GPU_SCENE_OBJ_VALID | GPU_SCENE_OBJ_VISIBLE | GPU_SCENE_OBJ_CAST_SHADOW | GPU_SCENE_OBJ_HAS_VALID_BOUNDS;
            return rec.slotIndex == expectedSlot &&
                   rec.slotIndex != kInvalidGPUSceneSlot &&
                   rec.generation != 0u &&
                   ((rec.objectFlags & kReq) == kReq) &&
                   (rec.indexCount > 0u || rec.vertexCount > 0u) &&
                   (rec.boundingSphere.w > 0.0f) &&
                   (rec.aabbMax.x >= rec.aabbMin.x) &&
                   (rec.aabbMax.y >= rec.aabbMin.y) &&
                   (rec.aabbMax.z >= rec.aabbMin.z);
        };

        auto emitInvBox = [&](const Vec3& bMin, const Vec3& bMax, UINT32 slot, UINT32 flags)
        {
            NVSDynamicCasterInvalidationBox box{};
            box.worldMin          = bMin;
            box.gpuSceneSlotIndex = slot;
            box.worldMax          = bMax;
            box.invalidationFlags = flags;
            inst.lastShadowInvalidationBoxes.push_back(box);
            if (inst.pendingShadowInvalidationBoxes.size() < static_cast<SIZE_T>(kMaxGPUSceneSlotCapacity) * 2u)
            {
                inst.pendingShadowInvalidationBoxes.push_back(box);
            }
        };

        for (UINT32 s : inst.dirtySlotIndices)
        {
            if (s >= inst.cpuRecords.size() || s >= inst.gpuCommittedRecords.size())
            {
                continue;
            }
            const GPUSceneObjectData& oldRec    = inst.gpuCommittedRecords[s];
            const GPUSceneObjectData& newRec    = inst.cpuRecords[s];
            const BOOL                wasCaster = isShadowCasterRecord(oldRec, s);
            const BOOL                isCaster  = isShadowCasterRecord(newRec, s);

            if (wasCaster && !isCaster)
            {
                emitInvBox(oldRec.aabbMin, oldRec.aabbMax, s, NVS_DYN_INV_FLAG_OLD_BOUNDS | NVS_DYN_INV_FLAG_REMOVED_CASTER);
            }
            else if (!wasCaster && isCaster)
            {
                emitInvBox(newRec.aabbMin, newRec.aabbMax, s, NVS_DYN_INV_FLAG_NEW_BOUNDS | NVS_DYN_INV_FLAG_ADDED_CASTER);
            }
            else if (wasCaster && isCaster)
            {
                const BOOL boundsChanged =
                    (oldRec.aabbMin.x != newRec.aabbMin.x) || (oldRec.aabbMin.y != newRec.aabbMin.y) || (oldRec.aabbMin.z != newRec.aabbMin.z) ||
                    (oldRec.aabbMax.x != newRec.aabbMax.x) || (oldRec.aabbMax.y != newRec.aabbMax.y) || (oldRec.aabbMax.z != newRec.aabbMax.z);
                const BOOL propChanged =
                    (oldRec.generation != newRec.generation) ||
                    (oldRec.meshId != newRec.meshId) ||
                    (oldRec.materialId != newRec.materialId) ||
                    (oldRec.batchGroupId != newRec.batchGroupId) ||
                    (oldRec.objectFlags != newRec.objectFlags) ||
                    (oldRec.lodAndMaterialFlags != newRec.lodAndMaterialFlags);
                const BOOL skinningChanged =
                    (s < inst.slots.size()) && inst.slots[s].skinningPoseDirty;

                if (boundsChanged)
                {
                    emitInvBox(oldRec.aabbMin, oldRec.aabbMax, s, NVS_DYN_INV_FLAG_OLD_BOUNDS | NVS_DYN_INV_FLAG_MOVED_OR_SCALED);
                    emitInvBox(newRec.aabbMin, newRec.aabbMax, s, NVS_DYN_INV_FLAG_NEW_BOUNDS | NVS_DYN_INV_FLAG_MOVED_OR_SCALED);
                }
                else if (skinningChanged)
                {
                    emitInvBox(newRec.aabbMin, newRec.aabbMax, s, NVS_DYN_INV_FLAG_OLD_BOUNDS | NVS_DYN_INV_FLAG_NEW_BOUNDS | NVS_DYN_INV_FLAG_MOVED_OR_SCALED);
                }
                else if (propChanged)
                {
                    emitInvBox(newRec.aabbMin, newRec.aabbMax, s, NVS_DYN_INV_FLAG_OLD_BOUNDS | NVS_DYN_INV_FLAG_NEW_BOUNDS | NVS_DYN_INV_FLAG_MATERIAL_OR_MESH);
                }
            }
        }

        for (const GPUSceneUploadRange& range : inst.lastUploadRanges)
        {
            for (UINT32 s = range.firstSlot;
                 s < (range.firstSlot + range.slotCount) && s < inst.cpuRecords.size();
                 ++s)
            {
                inst.gpuCommittedRecords[s] = inst.cpuRecords[s];
            }
        }

        for (UINT32 slotIdx : inst.dirtySlotIndices)
        {
            if (slotIdx < inst.slots.size())
            {
                inst.slots[slotIdx].queuedDirty       = false;
                inst.slots[slotIdx].skinningPoseDirty = false;
            }
        }
        inst.dirtySlotIndices.clear();

        const auto tUploadEnd = std::chrono::high_resolution_clock::now();

        if (dirtySlotCount > 0u || bufferCreatedOrResized)
        {
            ++inst.mutationEpoch;
        }
        inst.lastUpdateSucceeded                 = true;
        inst.lastStats.updateSucceeded           = true;
        inst.lastStats.activeObjectCount         = inst.activeObjectCount;
        inst.lastStats.virtualActiveObjectCount  = inst.virtualActiveCount;
        inst.lastStats.materialKernelObjectCount = inst.materialKernelActiveCount;
        inst.lastStats.mutationEpoch             = inst.mutationEpoch;
        inst.lastStats.highWaterMark        = inst.highWaterMark;
        inst.lastStats.capacitySlots        = inst.capacitySlots;
        inst.lastStats.freeSlotCount        = static_cast<UINT32>(inst.freeSlots.size());
        inst.lastStats.invalidSlotCount     = (inst.capacitySlots >= inst.activeObjectCount) ? (inst.capacitySlots - inst.activeObjectCount) : 0u;
        inst.lastStats.dirtyObjectCount     = dirtyObjectCount;
        inst.lastStats.dirtySlotCount       = dirtySlotCount;
        inst.lastStats.coalescedUploadCount = static_cast<UINT32>(inst.lastUploadRanges.size());
        inst.lastStats.uploadedBytes        = totalUploadedBytes;
        inst.lastStats.bufferResizeCount    = resizeCountThisUpdate;
        inst.lastStats.cpuDirtyCollectMs    = std::chrono::duration<FLOAT64, std::milli>(tDirtyCollectEnd - tTotalStart).count();
        inst.lastStats.cpuRecordBuildMs     = std::chrono::duration<FLOAT64, std::milli>(tRecordBuildEnd - tDirtyCollectEnd).count();
        inst.lastStats.cpuUploadSubmitMs    = std::chrono::duration<FLOAT64, std::milli>(tUploadEnd - tRecordBuildEnd).count();
        inst.lastStats.cpuTotalUpdateMs     = std::chrono::duration<FLOAT64, std::milli>(tUploadEnd - tTotalStart).count();
        inst.lastUpdatedFrameNumber         = frameNumber;

        scene->m_gpuSceneBuffer = inst.gpuBuffer;
        scene->m_gpuSceneStats  = inst.lastStats;
        return true;
    }

    BOOL CoreGPUSceneManager::ForceFullRebuildAndUpload(Scene* scene, GPUSceneUpdateStats* outStats)
    {
        if (!scene || !m_bufferManager)
            return false;

        // Ensure slots exist for all objects in scene first
        if (!FindInstance(scene) || !FindInstance(scene)->initialized || !scene->m_dirtyObjects.empty() || !scene->m_pendingRemovedSlots.empty())
        {
            if (!UpdateScene(scene, 0ull))
                return false;
        }

        GPUSceneInstance* inst = FindInstance(scene);
        if (!inst || !inst->gpuBuffer)
            return false;

        const auto tStart = std::chrono::high_resolution_clock::now();

        for (UINT32 s = 0; s < inst->highWaterMark; ++s)
        {
            const GPUSceneSlotEntry& se = inst->slots[s];
            if (se.occupied && se.object != nullptr)
            {
                const SceneObjectLink* link = se.object->FindSceneLink(scene);
                BuildObjectRecord(se.object, link, s, se.currentGeneration, inst->cpuRecords[s]);
            }
            else
            {
                BuildInvalidSlotRecord(s, inst->cpuRecords[s]);
            }
        }

        const auto tBuildEnd = std::chrono::high_resolution_clock::now();

        const UINT32 uploadBytes = inst->highWaterMark * static_cast<UINT32>(sizeof(GPUSceneObjectData));
        if (uploadBytes > 0u)
        {
            if (!m_bufferManager->UpdateGPUBuffer(inst->gpuBuffer, inst->cpuRecords.data(), uploadBytes, 0u))
            {
                return false;
            }
            if (inst->gpuCommittedRecords.size() < inst->cpuRecords.size())
            {
                inst->gpuCommittedRecords.resize(inst->cpuRecords.size());
            }
            for (UINT32 s = 0; s < inst->highWaterMark && s < inst->cpuRecords.size(); ++s)
            {
                inst->gpuCommittedRecords[s] = inst->cpuRecords[s];
            }
        }

        const auto tUploadEnd = std::chrono::high_resolution_clock::now();

        GPUSceneUpdateStats fullStats{};
        fullStats.activeObjectCount    = inst->activeObjectCount;
        fullStats.highWaterMark        = inst->highWaterMark;
        fullStats.capacitySlots        = inst->capacitySlots;
        fullStats.freeSlotCount        = static_cast<UINT32>(inst->freeSlots.size());
        fullStats.dirtyObjectCount     = inst->activeObjectCount;
        fullStats.dirtySlotCount       = inst->highWaterMark;
        fullStats.coalescedUploadCount = (uploadBytes > 0u) ? 1u : 0u;
        fullStats.uploadedBytes        = uploadBytes;
        fullStats.bufferResizeCount    = 0u;
        fullStats.cpuDirtyCollectMs    = 0.0;
        fullStats.cpuRecordBuildMs     = std::chrono::duration<FLOAT64, std::milli>(tBuildEnd - tStart).count();
        fullStats.cpuUploadSubmitMs    = std::chrono::duration<FLOAT64, std::milli>(tUploadEnd - tBuildEnd).count();
        fullStats.cpuTotalUpdateMs     = std::chrono::duration<FLOAT64, std::milli>(tUploadEnd - tStart).count();

        if (outStats)
        {
            *outStats = fullStats;
        }
        return true;
    }

    GPUBuffer* CoreGPUSceneManager::GetSceneGPUBuffer(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? inst->gpuBuffer : nullptr;
    }

    GPUBuffer* CoreGPUSceneManager::GetSceneSkinningPaletteBuffer(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? inst->activeSkinningBuffer : nullptr;
    }

    const std::vector<Mat4>* CoreGPUSceneManager::GetSceneSkinningPaletteCPUData(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? &inst->cpuSkinningPalette : nullptr;
    }

    GPUSceneHandle CoreGPUSceneManager::GetObjectHandle(const Scene* scene, const RenderObject* obj) const
    {
        if (!scene || !obj) return {};
        return obj->GetGPUSceneHandle(scene);
    }

    BOOL CoreGPUSceneManager::IsHandleValid(const Scene* scene, GPUSceneHandle handle) const
    {
        if (!scene || !handle.IsValid())
            return false;
        const GPUSceneInstance* inst = FindInstance(scene);
        if (!inst || handle.slotIndex >= inst->highWaterMark || handle.slotIndex >= inst->slots.size())
            return false;
        const GPUSceneSlotEntry& se = inst->slots[handle.slotIndex];
        return se.occupied && se.object != nullptr && se.currentGeneration == handle.generation;
    }

    const GPUSceneObjectData* CoreGPUSceneManager::GetCPURecord(const Scene* scene, UINT32 slotIndex) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        if (!inst || slotIndex >= inst->cpuRecords.size())
            return nullptr;
        return &inst->cpuRecords[slotIndex];
    }

    const GPUSceneUpdateStats* CoreGPUSceneManager::GetSceneStats(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? &inst->lastStats : nullptr;
    }

    const std::vector<GPUSceneUploadRange>* CoreGPUSceneManager::GetLastUploadRanges(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? &inst->lastUploadRanges : nullptr;
    }

    const std::vector<NVSDynamicCasterInvalidationBox>* CoreGPUSceneManager::GetLastShadowInvalidationBoxes(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? &inst->lastShadowInvalidationBoxes : nullptr;
    }

    const std::vector<NVSDynamicCasterInvalidationBox>* CoreGPUSceneManager::GetPendingShadowInvalidationBoxes(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? &inst->pendingShadowInvalidationBoxes : nullptr;
    }

    BOOL CoreGPUSceneManager::ConsumePendingShadowInvalidationBoxes(
        Scene*                                        scene,
        std::vector<NVSDynamicCasterInvalidationBox>& outBoxes)
    {
        outBoxes.clear();
        GPUSceneInstance* inst = FindInstance(scene);
        if (!inst || inst->pendingShadowInvalidationBoxes.empty())
        {
            return false;
        }
        outBoxes = std::move(inst->pendingShadowInvalidationBoxes);
        inst->pendingShadowInvalidationBoxes.clear();
        return true;
    }

    void CoreGPUSceneManager::ClearPendingShadowInvalidationBoxes(Scene* scene)
    {
        GPUSceneInstance* inst = FindInstance(scene);
        if (inst)
        {
            inst->pendingShadowInvalidationBoxes.clear();
        }
    }

    UINT32 CoreGPUSceneManager::GetTotalBufferResizes(const Scene* scene) const
    {
        const GPUSceneInstance* inst = FindInstance(scene);
        return inst ? inst->totalBufferResizes : 0u;
    }
}
