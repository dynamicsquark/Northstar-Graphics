#pragma once

#include "render_object_desc.h"
#include "../scene/gpu_scene_types.h"
#include <vector>

namespace Northstar
{
    class Scene;
    class RenderObject;

    // Intrusive per-Scene membership & O(1) dirty-tracking link stored on RenderObject.
    // The primary link is inline (zero heap allocation for the 99.9% single-scene case).
    struct SceneObjectLink
    {
        Scene*         scene                 = nullptr;
        GPUSceneHandle gpuHandle             = {};
        UINT32         pendingDirtyFlags     = 0u;
        UINT32         dirtyListIndex        = 0xFFFFFFFFu;
        UINT32         batchGroupId          = 0u;
        UINT32         batchSubmeshIndex     = 0u;
        UINT32         batchStartIndex       = 0u;
        UINT32         batchIndexCount       = 0u;
        UINT32         skinningPaletteOffset = 0u;
        UINT32         skinningBoneCount     = 0u;
        void (*onDirtyThunk)(Scene*, RenderObject*, SceneObjectLink&) = nullptr;
    };

    // Fills ENGINE_PROPERTIES (Transform & Skinning source) at draw:
    //   WORLD_MATRIX -> m_worldMatrix
    //   NORMAL_MATRIX / MVP_MATRIX derived with active Camera VP
    //   BONE_PALETTE_OFFSET / BONE_COUNT / SKINNING_REVISION / GLOBAL_BONE_MATRIX -> m_skinningMatrices
    class RenderObject
    {
    public:
        // ==================== GETTERS ====================
        Mesh*              GetMesh()                                    const { return m_mesh;        }
        Material*          GetMaterial()                                const { return m_material;    }
        LODGroup*          GetLODGroup()                                const { return m_lodGroup;    }
        const Mat4&        GetWorldMatrix()                             const { return m_worldMatrix; } // WORLD_MATRIX
        const AABB&        GetWorldAABB()                               const { return m_worldAABB;   }
        UINT32             GetLayerMask()                               const { return m_layerMask;   }
        RenderObjectFlags  GetFlags()                                   const { return m_flags;       }
        UINT32             GetStateRevision()                           const { return m_stateRevision; }

        const std::vector<Mat4>& GetSkinningMatrices()                  const { return m_skinningMatrices; }
        const Mat4*        GetSkinningMatrixData()                      const { return m_skinningMatrices.empty() ? nullptr : m_skinningMatrices.data(); }
        UINT32             GetSkinningMatrixCount()                     const { return static_cast<UINT32>(m_skinningMatrices.size()); }
        UINT32             GetSkinningRevision()                        const { return m_skinningRevision; }
        BOOL               HasSkinningMatrices()                        const { return !m_skinningMatrices.empty(); }
        BOOL               HasExplicitWorldAABB()                       const { return m_hasExplicitWorldAABB; }

        UINT32             GetSkinningPaletteOffset(const Scene* scene = nullptr) const
        {
            if (!scene)
            {
                if (m_primaryLink.scene != nullptr)
                    return m_primaryLink.skinningPaletteOffset;
                if (!m_extraLinks.empty())
                    return m_extraLinks.front().skinningPaletteOffset;
                return 0u;
            }
            if (const SceneObjectLink* link = FindSceneLink(scene))
                return link->skinningPaletteOffset;
            return 0u;
        }

        GPUSceneHandle     GetGPUSceneHandle(const Scene* scene = nullptr) const
        {
            if (!scene)
            {
                if (m_primaryLink.scene != nullptr)
                    return m_primaryLink.gpuHandle;
                if (!m_extraLinks.empty())
                    return m_extraLinks.front().gpuHandle;
                return {};
            }
            if (m_primaryLink.scene == scene)
                return m_primaryLink.gpuHandle;
            for (const auto& link : m_extraLinks)
            {
                if (link.scene == scene)
                    return link.gpuHandle;
            }
            return {};
        }

        // ==================== SETTERS ====================
        void SetMesh(Mesh* mesh)
        {
            m_mesh = mesh;
            m_hasExplicitWorldAABB = false;
            if (!m_mesh && !m_lodGroup)
            {
                m_worldAABB = AABB();
            }
            UpdateWorldAABB();
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_MESH | GPU_SCENE_DIRTY_BOUNDS);
        }

        void SetMaterial(Material* material)
        {
            m_material = material;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_MATERIAL);
        }

        void SetLODGroup(LODGroup* lodGroup)
        {
            m_lodGroup = lodGroup;
            if (m_lodGroup != nullptr)
                m_flags |= RenderObjectFlags::USE_LOD;
            else
                m_flags &= ~RenderObjectFlags::USE_LOD;
            if (!m_hasExplicitWorldAABB)
            {
                if (m_mesh || m_lodGroup)
                    m_worldAABB = ComputeAutomaticWorldAABB();
                else
                    m_worldAABB = AABB();
            }
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_LOD_GROUP | GPU_SCENE_DIRTY_FLAGS | GPU_SCENE_DIRTY_BOUNDS);
        }

        void SetWorldMatrix(const Mat4& matrix)
        {
            m_worldMatrix = matrix;
            if ((m_mesh || m_lodGroup) && !m_hasExplicitWorldAABB)
            {
                m_worldAABB = ComputeAutomaticWorldAABB();
            }
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_TRANSFORM | GPU_SCENE_DIRTY_BOUNDS);
        }

        void SetTransform(
            const Vec3& position,
            const Quat& rotation = Quat::Identity(),
            const Vec3& scale    = Vec3::One())
        {
            SetWorldMatrix(Transform(position, rotation, scale).ToMatrix());
        }

        void SetTransform(const Transform& transform)
        {
            SetWorldMatrix(transform.ToMatrix());
        }

        void SetWorldAABB(const AABB& aabb)
        {
            m_worldAABB = aabb;
            m_hasExplicitWorldAABB = true;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_BOUNDS);
        }

        void SetLayerMask(UINT32 mask)
        {
            m_layerMask = mask;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_FLAGS);
        }

        void SetFlags(RenderObjectFlags flags)
        {
            m_flags = flags;
            if (m_lodGroup != nullptr)
                m_flags |= RenderObjectFlags::USE_LOD;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_FLAGS);
        }

        void SetSkinningMatrices(const Mat4* matrices, UINT32 count)
        {
            if (!matrices || count == 0u)
            {
                ClearSkinningMatrices();
                return;
            }
            m_skinningMatrices.assign(matrices, matrices + count);
            ++m_skinningRevision;
            if ((m_mesh || m_lodGroup) && !m_hasExplicitWorldAABB)
            {
                m_worldAABB = ComputeAutomaticWorldAABB();
            }
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_SKINNING | GPU_SCENE_DIRTY_BOUNDS);
        }

        void ClearSkinningMatrices()
        {
            if (m_skinningMatrices.empty())
                return;
            m_skinningMatrices.clear();
            ++m_skinningRevision;
            if ((m_mesh || m_lodGroup) && !m_hasExplicitWorldAABB)
            {
                m_worldAABB = ComputeAutomaticWorldAABB();
            }
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_SKINNING | GPU_SCENE_DIRTY_BOUNDS);
        }

        // ==================== AABB UPDATE ====================
        AABB GetLocalAABB() const
        {
            if (m_mesh && m_mesh->GetBoundingBox().IsValid())
            {
                return m_mesh->GetBoundingBox();
            }
            if (m_lodGroup)
            {
                AABB merged{};
                BOOL hasValidLevelBox = false;
                for (const auto& lvl : m_lodGroup->GetLevels())
                {
                    if (lvl.mesh && lvl.mesh->GetBoundingBox().IsValid())
                    {
                        if (!hasValidLevelBox)
                        {
                            merged           = lvl.mesh->GetBoundingBox();
                            hasValidLevelBox = true;
                        }
                        else
                        {
                            merged = merged.Merge(lvl.mesh->GetBoundingBox());
                        }
                    }
                }
                if (hasValidLevelBox)
                {
                    return merged;
                }
                const FLOAT32 r = m_lodGroup->GetLocalRadius();
                if (r > 0.0f)
                {
                    const Vec3 c = m_lodGroup->GetLocalCenter();
                    return AABB(c - Vec3(r, r, r), c + Vec3(r, r, r));
                }
            }
            return m_mesh ? m_mesh->GetBoundingBox() : AABB();
        }

        AABB ComputeSkinnedLocalAABB() const
        {
            const AABB baseBox = GetLocalAABB();
            if (m_skinningMatrices.empty() || !baseBox.IsValid())
                return baseBox;

            AABB envelope = baseBox;
            for (const Mat4& boneMat : m_skinningMatrices)
            {
                const AABB boneBox = baseBox.Transformed(boneMat);
                if (boneBox.IsValid())
                    envelope = envelope.Merge(boneBox);
            }
            return envelope;
        }

        AABB ComputeAutomaticWorldAABB() const
        {
            const AABB localBox = ComputeSkinnedLocalAABB();
            if (!localBox.IsValid())
                return AABB();
            return localBox.Transformed(m_worldMatrix);
        }

        void UpdateWorldAABB()
        {
            m_hasExplicitWorldAABB = false;
            if (m_mesh || m_lodGroup)
            {
                m_worldAABB = ComputeAutomaticWorldAABB();
            }
            else
            {
                m_worldAABB = AABB();
            }
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_BOUNDS);
        }

        // ==================== FLAG HELPERS ====================
        BOOL HasFlag(RenderObjectFlags flag)                      const { return (m_flags & flag) == flag; }

        void AddFlag(RenderObjectFlags flag)
        {
            m_flags |= flag;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_FLAGS);
        }

        void RemoveFlag(RenderObjectFlags flag)
        {
            m_flags &= ~flag;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_FLAGS);
        }

        BOOL IsStatic()                                           const { return HasFlag(RenderObjectFlags::STATIC); }
        void SetStatic(BOOL isStatic)                                   { if (isStatic) AddFlag(RenderObjectFlags::STATIC); else RemoveFlag(RenderObjectFlags::STATIC); }

        BOOL IsVisible()                                          const { return HasFlag(RenderObjectFlags::VISIBLE); }
        void SetVisible(BOOL visible)                                   { if (visible) AddFlag(RenderObjectFlags::VISIBLE); else RemoveFlag(RenderObjectFlags::VISIBLE); }

        BOOL GetCastShadows()                                     const { return HasFlag(RenderObjectFlags::CAST_SHADOW); }
        void SetCastShadows(BOOL cast)                                  { if (cast) AddFlag(RenderObjectFlags::CAST_SHADOW); else RemoveFlag(RenderObjectFlags::CAST_SHADOW); }

        BOOL GetReceiveShadows()                                  const { return HasFlag(RenderObjectFlags::RECEIVE_SHADOW); }
        void SetReceiveShadows(BOOL receive)                            { if (receive) AddFlag(RenderObjectFlags::RECEIVE_SHADOW); else RemoveFlag(RenderObjectFlags::RECEIVE_SHADOW); }

        BOOL IsBakedIntoStaticBatch()                             const { return HasFlag(RenderObjectFlags::BAKED_STATIC); }
        void SetBakedIntoStaticBatch(BOOL baked)
        {
            if (baked) m_flags |= RenderObjectFlags::BAKED_STATIC;
            else       m_flags &= ~RenderObjectFlags::BAKED_STATIC;
            MarkGPUSceneDirty(GPU_SCENE_DIRTY_FLAGS | GPU_SCENE_DIRTY_BATCHING);
        }

        void MarkGPUSceneDirty(UINT32 dirtyFlags)
        {
            ++m_stateRevision;
            if (m_primaryLink.scene != nullptr)
            {
                const UINT32 prev = m_primaryLink.pendingDirtyFlags;
                m_primaryLink.pendingDirtyFlags |= dirtyFlags;
                if (prev == 0u && m_primaryLink.onDirtyThunk != nullptr)
                {
                    m_primaryLink.onDirtyThunk(m_primaryLink.scene, this, m_primaryLink);
                }
            }
            for (auto& link : m_extraLinks)
            {
                if (link.scene != nullptr)
                {
                    const UINT32 prev = link.pendingDirtyFlags;
                    link.pendingDirtyFlags |= dirtyFlags;
                    if (prev == 0u && link.onDirtyThunk != nullptr)
                    {
                        link.onDirtyThunk(link.scene, this, link);
                    }
                }
            }
        }

        SceneObjectLink* FindSceneLink(const Scene* scene)
        {
            if (!scene) return nullptr;
            if (m_primaryLink.scene == scene)
                return &m_primaryLink;
            for (auto& link : m_extraLinks)
            {
                if (link.scene == scene)
                    return &link;
            }
            return nullptr;
        }

        const SceneObjectLink* FindSceneLink(const Scene* scene) const
        {
            if (!scene) return nullptr;
            if (m_primaryLink.scene == scene)
                return &m_primaryLink;
            for (const auto& link : m_extraLinks)
            {
                if (link.scene == scene)
                    return &link;
            }
            return nullptr;
        }

        SceneObjectLink& AttachSceneLink(Scene* scene, void (*thunk)(Scene*, RenderObject*, SceneObjectLink&))
        {
            if (SceneObjectLink* existing = FindSceneLink(scene))
            {
                existing->onDirtyThunk = thunk;
                return *existing;
            }
            if (m_primaryLink.scene == nullptr)
            {
                m_primaryLink = SceneObjectLink{};
                m_primaryLink.scene        = scene;
                m_primaryLink.onDirtyThunk = thunk;
                return m_primaryLink;
            }
            SceneObjectLink link{};
            link.scene        = scene;
            link.onDirtyThunk = thunk;
            m_extraLinks.push_back(link);
            return m_extraLinks.back();
        }

        void DetachSceneLink(const Scene* scene)
        {
            if (!scene) return;
            if (m_primaryLink.scene == scene)
            {
                if (!m_extraLinks.empty())
                {
                    m_primaryLink = m_extraLinks.back();
                    m_extraLinks.pop_back();
                }
                else
                {
                    m_primaryLink = SceneObjectLink{};
                }
                return;
            }
            for (SIZE_T i = 0; i < m_extraLinks.size(); ++i)
            {
                if (m_extraLinks[i].scene == scene)
                {
                    m_extraLinks[i] = m_extraLinks.back();
                    m_extraLinks.pop_back();
                    return;
                }
            }
        }

        BOOL HasAnySceneLinks() const
        {
            return m_primaryLink.scene != nullptr || !m_extraLinks.empty();
        }

        Scene* GetFirstLinkedScene() const
        {
            if (m_primaryLink.scene != nullptr)
                return m_primaryLink.scene;
            if (!m_extraLinks.empty())
                return m_extraLinks.front().scene;
            return nullptr;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreSceneManager;
        friend class CoreGPUSceneManager;
        friend class Scene;

        RenderObject() = default;

        Mesh*              m_mesh                                       = nullptr;
        Material*          m_material                                   = nullptr;
        LODGroup*          m_lodGroup                                   = nullptr;
        Mat4               m_worldMatrix                                = Mat4::Identity();
        AABB               m_worldAABB                                  = {};
        UINT32             m_layerMask                                  = 0x00000001u;
        RenderObjectFlags  m_flags                                      = RenderObjectFlags::VISIBLE        |
                                                                          RenderObjectFlags::FRUSTUM_CULL   |
                                                                          RenderObjectFlags::CAST_SHADOW    |
                                                                          RenderObjectFlags::RECEIVE_SHADOW |
                                                                          RenderObjectFlags::DISTANCE_CULL;
        UINT32             m_stateRevision                              = 1u;
        UINT32             m_skinningRevision                           = 0u;
        BOOL               m_hasExplicitWorldAABB                       = false;
        std::vector<Mat4>  m_skinningMatrices                           = {};
        SceneObjectLink    m_primaryLink                                = {};
        std::vector<SceneObjectLink> m_extraLinks;
    };
}
