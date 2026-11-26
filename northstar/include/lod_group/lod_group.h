#pragma once

#include "lod_group_desc.h"
#include "../types/northstar_types.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace Northstar
{
    // ============ LODGroup ============
    class LODGroup
    {
    public:
        LODGroup() = default;

        LODGroup(const LODGroupDesc& desc)
            : m_levels(desc.levels), m_localCenter(desc.localCenter), m_localRadius(desc.localRadius)
        {
        }

        // ==================== GETTERS ====================
        const std::vector<LODLevel>& GetLevels()                               const { return m_levels;        }
        const Vec3&    GetLocalCenter()                                        const { return m_localCenter;   }
        FLOAT32        GetLocalRadius()                                        const { return m_localRadius;   }
        BOOL           IsUsingScreenSize()                                     const { return m_useScreenSize; }
        UINT32         GetGPUSceneLODGroupId()                                 const { return m_gpuSceneLODGroupId; }
        UINT32         GetStateRevision()                                      const { return m_stateRevision; }
        
        Mesh*          GetMeshForLOD(INT32 lodIndex) const
        {
            if (lodIndex >= 0 && lodIndex < static_cast<INT32>(m_levels.size()))
            {
                return m_levels[lodIndex].mesh;
            }
            return nullptr;
        }

        // ==================== SETTERS ====================
        void SetLevels(const std::vector<LODLevel>& levels)                    { m_levels = levels; ++m_stateRevision; }
        void AddLevel(Mesh* mesh, FLOAT32 screenRelativeTransitionHeight)      { m_levels.push_back({mesh, screenRelativeTransitionHeight}); ++m_stateRevision; }
        void SetBoundingSphere(const Vec3& localCenter, FLOAT32 localRadius)   { m_localCenter = localCenter; m_localRadius = localRadius; ++m_stateRevision; }
        void SetUseScreenSize(BOOL useScreenSize)                              { m_useScreenSize = useScreenSize; ++m_stateRevision; }
        void SetGPUSceneLODGroupId(UINT32 id)                                  { m_gpuSceneLODGroupId = id; ++m_stateRevision; }

        void OnMeshDestroyed(const Mesh* destroyedMesh)
        {
            if (!destroyedMesh) return;
            BOOL modified = false;
            for (auto& lvl : m_levels)
            {
                if (lvl.mesh == destroyedMesh)
                {
                    lvl.mesh = nullptr;
                    modified = true;
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
        }

        // ==================== METHODS ====================
        FLOAT32 CalculateScreenRelativeHeight(
            const Vec3& objectWorldPos, 
            const Vec3& objectWorldScale,
            const Vec3& cameraWorldPos, 
            FLOAT32 cameraFovY) const
        {
            Vec3 worldCenter = objectWorldPos + (m_localCenter * objectWorldScale);
            
            FLOAT32 maxScale = std::max({
                std::fabs(objectWorldScale.x),
                std::fabs(objectWorldScale.y),
                std::fabs(objectWorldScale.z)
            });
            FLOAT32 worldRadius = std::max(0.0f, m_localRadius) * maxScale;

            FLOAT32 distance = (worldCenter - cameraWorldPos).Length();

            if (distance <= worldRadius || distance < EPSILON)
            {
                return 1.0f; 
            }

            const FLOAT32 clampedFov = std::clamp(cameraFovY, 1e-4f, PI - 1e-4f);
            const FLOAT32 halfFovTan = std::tan(clampedFov * 0.5f);
            if (halfFovTan <= EPSILON)
            {
                return 1.0f;
            }
            const FLOAT32 screenRelativeHeight = worldRadius / (distance * halfFovTan);

            return std::max(0.0f, std::min(screenRelativeHeight, 1.0f));
        }

        INT32 GetCurrentLODIndex(
            const Vec3& objectWorldPos, 
            const Vec3& objectWorldScale,
            const Vec3& cameraWorldPos, 
            FLOAT32 cameraFovY_or_MaxDistance) const
        {
            if (m_levels.empty()) return -1;

            if (m_useScreenSize)
            {
                FLOAT32 screenHeight = CalculateScreenRelativeHeight(
                    objectWorldPos, objectWorldScale, cameraWorldPos, cameraFovY_or_MaxDistance);

                for (SIZE_T i = 0; i < m_levels.size(); ++i)
                {
                    if (screenHeight >= m_levels[i].screenRelativeTransitionHeight)
                    {
                        return static_cast<INT32>(i);
                    }
                }
            }
            else
            {
                Vec3 worldCenter = objectWorldPos + (m_localCenter * objectWorldScale);
                FLOAT32 distance = (worldCenter - cameraWorldPos).Length();

                for (SIZE_T i = 0; i < m_levels.size(); ++i)
                {
                    if (distance <= m_levels[i].screenRelativeTransitionHeight)
                    {
                        return static_cast<INT32>(i);
                    }
                }
            }

            return -1; 
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreGeometryManager;

        std::vector<LODLevel> m_levels;
        
        Vec3           m_localCenter                               = Vec3::Zero();
        FLOAT32        m_localRadius                               = 1.0f;
        
        BOOL           m_useScreenSize                             = true;
        UINT32         m_gpuSceneLODGroupId                        = 0;
        UINT32         m_stateRevision                             = 1;
    };
}
