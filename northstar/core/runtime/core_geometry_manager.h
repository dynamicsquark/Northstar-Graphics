#pragma once

#include "mesh/mesh.h"
#include "mesh/mesh_desc.h"
#include "lod_group/lod_group.h"
#include "lod_group/lod_group_desc.h"
#include "platform/northstar_backend.h"
#include "../nvg/core_nvg_manager.h"
#include <string>
#include <vector>

namespace Northstar
{
    class CoreGeometryManager
    {
    public:
        CoreGeometryManager() = default;
        ~CoreGeometryManager();

        void      SetNVGManager(CoreNVGManager* nvgManager) { m_nvgManager = nvgManager; }
        const std::string& GetLastError() const { return m_lastError; }

        Mesh*     CreateMesh(const MeshDesc& desc);
        void      DestroyMesh(Mesh* mesh);
        BOOL      IsManagedMesh(const Mesh* mesh) const;

        LODGroup* CreateLODGroup(const LODGroupDesc& desc);
        void      DestroyLODGroup(LODGroup* lodGroup);
        BOOL      IsManagedLODGroup(const LODGroup* lodGroup) const;

        void      RegisterWithBackend(NorthstarBackend* backend);
        void      UnregisterFromBackend(BOOL destroyNativeHandles = false);
        void      Shutdown();

    private:
        std::vector<Mesh*>     m_meshes;
        std::vector<LODGroup*> m_lodGroups;
        NorthstarBackend*      m_backend        = nullptr;
        CoreNVGManager*        m_nvgManager     = nullptr;
        UINT32                 m_nextMeshId     = 1;
        UINT32                 m_nextLODGroupId = 1;
        std::string            m_lastError;
    };
}
