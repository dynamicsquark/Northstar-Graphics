#include "core_geometry_manager.h"
#include "../nvg/core_nvg_manager.h"
#include "platform/northstar_backend.h"
#include <algorithm>

namespace Northstar
{
    CoreGeometryManager::~CoreGeometryManager()
    {
        Shutdown();
    }

    Mesh* CoreGeometryManager::CreateMesh(const MeshDesc& desc)
    {
        m_lastError.clear();
        Mesh* mesh = new Mesh();
        if (m_nextMeshId == 0u) m_nextMeshId = 1u;
        const UINT32 assignedId = m_nextMeshId++;
        if (m_nextMeshId == 0u) m_nextMeshId = 1u;
        mesh->SetGPUSceneMeshId(assignedId);

        if (desc.submeshCount > 0u)
        {
            if (desc.submeshes == nullptr)
            {
                m_lastError = "CreateMesh failed: submeshCount > 0 but submeshes pointer is nullptr.";
                delete mesh;
                return nullptr;
            }
            for (UINT32 si = 0u; si < desc.submeshCount; ++si)
            {
                const NVGSubmeshRange& sr = desc.submeshes[si];
                if (sr.materialSlot >= kNVGMaxMaterialSlots)
                {
                    m_lastError = "CreateMesh failed: submesh materialSlot exceeds maximum slot index (255).";
                    delete mesh;
                    return nullptr;
                }
                if (sr.indexCount == 0u || (sr.indexCount % 3u) != 0u)
                {
                    m_lastError = "CreateMesh failed: submesh indexCount must be non-zero and a multiple of 3.";
                    delete mesh;
                    return nullptr;
                }
                if (desc.indexCount > 0u &&
                    (static_cast<UINT64>(sr.startIndex) + static_cast<UINT64>(sr.indexCount) > static_cast<UINT64>(desc.indexCount)))
                {
                    m_lastError = "CreateMesh failed: submesh range [startIndex .. startIndex + indexCount] exceeds mesh indexCount.";
                    delete mesh;
                    return nullptr;
                }
            }
            mesh->SetSubmeshes(desc.submeshes, desc.submeshCount);
        }
        if (desc.pageStreamProvider)
        {
            mesh->SetPageStreamProvider(desc.pageStreamProvider);
        }

        if (desc.geometryMode == GeometryMode::VIRTUALIZED)
        {
            if (!m_nvgManager || !m_nvgManager->IsInitialized())
            {
                m_lastError = "CreateMesh(GeometryMode::VIRTUALIZED) failed: NVG subsystem is not initialized.";
                delete mesh;
                return nullptr;
            }

            NVGMeshInfo info{};
            AABB        localBounds{};
            std::string nvgErr;
            std::shared_ptr<INVGPageStreamProvider> resolvedProvider;
            if (!m_nvgManager->CreateAndRegisterVirtualMesh(desc, assignedId, info, localBounds, nvgErr, &resolvedProvider))
            {
                m_lastError = nvgErr.empty() ? "CreateMesh(GeometryMode::VIRTUALIZED) failed." : nvgErr;
                delete mesh;
                return nullptr;
            }

            if (resolvedProvider && !mesh->GetPageStreamProvider())
            {
                mesh->SetPageStreamProvider(std::move(resolvedProvider));
            }
            mesh->SetBoundingBox((desc.boundingBox.Size().LengthSq() > 0.0f) ? desc.boundingBox : localBounds);
            mesh->SetVirtualGeometryMetadata(GeometryMode::VIRTUALIZED, info);
            mesh->SetGPUHandles(0, 0);
            m_meshes.push_back(mesh);
            return mesh;
        }

        if ((m_backend && m_backend->IsHardwareBackend() && (desc.vertexCount == 0u || desc.vertices == nullptr)) ||
            (desc.vertexCount > 0u && desc.vertices == nullptr) ||
            (desc.indexCount > 0u && desc.indices == nullptr))
        {
            m_lastError = "CreateMesh failed: missing vertex data or nullptr geometry pointer.";
            delete mesh;
            return nullptr;
        }

        const UINT64 vbBytes64 = static_cast<UINT64>(desc.vertexCount) * static_cast<UINT64>(sizeof(Vertex));
        const UINT64 ibBytes64 = static_cast<UINT64>(desc.indexCount) * static_cast<UINT64>(sizeof(UINT32));
        if (vbBytes64 > 0x7FFFFFFFULL || ibBytes64 > 0x7FFFFFFFULL)
        {
            m_lastError = "CreateMesh failed: vertex or index buffer byte size exceeds 2 GiB limit.";
            delete mesh;
            return nullptr;
        }

        if (desc.vertices && desc.vertexCount > 0)
        {
            mesh->SetVertices(desc.vertices, desc.vertexCount);
        }
        if (desc.indices && desc.indexCount > 0)
        {
            mesh->SetIndices(desc.indices, desc.indexCount);
        }
        if (desc.boundingBox.Size().LengthSq() > 0.0f)
        {
            mesh->SetBoundingBox(desc.boundingBox);
        }
        else
        {
            mesh->CalculateBoundingBox();
        }

        if (m_backend)
        {
            NativeHandle vh = 0;
            NativeHandle ih = 0;
            if (desc.vertices && desc.vertexCount > 0)
            {
                vh = m_backend->CreateVertexBuffer(desc.vertices, static_cast<UINT32>(vbBytes64), sizeof(Vertex));
                if (vh == 0 && m_backend->IsHardwareBackend())
                {
                    m_lastError = "CreateMesh failed: hardware backend CreateVertexBuffer returned 0.";
                    delete mesh;
                    return nullptr;
                }
            }
            if (desc.indices && desc.indexCount > 0)
            {
                ih = m_backend->CreateIndexBuffer(desc.indices, static_cast<UINT32>(ibBytes64), true);
                if (ih == 0 && (vh != 0 || m_backend->IsHardwareBackend()))
                {
                    if (vh != 0)
                    {
                        m_backend->DestroyVertexBuffer(vh);
                    }
                    m_lastError = "CreateMesh failed: backend CreateIndexBuffer returned 0.";
                    delete mesh;
                    return nullptr;
                }
            }
            mesh->SetGPUHandles(vh, ih);
        }

        m_meshes.push_back(mesh);
        return mesh;
    }

    BOOL CoreGeometryManager::IsManagedMesh(const Mesh* mesh) const
    {
        if (!mesh) return false;
        return std::find(m_meshes.begin(), m_meshes.end(), mesh) != m_meshes.end();
    }

    void CoreGeometryManager::DestroyMesh(Mesh* mesh)
    {
        if (!mesh) return;
        auto it = std::find(m_meshes.begin(), m_meshes.end(), mesh);
        if (it != m_meshes.end())
        {
            for (LODGroup* lod : m_lodGroups)
            {
                if (lod)
                {
                    lod->OnMeshDestroyed(mesh);
                }
            }
            if (mesh->IsVirtualized() && m_nvgManager)
            {
                m_nvgManager->UnregisterVirtualMesh(mesh->GetGPUSceneMeshId());
            }
            if (m_backend)
            {
                if (mesh->GetVertexBufferHandle() != 0)
                {
                    m_backend->DestroyVertexBuffer(mesh->GetVertexBufferHandle());
                }
                if (mesh->GetIndexBufferHandle() != 0)
                {
                    m_backend->DestroyIndexBuffer(mesh->GetIndexBufferHandle());
                }
                mesh->SetGPUHandles(0, 0);
            }
            delete mesh;
            m_meshes.erase(it);
        }
    }

    LODGroup* CoreGeometryManager::CreateLODGroup(const LODGroupDesc& desc)
    {
        LODGroup* lodGroup = new LODGroup();
        if (m_nextLODGroupId == 0u) m_nextLODGroupId = 1u;
        lodGroup->SetGPUSceneLODGroupId(m_nextLODGroupId++);
        if (m_nextLODGroupId == 0u) m_nextLODGroupId = 1u;
        lodGroup->SetLevels(desc.levels);
        lodGroup->SetBoundingSphere(desc.localCenter, desc.localRadius);
        m_lodGroups.push_back(lodGroup);
        return lodGroup;
    }

    BOOL CoreGeometryManager::IsManagedLODGroup(const LODGroup* lodGroup) const
    {
        if (!lodGroup) return false;
        return std::find(m_lodGroups.begin(), m_lodGroups.end(), lodGroup) != m_lodGroups.end();
    }

    void CoreGeometryManager::DestroyLODGroup(LODGroup* lodGroup)
    {
        if (!lodGroup) return;
        auto it = std::find(m_lodGroups.begin(), m_lodGroups.end(), lodGroup);
        if (it != m_lodGroups.end())
        {
            delete lodGroup;
            m_lodGroups.erase(it);
        }
    }

    void CoreGeometryManager::RegisterWithBackend(NorthstarBackend* backend)
    {
        m_backend = backend;
        if (!m_backend) return;

        for (auto* mesh : m_meshes)
        {
            if (!mesh || mesh->IsVirtualized()) continue;
            NativeHandle vh = mesh->GetVertexBufferHandle();
            NativeHandle ih = mesh->GetIndexBufferHandle();

            if (vh == 0 && mesh->HasCPUData())
            {
                vh = m_backend->CreateVertexBuffer(mesh->GetVertices(), mesh->GetVertexCount() * sizeof(Vertex), sizeof(Vertex));
            }
            if (ih == 0 && mesh->HasIndices())
            {
                ih = m_backend->CreateIndexBuffer(mesh->GetIndices(), mesh->GetIndexCount() * sizeof(UINT32), true);
            }
            mesh->SetGPUHandles(vh, ih);
        }
    }

    void CoreGeometryManager::UnregisterFromBackend(BOOL destroyNativeHandles)
    {
        if (destroyNativeHandles && m_backend)
        {
            for (auto* mesh : m_meshes)
            {
                if (!mesh) continue;
                if (mesh->GetVertexBufferHandle() != 0)
                {
                    m_backend->DestroyVertexBuffer(mesh->GetVertexBufferHandle());
                }
                if (mesh->GetIndexBufferHandle() != 0)
                {
                    m_backend->DestroyIndexBuffer(mesh->GetIndexBufferHandle());
                }
                mesh->SetGPUHandles(0, 0);
            }
        }
        m_backend = nullptr;
    }

    void CoreGeometryManager::Shutdown()
    {
        for (auto lod : m_lodGroups) delete lod;
        m_lodGroups.clear();

        for (auto mesh : m_meshes)
        {
            if (m_backend)
            {
                if (mesh->GetVertexBufferHandle() != 0)
                {
                    m_backend->DestroyVertexBuffer(mesh->GetVertexBufferHandle());
                }
                if (mesh->GetIndexBufferHandle() != 0)
                {
                    m_backend->DestroyIndexBuffer(mesh->GetIndexBufferHandle());
                }
                mesh->SetGPUHandles(0, 0);
            }
            delete mesh;
        }
        m_meshes.clear();
        m_backend = nullptr;
    }
}
