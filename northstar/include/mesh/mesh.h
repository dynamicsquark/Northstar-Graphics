#pragma once

#include "vertex.h"
#include "mesh_desc.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include <vector>

namespace Northstar
{
    class Mesh
    {
    public:
        //==================== GETTERS ====================
        UINT32        GetVertexCount()       const { return static_cast<UINT32>(m_vertices.size()); }
        UINT32        GetIndexCount()        const { return static_cast<UINT32>(m_indices.size());  }

        const Vertex* GetVertices()          const { return m_vertices.empty() ? nullptr : m_vertices.data(); }
        const UINT32* GetIndices()           const { return m_indices.empty() ? nullptr : m_indices.data();   }

        const AABB&   GetBoundingBox()       const { return m_boundingBox; }
        BOOL          HasIndices()           const { return !m_indices.empty(); }
        BOOL          HasCPUData()           const { return !m_vertices.empty(); }

        //==================== VIRTUAL GEOMETRY (NVG) ====================
        GeometryMode       GetGeometryMode()        const { return m_geometryMode; }
        BOOL               IsVirtualized()          const { return m_geometryMode == GeometryMode::VIRTUALIZED && m_nvgInfo.valid; }
        UINT32             GetVirtualMeshId()       const { return m_nvgInfo.virtualMeshId; }
        const NVGMeshInfo& GetVirtualMeshInfo()     const { return m_nvgInfo; }

        //==================== GPU HANDLES ====================
        NativeHandle GetVertexBufferHandle() const { return m_vertexBuffer; }
        NativeHandle GetIndexBufferHandle()  const { return m_indexBuffer;  }
        UINT32       GetVertexBufferOffset() const { return m_vertexBufferOffset; }
        UINT32       GetIndexBufferOffset()  const { return m_indexBufferOffset;  }
        UINT32       GetGPUSceneMeshId()     const { return m_gpuSceneMeshId;     }
        UINT32       GetStateRevision()      const { return m_stateRevision;      }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreGeometryManager;
        friend class CoreNVGManager;

        Mesh() = default;

        void SetVirtualGeometryMetadata(GeometryMode mode, const NVGMeshInfo& info)
        {
            m_geometryMode = mode;
            m_nvgInfo      = info;
            ++m_stateRevision;
        }

        void SetVertices(const Vertex* vertices, UINT32 count)
        {
            m_vertices.clear();
            if (vertices && count > 0)
            {
                m_vertices.assign(vertices, vertices + count);
            }
            ++m_stateRevision;
        }

        void SetIndices(const UINT32* indices, UINT32 count)
        {
            m_indices.clear();
            if (indices && count > 0)
            {
                m_indices.assign(indices, indices + count);
            }
            ++m_stateRevision;
        }

        void SetBoundingBox(const AABB& aabb)
        {
            m_boundingBox = aabb;
            ++m_stateRevision;
        }

        void CalculateBoundingBox()
        {
            if (m_vertices.empty())
            {
                m_boundingBox = AABB();
                return;
            }

            Vec3 minPos = m_vertices[0].position;
            Vec3 maxPos = m_vertices[0].position;

            for (SIZE_T i = 1; i < m_vertices.size(); ++i)
            {
                const Vec3& pos = m_vertices[i].position;
                if (pos.x < minPos.x) minPos.x = pos.x;
                if (pos.y < minPos.y) minPos.y = pos.y;
                if (pos.z < minPos.z) minPos.z = pos.z;

                if (pos.x > maxPos.x) maxPos.x = pos.x;
                if (pos.y > maxPos.y) maxPos.y = pos.y;
                if (pos.z > maxPos.z) maxPos.z = pos.z;
            }

            m_boundingBox = AABB(minPos, maxPos);
            ++m_stateRevision;
        }

        void SetGPUHandles(NativeHandle vb, NativeHandle ib, UINT32 vbOffset = 0, UINT32 ibOffset = 0)
        {
            m_vertexBuffer       = vb;
            m_indexBuffer        = ib;
            m_vertexBufferOffset = vbOffset;
            m_indexBufferOffset  = ibOffset;
            ++m_stateRevision;
        }

        void SetGPUSceneMeshId(UINT32 id)
        {
            m_gpuSceneMeshId = id;
            ++m_stateRevision;
        }

        void ClearCPUData()
        {
            m_vertices.clear();
            m_indices.clear();
        }

        //==================== CPU DATA ====================
        std::vector<Vertex>      m_vertices;
        std::vector<UINT32>      m_indices;
        UINT32                   m_vertexBufferOffset = 0;
        UINT32                   m_indexBufferOffset  = 0;

        AABB                     m_boundingBox = {};

        //==================== GPU RUNTIME ====================
        NativeHandle             m_vertexBuffer   = 0;
        NativeHandle             m_indexBuffer    = 0;
        UINT32                   m_gpuSceneMeshId = 0;
        UINT32                   m_stateRevision  = 1;

        //==================== NVG METADATA ====================
        GeometryMode             m_geometryMode   = GeometryMode::STANDARD;
        NVGMeshInfo              m_nvgInfo        = {};
    };
}
