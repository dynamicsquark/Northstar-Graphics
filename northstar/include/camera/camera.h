#pragma once

#include "camera_types.h"
#include "camera_desc.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"
#include "../texture/texture.h"
#include <algorithm>
#include <cmath>

namespace Northstar
{
    class Camera
    {
    public:
        // ==================== TRANSFORM GETTERS ====================
        Vec3           GetPosition()                               const { return m_position;              }
        Quat           GetRotation()                               const { return m_rotation;              }
        Vec3           GetDirection()                              const { return m_rotation.GetForward(); }
        Vec3           GetForward()                                const { return m_rotation.GetForward(); }
        Vec3           GetRight()                                  const { return m_rotation.GetRight();   }
        Vec3           GetUp()                                     const { return m_rotation.GetUp();      }

        // ==================== PROJECTION GETTERS ====================
        ProjectionType GetProjectionType()                         const { return m_projectionType; }
        FLOAT32        GetFOV()                                    const { return m_fov;            }
        FLOAT32        GetAspectRatio()                            const { return m_aspectRatio;    }
        FLOAT32        GetNearPlane()                              const { return m_nearPlane;      }
        FLOAT32        GetFarPlane()                               const { return m_farPlane;       }
        FLOAT32        GetOrthoHeight()                            const { return m_orthoHeight;    }

        // ==================== RENDERING GETTERS ====================
        FLOAT32        GetDepth()                                  const { return m_depth;         }
        UINT32         GetCullingMask()                            const { return m_cullingMask;   }
        Texture*       GetRenderTarget()                           const { return m_renderTarget;  }
        Texture*       GetDepthStencilTarget()                     const { return m_depthStencilTarget; }
        Texture*       GetSceneDepthTexture()                      const { return m_depthStencilTarget ? m_depthStencilTarget : m_sceneDepthTexture; }
        // SceneDepth Validity Contract:
        //   - Engine automatically executes DEPTH_PREPASS whenever required by active frame passes
        //     (NVS virtual shadow feedback, POST_DEPTH compute kernels, explicit PassType::DEPTH shaders,
        //     or opaque materials sampling SceneDepth SRV against a Read-Only DSV).
        //   - SceneDepth is always populated and valid after the Opaque Geometry Pass (POST_OPAQUE, TRANSPARENT, POST_PASS).
        BOOL           IsSceneDepthValidAfterOpaque() const
        {
            return GetSceneDepthTexture() != nullptr;
        }
        CameraRenderType GetRenderType()                           const { return m_renderType;    }
        FLOAT32        GetViewportX()                              const { return m_viewportX;     }
        FLOAT32        GetViewportY()                              const { return m_viewportY;     }
        FLOAT32        GetViewportWidth()                          const { return m_viewportWidth; }
        FLOAT32        GetViewportHeight()                         const { return m_viewportHeight;}

        // ==================== TRANSFORM SETTERS ====================
        void SetPosition(const Vec3& pos)
        {
            if (std::isfinite(pos.x) && std::isfinite(pos.y) && std::isfinite(pos.z))
            {
                m_position = pos;
            }
        }
        void SetRotation(const Quat& rot)
        {
            if (std::isfinite(rot.x) && std::isfinite(rot.y) && std::isfinite(rot.z) && std::isfinite(rot.w))
            {
                const FLOAT32 lenSq = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w;
                m_rotation = (lenSq > 1e-12f) ? rot.Normalized() : Quat::Identity();
            }
        }
        void SetTransform(const Vec3& pos, const Quat& rot)
        {
            SetPosition(pos);
            SetRotation(rot);
        }
        void LookAt(const Vec3& target, const Vec3& up = Vec3::Up())
        {
            const Vec3 forward = target - m_position;
            if (forward.LengthSq() > 1e-10f)
            {
                SetRotation(Quat::FromLookRotation(forward, up));
            }
        }

        // ==================== PROJECTION SETTERS ====================
        void SetProjectionType(ProjectionType type)                { m_projectionType = type; }
        void SetFOV(FLOAT32 fov)
        {
            if (std::isfinite(fov) && fov > 0.1f && fov < 179.9f)
            {
                m_fov = fov;
            }
        }
        void SetAspectRatio(FLOAT32 aspect)
        {
            if (std::isfinite(aspect) && aspect > 1e-4f)
            {
                m_aspectRatio = aspect;
            }
        }
        void SetNearPlane(FLOAT32 nearP)
        {
            if (!std::isfinite(nearP)) return;
            if (m_projectionType == ProjectionType::ORTHOGRAPHIC || nearP > 1e-5f)
            {
                m_nearPlane = nearP;
            }
        }
        void SetFarPlane(FLOAT32 farP)
        {
            if (!std::isfinite(farP)) return;
            if (m_projectionType == ProjectionType::ORTHOGRAPHIC || farP > 1e-4f)
            {
                m_farPlane = farP;
            }
        }
        void SetOrthoHeight(FLOAT32 height)
        {
            if (std::isfinite(height) && height > 1e-4f)
            {
                m_orthoHeight = height;
            }
        }

        // ==================== RENDERING SETTERS ====================
        void SetDepth(FLOAT32 depth)                               { if (std::isfinite(depth)) m_depth = depth; }
        void SetCullingMask(UINT32 mask)                           { m_cullingMask = mask;          }
        void SetRenderTarget(Texture* rt)                          { m_renderTarget = rt;           }
        void SetDepthStencilTarget(Texture* dst)                   { m_depthStencilTarget = dst;    }
        void SetRenderType(CameraRenderType type)                  { m_renderType = type;           }
        void SetViewport(FLOAT32 x, FLOAT32 y, FLOAT32 width, FLOAT32 height)
        {
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height))
                return;
            if (width < 0.0f || height < 0.0f)
                return;
            m_viewportX      = x;
            m_viewportY      = y;
            m_viewportWidth  = width;
            m_viewportHeight = height;
        }

        // ==================== MATRIX & FRUSTUM COMPUTATIONS ====================
        Mat4 GetViewMatrix() const
        {
            return Mat4::LookAt(m_position, m_position + m_rotation.GetForward(), m_rotation.GetUp());
        }

        Mat4 GetProjectionMatrix() const
        {
            const FLOAT32 safeAsp = std::max(1e-4f, m_aspectRatio);
            if (m_projectionType == ProjectionType::ORTHOGRAPHIC)
            {
                const FLOAT32 orthoNear  = std::isfinite(m_nearPlane) ? m_nearPlane : 0.0f;
                const FLOAT32 orthoFar   = std::isfinite(m_farPlane) ? std::max(orthoNear + 1e-4f, m_farPlane) : (orthoNear + 1.0f);
                const FLOAT32 safeH      = std::max(1e-4f, m_orthoHeight);
                const FLOAT32 halfWidth  = (safeH * safeAsp) * 0.5f;
                const FLOAT32 halfHeight = safeH * 0.5f;
                return Mat4::Orthographic(-halfWidth, halfWidth, -halfHeight, halfHeight, orthoNear, orthoFar);
            }
            const FLOAT32 safeNear = std::max(1e-5f, m_nearPlane);
            const FLOAT32 safeFar  = std::max(safeNear + 1e-3f, m_farPlane);
            const FLOAT32 safeFov  = std::clamp(m_fov, 0.1f, 179.9f);
            return Mat4::Perspective(safeFov * DEG2RAD, safeAsp, safeNear, safeFar);
        }

        Mat4 GetViewProjectionMatrix() const
        {
            return GetProjectionMatrix() * GetViewMatrix();
        }

        Frustum GetFrustum() const
        {
            return Frustum::FromViewProjection(GetViewProjectionMatrix());
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreSceneManager;

        Camera() = default;

        CameraRenderType m_renderType                              = CameraRenderType::TO_WINDOW;

        // Transform
        Vec3           m_position                                  = Vec3::Zero();
        Quat           m_rotation                                  = Quat::Identity();

        // Projection
        ProjectionType m_projectionType                            = ProjectionType::PERSPECTIVE;
        FLOAT32        m_fov                                       = 60.0f;
        FLOAT32        m_aspectRatio                               = 16.0f / 9.0f;
        FLOAT32        m_nearPlane                                 = 0.1f;
        FLOAT32        m_farPlane                                  = 1000.0f;
        FLOAT32        m_orthoHeight                               = 10.0f;

        // Rendering
        FLOAT32        m_depth                                     = 0.0f;
        UINT32         m_cullingMask                               = 0xFFFFFFFF;
        Texture*       m_renderTarget                              = nullptr;
        Texture*       m_depthStencilTarget                        = nullptr;
        Texture*       m_sceneDepthTexture                         = nullptr;

        // Viewport (in pixels, 0 for full screen/target)
        FLOAT32        m_viewportX                                 = 0.0f;
        FLOAT32        m_viewportY                                 = 0.0f;
        FLOAT32        m_viewportWidth                             = 0.0f;
        FLOAT32        m_viewportHeight                            = 0.0f;
    };
}