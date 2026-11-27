#pragma once

#include "light_desc.h"
#include "../types/northstar_math.h"
#include <algorithm>
#include <cmath>

namespace Northstar
{
    // Fills ENGINE_PROPERTIES (Light source):
    //   LIGHT_POSITION / COLOR / DIRECTION / INTENSITY
    //   GLOBAL_LIGHT_* (same fields when packed into SB arrays)
    class Light
    {
    public:
        // ==================== CORE GETTERS ====================
        LightType              GetType()                                const { return m_type;                  }
        Color                  GetColor()                               const { return m_color;                 } // LIGHT_COLOR
        FLOAT32                GetIntensity()                           const { return m_intensity;             } // LIGHT_INTENSITY
        LightFlags             GetFlags()                               const { return m_flags;                 }

        // ==================== TRANSFORM GETTERS ====================
        Vec3                   GetPosition()                            const { return m_position;              } // LIGHT_POSITION
        Quat                   GetRotation()                            const { return m_rotation;              }
        Vec3                   GetDirection()                           const { return m_rotation.GetForward(); } // LIGHT_DIRECTION

        // ==================== ATTENUATION GETTERS (POINT / SPOT) ====================
        FLOAT32                GetRange()                               const { return m_range;                 }
        FLOAT32                GetConstantAttenuation()                 const { return m_constantAttenuation;   }
        FLOAT32                GetLinearAttenuation()                   const { return m_linearAttenuation;     }
        FLOAT32                GetQuadraticAttenuation()                const { return m_quadraticAttenuation;  }

        // ==================== SPOT GETTERS ====================
        FLOAT32                GetInnerConeAngle()                      const { return m_innerConeAngle;        }
        FLOAT32                GetOuterConeAngle()                      const { return m_outerConeAngle;        }

        // ==================== SHADOW GETTERS (NVS) ====================
        LightShadowResolution  GetShadowResolution()                    const { return m_shadowResolution;      }
        LightShadowSoftness    GetShadowSoftness()                      const { return m_shadowSoftness;        }
        FLOAT32                GetShadowBias()                          const { return m_shadowBias;            }
        FLOAT32                GetShadowNormalBias()                    const { return m_shadowNormalBias;      }
        FLOAT32                GetShadowNearPlane()                     const { return m_shadowNearPlane;       }
        FLOAT32                GetShadowMaxDistance()                   const { return m_shadowMaxDistance;     }

        // ==================== CULLING GETTERS ====================
        UINT32                 GetCullingMask()                         const { return m_cullingMask;           }
        UINT64                 GetRuntimeUid()                          const { return m_runtimeUid;            }

        // ==================== CORE SETTERS ====================
        void SetType(LightType type)                                    { m_type = type;                       }
        void SetColor(const Color& color)
        {
            if (std::isfinite(color.r) && std::isfinite(color.g) && std::isfinite(color.b) && std::isfinite(color.a))
            {
                m_color = color;
            }
        }
        void SetIntensity(FLOAT32 intensity)
        {
            if (std::isfinite(intensity))
            {
                m_intensity = std::max(0.0f, intensity);
            }
        }
        void SetFlags(LightFlags flags)                                 { m_flags = flags;                     }

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
        void SetDirection(const Vec3& direction, const Vec3& up = Vec3::Up())
        {
            if (std::isfinite(direction.x) && std::isfinite(direction.y) && std::isfinite(direction.z) &&
                direction.LengthSq() > 1e-10f)
            {
                SetRotation(Quat::FromLookRotation(direction, up));
            }
        }
        void LookAt(const Vec3& target, const Vec3& up = Vec3::Up())
        {
            SetDirection(target - m_position, up);
        }

        // ==================== ATTENUATION SETTERS (POINT / SPOT) ====================
        void SetRange(FLOAT32 range)
        {
            if (std::isfinite(range) && range >= 0.0f)
            {
                m_range = range;
            }
        }
        void SetConstantAttenuation(FLOAT32 val)                        { if (std::isfinite(val)) m_constantAttenuation = val;  }
        void SetLinearAttenuation(FLOAT32 val)                          { if (std::isfinite(val)) m_linearAttenuation = val;    }
        void SetQuadraticAttenuation(FLOAT32 val)                       { if (std::isfinite(val)) m_quadraticAttenuation = val; }

        // ==================== SPOT SETTERS ====================
        void SetInnerConeAngle(FLOAT32 degrees)
        {
            if (std::isfinite(degrees))
            {
                m_innerConeAngle = std::clamp(degrees, 0.0f, m_outerConeAngle);
            }
        }
        void SetOuterConeAngle(FLOAT32 degrees)
        {
            if (std::isfinite(degrees))
            {
                m_outerConeAngle = std::clamp(degrees, 0.5f, 179.5f);
                if (m_innerConeAngle > m_outerConeAngle)
                {
                    m_innerConeAngle = m_outerConeAngle;
                }
            }
        }

        // ==================== SHADOW SETTERS (NVS) ====================
        BOOL SetShadowResolution(LightShadowResolution res)
        {
            if (!IsValidLightShadowResolution(res))
            {
                return false;
            }
            m_shadowResolution = res;
            return true;
        }

        BOOL SetShadowSoftness(LightShadowSoftness softness)
        {
            if (!IsValidLightShadowSoftness(softness))
            {
                return false;
            }
            m_shadowSoftness = softness;
            return true;
        }

        // Dynamic texel-relative depth bias in [0.0, 1.0] (1.0 = 1.0x local virtual shadow texel depth offset)
        BOOL SetShadowBias(FLOAT32 bias)
        {
            if (!std::isfinite(bias) || bias < 0.0f || bias > 1.0f)
            {
                return false;
            }
            m_shadowBias = bias;
            return true;
        }

        // Dynamic texel-relative normal bias in [0.0, 1.0] (1.0 = 1.0x local virtual shadow texel world footprint)
        BOOL SetShadowNormalBias(FLOAT32 bias)
        {
            if (!std::isfinite(bias) || bias < 0.0f || bias > 1.0f)
            {
                return false;
            }
            m_shadowNormalBias = bias;
            return true;
        }

        BOOL SetShadowNearPlane(FLOAT32 nearP)
        {
            if (!std::isfinite(nearP) || nearP < 0.001f)
            {
                return false;
            }
            m_shadowNearPlane = nearP;
            return true;
        }

        BOOL SetShadowMaxDistance(FLOAT32 maxDist)
        {
            if (!std::isfinite(maxDist) || maxDist <= 0.0f)
            {
                return false;
            }
            m_shadowMaxDistance = maxDist;
            return true;
        }

        // ==================== CULLING SETTERS ====================
        void SetCullingMask(UINT32 mask)                                { m_cullingMask = mask;                }

        // ==================== FLAG HELPERS ====================
        BOOL HasFlag(LightFlags flag)                                   const { return (m_flags & flag) == flag; }
        void AddFlag(LightFlags flag)                                   { m_flags |= flag;                     }
        void RemoveFlag(LightFlags flag)                                { m_flags &= ~flag;                    }

        BOOL IsEnabled()                                                const { return HasFlag(LightFlags::ENABLED); }
        void SetEnabled(BOOL enabled)                                   { if (enabled) AddFlag(LightFlags::ENABLED); else RemoveFlag(LightFlags::ENABLED); }

        BOOL IsShadowCaster()                                           const { return HasFlag(LightFlags::CAST_SHADOW); }
        void SetCastShadows(BOOL enabled)                               { if (enabled) AddFlag(LightFlags::CAST_SHADOW); else RemoveFlag(LightFlags::CAST_SHADOW); }

        // ==================== SHADOW MATRIX COMPUTATIONS ====================
        Mat4 GetViewMatrix() const
        {
            return Mat4::LookAt(m_position, m_position + m_rotation.GetForward(), m_rotation.GetUp());
        }

        Mat4 GetProjectionMatrix() const
        {
            const FLOAT32 effFar = std::max(
                m_shadowNearPlane + 0.05f,
                (m_type == LightType::DIRECTIONAL) ? m_shadowMaxDistance : std::min(m_range, m_shadowMaxDistance));
            if (m_type == LightType::SPOT)
            {
                const FLOAT32 safeHalfDeg = std::clamp(m_outerConeAngle, 0.5f, 88.0f);
                return Mat4::Perspective(safeHalfDeg * 2.0f * DEG2RAD, 1.0f, m_shadowNearPlane, effFar);
            }
            if (m_type == LightType::DIRECTIONAL)
            {
                const FLOAT32 extent = std::max(0.05f, m_shadowMaxDistance);
                return Mat4::Orthographic(-extent, extent, -extent, extent, m_shadowNearPlane, effFar);
            }
            return Mat4::Perspective(90.0f * DEG2RAD, 1.0f, m_shadowNearPlane, effFar);
        }

        Mat4 GetViewProjectionMatrix() const
        {
            return GetProjectionMatrix() * GetViewMatrix();
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreSceneManager;

        Light() = default;

        // Core
        LightType              m_type                                   = LightType::POINT;
        Color                  m_color                                  = Color::White();
        FLOAT32                m_intensity                              = 1.0f;
        LightFlags             m_flags                                  = LightFlags::ENABLED | LightFlags::CAST_SHADOW;

        // Transform
        Vec3                   m_position                               = Vec3::Zero();
        Quat                   m_rotation                               = Quat::Identity();

        // Attenuation (Point / Spot)
        FLOAT32                m_range                                  = 10.0f;
        FLOAT32                m_constantAttenuation                    = 1.0f;
        FLOAT32                m_linearAttenuation                      = 0.09f;
        FLOAT32                m_quadraticAttenuation                   = 0.032f;

        // Spot
        FLOAT32                m_innerConeAngle                         = 25.0f;
        FLOAT32                m_outerConeAngle                         = 35.0f;

        // Shadow (NVS)
        LightShadowResolution  m_shadowResolution                       = LightShadowResolution::RES_16384;
        LightShadowSoftness    m_shadowSoftness                         = LightShadowSoftness::HARD;
        FLOAT32                m_shadowBias                             = 0.5f;
        FLOAT32                m_shadowNormalBias                       = 0.5f;
        FLOAT32                m_shadowNearPlane                        = 0.1f;
        FLOAT32                m_shadowMaxDistance                      = 150.0f;

        // Culling & Identity
        UINT32                    m_cullingMask                            = 0xFFFFFFFF;
        UINT64                    m_runtimeUid                             = 0ULL;
    };
}
