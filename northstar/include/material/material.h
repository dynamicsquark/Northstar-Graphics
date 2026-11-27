#pragma once

#include "material_types.h"
#include "material_property_store.h"
#include "../nvg/nvg_types.h"
#include "../compute/compute_kernel.h"
#include "../shader/shader_types.h"
#include "../shader_set/shader_set.h"
#include "../texture/texture.h"
#include "../sampler/sampler.h"
#include "../sampler/sampler_types.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"

#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

namespace Northstar
{
    class Material
    {
    public:
        ~Material() = default;

        static BOOL FieldNameEquals(const std::string& stored, const char* field)
        {
            if (!field) return false;
            if (stored == field) return true;

            const SIZE_T fieldLen = std::strlen(field);
            if (stored.size() > fieldLen + 2)
            {
                const SIZE_T sep = stored.size() - fieldLen - 2;
                if (stored[sep] == '_' && stored[sep + 1] == '_' &&
                    std::strcmp(stored.c_str() + sep + 2, field) == 0)
                    return true;
            }
            return false;
        }

        // ==================== INITIALIZATION & PASS MANAGEMENT ====================

        // Initializes the material for a specific render pass using a ShaderSet.
        // Defaults to PassType::GEOMETRY.
        void Init(ShaderSet* shaderSet, PassType pass = PassType::GEOMETRY)
        {
            if (pass >= PassType::COUNT) return;
            m_shaderSets[static_cast<SIZE_T>(pass)] = shaderSet;
            ++m_stateRevision;
            if (!shaderSet) return;

            // Clear previous slots for this pass
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                stageProps.buffers.clear();
                stageProps.textures.clear();
                stageProps.rwTextures.clear();
                stageProps.samplers.clear();
                stageProps.structuredBuffers.clear();
            }

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                if (stage == ShaderStage::VERTEX)
                {
                    // Inspect all registered, valid vertex variants so material reflects all variant properties
                    for (SIZE_T v = 0; v < static_cast<SIZE_T>(ShaderVariant::COUNT); ++v)
                    {
                        const Shader* vs = shaderSet->GetVertexShader(static_cast<ShaderVariant>(v));
                        if (vs && vs->IsValid())
                        {
                            PopulateStageSlots(pass, static_cast<GraphicsShaderSlot>(s), vs->GetReflection());
                        }
                    }
                }
                else
                {
                    const Shader* shader = shaderSet->GetShader(stage);
                    if (!shader || !shader->IsValid()) continue;

                    PopulateStageSlots(pass, static_cast<GraphicsShaderSlot>(s), shader->GetReflection());
                }
            }
        }

        // Explicit Geometry Pass ShaderSet configuration
        void SetGeometryShaderSet(ShaderSet* shaderSet)
        {
            Init(shaderSet, PassType::GEOMETRY);
        }

        ShaderSet* GetGeometryShaderSet() const
        {
            return GetShaderSet(PassType::GEOMETRY);
        }

        // Explicit Shadow Pass ShaderSet configuration (PassType::SHADOW)
        void SetShadowShaderSet(ShaderSet* shaderSet)
        {
            Init(shaderSet, PassType::SHADOW);
        }

        ShaderSet* GetShadowShaderSet() const
        {
            return GetShaderSet(PassType::SHADOW);
        }

        // Explicit Depth Prepass ShaderSet configuration (PassType::DEPTH)
        void SetDepthShaderSet(ShaderSet* shaderSet)
        {
            Init(shaderSet, PassType::DEPTH);
        }

        ShaderSet* GetDepthShaderSet() const
        {
            return GetShaderSet(PassType::DEPTH);
        }

        // Multi-pass configuration: Assign or update the ShaderSet used for a specific pass (e.g. SHADOW, DEPTH_PREPASS)
        void SetShaderSet(PassType pass, ShaderSet* shaderSet)
        {
            Init(shaderSet, pass);
        }

        ShaderSet* GetShaderSet(PassType pass = PassType::GEOMETRY) const
        {
            if (pass >= PassType::COUNT) return nullptr;
            return m_shaderSets[static_cast<SIZE_T>(pass)];
        }

        BOOL HasPass(PassType pass) const
        {
            if (pass >= PassType::COUNT) return false;
            return m_shaderSets[static_cast<SIZE_T>(pass)] != nullptr;
        }

        void ClearPass(PassType pass)
        {
            if (pass >= PassType::COUNT) return;
            m_shaderSets[static_cast<SIZE_T>(pass)] = nullptr;
            ++m_stateRevision;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                stageProps.buffers.clear();
                stageProps.textures.clear();
                stageProps.rwTextures.clear();
                stageProps.samplers.clear();
                stageProps.structuredBuffers.clear();
            }
        }

        // ==================== SHADER SET RESOLUTION ====================

        // Returns explicit user ShaderSet if set for this pass; otherwise returns nullptr (no fallback)
        ShaderSet* GetEffectiveShaderSet(PassType pass = PassType::GEOMETRY, BOOL = false) const
        {
            if (pass >= PassType::COUNT) return nullptr;
            return m_shaderSets[static_cast<SIZE_T>(pass)];
        }

        // ==================== COMPUTE KERNEL INTEGRATION ====================
        // Materials can attach one or more ComputeKernels (e.g., procedural texture/buffer generation).
        //
        // Automatic Execution Contract:
        // 1. Visibility Gating: A Material's attached ComputeKernels are scheduled in ComputePassPhase::PRE_PASS
        //    if and only if at least one RenderObject (or static batch) using this Material is visible in the
        //    rendered camera's frustum (Scene::GetVisibleObjects).
        // 2. Execution Scope: Each kernel's ComputeExecutionScope (PER_VIEW vs PER_FRAME) is respected across
        //    multi-camera renders within the same BeginFrame/EndFrame cycle.
        // 3. Scene + Material Deduplication: If the same ComputeKernel* is also registered on the Scene, it is
        //    deduplicated so it executes at most once per view (with explicit Scene phase registration taking
        //    precedence over Material's implicit PRE_PASS).

        void SetComputeKernel(ComputeKernel* kernel)
        {
            m_computeKernels.clear();
            if (kernel)
            {
                m_computeKernels.push_back(kernel);
            }
            ++m_stateRevision;
        }

        ComputeKernel* GetComputeKernel() const
        {
            return m_computeKernels.empty() ? nullptr : m_computeKernels[0];
        }

        void AddComputeKernel(ComputeKernel* kernel)
        {
            if (!kernel) return;
            if (std::find(m_computeKernels.begin(), m_computeKernels.end(), kernel) == m_computeKernels.end())
            {
                m_computeKernels.push_back(kernel);
                ++m_stateRevision;
            }
        }

        void RemoveComputeKernel(ComputeKernel* kernel)
        {
            auto it = std::find(m_computeKernels.begin(), m_computeKernels.end(), kernel);
            if (it != m_computeKernels.end())
            {
                m_computeKernels.erase(it);
                ++m_stateRevision;
            }
        }

        void ClearComputeKernels()
        {
            if (!m_computeKernels.empty())
            {
                m_computeKernels.clear();
                ++m_stateRevision;
            }
        }

        const std::vector<ComputeKernel*>& GetComputeKernels() const
        {
            return m_computeKernels;
        }

        BOOL HasComputeKernels() const
        {
            return !m_computeKernels.empty();
        }

        // ==================== RASTER & PIPELINE STATE ====================
        CullMode          GetCullMode()    const { return m_cullMode;    }
        BlendMode         GetBlendMode()   const { return m_blendMode;   }
        FillMode          GetFillMode()    const { return m_fillMode;    }
        const DepthState& GetDepthState()  const { return m_depthState;  }
        UINT32            GetGPUSceneMaterialId() const { return m_gpuSceneMaterialId; }
        UINT32            GetStateRevision()      const { return m_stateRevision; }

        void SetCullMode(CullMode mode)          { m_cullMode = mode;   ++m_stateRevision; }
        void SetBlendMode(BlendMode mode)        { m_blendMode = mode;  ++m_stateRevision; }
        void SetFillMode(FillMode mode)          { m_fillMode = mode;   ++m_stateRevision; }
        void SetDepthState(const DepthState& s)  { m_depthState = s;    ++m_stateRevision; }

        // ==================== VIRTUAL GEOMETRY (NVG) POLICY ====================
        const NVGMaterialPolicy& GetNVGPolicy() const { return m_nvgPolicy; }
        void SetNVGPolicy(const NVGMaterialPolicy& policy)
        {
            m_nvgPolicy = policy;
            ++m_stateRevision;
        }

        BOOL IsTransparent()       const { return m_blendMode != BlendMode::OPAQUE; }
        BOOL IsTwoSided()          const { return m_cullMode == CullMode::NONE;   }
        BOOL IsWireframe()         const { return m_fillMode == FillMode::WIREFRAME; }
        BOOL IsDepthTestEnabled()  const { return m_depthState.test; }
        BOOL IsDepthWriteEnabled() const { return m_depthState.write; }

        BOOL IsAlphaTested() const
        {
            for (SIZE_T p = 0; p < static_cast<SIZE_T>(PassType::COUNT); ++p)
            {
                const ShaderSet* ss = m_shaderSets[p];
                if (!ss) continue;
                const Shader* ps = ss->GetShader(ShaderStage::PIXEL);
                if (ps && ps->IsValid() && (ps->GetReflection().flags & NSL::NSC_FLAG_USES_DISCARD) != 0)
                {
                    return true;
                }
            }
            return false;
        }

        const MaterialPropertyStore& GetPropertyStore() const { return m_props; }
        MaterialPropertyStore&       GetPropertyStore()       { return m_props; }

        // =====================================================================
        //  INTROSPECTION / QUERYING
        // =====================================================================

        BOOL HasProperty(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* field = nullptr;
            return ResolveBufferFieldByNameConst(pass, name, slot, field);
        }

        BOOL HasProperty(const char* group, const char* name, PassType pass = PassType::GEOMETRY) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* field = nullptr;
            return ResolveGroupBufferFieldConst(pass, group, name, slot, field);
        }

        BOOL HasBuffer(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            if (pass >= PassType::COUNT || !name) return false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& sp = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (const auto& b : sp.buffers)
                {
                    if (b.resourceName == name || b.propertyName == name || (b.propertyName + "__" + b.resourceName == name))
                        return true;
                }
            }
            return false;
        }

        BOOL HasTexture(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetTextureByName(pass, name) != nullptr ||
                   FindTextureSlotConst(pass, name) != nullptr ||
                   FindRWTextureSlotConst(pass, name) != nullptr;
        }

        BOOL HasRWTexture(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return FindRWTextureSlotConst(pass, name) != nullptr;
        }

        BOOL HasSampler(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return FindSamplerSlotConst(pass, name) != nullptr;
        }

        BOOL HasStructuredBuffer(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return FindStructuredBufferSlotConst(pass, name) != nullptr;
        }

        BOOL HasRWStructuredBuffer(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(pass, name);
            return slot && slot->isUAV;
        }

        // =====================================================================
        //  DIRECT PROPERTY SETTERS (Default Pass: GEOMETRY)
        // =====================================================================

        void SetFloat (const char* name, FLOAT32 value)           { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetInt   (const char* name, INT32 value)             { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetUInt  (const char* name, UINT32 value)            { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetBool  (const char* name, BOOL value)              { const INT32 i = value ? 1 : 0; WriteFieldByName(PassType::GEOMETRY, name, &i, sizeof(i)); }
        void SetVec2  (const char* name, const Vec2& value)       { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetVec3  (const char* name, const Vec3& value)       { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetVec4  (const char* name, const Vec4& value)       { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetColor (const char* name, const Color& value)      { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }
        void SetMat3  (const char* name, const Mat3& value)       { WriteMat3ByName(PassType::GEOMETRY, name, value); }
        void SetMat4  (const char* name, const Mat4& value)       { WriteFieldByName(PassType::GEOMETRY, name, &value, sizeof(value)); }

        void SetTexture(const char* name, Texture* texture)
        {
            SetTextureByName(PassType::GEOMETRY, name, texture);
        }

        void SetSampler(const char* name, SamplerState* sampler)
        {
            SetSamplerByName(PassType::GEOMETRY, name, sampler);
        }

        // =====================================================================
        //  DIRECT PROPERTY GETTERS (Default Pass: GEOMETRY)
        // =====================================================================

        BOOL GetFloat (const char* name, FLOAT32& out, PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetInt   (const char* name, INT32& out,   PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetUInt  (const char* name, UINT32& out,  PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetBool  (const char* name, BOOL& out,    PassType pass = PassType::GEOMETRY) const
        {
            INT32 i = 0;
            if (!ReadFieldByName(pass, name, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* name, Vec2& out,    PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetVec3  (const char* name, Vec3& out,    PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetVec4  (const char* name, Vec4& out,    PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetColor (const char* name, Color& out,   PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }
        BOOL GetMat4  (const char* name, Mat4& out,    PassType pass = PassType::GEOMETRY) const { return ReadFieldByName(pass, name, &out, sizeof(out)); }

        FLOAT32 GetFloatOrDefault(const char* name, FLOAT32 defaultVal = 0.0f, PassType pass = PassType::GEOMETRY) const
        {
            FLOAT32 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        INT32 GetIntOrDefault(const char* name, INT32 defaultVal = 0, PassType pass = PassType::GEOMETRY) const
        {
            INT32 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        UINT32 GetUIntOrDefault(const char* name, UINT32 defaultVal = 0, PassType pass = PassType::GEOMETRY) const
        {
            UINT32 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        BOOL GetBoolOrDefault(const char* name, BOOL defaultVal = false, PassType pass = PassType::GEOMETRY) const
        {
            INT32 i = defaultVal ? 1 : 0;
            ReadFieldByName(pass, name, &i, sizeof(i));
            return (i != 0);
        }

        Vec2 GetVec2OrDefault(const char* name, const Vec2& defaultVal = Vec2(0,0), PassType pass = PassType::GEOMETRY) const
        {
            Vec2 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        Vec3 GetVec3OrDefault(const char* name, const Vec3& defaultVal = Vec3(0,0,0), PassType pass = PassType::GEOMETRY) const
        {
            Vec3 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        Vec4 GetVec4OrDefault(const char* name, const Vec4& defaultVal = Vec4(0,0,0,0), PassType pass = PassType::GEOMETRY) const
        {
            Vec4 val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        Color GetColorOrDefault(const char* name, const Color& defaultVal = Color(1,1,1,1), PassType pass = PassType::GEOMETRY) const
        {
            Color val = defaultVal;
            ReadFieldByName(pass, name, &val, sizeof(val));
            return val;
        }

        Texture* GetTexture(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetTextureByName(pass, name);
        }

        SamplerState* GetSampler(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetSamplerByName(pass, name);
        }

        void SetRWTexture(const char* name, Texture* texture, PassType pass = PassType::GEOMETRY)
        {
            SetRWTextureByName(pass, name, texture, 0u, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        }

        BOOL SetRWTexture(const char* name, Texture* texture, UINT32 mipSlice,
                          PassType pass = PassType::GEOMETRY,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            return SetRWTextureByName(pass, name, texture, mipSlice, uavAccess);
        }

        Texture* GetRWTexture(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetTextureByName(pass, name);
        }

        // =====================================================================
        //  GROUP-QUALIFIED SETTERS & GETTERS
        // =====================================================================

        void SetFloat (const char* group, const char* field, FLOAT32 value)      { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetInt   (const char* group, const char* field, INT32 value)        { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetUInt  (const char* group, const char* field, UINT32 value)       { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetBool  (const char* group, const char* field, BOOL value)         { const INT32 i = value ? 1 : 0; WriteGroupFieldByName(PassType::GEOMETRY, group, field, &i, sizeof(i)); }
        void SetVec2  (const char* group, const char* field, const Vec2& value)  { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetVec3  (const char* group, const char* field, const Vec3& value)  { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetVec4  (const char* group, const char* field, const Vec4& value)  { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetColor (const char* group, const char* field, const Color& value) { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }
        void SetMat4  (const char* group, const char* field, const Mat4& value)  { WriteGroupFieldByName(PassType::GEOMETRY, group, field, &value, sizeof(value)); }

        void SetTexture(const char* group, const char* name, Texture* texture)
        {
            SetTextureByGroupName(PassType::GEOMETRY, group, name, texture);
        }

        void SetSampler(const char* group, const char* name, SamplerState* sampler)
        {
            SetSamplerByGroupName(PassType::GEOMETRY, group, name, sampler);
        }

        BOOL GetFloat (const char* group, const char* field, FLOAT32& out, PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetInt   (const char* group, const char* field, INT32& out,   PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetUInt  (const char* group, const char* field, UINT32& out,  PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetBool  (const char* group, const char* field, BOOL& out,    PassType pass = PassType::GEOMETRY) const
        {
            INT32 i = 0;
            if (!ReadGroupFieldByName(pass, group, field, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* group, const char* field, Vec2& out,    PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetVec3  (const char* group, const char* field, Vec3& out,    PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetVec4  (const char* group, const char* field, Vec4& out,    PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetColor (const char* group, const char* field, Color& out,   PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }
        BOOL GetMat4  (const char* group, const char* field, Mat4& out,    PassType pass = PassType::GEOMETRY) const { return ReadGroupFieldByName(pass, group, field, &out, sizeof(out)); }

        // Fully disambiguated (property, resource, field) overloads across all stages of a pass
        void SetFloat (const char* property, const char* resource, const char* field, FLOAT32 value)      { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetInt   (const char* property, const char* resource, const char* field, INT32 value)        { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetUInt  (const char* property, const char* resource, const char* field, UINT32 value)       { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetBool  (const char* property, const char* resource, const char* field, BOOL value)         { const INT32 i = value ? 1 : 0; WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &i, sizeof(i)); }
        void SetVec2  (const char* property, const char* resource, const char* field, const Vec2& value)  { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetVec3  (const char* property, const char* resource, const char* field, const Vec3& value)  { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetVec4  (const char* property, const char* resource, const char* field, const Vec4& value)  { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetColor (const char* property, const char* resource, const char* field, const Color& value) { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }
        void SetMat3  (const char* property, const char* resource, const char* field, const Mat3& value)  { WriteQualifiedMat3ByName(PassType::GEOMETRY, property, resource, field, value); }
        void SetMat4  (const char* property, const char* resource, const char* field, const Mat4& value)  { WriteQualifiedFieldByName(PassType::GEOMETRY, property, resource, field, &value, sizeof(value)); }

        BOOL GetFloat (const char* property, const char* resource, const char* field, FLOAT32& out, PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetInt   (const char* property, const char* resource, const char* field, INT32& out,   PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetUInt  (const char* property, const char* resource, const char* field, UINT32& out,  PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetBool  (const char* property, const char* resource, const char* field, BOOL& out,    PassType pass = PassType::GEOMETRY) const
        {
            INT32 i = 0;
            if (!ReadQualifiedFieldByName(pass, property, resource, field, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* property, const char* resource, const char* field, Vec2& out,    PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetVec3  (const char* property, const char* resource, const char* field, Vec3& out,    PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetVec4  (const char* property, const char* resource, const char* field, Vec4& out,    PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetColor (const char* property, const char* resource, const char* field, Color& out,   PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }
        BOOL GetMat4  (const char* property, const char* resource, const char* field, Mat4& out,    PassType pass = PassType::GEOMETRY) const { return ReadQualifiedFieldByName(pass, property, resource, field, &out, sizeof(out)); }

        Texture* GetTexture(const char* group, const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetTextureByGroupName(pass, group, name);
        }

        SamplerState* GetSampler(const char* group, const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetSamplerByGroupName(pass, group, name);
        }

        void SetRWTexture(const char* group, const char* name, Texture* texture, PassType pass = PassType::GEOMETRY)
        {
            SetRWTextureByGroupName(pass, group, name, texture, 0u, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        }

        BOOL SetRWTexture(const char* group, const char* name, Texture* texture, UINT32 mipSlice,
                          PassType pass = PassType::GEOMETRY,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            return SetRWTextureByGroupName(pass, group, name, texture, mipSlice, uavAccess);
        }

        Texture* GetRWTexture(const char* group, const char* name, PassType pass = PassType::GEOMETRY) const
        {
            return GetTextureByGroupName(pass, group, name);
        }

        // =====================================================================
        //  PASS-AWARE ERGONOMIC SETTERS & GETTERS
        // =====================================================================

        void SetFloat (PassType pass, const char* name, FLOAT32 value)      { WriteFieldByName(pass, name, &value, sizeof(value)); }
        void SetColor (PassType pass, const char* name, const Color& value) { WriteFieldByName(pass, name, &value, sizeof(value)); }
        void SetTexture(PassType pass, const char* name, Texture* texture)  { SetTextureByName(pass, name, texture); }
        void SetSampler(PassType pass, const char* name, SamplerState* s)   { SetSamplerByName(pass, name, s); }

        // =====================================================================
        //  EXPLICIT LOW-LEVEL SETTERS (Full signature backward compatibility)
        // =====================================================================

        void SetFloat (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, FLOAT32 value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetInt (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, INT32 value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetUInt (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, UINT32 value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetBool (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, BOOL value)
        {
            const INT32 asInt = value ? 1 : 0;
            WriteBufferField(pass, stage, property, resource, field, &asInt, sizeof(asInt));
        }
        void SetVec2 (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Vec2& value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetVec3 (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Vec3& value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetVec4 (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Vec4& value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetColor (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Color& value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }
        void SetMat3 (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Mat3& value)
        {
            WriteMat3Field(pass, stage, property, resource, field, value);
        }
        void SetMat4 (PassType pass, ShaderStage stage, const char* property, const char* resource, const char* field, const Mat4& value)
        {
            WriteBufferField(pass, stage, property, resource, field, &value, sizeof(value));
        }

        void SetTexture(PassType pass, ShaderStage stage, const char* property, const char* resource, Texture* texture)
        {
            if (ShaderTextureSlot* slot = FindTextureSlot(pass, stage, property, resource))
            {
                slot->texture              = texture;
                slot->textureHandle        = texture ? texture->GetNativeHandle() : 0;
                slot->hasDestroyedResource = false;
                ++m_stateRevision;
                return;
            }
            if (ShaderRWTextureSlot* slot = FindRWTextureSlot(pass, stage, property, resource))
            {
                slot->texture              = texture;
                slot->textureHandle        = texture ? texture->GetNativeHandle() : 0;
                slot->hasDestroyedResource = false;
                ++m_stateRevision;
            }
        }

        void SetSampler(PassType pass, ShaderStage stage, const char* property, const char* resource, SamplerState* sampler)
        {
            if (ShaderSamplerSlot* slot = FindSamplerSlot(pass, stage, property, resource))
            {
                slot->sampler              = sampler;
                slot->samplerHandle        = sampler ? sampler->GetNativeHandle() : 0;
                slot->hasDestroyedResource = false;
                ++m_stateRevision;
            }
        }

        void SetStructuredBufferData(PassType pass, ShaderStage stage,
                                     const char* property, const char* resource,
                                     const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot || !elements || elementCount == 0 || slot->stride == 0)
                return;

            const UINT32 bytes = elementCount * slot->stride;
            while (slot->cpuData.size() < bytes)
                slot->cpuData.push_back(0);

            std::memcpy(slot->cpuData.data(), elements, bytes);
            slot->elementCount = elementCount;
            slot->dirty = true;
            ++m_stateRevision;
        }

        // ==================== USER-OWNED STRUCTURED BUFFER ELEMENT API ====================
        UINT32 AddStructuredBufferUserElement(PassType pass, ShaderStage stage,
                                              const char* property, const char* resource)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot || slot->stride == 0) return 0xFFFFFFFF;

            UINT32 userIndex = slot->userElementCount++;
            const SIZE_T newBytes = static_cast<SIZE_T>(slot->userElementCount) * slot->stride;
            while (slot->userData.size() < newBytes)
                slot->userData.push_back(0);

            slot->dirty = true;
            ++m_stateRevision;
            return userIndex;
        }

        UINT32 AddStructuredBufferUserElement(const char* name, PassType pass = PassType::GEOMETRY)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, name);
            if (!slot || slot->stride == 0) return 0xFFFFFFFF;

            UINT32 userIndex = slot->userElementCount++;
            const SIZE_T newBytes = static_cast<SIZE_T>(slot->userElementCount) * slot->stride;
            while (slot->userData.size() < newBytes)
                slot->userData.push_back(0);

            slot->dirty = true;
            ++m_stateRevision;
            return userIndex;
        }

        UINT32 AddStructuredBufferUserElement(const char* property, const char* resource, PassType pass = PassType::GEOMETRY)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, property, resource);
            if (!slot || slot->stride == 0) return 0xFFFFFFFF;

            UINT32 userIndex = slot->userElementCount++;
            const SIZE_T newBytes = static_cast<SIZE_T>(slot->userElementCount) * slot->stride;
            while (slot->userData.size() < newBytes)
                slot->userData.push_back(0);

            slot->dirty = true;
            ++m_stateRevision;
            return userIndex;
        }

        BOOL SetStructuredBufferUserField(PassType pass, ShaderStage stage,
                                          const char* property, const char* resource,
                                          UINT32 userIndex, const char* field,
                                          const void* data, UINT32 byteSize)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot || !data || byteSize == 0 || userIndex >= slot->userElementCount || slot->stride == 0)
                return false;

            const ShaderReflection* refl = GetStageReflection(pass, stage);
            if (!refl) return false;

            std::string qName = std::string(property) + "__" + resource;
            const NSL::NSCStructuredBuffer* sb = refl->FindStructuredBuffer(qName);
            if (!sb) sb = refl->FindStructuredBuffer(resource);
            if (sb)
            {
                for (const auto& f : sb->fields)
                {
                    if (f.name == field || FieldNameEquals(f.name, field))
                    {
                        // Reject writes to engine-managed Get() fields
                        if (f.isGet)
                            return false;

                        if (f.offset + byteSize > slot->stride)
                            return false;

                        BYTE* elemPtr = slot->userData.data() + userIndex * slot->stride;
                        std::memcpy(elemPtr + f.offset, data, (std::min<UINT32>)(byteSize, f.size));
                        slot->dirty = true;
                        ++m_stateRevision;
                        return true;
                    }
                }
            }
            else
            {
                const NSL::NSCUAVBuffer* uav = refl->FindUAV(qName);
                if (!uav) uav = refl->FindUAV(resource);
                if (uav)
                {
                    for (const auto& f : uav->fields)
                    {
                        if (f.name == field || FieldNameEquals(f.name, field))
                        {
                            if (f.isGet)
                                return false;

                            if (f.offset + byteSize > slot->stride)
                                return false;

                            BYTE* elemPtr = slot->userData.data() + userIndex * slot->stride;
                            std::memcpy(elemPtr + f.offset, data, (std::min<UINT32>)(byteSize, f.size));
                            slot->dirty = true;
                            ++m_stateRevision;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        BOOL SetStructuredBufferUserField(const char* name, UINT32 userIndex, const char* field,
                                          const void* data, UINT32 byteSize, PassType pass = PassType::GEOMETRY)
        {
            if (!name || !field || !data || byteSize == 0 || pass >= PassType::COUNT) return false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                    {
                        return SetStructuredBufferUserField(pass, stage,
                                                           stageProps.structuredBuffers[i].propertyName.c_str(),
                                                           stageProps.structuredBuffers[i].resourceName.c_str(),
                                                           userIndex, field, data, byteSize);
                    }
                }
            }
            return false;
        }

        BOOL SetStructuredBufferUserField(const char* property, const char* resource, UINT32 userIndex, const char* field,
                                          const void* data, UINT32 byteSize, PassType pass = PassType::GEOMETRY)
        {
            if (!property || !resource || !field || !data || byteSize == 0 || pass >= PassType::COUNT) return false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        return SetStructuredBufferUserField(pass, stage, property, resource,
                                                           userIndex, field, data, byteSize);
                    }
                }
            }
            return false;
        }

        void SetStructuredBufferUserData(PassType pass, ShaderStage stage,
                                         const char* property, const char* resource,
                                         const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot || !elements || elementCount == 0 || slot->stride == 0)
                return;

            const UINT32 bytes = elementCount * slot->stride;
            while (slot->userData.size() < bytes)
                slot->userData.push_back(0);

            std::memcpy(slot->userData.data(), elements, bytes);
            slot->userElementCount = elementCount;

            // Zero-out engine Get fields across all user elements for strict ownership integrity
            const ShaderReflection* refl = GetStageReflection(pass, stage);
            if (refl)
            {
                std::string qName = std::string(property) + "__" + resource;
                const NSL::NSCStructuredBuffer* sb = refl->FindStructuredBuffer(qName);
                if (!sb) sb = refl->FindStructuredBuffer(resource);
                if (sb)
                {
                    for (UINT32 u = 0; u < elementCount; ++u)
                    {
                        BYTE* elemPtr = slot->userData.data() + u * slot->stride;
                        for (const auto& f : sb->fields)
                        {
                            if (f.isGet && f.offset + f.size <= slot->stride)
                            {
                                std::memset(elemPtr + f.offset, 0, f.size);
                            }
                        }
                    }
                }
                else
                {
                    const NSL::NSCUAVBuffer* uav = refl->FindUAV(qName);
                    if (!uav) uav = refl->FindUAV(resource);
                    if (uav)
                    {
                        for (UINT32 u = 0; u < elementCount; ++u)
                        {
                            BYTE* elemPtr = slot->userData.data() + u * slot->stride;
                            for (const auto& f : uav->fields)
                            {
                                if (f.isGet && f.offset + f.size <= slot->stride)
                                {
                                    std::memset(elemPtr + f.offset, 0, f.size);
                                }
                            }
                        }
                    }
                }
            }

            slot->dirty = true;
            ++m_stateRevision;
        }

        void SetStructuredBufferUserData(const char* name, const void* elements, UINT32 elementCount, PassType pass = PassType::GEOMETRY)
        {
            if (!name || !elements || elementCount == 0 || pass >= PassType::COUNT) return;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                    {
                        SetStructuredBufferUserData(pass, stage,
                                                    stageProps.structuredBuffers[i].propertyName.c_str(),
                                                    stageProps.structuredBuffers[i].resourceName.c_str(),
                                                    elements, elementCount);
                        return;
                    }
                }
            }
        }

        void SetStructuredBufferUserData(const char* property, const char* resource, const void* elements, UINT32 elementCount, PassType pass = PassType::GEOMETRY)
        {
            if (!property || !resource || !elements || elementCount == 0 || pass >= PassType::COUNT) return;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        SetStructuredBufferUserData(pass, stage, property, resource, elements, elementCount);
                        return;
                    }
                }
            }
        }

        void ClearStructuredBufferUserElements(PassType pass, ShaderStage stage,
                                               const char* property, const char* resource)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot) return;

            slot->userData.clear();
            slot->userElementCount = 0;
            slot->dirty = true;
            ++m_stateRevision;
        }

        void ClearStructuredBufferUserElements(const char* name, PassType pass = PassType::GEOMETRY)
        {
            if (!name || pass >= PassType::COUNT) return;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                    {
                        ClearStructuredBufferUserElements(pass, stage,
                                                          stageProps.structuredBuffers[i].propertyName.c_str(),
                                                          stageProps.structuredBuffers[i].resourceName.c_str());
                        return;
                    }
                }
            }
        }

        void ClearStructuredBufferUserElements(const char* property, const char* resource, PassType pass = PassType::GEOMETRY)
        {
            if (!property || !resource || pass >= PassType::COUNT) return;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        ClearStructuredBufferUserElements(pass, stage, property, resource);
                        return;
                    }
                }
            }
        }

        BOOL RemoveStructuredBufferUserElement(PassType pass, ShaderStage stage,
                                               const char* property, const char* resource,
                                               UINT32 userIndex)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot || userIndex >= slot->userElementCount || slot->stride == 0)
                return false;

            const SIZE_T stride = slot->stride;
            BYTE* base = slot->userData.data();
            const SIZE_T remainingElements = static_cast<SIZE_T>(slot->userElementCount - 1 - userIndex);
            if (remainingElements > 0)
            {
                std::memmove(base + userIndex * stride, base + (userIndex + 1) * stride, remainingElements * stride);
            }
            slot->userElementCount--;
            slot->userData.resize(static_cast<SIZE_T>(slot->userElementCount) * stride);
            slot->dirty = true;
            ++m_stateRevision;
            return true;
        }

        BOOL RemoveStructuredBufferUserElement(const char* name, UINT32 userIndex, PassType pass = PassType::GEOMETRY)
        {
            if (!name || pass >= PassType::COUNT) return false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                    {
                        return RemoveStructuredBufferUserElement(pass, stage,
                                                                stageProps.structuredBuffers[i].propertyName.c_str(),
                                                                stageProps.structuredBuffers[i].resourceName.c_str(),
                                                                userIndex);
                    }
                }
            }
            return false;
        }

        BOOL RemoveStructuredBufferUserElement(const char* property, const char* resource, UINT32 userIndex, PassType pass = PassType::GEOMETRY)
        {
            if (!property || !resource || pass >= PassType::COUNT) return false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        return RemoveStructuredBufferUserElement(pass, stage, property, resource, userIndex);
                    }
                }
            }
            return false;
        }

        UINT32 GetStructuredBufferUserElementCount(PassType pass, ShaderStage stage,
                                                  const char* property, const char* resource) const
        {
            ShaderStageProps* stageProps = const_cast<MaterialPropertyStore&>(m_props).Get(pass, stage);
            if (!stageProps) return 0;
            for (SIZE_T i = 0; i < stageProps->structuredBuffers.size(); ++i)
            {
                if (stageProps->structuredBuffers[i].propertyName == property &&
                    stageProps->structuredBuffers[i].resourceName == resource)
                    return stageProps->structuredBuffers[i].userElementCount;
            }
            return 0;
        }

        UINT32 GetStructuredBufferUserElementCount(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(pass, name);
            return slot ? slot->userElementCount : 0;
        }

        UINT32 GetStructuredBufferUserElementCount(const char* property, const char* resource, PassType pass = PassType::GEOMETRY) const
        {
            if (!property || !resource || pass >= PassType::COUNT) return 0;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        return stageProps.structuredBuffers[i].userElementCount;
                    }
                }
            }
            return 0;
        }

        // ==================== PERSISTENT GPU BUFFER BINDING API ====================
        BOOL SetGPUBuffer(PassType pass, ShaderStage stage,
                          const char* property, const char* resource,
                          GPUBuffer* buffer,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(pass, stage, property, resource);
            if (!slot) return false;

            if (!buffer)
            {
                slot->gpuBuffer             = nullptr;
                slot->gpuBufferHandle       = 0;
                slot->gpuBufferElementCount = 0;
                slot->gpuBufferStride       = 0;
                slot->gpuBufferByteSize     = 0;
                slot->uavAccess             = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
                slot->hasDestroyedResource  = false;
                ++m_stateRevision;
                return true;
            }

            if (!buffer->IsValid() || buffer->GetStride() != slot->stride)
            {
                return false;
            }

            if (slot->isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV))
            {
                return false;
            }

            if (!slot->isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV))
            {
                return false;
            }

            slot->gpuBuffer             = buffer;
            slot->gpuBufferHandle       = buffer->GetNativeHandle();
            slot->gpuBufferElementCount = buffer->GetElementCount();
            slot->gpuBufferStride       = buffer->GetStride();
            slot->gpuBufferByteSize     = buffer->GetByteSize();
            slot->uavAccess             = uavAccess;
            slot->hasDestroyedResource  = false;
            ++m_stateRevision;
            return true;
        }

        BOOL SetGPUBuffer(const char* name, GPUBuffer* buffer, PassType pass = PassType::GEOMETRY,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            if (!name || pass >= PassType::COUNT) return false;
            BOOL boundAny = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                    {
                        if (SetGPUBuffer(pass, stage,
                                         stageProps.structuredBuffers[i].propertyName.c_str(),
                                         stageProps.structuredBuffers[i].resourceName.c_str(),
                                         buffer,
                                         uavAccess))
                        {
                            boundAny = true;
                        }
                    }
                }
            }
            return boundAny;
        }

        BOOL SetGPUBuffer(const char* property, const char* resource, GPUBuffer* buffer, PassType pass = PassType::GEOMETRY,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            if (!property || !resource || pass >= PassType::COUNT) return false;
            BOOL boundAny = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        if (SetGPUBuffer(pass, stage, property, resource, buffer, uavAccess))
                        {
                            boundAny = true;
                        }
                    }
                }
            }
            return boundAny;
        }

        GPUBuffer* GetGPUBuffer(const char* name, PassType pass = PassType::GEOMETRY) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(pass, name);
            return slot ? slot->gpuBuffer : nullptr;
        }

        GPUBuffer* GetGPUBuffer(const char* property, const char* resource, PassType pass = PassType::GEOMETRY) const
        {
            if (!property || !resource || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                    {
                        return stageProps.structuredBuffers[i].gpuBuffer;
                    }
                }
            }
            return nullptr;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreMaterialManager;

        Material() = default;
        Material(const Material& other) = default;
        Material& operator=(const Material& other) = default;

        void SetGPUSceneMaterialId(UINT32 id)    { m_gpuSceneMaterialId = id; ++m_stateRevision; }

        void OnTextureDestroyed(Texture* destroyedTex)
        {
            if (!destroyedTex) return;
            BOOL modified = false;
            for (UINT8 p = 0; p < static_cast<UINT8>(PassType::COUNT); ++p)
            {
                for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
                {
                    if (m_props.Get(static_cast<PassType>(p), static_cast<GraphicsShaderSlot>(s)).OnTextureDestroyed(destroyedTex))
                    {
                        modified = true;
                    }
                }
            }
            if (modified) ++m_stateRevision;
        }

        void OnSamplerDestroyed(SamplerState* destroyedSamp)
        {
            if (!destroyedSamp) return;
            BOOL modified = false;
            for (UINT8 p = 0; p < static_cast<UINT8>(PassType::COUNT); ++p)
            {
                for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
                {
                    if (m_props.Get(static_cast<PassType>(p), static_cast<GraphicsShaderSlot>(s)).OnSamplerDestroyed(destroyedSamp))
                    {
                        modified = true;
                    }
                }
            }
            if (modified) ++m_stateRevision;
        }

        void OnGPUBufferDestroyed(GPUBuffer* destroyedBuf)
        {
            if (!destroyedBuf) return;
            BOOL modified = false;
            for (UINT8 p = 0; p < static_cast<UINT8>(PassType::COUNT); ++p)
            {
                for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
                {
                    if (m_props.Get(static_cast<PassType>(p), static_cast<GraphicsShaderSlot>(s)).OnGPUBufferDestroyed(destroyedBuf))
                    {
                        modified = true;
                    }
                }
            }
            if (modified) ++m_stateRevision;
        }

        void OnShaderSetDestroyed(ShaderSet* destroyedSet)
        {
            if (!destroyedSet) return;
            BOOL modified = false;
            for (SIZE_T p = 0; p < static_cast<SIZE_T>(PassType::COUNT); ++p)
            {
                if (m_shaderSets[p] == destroyedSet)
                {
                    m_shaderSets[p] = nullptr;
                    modified = true;
                }
            }
            if (modified) ++m_stateRevision;
        }

        // -------------------- Slot Population Helper --------------------
        void PopulateStageSlots(PassType pass, GraphicsShaderSlot slotIdx, const ShaderReflection& refl)
        {
            PopulateShaderStageProps(m_props.Get(pass, slotIdx), SlotToShaderStage(slotIdx), refl);
        }

        // -------------------- Resolve helpers --------------------

        const ShaderReflection* GetStageReflection(PassType pass, ShaderStage stage) const
        {
            if (pass < PassType::COUNT && m_shaderSets[static_cast<SIZE_T>(pass)])
            {
                const ShaderSet* ss = m_shaderSets[static_cast<SIZE_T>(pass)];
                if (stage == ShaderStage::VERTEX)
                {
                    const Shader* vs = ss->GetVertexShader(ShaderVariant::STANDARD);
                    if (vs && vs->IsValid()) return &vs->GetReflection();
                    for (SIZE_T v = 0; v < static_cast<SIZE_T>(ShaderVariant::COUNT); ++v)
                    {
                        const Shader* alt = ss->GetVertexShader(static_cast<ShaderVariant>(v));
                        if (alt && alt->IsValid()) return &alt->GetReflection();
                    }
                }
                else
                {
                    const Shader* shader = ss->GetShader(stage);
                    if (shader && shader->IsValid())
                        return &shader->GetReflection();
                }
            }
            return nullptr;
        }

        const NSL::NSCBuffer* FindBufferInReflection(PassType pass, ShaderStage stage,
                                                     const ShaderBufferSlot& bSlot) const
        {
            if (pass >= PassType::COUNT || !m_shaderSets[static_cast<SIZE_T>(pass)]) return nullptr;
            const ShaderSet* ss = m_shaderSets[static_cast<SIZE_T>(pass)];

            auto searchInRefl = [&](const ShaderReflection& refl) -> const NSL::NSCBuffer* {
                const NSL::NSCBuffer* buf = refl.FindBuffer(bSlot.propertyName + "__" + bSlot.resourceName);
                if (buf) return buf;
                for (const auto& candidate : refl.buffers)
                {
                    if (candidate.name == bSlot.resourceName || candidate.name == bSlot.propertyName)
                        return &candidate;
                }
                return nullptr;
            };

            const ShaderReflection* primRefl = GetStageReflection(pass, stage);
            if (primRefl)
            {
                const NSL::NSCBuffer* buf = searchInRefl(*primRefl);
                if (buf) return buf;
            }

            if (stage == ShaderStage::VERTEX)
            {
                for (SIZE_T v = 0; v < static_cast<SIZE_T>(ShaderVariant::COUNT); ++v)
                {
                    const Shader* vs = ss->GetVertexShader(static_cast<ShaderVariant>(v));
                    if (vs && vs->IsValid())
                    {
                        const NSL::NSCBuffer* buf = searchInRefl(vs->GetReflection());
                        if (buf) return buf;
                    }
                }
            }
            return nullptr;
        }

        static const NSL::NSCField* FindAddField(const NSL::NSCBuffer& buffer, const char* field)
        {
            for (const NSL::NSCField& f : buffer.fields)
            {
                if (f.isGet) continue;
                if (FieldNameEquals(f.name, field)) return &f;
            }
            return nullptr;
        }

        BOOL HasUniqueUnqualifiedBufferField(PassType pass, const char* field,
                                             std::string& outProp, std::string& outRes) const
        {
            outProp.clear();
            outRes.clear();
            if (!field || pass >= PassType::COUNT) return false;

            BOOL found = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer || !FindAddField(*buffer, field)) continue;

                    if (!found)
                    {
                        outProp = bSlot.propertyName;
                        outRes  = bSlot.resourceName;
                        found   = true;
                    }
                    else if (bSlot.propertyName != outProp || bSlot.resourceName != outRes)
                    {
                        return false;
                    }
                }
            }
            return found;
        }

        BOOL HasUniqueGroupBufferField(PassType pass, const char* group, const char* field,
                                       std::string& outProp, std::string& outRes) const
        {
            outProp.clear();
            outRes.clear();
            if (!group || !field || pass >= PassType::COUNT) return false;

            // 1. Exact qualified match ("Property__Resource" or "Property.Resource") takes unambiguous priority
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if ((bSlot.propertyName + "__" + bSlot.resourceName) != group &&
                        (bSlot.propertyName + "." + bSlot.resourceName) != group)
                        continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (buffer && FindAddField(*buffer, field))
                    {
                        outProp = bSlot.propertyName;
                        outRes  = bSlot.resourceName;
                        return true;
                    }
                }
            }

            // 2. Partial match on propertyName or resourceName — reject if >1 distinct (propertyName, resourceName) match
            BOOL found = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.resourceName != group && bSlot.propertyName != group)
                        continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer || !FindAddField(*buffer, field)) continue;

                    if (!found)
                    {
                        outProp = bSlot.propertyName;
                        outRes  = bSlot.resourceName;
                        found   = true;
                    }
                    else if (bSlot.propertyName != outProp || bSlot.resourceName != outRes)
                    {
                        return false;
                    }
                }
            }
            return found;
        }

        // Resolves buffer field by unique field name across all stages of a pass
        BOOL ResolveBufferFieldByName(PassType pass, const char* field,
                                      ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField)
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(pass, field, targetProp, targetRes))
                return false;

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != targetProp || bSlot.resourceName != targetRes) continue;
                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField)
                    {
                        outSlot  = &bSlot;
                        outField = nscField;
                        return true;
                    }
                }
            }
            return false;
        }

        BOOL ResolveBufferFieldByNameConst(PassType pass, const char* field,
                                           const ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField) const
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(pass, field, targetProp, targetRes))
                return false;

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != targetProp || bSlot.resourceName != targetRes) continue;
                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField)
                    {
                        outSlot  = &bSlot;
                        outField = nscField;
                        return true;
                    }
                }
            }
            return false;
        }

        // Resolves buffer field with group qualification
        BOOL ResolveGroupBufferField(PassType pass, const char* group, const char* field,
                                     ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField)
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueGroupBufferField(pass, group, field, targetProp, targetRes))
                return false;

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != targetProp || bSlot.resourceName != targetRes)
                        continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField)
                    {
                        outSlot  = &bSlot;
                        outField = nscField;
                        return true;
                    }
                }
            }
            return false;
        }

        BOOL ResolveGroupBufferFieldConst(PassType pass, const char* group, const char* field,
                                          const ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField) const
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueGroupBufferField(pass, group, field, targetProp, targetRes))
                return false;

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != targetProp || bSlot.resourceName != targetRes)
                        continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField)
                    {
                        outSlot  = &bSlot;
                        outField = nscField;
                        return true;
                    }
                }
            }
            return false;
        }

        BOOL ResolveBufferField(PassType pass, ShaderStage stage,
                                const char* property, const char* resource, const char* field,
                                ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField)
        {
            outSlot  = nullptr;
            outField = nullptr;

            if (!property || !resource || !field) return false;

            ShaderBufferSlot* slot = FindBufferSlot(pass, stage, property, resource);
            if (!slot || slot->cpuData.size() == 0) return false;

            const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, *slot);
            if (!buffer) return false;

            const NSL::NSCField* nscField = FindAddField(*buffer, field);
            if (!nscField) return false;

            outSlot  = slot;
            outField = nscField;
            return true;
        }

        ShaderBufferSlot* FindBufferSlot(PassType pass, ShaderStage stage, const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->buffers.size(); ++i)
            {
                if (stageProps->buffers[i].propertyName == property &&
                    stageProps->buffers[i].resourceName == resource)
                    return &stageProps->buffers[i];
            }
            return nullptr;
        }

        const ShaderBufferSlot* FindBufferSlotConst(PassType pass, ShaderStage stage, const char* property, const char* resource) const
        {
            if (!property || !resource) return nullptr;
            const ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->buffers.size(); ++i)
            {
                if (stageProps->buffers[i].propertyName == property &&
                    stageProps->buffers[i].resourceName == resource)
                    return &stageProps->buffers[i];
            }
            return nullptr;
        }

        ShaderTextureSlot* FindTextureSlot(PassType pass, ShaderStage stage, const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->textures.size(); ++i)
            {
                if (stageProps->textures[i].propertyName == property &&
                    stageProps->textures[i].resourceName == resource)
                    return &stageProps->textures[i];
            }
            return nullptr;
        }

        const ShaderTextureSlot* FindTextureSlotConst(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.textures.size(); ++i)
                {
                    if (stageProps.textures[i].resourceName == name ||
                        stageProps.textures[i].propertyName == name ||
                        (stageProps.textures[i].propertyName + "__" + stageProps.textures[i].resourceName) == name)
                        return &stageProps.textures[i];
                }
            }
            return nullptr;
        }

        ShaderRWTextureSlot* FindRWTextureSlot(PassType pass, ShaderStage stage, const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->rwTextures.size(); ++i)
            {
                if (stageProps->rwTextures[i].propertyName == property &&
                    stageProps->rwTextures[i].resourceName == resource)
                    return &stageProps->rwTextures[i];
            }
            return nullptr;
        }

        const ShaderRWTextureSlot* FindRWTextureSlotConst(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (stageProps.rwTextures[i].resourceName == name ||
                        stageProps.rwTextures[i].propertyName == name ||
                        (stageProps.rwTextures[i].propertyName + "__" + stageProps.rwTextures[i].resourceName) == name)
                        return &stageProps.rwTextures[i];
                }
            }
            return nullptr;
        }

        ShaderSamplerSlot* FindSamplerSlot(PassType pass, ShaderStage stage, const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->samplers.size(); ++i)
            {
                if (stageProps->samplers[i].propertyName == property &&
                    stageProps->samplers[i].resourceName == resource)
                    return &stageProps->samplers[i];
            }
            return nullptr;
        }

        const ShaderSamplerSlot* FindSamplerSlotConst(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.samplers.size(); ++i)
                {
                    if (stageProps.samplers[i].resourceName == name ||
                        stageProps.samplers[i].propertyName == name ||
                        (stageProps.samplers[i].propertyName + "__" + stageProps.samplers[i].resourceName) == name)
                        return &stageProps.samplers[i];
                }
            }
            return nullptr;
        }

        ShaderStructuredBufferSlot* FindStructuredBufferSlot(PassType pass, ShaderStage stage, const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            ShaderStageProps* stageProps = m_props.Get(pass, stage);
            if (!stageProps) return nullptr;

            for (SIZE_T i = 0; i < stageProps->structuredBuffers.size(); ++i)
            {
                if (stageProps->structuredBuffers[i].propertyName == property &&
                    stageProps->structuredBuffers[i].resourceName == resource)
                    return &stageProps->structuredBuffers[i];
            }
            return nullptr;
        }

        ShaderStructuredBufferSlot* FindStructuredBufferSlot(PassType pass, const char* name)
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                        return &stageProps.structuredBuffers[i];
                }
            }
            return nullptr;
        }

        ShaderStructuredBufferSlot* FindStructuredBufferSlot(PassType pass, const char* property, const char* resource)
        {
            if (!property || !resource || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].propertyName == property &&
                        stageProps.structuredBuffers[i].resourceName == resource)
                        return &stageProps.structuredBuffers[i];
                }
            }
            return nullptr;
        }

        const ShaderStructuredBufferSlot* FindStructuredBufferSlotConst(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.structuredBuffers.size(); ++i)
                {
                    if (stageProps.structuredBuffers[i].resourceName == name ||
                        stageProps.structuredBuffers[i].propertyName == name ||
                        (stageProps.structuredBuffers[i].propertyName + "__" + stageProps.structuredBuffers[i].resourceName) == name)
                        return &stageProps.structuredBuffers[i];
                }
            }
            return nullptr;
        }

        // -------------------- Low-Level Read / Write --------------------

        void WriteQualifiedFieldByName(PassType pass, const char* property, const char* resource, const char* field,
                                       const void* src, UINT32 byteCount)
        {
            if (!property || !resource || !field || !src || byteCount == 0 || pass >= PassType::COUNT) return;

            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != property || bSlot.resourceName != resource) continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField && nscField->size >= byteCount && nscField->offset + byteCount <= bSlot.byteSize)
                    {
                        std::memcpy(bSlot.cpuData.data() + nscField->offset, src, byteCount);
                        bSlot.dirty = true;
                        modified    = true;
                    }
                }
            }

            if (modified)
            {
                ++m_stateRevision;
            }

            if (pass == PassType::GEOMETRY)
            {
                if (HasPass(PassType::SHADOW))
                {
                    WriteQualifiedFieldByName(PassType::SHADOW, property, resource, field, src, byteCount);
                }
                if (HasPass(PassType::DEPTH))
                {
                    WriteQualifiedFieldByName(PassType::DEPTH, property, resource, field, src, byteCount);
                }
            }
        }

        BOOL ReadQualifiedFieldByName(PassType pass, const char* property, const char* resource, const char* field,
                                      void* dst, UINT32 byteCount) const
        {
            if (!property || !resource || !field || !dst || byteCount == 0 || pass >= PassType::COUNT) return false;

            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    const ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != property || bSlot.resourceName != resource) continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField)
                    {
                        if (nscField->size < byteCount || nscField->offset + byteCount > bSlot.byteSize)
                            return false;
                        std::memcpy(dst, bSlot.cpuData.data() + nscField->offset, byteCount);
                        return true;
                    }
                }
            }
            return false;
        }

        void WriteQualifiedMat3ByName(PassType pass, const char* property, const char* resource, const char* field, const Mat3& value)
        {
            if (!property || !resource || !field || pass >= PassType::COUNT) return;

            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                {
                    ShaderBufferSlot& bSlot = stageProps.buffers[b];
                    if (bSlot.propertyName != property || bSlot.resourceName != resource) continue;

                    const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                    if (!buffer) continue;

                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (nscField && nscField->offset + nscField->size <= bSlot.byteSize)
                    {
                        UINT32 colStride = nscField->size / 3;
                        for (UINT32 col = 0; col < 3; ++col)
                        {
                            FLOAT32 packed[4] = { value.m[col * 3 + 0], value.m[col * 3 + 1], value.m[col * 3 + 2], 0.0f };
                            std::memcpy(bSlot.cpuData.data() + nscField->offset + col * colStride, packed, colStride);
                        }
                        bSlot.dirty = true;
                        modified    = true;
                    }
                }
            }

            if (modified)
            {
                ++m_stateRevision;
            }

            if (pass == PassType::GEOMETRY)
            {
                if (HasPass(PassType::SHADOW))
                {
                    WriteQualifiedMat3ByName(PassType::SHADOW, property, resource, field, value);
                }
                if (HasPass(PassType::DEPTH))
                {
                    WriteQualifiedMat3ByName(PassType::DEPTH, property, resource, field, value);
                }
            }
        }

        void WriteFieldByName(PassType pass, const char* field, const void* src, UINT32 byteCount)
        {
            if (!field || !src || byteCount == 0 || pass >= PassType::COUNT) return;

            std::string targetProp, targetRes;
            if (HasUniqueUnqualifiedBufferField(pass, field, targetProp, targetRes))
            {
                BOOL modified = false;
                for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
                {
                    ShaderStage stage = SlotToShaderStage(static_cast<GraphicsShaderSlot>(s));
                    ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                    for (SIZE_T b = 0; b < stageProps.buffers.size(); ++b)
                    {
                        ShaderBufferSlot& bSlot = stageProps.buffers[b];
                        if (bSlot.propertyName != targetProp || bSlot.resourceName != targetRes) continue;
                        const NSL::NSCBuffer* buffer = FindBufferInReflection(pass, stage, bSlot);
                        if (!buffer) continue;

                        const NSL::NSCField* nscField = FindAddField(*buffer, field);
                        if (nscField && nscField->size >= byteCount && nscField->offset + byteCount <= bSlot.byteSize)
                        {
                            std::memcpy(bSlot.cpuData.data() + nscField->offset, src, byteCount);
                            bSlot.dirty = true;
                            modified    = true;
                        }
                    }
                }

                if (modified)
                {
                    ++m_stateRevision;
                }
            }

            if (pass == PassType::GEOMETRY)
            {
                if (HasPass(PassType::SHADOW))
                {
                    WriteFieldByName(PassType::SHADOW, field, src, byteCount);
                }
                if (HasPass(PassType::DEPTH))
                {
                    WriteFieldByName(PassType::DEPTH, field, src, byteCount);
                }
            }
        }

        BOOL ReadFieldByName(PassType pass, const char* field, void* dst, UINT32 byteCount) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferFieldByNameConst(pass, field, slot, nscField))
                return false;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return false;

            std::memcpy(dst, slot->cpuData.data() + nscField->offset, byteCount);
            return true;
        }

        void WriteGroupFieldByName(PassType pass, const char* group, const char* field, const void* src, UINT32 byteCount)
        {
            if (!group || !field || !src || byteCount == 0 || pass >= PassType::COUNT) return;

            std::string targetProp, targetRes;
            if (HasUniqueGroupBufferField(pass, group, field, targetProp, targetRes))
            {
                WriteQualifiedFieldByName(pass, targetProp.c_str(), targetRes.c_str(), field, src, byteCount);
                return;
            }

            if (pass == PassType::GEOMETRY)
            {
                if (HasPass(PassType::SHADOW))
                {
                    WriteGroupFieldByName(PassType::SHADOW, group, field, src, byteCount);
                }
                if (HasPass(PassType::DEPTH))
                {
                    WriteGroupFieldByName(PassType::DEPTH, group, field, src, byteCount);
                }
            }
        }

        BOOL ReadGroupFieldByName(PassType pass, const char* group, const char* field, void* dst, UINT32 byteCount) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveGroupBufferFieldConst(pass, group, field, slot, nscField))
                return false;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return false;

            std::memcpy(dst, slot->cpuData.data() + nscField->offset, byteCount);
            return true;
        }

        void WriteBufferField(PassType pass, ShaderStage stage,
                              const char* property, const char* resource, const char* field,
                              const void* src, UINT32 byteCount)
        {
            ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferField(pass, stage, property, resource, field, slot, nscField))
                return;
            
            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return;

            std::memcpy(slot->cpuData.data() + nscField->offset, src, byteCount);
            slot->dirty = true;
            ++m_stateRevision;
        }

        void WriteMat3ByName(PassType pass, const char* field, const Mat3& value)
        {
            if (!field || pass >= PassType::COUNT) return;

            std::string targetProp, targetRes;
            if (HasUniqueUnqualifiedBufferField(pass, field, targetProp, targetRes))
            {
                WriteQualifiedMat3ByName(pass, targetProp.c_str(), targetRes.c_str(), field, value);
                return;
            }

            if (pass == PassType::GEOMETRY)
            {
                if (HasPass(PassType::SHADOW))
                {
                    WriteMat3ByName(PassType::SHADOW, field, value);
                }
                if (HasPass(PassType::DEPTH))
                {
                    WriteMat3ByName(PassType::DEPTH, field, value);
                }
            }
        }

        void WriteMat3Field(PassType pass, ShaderStage stage,
                            const char* property, const char* resource, const char* field,
                            const Mat3& value)
        {
            ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferField(pass, stage, property, resource, field, slot, nscField))
                return;

            if (nscField->offset + nscField->size > slot->byteSize) return;

            UINT32 colStride = nscField->size / 3;
            for (UINT32 col = 0; col < 3; ++col)
            {
                FLOAT32 packed[4] = { value.m[col * 3 + 0], value.m[col * 3 + 1], value.m[col * 3 + 2], 0.0f };
                std::memcpy(slot->cpuData.data() + nscField->offset + col * colStride, packed, colStride);
            }
            slot->dirty = true;
            ++m_stateRevision;
        }

        static BOOL MatchesGroupQualifier(const std::string& propName, const std::string& grpName, const char* group)
        {
            return propName == group ||
                   grpName == group ||
                   (propName + "__" + grpName) == group ||
                   (propName + "." + grpName) == group;
        }

        static BOOL MatchesUnqualifiedOrQualifiedSlot(
            const std::string& propName,
            const std::string& grpName,
            const std::string& resName,
            const char* name)
        {
            return resName == name ||
                   propName == name ||
                   grpName == name ||
                   (propName + "__" + resName) == name ||
                   (grpName + "__" + resName) == name ||
                   (propName + "__" + grpName + "__" + resName) == name;
        }

        void SetTextureByName(PassType pass, const char* name, Texture* texture)
        {
            if (!name || pass >= PassType::COUNT) return;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.textures.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.textures[i].propertyName,
                            stageProps.textures[i].groupName,
                            stageProps.textures[i].resourceName,
                            name))
                    {
                        stageProps.textures[i].texture              = texture;
                        stageProps.textures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.textures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.rwTextures[i].propertyName,
                            stageProps.rwTextures[i].groupName,
                            stageProps.rwTextures[i].resourceName,
                            name))
                    {
                        stageProps.rwTextures[i].texture              = texture;
                        stageProps.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.rwTextures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetTextureByName(PassType::DEPTH, name, texture);
            }
        }

        void SetTextureByGroupName(PassType pass, const char* group, const char* name, Texture* texture)
        {
            if (!group || !name || pass >= PassType::COUNT) return;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.textures.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.textures[i].propertyName, stageProps.textures[i].groupName, group) &&
                        stageProps.textures[i].resourceName == name)
                    {
                        stageProps.textures[i].texture              = texture;
                        stageProps.textures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.textures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.rwTextures[i].propertyName, stageProps.rwTextures[i].groupName, group) &&
                        stageProps.rwTextures[i].resourceName == name)
                    {
                        stageProps.rwTextures[i].texture              = texture;
                        stageProps.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.rwTextures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetTextureByGroupName(PassType::DEPTH, group, name, texture);
            }
        }

        BOOL SetRWTextureByName(PassType pass, const char* name, Texture* texture,
                                UINT32 mipSlice, GPUResourceAccess uavAccess)
        {
            if (!name || pass >= PassType::COUNT) return false;
            if (texture && (texture->GetUsage() != TextureUsage::STORAGE ||
                            (mipSlice != 0xFFFFFFFFu && mipSlice >= texture->GetMipLevels()))) return false;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.rwTextures[i].propertyName,
                            stageProps.rwTextures[i].groupName,
                            stageProps.rwTextures[i].resourceName,
                            name))
                    {
                        stageProps.rwTextures[i].texture              = texture;
                        stageProps.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.rwTextures[i].mipSlice             = mipSlice;
                        stageProps.rwTextures[i].uavAccess            = uavAccess;
                        stageProps.rwTextures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetRWTextureByName(PassType::DEPTH, name, texture, mipSlice, uavAccess);
            }
            return modified;
        }

        BOOL SetRWTextureByGroupName(PassType pass, const char* group, const char* name, Texture* texture,
                                     UINT32 mipSlice, GPUResourceAccess uavAccess)
        {
            if (!group || !name || pass >= PassType::COUNT) return false;
            if (texture && (texture->GetUsage() != TextureUsage::STORAGE ||
                            (mipSlice != 0xFFFFFFFFu && mipSlice >= texture->GetMipLevels()))) return false;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.rwTextures[i].propertyName, stageProps.rwTextures[i].groupName, group) &&
                        stageProps.rwTextures[i].resourceName == name)
                    {
                        stageProps.rwTextures[i].texture              = texture;
                        stageProps.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                        stageProps.rwTextures[i].mipSlice             = mipSlice;
                        stageProps.rwTextures[i].uavAccess            = uavAccess;
                        stageProps.rwTextures[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetRWTextureByGroupName(PassType::DEPTH, group, name, texture, mipSlice, uavAccess);
            }
            return modified;
        }

        Texture* GetTextureByName(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.textures.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.textures[i].propertyName,
                            stageProps.textures[i].groupName,
                            stageProps.textures[i].resourceName,
                            name))
                    {
                        return stageProps.textures[i].texture;
                    }
                }
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.rwTextures[i].propertyName,
                            stageProps.rwTextures[i].groupName,
                            stageProps.rwTextures[i].resourceName,
                            name))
                    {
                        return stageProps.rwTextures[i].texture;
                    }
                }
            }
            return nullptr;
        }

        Texture* GetTextureByGroupName(PassType pass, const char* group, const char* name) const
        {
            if (!group || !name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.textures.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.textures[i].propertyName, stageProps.textures[i].groupName, group) &&
                        stageProps.textures[i].resourceName == name)
                    {
                        return stageProps.textures[i].texture;
                    }
                }
                for (SIZE_T i = 0; i < stageProps.rwTextures.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.rwTextures[i].propertyName, stageProps.rwTextures[i].groupName, group) &&
                        stageProps.rwTextures[i].resourceName == name)
                    {
                        return stageProps.rwTextures[i].texture;
                    }
                }
            }
            return nullptr;
        }

        void SetSamplerByName(PassType pass, const char* name, SamplerState* sampler)
        {
            if (!name || pass >= PassType::COUNT) return;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.samplers.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.samplers[i].propertyName,
                            stageProps.samplers[i].groupName,
                            stageProps.samplers[i].resourceName,
                            name))
                    {
                        stageProps.samplers[i].sampler              = sampler;
                        stageProps.samplers[i].samplerHandle        = sampler ? sampler->GetNativeHandle() : 0;
                        stageProps.samplers[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetSamplerByName(PassType::DEPTH, name, sampler);
            }
        }

        void SetSamplerByGroupName(PassType pass, const char* group, const char* name, SamplerState* sampler)
        {
            if (!group || !name || pass >= PassType::COUNT) return;
            BOOL modified = false;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.samplers.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.samplers[i].propertyName, stageProps.samplers[i].groupName, group) &&
                        stageProps.samplers[i].resourceName == name)
                    {
                        stageProps.samplers[i].sampler              = sampler;
                        stageProps.samplers[i].samplerHandle        = sampler ? sampler->GetNativeHandle() : 0;
                        stageProps.samplers[i].hasDestroyedResource = false;
                        modified = true;
                    }
                }
            }
            if (modified)
            {
                ++m_stateRevision;
            }
            if (pass == PassType::GEOMETRY && HasPass(PassType::DEPTH))
            {
                SetSamplerByGroupName(PassType::DEPTH, group, name, sampler);
            }
        }

        SamplerState* GetSamplerByName(PassType pass, const char* name) const
        {
            if (!name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.samplers.size(); ++i)
                {
                    if (MatchesUnqualifiedOrQualifiedSlot(
                            stageProps.samplers[i].propertyName,
                            stageProps.samplers[i].groupName,
                            stageProps.samplers[i].resourceName,
                            name))
                    {
                        return stageProps.samplers[i].sampler;
                    }
                }
            }
            return nullptr;
        }

        SamplerState* GetSamplerByGroupName(PassType pass, const char* group, const char* name) const
        {
            if (!group || !name || pass >= PassType::COUNT) return nullptr;
            for (UINT8 s = 0; s < static_cast<UINT8>(GraphicsShaderSlot::COUNT); ++s)
            {
                const ShaderStageProps& stageProps = m_props.Get(pass, static_cast<GraphicsShaderSlot>(s));
                for (SIZE_T i = 0; i < stageProps.samplers.size(); ++i)
                {
                    if (MatchesGroupQualifier(stageProps.samplers[i].propertyName, stageProps.samplers[i].groupName, group) &&
                        stageProps.samplers[i].resourceName == name)
                    {
                        return stageProps.samplers[i].sampler;
                    }
                }
            }
            return nullptr;
        }

        // ==================== MEMBERS ====================
        ShaderSet* m_shaderSets[static_cast<SIZE_T>(PassType::COUNT)] = {};
        std::vector<ComputeKernel*> m_computeKernels;

        CullMode     m_cullMode    = CullMode::BACK;
        BlendMode    m_blendMode      = BlendMode::OPAQUE;
        FillMode     m_fillMode       = FillMode::SOLID;
        DepthState   m_depthState;
        UINT32            m_gpuSceneMaterialId = 0;
        UINT32            m_stateRevision      = 1;
        NVGMaterialPolicy m_nvgPolicy{};

        MaterialPropertyStore m_props;
    };
}
