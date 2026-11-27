#pragma once

#include "compute_kernel_desc.h"
#include "../shader/shader_property_store.h"
#include "../buffer/gpu_indirect_args.h"
#include "../shader/shader_types.h"
#include "../shader/shader.h"
#include "../texture/texture.h"
#include "../sampler/sampler.h"
#include "../types/northstar_types.h"
#include "../types/northstar_math.h"

#include <cstring>
#include <string>
#include <algorithm>
#include <cctype>

namespace Northstar
{
    class ComputeKernel
    {
    public:
        ~ComputeKernel() = default;

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

        // ==================== INITIALIZATION & SHADER MANAGEMENT ====================

        void Init(Shader* computeShader)
        {
            m_props.Clear();
            m_computeShader            = nullptr;
            m_threadGroupSizeX         = 0;
            m_threadGroupSizeY         = 0;
            m_threadGroupSizeZ         = 0;
            m_hasValidThreadGroupSize  = false;
            m_indirectArgsBuffer       = nullptr;
            m_indirectArgsBufferHandle = 0;
            m_indirectArgsOffset       = 0;
            m_useRootConstants         = false;
            m_hasDestroyedShader       = false;
            m_hasDestroyedIndirectBuffer = false;
            std::memset(m_rootConstants, 0, sizeof(m_rootConstants));

            if (!computeShader || computeShader->GetStage() != ShaderStage::COMPUTE)
                return;

            m_computeShader = computeShader;
            if (computeShader->IsValid())
            {
                const ShaderReflection& refl = computeShader->GetReflection();
                PopulateShaderStageProps(m_props, ShaderStage::COMPUTE, refl);

                // Primary: API-agnostic binary .nsc reflection metadata ([numthreads(X,Y,Z)])
                if (refl.HasValidNumThreads())
                {
                    m_threadGroupSizeX        = refl.numThreadsX;
                    m_threadGroupSizeY        = refl.numThreadsY;
                    m_threadGroupSizeZ        = refl.numThreadsZ;
                    m_hasValidThreadGroupSize = true;
                }
                else if (TryParseNumThreadsFromSource(refl.sourceCode, m_threadGroupSizeX, m_threadGroupSizeY, m_threadGroupSizeZ))
                {
                    m_hasValidThreadGroupSize = true;
                }
            }
        }

        void SetComputeShader(Shader* computeShader)
        {
            Init(computeShader);
        }

        Shader* GetComputeShader() const
        {
            return m_computeShader;
        }

        BOOL IsValid() const
        {
            return !m_hasDestroyedShader &&
                   !m_hasDestroyedIndirectBuffer &&
                   !m_props.HasDestroyedResource() &&
                   m_computeShader != nullptr &&
                   m_computeShader->IsValid() &&
                   m_computeShader->GetStage() == ShaderStage::COMPUTE;
        }

        BOOL HasDestroyedResource() const
        {
            return m_hasDestroyedShader || m_hasDestroyedIndirectBuffer || m_props.HasDestroyedResource();
        }

        void SetName(const std::string& name) { m_name = name; }
        const std::string& GetName() const    { return m_name; }

        void SetExecutionScope(ComputeExecutionScope scope) { m_executionScope = scope; }
        ComputeExecutionScope GetExecutionScope() const     { return m_executionScope; }

        const ShaderStageProps& GetStageProps() const { return m_props; }
        ShaderStageProps&       GetStageProps()       { return m_props; }

        // ==================== DISPATCH & THREAD GROUP CONFIGURATION ====================

        BOOL   HasValidThreadGroupSize() const { return m_hasValidThreadGroupSize; }
        UINT32 GetThreadGroupSizeX() const { return m_threadGroupSizeX; }
        UINT32 GetThreadGroupSizeY() const { return m_threadGroupSizeY; }
        UINT32 GetThreadGroupSizeZ() const { return m_threadGroupSizeZ; }

        UINT32 GetGroupThreadCountX() const { return m_threadGroupSizeX; }
        UINT32 GetGroupThreadCountY() const { return m_threadGroupSizeY; }
        UINT32 GetGroupThreadCountZ() const { return m_threadGroupSizeZ; }

        void SetThreadGroupCount(UINT32 groupX, UINT32 groupY = 1, UINT32 groupZ = 1)
        {
            m_threadGroupCountX          = groupX;
            m_threadGroupCountY          = groupY;
            m_threadGroupCountZ          = groupZ;
            m_indirectArgsBuffer         = nullptr;
            m_indirectArgsBufferHandle   = 0;
            m_indirectArgsOffset         = 0;
            m_hasDestroyedIndirectBuffer = false;
        }

        // Automatically computes ceiling(threads / numthreads) using the shader's [numthreads(X, Y, Z)] reflection.
        // Returns false and sets group counts to (0, 0, 0) if thread-group dimensions cannot be reliably determined
        // or if any requested thread count is 0. Never silently divides by a guessed (1, 1, 1) fallback.
        BOOL SetDispatchThreadCount(UINT32 threadsX, UINT32 threadsY = 1, UINT32 threadsZ = 1)
        {
            m_indirectArgsBuffer         = nullptr;
            m_indirectArgsBufferHandle   = 0;
            m_indirectArgsOffset         = 0;
            m_hasDestroyedIndirectBuffer = false;

            if (!m_hasValidThreadGroupSize ||
                m_threadGroupSizeX == 0 || m_threadGroupSizeY == 0 || m_threadGroupSizeZ == 0 ||
                threadsX == 0 || threadsY == 0 || threadsZ == 0)
            {
                m_threadGroupCountX = 0;
                m_threadGroupCountY = 0;
                m_threadGroupCountZ = 0;
                return false;
            }

            m_threadGroupCountX = (threadsX + m_threadGroupSizeX - 1) / m_threadGroupSizeX;
            m_threadGroupCountY = (threadsY + m_threadGroupSizeY - 1) / m_threadGroupSizeY;
            m_threadGroupCountZ = (threadsZ + m_threadGroupSizeZ - 1) / m_threadGroupSizeZ;
            return true;
        }

        UINT32 GetThreadGroupCountX() const { return m_threadGroupCountX; }
        UINT32 GetThreadGroupCountY() const { return m_threadGroupCountY; }
        UINT32 GetThreadGroupCountZ() const { return m_threadGroupCountZ; }

        // Configures GPU-driven DispatchIndirect using a GPUBuffer with GPUBufferUsage::INDIRECT_ARGUMENT.
        // Pass nullptr to switch back to direct Dispatch(X, Y, Z).
        BOOL SetIndirectArgsBuffer(GPUBuffer* buffer, UINT32 byteOffset = 0)
        {
            if (!buffer)
            {
                m_indirectArgsBuffer         = nullptr;
                m_indirectArgsBufferHandle   = 0;
                m_indirectArgsOffset         = 0;
                m_hasDestroyedIndirectBuffer = false;
                return true;
            }

            if (!buffer->IsValid() || !buffer->HasUsage(GPUBufferUsage::INDIRECT_ARGUMENT))
                return false;

            if ((byteOffset % 4u) != 0u)
                return false;

            if (static_cast<UINT64>(byteOffset) + sizeof(GPUIndirectDispatchArgs) > static_cast<UINT64>(buffer->GetByteSize()))
                return false;

            m_indirectArgsBuffer         = buffer;
            m_indirectArgsBufferHandle   = buffer->GetNativeHandle();
            m_indirectArgsOffset         = byteOffset;
            m_hasDestroyedIndirectBuffer = false;
            return true;
        }

        GPUBuffer*   GetIndirectArgsBuffer()       const { return m_indirectArgsBuffer; }
        NativeHandle GetIndirectArgsBufferHandle() const { return m_indirectArgsBufferHandle; }
        UINT32       GetIndirectArgsOffset()       const { return m_indirectArgsOffset; }
        BOOL         IsIndirectDispatch()          const { return m_indirectArgsBufferHandle != 0; }

        // ==================== ROOT CONSTANTS (b0, space4, 4 DWORDs) ====================

        void SetRootConstants(UINT32 c0, UINT32 c1 = 0, UINT32 c2 = 0, UINT32 c3 = 0)
        {
            m_rootConstants[0] = c0;
            m_rootConstants[1] = c1;
            m_rootConstants[2] = c2;
            m_rootConstants[3] = c3;
            m_useRootConstants = true;
        }

        void ClearRootConstants()
        {
            std::memset(m_rootConstants, 0, sizeof(m_rootConstants));
            m_useRootConstants = false;
        }

        BOOL          UsesRootConstants() const { return m_useRootConstants; }
        const UINT32* GetRootConstants()  const { return m_rootConstants; }

        // =====================================================================
        //  INTROSPECTION / QUERYING
        // =====================================================================

        BOOL HasProperty(const char* name) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* field = nullptr;
            return ResolveBufferFieldByNameConst(name, slot, field);
        }

        BOOL HasProperty(const char* group, const char* name) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* field = nullptr;
            return ResolveGroupBufferFieldConst(group, name, slot, field);
        }

        BOOL HasBuffer(const char* name) const
        {
            if (!name) return false;
            for (const auto& b : m_props.buffers)
            {
                if (b.resourceName == name || b.propertyName == name || (b.propertyName + "__" + b.resourceName == name))
                    return true;
            }
            return false;
        }

        BOOL HasTexture(const char* name) const
        {
            return FindTextureSlotConst(name) != nullptr || FindRWTextureSlotConst(name) != nullptr;
        }

        BOOL HasRWTexture(const char* name) const
        {
            return FindRWTextureSlotConst(name) != nullptr;
        }

        BOOL HasSampler(const char* name) const
        {
            return FindSamplerSlotConst(name) != nullptr;
        }

        BOOL HasStructuredBuffer(const char* name) const
        {
            return FindStructuredBufferSlotConst(name) != nullptr;
        }

        BOOL HasRWStructuredBuffer(const char* name) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(name);
            return slot && slot->isUAV;
        }

        // =====================================================================
        //  DIRECT PROPERTY SETTERS & GETTERS
        // =====================================================================

        void SetFloat (const char* name, FLOAT32 value)           { WriteFieldByName(name, &value, sizeof(value)); }
        void SetInt   (const char* name, INT32 value)             { WriteFieldByName(name, &value, sizeof(value)); }
        void SetUInt  (const char* name, UINT32 value)            { WriteFieldByName(name, &value, sizeof(value)); }
        void SetBool  (const char* name, BOOL value)              { const INT32 i = value ? 1 : 0; WriteFieldByName(name, &i, sizeof(i)); }
        void SetVec2  (const char* name, const Vec2& value)       { WriteFieldByName(name, &value, sizeof(value)); }
        void SetVec3  (const char* name, const Vec3& value)       { WriteFieldByName(name, &value, sizeof(value)); }
        void SetVec4  (const char* name, const Vec4& value)       { WriteFieldByName(name, &value, sizeof(value)); }
        void SetColor (const char* name, const Color& value)      { WriteFieldByName(name, &value, sizeof(value)); }
        void SetMat3  (const char* name, const Mat3& value)       { WriteMat3ByName(name, value); }
        void SetMat4  (const char* name, const Mat4& value)       { WriteFieldByName(name, &value, sizeof(value)); }
        void SetStruct(const char* name, const void* data, UINT32 byteSize) { WriteFieldByName(name, data, byteSize); }
        void SetStruct(const char* group, const char* field, const void* data, UINT32 byteSize) { WriteGroupFieldByName(group, field, data, byteSize); }

        BOOL GetFloat (const char* name, FLOAT32& out) const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetInt   (const char* name, INT32& out)   const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetUInt  (const char* name, UINT32& out)  const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetBool  (const char* name, BOOL& out)    const
        {
            INT32 i = 0;
            if (!ReadFieldByName(name, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* name, Vec2& out)    const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetVec3  (const char* name, Vec3& out)    const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetVec4  (const char* name, Vec4& out)    const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetColor (const char* name, Color& out)   const { return ReadFieldByName(name, &out, sizeof(out)); }
        BOOL GetMat4  (const char* name, Mat4& out)    const { return ReadFieldByName(name, &out, sizeof(out)); }

        FLOAT32 GetFloatOrDefault(const char* name, FLOAT32 defaultVal = 0.0f) const
        {
            FLOAT32 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        INT32 GetIntOrDefault(const char* name, INT32 defaultVal = 0) const
        {
            INT32 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        UINT32 GetUIntOrDefault(const char* name, UINT32 defaultVal = 0) const
        {
            UINT32 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        BOOL GetBoolOrDefault(const char* name, BOOL defaultVal = false) const
        {
            INT32 i = defaultVal ? 1 : 0;
            ReadFieldByName(name, &i, sizeof(i));
            return (i != 0);
        }

        Vec2 GetVec2OrDefault(const char* name, const Vec2& defaultVal = Vec2(0,0)) const
        {
            Vec2 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        Vec3 GetVec3OrDefault(const char* name, const Vec3& defaultVal = Vec3(0,0,0)) const
        {
            Vec3 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        Vec4 GetVec4OrDefault(const char* name, const Vec4& defaultVal = Vec4(0,0,0,0)) const
        {
            Vec4 val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        Color GetColorOrDefault(const char* name, const Color& defaultVal = Color(1,1,1,1)) const
        {
            Color val = defaultVal;
            ReadFieldByName(name, &val, sizeof(val));
            return val;
        }

        // =====================================================================
        //  GROUP-QUALIFIED SETTERS & GETTERS
        // =====================================================================

        void SetFloat (const char* group, const char* field, FLOAT32 value)      { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetInt   (const char* group, const char* field, INT32 value)        { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetUInt  (const char* group, const char* field, UINT32 value)       { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetBool  (const char* group, const char* field, BOOL value)         { const INT32 i = value ? 1 : 0; WriteGroupFieldByName(group, field, &i, sizeof(i)); }
        void SetVec2  (const char* group, const char* field, const Vec2& value)  { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetVec3  (const char* group, const char* field, const Vec3& value)  { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetVec4  (const char* group, const char* field, const Vec4& value)  { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetColor (const char* group, const char* field, const Color& value) { WriteGroupFieldByName(group, field, &value, sizeof(value)); }
        void SetMat3  (const char* group, const char* field, const Mat3& value)  { WriteGroupMat3ByName(group, field, value); }
        void SetMat4  (const char* group, const char* field, const Mat4& value)  { WriteGroupFieldByName(group, field, &value, sizeof(value)); }

        void SetFloat (const char* property, const char* resource, const char* field, FLOAT32 value)      { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetInt   (const char* property, const char* resource, const char* field, INT32 value)        { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetUInt  (const char* property, const char* resource, const char* field, UINT32 value)       { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetBool  (const char* property, const char* resource, const char* field, BOOL value)         { const INT32 i = value ? 1 : 0; WriteBufferField(property, resource, field, &i, sizeof(i)); }
        void SetVec2  (const char* property, const char* resource, const char* field, const Vec2& value)  { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetVec3  (const char* property, const char* resource, const char* field, const Vec3& value)  { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetVec4  (const char* property, const char* resource, const char* field, const Vec4& value)  { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetColor (const char* property, const char* resource, const char* field, const Color& value) { WriteBufferField(property, resource, field, &value, sizeof(value)); }
        void SetMat3  (const char* property, const char* resource, const char* field, const Mat3& value)  { WriteMat3Field(property, resource, field, value); }
        void SetMat4  (const char* property, const char* resource, const char* field, const Mat4& value)  { WriteBufferField(property, resource, field, &value, sizeof(value)); }

        BOOL GetFloat (const char* group, const char* field, FLOAT32& out) const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetInt   (const char* group, const char* field, INT32& out)   const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetUInt  (const char* group, const char* field, UINT32& out)  const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetBool  (const char* group, const char* field, BOOL& out)    const
        {
            INT32 i = 0;
            if (!ReadGroupFieldByName(group, field, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* group, const char* field, Vec2& out)    const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetVec3  (const char* group, const char* field, Vec3& out)    const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetVec4  (const char* group, const char* field, Vec4& out)    const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetColor (const char* group, const char* field, Color& out)   const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }
        BOOL GetMat4  (const char* group, const char* field, Mat4& out)    const { return ReadGroupFieldByName(group, field, &out, sizeof(out)); }

        BOOL GetFloat (const char* property, const char* resource, const char* field, FLOAT32& out) const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetInt   (const char* property, const char* resource, const char* field, INT32& out)   const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetUInt  (const char* property, const char* resource, const char* field, UINT32& out)  const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetBool  (const char* property, const char* resource, const char* field, BOOL& out)    const
        {
            INT32 i = 0;
            if (!ReadBufferField(property, resource, field, &i, sizeof(i))) return false;
            out = (i != 0);
            return true;
        }
        BOOL GetVec2  (const char* property, const char* resource, const char* field, Vec2& out)    const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetVec3  (const char* property, const char* resource, const char* field, Vec3& out)    const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetVec4  (const char* property, const char* resource, const char* field, Vec4& out)    const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetColor (const char* property, const char* resource, const char* field, Color& out)   const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }
        BOOL GetMat4  (const char* property, const char* resource, const char* field, Mat4& out)    const { return ReadBufferField(property, resource, field, &out, sizeof(out)); }

        // =====================================================================
        //  TEXTURE, RWTEXTURE & SAMPLER BINDING API
        // =====================================================================

        BOOL SetTexture(const char* name, Texture* texture, UINT32 mipSlice = 0xFFFFFFFFu)
        {
            if (!name) return false;
            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.textures.size(); ++i)
            {
                if (MatchesTextureSlot(m_props.textures[i], name))
                {
                    m_props.textures[i].texture              = texture;
                    m_props.textures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                    m_props.textures[i].mipSlice             = mipSlice;
                    m_props.textures[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            if (!matched)
            {
                return SetRWTexture(name, texture, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE, mipSlice == 0xFFFFFFFFu ? 0u : mipSlice);
            }
            return matched;
        }

        BOOL SetTexture(const char* group, const char* name, Texture* texture, UINT32 mipSlice = 0xFFFFFFFFu)
        {
            if (!group || !name) return false;
            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.textures.size(); ++i)
            {
                if (MatchesGroupTextureSlot(m_props.textures[i], group, name))
                {
                    m_props.textures[i].texture              = texture;
                    m_props.textures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                    m_props.textures[i].mipSlice             = mipSlice;
                    m_props.textures[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            if (!matched)
            {
                return SetRWTexture(group, name, texture, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE, mipSlice == 0xFFFFFFFFu ? 0u : mipSlice);
            }
            return matched;
        }

        Texture* GetTexture(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.textures.size(); ++i)
            {
                if (MatchesTextureSlot(m_props.textures[i], name))
                    return m_props.textures[i].texture;
            }
            return GetRWTexture(name);
        }

        Texture* GetTexture(const char* group, const char* name) const
        {
            if (!group || !name) return nullptr;
            for (SIZE_T i = 0; i < m_props.textures.size(); ++i)
            {
                if (MatchesGroupTextureSlot(m_props.textures[i], group, name))
                    return m_props.textures[i].texture;
            }
            return GetRWTexture(group, name);
        }

        BOOL SetRWTexture(const char* name, Texture* texture,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE,
                          UINT32 mipSlice = 0u)
        {
            if (!name) return false;
            if (texture && (texture->GetUsage() != TextureUsage::STORAGE ||
                            (mipSlice != 0xFFFFFFFFu && mipSlice >= texture->GetMipLevels())))
                return false;

            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.rwTextures.size(); ++i)
            {
                if (MatchesRWTextureSlot(m_props.rwTextures[i], name))
                {
                    m_props.rwTextures[i].texture              = texture;
                    m_props.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                    m_props.rwTextures[i].mipSlice             = mipSlice;
                    m_props.rwTextures[i].uavAccess            = uavAccess;
                    m_props.rwTextures[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            return matched;
        }

        BOOL SetRWTexture(const char* group, const char* name, Texture* texture,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE,
                          UINT32 mipSlice = 0u)
        {
            if (!group || !name) return false;
            if (texture && (texture->GetUsage() != TextureUsage::STORAGE ||
                            (mipSlice != 0xFFFFFFFFu && mipSlice >= texture->GetMipLevels())))
                return false;

            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.rwTextures.size(); ++i)
            {
                if (MatchesGroupRWTextureSlot(m_props.rwTextures[i], group, name))
                {
                    m_props.rwTextures[i].texture              = texture;
                    m_props.rwTextures[i].textureHandle        = texture ? texture->GetNativeHandle() : 0;
                    m_props.rwTextures[i].mipSlice             = mipSlice;
                    m_props.rwTextures[i].uavAccess            = uavAccess;
                    m_props.rwTextures[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            return matched;
        }

        Texture* GetRWTexture(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.rwTextures.size(); ++i)
            {
                if (MatchesRWTextureSlot(m_props.rwTextures[i], name))
                    return m_props.rwTextures[i].texture;
            }
            return nullptr;
        }

        Texture* GetRWTexture(const char* group, const char* name) const
        {
            if (!group || !name) return nullptr;
            for (SIZE_T i = 0; i < m_props.rwTextures.size(); ++i)
            {
                if (MatchesGroupRWTextureSlot(m_props.rwTextures[i], group, name))
                    return m_props.rwTextures[i].texture;
            }
            return nullptr;
        }

        BOOL SetSampler(const char* name, SamplerState* sampler)
        {
            if (!name) return false;
            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.samplers.size(); ++i)
            {
                if (MatchesSamplerSlot(m_props.samplers[i], name))
                {
                    m_props.samplers[i].sampler              = sampler;
                    m_props.samplers[i].samplerHandle        = sampler ? sampler->GetNativeHandle() : 0;
                    m_props.samplers[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            return matched;
        }

        BOOL SetSampler(const char* group, const char* name, SamplerState* sampler)
        {
            if (!group || !name) return false;
            BOOL matched = false;
            for (SIZE_T i = 0; i < m_props.samplers.size(); ++i)
            {
                if (MatchesGroupSamplerSlot(m_props.samplers[i], group, name))
                {
                    m_props.samplers[i].sampler              = sampler;
                    m_props.samplers[i].samplerHandle        = sampler ? sampler->GetNativeHandle() : 0;
                    m_props.samplers[i].hasDestroyedResource = false;
                    matched = true;
                }
            }
            return matched;
        }

        SamplerState* GetSampler(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.samplers.size(); ++i)
            {
                if (MatchesSamplerSlot(m_props.samplers[i], name))
                    return m_props.samplers[i].sampler;
            }
            return nullptr;
        }

        SamplerState* GetSampler(const char* group, const char* name) const
        {
            if (!group || !name) return nullptr;
            for (SIZE_T i = 0; i < m_props.samplers.size(); ++i)
            {
                if (MatchesGroupSamplerSlot(m_props.samplers[i], group, name))
                    return m_props.samplers[i].sampler;
            }
            return nullptr;
        }

        // =====================================================================
        //  STRUCTURED BUFFER & PERSISTENT GPUBUFFER BINDING API
        // =====================================================================

        BOOL SetGPUBuffer(const char* property, const char* resource,
                          GPUBuffer* buffer,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot) return false;
            return ApplyGPUBufferToSlot(*slot, buffer, uavAccess);
        }

        BOOL SetGPUBuffer(const char* name,
                          GPUBuffer* buffer,
                          GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            if (!name) return false;
            BOOL boundAny = false;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (MatchesStructuredBufferSlot(m_props.structuredBuffers[i], name))
                {
                    if (ApplyGPUBufferToSlot(m_props.structuredBuffers[i], buffer, uavAccess))
                    {
                        boundAny = true;
                    }
                }
            }
            return boundAny;
        }

        GPUBuffer* GetGPUBuffer(const char* name) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(name);
            return slot ? slot->gpuBuffer : nullptr;
        }

        GPUBuffer* GetGPUBuffer(const char* property, const char* resource) const
        {
            if (!property || !resource) return nullptr;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (m_props.structuredBuffers[i].propertyName == property &&
                    m_props.structuredBuffers[i].resourceName == resource)
                {
                    return m_props.structuredBuffers[i].gpuBuffer;
                }
            }
            return nullptr;
        }

        void SetConstantBufferSlotData(UINT16 slotIndex, UINT16 registerSpace, const void* data, UINT32 byteSize)
        {
            if (!data || byteSize == 0 || slotIndex == 0xFFFF) return;
            for (SIZE_T i = 0; i < m_props.buffers.size(); ++i)
            {
                if (m_props.buffers[i].slotIndex == slotIndex &&
                    m_props.buffers[i].registerSpace == registerSpace)
                {
                    m_props.buffers[i].byteSize = byteSize;
                    m_props.buffers[i].cpuData.assign(static_cast<const BYTE*>(data),
                                                      static_cast<const BYTE*>(data) + byteSize);
                    m_props.buffers[i].dirty = true;
                    return;
                }
            }

            ShaderBufferSlot slot{};
            slot.slotIndex     = slotIndex;
            slot.registerSpace = registerSpace;
            slot.byteSize      = byteSize;
            slot.cpuData.assign(static_cast<const BYTE*>(data), static_cast<const BYTE*>(data) + byteSize);
            slot.dirty         = true;
            m_props.buffers.push_back(std::move(slot));
        }

        BOOL SetStructuredBufferSlot(UINT16 slotIndex, UINT16 registerSpace, GPUBuffer* buffer,
                                     BOOL isUAV = false,
                                     GPUResourceAccess uavAccess = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE)
        {
            if (slotIndex == 0xFFFF) return false;
            if (buffer)
            {
                if (!buffer->IsValid()) return false;
                if (isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV)) return false;
                if (!isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV)) return false;
            }

            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (m_props.structuredBuffers[i].slotIndex == slotIndex &&
                    m_props.structuredBuffers[i].registerSpace == registerSpace &&
                    m_props.structuredBuffers[i].isUAV == isUAV)
                {
                    if (!buffer)
                    {
                        m_props.structuredBuffers[i].gpuBuffer             = nullptr;
                        m_props.structuredBuffers[i].gpuBufferHandle       = 0;
                        m_props.structuredBuffers[i].gpuBufferElementCount = 0;
                        m_props.structuredBuffers[i].gpuBufferStride       = 0;
                        m_props.structuredBuffers[i].gpuBufferByteSize     = 0;
                        m_props.structuredBuffers[i].hasDestroyedResource  = false;
                        return true;
                    }
                    m_props.structuredBuffers[i].stride                = buffer->GetStride();
                    m_props.structuredBuffers[i].gpuBuffer             = buffer;
                    m_props.structuredBuffers[i].gpuBufferHandle       = buffer->GetNativeHandle();
                    m_props.structuredBuffers[i].gpuBufferElementCount = buffer->GetElementCount();
                    m_props.structuredBuffers[i].gpuBufferStride       = buffer->GetStride();
                    m_props.structuredBuffers[i].gpuBufferByteSize     = buffer->GetByteSize();
                    m_props.structuredBuffers[i].uavAccess             = uavAccess;
                    m_props.structuredBuffers[i].hasDestroyedResource  = false;
                    return true;
                }
            }

            if (!buffer) return true;

            ShaderStructuredBufferSlot slot{};
            slot.slotIndex             = slotIndex;
            slot.registerSpace         = registerSpace;
            slot.stride                = buffer->GetStride();
            slot.isUAV                 = isUAV;
            slot.uavAccess             = uavAccess;
            slot.gpuBuffer             = buffer;
            slot.gpuBufferHandle       = buffer->GetNativeHandle();
            slot.gpuBufferElementCount = buffer->GetElementCount();
            slot.gpuBufferStride       = buffer->GetStride();
            slot.gpuBufferByteSize     = buffer->GetByteSize();
            m_props.structuredBuffers.push_back(std::move(slot));
            return true;
        }

        void SetStructuredBufferData(const char* property, const char* resource,
                                     const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot || !elements || elementCount == 0 || slot->stride == 0)
                return;

            const UINT32 bytes = elementCount * slot->stride;
            while (slot->cpuData.size() < bytes)
                slot->cpuData.push_back(0);

            std::memcpy(slot->cpuData.data(), elements, bytes);
            slot->elementCount = elementCount;
            slot->dirty = true;
        }

        void SetStructuredBufferData(const char* name, const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return;
            SetStructuredBufferData(slot->propertyName.c_str(), slot->resourceName.c_str(), elements, elementCount);
        }

        UINT32 AddStructuredBufferUserElement(const char* property, const char* resource)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot || slot->stride == 0) return 0xFFFFFFFF;

            UINT32 userIndex = slot->userElementCount++;
            const SIZE_T newBytes = static_cast<SIZE_T>(slot->userElementCount) * slot->stride;
            while (slot->userData.size() < newBytes)
                slot->userData.push_back(0);

            slot->dirty = true;
            return userIndex;
        }

        UINT32 AddStructuredBufferUserElement(const char* name)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return 0xFFFFFFFF;
            return AddStructuredBufferUserElement(slot->propertyName.c_str(), slot->resourceName.c_str());
        }

        BOOL SetStructuredBufferUserField(const char* property, const char* resource,
                                          UINT32 userIndex, const char* field,
                                          const void* data, UINT32 byteSize)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot || !data || !field || byteSize == 0 || userIndex >= slot->userElementCount || slot->stride == 0)
                return false;

            if (!m_computeShader || !m_computeShader->IsValid())
                return false;

            const ShaderReflection& refl = m_computeShader->GetReflection();
            std::string qName = std::string(property) + "__" + resource;
            const NSL::NSCStructuredBuffer* sb = refl.FindStructuredBuffer(qName);
            if (!sb) sb = refl.FindStructuredBuffer(resource);
            if (sb)
            {
                for (const auto& f : sb->fields)
                {
                    if (f.name == field || FieldNameEquals(f.name, field))
                    {
                        if (f.isGet) return false;
                        if (f.offset + byteSize > slot->stride) return false;

                        BYTE* elemPtr = slot->userData.data() + userIndex * slot->stride;
                        std::memcpy(elemPtr + f.offset, data, (std::min<UINT32>)(byteSize, f.size));
                        slot->dirty = true;
                        return true;
                    }
                }
            }
            else
            {
                const NSL::NSCUAVBuffer* uav = refl.FindUAV(qName);
                if (!uav) uav = refl.FindUAV(resource);
                if (uav)
                {
                    for (const auto& f : uav->fields)
                    {
                        if (f.name == field || FieldNameEquals(f.name, field))
                        {
                            if (f.isGet) return false;
                            if (f.offset + byteSize > slot->stride) return false;

                            BYTE* elemPtr = slot->userData.data() + userIndex * slot->stride;
                            std::memcpy(elemPtr + f.offset, data, (std::min<UINT32>)(byteSize, f.size));
                            slot->dirty = true;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        BOOL SetStructuredBufferUserField(const char* name, UINT32 userIndex, const char* field,
                                          const void* data, UINT32 byteSize)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return false;
            return SetStructuredBufferUserField(slot->propertyName.c_str(), slot->resourceName.c_str(),
                                                userIndex, field, data, byteSize);
        }

        void SetStructuredBufferUserData(const char* property, const char* resource,
                                         const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot || !elements || elementCount == 0 || slot->stride == 0)
                return;

            const UINT32 bytes = elementCount * slot->stride;
            while (slot->userData.size() < bytes)
                slot->userData.push_back(0);

            std::memcpy(slot->userData.data(), elements, bytes);
            slot->userElementCount = elementCount;

            if (m_computeShader && m_computeShader->IsValid())
            {
                const ShaderReflection& refl = m_computeShader->GetReflection();
                std::string qName = std::string(property) + "__" + resource;
                const NSL::NSCStructuredBuffer* sb = refl.FindStructuredBuffer(qName);
                if (!sb) sb = refl.FindStructuredBuffer(resource);
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
                    const NSL::NSCUAVBuffer* uav = refl.FindUAV(qName);
                    if (!uav) uav = refl.FindUAV(resource);
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
        }

        void SetStructuredBufferUserData(const char* name, const void* elements, UINT32 elementCount)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return;
            SetStructuredBufferUserData(slot->propertyName.c_str(), slot->resourceName.c_str(), elements, elementCount);
        }

        void ClearStructuredBufferUserElements(const char* property, const char* resource)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
            if (!slot) return;
            slot->userData.clear();
            slot->userElementCount = 0;
            slot->dirty = true;
        }

        void ClearStructuredBufferUserElements(const char* name)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return;
            ClearStructuredBufferUserElements(slot->propertyName.c_str(), slot->resourceName.c_str());
        }

        BOOL RemoveStructuredBufferUserElement(const char* property, const char* resource, UINT32 userIndex)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(property, resource);
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
            return true;
        }

        BOOL RemoveStructuredBufferUserElement(const char* name, UINT32 userIndex)
        {
            ShaderStructuredBufferSlot* slot = FindStructuredBufferSlot(name);
            if (!slot) return false;
            return RemoveStructuredBufferUserElement(slot->propertyName.c_str(), slot->resourceName.c_str(), userIndex);
        }

        UINT32 GetStructuredBufferUserElementCount(const char* property, const char* resource) const
        {
            if (!property || !resource) return 0;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (m_props.structuredBuffers[i].propertyName == property &&
                    m_props.structuredBuffers[i].resourceName == resource)
                    return m_props.structuredBuffers[i].userElementCount;
            }
            return 0;
        }

        UINT32 GetStructuredBufferUserElementCount(const char* name) const
        {
            const ShaderStructuredBufferSlot* slot = FindStructuredBufferSlotConst(name);
            return slot ? slot->userElementCount : 0;
        }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreMaterialManager;
        friend class CoreComputeManager;
        friend class CoreRenderPipeline;

        ComputeKernel() = default;
        ComputeKernel(const ComputeKernel& other) = default;
        ComputeKernel& operator=(const ComputeKernel& other) = default;

        void OnShaderDestroyed(Shader* destroyedShader)
        {
            if (destroyedShader && m_computeShader == destroyedShader)
            {
                m_computeShader      = nullptr;
                m_hasDestroyedShader = true;
            }
        }

        void OnTextureDestroyed(Texture* destroyedTex)
        {
            m_props.OnTextureDestroyed(destroyedTex);
        }

        void OnSamplerDestroyed(SamplerState* destroyedSamp)
        {
            m_props.OnSamplerDestroyed(destroyedSamp);
        }

        void OnGPUBufferDestroyed(GPUBuffer* destroyedBuf)
        {
            if (!destroyedBuf) return;
            m_props.OnGPUBufferDestroyed(destroyedBuf);
            if (m_indirectArgsBuffer == destroyedBuf)
            {
                m_indirectArgsBuffer         = nullptr;
                m_indirectArgsBufferHandle   = 0;
                m_indirectArgsOffset         = 0;
                m_hasDestroyedIndirectBuffer = true;
            }
        }

        // Fallback parser used only when binary .nsc reflection does not provide [numthreads(X,Y,Z)]
        // (e.g., raw inline HLSL compute shaders). Returns false and leaves outX/Y/Z = 0 if not found.
        static BOOL TryParseNumThreadsFromSource(const std::string& src, UINT32& outX, UINT32& outY, UINT32& outZ)
        {
            outX = 0;
            outY = 0;
            outZ = 0;
            if (src.empty()) return false;

            SIZE_T pos = src.find("numthreads");
            if (pos == std::string::npos) return false;

            SIZE_T openParen = src.find('(', pos);
            SIZE_T closeParen = (openParen != std::string::npos) ? src.find(')', openParen) : std::string::npos;
            if (openParen == std::string::npos || closeParen == std::string::npos || closeParen <= openParen + 1)
                return false;

            std::string args = src.substr(openParen + 1, closeParen - openParen - 1);
            UINT32 vals[3] = { 0, 0, 0 };
            SIZE_T start = 0;
            int count = 0;
            for (; count < 3 && start < args.size(); ++count)
            {
                SIZE_T comma = args.find(',', start);
                if (comma == std::string::npos) comma = args.size();
                std::string token = args.substr(start, comma - start);
                SIZE_T p = 0;
                while (p < token.size() && std::isspace(static_cast<unsigned char>(token[p]))) ++p;
                UINT32 parsed = 0;
                bool hasDigit = false;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p])))
                {
                    hasDigit = true;
                    parsed = parsed * 10u + static_cast<UINT32>(token[p] - '0');
                    ++p;
                }
                if (p < token.size() && (token[p] == 'u' || token[p] == 'U')) ++p;
                while (p < token.size() && std::isspace(static_cast<unsigned char>(token[p]))) ++p;
                if (!hasDigit || parsed == 0 || p != token.size())
                    return false;

                vals[count] = parsed;
                start = comma + 1;
            }
            if (count != 3) return false;
            outX = vals[0];
            outY = vals[1];
            outZ = vals[2];
            return true;
        }

        static BOOL MatchesGroupQualifier(const std::string& propName, const std::string& grpName, const char* group)
        {
            return propName == group ||
                   grpName == group ||
                   (propName + "__" + grpName) == group ||
                   (propName + "." + grpName) == group;
        }

        static BOOL MatchesTextureSlot(const ShaderTextureSlot& slot, const char* name)
        {
            return slot.resourceName == name ||
                   slot.groupName == name ||
                   slot.propertyName == name ||
                   (slot.propertyName + "__" + slot.resourceName) == name ||
                   (slot.groupName + "__" + slot.resourceName) == name ||
                   (slot.propertyName + "__" + slot.groupName + "__" + slot.resourceName) == name;
        }

        static BOOL MatchesGroupTextureSlot(const ShaderTextureSlot& slot, const char* group, const char* name)
        {
            return MatchesGroupQualifier(slot.propertyName, slot.groupName, group) &&
                   slot.resourceName == name;
        }

        static BOOL MatchesRWTextureSlot(const ShaderRWTextureSlot& slot, const char* name)
        {
            return slot.resourceName == name ||
                   slot.groupName == name ||
                   slot.propertyName == name ||
                   (slot.propertyName + "__" + slot.resourceName) == name ||
                   (slot.groupName + "__" + slot.resourceName) == name ||
                   (slot.propertyName + "__" + slot.groupName + "__" + slot.resourceName) == name;
        }

        static BOOL MatchesGroupRWTextureSlot(const ShaderRWTextureSlot& slot, const char* group, const char* name)
        {
            return MatchesGroupQualifier(slot.propertyName, slot.groupName, group) &&
                   slot.resourceName == name;
        }

        static BOOL MatchesSamplerSlot(const ShaderSamplerSlot& slot, const char* name)
        {
            return slot.resourceName == name ||
                   slot.groupName == name ||
                   slot.propertyName == name ||
                   (slot.propertyName + "__" + slot.resourceName) == name ||
                   (slot.groupName + "__" + slot.resourceName) == name ||
                   (slot.propertyName + "__" + slot.groupName + "__" + slot.resourceName) == name;
        }

        static BOOL MatchesGroupSamplerSlot(const ShaderSamplerSlot& slot, const char* group, const char* name)
        {
            return MatchesGroupQualifier(slot.propertyName, slot.groupName, group) &&
                   slot.resourceName == name;
        }

        static BOOL MatchesStructuredBufferSlot(const ShaderStructuredBufferSlot& slot, const char* name)
        {
            return slot.resourceName == name ||
                   slot.propertyName == name ||
                   (slot.propertyName + "__" + slot.resourceName) == name;
        }

        static BOOL ApplyGPUBufferToSlot(ShaderStructuredBufferSlot& slot, GPUBuffer* buffer, GPUResourceAccess uavAccess)
        {
            if (!buffer)
            {
                slot.gpuBuffer             = nullptr;
                slot.gpuBufferHandle       = 0;
                slot.gpuBufferElementCount = 0;
                slot.gpuBufferStride       = 0;
                slot.gpuBufferByteSize     = 0;
                slot.uavAccess             = GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE;
                slot.hasDestroyedResource  = false;
                return true;
            }

            if (!buffer->IsValid() || buffer->GetStride() != slot.stride)
                return false;

            if (slot.isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_UAV))
                return false;

            if (!slot.isUAV && !buffer->HasUsage(GPUBufferUsage::STRUCTURED_SRV))
                return false;

            slot.gpuBuffer             = buffer;
            slot.gpuBufferHandle       = buffer->GetNativeHandle();
            slot.gpuBufferElementCount = buffer->GetElementCount();
            slot.gpuBufferStride       = buffer->GetStride();
            slot.gpuBufferByteSize     = buffer->GetByteSize();
            slot.uavAccess             = uavAccess;
            slot.hasDestroyedResource  = false;
            return true;
        }

        const NSL::NSCBuffer* FindBufferInReflection(const ShaderBufferSlot& bSlot) const
        {
            if (!m_computeShader || !m_computeShader->IsValid()) return nullptr;
            const ShaderReflection& refl = m_computeShader->GetReflection();

            const NSL::NSCBuffer* buf = refl.FindBuffer(bSlot.propertyName + "__" + bSlot.resourceName);
            if (buf) return buf;
            for (const auto& candidate : refl.buffers)
            {
                if (candidate.name == bSlot.resourceName || candidate.name == bSlot.propertyName)
                    return &candidate;
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

        BOOL HasUniqueUnqualifiedBufferField(const char* field,
                                             std::string& outProp, std::string& outRes) const
        {
            outProp.clear();
            outRes.clear();
            if (!field) return false;

            BOOL found = false;
            for (SIZE_T b = 0; b < m_props.buffers.size(); ++b)
            {
                const ShaderBufferSlot& bSlot = m_props.buffers[b];
                const NSL::NSCBuffer* buffer = FindBufferInReflection(bSlot);
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
            return found;
        }

        BOOL HasUniqueGroupBufferField(const char* group, const char* field,
                                       std::string& outProp, std::string& outRes) const
        {
            outProp.clear();
            outRes.clear();
            if (!group || !field) return false;

            for (SIZE_T b = 0; b < m_props.buffers.size(); ++b)
            {
                const ShaderBufferSlot& bSlot = m_props.buffers[b];
                if ((bSlot.propertyName + "__" + bSlot.resourceName) != group &&
                    (bSlot.propertyName + "." + bSlot.resourceName) != group)
                    continue;

                const NSL::NSCBuffer* buffer = FindBufferInReflection(bSlot);
                if (buffer && FindAddField(*buffer, field))
                {
                    outProp = bSlot.propertyName;
                    outRes  = bSlot.resourceName;
                    return true;
                }
            }

            BOOL found = false;
            for (SIZE_T b = 0; b < m_props.buffers.size(); ++b)
            {
                const ShaderBufferSlot& bSlot = m_props.buffers[b];
                if (bSlot.resourceName != group && bSlot.propertyName != group)
                    continue;

                const NSL::NSCBuffer* buffer = FindBufferInReflection(bSlot);
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
            return found;
        }

        BOOL ResolveBufferFieldByName(const char* field,
                                      ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField)
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(field, targetProp, targetRes))
                return false;
            return ResolveBufferField(targetProp.c_str(), targetRes.c_str(), field, outSlot, outField);
        }

        BOOL ResolveBufferFieldByNameConst(const char* field,
                                           const ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField) const
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(field, targetProp, targetRes))
                return false;
            return ResolveBufferFieldConst(targetProp.c_str(), targetRes.c_str(), field, outSlot, outField);
        }

        BOOL ResolveGroupBufferFieldConst(const char* group, const char* field,
                                          const ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField) const
        {
            outSlot  = nullptr;
            outField = nullptr;
            std::string targetProp, targetRes;
            if (!HasUniqueGroupBufferField(group, field, targetProp, targetRes))
                return false;
            return ResolveBufferFieldConst(targetProp.c_str(), targetRes.c_str(), field, outSlot, outField);
        }

        BOOL ResolveBufferFieldConst(const char* property, const char* resource, const char* field,
                                     const ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField) const
        {
            outSlot  = nullptr;
            outField = nullptr;
            if (!property || !resource || !field) return false;

            for (SIZE_T i = 0; i < m_props.buffers.size(); ++i)
            {
                if (m_props.buffers[i].propertyName == property &&
                    m_props.buffers[i].resourceName == resource)
                {
                    const ShaderBufferSlot& slot = m_props.buffers[i];
                    if (slot.cpuData.empty()) return false;
                    const NSL::NSCBuffer* buffer = FindBufferInReflection(slot);
                    if (!buffer) return false;
                    const NSL::NSCField* nscField = FindAddField(*buffer, field);
                    if (!nscField) return false;
                    outSlot  = &slot;
                    outField = nscField;
                    return true;
                }
            }
            return false;
        }

        BOOL ResolveBufferField(const char* property, const char* resource, const char* field,
                                ShaderBufferSlot*& outSlot, const NSL::NSCField*& outField)
        {
            outSlot  = nullptr;
            outField = nullptr;
            if (!property || !resource || !field) return false;

            ShaderBufferSlot* slot = FindBufferSlot(property, resource);
            if (!slot || slot->cpuData.empty()) return false;

            const NSL::NSCBuffer* buffer = FindBufferInReflection(*slot);
            if (!buffer) return false;

            const NSL::NSCField* nscField = FindAddField(*buffer, field);
            if (!nscField) return false;

            outSlot  = slot;
            outField = nscField;
            return true;
        }

        ShaderBufferSlot* FindBufferSlot(const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            for (SIZE_T i = 0; i < m_props.buffers.size(); ++i)
            {
                if (m_props.buffers[i].propertyName == property &&
                    m_props.buffers[i].resourceName == resource)
                    return &m_props.buffers[i];
            }
            return nullptr;
        }

        const ShaderTextureSlot* FindTextureSlotConst(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.textures.size(); ++i)
            {
                if (MatchesTextureSlot(m_props.textures[i], name))
                    return &m_props.textures[i];
            }
            return nullptr;
        }

        const ShaderRWTextureSlot* FindRWTextureSlotConst(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.rwTextures.size(); ++i)
            {
                if (MatchesRWTextureSlot(m_props.rwTextures[i], name))
                    return &m_props.rwTextures[i];
            }
            return nullptr;
        }

        const ShaderSamplerSlot* FindSamplerSlotConst(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.samplers.size(); ++i)
            {
                if (MatchesSamplerSlot(m_props.samplers[i], name))
                    return &m_props.samplers[i];
            }
            return nullptr;
        }

        ShaderStructuredBufferSlot* FindStructuredBufferSlot(const char* property, const char* resource)
        {
            if (!property || !resource) return nullptr;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (m_props.structuredBuffers[i].propertyName == property &&
                    m_props.structuredBuffers[i].resourceName == resource)
                    return &m_props.structuredBuffers[i];
            }
            return nullptr;
        }

        ShaderStructuredBufferSlot* FindStructuredBufferSlot(const char* name)
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (MatchesStructuredBufferSlot(m_props.structuredBuffers[i], name))
                    return &m_props.structuredBuffers[i];
            }
            return nullptr;
        }

        const ShaderStructuredBufferSlot* FindStructuredBufferSlotConst(const char* name) const
        {
            if (!name) return nullptr;
            for (SIZE_T i = 0; i < m_props.structuredBuffers.size(); ++i)
            {
                if (MatchesStructuredBufferSlot(m_props.structuredBuffers[i], name))
                    return &m_props.structuredBuffers[i];
            }
            return nullptr;
        }

        void WriteFieldByName(const char* field, const void* src, UINT32 byteCount)
        {
            if (!field || !src || byteCount == 0) return;

            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(field, targetProp, targetRes))
                return;
            WriteBufferField(targetProp.c_str(), targetRes.c_str(), field, src, byteCount);
        }

        BOOL ReadFieldByName(const char* field, void* dst, UINT32 byteCount) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferFieldByNameConst(field, slot, nscField))
                return false;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return false;

            std::memcpy(dst, slot->cpuData.data() + nscField->offset, byteCount);
            return true;
        }

        void WriteGroupFieldByName(const char* group, const char* field, const void* src, UINT32 byteCount)
        {
            if (!group || !field || !src || byteCount == 0) return;

            std::string targetProp, targetRes;
            if (!HasUniqueGroupBufferField(group, field, targetProp, targetRes))
                return;
            WriteBufferField(targetProp.c_str(), targetRes.c_str(), field, src, byteCount);
        }

        BOOL ReadGroupFieldByName(const char* group, const char* field, void* dst, UINT32 byteCount) const
        {
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveGroupBufferFieldConst(group, field, slot, nscField))
                return false;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return false;

            std::memcpy(dst, slot->cpuData.data() + nscField->offset, byteCount);
            return true;
        }

        void WriteBufferField(const char* property, const char* resource, const char* field,
                              const void* src, UINT32 byteCount)
        {
            ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferField(property, resource, field, slot, nscField))
                return;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return;

            std::memcpy(slot->cpuData.data() + nscField->offset, src, byteCount);
            slot->dirty = true;
        }

        BOOL ReadBufferField(const char* property, const char* resource, const char* field,
                             void* dst, UINT32 byteCount) const
        {
            if (!dst || byteCount == 0) return false;
            const ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferFieldConst(property, resource, field, slot, nscField))
                return false;

            if (nscField->size < byteCount || nscField->offset + byteCount > slot->byteSize)
                return false;

            std::memcpy(dst, slot->cpuData.data() + nscField->offset, byteCount);
            return true;
        }

        void WriteMat3ByName(const char* field, const Mat3& value)
        {
            if (!field) return;
            std::string targetProp, targetRes;
            if (!HasUniqueUnqualifiedBufferField(field, targetProp, targetRes))
                return;
            WriteMat3Field(targetProp.c_str(), targetRes.c_str(), field, value);
        }

        void WriteGroupMat3ByName(const char* group, const char* field, const Mat3& value)
        {
            if (!group || !field) return;
            std::string targetProp, targetRes;
            if (!HasUniqueGroupBufferField(group, field, targetProp, targetRes))
                return;
            WriteMat3Field(targetProp.c_str(), targetRes.c_str(), field, value);
        }

        void WriteMat3Field(const char* property, const char* resource, const char* field, const Mat3& value)
        {
            ShaderBufferSlot* slot = nullptr;
            const NSL::NSCField* nscField = nullptr;
            if (!ResolveBufferField(property, resource, field, slot, nscField))
                return;

            if (nscField->offset + nscField->size > slot->byteSize) return;

            UINT32 colStride = nscField->size / 3;
            for (UINT32 col = 0; col < 3; ++col)
            {
                FLOAT32 packed[4] = { value.m[col * 3 + 0], value.m[col * 3 + 1], value.m[col * 3 + 2], 0.0f };
                std::memcpy(slot->cpuData.data() + nscField->offset + col * colStride, packed, colStride);
            }
            slot->dirty = true;
        }

        // ==================== MEMBERS ====================
        Shader*               m_computeShader            = nullptr;
        std::string           m_name;
        UINT32                m_threadGroupSizeX         = 0;
        UINT32                m_threadGroupSizeY         = 0;
        UINT32                m_threadGroupSizeZ         = 0;
        BOOL                  m_hasValidThreadGroupSize  = false;
        UINT32                m_threadGroupCountX        = 1;
        UINT32                m_threadGroupCountY        = 1;
        UINT32                m_threadGroupCountZ        = 1;
        ComputeExecutionScope m_executionScope           = ComputeExecutionScope::PER_VIEW;
        GPUBuffer*            m_indirectArgsBuffer       = nullptr;
        NativeHandle          m_indirectArgsBufferHandle = 0;
        UINT32                m_indirectArgsOffset       = 0;
        BOOL                  m_useRootConstants         = false;
        UINT32                m_rootConstants[4]         = { 0, 0, 0, 0 };
        BOOL                  m_hasDestroyedShader       = false;
        BOOL                  m_hasDestroyedIndirectBuffer = false;
        ShaderStageProps      m_props;
    };
}
