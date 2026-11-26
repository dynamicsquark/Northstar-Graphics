#pragma once

#include "shader_types.h"
#include "shader_desc.h"
#include "../types/northstar_types.h"

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace Northstar
{
    class Shader
    {
    public:
        // ==================== IDENTIFICATION & STAGE ====================
        NativeHandle            GetNativeHandle()  const { return m_nativeHandle; }
        UINT64                  GetStateRevision() const { return m_stateRevision; }
        ShaderStage             GetStage()         const { return m_reflection.stage; }
        TargetAPI               GetTargetAPI()     const { return m_reflection.api; }
        UINT32                  GetNSCVersion()    const { return m_reflection.nscVersion; }
        const std::string&      GetName()          const { return m_name; }

        // ==================== PIPELINE CLASSIFICATION ====================
        BOOL IsValid()    const { return m_reflection.stage != ShaderStage::NONE; }
        BOOL IsGraphics() const { return IsValid() && m_reflection.stage != ShaderStage::COMPUTE; }
        BOOL IsCompute()  const { return IsValid() && m_reflection.stage == ShaderStage::COMPUTE; }

        // ==================== CODE / BYTECODE ====================
        const std::string&          GetSourceCode()   const { return m_reflection.sourceCode; }
        BOOL                        HasSourceCode()   const { return !m_reflection.sourceCode.empty(); }
        const std::vector<BYTE>&    GetBytecode()     const { return m_bytecode; }
        const BYTE*                 GetBytecodeData() const { return m_bytecode.data(); }
        SIZE_T                      GetBytecodeSize() const { return m_bytecode.size(); }
        BOOL                        HasBytecode()     const { return !m_bytecode.empty(); }

        // ==================== REFLECTION & DATA ACCESS ====================
        const ShaderReflection&     GetReflection()   const { return m_reflection; }

        // Helper lookups for reflection resources
        const NSCBuffer* FindBuffer(const std::string& name) const
        {
            return m_reflection.FindBuffer(name);
        }

        const NSCTexture* FindTexture(const std::string& groupQualifiedName, const std::string& entryName) const
        {
            return m_reflection.FindTexture(groupQualifiedName, entryName);
        }

        const NSCTexture* FindTexture(const std::string& name) const
        {
            for (const auto& t : m_reflection.textures)
                if (t.name == name || t.groupQualifiedName == name) return &t;
            return nullptr;
        }

        const NSCSampler* FindSampler(const std::string& groupQualifiedName, const std::string& entryName) const
        {
            return m_reflection.FindSampler(groupQualifiedName, entryName);
        }

        const NSCSampler* FindSampler(const std::string& name) const
        {
            for (const auto& s : m_reflection.samplers)
                if (s.name == name || s.groupQualifiedName == name) return &s;
            return nullptr;
        }

        const NSCStructuredBuffer* FindStructuredBuffer(const std::string& name) const
        {
            return m_reflection.FindStructuredBuffer(name);
        }

        const NSCUAVBuffer* FindUAVBuffer(const std::string& name) const
        {
            return m_reflection.FindUAV(name);
        }

        const NSCPropertyGroup* FindPropertyGroup(const std::string& name) const
        {
            return m_reflection.FindPropertyGroup(name);
        }

        const NSCPropertyGroup* FindPropertyGroupInBlock(const std::string& blockName, const std::string& groupName) const
        {
            return m_reflection.FindPropertyGroupInBlock(blockName, groupName);
        }

        const std::vector<NSCStageIOEntry>& GetInputLayout()  const { return m_reflection.inputLayout; }
        const std::vector<NSCStageIOEntry>& GetOutputLayout() const { return m_reflection.outputLayout; }

    private:
        friend class NorthstarEngine;
        friend class NorthstarCore;
        friend class CoreShaderManager;
        friend class ShaderLoader;
        friend class InternalResourceManager;

        Shader() = default;

        void SetReflection(const ShaderReflection& reflection)
        {
            m_reflection = reflection;
            ++m_stateRevision;
        }

        void SetReflection(ShaderReflection&& reflection)
        {
            m_reflection = std::move(reflection);
            ++m_stateRevision;
        }

        void SetBytecode(const BYTE* data, SIZE_T size)
        {
            if (data && size > 0)
                m_bytecode.assign(data, data + size);
            else
                m_bytecode.clear();
            ++m_stateRevision;
        }

        void SetBytecode(std::vector<BYTE> bytecode)
        {
            m_bytecode = std::move(bytecode);
            ++m_stateRevision;
        }

        void SetNativeHandle(NativeHandle handle)
        {
            m_nativeHandle = handle;
            ++m_stateRevision;
        }

        void SetName(const std::string& name)
        {
            m_name = name;
        }

        void BumpStateRevision()
        {
            ++m_stateRevision;
        }

        std::string          m_name;
        NativeHandle         m_nativeHandle  = 0;
        UINT64               m_stateRevision = 1u;
        ShaderReflection     m_reflection;
        std::vector<BYTE>    m_bytecode;
    };
}
