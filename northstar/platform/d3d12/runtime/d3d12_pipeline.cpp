#include "d3d12_pipeline.h"
#include "d3d12_resource_manager.h"
#include <algorithm>
#include <iostream>
#include <cctype>

namespace Northstar
{
    using namespace D3D12Util;

    BOOL D3D12Pipeline::Initialize(D3D12DeviceContext& ctx)
    {
        m_device = ctx.GetDevice();
        if (!m_device) return false;

        if (!CreateRootSignature(m_device))
            return false;

        if (!CreateComputeRootSignature(m_device))
            return false;

        if (!CreateCommandSignatures(m_device))
            return false;

        if (!CreateInstanceBuffer(m_device))
            return false;

        if (!CreateCBufferRing(m_device))
            return false;

        if (!CreateSBufferRing(m_device))
            return false;

        return true;
    }

    void D3D12Pipeline::Shutdown()
    {
        m_retiredBuffers.clear();
        m_retiredPSOs.clear();
        m_currentRetireFenceValue = 0;

        if (m_sbufferUploadBuffer && m_sbufferMappedData)
        {
            m_sbufferUploadBuffer->Unmap(0, nullptr);
            m_sbufferMappedData = nullptr;
        }
        m_sbufferUploadBuffer.Reset();
        m_sbufferOffset = 0;
        m_sbufferRingSize = kInitialSBufferRingSize;

        if (m_cbufferUploadBuffer && m_cbufferMappedData)
        {
            m_cbufferUploadBuffer->Unmap(0, nullptr);
            m_cbufferMappedData = nullptr;
        }
        m_cbufferUploadBuffer.Reset();
        m_cbufferOffset = 0;
        m_cbufferRingSize = kInitialCBufferRingSize;

        if (m_instanceUploadBuffer && m_instanceMappedData)
        {
            m_instanceUploadBuffer->Unmap(0, nullptr);
            m_instanceMappedData = nullptr;
        }
        m_instanceUploadBuffer.Reset();
        m_instanceOffset = 0;
        m_instanceRingCount = kInitialInstanceRingCount;
        m_instanceRingSize = sizeof(D3D12InstancePayload) * kInitialInstanceRingCount;

        m_psoCache.clear();
        m_computePsoCache.clear();
        m_indirectRootObjects.clear();
        m_dispatchIndirectSignature.Reset();
        m_nvsDrawIndexedIndirectSignature.Reset();
        m_drawIndexedIndirectSignature.Reset();
        m_drawIndirectSignature.Reset();
        m_computeRootSignature.Reset();
        m_rootSignature.Reset();
        m_device = nullptr;
    }

    void D3D12Pipeline::ReclaimRetiredResources(UINT64 completedFenceValue)
    {
        if (completedFenceValue == UINT64_MAX)
        {
            // Device removed: do not interpret UINT64_MAX as normal fence completion here;
            // explicit Shutdown / FlushAllRetiredResources handles teardown.
            return;
        }

        m_retiredBuffers.erase(
            std::remove_if(
                m_retiredBuffers.begin(),
                m_retiredBuffers.end(),
                [completedFenceValue](const RetiredBufferEntry& e) {
                    return completedFenceValue >= e.fenceValue;
                }),
            m_retiredBuffers.end());

        m_retiredPSOs.erase(
            std::remove_if(
                m_retiredPSOs.begin(),
                m_retiredPSOs.end(),
                [completedFenceValue](const RetiredPSOEntry& e) {
                    return completedFenceValue >= e.fenceValue;
                }),
            m_retiredPSOs.end());
    }

    void D3D12Pipeline::BeginFrameBuffers(UINT32 bbIndex, UINT64 completedFenceValue, UINT64 upcomingFenceValue)
    {
        m_activeBB = bbIndex % kBackBufferCount;
        if (upcomingFenceValue > m_currentRetireFenceValue)
        {
            m_currentRetireFenceValue = upcomingFenceValue;
        }
        if (completedFenceValue > 0 && completedFenceValue != UINT64_MAX)
        {
            ReclaimRetiredResources(completedFenceValue);
        }

        const UINT64 cbufferPerBB = m_cbufferRingSize / kBackBufferCount;
        const UINT64 sbufferPerBB = m_sbufferRingSize / kBackBufferCount;
        const UINT64 instPerBB    = m_instanceRingSize / kBackBufferCount;

        m_cbufferOffset  = m_activeBB * cbufferPerBB;
        m_sbufferOffset  = m_activeBB * sbufferPerBB;
        m_instanceOffset = m_activeBB * instPerBB;
    }

    void D3D12Pipeline::FlushAllRetiredResources()
    {
        m_retiredBuffers.clear();
        m_retiredPSOs.clear();
    }

    BOOL D3D12Pipeline::CreateRootSignature(ID3D12Device* device)
    {
        if (!device) return false;

        // Base Graphics Root Signature (45 parameters = 48 DWORDs, zero space4 graphics slots):
        //   Params  0..3  : Vertex   space0..3 CBV+SRV Descriptor Tables (48 desc: 16 CBV + 32 SRV)
        //   Params  4..7  : Pixel    space0..3 CBV+SRV Descriptor Tables (48 desc: 16 CBV + 32 SRV)
        //   Params  8..11 : Vertex   space0..3 Sampler Descriptor Tables (16 samplers, VISIBILITY_VERTEX)
        //   Params 12..15 : Pixel    space0..3 Sampler Descriptor Tables (16 samplers, VISIBILITY_PIXEL)
        //   Param  16     : 4x 32-bit Root Constants at (b1, space3, VISIBILITY_ALL)
        //   Params 17..20 : Geometry space0..3 CBV+SRV Descriptor Tables (48 desc, VISIBILITY_GEOMETRY)
        //   Params 21..24 : Hull     space0..3 CBV+SRV Descriptor Tables (48 desc, VISIBILITY_HULL)
        //   Params 25..28 : Domain   space0..3 CBV+SRV Descriptor Tables (48 desc, VISIBILITY_DOMAIN)
        //   Params 29..32 : Geometry space0..3 Sampler Descriptor Tables (16 samplers, VISIBILITY_GEOMETRY)
        //   Params 33..36 : Hull     space0..3 Sampler Descriptor Tables (16 samplers, VISIBILITY_HULL)
        //   Params 37..40 : Domain   space0..3 Sampler Descriptor Tables (16 samplers, VISIBILITY_DOMAIN)
        //   Params 41..44 : Graphics space0..3 UAV Descriptor Tables (16 UAVs u0..15, VISIBILITY_ALL)
        D3D12_DESCRIPTOR_RANGE vtxRanges[kMaxSpaces][2]{};
        D3D12_DESCRIPTOR_RANGE pixRanges[kMaxSpaces][2]{};
        D3D12_DESCRIPTOR_RANGE geoRanges[kMaxSpaces][2]{};
        D3D12_DESCRIPTOR_RANGE hullRanges[kMaxSpaces][2]{};
        D3D12_DESCRIPTOR_RANGE domRanges[kMaxSpaces][2]{};
        D3D12_DESCRIPTOR_RANGE space3Ranges[3]{};
        D3D12_DESCRIPTOR_RANGE samplerRanges[kMaxSpaces]{};
        D3D12_DESCRIPTOR_RANGE uavRanges[kMaxSpaces]{};
        D3D12_ROOT_PARAMETER   rootParams[kGraphicsBaseRootParamCount]{};

        // Shared Space 3 Descriptor Ranges (b0 at offset 0, b2..b15 at offset 2, t0..t31 at offset 16; skips b1, space3)
        space3Ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        space3Ranges[0].NumDescriptors                    = 1;
        space3Ranges[0].BaseShaderRegister                = 0;
        space3Ranges[0].RegisterSpace                     = 3;
        space3Ranges[0].OffsetInDescriptorsFromTableStart = 0;

        space3Ranges[1].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        space3Ranges[1].NumDescriptors                    = kCBVPerSpace - 2;
        space3Ranges[1].BaseShaderRegister                = 2;
        space3Ranges[1].RegisterSpace                     = 3;
        space3Ranges[1].OffsetInDescriptorsFromTableStart = 2;

        space3Ranges[2].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        space3Ranges[2].NumDescriptors                    = kSRVPerSpace;
        space3Ranges[2].BaseShaderRegister                = 0;
        space3Ranges[2].RegisterSpace                     = 3;
        space3Ranges[2].OffsetInDescriptorsFromTableStart = kCBVPerSpace;

        auto configureSpaceTableParam = [&](UINT32 paramIdx, UINT32 s, D3D12_DESCRIPTOR_RANGE (&ranges)[kMaxSpaces][2], D3D12_SHADER_VISIBILITY visibility)
        {
            if (s == 3)
            {
                rootParams[paramIdx].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParams[paramIdx].DescriptorTable.NumDescriptorRanges = 3;
                rootParams[paramIdx].DescriptorTable.pDescriptorRanges   = space3Ranges;
                rootParams[paramIdx].ShaderVisibility                    = visibility;
            }
            else
            {
                ranges[s][0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                ranges[s][0].NumDescriptors                    = kCBVPerSpace;
                ranges[s][0].BaseShaderRegister                = 0;
                ranges[s][0].RegisterSpace                     = s;
                ranges[s][0].OffsetInDescriptorsFromTableStart = 0;

                ranges[s][1].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                ranges[s][1].NumDescriptors                    = kSRVPerSpace;
                ranges[s][1].BaseShaderRegister                = 0;
                ranges[s][1].RegisterSpace                     = s;
                ranges[s][1].OffsetInDescriptorsFromTableStart = kCBVPerSpace;

                rootParams[paramIdx].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParams[paramIdx].DescriptorTable.NumDescriptorRanges = 2;
                rootParams[paramIdx].DescriptorTable.pDescriptorRanges   = ranges[s];
                rootParams[paramIdx].ShaderVisibility                    = visibility;
            }
        };

        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            samplerRanges[s].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            samplerRanges[s].NumDescriptors                    = 16;
            samplerRanges[s].BaseShaderRegister                = 0;
            samplerRanges[s].RegisterSpace                     = s;
            samplerRanges[s].OffsetInDescriptorsFromTableStart = 0;

            uavRanges[s].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            uavRanges[s].NumDescriptors                    = kUAVPerSpace;
            uavRanges[s].BaseShaderRegister                = 0;
            uavRanges[s].RegisterSpace                     = s;
            uavRanges[s].OffsetInDescriptorsFromTableStart = 0;
        }

        auto configureSamplerStageParams = [&](UINT32 paramBase, D3D12_SHADER_VISIBILITY visibility)
        {
            for (UINT32 s = 0; s < kMaxSpaces; ++s)
            {
                rootParams[paramBase + s].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParams[paramBase + s].DescriptorTable.NumDescriptorRanges = 1;
                rootParams[paramBase + s].DescriptorTable.pDescriptorRanges   = &samplerRanges[s];
                rootParams[paramBase + s].ShaderVisibility                    = visibility;
            }
        };

        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            configureSpaceTableParam(kGraphicsVertexSpaceParamBase + s,   s, vtxRanges,  D3D12_SHADER_VISIBILITY_VERTEX);
            configureSpaceTableParam(kGraphicsPixelSpaceParamBase + s,    s, pixRanges,  D3D12_SHADER_VISIBILITY_PIXEL);
            configureSpaceTableParam(kGraphicsGeometrySpaceParamBase + s, s, geoRanges,  D3D12_SHADER_VISIBILITY_GEOMETRY);
            configureSpaceTableParam(kGraphicsHullSpaceParamBase + s,     s, hullRanges, D3D12_SHADER_VISIBILITY_HULL);
            configureSpaceTableParam(kGraphicsDomainSpaceParamBase + s,   s, domRanges,  D3D12_SHADER_VISIBILITY_DOMAIN);
        }

        configureSamplerStageParams(kGraphicsVertexSamplerParamBase,   D3D12_SHADER_VISIBILITY_VERTEX);
        configureSamplerStageParams(kGraphicsPixelSamplerParamBase,    D3D12_SHADER_VISIBILITY_PIXEL);
        configureSamplerStageParams(kGraphicsGeometrySamplerParamBase, D3D12_SHADER_VISIBILITY_GEOMETRY);
        configureSamplerStageParams(kGraphicsHullSamplerParamBase,     D3D12_SHADER_VISIBILITY_HULL);
        configureSamplerStageParams(kGraphicsDomainSamplerParamBase,   D3D12_SHADER_VISIBILITY_DOMAIN);

        // Param 16: 32-Bit Root Constants at (b1, space3)
        rootParams[kGraphicsRootConstantsParamIndex].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.ShaderRegister = 1;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.RegisterSpace  = 3;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.Num32BitValues = 4;
        rootParams[kGraphicsRootConstantsParamIndex].ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;

        // Params 41..44: Graphics UAV Tables (u0..u15 in space0..space3, VISIBILITY_ALL)
        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            rootParams[kGraphicsUAVSpaceParamBase + s].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParams[kGraphicsUAVSpaceParamBase + s].DescriptorTable.NumDescriptorRanges = 1;
            rootParams[kGraphicsUAVSpaceParamBase + s].DescriptorTable.pDescriptorRanges   = &uavRanges[s];
            rootParams[kGraphicsUAVSpaceParamBase + s].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        }

        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters     = kGraphicsBaseRootParamCount;
        rootDesc.pParameters       = rootParams;
        rootDesc.NumStaticSamplers = 0;
        rootDesc.pStaticSamplers   = nullptr;
        rootDesc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> signatureBlob;
        ComPtr<ID3DBlob> errorBlob;
        if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureBlob, &errorBlob)))
        {
            if (errorBlob)
            {
                std::cerr << "[D3D12Pipeline] Root Signature Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            }
            return false;
        }

        return SUCCEEDED(device->CreateRootSignature(
            0,
            signatureBlob->GetBufferPointer(),
            signatureBlob->GetBufferSize(),
            IID_PPV_ARGS(&m_rootSignature)
        ));
    }

    ID3D12RootSignature* D3D12Pipeline::GetOrCreateIndirectRootSignature(
        ID3D12Device*                 device,
        const IndirectRootBindingKey& key)
    {
        if (!device) return nullptr;
        if (!key.enabled) return m_rootSignature.Get();

        auto it = m_indirectRootObjects.find(key);
        if (it != m_indirectRootObjects.end() && it->second.rootSignature)
        {
            return it->second.rootSignature.Get();
        }

        auto mapStage = [](ShaderStage stage, UINT32& outStageIdx, D3D12_SHADER_VISIBILITY& outVis) -> bool
        {
            switch (stage)
            {
                case ShaderStage::VERTEX:   outStageIdx = 0; outVis = D3D12_SHADER_VISIBILITY_VERTEX;   return true;
                case ShaderStage::PIXEL:    outStageIdx = 1; outVis = D3D12_SHADER_VISIBILITY_PIXEL;    return true;
                case ShaderStage::GEOMETRY: outStageIdx = 2; outVis = D3D12_SHADER_VISIBILITY_GEOMETRY; return true;
                case ShaderStage::HULL:     outStageIdx = 3; outVis = D3D12_SHADER_VISIBILITY_HULL;     return true;
                case ShaderStage::DOMAIN:   outStageIdx = 4; outVis = D3D12_SHADER_VISIBILITY_DOMAIN;   return true;
                default:                    outStageIdx = 0; outVis = D3D12_SHADER_VISIBILITY_ALL;      return false;
            }
        };

        UINT32 cbvMask[5][kMaxSpaces]{};
        UINT32 srvMask[5][kMaxSpaces]{};
        for (UINT32 st = 0; st < 5; ++st)
        {
            for (UINT32 s = 0; s < kMaxSpaces; ++s)
            {
                cbvMask[st][s] = (1u << kCBVPerSpace) - 1u;
                if (s == 3)
                {
                    cbvMask[st][s] &= ~(1u << 1u); // Skip b1, space3 (reserved for 32-bit Root Constants)
                }
                srvMask[st][s] = 0xFFFFFFFFu;
            }
        }

        D3D12_ROOT_PARAMETER   rootParams[kGraphicsRootParamCount]{};
        D3D12_DESCRIPTOR_RANGE samplerRanges[kMaxSpaces]{};
        D3D12_DESCRIPTOR_RANGE uavRanges[kMaxSpaces]{};

        // Params 45..47: Promoted per-draw Root CBVs 0..2
        for (UINT32 i = 0; i < kGraphicsIndirectCBVParamCount; ++i)
        {
            const UINT32 paramIdx = kGraphicsIndirectCBVParamBase + i;
            UINT32                  stIdx = 0;
            D3D12_SHADER_VISIBILITY vis   = D3D12_SHADER_VISIBILITY_ALL;

            if (i < key.cbvCount &&
                key.cbvSpace[i] < kMaxSpaces &&
                key.cbvSlot[i] < kCBVPerSpace &&
                !(key.cbvSpace[i] == 3 && key.cbvSlot[i] == 1) &&
                mapStage(key.cbvStage[i], stIdx, vis))
            {
                cbvMask[stIdx][key.cbvSpace[i]] &= ~(1u << key.cbvSlot[i]);
                rootParams[paramIdx].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
                rootParams[paramIdx].Descriptor.ShaderRegister = key.cbvSlot[i];
                rootParams[paramIdx].Descriptor.RegisterSpace  = key.cbvSpace[i];
                rootParams[paramIdx].ShaderVisibility          = vis;
            }
            else
            {
                rootParams[paramIdx].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
                rootParams[paramIdx].Descriptor.ShaderRegister = i; // b0..b2, space5 (unreferenced placeholder)
                rootParams[paramIdx].Descriptor.RegisterSpace  = 5;
                rootParams[paramIdx].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            }
        }

        // Param 48: Promoted per-batch Root SRV 0
        {
            UINT32                  stIdx = 0;
            D3D12_SHADER_VISIBILITY vis   = D3D12_SHADER_VISIBILITY_ALL;
            if (key.hasPromotedSRV &&
                key.srvSpace < kMaxSpaces &&
                key.srvSlot < kSRVPerSpace &&
                mapStage(key.srvStage, stIdx, vis))
            {
                srvMask[stIdx][key.srvSpace] &= ~(1u << key.srvSlot);
                rootParams[kGraphicsIndirectSRVParamIndex].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
                rootParams[kGraphicsIndirectSRVParamIndex].Descriptor.ShaderRegister = key.srvSlot;
                rootParams[kGraphicsIndirectSRVParamIndex].Descriptor.RegisterSpace  = key.srvSpace;
                rootParams[kGraphicsIndirectSRVParamIndex].ShaderVisibility          = vis;
            }
            else
            {
                rootParams[kGraphicsIndirectSRVParamIndex].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
                rootParams[kGraphicsIndirectSRVParamIndex].Descriptor.ShaderRegister = 0; // t0, space5 (unreferenced placeholder)
                rootParams[kGraphicsIndirectSRVParamIndex].Descriptor.RegisterSpace  = 5;
                rootParams[kGraphicsIndirectSRVParamIndex].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            }
        }

        std::vector<D3D12_DESCRIPTOR_RANGE> rangeStorage;
        rangeStorage.reserve(128);

        const UINT32 stageParamBases[5] = {
            kGraphicsVertexSpaceParamBase,
            kGraphicsPixelSpaceParamBase,
            kGraphicsGeometrySpaceParamBase,
            kGraphicsHullSpaceParamBase,
            kGraphicsDomainSpaceParamBase
        };
        const UINT32 stageSamplerParamBases[5] = {
            kGraphicsVertexSamplerParamBase,
            kGraphicsPixelSamplerParamBase,
            kGraphicsGeometrySamplerParamBase,
            kGraphicsHullSamplerParamBase,
            kGraphicsDomainSamplerParamBase
        };
        const D3D12_SHADER_VISIBILITY stageVisibilities[5] = {
            D3D12_SHADER_VISIBILITY_VERTEX,
            D3D12_SHADER_VISIBILITY_PIXEL,
            D3D12_SHADER_VISIBILITY_GEOMETRY,
            D3D12_SHADER_VISIBILITY_HULL,
            D3D12_SHADER_VISIBILITY_DOMAIN
        };

        for (UINT32 st = 0; st < 5; ++st)
        {
            for (UINT32 s = 0; s < kMaxSpaces; ++s)
            {
                const SIZE_T firstRangeIdx = rangeStorage.size();

                UINT32 r = 0;
                while (r < kCBVPerSpace)
                {
                    if ((cbvMask[st][s] & (1u << r)) == 0u)
                    {
                        ++r;
                        continue;
                    }
                    const UINT32 runStart = r;
                    while (r < kCBVPerSpace && (cbvMask[st][s] & (1u << r)) != 0u)
                    {
                        ++r;
                    }
                    D3D12_DESCRIPTOR_RANGE rng{};
                    rng.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                    rng.NumDescriptors                    = r - runStart;
                    rng.BaseShaderRegister                = runStart;
                    rng.RegisterSpace                     = s;
                    rng.OffsetInDescriptorsFromTableStart = runStart;
                    rangeStorage.push_back(rng);
                }

                r = 0;
                while (r < kSRVPerSpace)
                {
                    if ((srvMask[st][s] & (1u << r)) == 0u)
                    {
                        ++r;
                        continue;
                    }
                    const UINT32 runStart = r;
                    while (r < kSRVPerSpace && (srvMask[st][s] & (1u << r)) != 0u)
                    {
                        ++r;
                    }
                    D3D12_DESCRIPTOR_RANGE rng{};
                    rng.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                    rng.NumDescriptors                    = r - runStart;
                    rng.BaseShaderRegister                = runStart;
                    rng.RegisterSpace                     = s;
                    rng.OffsetInDescriptorsFromTableStart = kCBVPerSpace + runStart;
                    rangeStorage.push_back(rng);
                }

                const UINT32 paramIdx = stageParamBases[st] + s;
                rootParams[paramIdx].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParams[paramIdx].DescriptorTable.NumDescriptorRanges = static_cast<UINT32>(rangeStorage.size() - firstRangeIdx);
                rootParams[paramIdx].DescriptorTable.pDescriptorRanges   = rangeStorage.data() + firstRangeIdx;
                rootParams[paramIdx].ShaderVisibility                    = stageVisibilities[st];
            }
        }

        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            samplerRanges[s].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            samplerRanges[s].NumDescriptors                    = 16;
            samplerRanges[s].BaseShaderRegister                = 0;
            samplerRanges[s].RegisterSpace                     = s;
            samplerRanges[s].OffsetInDescriptorsFromTableStart = 0;

            uavRanges[s].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            uavRanges[s].NumDescriptors                    = kUAVPerSpace;
            uavRanges[s].BaseShaderRegister                = 0;
            uavRanges[s].RegisterSpace                     = s;
            uavRanges[s].OffsetInDescriptorsFromTableStart = 0;

            for (UINT32 st = 0; st < 5; ++st)
            {
                const UINT32 sampParam = stageSamplerParamBases[st] + s;
                rootParams[sampParam].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                rootParams[sampParam].DescriptorTable.NumDescriptorRanges = 1;
                rootParams[sampParam].DescriptorTable.pDescriptorRanges   = &samplerRanges[s];
                rootParams[sampParam].ShaderVisibility                    = stageVisibilities[st];
            }

            rootParams[kGraphicsUAVSpaceParamBase + s].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParams[kGraphicsUAVSpaceParamBase + s].DescriptorTable.NumDescriptorRanges = 1;
            rootParams[kGraphicsUAVSpaceParamBase + s].DescriptorTable.pDescriptorRanges   = &uavRanges[s];
            rootParams[kGraphicsUAVSpaceParamBase + s].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        }

        // Param 16: 32-Bit Root Constants at (b1, space3)
        rootParams[kGraphicsRootConstantsParamIndex].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.ShaderRegister = 1;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.RegisterSpace  = 3;
        rootParams[kGraphicsRootConstantsParamIndex].Constants.Num32BitValues = 4;
        rootParams[kGraphicsRootConstantsParamIndex].ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters     = kGraphicsRootParamCount;
        rootDesc.pParameters       = rootParams;
        rootDesc.NumStaticSamplers = 0;
        rootDesc.pStaticSamplers   = nullptr;
        rootDesc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> signatureBlob;
        ComPtr<ID3DBlob> errorBlob;
        if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureBlob, &errorBlob)))
        {
            if (errorBlob)
            {
                std::cerr << "[D3D12Pipeline] Indirect Root Signature Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            }
            return nullptr;
        }

        ComPtr<ID3D12RootSignature> createdRootSig;
        if (FAILED(device->CreateRootSignature(
                0,
                signatureBlob->GetBufferPointer(),
                signatureBlob->GetBufferSize(),
                IID_PPV_ARGS(&createdRootSig))))
        {
            return nullptr;
        }

        IndirectRootSignatureEntry& entry = m_indirectRootObjects[key];
        entry.rootSignature               = createdRootSig;
        return entry.rootSignature.Get();
    }

    ID3D12CommandSignature* D3D12Pipeline::GetOrCreateIndirectCommandSignature(
        ID3D12Device*                 device,
        const IndirectRootBindingKey& key)
    {
        if (!device) return nullptr;

        IndirectRootBindingKey effKey = key;
        effKey.enabled                = true;

        ID3D12RootSignature* rootSig = GetOrCreateIndirectRootSignature(device, effKey);
        if (!rootSig) return nullptr;

        IndirectRootSignatureEntry& entry = m_indirectRootObjects[effKey];
        if (entry.commandSignature)
        {
            return entry.commandSignature.Get();
        }

        D3D12_INDIRECT_ARGUMENT_DESC argDescs[7]{};
        argDescs[0].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
        argDescs[0].VertexBuffer.Slot                     = 0;

        argDescs[1].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;

        argDescs[2].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
        argDescs[2].ConstantBufferView.RootParameterIndex = kGraphicsIndirectCBVParamBase + 0;

        argDescs[3].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
        argDescs[3].ConstantBufferView.RootParameterIndex = kGraphicsIndirectCBVParamBase + 1;

        argDescs[4].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
        argDescs[4].ConstantBufferView.RootParameterIndex = kGraphicsIndirectCBVParamBase + 2;

        argDescs[5].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW;
        argDescs[5].ShaderResourceView.RootParameterIndex = kGraphicsIndirectSRVParamIndex;

        argDescs[6].Type                                  = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

        D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
        sigDesc.ByteStride       = kNVSIndirectCommandByteStride;
        sigDesc.NumArgumentDescs = 7;
        sigDesc.pArgumentDescs   = argDescs;
        sigDesc.NodeMask         = 0;

        ComPtr<ID3D12CommandSignature> cmdSig;
        if (FAILED(device->CreateCommandSignature(&sigDesc, rootSig, IID_PPV_ARGS(&cmdSig))))
        {
            return nullptr;
        }

        entry.commandSignature = cmdSig;
        return entry.commandSignature.Get();
    }

    BOOL D3D12Pipeline::CreateComputeRootSignature(ID3D12Device* device)
    {
        if (!device) return false;

        D3D12_DESCRIPTOR_RANGE csRanges[kMaxSpaces][3]{};
        D3D12_DESCRIPTOR_RANGE samplerRanges[kMaxSpaces]{};
        D3D12_ROOT_PARAMETER   rootParams[9]{};

        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            csRanges[s][0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            csRanges[s][0].NumDescriptors                    = kCBVPerSpace;
            csRanges[s][0].BaseShaderRegister                = 0;
            csRanges[s][0].RegisterSpace                     = s;
            csRanges[s][0].OffsetInDescriptorsFromTableStart = 0;

            csRanges[s][1].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            csRanges[s][1].NumDescriptors                    = kSRVPerSpace;
            csRanges[s][1].BaseShaderRegister                = 0;
            csRanges[s][1].RegisterSpace                     = s;
            csRanges[s][1].OffsetInDescriptorsFromTableStart = kCBVPerSpace;

            csRanges[s][2].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            csRanges[s][2].NumDescriptors                    = kUAVPerSpace;
            csRanges[s][2].BaseShaderRegister                = 0;
            csRanges[s][2].RegisterSpace                     = s;
            csRanges[s][2].OffsetInDescriptorsFromTableStart = kCBVPerSpace + kSRVPerSpace;

            rootParams[s].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParams[s].DescriptorTable.NumDescriptorRanges = 3;
            rootParams[s].DescriptorTable.pDescriptorRanges   = csRanges[s];
            rootParams[s].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        }

        for (UINT32 s = 0; s < kMaxSpaces; ++s)
        {
            samplerRanges[s].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            samplerRanges[s].NumDescriptors                    = 16;
            samplerRanges[s].BaseShaderRegister                = 0;
            samplerRanges[s].RegisterSpace                     = s;
            samplerRanges[s].OffsetInDescriptorsFromTableStart = 0;

            rootParams[kComputeSamplerParamBase + s].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParams[kComputeSamplerParamBase + s].DescriptorTable.NumDescriptorRanges = 1;
            rootParams[kComputeSamplerParamBase + s].DescriptorTable.pDescriptorRanges   = &samplerRanges[s];
            rootParams[kComputeSamplerParamBase + s].ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        }

        rootParams[kComputeRootConstantsParamIndex].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[kComputeRootConstantsParamIndex].Constants.ShaderRegister = 0;
        rootParams[kComputeRootConstantsParamIndex].Constants.RegisterSpace  = 4; // space4 (isolated from NSL space0..space3)
        rootParams[kComputeRootConstantsParamIndex].Constants.Num32BitValues = 4;
        rootParams[kComputeRootConstantsParamIndex].ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters     = 9;
        rootDesc.pParameters       = rootParams;
        rootDesc.NumStaticSamplers = 0;
        rootDesc.pStaticSamplers   = nullptr;
        rootDesc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> signatureBlob;
        ComPtr<ID3DBlob> errorBlob;
        if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureBlob, &errorBlob)))
        {
            if (errorBlob)
            {
                std::cerr << "[D3D12Pipeline] Compute Root Signature Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            }
            return false;
        }

        return SUCCEEDED(device->CreateRootSignature(
            0,
            signatureBlob->GetBufferPointer(),
            signatureBlob->GetBufferSize(),
            IID_PPV_ARGS(&m_computeRootSignature)
        ));
    }

    BOOL D3D12Pipeline::CreateCommandSignatures(ID3D12Device* device)
    {
        if (!device) return false;

        // 1. DrawInstanced (16 bytes — GPUIndirectDrawArgs)
        {
            D3D12_INDIRECT_ARGUMENT_DESC argDesc{};
            argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;

            D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
            sigDesc.ByteStride       = sizeof(GPUIndirectDrawArgs);
            sigDesc.NumArgumentDescs = 1;
            sigDesc.pArgumentDescs   = &argDesc;
            sigDesc.NodeMask         = 0;

            if (FAILED(device->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&m_drawIndirectSignature))))
            {
                return false;
            }
        }

        // 2. DrawIndexedInstanced (20 bytes — GPUIndirectDrawIndexedArgs)
        {
            D3D12_INDIRECT_ARGUMENT_DESC argDesc{};
            argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;

            D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
            sigDesc.ByteStride       = sizeof(GPUIndirectDrawIndexedArgs);
            sigDesc.NumArgumentDescs = 1;
            sigDesc.pArgumentDescs   = &argDesc;
            sigDesc.NodeMask         = 0;

            if (FAILED(device->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&m_drawIndexedIndirectSignature))))
            {
                return false;
            }
        }

        // 3. DispatchIndirect (12 bytes — GPUIndirectDispatchArgs)
        {
            D3D12_INDIRECT_ARGUMENT_DESC argDesc{};
            argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

            D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
            sigDesc.ByteStride       = sizeof(GPUIndirectDispatchArgs);
            sigDesc.NumArgumentDescs = 1;
            sigDesc.pArgumentDescs   = &argDesc;
            sigDesc.NodeMask         = 0;

            if (FAILED(device->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&m_dispatchIndirectSignature))))
            {
                return false;
            }
        }

        // 4. Default NVS GPU-Driven Multi-Mesh DrawIndexedInstanced (96 bytes — NVSIndirectDrawCommand)
        {
            IndirectRootBindingKey defaultNvsKey{};
            defaultNvsKey.enabled = true;
            ID3D12CommandSignature* nvsCmdSig = GetOrCreateIndirectCommandSignature(device, defaultNvsKey);
            if (!nvsCmdSig)
            {
                return false;
            }
            m_nvsDrawIndexedIndirectSignature = nvsCmdSig;
        }

        return true;
    }

    namespace
    {
        DXGI_FORMAT MapDataTypeToDXGIFormat(DataTypeID typeID)
        {
            switch (typeID)
            {
                case DataTypeID::FLOAT:  return DXGI_FORMAT_R32_FLOAT;
                case DataTypeID::VEC2:   return DXGI_FORMAT_R32G32_FLOAT;
                case DataTypeID::VEC3:   return DXGI_FORMAT_R32G32B32_FLOAT;
                case DataTypeID::VEC4:   return DXGI_FORMAT_R32G32B32A32_FLOAT;
                case DataTypeID::INT:    return DXGI_FORMAT_R32_SINT;
                case DataTypeID::INT2:   return DXGI_FORMAT_R32G32_SINT;
                case DataTypeID::INT3:   return DXGI_FORMAT_R32G32B32_SINT;
                case DataTypeID::INT4:   return DXGI_FORMAT_R32G32B32A32_SINT;
                case DataTypeID::UINT:   return DXGI_FORMAT_R32_UINT;
                case DataTypeID::UINT2:  return DXGI_FORMAT_R32G32_UINT;
                case DataTypeID::UINT3:  return DXGI_FORMAT_R32G32B32_UINT;
                case DataTypeID::UINT4:  return DXGI_FORMAT_R32G32B32A32_UINT;
                default:                 return DXGI_FORMAT_UNKNOWN;
            }
        }

        UINT32 MapDataTypeByteSize(DataTypeID typeID)
        {
            switch (typeID)
            {
                case DataTypeID::FLOAT:
                case DataTypeID::INT:
                case DataTypeID::UINT:   return 4u;
                case DataTypeID::VEC2:
                case DataTypeID::INT2:
                case DataTypeID::UINT2:  return 8u;
                case DataTypeID::VEC3:
                case DataTypeID::INT3:
                case DataTypeID::UINT3:  return 12u;
                case DataTypeID::VEC4:
                case DataTypeID::INT4:
                case DataTypeID::UINT4:  return 16u;
                default:                 return 0u;
            }
        }

        UINT32 MapSemanticToAlignedOffset(const std::string& name, const std::string& semantic)
        {
            if (name == "VERTEX_POSITION"     || semantic == "POSITION"     || semantic == "VERTEX_POSITION")     return 0u;
            if (name == "VERTEX_NORMAL"       || semantic == "NORMAL"       || semantic == "VERTEX_NORMAL")       return 12u;
            if (name == "VERTEX_TANGENT"      || semantic == "TANGENT"      || semantic == "VERTEX_TANGENT")      return 24u;
            if (name == "VERTEX_BITANGENT"    || semantic == "BINORMAL"     || semantic == "VERTEX_BITANGENT")    return 36u;
            if (name == "VERTEX_UV0"          || semantic == "TEXCOORD0"    || semantic == "VERTEX_UV0")          return 48u;
            if (name == "VERTEX_UV1"          || semantic == "TEXCOORD1"    || semantic == "VERTEX_UV1")          return 56u;
            if (name == "VERTEX_COLOR"        || name == "VERTEX_COLOR0"    || semantic == "COLOR0" || semantic == "COLOR" || semantic == "VERTEX_COLOR" || semantic == "VERTEX_COLOR0") return 64u;
            if (name == "VERTEX_BONE_INDICES" || semantic == "BLENDINDICES" || semantic == "VERTEX_BONE_INDICES") return 80u;
            if (name == "VERTEX_BONE_WEIGHTS" || semantic == "BLENDWEIGHT"  || semantic == "VERTEX_BONE_WEIGHTS") return 96u;
            // Attributes outside Northstar's 112-byte Vertex contract (e.g. VERTEX_UV2/UV3/COLOR1) are rejected explicitly
            return UINT32_MAX;
        }

        void ParseSemanticNameAndIndex(const std::string& rawSemantic, const std::string& name, std::string& outName, UINT32& outIndex)
        {
            if (name == "VERTEX_POSITION"     || rawSemantic == "POSITION")     { outName = "POSITION";     outIndex = 0; return; }
            if (name == "VERTEX_NORMAL"       || rawSemantic == "NORMAL")       { outName = "NORMAL";       outIndex = 0; return; }
            if (name == "VERTEX_TANGENT"      || rawSemantic == "TANGENT")      { outName = "TANGENT";      outIndex = 0; return; }
            if (name == "VERTEX_BITANGENT"    || rawSemantic == "BINORMAL")     { outName = "BINORMAL";     outIndex = 0; return; }
            if (name == "VERTEX_UV0"          || rawSemantic == "TEXCOORD0")    { outName = "TEXCOORD";     outIndex = 0; return; }
            if (name == "VERTEX_UV1"          || rawSemantic == "TEXCOORD1")    { outName = "TEXCOORD";     outIndex = 1; return; }
            if (name == "VERTEX_COLOR" || name == "VERTEX_COLOR0" || rawSemantic == "COLOR0" || rawSemantic == "COLOR") { outName = "COLOR"; outIndex = 0; return; }
            if (name == "VERTEX_BONE_INDICES" || rawSemantic == "BLENDINDICES") { outName = "BLENDINDICES"; outIndex = 0; return; }
            if (name == "VERTEX_BONE_WEIGHTS" || rawSemantic == "BLENDWEIGHT")  { outName = "BLENDWEIGHT";  outIndex = 0; return; }

            std::string target = !rawSemantic.empty() ? rawSemantic : name;
            outIndex = 0;
            if (!target.empty() && std::isdigit(static_cast<unsigned char>(target.back())))
            {
                size_t lastNonDigit = target.find_last_not_of("0123456789");
                if (lastNonDigit != std::string::npos)
                {
                    outIndex = static_cast<UINT32>(std::stoul(target.substr(lastNonDigit + 1)));
                    outName = target.substr(0, lastNonDigit + 1);
                    return;
                }
            }
            outName = target;
        }
    }

    ID3D12PipelineState* D3D12Pipeline::GetOrCreatePSO(
        ID3D12Device* device,
        const D3D12ResourceManager& resMgr,
        NativeHandle vsHandle,
        NativeHandle psHandle,
        CullMode cullMode,
        BlendMode blendMode,
        FillMode fillMode,
        const DepthState& depthState,
        BOOL shadowDepthBias,
        INT32 customDepthBias,
        FLOAT32 customSlopeScaledDepthBias,
        FLOAT32 customDepthBiasClamp,
        NativeHandle gsHandle,
        BOOL depthOnlyTarget,
        NativeHandle hsHandle,
        NativeHandle dsHandle,
        const IndirectRootBindingKey& indirectRootKey,
        DXGI_FORMAT rtvFormat,
        DXGI_FORMAT dsvFormat)
    {
        if (!device || !m_rootSignature) return nullptr;

        ID3D12RootSignature* targetRootSig = indirectRootKey.enabled
            ? GetOrCreateIndirectRootSignature(device, indirectRootKey)
            : m_rootSignature.Get();
        if (!targetRootSig)
        {
            return nullptr;
        }

        const BOOL effectiveDepthOnly = (depthOnlyTarget != FALSE) || (psHandle == 0);

        const D3D12ShaderResource* vsRes = resMgr.GetShader(vsHandle);
        const D3D12ShaderResource* psRes = (psHandle != 0) ? resMgr.GetShader(psHandle) : nullptr;
        const D3D12ShaderResource* gsRes = (gsHandle != 0) ? resMgr.GetShader(gsHandle) : nullptr;
        const D3D12ShaderResource* hsRes = (hsHandle != 0) ? resMgr.GetShader(hsHandle) : nullptr;
        const D3D12ShaderResource* dsRes = (dsHandle != 0) ? resMgr.GetShader(dsHandle) : nullptr;
        if (!vsRes) return nullptr;
        if (psHandle != 0 && !psRes) return nullptr;
        if (gsHandle != 0 && !gsRes) return nullptr;
        if (hsHandle != 0 && !hsRes) return nullptr;
        if (dsHandle != 0 && !dsRes) return nullptr;

        UINT64 genHash = vsRes->generation;
        if (psRes) genHash ^= psRes->generation + 0x9e3779b97f4a7c15ULL + (genHash << 6) + (genHash >> 2);
        if (gsRes) genHash ^= gsRes->generation + 0x9e3779b97f4a7c15ULL + (genHash << 6) + (genHash >> 2);
        if (hsRes) genHash ^= hsRes->generation + 0x9e3779b97f4a7c15ULL + (genHash << 6) + (genHash >> 2);
        if (dsRes) genHash ^= dsRes->generation + 0x9e3779b97f4a7c15ULL + (genHash << 6) + (genHash >> 2);

        UINT64 ilHash = 0;
        for (const auto& entry : vsRes->inputLayout)
        {
            ilHash ^= std::hash<std::string>{}(entry.name) + 0x9e3779b97f4a7c15ULL + (ilHash << 6) + (ilHash >> 2);
            ilHash ^= std::hash<std::string>{}(entry.semantic) + 0x9e3779b97f4a7c15ULL + (ilHash << 6) + (ilHash >> 2);
            ilHash ^= static_cast<UINT64>(entry.typeID) + 0x9e3779b97f4a7c15ULL + (ilHash << 6) + (ilHash >> 2);
        }

        const D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType = (hsHandle != 0 || dsHandle != 0)
            ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH
            : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        const DXGI_FORMAT resolvedRtvFmt = (rtvFormat != DXGI_FORMAT_UNKNOWN) ? rtvFormat : DXGI_FORMAT_R8G8B8A8_UNORM;
        const DXGI_FORMAT rtvFmt0        = (!effectiveDepthOnly && psRes) ? resolvedRtvFmt : DXGI_FORMAT_UNKNOWN;
        const BOOL        hasValidDsv    = (dsvFormat != DXGI_FORMAT_UNKNOWN);
        const BOOL        effDepthTest   = hasValidDsv && (depthState.test != FALSE);
        const BOOL        effDepthWrite  = hasValidDsv && (depthState.write != FALSE);
        const DXGI_FORMAT dsvFmt         = hasValidDsv ? dsvFormat : DXGI_FORMAT_UNKNOWN;

        PSOCacheKey key{};
        key.vsHandle              = vsHandle;
        key.psHandle              = psHandle;
        key.gsHandle              = gsHandle;
        key.hsHandle              = hsHandle;
        key.dsHandle              = dsHandle;
        key.shaderGenerationHash  = genHash;
        key.inputLayoutHash       = ilHash;
        key.cullMode              = cullMode;
        key.blendMode             = blendMode;
        key.fillMode              = fillMode;
        key.depthTest             = effDepthTest;
        key.depthWrite            = effDepthWrite;
        key.depthCompare          = depthState.compare;
        key.shadowDepthBias       = ((!psHandle) && (shadowDepthBias != FALSE));
        key.depthOnlyTarget       = effectiveDepthOnly;
        key.depthBias             = customDepthBias;
        key.slopeScaledDepthBias  = customSlopeScaledDepthBias;
        key.depthBiasClamp        = customDepthBiasClamp;
        key.primitiveTopologyType = topoType;
        key.rtvFormat0            = rtvFmt0;
        key.dsvFormat             = dsvFmt;
        key.indirectRootKey       = indirectRootKey;

        auto it = m_psoCache.find(key);
        if (it != m_psoCache.end())
        {
            ++m_psoCacheHits;
            ++m_totalPsoCacheHits;
            return it->second.Get();
        }

        ++m_psoCacheMisses;
        ++m_totalPsoCacheMisses;

        const void* vsPtr = vsRes->blob ? vsRes->blob->GetBufferPointer() : vsRes->bytecode.data();
        SIZE_T      vsSize = vsRes->blob ? vsRes->blob->GetBufferSize() : vsRes->bytecode.size();
        const void* psPtr = (psRes && (psRes->blob || !psRes->bytecode.empty()))
            ? (psRes->blob ? psRes->blob->GetBufferPointer() : psRes->bytecode.data())
            : nullptr;
        SIZE_T      psSize = (psRes && (psRes->blob || !psRes->bytecode.empty()))
            ? (psRes->blob ? psRes->blob->GetBufferSize() : psRes->bytecode.size())
            : 0;
        const void* gsPtr = (gsRes && (gsRes->blob || !gsRes->bytecode.empty()))
            ? (gsRes->blob ? gsRes->blob->GetBufferPointer() : gsRes->bytecode.data())
            : nullptr;
        SIZE_T      gsSize = (gsRes && (gsRes->blob || !gsRes->bytecode.empty()))
            ? (gsRes->blob ? gsRes->blob->GetBufferSize() : gsRes->bytecode.size())
            : 0;
        const void* hsPtr = (hsRes && (hsRes->blob || !hsRes->bytecode.empty()))
            ? (hsRes->blob ? hsRes->blob->GetBufferPointer() : hsRes->bytecode.data())
            : nullptr;
        SIZE_T      hsSize = (hsRes && (hsRes->blob || !hsRes->bytecode.empty()))
            ? (hsRes->blob ? hsRes->blob->GetBufferSize() : hsRes->bytecode.size())
            : 0;
        const void* dsPtr = (dsRes && (dsRes->blob || !dsRes->bytecode.empty()))
            ? (dsRes->blob ? dsRes->blob->GetBufferPointer() : dsRes->bytecode.data())
            : nullptr;
        SIZE_T      dsSize = (dsRes && (dsRes->blob || !dsRes->bytecode.empty()))
            ? (dsRes->blob ? dsRes->blob->GetBufferSize() : dsRes->bytecode.size())
            : 0;

        if (!vsPtr || vsSize == 0) return nullptr;
        if (psHandle != 0 && (!psPtr || psSize == 0)) return nullptr;
        if (gsHandle != 0 && (!gsPtr || gsSize == 0)) return nullptr;
        if (hsHandle != 0 && (!hsPtr || hsSize == 0)) return nullptr;
        if (dsHandle != 0 && (!dsPtr || dsSize == 0)) return nullptr;

        const D3D12_INPUT_ELEMENT_DESC defaultInputLayout[] = {
            { "POSITION",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "NORMAL",       0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TANGENT",      0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "BINORMAL",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD",     0, DXGI_FORMAT_R32G32_FLOAT,       0, 48, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD",     1, DXGI_FORMAT_R32G32_FLOAT,       0, 56, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "COLOR",        0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 64, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "BLENDINDICES", 0, DXGI_FORMAT_R32G32B32A32_UINT,  0, 80, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "BLENDWEIGHT",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 96, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        };

        std::vector<D3D12_INPUT_ELEMENT_DESC> dynamicInputLayout;
        std::vector<std::string> semanticNameStorage;

        if (!vsRes->inputLayout.empty())
        {
            semanticNameStorage.reserve(vsRes->inputLayout.size());
            dynamicInputLayout.reserve(vsRes->inputLayout.size());

            for (const auto& entry : vsRes->inputLayout)
            {
                if (entry.category == 2 || entry.name == "VERTEX_ID" || entry.name == "INSTANCE_ID" || entry.semantic.rfind("SV_", 0) == 0)
                {
                    continue;
                }

                const UINT32 alignedOffset = MapSemanticToAlignedOffset(entry.name, entry.semantic);
                const UINT32 elemByteSize  = MapDataTypeByteSize(entry.typeID);
                const DXGI_FORMAT dxgiFmt  = MapDataTypeToDXGIFormat(entry.typeID);
                if (alignedOffset == UINT32_MAX || elemByteSize == 0u || dxgiFmt == DXGI_FORMAT_UNKNOWN ||
                    (alignedOffset + elemByteSize) > sizeof(Vertex))
                {
                    std::cerr << "[D3D12Pipeline] Rejected vertex input attribute '" << entry.name
                              << "' (semantic='" << entry.semantic << "'): out of 112-byte Vertex bounds.\n";
                    return nullptr;
                }

                std::string sName;
                UINT32 sIndex = 0;
                ParseSemanticNameAndIndex(entry.semantic, entry.name, sName, sIndex);
                semanticNameStorage.push_back(std::move(sName));

                D3D12_INPUT_ELEMENT_DESC elem{};
                elem.SemanticName         = semanticNameStorage.back().c_str();
                elem.SemanticIndex        = sIndex;
                elem.Format               = dxgiFmt;
                elem.InputSlot            = 0;
                elem.AlignedByteOffset    = alignedOffset;
                elem.InputSlotClass       = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
                elem.InstanceDataStepRate = 0;

                dynamicInputLayout.push_back(elem);
            }
        }

        const D3D12_INPUT_ELEMENT_DESC* chosenLayout = vsRes->inputLayout.empty() ? defaultInputLayout : dynamicInputLayout.data();
        const UINT32 chosenLayoutCount = vsRes->inputLayout.empty() ? _countof(defaultInputLayout) : static_cast<UINT32>(dynamicInputLayout.size());

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = targetRootSig;
        psoDesc.VS = { vsPtr, vsSize };
        psoDesc.PS = { psPtr, psSize };
        psoDesc.GS = { gsPtr, gsSize };
        psoDesc.HS = { hsPtr, hsSize };
        psoDesc.DS = { dsPtr, dsSize };
        psoDesc.InputLayout = { chosenLayout, chosenLayoutCount };
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = topoType;
        if (!effectiveDepthOnly && psPtr && psSize > 0)
        {
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = rtvFmt0;
        }
        else
        {
            psoDesc.NumRenderTargets = 0;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        }
        psoDesc.DSVFormat = dsvFmt;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.SampleDesc.Quality = 0;

        psoDesc.RasterizerState.FillMode = (fillMode == FillMode::WIREFRAME) ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
        if (cullMode == CullMode::NONE)
            psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        else if (cullMode == CullMode::FRONT)
            psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
        else
            psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;

        psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
        if (customDepthBias != 0 || customSlopeScaledDepthBias != 0.0f || customDepthBiasClamp != 0.0f)
        {
            psoDesc.RasterizerState.DepthBias            = customDepthBias;
            psoDesc.RasterizerState.DepthBiasClamp       = customDepthBiasClamp;
            psoDesc.RasterizerState.SlopeScaledDepthBias = customSlopeScaledDepthBias;
        }
        else if ((!psPtr || psSize == 0) && shadowDepthBias)
        {
            const BOOL isReversedZPass =
                (depthState.compare == DepthCompare::GREATER ||
                 depthState.compare == DepthCompare::GREATER_EQUAL);
            psoDesc.RasterizerState.DepthBias            = isReversedZPass ? -1 : 1;
            psoDesc.RasterizerState.DepthBiasClamp       = 0.0f;
            psoDesc.RasterizerState.SlopeScaledDepthBias = isReversedZPass ? -1.0f : 1.0f;
        }
        else
        {
            psoDesc.RasterizerState.DepthBias            = D3D12_DEFAULT_DEPTH_BIAS;
            psoDesc.RasterizerState.DepthBiasClamp       = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
            psoDesc.RasterizerState.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        }
        psoDesc.RasterizerState.DepthClipEnable = TRUE;

        psoDesc.BlendState.AlphaToCoverageEnable  = FALSE;
        psoDesc.BlendState.IndependentBlendEnable = FALSE;
        D3D12_RENDER_TARGET_BLEND_DESC& rtBlend   = psoDesc.BlendState.RenderTarget[0];
        rtBlend.RenderTargetWriteMask             = effectiveDepthOnly ? 0 : D3D12_COLOR_WRITE_ENABLE_ALL;

        if (blendMode == BlendMode::ALPHA_BLEND)
        {
            rtBlend.BlendEnable    = TRUE;
            rtBlend.SrcBlend       = D3D12_BLEND_SRC_ALPHA;
            rtBlend.DestBlend      = D3D12_BLEND_INV_SRC_ALPHA;
            rtBlend.BlendOp        = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha  = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha   = D3D12_BLEND_OP_ADD;
        }
        else if (blendMode == BlendMode::ADDITIVE)
        {
            rtBlend.BlendEnable    = TRUE;
            rtBlend.SrcBlend       = D3D12_BLEND_ONE;
            rtBlend.DestBlend      = D3D12_BLEND_ONE;
            rtBlend.BlendOp        = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha  = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha   = D3D12_BLEND_OP_ADD;
        }
        else if (blendMode == BlendMode::MULTIPLY)
        {
            rtBlend.BlendEnable    = TRUE;
            rtBlend.SrcBlend       = D3D12_BLEND_DEST_COLOR;
            rtBlend.DestBlend      = D3D12_BLEND_ZERO;
            rtBlend.BlendOp        = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha  = D3D12_BLEND_DEST_ALPHA;
            rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
            rtBlend.BlendOpAlpha   = D3D12_BLEND_OP_ADD;
        }
        else if (blendMode == BlendMode::PREMULTIPLIED_ALPHA)
        {
            rtBlend.BlendEnable    = TRUE;
            rtBlend.SrcBlend       = D3D12_BLEND_ONE;
            rtBlend.DestBlend      = D3D12_BLEND_INV_SRC_ALPHA;
            rtBlend.BlendOp        = D3D12_BLEND_OP_ADD;
            rtBlend.SrcBlendAlpha  = D3D12_BLEND_ONE;
            rtBlend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
            rtBlend.BlendOpAlpha   = D3D12_BLEND_OP_ADD;
        }
        else
        {
            rtBlend.BlendEnable = FALSE;
        }

        static auto mapDepthFunc = [](DepthCompare cmp) -> D3D12_COMPARISON_FUNC {
            switch (cmp)
            {
                case DepthCompare::NEVER:         return D3D12_COMPARISON_FUNC_NEVER;
                case DepthCompare::LESS:          return D3D12_COMPARISON_FUNC_LESS;
                case DepthCompare::EQUAL:         return D3D12_COMPARISON_FUNC_EQUAL;
                case DepthCompare::LESS_EQUAL:    return D3D12_COMPARISON_FUNC_LESS_EQUAL;
                case DepthCompare::GREATER:       return D3D12_COMPARISON_FUNC_GREATER;
                case DepthCompare::GREATER_EQUAL: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
                case DepthCompare::NOT_EQUAL:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
                case DepthCompare::ALWAYS:        return D3D12_COMPARISON_FUNC_ALWAYS;
                default:                          return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            }
        };

        psoDesc.DepthStencilState.DepthEnable    = effDepthTest ? TRUE : FALSE;
        psoDesc.DepthStencilState.DepthWriteMask = effDepthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc      = mapDepthFunc(depthState.compare);
        psoDesc.DepthStencilState.StencilEnable  = FALSE;

        ComPtr<ID3D12PipelineState> pso;
        HRESULT hr = device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
        if (FAILED(hr))
        {
            std::cerr << "[D3D12Pipeline] Failed to create PSO: 0x" << std::hex << hr << std::dec << "\n";
            return nullptr;
        }

        m_psoCache[key] = pso;
        return pso.Get();
    }

    ID3D12PipelineState* D3D12Pipeline::GetOrCreateComputePSO(
        ID3D12Device* device,
        const D3D12ResourceManager& resMgr,
        NativeHandle csHandle)
    {
        if (!device || !m_computeRootSignature || csHandle == 0) return nullptr;

        const D3D12ShaderResource* csRes = resMgr.GetShader(csHandle);
        if (!csRes || csRes->stage != ShaderStage::COMPUTE)
        {
            return nullptr;
        }

        auto it = m_computePsoCache.find(csHandle);
        if (it != m_computePsoCache.end())
        {
            return it->second.Get();
        }

        const void* csPtr  = csRes->blob ? csRes->blob->GetBufferPointer() : csRes->bytecode.data();
        SIZE_T      csSize = csRes->blob ? csRes->blob->GetBufferSize()    : csRes->bytecode.size();
        if (!csPtr || csSize == 0)
        {
            return nullptr;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = m_computeRootSignature.Get();
        psoDesc.CS             = { csPtr, csSize };
        psoDesc.NodeMask       = 0;
        psoDesc.Flags          = D3D12_PIPELINE_STATE_FLAG_NONE;

        ComPtr<ID3D12PipelineState> pso;
        HRESULT hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
        if (FAILED(hr) || !pso)
        {
            std::cerr << "[D3D12Pipeline] Failed to create Compute PSO: 0x" << std::hex << hr << std::dec << "\n";
            return nullptr;
        }

        m_computePsoCache[csHandle] = pso;
        return pso.Get();
    }

    void D3D12Pipeline::InvalidatePSOsForShader(NativeHandle shaderHandle, UINT64 retireFenceValue)
    {
        if (shaderHandle == 0) return;
        const UINT64 safeFence = (std::max)(retireFenceValue, m_currentRetireFenceValue);

        for (auto it = m_psoCache.begin(); it != m_psoCache.end(); )
        {
            const PSOCacheKey& k = it->first;
            if (k.vsHandle == shaderHandle ||
                k.psHandle == shaderHandle ||
                k.gsHandle == shaderHandle ||
                k.hsHandle == shaderHandle ||
                k.dsHandle == shaderHandle)
            {
                if (it->second)
                {
                    m_retiredPSOs.push_back({ safeFence, std::move(it->second) });
                }
                it = m_psoCache.erase(it);
            }
            else
            {
                ++it;
            }
        }

        auto csIt = m_computePsoCache.find(shaderHandle);
        if (csIt != m_computePsoCache.end())
        {
            if (csIt->second)
            {
                m_retiredPSOs.push_back({ safeFence, std::move(csIt->second) });
            }
            m_computePsoCache.erase(csIt);
        }
    }

    UINT32 D3D12Pipeline::GetRetiredPSOCount() const
    {
        return static_cast<UINT32>(m_retiredPSOs.size());
    }

    BOOL D3D12Pipeline::CreateInstanceBuffer(ID3D12Device* device)
    {
        if (!device) return false;

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(m_instanceRingSize);

        if (FAILED(device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_instanceUploadBuffer))))
        {
            return false;
        }

        D3D12_RANGE readRange{ 0, 0 };
        return SUCCEEDED(m_instanceUploadBuffer->Map(0, &readRange, &m_instanceMappedData));
    }

    BOOL D3D12Pipeline::GrowInstanceBuffer(UINT64 minRequiredCount)
    {
        if (!m_device) return false;

        UINT64 targetCount = m_instanceRingCount * 2;
        if (targetCount < minRequiredCount * kBackBufferCount)
        {
            targetCount = minRequiredCount * kBackBufferCount;
        }
        const UINT64 targetSize = sizeof(D3D12InstancePayload) * targetCount;

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(targetSize);

        ComPtr<ID3D12Resource> newBuffer;
        if (FAILED(m_device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&newBuffer))))
        {
            return false;
        }

        void* newMapped = nullptr;
        D3D12_RANGE readRange{ 0, 0 };
        if (FAILED(newBuffer->Map(0, &readRange, &newMapped)) || !newMapped)
        {
            return false;
        }

        if (m_instanceUploadBuffer)
        {
            if (m_instanceMappedData)
            {
                m_instanceUploadBuffer->Unmap(0, nullptr);
                m_instanceMappedData = nullptr;
            }
            m_retiredBuffers.push_back({ m_currentRetireFenceValue, std::move(m_instanceUploadBuffer) });
            m_instanceUploadBuffer.Reset();
        }

        m_instanceRingCount    = targetCount;
        m_instanceRingSize     = targetSize;
        m_instanceUploadBuffer = std::move(newBuffer);
        m_instanceMappedData   = newMapped;
        m_instanceOffset       = m_activeBB * (m_instanceRingSize / kBackBufferCount);
        return true;
    }

    UINT32 D3D12Pipeline::AllocateInstances(const D3D12InstancePayload* data, UINT32 count)
    {
        if (!data || count == 0) return 0;

        UINT64 instPerBB = m_instanceRingCount / kBackBufferCount;
        UINT64 byteSize = static_cast<UINT64>(count) * sizeof(D3D12InstancePayload);
        UINT64 bbBase = m_activeBB * (m_instanceRingSize / kBackBufferCount);
        UINT64 bbLimit = bbBase + (m_instanceRingSize / kBackBufferCount);

        if (count > instPerBB || m_instanceOffset < bbBase || m_instanceOffset + byteSize > bbLimit)
        {
            if (!GrowInstanceBuffer(count))
            {
                return 0;
            }
            instPerBB = m_instanceRingCount / kBackBufferCount;
            bbBase    = m_activeBB * (m_instanceRingSize / kBackBufferCount);
            bbLimit   = bbBase + (m_instanceRingSize / kBackBufferCount);
        }

        if (!m_instanceMappedData || m_instanceOffset + byteSize > bbLimit) return 0;

        UINT8* dest = static_cast<UINT8*>(m_instanceMappedData) + m_instanceOffset;
        std::memcpy(dest, data, byteSize);

        UINT32 firstElement = static_cast<UINT32>(m_instanceOffset / sizeof(D3D12InstancePayload));
        m_instanceOffset += byteSize;
        return firstElement;
    }

    BOOL D3D12Pipeline::CreateCBufferRing(ID3D12Device* device)
    {
        if (!device) return false;

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(m_cbufferRingSize);

        if (FAILED(device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_cbufferUploadBuffer))))
        {
            return false;
        }

        D3D12_RANGE readRange{ 0, 0 };
        return SUCCEEDED(m_cbufferUploadBuffer->Map(0, &readRange, &m_cbufferMappedData));
    }

    BOOL D3D12Pipeline::GrowCBuffer(UINT64 minRequiredBytes)
    {
        if (!m_device) return false;

        UINT64 targetSize = m_cbufferRingSize * 2;
        if (targetSize < minRequiredBytes * kBackBufferCount)
        {
            targetSize = minRequiredBytes * kBackBufferCount;
        }

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(targetSize);

        ComPtr<ID3D12Resource> newBuffer;
        if (FAILED(m_device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&newBuffer))))
        {
            return false;
        }

        void* newMapped = nullptr;
        D3D12_RANGE readRange{ 0, 0 };
        if (FAILED(newBuffer->Map(0, &readRange, &newMapped)) || !newMapped)
        {
            return false;
        }

        if (m_cbufferUploadBuffer)
        {
            if (m_cbufferMappedData)
            {
                m_cbufferUploadBuffer->Unmap(0, nullptr);
                m_cbufferMappedData = nullptr;
            }
            m_retiredBuffers.push_back({ m_currentRetireFenceValue, std::move(m_cbufferUploadBuffer) });
            m_cbufferUploadBuffer.Reset();
        }

        m_cbufferRingSize     = targetSize;
        m_cbufferUploadBuffer = std::move(newBuffer);
        m_cbufferMappedData   = newMapped;
        m_cbufferOffset       = m_activeBB * (m_cbufferRingSize / kBackBufferCount);
        return true;
    }

    D3D12_GPU_VIRTUAL_ADDRESS D3D12Pipeline::AllocateCBuffer(const void* data, UINT32 byteSize)
    {
        if (!data || byteSize == 0) return 0;

        const UINT64 alignedSize = (static_cast<UINT64>(byteSize) + 255ULL) & ~255ULL;
        UINT64 cbufferPerBB = m_cbufferRingSize / kBackBufferCount;
        UINT64 bbBase = m_activeBB * cbufferPerBB;
        UINT64 bbLimit = bbBase + cbufferPerBB;

        if (alignedSize > cbufferPerBB || m_cbufferOffset < bbBase || m_cbufferOffset + alignedSize > bbLimit)
        {
            if (!GrowCBuffer(alignedSize))
            {
                return 0;
            }
            cbufferPerBB = m_cbufferRingSize / kBackBufferCount;
            bbBase       = m_activeBB * cbufferPerBB;
            bbLimit      = bbBase + cbufferPerBB;
        }

        if (!m_cbufferMappedData || m_cbufferOffset + alignedSize > bbLimit) return 0;

        UINT8* dest = static_cast<UINT8*>(m_cbufferMappedData) + m_cbufferOffset;
        std::memcpy(dest, data, byteSize);

        D3D12_GPU_VIRTUAL_ADDRESS gpuVA = m_cbufferUploadBuffer->GetGPUVirtualAddress() + m_cbufferOffset;
        m_cbufferOffset += alignedSize;
        return gpuVA;
    }

    BOOL D3D12Pipeline::CreateSBufferRing(ID3D12Device* device)
    {
        if (!device) return false;

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(m_sbufferRingSize);

        if (FAILED(device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_sbufferUploadBuffer))))
        {
            return false;
        }

        D3D12_RANGE readRange{ 0, 0 };
        return SUCCEEDED(m_sbufferUploadBuffer->Map(0, &readRange, &m_sbufferMappedData));
    }

    BOOL D3D12Pipeline::GrowSBuffer(UINT64 minRequiredBytes)
    {
        if (!m_device) return false;

        UINT64 targetSize = m_sbufferRingSize * 2;
        if (targetSize < minRequiredBytes * kBackBufferCount)
        {
            targetSize = minRequiredBytes * kBackBufferCount;
        }

        D3D12_HEAP_PROPERTIES uploadProps = GetUploadHeapProperties();
        D3D12_RESOURCE_DESC bufferDesc = CreateBufferResourceDesc(targetSize);

        ComPtr<ID3D12Resource> newBuffer;
        if (FAILED(m_device->CreateCommittedResource(
            &uploadProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&newBuffer))))
        {
            return false;
        }

        void* newMapped = nullptr;
        D3D12_RANGE readRange{ 0, 0 };
        if (FAILED(newBuffer->Map(0, &readRange, &newMapped)) || !newMapped)
        {
            return false;
        }

        if (m_sbufferUploadBuffer)
        {
            if (m_sbufferMappedData)
            {
                m_sbufferUploadBuffer->Unmap(0, nullptr);
                m_sbufferMappedData = nullptr;
            }
            m_retiredBuffers.push_back({ m_currentRetireFenceValue, std::move(m_sbufferUploadBuffer) });
            m_sbufferUploadBuffer.Reset();
        }

        m_sbufferRingSize     = targetSize;
        m_sbufferUploadBuffer = std::move(newBuffer);
        m_sbufferMappedData   = newMapped;
        m_sbufferOffset       = m_activeBB * (m_sbufferRingSize / kBackBufferCount);
        return true;
    }

    D3D12_GPU_VIRTUAL_ADDRESS D3D12Pipeline::AllocateStructuredBuffer(const void* data, UINT32 byteSize, UINT32 stride)
    {
        if (!data || byteSize == 0) return 0;

        const UINT64 align = (stride > 0) ? static_cast<UINT64>(stride) : 16ULL;
        UINT64 sbufferPerBB = m_sbufferRingSize / kBackBufferCount;
        UINT64 bbBase = m_activeBB * sbufferPerBB;
        UINT64 bbLimit = bbBase + sbufferPerBB;

        UINT64 alignedOffset = (m_sbufferOffset < bbBase) ? bbBase : m_sbufferOffset;
        alignedOffset = ((alignedOffset + align - 1ULL) / align) * align;

        if (static_cast<UINT64>(byteSize) + align > sbufferPerBB ||
            alignedOffset > bbLimit ||
            static_cast<UINT64>(byteSize) > (bbLimit - alignedOffset))
        {
            if (!GrowSBuffer(static_cast<UINT64>(byteSize) + align))
            {
                return 0;
            }
            sbufferPerBB  = m_sbufferRingSize / kBackBufferCount;
            bbBase        = m_activeBB * sbufferPerBB;
            bbLimit       = bbBase + sbufferPerBB;
            alignedOffset = ((bbBase + align - 1ULL) / align) * align;
        }

        if (!m_sbufferMappedData || alignedOffset > bbLimit || static_cast<UINT64>(byteSize) > (bbLimit - alignedOffset))
        {
            return 0;
        }

        UINT8* dest = static_cast<UINT8*>(m_sbufferMappedData) + alignedOffset;
        std::memcpy(dest, data, byteSize);

        D3D12_GPU_VIRTUAL_ADDRESS gpuVA = m_sbufferUploadBuffer->GetGPUVirtualAddress() + alignedOffset;
        m_sbufferOffset = alignedOffset + byteSize;
        return gpuVA;
    }
}
