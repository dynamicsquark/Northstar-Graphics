#include "core_nvg_manager.h"
#include "../runtime/core_render_pipeline.h"
#include "../runtime/core_nvs_shadow_rasterizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace Northstar
{
    namespace
    {
        inline constexpr UINT32 kNVGNodeQueueCap            = 524288u;
        inline constexpr UINT32 kNVGCandidateCap            = 1048576u;
        inline constexpr UINT32 kNVGVisibleCap              = 524288u;
        inline constexpr UINT32 kNVGShadowVisibleCap        = 1048576u;
        inline constexpr UINT32 kNVGMaxQueueCap             = 8388480u; // Grid-stride 2x 65535*64 threads for Billion-Polygon bursts
        inline constexpr UINT32 kNVGMaxVisibleCap           = 8388480u; // Scales dynamically up to 8.38M camera visible clusters
        inline constexpr UINT32 kNVGMaxShadowVisibleCap     = 8388480u; // Scales up to 8.38M shadow visible clusters on cold Billion-Polygon bursts
        inline constexpr UINT32 kNVGOccludedObjCap          = 65536u;
        inline constexpr UINT32 kNVGOccludedClusterCap      = 524288u;
        inline constexpr UINT32 kNVGDefaultDecodedVerts     = 524288u;  // Baseline fallback capacity; scaled dynamically for Path-B
        inline constexpr UINT32 kNVGDefaultDecodedIdxs      = 1572864u;
        inline constexpr UINT32 kNVGMinimalPullDecodedVerts = 8192u;    // Minimal placeholder when only Path-A Vertex-Pull buckets exist
        inline constexpr UINT32 kNVGMinimalPullDecodedIdxs  = 24576u;
        inline constexpr UINT32 kNVGMaxUniqueJobs           = 32768u;
        inline constexpr UINT32 kNVGPerDrawCBVWords         = 8388608u; // 32 MiB -> up to 65536 promoted 256B CBVs (grows up to 128 MiB)
        inline constexpr UINT32 kNVGMaxPerDrawCBVWords      = 33554432u;// 128 MiB -> up to 262144 promoted 256B CBVs
        inline constexpr UINT32 kNVGPerBatchInstWords       = 1048576u; // 4 MiB  -> 65536 64B instance payloads
        inline constexpr UINT32 kNVGShadowPerDrawCBVWords   = 8388608u;
        inline constexpr UINT32 kNVGShadowPerBatchInstWords = 1048576u;
        inline constexpr UINT32 kNVGMaxShadowPages          = 2048u;
        inline constexpr UINT32 kNVGMaxShadowSWTiles        = 512u;       // Up to 512 active 128x128 atomic shadow tiles/frame (32 MiB)
        inline constexpr UINT32 kNVGShadowSWTileTexels      = 16384u;     // 128 x 128 texels per NVS physical shadow page
        inline constexpr UINT32 kNVGMaxCameraCmdsPerBucket  = 65536u;
        inline constexpr UINT32 kNVGMaxShadowCmdsPerBucket  = 65536u;
        inline constexpr UINT32 kNVGDispatchArgWords        = 80u;        // Indirect dispatch arg triplets (camera + shadow + SW raster)

        inline AABB TransformLocalAABBToWorldCPU(const Mat4& wm, const AABB& localB, FLOAT32 pad) noexcept
        {
            const Vec3 col0(wm.m[0], wm.m[1], wm.m[2]);
            const Vec3 col1(wm.m[4], wm.m[5], wm.m[6]);
            const Vec3 col2(wm.m[8], wm.m[9], wm.m[10]);
            const Vec3 trans(wm.m[12], wm.m[13], wm.m[14]);
            const Vec3 lCenter = (localB.m_Min + localB.m_Max) * 0.5f;
            const Vec3 lExt    = (localB.m_Max - localB.m_Min) * 0.5f;
            const Vec3 wCenter = col0 * lCenter.x + col1 * lCenter.y + col2 * lCenter.z + trans;
            const Vec3 abs0(std::fabs(col0.x), std::fabs(col0.y), std::fabs(col0.z));
            const Vec3 abs1(std::fabs(col1.x), std::fabs(col1.y), std::fabs(col1.z));
            const Vec3 abs2(std::fabs(col2.x), std::fabs(col2.y), std::fabs(col2.z));
            const Vec3 wExt = abs0 * lExt.x + abs1 * lExt.y + abs2 * lExt.z + Vec3(pad, pad, pad);
            return AABB(wCenter - wExt, wCenter + wExt);
        }

        // Exact Gershgorin-Frobenius singular-value upper bound matching NVGComputeMaxWorldScaleUpperBound in NVGRasterMath.nsl:
        // Returns -1.0f if wm contains non-finite entries or extreme (> 1e12) overflow values.
        inline FLOAT32 ComputeMaxWorldScaleUpperBoundCPU(const Mat4& wm) noexcept
        {
            const Vec3 c0(wm.m[0], wm.m[1], wm.m[2]);
            const Vec3 c1(wm.m[4], wm.m[5], wm.m[6]);
            const Vec3 c2(wm.m[8], wm.m[9], wm.m[10]);
            const Vec3 tr(wm.m[12], wm.m[13], wm.m[14]);
            const FLOAT32 g00     = c0.Dot(c0);
            const FLOAT32 g11     = c1.Dot(c1);
            const FLOAT32 g22     = c2.Dot(c2);
            const FLOAT32 fNormSq = g00 + g11 + g22;
            const FLOAT32 trLenSq = tr.Dot(tr);
            if (!std::isfinite(fNormSq) || !std::isfinite(trLenSq) || fNormSq > 1.0e24f || trLenSq > 1.0e24f)
            {
                return -1.0f;
            }
            const FLOAT32 g01   = std::fabs(c0.Dot(c1));
            const FLOAT32 g02   = std::fabs(c0.Dot(c2));
            const FLOAT32 g12   = std::fabs(c1.Dot(c2));
            const FLOAT32 r0    = g00 + g01 + g02;
            const FLOAT32 r1    = g11 + g01 + g12;
            const FLOAT32 r2    = g22 + g02 + g12;
            const FLOAT32 maxSq = (std::min)(fNormSq, (std::max)({ r0, r1, r2 }));
            return std::sqrt((std::max)(maxSq, 1.0e-12f));
        }

        inline UINT32 HashMaterialShadowPolicy(const Material* mat) noexcept
        {
            if (!mat) return 0u;
            const NVGMaterialPolicy& pol = mat->GetNVGPolicy();
            UINT32 h = 2166136261u;
            auto mixU32 = [&](UINT32 v) noexcept {
                h ^= v;
                h *= 16777619u;
            };
            UINT32 bScale = 0u;
            UINT32 bDisp  = 0u;
            std::memcpy(&bScale, &pol.shadowLodErrorScale, sizeof(UINT32));
            std::memcpy(&bDisp,  &pol.maxWorldDisplacement, sizeof(UINT32));
            mixU32(bScale);
            mixU32(bDisp);
            mixU32(pol.lockToLOD0 ? 1u : 0u);
            mixU32(static_cast<UINT32>(mat->GetCullMode()));
            mixU32(mat->IsAlphaTested() ? 1u : 0u);
            const ShaderSet* shSet = mat->GetShaderSet(PassType::SHADOW);
            const UINT64 shPtr = static_cast<UINT64>(reinterpret_cast<std::uintptr_t>(shSet));
            mixU32(static_cast<UINT32>(shPtr & 0xFFFFFFFFull));
            mixU32(static_cast<UINT32>((shPtr >> 32u) & 0xFFFFFFFFull));
            const UINT64 shRev = shSet ? shSet->GetStateRevision() : 0ull;
            mixU32(static_cast<UINT32>(shRev & 0xFFFFFFFFull));
            mixU32(static_cast<UINT32>((shRev >> 32u) & 0xFFFFFFFFull));
            return (h == 0u) ? 1u : h;
        }

        inline UINT32 PrevPow2(UINT32 v) noexcept
        {
            if (v <= 1u) return 1u;
            UINT32 p = 1u;
            while ((p << 1u) <= v && p < 0x40000000u)
            {
                p <<= 1u;
            }
            return p;
        }

        inline UINT32 ComputeMipCount(UINT32 w, UINT32 h) noexcept
        {
            UINT32 m = 1u;
            while (w > 1u || h > 1u)
            {
                w = (w > 1u) ? (w >> 1u) : 1u;
                h = (h > 1u) ? (h >> 1u) : 1u;
                ++m;
            }
            return m;
        }

        inline bool EqualsNoCase(const std::string& a, const char* b) noexcept
        {
            if (!b) return false;
            const SIZE_T len = std::strlen(b);
            if (a.size() != len) return false;
            for (SIZE_T i = 0; i < len; ++i)
            {
                unsigned char ca = static_cast<unsigned char>(a[i]);
                unsigned char cb = static_cast<unsigned char>(b[i]);
                if (ca >= 'A' && ca <= 'Z') ca = static_cast<unsigned char>(ca - 'A' + 'a');
                if (cb >= 'A' && cb <= 'Z') cb = static_cast<unsigned char>(cb - 'A' + 'a');
                if (ca != cb) return false;
            }
            return true;
        }

        inline std::string StripNSLPrefix(const std::string& s)
        {
            const SIZE_T pos = s.rfind("__");
            if (pos != std::string::npos && pos + 2u < s.size())
            {
                return s.substr(pos + 2u);
            }
            return s;
        }

        inline void ExtractFrustumPlanesFromVP(const Mat4& vp, Vec4 outPlanes[6]) noexcept
        {
            // Column-major Mat4: row r = (vp(r,0), vp(r,1), vp(r,2), vp(r,3))
            const Vec4 r0(vp.m[0], vp.m[4], vp.m[8],  vp.m[12]);
            const Vec4 r1(vp.m[1], vp.m[5], vp.m[9],  vp.m[13]);
            const Vec4 r2(vp.m[2], vp.m[6], vp.m[10], vp.m[14]);
            const Vec4 r3(vp.m[3], vp.m[7], vp.m[11], vp.m[15]);

            outPlanes[0] = r3 + r0; // Left:   x + w >= 0
            outPlanes[1] = r3 - r0; // Right:  w - x >= 0
            outPlanes[2] = r3 + r1; // Bottom: y + w >= 0
            outPlanes[3] = r3 - r1; // Top:    w - y >= 0
            outPlanes[4] = r2;      // Near (D3D 0..1): z >= 0
            outPlanes[5] = r3 - r2; // Far  (D3D 0..1): w - z >= 0

            for (int i = 0; i < 6; ++i)
            {
                const FLOAT32 len = std::sqrt(
                    outPlanes[i].x * outPlanes[i].x +
                    outPlanes[i].y * outPlanes[i].y +
                    outPlanes[i].z * outPlanes[i].z);
                if (len > 1e-6f)
                {
                    const FLOAT32 inv = 1.0f / len;
                    outPlanes[i] = Vec4(
                        outPlanes[i].x * inv,
                        outPlanes[i].y * inv,
                        outPlanes[i].z * inv,
                        outPlanes[i].w * inv);
                }
            }
        }

        inline UINT64 ComputeDrawSortKey(const DrawCommand& cmd) noexcept
        {
            UINT64 k = static_cast<UINT64>(cmd.vertexShader & 0xFFFFu) << 40u;
            k |= static_cast<UINT64>(cmd.pixelShader & 0xFFFFu) << 24u;
            k |= static_cast<UINT64>(static_cast<UINT8>(cmd.blendMode) & 0xFu) << 16u;
            k |= static_cast<UINT64>(static_cast<UINT8>(cmd.cullMode) & 0xFu) << 8u;
            k |= static_cast<UINT64>(cmd.indirectArgsOffset & 0xFFu);
            return k;
        }

        inline ShaderVariant ResolveEffectiveNVGVariant(
            const ShaderSet*         shaderSet,
            BOOL                     isSkinned,
            const NVGMaterialPolicy& policy,
            NVGInternalExecutionStrategy (*planFn)(const ShaderSet*, ShaderVariant, const NVGMaterialPolicy&, BOOL*, BOOL*))
        {
            if (!shaderSet) return ShaderVariant::COUNT;
            auto isValidStage = [](const Shader* s) -> BOOL {
                return (s != nullptr && s->IsValid() && s->GetNativeHandle() != 0);
            };
            if (!isSkinned)
            {
                if (isValidStage(shaderSet->GetVertexShader(ShaderVariant::STANDARD)))  return ShaderVariant::STANDARD;
                if (isValidStage(shaderSet->GetVertexShader(ShaderVariant::INSTANCED))) return ShaderVariant::INSTANCED;
                return ShaderVariant::COUNT;
            }
            if (isValidStage(shaderSet->GetVertexShader(ShaderVariant::SKINNED)))           return ShaderVariant::SKINNED;
            if (isValidStage(shaderSet->GetVertexShader(ShaderVariant::INSTANCED_SKINNED))) return ShaderVariant::INSTANCED_SKINNED;
            const Shader* stdVs = shaderSet->GetVertexShader(ShaderVariant::STANDARD);
            if (isValidStage(stdVs))
            {
                BOOL usesIA = false;
                BOOL usesPull = false;
                planFn(shaderSet, ShaderVariant::STANDARD, policy, &usesIA, &usesPull);
                if ((usesPull && !usesIA) || CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(stdVs->GetReflection()))
                {
                    return ShaderVariant::STANDARD;
                }
            }
            const Shader* instVs = shaderSet->GetVertexShader(ShaderVariant::INSTANCED);
            if (isValidStage(instVs) && CoreResourceBindingResolver::DoesShaderRequireEngineSkinning(instVs->GetReflection()))
            {
                return ShaderVariant::INSTANCED;
            }
            return ShaderVariant::COUNT;
        }

        inline void ComputeRequiredIAVertexWordsAndMask(
            const Shader* vs,
            UINT32&       outVertexWords,
            UINT32&       outAttrMask) noexcept
        {
            outVertexWords = 0u;
            outAttrMask    = 0u;
            if (!vs || !vs->IsValid()) return;

            for (const auto& inp : vs->GetReflection().inputLayout)
            {
                if (inp.category == 2 ||
                    EqualsNoCase(inp.semantic, "SV_VertexID") ||
                    EqualsNoCase(inp.semantic, "SV_InstanceID") ||
                    EqualsNoCase(inp.name, "VERTEX_ID") ||
                    EqualsNoCase(inp.name, "INSTANCE_ID") ||
                    inp.semantic.rfind("SV_", 0) == 0)
                {
                    continue;
                }

                if (EqualsNoCase(inp.name, "VERTEX_POSITION") || EqualsNoCase(inp.name, "POSITION") ||
                    EqualsNoCase(inp.semantic, "POSITION") || EqualsNoCase(inp.semantic, "POSITION0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 3u);
                    outAttrMask   |= (1u << 0);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_NORMAL") || EqualsNoCase(inp.name, "NORMAL") ||
                         EqualsNoCase(inp.semantic, "NORMAL") || EqualsNoCase(inp.semantic, "NORMAL0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 6u);
                    outAttrMask   |= (1u << 1);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_TANGENT") || EqualsNoCase(inp.name, "TANGENT") ||
                         EqualsNoCase(inp.semantic, "TANGENT") || EqualsNoCase(inp.semantic, "TANGENT0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 9u);
                    outAttrMask   |= (1u << 2);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_BITANGENT") || EqualsNoCase(inp.name, "BITANGENT") ||
                         EqualsNoCase(inp.name, "BINORMAL") || EqualsNoCase(inp.semantic, "BINORMAL") ||
                         EqualsNoCase(inp.semantic, "BINORMAL0") || EqualsNoCase(inp.semantic, "BITANGENT") ||
                         EqualsNoCase(inp.semantic, "BITANGENT0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 12u);
                    outAttrMask   |= (1u << 3);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_UV0") || EqualsNoCase(inp.name, "UV0") ||
                         EqualsNoCase(inp.semantic, "TEXCOORD") || EqualsNoCase(inp.semantic, "TEXCOORD0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 14u);
                    outAttrMask   |= (1u << 4);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_UV1") || EqualsNoCase(inp.name, "UV1") ||
                         EqualsNoCase(inp.semantic, "TEXCOORD1"))
                {
                    outVertexWords = (std::max)(outVertexWords, 16u);
                    outAttrMask   |= (1u << 5);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_COLOR") || EqualsNoCase(inp.name, "COLOR") ||
                         EqualsNoCase(inp.semantic, "COLOR") || EqualsNoCase(inp.semantic, "COLOR0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 20u);
                    outAttrMask   |= (1u << 6);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_BONE_INDICES") || EqualsNoCase(inp.name, "BONE_INDICES") ||
                         EqualsNoCase(inp.semantic, "BLENDINDICES") || EqualsNoCase(inp.semantic, "BLENDINDICES0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 24u);
                    outAttrMask   |= (1u << 7);
                }
                else if (EqualsNoCase(inp.name, "VERTEX_BONE_WEIGHTS") || EqualsNoCase(inp.name, "BONE_WEIGHTS") ||
                         EqualsNoCase(inp.semantic, "BLENDWEIGHT") || EqualsNoCase(inp.semantic, "BLENDWEIGHT0"))
                {
                    outVertexWords = (std::max)(outVertexWords, 28u);
                    outAttrMask   |= (1u << 8);
                }
                else
                {
                    outVertexWords = 28u;
                    outAttrMask    = 0x1FFu;
                }
            }
        }

        inline BOOL DoesShaderDeclareClusterOffsetField(const Shader* st) noexcept
        {
            if (!st || !st->IsValid()) return false;
            for (const auto& nb : st->GetReflection().buffers)
            {
                for (const auto& f : nb.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    if (pName == "NVG_VISIBLE_CLUSTER_OFFSET")
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        inline BOOL DoesShaderReferenceVisBuffer(const Shader* st) noexcept
        {
            if (!st || !st->IsValid()) return false;
            for (const auto& tex : st->GetReflection().textures)
            {
                if (tex.IsEngineProvided() && tex.propertyName == "NVG_VISIBILITY_BUFFER")
                {
                    return true;
                }
            }
            return false;
        }

        inline BOOL DoesComputeShaderReferenceNVG(const Shader* cs) noexcept
        {
            if (!cs || !cs->IsValid()) return false;
            if (DoesShaderReferenceVisBuffer(cs)) return true;
            const ShaderReflection& r = cs->GetReflection();
            for (const auto& sb : r.structuredBuffers)
            {
                const EngineResourceDomain dom = CoreResourceBindingResolver::DetectStructuredBufferDomain(sb, ShaderStage::COMPUTE);
                if (dom == EngineResourceDomain::NVG_PAGE_POOL ||
                    dom == EngineResourceDomain::NVG_BUCKET_RANGES ||
                    dom == EngineResourceDomain::NVG_VISIBLE_CLUSTERS)
                {
                    return true;
                }
            }
            for (const auto& nb : r.buffers)
            {
                for (const auto& f : nb.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    if (pName == "NVG_DEBUG_MODE" ||
                        pName == "NVG_MAX_SCREEN_ERROR_PX" ||
                        pName == "NVG_PHYSICAL_POOL_CAPACITY")
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        inline BOOL IsCameraBucketEligibleForSWRaster(
            const NVGDrawBucketEntry& bucket,
            BOOL                      isReversedZ,
            NVGInternalExecutionStrategy (*planFn)(const ShaderSet*, ShaderVariant, const NVGMaterialPolicy&, BOOL*, BOOL*)) noexcept
        {
            if (bucket.isShadowPass || bucket.isDepthPass || bucket.isDepthCompanion ||
                bucket.isTransparent || bucket.hasRejectedCustomDepth || bucket.isVertexPull)
            {
                return false;
            }
            const Material*  mat = bucket.material;
            const ShaderSet* set = bucket.shaderSet;
            if (!mat || !set) return false;

            // 1. Exclude transparency / non-opaque blending
            if (mat->IsTransparent() || mat->GetBlendMode() != BlendMode::OPAQUE)
                return false;

            // 2. Exclude alpha clipping / discard
            if (mat->IsAlphaTested())
                return false;

            // 3. Require standard nearest depth test & depth write
            const DepthState& ds = mat->GetDepthState();
            if (!ds.test || !ds.write)
                return false;
            if (isReversedZ)
            {
                if (ds.compare != DepthCompare::GREATER && ds.compare != DepthCompare::GREATER_EQUAL)
                    return false;
            }
            else
            {
                if (ds.compare != DepthCompare::LESS && ds.compare != DepthCompare::LESS_EQUAL)
                    return false;
            }

            // 4. Exclude wireframe fill mode
            if (mat->GetFillMode() != FillMode::SOLID)
                return false;

            // 5. Exclude custom PassType::DEPTH ShaderSet
            if (mat->HasPass(PassType::DEPTH))
                return false;

            // 6. Exclude tessellation (Hull/Domain) and Geometry Shader stages
            if (set->GetHullShader() != nullptr ||
                set->GetDomainShader() != nullptr ||
                set->GetGeometryShader() != nullptr)
            {
                return false;
            }

            // 7. Exclude PS discard, custom SV_Depth, and PS UAV side effects
            const Shader* ps = set->GetPixelShader();
            if (ps && ps->IsValid())
            {
                const ShaderReflection& pr = ps->GetReflection();
                if ((pr.flags & (NSL::NSC_FLAG_USES_DISCARD | NSL::NSC_FLAG_WRITES_DEPTH | NSL::NSC_FLAG_HAS_UAV)) != 0u)
                    return false;
                if (!pr.uavBuffers.empty())
                    return false;
                for (const auto& outp : pr.outputLayout)
                {
                    if (EqualsNoCase(outp.semantic, "SV_Depth") ||
                        EqualsNoCase(outp.semantic, "SV_DepthGreaterEqual") ||
                        EqualsNoCase(outp.semantic, "SV_DepthLessEqual") ||
                        EqualsNoCase(outp.name, "DEPTH") ||
                        EqualsNoCase(outp.name, "FRAG_DEPTH"))
                    {
                        return false;
                    }
                }
            }

            // 8. Exclude WPO / custom vertex deformation / procedural system-ID-only VS
            const NVGMaterialPolicy& pol = mat->GetNVGPolicy();
            if (pol.maxWorldDisplacement > 0.0f)
                return false;

            const Shader* vs = set->GetVertexShader(bucket.variant);
            if (!vs || !vs->IsValid())
                return false;

            BOOL usesIA   = false;
            BOOL usesPull = false;
            planFn(set, bucket.variant, pol, &usesIA, &usesPull);
            if (!usesIA && !usesPull)
                return false;

            const ShaderReflection& vr = vs->GetReflection();
            if (!vr.textures.empty() || !vr.samplers.empty() || !vr.uavBuffers.empty() ||
                (vr.flags & NSL::NSC_FLAG_HAS_UAV) != 0u)
            {
                return false;
            }
            for (const auto& nb : vr.buffers)
            {
                for (const auto& nf : nb.fields)
                {
                    if (!nf.isGet)
                        return false;
                    const std::string& pName = nf.propertyName.empty() ? nf.name : nf.propertyName;
                    if (pName == "TIME" || pName == "ENGINE_TIME" || pName == "DELTA_TIME" || pName == "FRAME_INDEX")
                        return false;
                }
            }
            for (const auto& nsb : vr.structuredBuffers)
            {
                const StructuredBufferLayoutPlan plan =
                    CoreResourceBindingResolver::AnalyzeStructuredBuffer(nsb, ShaderStage::VERTEX, 0u);
                if (!plan.valid || plan.domain == EngineResourceDomain::USER_ONLY)
                    return false;
                if (plan.hasAddFields && !CoreResourceBindingResolver::IsTier1InstanceStructuredBuffer(nsb, ShaderStage::VERTEX))
                    return false;
            }
            return true;
        }

        inline BOOL DoesShaderReferenceShadowProp(const Shader* st, const char* propA, const char* propB) noexcept
        {
            if (!st || !st->IsValid()) return false;
            const ShaderReflection& r = st->GetReflection();
            for (const auto& cb : r.buffers)
            {
                for (const auto& f : cb.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    if ((propA && pName == propA) || (propB && pName == propB)) return true;
                }
            }
            for (const auto& sb : r.structuredBuffers)
            {
                for (const auto& f : sb.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    if ((propA && pName == propA) || (propB && pName == propB)) return true;
                }
            }
            return false;
        }

        inline BOOL IsShadowBucketEligibleForSWRaster(
            const NVGDrawBucketEntry& bucket,
            NVGInternalExecutionStrategy (*planFn)(const ShaderSet*, ShaderVariant, const NVGMaterialPolicy&, BOOL*, BOOL*)) noexcept
        {
            if (!bucket.isShadowPass || bucket.isTransparent)
                return false;

            const Material*  mat = bucket.material;
            const ShaderSet* set = bucket.shaderSet;
            if (!mat || !set)
                return false;

            // Note: check alpha-clipping/discard strictly on PassType::SHADOW's ShaderSet (below), NOT mat->IsAlphaTested()
            // (which inspects PassType::GEOMETRY too, where StandardLitPS declares optional alphaCutoff discard even when
            // PassType::SHADOW binds an opaque depth-only shadow ShaderSet with pixelShader == nullptr!).
            if (mat->IsTransparent() || mat->GetFillMode() != FillMode::SOLID)
                return false;

            const DepthState& ds = mat->GetDepthState();
            if (!ds.test || !ds.write)
                return false;
            if (ds.compare != DepthCompare::LESS &&
                ds.compare != DepthCompare::LESS_EQUAL &&
                ds.compare != DepthCompare::GREATER &&
                ds.compare != DepthCompare::GREATER_EQUAL)
            {
                return false;
            }

            if (set->GetHullShader() != nullptr ||
                set->GetDomainShader() != nullptr ||
                set->GetGeometryShader() != nullptr)
            {
                return false;
            }

            const Shader* ps = set->GetPixelShader();
            if (ps && ps->IsValid())
            {
                const ShaderReflection& pr = ps->GetReflection();
                if ((pr.flags & (NSL::NSC_FLAG_USES_DISCARD | NSL::NSC_FLAG_WRITES_DEPTH | NSL::NSC_FLAG_HAS_UAV)) != 0u)
                    return false;
                if (!pr.uavBuffers.empty())
                    return false;
                for (const auto& outp : pr.outputLayout)
                {
                    if (EqualsNoCase(outp.semantic, "SV_Depth") ||
                        EqualsNoCase(outp.semantic, "SV_DepthGreaterEqual") ||
                        EqualsNoCase(outp.semantic, "SV_DepthLessEqual") ||
                        EqualsNoCase(outp.name, "DEPTH") ||
                        EqualsNoCase(outp.name, "FRAG_DEPTH"))
                    {
                        return false;
                    }
                }
            }

            const NVGMaterialPolicy& pol = mat->GetNVGPolicy();
            if (pol.maxWorldDisplacement > 0.0f)
                return false;

            const Shader* vs = set->GetVertexShader(bucket.variant);
            if (!vs || !vs->IsValid())
                return false;

            BOOL usesIA   = false;
            BOOL usesPull = false;
            planFn(set, bucket.variant, pol, &usesIA, &usesPull);
            if (!usesIA && !usesPull)
                return false;

            const ShaderReflection& vr = vs->GetReflection();
            if (!vr.textures.empty() || !vr.samplers.empty() || !vr.uavBuffers.empty() ||
                (vr.flags & NSL::NSC_FLAG_HAS_UAV) != 0u)
            {
                return false;
            }
            for (const auto& nb : vr.buffers)
            {
                for (const auto& nf : nb.fields)
                {
                    if (!nf.isGet)
                        return false;
                    const std::string& pName = nf.propertyName.empty() ? nf.name : nf.propertyName;
                    if (pName == "TIME" || pName == "ENGINE_TIME" || pName == "DELTA_TIME" || pName == "FRAME_INDEX")
                        return false;
                }
            }
            for (const auto& nsb : vr.structuredBuffers)
            {
                const StructuredBufferLayoutPlan plan =
                    CoreResourceBindingResolver::AnalyzeStructuredBuffer(nsb, ShaderStage::VERTEX, 0u);
                if (!plan.valid || plan.domain == EngineResourceDomain::USER_ONLY)
                    return false;
                if (plan.hasAddFields && !CoreResourceBindingResolver::IsTier1InstanceStructuredBuffer(nsb, ShaderStage::VERTEX))
                    return false;
            }
            return true;
        }

        inline UINT32 EstimatePromotedGraphicsRootDwords(
            const Shader* const stages[5],
            UINT32              promotedCBVs,
            UINT32              promotedSRVs) noexcept
        {
            UINT16 uavMask = 0u;
            UINT32 tableCount = 0u;
            for (UINT32 si = 0u; si < 5u; ++si)
            {
                const Shader* st = stages[si];
                if (!st || !st->IsValid()) continue;
                const ShaderReflection& r = st->GetReflection();
                UINT16 cbvSrv = (si == 0u) ? 0x0001u : 0x0000u;
                UINT16 samp   = 0u;
                for (const auto& b : r.buffers)           if (b.set < 16u)  cbvSrv = static_cast<UINT16>(cbvSrv | (1u << b.set));
                for (const auto& s : r.structuredBuffers) if (r.FindUAV(s.name) == nullptr && s.set < 16u) cbvSrv = static_cast<UINT16>(cbvSrv | (1u << s.set));
                for (const auto& t : r.textures)          if (t.set < 16u)  cbvSrv = static_cast<UINT16>(cbvSrv | (1u << t.set));
                for (const auto& m : r.samplers)          if (m.set < 16u)  samp   = static_cast<UINT16>(samp   | (1u << m.set));
                for (const auto& u : r.uavBuffers)        if (u.set < 16u)  uavMask = static_cast<UINT16>(uavMask | (1u << u.set));
                for (UINT32 b = 0u; b < 16u; ++b)
                {
                    if ((cbvSrv & (1u << b)) != 0u) ++tableCount;
                    if ((samp   & (1u << b)) != 0u) ++tableCount;
                }
            }
            for (UINT32 b = 0u; b < 16u; ++b)
            {
                if ((uavMask & (1u << b)) != 0u) ++tableCount;
            }
            const BOOL extended = (promotedCBVs > 3u) || (promotedSRVs > 1u);
            const UINT32 rootCbvSlots = extended ? 8u : 3u;
            const UINT32 rootSrvSlots = extended ? 2u : 1u;
            // 4 DWORDs (RootConstants) + 1 DWORD per descriptor table + 2 DWORDs per 64-bit Root Descriptor
            return 4u + tableCount + (rootCbvSlots + rootSrvSlots) * 2u;
        }
    }

    CoreNVGManager::~CoreNVGManager()
    {
        Shutdown();
    }

    BOOL CoreNVGManager::Initialize(
        NorthstarBackend*        backend,
        CoreBufferManager*       bufferManager,
        CoreTextureManager*      textureManager,
        CoreComputeManager*      computeManager,
        InternalResourceManager* internalResources)
    {
        if (m_initialized) return true;
        if (!backend || !bufferManager || !textureManager || !computeManager || !internalResources)
            return false;

        m_backend        = backend;
        m_bufferManager  = bufferManager;
        m_textureManager = textureManager;
        m_computeManager = computeManager;
        m_internalRes    = internalResources;

        if (!m_residencyManager.Initialize(backend, bufferManager, m_config))
        {
            Shutdown();
            return false;
        }

        // Allocate metadata & draw-group reflection tables
        if (!EnsureBuffer(m_materialTableBuffer, sizeof(NVGMaterialGPUData), 16u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_cameraBucketFlagsBuf, sizeof(UINT32), kNVGMaxDrawBuckets, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_shadowBucketFlagsBuf, sizeof(UINT32), kNVGMaxDrawBuckets, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_cameraBindingsBuf, sizeof(NVGDrawGroupShaderBindingDesc), kNVGMaxDrawBuckets, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_cameraPatchesBuf, sizeof(NVSDrawGroupFieldPatch), 256u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_cameraTemplatesBuf, sizeof(UINT32), 1024u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_shadowBindingsBuf, sizeof(NVGDrawGroupShaderBindingDesc), kNVGMaxDrawBuckets, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_shadowPatchesBuf, sizeof(NVSDrawGroupFieldPatch), 256u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_shadowTemplatesBuf, sizeof(UINT32), 1024u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_dummyCommitRecordsBuf, sizeof(NVSPageRenderCommitRecord), 4u, GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV) ||
            !EnsureBuffer(m_dummySkinnedBoundsBuf, sizeof(GPUSceneSkinningBoundsData), 4u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_dummyObjectMatSlotsBuf, sizeof(UINT32), 4u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_dummyBonePaletteBuf, sizeof(Mat4), 4u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(m_dummySWMaskBuf, sizeof(UINT32), 4u, GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV))
        {
            Shutdown();
            return false;
        }

        m_initialized   = true;
        m_kernelsLoaded = false;
        return true;
    }

    void CoreNVGManager::Shutdown()
    {
        for (auto& kv : m_cameraContexts)
        {
            if (kv.second)
            {
                DestroyCameraContextResources(*kv.second);
            }
        }
        m_cameraContexts.clear();

        if (m_bufferManager)
        {
            auto destroyBuf = [&](GPUBuffer*& b) {
                if (b) { m_bufferManager->DestroyGPUBuffer(b); b = nullptr; }
            };
            destroyBuf(m_materialTableBuffer);
            destroyBuf(m_cameraBucketFlagsBuf);
            destroyBuf(m_shadowBucketFlagsBuf);
            destroyBuf(m_cameraBindingsBuf);
            destroyBuf(m_cameraPatchesBuf);
            destroyBuf(m_cameraTemplatesBuf);
            destroyBuf(m_shadowBindingsBuf);
            destroyBuf(m_shadowPatchesBuf);
            destroyBuf(m_shadowTemplatesBuf);
            destroyBuf(m_dummyCommitRecordsBuf);
            destroyBuf(m_dummySkinnedBoundsBuf);
            destroyBuf(m_dummyObjectMatSlotsBuf);
            destroyBuf(m_dummyBonePaletteBuf);
            destroyBuf(m_dummySWMaskBuf);
        }

        m_residencyManager.Shutdown();

        m_kHZBBuild            = nullptr;
        m_kResidencyMask       = nullptr;
        m_kInstanceCull        = nullptr;
        m_kHierarchyCull       = nullptr;
        m_kCompactionDecode    = nullptr;
        m_kIndirectDraw        = nullptr;
        m_kShadowCull              = nullptr;
        m_kShadowMicropolyRaster   = nullptr;
        m_kMicropolyRasterAtomic64 = nullptr;
        m_kMicropolyRasterAtomic32 = nullptr;
        m_kDepthExport             = nullptr;
        m_shadowSWMergeVS          = nullptr;
        m_shadowSWMergeGS          = nullptr;
        m_shadowSWMergePS          = nullptr;
        m_hasMockCapsOverride      = false;
        m_mockCapabilities         = BackendCapabilities{};
        m_kernelsLoaded     = false;

        m_materialTable.clear();
        m_cameraBuckets.clear();
        m_shadowBuckets.clear();
        m_cameraBucketFlags.clear();
        m_shadowBucketFlags.clear();
        m_cameraBindingDescs.clear();
        m_cameraFieldPatches.clear();
        m_cameraTemplateWords.clear();
        m_shadowBindingDescs.clear();
        m_shadowFieldPatches.clear();
        m_shadowTemplateWords.clear();
        m_cachedSyncScene      = nullptr;
        m_cachedSyncObjCount   = 0u;
        m_cachedSyncSceneEpoch = ~0ull;
        m_cachedSyncMaterials.clear();
        m_lastMaterialShadowHash.clear();
        m_hasLastShadowConfigState = false;
        m_sceneHasAnimatedDeformation = false;
        m_maxMeshBVHDepth          = 1u;

        m_stats                = NVGStats{};
        m_backend              = nullptr;
        m_bufferManager        = nullptr;
        m_textureManager       = nullptr;
        m_computeManager       = nullptr;
        m_internalRes          = nullptr;
        m_resourceMaterializer = nullptr;
        m_initialized          = false;
    }

    BOOL CoreNVGManager::SetConfig(const NVGConfig& config)
    {
        if (!IsValidNVGQualityPreset(config.qualityPreset) ||
            !IsValidNVGRasterAtomicStrategy(config.swRasterAtomicStrategy))
        {
            return false;
        }
        if (!std::isfinite(config.maxScreenErrorPixels) ||
            !std::isfinite(config.shadowMaxTexelError) ||
            !std::isfinite(config.lodHysteresis) ||
            !std::isfinite(config.thinFeatureShadowBoost) ||
            !std::isfinite(config.smallClusterPixelThreshold) ||
            !std::isfinite(config.swRasterMaxEdgePixels))
        {
            return false;
        }
        if (config.qualityPreset == NVGQualityPreset::CUSTOM &&
            (config.maxScreenErrorPixels <= 0.0f || config.shadowMaxTexelError <= 0.0f))
        {
            return false;
        }
        if (config.lodHysteresis < 0.0f || config.lodHysteresis > 0.5f ||
            config.thinFeatureShadowBoost < 0.0f ||
            config.smallClusterPixelThreshold < 0.0f ||
            config.swRasterMaxEdgePixels <= 0.0f || config.swRasterMaxEdgePixels > 64.0f)
        {
            return false;
        }
        if (config.physicalPagePoolCapacity != kNVGAutoPoolCapacity &&
            (config.physicalPagePoolCapacity < kNVGMinPhysicalPageCapacity ||
             config.physicalPagePoolCapacity > kNVGMaxPhysicalPageCapacity))
        {
            return false;
        }

        // Zero-Handholding capability validation: explicitly reject forced strategies unsupported by active hardware
        if (m_initialized)
        {
            const BackendCapabilities effCaps = GetEffectiveCapabilities();
            if (config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS)
            {
                if (!effCaps.SupportsNVGRasterAtomic64())
                {
                    return false;
                }
                if (!EnsureRuntimeKernelsLoaded() || m_kMicropolyRasterAtomic64 == nullptr)
                {
                    return false;
                }
            }
            else if (config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS)
            {
                if (!effCaps.SupportsNVGRasterAtomic32())
                {
                    return false;
                }
                if (!EnsureRuntimeKernelsLoaded() || m_kMicropolyRasterAtomic32 == nullptr)
                {
                    return false;
                }
            }
        }

        const NVGConfig prevConfig = m_config;
        m_config = config;
        if (m_config.qualityPreset != NVGQualityPreset::CUSTOM)
        {
            m_config.maxScreenErrorPixels = (std::max)(0.05f, m_config.maxScreenErrorPixels);
            m_config.shadowMaxTexelError  = (std::max)(0.05f, m_config.shadowMaxTexelError);
        }
        m_config.minInFlightProtectFrames = (std::max)(1u, m_config.minInFlightProtectFrames);

        if (m_initialized)
        {
            if (!m_residencyManager.ApplyConfig(m_config, m_currentFrameIndex))
            {
                m_config = prevConfig;
                return false;
            }
        }

        // Invalidate cached material CBV templates if debug mode or screen error changed
        m_cachedSyncScene = nullptr;
        return true;
    }

    FLOAT32 CoreNVGManager::ResolveEffectiveScreenErrorPixels() const noexcept
    {
        const FLOAT32 basePx = ResolveNVGQualityScreenError(m_config.qualityPreset, m_config.maxScreenErrorPixels);
        return (std::max)(0.05f, basePx * m_residencyManager.GetAdaptiveErrorScale());
    }

    FLOAT32 CoreNVGManager::ResolveEffectiveShadowTexelError() const noexcept
    {
        const FLOAT32 baseTx = (m_config.qualityPreset == NVGQualityPreset::CUSTOM)
            ? (std::max)(0.05f, m_config.shadowMaxTexelError)
            : ResolveNVGQualityScreenError(m_config.qualityPreset, m_config.shadowMaxTexelError);
        const FLOAT32 adaptiveScale = m_residencyManager.GetAdaptiveErrorScale();
        // Under constrained VRAM (adaptiveScale > 1.0), harmonize shadow texel error quadratically with coarse camera LOD
        // so high-res NVS shadow cascades neither starve camera residency nor slice self-shadow stripes across coarse facets.
        const FLOAT32 shadowScale = (adaptiveScale > 1.001f) ? (adaptiveScale * adaptiveScale) : 1.0f;
        return (std::max)(0.05f, baseTx * shadowScale);
    }

    BOOL CoreNVGManager::EnsureBuffer(GPUBuffer*& buf, UINT32 stride, UINT32 elemCount, GPUBufferUsage usage)
    {
        if (!m_bufferManager) return false;
        const UINT32 req = (std::max)(elemCount, 1u);
        if (buf != nullptr)
        {
            if (buf->GetElementCount() >= req) return true;
            return m_bufferManager->ResizeGPUBuffer(buf, req, nullptr);
        }
        GPUBufferDesc desc{};
        desc.elementCount = req;
        desc.stride       = stride;
        desc.usage        = usage;
        desc.initialData  = nullptr;
        buf = m_bufferManager->CreateGPUBuffer(desc);
        return (buf != nullptr);
    }

    BackendCapabilities CoreNVGManager::GetEffectiveCapabilities() const noexcept
    {
        if (m_hasMockCapsOverride)
        {
            return m_mockCapabilities;
        }
        return m_backend ? m_backend->GetCapabilities() : BackendCapabilities{};
    }

    BOOL CoreNVGManager::SetMockCapabilitiesOverride(BOOL enableOverride, const BackendCapabilities& mockCaps)
    {
        if (enableOverride)
        {
            if (m_config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS &&
                !mockCaps.SupportsNVGRasterAtomic64())
            {
                return false;
            }
            if (m_config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS &&
                !mockCaps.SupportsNVGRasterAtomic32())
            {
                return false;
            }
            m_hasMockCapsOverride = true;
            m_mockCapabilities    = mockCaps;
            if (m_backend)
            {
                m_backend->SetRestrictedCapabilitiesOverride(true, mockCaps);
            }
        }
        else
        {
            m_hasMockCapsOverride = false;
            m_mockCapabilities    = BackendCapabilities{};
            if (m_backend)
            {
                m_backend->SetRestrictedCapabilitiesOverride(false, BackendCapabilities{});
            }
        }
        return true;
    }

    BOOL CoreNVGManager::DebugReloadRasterKernels(
        BOOL    bypassDiskCache,
        BYTE*   outAtomic32Bytecode,
        UINT32  maxAtomic32Bytes,
        UINT32* outAtomic32ActualBytes,
        BYTE*   outAtomic64Bytecode,
        UINT32  maxAtomic64Bytes,
        UINT32* outAtomic64ActualBytes)
    {
        if (outAtomic32ActualBytes) *outAtomic32ActualBytes = 0u;
        if (outAtomic64ActualBytes) *outAtomic64ActualBytes = 0u;
        if (!m_initialized || !m_internalRes || !m_backend)
        {
            return false;
        }
        m_backend->WaitForGPU();
        if (!EnsureRuntimeKernelsLoaded())
        {
            return false;
        }
        if (!m_internalRes->RecompileLoadedNativeShaders(bypassDiskCache))
        {
            return false;
        }

        const BackendCapabilities effCaps = GetEffectiveCapabilities();
        if (effCaps.SupportsNVGRasterAtomic32())
        {
            m_kMicropolyRasterAtomic32 = m_internalRes->AcquireFeatureKernel("Runtime.NVGMicropolyRasterAtomic32");
            Shader* sh32 = m_internalRes->GetShaderByName("Runtime.NVGMicropolyRasterAtomic32");
            if (!m_kMicropolyRasterAtomic32 || !sh32 || sh32->GetNativeHandle() == 0)
            {
                return false;
            }
            std::vector<BYTE> bc32;
            if (m_backend->GetShaderCompiledBytecode(sh32->GetNativeHandle(), bc32))
            {
                if (outAtomic32ActualBytes) *outAtomic32ActualBytes = static_cast<UINT32>(bc32.size());
                if (outAtomic32Bytecode && maxAtomic32Bytes > 0u && !bc32.empty())
                {
                    std::memcpy(outAtomic32Bytecode, bc32.data(), (std::min<SIZE_T>)(bc32.size(), maxAtomic32Bytes));
                }
            }
        }
        else
        {
            m_kMicropolyRasterAtomic32 = nullptr;
        }

        if (effCaps.SupportsNVGRasterAtomic64())
        {
            m_kMicropolyRasterAtomic64 = m_internalRes->AcquireFeatureKernel("Runtime.NVGMicropolyRasterAtomic64");
            Shader* sh64 = m_internalRes->GetShaderByName("Runtime.NVGMicropolyRasterAtomic64");
            if (!m_kMicropolyRasterAtomic64 || !sh64 || sh64->GetNativeHandle() == 0)
            {
                return false;
            }
            std::vector<BYTE> bc64;
            if (m_backend->GetShaderCompiledBytecode(sh64->GetNativeHandle(), bc64))
            {
                if (outAtomic64ActualBytes) *outAtomic64ActualBytes = static_cast<UINT32>(bc64.size());
                if (outAtomic64Bytecode && maxAtomic64Bytes > 0u && !bc64.empty())
                {
                    std::memcpy(outAtomic64Bytecode, bc64.data(), (std::min<SIZE_T>)(bc64.size(), maxAtomic64Bytes));
                }
            }
        }
        else
        {
            m_kMicropolyRasterAtomic64 = nullptr;
        }
        return true;
    }

    NVGRasterAtomicStrategy CoreNVGManager::ResolveActiveCameraRasterStrategy() const noexcept
    {
        if (!m_config.enableSoftwareRasterizer || m_config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::DISABLED)
        {
            return NVGRasterAtomicStrategy::DISABLED;
        }
        const BackendCapabilities caps = GetEffectiveCapabilities();
        if (m_config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS)
        {
            return (caps.SupportsNVGRasterAtomic64() && m_kMicropolyRasterAtomic64 != nullptr)
                ? NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS
                : NVGRasterAtomicStrategy::DISABLED;
        }
        if (m_config.swRasterAtomicStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS)
        {
            return (caps.SupportsNVGRasterAtomic32() && m_kMicropolyRasterAtomic32 != nullptr)
                ? NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS
                : NVGRasterAtomicStrategy::DISABLED;
        }
        // AUTO: prefer single-pass 64-bit descriptor-heap atomic commit when supported, else universal two-pass 2x32-bit
        if (caps.SupportsNVGRasterAtomic64() && m_kMicropolyRasterAtomic64 != nullptr)
        {
            return NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS;
        }
        if (caps.SupportsNVGRasterAtomic32() && m_kMicropolyRasterAtomic32 != nullptr)
        {
            return NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS;
        }
        return NVGRasterAtomicStrategy::DISABLED;
    }

    BOOL CoreNVGManager::EnsureRuntimeKernelsLoaded()
    {
        if (m_kernelsLoaded) return true;
        if (!m_internalRes || !m_internalRes->IsInitialized()) return false;

        m_kHZBBuild                = m_internalRes->AcquireFeatureKernel("Runtime.NVGHZBBuild");
        m_kResidencyMask           = m_internalRes->AcquireFeatureKernel("Runtime.NVGResidencyMask");
        m_kInstanceCull            = m_internalRes->AcquireFeatureKernel("Runtime.NVGInstanceCull");
        m_kHierarchyCull           = m_internalRes->AcquireFeatureKernel("Runtime.NVGHierarchyCull");
        m_kCompactionDecode        = m_internalRes->AcquireFeatureKernel("Runtime.NVGCompactionAndDecode");
        m_kIndirectDraw            = m_internalRes->AcquireFeatureKernel("Runtime.NVGIndirectDraw");
        m_kShadowCull              = m_internalRes->AcquireFeatureKernel("Runtime.NVGShadowCull");
        m_kShadowMicropolyRaster   = m_internalRes->AcquireFeatureKernel("Runtime.NVGShadowMicropolyRaster");
        m_kMicropolyRasterAtomic32 = m_internalRes->AcquireFeatureKernel("Runtime.NVGMicropolyRasterAtomic32");
        m_kDepthExport             = m_internalRes->AcquireFeatureKernel("Runtime.NVGDepthExport");
        m_shadowSWMergeVS          = m_internalRes->AcquireFeatureGraphicsShader("Runtime.NVGShadowSWMergeVS");
        m_shadowSWMergeGS          = m_internalRes->AcquireFeatureGraphicsShader("Runtime.NVGShadowSWMergeGS");
        m_shadowSWMergePS          = m_internalRes->AcquireFeatureGraphicsShader("Runtime.NVGShadowSWMergePS");
        if (m_backend && m_backend->GetCapabilities().SupportsNVGRasterAtomic64())
        {
            m_kMicropolyRasterAtomic64 = m_internalRes->AcquireFeatureKernel("Runtime.NVGMicropolyRasterAtomic64");
        }

        m_kernelsLoaded = (m_kHZBBuild != nullptr &&
                           m_kResidencyMask != nullptr &&
                           m_kInstanceCull != nullptr &&
                           m_kHierarchyCull != nullptr &&
                           m_kCompactionDecode != nullptr &&
                           m_kIndirectDraw != nullptr &&
                           m_kShadowCull != nullptr &&
                           m_kShadowMicropolyRaster != nullptr &&
                           m_shadowSWMergeVS != nullptr &&
                           m_shadowSWMergePS != nullptr &&
                           m_kDepthExport != nullptr &&
                           (m_kMicropolyRasterAtomic64 != nullptr || m_kMicropolyRasterAtomic32 != nullptr));
        return m_kernelsLoaded;
    }

    BOOL CoreNVGManager::CreateAndRegisterVirtualMesh(
        const MeshDesc&                          desc,
        UINT32                                   meshId,
        NVGMeshInfo&                             outInfo,
        AABB&                                    outLocalBounds,
        std::string&                             outError,
        std::shared_ptr<INVGPageStreamProvider>* outResolvedStreamProvider)
    {
        outInfo = NVGMeshInfo{};
        outError.clear();
        if (outResolvedStreamProvider) *outResolvedStreamProvider = nullptr;
        if (!m_initialized)
        {
            outError = "NVG manager is not initialized.";
            return false;
        }

        NVGVirtualMeshAsset asset{};
        std::shared_ptr<INVGPageStreamProvider> activeStreamProvider = desc.pageStreamProvider;

        if (desc.virtualGeometryFilePath != nullptr && desc.virtualGeometryFilePath[0] != '\0')
        {
            auto fileProvider = NVGFilePageStreamProvider::OpenNVG2File(
                std::filesystem::path(reinterpret_cast<const char8_t*>(desc.virtualGeometryFilePath)),
                asset,
                outError);
            if (!fileProvider)
            {
                return false;
            }
            activeStreamProvider = std::move(fileProvider);
        }
        else if (desc.serializedVirtualGeometryData != nullptr && desc.serializedVirtualGeometrySize > 0u)
        {
            UINT32 peekMagic = 0u;
            if (desc.serializedVirtualGeometrySize >= sizeof(UINT32))
            {
                std::memcpy(&peekMagic, desc.serializedVirtualGeometryData, sizeof(UINT32));
            }

            if (activeStreamProvider != nullptr && peekMagic == kNVG2BinaryMagic)
            {
                // Zero-RAM streaming registration: parse only [header + tables + page directory] without decompressing pages!
                NVG2BinaryHeader hdr{};
                std::vector<NVG2PageDirectoryEntry> dir;
                if (!NVGBuilder::ParseNVG2MetadataAndDirectory(
                        static_cast<const BYTE*>(desc.serializedVirtualGeometryData),
                        desc.serializedVirtualGeometrySize,
                        hdr,
                        asset,
                        dir,
                        outError))
                {
                    return false;
                }
            }
            else if (!NVGBuilder::Deserialize(
                    static_cast<const BYTE*>(desc.serializedVirtualGeometryData),
                    desc.serializedVirtualGeometrySize,
                    asset,
                    outError))
            {
                return false;
            }
        }
        else
        {
            if (!NVGBuilder::BuildVirtualMesh(
                    desc.vertices,
                    desc.vertexCount,
                    desc.indices,
                    desc.indexCount,
                    desc.submeshes,
                    desc.submeshCount,
                    desc.nvgBuildConfig,
                    asset))
            {
                outError = asset.buildReport.diagnosticMessage.empty()
                    ? "NVG virtual mesh build failed."
                    : asset.buildReport.diagnosticMessage;
                return false;
            }
        }

        const UINT32 totalVirtualPages = activeStreamProvider
            ? activeStreamProvider->GetVirtualPageCount()
            : static_cast<UINT32>(asset.pages.size());

        outInfo.valid                = true;
        outInfo.virtualMeshId        = meshId;
        outInfo.totalClusterCount    = static_cast<UINT32>(asset.clusters.size());
        outInfo.lod0ClusterCount     = asset.buildReport.lod0ClusterCount;
        outInfo.lodLevelCount        = asset.lodLevelCount;
        outInfo.virtualPageCount     = totalVirtualPages;
        outInfo.rootPageCount        = asset.rootPageCount;
        outInfo.bvhNodeCount         = static_cast<UINT32>(asset.bvhNodes.size());
        outInfo.hasSkinning          = asset.hasSkinning;
        outInfo.compressionRatio     = asset.buildReport.compressionRatio;
        outInfo.maxQuantizationError = asset.buildReport.maxQuantizationError;
        outInfo.buildReport          = asset.buildReport;
        outLocalBounds               = asset.localBounds;

        // Compute exact BVH tree depth from built 8-ary hierarchy (eliminates redundant empty BVH dispatches)
        UINT32 exactDepth = 1u;
        if (!asset.bvhNodes.empty())
        {
            std::vector<std::pair<UINT32, UINT32>> stack;
            stack.reserve(64u);
            stack.push_back({ 0u, 1u });
            while (!stack.empty())
            {
                const auto [nodeIdx, depth] = stack.back();
                stack.pop_back();
                if (depth > exactDepth) exactDepth = depth;
                if (nodeIdx >= asset.bvhNodes.size()) continue;
                const NVGHierarchyNodeGPUData& nd = asset.bvhNodes[nodeIdx];
                const BOOL isLeaf = ((nd.childCountAndFlags & 65536u) != 0u);
                const UINT32 cnt  = nd.childCountAndFlags & 65535u;
                if (!isLeaf && depth < kNVGMaxBVHDepth)
                {
                    for (UINT32 c = 0u; c < cnt; ++c)
                    {
                        const UINT32 childIdx = nd.childStart + c;
                        if (childIdx < asset.bvhNodes.size())
                        {
                            stack.push_back({ childIdx, depth + 1u });
                        }
                    }
                }
            }
        }
        if (exactDepth > m_maxMeshBVHDepth)
        {
            m_maxMeshBVHDepth = (std::min)(exactDepth, kNVGMaxBVHDepth);
        }

        if (!m_residencyManager.RegisterVirtualMesh(
                meshId,
                std::move(asset),
                desc.customPageDataSource,
                m_currentFrameIndex,
                outError,
                activeStreamProvider))
        {
            outInfo = NVGMeshInfo{};
            return false;
        }

        if (outResolvedStreamProvider)
        {
            *outResolvedStreamProvider = std::move(activeStreamProvider);
        }
        return true;
    }

    void CoreNVGManager::UnregisterVirtualMesh(UINT32 meshId)
    {
        if (!m_initialized) return;
        m_residencyManager.UnregisterVirtualMesh(meshId, m_currentFrameIndex);
    }

    BOOL CoreNVGManager::SerializeVirtualMesh(UINT32 meshId, std::vector<BYTE>& outBlob) const
    {
        if (!m_initialized) return false;
        return m_residencyManager.SerializeVirtualMesh(meshId, outBlob);
    }

    NVGInternalExecutionStrategy CoreNVGManager::PlanExecutionStrategy(
        const ShaderSet*         shaderSet,
        ShaderVariant            variant,
        const NVGMaterialPolicy& policy,
        BOOL*                    outUsesIAVertexAttribs,
        BOOL*                    outUsesNVGPullBuffers)
    {
        if (outUsesIAVertexAttribs) *outUsesIAVertexAttribs = false;
        if (outUsesNVGPullBuffers)  *outUsesNVGPullBuffers  = false;
        if (!shaderSet) return NVGInternalExecutionStrategy::DECODED_IA;

        const Shader* vs = shaderSet->GetVertexShader(variant);
        if (!vs || !vs->IsValid()) return NVGInternalExecutionStrategy::DECODED_IA;

        // 1. Check whether VS declares classic Input Assembler vertex attributes (beyond system IDs)
        BOOL usesIAVertexAttribs = false;
        for (const auto& inp : vs->GetReflection().inputLayout)
        {
            if (EqualsNoCase(inp.semantic, "SV_VertexID") ||
                EqualsNoCase(inp.semantic, "SV_InstanceID") ||
                EqualsNoCase(inp.name, "VERTEX_ID") ||
                EqualsNoCase(inp.name, "INSTANCE_ID"))
            {
                continue;
            }
            usesIAVertexAttribs = true;
            break;
        }

        // 2. Check whether any stage explicitly demands NVG cluster/page-pool pulling buffers
        //    (GPUSceneObjectData alone does NOT imply NVG vertex pulling!)
        BOOL usesNVGPullBuffers = false;
        const Shader* stages[5] = {
            vs,
            shaderSet->GetPixelShader(),
            shaderSet->GetGeometryShader(),
            shaderSet->GetHullShader(),
            shaderSet->GetDomainShader()
        };
        const ShaderStage stageTypes[5] = {
            ShaderStage::VERTEX, ShaderStage::PIXEL, ShaderStage::GEOMETRY,
            ShaderStage::HULL, ShaderStage::DOMAIN
        };
        for (UINT32 si = 0u; si < 5u && !usesNVGPullBuffers; ++si)
        {
            if (!stages[si] || !stages[si]->IsValid()) continue;
            for (const auto& sb : stages[si]->GetReflection().structuredBuffers)
            {
                const EngineResourceDomain dom = CoreResourceBindingResolver::DetectStructuredBufferDomain(sb, stageTypes[si]);
                if (dom == EngineResourceDomain::NVG_PAGE_POOL ||
                    dom == EngineResourceDomain::NVG_BUCKET_RANGES ||
                    dom == EngineResourceDomain::NVG_VISIBLE_CLUSTERS)
                {
                    usesNVGPullBuffers = true;
                    break;
                }
            }
        }

        if (outUsesIAVertexAttribs) *outUsesIAVertexAttribs = usesIAVertexAttribs;
        if (outUsesNVGPullBuffers)  *outUsesNVGPullBuffers  = usesNVGPullBuffers;

        // Universal Reflection-Driven Strategy Selection:
        // - If the shader consumes IA vertex attributes (even if it ALSO reads NVG StructuredBuffers!),
        //   execute via GPU-Decoded IA stream while ALSO binding all requested NVG/GPUScene buffers live.
        // - If the shader has NO IA vertex attributes AND reads NVG pulling buffers, use zero-decode
        //   Vertex Pulling (unless the user explicitly hints DECODED_IA).
        // - If the shader has NO IA vertex attributes AND NO NVG pulling buffers (procedural/system-ID
        //   shader using only VERTEX_ID / INSTANCE_ID), NEVER misclassify it as NVG Vertex Pulling —
        //   route through Decoded-IA / per-cluster indirect draws so per-object transforms & IDs stay valid.
        if (usesIAVertexAttribs)
        {
            return NVGInternalExecutionStrategy::DECODED_IA;
        }
        if (usesNVGPullBuffers)
        {
            // Reflection Lock: an attribute-less vertex-pulling shader (!usesIAVertexAttribs && usesNVGPullBuffers)
            // decodes compressed cluster words via SV_VertexID / SV_InstanceID and CANNOT be fed an indexed IA stream.
            // Ignore any incompatible DECODED_IA hint to guarantee 100% reflection-safe execution.
            (void)policy;
            return NVGInternalExecutionStrategy::VERTEX_PULL;
        }
        return NVGInternalExecutionStrategy::DECODED_IA;
    }

    NVGCompatibilityStatus CoreNVGManager::CheckMaterialCompatibility(
        const Material* material,
        PassType        pass,
        BOOL            isSkinned,
        std::string*    outReason) const
    {
        auto fail = [&](NVGCompatibilityStatus st, const std::string& msg) {
            if (outReason) *outReason = msg;
            return st;
        };

        if (!material)
            return fail(NVGCompatibilityStatus::REJECTED_MISSING_SHADERSET, "Material is null.");

        const ShaderSet* shaderSet = material->GetShaderSet(pass);
        if (!shaderSet)
            return fail(NVGCompatibilityStatus::REJECTED_MISSING_SHADERSET, "Material has no ShaderSet bound for requested PassType.");

        const NVGMaterialPolicy& policy = material->GetNVGPolicy();
        const ShaderVariant effectiveVariant = ResolveEffectiveNVGVariant(shaderSet, isSkinned, policy, &CoreNVGManager::PlanExecutionStrategy);
        if (effectiveVariant == ShaderVariant::COUNT)
        {
            return fail(NVGCompatibilityStatus::REJECTED_MISSING_VARIANT, "ShaderSet lacks a valid Vertex Shader for requested variant.");
        }
        const Shader* vs = shaderSet->GetVertexShader(effectiveVariant);

        const Shader* ps = shaderSet->GetPixelShader();
        const Shader* gs = shaderSet->GetGeometryShader();
        const Shader* hs = shaderSet->GetHullShader();
        const Shader* ds = shaderSet->GetDomainShader();

        // Hardware contract: Tessellation requires BOTH Hull and Domain shaders (no opt-in flag required!)
        if ((hs != nullptr) != (ds != nullptr))
        {
            return fail(NVGCompatibilityStatus::REJECTED_INCOMPLETE_TESSELLATION,
                        "Incomplete tessellation pipeline (both Hull and Domain shaders are required by hardware).");
        }

        // Shadow pass hardware contract: SV_RenderTargetArrayIndex output on final pre-raster stage
        if (pass == PassType::SHADOW)
        {
            const Shader* finalPre = gs ? gs : (ds ? ds : vs);
            if (!CoreNVSShadowRasterizer::DoesShaderDeclareRenderTargetArrayIndexOutput(finalPre))
            {
                return fail(NVGCompatibilityStatus::REJECTED_MISSING_RT_ARRAY_IDX,
                            "Shadow ShaderSet lacks SV_RenderTargetArrayIndex output required for NVS layered shadow rasterization.");
            }
        }

        // StructuredBuffer layout validation across all active stages
        const Shader* stages[5] = { vs, ps, gs, hs, ds };
        const ShaderStage stageTypes[5] = {
            ShaderStage::VERTEX, ShaderStage::PIXEL,
            ShaderStage::GEOMETRY, ShaderStage::HULL, ShaderStage::DOMAIN
        };
        const GraphicsShaderSlot slots[5] = {
            GraphicsShaderSlot::VERTEX, GraphicsShaderSlot::PIXEL,
            GraphicsShaderSlot::GEOMETRY, GraphicsShaderSlot::HULL, GraphicsShaderSlot::DOMAIN_STAGE
        };

        for (UINT32 si = 0u; si < 5u; ++si)
        {
            const Shader* st = stages[si];
            if (!st) continue;
            if (!st->IsValid() || st->GetNativeHandle() == 0)
            {
                return fail(NVGCompatibilityStatus::REJECTED_MISSING_VARIANT, "ShaderSet contains an invalid shader stage.");
            }
            const ShaderStageProps& props   = material->GetPropertyStore().Get(pass, slots[si]);
            const ShaderStageProps& fbProps = material->GetPropertyStore().Get(PassType::GEOMETRY, slots[si]);
            if (props.HasDestroyedResource() || fbProps.HasDestroyedResource())
            {
                return fail(NVGCompatibilityStatus::REJECTED_MISSING_USER_BINDING, "Material references a destroyed user texture, sampler, or GPUBuffer.");
            }

            const ShaderReflection& refl = st->GetReflection();
            for (const auto& tex : refl.textures)
            {
                if (tex.binding == 0xFFFF || tex.set >= 16u || tex.IsEngineProvided()) continue;
                BOOL found = false;
                auto checkTex = [&](const ShaderStageProps& sp) {
                    for (const auto& mt : sp.textures)
                    {
                        if ((mt.resourceName == tex.name ||
                             (mt.propertyName + "__" + mt.resourceName) == tex.groupQualifiedName ||
                             (mt.slotIndex == tex.binding && mt.registerSpace == tex.set)) &&
                            mt.texture != nullptr && mt.texture->GetNativeHandle() != 0)
                        {
                            found = true;
                            return;
                        }
                    }
                };
                checkTex(props);
                if (!found && pass != PassType::GEOMETRY) checkTex(fbProps);
                if (!found)
                {
                    return fail(NVGCompatibilityStatus::REJECTED_MISSING_USER_BINDING, "Material is missing a required user Add() Texture binding.");
                }
            }
            for (const auto& samp : refl.samplers)
            {
                if (samp.binding == 0xFFFF || samp.set >= 16u || samp.IsEngineProvided()) continue;
                BOOL found = false;
                auto checkSamp = [&](const ShaderStageProps& sp) {
                    for (const auto& ms : sp.samplers)
                    {
                        if ((ms.resourceName == samp.name ||
                             (ms.propertyName + "__" + ms.resourceName) == samp.groupQualifiedName ||
                             (ms.slotIndex == samp.binding && ms.registerSpace == samp.set)) &&
                            ms.sampler != nullptr && ms.sampler->GetNativeHandle() != 0 &&
                            ms.sampler->IsComparison() == samp.isComparison)
                        {
                            found = true;
                            return;
                        }
                    }
                };
                checkSamp(props);
                if (!found && pass != PassType::GEOMETRY) checkSamp(fbProps);
                if (!found)
                {
                    return fail(NVGCompatibilityStatus::REJECTED_MISSING_USER_BINDING, "Material is missing a required or matching user Add() Sampler binding.");
                }
            }
            for (const auto& nscSb : refl.structuredBuffers)
            {
                if (nscSb.binding == 0xFFFF || nscSb.set >= 16u || nscSb.stride == 0u) continue;
                const BOOL isReflUav = (refl.FindUAV(nscSb.name) != nullptr);
                if (isReflUav) continue;

                const StructuredBufferLayoutPlan plan = CoreResourceBindingResolver::AnalyzeStructuredBuffer(nscSb, stageTypes[si], 0u);
                if (!plan.valid || plan.domain == EngineResourceDomain::INVALID_MIXED_DOMAINS)
                {
                    return fail(NVGCompatibilityStatus::REJECTED_UNSUPPORTED_LAYOUT, plan.diagnostic);
                }

                if (plan.domain == EngineResourceDomain::USER_ONLY)
                {
                    BOOL foundUserSb = false;
                    auto checkUserSb = [&](const ShaderStageProps& sp) {
                        for (const auto& ms : sp.structuredBuffers)
                        {
                            if (!ms.isUAV &&
                                ((ms.propertyName + "__" + ms.resourceName) == nscSb.name ||
                                 ms.resourceName == nscSb.name ||
                                 (ms.slotIndex == nscSb.binding && ms.registerSpace == nscSb.set)))
                            {
                                const NativeHandle h = ms.gpuBuffer ? ms.gpuBuffer->GetNativeHandle() : ms.gpuBufferHandle;
                                const UINT32 stStride = ms.gpuBuffer ? ms.gpuBuffer->GetStride() : ms.gpuBufferStride;
                                const UINT32 cnt = (ms.userElementCount > 0u) ? ms.userElementCount : ms.elementCount;
                                if ((h != 0 && stStride == nscSb.stride) || (cnt > 0u && (!ms.userData.empty() || !ms.cpuData.empty())))
                                {
                                    foundUserSb = true;
                                    return;
                                }
                            }
                        }
                    };
                    checkUserSb(props);
                    if (!foundUserSb && pass != PassType::GEOMETRY) checkUserSb(fbProps);
                    if (!foundUserSb)
                    {
                        return fail(NVGCompatibilityStatus::REJECTED_MISSING_USER_BINDING,
                                    "Material is missing a required user Add() StructuredBuffer or GPUBuffer binding.");
                    }
                }
            }
        }

        const NVGInternalExecutionStrategy strategy = PlanExecutionStrategy(shaderSet, effectiveVariant, policy);
        const BOOL isUnindexedRawPull = (strategy == NVGInternalExecutionStrategy::VERTEX_PULL) &&
                                        !DoesShaderDeclareClusterOffsetField(vs);
        if (isUnindexedRawPull)
        {
            if (outReason) *outReason = "Compatible (internal strategy: raw vertex pulling).";
            return NVGCompatibilityStatus::COMPATIBLE;
        }

        // Indexed Cluster Pull & Decoded-IA indirect strategy: verify D3D12 ExecuteIndirect promoted descriptor budget
        // (up to 8 per-draw CBVs, up to 2 promoted Root SRVs [1 Instance + 1 Bone Slice], within <= 64 Root Signature DWORDs).
        UINT32 promotedCBVs      = 0u;
        UINT32 promotedInstSRVs  = 0u;
        UINT32 promotedBoneSRVs  = 0u;
        for (UINT32 si = 0u; si < 5u; ++si)
        {
            const Shader* st = stages[si];
            if (!st) continue;
            const ShaderReflection& refl = st->GetReflection();
            for (const auto& buf : refl.buffers)
            {
                if (buf.binding == 0xFFFF || buf.set >= 16u || buf.totalSize == 0) continue;
                BOOL needsPerDrawPatch = false;
                for (const auto& f : buf.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    if (pass == PassType::SHADOW)
                    {
                        if (CoreResourceBindingResolver::IsCBufferFieldShadowPageDynamic(pName))
                        {
                            needsPerDrawPatch = true;
                            break;
                        }
                    }
                    else if (CoreResourceBindingResolver::IsCBufferFieldPerDrawDynamic(pName))
                    {
                        needsPerDrawPatch = true;
                        break;
                    }
                }
                if (needsPerDrawPatch) ++promotedCBVs;
            }
            for (const auto& sb : refl.structuredBuffers)
            {
                if (sb.binding == 0xFFFF || sb.set >= 16u) continue;
                const EngineResourceDomain dom = CoreResourceBindingResolver::DetectStructuredBufferDomain(sb, stageTypes[si]);
                if (dom == EngineResourceDomain::INSTANCES || dom == EngineResourceDomain::INSTANCE_SKINNING)
                {
                    ++promotedInstSRVs;
                }
                else if (dom == EngineResourceDomain::PER_OBJECT_BONE_SLICE)
                {
                    ++promotedBoneSRVs;
                }
            }
        }

        const UINT32 promotedSRVs = promotedInstSRVs + promotedBoneSRVs;
        const UINT32 totalRootDwords = EstimatePromotedGraphicsRootDwords(stages, promotedCBVs, promotedSRVs);
        if (promotedCBVs > 8u || promotedInstSRVs > 1u || promotedBoneSRVs > 1u || promotedSRVs > 2u || totalRootDwords > 64u)
        {
            return fail(NVGCompatibilityStatus::REJECTED_ROOT_PROMOTION_LIMIT,
                        "Shader exceeds D3D12 indirect promoted root signature budget (>8 per-draw CBVs, >2 promoted SRVs, or >64 Root Signature DWORDs).");
        }

        if (outReason)
        {
            *outReason = (strategy == NVGInternalExecutionStrategy::VERTEX_PULL)
                ? "Compatible (internal strategy: indexed cluster pull)."
                : "Compatible (internal strategy: compact decoded IA / indirect).";
        }
        return NVGCompatibilityStatus::COMPATIBLE;
    }

    BOOL CoreNVGManager::HasVirtualObjectsInScene(const Scene* scene) const
    {
        if (!scene || !m_initialized || !m_config.enabled || m_residencyManager.GetRegisteredMeshCount() == 0u)
            return false;

        // O(1) fast path via incrementally tracked GPUScene virtual object counter
        const GPUSceneUpdateStats& gs = scene->GetGPUSceneStats();
        if (gs.activeObjectCount > 0u && gs.activeObjectCount == static_cast<UINT32>(scene->GetRenderObjects().size()))
        {
            return gs.virtualActiveObjectCount > 0u;
        }

        for (RenderObject* obj : scene->GetRenderObjects())
        {
            if (!obj || !obj->HasFlag(RenderObjectFlags::VISIBLE)) continue;
            const Mesh* m = obj->GetMesh();
            if (m && m->IsVirtualized()) return true;
        }
        return false;
    }

    void CoreNVGManager::BeginFrameTick(
        UINT64                                        frameIndex,
        Scene*                                        activeScene,
        std::vector<NVSDynamicCasterInvalidationBox>& outNvsInvalidations)
    {
        m_currentFrameIndex = frameIndex;
        m_residencyMaskDispatchedThisFrame = false;
        m_stats.ResetPerFrame();
        m_stats.active = false;

        if (!m_initialized || !m_config.enabled)
        {
            if (m_hasLastShadowConfigState && m_lastShadowEnabled && activeScene != nullptr)
            {
                // NVG was just disabled: invalidate shadow volumes of virtualized casters so NVS refreshes cleanly
                for (RenderObject* obj : activeScene->GetRenderObjects())
                {
                    if (!obj || !obj->HasFlag(RenderObjectFlags::VISIBLE) || !obj->HasFlag(RenderObjectFlags::CAST_SHADOW)) continue;
                    const Mesh* m = obj->GetMesh();
                    if (!m || !m->IsVirtualized()) continue;
                    const AABB& wb = obj->GetWorldAABB();
                    NVSDynamicCasterInvalidationBox box{};
                    box.worldMin          = wb.m_Min;
                    box.worldMax          = wb.m_Max;
                    box.gpuSceneSlotIndex = obj->GetGPUSceneHandle(activeScene).slotIndex;
                    box.invalidationFlags = 1u;
                    outNvsInvalidations.push_back(box);
                }
                m_lastShadowEnabled = false;
            }
            return;
        }

        // 1. Run non-blocking virtual geometry streaming, leaf-first LRU, elastic pool & telemetry refresh
        std::vector<UINT32>                    residencyChangedMeshIds;
        std::vector<NVGResidencyChangedRegion> changedRegions;
        m_residencyManager.ProcessStreamingFrame(frameIndex, residencyChangedMeshIds, m_stats, &changedRegions);
        if (m_residencyManager.GetRegisteredMeshCount() == 0u)
        {
            return;
        }

        // 2. Detect global NVG shadow config or per-material shadow policy mutations, accumulate effectiveSceneTriangles,
        //    and emit localized sub-mesh NVS shadow invalidations (instead of invalidating entire mountain AABBs!)
        const FLOAT32 curEffShadowErr = ResolveEffectiveShadowTexelError();
        const FLOAT32 prevShadowRef   = (std::max)(m_lastShadowBaseTexelError, 0.05f);
        const BOOL    shadowErrMoved  = std::fabs(curEffShadowErr - m_lastShadowBaseTexelError) > (prevShadowRef * 0.15f + 1e-4f);
        const BOOL globalShadowCfgChanged = m_hasLastShadowConfigState &&
            (!m_lastShadowEnabled ||
             shadowErrMoved ||
             std::fabs(m_config.thinFeatureShadowBoost - m_lastThinShadowBoost) > 1e-4f);
        m_hasLastShadowConfigState = true;
        m_lastShadowEnabled        = true;
        if (!m_hasLastShadowConfigState || shadowErrMoved || m_lastShadowBaseTexelError <= 0.0f)
        {
            m_lastShadowBaseTexelError = curEffShadowErr;
        }
        m_lastThinShadowBoost = m_config.thinFeatureShadowBoost;

        if (activeScene != nullptr)
        {
            struct MeshDirtyRegionList
            {
                std::vector<NVGResidencyChangedRegion> regions;
                AABB                                   mergedLocalBounds{};
                FLOAT32                                maxParentError = 0.0f;
                BOOL                                   hasMerged      = false;
            };
            std::unordered_map<UINT32, MeshDirtyRegionList> meshDirtyRegions;
            if (!changedRegions.empty())
            {
                for (const NVGResidencyChangedRegion& reg : changedRegions)
                {
                    MeshDirtyRegionList& dst = meshDirtyRegions[reg.meshId];
                    dst.regions.push_back(reg);
                    if (!dst.hasMerged)
                    {
                        dst.mergedLocalBounds = reg.localBounds;
                        dst.hasMerged         = true;
                    }
                    else
                    {
                        dst.mergedLocalBounds.Expand(reg.localBounds.m_Min);
                        dst.mergedLocalBounds.Expand(reg.localBounds.m_Max);
                    }
                    if (reg.maxShadowScaledParentError > dst.maxParentError)
                    {
                        dst.maxParentError = reg.maxShadowScaledParentError;
                    }
                }
            }

            // Check whether scene has any non-directional (Point/Spot) shadow light
            BOOL hasLocalShadowLights = false;
            for (const Light* l : activeScene->GetLights())
            {
                if (l && l->HasFlag(LightFlags::ENABLED) && l->HasFlag(LightFlags::CAST_SHADOW) &&
                    l->GetType() != LightType::DIRECTIONAL)
                {
                    hasLocalShadowLights = true;
                    break;
                }
            }
            // Gather ALL active camera origins in the scene so distant directional invalidation culling
            // is conservative across every active viewpoint in multi-camera scenes (NVG-018)
            std::vector<Vec3> activeCamOrigins;
            for (const Camera* scCam : activeScene->GetCameras())
            {
                if (scCam) activeCamOrigins.push_back(scCam->GetPosition());
            }
            if (activeCamOrigins.empty() && activeScene->GetMainCamera() != nullptr)
            {
                activeCamOrigins.push_back(activeScene->GetMainCamera()->GetPosition());
            }
            const BOOL canCullFarDirInvalidations = (!hasLocalShadowLights && !activeCamOrigins.empty());

            std::unordered_map<const Material*, BOOL> matShadowDirty;
            UINT64 sceneVirtualTris = 0ull;

            for (RenderObject* obj : activeScene->GetRenderObjects())
            {
                if (!obj || !obj->HasFlag(RenderObjectFlags::VISIBLE)) continue;
                const Mesh* m = obj->GetMesh();
                if (!m || !m->IsVirtualized()) continue;

                const NVGMeshInfo& nvgInfo = m->GetVirtualMeshInfo();
                const UINT32 meshTris = (nvgInfo.buildReport.validTriangleCount > 0u)
                    ? nvgInfo.buildReport.validTriangleCount
                    : (m->GetIndexCount() / 3u);
                sceneVirtualTris += static_cast<UINT64>(meshTris);

                if (!obj->HasFlag(RenderObjectFlags::CAST_SHADOW)) continue;

                FLOAT32 disp       = 0.0f;
                FLOAT32 matShScale = 1.0f;
                BOOL    matLock0   = false;
                BOOL    matDirty   = false;
                BOOL    foundShMat = false;
                const UINT32 objMatSlots = (std::max)(1u, obj->GetMaterialSlotCount());
                for (UINT32 ms = 0u; ms < objMatSlots; ++ms)
                {
                    const Material* mat = obj->GetMaterial(ms);
                    if (!mat) continue;
                    const NVGMaterialPolicy& mp = mat->GetNVGPolicy();
                    disp = (std::max)(disp, (std::max)(0.0f, mp.maxWorldDisplacement));
                    if (!foundShMat || mp.shadowLodErrorScale < matShScale)
                    {
                        matShScale = (std::max)(0.001f, mp.shadowLodErrorScale);
                        foundShMat = true;
                    }
                    if (mp.lockToLOD0) matLock0 = true;

                    auto itMD = matShadowDirty.find(mat);
                    if (itMD != matShadowDirty.end())
                    {
                        if (itMD->second) matDirty = true;
                    }
                    else
                    {
                        const UINT32 curHash = HashMaterialShadowPolicy(mat);
                        auto itPrev = m_lastMaterialShadowHash.find(mat);
                        const BOOL thisDirty = (itPrev != m_lastMaterialShadowHash.end() && itPrev->second != curHash);
                        m_lastMaterialShadowHash[mat] = curHash;
                        matShadowDirty[mat]           = thisDirty;
                        if (thisDirty) matDirty = true;
                    }
                }

                const UINT32 slotIdx = obj->GetGPUSceneHandle(activeScene).slotIndex;
                if (globalShadowCfgChanged || matDirty)
                {
                    const AABB& wb = obj->GetWorldAABB();
                    NVSDynamicCasterInvalidationBox box{};
                    box.worldMin          = wb.m_Min - Vec3(disp, disp, disp);
                    box.worldMax          = wb.m_Max + Vec3(disp, disp, disp);
                    box.gpuSceneSlotIndex = slotIdx;
                    box.invalidationFlags = 1u;
                    outNvsInvalidations.push_back(box);
                }
                else if (!meshDirtyRegions.empty())
                {
                    const UINT32 mid = m->GetGPUSceneMeshId();
                    const auto itReg = meshDirtyRegions.find(mid);
                    if (itReg != meshDirtyRegions.end() && itReg->second.hasMerged)
                    {
                        const Mat4&   wm           = obj->GetWorldMatrix();
                        const FLOAT32 maxAxisScale = ComputeMaxWorldScaleUpperBoundCPU(wm);

                        const AABB&   wb     = obj->GetWorldAABB();
                        const FLOAT32 objRad = wb.Extents().Length() + disp;
                        FLOAT32 minCamDist   = activeCamOrigins.empty() ? 0.0f : 1.0e30f;
                        for (const Vec3& co : activeCamOrigins)
                        {
                            const FLOAT32 d = (wb.Center() - co).Length();
                            if (d < minCamDist) minCamDist = d;
                        }
                        const FLOAT32 safeDist = (std::max)(0.0f, minCamDist - objRad);

                        // Exact directional clipmap lower-bound: any directional cascade overlapping obj has half-extent >= safeDist
                        // and virtual resolution <= 16384 -> texelWorldSize >= (2 * safeDist) / 16384 = safeDist / 8192.
                        // If even the coarsest changed page's parent error is strictly below threshold, obj's shadow cut is unchanged!
                        if (canCullFarDirInvalidations && !matLock0 && maxAxisScale >= 0.0f && safeDist > 48.0f)
                        {
                            const FLOAT32 minCascadeTexel = safeDist * (1.0f / 8192.0f);
                            const FLOAT32 maxWorldErr     = itReg->second.maxParentError * maxAxisScale;
                            if (maxWorldErr <= minCascadeTexel * curEffShadowErr * matShScale)
                            {
                                continue;
                            }
                        }

                        // Coalesce into 1 merged box for distant instances or high-instance-count scenes
                        if (safeDist > 48.0f || outNvsInvalidations.size() >= 128u)
                        {
                            const AABB tightWB = TransformLocalAABBToWorldCPU(wm, itReg->second.mergedLocalBounds, disp);
                            NVSDynamicCasterInvalidationBox box{};
                            box.worldMin          = tightWB.m_Min;
                            box.worldMax          = tightWB.m_Max;
                            box.gpuSceneSlotIndex = slotIdx;
                            box.invalidationFlags = 1u;
                            outNvsInvalidations.push_back(box);
                        }
                        else
                        {
                            for (const NVGResidencyChangedRegion& reg : itReg->second.regions)
                            {
                                const AABB tightWB = TransformLocalAABBToWorldCPU(wm, reg.localBounds, disp);
                                NVSDynamicCasterInvalidationBox box{};
                                box.worldMin          = tightWB.m_Min;
                                box.worldMax          = tightWB.m_Max;
                                box.gpuSceneSlotIndex = slotIdx;
                                box.invalidationFlags = 1u;
                                outNvsInvalidations.push_back(box);
                            }
                        }
                    }
                }
            }
            m_stats.effectiveSceneTriangles = sceneVirtualTris;
        }

        // 3. Poll non-blocking GPU telemetry readback from completed camera frames & grow queues 2x on demand
        UINT64 totalCamLodSum = 0ull;
        UINT64 totalCamVisCnt = 0ull;
        UINT64 totalShLodSum  = 0ull;
        UINT64 totalShVisCnt  = 0ull;
        UINT32 maxNodeCapStat = kNVGNodeQueueCap;
        UINT32 maxCandCapStat = kNVGCandidateCap;

        for (auto& kv : m_cameraContexts)
        {
            CameraNVGContext* ctx = kv.second.get();
            if (!ctx) continue;

            if (ctx->telemetryPending && ctx->telemetryBuffer != nullptr)
            {
                NVGTelemetryGPUData gpuTel{};
                if (m_bufferManager->TryGetGPUBufferReadbackData(ctx->telemetryBuffer, &gpuTel, sizeof(gpuTel)))
                {
                    ctx->telemetryPending = false;
                    std::memcpy(ctx->lastTelemetryWords, &gpuTel, sizeof(gpuTel));
                    ctx->hasLastTelemetry = true;

                    // Autonomous 2x GPU queue & Path-B pool growth on overflow or >75% camera/shadow watermark
                    const UINT32 shOverflowPages = gpuTel.reserved0;
                    const UINT32 shPeakDemand    = gpuTel.reserved1;
                    const UINT32 camVisTotal     = gpuTel.visibleOpaqueClusters + gpuTel.visibleTransparentClusters;
                    if (gpuTel.overflowDroppedClusters > 0u ||
                        shOverflowPages > 0u ||
                        gpuTel.clustersTested > (ctx->dynCandidateCap * 3u) / 4u ||
                        camVisTotal > (ctx->dynVisibleCap * 3u) / 4u ||
                        gpuTel.clustersHZBCulled > (ctx->dynOccludedClusterCap * 3u) / 4u ||
                        gpuTel.decodedUniqueClusters > (ctx->dynDecodeClusterCap * 3u) / 4u ||
                        shPeakDemand > (ctx->dynCandidateCap * 3u) / 4u ||
                        shPeakDemand > (ctx->dynNodeQueueCap * 3u) / 4u ||
                        shPeakDemand > (ctx->dynShadowVisibleCap * 3u) / 4u)
                    {
                        ctx->dynNodeQueueCap        = (std::min)(kNVGMaxQueueCap, ctx->dynNodeQueueCap * 2u);
                        ctx->dynCandidateCap        = (std::min)(kNVGMaxQueueCap, ctx->dynCandidateCap * 2u);
                        ctx->dynVisibleCap          = (std::min)(kNVGMaxVisibleCap, ctx->dynVisibleCap * 2u);
                        ctx->dynShadowVisibleCap    = (std::min)(kNVGMaxShadowVisibleCap, ctx->dynShadowVisibleCap * 2u);
                        ctx->dynOccludedClusterCap  = (std::min)(kNVGMaxQueueCap, ctx->dynOccludedClusterCap * 2u);
                        ctx->dynDecodeClusterCap    = (std::min)(262144u, ctx->dynDecodeClusterCap * 2u);
                        ctx->dynCameraCmdsPerBucket = (std::min)(kNVGMaxCommandsPerBucket, ctx->dynCameraCmdsPerBucket * 2u);
                        ctx->dynShadowCmdsPerBucket = (std::min)(kNVGMaxCommandsPerBucket, ctx->dynShadowCmdsPerBucket * 2u);
                        ctx->dynCBVWords            = (std::min)(kNVGMaxPerDrawCBVWords, ctx->dynCBVWords * 2u);
                    }
                }
            }

            if (ctx->dynNodeQueueCap > maxNodeCapStat) maxNodeCapStat = ctx->dynNodeQueueCap;
            if (ctx->dynCandidateCap > maxCandCapStat) maxCandCapStat = ctx->dynCandidateCap;

            if (ctx->hasLastTelemetry && ctx->usedThisFrame)
            {
                NVGTelemetryGPUData gpuTel{};
                std::memcpy(&gpuTel, ctx->lastTelemetryWords, sizeof(gpuTel));
                m_stats.nvgObjectsEvaluated           += gpuTel.objectsEvaluated;
                m_stats.nvgObjectsFrustumCulled       += gpuTel.objectsFrustumCulled;
                m_stats.nvgObjectsHZBCulled           += gpuTel.objectsHZBCulled;
                m_stats.clustersTested                += gpuTel.clustersTested;
                m_stats.clustersFrustumCulled         += gpuTel.clustersFrustumCulled;
                m_stats.clustersConeCulled            += gpuTel.clustersConeCulled;
                m_stats.clustersHZBCulled             += gpuTel.clustersHZBCulled;
                m_stats.clustersSmallCulled           += gpuTel.clustersSmallCulled;
                m_stats.visibleOpaqueClusters         += gpuTel.visibleOpaqueClusters;
                m_stats.visibleTransparentClusters    += gpuTel.visibleTransparentClusters;
                m_stats.clustersDisocclusionRecovered += gpuTel.disocclusionRecovered;
                m_stats.fallbackClustersThisFrame     += gpuTel.fallbackClusters;
                m_stats.pageFaultsThisFrame           += gpuTel.pageFaults;
                m_stats.decodedUniqueClusters         += gpuTel.decodedUniqueClusters;
                m_stats.indirectDrawCommandsGenerated += gpuTel.indirectDrawsGenerated;
                m_stats.overflowDroppedClusters       += gpuTel.overflowDroppedClusters;
                m_stats.shadowCastersEvaluated        += gpuTel.shadowCastersEvaluated;
                m_stats.shadowClustersTested          += gpuTel.shadowClustersTested;
                m_stats.shadowClustersVisible         += gpuTel.shadowClustersVisible;
                m_stats.shadowSWRasterizedClusters    += gpuTel.shadowSWRasterizedClusters;
                m_stats.shadowIndirectDrawsGenerated  += gpuTel.shadowIndirectDraws;
                m_stats.shadowOverflowPages           += gpuTel.reserved0;
                m_stats.swRasterizedClusters          += gpuTel.swRasterizedClusters;
                m_stats.swFallbackToHWClusters        += gpuTel.swFallbackToHWClusters;
                m_stats.swRasterizedTriangles         += gpuTel.swRasterizedTriangles;
                m_stats.swCoveredPixels               += gpuTel.swCoveredPixels;

                for (UINT32 i = 0u; i < 8u; ++i)
                {
                    m_stats.cameraLODHistogram[i] += gpuTel.cameraLODHistogram[i];
                    m_stats.shadowLODHistogram[i] += gpuTel.shadowLODHistogram[i];
                }
                const UINT32 totalCamVis = gpuTel.visibleOpaqueClusters + gpuTel.visibleTransparentClusters;
                totalCamLodSum += static_cast<UINT64>(gpuTel.cameraLODSum);
                totalCamVisCnt += static_cast<UINT64>(totalCamVis);
                totalShLodSum  += static_cast<UINT64>(gpuTel.shadowLODSum);
                totalShVisCnt  += static_cast<UINT64>(gpuTel.shadowClustersVisible);
            }

            ctx->usedThisFrame          = false;
            ctx->cameraPassRanThisFrame = false;
        }
        m_stats.shadowNodeQueueCapacity = maxNodeCapStat;
        m_stats.shadowCandidateCapacity = maxCandCapStat;
        if (totalCamVisCnt > 0ull)
        {
            m_stats.cameraAverageLODLevel = static_cast<FLOAT32>(static_cast<double>(totalCamLodSum) / static_cast<double>(totalCamVisCnt));
        }
        if (totalShVisCnt > 0ull)
        {
            m_stats.shadowAverageLODLevel = static_cast<FLOAT32>(static_cast<double>(totalShLodSum) / static_cast<double>(totalShVisCnt));
        }

        // 4. Sync active scene's NVG material compatibility & draw bucket routing tables
        if (activeScene && HasVirtualObjectsInScene(activeScene))
        {
            if (EnsureRuntimeKernelsLoaded())
            {
                SyncSceneMaterialsAndBuckets(activeScene);
                m_stats.active = true;
            }
        }
    }

    BOOL CoreNVGManager::BuildReflectedBindingForBucket(
        NVGDrawBucketEntry&                  bucket,
        Scene*                               scene,
        FLOAT32                              timeSeconds,
        std::vector<NVSDrawGroupFieldPatch>& ioPatches,
        std::vector<UINT32>&                 ioTemplateWords)
    {
        bucket.indirectRootKey = IndirectRootBindingKey{};
        bucket.bindingDesc     = NVGDrawGroupShaderBindingDesc{};
        bucket.promotedCBVSpans.clear();
        bucket.decodeVertexWords = 0u;
        bucket.decodeAttrMask    = 0u;

        if (bucket.isVertexPull || !bucket.material || !bucket.shaderSet)
        {
            return true;
        }

        const Shader* vs = bucket.shaderSet->GetVertexShader(bucket.variant);
        const Shader* ps = bucket.shaderSet->GetPixelShader();
        const Shader* gs = bucket.shaderSet->GetGeometryShader();
        const Shader* hs = bucket.shaderSet->GetHullShader();
        const Shader* ds = bucket.shaderSet->GetDomainShader();

        if (bucket.isIndexedPull)
        {
            bucket.decodeVertexWords = 0u;
            bucket.decodeAttrMask    = 0u;
        }
        else
        {
            ComputeRequiredIAVertexWordsAndMask(vs, bucket.decodeVertexWords, bucket.decodeAttrMask);
            if (bucket.decodeVertexWords == 0u)
            {
                // Procedural / system-ID-only VS without NVG pull buffers: no IA vertex attributes needed
                bucket.decodeAttrMask = 0u;
            }
        }

        const Shader* stages[5] = { vs, ps, gs, hs, ds };
        const ShaderStage stageTypes[5] = {
            ShaderStage::VERTEX, ShaderStage::PIXEL, ShaderStage::GEOMETRY,
            ShaderStage::HULL, ShaderStage::DOMAIN
        };
        const GraphicsShaderSlot slots[5] = {
            GraphicsShaderSlot::VERTEX, GraphicsShaderSlot::PIXEL, GraphicsShaderSlot::GEOMETRY,
            GraphicsShaderSlot::HULL, GraphicsShaderSlot::DOMAIN_STAGE
        };

        NVGDrawGroupShaderBindingDesc bd{};
        bd.bindingFlags      = 1u;
        if (bucket.decodeVertexWords == 0u)
        {
            bd.bindingFlags |= 4u; // Null VBV (Indexed Cluster Pull or system-ID-only shader)
        }
        bd.patchFirstIndex   = static_cast<UINT32>(ioPatches.size());
        bd.drawGroupIndex    = bucket.bucketIndex;
        bd.decodeVertexWords = bucket.decodeVertexWords;
        bd.decodeAttrMask    = bucket.decodeAttrMask;

        IndirectRootBindingKey rkey{};
        rkey.enabled = true;

        UINT32 cbvSizes[8]      = {};
        UINT32 cbvTplOffsets[8] = {};

        StageResolveContext initCtx{};
        initCtx.executionContextMask = EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_DRAW | EXEC_CTX_CLUSTER |
                                       (bucket.isShadowPass ? EXEC_CTX_SHADOW_PAGE : EXEC_CTX_NONE);
        initCtx.scene                = scene;
        initCtx.material             = bucket.material;
        initCtx.passType             = bucket.passType;
        initCtx.geometryMode         = static_cast<UINT32>(GeometryMode::VIRTUALIZED);
        initCtx.nvgDrawBucketId      = bucket.bucketIndex;
        initCtx.nvgVisibleClusterOff = kNVGInvalidIndex;
        initCtx.nvgVisibleClusterCnt = 0u;
        initCtx.nvgDebugMode         = static_cast<UINT32>(m_config.debugVisualizationMode);
        initCtx.nvgMaxScreenErrorPx  = ResolveEffectiveScreenErrorPixels();
        initCtx.nvgPhysicalPoolCap   = m_residencyManager.GetPhysicalPoolCapacity();

        for (UINT32 si = 0u; si < 5u; ++si)
        {
            const Shader* st = stages[si];
            if (!st || !st->IsValid()) continue;
            bd.activeStageMask |= (1u << si);

            const ShaderStageProps& props   = bucket.material->GetPropertyStore().Get(bucket.passType, slots[si]);
            const ShaderStageProps& fbProps = bucket.material->GetPropertyStore().Get(PassType::GEOMETRY, slots[si]);
            const ShaderReflection& refl = st->GetReflection();

            // Promote up to 8 per-draw CBVs containing dynamic transform / camera / NVS / skinning / NVG cluster fields
            for (const auto& nscBuf : refl.buffers)
            {
                if (nscBuf.binding == 0xFFFF || nscBuf.set >= 16u || nscBuf.totalSize == 0) continue;

                std::vector<NVSDrawGroupFieldPatch> localPatches;
                BOOL hasTriggerSemantic = false;
                for (const auto& f : nscBuf.fields)
                {
                    if (!f.isGet) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    const UINT32 sem = CoreResourceBindingResolver::MapPropertyNameToPatchSemantic(pName);
                    if (sem != NVS_PATCH_SEMANTIC_NONE && !IsNVSInstancePatchSemantic(sem))
                    {
                        NVSDrawGroupFieldPatch p{};
                        p.semanticId = sem;
                        p.byteOffset = f.offset;
                        p.byteSize   = f.size;
                        localPatches.push_back(p);
                        if (bucket.isShadowPass
                            ? CoreResourceBindingResolver::IsCBufferFieldShadowPageDynamic(pName)
                            : CoreResourceBindingResolver::IsCBufferFieldPerDrawDynamic(pName))
                        {
                            hasTriggerSemantic = true;
                        }
                    }
                }
                if (localPatches.empty() || !hasTriggerSemantic) continue;
                if (rkey.cbvCount >= 8u)
                {
                    return false;
                }

                const UINT32 slotIdx   = rkey.cbvCount++;
                rkey.cbvSpace[slotIdx] = static_cast<UINT16>(nscBuf.set);
                rkey.cbvSlot[slotIdx]  = static_cast<UINT16>(nscBuf.binding);
                rkey.cbvStage[slotIdx] = stageTypes[si];

                const UINT32 wordCount = (nscBuf.totalSize + 3u) >> 2u;
                std::vector<BYTE> tplBytes(static_cast<SIZE_T>(wordCount) * 4u, 0u);

                for (const auto& f : nscBuf.fields)
                {
                    if (!f.isGet && !f.defaultData.empty() && f.offset + f.defaultData.size() <= tplBytes.size())
                    {
                        std::memcpy(tplBytes.data() + f.offset, f.defaultData.data(), f.defaultData.size());
                    }
                }
                auto copyMatBuf = [&](const ShaderStageProps& sp) -> BOOL {
                    for (const auto& mbs : sp.buffers)
                    {
                        if ((mbs.propertyName + "__" + mbs.resourceName) == nscBuf.name ||
                            (mbs.registerSpace == nscBuf.set && mbs.slotIndex == nscBuf.binding))
                        {
                            if (!mbs.cpuData.empty())
                            {
                                const SIZE_T nCopy = (std::min)(tplBytes.size(), mbs.cpuData.size());
                                std::memcpy(tplBytes.data(), mbs.cpuData.data(), nCopy);
                                return true;
                            }
                        }
                    }
                    return false;
                };
                if (!copyMatBuf(props) && bucket.passType != PassType::GEOMETRY)
                {
                    copyMatBuf(fbProps);
                }

                for (const auto& f : nscBuf.fields)
                {
                    if (!f.isGet || f.offset + f.size > tplBytes.size()) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    CoreResourceBindingResolver::FillEngineConstantBufferField(
                        pName, initCtx, tplBytes.data(), f.offset, f.size, nscBuf.totalSize);
                    if ((pName == "TIME" || pName == "ENGINE_TIME") && f.size >= sizeof(FLOAT32))
                    {
                        std::memcpy(tplBytes.data() + f.offset, &timeSeconds, sizeof(FLOAT32));
                    }
                }

                cbvSizes[slotIdx]      = nscBuf.totalSize;
                cbvTplOffsets[slotIdx] = static_cast<UINT32>(ioTemplateWords.size());

                NVGPromotedCBVTemplateSpan span{};
                span.nscBuf             = &nscBuf;
                span.stage              = stageTypes[si];
                span.templateWordOffset = cbvTplOffsets[slotIdx];
                span.byteSize           = nscBuf.totalSize;
                bucket.promotedCBVSpans.push_back(span);

                for (UINT32 w = 0u; w < wordCount; ++w)
                {
                    UINT32 val = 0u;
                    std::memcpy(&val, tplBytes.data() + static_cast<SIZE_T>(w) * 4u, sizeof(UINT32));
                    ioTemplateWords.push_back(val);
                }

                for (auto& p : localPatches)
                {
                    p.targetSlot = slotIdx;
                    ioPatches.push_back(p);
                }
            }

            // Promote up to 2 Root SRVs (Instance Pool + Per-Object Bone Slice; GLOBAL_BONE_MATRIX stays in descriptor table)
            for (const auto& nscSb : refl.structuredBuffers)
            {
                if (nscSb.binding == 0xFFFF || nscSb.set >= 16u) continue;
                const EngineResourceDomain dom = CoreResourceBindingResolver::DetectStructuredBufferDomain(nscSb, stageTypes[si]);
                const BOOL isInst      = (dom == EngineResourceDomain::INSTANCES || dom == EngineResourceDomain::INSTANCE_SKINNING);
                const BOOL isBoneSlice = (dom == EngineResourceDomain::PER_OBJECT_BONE_SLICE);

                if (isInst || isBoneSlice)
                {
                    if (rkey.srvCount >= 2u) return false;
                    const UINT32 srvSlotIdx = rkey.srvCount++;
                    rkey.hasPromotedSRV     = true;
                    if (srvSlotIdx == 0u)
                    {
                        rkey.srvSpace = static_cast<UINT16>(nscSb.set);
                        rkey.srvSlot  = static_cast<UINT16>(nscSb.binding);
                        rkey.srvStage = stageTypes[si];
                    }
                    else
                    {
                        rkey.srv2Space = static_cast<UINT16>(nscSb.set);
                        rkey.srv2Slot  = static_cast<UINT16>(nscSb.binding);
                        rkey.srv2Stage = stageTypes[si];
                    }

                    const UINT32 modeVal = isInst ? NVS_INDIRECT_SRV_INSTANCE_POOL : NVS_INDIRECT_SRV_BONE_PALETTE_SLICE;
                    if (srvSlotIdx == 0u) bd.srv0Mode = modeVal;
                    else                  bd.srv1Mode = modeVal;

                    if (isInst && bd.instanceStrideBytes == 0u)
                    {
                        const BOOL allowTier1Names = CoreResourceBindingResolver::IsTier1InstanceStructuredBuffer(nscSb, stageTypes[si]);
                        std::vector<NVSDrawGroupFieldPatch> instPatches;
                        for (const auto& f : nscSb.fields)
                        {
                            UINT32 sem = NVS_PATCH_SEMANTIC_NONE;
                            if (f.isGet)
                            {
                                const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                                const UINT32 s = CoreResourceBindingResolver::MapPropertyNameToPatchSemantic(pName);
                                if (IsNVSInstancePatchSemantic(s)) sem = s;
                            }
                            else if (allowTier1Names)
                            {
                                if (f.name == "worldMatrix" && f.size == 64u) sem = NVS_PATCH_INSTANCE_WORLD_MATRIX;
                                else if (f.name == "instanceId" && f.size == 4u) sem = NVS_PATCH_INSTANCE_ID;
                                else if (f.name == "customData" && f.size == 4u) sem = NVS_PATCH_INSTANCE_CUSTOM_DATA;
                            }
                            if (sem != NVS_PATCH_SEMANTIC_NONE && f.offset + f.size <= nscSb.stride)
                            {
                                NVSDrawGroupFieldPatch p{};
                                p.semanticId = sem;
                                p.targetSlot = 8u; // Slot 8 = Promoted Instance SRV
                                p.byteOffset = f.offset;
                                p.byteSize   = f.size;
                                instPatches.push_back(p);
                            }
                        }
                        bd.instanceStrideBytes        = (std::max)(nscSb.stride, 16u);
                        bd.instanceTemplateWordOffset = static_cast<UINT32>(ioTemplateWords.size());
                        const UINT32 iWords           = (bd.instanceStrideBytes + 3u) >> 2u;
                        for (UINT32 w = 0u; w < iWords; ++w) ioTemplateWords.push_back(0u);
                        for (const auto& p : instPatches) ioPatches.push_back(p);
                    }
                }
            }
        }

        rkey.extended = (rkey.cbvCount > 3u) || (rkey.srvCount > 1u);
        if (rkey.extended)
        {
            bd.bindingFlags |= 2u; // Extended 144-byte indirect command record
        }

        bd.cbvCount = rkey.cbvCount;
        for (UINT32 c = 0u; c < 4u; ++c)
        {
            bd.cbvByteSize[c]              = cbvSizes[c];
            bd.cbvExtByteSize[c]           = cbvSizes[4u + c];
            bd.cbvTemplateWordOffset[c]    = cbvTplOffsets[c];
            bd.cbvExtTemplateWordOffset[c] = cbvTplOffsets[4u + c];
        }
        bd.patchCount = static_cast<UINT32>(ioPatches.size()) - bd.patchFirstIndex;

        bucket.indirectRootKey = rkey;
        bucket.bindingDesc     = bd;
        return true;
    }

    void CoreNVGManager::RefreshBucketTemplatesForView(
        const FrameView& frameView,
        Scene*           scene,
        BOOL             isShadowPass)
    {
        auto& buckets  = isShadowPass ? m_shadowBuckets       : m_cameraBuckets;
        auto& tpls     = isShadowPass ? m_shadowTemplateWords : m_cameraTemplateWords;
        GPUBuffer*& tb = isShadowPass ? m_shadowTemplatesBuf  : m_cameraTemplatesBuf;
        if (buckets.empty() || tpls.empty() || !tb)
        {
            return;
        }

        StageResolveContext ctx{};
        ctx.executionContextMask = EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_DRAW | EXEC_CTX_CLUSTER |
                                   (isShadowPass ? EXEC_CTX_SHADOW_PAGE : EXEC_CTX_NONE);
        ctx.frameView            = &frameView;
        ctx.scene                = scene;
        ctx.geometryMode         = static_cast<UINT32>(GeometryMode::VIRTUALIZED);
        ctx.nvgVisibleClusterOff = kNVGInvalidIndex;
        ctx.nvgVisibleClusterCnt = 0u;
        ctx.nvgDebugMode         = static_cast<UINT32>(m_config.debugVisualizationMode);
        ctx.nvgMaxScreenErrorPx  = ResolveEffectiveScreenErrorPixels();
        ctx.nvgPhysicalPoolCap   = m_residencyManager.GetPhysicalPoolCapacity();

        BOOL anyModified = false;
        for (const auto& b : buckets)
        {
            if (b.promotedCBVSpans.empty() || !b.material) continue;
            ctx.material        = b.material;
            ctx.passType        = b.passType;
            ctx.nvgDrawBucketId = b.bucketIndex;

            for (const auto& sp : b.promotedCBVSpans)
            {
                if (!sp.nscBuf || sp.byteSize == 0u) continue;
                const UINT32 wordCount = (sp.byteSize + 3u) >> 2u;
                if (static_cast<SIZE_T>(sp.templateWordOffset) + wordCount > tpls.size()) continue;
                BYTE* dstBytes = reinterpret_cast<BYTE*>(tpls.data() + sp.templateWordOffset);
                for (const auto& f : sp.nscBuf->fields)
                {
                    if (!f.isGet || f.offset + f.size > wordCount * 4u) continue;
                    const std::string& pName = f.propertyName.empty() ? f.name : f.propertyName;
                    CoreResourceBindingResolver::FillEngineConstantBufferField(
                        pName, ctx, dstBytes, f.offset, f.size, sp.byteSize);
                    anyModified = true;
                }
            }
        }

        if (anyModified)
        {
            m_bufferManager->UpdateGPUBuffer(
                tb,
                tpls.data(),
                static_cast<UINT32>(tpls.size() * sizeof(UINT32)),
                0u);
        }
    }

    UINT32 CoreNVGManager::RegisterDrawBucket(
        Material*     material,
        PassType      passType,
        ShaderVariant variant,
        BOOL          isVertexPull,
        BOOL          isTransparent,
        BOOL          isShadowPass,
        BOOL          isDepthPass,
        Scene*        scene,
        FLOAT32       timeSeconds)
    {
        auto& buckets  = isShadowPass ? m_shadowBuckets       : m_cameraBuckets;
        auto& flags    = isShadowPass ? m_shadowBucketFlags   : m_cameraBucketFlags;
        auto& descs    = isShadowPass ? m_shadowBindingDescs  : m_cameraBindingDescs;
        auto& patches  = isShadowPass ? m_shadowFieldPatches  : m_cameraFieldPatches;
        auto& tpls     = isShadowPass ? m_shadowTemplateWords : m_cameraTemplateWords;

        for (const auto& existing : buckets)
        {
            if (existing.material == material &&
                existing.passType == passType &&
                existing.variant == variant &&
                existing.isDepthPass == isDepthPass)
            {
                return existing.bucketIndex;
            }
        }

        if (buckets.size() >= kNVGMaxDrawBuckets)
        {
            return kNVGInvalidIndex;
        }

        NVGDrawBucketEntry entry{};
        entry.bucketIndex   = static_cast<UINT32>(buckets.size());
        entry.material      = material;
        entry.shaderSet     = material ? material->GetShaderSet(passType) : nullptr;
        entry.passType      = passType;
        entry.variant       = variant;
        entry.isVertexPull  = isVertexPull;
        entry.isIndexedPull = false;
        entry.isTransparent = isTransparent;
        entry.isShadowPass  = isShadowPass;
        entry.isDepthPass   = isDepthPass;

        const Shader* vs = entry.shaderSet ? entry.shaderSet->GetVertexShader(variant) : nullptr;
        if (entry.isVertexPull && DoesShaderDeclareClusterOffsetField(vs))
        {
            // Upgrade attribute-less NVG pulling shader to Path-A Indexed Cluster Pull:
            // 0 vertex words decoded, unpacked cluster index buffer, 1x vertex shading, genuine SV_VertexID!
            entry.isVertexPull  = false;
            entry.isIndexedPull = true;
        }

        if (!BuildReflectedBindingForBucket(entry, scene, timeSeconds, patches, tpls))
        {
            return kNVGInvalidIndex;
        }

        UINT32 bFlags = 0u;
        if (entry.isIndexedPull)       bFlags |= NVG_BUCKET_FLAG_INDEXED_PULL; // bit 2 = Path-A Indexed Pull (0 VB words)
        else if (!entry.isVertexPull)  bFlags |= NVG_BUCKET_FLAG_IA_STREAM;    // bit 0 = Path-B Compact IA-Stream
        if (isTransparent)             bFlags |= NVG_BUCKET_FLAG_TRANSPARENT;  // bit 1 = Transparent
        if (!isShadowPass)
        {
            if (IsCameraBucketEligibleForSWRaster(entry, false, &CoreNVGManager::PlanExecutionStrategy))
            {
                bFlags |= NVG_BUCKET_FLAG_SW_ELIGIBLE;      // bit 3 = Standard-Z SW Micropoly Eligible
            }
            if (IsCameraBucketEligibleForSWRaster(entry, true, &CoreNVGManager::PlanExecutionStrategy))
            {
                bFlags |= NVG_BUCKET_FLAG_SW_ELIGIBLE_REVZ; // bit 6 = Reversed-Z SW Micropoly Eligible
            }
        }
        if (isShadowPass)
        {
            if (IsShadowBucketEligibleForSWRaster(entry, &CoreNVGManager::PlanExecutionStrategy))
            {
                bFlags |= NVG_BUCKET_FLAG_SW_ELIGIBLE;
            }
            if (DoesShaderReferenceShadowProp(vs, "NVS_DEPTH_BIAS_NDC", "NVG_SHADOW_PAGE_DEPTH_BIAS"))
            {
                bFlags |= NVG_BUCKET_FLAG_SH_READS_DEPTH_BIAS;
            }
            if (DoesShaderReferenceShadowProp(vs, "NVS_NORMAL_BIAS_WORLD", "NVG_SHADOW_PAGE_NORMAL_BIAS"))
            {
                bFlags |= NVG_BUCKET_FLAG_SH_READS_NORMAL_BIAS;
            }
        }

        buckets.push_back(entry);
        flags.push_back(bFlags);
        descs.push_back(entry.bindingDesc);
        return entry.bucketIndex;
    }

    void CoreNVGManager::SyncSceneMaterialsAndBuckets(Scene* scene)
    {
        if (!scene)
        {
            m_cachedSyncScene = nullptr;
            m_cachedSyncMaterials.clear();
            m_materialTable.clear();
            m_cameraBuckets.clear();
            m_shadowBuckets.clear();
            return;
        }

        // O(M) incremental cache check: skip O(N) scene scan & redundant GPU buffer uploads when scene/materials unchanged
        const GPUSceneUpdateStats& gs = scene->GetGPUSceneStats();
        const SIZE_T objCount         = scene->GetRenderObjects().size();
        if (m_cachedSyncScene == scene &&
            m_cachedSyncObjCount == objCount &&
            m_cachedSyncSceneEpoch == gs.mutationEpoch &&
            !m_cachedSyncMaterials.empty() &&
            !m_materialTable.empty() &&
            m_materialTableBuffer != nullptr)
        {
            BOOL allMatsUnchanged = true;
            for (const auto& cm : m_cachedSyncMaterials)
            {
                if (!cm.material ||
                    cm.material->GetStateRevision() != cm.matRev ||
                    cm.material->GetGPUSceneMaterialId() != cm.gpuMatId)
                {
                    allMatsUnchanged = false;
                    break;
                }
                const ShaderSet* gSet = cm.material->GetShaderSet(PassType::GEOMETRY);
                const ShaderSet* dSet = cm.material->GetShaderSet(PassType::DEPTH);
                const ShaderSet* sSet = cm.material->GetShaderSet(PassType::SHADOW);
                if ((gSet ? gSet->GetStateRevision() : 0u) != cm.geomShaderRev ||
                    (dSet ? dSet->GetStateRevision() : 0u) != cm.depthShaderRev ||
                    (sSet ? sSet->GetStateRevision() : 0u) != cm.shShaderRev)
                {
                    allMatsUnchanged = false;
                    break;
                }
            }
            if (allMatsUnchanged)
            {
                return;
            }
        }

        m_materialTable.clear();
        m_cameraBuckets.clear();
        m_shadowBuckets.clear();
        m_cameraBucketFlags.clear();
        m_shadowBucketFlags.clear();
        m_cameraBindingDescs.clear();
        m_cameraFieldPatches.clear();
        m_cameraTemplateWords.clear();
        m_shadowBindingDescs.clear();
        m_shadowFieldPatches.clear();
        m_shadowTemplateWords.clear();
        m_cachedSyncMaterials.clear();
        m_sceneHasAnimatedDeformation = false;

        // Reserve slot 0 sentinel
        m_materialTable.push_back(NVGMaterialGPUData{});
        const FLOAT32 timeSec = static_cast<FLOAT32>(m_currentFrameIndex) * 0.016666667f;

        std::unordered_map<UINT32, Material*> seenMaterials;
        for (RenderObject* obj : scene->GetRenderObjects())
        {
            if (!obj || !obj->HasFlag(RenderObjectFlags::VISIBLE)) continue;
            const Mesh* m = obj->GetMesh();
            if (!m || !m->IsVirtualized()) continue;
            const UINT32 matSlotCnt = (std::max)(1u, obj->GetMaterialSlotCount());
            for (UINT32 ms = 0u; ms < matSlotCnt; ++ms)
            {
                Material* mat = obj->GetMaterial(ms);
                if (!mat) continue;
                const UINT32 matId = mat->GetGPUSceneMaterialId();
                if (matId == 0u) continue;
                seenMaterials[matId] = mat;
            }
        }

        for (const auto& kv : seenMaterials)
        {
            const UINT32 matId = kv.first;
            Material* mat      = kv.second;
            if (matId >= m_materialTable.size())
            {
                m_materialTable.resize(static_cast<SIZE_T>(matId) + 1u, NVGMaterialGPUData{});
            }

            NVGMaterialGPUData gm{};
            gm.materialId           = matId;
            const NVGMaterialPolicy& pol = mat->GetNVGPolicy();
            gm.maxWorldDisplacement = (std::max)(0.0f, pol.maxWorldDisplacement);
            if (gm.maxWorldDisplacement > 0.0f)
            {
                m_sceneHasAnimatedDeformation = true;
            }
            if (!m_sceneHasAnimatedDeformation)
            {
                const ShaderSet* gSet = mat->GetShaderSet(PassType::GEOMETRY);
                if (gSet != nullptr)
                {
                    const Shader* preStages[7] = {
                        gSet->GetVertexShader(ShaderVariant::STANDARD),
                        gSet->GetVertexShader(ShaderVariant::INSTANCED),
                        gSet->GetVertexShader(ShaderVariant::SKINNED),
                        gSet->GetVertexShader(ShaderVariant::INSTANCED_SKINNED),
                        gSet->GetHullShader(),
                        gSet->GetDomainShader(),
                        gSet->GetGeometryShader()
                    };
                    for (const Shader* st : preStages)
                    {
                        if (!st || !st->IsValid()) continue;
                        for (const auto& nb : st->GetReflection().buffers)
                        {
                            for (const auto& nf : nb.fields)
                            {
                                if (!nf.isGet) continue;
                                const std::string& pName = nf.propertyName.empty() ? nf.name : nf.propertyName;
                                if (pName == "TIME" || pName == "ENGINE_TIME")
                                {
                                    m_sceneHasAnimatedDeformation = true;
                                    break;
                                }
                            }
                            if (m_sceneHasAnimatedDeformation) break;
                        }
                        if (m_sceneHasAnimatedDeformation) break;
                    }
                }
            }
            gm.lodErrorScale        = (std::max)(0.01f, pol.lodErrorScale);
            gm.shadowLodErrorScale  = (std::max)(0.01f, pol.shadowLodErrorScale);

            const NVGCompatibilityStatus geomSt = CheckMaterialCompatibility(mat, PassType::GEOMETRY, false);
            gm.compatStatus = static_cast<UINT32>(geomSt);

            if (IsNVGCompatible(geomSt))
            {
                const ShaderSet* geomSet  = mat->GetShaderSet(PassType::GEOMETRY);
                const ShaderVariant mainVar = ResolveEffectiveNVGVariant(geomSet, false, pol, &CoreNVGManager::PlanExecutionStrategy);
                const BOOL isPull  = (PlanExecutionStrategy(geomSet, mainVar, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                const BOOL isTrans = mat->IsTransparent();

                gm.nvgFlags |= NVG_MAT_FLAG_VALID;
                gm.nvgFlags |= isPull ? NVG_MAT_FLAG_VERTEX_PULL : NVG_MAT_FLAG_IA_STREAM;
                if (mat->IsAlphaTested())                    gm.nvgFlags |= NVG_MAT_FLAG_ALPHA_TESTED;
                if (mat->GetCullMode() == CullMode::NONE)    gm.nvgFlags |= NVG_MAT_FLAG_TWO_SIDED;
                if (mat->GetCullMode() == CullMode::FRONT)   gm.nvgFlags |= NVG_MAT_FLAG_FRONT_CULL;
                if (isTrans)                                 gm.nvgFlags |= NVG_MAT_FLAG_TRANSPARENT;
                if (pol.lockToLOD0)                          gm.nvgFlags |= NVG_MAT_FLAG_LOCK_LOD0;

                const UINT32 mainBucket = RegisterDrawBucket(
                    mat, PassType::GEOMETRY, mainVar, isPull, isTrans, false, false, scene, timeSec);
                if (isTrans)
                {
                    gm.transparentBucketIndex = mainBucket;
                }
                else
                {
                    gm.opaqueBucketIndex = mainBucket;
                    if (mat->GetDepthState().test && mat->GetDepthState().write && mainBucket != kNVGInvalidIndex)
                    {
                        if (mat->HasPass(PassType::DEPTH))
                        {
                            const NVGCompatibilityStatus dpSt = CheckMaterialCompatibility(mat, PassType::DEPTH, false);
                            if (IsNVGCompatible(dpSt))
                            {
                                const ShaderSet* dpSet = mat->GetShaderSet(PassType::DEPTH);
                                const ShaderVariant dpVar = ResolveEffectiveNVGVariant(dpSet, false, pol, &CoreNVGManager::PlanExecutionStrategy);
                                const BOOL dpPull = (PlanExecutionStrategy(dpSet, dpVar, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                                const UINT32 dpBucket = RegisterDrawBucket(
                                    mat, PassType::DEPTH, dpVar, dpPull, false, false, true, scene, timeSec);
                                if (dpBucket != kNVGInvalidIndex && dpBucket != mainBucket)
                                {
                                    m_cameraBuckets[dpBucket].isDepthCompanion            = true;
                                    m_cameraBucketFlags[dpBucket]                        |= ((mainBucket + 1u) << 8u);
                                    m_cameraBuckets[mainBucket].depthCompanionBucketIndex = dpBucket;
                                    m_cameraBindingDescs[mainBucket].drawGroupIndex      |= ((dpBucket + 1u) << 16u);
                                    const UINT32 sharedWords = (std::max)(m_cameraBuckets[mainBucket].decodeVertexWords, m_cameraBuckets[dpBucket].decodeVertexWords);
                                    const UINT32 sharedMask  = m_cameraBuckets[mainBucket].decodeAttrMask | m_cameraBuckets[dpBucket].decodeAttrMask;
                                    m_cameraBuckets[mainBucket].decodeVertexWords         = sharedWords;
                                    m_cameraBuckets[mainBucket].decodeAttrMask            = sharedMask;
                                    m_cameraBindingDescs[mainBucket].decodeVertexWords    = sharedWords;
                                    m_cameraBindingDescs[mainBucket].decodeAttrMask       = sharedMask;
                                    gm.nvgFlags        |= NVG_MAT_FLAG_HAS_DEPTH_PASS;
                                    gm.depthBucketIndex = dpBucket;
                                }
                                else if (dpBucket == mainBucket)
                                {
                                    gm.nvgFlags        |= NVG_MAT_FLAG_HAS_DEPTH_PASS;
                                    gm.depthBucketIndex = mainBucket;
                                }
                                else
                                {
                                    m_cameraBuckets[mainBucket].hasRejectedCustomDepth = true;
                                }
                            }
                            else
                            {
                                m_cameraBuckets[mainBucket].hasRejectedCustomDepth = true;
                            }
                        }
                        else
                        {
                            gm.nvgFlags        |= NVG_MAT_FLAG_HAS_DEPTH_PASS;
                            gm.depthBucketIndex = mainBucket;
                        }
                    }
                }

                // Skinned variant bucket (SKINNED, INSTANCED_SKINNED, or unified/pull STANDARD)
                const NVGCompatibilityStatus skinSt = CheckMaterialCompatibility(mat, PassType::GEOMETRY, true);
                if (IsNVGCompatible(skinSt))
                {
                    const ShaderVariant sv = ResolveEffectiveNVGVariant(geomSet, true, pol, &CoreNVGManager::PlanExecutionStrategy);
                    const BOOL skinPull = (PlanExecutionStrategy(geomSet, sv, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                    const UINT32 skinBucket = RegisterDrawBucket(
                        mat, PassType::GEOMETRY, sv, skinPull, isTrans, false, false, scene, timeSec);
                    if (skinBucket != kNVGInvalidIndex)
                    {
                        gm.nvgFlags |= NVG_MAT_FLAG_HAS_SKINNING;
                        if (isTrans)
                        {
                            gm.transInstBucketIndex = skinBucket; // Skinned transparent bucket (NVG-037)
                        }
                        else
                        {
                            gm.skinnedOpaqueBucketIndex = skinBucket;
                            if (mat->GetDepthState().test && mat->GetDepthState().write)
                            {
                                if (mat->HasPass(PassType::DEPTH))
                                {
                                    const NVGCompatibilityStatus dpSkSt = CheckMaterialCompatibility(mat, PassType::DEPTH, true);
                                    if (IsNVGCompatible(dpSkSt))
                                    {
                                        const ShaderSet* dpSet = mat->GetShaderSet(PassType::DEPTH);
                                        const ShaderVariant dpVar = ResolveEffectiveNVGVariant(dpSet, true, pol, &CoreNVGManager::PlanExecutionStrategy);
                                        const BOOL dpPull = (PlanExecutionStrategy(dpSet, dpVar, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                                        const UINT32 dpSkBucket = RegisterDrawBucket(
                                            mat, PassType::DEPTH, dpVar, dpPull, false, false, true, scene, timeSec);
                                        if (dpSkBucket != kNVGInvalidIndex && dpSkBucket != skinBucket)
                                        {
                                            m_cameraBuckets[dpSkBucket].isDepthCompanion            = true;
                                            m_cameraBucketFlags[dpSkBucket]                        |= ((skinBucket + 1u) << 8u);
                                            m_cameraBuckets[skinBucket].depthCompanionBucketIndex   = dpSkBucket;
                                            m_cameraBindingDescs[skinBucket].drawGroupIndex        |= ((dpSkBucket + 1u) << 16u);
                                            const UINT32 sharedWords = (std::max)(m_cameraBuckets[skinBucket].decodeVertexWords, m_cameraBuckets[dpSkBucket].decodeVertexWords);
                                            const UINT32 sharedMask  = m_cameraBuckets[skinBucket].decodeAttrMask | m_cameraBuckets[dpSkBucket].decodeAttrMask;
                                            m_cameraBuckets[skinBucket].decodeVertexWords           = sharedWords;
                                            m_cameraBuckets[skinBucket].decodeAttrMask              = sharedMask;
                                            m_cameraBindingDescs[skinBucket].decodeVertexWords      = sharedWords;
                                            m_cameraBindingDescs[skinBucket].decodeAttrMask         = sharedMask;
                                            gm.depthInstBucketIndex = dpSkBucket;
                                        }
                                        else if (dpSkBucket == skinBucket)
                                        {
                                            gm.depthInstBucketIndex = skinBucket;
                                        }
                                        else
                                        {
                                            m_cameraBuckets[skinBucket].hasRejectedCustomDepth = true;
                                        }
                                    }
                                    else
                                    {
                                        m_cameraBuckets[skinBucket].hasRejectedCustomDepth = true;
                                    }
                                }
                                else
                                {
                                    gm.depthInstBucketIndex = skinBucket;
                                }
                            }
                        }
                    }
                }
            }

            // Shadow pass compatibility (independent of geometry pass!)
            const NVGCompatibilityStatus shSt = CheckMaterialCompatibility(mat, PassType::SHADOW, false);
            if (IsNVGCompatible(shSt))
            {
                const ShaderSet* shSet = mat->GetShaderSet(PassType::SHADOW);
                const ShaderVariant shVar = ResolveEffectiveNVGVariant(shSet, false, pol, &CoreNVGManager::PlanExecutionStrategy);
                const BOOL shPull = (PlanExecutionStrategy(shSet, shVar, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                const UINT32 shBucket = RegisterDrawBucket(
                    mat, PassType::SHADOW, shVar, shPull, false, true, false, scene, timeSec);
                if (shBucket != kNVGInvalidIndex)
                {
                    gm.shadowBucketIndex = shBucket;
                    gm.nvgFlags |= NVG_MAT_FLAG_HAS_SHADOW_PASS;
                    if (shPull) gm.nvgFlags |= NVG_MAT_FLAG_SHADOW_PULL;
                }

                const NVGCompatibilityStatus shSkinSt = CheckMaterialCompatibility(mat, PassType::SHADOW, true);
                if (IsNVGCompatible(shSkinSt))
                {
                    const ShaderVariant sv = ResolveEffectiveNVGVariant(shSet, true, pol, &CoreNVGManager::PlanExecutionStrategy);
                    const BOOL shSkinPull = (PlanExecutionStrategy(shSet, sv, pol) == NVGInternalExecutionStrategy::VERTEX_PULL);
                    gm.skinnedShadowBucketIndex = RegisterDrawBucket(
                        mat, PassType::SHADOW, sv, shSkinPull, false, true, false, scene, timeSec);
                }
            }

            m_materialTable[matId] = gm;
        }

        // Upload synced material & reflection tables to GPU
        if (!m_materialTable.empty() &&
            EnsureBuffer(m_materialTableBuffer, sizeof(NVGMaterialGPUData), static_cast<UINT32>(m_materialTable.size()), GPUBufferUsage::STRUCTURED_SRV))
        {
            m_bufferManager->UpdateGPUBuffer(
                m_materialTableBuffer, m_materialTable.data(),
                static_cast<UINT32>(m_materialTable.size() * sizeof(NVGMaterialGPUData)), 0u);
        }

        auto uploadU32Vec = [&](GPUBuffer*& buf, const std::vector<UINT32>& vec) {
            if (vec.empty()) return;
            if (EnsureBuffer(buf, sizeof(UINT32), static_cast<UINT32>(vec.size()), GPUBufferUsage::STRUCTURED_SRV))
            {
                m_bufferManager->UpdateGPUBuffer(buf, vec.data(), static_cast<UINT32>(vec.size() * sizeof(UINT32)), 0u);
            }
        };
        auto uploadBindings = [&](GPUBuffer*& buf, const std::vector<NVGDrawGroupShaderBindingDesc>& vec) {
            if (vec.empty()) return;
            if (EnsureBuffer(buf, sizeof(NVGDrawGroupShaderBindingDesc), static_cast<UINT32>(vec.size()), GPUBufferUsage::STRUCTURED_SRV))
            {
                m_bufferManager->UpdateGPUBuffer(buf, vec.data(), static_cast<UINT32>(vec.size() * sizeof(NVGDrawGroupShaderBindingDesc)), 0u);
            }
        };
        auto uploadPatches = [&](GPUBuffer*& buf, const std::vector<NVSDrawGroupFieldPatch>& vec) {
            if (vec.empty()) return;
            if (EnsureBuffer(buf, sizeof(NVSDrawGroupFieldPatch), static_cast<UINT32>(vec.size()), GPUBufferUsage::STRUCTURED_SRV))
            {
                m_bufferManager->UpdateGPUBuffer(buf, vec.data(), static_cast<UINT32>(vec.size() * sizeof(NVSDrawGroupFieldPatch)), 0u);
            }
        };

        uploadU32Vec(m_cameraBucketFlagsBuf, m_cameraBucketFlags);
        uploadU32Vec(m_shadowBucketFlagsBuf, m_shadowBucketFlags);
        uploadBindings(m_cameraBindingsBuf, m_cameraBindingDescs);
        uploadPatches(m_cameraPatchesBuf, m_cameraFieldPatches);
        uploadU32Vec(m_cameraTemplatesBuf, m_cameraTemplateWords);
        uploadBindings(m_shadowBindingsBuf, m_shadowBindingDescs);
        uploadPatches(m_shadowPatchesBuf, m_shadowFieldPatches);
        uploadU32Vec(m_shadowTemplatesBuf, m_shadowTemplateWords);

        // Save O(M) incremental sync signature
        m_cachedSyncScene      = scene;
        m_cachedSyncObjCount   = objCount;
        m_cachedSyncSceneEpoch = gs.mutationEpoch;
        m_cachedSyncMaterials.reserve(seenMaterials.size());
        for (const auto& kv : seenMaterials)
        {
            const Material* m = kv.second;
            CachedSyncMaterialState st{};
            st.material      = m;
            st.matRev        = m->GetStateRevision();
            st.gpuMatId      = m->GetGPUSceneMaterialId();
            const ShaderSet* gSet = m->GetShaderSet(PassType::GEOMETRY);
            const ShaderSet* dSet = m->GetShaderSet(PassType::DEPTH);
            const ShaderSet* sSet = m->GetShaderSet(PassType::SHADOW);
            st.geomShaderRev  = gSet ? gSet->GetStateRevision() : 0u;
            st.depthShaderRev = dSet ? dSet->GetStateRevision() : 0u;
            st.shShaderRev    = sSet ? sSet->GetStateRevision() : 0u;
            m_cachedSyncMaterials.push_back(st);
        }
    }

    BOOL CoreNVGManager::EnsureCameraContextResources(CameraNVGContext& ctx, UINT32 vpW, UINT32 vpH, UINT32 gpuSceneSlotCount)
    {
        const UINT32 desiredHzbW = PrevPow2((std::max)(vpW, 1u));
        const UINT32 desiredHzbH = PrevPow2((std::max)(vpH, 1u));

        if (!ctx.hzbTexture || ctx.hzbWidth != desiredHzbW || ctx.hzbHeight != desiredHzbH)
        {
            if (ctx.hzbTexture)
            {
                m_textureManager->DestroyTexture(ctx.hzbTexture);
                ctx.hzbTexture = nullptr;
            }

            TextureDesc td{};
            td.width       = desiredHzbW;
            td.height      = desiredHzbH;
            td.depth       = 1u;
            td.mipLevels   = ComputeMipCount(desiredHzbW, desiredHzbH);
            td.arrayLayers = 1u;
            td.format      = TextureFormat::R32_FLOAT;
            td.type        = TextureType::TEXTURE_2D;
            td.usage       = TextureUsage::STORAGE;
            td.dataSource  = TextureDataSource::EMPTY;

            ctx.hzbTexture = m_textureManager->CreateTexture(td);
            if (!ctx.hzbTexture || ctx.hzbTexture->GetNativeHandle() == 0)
            {
                return false;
            }
            ctx.hzbWidth        = desiredHzbW;
            ctx.hzbHeight       = desiredHzbH;
            ctx.hzbMipCount     = td.mipLevels;
            ctx.hasValidPrevHZB = false;
        }

        if (ctx.lastViewportW != vpW || ctx.lastViewportH != vpH)
        {
            ctx.lastViewportW   = vpW;
            ctx.lastViewportH   = vpH;
            ctx.hasValidPrevHZB = false;
        }

        const UINT32 safeSwW = (std::max)(vpW, 1u);
        const UINT32 safeSwH = (std::max)(vpH, 1u);
        if (!ctx.swVisTexture || !ctx.swDepthTexture || ctx.swTexWidth != safeSwW || ctx.swTexHeight != safeSwH)
        {
            if (ctx.swVisTexture)
            {
                m_textureManager->DestroyTexture(ctx.swVisTexture);
                ctx.swVisTexture = nullptr;
            }
            if (ctx.swDepthTexture)
            {
                m_textureManager->DestroyTexture(ctx.swDepthTexture);
                ctx.swDepthTexture = nullptr;
            }

            TextureDesc visTd{};
            visTd.width       = safeSwW;
            visTd.height      = safeSwH;
            visTd.depth       = 1u;
            visTd.mipLevels   = 1u;
            visTd.arrayLayers = 1u;
            visTd.format      = TextureFormat::RG32_UINT;
            visTd.type        = TextureType::TEXTURE_2D;
            visTd.usage       = TextureUsage::STORAGE;
            visTd.dataSource  = TextureDataSource::EMPTY;
            ctx.swVisTexture  = m_textureManager->CreateTexture(visTd);

            TextureDesc swDepTd = visTd;
            swDepTd.format      = TextureFormat::R32_FLOAT;
            ctx.swDepthTexture  = m_textureManager->CreateTexture(swDepTd);

            if (!ctx.swVisTexture || ctx.swVisTexture->GetNativeHandle() == 0 ||
                !ctx.swDepthTexture || ctx.swDepthTexture->GetNativeHandle() == 0)
            {
                return false;
            }
            ctx.swTexWidth  = safeSwW;
            ctx.swTexHeight = safeSwH;
        }

        const UINT32 totalClusters  = (std::max)(m_residencyManager.GetTotalClusterCount(), 1024u);
        const UINT32 camIndBuckets  = (std::min)((std::max)(static_cast<UINT32>(m_cameraBuckets.size()), 16u), kNVGMaxDrawBuckets);
        const UINT32 shIndBuckets   = (std::min)((std::max)(static_cast<UINT32>(m_shadowBuckets.size()), 4u), kNVGMaxDrawBuckets);
        const BOOL   massiveScene   = (m_stats.effectiveSceneTriangles >= 250000000ull || gpuSceneSlotCount >= 512u);
        const UINT32 floorNodeCap   = massiveScene ? kNVGMaxQueueCap : kNVGNodeQueueCap;
        const UINT32 floorCandCap   = massiveScene ? kNVGMaxQueueCap : kNVGCandidateCap;
        const UINT32 floorVisCap    = massiveScene ? kNVGMaxVisibleCap : kNVGVisibleCap;
        const UINT32 floorShVisCap  = massiveScene ? kNVGMaxShadowVisibleCap : kNVGShadowVisibleCap;
        const UINT32 floorOccClCap  = massiveScene ? 4194304u : kNVGOccludedClusterCap;

        const UINT32 reqNodeCap     = (std::min)((std::max)({ ctx.dynNodeQueueCap, floorNodeCap, gpuSceneSlotCount * 2u }), kNVGMaxQueueCap);
        const UINT32 reqCandCap     = (std::min)((std::max)(ctx.dynCandidateCap, floorCandCap), kNVGMaxQueueCap);
        const UINT32 reqVisCap      = (std::min)((std::max)(ctx.dynVisibleCap, floorVisCap), kNVGMaxVisibleCap);
        const UINT32 reqShVisCap    = (std::min)((std::max)(ctx.dynShadowVisibleCap, floorShVisCap), kNVGMaxShadowVisibleCap);
        const UINT32 reqOccClCap    = (std::min)((std::max)(ctx.dynOccludedClusterCap, floorOccClCap), kNVGMaxQueueCap);
        const UINT32 reqOccObjCap   = (std::min)((std::max)(kNVGOccludedObjCap, gpuSceneSlotCount), 4194240u);
        const GPUBufferUsage uavSrv = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        const GPUBufferUsage uavInd = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::INDIRECT_ARGUMENT;
        const GPUBufferUsage uavRead= GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::CPU_READBACK;

        ctx.dynNodeQueueCap       = reqNodeCap;
        ctx.dynCandidateCap       = reqCandCap;
        ctx.dynVisibleCap         = reqVisCap;
        ctx.dynShadowVisibleCap   = reqShVisCap;
        ctx.dynOccludedClusterCap = reqOccClCap;

        BOOL   needCameraIndirect  = false;
        UINT32 maxCamDecodeWords   = 0u;
        for (const auto& b : m_cameraBuckets)
        {
            if (!b.isVertexPull)
            {
                needCameraIndirect = true;
                maxCamDecodeWords  = (std::max)(maxCamDecodeWords, b.decodeVertexWords);
            }
        }
        BOOL   needShadowIndirect  = false;
        UINT32 maxShDecodeWords    = 0u;
        for (const auto& b : m_shadowBuckets)
        {
            if (!b.isVertexPull)
            {
                needShadowIndirect = true;
                maxShDecodeWords   = (std::max)(maxShDecodeWords, b.decodeVertexWords);
            }
        }

        // Scale visible-only decode pools by active dynDecodeClusterCap & compact IA stride (never totalClusters -> 0 2GiB overflow!)
        constexpr UINT64 kMaxSafePoolWords = 0x70000000ull / sizeof(UINT32);
        const UINT32 dynDecodeMax        = (std::clamp)(ctx.dynDecodeClusterCap, 32768u, kNVGDecodedHashTableCapacity);
        const UINT32 camDecodeClusterCap = (std::min)(dynDecodeMax, (std::max)(totalClusters, 16384u));
        const UINT32 camDecodedVertWords = (needCameraIndirect && maxCamDecodeWords > 0u)
            ? static_cast<UINT32>((std::min<UINT64>)(static_cast<UINT64>(camDecodeClusterCap) * kNVGMaxClusterVertices * maxCamDecodeWords, kMaxSafePoolWords))
            : 256u;
        const UINT32 camDecodedIdxs      = needCameraIndirect ? (camDecodeClusterCap * kNVGMaxClusterTriangles * 3u) : kNVGMinimalPullDecodedIdxs;
        const UINT32 camUniqueJobs       = needCameraIndirect ? (std::max)(camDecodeClusterCap, kNVGMaxUniqueJobs) : 1024u;
        const UINT32 camCBVWords         = needCameraIndirect ? (std::max)(ctx.dynCBVWords, kNVGPerDrawCBVWords) : 4096u;
        const UINT32 camInstWords        = needCameraIndirect ? kNVGPerBatchInstWords : 4096u;
        const UINT32 camIndCmdWords      = needCameraIndirect ? (camIndBuckets * ctx.dynCameraCmdsPerBucket * 36u) : 1024u;

        const UINT32 shDecodeClusterCap  = (std::min)(dynDecodeMax, (std::max)(totalClusters, 16384u));
        const UINT32 shDecodedVertWords  = (needShadowIndirect && maxShDecodeWords > 0u)
            ? static_cast<UINT32>((std::min<UINT64>)(static_cast<UINT64>(shDecodeClusterCap) * kNVGMaxClusterVertices * maxShDecodeWords, kMaxSafePoolWords))
            : 256u;
        const UINT32 shDecodedIdxs       = needShadowIndirect ? (shDecodeClusterCap * kNVGMaxClusterTriangles * 3u) : kNVGMinimalPullDecodedIdxs;
        const UINT32 shUniqueJobs        = needShadowIndirect ? (std::max)(shDecodeClusterCap, kNVGMaxUniqueJobs) : 1024u;
        const UINT32 shCBVWords          = needShadowIndirect ? (std::max)(ctx.dynCBVWords, kNVGShadowPerDrawCBVWords) : 4096u;
        const UINT32 shInstWords         = needShadowIndirect ? kNVGShadowPerBatchInstWords : 4096u;
        const UINT32 shIndCmdWords       = needShadowIndirect ? (shIndBuckets * ctx.dynShadowCmdsPerBucket * 36u) : 1024u;

        const UINT32 swTotalPixels      = (std::max)(safeSwW * safeSwH, 1u);
        const auto   resolvedSwStrategy = ResolveActiveCameraRasterStrategy();
        const BOOL   useSplitAtomic32   = (resolvedSwStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS);
        const UINT32 vis64ElemCount     = (!useSplitAtomic32) ? swTotalPixels : 4u;
        const UINT32 atomic32ElemCount  = useSplitAtomic32    ? swTotalPixels : 4u;
        const UINT32 swMaskWords        = (reqVisCap + 31u) / 32u;

        if (!EnsureBuffer(ctx.nodeQueueA, sizeof(NVGTraversalItemGPUData), reqNodeCap, uavSrv) ||
            !EnsureBuffer(ctx.nodeQueueB, sizeof(NVGTraversalItemGPUData), reqNodeCap, uavSrv) ||
            !EnsureBuffer(ctx.candidateClusters, sizeof(NVGTraversalItemGPUData), reqCandCap, uavSrv) ||
            !EnsureBuffer(ctx.occludedObjects, sizeof(NVGTraversalItemGPUData), reqOccObjCap, uavSrv) ||
            !EnsureBuffer(ctx.occludedNodes, sizeof(NVGTraversalItemGPUData), reqNodeCap, uavSrv) ||
            !EnsureBuffer(ctx.occludedClusters, sizeof(NVGTraversalItemGPUData), reqOccClCap, uavSrv) ||
            !EnsureBuffer(ctx.rawVisibleClusters, sizeof(NVGVisibleClusterGPUData), reqVisCap, uavSrv) ||
            !EnsureBuffer(ctx.compactedVisibleClusters, sizeof(NVGVisibleClusterGPUData), reqVisCap, uavSrv) ||
            !EnsureBuffer(ctx.bucketRanges, sizeof(NVGBucketRangeGPUData), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.bucketCounts, sizeof(UINT32), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.bucketWriteCursors, sizeof(UINT32), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.pathAIndirectArgs, sizeof(GPUIndirectDrawArgs), kNVGMaxDrawBuckets, uavInd) ||
            !EnsureBuffer(ctx.queueCounters, sizeof(UINT32), 32u, uavSrv) ||
            !EnsureBuffer(ctx.dispatchArgs, sizeof(UINT32), kNVGDispatchArgWords, uavInd) ||
            !EnsureBuffer(ctx.decodedClusterTable, sizeof(NVGDecodedClusterRecordGPUData), kNVGDecodedHashTableCapacity, uavSrv) ||
            !EnsureBuffer(ctx.uniqueDecodeJobs, sizeof(UINT32) * 8u, camUniqueJobs, uavSrv) ||
            !EnsureBuffer(ctx.decodedVertexPool, sizeof(UINT32), camDecodedVertWords, uavSrv) ||
            !EnsureBuffer(ctx.decodedIndexPool, sizeof(UINT32), camDecodedIdxs, uavSrv) ||
            !EnsureBuffer(ctx.indirectDrawCommands, sizeof(UINT32), camIndCmdWords, uavInd) ||
            !EnsureBuffer(ctx.perDrawCBVWords, sizeof(UINT32), camCBVWords, uavSrv) ||
            !EnsureBuffer(ctx.perBatchInstanceWords, sizeof(UINT32), camInstWords, uavSrv) ||
            !EnsureBuffer(ctx.bucketCommandCounts, sizeof(UINT32), kNVGMaxDrawBuckets, uavInd) ||
            !EnsureBuffer(ctx.telemetryBuffer, sizeof(UINT32), 48u, uavRead) ||
            !EnsureBuffer(ctx.visBuffer64, sizeof(UINT64), vis64ElemCount, uavRead) ||
            !EnsureBuffer(ctx.swAtomicDepthKey32, sizeof(UINT32), atomic32ElemCount, uavRead) ||
            !EnsureBuffer(ctx.swAtomicPayload32, sizeof(UINT32), atomic32ElemCount, uavRead) ||
            !EnsureBuffer(ctx.swClusterList, sizeof(UINT32), reqVisCap, uavSrv) ||
            !EnsureBuffer(ctx.swScheduledMask, sizeof(UINT32), swMaskWords, uavRead) ||
            !EnsureBuffer(ctx.swWinningClusterMask, sizeof(UINT32), swMaskWords, uavRead) ||
            // Dedicated Phase-2 disocclusion draw output buffers (keeps Phase-1 Opaque & Transparent draws intact!)
            !EnsureBuffer(ctx.p2CompactedVisible, sizeof(NVGVisibleClusterGPUData), reqVisCap, uavSrv) ||
            !EnsureBuffer(ctx.p2BucketRanges, sizeof(NVGBucketRangeGPUData), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.p2PathAIndirectArgs, sizeof(GPUIndirectDrawArgs), kNVGMaxDrawBuckets, uavInd) ||
            (needCameraIndirect && (
                !EnsureBuffer(ctx.p2IndirectCommands, sizeof(UINT32), camIndCmdWords, uavInd) ||
                !EnsureBuffer(ctx.p2BucketCmdCounts, sizeof(UINT32), kNVGMaxDrawBuckets, uavInd) ||
                !EnsureBuffer(ctx.p2PerDrawCBVWords, sizeof(UINT32), camCBVWords, uavSrv) ||
                !EnsureBuffer(ctx.p2PerBatchInstWords, sizeof(UINT32), camInstWords, uavSrv) ||
                !EnsureBuffer(ctx.p2DecodedVertexPool, sizeof(UINT32), camDecodedVertWords, uavSrv) ||
                !EnsureBuffer(ctx.p2DecodedIndexPool, sizeof(UINT32), camDecodedIdxs, uavSrv))) ||
            // Shadow working buffers
            !EnsureBuffer(ctx.shadowPageViews, sizeof(NVGShadowPageViewGPUData), kNVGMaxShadowPages, uavSrv) ||
            !EnsureBuffer(ctx.shadowViewMasks, sizeof(UINT32), 4096u, GPUBufferUsage::STRUCTURED_SRV) ||
            !EnsureBuffer(ctx.shadowPageMasks, sizeof(UINT32), kNVGMaxShadowPages, uavSrv) ||
            !EnsureBuffer(ctx.shadowRawVisible, sizeof(NVGVisibleClusterGPUData), reqShVisCap, uavSrv) ||
            !EnsureBuffer(ctx.shadowCompactedVisible, sizeof(NVGVisibleClusterGPUData), reqShVisCap, uavSrv) ||
            !EnsureBuffer(ctx.shadowBucketRanges, sizeof(NVGBucketRangeGPUData), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.shadowBucketCounts, sizeof(UINT32), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.shadowBucketWriteCursors, sizeof(UINT32), kNVGMaxDrawBuckets, uavSrv) ||
            !EnsureBuffer(ctx.shadowPathAIndirectArgs, sizeof(GPUIndirectDrawArgs), kNVGMaxDrawBuckets, uavInd) ||
            !EnsureBuffer(ctx.shadowIndirectCommands, sizeof(UINT32), shIndCmdWords, uavInd) ||
            !EnsureBuffer(ctx.shadowBucketCmdCounts, sizeof(UINT32), kNVGMaxDrawBuckets, uavInd) ||
            !EnsureBuffer(ctx.shadowPerDrawCBVWords, sizeof(UINT32), shCBVWords, uavSrv) ||
            !EnsureBuffer(ctx.shadowPerBatchInstWords, sizeof(UINT32), shInstWords, uavSrv) ||
            !EnsureBuffer(ctx.shadowDecodedVertexPool, sizeof(UINT32), shDecodedVertWords, uavSrv) ||
            !EnsureBuffer(ctx.shadowDecodedIndexPool, sizeof(UINT32), shDecodedIdxs, uavSrv) ||
            !EnsureBuffer(ctx.shadowDecodedClusterTbl, sizeof(NVGDecodedClusterRecordGPUData), kNVGDecodedHashTableCapacity, uavSrv) ||
            !EnsureBuffer(ctx.shadowUniqueDecodeJobs, sizeof(UINT32) * 8u, shUniqueJobs, uavSrv))
        {
            return false;
        }

        return true;
    }

    void CoreNVGManager::DestroyCameraContextResources(CameraNVGContext& ctx)
    {
        if (m_textureManager)
        {
            if (ctx.hzbTexture)
            {
                m_textureManager->DestroyTexture(ctx.hzbTexture);
                ctx.hzbTexture = nullptr;
            }
            if (ctx.swVisTexture)
            {
                m_textureManager->DestroyTexture(ctx.swVisTexture);
                ctx.swVisTexture = nullptr;
            }
            if (ctx.swDepthTexture)
            {
                m_textureManager->DestroyTexture(ctx.swDepthTexture);
                ctx.swDepthTexture = nullptr;
            }
            ctx.swTexWidth  = 0u;
            ctx.swTexHeight = 0u;
        }
        if (m_bufferManager)
        {
            auto d = [&](GPUBuffer*& b) { if (b) { m_bufferManager->DestroyGPUBuffer(b); b = nullptr; } };
            d(ctx.visBuffer64);
            d(ctx.swAtomicDepthKey32);
            d(ctx.swAtomicPayload32);
            d(ctx.swClusterList);
            d(ctx.swScheduledMask);
            d(ctx.swWinningClusterMask);
            d(ctx.nodeQueueA);
            d(ctx.nodeQueueB);
            d(ctx.candidateClusters);
            d(ctx.occludedObjects);
            d(ctx.occludedNodes);
            d(ctx.occludedClusters);
            d(ctx.rawVisibleClusters);
            d(ctx.compactedVisibleClusters);
            d(ctx.bucketRanges);
            d(ctx.bucketCounts);
            d(ctx.bucketWriteCursors);
            d(ctx.pathAIndirectArgs);
            d(ctx.queueCounters);
            d(ctx.dispatchArgs);
            d(ctx.decodedClusterTable);
            d(ctx.uniqueDecodeJobs);
            d(ctx.decodedVertexPool);
            d(ctx.decodedIndexPool);
            d(ctx.indirectDrawCommands);
            d(ctx.perDrawCBVWords);
            d(ctx.perBatchInstanceWords);
            d(ctx.bucketCommandCounts);
            d(ctx.telemetryBuffer);
            d(ctx.p2CompactedVisible);
            d(ctx.p2BucketRanges);
            d(ctx.p2PathAIndirectArgs);
            d(ctx.p2IndirectCommands);
            d(ctx.p2BucketCmdCounts);
            d(ctx.p2PerDrawCBVWords);
            d(ctx.p2PerBatchInstWords);
            d(ctx.p2DecodedVertexPool);
            d(ctx.p2DecodedIndexPool);
            d(ctx.shadowPageViews);
            d(ctx.shadowViewMasks);
            d(ctx.shadowPageMasks);
            d(ctx.shadowRawVisible);
            d(ctx.shadowCompactedVisible);
            d(ctx.shadowBucketRanges);
            d(ctx.shadowBucketCounts);
            d(ctx.shadowBucketWriteCursors);
            d(ctx.shadowPathAIndirectArgs);
            d(ctx.shadowIndirectCommands);
            d(ctx.shadowBucketCmdCounts);
            d(ctx.shadowPerDrawCBVWords);
            d(ctx.shadowPerBatchInstWords);
            d(ctx.shadowDecodedVertexPool);
            d(ctx.shadowDecodedIndexPool);
            d(ctx.shadowDecodedClusterTbl);
            d(ctx.shadowUniqueDecodeJobs);
            d(ctx.shadowSWDepthKeyPool);
            d(ctx.shadowSWClusterList);
            d(ctx.shadowSWScheduledMask);
            d(ctx.shadowSWFallbackMask);
            d(ctx.shadowSWPageNeedsTile);
            d(ctx.shadowSWPageTileMap);
            d(ctx.shadowSWDirtyTileSlices);
            d(ctx.shadowSWMergeIndirectArgs);
        }
    }

    BOOL CoreNVGManager::AppendResidencyMaskDispatch(std::vector<FramePassItem>& outItems)
    {
        if (!m_kResidencyMask)
            return true;

        const UINT64 curRev = m_residencyManager.GetResidencyRevision();
        if (m_residencyMaskDispatchedThisFrame || (m_lastMaskRevision == curRev && curRev != 0ull))
            return true;

        const UINT32 groupCount = m_residencyManager.GetTotalClusterGroupCount();
        const UINT32 vpCount    = m_residencyManager.GetTotalVirtualPageCount();
        if (groupCount == 0u || vpCount == 0u)
            return true;

        m_kResidencyMask->SetUInt("NVGResidencyMaskParams", "Config", "groupCount", groupCount);
        m_kResidencyMask->SetUInt("NVGResidencyMaskParams", "Config", "virtualPageCount", vpCount);
        m_kResidencyMask->SetGPUBuffer("NVGResidencyMaskParams", "InClusterGroups", m_residencyManager.GetClusterGroupBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kResidencyMask->SetGPUBuffer("NVGResidencyMaskParams", "InVirtualPageTable", m_residencyManager.GetVirtualPageTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kResidencyMask->SetGPUBuffer("NVGResidencyMaskParams", "OutGroupActiveMask", m_residencyManager.GetGroupActiveMaskBuffer(), GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kResidencyMask->SetIndirectArgsBuffer(nullptr, 0u);
        m_kResidencyMask->SetDispatchThreadCount((std::min)(groupCount, 4194240u), 1u, 1u);

        FramePassItem item{};
        item.type = FramePassItemType::COMPUTE_DISPATCH;
        CoreRenderPipeline dummyPipe;
        FrameView dummyFv;
        if (!dummyPipe.BuildComputeDispatchCommand(m_kResidencyMask, dummyFv, item.computeDispatch))
            return false;

        item.computeDispatch.profileGroup  = ComputeDispatchProfileGroup::NVG_CULL_LOD;
        outItems.push_back(std::move(item));
        m_residencyMaskDispatchedThisFrame = true;
        m_lastMaskRevision                 = curRev;
        return true;
    }

    BOOL CoreNVGManager::AppendHZBBuildDispatches(
        CameraNVGContext&           ctx,
        Texture*                    sceneDepthTex,
        UINT32                      vpX,
        UINT32                      vpY,
        UINT32                      vpW,
        UINT32                      vpH,
        BOOL                        isReversedZ,
        std::vector<FramePassItem>& outItems,
        Texture*                    swDepthTex)
    {
        if (!m_kHZBBuild || !ctx.hzbTexture || !sceneDepthTex || sceneDepthTex->GetNativeHandle() == 0)
            return false;

        CoreRenderPipeline dummyPipe;
        FrameView dummyFv;

        const BOOL hasValidSWDepth = (swDepthTex != nullptr && swDepthTex->GetNativeHandle() != 0);
        Texture*   effSWDepthTex   = hasValidSWDepth ? swDepthTex : sceneDepthTex;

        // Mip 0: Conservative reduction from SceneDepth (+ isolated SWDepthTexture when active) -> HZB Mip 0
        {
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dispatchMode", 0u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcWidth", vpW);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcHeight", vpH);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dstWidth", ctx.hzbWidth);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dstHeight", ctx.hzbHeight);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "isReversedZ", isReversedZ ? 1u : 0u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcOffsetX", vpX);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcOffsetY", vpY);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "hasSWDepth", hasValidSWDepth ? 1u : 0u);
            m_kHZBBuild->SetTexture("NVGHZBBuildParams", "SrcDepth", sceneDepthTex, 0xFFFFFFFFu);
            m_kHZBBuild->SetTexture("NVGHZBBuildParams", "SrcSWDepth", effSWDepthTex, 0xFFFFFFFFu);
            m_kHZBBuild->SetRWTexture("NVGHZBBuildParams", "DstHZB", ctx.hzbTexture, GPUResourceAccess::UNORDERED_ACCESS_WRITE, 0u);
            m_kHZBBuild->SetIndirectArgsBuffer(nullptr, 0u);
            m_kHZBBuild->SetDispatchThreadCount(ctx.hzbWidth, ctx.hzbHeight, 1u);

            FramePassItem item{};
            item.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kHZBBuild, dummyFv, item.computeDispatch))
                return false;

            item.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_HZB;
            outItems.push_back(std::move(item));
        }

        // Mips 1 .. hzbMipCount - 1: Exact 2x2 conservative pyramid downsampling
        UINT32 curW = ctx.hzbWidth;
        UINT32 curH = ctx.hzbHeight;
        for (UINT32 mip = 1u; mip < ctx.hzbMipCount; ++mip)
        {
            const UINT32 dstW = (curW > 1u) ? (curW >> 1u) : 1u;
            const UINT32 dstH = (curH > 1u) ? (curH >> 1u) : 1u;

            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dispatchMode", 1u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcWidth", curW);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcHeight", curH);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dstWidth", dstW);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "dstHeight", dstH);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "isReversedZ", isReversedZ ? 1u : 0u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcOffsetX", 0u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "srcOffsetY", 0u);
            m_kHZBBuild->SetUInt("NVGHZBBuildParams", "Config", "hasSWDepth", 0u);
            m_kHZBBuild->SetTexture("NVGHZBBuildParams", "SrcDepth", ctx.hzbTexture, mip - 1u);
            m_kHZBBuild->SetTexture("NVGHZBBuildParams", "SrcSWDepth", ctx.hzbTexture, mip - 1u);
            m_kHZBBuild->SetRWTexture("NVGHZBBuildParams", "DstHZB", ctx.hzbTexture, GPUResourceAccess::UNORDERED_ACCESS_WRITE, mip);
            m_kHZBBuild->SetIndirectArgsBuffer(nullptr, 0u);
            m_kHZBBuild->SetDispatchThreadCount(dstW, dstH, 1u);

            FramePassItem item{};
            item.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kHZBBuild, dummyFv, item.computeDispatch))
                return false;

            item.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_HZB;
            outItems.push_back(std::move(item));

            curW = dstW;
            curH = dstH;
        }
        return true;
    }

    BOOL CoreNVGManager::AppendPrepareDispatchArgs(
        CameraNVGContext&           ctx,
        UINT32                      srcCounterIdx,
        UINT32                      dstArgsWordOffset,
        UINT32                      groupSizeDivisor,
        UINT32                      resetNodeCounterIdx,
        std::vector<FramePassItem>& outItems,
        ComputeDispatchProfileGroup profileGroup)
    {
        if (!m_kInstanceCull) return false;
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dispatchMode", 2u);
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "srcCounterIndex", srcCounterIdx);
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dstArgsWordOffset", dstArgsWordOffset);
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "groupSizeDivisor", groupSizeDivisor);
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "maxDispatchGroups", 65535u);
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "resetNodeCounterIdx", resetNodeCounterIdx);
        m_kInstanceCull->SetTexture("NVGInstanceCullParams", "HZBTexture", ctx.hzbTexture, 0xFFFFFFFFu);
        m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMeshTable", m_residencyManager.GetMeshTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kInstanceCull->SetIndirectArgsBuffer(nullptr, 0u);
        m_kInstanceCull->SetThreadGroupCount(1u, 1u, 1u);

        FramePassItem item{};
        item.type = FramePassItemType::COMPUTE_DISPATCH;
        CoreRenderPipeline dummyPipe;
        FrameView dummyFv;
        if (!dummyPipe.BuildComputeDispatchCommand(m_kInstanceCull, dummyFv, item.computeDispatch))
            return false;

        item.computeDispatch.profileGroup = profileGroup;
        outItems.push_back(std::move(item));
        return true;
    }

    BOOL CoreNVGManager::BuildVertexPullDrawCommand(
        const NVGDrawBucketEntry&            bucket,
        const FrameView&                     frameView,
        Scene*                               scene,
        GPUBuffer*                           gpuSceneBuffer,
        GPUBuffer*                           bucketRangesBuf,
        GPUBuffer*                           compactedVisBuf,
        GPUBuffer*                           pathAIndirectArgsBuf,
        GPUBuffer*                           shadowPageViewsBuf,
        UINT64                               passRevision,
        BOOL                                 depthWriteOverride,
        DepthCompare                         depthCompareOverride,
        BOOL                                 applyDepthOverride,
        std::vector<ComputeDispatchCommand>& outMaterializeDispatches,
        DrawCommand&                         outCmd)
    {
        outCmd = DrawCommand{};
        if (!bucket.material || !bucket.shaderSet || !pathAIndirectArgsBuf)
            return false;

        const Shader* vs = bucket.shaderSet->GetVertexShader(bucket.variant);
        const Shader* ps = bucket.shaderSet->GetPixelShader();
        const Shader* gs = bucket.shaderSet->GetGeometryShader();
        const Shader* hs = bucket.shaderSet->GetHullShader();
        const Shader* ds = bucket.shaderSet->GetDomainShader();
        if (!vs || !vs->IsValid() || vs->GetNativeHandle() == 0)
            return false;

        outCmd.vertexShader   = vs->GetNativeHandle();
        outCmd.pixelShader    = ps ? ps->GetNativeHandle() : 0;
        outCmd.geometryShader = gs ? gs->GetNativeHandle() : 0;
        outCmd.hullShader     = hs ? hs->GetNativeHandle() : 0;
        outCmd.domainShader   = ds ? ds->GetNativeHandle() : 0;

        outCmd.vertexBuffer             = 0;
        outCmd.indexBuffer              = 0;
        outCmd.vertexCount              = kNVGMaxClusterTriangles * 3u;
        outCmd.instanceCount            = 1u;
        outCmd.useVertexPullingIndirect = true;
        outCmd.indirectArgsBuffer       = pathAIndirectArgsBuf->GetNativeHandle();
        outCmd.indirectArgsOffset       = bucket.bucketIndex * static_cast<UINT32>(sizeof(GPUIndirectDrawArgs));
        outCmd.maxIndirectCommandCount  = 1u;

        outCmd.cullMode    = bucket.material->GetCullMode();
        outCmd.blendMode   = bucket.isShadowPass ? BlendMode::OPAQUE : bucket.material->GetBlendMode();
        outCmd.fillMode    = bucket.material->GetFillMode();
        outCmd.depthState  = bucket.material->GetDepthState();
        outCmd.pixelShaderModifiesCoverage = bucket.material->IsAlphaTested();
        if (applyDepthOverride)
        {
            outCmd.depthState.write   = depthWriteOverride;
            outCmd.depthState.compare = depthCompareOverride;
        }

        GPUBuffer* pagePoolBuf = m_residencyManager.GetPhysicalPagePoolBuffer();
        StageResolveContext ctx{};
        ctx.executionContextMask  = EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_DRAW | EXEC_CTX_CLUSTER |
                                    (bucket.isShadowPass ? EXEC_CTX_SHADOW_PAGE : EXEC_CTX_NONE);
        ctx.frameView             = &frameView;
        ctx.scene                 = scene;
        ctx.renderObject          = nullptr;
        ctx.material              = bucket.material;
        ctx.passType              = bucket.isShadowPass ? PassType::SHADOW : (bucket.isDepthPass ? PassType::DEPTH : PassType::GEOMETRY);
        ctx.worldMatrix           = Mat4::Identity();
        ctx.gpuSceneObjectSlot    = kInvalidGPUSceneSlot;
        ctx.geometryMode          = static_cast<UINT32>(GeometryMode::VIRTUALIZED);
        ctx.nvgDrawBucketId       = bucket.bucketIndex;
        ctx.nvgVisibleClusterOff  = 0u;
        ctx.nvgVisibleClusterCnt  = 0u;
        ctx.nvgDebugMode          = static_cast<UINT32>(m_config.debugVisualizationMode);
        ctx.nvgMaxScreenErrorPx   = ResolveEffectiveScreenErrorPixels();
        ctx.nvgPhysicalPoolCap    = m_residencyManager.GetPhysicalPoolCapacity();
        ctx.nvgPagePoolBuffer     = pagePoolBuf ? pagePoolBuf->GetNativeHandle() : 0;
        ctx.nvgPagePoolWords      = pagePoolBuf ? pagePoolBuf->GetElementCount() : 0u;
        ctx.nvgBucketRangesBuffer = bucketRangesBuf ? bucketRangesBuf->GetNativeHandle() : 0;
        ctx.nvgBucketRangesCount  = bucketRangesBuf ? bucketRangesBuf->GetElementCount() : 0u;
        ctx.nvgVisibleClustersBuf = compactedVisBuf ? compactedVisBuf->GetNativeHandle() : 0;
        ctx.nvgVisibleClustersCnt = compactedVisBuf ? compactedVisBuf->GetElementCount() : 0u;
        ctx.nvgShadowPageViewsBuf = (bucket.isShadowPass && shadowPageViewsBuf) ? shadowPageViewsBuf->GetNativeHandle() : 0;
        ctx.nvgShadowPageViewsCnt = (bucket.isShadowPass && shadowPageViewsBuf) ? shadowPageViewsBuf->GetElementCount() : 0u;
        ctx.nvgPassRevision       = passRevision;
        ctx.promotedRootKey       = nullptr;
        ctx.materializer          = m_resourceMaterializer;
        ctx.outPreDrawDispatches  = &outMaterializeDispatches;

        (void)gpuSceneBuffer;

        const Shader* stages[5] = { vs, ps, gs, hs, ds };
        const ShaderStage stageTypes[5] = {
            ShaderStage::VERTEX, ShaderStage::PIXEL, ShaderStage::GEOMETRY,
            ShaderStage::HULL, ShaderStage::DOMAIN
        };

        for (UINT32 si = 0u; si < 5u; ++si)
        {
            const Shader* st = stages[si];
            if (!st) continue;
            const ShaderStage stType       = stageTypes[si];
            const GraphicsShaderSlot gSlot = ShaderStageToSlot(stType);
            const ShaderStageProps& props   = bucket.material->GetPropertyStore().Get(bucket.passType, gSlot);
            const ShaderStageProps* fbProps = (bucket.passType != PassType::GEOMETRY)
                ? &bucket.material->GetPropertyStore().Get(PassType::GEOMETRY, gSlot)
                : nullptr;

            if (!CoreResourceBindingResolver::ResolveGraphicsStageBindings(
                    st,
                    stType,
                    props,
                    fbProps,
                    ctx,
                    outCmd.constantBuffers,
                    outCmd.textures,
                    outCmd.samplers,
                    outCmd.structuredBuffers,
                    nullptr, nullptr, nullptr, nullptr))
            {
                return false;
            }
        }

        outCmd.sortKey = ComputeDrawSortKey(outCmd);
        return true;
    }

    BOOL CoreNVGManager::BuildDecodedIADrawCommand(
        const NVGDrawBucketEntry&            bucket,
        const FrameView&                     frameView,
        Scene*                               scene,
        GPUBuffer*                           gpuSceneBuffer,
        GPUBuffer*                           bucketRangesBuf,
        GPUBuffer*                           compactedVisBuf,
        GPUBuffer*                           shadowPageViewsBuf,
        GPUBuffer*                           indirectCmdBuf,
        GPUBuffer*                           bucketCmdCountBuf,
        GPUBuffer*                           decodedVertexPoolBuf,
        GPUBuffer*                           decodedIndexPoolBuf,
        GPUBuffer*                           perDrawCBVBuf,
        GPUBuffer*                           perBatchInstBuf,
        UINT32                               maxCmdsPerBucket,
        UINT64                               passRevision,
        BOOL                                 depthWriteOverride,
        DepthCompare                         depthCompareOverride,
        BOOL                                 applyDepthOverride,
        std::vector<ComputeDispatchCommand>& outMaterializeDispatches,
        DrawCommand&                         outCmd)
    {
        outCmd = DrawCommand{};
        if (!bucket.material || !bucket.shaderSet || !indirectCmdBuf || !bucketCmdCountBuf ||
            !decodedVertexPoolBuf || !decodedIndexPoolBuf || maxCmdsPerBucket == 0u)
        {
            return false;
        }

        const Shader* vs = bucket.shaderSet->GetVertexShader(bucket.variant);
        const Shader* ps = bucket.shaderSet->GetPixelShader();
        const Shader* gs = bucket.shaderSet->GetGeometryShader();
        const Shader* hs = bucket.shaderSet->GetHullShader();
        const Shader* ds = bucket.shaderSet->GetDomainShader();
        if (!vs || !vs->IsValid() || vs->GetNativeHandle() == 0)
            return false;

        outCmd.vertexShader   = vs->GetNativeHandle();
        outCmd.pixelShader    = ps ? ps->GetNativeHandle() : 0;
        outCmd.geometryShader = gs ? gs->GetNativeHandle() : 0;
        outCmd.hullShader     = hs ? hs->GetNativeHandle() : 0;
        outCmd.domainShader   = ds ? ds->GetNativeHandle() : 0;

        outCmd.useNVSIndirectCommandSignature = true;
        outCmd.indirectRootKey                = bucket.indirectRootKey;
        outCmd.indirectArgsBuffer             = indirectCmdBuf->GetNativeHandle();
        outCmd.indirectArgsOffset             = bucket.bucketIndex * maxCmdsPerBucket * 144u; // 36 UINT32 words/slot
        outCmd.maxIndirectCommandCount        = maxCmdsPerBucket;
        outCmd.indirectCountBuffer            = bucketCmdCountBuf->GetNativeHandle();
        outCmd.indirectCountBufferOffset      = bucket.bucketIndex * static_cast<UINT32>(sizeof(UINT32));
        outCmd.indirectVertexPoolBuffer       = (bucket.decodeVertexWords > 0u) ? decodedVertexPoolBuf->GetNativeHandle() : 0;
        outCmd.indirectIndexPoolBuffer        = decodedIndexPoolBuf->GetNativeHandle();
        outCmd.indirectPerDrawCBVBuffer       = perDrawCBVBuf ? perDrawCBVBuf->GetNativeHandle() : 0;
        outCmd.indirectPerBatchInstanceBuffer = perBatchInstBuf ? perBatchInstBuf->GetNativeHandle() : 0;
        outCmd.indirectBonePaletteBuffer      = frameView.gpuSkinningPaletteBuffer;

        outCmd.cullMode    = bucket.material->GetCullMode();
        outCmd.blendMode   = bucket.isShadowPass ? BlendMode::OPAQUE : bucket.material->GetBlendMode();
        outCmd.fillMode    = bucket.material->GetFillMode();
        outCmd.depthState  = bucket.material->GetDepthState();
        outCmd.pixelShaderModifiesCoverage = bucket.material->IsAlphaTested();
        if (applyDepthOverride)
        {
            outCmd.depthState.write   = depthWriteOverride;
            outCmd.depthState.compare = depthCompareOverride;
        }

        // Bind ALL non-promoted stage resources INCLUDING live NVG PagePool, BucketRanges,
        // VisibleClusters, GPUSceneObjects, and (in shadow pass) ShadowPageViews!
        // Enables full hybrid IA + NVG + Add(bool) dynamic path switching inside the same shader!
        GPUBuffer* pagePoolBuf = m_residencyManager.GetPhysicalPagePoolBuffer();
        StageResolveContext ctx{};
        ctx.executionContextMask  = EXEC_CTX_SCENE | EXEC_CTX_FRAME | EXEC_CTX_CAMERA_VIEW | EXEC_CTX_DRAW | EXEC_CTX_CLUSTER |
                                    (bucket.isShadowPass ? EXEC_CTX_SHADOW_PAGE : EXEC_CTX_NONE);
        ctx.frameView             = &frameView;
        ctx.scene                 = scene;
        ctx.renderObject          = nullptr;
        ctx.material              = bucket.material;
        ctx.passType              = bucket.isShadowPass ? PassType::SHADOW : (bucket.isDepthPass ? PassType::DEPTH : PassType::GEOMETRY);
        ctx.worldMatrix           = Mat4::Identity();
        ctx.gpuSceneObjectSlot    = kInvalidGPUSceneSlot; // GPU patches exact per-cluster slot in promoted CBV
        ctx.geometryMode          = static_cast<UINT32>(GeometryMode::VIRTUALIZED);
        ctx.nvgDrawBucketId       = bucket.bucketIndex;
        ctx.nvgVisibleClusterOff  = kNVGInvalidIndex;     // GPU patches exact per-cluster offset in promoted CBV
        ctx.nvgVisibleClusterCnt  = 0u;                   // GPU patches exact bucket count in promoted CBV
        ctx.nvgDebugMode          = static_cast<UINT32>(m_config.debugVisualizationMode);
        ctx.nvgMaxScreenErrorPx   = ResolveEffectiveScreenErrorPixels();
        ctx.nvgPhysicalPoolCap    = m_residencyManager.GetPhysicalPoolCapacity();
        ctx.nvgPagePoolBuffer     = pagePoolBuf ? pagePoolBuf->GetNativeHandle() : 0;
        ctx.nvgPagePoolWords      = pagePoolBuf ? pagePoolBuf->GetElementCount() : 0u;
        ctx.nvgBucketRangesBuffer = bucketRangesBuf ? bucketRangesBuf->GetNativeHandle() : 0;
        ctx.nvgBucketRangesCount  = bucketRangesBuf ? bucketRangesBuf->GetElementCount() : 0u;
        ctx.nvgVisibleClustersBuf = compactedVisBuf ? compactedVisBuf->GetNativeHandle() : 0;
        ctx.nvgVisibleClustersCnt = compactedVisBuf ? compactedVisBuf->GetElementCount() : 0u;
        ctx.nvgShadowPageViewsBuf = (bucket.isShadowPass && shadowPageViewsBuf) ? shadowPageViewsBuf->GetNativeHandle() : 0;
        ctx.nvgShadowPageViewsCnt = (bucket.isShadowPass && shadowPageViewsBuf) ? shadowPageViewsBuf->GetElementCount() : 0u;
        ctx.nvgPassRevision       = passRevision;
        ctx.promotedRootKey       = &bucket.indirectRootKey;
        ctx.materializer          = m_resourceMaterializer;
        ctx.outPreDrawDispatches  = &outMaterializeDispatches;

        (void)gpuSceneBuffer;

        const Shader* stages[5] = { vs, ps, gs, hs, ds };
        const ShaderStage stageTypes[5] = {
            ShaderStage::VERTEX, ShaderStage::PIXEL, ShaderStage::GEOMETRY,
            ShaderStage::HULL, ShaderStage::DOMAIN
        };

        for (UINT32 si = 0u; si < 5u; ++si)
        {
            const Shader* st = stages[si];
            if (!st) continue;
            const ShaderStage stType       = stageTypes[si];
            const GraphicsShaderSlot gSlot = ShaderStageToSlot(stType);
            const ShaderStageProps& props   = bucket.material->GetPropertyStore().Get(bucket.passType, gSlot);
            const ShaderStageProps* fbProps = (bucket.passType != PassType::GEOMETRY)
                ? &bucket.material->GetPropertyStore().Get(PassType::GEOMETRY, gSlot)
                : nullptr;

            if (!CoreResourceBindingResolver::ResolveGraphicsStageBindings(
                    st,
                    stType,
                    props,
                    fbProps,
                    ctx,
                    outCmd.constantBuffers,
                    outCmd.textures,
                    outCmd.samplers,
                    outCmd.structuredBuffers,
                    nullptr, nullptr, nullptr, nullptr))
            {
                return false;
            }
        }

        outCmd.sortKey = ComputeDrawSortKey(outCmd);
        return true;
    }

    BOOL CoreNVGManager::InjectCameraNVGPasses(
        Scene*                    scene,
        Camera*                   camera,
        const NativeRenderTarget& defaultRenderTarget,
        FrameView&                ioFrameView,
        GPUBuffer*                gpuSceneBuffer,
        UINT32                    gpuSceneSlotCount,
        CoreRenderPipeline*       renderPipeline,
        RenderStats&              stats,
        BOOL                      isFirstCameraInFrame,
        BOOL                      isFinalCameraInFrame)
    {
        (void)defaultRenderTarget;
        (void)renderPipeline;
        if (!m_initialized || !m_config.enabled || !scene || !camera || !gpuSceneBuffer || gpuSceneSlotCount == 0u)
            return false;
        if (m_cameraBuckets.empty() || m_residencyManager.GetRegisteredMeshCount() == 0u)
            return false;
        if (!EnsureRuntimeKernelsLoaded())
            return false;

        m_residencyManager.SyncDirtyMetadataToGPU();

        auto& ctxPtr = m_cameraContexts[camera];
        if (!ctxPtr)
        {
            ctxPtr = std::make_unique<CameraNVGContext>();
            ctxPtr->camera = camera;
        }
        CameraNVGContext& ctx = *ctxPtr;
        ctx.usedThisFrame          = true;
        ctx.cameraPassRanThisFrame = true;

        Texture* depthForSize = camera->GetSceneDepthTexture();
        const UINT32 vpX = static_cast<UINT32>((std::max)(0.0f, ioFrameView.camera.viewportX));
        const UINT32 vpY = static_cast<UINT32>((std::max)(0.0f, ioFrameView.camera.viewportY));
        const UINT32 vpW = (ioFrameView.camera.viewportWidth > 0.0f)
            ? static_cast<UINT32>(ioFrameView.camera.viewportWidth)
            : (depthForSize && depthForSize->GetWidth() > 0u ? depthForSize->GetWidth() : 1280u);
        const UINT32 vpH = (ioFrameView.camera.viewportHeight > 0.0f)
            ? static_cast<UINT32>(ioFrameView.camera.viewportHeight)
            : (depthForSize && depthForSize->GetHeight() > 0u ? depthForSize->GetHeight() : 720u);

        if (!EnsureCameraContextResources(ctx, vpW, vpH, gpuSceneSlotCount))
            return false;

        RefreshBucketTemplatesForView(ioFrameView, scene, false);

        GPUBuffer* objMatSlotsBuf = scene->GetGPUMaterialSlotTableBuffer();
        const UINT32 objMatSlotsCnt = (objMatSlotsBuf && objMatSlotsBuf->GetNativeHandle() != 0)
            ? scene->GetGPUMaterialSlotTableCount()
            : 0u;
        if (!objMatSlotsBuf || objMatSlotsBuf->GetNativeHandle() == 0)
        {
            objMatSlotsBuf = m_dummyObjectMatSlotsBuf;
        }
        m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "objectMatSlotTableCount", objMatSlotsCnt);
        m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InObjectMaterialSlots", objMatSlotsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);

        // Detect camera teleport/cut -> invalidate prev-frame HZB so Phase 1 never over-occludes
        const Vec3 camPos = ioFrameView.camera.position;
        if (ctx.hasValidPrevHZB)
        {
            if ((camPos - ctx.prevCameraPos).Length() > 25.0f)
            {
                ctx.hasValidPrevHZB = false;
            }
        }

        // Crack-free temporal LOD hysteresis via camera-position deadband
        if (!ctx.hasLodHysteresis || m_config.lodHysteresis <= 0.0f)
        {
            ctx.lodHysteresisPos = camPos;
            ctx.hasLodHysteresis = true;
        }
        else
        {
            const FLOAT32 deadband = m_config.lodHysteresis * (std::max)(ioFrameView.camera.nearPlane, 0.1f);
            if ((camPos - ctx.lodHysteresisPos).Length() > deadband)
            {
                ctx.lodHysteresisPos = camPos;
            }
        }

        const BOOL isOrtho = (camera->GetProjectionType() == ProjectionType::ORTHOGRAPHIC);
        const FLOAT32 projM11 = std::fabs(ioFrameView.camera.projectionMatrix.m[5]);
        const FLOAT32 lodProjScale = 0.5f * static_cast<FLOAT32>(vpH) * (std::max)(projM11, 1e-5f);

        Vec4 frustumPlanes[6]{};
        ExtractFrustumPlanesFromVP(ioFrameView.camera.viewProjectionMatrix, frustumPlanes);

        const UINT32 bucketCount  = static_cast<UINT32>(m_cameraBuckets.size());
        const UINT32 vpCount      = m_residencyManager.GetTotalVirtualPageCount();
        const UINT32 bvhNodeCount = m_residencyManager.GetTotalBVHNodeCount();
        const UINT32 clusterCount = m_residencyManager.GetTotalClusterCount();
        const UINT32 groupCount   = m_residencyManager.GetTotalClusterGroupCount();
        const UINT32 meshTabCount = m_residencyManager.GetMeshTableCapacity();
        const UINT32 matTabCount  = static_cast<UINT32>(m_materialTable.size());
        const BOOL   hasPrevHZB   = (m_config.enableHZBOcclusion && ctx.hasValidPrevHZB && ctx.hzbTexture != nullptr);
        const BOOL   isDepthPre   = ioFrameView.hasDepthPrepass;
        const BOOL   isRevZ       = false; // Camera scene depth is Standard-Z (0..1)

        // Check if any camera bucket uses Path B (IA-Stream), SW eligibility, or user Get(NVG_VISIBILITY_BUFFER)
        const UINT32 reqCamSWFlag   = isRevZ ? NVG_BUCKET_FLAG_SW_ELIGIBLE_REVZ : NVG_BUCKET_FLAG_SW_ELIGIBLE;
        BOOL hasAnyPathB            = false;
        BOOL hasAnySWEligibleBucket = false;
        BOOL userConsumesVisBuffer  = false;
        std::vector<UINT32> transparentBucketIds;
        for (const auto& b : m_cameraBuckets)
        {
            if (!b.isVertexPull) hasAnyPathB = true;
            if (b.isTransparent) transparentBucketIds.push_back(b.bucketIndex);
            if (b.bucketIndex < m_cameraBucketFlags.size() &&
                (m_cameraBucketFlags[b.bucketIndex] & reqCamSWFlag) != 0u)
            {
                hasAnySWEligibleBucket = true;
            }
            if (b.shaderSet)
            {
                if (DoesShaderReferenceVisBuffer(b.shaderSet->GetVertexShader(b.variant)) ||
                    DoesShaderReferenceVisBuffer(b.shaderSet->GetPixelShader()))
                {
                    userConsumesVisBuffer = true;
                }
            }
        }
        for (const auto& ke : scene->GetComputeKernelEntries())
        {
            if (ke.kernel && ke.kernel->IsValid() && DoesShaderReferenceVisBuffer(ke.kernel->GetComputeShader()))
            {
                userConsumesVisBuffer = true;
                break;
            }
        }

        const NVGRasterAtomicStrategy activeStrategy = ResolveActiveCameraRasterStrategy();
        m_stats.activeSWRasterStrategy = activeStrategy;

        const BOOL runSWRasterPhase1 = m_config.enableSoftwareRasterizer &&
                                       (activeStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS ||
                                        activeStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS) &&
                                       (hasAnySWEligibleBucket || userConsumesVisBuffer) &&
                                       ctx.swVisTexture != nullptr &&
                                       ctx.swDepthTexture != nullptr &&
                                       m_kDepthExport != nullptr &&
                                       ((activeStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS && m_kMicropolyRasterAtomic64 != nullptr) ||
                                        (activeStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC32_TWO_PASS   && m_kMicropolyRasterAtomic32 != nullptr));
        const BOOL effHasPrevHZB     = hasPrevHZB && !userConsumesVisBuffer;

        // Populate FrameView NVG handles upfront so both NVG graphics draws and re-resolved user compute kernels see live handles
        ioFrameView.nvgPagePoolBuffer        = m_residencyManager.GetPhysicalPagePoolBuffer()->GetNativeHandle();
        ioFrameView.nvgPagePoolWordCount     = m_residencyManager.GetPhysicalPagePoolBuffer()->GetElementCount();
        ioFrameView.nvgVisibleClustersBuffer = ctx.compactedVisibleClusters->GetNativeHandle();
        ioFrameView.nvgVisibleClustersCount  = ctx.compactedVisibleClusters->GetElementCount();
        ioFrameView.nvgBucketRangesBuffer    = ctx.bucketRanges->GetNativeHandle();
        ioFrameView.nvgBucketRangesCount     = ctx.bucketRanges->GetElementCount();
        ioFrameView.nvgPhysicalPoolCapacity  = m_residencyManager.GetPhysicalPoolCapacity();
        ioFrameView.nvgResidencyRevision     = m_residencyManager.GetResidencyRevision();
        ioFrameView.nvgDebugMode             = static_cast<UINT32>(m_config.debugVisualizationMode);
        ioFrameView.nvgMaxScreenErrorPx      = ResolveEffectiveScreenErrorPixels();
        ioFrameView.nvgVisibilityTexture     = (runSWRasterPhase1 && ctx.swVisTexture) ? ctx.swVisTexture->GetNativeHandle() : 0;

        // Refresh any POST_DEPTH / POST_OPAQUE / POST_PASS user compute dispatches built earlier in BuildFrameView
        // that consume Get(NVG_VISIBILITY_BUFFER) or NVG StructuredBuffers / CBuffer metadata
        if (renderPipeline != nullptr)
        {
            for (const auto& ke : scene->GetComputeKernelEntries())
            {
                if (!ke.kernel || !ke.kernel->IsValid() || ke.phase == ComputePassPhase::PRE_PASS)
                    continue;
                const Shader* cs = ke.kernel->GetComputeShader();
                if (!DoesComputeShaderReferenceNVG(cs))
                    continue;

                ComputeDispatchCommand refreshedCmd{};
                std::vector<ComputeDispatchCommand> prepDispatches;
                if (!renderPipeline->BuildComputeDispatchCommand(ke.kernel, ioFrameView, refreshedCmd, &prepDispatches))
                    continue;

                const NativeHandle csHandle = cs->GetNativeHandle();
                if (ke.phase == ComputePassPhase::POST_PASS)
                {
                    for (SIZE_T di = 0u; di < ioFrameView.postPassComputeDispatches.size(); ++di)
                    {
                        if (ioFrameView.postPassComputeDispatches[di].computeShader == csHandle)
                        {
                            ioFrameView.postPassComputeDispatches[di] = std::move(refreshedCmd);
                            if (!prepDispatches.empty())
                            {
                                ioFrameView.postPassComputeDispatches.insert(
                                    ioFrameView.postPassComputeDispatches.begin() + static_cast<std::ptrdiff_t>(di),
                                    std::make_move_iterator(prepDispatches.begin()),
                                    std::make_move_iterator(prepDispatches.end()));
                            }
                            break;
                        }
                    }
                }
                else
                {
                    for (SIZE_T ii = 0u; ii < ioFrameView.interleavedPassItems.size(); ++ii)
                    {
                        if (ioFrameView.interleavedPassItems[ii].type == FramePassItemType::COMPUTE_DISPATCH &&
                            ioFrameView.interleavedPassItems[ii].computeDispatch.computeShader == csHandle)
                        {
                            ioFrameView.interleavedPassItems[ii].computeDispatch = std::move(refreshedCmd);
                            if (!prepDispatches.empty())
                            {
                                std::vector<FramePassItem> prepItems;
                                prepItems.reserve(prepDispatches.size());
                                for (auto& pd : prepDispatches)
                                {
                                    FramePassItem pi{};
                                    pi.type            = FramePassItemType::COMPUTE_DISPATCH;
                                    pi.computeDispatch = std::move(pd);
                                    prepItems.push_back(std::move(pi));
                                }
                                ioFrameView.interleavedPassItems.insert(
                                    ioFrameView.interleavedPassItems.begin() + static_cast<std::ptrdiff_t>(ii),
                                    std::make_move_iterator(prepItems.begin()),
                                    std::make_move_iterator(prepItems.end()));
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Helper to append BVH traversal + cluster cull + compaction + decode + Path-B indirect draw
        CoreRenderPipeline dummyPipe;
        auto appendCullCompactAndIndirect = [&](BOOL isPhase2, std::vector<FramePassItem>& items) -> BOOL {
            if (!isPhase2)
            {
                // Reset frame counters, buckets, telemetry, and (ONLY on the first camera of the frame after a
                // completed readback) page requests — subsequent cameras accumulate via InterlockedMax (NVG-017)!
                const BOOL clearReqsNow = isFirstCameraInFrame ? m_residencyManager.ConsumeClearPageRequestsFlag() : false;
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dispatchMode", 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "bucketCount", bucketCount);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "virtualPageCount", clearReqsNow ? vpCount : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "clearPageRequests", clearReqsNow ? 1u : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "resetFlags", 0u);
                m_kInstanceCull->SetTexture("NVGInstanceCullParams", "HZBTexture", ctx.hzbTexture);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMeshTable", m_residencyManager.GetMeshTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutNodeQueue", ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutOccludedObjects", ctx.occludedObjects, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutBucketCounts", ctx.bucketCounts, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutPageRequests", m_residencyManager.GetPageRequestBuffer(), GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetIndirectArgsBuffer(nullptr, 0u);
                const UINT32 resetThreads = (std::min)((std::max)({ bucketCount, clearReqsNow ? vpCount : 0u, 64u }), 4194240u);
                m_kInstanceCull->SetDispatchThreadCount(resetThreads, 1u, 1u);

                FramePassItem rstItem{};
                rstItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kInstanceCull, ioFrameView, rstItem.computeDispatch))
                    return false;
                rstItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(rstItem));

                // Object culling (Mode 1)
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dispatchMode", 1u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "sceneSlotCount", gpuSceneSlotCount);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "meshTableCount", meshTabCount);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "materialTableCount", matTabCount);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "maxNodeQueueCapacity", ctx.nodeQueueA->GetElementCount());
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "maxOccludedObjCapacity", ctx.occludedObjects->GetElementCount());
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "cameraCullingMask", camera->GetCullingMask());
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "passKind", 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "enableHZBOcclusion", m_config.enableHZBOcclusion ? 1u : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "enableTwoPhase", m_config.enableTwoPhaseDisocclusion ? 1u : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "hasPrevHZB", effHasPrevHZB ? 1u : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "isReversedZ", isRevZ ? 1u : 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "hzbWidth", ctx.hzbWidth);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "hzbHeight", ctx.hzbHeight);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "hzbMipCount", ctx.hzbMipCount);
                m_kInstanceCull->SetVec3("NVGInstanceCullParams", "Config", "cameraPosition", ioFrameView.camera.position);
                m_kInstanceCull->SetFloat("NVGInstanceCullParams", "Config", "cameraFarPlane", ioFrameView.camera.farPlane);
                m_kInstanceCull->SetVec3("NVGInstanceCullParams", "Config", "cameraForward", ioFrameView.camera.direction);
                m_kInstanceCull->SetFloat("NVGInstanceCullParams", "Config", "smallObjectPixelThresh", (std::max)(m_config.smallClusterPixelThreshold, 0.5f));
                m_kInstanceCull->SetFloat("NVGInstanceCullParams", "Config", "viewportWidth", static_cast<FLOAT32>(vpW));
                m_kInstanceCull->SetFloat("NVGInstanceCullParams", "Config", "viewportHeight", static_cast<FLOAT32>(vpH));
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane0", frustumPlanes[0]);
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane1", frustumPlanes[1]);
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane2", frustumPlanes[2]);
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane3", frustumPlanes[3]);
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane4", frustumPlanes[4]);
                m_kInstanceCull->SetVec4("NVGInstanceCullParams", "Config", "frustumPlane5", frustumPlanes[5]);
                m_kInstanceCull->SetMat4("NVGInstanceCullParams", "Config", "prevViewProj", ctx.prevViewProj);
                m_kInstanceCull->SetMat4("NVGInstanceCullParams", "Config", "currViewProj", ioFrameView.camera.viewProjectionMatrix);
                m_kInstanceCull->SetIndirectArgsBuffer(nullptr, 0u);
                m_kInstanceCull->SetDispatchThreadCount((std::min)(gpuSceneSlotCount, 4194240u), 1u, 1u);

                FramePassItem objItem{};
                objItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kInstanceCull, ioFrameView, objItem.computeDispatch))
                    return false;
                objItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(objItem));
            }
            else
            {
                // Fast single-dispatch Phase-2 Reset: zeroes counters 0..3 & bucketCounts, prepares OutDispatchArgs[0..2]
                // for occluded objects (counter 6), while preserving counter 4 (occludedClusters), telemetry & page requests!
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dispatchMode", 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "bucketCount", bucketCount);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "virtualPageCount", 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "clearPageRequests", 0u);
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "resetFlags", 1u);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kInstanceCull->SetIndirectArgsBuffer(nullptr, 0u);
                m_kInstanceCull->SetDispatchThreadCount((std::max)(bucketCount, 64u), 1u, 1u);
                {
                    FramePassItem p2Rst{};
                    p2Rst.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kInstanceCull, ioFrameView, p2Rst.computeDispatch))
                        return false;
                    p2Rst.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                    items.push_back(std::move(p2Rst));
                }

                // Mode 3: Re-test occluded objects against fresh current-frame HZB
                // Unbind OutDispatchArgs UAV before using ctx.dispatchArgs as indirect dispatch args!
                m_kInstanceCull->SetUInt("NVGInstanceCullParams", "Config", "dispatchMode", 3u);
                m_kInstanceCull->SetTexture("NVGInstanceCullParams", "HZBTexture", ctx.hzbTexture, 0xFFFFFFFFu);
                m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutDispatchArgs", nullptr);
                m_kInstanceCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 0u);
                FramePassItem p2Obj{};
                p2Obj.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kInstanceCull, ioFrameView, p2Obj.computeDispatch))
                    return false;
                p2Obj.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(p2Obj));
            }

            // Common HierarchyCull config & resource bindings
            GPUBuffer* skinBoundsBuf = scene->GetGPUSkinningBoundsBuffer();
            const UINT32 skinBoundsCnt = (skinBoundsBuf && skinBoundsBuf->GetNativeHandle() != 0)
                ? scene->GetGPUSkinningBoundsCount()
                : 0u;
            if (!skinBoundsBuf || skinBoundsBuf->GetNativeHandle() == 0)
            {
                skinBoundsBuf = m_dummySkinnedBoundsBuf;
            }

            const FLOAT32 effScreenErr = ResolveEffectiveScreenErrorPixels();
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "maxNodeQueueCapacity", ctx.nodeQueueA->GetElementCount());
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "maxCandidateCapacity", ctx.candidateClusters->GetElementCount());
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "maxVisibleCapacity", ctx.rawVisibleClusters->GetElementCount());
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "maxOccludedClusterCap", ctx.occludedClusters->GetElementCount());
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "maxOccludedNodeCap", ctx.occludedNodes->GetElementCount());
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "skinBoundsCount", skinBoundsCnt);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "isPhase2Pass", isPhase2 ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "bvhNodeCount", bvhNodeCount);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "clusterMetaCount", clusterCount);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "clusterGroupCount", groupCount);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "virtualPageCount", vpCount);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "materialTableCount", matTabCount);
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "maxScreenErrorPixels", effScreenErr);
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "lodProjScale", lodProjScale);
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "cameraNearPlane", (std::max)(ioFrameView.camera.nearPlane, 0.001f));
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "isOrthographic", isOrtho ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "enableConeCulling", m_config.enableBackfaceConeCulling ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "enableSmallClusterCull", m_config.enableSmallClusterCulling ? 1u : 0u);
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "smallClusterPixelThresh", m_config.smallClusterPixelThreshold);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "enableHZBOcclusion", m_config.enableHZBOcclusion ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "enableTwoPhase", m_config.enableTwoPhaseDisocclusion ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "hasPrevHZB", effHasPrevHZB ? 1u : 0u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "isReversedZ", isRevZ ? 1u : 0u);
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "viewportWidth", static_cast<FLOAT32>(vpW));
            m_kHierarchyCull->SetFloat("NVGHierarchyCullParams", "Config", "viewportHeight", static_cast<FLOAT32>(vpH));
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "hzbWidth", ctx.hzbWidth);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "hzbHeight", ctx.hzbHeight);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "hzbMipCount", ctx.hzbMipCount);
            m_kHierarchyCull->SetVec3("NVGHierarchyCullParams", "Config", "lodCameraPos", ctx.lodHysteresisPos);
            m_kHierarchyCull->SetVec3("NVGHierarchyCullParams", "Config", "cameraForward", ioFrameView.camera.direction);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane0", frustumPlanes[0]);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane1", frustumPlanes[1]);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane2", frustumPlanes[2]);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane3", frustumPlanes[3]);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane4", frustumPlanes[4]);
            m_kHierarchyCull->SetVec4("NVGHierarchyCullParams", "Config", "frustumPlane5", frustumPlanes[5]);
            m_kHierarchyCull->SetMat4("NVGHierarchyCullParams", "Config", "prevViewProj", ctx.prevViewProj);
            m_kHierarchyCull->SetMat4("NVGHierarchyCullParams", "Config", "currViewProj", ioFrameView.camera.viewProjectionMatrix);

            m_kHierarchyCull->SetTexture("NVGHierarchyCullParams", "HZBTexture", ctx.hzbTexture);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InObjectMaterialSlots", objMatSlotsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "objectMatSlotTableCount", objMatSlotsCnt);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "passKind", 0u);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InBVHNodes", m_residencyManager.GetBVHNodeBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InClusterMeta", m_residencyManager.GetClusterMetaBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InClusterGroups", m_residencyManager.GetClusterGroupBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InVirtualPageTable", m_residencyManager.GetVirtualPageTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InGroupActiveMask", m_residencyManager.GetGroupActiveMaskBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InSkinnedBounds", skinBoundsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutOccludedNodes", ctx.occludedNodes, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutCandidateClusters", ctx.candidateClusters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutOccludedClusters", ctx.occludedClusters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutVisibleClusters", ctx.rawVisibleClusters, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutBucketCounts", ctx.bucketCounts, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutPageRequests", m_residencyManager.GetPageRequestBuffer(), GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);

            // In Phase 2, re-test previously PrevHZB-occluded inner BVH4 subtrees against current-frame HZB BEFORE BVH traversal!
            if (isPhase2)
            {
                m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "dispatchMode", 3u);
                m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InNodeQueue", ctx.nodeQueueB, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutNodeQueue", ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kHierarchyCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 40u * sizeof(UINT32));

                FramePassItem occNodeItem{};
                occNodeItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kHierarchyCull, ioFrameView, occNodeItem.computeDispatch))
                    return false;
                occNodeItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(occNodeItem));
            }

            // Ping-pong 4-wide BVH4 cooperative traversal (16 parent nodes x 4 lanes = 64 threads/group; zero-cost GPU no-op once empty)
            const UINT32 steps = (std::min)((std::max)(m_maxMeshBVHDepth, 1u), kNVGMaxBVHDepth);
            for (UINT32 step = 0u; step < steps; ++step)
            {
                const BOOL   even   = ((step & 1u) == 0u);
                const UINT32 srcCnt = even ? 0u : 1u;
                const UINT32 dstCnt = even ? 1u : 0u;

                AppendPrepareDispatchArgs(ctx, srcCnt, 0u, 16u, dstCnt, items, ComputeDispatchProfileGroup::NVG_CULL_LOD);

                m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "dispatchMode", 0u);
                m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "srcCounterIndex", srcCnt);
                m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "dstNodeCounterIndex", dstCnt);
                m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InNodeQueue", even ? ctx.nodeQueueA : ctx.nodeQueueB, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutNodeQueue", even ? ctx.nodeQueueB : ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kHierarchyCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 0u);

                FramePassItem bvhItem{};
                bvhItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kHierarchyCull, ioFrameView, bvhItem.computeDispatch))
                    return false;
                bvhItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(bvhItem));
            }

            // Evaluate leaf candidate clusters (Mode 1)
            AppendPrepareDispatchArgs(ctx, 2u, 4u, 64u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_CULL_LOD);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "dispatchMode", 1u);
            m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "hasPrevHZB", isPhase2 ? 0u : (effHasPrevHZB ? 1u : 0u));
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "InNodeQueue", ctx.nodeQueueA, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kHierarchyCull->SetGPUBuffer("NVGHierarchyCullParams", "OutNodeQueue", ctx.nodeQueueB, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kHierarchyCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 4u * sizeof(UINT32));
            {
                FramePassItem clItem{};
                clItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kHierarchyCull, ioFrameView, clItem.computeDispatch))
                    return false;
                clItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(clItem));
            }

            // In Phase 2, ALSO re-evaluate previously occluded clusters from Phase 1 (Mode 2)
            if (isPhase2)
            {
                AppendPrepareDispatchArgs(ctx, 4u, 8u, 64u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_CULL_LOD);
                m_kHierarchyCull->SetUInt("NVGHierarchyCullParams", "Config", "dispatchMode", 2u);
                m_kHierarchyCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 8u * sizeof(UINT32));
                FramePassItem cl2Item{};
                cl2Item.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kHierarchyCull, ioFrameView, cl2Item.computeDispatch))
                    return false;
                cl2Item.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_CULL_LOD;
                items.push_back(std::move(cl2Item));
            }

            // Compaction Mode 0: Prefix Sum + Path-A DrawInstanced Indirect Args
            GPUBuffer* outBucketRanges = isPhase2 ? ctx.p2BucketRanges       : ctx.bucketRanges;
            GPUBuffer* outPathAArgs    = isPhase2 ? ctx.p2PathAIndirectArgs  : ctx.pathAIndirectArgs;
            GPUBuffer* outCompactedVis = isPhase2 ? ctx.p2CompactedVisible   : ctx.compactedVisibleClusters;
            GPUBuffer* outDecVerts     = (isPhase2 && ctx.p2DecodedVertexPool) ? ctx.p2DecodedVertexPool : ctx.decodedVertexPool;
            GPUBuffer* outDecIdxs      = (isPhase2 && ctx.p2DecodedIndexPool)  ? ctx.p2DecodedIndexPool  : ctx.decodedIndexPool;

            ++ctx.frameTagCounter;
            if (ctx.frameTagCounter == 0u) ctx.frameTagCounter = 1u;
            const UINT32 frameTag = ctx.frameTagCounter;

            const BOOL   useSWCullHere = (!isPhase2 && runSWRasterPhase1);
            const UINT32 swMaskWordCnt = useSWCullHere ? ctx.swScheduledMask->GetElementCount() : 0u;
            GPUBuffer*   schedMaskBuf  = useSWCullHere ? ctx.swScheduledMask    : m_dummySWMaskBuf;
            GPUBuffer*   winMaskBuf    = useSWCullHere ? ctx.swWinningClusterMask : m_dummySWMaskBuf;

            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 0u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "bucketCount", bucketCount);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxVisibleCapacity", ctx.rawVisibleClusters->GetElementCount());
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "rawVisibleCounterIdx", 3u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "frameTag", frameTag);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "pageWordCount", kNVGPageWordCount);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxDecodedVertexWords", outDecVerts->GetElementCount());
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxDecodedIndices", outDecIdxs->GetElementCount());
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxUniqueDecodeJobs", ctx.uniqueDecodeJobs->GetElementCount());
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "decodedTableCapacity", ctx.decodedClusterTable->GetElementCount());
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "shadowCommitRecordCap", 0u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "drawGroupBindingCount", static_cast<UINT32>(m_cameraBindingDescs.size()));
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "enableSWClusterCull", 0u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "swMaskWordCount", 0u);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InPagePool", m_residencyManager.GetPhysicalPagePoolBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InBucketCounts", ctx.bucketCounts, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InBucketFlags", m_cameraBucketFlagsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InDrawGroupBindings", m_cameraBindingsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InRawVisibleClusters", ctx.rawVisibleClusters, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InSWScheduledMask", m_dummySWMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InWinningClusterMask", m_dummySWMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutBucketRanges", outBucketRanges, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutBucketWriteCursors", ctx.bucketWriteCursors, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutPathAIndirectArgs", outPathAArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutCompactedVisibleClusters", outCompactedVis, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedClusterTable", ctx.decodedClusterTable, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutUniqueDecodeJobs", ctx.uniqueDecodeJobs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedVertexWords", outDecVerts, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedIndexPool", outDecIdxs, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InOutPageCommitRecords", m_dummyCommitRecordsBuf, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kCompactionDecode->SetIndirectArgsBuffer(nullptr, 0u);
            const UINT32 clearGroups = (ctx.decodedClusterTable->GetElementCount() + 127u) / 128u;
            m_kCompactionDecode->SetThreadGroupCount((std::max)(clearGroups, 1u), 1u, 1u);
            {
                FramePassItem psItem{};
                psItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, psItem.computeDispatch))
                    return false;
                psItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                items.push_back(std::move(psItem));
            }

            // Compaction Mode 1: Scatter Raw Visible -> Compacted Per-Bucket Ranges
            AppendPrepareDispatchArgs(ctx, 3u, 12u, 128u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_DRAW_GEN);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 1u);
            m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 12u * sizeof(UINT32));
            {
                FramePassItem scItem{};
                scItem.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, scItem.computeDispatch))
                    return false;
                scItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                items.push_back(std::move(scItem));
            }

            // Hybrid Single-Dispatch Groupshared Tile Sort (Mode 5) + Inter-Tile Merge (Mode 2) up to 16,384 Transparent Clusters:
            if (!transparentBucketIds.empty())
            {
                const UINT32 sortCap = 16384u;
                for (UINT32 tb : transparentBucketIds)
                {
                    m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 5u);
                    m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortBucketIndex", tb);
                    m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortMaxElements", sortCap);
                    m_kCompactionDecode->SetIndirectArgsBuffer(nullptr, 0u);
                    m_kCompactionDecode->SetThreadGroupCount(sortCap / 1024u, 1u, 1u);
                    {
                        FramePassItem tileSrt{};
                        tileSrt.type = FramePassItemType::COMPUTE_DISPATCH;
                        if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, tileSrt.computeDispatch))
                            return false;
                        tileSrt.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                        items.push_back(std::move(tileSrt));
                    }

                    for (UINT32 p = 1024u; p < sortCap; p <<= 1u)
                    {
                        for (UINT32 k = p; k >= 1u; k >>= 1u)
                        {
                            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 2u);
                            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortBucketIndex", tb);
                            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortK", p);
                            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortJ", k);
                            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "sortMaxElements", sortCap);
                            m_kCompactionDecode->SetIndirectArgsBuffer(nullptr, 0u);
                            m_kCompactionDecode->SetDispatchThreadCount(sortCap, 1u, 1u);
                            FramePassItem srtItem{};
                            srtItem.type = FramePassItemType::COMPUTE_DISPATCH;
                            if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, srtItem.computeDispatch))
                                return false;
                            srtItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                            items.push_back(std::move(srtItem));
                        }
                    }
                }
            }

            // Compute Software Micropoly Rasterizer + Isolated Visibility/Depth Export (Phase 1, Unified Single Source / Dual Atomic Strategy)
            if (useSWCullHere)
            {
                const BOOL     useAtomic64 = (activeStrategy == NVGRasterAtomicStrategy::FORCE_ATOMIC64_SINGLE_PASS);
                ComputeKernel* kSW         = useAtomic64 ? m_kMicropolyRasterAtomic64 : m_kMicropolyRasterAtomic32;
                const char*    swProp      = "NVGMicropolyParams";
                ctx.lastUsedSplitAtomic32  = !useAtomic64;

                GPUBuffer* bonePalBuf = scene->GetGPUSkinningPaletteBuffer();
                const UINT32 bonePalCnt = (bonePalBuf && bonePalBuf->GetNativeHandle() != 0)
                    ? scene->GetGPUSkinningPaletteCount()
                    : 0u;
                if (!bonePalBuf || bonePalBuf->GetNativeHandle() == 0)
                {
                    bonePalBuf = m_dummyBonePaletteBuf;
                }

                kSW->SetUInt(swProp, "Config", "viewportWidth", vpW);
                kSW->SetUInt(swProp, "Config", "viewportHeight", vpH);
                kSW->SetUInt(swProp, "Config", "maxVisibleCapacity", outCompactedVis->GetElementCount());
                kSW->SetUInt(swProp, "Config", "rawVisibleCounterIdx", 3u);
                kSW->SetUInt(swProp, "Config", "swClusterCap", ctx.swClusterList->GetElementCount());
                kSW->SetUInt(swProp, "Config", "maskWordCount", swMaskWordCnt);
                kSW->SetUInt(swProp, "Config", "pageWordCount", kNVGPageWordCount);
                kSW->SetUInt(swProp, "Config", "clusterMetaCount", clusterCount);
                kSW->SetUInt(swProp, "Config", "materialTableCount", matTabCount);
                kSW->SetUInt(swProp, "Config", "bonePaletteCount", bonePalCnt);
                kSW->SetUInt(swProp, "Config", "skinBoundsCount", skinBoundsCnt);
                kSW->SetUInt(swProp, "Config", "isOrthographic", isOrtho ? 1u : 0u);
                kSW->SetUInt(swProp, "Config", "isReversedZ", isRevZ ? 1u : 0u);
                kSW->SetUInt(swProp, "Config", "forceAllEligible", userConsumesVisBuffer ? 1u : 0u);
                kSW->SetUInt(swProp, "Config", "maxSWTriangleSpanPx", userConsumesVisBuffer ? 2048u : 64u);
                kSW->SetFloat(swProp, "Config", "cameraNearPlane", (std::max)(ioFrameView.camera.nearPlane, 0.0001f));
                kSW->SetFloat(swProp, "Config", "lodProjScale", lodProjScale);
                kSW->SetFloat(swProp, "Config", "swMaxEdgePixels", m_config.swRasterMaxEdgePixels);
                kSW->SetVec3(swProp, "Config", "cameraPosition", ioFrameView.camera.position);
                kSW->SetMat4(swProp, "Config", "vpMatrix", ioFrameView.camera.viewProjectionMatrix);

                kSW->SetGPUBuffer(swProp, "InPagePool", m_residencyManager.GetPhysicalPagePoolBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InCompactedVisibleClusters", outCompactedVis, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InBucketRanges", outBucketRanges, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InClusterMeta", m_residencyManager.GetClusterMetaBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InBonePalette", bonePalBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                kSW->SetGPUBuffer(swProp, "InSkinnedBounds", skinBoundsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                if (useAtomic64)
                {
                    kSW->SetGPUBuffer(swProp, "OutVisBuffer64", ctx.visBuffer64, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                }
                else
                {
                    kSW->SetGPUBuffer(swProp, "OutAtomicDepthKey32", ctx.swAtomicDepthKey32, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                    kSW->SetGPUBuffer(swProp, "OutAtomicPayload32", ctx.swAtomicPayload32, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                }
                kSW->SetGPUBuffer(swProp, "OutSWClusterList", ctx.swClusterList, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                kSW->SetGPUBuffer(swProp, "OutSWScheduledMask", ctx.swScheduledMask, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                kSW->SetGPUBuffer(swProp, "OutWinningClusterMask", ctx.swWinningClusterMask, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                kSW->SetGPUBuffer(swProp, "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                kSW->SetGPUBuffer(swProp, "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);

                // Mode 0: Clear VisBuffer64 / AtomicDepthKey32+Payload32 + Scheduled/Winning Masks + Counter[20]
                kSW->SetUInt(swProp, "Config", "dispatchMode", 0u);
                kSW->SetIndirectArgsBuffer(nullptr, 0u);
                const UINT32 swClearThreads = (std::min)((std::max)(vpW * vpH, swMaskWordCnt), 4194240u);
                kSW->SetDispatchThreadCount((std::max)(swClearThreads, 64u), 1u, 1u);
                {
                    FramePassItem swClr{};
                    swClr.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(kSW, ioFrameView, swClr.computeDispatch))
                        return false;
                    swClr.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                    items.push_back(std::move(swClr));
                }

                // Mode 1: Classify SW-Eligible Micro-Clusters (indirect over rawVisibleCounterIdx = 3)
                AppendPrepareDispatchArgs(ctx, 3u, 56u, 64u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_SW_RASTER);
                kSW->SetUInt(swProp, "Config", "dispatchMode", 1u);
                kSW->SetIndirectArgsBuffer(ctx.dispatchArgs, 56u * sizeof(UINT32));
                {
                    FramePassItem swCls{};
                    swCls.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(kSW, ioFrameView, swCls.computeDispatch))
                        return false;
                    swCls.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                    items.push_back(std::move(swCls));
                }

                // Mode 2 (+ Mode 3 on ATOMIC32_TWO_PASS): 1 Threadgroup (64 threads) per SW Cluster
                AppendPrepareDispatchArgs(ctx, 20u, 52u, 1u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_SW_RASTER);
                kSW->SetUInt(swProp, "Config", "dispatchMode", 2u);
                kSW->SetIndirectArgsBuffer(ctx.dispatchArgs, 52u * sizeof(UINT32));
                {
                    FramePassItem swRst{};
                    swRst.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(kSW, ioFrameView, swRst.computeDispatch))
                        return false;
                    swRst.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER_PASS1;
                    items.push_back(std::move(swRst));
                }
                if (!useAtomic64)
                {
                    kSW->SetUInt(swProp, "Config", "dispatchMode", 3u);
                    kSW->SetIndirectArgsBuffer(ctx.dispatchArgs, 52u * sizeof(UINT32));
                    FramePassItem swRst2{};
                    swRst2.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(kSW, ioFrameView, swRst2.computeDispatch))
                        return false;
                    swRst2.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER_PASS2;
                    items.push_back(std::move(swRst2));
                }

                // Depth & Visibility Export -> OutVisTexture (RG32_UINT), OutSWDepth (R32_FLOAT), OutWinningClusterMask
                m_kDepthExport->SetUInt("NVGDepthExportParams", "Config", "viewportWidth", vpW);
                m_kDepthExport->SetUInt("NVGDepthExportParams", "Config", "viewportHeight", vpH);
                m_kDepthExport->SetUInt("NVGDepthExportParams", "Config", "isReversedZ", isRevZ ? 1u : 0u);
                m_kDepthExport->SetUInt("NVGDepthExportParams", "Config", "useSplitAtomic32", useAtomic64 ? 0u : 1u);
                m_kDepthExport->SetUInt("NVGDepthExportParams", "Config", "maskWordCount", swMaskWordCnt);
                m_kDepthExport->SetGPUBuffer("NVGDepthExportParams", "InVisBuffer64", ctx.visBuffer64, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kDepthExport->SetGPUBuffer("NVGDepthExportParams", "InAtomicDepthKey32", ctx.swAtomicDepthKey32, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kDepthExport->SetGPUBuffer("NVGDepthExportParams", "InAtomicPayload32", ctx.swAtomicPayload32, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kDepthExport->SetRWTexture("NVGDepthExportParams", "OutVisTexture", ctx.swVisTexture, GPUResourceAccess::UNORDERED_ACCESS_WRITE, 0u);
                m_kDepthExport->SetRWTexture("NVGDepthExportParams", "OutSWDepth", ctx.swDepthTexture, GPUResourceAccess::UNORDERED_ACCESS_WRITE, 0u);
                m_kDepthExport->SetGPUBuffer("NVGDepthExportParams", "OutWinningClusterMask", ctx.swWinningClusterMask, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                m_kDepthExport->SetIndirectArgsBuffer(nullptr, 0u);
                m_kDepthExport->SetDispatchThreadCount(vpW, vpH, 1u);
                {
                    FramePassItem expItem{};
                    expItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kDepthExport, ioFrameView, expItem.computeDispatch))
                        return false;
                    expItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                    items.push_back(std::move(expItem));
                }
                m_stats.swRasterDispatches += useAtomic64 ? 4u : 5u;
            }

            // Indirect Cluster Pull & Compact IA Decode + 96B/144B Promoted Indirect Draw Generation
            if (hasAnyPathB)
            {
                // Mode 3: Claim unique decode jobs over [indBase .. indEnd) (counter 18 = indirectSpanCount)
                AppendPrepareDispatchArgs(ctx, 18u, 44u, 128u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_DRAW_GEN);
                m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 3u);
                m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "enableSWClusterCull", useSWCullHere ? 1u : 0u);
                m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "swMaskWordCount", swMaskWordCnt);
                m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InSWScheduledMask", schedMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InWinningClusterMask", winMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 44u * sizeof(UINT32));
                {
                    FramePassItem clmItem{};
                    clmItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, clmItem.computeDispatch))
                        return false;
                    clmItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                    items.push_back(std::move(clmItem));
                }

                // Mode 4: Decompress 1 threadgroup (128 threads) per unique cluster
                AppendPrepareDispatchArgs(ctx, 5u, 16u, 1u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_DRAW_GEN);
                m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 4u);
                m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 16u * sizeof(UINT32));
                {
                    FramePassItem decItem{};
                    decItem.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, ioFrameView, decItem.computeDispatch))
                        return false;
                    decItem.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                    items.push_back(std::move(decItem));
                }

                // IndirectDraw Mode 0: Reset bucket command counts & CBV/Instance pool cursors
                GPUBuffer* outIndCmds     = (isPhase2 && ctx.p2IndirectCommands)  ? ctx.p2IndirectCommands  : ctx.indirectDrawCommands;
                GPUBuffer* outPerDrawCBV  = (isPhase2 && ctx.p2PerDrawCBVWords)   ? ctx.p2PerDrawCBVWords   : ctx.perDrawCBVWords;
                GPUBuffer* outPerBatchIn  = (isPhase2 && ctx.p2PerBatchInstWords) ? ctx.p2PerBatchInstWords : ctx.perBatchInstanceWords;
                GPUBuffer* outCmdCounts   = (isPhase2 && ctx.p2BucketCmdCounts)   ? ctx.p2BucketCmdCounts   : ctx.bucketCommandCounts;

                const UINT64 vbAddr   = m_backend->GetGPUBufferGPUVirtualAddress(outDecVerts->GetNativeHandle());
                const UINT64 ibAddr   = m_backend->GetGPUBufferGPUVirtualAddress(outDecIdxs->GetNativeHandle());
                const UINT64 cbvAddr  = m_backend->GetGPUBufferGPUVirtualAddress(outPerDrawCBV->GetNativeHandle());
                const UINT64 instAddr = m_backend->GetGPUBufferGPUVirtualAddress(outPerBatchIn->GetNativeHandle());
                const NativeHandle boneHandle = ioFrameView.gpuSkinningPaletteBuffer;
                const UINT64 boneAddr = (boneHandle != 0) ? m_backend->GetGPUBufferGPUVirtualAddress(boneHandle) : 0ull;

                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "dispatchMode", 0u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bucketCount", bucketCount);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxVisibleCapacity", ctx.rawVisibleClusters->GetElementCount());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "rawVisibleCounterIdx", 3u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxCommandsPerBucket", ctx.dynCameraCmdsPerBucket);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "drawGroupBindingCount", static_cast<UINT32>(m_cameraBindingDescs.size()));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedTableCapacity", ctx.decodedClusterTable->GetElementCount());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "isShadowPass", 0u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "shadowCommitRecordCap", 0u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bucketWordStride", 36u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "enableSWClusterCull", useSWCullHere ? 1u : 0u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "swMaskWordCount", swMaskWordCnt);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBAddressLo", static_cast<UINT32>(vbAddr & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBAddressHi", static_cast<UINT32>((vbAddr >> 32u) & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBSizeBytes", outDecVerts->GetByteSize());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBAddressLo", static_cast<UINT32>(ibAddr & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBAddressHi", static_cast<UINT32>((ibAddr >> 32u) & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBSizeBytes", outDecIdxs->GetByteSize());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "cbvPoolBaseLo", static_cast<UINT32>(cbvAddr & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "cbvPoolBaseHi", static_cast<UINT32>((cbvAddr >> 32u) & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "instancePoolBaseLo", static_cast<UINT32>(instAddr & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "instancePoolBaseHi", static_cast<UINT32>((instAddr >> 32u) & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bonePaletteBaseLo", static_cast<UINT32>(boneAddr & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bonePaletteBaseHi", static_cast<UINT32>((boneAddr >> 32u) & 0xFFFFFFFFull));
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxPerDrawCBVWords", outPerDrawCBV->GetElementCount());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxPerBatchInstanceWords", outPerBatchIn->GetElementCount());
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "shadowPageViewCount", 0u);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "skinningRevision", 1u);
                m_kIndirectDraw->SetVec3("NVGIndirectDrawParams", "Config", "cameraPosition", ioFrameView.camera.position);
                m_kIndirectDraw->SetFloat("NVGIndirectDrawParams", "Config", "cameraNearPlane", ioFrameView.camera.nearPlane);
                m_kIndirectDraw->SetVec3("NVGIndirectDrawParams", "Config", "cameraDirection", ioFrameView.camera.direction);
                m_kIndirectDraw->SetFloat("NVGIndirectDrawParams", "Config", "cameraFarPlane", ioFrameView.camera.farPlane);
                m_kIndirectDraw->SetFloat("NVGIndirectDrawParams", "Config", "cameraFov", ioFrameView.camera.fov);
                m_kIndirectDraw->SetFloat("NVGIndirectDrawParams", "Config", "cameraAspect", ioFrameView.camera.aspectRatio);
                Mat4 invViewCam = ioFrameView.camera.viewMatrix;           invViewCam.Inverse();
                Mat4 invProjCam = ioFrameView.camera.projectionMatrix;     invProjCam.Inverse();
                Mat4 invVPCam   = ioFrameView.camera.viewProjectionMatrix; invVPCam.Inverse();
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "viewMatrix", ioFrameView.camera.viewMatrix);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "invViewMatrix", invViewCam);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "projMatrix", ioFrameView.camera.projectionMatrix);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "invProjMatrix", invProjCam);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "vpMatrix", ioFrameView.camera.viewProjectionMatrix);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "invVPMatrix", invVPCam);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "prevViewMatrix",
                    ctx.hasValidPrevHZB ? ctx.prevViewMatrix : ioFrameView.camera.viewMatrix);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "prevProjMatrix",
                    ctx.hasValidPrevHZB ? ctx.prevProjMatrix : ioFrameView.camera.projectionMatrix);
                m_kIndirectDraw->SetMat4("NVGIndirectDrawParams", "Config", "prevVPMatrix",
                    ctx.hasValidPrevHZB ? ctx.prevViewProj : ioFrameView.camera.viewProjectionMatrix);

                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InCompactedVisibleClusters", outCompactedVis, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InBucketRanges", outBucketRanges, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDecodedClusterTable", ctx.decodedClusterTable, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupBindings", m_cameraBindingsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupFieldPatches", m_cameraPatchesBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupTemplateWords", m_cameraTemplatesBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InShadowPageViews", ctx.shadowPageViews, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InSWScheduledMask", schedMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InWinningClusterMask", winMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutIndirectDrawCommands", outIndCmds, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutPerDrawCBVWords", outPerDrawCBV, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutPerBatchInstanceWords", outPerBatchIn, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutBucketCommandCounts", outCmdCounts, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InOutPageCommitRecords", m_dummyCommitRecordsBuf, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
                m_kIndirectDraw->SetIndirectArgsBuffer(nullptr, 0u);
                m_kIndirectDraw->SetDispatchThreadCount((std::max)(bucketCount, 64u), 1u, 1u);
                {
                    FramePassItem rstInd{};
                    rstInd.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kIndirectDraw, ioFrameView, rstInd.computeDispatch))
                        return false;
                    rstInd.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                    items.push_back(std::move(rstInd));
                }

                // IndirectDraw Mode 1: Emit 96B / 144B promoted ExecuteIndirect commands over [indBase .. indEnd) (counter 18)
                AppendPrepareDispatchArgs(ctx, 18u, 20u, 64u, kNVGInvalidIndex, items, ComputeDispatchProfileGroup::NVG_DRAW_GEN);
                m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "dispatchMode", 1u);
                m_kIndirectDraw->SetIndirectArgsBuffer(ctx.dispatchArgs, 20u * sizeof(UINT32));
                {
                    FramePassItem genInd{};
                    genInd.type = FramePassItemType::COMPUTE_DISPATCH;
                    if (!dummyPipe.BuildComputeDispatchCommand(m_kIndirectDraw, ioFrameView, genInd.computeDispatch))
                        return false;
                    genInd.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_DRAW_GEN;
                    items.push_back(std::move(genInd));
                }
            }

            return true;
        };

        const UINT64 residencyRev = m_residencyManager.GetResidencyRevision();
        const UINT64 phase1Rev    = (m_currentFrameIndex << 16u) ^ (residencyRev << 4u) ^ 1ull;
        const UINT64 phase2Rev    = (m_currentFrameIndex << 16u) ^ (residencyRev << 4u) ^ 2ull;

        auto emitBucketDraws = [&](BOOL transparentOnly, BOOL depthPrepassDraws, BOOL reuseDepthEqual, BOOL usePhase2Outputs,
                                   UINT64 passRev, std::vector<ComputeDispatchCommand>& ioMatDispatches, std::vector<DrawCommand>& outCmds) {
            GPUBuffer* brBuf      = usePhase2Outputs ? ctx.p2BucketRanges       : ctx.bucketRanges;
            GPUBuffer* compVisBuf = usePhase2Outputs ? ctx.p2CompactedVisible   : ctx.compactedVisibleClusters;
            GPUBuffer* pathABuf   = usePhase2Outputs ? ctx.p2PathAIndirectArgs  : ctx.pathAIndirectArgs;
            GPUBuffer* indCmdBuf  = (usePhase2Outputs && ctx.p2IndirectCommands)  ? ctx.p2IndirectCommands  : ctx.indirectDrawCommands;
            GPUBuffer* cmdCntBuf  = (usePhase2Outputs && ctx.p2BucketCmdCounts)   ? ctx.p2BucketCmdCounts   : ctx.bucketCommandCounts;
            GPUBuffer* decVertBuf = (usePhase2Outputs && ctx.p2DecodedVertexPool) ? ctx.p2DecodedVertexPool : ctx.decodedVertexPool;
            GPUBuffer* decIdxBuf  = (usePhase2Outputs && ctx.p2DecodedIndexPool)  ? ctx.p2DecodedIndexPool  : ctx.decodedIndexPool;
            GPUBuffer* cbvWordBuf = (usePhase2Outputs && ctx.p2PerDrawCBVWords)   ? ctx.p2PerDrawCBVWords   : ctx.perDrawCBVWords;
            GPUBuffer* instWrdBuf = (usePhase2Outputs && ctx.p2PerBatchInstWords) ? ctx.p2PerBatchInstWords : ctx.perBatchInstanceWords;

            for (const auto& b : m_cameraBuckets)
            {
                if (b.isTransparent != transparentOnly) continue;
                if (depthPrepassDraws)
                {
                    if (!b.material || !b.material->GetDepthState().test || !b.material->GetDepthState().write)
                        continue;
                    if (b.hasRejectedCustomDepth)
                        continue; // Zero-Handholding: broken custom PassType::DEPTH ShaderSet produces 0 depth draws
                    if (!b.isDepthCompanion && b.depthCompanionBucketIndex != kNVGInvalidIndex)
                        continue; // Custom PassType::DEPTH companion bucket handles depth prepass for this material
                }
                else if (b.isDepthCompanion)
                {
                    continue; // Companion depth buckets execute exclusively in Depth Prepass
                }

                const BOOL applyDepthOv = reuseDepthEqual && !transparentOnly && b.material &&
                                          b.material->GetDepthState().test && b.material->GetDepthState().write;
                DepthCompare cmpOv = b.material ? b.material->GetDepthState().compare : DepthCompare::LESS_EQUAL;
                if (cmpOv == DepthCompare::LESS)
                {
                    cmpOv = DepthCompare::LESS_EQUAL;
                }
                else if (cmpOv == DepthCompare::GREATER)
                {
                    cmpOv = DepthCompare::GREATER_EQUAL;
                }

                DrawCommand cmd{};
                BOOL ok = false;
                if (b.isVertexPull)
                {
                    ok = BuildVertexPullDrawCommand(
                        b, ioFrameView, scene, gpuSceneBuffer,
                        brBuf, compVisBuf, pathABuf,
                        nullptr, passRev, false, cmpOv, applyDepthOv, ioMatDispatches, cmd);
                }
                else
                {
                    ok = BuildDecodedIADrawCommand(
                        b, ioFrameView, scene, gpuSceneBuffer,
                        brBuf, compVisBuf, nullptr,
                        indCmdBuf, cmdCntBuf,
                        decVertBuf, decIdxBuf,
                        cbvWordBuf, instWrdBuf,
                        ctx.dynCameraCmdsPerBucket,
                        passRev, false, cmpOv, applyDepthOv, ioMatDispatches, cmd);
                }
                if (ok)
                {
                    if (depthPrepassDraws && b.material && !b.material->IsAlphaTested())
                    {
                        BOOL psPromoted = false;
                        if (cmd.indirectRootKey.enabled)
                        {
                            for (UINT8 r = 0u; r < cmd.indirectRootKey.cbvCount; ++r)
                            {
                                if (cmd.indirectRootKey.cbvStage[r] == ShaderStage::PIXEL) { psPromoted = true; break; }
                            }
                            if (cmd.indirectRootKey.hasPromotedSRV && cmd.indirectRootKey.srvStage == ShaderStage::PIXEL)
                            {
                                psPromoted = true;
                            }
                        }
                        if (!psPromoted)
                        {
                            cmd.pixelShader = 0;
                            auto isPixel = [](const auto& x) { return x.stage == ShaderStage::PIXEL; };
                            cmd.constantBuffers.erase(std::remove_if(cmd.constantBuffers.begin(), cmd.constantBuffers.end(), isPixel), cmd.constantBuffers.end());
                            cmd.textures.erase(std::remove_if(cmd.textures.begin(), cmd.textures.end(), isPixel), cmd.textures.end());
                            cmd.samplers.erase(std::remove_if(cmd.samplers.begin(), cmd.samplers.end(), isPixel), cmd.samplers.end());
                            cmd.structuredBuffers.erase(std::remove_if(cmd.structuredBuffers.begin(), cmd.structuredBuffers.end(), isPixel), cmd.structuredBuffers.end());
                        }
                    }
                    if (transparentOnly && b.material && scene)
                    {
                        const Vec3 camFwd    = ioFrameView.camera.direction;
                        const BOOL hasFwdDir = (camFwd.LengthSq() > 1e-6f);
                        const Vec3 fwdNorm   = hasFwdDir ? camFwd.Normalized() : Vec3(0.0f, 0.0f, 1.0f);
                        FLOAT32    sumDepth  = 0.0f;
                        UINT32     matchCnt  = 0u;
                        for (const RenderObject* ro : scene->GetRenderObjects())
                        {
                            if (!ro || !ro->HasFlag(RenderObjectFlags::VISIBLE)) continue;
                            const Mesh* rm = ro->GetMesh();
                            if (!rm || !rm->IsVirtualized()) continue;
                            if ((camera->GetCullingMask() & ro->GetLayerMask()) == 0u) continue;
                            BOOL usesMat = false;
                            const UINT32 slotCnt = (std::max)(1u, ro->GetMaterialSlotCount());
                            for (UINT32 ms = 0u; ms < slotCnt; ++ms)
                            {
                                if (ro->GetMaterial(ms) == b.material) { usesMat = true; break; }
                            }
                            if (!usesMat) continue;
                            const Vec3 centerP = ro->GetWorldAABB().IsValid()
                                ? ro->GetWorldAABB().Center()
                                : ro->GetWorldMatrix().GetTranslation();
                            const Vec3 delta = centerP - camPos;
                            sumDepth += hasFwdDir ? delta.Dot(fwdNorm) : delta.Length();
                            ++matchCnt;
                        }
                        if (matchCnt > 0u)
                        {
                            cmd.depthKey = sumDepth / static_cast<FLOAT32>(matchCnt);
                        }
                    }
                    outCmds.push_back(std::move(cmd));
                }
            }
        };

        // Ensure all standard geometry draws are in interleavedPassItems so NVG interleaves cleanly in exact order
        if (!ioFrameView.geometryPass.opaqueCommands.empty())
        {
            FramePassItem stdOpaque{};
            stdOpaque.type         = FramePassItemType::GRAPHICS_DRAWS;
            stdOpaque.drawCommands = std::move(ioFrameView.geometryPass.opaqueCommands);
            ioFrameView.geometryPass.opaqueCommands.clear();
            auto existingDpIt = std::find_if(
                ioFrameView.interleavedPassItems.begin(),
                ioFrameView.interleavedPassItems.end(),
                [](const FramePassItem& it) { return it.type == FramePassItemType::DEPTH_PREPASS; });
            if (existingDpIt != ioFrameView.interleavedPassItems.end())
            {
                ioFrameView.interleavedPassItems.insert(existingDpIt + 1, std::move(stdOpaque));
            }
            else
            {
                ioFrameView.interleavedPassItems.insert(ioFrameView.interleavedPassItems.begin(), std::move(stdOpaque));
            }
        }
        std::vector<DrawCommand> stdTransparentDraws = std::move(ioFrameView.geometryPass.transparentCommands);
        ioFrameView.geometryPass.transparentCommands.clear();

        // Build Phase-1 NVG compute burst (ResidencyMask + Cull + BVH + LOD + Compaction + Decode + Indirect)
        std::vector<FramePassItem> nvgPhase1Compute;
        if (!AppendResidencyMaskDispatch(nvgPhase1Compute))
            return false;
        if (!appendCullCompactAndIndirect(false, nvgPhase1Compute))
            return false;

        std::vector<ComputeDispatchCommand> phase1MatDispatches;
        std::vector<DrawCommand> nvgOpaqueDraws;
        std::vector<DrawCommand> nvgTransparentDraws;
        emitBucketDraws(false, false, isDepthPre, false, phase1Rev, phase1MatDispatches, nvgOpaqueDraws);
        emitBucketDraws(true,  false, false,      false, phase1Rev, phase1MatDispatches, nvgTransparentDraws);

        stats.drawCalls += static_cast<UINT32>(nvgOpaqueDraws.size() + nvgTransparentDraws.size());

        Texture* sceneDepthTex = camera->GetSceneDepthTexture();

        BOOL needPhase2Pass = false;
        if (m_config.enableHZBOcclusion && m_config.enableTwoPhaseDisocclusion && effHasPrevHZB && sceneDepthTex)
        {
            const BOOL cameraStatic    = (std::memcmp(&ctx.prevViewProj, &ioFrameView.camera.viewProjectionMatrix, sizeof(Mat4)) == 0);
            const BOOL sceneStatic     = (ctx.lastSceneMutationEpoch == ioFrameView.gpuSceneMutationEpoch);
            const BOOL residencyStatic = (ctx.lastResidencyRevision == residencyRev);
            NVGTelemetryGPUData lastTel{};
            if (ctx.hasLastTelemetry)
            {
                std::memcpy(&lastTel, ctx.lastTelemetryWords, sizeof(lastTel));
            }
            needPhase2Pass = !cameraStatic || !sceneStatic || !residencyStatic ||
                             m_sceneHasAnimatedDeformation || !ctx.hasLastTelemetry ||
                             (lastTel.disocclusionRecovered > 0u);
        }

        if (isDepthPre)
        {
            std::vector<DrawCommand> nvgDepthDraws;
            emitBucketDraws(false, true, false, false, phase1Rev, phase1MatDispatches, nvgDepthDraws);
            stats.depthPrepassDrawCalls += static_cast<UINT32>(nvgDepthDraws.size());

            // Append any reflection GPU materialization dispatches right after NVG compaction & before Depth Prepass
            for (auto& md : phase1MatDispatches)
            {
                FramePassItem matItem{};
                matItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                matItem.computeDispatch = std::move(md);
                nvgPhase1Compute.push_back(std::move(matItem));
            }

            auto depthIt = std::find_if(
                ioFrameView.interleavedPassItems.begin(),
                ioFrameView.interleavedPassItems.end(),
                [](const FramePassItem& it) { return it.type == FramePassItemType::DEPTH_PREPASS; });

            if (depthIt == ioFrameView.interleavedPassItems.end())
            {
                FramePassItem dp{};
                dp.type         = FramePassItemType::DEPTH_PREPASS;
                dp.drawCommands = std::move(nvgDepthDraws);
                ioFrameView.interleavedPassItems.insert(ioFrameView.interleavedPassItems.begin(), std::move(dp));
                depthIt = ioFrameView.interleavedPassItems.begin();
            }
            else
            {
                for (auto& c : nvgDepthDraws) depthIt->drawCommands.push_back(std::move(c));
            }

            // Insert Phase-1 compute right before DEPTH_PREPASS
            const SIZE_T dpIdx = static_cast<SIZE_T>(depthIt - ioFrameView.interleavedPassItems.begin());
            ioFrameView.interleavedPassItems.insert(
                ioFrameView.interleavedPassItems.begin() + static_cast<std::ptrdiff_t>(dpIdx),
                std::make_move_iterator(nvgPhase1Compute.begin()),
                std::make_move_iterator(nvgPhase1Compute.end()));

            // Immediately after Phase-1 DEPTH_PREPASS (and BEFORE POST_DEPTH / NVS Feedback!):
            //   1. Build HZB from Phase-1 SceneDepth (STANDARD + NVG Phase-1)
            //   2. Run Phase-2 Disocclusion Recovery into dedicated ctx.p2* buffers
            //   3. Execute Phase-2 DEPTH_PREPASS draws + HZB Refresh so POST_DEPTH & NVS Feedback see 100% disoccluded depth!
            const SIZE_T afterDpIdx = dpIdx + nvgPhase1Compute.size() + 1u;
            std::vector<FramePassItem> postDpItems;
            std::vector<DrawCommand> p2OpaqueDraws;
            std::vector<DrawCommand> p2TransparentDraws;
            if (sceneDepthTex)
            {
                Texture* swDepthForHzb = runSWRasterPhase1 ? ctx.swDepthTexture : nullptr;
                AppendHZBBuildDispatches(ctx, sceneDepthTex, vpX, vpY, vpW, vpH, isRevZ, postDpItems, swDepthForHzb);
                if (needPhase2Pass && appendCullCompactAndIndirect(true, postDpItems))
                {
                    std::vector<ComputeDispatchCommand> phase2MatDispatches;
                    std::vector<DrawCommand> p2DepthDraws;
                    emitBucketDraws(false, true,  false, true, phase2Rev, phase2MatDispatches, p2DepthDraws);
                    emitBucketDraws(false, false, true,  true, phase2Rev, phase2MatDispatches, p2OpaqueDraws);
                    emitBucketDraws(true,  false, false, true, phase2Rev, phase2MatDispatches, p2TransparentDraws);
                    for (auto& md : phase2MatDispatches)
                    {
                        FramePassItem matItem{};
                        matItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                        matItem.computeDispatch = std::move(md);
                        postDpItems.push_back(std::move(matItem));
                    }
                    if (!p2DepthDraws.empty())
                    {
                        FramePassItem p2Dp{};
                        p2Dp.type         = FramePassItemType::DEPTH_PREPASS;
                        p2Dp.drawCommands = std::move(p2DepthDraws);
                        postDpItems.push_back(std::move(p2Dp));
                        AppendHZBBuildDispatches(ctx, sceneDepthTex, vpX, vpY, vpW, vpH, isRevZ, postDpItems, swDepthForHzb);
                    }
                }
                ioFrameView.interleavedPassItems.insert(
                    ioFrameView.interleavedPassItems.begin() + static_cast<std::ptrdiff_t>(afterDpIdx),
                    std::make_move_iterator(postDpItems.begin()),
                    std::make_move_iterator(postDpItems.end()));
            }

            // Insert NVG Phase-1 + Phase-2 Main Opaque draws into Main Opaque GRAPHICS_DRAWS (Read-Only DSV LESS_EQUAL)
            if (!nvgOpaqueDraws.empty() || !p2OpaqueDraws.empty())
            {
                auto opqIt = std::find_if(
                    ioFrameView.interleavedPassItems.begin() + static_cast<std::ptrdiff_t>(afterDpIdx + postDpItems.size()),
                    ioFrameView.interleavedPassItems.end(),
                    [](const FramePassItem& it) { return it.type == FramePassItemType::GRAPHICS_DRAWS; });
                if (opqIt != ioFrameView.interleavedPassItems.end())
                {
                    for (auto& c : nvgOpaqueDraws) opqIt->drawCommands.push_back(std::move(c));
                    for (auto& c : p2OpaqueDraws)  opqIt->drawCommands.push_back(std::move(c));
                }
                else
                {
                    FramePassItem gi{};
                    gi.type         = FramePassItemType::GRAPHICS_DRAWS;
                    gi.drawCommands = std::move(nvgOpaqueDraws);
                    for (auto& c : p2OpaqueDraws) gi.drawCommands.push_back(std::move(c));
                    ioFrameView.interleavedPassItems.push_back(std::move(gi));
                }
            }
            for (auto& c : p2TransparentDraws) nvgTransparentDraws.push_back(std::move(c));
        }
        else
        {
            // DEPTH PREPASS NOT REQUIRED:
            //   1. NVG Phase-1 Compute -> runs BEFORE Main Opaque GRAPHICS_DRAWS
            //   2. Standard + NVG Phase-1 Opaque GRAPHICS_DRAWS execute together into SceneDepth
            //   3. Build HZB from complete (STANDARD + NVG) SceneDepth
            //   4. If two-phase disocclusion & had prev HZB:
            //        NVG Phase-2 Disocclusion Compute (ctx.p2*) -> Phase-2 Opaque GRAPHICS_DRAWS -> HZB Refresh
            //   5. Any remaining POST_OPAQUE / Transparent passes (Phase-1 ctx.* remains 100% intact for Transparent!)
            std::vector<FramePassItem> preOpaqueItems;
            for (auto& it : nvgPhase1Compute) preOpaqueItems.push_back(std::move(it));
            for (auto& md : phase1MatDispatches)
            {
                FramePassItem matItem{};
                matItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                matItem.computeDispatch = std::move(md);
                preOpaqueItems.push_back(std::move(matItem));
            }

            // Ensure a Main Opaque GRAPHICS_DRAWS slot exists at the start of interleavedPassItems (before POST_OPAQUE / Transparent)
            if (ioFrameView.interleavedPassItems.empty() ||
                ioFrameView.interleavedPassItems.front().type != FramePassItemType::GRAPHICS_DRAWS)
            {
                FramePassItem opqAnchor{};
                opqAnchor.type = FramePassItemType::GRAPHICS_DRAWS;
                ioFrameView.interleavedPassItems.insert(ioFrameView.interleavedPassItems.begin(), std::move(opqAnchor));
            }
            for (auto& c : nvgOpaqueDraws)
            {
                ioFrameView.interleavedPassItems.front().drawCommands.push_back(std::move(c));
            }

            // Prepend Phase-1 compute right before Main Opaque GRAPHICS_DRAWS
            ioFrameView.interleavedPassItems.insert(
                ioFrameView.interleavedPassItems.begin(),
                std::make_move_iterator(preOpaqueItems.begin()),
                std::make_move_iterator(preOpaqueItems.end()));

            // Build HZB + Phase-2 Disocclusion immediately AFTER Main Opaque GRAPHICS_DRAWS
            const SIZE_T afterMainOpaqueIdx = preOpaqueItems.size() + 1u;
            std::vector<FramePassItem> postOpaqueHzbItems;
            Texture* swDepthForHzb = runSWRasterPhase1 ? ctx.swDepthTexture : nullptr;
            if (sceneDepthTex)
            {
                AppendHZBBuildDispatches(ctx, sceneDepthTex, vpX, vpY, vpW, vpH, isRevZ, postOpaqueHzbItems, swDepthForHzb);
            }

            if (needPhase2Pass && appendCullCompactAndIndirect(true, postOpaqueHzbItems))
            {
                std::vector<ComputeDispatchCommand> phase2MatDispatches;
                std::vector<DrawCommand> p2Draws;
                std::vector<DrawCommand> p2TransDraws;
                emitBucketDraws(false, false, false, true, phase2Rev, phase2MatDispatches, p2Draws);
                emitBucketDraws(true,  false, false, true, phase2Rev, phase2MatDispatches, p2TransDraws);
                for (auto& md : phase2MatDispatches)
                {
                    FramePassItem matItem{};
                    matItem.type            = FramePassItemType::COMPUTE_DISPATCH;
                    matItem.computeDispatch = std::move(md);
                    postOpaqueHzbItems.push_back(std::move(matItem));
                }
                if (!p2Draws.empty())
                {
                    FramePassItem p2Gfx{};
                    p2Gfx.type         = FramePassItemType::GRAPHICS_DRAWS;
                    p2Gfx.drawCommands = std::move(p2Draws);
                    postOpaqueHzbItems.push_back(std::move(p2Gfx));
                    AppendHZBBuildDispatches(ctx, sceneDepthTex, vpX, vpY, vpW, vpH, isRevZ, postOpaqueHzbItems, swDepthForHzb);
                }
                for (auto& c : p2TransDraws) nvgTransparentDraws.push_back(std::move(c));
            }

            if (!postOpaqueHzbItems.empty())
            {
                ioFrameView.interleavedPassItems.insert(
                    ioFrameView.interleavedPassItems.begin() + static_cast<std::ptrdiff_t>(afterMainOpaqueIdx),
                    std::make_move_iterator(postOpaqueHzbItems.begin()),
                    std::make_move_iterator(postOpaqueHzbItems.end()));
            }
        }

        // Merge & sort Standard + NVG Transparent draws back-to-front after all opaque & post-opaque passes
        if (!stdTransparentDraws.empty() || !nvgTransparentDraws.empty())
        {
            std::vector<DrawCommand> mergedTransparent;
            mergedTransparent.reserve(stdTransparentDraws.size() + nvgTransparentDraws.size());
            for (auto& c : stdTransparentDraws) mergedTransparent.push_back(std::move(c));
            for (auto& c : nvgTransparentDraws) mergedTransparent.push_back(std::move(c));
            std::stable_sort(mergedTransparent.begin(), mergedTransparent.end(),
                             [](const DrawCommand& a, const DrawCommand& b) {
                                 return a.depthKey > b.depthKey;
                             });
            FramePassItem trItem{};
            trItem.type         = FramePassItemType::GRAPHICS_DRAWS;
            trItem.drawCommands = std::move(mergedTransparent);
            ioFrameView.interleavedPassItems.push_back(std::move(trItem));
        }

        // Populate FrameView NVG fields for any diagnostic inspection
        ioFrameView.nvgPagePoolBuffer       = m_residencyManager.GetPhysicalPagePoolBuffer()->GetNativeHandle();
        ioFrameView.nvgPagePoolWordCount    = m_residencyManager.GetPhysicalPagePoolBuffer()->GetElementCount();
        ioFrameView.nvgVisibleClustersBuffer= ctx.compactedVisibleClusters->GetNativeHandle();
        ioFrameView.nvgVisibleClustersCount = ctx.compactedVisibleClusters->GetElementCount();
        ioFrameView.nvgBucketRangesBuffer   = ctx.bucketRanges->GetNativeHandle();
        ioFrameView.nvgBucketRangesCount    = ctx.bucketRanges->GetElementCount();
        ioFrameView.nvgPhysicalPoolCapacity = m_residencyManager.GetPhysicalPoolCapacity();
        ioFrameView.nvgResidencyRevision    = m_residencyManager.GetResidencyRevision();
        ioFrameView.nvgDebugMode            = static_cast<UINT32>(m_config.debugVisualizationMode);
        ioFrameView.nvgMaxScreenErrorPx     = ResolveEffectiveScreenErrorPixels();

        // Store camera & scene mutation state for next frame's Phase-1 HZB reprojection & Phase-2 skip check
        ctx.prevViewMatrix         = ioFrameView.camera.viewMatrix;
        ctx.prevProjMatrix         = ioFrameView.camera.projectionMatrix;
        ctx.prevViewProj           = ioFrameView.camera.viewProjectionMatrix;
        ctx.prevCameraPos          = camPos;
        ctx.lastSceneMutationEpoch = ioFrameView.gpuSceneMutationEpoch;
        ctx.lastResidencyRevision  = residencyRev;
        ctx.hasValidPrevHZB        = (sceneDepthTex != nullptr);

        // Queue async page-request readback ONLY after the final camera of the frame has accumulated its requests (NVG-017)
        if (isFinalCameraInFrame)
        {
            m_residencyManager.QueuePageRequestReadback();
        }
        if (!ctx.telemetryPending && ctx.telemetryBuffer != nullptr)
        {
            if (m_bufferManager->RequestGPUBufferReadback(ctx.telemetryBuffer, 0u, sizeof(NVGTelemetryGPUData)))
            {
                ctx.telemetryPending = true;
            }
        }
        return true;
    }

    BOOL CoreNVGManager::InjectNVSShadowPasses(
        Scene*                         scene,
        Camera*                        camera,
        const NVSShadowRasterPassDesc& rasterDesc,
        GPUBuffer*                     gpuSceneBuffer,
        UINT32                         gpuSceneSlotCount,
        GPUBuffer*                     projCounterBuf,
        GPUBuffer*                     activeRenderPageListBuf,
        GPUBuffer*                     pageRenderDataBuf,
        UINT32                         activePageCap,
        std::vector<FramePassItem>&    outPreRasterItems,
        LayeredDepthPassView&          ioShadowPassView,
        CoreRenderPipeline*            renderPipeline,
        const FrameView&               cameraFrameView,
        GPUBuffer*                     pageCommitRecordsBuf)
    {
        (void)renderPipeline;
        if (!m_initialized || !m_config.enabled || !scene || !camera || !gpuSceneBuffer || gpuSceneSlotCount == 0u)
            return false;
        if (m_shadowBuckets.empty() || m_residencyManager.GetRegisteredMeshCount() == 0u)
            return false;
        if (!projCounterBuf || !activeRenderPageListBuf || !pageRenderDataBuf)
            return false;
        if (!EnsureRuntimeKernelsLoaded())
            return false;

        m_residencyManager.SyncDirtyMetadataToGPU();

        auto& ctxPtr = m_cameraContexts[camera];
        if (!ctxPtr)
        {
            ctxPtr = std::make_unique<CameraNVGContext>();
            ctxPtr->camera = camera;
        }
        CameraNVGContext& ctx = *ctxPtr;
        ctx.usedThisFrame = true;
        // Note: do NOT set ctx.cameraPassRanThisFrame here! In camera renders, InjectCameraNVGPasses prepends
        // nvgPhase1Compute BEFORE shadow cull on the GPU timeline and owns page-request clearing & readback;
        // in standalone camera-less shadow renders, !hasCameraGeomPass below handles clearing & readback (NVG-017).
        const BOOL isStandaloneShadowOnly = !HasVirtualObjectsInScene(scene) || m_cameraBuckets.empty();

        Texture* depthForSize = camera->GetSceneDepthTexture();
        const UINT32 vpW = (cameraFrameView.camera.viewportWidth > 0.0f)
            ? static_cast<UINT32>(cameraFrameView.camera.viewportWidth)
            : (depthForSize && depthForSize->GetWidth() > 0u ? depthForSize->GetWidth() : 1280u);
        const UINT32 vpH = (cameraFrameView.camera.viewportHeight > 0.0f)
            ? static_cast<UINT32>(cameraFrameView.camera.viewportHeight)
            : (depthForSize && depthForSize->GetHeight() > 0u ? depthForSize->GetHeight() : 720u);
        if (!EnsureCameraContextResources(ctx, vpW, vpH, gpuSceneSlotCount))
            return false;

        RefreshBucketTemplatesForView(cameraFrameView, scene, true);

        GPUBuffer* shObjMatSlotsBuf = scene->GetGPUMaterialSlotTableBuffer();
        const UINT32 shObjMatSlotsCnt = (shObjMatSlotsBuf && shObjMatSlotsBuf->GetNativeHandle() != 0)
            ? scene->GetGPUMaterialSlotTableCount()
            : 0u;
        if (!shObjMatSlotsBuf || shObjMatSlotsBuf->GetNativeHandle() == 0)
        {
            shObjMatSlotsBuf = m_dummyObjectMatSlotsBuf;
        }

        if (isStandaloneShadowOnly)
        {
            if (!AppendResidencyMaskDispatch(outPreRasterItems))
                return false;
        }

        const UINT32 shBucketCount   = static_cast<UINT32>(m_shadowBuckets.size());
        const UINT32 clampedPages    = (std::min)((std::max)(activePageCap, 1u), 65535u);
        const UINT32 pageViewBufCap  = (std::max)(clampedPages, kNVGMaxShadowPages);
        const UINT32 physPoolCap     = (std::max)(ioShadowPassView.arrayLayerCount, activePageCap);
        const UINT32 pageRenderCnt   = pageRenderDataBuf->GetElementCount();
        const UINT32 bvhNodeCount    = m_residencyManager.GetTotalBVHNodeCount();
        const UINT32 clusterCount    = m_residencyManager.GetTotalClusterCount();
        const UINT32 groupCount      = m_residencyManager.GetTotalClusterGroupCount();
        const UINT32 vpCount         = m_residencyManager.GetTotalVirtualPageCount();
        const UINT32 meshTabCount    = m_residencyManager.GetMeshTableCapacity();
        const UINT32 matTabCount     = static_cast<UINT32>(m_materialTable.size());
        const GPUBufferUsage uavSrv  = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV;
        const GPUBufferUsage uavInd  = GPUBufferUsage::STRUCTURED_SRV | GPUBufferUsage::STRUCTURED_UAV | GPUBufferUsage::INDIRECT_ARGUMENT;

        if (!EnsureBuffer(ctx.shadowPageViews, sizeof(NVGShadowPageViewGPUData), pageViewBufCap, uavSrv) ||
            !EnsureBuffer(ctx.shadowPageMasks, sizeof(UINT32), pageViewBufCap, uavSrv))
        {
            return false;
        }

        BOOL hasShadowPathA        = false;
        BOOL hasShadowPathB        = false;
        BOOL hasAnyShadowSWEligible = false;
        for (const auto& b : m_shadowBuckets)
        {
            if (b.isVertexPull) hasShadowPathA = true;
            else                hasShadowPathB = true;
            if (b.bucketIndex < m_shadowBucketFlags.size() &&
                (m_shadowBucketFlags[b.bucketIndex] & NVG_BUCKET_FLAG_SW_ELIGIBLE) != 0u)
            {
                hasAnyShadowSWEligible = true;
            }
        }

        CoreRenderPipeline dummyPipe;

        const FLOAT32 adaptiveScale    = m_residencyManager.GetAdaptiveErrorScale();
        const FLOAT32 effShadowErr     = ResolveEffectiveShadowTexelError();
        const FLOAT32 effThinBoost     = (adaptiveScale > 1.01f)
            ? (m_config.thinFeatureShadowBoost / (adaptiveScale * adaptiveScale))
            : m_config.thinFeatureShadowBoost;
        const FLOAT32 coarseBiasBoost  = (adaptiveScale > 1.05f)
            ? (1.0f + (adaptiveScale - 1.0f) * 0.65f)
            : 1.0f;
        const BOOL    hasCommitBuf     = (pageCommitRecordsBuf != nullptr && pageCommitRecordsBuf->GetNativeHandle() != 0);
        GPUBuffer*    commitBuf        = hasCommitBuf ? pageCommitRecordsBuf : m_dummyCommitRecordsBuf;
        const UINT32  commitCap        = hasCommitBuf ? pageCommitRecordsBuf->GetElementCount() : 0u;
        const BOOL    clearReqsSh      = isStandaloneShadowOnly ? m_residencyManager.ConsumeClearPageRequestsFlag() : false;

        // Upload per-shadow-view light cullingMasks for NVG shadow caster layerMask filtering
        UINT32 maxShadowViewSlot = 16u;
        for (const FrameLightData& ld : cameraFrameView.lights)
        {
            if (ld.castsShadow != 0u && ld.shadowViewCount > 0u)
            {
                maxShadowViewSlot = (std::max)(maxShadowViewSlot, ld.shadowViewBase + ld.shadowViewCount);
            }
        }
        std::vector<UINT32> shadowViewMasksCpu(static_cast<SIZE_T>(maxShadowViewSlot), 0xFFFFFFFFu);
        for (const FrameLightData& ld : cameraFrameView.lights)
        {
            if (ld.castsShadow != 0u && ld.shadowViewCount > 0u)
            {
                for (UINT32 vi = 0u; vi < ld.shadowViewCount; ++vi)
                {
                    shadowViewMasksCpu[ld.shadowViewBase + vi] = ld.cullingMask;
                }
            }
        }
        if (!EnsureBuffer(ctx.shadowViewMasks, sizeof(UINT32), maxShadowViewSlot, GPUBufferUsage::STRUCTURED_SRV))
        {
            return false;
        }
        m_bufferManager->UpdateGPUBuffer(
            ctx.shadowViewMasks,
            shadowViewMasksCpu.data(),
            maxShadowViewSlot * static_cast<UINT32>(sizeof(UINT32)),
            0u);

        GPUBuffer* shSkinBoundsBuf = scene->GetGPUSkinningBoundsBuffer();
        const UINT32 shSkinBoundsCnt = (shSkinBoundsBuf && shSkinBoundsBuf->GetNativeHandle() != 0)
            ? scene->GetGPUSkinningBoundsCount()
            : 0u;
        if (!shSkinBoundsBuf || shSkinBoundsBuf->GetNativeHandle() == 0)
        {
            shSkinBoundsBuf = m_dummySkinnedBoundsBuf;
        }

        // Configure ShadowCull common params
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "sceneSlotCount", gpuSceneSlotCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "meshTableCount", meshTabCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "materialTableCount", matTabCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "bvhNodeCount", bvhNodeCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "clusterMetaCount", clusterCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "clusterGroupCount", groupCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "virtualPageCount", vpCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "bucketCount", shBucketCount);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "maxActiveShadowPages", clampedPages);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "maxNodeQueueCapacity", ctx.nodeQueueA->GetElementCount());
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "maxCandidateCapacity", ctx.candidateClusters->GetElementCount());
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "maxVisibleCapacity", ctx.shadowRawVisible->GetElementCount());
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "useGPUActivePageCount", 1u);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "explicitActivePageCount", 0u);
        m_kShadowCull->SetFloat("NVGShadowCullParams", "Config", "shadowMaxTexelError", effShadowErr);
        m_kShadowCull->SetFloat("NVGShadowCullParams", "Config", "thinFeatureShadowBoost", effThinBoost);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "applyShaderBias", rasterDesc.applyShaderBias ? 1u : 0u);
        m_kShadowCull->SetFloat("NVGShadowCullParams", "Config", "depthBiasScale", rasterDesc.depthBiasScale * coarseBiasBoost);
        m_kShadowCull->SetFloat("NVGShadowCullParams", "Config", "normalBiasScale", rasterDesc.normalBiasScale * coarseBiasBoost);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "hasCommitRecords", hasCommitBuf ? 1u : 0u);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "clearPageRequests", clearReqsSh ? 1u : 0u);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "skinBoundsCount", shSkinBoundsCnt);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "shadowViewMaskCount", maxShadowViewSlot);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "objectMatSlotTableCount", shObjMatSlotsCnt);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "pageRenderDataCount", pageRenderCnt);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "physicalPoolCapacity", physPoolCap);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "commitRecordCap", commitCap);

        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InProjectionCounter", projCounterBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InCasterPageRanges", activeRenderPageListBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InPageRenderData", pageRenderDataBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InMeshTable", m_residencyManager.GetMeshTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InObjectMaterialSlots", shObjMatSlotsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InBVHNodes", m_residencyManager.GetBVHNodeBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InClusterMeta", m_residencyManager.GetClusterMetaBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InClusterGroups", m_residencyManager.GetClusterGroupBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InVirtualPageTable", m_residencyManager.GetVirtualPageTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InGroupActiveMask", m_residencyManager.GetGroupActiveMaskBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InSkinnedBounds", shSkinBoundsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InShadowViewMasks", ctx.shadowViewMasks, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InNodeQueue", ctx.nodeQueueB, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutNodeQueue", ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutCandidateClusters", ctx.candidateClusters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutShadowPageViews", ctx.shadowPageViews, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutShadowPageMasks", ctx.shadowPageMasks, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutVisibleClusters", ctx.shadowRawVisible, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutBucketCounts", ctx.shadowBucketCounts, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutPageRequests", m_residencyManager.GetPageRequestBuffer(), GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InOutPageCommitRecords", commitBuf, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);

        // Mode 4: Race-free single-group reset of shadow counters, buckets & telemetry before multi-group Mode 0
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dispatchMode", 4u);
        m_kShadowCull->SetIndirectArgsBuffer(nullptr, 0u);
        m_kShadowCull->SetDispatchThreadCount(64u, 1u, 1u);
        {
            FramePassItem m4{};
            m4.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowCull, cameraFrameView, m4.computeDispatch)) return false;
            m4.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(m4));
        }

        // Mode 0: Prepare shadow page views & atomic overflow marks across active/overflow pages
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dispatchMode", 0u);
        m_kShadowCull->SetIndirectArgsBuffer(nullptr, 0u);
        m_kShadowCull->SetDispatchThreadCount((std::min)((std::max)({ clampedPages, commitCap, shBucketCount, clearReqsSh ? vpCount : 0u, 64u }), 4194240u), 1u, 1u);
        {
            FramePassItem m0{};
            m0.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowCull, cameraFrameView, m0.computeDispatch)) return false;
            m0.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(m0));
        }

        // Mode 1: Cull virtual shadow casters against ALL active dirty NVS shadow pages (off-screen safe!)
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dispatchMode", 1u);
        m_kShadowCull->SetIndirectArgsBuffer(nullptr, 0u);
        m_kShadowCull->SetDispatchThreadCount((std::min)(gpuSceneSlotCount, 4194240u), 1u, 1u);
        {
            FramePassItem m1{};
            m1.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowCull, cameraFrameView, m1.computeDispatch)) return false;
            m1.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(m1));
        }

        // Ensure m_kInstanceCull has all buffers bound before AppendPrepareDispatchArgs (Mode 2)
        if (m_kInstanceCull)
        {
            m_kInstanceCull->SetTexture("NVGInstanceCullParams", "HZBTexture", ctx.hzbTexture, 0xFFFFFFFFu);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMeshTable", m_residencyManager.GetMeshTableBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "InObjectMaterialSlots", shObjMatSlotsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutNodeQueue", ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutOccludedObjects", ctx.occludedObjects, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutBucketCounts", ctx.bucketCounts, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutPageRequests", m_residencyManager.GetPageRequestBuffer(), GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kInstanceCull->SetGPUBuffer("NVGInstanceCullParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        }

        // Mode 2: 4-wide BVH4 cooperative shadow traversal ping-pong (counters 10 <-> 11, 16 parent items x 4 lanes = 64 threads/group)
        const UINT32 steps = (std::min)((std::max)(m_maxMeshBVHDepth, 1u), kNVGMaxBVHDepth);
        for (UINT32 step = 0u; step < steps; ++step)
        {
            const BOOL   even   = ((step & 1u) == 0u);
            const UINT32 srcCnt = even ? 10u : 11u;
            const UINT32 dstCnt = even ? 11u : 10u;

            AppendPrepareDispatchArgs(ctx, srcCnt, 24u, 16u, dstCnt, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);

            m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dispatchMode", 2u);
            m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "srcCounterIndex", srcCnt);
            m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dstNodeCounterIndex", dstCnt);
            m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InNodeQueue", even ? ctx.nodeQueueA : ctx.nodeQueueB, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutNodeQueue", even ? ctx.nodeQueueB : ctx.nodeQueueA, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kShadowCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 24u * sizeof(UINT32));

            FramePassItem m2{};
            m2.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowCull, cameraFrameView, m2.computeDispatch)) return false;
            m2.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(m2));
        }

        // Mode 3: Independent Shadow-Texel LOD Cut & Thin-Feature Silhouette Preservation
        AppendPrepareDispatchArgs(ctx, 12u, 28u, 64u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);
        m_kShadowCull->SetUInt("NVGShadowCullParams", "Config", "dispatchMode", 3u);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "InNodeQueue", ctx.nodeQueueA, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kShadowCull->SetGPUBuffer("NVGShadowCullParams", "OutNodeQueue", ctx.nodeQueueB, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kShadowCull->SetIndirectArgsBuffer(ctx.dispatchArgs, 28u * sizeof(UINT32));
        {
            FramePassItem m3{};
            m3.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowCull, cameraFrameView, m3.computeDispatch)) return false;
            m3.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(m3));
        }

        // Compact shadow visible clusters + Path-A shadow indirect args + optional Path-B shadow decode & indirect draw
        ++ctx.frameTagCounter;
        if (ctx.frameTagCounter == 0u) ctx.frameTagCounter = 1u;
        const UINT32 frameTag = ctx.frameTagCounter;

        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 0u);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "bucketCount", shBucketCount);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxVisibleCapacity", ctx.shadowRawVisible->GetElementCount());
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "rawVisibleCounterIdx", 13u);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "frameTag", frameTag);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "pageWordCount", kNVGPageWordCount);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxDecodedVertexWords", ctx.shadowDecodedVertexPool->GetElementCount());
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxDecodedIndices", ctx.shadowDecodedIndexPool->GetElementCount());
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "maxUniqueDecodeJobs", ctx.shadowUniqueDecodeJobs->GetElementCount());
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "decodedTableCapacity", ctx.shadowDecodedClusterTbl->GetElementCount());
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "shadowCommitRecordCap", commitCap);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "drawGroupBindingCount", static_cast<UINT32>(m_shadowBindingDescs.size()));
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "enableSWClusterCull", 0u);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "swMaskWordCount", 0u);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InPagePool", m_residencyManager.GetPhysicalPagePoolBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InBucketCounts", ctx.shadowBucketCounts, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InBucketFlags", m_shadowBucketFlagsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InDrawGroupBindings", m_shadowBindingsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InRawVisibleClusters", ctx.shadowRawVisible, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InSWScheduledMask", m_dummySWMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InWinningClusterMask", m_dummySWMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutBucketRanges", ctx.shadowBucketRanges, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutBucketWriteCursors", ctx.shadowBucketWriteCursors, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutPathAIndirectArgs", ctx.shadowPathAIndirectArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutCompactedVisibleClusters", ctx.shadowCompactedVisible, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedClusterTable", ctx.shadowDecodedClusterTbl, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutUniqueDecodeJobs", ctx.shadowUniqueDecodeJobs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedVertexWords", ctx.shadowDecodedVertexPool, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutDecodedIndexPool", ctx.shadowDecodedIndexPool, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InOutPageCommitRecords", commitBuf, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
        m_kCompactionDecode->SetIndirectArgsBuffer(nullptr, 0u);
        const UINT32 shClearGroups = (ctx.shadowDecodedClusterTbl->GetElementCount() + 127u) / 128u;
        m_kCompactionDecode->SetThreadGroupCount((std::max)(shClearGroups, 1u), 1u, 1u);
        {
            FramePassItem ps{};
            ps.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, cameraFrameView, ps.computeDispatch)) return false;
            ps.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(ps));
        }

        AppendPrepareDispatchArgs(ctx, 13u, 32u, 128u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);
        m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 1u);
        m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 32u * sizeof(UINT32));
        {
            FramePassItem sc{};
            sc.type = FramePassItemType::COMPUTE_DISPATCH;
            if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, cameraFrameView, sc.computeDispatch)) return false;
            sc.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
            outPreRasterItems.push_back(std::move(sc));
        }

        const BOOL isRevZShadow     = (rasterDesc.depthMode == NVS_DEPTH_MODE_REVERSED_ONE_TO_ZERO);
        BOOL       runShadowSWRaster = m_config.enableSoftwareRasterizer &&
                                       hasAnyShadowSWEligible &&
                                       m_kShadowMicropolyRaster != nullptr &&
                                       m_shadowSWMergeVS != nullptr &&
                                       m_shadowSWMergePS != nullptr;
        UINT32     shSWMaskWords    = 0u;

        if (runShadowSWRaster)
        {
            const UINT32 shVisCap = ctx.shadowRawVisible->GetElementCount();
            shSWMaskWords         = (shVisCap + 31u) / 32u;
            if (!EnsureBuffer(ctx.shadowSWDepthKeyPool, sizeof(UINT32), kNVGMaxShadowSWTiles * kNVGShadowSWTileTexels, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWClusterList, sizeof(UINT32), shVisCap, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWScheduledMask, sizeof(UINT32), shSWMaskWords, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWFallbackMask, sizeof(UINT32), shSWMaskWords, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWPageNeedsTile, sizeof(UINT32), pageViewBufCap, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWPageTileMap, sizeof(UINT32), pageViewBufCap, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWDirtyTileSlices, sizeof(UINT32), kNVGMaxShadowSWTiles, uavSrv) ||
                !EnsureBuffer(ctx.shadowSWMergeIndirectArgs, sizeof(UINT32), 4u, uavInd))
            {
                runShadowSWRaster = false;
                shSWMaskWords     = 0u;
            }
        }

        if (runShadowSWRaster)
        {
            GPUBuffer* shBonePalBuf = scene->GetGPUSkinningPaletteBuffer();
            const UINT32 shBonePalCnt = (shBonePalBuf && shBonePalBuf->GetNativeHandle() != 0)
                ? scene->GetGPUSkinningPaletteCount()
                : 0u;
            if (!shBonePalBuf || shBonePalBuf->GetNativeHandle() == 0)
            {
                shBonePalBuf = m_dummyBonePaletteBuf;
            }

            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "bucketCount", shBucketCount);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "maxActiveShadowPages", clampedPages);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "physicalPoolCapacity", physPoolCap);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "maxSWTiles", kNVGMaxShadowSWTiles);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "maxVisibleCapacity", ctx.shadowCompactedVisible->GetElementCount());
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "swClusterCap", ctx.shadowSWClusterList->GetElementCount());
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "maskWordCount", shSWMaskWords);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "pageWordCount", kNVGPageWordCount);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "clusterMetaCount", clusterCount);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "materialTableCount", matTabCount);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "bonePaletteCount", shBonePalCnt);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "skinBoundsCount", shSkinBoundsCnt);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "isReversedZ", isRevZShadow ? 1u : 0u);
            m_kShadowMicropolyRaster->SetInt("NVGShadowMicropolyParams", "Config", "rasterDepthBias", ioShadowPassView.depthBias);
            m_kShadowMicropolyRaster->SetFloat("NVGShadowMicropolyParams", "Config", "rasterSlopeScaledDepthBias", ioShadowPassView.slopeScaledDepthBias);
            m_kShadowMicropolyRaster->SetFloat("NVGShadowMicropolyParams", "Config", "rasterDepthBiasClamp", ioShadowPassView.depthBiasClamp);
            m_kShadowMicropolyRaster->SetFloat("NVGShadowMicropolyParams", "Config", "swMaxEdgePixels", m_config.swRasterMaxEdgePixels);

            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InPagePool", m_residencyManager.GetPhysicalPagePoolBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InOutCompactedVisibleClusters", ctx.shadowCompactedVisible, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InOutBucketRanges", ctx.shadowBucketRanges, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutPathAIndirectArgs", ctx.shadowPathAIndirectArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InShadowPageViews", ctx.shadowPageViews, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InClusterMeta", m_residencyManager.GetClusterMetaBuffer(), GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InMaterialTable", m_materialTableBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InBonePalette", shBonePalBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "InSkinnedBounds", shSkinBoundsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);

            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutShadowSWDepthKey", ctx.shadowSWDepthKeyPool, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutShadowSWClusterList", ctx.shadowSWClusterList, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutShadowSWScheduledMask", ctx.shadowSWScheduledMask, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutShadowSWFallbackMask", ctx.shadowSWFallbackMask, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutPageNeedsSWTile", ctx.shadowSWPageNeedsTile, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutPageTileMap", ctx.shadowSWPageTileMap, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDirtySWTileSlices", ctx.shadowSWDirtyTileSlices, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutMergeIndirectArgs", ctx.shadowSWMergeIndirectArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);

            // Mode 0: Reset page tile maps, masks, counters & merge indirect args
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 0u);
            m_kShadowMicropolyRaster->SetIndirectArgsBuffer(nullptr, 0u);
            const UINT32 shSWResetThreads = (std::min)((std::max)({ clampedPages, shSWMaskWords, 64u }), 4194240u);
            m_kShadowMicropolyRaster->SetDispatchThreadCount(shSWResetThreads, 1u, 1u);
            {
                FramePassItem sw0{};
                sw0.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw0.computeDispatch)) return false;
                sw0.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw0));
            }

            // Mode 1: Classify SW-eligible shadow micro-clusters (indirect over counter[13])
            AppendPrepareDispatchArgs(ctx, 13u, 60u, 64u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::NVG_SW_RASTER);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 1u);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", nullptr);
            m_kShadowMicropolyRaster->SetIndirectArgsBuffer(ctx.dispatchArgs, 60u * sizeof(UINT32));
            {
                FramePassItem sw1{};
                sw1.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw1.computeDispatch)) return false;
                sw1.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw1));
            }

            // Mode 2: Allocate dense 128x128 tile slots for dirty SW shadow pages
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 2u);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", ctx.dispatchArgs, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kShadowMicropolyRaster->SetIndirectArgsBuffer(nullptr, 0u);
            m_kShadowMicropolyRaster->SetDispatchThreadCount((std::min)((std::max)(clampedPages, 64u), 4194240u), 1u, 1u);
            {
                FramePassItem sw2{};
                sw2.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw2.computeDispatch)) return false;
                sw2.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw2));
            }

            // Mode 3: Indirectly clear ONLY active tileCount * 16,384 texels
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 3u);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", nullptr);
            m_kShadowMicropolyRaster->SetIndirectArgsBuffer(ctx.dispatchArgs, 64u * sizeof(UINT32));
            {
                FramePassItem sw3{};
                sw3.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw3.computeDispatch)) return false;
                sw3.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw3));
            }

            // Mode 4: 1 Threadgroup (64 threads) per SW Shadow Cluster
            AppendPrepareDispatchArgs(ctx, 21u, 68u, 1u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::NVG_SW_RASTER);
            m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 4u);
            m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", nullptr);
            m_kShadowMicropolyRaster->SetIndirectArgsBuffer(ctx.dispatchArgs, 68u * sizeof(UINT32));
            {
                FramePassItem sw4{};
                sw4.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw4.computeDispatch)) return false;
                sw4.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw4));
            }

            // Mode 5: Compact remaining HW clusters & update OutPathAIndirectArgs for Path-A shadow buckets
            if (hasShadowPathA && shBucketCount > 0u)
            {
                m_kShadowMicropolyRaster->SetUInt("NVGShadowMicropolyParams", "Config", "dispatchMode", 5u);
                m_kShadowMicropolyRaster->SetGPUBuffer("NVGShadowMicropolyParams", "OutDispatchArgs", nullptr);
                m_kShadowMicropolyRaster->SetIndirectArgsBuffer(nullptr, 0u);
                m_kShadowMicropolyRaster->SetDispatchThreadCount((std::max)(shBucketCount, 64u), 1u, 1u);
                FramePassItem sw5{};
                sw5.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kShadowMicropolyRaster, cameraFrameView, sw5.computeDispatch)) return false;
                sw5.computeDispatch.profileGroup = ComputeDispatchProfileGroup::NVG_SW_RASTER;
                outPreRasterItems.push_back(std::move(sw5));
            }
        }

        GPUBuffer* shSchedMaskBuf = runShadowSWRaster ? ctx.shadowSWScheduledMask : m_dummySWMaskBuf;
        GPUBuffer* shWinMaskBuf   = runShadowSWRaster ? ctx.shadowSWFallbackMask  : m_dummySWMaskBuf;

        if (hasShadowPathB)
        {
            AppendPrepareDispatchArgs(ctx, 18u, 48u, 128u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 3u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "enableSWClusterCull", runShadowSWRaster ? 1u : 0u);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "swMaskWordCount", shSWMaskWords);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InSWScheduledMask", shSchedMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetGPUBuffer("NVGCompactionParams", "InWinningClusterMask", shWinMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 48u * sizeof(UINT32));
            {
                FramePassItem clm{};
                clm.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, cameraFrameView, clm.computeDispatch)) return false;
                clm.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
                outPreRasterItems.push_back(std::move(clm));
            }

            AppendPrepareDispatchArgs(ctx, 5u, 36u, 1u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);
            m_kCompactionDecode->SetUInt("NVGCompactionParams", "Config", "dispatchMode", 4u);
            m_kCompactionDecode->SetIndirectArgsBuffer(ctx.dispatchArgs, 36u * sizeof(UINT32));
            {
                FramePassItem dec{};
                dec.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kCompactionDecode, cameraFrameView, dec.computeDispatch)) return false;
                dec.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
                outPreRasterItems.push_back(std::move(dec));
            }

            const UINT64 vbAddr   = m_backend->GetGPUBufferGPUVirtualAddress(ctx.shadowDecodedVertexPool->GetNativeHandle());
            const UINT64 ibAddr   = m_backend->GetGPUBufferGPUVirtualAddress(ctx.shadowDecodedIndexPool->GetNativeHandle());
            const UINT64 cbvAddr  = m_backend->GetGPUBufferGPUVirtualAddress(ctx.shadowPerDrawCBVWords->GetNativeHandle());
            const UINT64 instAddr = m_backend->GetGPUBufferGPUVirtualAddress(ctx.shadowPerBatchInstWords->GetNativeHandle());
            const NativeHandle boneHandle = cameraFrameView.gpuSkinningPaletteBuffer;
            const UINT64 boneAddr = (boneHandle != 0) ? m_backend->GetGPUBufferGPUVirtualAddress(boneHandle) : 0ull;

            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "dispatchMode", 0u);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bucketCount", shBucketCount);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxVisibleCapacity", ctx.shadowRawVisible->GetElementCount());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "rawVisibleCounterIdx", 13u);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxCommandsPerBucket", ctx.dynShadowCmdsPerBucket);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "drawGroupBindingCount", static_cast<UINT32>(m_shadowBindingDescs.size()));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedTableCapacity", ctx.shadowDecodedClusterTbl->GetElementCount());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "isShadowPass", 1u);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "shadowCommitRecordCap", commitCap);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bucketWordStride", 36u);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "enableSWClusterCull", runShadowSWRaster ? 1u : 0u);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "swMaskWordCount", shSWMaskWords);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBAddressLo", static_cast<UINT32>(vbAddr & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBAddressHi", static_cast<UINT32>((vbAddr >> 32u) & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedVBSizeBytes", ctx.shadowDecodedVertexPool->GetByteSize());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBAddressLo", static_cast<UINT32>(ibAddr & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBAddressHi", static_cast<UINT32>((ibAddr >> 32u) & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "decodedIBSizeBytes", ctx.shadowDecodedIndexPool->GetByteSize());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "cbvPoolBaseLo", static_cast<UINT32>(cbvAddr & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "cbvPoolBaseHi", static_cast<UINT32>((cbvAddr >> 32u) & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "instancePoolBaseLo", static_cast<UINT32>(instAddr & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "instancePoolBaseHi", static_cast<UINT32>((instAddr >> 32u) & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bonePaletteBaseLo", static_cast<UINT32>(boneAddr & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "bonePaletteBaseHi", static_cast<UINT32>((boneAddr >> 32u) & 0xFFFFFFFFull));
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxPerDrawCBVWords", ctx.shadowPerDrawCBVWords->GetElementCount());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "maxPerBatchInstanceWords", ctx.shadowPerBatchInstWords->GetElementCount());
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "shadowPageViewCount", clampedPages);

            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InCompactedVisibleClusters", ctx.shadowCompactedVisible, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InBucketRanges", ctx.shadowBucketRanges, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDecodedClusterTable", ctx.shadowDecodedClusterTbl, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InGPUSceneObjects", gpuSceneBuffer, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupBindings", m_shadowBindingsBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupFieldPatches", m_shadowPatchesBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InDrawGroupTemplateWords", m_shadowTemplatesBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InShadowPageViews", ctx.shadowPageViews, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InSWScheduledMask", shSchedMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InWinningClusterMask", shWinMaskBuf, GPUResourceAccess::NON_PIXEL_SHADER_RESOURCE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutIndirectDrawCommands", ctx.shadowIndirectCommands, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutPerDrawCBVWords", ctx.shadowPerDrawCBVWords, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutPerBatchInstanceWords", ctx.shadowPerBatchInstWords, GPUResourceAccess::UNORDERED_ACCESS_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutBucketCommandCounts", ctx.shadowBucketCmdCounts, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutQueueCounters", ctx.queueCounters, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "InOutPageCommitRecords", commitBuf, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kIndirectDraw->SetGPUBuffer("NVGIndirectDrawParams", "OutTelemetry", ctx.telemetryBuffer, GPUResourceAccess::UNORDERED_ACCESS_READ_WRITE);
            m_kIndirectDraw->SetIndirectArgsBuffer(nullptr, 0u);
            m_kIndirectDraw->SetDispatchThreadCount((std::max)(shBucketCount, 64u), 1u, 1u);
            {
                FramePassItem r0{};
                r0.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kIndirectDraw, cameraFrameView, r0.computeDispatch)) return false;
                r0.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
                outPreRasterItems.push_back(std::move(r0));
            }

            AppendPrepareDispatchArgs(ctx, 18u, 40u, 64u, kNVGInvalidIndex, outPreRasterItems, ComputeDispatchProfileGroup::RESIDENCY_CULL);
            m_kIndirectDraw->SetUInt("NVGIndirectDrawParams", "Config", "dispatchMode", 1u);
            m_kIndirectDraw->SetIndirectArgsBuffer(ctx.dispatchArgs, 40u * sizeof(UINT32));
            {
                FramePassItem r1{};
                r1.type = FramePassItemType::COMPUTE_DISPATCH;
                if (!dummyPipe.BuildComputeDispatchCommand(m_kIndirectDraw, cameraFrameView, r1.computeDispatch)) return false;
                r1.computeDispatch.profileGroup = ComputeDispatchProfileGroup::RESIDENCY_CULL;
                outPreRasterItems.push_back(std::move(r1));
            }

            ioShadowPassView.indirectVertexPoolBuffer = ctx.shadowDecodedVertexPool->GetNativeHandle();
            ioShadowPassView.indirectIndexPoolBuffer  = ctx.shadowDecodedIndexPool->GetNativeHandle();
        }

        // Append NVG Shadow DrawCommands directly into NVS's LayeredDepthPassView!
        if (runShadowSWRaster && m_shadowSWMergeVS && m_shadowSWMergePS)
        {
            const BOOL needMergeGS = rasterDesc.forceGeometryShaderFallback ||
                                     (m_backend && !m_backend->GetCapabilities().supportsVPAndRTArrayIndexWithoutGS);
            if (!needMergeGS || (m_shadowSWMergeGS && m_shadowSWMergeGS->IsValid()))
            {
                DrawCommand mergeCmd{};
                mergeCmd.useNVSIndirectCommandSignature = false;
                mergeCmd.useVertexPullingIndirect       = true;
                mergeCmd.indirectArgsBuffer             = ctx.shadowSWMergeIndirectArgs->GetNativeHandle();
                mergeCmd.indirectArgsOffset             = 0u;
                mergeCmd.maxIndirectCommandCount        = 1u;
                mergeCmd.indirectCountBuffer            = 0;
                mergeCmd.vertexShader                   = m_shadowSWMergeVS->GetNativeHandle();
                mergeCmd.pixelShader                    = m_shadowSWMergePS->GetNativeHandle();
                mergeCmd.geometryShader                 = needMergeGS ? m_shadowSWMergeGS->GetNativeHandle() : 0;
                mergeCmd.cullMode                       = CullMode::NONE;
                mergeCmd.blendMode                      = BlendMode::OPAQUE;
                mergeCmd.fillMode                       = FillMode::SOLID;
                mergeCmd.depthState.test                = true;
                mergeCmd.depthState.write               = true;
                mergeCmd.depthState.compare             = isRevZShadow ? DepthCompare::GREATER_EQUAL : DepthCompare::LESS_EQUAL;

                const ShaderReflection& mvr = m_shadowSWMergeVS->GetReflection();
                for (const auto& nb : mvr.buffers)
                {
                    if (nb.binding == 0xFFFF || nb.totalSize == 0u) continue;
                    NativeConstantBufferBinding cb{};
                    cb.slotIndex     = nb.binding;
                    cb.registerSpace = nb.set;
                    cb.stage         = ShaderStage::VERTEX;
                    cb.byteSize      = nb.totalSize;
                    cb.ownedData.assign(nb.totalSize, 0u);
                    for (const auto& f : nb.fields)
                    {
                        if (f.offset + sizeof(UINT32) > nb.totalSize) continue;
                        UINT32 v = 0u;
                        if (f.name == "maxArrayLayers") v = physPoolCap;
                        else if (f.name == "maxSWTiles") v = kNVGMaxShadowSWTiles;
                        else continue;
                        std::memcpy(cb.ownedData.data() + f.offset, &v, sizeof(UINT32));
                    }
                    mergeCmd.constantBuffers.push_back(std::move(cb));
                }
                for (const auto& sb : mvr.structuredBuffers)
                {
                    if (sb.binding == 0xFFFF) continue;
                    NativeStructuredBufferBinding b{};
                    b.slotIndex     = sb.binding;
                    b.registerSpace = sb.set;
                    b.stage         = ShaderStage::VERTEX;
                    b.isUAV         = false;
                    b.buffer        = ctx.shadowSWDirtyTileSlices->GetNativeHandle();
                    b.stride        = ctx.shadowSWDirtyTileSlices->GetStride();
                    b.elementCount  = ctx.shadowSWDirtyTileSlices->GetElementCount();
                    b.byteSize      = ctx.shadowSWDirtyTileSlices->GetByteSize();
                    mergeCmd.structuredBuffers.push_back(std::move(b));
                }

                const ShaderReflection& mpr = m_shadowSWMergePS->GetReflection();
                for (const auto& nb : mpr.buffers)
                {
                    if (nb.binding == 0xFFFF || nb.totalSize == 0u) continue;
                    NativeConstantBufferBinding cb{};
                    cb.slotIndex     = nb.binding;
                    cb.registerSpace = nb.set;
                    cb.stage         = ShaderStage::PIXEL;
                    cb.byteSize      = nb.totalSize;
                    cb.ownedData.assign(nb.totalSize, 0u);
                    for (const auto& f : nb.fields)
                    {
                        if (f.offset + sizeof(UINT32) > nb.totalSize) continue;
                        UINT32 v = 0u;
                        if (f.name == "isReversedZ")     v = isRevZShadow ? 1u : 0u;
                        else if (f.name == "maxSWTiles") v = kNVGMaxShadowSWTiles;
                        else continue;
                        std::memcpy(cb.ownedData.data() + f.offset, &v, sizeof(UINT32));
                    }
                    mergeCmd.constantBuffers.push_back(std::move(cb));
                }
                for (const auto& sb : mpr.structuredBuffers)
                {
                    if (sb.binding == 0xFFFF) continue;
                    NativeStructuredBufferBinding b{};
                    b.slotIndex     = sb.binding;
                    b.registerSpace = sb.set;
                    b.stage         = ShaderStage::PIXEL;
                    b.isUAV         = false;
                    b.buffer        = ctx.shadowSWDepthKeyPool->GetNativeHandle();
                    b.stride        = ctx.shadowSWDepthKeyPool->GetStride();
                    b.elementCount  = ctx.shadowSWDepthKeyPool->GetElementCount();
                    b.byteSize      = ctx.shadowSWDepthKeyPool->GetByteSize();
                    mergeCmd.structuredBuffers.push_back(std::move(b));
                }

                ioShadowPassView.drawCommands.push_back(std::move(mergeCmd));
            }
        }

        const UINT64 shadowRev  = (m_currentFrameIndex << 16u) ^ (m_residencyManager.GetResidencyRevision() << 4u) ^ 3ull;
        std::vector<ComputeDispatchCommand> shadowMatDispatches;
        for (const auto& b : m_shadowBuckets)
        {
            DrawCommand cmd{};
            BOOL ok = false;
            if (b.isVertexPull)
            {
                ok = BuildVertexPullDrawCommand(
                    b, cameraFrameView, scene, gpuSceneBuffer,
                    ctx.shadowBucketRanges, ctx.shadowCompactedVisible, ctx.shadowPathAIndirectArgs,
                    ctx.shadowPageViews, shadowRev, true,
                    isRevZShadow ? DepthCompare::GREATER_EQUAL : DepthCompare::LESS_EQUAL,
                    true, shadowMatDispatches, cmd);
            }
            else
            {
                ok = BuildDecodedIADrawCommand(
                    b, cameraFrameView, scene, gpuSceneBuffer,
                    ctx.shadowBucketRanges, ctx.shadowCompactedVisible, ctx.shadowPageViews,
                    ctx.shadowIndirectCommands, ctx.shadowBucketCmdCounts,
                    ctx.shadowDecodedVertexPool, ctx.shadowDecodedIndexPool,
                    ctx.shadowPerDrawCBVWords, ctx.shadowPerBatchInstWords,
                    ctx.dynShadowCmdsPerBucket,
                    shadowRev, true,
                    isRevZShadow ? DepthCompare::GREATER_EQUAL : DepthCompare::LESS_EQUAL,
                    true, shadowMatDispatches, cmd);
            }
            if (ok)
            {
                ioShadowPassView.drawCommands.push_back(std::move(cmd));
            }
        }
        for (auto& md : shadowMatDispatches)
        {
            FramePassItem matItem{};
            matItem.type            = FramePassItemType::COMPUTE_DISPATCH;
            matItem.computeDispatch = std::move(md);
            outPreRasterItems.push_back(std::move(matItem));
        }

        // In shadow-only frames (where camera NVG pass will not run), ensure page requests & telemetry still read back!
        if (isStandaloneShadowOnly)
        {
            m_residencyManager.QueuePageRequestReadback();
            if (!ctx.telemetryPending && ctx.telemetryBuffer != nullptr)
            {
                if (m_bufferManager->RequestGPUBufferReadback(ctx.telemetryBuffer, 0u, sizeof(NVGTelemetryGPUData)))
                {
                    ctx.telemetryPending = true;
                }
            }
        }

        return true;
    }

    void CoreNVGManager::OnAfterExecuteFrame(Camera* /*camera*/)
    {
        // Async page-request & telemetry readbacks are queued inside InjectCameraNVGPasses
        // before ExecuteFrame so D3D12 frame execution captures the completed pass output cleanly
        // without overwriting pending readback fences.
    }

    void CoreNVGManager::AccumulateBackendTiming(const BackendFrameMetrics& bm)
    {
        m_stats.gpuHZBBuildMs         += bm.gpuNVGHZBMs;
        m_stats.gpuCullAndLODMs       += bm.gpuNVGCullLodMs;
        m_stats.gpuDecodeAndDrawGenMs += bm.gpuNVGDrawGenMs;
        m_stats.gpuSWRasterMs         += bm.gpuNVGSWRasterMs;
        m_stats.gpuSWRasterPass1Ms    += bm.gpuNVGSWRasterPass1Ms;
        m_stats.gpuSWRasterPass2Ms    += bm.gpuNVGSWRasterPass2Ms;
        if (m_stats.active)
        {
            m_stats.gpuDepthPrepassMs   += bm.gpuDepthPrepassMs;
            m_stats.gpuGeometryRasterMs += bm.gpuGeometryPassMs;
        }
    }

    void CoreNVGManager::OnCameraDestroyed(const Camera* camera)
    {
        auto it = m_cameraContexts.find(camera);
        if (it != m_cameraContexts.end())
        {
            if (it->second) DestroyCameraContextResources(*it->second);
            m_cameraContexts.erase(it);
        }
    }

    void CoreNVGManager::OnMaterialDestroyed(const Material* material)
    {
        m_lastMaterialShadowHash.erase(material);
        m_cachedSyncScene = nullptr;
        m_cachedSyncMaterials.clear();
        m_cameraBuckets.clear();
        m_shadowBuckets.clear();
    }

    void CoreNVGManager::OnShaderSetDestroyed(const ShaderSet* /*shaderSet*/)
    {
        m_cachedSyncScene = nullptr;
        m_cachedSyncMaterials.clear();
        m_cameraBuckets.clear();
        m_shadowBuckets.clear();
    }

    void CoreNVGManager::OnSceneDestroyed(const Scene* /*scene*/)
    {
        m_cachedSyncScene = nullptr;
        m_cachedSyncMaterials.clear();
        m_cameraBuckets.clear();
        m_shadowBuckets.clear();
    }

    void CoreNVGManager::OnResize()
    {
        for (auto& kv : m_cameraContexts)
        {
            if (kv.second)
            {
                kv.second->hasValidPrevHZB = false;
            }
        }
    }
}
