#include <northstar.h>
#include "ns_window.h"
#include "ns_resource_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    using namespace Northstar;

    UniqueTexture CreateSolidTexture2D(NorthstarEngine& engine, BYTE r, BYTE g, BYTE b, BYTE a, bool srgb = false)
    {
        const BYTE pixel[4] = { r, g, b, a };
        TextureDesc desc{};
        desc.dataSource      = TextureDataSource::MEMORY;
        desc.initialData     = pixel;
        desc.initialDataSize = sizeof(pixel);
        desc.width           = 1;
        desc.height          = 1;
        desc.mipLevels       = 1;
        desc.format          = srgb ? TextureFormat::RGBA8_SRGB : TextureFormat::RGBA8_UNORM;
        desc.type            = TextureType::TEXTURE_2D;
        desc.usage           = TextureUsage::SAMPLED;
        return MakeUniqueTexture(engine, desc);
    }

    UniqueTexture CreateCheckerboardTexture2D(NorthstarEngine& engine, UINT32 size, UINT32 tilePixels)
    {
        std::vector<BYTE> pixels(static_cast<SIZE_T>(size) * static_cast<SIZE_T>(size) * 4u);
        for (UINT32 y = 0; y < size; ++y)
        {
            for (UINT32 x = 0; x < size; ++x)
            {
                const bool dark = (((x / tilePixels) + (y / tilePixels)) & 1u) != 0u;
                const BYTE c    = dark ? 58u : 186u;
                const SIZE_T idx = (static_cast<SIZE_T>(y) * size + x) * 4u;
                pixels[idx + 0] = c;
                pixels[idx + 1] = c;
                pixels[idx + 2] = static_cast<BYTE>(c + 8u);
                pixels[idx + 3] = 255u;
            }
        }

        TextureDesc desc{};
        desc.dataSource      = TextureDataSource::MEMORY;
        desc.initialData     = pixels.data();
        desc.initialDataSize = pixels.size();
        desc.width           = size;
        desc.height          = size;
        desc.mipLevels       = 0; // Auto-generate full mip chain
        desc.format          = TextureFormat::RGBA8_SRGB;
        desc.type            = TextureType::TEXTURE_2D;
        desc.usage           = TextureUsage::SAMPLED;
        return MakeUniqueTexture(engine, desc);
    }

    UniqueMesh CreateUVSphereMesh(NorthstarEngine& engine, FLOAT32 radius, UINT32 slices, UINT32 stacks)
    {
        std::vector<Vertex> vertices;
        std::vector<UINT32> indices;
        vertices.reserve(static_cast<SIZE_T>(stacks + 1u) * (slices + 1u));
        indices.reserve(static_cast<SIZE_T>(stacks) * slices * 6u);

        for (UINT32 st = 0; st <= stacks; ++st)
        {
            const FLOAT32 v   = static_cast<FLOAT32>(st) / static_cast<FLOAT32>(stacks);
            const FLOAT32 phi = v * PI;
            const FLOAT32 sinPhi = std::sin(phi);
            const FLOAT32 cosPhi = std::cos(phi);

            for (UINT32 sl = 0; sl <= slices; ++sl)
            {
                const FLOAT32 u     = static_cast<FLOAT32>(sl) / static_cast<FLOAT32>(slices);
                const FLOAT32 theta = u * 2.0f * PI;
                const FLOAT32 sinTheta = std::sin(theta);
                const FLOAT32 cosTheta = std::cos(theta);

                const Vec3 normal(sinPhi * cosTheta, cosPhi, sinPhi * sinTheta);
                Vec3 tangent(-sinTheta, 0.0f, cosTheta);
                if (tangent.LengthSq() < 1e-6f)
                {
                    tangent = Vec3::Right();
                }
                else
                {
                    tangent = tangent.Normalized();
                }
                const Vec3 bitangent = normal.Cross(tangent).Normalized();

                Vertex vert{};
                vert.position  = normal * radius;
                vert.normal    = normal;
                vert.tangent   = tangent;
                vert.bitangent = bitangent;
                vert.texCoord0 = Vec2(u, v);
                vert.texCoord1 = Vec2(u, v);
                vert.color     = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
                vertices.push_back(vert);
            }
        }

        const UINT32 ringStride = slices + 1u;
        for (UINT32 st = 0; st < stacks; ++st)
        {
            for (UINT32 sl = 0; sl < slices; ++sl)
            {
                const UINT32 i0 = st * ringStride + sl;
                const UINT32 i1 = i0 + 1u;
                const UINT32 i2 = (st + 1u) * ringStride + sl;
                const UINT32 i3 = i2 + 1u;

                // Clockwise front-facing winding (DirectX LH)
                indices.push_back(i0);
                indices.push_back(i1);
                indices.push_back(i2);

                indices.push_back(i1);
                indices.push_back(i3);
                indices.push_back(i2);
            }
        }

        MeshDesc desc{};
        desc.vertices     = vertices.data();
        desc.vertexCount  = static_cast<UINT32>(vertices.size());
        desc.indices      = indices.data();
        desc.indexCount   = static_cast<UINT32>(indices.size());
        desc.geometryMode = GeometryMode::Standard;
        return MakeUniqueMesh(engine, desc);
    }

    UniqueMesh CreateGroundPlaneMesh(NorthstarEngine& engine, FLOAT32 halfExtent, FLOAT32 uvRepeat)
    {
        Vertex verts[4]{};
        const Vec3 positions[4] = {
            Vec3(-halfExtent, 0.0f, -halfExtent),
            Vec3(-halfExtent, 0.0f,  halfExtent),
            Vec3( halfExtent, 0.0f,  halfExtent),
            Vec3( halfExtent, 0.0f, -halfExtent)
        };
        const Vec2 uvs[4] = {
            Vec2(0.0f,     uvRepeat),
            Vec2(0.0f,     0.0f),
            Vec2(uvRepeat, 0.0f),
            Vec2(uvRepeat, uvRepeat)
        };

        for (UINT32 i = 0; i < 4u; ++i)
        {
            verts[i].position  = positions[i];
            verts[i].normal    = Vec3::Up();
            verts[i].tangent   = Vec3::Right();
            verts[i].bitangent = Vec3::Forward();
            verts[i].texCoord0 = uvs[i];
            verts[i].texCoord1 = uvs[i];
            verts[i].color     = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
        }

        const UINT32 indices[6] = { 0u, 1u, 2u, 0u, 2u, 3u };

        MeshDesc desc{};
        desc.vertices     = verts;
        desc.vertexCount  = 4u;
        desc.indices      = indices;
        desc.indexCount   = 6u;
        desc.geometryMode = GeometryMode::Standard;
        return MakeUniqueMesh(engine, desc);
    }

    UniqueShader LoadCompiledShader(NorthstarEngine& engine, const std::filesystem::path& path, const std::string& debugName)
    {
        const auto loaded = Tools::ResourceLoader::LoadShaderFromFile(path.wstring(), debugName);
        if (!loaded.isValid)
        {
            std::cerr << "[HelloPBR] Failed to load compiled shader: " << path.string() << "\n";
            return {};
        }
        return MakeUniqueShader(engine, loaded.AsDesc());
    }
}

int main(int argc, char** argv)
{
    using namespace Northstar;

    UINT64 maxFrames = 0;
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::string(argv[i]) == "--frames")
        {
            maxFrames = static_cast<UINT64>(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }

    // 1. Create platform window
    Tools::WindowDesc winDesc{};
    winDesc.width          = 1280;
    winDesc.height         = 720;
    winDesc.title          = L"Northstar Graphics - 01 Hello PBR & Virtual Shadows";
    winDesc.enableDarkMode = true;

    Tools::Window window;
    if (!window.Create(winDesc))
    {
        std::cerr << "[HelloPBR] Failed to create Win32 window.\n";
        return 1;
    }

    // 2. Initialize Northstar Runtime Session (loads NorthstarCore & NorthstarD3D12, validates ABI)
    RuntimeConfig rtCfg{};
    rtCfg.backend      = BackendAPI::D3D12;
    rtCfg.renderTarget = window.GetNativeRenderTarget();
    rtCfg.vsync        = false;

    RuntimeSession runtime;
    if (!runtime.Initialize(rtCfg))
    {
        std::cerr << "[HelloPBR] Runtime initialization failed: " << runtime.GetLastError() << "\n";
        return 1;
    }

    NorthstarEngine& engine = runtime.GetEngine();
    window.SetOnResize([&engine](UINT32 w, UINT32 h) {
        engine.OnResize(w, h);
    });
    window.SetOnKey([&window](Tools::KeyCode key, bool isDown, bool) {
        if (key == Tools::KeyCode::Escape && isDown)
        {
            window.Close();
        }
    });

    // 3. Load built-in StandardLit shaders (.nsc generated by build.bat)
    const std::filesystem::path shaderDir = RuntimeSession::ResolveExecutableDirectory() / "shaders" / "StandardLit";
    auto vsStd        = LoadCompiledShader(engine, shaderDir / "StandardLitVS_hlsl12.nsc", "StandardLitVS");
    auto vsInst       = LoadCompiledShader(engine, shaderDir / "StandardLitInstancedVS_hlsl12.nsc", "StandardLitInstancedVS");
    auto psStd        = LoadCompiledShader(engine, shaderDir / "StandardLitPS_hlsl12.nsc", "StandardLitPS");
    auto shadowVsStd  = LoadCompiledShader(engine, shaderDir / "StandardLitShadowVS_hlsl12.nsc", "StandardLitShadowVS");
    auto shadowVsInst = LoadCompiledShader(engine, shaderDir / "StandardLitShadowInstancedVS_hlsl12.nsc", "StandardLitShadowInstancedVS");

    if (!vsStd || !vsInst || !psStd || !shadowVsStd || !shadowVsInst)
    {
        std::cerr << "[HelloPBR] Missing StandardLit .nsc shaders in '" << shaderDir.string() << "'. Run build.bat first.\n";
        return 1;
    }

    ShaderSetDesc geomSetDesc{};
    geomSetDesc.name                  = "StandardLitGeometry";
    geomSetDesc.vertexShader          = vsStd.get();
    geomSetDesc.instancedVertexShader = vsInst.get();
    geomSetDesc.pixelShader           = psStd.get();
    auto geomShaderSet = MakeUniqueShaderSet(engine, geomSetDesc);

    ShaderSetDesc shadowSetDesc{};
    shadowSetDesc.name                  = "StandardLitShadow";
    shadowSetDesc.vertexShader          = shadowVsStd.get();
    shadowSetDesc.instancedVertexShader = shadowVsInst.get();
    auto shadowShaderSet = MakeUniqueShaderSet(engine, shadowSetDesc);

    // 4. Create default textures & anisotropic sampler
    auto whiteTex   = CreateSolidTexture2D(engine, 255, 255, 255, 255, true);
    auto flatNrmTex = CreateSolidTexture2D(engine, 128, 128, 255, 255, false);
    auto blackTex   = CreateSolidTexture2D(engine, 0, 0, 0, 255, true);
    auto checkerTex = CreateCheckerboardTexture2D(engine, 512, 32);
    auto sampler    = MakeUniqueSampler(engine, SamplerDesc::AnisotropicWrap(16));

    auto bindDefaultMaterialMaps = [&](Material& mat, Texture* baseColorMap, bool useBaseMap) {
        mat.SetTexture("BaseColorMap", baseColorMap);
        mat.SetTexture("NormalMap", flatNrmTex.get());
        mat.SetTexture("MetallicRoughnessMap", whiteTex.get());
        mat.SetTexture("EmissiveMap", blackTex.get());
        mat.SetTexture("AOMap", whiteTex.get());
        mat.SetSampler("LinearWrapSampler", sampler.get());
        mat.SetBool("useBaseColorMap", useBaseMap);
        mat.SetBool("useNormalMap", false);
        mat.SetBool("useRoughnessMap", false);
        mat.SetBool("useEmissiveMap", false);
        mat.SetBool("useAOMap", false);
        mat.SetBool("receiveShadows", true);
    };

    // 5. Build meshes, materials, lights & scene
    auto sphereMesh = CreateUVSphereMesh(engine, 0.72f, 48, 32);
    auto groundMesh = CreateGroundPlaneMesh(engine, 24.0f, 12.0f);

    MaterialDesc baseMatDesc{};
    baseMatDesc.geometryShaderSet = geomShaderSet.get();
    baseMatDesc.shadowShaderSet   = shadowShaderSet.get();
    baseMatDesc.cullMode          = CullMode::Back;
    baseMatDesc.blendMode         = BlendMode::Opaque;

    auto groundMat = MakeUniqueMaterial(engine, baseMatDesc);
    bindDefaultMaterialMaps(*groundMat, checkerTex.get(), true);
    groundMat->SetFloat("roughness", 0.75f);
    groundMat->SetFloat("metallic", 0.05f);

    RenderObjectDesc groundObjDesc{};
    groundObjDesc.mesh     = groundMesh.get();
    groundObjDesc.material = groundMat.get();
    auto groundObject = MakeUniqueRenderObject(engine, groundObjDesc);

    auto scene = MakeUniqueScene(engine);
    scene->SetAmbientColor(Color(0.18f, 0.22f, 0.30f, 1.0f));
    scene->SetAmbientIntensity(0.85f);
    scene->AddRenderObject(groundObject.get());

    constexpr UINT32 kGridRows = 4;
    constexpr UINT32 kGridCols = 5;
    std::vector<UniqueMaterial>     sphereMaterials;
    std::vector<UniqueRenderObject> sphereObjects;
    sphereMaterials.reserve(kGridRows * kGridCols);
    sphereObjects.reserve(kGridRows * kGridCols);

    const Vec4 rowPalette[kGridRows] = {
        Vec4(0.95f, 0.78f, 0.36f, 1.0f), // Gold
        Vec4(0.92f, 0.26f, 0.24f, 1.0f), // Crimson
        Vec4(0.24f, 0.62f, 0.95f, 1.0f), // Cobalt
        Vec4(0.88f, 0.90f, 0.92f, 1.0f)  // Platinum
    };

    for (UINT32 r = 0; r < kGridRows; ++r)
    {
        const FLOAT32 metallic = static_cast<FLOAT32>(r) / static_cast<FLOAT32>(kGridRows - 1u);
        for (UINT32 c = 0; c < kGridCols; ++c)
        {
            const FLOAT32 roughness = 0.08f + 0.86f * (static_cast<FLOAT32>(c) / static_cast<FLOAT32>(kGridCols - 1u));

            auto mat = MakeUniqueMaterial(engine, baseMatDesc);
            bindDefaultMaterialMaps(*mat, whiteTex.get(), false);
            mat->SetVec4("baseColor", rowPalette[r]);
            mat->SetFloat("metallic", metallic);
            mat->SetFloat("roughness", roughness);

            RenderObjectDesc objDesc{};
            objDesc.mesh     = sphereMesh.get();
            objDesc.material = mat.get();
            auto obj = MakeUniqueRenderObject(engine, objDesc);

            const FLOAT32 x = (static_cast<FLOAT32>(c) - 2.0f) * 2.15f;
            const FLOAT32 z = (static_cast<FLOAT32>(r) - 1.5f) * 2.15f;
            obj->SetTransform(Vec3(x, 0.85f, z));

            scene->AddRenderObject(obj.get());
            sphereMaterials.push_back(std::move(mat));
            sphereObjects.push_back(std::move(obj));
        }
    }

    // Directional Sun Light with Northstar Virtual Shadows (NVS)
    LightDesc sunDesc{};
    sunDesc.type              = LightType::Directional;
    sunDesc.color             = Color(1.0f, 0.96f, 0.90f, 1.0f);
    sunDesc.intensity         = 3.2f;
    sunDesc.shadowResolution  = LightShadowResolution::Res16384;
    sunDesc.shadowSoftness    = LightShadowSoftness::MediumHard;
    sunDesc.shadowMaxDistance = 80.0f;
    auto sunLight = MakeUniqueLight(engine, sunDesc);
    sunLight->SetDirection(Vec3(-0.55f, -0.82f, 0.38f));
    scene->AddLight(sunLight.get());
    scene->SetDirectionalLight(sunLight.get());

    // Accent Point Lights with 16K NVS cubemap virtual shadows (stationary so NVS caches pages)
    LightDesc pointDesc{};
    pointDesc.type             = LightType::Point;
    pointDesc.intensity        = 14.0f;
    pointDesc.range            = 16.0f;
    pointDesc.shadowResolution = LightShadowResolution::Res16384;
    pointDesc.shadowSoftness   = LightShadowSoftness::Hard;

    pointDesc.color    = Color(0.25f, 0.72f, 1.0f, 1.0f);
    pointDesc.position = Vec3(5.5f, 3.0f, -4.0f);
    auto cyanLight     = MakeUniqueLight(engine, pointDesc);
    scene->AddLight(cyanLight.get());

    pointDesc.color    = Color(1.0f, 0.48f, 0.18f, 1.0f);
    pointDesc.position = Vec3(-5.5f, 3.2f, 4.0f);
    auto amberLight    = MakeUniqueLight(engine, pointDesc);
    scene->AddLight(amberLight.get());

    // Main Perspective Camera
    CameraDesc camDesc{};
    camDesc.fov         = 55.0f;
    camDesc.aspectRatio = static_cast<FLOAT32>(window.GetWidth()) / static_cast<FLOAT32>(window.GetHeight());
    camDesc.nearPlane   = 0.1f;
    camDesc.farPlane    = 250.0f;
    auto camera = MakeUniqueCamera(engine, camDesc);
    scene->AddCamera(camera.get());
    scene->SetMainCamera(camera.get());

    engine.SetActiveScene(scene.get());

    // 6. Real-time frame loop
    FLOAT32 titleUpdateAccum = 0.0f;
    while (window.IsRunning())
    {
        window.PollMessages();
        if (!window.IsRunning())
        {
            break;
        }

        engine.BeginFrame();
        const RenderStats& prevStats = engine.GetStats();
        const FLOAT32 t = static_cast<FLOAT32>(prevStats.elapsedTimeSeconds);

        // Smooth orbital camera
        const FLOAT32 camAngle = t * 0.22f;
        camera->SetPosition(Vec3(std::sin(camAngle) * 9.5f, 5.2f, -std::cos(camAngle) * 9.5f));
        camera->LookAt(Vec3(0.0f, 0.65f, 0.0f));

        engine.RenderFrame();
        engine.EndFrame();

        const RenderStats& stats = engine.GetStats();
        titleUpdateAccum += stats.deltaTimeSeconds;
        if (titleUpdateAccum >= 0.25f)
        {
            titleUpdateAccum = 0.0f;
            const std::string title =
                "Northstar Graphics - 01 Hello PBR  |  " +
                std::to_string(static_cast<UINT32>(stats.fps + 0.5f)) + " FPS (" +
                std::to_string(stats.frameTimeMs).substr(0, 5) + " ms)  |  Visible: " +
                std::to_string(stats.visibleObjects) + "  |  Draws: " +
                std::to_string(stats.drawCalls);
            window.SetTitle(title);
        }

        if (maxFrames > 0 && stats.frameIndex >= maxFrames)
        {
            std::cout << "[HelloPBR] Completed " << stats.frameIndex
                      << " frames | FrameTimeMs=" << stats.frameTimeMs
                      << " (" << stats.fps << " FPS)"
                      << " | GPUTotalMs=" << stats.gpuTotalFrameMs
                      << " | GPUShadowMs=" << stats.gpuShadowPassMs
                      << " | GPUGeomMs=" << stats.gpuGeometryPassMs
                      << " | GPUNVSFeedbackMs=" << stats.gpuNVSFeedbackMs
                      << " | GPUNVSCullMs=" << stats.gpuNVSResidencyCullMs
                      << " | RenderedTiles=" << stats.renderedTileCount << "/" << stats.logicalTileCount
                      << " | ShadowDraws=" << stats.shadowDrawCalls
                      << " | FenceWaitCpuMs=" << stats.fenceWaitCpuMs
                      << " | PresentCpuMs=" << stats.presentCpuMs << "\n";
            break;
        }
    }

    engine.WaitForGPU();
    return 0;
}
