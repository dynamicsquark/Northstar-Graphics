#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <northstar.h>
#include "ns_window.h"
#include "ns_resource_loader.h"

#include <windows.h>
#include <objbase.h>
#include <commdlg.h>
#include <d3d12.h>
#include "types/northstar_platform_guard.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace
{
    using namespace Northstar;

    // Isolated shader-visible SRV descriptor heap allocator for Dear ImGui D3D12
    struct ImGuiDescriptorAllocator
    {
        ID3D12DescriptorHeap*       heap           = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuStart       = {};
        D3D12_GPU_DESCRIPTOR_HANDLE gpuStart       = {};
        UINT                        descriptorSize = 0;
        std::vector<UINT>           freeIndices;

        bool Create(ID3D12Device* device, UINT capacity = 64u)
        {
            if (!device || capacity == 0u) return false;

            D3D12_DESCRIPTOR_HEAP_DESC desc{};
            desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = capacity;
            desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap))))
            {
                return false;
            }

            cpuStart       = heap->GetCPUDescriptorHandleForHeapStart();
            gpuStart       = heap->GetGPUDescriptorHandleForHeapStart();
            descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            freeIndices.clear();
            freeIndices.reserve(capacity);
            for (UINT i = capacity; i > 0u; --i)
            {
                freeIndices.push_back(i - 1u);
            }
            return true;
        }

        void Destroy()
        {
            if (heap)
            {
                heap->Release();
                heap = nullptr;
            }
            freeIndices.clear();
        }

        void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* outCpu, D3D12_GPU_DESCRIPTOR_HANDLE* outGpu)
        {
            const UINT idx = freeIndices.empty() ? 0u : freeIndices.back();
            if (!freeIndices.empty()) freeIndices.pop_back();
            outCpu->ptr = cpuStart.ptr + static_cast<SIZE_T>(idx) * descriptorSize;
            outGpu->ptr = gpuStart.ptr + static_cast<UINT64>(idx) * descriptorSize;
        }

        void Free(D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE)
        {
            if (descriptorSize > 0u && cpu.ptr >= cpuStart.ptr)
            {
                const UINT idx = static_cast<UINT>((cpu.ptr - cpuStart.ptr) / descriptorSize);
                freeIndices.push_back(idx);
            }
        }
    };

    void ApplyNorthstarDarkTheme()
    {
        ImGui::StyleColorsDark();
        ImGuiStyle& style      = ImGui::GetStyle();
        style.WindowRounding   = 6.0f;
        style.FrameRounding    = 4.0f;
        style.GrabRounding     = 4.0f;
        style.WindowBorderSize = 1.0f;
        style.FramePadding     = ImVec2(8.0f, 5.0f);

        ImVec4* colors                         = style.Colors;
        colors[ImGuiCol_WindowBg]              = ImVec4(0.08f, 0.09f, 0.12f, 0.94f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.06f, 0.07f, 0.09f, 1.00f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.10f, 0.13f, 0.18f, 1.00f);
        colors[ImGuiCol_Header]                = ImVec4(0.16f, 0.24f, 0.36f, 0.80f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.22f, 0.34f, 0.50f, 0.90f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.26f, 0.40f, 0.58f, 1.00f);
        colors[ImGuiCol_Button]                = ImVec4(0.16f, 0.26f, 0.40f, 0.90f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.22f, 0.36f, 0.55f, 1.00f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.14f, 0.44f, 0.68f, 1.00f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.13f, 0.15f, 0.20f, 1.00f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(0.28f, 0.56f, 0.88f, 1.00f);
        colors[ImGuiCol_CheckMark]             = ImVec4(0.34f, 0.68f, 0.98f, 1.00f);
    }

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

    UniqueTexture CreateGridFloorTexture2D(NorthstarEngine& engine, UINT32 size = 512u)
    {
        std::vector<BYTE> pixels(static_cast<SIZE_T>(size) * size * 4u);
        for (UINT32 y = 0; y < size; ++y)
        {
            for (UINT32 x = 0; x < size; ++x)
            {
                const bool majorLine = ((x % 64u) < 2u) || ((y % 64u) < 2u);
                const bool minorLine = ((x % 16u) == 0u) || ((y % 16u) == 0u);
                const BYTE val       = majorLine ? 125u : (minorLine ? 68u : 46u);
                const SIZE_T idx     = (static_cast<SIZE_T>(y) * size + x) * 4u;
                pixels[idx + 0]      = val;
                pixels[idx + 1]      = static_cast<BYTE>(val + 4u);
                pixels[idx + 2]      = static_cast<BYTE>(val + 10u);
                pixels[idx + 3]      = 255u;
            }
        }

        TextureDesc desc{};
        desc.dataSource      = TextureDataSource::MEMORY;
        desc.initialData     = pixels.data();
        desc.initialDataSize = pixels.size();
        desc.width           = size;
        desc.height          = size;
        desc.mipLevels       = 0;
        desc.format          = TextureFormat::RGBA8_SRGB;
        desc.type            = TextureType::TEXTURE_2D;
        desc.usage           = TextureUsage::SAMPLED;
        return MakeUniqueTexture(engine, desc);
    }

    // Procedural high-density Torus Knot mesh built directly as Northstar Virtual Geometry (NVG)
    UniqueMesh CreateVirtualizedTorusKnot(
        NorthstarEngine& engine,
        UINT32           tubularSegments = 512u,
        UINT32           radialSegments  = 64u,
        FLOAT32          p               = 2.0f,
        FLOAT32          q               = 3.0f,
        FLOAT32          majorScale      = 0.95f,
        FLOAT32          tubeRadius      = 0.26f)
    {
        std::vector<Vertex> vertices;
        std::vector<UINT32> indices;
        vertices.reserve(static_cast<SIZE_T>(tubularSegments + 1u) * (radialSegments + 1u));
        indices.reserve(static_cast<SIZE_T>(tubularSegments) * radialSegments * 6u);

        auto evalCurve = [&](FLOAT32 t) -> Vec3 {
            const FLOAT32 r = (2.0f + std::cos(q * t)) * 0.5f * majorScale;
            return Vec3(
                r * std::cos(p * t),
                (std::sin(q * t) * 0.55f) * majorScale,
                r * std::sin(p * t));
        };

        for (UINT32 i = 0; i <= tubularSegments; ++i)
        {
            const FLOAT32 u  = static_cast<FLOAT32>(i) / static_cast<FLOAT32>(tubularSegments);
            const FLOAT32 t0 = u * 2.0f * PI;
            const Vec3    p0 = evalCurve(t0);
            const Vec3    p1 = evalCurve(t0 + 0.0005f);
            const Vec3    T  = (p1 - p0).Normalized();
            Vec3          N  = T.Cross(Vec3::Up());
            if (N.LengthSq() < 1e-6f) N = T.Cross(Vec3::Right());
            N = N.Normalized();
            const Vec3 B = T.Cross(N).Normalized();

            for (UINT32 j = 0; j <= radialSegments; ++j)
            {
                const FLOAT32 v     = static_cast<FLOAT32>(j) / static_cast<FLOAT32>(radialSegments);
                const FLOAT32 angle = v * 2.0f * PI;
                const FLOAT32 ca    = std::cos(angle);
                const FLOAT32 sa    = std::sin(angle);

                const Vec3 normal    = (N * ca + B * sa).Normalized();
                const Vec3 tangent   = T;
                const Vec3 bitangent = (-N * sa + B * ca).Normalized();

                Vertex vert{};
                vert.position  = p0 + normal * tubeRadius;
                vert.normal    = normal;
                vert.tangent   = tangent;
                vert.bitangent = bitangent;
                vert.texCoord0 = Vec2(u * 12.0f, v * 2.0f);
                vert.texCoord1 = Vec2(u, v);
                vert.color     = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
                vertices.push_back(vert);
            }
        }

        const UINT32 stride = radialSegments + 1u;
        for (UINT32 i = 0; i < tubularSegments; ++i)
        {
            for (UINT32 j = 0; j < radialSegments; ++j)
            {
                const UINT32 i0 = i * stride + j;
                const UINT32 i1 = i0 + 1u;
                const UINT32 i2 = (i + 1u) * stride + j;
                const UINT32 i3 = i2 + 1u;

                // Clockwise front-facing winding (DirectX LH: (i1 - i0) x (i2 - i0) = +normal)
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
        desc.geometryMode = GeometryMode::Virtualized;
        return MakeUniqueMesh(engine, desc);
    }

    UniqueMesh CreateGroundMesh(NorthstarEngine& engine, FLOAT32 halfExtent, FLOAT32 uvScale)
    {
        Vertex verts[4]{};
        const Vec3 pos[4] = {
            Vec3(-halfExtent, 0.0f, -halfExtent),
            Vec3(-halfExtent, 0.0f,  halfExtent),
            Vec3( halfExtent, 0.0f,  halfExtent),
            Vec3( halfExtent, 0.0f, -halfExtent)
        };
        const Vec2 uv[4] = {
            Vec2(0.0f,    uvScale),
            Vec2(0.0f,    0.0f),
            Vec2(uvScale, 0.0f),
            Vec2(uvScale, uvScale)
        };
        for (UINT32 i = 0; i < 4u; ++i)
        {
            verts[i].position  = pos[i];
            verts[i].normal    = Vec3::Up();
            verts[i].tangent   = Vec3::Right();
            verts[i].bitangent = Vec3::Forward();
            verts[i].texCoord0 = uv[i];
            verts[i].texCoord1 = uv[i];
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

    UniqueMesh ImportVirtualizedAssimpMesh(NorthstarEngine& engine, const std::string& filePath)
    {
        Assimp::Importer importer;
        const unsigned int flags =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ConvertToLeftHanded;

        const aiScene* aiScn = importer.ReadFile(filePath, flags);
        if (!aiScn || !aiScn->HasMeshes())
        {
            std::cerr << "[Studio] Assimp failed to load '" << filePath << "': " << importer.GetErrorString() << "\n";
            return {};
        }

        std::vector<Vertex> vertices;
        std::vector<UINT32> indices;

        for (unsigned int m = 0; m < aiScn->mNumMeshes; ++m)
        {
            const aiMesh* sub = aiScn->mMeshes[m];
            if (!sub || sub->mNumVertices == 0 || sub->mNumFaces == 0) continue;

            const UINT32 baseVertex = static_cast<UINT32>(vertices.size());
            vertices.reserve(vertices.size() + sub->mNumVertices);

            for (unsigned int v = 0; v < sub->mNumVertices; ++v)
            {
                Vertex out{};
                out.position  = Vec3(sub->mVertices[v].x, sub->mVertices[v].y, sub->mVertices[v].z);
                out.normal    = sub->HasNormals()
                    ? Vec3(sub->mNormals[v].x, sub->mNormals[v].y, sub->mNormals[v].z).Normalized()
                    : Vec3::Up();
                out.tangent   = sub->HasTangentsAndBitangents()
                    ? Vec3(sub->mTangents[v].x, sub->mTangents[v].y, sub->mTangents[v].z).Normalized()
                    : Vec3::Right();
                out.bitangent = sub->HasTangentsAndBitangents()
                    ? Vec3(sub->mBitangents[v].x, sub->mBitangents[v].y, sub->mBitangents[v].z).Normalized()
                    : Vec3::Forward();
                out.texCoord0 = sub->HasTextureCoords(0)
                    ? Vec2(sub->mTextureCoords[0][v].x, sub->mTextureCoords[0][v].y)
                    : Vec2(0.0f, 0.0f);
                out.texCoord1 = out.texCoord0;
                out.color     = Vec4(1.0f, 1.0f, 1.0f, 1.0f);
                vertices.push_back(out);
            }

            for (unsigned int f = 0; f < sub->mNumFaces; ++f)
            {
                const aiFace& face = sub->mFaces[f];
                if (face.mNumIndices == 3u)
                {
                    indices.push_back(baseVertex + face.mIndices[0]);
                    indices.push_back(baseVertex + face.mIndices[1]);
                    indices.push_back(baseVertex + face.mIndices[2]);
                }
            }
        }

        if (vertices.empty() || indices.empty())
        {
            return {};
        }

        // Normalize imported asset bounds to ~3.0 world units height for comfortable inspection
        AABB box(vertices[0].position, vertices[0].position);
        for (const auto& v : vertices)
        {
            box.Expand(v.position);
        }
        const Vec3    ext    = box.Size();
        const FLOAT32 maxDim = std::max({ ext.x, ext.y, ext.z, 1e-4f });
        const FLOAT32 scale  = 3.2f / maxDim;
        const Vec3    center = box.Center();

        for (auto& v : vertices)
        {
            v.position = (v.position - Vec3(center.x, box.m_Min.y, center.z)) * scale;
        }

        MeshDesc desc{};
        desc.vertices     = vertices.data();
        desc.vertexCount  = static_cast<UINT32>(vertices.size());
        desc.indices      = indices.data();
        desc.indexCount   = static_cast<UINT32>(indices.size());
        desc.geometryMode = GeometryMode::Virtualized;
        return MakeUniqueMesh(engine, desc);
    }

    std::string OpenModelFileDialog(HWND ownerHwnd)
    {
        char filePath[1024]{};
        OPENFILENAMEA ofn{};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner   = ownerHwnd;
        ofn.lpstrFile   = filePath;
        ofn.nMaxFile    = sizeof(filePath);
        ofn.lpstrFilter = "3D Models (*.obj;*.fbx;*.gltf;*.glb;*.ply)\0*.obj;*.fbx;*.gltf;*.glb;*.ply\0All Files (*.*)\0*.*\0";
        ofn.nFilterIndex = 1;
        ofn.Flags       = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
        if (::GetOpenFileNameA(&ofn))
        {
            return std::string(filePath);
        }
        return {};
    }

    UniqueShader LoadCompiledShader(NorthstarEngine& engine, const std::filesystem::path& path, const std::string& name)
    {
        const auto loaded = Tools::ResourceLoader::LoadShaderFromFile(path.wstring(), name);
        if (!loaded.isValid)
        {
            std::cerr << "[Studio] Failed to load shader: " << path.string() << "\n";
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

    // 1. Create Studio Window
    Tools::WindowDesc winDesc{};
    winDesc.width          = 1600;
    winDesc.height         = 900;
    winDesc.title          = L"Northstar Graphics Studio - Virtual Geometry (NVG) & Virtual Shadows (NVS)";
    winDesc.enableDarkMode = true;

    Tools::Window window;
    if (!window.Create(winDesc))
    {
        std::cerr << "[Studio] Failed to create window.\n";
        return 1;
    }

    // 2. Initialize Northstar Runtime Session
    RuntimeConfig rtCfg{};
    rtCfg.backend      = BackendAPI::D3D12;
    rtCfg.renderTarget = window.GetNativeRenderTarget();
    rtCfg.vsync        = false;

    RuntimeSession runtime;
    if (!runtime.Initialize(rtCfg))
    {
        std::cerr << "[Studio] Runtime initialization failed: " << runtime.GetLastError() << "\n";
        return 1;
    }

    NorthstarEngine& engine = runtime.GetEngine();

    // 3. Initialize Dear ImGui over NativeContext (isolated shader-visible SRV heap)
    NativeContext nativeCtx = engine.GetNativeContext();
    auto* d3d12Device       = static_cast<ID3D12Device*>(nativeCtx.nativeDevice);
    auto* d3d12Queue        = static_cast<ID3D12CommandQueue*>(nativeCtx.nativeCommandQueue);

    ImGuiDescriptorAllocator imguiHeap;
    if (!imguiHeap.Create(d3d12Device, 64u))
    {
        std::cerr << "[Studio] Failed to allocate ImGui D3D12 SRV descriptor heap.\n";
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io    = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ApplyNorthstarDarkTheme();

    ImGui_ImplWin32_Init(window.GetHWND());

    ImGui_ImplDX12_InitInfo dx12Info{};
    dx12Info.Device            = d3d12Device;
    dx12Info.CommandQueue      = d3d12Queue;
    dx12Info.NumFramesInFlight = static_cast<int>(nativeCtx.backBufferCount > 0 ? nativeCtx.backBufferCount : 2);
    dx12Info.RTVFormat         = (nativeCtx.renderTargetFormat != 0)
        ? static_cast<DXGI_FORMAT>(nativeCtx.renderTargetFormat)
        : DXGI_FORMAT_R8G8B8A8_UNORM;
    dx12Info.DSVFormat         = DXGI_FORMAT_UNKNOWN;
    dx12Info.UserData          = &imguiHeap;
    dx12Info.SrvDescriptorHeap = imguiHeap.heap;
    dx12Info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE* outCpu, D3D12_GPU_DESCRIPTOR_HANDLE* outGpu) {
        static_cast<ImGuiDescriptorAllocator*>(info->UserData)->Alloc(outCpu, outGpu);
    };
    dx12Info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* info, D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE gpu) {
        static_cast<ImGuiDescriptorAllocator*>(info->UserData)->Free(cpu, gpu);
    };
    ImGui_ImplDX12_Init(&dx12Info);

    engine.SetPrePresentCallback([&imguiHeap](void* nativeCmdList) {
        ImDrawData* drawData = ImGui::GetDrawData();
        if (drawData && nativeCmdList && imguiHeap.heap)
        {
            auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(nativeCmdList);
            ID3D12DescriptorHeap* heaps[] = { imguiHeap.heap };
            cmdList->SetDescriptorHeaps(1, heaps);
            ImGui_ImplDX12_RenderDrawData(drawData, cmdList);
        }
    });

    // Camera & input state
    Vec3    camPos(0.0f, 5.5f, -15.5f);
    FLOAT32 camYawDeg   = 0.0f;
    FLOAT32 camPitchDeg = 16.0f;
    FLOAT32 moveSpeed   = 12.0f;
    bool    rmbLooking  = false;
    INT32   lastMouseX  = 0;
    INT32   lastMouseY  = 0;
    bool    keyState[256]{};

    window.SetNativeMessageHook([](void* hwnd, UINT32 msg, std::uintptr_t wParam, std::intptr_t lParam) -> std::optional<std::intptr_t> {
        if (ImGui_ImplWin32_WndProcHandler(static_cast<HWND>(hwnd), msg, static_cast<WPARAM>(wParam), static_cast<LPARAM>(lParam)))
        {
            return 1;
        }
        return std::nullopt;
    });

    window.SetOnResize([&engine](UINT32 w, UINT32 h) {
        engine.OnResize(w, h);
    });
    window.SetOnKey([&](Tools::KeyCode key, bool isDown, bool) {
        const auto idx = static_cast<SIZE_T>(key);
        if (idx < 256u) keyState[idx] = isDown;
        if (key == Tools::KeyCode::Escape && isDown) window.Close();
    });
    window.SetOnMouseButton([&](Tools::MouseButton button, bool isDown, INT32 x, INT32 y) {
        if (button == Tools::MouseButton::Right)
        {
            rmbLooking = isDown;
            lastMouseX = x;
            lastMouseY = y;
        }
    });
    window.SetOnMouseMove([&](INT32 x, INT32 y) {
        if (rmbLooking)
        {
            const INT32 dx = x - lastMouseX;
            const INT32 dy = y - lastMouseY;
            camYawDeg   += static_cast<FLOAT32>(dx) * 0.18f;
            camPitchDeg  = std::clamp(camPitchDeg + static_cast<FLOAT32>(dy) * 0.18f, -85.0f, 85.0f);
        }
        lastMouseX = x;
        lastMouseY = y;
    });
    window.SetOnMouseWheel([&](FLOAT32 delta, INT32, INT32) {
        if (!ImGui::GetIO().WantCaptureMouse)
        {
            moveSpeed = std::clamp(moveSpeed * (delta > 0.0f ? 1.15f : 0.87f), 1.5f, 80.0f);
        }
    });

    // 4. Load Unified IA + NVG Hybrid StandardLit Shaders (one shared 4-variant ShaderSet across STANDARD & VIRTUALIZED meshes)
    const std::filesystem::path shaderDir = RuntimeSession::ResolveExecutableDirectory() / "shaders" / "StandardLit";
    auto vsStd             = LoadCompiledShader(engine, shaderDir / "StandardLitVS_hlsl12.nsc", "StandardLitVS");
    auto vsInst            = LoadCompiledShader(engine, shaderDir / "StandardLitInstancedVS_hlsl12.nsc", "StandardLitInstancedVS");
    auto vsSkin            = LoadCompiledShader(engine, shaderDir / "StandardLitSkinnedVS_hlsl12.nsc", "StandardLitSkinnedVS");
    auto vsInstSkin        = LoadCompiledShader(engine, shaderDir / "StandardLitInstancedSkinnedVS_hlsl12.nsc", "StandardLitInstancedSkinnedVS");
    auto psStd             = LoadCompiledShader(engine, shaderDir / "StandardLitPS_hlsl12.nsc", "StandardLitPS");
    auto shadowVsStd       = LoadCompiledShader(engine, shaderDir / "StandardLitShadowVS_hlsl12.nsc", "StandardLitShadowVS");
    auto shadowVsInst      = LoadCompiledShader(engine, shaderDir / "StandardLitShadowInstancedVS_hlsl12.nsc", "StandardLitShadowInstancedVS");
    auto shadowVsSkin      = LoadCompiledShader(engine, shaderDir / "StandardLitShadowSkinnedVS_hlsl12.nsc", "StandardLitShadowSkinnedVS");
    auto shadowVsInstSkin  = LoadCompiledShader(engine, shaderDir / "StandardLitShadowInstancedSkinnedVS_hlsl12.nsc", "StandardLitShadowInstancedSkinnedVS");

    if (!vsStd || !vsInst || !vsSkin || !vsInstSkin || !psStd ||
        !shadowVsStd || !shadowVsInst || !shadowVsSkin || !shadowVsInstSkin)
    {
        std::cerr << "[Studio] Missing compiled StandardLit shaders in '" << shaderDir.string() << "'.\n";
        return 1;
    }

    ShaderSetDesc geomSetDesc{};
    geomSetDesc.name                         = "StudioUnifiedStandardLitGeometry";
    geomSetDesc.vertexShader                 = vsStd.get();
    geomSetDesc.instancedVertexShader        = vsInst.get();
    geomSetDesc.skinnedVertexShader          = vsSkin.get();
    geomSetDesc.instancedSkinnedVertexShader = vsInstSkin.get();
    geomSetDesc.pixelShader                  = psStd.get();
    auto geomShaderSet = MakeUniqueShaderSet(engine, geomSetDesc);

    ShaderSetDesc shadowSetDesc{};
    shadowSetDesc.name                         = "StudioUnifiedStandardLitShadow";
    shadowSetDesc.vertexShader                 = shadowVsStd.get();
    shadowSetDesc.instancedVertexShader        = shadowVsInst.get();
    shadowSetDesc.skinnedVertexShader          = shadowVsSkin.get();
    shadowSetDesc.instancedSkinnedVertexShader = shadowVsInstSkin.get();
    auto shadowShaderSet = MakeUniqueShaderSet(engine, shadowSetDesc);

    auto whiteTex   = CreateSolidTexture2D(engine, 255, 255, 255, 255, true);
    auto flatNrmTex = CreateSolidTexture2D(engine, 128, 128, 255, 255, false);
    auto blackTex   = CreateSolidTexture2D(engine, 0, 0, 0, 255, true);
    auto floorTex   = CreateGridFloorTexture2D(engine, 512u);
    auto sampler    = MakeUniqueSampler(engine, SamplerDesc::AnisotropicWrap(16));

    auto configureStudioMaterial = [&](Material& mat, Texture* baseMap, bool useMap, const Vec4& albedo, FLOAT32 metallic, FLOAT32 roughness) {
        mat.SetTexture("BaseColorMap", baseMap);
        mat.SetTexture("NormalMap", flatNrmTex.get());
        mat.SetTexture("MetallicRoughnessMap", whiteTex.get());
        mat.SetTexture("EmissiveMap", blackTex.get());
        mat.SetTexture("AOMap", whiteTex.get());
        mat.SetSampler("LinearWrapSampler", sampler.get());
        mat.SetVec4("baseColor", albedo);
        mat.SetFloat("metallic", metallic);
        mat.SetFloat("roughness", roughness);
        mat.SetBool("useBaseColorMap", useMap);
        mat.SetBool("useNormalMap", false);
        mat.SetBool("useRoughnessMap", false);
        mat.SetBool("useEmissiveMap", false);
        mat.SetBool("useAOMap", false);
        mat.SetBool("receiveShadows", true);
    };

    // 5. Configure NVG & Build Virtualized Gallery Scene
    NVGConfig nvgCfg = engine.GetNVGConfig();
    nvgCfg.enabled          = true;
    nvgCfg.qualityPreset    = NVGQualityPreset::Balanced;
    nvgCfg.memoryBudget     = NVGMemoryBudget::Auto512MB;
    engine.SetNVGConfig(nvgCfg);

    auto knotMesh   = CreateVirtualizedTorusKnot(engine, 384u, 54u, 2.0f, 3.0f, 1.05f, 0.27f); // 41,472 tris/instance (~1.04M total)
    auto groundMesh = CreateGroundMesh(engine, 60.0f, 24.0f);
    UniqueMesh importedMesh;

    MaterialDesc matDesc{};
    matDesc.geometryShaderSet = geomShaderSet.get();
    matDesc.shadowShaderSet   = shadowShaderSet.get();
    matDesc.cullMode          = CullMode::Back;
    matDesc.blendMode         = BlendMode::Opaque;

    auto floorMat = MakeUniqueMaterial(engine, matDesc);
    configureStudioMaterial(*floorMat, floorTex.get(), true, Vec4(1.0f, 1.0f, 1.0f, 1.0f), 0.05f, 0.80f);

    constexpr SIZE_T kMaterialCount = 4;
    std::vector<UniqueMaterial> galleryMaterials;
    const Vec4 palette[kMaterialCount] = {
        Vec4(0.94f, 0.77f, 0.42f, 1.0f),
        Vec4(0.86f, 0.89f, 0.94f, 1.0f),
        Vec4(0.24f, 0.66f, 0.94f, 1.0f),
        Vec4(0.92f, 0.32f, 0.26f, 1.0f)
    };
    for (SIZE_T i = 0; i < kMaterialCount; ++i)
    {
        auto m = MakeUniqueMaterial(engine, matDesc);
        configureStudioMaterial(*m, whiteTex.get(), false, palette[i], (i < 2) ? 0.88f : 0.12f, 0.16f + 0.18f * static_cast<FLOAT32>(i));
        galleryMaterials.push_back(std::move(m));
    }

    auto scene = MakeUniqueScene(engine);
    scene->SetAmbientColor(Color(0.16f, 0.20f, 0.28f, 1.0f));
    scene->SetAmbientIntensity(0.90f);

    RenderObjectDesc floorObjDesc{};
    floorObjDesc.mesh     = groundMesh.get();
    floorObjDesc.material = floorMat.get();
    auto floorObj = MakeUniqueRenderObject(engine, floorObjDesc);
    scene->AddRenderObject(floorObj.get());

    std::vector<UniqueRenderObject> virtualObjects;
    auto rebuildVirtualGrid = [&](Mesh* activeMesh, INT32 gridRadius) {
        for (auto& obj : virtualObjects)
        {
            scene->RemoveRenderObject(obj.get());
        }
        virtualObjects.clear();
        if (!activeMesh) return;

        UINT32 idx = 0;
        for (INT32 gz = -gridRadius; gz <= gridRadius; ++gz)
        {
            for (INT32 gx = -gridRadius; gx <= gridRadius; ++gx)
            {
                RenderObjectDesc roDesc{};
                roDesc.mesh     = activeMesh;
                roDesc.material = galleryMaterials[idx % kMaterialCount].get();
                auto ro = MakeUniqueRenderObject(engine, roDesc);
                ro->SetTransform(Vec3(static_cast<FLOAT32>(gx) * 3.8f, 1.55f, static_cast<FLOAT32>(gz) * 3.8f));
                scene->AddRenderObject(ro.get());
                virtualObjects.push_back(std::move(ro));
                ++idx;
            }
        }
    };

    INT32 gridRadius = 2; // 5x5 = 25 virtualized instances (~1.64M source triangles)
    rebuildVirtualGrid(knotMesh.get(), gridRadius);

    // Lighting (Directional Sun + 2 Accent Point Lights — cached in NVS unless animated)
    FLOAT32 sunAzimuthDeg   = 38.0f;
    FLOAT32 sunElevationDeg = 48.0f;
    int     shadowSoftness  = static_cast<int>(LightShadowSoftness::MediumHard);
    bool    animateObjects  = false;
    bool    enablePtLights  = false;
    bool    animateLights   = false;
    bool    vsyncEnabled    = false;

    const auto computeSunDirection = [](FLOAT32 elevDeg, FLOAT32 azimDeg) -> Vec3 {
        const FLOAT32 elRad = elevDeg * DEG2RAD;
        const FLOAT32 azRad = azimDeg * DEG2RAD;
        return Vec3(
            -std::cos(elRad) * std::sin(azRad),
            -std::sin(elRad),
            -std::cos(elRad) * std::cos(azRad));
    };

    LightDesc sunDesc{};
    sunDesc.type              = LightType::Directional;
    sunDesc.color             = Color(1.0f, 0.96f, 0.90f, 1.0f);
    sunDesc.intensity         = 3.4f;
    sunDesc.shadowResolution  = LightShadowResolution::Res16384;
    sunDesc.shadowSoftness    = LightShadowSoftness::MediumHard;
    sunDesc.shadowMaxDistance = 140.0f;
    auto sunLight = MakeUniqueLight(engine, sunDesc);
    sunLight->SetDirection(computeSunDirection(sunElevationDeg, sunAzimuthDeg));
    scene->AddLight(sunLight.get());
    scene->SetDirectionalLight(sunLight.get());

    LightDesc ptDesc{};
    ptDesc.type             = LightType::Point;
    ptDesc.intensity        = 18.0f;
    ptDesc.range            = 20.0f;
    ptDesc.shadowResolution = LightShadowResolution::Res16384;
    ptDesc.shadowSoftness   = LightShadowSoftness::Hard;
    ptDesc.color            = Color(0.28f, 0.74f, 1.0f, 1.0f);
    ptDesc.position         = Vec3(8.5f, 3.8f, -6.5f);
    auto fillLightA         = MakeUniqueLight(engine, ptDesc);
    fillLightA->SetEnabled(enablePtLights);
    scene->AddLight(fillLightA.get());

    ptDesc.color            = Color(1.0f, 0.50f, 0.20f, 1.0f);
    ptDesc.position         = Vec3(-8.5f, 3.8f, 6.5f);
    auto fillLightB         = MakeUniqueLight(engine, ptDesc);
    fillLightB->SetEnabled(enablePtLights);
    scene->AddLight(fillLightB.get());

    CameraDesc camDesc{};
    camDesc.fov         = 60.0f;
    camDesc.aspectRatio = static_cast<FLOAT32>(window.GetWidth()) / static_cast<FLOAT32>(window.GetHeight());
    camDesc.nearPlane   = 0.1f;
    camDesc.farPlane    = 500.0f;
    auto camera = MakeUniqueCamera(engine, camDesc);
    scene->AddCamera(camera.get());
    scene->SetMainCamera(camera.get());

    engine.SetActiveScene(scene.get());

    int qualityIdx = 2; // Balanced (1.5px)
    int budgetIdx  = 2; // 512 MB
    int debugMode  = 0; // Lit PBR

    // 6. Main Studio Loop
    while (window.IsRunning())
    {
        window.PollMessages();
        if (!window.IsRunning())
        {
            break;
        }

        // Snapshot completed previous-frame telemetry BEFORE BeginFrame() resets per-frame counters
        const RenderStats prevStats = engine.GetStats();

        engine.BeginFrame();
        const RenderStats& frameTiming = engine.GetStats();
        const FLOAT32 dt = frameTiming.deltaTimeSeconds;
        const FLOAT32 t  = static_cast<FLOAT32>(frameTiming.elapsedTimeSeconds);

        // Free-fly camera movement
        const Quat camRot = Quat::FromEuler(camPitchDeg * DEG2RAD, camYawDeg * DEG2RAD, 0.0f);
        if (!ImGui::GetIO().WantCaptureKeyboard)
        {
            const FLOAT32 boost = (keyState[static_cast<SIZE_T>(Tools::KeyCode::LeftShift)] ? 2.5f : 1.0f) * moveSpeed * dt;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::W)]) camPos = camPos + camRot.GetForward() * boost;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::S)]) camPos = camPos - camRot.GetForward() * boost;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::D)]) camPos = camPos + camRot.GetRight() * boost;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::A)]) camPos = camPos - camRot.GetRight() * boost;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::E)]) camPos = camPos + Vec3::Up() * boost;
            if (keyState[static_cast<SIZE_T>(Tools::KeyCode::Q)]) camPos = camPos - Vec3::Up() * boost;
        }
        camera->SetTransform(camPos, camRot);

        if (animateLights)
        {
            fillLightA->SetPosition(Vec3(std::cos(t * 0.6f) * 9.5f, 3.6f, std::sin(t * 0.6f) * 9.5f));
            fillLightB->SetPosition(Vec3(-std::cos(t * 0.5f) * 9.5f, 3.6f, -std::sin(t * 0.5f) * 9.5f));
        }

        if (animateObjects && !importedMesh)
        {
            for (SIZE_T i = 0; i < virtualObjects.size(); ++i)
            {
                const Mat4& cur = virtualObjects[i]->GetWorldMatrix();
                const Vec3 pos(cur.m[12], cur.m[13], cur.m[14]);
                const Quat rot = Quat::FromEuler(0.0f, t * 0.45f + static_cast<FLOAT32>(i) * 0.2f, 0.0f);
                virtualObjects[i]->SetTransform(pos, rot);
            }
        }

        // Build Dear ImGui Overlay
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(410.0f, 620.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Northstar Graphics Studio"))
        {
            const NVGStats& nvg = prevStats.nvgStats;
            ImGui::Text("Northstar v%s  |  D3D12 Hardware Runtime", VERSION_STRING);
            ImGui::Text("Frame: %.2f ms (%.0f FPS)  |  GPU Total: %.2f ms",
                        prevStats.frameTimeMs, prevStats.fps, prevStats.gpuTotalFrameMs);
            ImGui::Separator();

            if (ImGui::CollapsingHeader("Virtual Geometry (NVG) & Shadow Telemetry", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Text("Effective Scene Triangles : %llu", static_cast<unsigned long long>(nvg.effectiveSceneTriangles));
                ImGui::Text("Clusters Tested / Visible : %u / %u", nvg.clustersTested, nvg.visibleOpaqueClusters);
                ImGui::Text("HZB / Frustum / Cone Cull : %u / %u / %u",
                            nvg.clustersHZBCulled, nvg.clustersFrustumCulled, nvg.clustersConeCulled);
                ImGui::Text("Resident / Virtual Pages  : %u / %u (%.1f MB)",
                            nvg.residentPhysicalPages, nvg.totalVirtualPages, nvg.geometryPoolVramMB);
                ImGui::Text("Shadow Clusters / Tiles   : %u / %u",
                            nvg.shadowClustersVisible, prevStats.renderedTileCount);
                ImGui::Text("GPU Cull / Geom / Shadow  : %.2f / %.2f / %.2f ms",
                            nvg.gpuCullAndLODMs, prevStats.gpuGeometryPassMs, prevStats.gpuShadowPassMs);
            }

            if (ImGui::CollapsingHeader("Virtual Geometry Controls", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool cfgChanged = false;
                const char* qualities[] = { "Ultra (0.5px)", "High (1.0px)", "Balanced (1.5px)", "Performance (2.25px)" };
                if (ImGui::Combo("Quality Preset", &qualityIdx, qualities, 4))
                {
                    nvgCfg.qualityPreset = static_cast<NVGQualityPreset>(qualityIdx);
                    cfgChanged = true;
                }

                const char* budgets[] = { "128 MB", "256 MB", "512 MB", "1024 MB" };
                if (ImGui::Combo("Streaming Pool", &budgetIdx, budgets, 4))
                {
                    const NVGMemoryBudget mapB[4] = {
                        NVGMemoryBudget::Auto128MB, NVGMemoryBudget::Auto256MB,
                        NVGMemoryBudget::Auto512MB, NVGMemoryBudget::Auto1024MB
                    };
                    nvgCfg.memoryBudget = mapB[budgetIdx];
                    cfgChanged = true;
                }

                const char* debugModes[] = {
                    "Lit Physically-Based",
                    "Cluster ID",
                    "LOD Level",
                    "Virtual Page",
                    "Physical Page",
                    "Geometric Error",
                    "Residency Fallback",
                    "Normal Cone"
                };
                if (ImGui::Combo("Visualization", &debugMode, debugModes, 8))
                {
                    nvgCfg.debugVisualizationMode = static_cast<NVGDebugVisualizationMode>(debugMode);
                    cfgChanged = true;
                    floorMat->SetInt("nvgDebugActive", debugMode);
                    for (auto& m : galleryMaterials)
                    {
                        m->SetInt("nvgDebugActive", debugMode);
                    }
                }

                bool hzb = nvgCfg.enableHZBOcclusion;
                if (ImGui::Checkbox("Two-Phase HZB Occlusion Culling", &hzb))
                {
                    nvgCfg.enableHZBOcclusion = hzb;
                    cfgChanged = true;
                }

                if (cfgChanged)
                {
                    engine.SetNVGConfig(nvgCfg);
                }
            }

            if (ImGui::CollapsingHeader("Scene, Assets & Virtual Shadows", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (ImGui::Checkbox("VSync", &vsyncEnabled))
                {
                    engine.SetVSync(vsyncEnabled);
                }
                ImGui::SameLine();
                ImGui::Checkbox("Rotate Instances", &animateObjects);
                if (ImGui::Checkbox("Accent Point Lights (2x 16K)", &enablePtLights))
                {
                    fillLightA->SetEnabled(enablePtLights);
                    fillLightB->SetEnabled(enablePtLights);
                }
                ImGui::SameLine();
                ImGui::Checkbox("Orbit Point Lights", &animateLights);

                if (ImGui::SliderInt("Instance Grid Radius", &gridRadius, 0, 5))
                {
                    rebuildVirtualGrid(importedMesh ? importedMesh.get() : knotMesh.get(), gridRadius);
                }

                if (ImGui::Button("Import 3D Model (.obj / .fbx / .gltf)..."))
                {
                    const std::string chosen = OpenModelFileDialog(window.GetHWND());
                    if (!chosen.empty())
                    {
                        engine.WaitForGPU();
                        if (auto loadedMesh = ImportVirtualizedAssimpMesh(engine, chosen))
                        {
                            importedMesh = std::move(loadedMesh);
                            gridRadius   = 0;
                            rebuildVirtualGrid(importedMesh.get(), gridRadius);
                        }
                    }
                }
                if (importedMesh)
                {
                    ImGui::SameLine();
                    if (ImGui::Button("Reset to Procedural"))
                    {
                        engine.WaitForGPU();
                        rebuildVirtualGrid(knotMesh.get(), gridRadius);
                        importedMesh.reset();
                    }
                }

                bool sunChanged = false;
                sunChanged |= ImGui::SliderFloat("Sun Azimuth", &sunAzimuthDeg, -180.0f, 180.0f);
                sunChanged |= ImGui::SliderFloat("Sun Elevation", &sunElevationDeg, 8.0f, 85.0f);
                if (sunChanged)
                {
                    sunLight->SetDirection(computeSunDirection(sunElevationDeg, sunAzimuthDeg));
                }
                const char* softModes[] = { "Hard", "Medium Hard", "Medium", "Medium Soft", "Soft (Vogel 16-Tap)" };
                if (ImGui::Combo("NVS Softness", &shadowSoftness, softModes, 5))
                {
                    sunLight->SetShadowSoftness(static_cast<LightShadowSoftness>(shadowSoftness));
                }
            }

            ImGui::Separator();
            ImGui::TextDisabled("Controls: Hold RMB + WASD/QE to fly, Scroll = Speed");
        }
        ImGui::End();

        ImGui::Render();

        engine.RenderFrame();
        engine.EndFrame();

        const RenderStats& stats = engine.GetStats();
        if (maxFrames > 0 && stats.frameIndex >= maxFrames)
        {
            std::cout << "[Studio] Completed " << stats.frameIndex
                      << " frames | FrameTimeMs=" << stats.frameTimeMs
                      << " (" << stats.fps << " FPS)"
                      << " | GPUTotalMs=" << stats.gpuTotalFrameMs
                      << " | GPUShadowMs=" << stats.gpuShadowPassMs
                      << " | GPUDepthMs=" << stats.gpuDepthPrepassMs
                      << " | GPUGeomMs=" << stats.gpuGeometryPassMs
                      << " | NVGCullMs=" << stats.nvgStats.gpuCullAndLODMs
                      << " | NVGDecodeMs=" << stats.nvgStats.gpuDecodeAndDrawGenMs
                      << " | GPUNVSFeedbackMs=" << stats.gpuNVSFeedbackMs
                      << " | GPUNVSCullMs=" << stats.gpuNVSResidencyCullMs
                      << " | RenderedTiles=" << stats.renderedTileCount << "/" << stats.logicalTileCount
                      << " | ShadowClusters=" << stats.nvgStats.shadowClustersVisible
                      << " | DecodedClusters=" << stats.nvgStats.decodedUniqueClusters
                      << " | FenceWaitCpuMs=" << stats.fenceWaitCpuMs
                      << " | PresentCpuMs=" << stats.presentCpuMs << "\n";
            break;
        }
    }

    // 7. Deterministic Shutdown
    engine.WaitForGPU();
    engine.SetPrePresentCallback(nullptr);
    window.SetNativeMessageHook(nullptr);

    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    imguiHeap.Destroy();

    return 0;
}
