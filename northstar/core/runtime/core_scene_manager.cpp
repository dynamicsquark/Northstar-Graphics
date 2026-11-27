#include "core_scene_manager.h"
#include <algorithm>
#include <cmath>

namespace Northstar
{
    CoreSceneManager::~CoreSceneManager()
    {
        Shutdown();
    }

    Camera* CoreSceneManager::CreateCamera(const CameraDesc& desc)
    {
        if (!std::isfinite(desc.position.x) || !std::isfinite(desc.position.y) || !std::isfinite(desc.position.z) ||
            !std::isfinite(desc.rotation.x) || !std::isfinite(desc.rotation.y) || !std::isfinite(desc.rotation.z) || !std::isfinite(desc.rotation.w) ||
            !std::isfinite(desc.fov) || !std::isfinite(desc.aspectRatio) ||
            !std::isfinite(desc.nearPlane) || !std::isfinite(desc.farPlane) ||
            !std::isfinite(desc.orthoHeight) || !std::isfinite(desc.depth) ||
            !std::isfinite(desc.viewportX) || !std::isfinite(desc.viewportY) ||
            !std::isfinite(desc.viewportWidth) || !std::isfinite(desc.viewportHeight))
        {
            return nullptr;
        }

        const FLOAT32 rotLenSq =
            desc.rotation.x * desc.rotation.x +
            desc.rotation.y * desc.rotation.y +
            desc.rotation.z * desc.rotation.z +
            desc.rotation.w * desc.rotation.w;
        if (rotLenSq <= 1e-12f)
        {
            return nullptr;
        }

        if (desc.farPlane <= desc.nearPlane + 1e-5f || desc.aspectRatio <= 1e-4f)
        {
            return nullptr;
        }

        if (desc.projectionType == ProjectionType::PERSPECTIVE)
        {
            if (desc.nearPlane <= 1e-5f || desc.fov <= 0.1f || desc.fov >= 179.9f)
            {
                return nullptr;
            }
        }
        else if (desc.projectionType == ProjectionType::ORTHOGRAPHIC)
        {
            if (desc.orthoHeight <= 1e-4f)
            {
                return nullptr;
            }
        }
        else
        {
            return nullptr;
        }

        if (desc.viewportWidth < 0.0f || desc.viewportHeight < 0.0f)
        {
            return nullptr;
        }

        Camera* camera = new Camera();
        camera->SetRenderType(desc.renderType);
        camera->SetPosition(desc.position);
        camera->SetRotation(desc.rotation);
        camera->SetProjectionType(desc.projectionType);
        camera->SetFOV(desc.fov);
        camera->SetAspectRatio(desc.aspectRatio);
        camera->SetNearPlane(desc.nearPlane);
        camera->SetFarPlane(desc.farPlane);
        camera->SetOrthoHeight(desc.orthoHeight);
        camera->SetDepth(desc.depth);
        camera->SetCullingMask(desc.cullingMask);
        camera->SetRenderTarget(desc.renderTarget);
        camera->SetDepthStencilTarget(desc.depthStencilTarget);
        camera->SetViewport(desc.viewportX, desc.viewportY, desc.viewportWidth, desc.viewportHeight);

        m_cameras.push_back(camera);
        return camera;
    }

    BOOL CoreSceneManager::IsManagedCamera(const Camera* camera) const
    {
        if (!camera) return false;
        return std::find(m_cameras.begin(), m_cameras.end(), camera) != m_cameras.end();
    }

    void CoreSceneManager::DestroyCamera(Camera* camera)
    {
        if (!camera) return;
        auto it = std::find(m_cameras.begin(), m_cameras.end(), camera);
        if (it == m_cameras.end()) return;

        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->RemoveCamera(camera);
            }
        }
        m_cameras.erase(it);
        delete camera;
    }

    Camera* CoreSceneManager::GetMainCamera() const
    {
        return m_activeScene ? m_activeScene->GetMainCamera() : nullptr;
    }

    RenderObject* CoreSceneManager::CreateRenderObject(const RenderObjectDesc& desc)
    {
        if (desc.materialCount > 0u && desc.materials == nullptr)
        {
            return nullptr;
        }
        if (desc.materialCount > kNVGMaxMaterialSlots)
        {
            return nullptr;
        }
        RenderObject* renderObject = new RenderObject();
        renderObject->SetMesh(desc.mesh);
        renderObject->SetMaterial(desc.material);
        if (desc.materials != nullptr && desc.materialCount > 0u)
        {
            for (UINT32 slot = 0u; slot < desc.materialCount; ++slot)
            {
                renderObject->SetMaterial(slot, desc.materials[slot]);
            }
        }
        renderObject->SetLODGroup(desc.lodGroup);
        renderObject->SetWorldMatrix(desc.worldMatrix);
        renderObject->SetLayerMask(desc.layerMask);
        renderObject->SetFlags(desc.flags);
        if (desc.worldAABB.Size().LengthSq() > 0.0f)
        {
            renderObject->SetWorldAABB(desc.worldAABB);
        }
        else
        {
            renderObject->UpdateWorldAABB();
        }

        m_renderObjects.push_back(renderObject);
        return renderObject;
    }

    BOOL CoreSceneManager::IsManagedRenderObject(const RenderObject* renderObject) const
    {
        if (!renderObject) return false;
        return std::find(m_renderObjects.begin(), m_renderObjects.end(), renderObject) != m_renderObjects.end();
    }

    void CoreSceneManager::DestroyRenderObject(RenderObject* renderObject)
    {
        if (!renderObject) return;
        auto it = std::find(m_renderObjects.begin(), m_renderObjects.end(), renderObject);
        if (it == m_renderObjects.end()) return;

        while (Scene* linkedScene = renderObject->GetFirstLinkedScene())
        {
            linkedScene->RemoveRenderObject(renderObject);
        }

        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->RemoveRenderObject(renderObject);
            }
        }

        m_renderObjects.erase(it);
        delete renderObject;
    }

    Light* CoreSceneManager::CreateLight(const LightDesc& desc)
    {
        if (desc.type != LightType::DIRECTIONAL &&
            desc.type != LightType::POINT &&
            desc.type != LightType::SPOT)
        {
            return nullptr;
        }

        if (!std::isfinite(desc.color.r) || !std::isfinite(desc.color.g) ||
            !std::isfinite(desc.color.b) || !std::isfinite(desc.color.a) ||
            !std::isfinite(desc.intensity) || desc.intensity < 0.0f ||
            !std::isfinite(desc.position.x) || !std::isfinite(desc.position.y) || !std::isfinite(desc.position.z) ||
            !std::isfinite(desc.rotation.x) || !std::isfinite(desc.rotation.y) ||
            !std::isfinite(desc.rotation.z) || !std::isfinite(desc.rotation.w) ||
            !std::isfinite(desc.range) ||
            !std::isfinite(desc.constantAttenuation) ||
            !std::isfinite(desc.linearAttenuation) ||
            !std::isfinite(desc.quadraticAttenuation) ||
            !std::isfinite(desc.innerConeAngle) ||
            !std::isfinite(desc.outerConeAngle))
        {
            return nullptr;
        }

        const FLOAT32 rotLenSq =
            desc.rotation.x * desc.rotation.x +
            desc.rotation.y * desc.rotation.y +
            desc.rotation.z * desc.rotation.z +
            desc.rotation.w * desc.rotation.w;
        if (rotLenSq <= 1e-12f)
        {
            return nullptr;
        }

        if ((desc.type == LightType::POINT || desc.type == LightType::SPOT) && desc.range <= 0.0f)
        {
            return nullptr;
        }

        if (desc.type == LightType::SPOT &&
            (desc.innerConeAngle < 0.0f || desc.outerConeAngle <= 0.0f ||
             desc.outerConeAngle >= 180.0f || desc.innerConeAngle > desc.outerConeAngle))
        {
            return nullptr;
        }

        Light* light = new Light();
        light->SetType(desc.type);
        light->SetColor(desc.color);
        light->SetIntensity(desc.intensity);
        light->SetFlags(desc.flags);
        light->SetPosition(desc.position);
        light->SetRotation(desc.rotation);
        light->SetRange(desc.range);
        light->SetConstantAttenuation(desc.constantAttenuation);
        light->SetLinearAttenuation(desc.linearAttenuation);
        light->SetQuadraticAttenuation(desc.quadraticAttenuation);
        light->SetOuterConeAngle(desc.outerConeAngle);
        light->SetInnerConeAngle(desc.innerConeAngle);
        if (!light->SetShadowResolution(desc.shadowResolution) ||
            !light->SetShadowSoftness(desc.shadowSoftness) ||
            !light->SetShadowBias(desc.shadowBias) ||
            !light->SetShadowNormalBias(desc.shadowNormalBias) ||
            !light->SetShadowMaxDistance(desc.shadowMaxDistance) ||
            !light->SetShadowNearPlane(desc.shadowNearPlane))
        {
            delete light;
            return nullptr;
        }
        light->SetCullingMask(desc.cullingMask);
        light->m_runtimeUid = ++m_nextLightRuntimeUid;

        m_lights.push_back(light);
        return light;
    }

    BOOL CoreSceneManager::IsManagedLight(const Light* light) const
    {
        if (!light) return false;
        return std::find(m_lights.begin(), m_lights.end(), light) != m_lights.end();
    }

    void CoreSceneManager::DestroyLight(Light* light)
    {
        if (!light) return;
        auto it = std::find(m_lights.begin(), m_lights.end(), light);
        if (it == m_lights.end()) return;

        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->RemoveLight(light);
            }
        }
        *it = m_lights.back();
        m_lights.pop_back();
        delete light;
    }

    Scene* CoreSceneManager::CreateScene(const SceneDesc& desc)
    {
        Scene* scene = new Scene();
        for (SIZE_T i = 0; i < desc.renderObjects.size(); ++i)
            scene->AddRenderObject(desc.renderObjects[i]);
        for (SIZE_T i = 0; i < desc.cameras.size(); ++i)
            scene->AddCamera(desc.cameras[i]);
        for (SIZE_T i = 0; i < desc.lights.size(); ++i)
            scene->AddLight(desc.lights[i]);

        scene->SetMainCamera(desc.mainCamera);
        scene->SetDirectionalLight(desc.directionalLight);
        scene->SetAmbientColor(desc.ambientColor);
        scene->SetAmbientIntensity(desc.ambientIntensity);

        m_scenes.push_back(scene);
        return scene;
    }

    BOOL CoreSceneManager::IsManagedScene(const Scene* scene) const
    {
        if (!scene) return false;
        return std::find(m_scenes.begin(), m_scenes.end(), scene) != m_scenes.end();
    }

    void CoreSceneManager::DestroyScene(Scene* scene)
    {
        if (!scene) return;
        auto it = std::find(m_scenes.begin(), m_scenes.end(), scene);
        if (it == m_scenes.end()) return;

        if (m_activeScene == scene)
            m_activeScene = nullptr;

        m_scenes.erase(it);
        delete scene;
    }

    void CoreSceneManager::SetActiveScene(Scene* scene)
    {
        if (scene != nullptr && !IsManagedScene(scene))
            return;
        m_activeScene = scene;
    }

    Scene* CoreSceneManager::GetActiveScene() const
    {
        return m_activeScene;
    }

    void CoreSceneManager::OnComputeKernelDestroyed(ComputeKernel* destroyedKernel)
    {
        if (!destroyedKernel) return;
        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->RemoveComputeKernel(destroyedKernel);
            }
        }
    }

    void CoreSceneManager::OnMaterialDestroyed(Material* destroyedMaterial)
    {
        if (!destroyedMaterial) return;
        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->OnMaterialDestroyed(destroyedMaterial);
            }
        }
        for (RenderObject* ro : m_renderObjects)
        {
            if (ro)
            {
                ro->ClearDestroyedMaterial(destroyedMaterial);
            }
        }
    }

    void CoreSceneManager::OnMeshDestroyed(Mesh* destroyedMesh)
    {
        if (!destroyedMesh) return;
        for (Scene* scene : m_scenes)
        {
            if (scene)
            {
                scene->OnMeshDestroyed(destroyedMesh);
            }
        }
        for (RenderObject* ro : m_renderObjects)
        {
            if (ro && ro->GetMesh() == destroyedMesh)
            {
                ro->SetMesh(nullptr);
            }
        }
    }

    void CoreSceneManager::OnLODGroupDestroyed(LODGroup* destroyedLODGroup)
    {
        if (!destroyedLODGroup) return;
        for (RenderObject* ro : m_renderObjects)
        {
            if (ro && ro->GetLODGroup() == destroyedLODGroup)
            {
                ro->SetLODGroup(nullptr);
            }
        }
    }

    void CoreSceneManager::OnTextureDestroyed(Texture* destroyedTexture)
    {
        if (!destroyedTexture) return;
        for (Camera* cam : m_cameras)
        {
            if (!cam) continue;
            if (cam->m_renderTarget == destroyedTexture)
            {
                cam->m_renderTarget = nullptr;
            }
            if (cam->m_depthStencilTarget == destroyedTexture)
            {
                cam->m_depthStencilTarget = nullptr;
            }
            if (cam->m_sceneDepthTexture == destroyedTexture)
            {
                cam->m_sceneDepthTexture = nullptr;
            }
        }
    }

    void CoreSceneManager::Shutdown()
    {
        for (auto scene : m_scenes) delete scene;
        m_scenes.clear();

        for (auto renderObj : m_renderObjects) delete renderObj;
        m_renderObjects.clear();

        for (auto camera : m_cameras) delete camera;
        m_cameras.clear();

        for (auto light : m_lights) delete light;
        m_lights.clear();

        m_activeScene = nullptr;
    }
}
