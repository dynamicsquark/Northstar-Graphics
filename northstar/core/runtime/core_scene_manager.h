#pragma once

#include "camera/camera.h"
#include "camera/camera_desc.h"
#include "light/light.h"
#include "light/light_desc.h"
#include "render_object/render_object.h"
#include "render_object/render_object_desc.h"
#include "scene/scene.h"
#include "scene/scene_desc.h"
#include <vector>

namespace Northstar
{
    class CoreSceneManager
    {
    public:
        CoreSceneManager() = default;
        ~CoreSceneManager();

        Camera*       CreateCamera(const CameraDesc& desc);
        void          DestroyCamera(Camera* camera);
        BOOL          IsManagedCamera(const Camera* camera) const;
        Camera*       GetMainCamera() const;
        const std::vector<Camera*>& GetCameras() const { return m_cameras; }

        RenderObject* CreateRenderObject(const RenderObjectDesc& desc);
        void          DestroyRenderObject(RenderObject* renderObject);
        BOOL          IsManagedRenderObject(const RenderObject* renderObject) const;

        Light*        CreateLight(const LightDesc& desc);
        void          DestroyLight(Light* light);
        BOOL          IsManagedLight(const Light* light) const;

        Scene*        CreateScene(const SceneDesc& desc);
        void          DestroyScene(Scene* scene);
        BOOL          IsManagedScene(const Scene* scene) const;
        void          SetActiveScene(Scene* scene);
        Scene*        GetActiveScene() const;
        const std::vector<Scene*>& GetScenes() const { return m_scenes; }

        void          OnComputeKernelDestroyed(ComputeKernel* destroyedKernel);
        void          OnMaterialDestroyed(Material* destroyedMaterial);
        void          OnMeshDestroyed(Mesh* destroyedMesh);
        void          OnLODGroupDestroyed(LODGroup* destroyedLODGroup);
        void          OnTextureDestroyed(Texture* destroyedTexture);

        void          Shutdown();

    private:
        Scene*                   m_activeScene = nullptr;
        std::vector<Camera*>       m_cameras;
        std::vector<RenderObject*> m_renderObjects;
        std::vector<Light*>        m_lights;
        std::vector<Scene*>        m_scenes;
        UINT64                     m_nextLightRuntimeUid = 0ULL;
    };
}
