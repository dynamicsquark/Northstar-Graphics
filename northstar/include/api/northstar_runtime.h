#pragma once

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#include "../types/northstar_platform_guard.h"
#include "northstar_engine.h"
#include "northstar_handles.h"

#include <filesystem>
#include <string>
#include <utility>

namespace Northstar
{
    /**
     * @brief Configuration descriptor for initializing a complete Northstar runtime session.
     */
    struct RuntimeConfig
    {
        BackendAPI            backend                   = BackendAPI::D3D12;
        NativeRenderTarget    renderTarget              = {};
        BOOL                  vsync                     = true;
        BOOL                  enableValidationLayer     = false;
        BOOL                  enableGPUBasedValidation  = false;
        NorthstarBackend*     customBackend             = nullptr; // Used when backend == BackendAPI::Custom
        std::filesystem::path moduleDirectory           = {};      // Defaults to host executable directory
    };

    /**
     * @brief RAII host session managing Northstar dynamic shared modules, ABI validation,
     *        hardware backend lifetime, and NorthstarEngine teardown in strict reverse order.
     */
    class RuntimeSession
    {
    public:
        RuntimeSession() = default;

        ~RuntimeSession()
        {
            Shutdown();
        }

        RuntimeSession(const RuntimeSession&)            = delete;
        RuntimeSession& operator=(const RuntimeSession&) = delete;

        RuntimeSession(RuntimeSession&& other) noexcept
        {
            MoveFrom(std::move(other));
        }

        RuntimeSession& operator=(RuntimeSession&& other) noexcept
        {
            if (this != &other)
            {
                Shutdown();
                MoveFrom(std::move(other));
            }
            return *this;
        }

        [[nodiscard]] BOOL Initialize(const RuntimeConfig& config)
        {
            Shutdown();
            m_lastError.clear();

            const std::filesystem::path baseDir = config.moduleDirectory.empty()
                ? ResolveExecutableDirectory()
                : config.moduleDirectory;

            // 1. Load NorthstarCore module and verify ABI contract
            m_coreModule = LoadSharedModule(baseDir, "NorthstarCore");
            if (!m_coreModule)
            {
                m_lastError = "RuntimeSession: Failed to load NorthstarCore shared library from '" + baseDir.string() + "'.";
                return false;
            }

            const auto getCoreVerFn = LoadSymbol<GetNorthstarAPIVersionFn>(m_coreModule, "Northstar_GetAPIVersion");
            m_createEngineFn        = LoadSymbol<CreateNorthstarEngineFn>(m_coreModule, "CreateNorthstarEngine");
            m_destroyEngineFn       = LoadSymbol<DestroyNorthstarEngineFn>(m_coreModule, "DestroyNorthstarEngine");

            if (!getCoreVerFn || !m_createEngineFn || !m_destroyEngineFn)
            {
                m_lastError = "RuntimeSession: NorthstarCore shared library is missing required entry points.";
                Shutdown();
                return false;
            }

            const UINT32 coreVer = getCoreVerFn();
            if (coreVer != NORTHSTAR_ENGINE_API_VERSION)
            {
                m_lastError = "RuntimeSession: NorthstarCore ABI mismatch (expected " +
                              std::to_string(NORTHSTAR_ENGINE_API_VERSION) + ", found " +
                              std::to_string(coreVer) + ").";
                Shutdown();
                return false;
            }

            // 2. Acquire or dynamically load the GPU backend
            if (config.backend == BackendAPI::Custom)
            {
                if (!config.customBackend)
                {
                    m_lastError = "RuntimeSession: BackendAPI::Custom selected but customBackend is nullptr.";
                    Shutdown();
                    return false;
                }
                m_backend     = config.customBackend;
                m_ownsBackend = false;
            }
            else
            {
                const char* backendLibName = (config.backend == BackendAPI::D3D11) ? "NorthstarD3D11" : "NorthstarD3D12";
                const char* createSym      = (config.backend == BackendAPI::D3D11) ? "CreateNorthstarD3D11Backend" : "CreateNorthstarD3D12Backend";
                const char* destroySym     = (config.backend == BackendAPI::D3D11) ? "DestroyNorthstarD3D11Backend" : "DestroyNorthstarD3D12Backend";

                m_backendModule = LoadSharedModule(baseDir, backendLibName);
                if (!m_backendModule)
                {
                    m_lastError = std::string("RuntimeSession: Failed to load ") + backendLibName +
                                  " shared library from '" + baseDir.string() + "'.";
                    Shutdown();
                    return false;
                }

                const auto getBackendVerFn = LoadSymbol<GetNorthstarAPIVersionFn>(m_backendModule, "Northstar_GetAPIVersion");
                m_createBackendFn          = LoadSymbol<CreateNorthstarBackendFn>(m_backendModule, createSym);
                m_destroyBackendFn         = LoadSymbol<DestroyNorthstarBackendFn>(m_backendModule, destroySym);

                if (!getBackendVerFn || !m_createBackendFn || !m_destroyBackendFn)
                {
                    m_lastError = std::string("RuntimeSession: ") + backendLibName + " is missing required entry points.";
                    Shutdown();
                    return false;
                }

                const UINT32 backendVer = getBackendVerFn();
                if (backendVer != NORTHSTAR_ENGINE_API_VERSION)
                {
                    m_lastError = std::string("RuntimeSession: ") + backendLibName + " ABI mismatch (expected " +
                                  std::to_string(NORTHSTAR_ENGINE_API_VERSION) + ", found " +
                                  std::to_string(backendVer) + ").";
                    Shutdown();
                    return false;
                }

                m_backend = m_createBackendFn();
                if (!m_backend)
                {
                    m_lastError = std::string("RuntimeSession: ") + createSym + " returned nullptr (backend may be unsupported on this platform).";
                    Shutdown();
                    return false;
                }
                m_ownsBackend = true;
            }

            if (config.enableValidationLayer || config.enableGPUBasedValidation)
            {
                m_backend->SetValidationConfig(config.enableValidationLayer, config.enableGPUBasedValidation);
            }

            // 3. Instantiate and initialize NorthstarEngine
            m_engine = m_createEngineFn();
            if (!m_engine)
            {
                m_lastError = "RuntimeSession: CreateNorthstarEngine returned nullptr.";
                Shutdown();
                return false;
            }

            m_engine->SetVSync(config.vsync);
            if (!m_engine->Initialize(m_backend, config.renderTarget))
            {
                m_lastError = m_engine->GetLastError().empty()
                    ? "RuntimeSession: NorthstarEngine::Initialize failed."
                    : m_engine->GetLastError();
                Shutdown();
                return false;
            }

            return true;
        }

        void Shutdown() noexcept
        {
            if (m_engine)
            {
                if (m_engine->IsInitialized())
                {
                    m_engine->WaitForGPU();
                    m_engine->Shutdown();
                }
                if (m_destroyEngineFn)
                {
                    m_destroyEngineFn(m_engine);
                }
                m_engine = nullptr;
            }

            if (m_backend && m_ownsBackend)
            {
                m_backend->Shutdown();
                if (m_destroyBackendFn)
                {
                    m_destroyBackendFn(m_backend);
                }
            }
            m_backend     = nullptr;
            m_ownsBackend = false;

            m_createEngineFn   = nullptr;
            m_destroyEngineFn  = nullptr;
            m_createBackendFn  = nullptr;
            m_destroyBackendFn = nullptr;

            if (m_backendModule)
            {
                UnloadSharedModule(m_backendModule);
                m_backendModule = nullptr;
            }
            if (m_coreModule)
            {
                UnloadSharedModule(m_coreModule);
                m_coreModule = nullptr;
            }
        }

        [[nodiscard]] BOOL               IsInitialized() const noexcept { return m_engine != nullptr && m_engine->IsInitialized(); }
        [[nodiscard]] NorthstarEngine*   GetEnginePtr()  const noexcept { return m_engine; }
        [[nodiscard]] NorthstarEngine&   GetEngine()     const noexcept { return *m_engine; }
        [[nodiscard]] NorthstarBackend*  GetBackend()    const noexcept { return m_backend; }
        [[nodiscard]] const std::string& GetLastError()  const noexcept { return m_lastError; }

        NorthstarEngine* operator->() const noexcept { return m_engine; }
        NorthstarEngine& operator*()  const noexcept { return *m_engine; }
        explicit operator bool()      const noexcept { return IsInitialized(); }

        [[nodiscard]] static std::filesystem::path ResolveExecutableDirectory()
        {
#ifdef _WIN32
            wchar_t buffer[2048]{};
            const DWORD len = ::GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
            if (len > 0 && len < std::size(buffer))
            {
                return std::filesystem::path(buffer).parent_path();
            }
#endif
            return std::filesystem::current_path();
        }

    private:
        static void* LoadSharedModule(const std::filesystem::path& dir, const char* baseName)
        {
#ifdef _WIN32
    #ifdef _DEBUG
            const std::filesystem::path primary   = dir / (std::string(baseName) + "d.dll");
            const std::filesystem::path secondary = dir / (std::string(baseName) + ".dll");
    #else
            const std::filesystem::path primary   = dir / (std::string(baseName) + ".dll");
            const std::filesystem::path secondary = dir / (std::string(baseName) + "d.dll");
    #endif
            if (HMODULE mod = ::LoadLibraryW(primary.c_str()))
            {
                return static_cast<void*>(mod);
            }
            if (HMODULE mod = ::LoadLibraryW(secondary.c_str()))
            {
                return static_cast<void*>(mod);
            }
#else
            (void)dir;
            (void)baseName;
#endif
            return nullptr;
        }

        static void UnloadSharedModule(void* handle) noexcept
        {
#ifdef _WIN32
            if (handle)
            {
                ::FreeLibrary(static_cast<HMODULE>(handle));
            }
#else
            (void)handle;
#endif
        }

        template <typename Fn>
        static Fn LoadSymbol(void* handle, const char* symbolName) noexcept
        {
#ifdef _WIN32
            if (!handle || !symbolName)
            {
                return nullptr;
            }
            const FARPROC proc = ::GetProcAddress(static_cast<HMODULE>(handle), symbolName);
            return reinterpret_cast<Fn>(reinterpret_cast<void*>(proc));
#else
            (void)handle;
            (void)symbolName;
            return nullptr;
#endif
        }

        void MoveFrom(RuntimeSession&& other) noexcept
        {
            m_coreModule       = std::exchange(other.m_coreModule, nullptr);
            m_backendModule    = std::exchange(other.m_backendModule, nullptr);
            m_backend          = std::exchange(other.m_backend, nullptr);
            m_engine           = std::exchange(other.m_engine, nullptr);
            m_ownsBackend      = std::exchange(other.m_ownsBackend, false);
            m_createEngineFn   = std::exchange(other.m_createEngineFn, nullptr);
            m_destroyEngineFn  = std::exchange(other.m_destroyEngineFn, nullptr);
            m_createBackendFn  = std::exchange(other.m_createBackendFn, nullptr);
            m_destroyBackendFn = std::exchange(other.m_destroyBackendFn, nullptr);
            m_lastError        = std::move(other.m_lastError);
        }

        void*                    m_coreModule       = nullptr;
        void*                    m_backendModule    = nullptr;
        NorthstarBackend*        m_backend          = nullptr;
        NorthstarEngine*         m_engine           = nullptr;
        BOOL                     m_ownsBackend      = false;
        CreateNorthstarEngineFn  m_createEngineFn   = nullptr;
        DestroyNorthstarEngineFn m_destroyEngineFn  = nullptr;
        CreateNorthstarBackendFn m_createBackendFn  = nullptr;
        DestroyNorthstarBackendFn m_destroyBackendFn = nullptr;
        std::string              m_lastError;
    };
}
