#pragma once

#include "../../include/nvg/nvg_types.h"
#include "nvg_builder.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Northstar
{
    // Thread-safe disk-backed NVG2 page stream provider.
    // Parses only [header + metadata tables + page directory] into RAM (zero streaming page retention)
    // and performs concurrent positioned file reads for background decompression workers.
    class NVGFilePageStreamProvider final : public INVGPageStreamProvider
    {
    public:
        NVGFilePageStreamProvider() = default;
        ~NVGFilePageStreamProvider() override;

        NVGFilePageStreamProvider(const NVGFilePageStreamProvider&)            = delete;
        NVGFilePageStreamProvider& operator=(const NVGFilePageStreamProvider&) = delete;

        static std::shared_ptr<NVGFilePageStreamProvider> OpenNVG2File(
            const std::filesystem::path& filePath,
            NVGVirtualMeshAsset&         outMetadataOnlyAsset,
            std::string&                 outError);

        UINT32 GetVirtualPageCount() const noexcept override
        {
            return static_cast<UINT32>(m_directory.size());
        }

        BOOL ReadCompressedPage(
            UINT32  localPageIndex,
            BYTE*   outCompressedBytes,
            UINT32  maxBytes,
            UINT32& outBytesRead,
            UINT32& outUncompressedBytes,
            UINT32& outCodecFlags,
            UINT32& outExpectedCrc32c) const override;

        BOOL ReadDecompressedPage(
            UINT32 localPageIndex,
            BYTE*  outPageBytes,
            UINT32 maxBytes) const override;

    private:
        BOOL ReadFileRangeAt(UINT64 fileOffset, BYTE* dst, UINT32 byteCount) const;

        std::filesystem::path               m_filePath;
        UINT64                              m_fileByteSize = 0ull;
        std::vector<NVG2PageDirectoryEntry> m_directory;
#ifdef _WIN32
        void*                               m_fileHandle   = nullptr; // Shared read-only HANDLE opened for positioned OVERLAPPED reads
#endif
    };

    struct NVGAsyncStreamJob
    {
        UINT32                                  meshId          = 0u;
        UINT32                                  globalVP        = 0u;
        UINT32                                  localPageIndex  = 0u;
        UINT32                                  expectedGen     = 0u;
        UINT32                                  priority        = 0u;
        UINT32                                  lodLevel        = 0u;
        std::shared_ptr<INVGPageStreamProvider> provider;
    };

    struct NVGCompletedStreamPage
    {
        UINT32              meshId         = 0u;
        UINT32              globalVP       = 0u;
        UINT32              localPageIndex = 0u;
        UINT32              expectedGen    = 0u;
        UINT32              priority       = 0u;
        UINT32              lodLevel       = 0u;
        BOOL                succeeded      = false;
        std::vector<BYTE>   pageBytes; // Exactly kNVGPageSizeBytes (32 KiB)
    };

    // Multi-threaded background I/O + LZ4 decompression + CRC32C verification queue with bounded 32 KiB staging buffers.
    class NVGAsyncPageStreamer
    {
    public:
        NVGAsyncPageStreamer() = default;
        ~NVGAsyncPageStreamer();

        NVGAsyncPageStreamer(const NVGAsyncPageStreamer&)            = delete;
        NVGAsyncPageStreamer& operator=(const NVGAsyncPageStreamer&) = delete;

        void Initialize(UINT32 workerCount = 2u, UINT32 maxStagingPages = 256u);
        void Shutdown();

        BOOL IsInitialized() const noexcept { return m_running.load(std::memory_order_relaxed); }

        // Enqueues or boosts priority of a non-resident streaming page job without blocking the render thread.
        void EnqueueOrUpdateJob(const NVGAsyncStreamJob& job);

        // Cancels queued jobs belonging to an unregistered mesh.
        void CancelMeshJobs(UINT32 meshId);

        // Non-blocking drain of completed 32 KiB decompressed pages onto caller's vector; recycles staging buffers via RecyclePageBuffer.
        void DrainCompletedPages(std::vector<NVGCompletedStreamPage>& outCompleted, UINT32 maxDrainCount);

        void RecyclePageBuffer(std::vector<BYTE>&& buffer);

    private:
        void WorkerLoop();

        std::atomic<bool>                   m_running{ false };
        UINT32                              m_maxStagingPages = 256u;
        std::vector<std::thread>            m_workers;

        std::mutex                          m_mutex;
        std::condition_variable             m_cv;
        std::vector<NVGAsyncStreamJob>      m_pendingJobs;
        std::vector<UINT32>                 m_inFlightGlobalVPs;
        std::deque<NVGCompletedStreamPage>  m_completedPages;
        std::vector<std::vector<BYTE>>      m_freePageBuffers;
    };
}
