#include "nvg_async_stream_io.h"
#include "nvg_codec_lz4.h"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include "../../include/types/northstar_platform_guard.h"
#endif

#include <algorithm>
#include <cstring>
#include <fstream>

namespace Northstar
{
    // =====================================================================
    //  NVGFilePageStreamProvider
    // =====================================================================

    NVGFilePageStreamProvider::~NVGFilePageStreamProvider()
    {
#ifdef _WIN32
        if (m_fileHandle != nullptr && m_fileHandle != INVALID_HANDLE_VALUE)
        {
            ::CloseHandle(static_cast<HANDLE>(m_fileHandle));
            m_fileHandle = nullptr;
        }
#endif
    }

    std::shared_ptr<NVGFilePageStreamProvider> NVGFilePageStreamProvider::OpenNVG2File(
        const std::filesystem::path& filePath,
        NVGVirtualMeshAsset&         outMetadataOnlyAsset,
        std::string&                 outError)
    {
        outMetadataOnlyAsset = NVGVirtualMeshAsset{};
        outError.clear();

        if (filePath.empty())
        {
            outError = "NVGFilePageStreamProvider: empty file path.";
            return nullptr;
        }

        auto provider = std::make_shared<NVGFilePageStreamProvider>();
        provider->m_filePath = filePath;

#ifdef _WIN32
        // Open with FILE_FLAG_OVERLAPPED so multiple background worker threads can issue concurrent
        // lock-free positioned ReadFile calls at independent file offsets without mutating a shared file pointer.
        HANDLE hFile = ::CreateFileW(
            filePath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
        {
            outError = "NVGFilePageStreamProvider: failed to open NVG2 file for reading.";
            return nullptr;
        }
        provider->m_fileHandle = hFile;

        LARGE_INTEGER fileSizeLI{};
        if (!::GetFileSizeEx(hFile, &fileSizeLI) || fileSizeLI.QuadPart < static_cast<LONGLONG>(sizeof(NVG2BinaryHeader)))
        {
            outError = "NVGFilePageStreamProvider: file smaller than NVG2BinaryHeader.";
            return nullptr;
        }
        provider->m_fileByteSize = static_cast<UINT64>(fileSizeLI.QuadPart);
#else
        std::ifstream ifs(filePath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open())
        {
            outError = "NVGFilePageStreamProvider: failed to open NVG2 file.";
            return nullptr;
        }
        const std::streamoff endPos = ifs.tellg();
        if (endPos < static_cast<std::streamoff>(sizeof(NVG2BinaryHeader)))
        {
            outError = "NVGFilePageStreamProvider: file smaller than NVG2BinaryHeader.";
            return nullptr;
        }
        provider->m_fileByteSize = static_cast<UINT64>(endPos);
#endif

        // Step 1: Read ONLY the 192-byte NVG2BinaryHeader first to find exact pageDataByteOffset
        NVG2BinaryHeader peekHdr{};
        if (!provider->ReadFileRangeAt(0ull, reinterpret_cast<BYTE*>(&peekHdr), sizeof(NVG2BinaryHeader)))
        {
            outError = "NVGFilePageStreamProvider: failed to read 192-byte NVG2 header.";
            return nullptr;
        }
        if (peekHdr.magic != kNVG2BinaryMagic || peekHdr.version != kNVG2BinaryVersion)
        {
            outError = "NVGFilePageStreamProvider: file is not a valid NVG2 container.";
            return nullptr;
        }
        if (peekHdr.pageDataByteOffset < sizeof(NVG2BinaryHeader) ||
            peekHdr.pageDataByteOffset > provider->m_fileByteSize ||
            peekHdr.totalFileByteSize != provider->m_fileByteSize ||
            peekHdr.pageDataByteOffset > 0x20000000ull)
        {
            outError = "NVGFilePageStreamProvider: invalid header byte offsets or file size mismatch.";
            return nullptr;
        }

        // Step 2: Read ONLY [0 .. pageDataByteOffset) (header + tables + page directory) — zero streaming pages in RAM!
        std::vector<BYTE> metaPrefix(static_cast<SIZE_T>(peekHdr.pageDataByteOffset));
        if (!provider->ReadFileRangeAt(0ull, metaPrefix.data(), static_cast<UINT32>(metaPrefix.size())))
        {
            outError = "NVGFilePageStreamProvider: failed to read NVG2 metadata prefix.";
            return nullptr;
        }

        NVG2BinaryHeader parsedHdr{};
        if (!NVGBuilder::ParseNVG2MetadataAndDirectory(
                metaPrefix.data(),
                metaPrefix.size(),
                parsedHdr,
                outMetadataOnlyAsset,
                provider->m_directory,
                outError))
        {
            return nullptr;
        }

        return provider;
    }

    BOOL NVGFilePageStreamProvider::ReadFileRangeAt(UINT64 fileOffset, BYTE* dst, UINT32 byteCount) const
    {
        if (!dst) return false;
        if (byteCount == 0u) return true;
        if (fileOffset > m_fileByteSize || byteCount > (m_fileByteSize - fileOffset))
            return false;

#ifdef _WIN32
        if (m_fileHandle == nullptr || m_fileHandle == INVALID_HANDLE_VALUE)
            return false;

        // Per-thread cached manual-reset event avoids kernel CreateEvent overhead on every 32 KiB page read
        struct ThreadLocalOverlappedEvent
        {
            HANDLE hEvent = nullptr;
            ThreadLocalOverlappedEvent()
            {
                hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
            }
            ~ThreadLocalOverlappedEvent()
            {
                if (hEvent != nullptr) ::CloseHandle(hEvent);
            }
        };
        thread_local ThreadLocalOverlappedEvent tlEvent;
        if (tlEvent.hEvent == nullptr)
            return false;
        ::ResetEvent(tlEvent.hEvent);

        OVERLAPPED ov{};
        ov.Offset     = static_cast<DWORD>(fileOffset & 0xFFFFFFFFull);
        ov.OffsetHigh = static_cast<DWORD>((fileOffset >> 32u) & 0xFFFFFFFFull);
        ov.hEvent     = tlEvent.hEvent;

        DWORD bytesRead = 0u;
        BOOL ok = ::ReadFile(static_cast<HANDLE>(m_fileHandle), dst, byteCount, &bytesRead, &ov);
        if (!ok)
        {
            if (::GetLastError() != ERROR_IO_PENDING)
                return false;
            if (!::GetOverlappedResult(static_cast<HANDLE>(m_fileHandle), &ov, &bytesRead, TRUE))
                return false;
        }
        return (bytesRead == byteCount);
#else
        std::ifstream ifs(m_filePath, std::ios::binary);
        if (!ifs.is_open()) return false;
        ifs.seekg(static_cast<std::streamoff>(fileOffset), std::ios::beg);
        ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(byteCount));
        return ifs.good() && (static_cast<UINT32>(ifs.gcount()) == byteCount);
#endif
    }

    BOOL NVGFilePageStreamProvider::ReadCompressedPage(
        UINT32  localPageIndex,
        BYTE*   outCompressedBytes,
        UINT32  maxBytes,
        UINT32& outBytesRead,
        UINT32& outUncompressedBytes,
        UINT32& outCodecFlags,
        UINT32& outExpectedCrc32c) const
    {
        outBytesRead         = 0u;
        outUncompressedBytes = 0u;
        outCodecFlags        = 0u;
        outExpectedCrc32c    = 0u;

        if (localPageIndex >= m_directory.size() || !outCompressedBytes)
            return false;

        const NVG2PageDirectoryEntry& de = m_directory[localPageIndex];
        if (de.compressedByteSize > maxBytes || de.uncompressedByteSize > kNVGPageSizeBytes)
            return false;

        if (!ReadFileRangeAt(de.byteOffsetInFile, outCompressedBytes, de.compressedByteSize))
            return false;

        outBytesRead         = de.compressedByteSize;
        outUncompressedBytes = de.uncompressedByteSize;
        outCodecFlags        = de.codecAndFlags;
        outExpectedCrc32c    = de.crc32c;
        return true;
    }

    BOOL NVGFilePageStreamProvider::ReadDecompressedPage(
        UINT32 localPageIndex,
        BYTE*  outPageBytes,
        UINT32 maxBytes) const
    {
        if (!outPageBytes || maxBytes < kNVGPageSizeBytes || localPageIndex >= m_directory.size())
            return false;

        thread_local std::vector<BYTE> tlCompressedScratch(kNVGPageSizeBytes + 256u);
        UINT32 compBytes   = 0u;
        UINT32 uncompBytes = 0u;
        UINT32 codecFlags  = 0u;
        UINT32 expectedCrc = 0u;
        if (!ReadCompressedPage(
                localPageIndex,
                tlCompressedScratch.data(),
                static_cast<UINT32>(tlCompressedScratch.size()),
                compBytes,
                uncompBytes,
                codecFlags,
                expectedCrc))
        {
            return false;
        }

        std::memset(outPageBytes, 0, kNVGPageSizeBytes);
        if ((codecFlags & 2u) != 0u)
        {
            if (!NVGCodecLZ4::DecompressBlockExact(tlCompressedScratch.data(), compBytes, outPageBytes, uncompBytes))
                return false;
        }
        else if (uncompBytes > 0u)
        {
            std::memcpy(outPageBytes, tlCompressedScratch.data(), uncompBytes);
        }

        const UINT32 actualCrc = NVGCodecLZ4::ComputeCRC32C(outPageBytes, uncompBytes);
        return (actualCrc == expectedCrc);
    }

    // =====================================================================
    //  NVGAsyncPageStreamer
    // =====================================================================

    NVGAsyncPageStreamer::~NVGAsyncPageStreamer()
    {
        Shutdown();
    }

    void NVGAsyncPageStreamer::Initialize(UINT32 workerCount, UINT32 maxStagingPages)
    {
        if (m_running.load(std::memory_order_relaxed))
            return;

        m_maxStagingPages = (std::max)(16u, maxStagingPages);
        const UINT32 clampedWorkers = (std::clamp)(workerCount, 1u, 8u);

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_pendingJobs.clear();
            m_inFlightGlobalVPs.clear();
            m_completedPages.clear();
            m_freePageBuffers.clear();
            m_freePageBuffers.reserve(64u);
            for (UINT32 i = 0u; i < 32u; ++i)
            {
                m_freePageBuffers.emplace_back(kNVGPageSizeBytes, static_cast<BYTE>(0));
            }
            m_running.store(true, std::memory_order_release);
        }

        m_workers.reserve(clampedWorkers);
        for (UINT32 w = 0u; w < clampedWorkers; ++w)
        {
            m_workers.emplace_back([this]() { WorkerLoop(); });
        }
    }

    void NVGAsyncPageStreamer::Shutdown()
    {
        if (!m_running.exchange(false, std::memory_order_acq_rel))
            return;

        m_cv.notify_all();
        for (std::thread& th : m_workers)
        {
            if (th.joinable()) th.join();
        }
        m_workers.clear();

        std::lock_guard<std::mutex> lock(m_mutex);
        m_pendingJobs.clear();
        m_inFlightGlobalVPs.clear();
        m_completedPages.clear();
        m_freePageBuffers.clear();
    }

    void NVGAsyncPageStreamer::EnqueueOrUpdateJob(const NVGAsyncStreamJob& job)
    {
        if (!job.provider || !m_running.load(std::memory_order_relaxed))
            return;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            // Skip if already in flight on a worker thread or already waiting in completedPages
            for (UINT32 vp : m_inFlightGlobalVPs)
            {
                if (vp == job.globalVP) return;
            }
            for (const NVGCompletedStreamPage& cp : m_completedPages)
            {
                if (cp.globalVP == job.globalVP && cp.expectedGen == job.expectedGen)
                    return;
            }
            // Boost priority if already queued in m_pendingJobs
            for (NVGAsyncStreamJob& pj : m_pendingJobs)
            {
                if (pj.globalVP == job.globalVP)
                {
                    pj.expectedGen = job.expectedGen;
                    if (job.priority > pj.priority) pj.priority = job.priority;
                    return;
                }
            }
            // Cap pending queue to prevent unbounded backlog; evict lowest-priority entry if full
            if (m_pendingJobs.size() >= m_maxStagingPages * 4u)
            {
                auto minIt = std::min_element(m_pendingJobs.begin(), m_pendingJobs.end(),
                    [](const NVGAsyncStreamJob& a, const NVGAsyncStreamJob& b) {
                        if (a.priority != b.priority) return a.priority < b.priority;
                        return a.lodLevel < b.lodLevel;
                    });
                if (minIt != m_pendingJobs.end() && job.priority > minIt->priority)
                {
                    *minIt = job;
                }
                else
                {
                    return;
                }
            }
            else
            {
                m_pendingJobs.push_back(job);
            }
        }
        m_cv.notify_one();
    }

    void NVGAsyncPageStreamer::CancelMeshJobs(UINT32 meshId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pendingJobs.erase(
            std::remove_if(m_pendingJobs.begin(), m_pendingJobs.end(),
                [meshId](const NVGAsyncStreamJob& j) { return j.meshId == meshId; }),
            m_pendingJobs.end());

        auto it = m_completedPages.begin();
        while (it != m_completedPages.end())
        {
            if (it->meshId == meshId)
            {
                if (it->pageBytes.size() == kNVGPageSizeBytes && m_freePageBuffers.size() < m_maxStagingPages)
                {
                    m_freePageBuffers.push_back(std::move(it->pageBytes));
                }
                it = m_completedPages.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void NVGAsyncPageStreamer::DrainCompletedPages(
        std::vector<NVGCompletedStreamPage>& outCompleted,
        UINT32                               maxDrainCount)
    {
        outCompleted.clear();
        if (maxDrainCount == 0u) return;

        std::lock_guard<std::mutex> lock(m_mutex);
        UINT32 drained = 0u;
        while (!m_completedPages.empty() && drained < maxDrainCount)
        {
            outCompleted.push_back(std::move(m_completedPages.front()));
            m_completedPages.pop_front();
            ++drained;
        }
        if (drained > 0u)
        {
            m_cv.notify_all();
        }
    }

    void NVGAsyncPageStreamer::RecyclePageBuffer(std::vector<BYTE>&& buffer)
    {
        if (buffer.size() != kNVGPageSizeBytes) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_freePageBuffers.size() < m_maxStagingPages)
        {
            m_freePageBuffers.push_back(std::move(buffer));
        }
    }

    void NVGAsyncPageStreamer::WorkerLoop()
    {
        std::vector<BYTE> compressedScratch(kNVGPageSizeBytes + 256u);

        for (;;)
        {
            NVGAsyncStreamJob job{};
            std::vector<BYTE> dstPage;

            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this]() {
                    return !m_running.load(std::memory_order_relaxed) ||
                           (!m_pendingJobs.empty() && m_completedPages.size() < m_maxStagingPages);
                });

                if (!m_running.load(std::memory_order_relaxed))
                    break;
                if (m_pendingJobs.empty() || m_completedPages.size() >= m_maxStagingPages)
                    continue;

                // Pop highest-priority job (tie-break coarser LOD first so parent pages finish before children!)
                auto bestIt = std::max_element(m_pendingJobs.begin(), m_pendingJobs.end(),
                    [](const NVGAsyncStreamJob& a, const NVGAsyncStreamJob& b) {
                        if (a.priority != b.priority) return a.priority < b.priority;
                        if (a.lodLevel != b.lodLevel) return a.lodLevel < b.lodLevel;
                        return a.globalVP > b.globalVP;
                    });
                job = std::move(*bestIt);
                *bestIt = std::move(m_pendingJobs.back());
                m_pendingJobs.pop_back();
                m_inFlightGlobalVPs.push_back(job.globalVP);

                if (!m_freePageBuffers.empty())
                {
                    dstPage = std::move(m_freePageBuffers.back());
                    m_freePageBuffers.pop_back();
                }
            }

            if (dstPage.size() != kNVGPageSizeBytes)
            {
                dstPage.assign(kNVGPageSizeBytes, static_cast<BYTE>(0));
            }
            else
            {
                std::memset(dstPage.data(), 0, kNVGPageSizeBytes);
            }

            // Execute disk I/O + LZ4 decompression + CRC32C verification completely lock-free off the render thread
            BOOL ok = false;
            if (job.provider)
            {
                UINT32 compBytes   = 0u;
                UINT32 uncompBytes = 0u;
                UINT32 codecFlags  = 0u;
                UINT32 expectedCrc = 0u;
                if (job.provider->ReadCompressedPage(
                        job.localPageIndex,
                        compressedScratch.data(),
                        static_cast<UINT32>(compressedScratch.size()),
                        compBytes,
                        uncompBytes,
                        codecFlags,
                        expectedCrc))
                {
                    BOOL decompOk = true;
                    if ((codecFlags & 2u) != 0u)
                    {
                        decompOk = NVGCodecLZ4::DecompressBlockExact(
                            compressedScratch.data(), compBytes, dstPage.data(), uncompBytes);
                    }
                    else if (uncompBytes > 0u)
                    {
                        std::memcpy(dstPage.data(), compressedScratch.data(), uncompBytes);
                    }

                    if (decompOk)
                    {
                        const UINT32 actualCrc = NVGCodecLZ4::ComputeCRC32C(dstPage.data(), uncompBytes);
                        ok = (actualCrc == expectedCrc);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_inFlightGlobalVPs.erase(
                    std::remove(m_inFlightGlobalVPs.begin(), m_inFlightGlobalVPs.end(), job.globalVP),
                    m_inFlightGlobalVPs.end());

                if (m_running.load(std::memory_order_relaxed))
                {
                    NVGCompletedStreamPage done{};
                    done.meshId         = job.meshId;
                    done.globalVP       = job.globalVP;
                    done.localPageIndex = job.localPageIndex;
                    done.expectedGen    = job.expectedGen;
                    done.priority       = job.priority;
                    done.lodLevel       = job.lodLevel;
                    done.succeeded      = ok;
                    done.pageBytes      = std::move(dstPage);
                    m_completedPages.push_back(std::move(done));
                }
            }
        }
    }
}
