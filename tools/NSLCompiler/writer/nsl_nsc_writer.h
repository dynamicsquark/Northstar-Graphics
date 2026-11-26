#pragma once

#include "NSLCommon/nsl_common_types.h"
#include "NSLCommon/nsl_shader_data.h"
#include "../property/nsl_property_parser.h"
#include "../stage_io/nsl_stage_io_parser.h"
#include "../aligner/nsl_aligner.h"
#include <string>
#include <vector>
#include <unordered_map>

// =====================================================================
//  NSC Writer — Binary .nsc v6 Format
//
//  v6 replaces 1D int16 slot with 2D (uint16 set, uint16 binding).
//  v5 adds INPUT_LAYOUT / OUTPUT_LAYOUT reflection sections.
//  Header remains 36 bytes: the former 2-byte pad is now
//    inputLayoutCount (u8) + outputLayoutCount (u8).
// =====================================================================

namespace NSL
{
    static constexpr uint32_t NSC_MAGIC   = 0x4E534300; // "NSC\0"
    static constexpr uint32_t NSC_VERSION = 6;

    class NSCWriter
    {
    public:
        bool Write(
            const std::string&                                    outputPath,
            ShaderStage                                           stage,
            TargetAPI                                             api,
            const ParsedProperties&                               properties,
            const ParsedStageIO&                                  stageIO,
            const std::unordered_map<std::string, AlignedBuffer>& alignedBuffers,
            const std::unordered_map<std::string, AlignedStruct>& alignedStructs,
            const std::string&                                    sourceCode,
            uint16_t                                              extraFlags = NSC_FLAG_NONE,
            uint32_t                                              numThreadsX = 0,
            uint32_t                                              numThreadsY = 0,
            uint32_t                                              numThreadsZ = 0
        );

    private:
        void WritePropertyBlockTable(std::vector<uint8_t>& buf, const ParsedProperties& props);
        void WritePropertyTable(std::vector<uint8_t>& buf, const ParsedProperties& props);
        void WriteBufferSection(std::vector<uint8_t>& buf, const ParsedProperties& props,
                                const std::unordered_map<std::string, AlignedBuffer>& aligned);
        void WriteTextureSection(std::vector<uint8_t>& buf, const ParsedProperties& props);
        void WriteSamplerSection(std::vector<uint8_t>& buf, const ParsedProperties& props);
        void WriteStructuredBufferSection(std::vector<uint8_t>& buf, const ParsedProperties& props,
                                          const std::unordered_map<std::string, AlignedStruct>& aligned);
        void WriteUAVSection(std::vector<uint8_t>& buf, const ParsedProperties& props,
                             const std::unordered_map<std::string, AlignedStruct>& aligned);
        void WriteStageIOSection(std::vector<uint8_t>& buf, const std::vector<StageIOEntry>& entries);
        void WriteShaderSource(std::vector<uint8_t>& buf, const std::string& sourceCode);

        static void WriteU8 (std::vector<uint8_t>& buf, uint8_t  v);
        static void WriteU16(std::vector<uint8_t>& buf, uint16_t v);
        static void WriteU32(std::vector<uint8_t>& buf, uint32_t v);
        static void WriteI16(std::vector<uint8_t>& buf, int16_t  v);
        static void WriteStr(std::vector<uint8_t>& buf, const std::string& s);
        static void PatchU32(std::vector<uint8_t>& buf, size_t pos, uint32_t v);
    };

} // namespace NSL
