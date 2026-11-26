#pragma once

#include "nsl_property_parser.h"
#include "../core/nsl_error.h"
#include "NSLCommon/nsl_common_types.h"

// =====================================================================
//  NSL Binding Resolver
//
//  Decouples logical PROPERTY block count from physical descriptor spaces.
//  Assigns valid, collision-free (set, binding) register coordinates
//  strictly within the target API's real hardware/root-signature budgets:
//
//    HLSL12 (D3D12 Root Signature: space0 .. space3):
//      - Arbitrary number of logical PROPERTY blocks supported.
//      - CBV:     16 per space (b0..b15; b1 in space3 reserved for graphics
//                 32-bit root constants -> 63 graphics CBVs / 64 compute CBVs).
//      - SRV:     32 per space (t0..t31 -> 128 total SRVs across space0..space3).
//      - Sampler: 16 per space (s0..s15 -> 64 total samplers across space0..space3).
//      - UAV:     16 per space (u0..u15 -> 64 total UAVs across space0..space3).
//
//    HLSL11 (D3D11 SM 5.0: single register space space0):
//      - Arbitrary number of logical PROPERTY blocks supported, packed into space0.
//      - CBV:     16 (b0..b15).
//      - SRV:     32 (t0..t31).
//      - Sampler: 16 (s0..s15).
//      - UAV:     16 (u0..u15).
// =====================================================================

namespace NSL
{
    class BindingResolver
    {
    public:
        static constexpr uint16_t kMaxSpaces          = 4;   // space0 .. space3
        static constexpr uint16_t kMaxCBVBindings     = 16;  // b0 .. b15 per space
        static constexpr uint16_t kMaxSRVBindings     = 32;  // t0 .. t31 per space
        static constexpr uint16_t kMaxUAVBindings     = 16;  // u0 .. u15 per space
        static constexpr uint16_t kMaxSamplerBindings = 16;  // s0 .. s15 per space

        // Resolves and assigns (set, binding) to all groups & entries in `props`.
        // Returns false if a real target hardware register budget is exceeded.
        bool Resolve(
            ParsedProperties& props,
            TargetAPI         targetAPI,
            ShaderStage       stage,
            DiagnosticEngine& diag);

        // Compatibility overload defaulting to HLSL12 / VERTEX
        bool Resolve(ParsedProperties& props, DiagnosticEngine& diag)
        {
            return Resolve(props, TargetAPI::HLSL12, ShaderStage::VERTEX, diag);
        }
    };

} // namespace NSL
