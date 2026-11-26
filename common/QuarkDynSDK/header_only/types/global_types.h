#pragma once

#include <cstdint>
#include <cstddef>

// =====================================================================
//  global_types.h — QuarkDyn SDK Universal Primitive Engine Types
// =====================================================================

namespace QuarkDyn
{
    // Unsigned integers
    using UINT8   = uint8_t;
    using UINT16  = uint16_t;
    using UINT32  = uint32_t;
    using UINT64  = uint64_t;

    // Signed integers
    using INT8    = int8_t;
    using INT16   = int16_t;
    using INT32   = int32_t;
    using INT64   = int64_t;

    // Floating point
    using FLOAT32 = float;
    using FLOAT64 = double;

    // Memory & Byte types
    using SIZE_T  = size_t;
    using BYTE    = uint8_t;

#ifndef GLOBAL_TYPES_SKIP_BOOL
    using BOOL    = bool;
#endif

    using NativeHandle = UINT64;
} // namespace QuarkDyn
