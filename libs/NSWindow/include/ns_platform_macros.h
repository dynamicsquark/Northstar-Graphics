#pragma once

// =====================================================================
//  Northstar Platform & Architecture Detection Macros
// =====================================================================

// Operating System Detection
#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || defined(__CYGWIN__)
    #define NS_PLATFORM_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #define NS_PLATFORM_IOS 1
    #elif TARGET_OS_MAC
        #define NS_PLATFORM_MACOS 1
    #endif
#elif defined(__ANDROID__)
    #define NS_PLATFORM_ANDROID 1
    #define NS_PLATFORM_LINUX 1
#elif defined(__linux__) || defined(__gnu_linux__)
    #define NS_PLATFORM_LINUX 1
#elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
    #define NS_PLATFORM_FREEBSD 1
#else
    #define NS_PLATFORM_UNKNOWN 1
#endif

// Architecture Detection
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    #define NS_ARCH_X64 1
#elif defined(__i386__) || defined(_M_IX86)
    #define NS_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define NS_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
    #define NS_ARCH_ARM 1
#else
    #define NS_ARCH_UNKNOWN 1
#endif

// Compiler Detection
#if defined(_MSC_VER)
    #define NS_COMPILER_MSVC 1
#elif defined(__clang__)
    #define NS_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define NS_COMPILER_GCC 1
#else
    #define NS_COMPILER_UNKNOWN 1
#endif

// Configuration Detection
#if defined(_DEBUG) || defined(DEBUG)
    #define NS_CONFIG_DEBUG 1
#else
    #define NS_CONFIG_RELEASE 1
#endif

// Compiler-Specific Keywords & Utilities
#if defined(NS_COMPILER_MSVC)
    #define NS_FORCEINLINE __forceinline
    #define NS_INLINE      inline
#elif defined(NS_COMPILER_CLANG) || defined(NS_COMPILER_GCC)
    #define NS_FORCEINLINE inline __attribute__((always_inline))
    #define NS_INLINE      inline
#else
    #define NS_FORCEINLINE inline
    #define NS_INLINE      inline
#endif

#define NS_NODISCARD [[nodiscard]]
#define NS_UNUSED(var) (void)(var)
