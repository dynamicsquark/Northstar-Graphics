@echo off
setlocal enabledelayedexpansion

echo ============================================================================
echo   Northstar Graphics - Unified Build Pipeline
echo ============================================================================

cd /d "%~dp0"

set "TARGET_CONFIG=%~1"
if "%TARGET_CONFIG%"=="" set "TARGET_CONFIG=all"

if /i "%TARGET_CONFIG%"=="clean" (
    echo [CLEAN] Removing build directory...
    if exist "build" rmdir /s /q "build"
    echo [CLEAN] Clean completed.
    exit /b 0
)

if /i not "%TARGET_CONFIG%"=="release" (
    if /i not "%TARGET_CONFIG%"=="debug" (
        if /i not "%TARGET_CONFIG%"=="all" (
            echo [ERROR] Unknown configuration "%TARGET_CONFIG%".
            echo         Usage: build.bat [release ^| debug ^| all ^| clean]
            exit /b 1
        )
    )
)

:: ----------------------------------------------------------------------------
:: 1. Locate CMake
:: ----------------------------------------------------------------------------
set "CMAKE_EXE="
where cmake >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "delims=" %%I in ('where cmake') do (
        if not defined CMAKE_EXE set "CMAKE_EXE=%%I"
    )
)

if not defined CMAKE_EXE (
    set "VSWHERE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE_EXE!" (
        for /f "usebackq delims=" %%I in (`"!VSWHERE_EXE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VS_CMAKE=%%I\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if exist "!VS_CMAKE!" if not defined CMAKE_EXE set "CMAKE_EXE=!VS_CMAKE!"
        )
    )
)

if not defined CMAKE_EXE (
    if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
)

if not defined CMAKE_EXE (
    echo [ERROR] CMake 3.20+ not found. Run setup.bat after installing CMake or Visual Studio C++ tools.
    exit /b 1
)

:: ----------------------------------------------------------------------------
:: 2. Configure Unified Build Tree (if needed or relocated)
:: ----------------------------------------------------------------------------
set "NEED_CONFIGURE=0"
if not exist "build\CMakeCache.txt" (
    set "NEED_CONFIGURE=1"
) else (
    set "NORM_CD=%CD:\=/%"
    findstr /i /c:"%CD%" /c:"!NORM_CD!" "build\CMakeCache.txt" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Detected stale CMake cache path. Resetting cache...
        del /f /q "build\CMakeCache.txt" >nul 2>&1
        if exist "build\CMakeFiles" rmdir /s /q "build\CMakeFiles" >nul 2>&1
        set "NEED_CONFIGURE=1"
    )
)

if "!NEED_CONFIGURE!"=="1" (
    echo [1/2] Configuring CMake workspace [x64]...
    "!CMAKE_EXE!" -S . -B build -A x64
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] CMake configuration failed.
        exit /b !ERRORLEVEL!
    )
)

:: ----------------------------------------------------------------------------
:: 3. Build Engine, NSLCompiler, Runtime/Standard .nsc Shaders & Examples
:: ----------------------------------------------------------------------------
set "IS_NINJA=0"
findstr /i /c:"CMAKE_GENERATOR:INTERNAL=Ninja" "build\CMakeCache.txt" >nul 2>&1
if !ERRORLEVEL! equ 0 set "IS_NINJA=1"

if /i "%TARGET_CONFIG%"=="release" (
    call :run_build "Release"
    if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!
) else if /i "%TARGET_CONFIG%"=="debug" (
    call :run_build "Debug"
    if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!
) else (
    call :run_build "Release"
    if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!
    call :run_build "Debug"
    if !ERRORLEVEL! neq 0 exit /b !ERRORLEVEL!
)

echo.
echo ============================================================================
echo   Northstar Graphics - Build Completed Successfully!
echo   Output Binaries : build\bin\win64-release  ^|  build\bin\win64-debug
echo ============================================================================
exit /b 0

:run_build
set "CFG=%~1"
echo.
echo [BUILD] Building Northstar Graphics [%CFG% x64]...
if "!IS_NINJA!"=="1" (
    "!CMAKE_EXE!" --build build --config "%CFG%" --parallel
) else (
    "!CMAKE_EXE!" --build build --config "%CFG%" -- /m:1 /nr:false
)
set "BUILD_RC=!ERRORLEVEL!"
if !BUILD_RC! neq 0 (
    echo [ERROR] %CFG% build failed.
    exit /b !BUILD_RC!
)
exit /b 0
