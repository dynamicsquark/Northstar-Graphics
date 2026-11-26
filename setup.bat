@echo off
setlocal enabledelayedexpansion

echo ============================================================================
echo   Northstar Graphics SDK - Developer Environment Setup
echo ============================================================================
echo.

cd /d "%~dp0"

:: ----------------------------------------------------------------------------
:: 1. Locate CMake (System PATH or Visual Studio Bundled Installation)
:: ----------------------------------------------------------------------------
set "CMAKE_EXE="
where cmake >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "delims=" %%I in ('where cmake') do (
        if not defined CMAKE_EXE set "CMAKE_EXE=%%I"
    )
)

set "VSWHERE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALL_DIR="
if exist "%VSWHERE_EXE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE_EXE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if not defined VS_INSTALL_DIR set "VS_INSTALL_DIR=%%I"
    )
)

if not defined CMAKE_EXE (
    if defined VS_INSTALL_DIR (
        set "VS_CMAKE=!VS_INSTALL_DIR!\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if exist "!VS_CMAKE!" set "CMAKE_EXE=!VS_CMAKE!"
    )
)

if not defined CMAKE_EXE (
    if exist "%ProgramFiles%\CMake\bin\cmake.exe" (
        set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
    )
)

if not defined CMAKE_EXE (
    echo [ERROR] CMake 3.20+ was not found on PATH or inside Visual Studio.
    echo         Please install CMake or the Visual Studio "C++ CMake tools for Windows" component.
    goto :fail
)

echo [OK] CMake Toolchain : "!CMAKE_EXE!"
if defined VS_INSTALL_DIR (
    echo [OK] Visual Studio   : "!VS_INSTALL_DIR!"
) else (
    echo [WARN] Visual Studio C++ x64 toolset was not detected via vswhere; relying on system CMake generator.
)

:: ----------------------------------------------------------------------------
:: 2. Verify Vendored Third-Party Dependencies
:: ----------------------------------------------------------------------------
set "MISSING_DEPS=0"
if not exist "thirdparty\imgui\imgui.h" (
    echo [ERROR] Missing dependency: thirdparty\imgui
    set "MISSING_DEPS=1"
)
if not exist "thirdparty\stb\stb_image.h" (
    echo [ERROR] Missing dependency: thirdparty\stb
    set "MISSING_DEPS=1"
)
if not exist "thirdparty\assimp\include\assimp\Importer.hpp" (
    echo [ERROR] Missing dependency: thirdparty\assimp\include
    set "MISSING_DEPS=1"
)
if not exist "thirdparty\assimp\lib\assimp-vc143-mt.lib" (
    echo [ERROR] Missing dependency: thirdparty\assimp\lib\assimp-vc143-mt.lib
    set "MISSING_DEPS=1"
)
if not exist "thirdparty\assimp\bin\assimp.dll" (
    echo [ERROR] Missing dependency: thirdparty\assimp\bin\assimp.dll
    set "MISSING_DEPS=1"
)

if "!MISSING_DEPS!"=="1" goto :fail
echo [OK] Third-Party SDK : Verified (Dear ImGui, ImGuizmo, stb, Assimp x64)

:: ----------------------------------------------------------------------------
:: 3. Reset Stale CMake Cache (if repository directory was moved)
:: ----------------------------------------------------------------------------
if exist "build\CMakeCache.txt" (
    set "NORM_CD=%CD:\=/%"
    findstr /i /c:"%CD%" /c:"!NORM_CD!" "build\CMakeCache.txt" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Detected relocated workspace path. Refreshing CMake cache...
        del /f /q "build\CMakeCache.txt" >nul 2>&1
        if exist "build\CMakeFiles" rmdir /s /q "build\CMakeFiles" >nul 2>&1
    )
)

:: ----------------------------------------------------------------------------
:: 4. Generate Unified x64 Build System & Visual Studio Solution
:: ----------------------------------------------------------------------------
echo.
echo [SETUP] Configuring Northstar Graphics x64 project into .\build ...
"!CMAKE_EXE!" -S . -B build -A x64
if !ERRORLEVEL! neq 0 (
    echo [ERROR] CMake project generation failed.
    goto :fail
)

echo.
echo ============================================================================
echo   Setup Completed Successfully!
echo ----------------------------------------------------------------------------
echo   Visual Studio Solution : build\NorthstarGraphics.sln
echo   Command-Line Build     : build.bat [release ^| debug ^| all ^| clean]
echo ============================================================================
goto :done

:fail
echo.
echo [FAILED] Environment setup did not complete.
exit /b 1

:done
exit /b 0
