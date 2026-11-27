# =====================================================================
#  compile_shaders.ps1
#  Compiles NorthstarRuntime and StandardLit .nsl shaders directly into
#  build output directories (never writes .nsc artifacts into shaders/).
#  RuntimeNVGMicropolyRaster.nsl is compiled from a single authoritative
#  NSL source into Atomic64 (NVG_RASTER_ATOMIC64=1) and Atomic32
#  (NVG_RASTER_ATOMIC64=0) .nsc binaries.
# =====================================================================

param(
    [ValidateSet("Release", "Debug", "All")]
    [string]$Configuration = "All"
)

$ErrorActionPreference = "Stop"
$repoRoot   = (Resolve-Path "$PSScriptRoot/..").Path
$includeDir = Join-Path $repoRoot "shaders"

$compilerCandidates = @(
    (Join-Path $repoRoot "build\bin\win64-release\NSLCompiler.exe"),
    (Join-Path $repoRoot "build\bin\win64-debug\NSLCompiler.exe")
)
$compiler = $compilerCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $compiler) {
    Write-Error "NSLCompiler.exe not found under build\bin\. Run build.bat first."
    exit 1
}

$configs = if ($Configuration -eq "All") { @("win64-release", "win64-debug") }
           elseif ($Configuration -eq "Debug") { @("win64-debug") }
           else { @("win64-release") }

function Compile-ShaderDirectory([string]$srcDir, [string]$relOutDir) {
    if (-not (Test-Path $srcDir)) { return }
    $nslFiles = Get-ChildItem -Path $srcDir -File -Filter "*.nsl"
    foreach ($cfg in $configs) {
        $binRoot = Join-Path $repoRoot "build\bin\$cfg"
        if (-not (Test-Path $binRoot) -and $cfg -ne $configs[0]) { continue }
        $outDir = Join-Path $binRoot $relOutDir
        if (-not (Test-Path $outDir)) {
            New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        }
        foreach ($file in $nslFiles) {
            if ($file.Name -eq "RuntimeNVGMicropolyRaster.nsl") {
                $staleA64Hlsl11 = Join-Path $outDir "RuntimeNVGMicropolyRasterAtomic64_hlsl11.nsc"
                if (Test-Path $staleA64Hlsl11) {
                    Remove-Item -Path $staleA64Hlsl11 -Force
                }
                $specializations = @(
                    @{ OutBase = "RuntimeNVGMicropolyRasterAtomic64"; Define = "NVG_RASTER_ATOMIC64=1"; Targets = @("hlsl12") },
                    @{ OutBase = "RuntimeNVGMicropolyRasterAtomic32"; Define = "NVG_RASTER_ATOMIC64=0"; Targets = @("hlsl11", "hlsl12") }
                )
                foreach ($spec in $specializations) {
                    foreach ($api in $spec.Targets) {
                        $outNsc = Join-Path $outDir "$($spec.OutBase)_${api}.nsc"
                        & $compiler --input $file.FullName --target $api --output $outNsc -I $includeDir -D $spec.Define
                        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outNsc)) {
                            Write-Error "Failed to compile $($file.FullName) [$($spec.Define)] ($api)"
                            exit 1
                        }
                        Write-Host "[OK] [$cfg] $($file.Name) [$($spec.Define)] ($api) -> $outNsc" -ForegroundColor Green
                    }
                }
                continue
            }
            foreach ($api in @("hlsl11", "hlsl12")) {
                $outNsc = Join-Path $outDir "$($file.BaseName)_${api}.nsc"
                & $compiler --input $file.FullName --target $api --output $outNsc -I $includeDir
                if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outNsc)) {
                    Write-Error "Failed to compile $($file.FullName) ($api)"
                    exit 1
                }
                Write-Host "[OK] [$cfg] $($file.Name) ($api) -> $outNsc" -ForegroundColor Green
            }
        }
    }
}

Compile-ShaderDirectory (Join-Path $repoRoot "shaders\NorthstarRuntime\Compute") "NorthstarRuntime\Compute"
Compile-ShaderDirectory (Join-Path $repoRoot "shaders\NorthstarRuntime\Depth")   "NorthstarRuntime\Depth"
Compile-ShaderDirectory (Join-Path $repoRoot "shaders\StandardLit")              "shaders\StandardLit"
