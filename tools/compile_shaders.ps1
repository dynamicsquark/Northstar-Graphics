# =====================================================================
#  compile_shaders.ps1
#  Compiles NorthstarRuntime and StandardLit .nsl shaders directly into
#  build output directories (never writes .nsc artifacts into shaders/).
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
