# =====================================================================
#  install.ps1 — Installs / Updates NSL Extension for Cursor & VS Code
# =====================================================================

$src = $PSScriptRoot

# 1. Cursor
$cursorExtDir = "$env:USERPROFILE\.cursor\extensions"
if (Test-Path $cursorExtDir) {
    Remove-Item -Recurse -Force "$cursorExtDir\northstar.nsl-*" -ErrorAction SilentlyContinue
    $cursorDest = "$cursorExtDir\northstar.nsl-3.2.0"
    Copy-Item -Recurse $src $cursorDest
    Write-Host "[OK] Installed NSL extension to Cursor: $cursorDest" -ForegroundColor Green
} else {
    Write-Host "[INFO] Cursor extensions folder not found at $cursorExtDir" -ForegroundColor Yellow
}

# 2. VS Code
$vscodeExtDir = "$env:USERPROFILE\.vscode\extensions"
if (Test-Path $vscodeExtDir) {
    Remove-Item -Recurse -Force "$vscodeExtDir\northstar.nsl-*" -ErrorAction SilentlyContinue
    $vscodeDest = "$vscodeExtDir\northstar.nsl-3.2.0"
    Copy-Item -Recurse $src $vscodeDest
    Write-Host "[OK] Installed NSL extension to VS Code: $vscodeDest" -ForegroundColor Green
} else {
    Write-Host "[INFO] VS Code extensions folder not found at $vscodeExtDir" -ForegroundColor Yellow
}

Write-Host "`nTo apply changes: Press Ctrl+Shift+P in Cursor/VS Code and run 'Developer: Reload Window'." -ForegroundColor Cyan

