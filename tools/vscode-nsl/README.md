# Northstar Shader Language (NSL) — VS Code / Cursor Extension

Official VS Code & Cursor extension for **Northstar Shader Language (.nsl)**.
Provides rich syntax highlighting, language configuration, code folding, and production-ready snippets for the NSL 3.2 compiler.

---

## Features (v3.2.0)

* **Stage Directives**:
  * First-class syntax highlighting for `STAGE <stage>` and `STAGE_DEF <stage>` (`VERTEX`, `PIXEL`, `COMPUTE`, `GEOMETRY`, `HULL`, `DOMAIN`).
  * Backward-compatible support for `#shader <stage>`.
* **Module & Library System**:
  * `LIBRARY_DECLARATION { Add("...") }` block highlighting for native NSL imports.
  * `LIBRARY { ... }` blocks for reusable shader libraries and header-only modules.
* **Top-Level Block Recognition**:
  * `PROPERTY`, `INPUT_LAYOUT`, `OUTPUT_LAYOUT`, `SHADER`, `LIBRARY_DECLARATION`, `LIBRARY`.
* **Resource Types**:
  * `Buffer`, `Texture2D`, `TextureCube`, `Texture2DArray`, `TextureCubeArray`, `RWTexture2D`, `SamplerState`, `SamplerComparisonState`, `StructuredBuffer`, `RWStructuredBuffer`.
* **Data Types**:
  * Scalars: `Float`, `Double`, `Bool`, `Int`, `UInt`, `void`.
  * Vectors: `Vec2`, `Vec3`, `Vec4`, `Int2`, `Int3`, `Int4`, `UInt2`, `UInt3`, `UInt4`.
  * Matrices: `Mat3`, `Mat4`.
  * Colors & Rotations: `Color`, `Quat`.
* **Engine Catalog & Stage I/O**:
  * Full autocomplete & syntax highlighting for all engine `Get(...)` properties (Camera, Lighting, Geometry/GPUScene `GEOMETRY_MODE` / `GPU_SCENE_OBJECT_SLOT`, Northstar Virtual Geometry (`NVG_DRAW_BUCKET_ID`, `NVG_VISIBLE_CLUSTER_COUNT`, `NVG_VISIBLE_CLUSTER_OFFSET`, `NVG_PAGE_POOL_WORD`, `NVG_BUCKET_*`, `NVG_VIS_*`, `NVG_OBJ_*`, `NVG_SHADOW_PAGE_*`), Northstar Virtual Shadows (`NVS_SHADOW_MAP`, `NVS_COMPARISON_SAMPLER`, `NVS_DEPTH_SAMPLER`, `NVS_VPAGE_*`, `NVS_PPAGE_*`, `NVS_VIEW_*`, `NVS_SPARSE_*`, `NVS_SCREEN_PIXEL_WORLD_SCALE`), and GPU Skinning Palette `GLOBAL_BONE_MATRIX` / `BONE_MATRIX` / `BONE_PALETTE_OFFSET` / `BONE_COUNT`).
  * Stage inputs/outputs: `VERTEX_POSITION`, `VERTEX_BONE_INDICES`, `VERTEX_BONE_WEIGHTS`, `CLIP_POSITION`, `RENDER_TARGET_ARRAY_INDEX`, `VIEWPORT_ARRAY_INDEX`, `WORLD_POSITION`, `COLOR_TARGET0..7`, `DISPATCH_THREAD_ID`, etc.
* **Intrinsics**:
  * Vector math: `Mul`, `Dot`, `Cross`, `Normalize`, `Length`, `Distance`, `Reflect`, `Refract`, `Transpose`, `Determinant`.
  * Bitcast & bit-ops: `AsFloat`, `AsInt`, `AsUInt`, `CountBits`, `FirstBitHigh`, `FirstBitLow`.
  * Sampling, loading & discard: `Sample`, `SampleLevel`, `SampleCmp`, `SampleCmpLevelZero`, `Load`, `Clip`, `GetDimensions`.
  * Compute atomics & barriers: `InterlockedAdd`, `InterlockedOr`, `InterlockedAnd`, `InterlockedMin`, `InterlockedMax`, `InterlockedExchange`, `InterlockedCompareExchange`, `GroupMemoryBarrierWithGroupSync`, `DeviceMemoryBarrierWithGroupSync`, `AllMemoryBarrierWithGroupSync`.
  * Streams: `Append`, `RestartStrip`.
* **Production Snippets**:
  * Modern stage skeletons (`nsl-vertex`, `nsl-vertex-skinned`, `nsl-pixel`, `nsl-compute`, `nsl-geometry`, `nsl-hull`, `nsl-domain`).
  * Library module skeleton (`nsl-library`).
  * Library import block (`libdecl`).
  * Northstar Virtual Shadows helpers (`nsl-nvs-eval`, `nsl-nvs-eval-5x5`, `nsl-nvs-mip-pcf3`, `nsl-nvs-lookup`, `nsl-nvs-prop`).

---

## Installation

### For Cursor:

Open PowerShell and run:

```powershell
$src  = "tools\vscode-nsl"
$dest = "$env:USERPROFILE\.cursor\extensions\northstar.nsl-3.2.0"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cursor\extensions\northstar.nsl-*" -ErrorAction SilentlyContinue
Copy-Item -Recurse $src $dest
```

Then in Cursor: `Ctrl+Shift+P` → **Developer: Reload Window**.

---

### For VS Code:

Open PowerShell and run:

```powershell
$src  = "tools\vscode-nsl"
$dest = "$env:USERPROFILE\.vscode\extensions\northstar.nsl-3.2.0"
Remove-Item -Recurse -Force "$env:USERPROFILE\.vscode\extensions\northstar.nsl-*" -ErrorAction SilentlyContinue
Copy-Item -Recurse $src $dest
```

Then in VS Code: `Ctrl+Shift+P` → **Developer: Reload Window**.

---

## Snippet Reference

| Prefix | Description |
|---|---|
| `stage` | `STAGE <stage>` directive (modern) |
| `stage_def` | `STAGE_DEF <stage>` directive |
| `shader` | `#shader <stage>` legacy directive |
| `libdecl` | `LIBRARY_DECLARATION { Add(...) }` import block |
| `nsl-library` | Full NSL reusable library module skeleton |
| `nsl-vertex` | Vertex shader skeleton |
| `nsl-pixel` | Lit pixel shader skeleton with shadow evaluation |
| `nsl-compute` | Compute shader skeleton with `[numthreads]` |
| `nsl-geometry` | Geometry shader skeleton with stream `Append` |
| `nsl-hull` | Hull shader skeleton with `PatchConstant` |
| `nsl-domain` | Domain shader skeleton with `PATCH_CONSTANT_DATA` |
| `nsl-nvs-eval` | Northstar Virtual Shadows 3x3 PCF shadow evaluation |
| `nsl-nvs-eval-5x5` | Northstar Virtual Shadows 5x5 PCF shadow evaluation |
| `nsl-nvs-mip-pcf3` | Northstar Virtual Shadows Spot/Point virtual mip-hierarchy 3x3 PCF |
| `nsl-nvs-lookup` | Explicit Northstar Virtual Shadows virtual-to-physical lookup |
| `nsl-nvs-prop` | Engine-owned `PROPERTY NorthstarNVS` block |
| `alias` | `Type name = Property.Resource.Data;` alias declaration |
| `property` | Resource group block definition |
| `input_layout` | `INPUT_LAYOUT { Get(...) }` block |
| `output_layout` | `OUTPUT_LAYOUT { Add(...) }` block |
| `main` | Entry point `STAGE_OUTPUT Main(STAGE_INPUT input)` |

---

## Package as VSIX (Optional)

```powershell
npm install -g @vscode/vsce
cd tools/vscode-nsl
vsce package --allow-missing-repository
```
