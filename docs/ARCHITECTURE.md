# Northstar Graphics — Architectural Summary

For rich interactive documentation, open [`docs/index.html`](index.html) in any web browser.

---

## Module Hierarchy

1. **`common/QuarkDynSDK`** — Left-handed (`+X` Right, `+Y` Up, `+Z` Forward) C++20 math library (`Vec2/3/4`, `Mat3/4`, `Quat`, `Transform`, `AABB`, `Frustum`, `Ray`, `Sphere`) and header-only containers.
2. **`common/NSLCommon` & `libs/NSCReader`** — Binary `.nsc` shader reflection format and reader shared between `NSLCompiler` and `NorthstarCore`.
3. **`tools/NSLCompiler`** — Lexer, parser, semantic analyzer, HLSL11/HLSL12 backend transpiler, constant-buffer aligner, and `.nsc` binary writer.
4. **`northstar/core` (`NorthstarCore.dll`)** — Platform-independent rendering runtime:
   - Incremental O(1) persistent GPU Scene & skinning palette manager.
   - Reflection-driven GPU constant/structured buffer materializer.
   - **Northstar Virtual Geometry (`NVG`)**: Cluster DAG builder, 32 KiB compressed virtual page pool, elastic streaming scheduler, and two-phase HZB GPU culling pipeline.
   - **Northstar Virtual Shadows (`NVS`)**: 2-level sparse virtual shadow page hierarchy, screen-space depth feedback, indirect caster culling, and multi-bucket shadow rasterization.
5. **`northstar/platform/d3d12` (`NorthstarD3D12.dll`)** — Direct3D 12 hardware backend implementing descriptor pooling, automatic resource state tracking, PSO caching, upload rings, and `ExecuteIndirect` submission.
6. **`northstar/include` (`<northstar.h>`)** — Host C++20 SDK providing `RuntimeSession` (dynamic module loader + ABI version check), RAII `Unique*` resource handles, and Win32 macro guards.

---

## Frame Execution Flow

```text
Host Application
  ├─ engine.BeginFrame()
  │    └─ Prune retired GPU handles, advance frame clock & reset per-frame telemetry
  ├─ Scene / Camera / Light / Material updates (O(1) dirty slot tracking)
  ├─ engine.RenderFrame()
  │    ├─ Incremental GPUScene upload (coalesced dirty ranges)
  │    ├─ Per-Camera Render Loop (sorted by ascending Camera::GetDepth()):
  │    │    ├─ Depth Prepass (when required by NVS feedback or POST_DEPTH compute)
  │    │    ├─ NVS Feedback -> Residency Allocation -> Selective Slice Clear -> Caster Cull -> Indirect Shadow Raster -> Commit
  │    │    ├─ NVG Two-Phase Instance & BVH Hierarchy Cull + HZB Build + Cluster Decode / Indirect Draw Gen
  │    │    ├─ Opaque & Transparent Geometry Passes + Scheduled Scene/Material Compute Kernels
  │    │    └─ Backend ExecuteFrame() on D3D12 Command Queue
  └─ engine.EndFrame()
       └─ PrePresentCallback (e.g. Dear ImGui overlay) -> SwapChain Present()
```
