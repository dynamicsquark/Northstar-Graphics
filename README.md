# Northstar Graphics

### A standalone, shader-agnostic graphics engine.

Developed by **Quark Dynamics**.

---

## Overview

Northstar Graphics is a modern, high-performance graphics engine designed to provide developers with advanced rendering technologies, flexibility, and complete control over their rendering pipelines.

Rather than enforcing predefined rendering techniques, Northstar provides the infrastructure and technologies necessary to create custom rendering solutions.

Built from the ground up with independence, modularity, and performance in mind.

## Core Technologies

### Universal Reflection

The foundation of Northstar Graphics.

Universal Reflection establishes a unified interface between the engine and developer-authored shaders, enabling flexible resource management, custom material definitions, and seamless access to engine-provided data.

Its integrated Complex Structure Technology extends these capabilities to complex GPU data structures.

### Northstar Shading Language

A dedicated shader language, standard library (`NorthstarLibrary`), and compilation toolchain designed specifically for Northstar Graphics.

NSL empowers developers to author custom shaders once while maintaining seamless integration with the engine's rendering technologies through Universal Reflection — demonstrated by the built-in `StandardLit` package (4 unified `IA + NVG` geometry variants, 4 unified `IA + NVG` shadow variants, and 1 physically-based pixel shader).

### Northstar Geometry Processing Pipeline

A unified geometry processing infrastructure supporting conventional Input-Assembler (`IA`) streams, hardware instancing, static batching, 4-bone skeletal skinning, and GPU-driven virtualized cluster rendering through a single shared `ShaderSet` and `Material`.

### Tile-Based Shadow Mapping

A scalable shadow resource management technology that provides the physical foundation for Northstar's virtualized shadow architecture.

### Northstar Virtual Shadows

An advanced virtualized shadow technology designed to deliver high-detail, scalable, and resource-efficient shadows across complex environments, supporting both conventional and virtualized shadow casters.

### Northstar Virtual Geometry

A cluster-DAG virtualized geometry architecture (`NVG2`) featuring boundary-locked QEM simplification, 32 KiB compressed streaming pages, two-phase HZB occlusion culling, and a unified **Compute Software Micropolygon + Hardware Rasterizer** (`Atomic64` single-pass / `2×Atomic32` two-pass capability-gated execution) with full developer pixel-shader sovereignty.

## Capabilities

- Universal shader reflection & Complex Structure Technology (`CST`)
- Dedicated shader language (`NSL`), standard library (`NorthstarLibrary`), and compiler (`NSLCompiler`)
- Virtualized cluster-DAG geometry (`NVG`) with streaming page residency
- Hybrid compute software micropolygon & hardware `ExecuteIndirect` rasterization
- Unified `IA + NVG` shader sets (`STANDARD`, `INSTANCED`, `SKINNED`, `INSTANCED_SKINNED`)
- Virtualized shadows (`NVS`) with software + hardware shadow page rasterization
- GPU-driven persistent scene & 3-tier buffer architecture
- Hardware instancing, real static batching & GPU skeletal skinning
- Custom rendering pipelines & flexible material systems
- Compute shader execution & programmable post-processing
- Render-to-texture & shared GPU resources
- Multiple scenes and cameras
- Modular, backend-isolated rendering architecture

## Design Philosophy

### Shader Agnostic

Northstar Graphics does not impose predefined lighting algorithms, shading models, or material workflows.

Developers decide how their worlds are rendered.

### Data-Provider Architecture

**The developer owns the algorithms. Northstar owns the infrastructure.**

Northstar provides the data, resources, and rendering technologies required to bring developer-defined rendering techniques to life.

### Zero-Handholding

No hidden defaults. No implicit fallbacks. No imposed rendering behavior.

Northstar is built around explicit control and developer freedom.

### Standalone by Design

Northstar Graphics is an independent graphics engine. While developed as part of the Quark Engine ecosystem, it can operate without Quark Engine or any specific game engine architecture.

---

## Building from Source

### Requirements

- **OS:** Windows 10 / 11 (x64)
- **Compiler:** Visual Studio 2022 or newer with the **Desktop development with C++** workload (C++20)
- **Build System:** CMake 3.20+
- **GPU:** Direct3D 12 feature level 12_0+ hardware (Shader Model 6.6; capability-gated 64-bit / 2×32-bit atomic software rasterization)

### 1. Generate IDE Solution

Run `setup.bat` to verify your developer toolchain and generate the Visual Studio x64 solution inside `build\NorthstarGraphics.sln`:

```bat
setup.bat
```

### 2. Build Runtime, Shader Toolchain & Examples

Run `build.bat` to build `NSLCompiler`, compile all runtime and standard `.nsl` shaders into `build\`, and compile the engine and example applications:

```bat
:: Build both Release and Debug x64 configurations
build.bat

:: Build a single configuration
build.bat release
build.bat debug

:: Clean all generated build outputs
build.bat clean
```

Compiled binaries and runtime shader packages are placed under `build\bin\win64-release\` and `build\bin\win64-debug\`:

- **`HelloPBR.exe`** — Minimal C++20 host application demonstrating `<northstar.h>`, `RuntimeSession`, physically-based `StandardLit` materials, and virtual shadows.
- **`VirtualGeometryStudio.exe`** — Interactive inspection studio showcasing Northstar Virtual Geometry (`NVG`), Northstar Virtual Shadows (`NVS`), Dear ImGui telemetry, and Assimp 3D asset import.

---

## Documentation

Offline technical documentation is available in [`docs/`](docs/):

- [**Documentation Portal (`docs/index.html`)**](docs/index.html) — Getting started, build workflow, and example overview
- [**Architecture Guide (`docs/architecture.html`)**](docs/architecture.html) — Runtime/backend separation, GPU Scene, NVG, and NVS pipelines
- [**C++20 Host API Reference (`docs/api-reference.html`)**](docs/api-reference.html) — `RuntimeSession`, `NorthstarEngine`, RAII handles, scenes, and materials
- [**Northstar Shading Language (`docs/nsl-language.html`)**](docs/nsl-language.html) — NSL syntax, `PROPERTY` blocks, Universal Reflection, and `NSLCompiler`
- [**Architecture Summary (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md) — Markdown architectural overview

---

## Development Status

**Active Development**

Northstar Graphics is currently under active development. Technologies and capabilities are continuously being developed, tested, refined, and optimized.

Some features are experimental and are not yet considered production-ready.

## License

**Proprietary — Source-Available for Inspection Only (`All Rights Reserved`).**

This repository is published strictly for personal viewing and technical inspection. Copying, modification, redistribution, derivative works, commercial or internal use, and automated model training are strictly prohibited without prior written permission from **Quark Dynamics**. See [`LICENSE.md`](LICENSE.md) for full legal terms.

## About Quark Dynamics

Quark Dynamics is an independent technology development initiative focused on game engine development and advanced graphics technologies.

[GitHub](https://github.com/dynamicsquark)
