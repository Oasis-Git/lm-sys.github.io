---
title: "SGLang Low-Latency Mode: Kernels and Runtime"
author: "Authors to be confirmed"
date: "October 18, 2026"
previewImg: /images/blog/sglang-vendor-kernel-runtime/runtime.svg
type: blog
---

Low-latency inference offers many opportunities for optimizations tailored to specific models, hardware and workloads. These optimizations often require extensive kernel fusion and duplicate code across implementations, making them difficult to maintain. To support this specialization, we introduce **SGLang Low-Latency Mode**. It brings specialized kernels into SGLang through a plugin framework and delivers substantial latency reductions on both AMD and NVIDIA GPUs.

<p align="center"><img src="/images/blog/sglang-vendor-kernel-runtime/amd-agentx-bs1.svg" width="100%" alt="AgentX P90 interactivity at one concurrent user for DeepSeek-V4.1-Flash on AMD Instinct MI355X: 406.5 tokens per second per user with SGLang's default path and 610.3 with Low-Latency Mode, a 50 percent gain."></p>

*Figure 1. AgentX at one concurrent user: DeepSeek-V4.1-Flash on AMD Instinct™ MI355X (TP4, DSpark). Low-Latency Mode raises P90 interactivity from 406.5 to 610.3 tokens/s per user.*

## Kernels for Low-Latency Inference

### AMD Low-Latency Kernels

Agentic workloads keep only a few requests in flight, so each decode step has little work to spread across the GPU. On AMD Instinct™ MI350 Series GPUs, SGLang's standard path runs each DeepSeek-V4.1-Flash decoder layer as dozens of small kernels and two tensor-parallel all-reduces. At batch size one, a DSpark draft-and-verify cycle runs about 1,300 kernels in 9.1 ms. At this scale, the time between kernels matters more than the computation inside them:

- **Launch overhead:** HIP graphs remove the CPU's launch work, but each small kernel still costs 2-4 µs on the GPU.
- **Idle boundaries:** Each kernel must finish before the next starts, so compute units idle while one kernel drains and the next ramps up, and weight loading pauses.
- **Communication stops:** Each layer waits twice for an all-reduce across GPUs, with no computation running.

[sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) addresses these costs with mono kernels: persistent kernels, written in FlyDSL, that each run a large part of a layer in one launch. A mono kernel keeps one workgroup on every compute unit for the whole launch. Workgroups move from stage to stage, load the next stage's weights while earlier stages compute, and pass partial results through tagged buffers, so each stage starts on the data that is ready. Each GPU writes partial sums directly into its peers' memory, so the all-reduce overlaps with computation.

![Schematic timeline of one DeepSeek-V4.1 layer at batch size 1. The standard path runs dozens of small kernels with an idle gap at every launch boundary and two all-reduces that stop computation. Low-Latency Mode runs the layer as two FlyDSL launches, K1 and K2, where weights load early and partial sums go to peer GPUs while computation continues.](/images/blog/sglang-vendor-kernel-runtime/amd-mono-layer.svg)

*One DeepSeek-V4.1 layer at batch size 1 on SGLang's standard path and in Low-Latency Mode. Numbered markers pair each cost of the standard path with the mono-kernel idea that removes it. The timeline is schematic, not measured.*

For DeepSeek-V4.1, each decoder layer runs as two launches. K1 prepares attention: it applies the mHC update, computes the query and key-value projections and writes the sliding-window KV cache. K2 runs the rest of the layer: sparse attention, the output projections, the router, the experts and both all-reduces. On the eight layers with a compressed attention index, SGLang's compressor runs between the launches, and a third launch, K2i, scores the FP4 index keys for SGLang's top-k. The layer is split at attention because attention reads KV entries written by every workgroup of K1. MI350 Series GPUs divide their compute units across eight chiplets with separate L2 caches, so a launch boundary is the cheapest way to make those writes visible across the GPU. The DSpark draft model runs on the same kernels.

FlyDSL, AMD's MLIR-based Python kernel language, gives explicit control over work assignment, memory layout, cross-GPU synchronization and instruction selection. The kernels compile at startup, run inside SGLang's HIP graphs and read the weights SGLang has already loaded. Every cross-GPU wait has a deadline below SGLang's watchdog timeout, so a failure stops the server instead of hanging it or returning a token from a failed step.

On MI355X with TP4 and DSpark, Low-Latency Mode raises AgentX P90 interactivity at one concurrent user by 50%, from 406.5 to 610.3 tokens/s per user (Figure 1), and total throughput per GPU by 23%. The gain holds at four and eight users, at 56% and 45%. Steps larger than 48 rows run SGLang's standard path. Accuracy remains comparable: GSM8K scores 0.901 with Low-Latency Mode and 0.911 without it, and needle-in-a-haystack retrieval passes 60 of 60 at up to 120K tokens.

### NVIDIA Low-Latency Kernels

## Plugin System: Design and Integration

Specialized kernels can require substantial amounts of code. Each combination of hardware, model architecture and parallel configuration may need its own implementation. Different workloads add further specialization. More combinations mean more code to review, test and maintain. Bringing every implementation into SGLang would make the core codebase increasingly difficult to maintain and place a growing workload on its reviewers and maintainers.

![External kernel libraries register model specifications with SGLang, which validates configurations, builds the plugin and dispatches supported steps.](/images/blog/sglang-vendor-kernel-runtime/plugin-architecture.svg)

*External kernel libraries connect to SGLang through registration, validation, build and runtime dispatch. The diagram shows the shared design; the AMD and NVIDIA interfaces are still evolving.*

To manage these low-latency improvements, we introduce a plugin system for external kernel libraries. [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) keep specialized kernels, weight preparation and model-specific execution code together so developers can optimize them as a unit. Each library declares its requirements and supported configurations through model specifications. SGLang uses these specifications to integrate the implementation with its serving runtime. Kernel teams can iterate in their own repositories while reusing SGLang's scheduling and runtime resources.

The integration separates startup preparation from execution:

- **Register:** A plugin registers a specification for each model architecture it supports. The specification describes the implementation's requirements and provides the entry point for building it. This gives SGLang a way to discover external implementations while keeping the details of model support in the library that owns the kernels.
- **Guard:** Specialized kernels make assumptions about hardware, precision, parallelism and serving features. SGLang checks the declared requirements before loading weights so an incompatible configuration fails early. Additional checks on the loaded model run before the plugin is built. These checks distinguish an unsupported server configuration from a supported server encountering a step outside the plugin's coverage.
- **Build:** Weight preparation and execution resources are set up during initialization so requests can reuse them. The integration accommodates the needs of each implementation: AMD can retain checkpoint tensors before weight conversion, while NVIDIA receives a binding through which it can access runtime state as that state becomes available. SGLang builds the plugin before sizing the KV cache so its allocations are accounted for, then installs dispatch in the model's forward path.
- **Runtime:** Specialized and general execution coexist because a plugin may cover only certain step types and token counts. The installed dispatcher uses the plugin's support conditions to select its forward or SGLang's original forward. Tensor-parallel ranks must agree on coverage to avoid entering incompatible collectives. This selection fits into ModelRunner's existing execution flow. During graph capture, the chosen path is recorded for that shape; serving can then replay the graph without repeating Python dispatch.

This design lets kernel teams ship focused optimizations without adding every specialized implementation to SGLang's core codebase. It concentrates integration work at explicit interfaces and allows each library to expand its coverage incrementally. Specialized kernels can serve the cases they optimize while SGLang's general forward handles the remaining steps within a supported configuration.

## Future Plans

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Broader Model Support</p>

Megakernel development is underway for GLM-5.3, K3, MiniMax M3 and other models on both AMD and NVIDIA GPUs. We plan to integrate these implementations into the plugin framework as they become ready. These low-latency kernels show promising potential for further performance improvements.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">CI and Performance Validation</p>

External kernel implementations sit outside SGLang's regular code review process. Strong CI coverage at the plugin interface is therefore essential to verify that these implementations meet SGLang's correctness and runtime requirements. We plan to test compatibility checks, weight binding, forward outputs and state updates, including KV-cache writes. End-to-end tests will also cover graph execution, fallback behavior and performance regressions as SGLang and the external libraries evolve.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Evolving Shared Plugin Interfaces</p>

Although the plugin framework is now in place, it is still at an early stage. We expect to refine its interfaces and design through long-term iteration as we integrate more kernels and learn from their use in production.
