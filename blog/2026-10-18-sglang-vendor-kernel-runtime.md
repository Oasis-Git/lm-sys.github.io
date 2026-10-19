---
title: "SGLang Low-Latency Mode: Kernels and Runtime"
author: "Authors to be confirmed"
date: "October 18, 2026"
previewImg: /images/blog/sglang-vendor-kernel-runtime/runtime.svg
type: blog
---

Low-latency inference offers many opportunities for optimizations tailored to specific models, hardware and workloads. These optimizations often require extensive kernel fusion and duplicate code across implementations, making them difficult to maintain. To support this specialization, we introduce **SGLang Low-Latency Mode**. It brings specialized kernels into SGLang through a plugin framework and delivers substantial latency reductions on both AMD and NVIDIA GPUs.

*[Figure 1: Low-latency inference results]*

## Kernels for Low-Latency Inference

### AMD Low-Latency Kernels

[`sgl-amd-mono`](https://github.com/sgl-project/sgl-amd-mono) provides model-specific kernels for AMD Instinct™ MI350 Series GPUs through `--monokernel-backend amd`. Its first implementation supports **DeepSeek-V4.1-Flash with DSpark speculative decoding on MI350X and MI355X (TP2 or TP4)**. It reuses the weights, KV cache and serving scheduler managed by SGLang, and runs both the target verify step and the DSpark draft forward on its own kernels.

At low concurrency, SGLang's standard path runs each decoder layer as dozens of small kernels and two tensor-parallel all-reduces. With a draft block size of five, one request produces six rows in a target verify step. At one or two concurrent requests, one draft-and-verify cycle runs about 1,300 kernels in about 9.1 ms on MI350X. Three costs dominate at this size:

- **Launch overhead:** HIP graphs remove the CPU's launch work, but each small kernel still costs 2-4 µs on the GPU.
- **Idle boundaries:** Each kernel must finish before the next starts, so compute units idle while one kernel drains and the next ramps up, and weight loading pauses.
- **Communication stops:** Each layer waits twice for an all-reduce across GPUs, with no computation running.

**Two persistent launches per layer.** The plugin replaces this chain with mono kernels: persistent FlyDSL kernels that each run a large part of a layer in one launch, with one workgroup resident on each of the GPU's 256 compute units. **K1** applies the mHC update, computes the query and key-value projections and writes the sliding-window KV cache. **K2** runs sparse attention, the output projections, the second mHC update, the router, the shared and routed experts, and both all-reduces. On the eight layers with a compressed attention index, SGLang's compressor runs between the launches, and a third launch, **K2i**, scores the FP4 index keys for SGLang's top-k. The layer splits at attention because attention reads KV entries written by every workgroup of K1. Each of the GPU's eight XCDs has its own L2 cache, so a launch boundary is the cheapest way to make those writes visible across the GPU.

**Overlapping stages inside a launch.** Each value a stage produces is stored together with a tag for the current decode step, so the next stage, on the same GPU or another one, starts on a value as soon as its tag matches. No separate flag or reset between steps is needed. Weights that do not depend on routing load before their inputs arrive: K1's projection weights before the norm finishes, and in K2 the mHC mix weights and the shared expert's weights before the data they multiply. Routed-expert weights load once the router has chosen the experts.

**Bringing the all-reduce into the kernel.** As each output-projection tile is computed, K2 pushes it to every GPU's memory, and each GPU sums the pieces in rank order while other workgroups are still producing tiles ([push and reduce stages](https://github.com/sgl-project/sgl-amd-mono/blob/e952b221aafc05ef5adb84d2caf942f7283c1f0b/sgl_amd_mono/deepseek_v41/gfx950/stages/seam.py)). The MoE output is reduced the same way. Every cross-GPU wait has a deadline below SGLang's watchdog timeout; a missed deadline stops the launch on every GPU instead of hanging the server.

![Schematic timeline of one DeepSeek-V4.1 layer at batch size 1. The standard path runs dozens of small kernels with an idle gap at every launch boundary and two all-reduces that stop computation. With the AMD plugin, the layer runs as two FlyDSL launches, K1 and K2, where weights that do not depend on routing load early and partial sums go to peer GPUs while computation continues.](/images/blog/sglang-vendor-kernel-runtime/amd-mono-layer.svg)

*One DeepSeek-V4.1 layer at batch size 1 on SGLang's standard path and with the AMD plugin. Numbered markers pair each cost of the standard path with the change that removes it. The timeline is schematic; it does not represent measured durations.*

The supported configuration uses decode graphs. Kernels compile and buffers are allocated at model initialization, and the graph records the selected path for each covered shape. Steps wider than 48 rows (more than eight requests at block size 5), prefill and unsupported configurations use SGLang's original forward.

**AgentX results.** The comparison runs DeepSeek-V4.1-Flash on four MI355X GPUs with TP4 and DSpark block size 5. It uses the in-tree prototype of these kernels and compares it with SGLang's AgentX baseline on the same recipe. With the plugin, P90 interactivity improves by 50%, 56% and 45% at one, four and eight concurrent AgentX sessions, and total throughput per GPU improves by 23% at one session.

![Reported AgentX P90 interactivity for DeepSeek-V4.1-Flash with DSpark on four MI355X GPUs, comparing SGLang with and without the AMD plugin](/images/blog/sglang-vendor-kernel-runtime/amd-agentx-interactivity.svg)

*DeepSeek-V4.1-Flash · 4 × MI355X · TP4 · DSpark block size 5 · AgentX trace replay. "Plugin off" is SGLang's AgentX baseline; "plugin on" adds `--monokernel-backend amd`. Speculative acceptance is fixed at 3.51 tokens per verify step in both arms. P90 interactivity is the inverse of the 90th-percentile inter-token latency.*

Accuracy is reported separately. The report gives GSM8K scores of 0.911 without the plugin and 0.901 with it (1,319 questions, TP4), and 60 of 60 on needle-in-a-haystack retrieval at 32K, 64K and 120K tokens. These are the observed results; the one-point GSM8K difference does not establish an accuracy change or numerical equivalence.

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
