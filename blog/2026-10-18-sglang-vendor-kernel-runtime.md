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

On AMD GPUs, we reduce latency by combining much of a decoder layer’s computation and communication into a few persistent kernels. These kernels live in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) which accelerates both target verification and draft execution for DeepSeek-V4.1-Flash with DSpark on AMD Instinct MI355X and MI350X GPUs.

At low concurrency, each decoder layer runs dozens of small kernels and two tensor-parallel all-reduces. Three costs dominate:

- **Launch overhead:** HIP graphs remove the CPU's launch work, but each small kernel still costs 2-4 µs on the GPU.
- **Idle boundaries:** Each kernel must finish before the next starts, so compute units idle while one kernel drains and the next ramps up, and weight loading pauses.
- **Communication stops:** Each layer waits twice for an all-reduce across GPUs, with no computation running.

![Schematic timeline of one DeepSeek-V4.1 layer at batch size 1. The standard path runs dozens of small kernels with an idle gap at every launch boundary and two all-reduces that stop computation. With the AMD plugin, the layer runs as two FlyDSL launches, K1 and K2, where weights that do not depend on routing load early and partial sums go to peer GPUs while computation continues.](/images/blog/sglang-vendor-kernel-runtime/amd-mono-layer.svg)

*One DeepSeek-V4.1 layer at batch size 1 on SGLang's standard path and with the AMD plugin. Numbered markers pair each cost of the standard path with the change that removes it. The timeline is schematic; it does not represent measured durations.*

**Two persistent launches per layer.** Persistent FlyDSL kernels combine most of each layer into two launches. **K1** handles the mHC update, query and KV projections and KV-cache writes. **K2** handles attention, output projections, the remaining mHC update, MoE and both all-reduces. Index layers add compression, **K2i** scoring and top-k between them. The launch boundary makes K1’s KV writes visible across the GPU before K2 reads them.

**Overlapping stages inside a launch.** Each value a stage produces is stored together with a tag for the current decode step, so the next stage, on the same GPU or another one, starts on a value as soon as its tag matches. No separate flag or reset between steps is needed. Weights that do not depend on routing load before their inputs arrive: K1's projection weights before the norm finishes, and in K2 the mHC mix weights and the shared expert's weights before the data they multiply. Routed-expert weights load once the router has chosen the experts.

**Bringing the all-reduce into the kernel.** As each output-projection tile is computed, K2 pushes it to every GPU's memory, and each GPU sums the pieces in rank order while other workgroups are still producing tiles ([push and reduce stages](https://github.com/sgl-project/sgl-amd-mono/blob/e952b221aafc05ef5adb84d2caf942f7283c1f0b/sgl_amd_mono/deepseek_v41/gfx950/stages/seam.py)). The MoE output is reduced the same way. Every cross-GPU wait has a deadline below SGLang's watchdog timeout; a missed deadline stops the launch on every GPU instead of hanging the server.

The AMD implementation is compatible with decode graphs and integrates directly with SGLang’s graph runner. Kernels are compiled and buffers are prepared during initialization. SGLang then captures the specialized execution path for each supported shape and replays it through the existing serving flow.

### NVIDIA Low-Latency Kernels

On NVIDIA GPUs, we reduce latency by overlapping dependent kernels and communication while specializing attention, MoE and draft sampling for small batches. These kernels live in [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) which accelerates both target verification and draft execution for DeepSeek-V4.1-Flash with DSpark on NVIDIA GB300 GPUs.

**Where the Time Goes.** At one request, DSpark verifies six token rows per step, leaving little work for each kernel while dependencies and cross-GPU communication add latency. In warmed CUDA-graph profiles, the plugin reduces the verify graph from 1,205 kernels to 1,189 but cuts the median decoder-layer latency from 96.8 µs to 68.4 µs. The gains come largely from reducing waits and overlapping work, with the largest reduction in the attention region. The table breaks down these changes before we walk through the main optimizations.

| Region of one decoder layer, one request (median) | SGLang | With plugin |
| --- | ---: | ---: |
| Attention, from the previous MoE output to the output projection | 33.4 µs | 18.3 µs |
| Attention output projection and tensor-parallel reduction | 10.9 µs | 9.4 µs |
| MoE, from routing to the reduced output | 44.4 µs | 36.5 µs |
| Whole layer, from one routing step to the next | 96.8 µs | 68.4 µs |

*Medians over 780–800 instances per arm on four GPUs. Region medians do not add up to the whole-layer median. Profiling affects timings; these measurements explain the mechanisms rather than report serving performance.*

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Low-Latency-Specific Optimizations</p>

**1. Overlap dependent kernels.** Programmatic dependent launch (PDL) lets a kernel begin setup and weight loading before its predecessor finishes. SGLang already uses PDL; the plugin releases dependencies earlier so more preparation overlaps with ongoing computation. Small-batch projection kernels and RoPE fused into query loading further shorten the attention path. At one request, independent Engram lookups and projections also run ahead on a side stream.

**2. Overlap computation and communication.** The plugin moves the cross-GPU send into the output GEMM’s epilogue. Each completed tile goes directly to peer GPUs while other tiles are still being computed. A second kernel reduces the results and applies the mHC update and normalization. Both paths still use two kernels, but the plugin starts communication earlier and avoids rereading the partial output just to send it.

![Attention output handoff in SGLang and the NVIDIA plugin for DeepSeek-V4.1-Flash on GB300](/images/blog/sglang-vendor-kernel-runtime/nvidia-output-handoff.svg)

*The plugin moves the cross-GPU send into the GEMM epilogue. Both paths use two kernels; box sizes do not represent durations.*

**3. Start expert GEMMs earlier.** For small batches, the plugin simplifies routing with a bitmap-based layout that avoids scanning and sorting. It also launches the first expert GEMM before routing finishes so setup overlaps with routing. The GEMM skips padded weight tiles that contain only zeros. Together these changes shorten the MoE path without changing the router’s selections.

![Measured MoE timeline at one request with and without the NVIDIA plugin](/images/blog/sglang-vendor-kernel-runtime/nvidia-moe-timeline.svg)

*One matched MoE trace pair. GEMM1 starts earlier and overlaps routing; its bar includes time spent waiting for routing to finish.*

**4. Fuse sequential draft sampling.** DSpark proposes draft tokens sequentially, with each proposal depending on the previous token. The plugin handles all five proposals in one cooperative kernel launch per GPU instead of a chain of sampling kernels. It keeps the head weights in registers and coordinates token selection across GPUs within the kernel. SGLang’s speculative worker and acceptance logic remain unchanged.

**Coverage.** Each path handles a range of verify rows, and dispatch looks at the row count of the CUDA graph being replayed. The table shows which of the paths above run at each concurrency we measured; the headers give requests, with verify rows in parentheses.

| Plugin path | Rows | 1 (6) | 2 (12) | 4 (24) | 8 (48) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Attention chain: early-released norm, `wq_b` and `wo_a` | 1–48 | ✓ | ✓ | ✓ | ✓ |
| Small-batch sparse attention with RoPE on load | 1–24 | ✓ | ✓ | ✓ | |
| Publishing GEMM and mailbox reduction (attention and MoE outputs) | 1–16 | ✓ | ✓ | | |
| MoE top-k, route kernel and expert GEMM pair | 1–48 | ✓ | ✓ | ✓ | ✓ |
| Low-latency `wqkv_a` GEMM and Engram prefetch | 1–8 | ✓ | | | |

Fewer of these paths apply as concurrency rises, and each kernel gets more rows to work on; over the same range the per-user gain falls from 38% to 12%. When a shape is outside a path's range, only that operation falls back to SGLang's kernel and the rest of the step stays on the plugin. The supported setup needs full decode CUDA graphs. Kernels and buffers are prepared at startup, and every captured graph records which path it took. Prefill and eager fallback steps use SGLang's forward. The plugin's model specification lists a [numerical tolerance for every replaced operation](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/spec.py#L131-L136). Several of them are exactly zero, and some replacements are not bitwise, such as the TF32 GEMM that computes the mHC coefficients.

**Serving results.** Both arms ran on the same four-GPU GB300 node with the same software, effective server arguments and environment. The only difference was `--enable-nvidia-megakernel`. We started four fresh servers in off/on/on/off order, and each one measured concurrency 1, 2, 4 and 8 forward and then in reverse. Inputs were 8,192 tokens and outputs 1,024 tokens, with identical prompts in both arms. At concurrency one, each pass warmed up for at least 60 seconds and then timed 30 requests; higher concurrency used 2C warmup and 5C timed requests. We kept all 800 timed requests.

![Per-user decode speed versus aggregate output throughput per GPU for DeepSeek-V4.1-Flash on four GB300 GPUs, with and without the NVIDIA plugin](/images/blog/sglang-vendor-kernel-runtime/nvidia-low-concurrency.svg)

*DeepSeek-V4.1-Flash · 4 × GB300 · TP4 · DSpark block size 5 · FP8 KV cache · simulated mean acceptance length 3.51 in both arms. Large markers average the four pass summaries per arm, small dots are single passes, and bars span the two fresh-launch means (they are not confidence intervals). Lines connect measured points only.*

| Requests | Per user, off | Per user, on | Change | Per GPU, off | Per GPU, on |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 745.6 | 1,030.2 | +38.2% | 165.8 | 219.3 |
| 2 | 647.6 | 837.5 | +29.3% | 266.3 | 327.1 |
| 4 | 541.0 | 658.1 | +21.6% | 408.5 | 468.6 |
| 8 | 373.0 | 416.3 | +11.6% | 552.8 | 599.2 |

Per-user decode speed, in tokens/s per user, is `(output tokens − 1) / (HTTP completion time − first text arrival)`, using the client's p50 summary. At one request, a new chunk reaches the client every 3.40 ms instead of 4.70 ms (median gap between streamed chunks, one chunk per decoding step). Aggregate throughput, in output tokens/s per GPU, is measured on its own: total output tokens over the whole timed interval, prefill included, divided by four GPUs. Time to first token barely moves (173.9 ms off, 175.5 ms on at one request) because prefill runs SGLang's path. That unchanged prefill time is why the throughput gains are smaller than the decode gains.

Both arms simulate a mean acceptance length of 3.51 so that they do the same amount of speculative work. This is a control, not a measured acceptance rate for a real workload; completed tokens per verify call ranged from 3.483 to 3.524 across the measured points. Four low-concurrency points are not a full Pareto frontier. Both arms use per-rank Engram host tables, with 100% huge-page coverage reported for both Engram layers on every rank. Pinned versions: SGLang `7b7015e72e4f` (the integration in [PR #44945](https://github.com/sgl-project/sglang/pull/44945)), plugin `d47deb3390e1`, model revision `314113f8a7da`, PyTorch 2.14.1+cu130 on CUDA 13.

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
