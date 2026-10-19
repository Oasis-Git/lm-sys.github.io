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

On AMD GPUs, we reduce latency by combining much of a decoder layer’s computation and communication into a few persistent kernels. These kernels live in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono), which accelerates both target verification and draft execution for DeepSeek-V4.1-Flash with DSpark. The initial implementation supports AMD Instinct MI355X and MI350X GPUs.

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

Accuracy is reported separately. The report gives GSM8K scores of 0.911 without the plugin and 0.901 with it (1,319 questions, TP4), and 60 of 60 on needle-in-a-haystack retrieval at 32K, 64K and 120K tokens. These are the observed results; the one-point GSM8K difference does not establish an accuracy change or numerical equivalence.

### NVIDIA Low-Latency Kernels

[`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega) is the NVIDIA plugin, turned on with `--enable-nvidia-megakernel`. For now it targets one configuration: **DeepSeek-V4.1-Flash with DSpark speculative decoding on four GB300 GPUs (TP4)**. The plugin works on the weights, KV cache and scheduler that SGLang already owns and takes over the covered target-verify and draft steps. Prefill and anything outside its coverage run SGLang's normal forward. On our node it lifts per-user decode speed from 745.6 to 1,030.2 tokens/s at one request (+38%). At eight concurrent requests the gain is 12%.

**What batch one looks like.** DSpark drafts five tokens per step, and the target model checks them together with the last accepted token, so every request adds six rows to a verify step. One, two, four and eight requests mean 6, 12, 24 and 48 rows. Six rows is very little work for a kernel. The attention `wo_a` projection streams 16 MiB of weights per GPU to produce them, and every tensor-parallel boundary waits for the other three GPUs. In SGLang's stock path, one verify step at one request runs 1,205 GPU kernels.

**Where the time goes.** We profiled warmed CUDA-graph replays at one request: five target-verify and five draft replays on each of the four GPUs, with CPU stage annotations and launch correlation IDs assigning GPU work to replays. The plugin removes only 16 kernels from the verify graph (1,189 instead of 1,205), yet the median decoder layer drops from 96.8 µs to 68.4 µs. Summed kernel time per replay falls by 2–5% while the replay's span falls by about a fifth, so most of the saving sits on the critical path rather than in the kernels' own work.

| Region of one decoder layer, one request (median) | SGLang | With plugin |
| --- | ---: | ---: |
| Attention, from the previous MoE output to the output projection | 33.4 µs | 18.3 µs |
| Attention output projection and tensor-parallel reduction | 10.9 µs | 9.4 µs |
| MoE, from routing to the reduced output | 44.4 µs | 36.5 µs |
| Whole layer, from one routing step to the next | 96.8 µs | 68.4 µs |

*Medians over 780–800 instances per arm (40 layers × 5 replays × 4 GPUs). Region medians do not add up to the layer median, and layers with Engram or an indexer are slower than the rest. The profiler stretches kernel durations, so we use these traces only to explain mechanisms; the serving numbers at the end come from separate runs without it.*

The output-handoff fusion is the easiest change to point at, but the attention region in front of it moved the most. We go through the layer in execution order.

**1. Launch the next kernel before its input exists.** [Programmatic dependent launch (PDL)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) lets a kernel start while its predecessor is still running. The new kernel calls `griddepcontrol.wait` only right before it reads the predecessor's output, so barrier and tensor-memory setup, weight loads and reads of KV entries the graph already wrote can all happen earlier. How early is up to the predecessor, which releases its dependents with `griddepcontrol.launch_dependents` or by finishing.

SGLang's kernels already use PDL, and some are written for it. The fused `wo_a` kernel, for example, [starts streaming its weights before it waits](https://github.com/sgl-project/sglang/blob/7b7015e72e4ffd6641159de0d8f3f466313f4e8b/python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh#L444-L461). In the stock chain, though, most kernels release their dependents late. The plugin moves those releases earlier. The MoE reduction lets the next layer's first projection launch [as soon as it starts polling](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/runtime/__init__.py#L40-L43). The [query RMSNorm](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/q_norm_quant.py#L88-L92) and the [attention combine](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/swapab.py#L42-L48) release on entry, and [`wo_a`](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/src/wo_a_fused.cu#L637-L640) releases `wo_b` right after its setup.

A few kernels in the chain are replaced too. The query and KV projections move from FlashInfer's persistent dense block-scaled GEMM to its low-latency variant. That kernel puts the weight on the 128-wide MMA dimension and the handful of token rows on the other, so the first projection runs on 56 CTAs instead of 14. The sparse attention [applies RoPE while it loads the query](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/chain.py#L1-L10), which removes the separate query RoPE kernel. In layers whose KV and index streams are joined before the query, it also [gathers KV before its wait](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/step.py#L258-L264).

In the 600 layer instances without Engram or an indexer, the next layer's first projection now launches 3.4 µs before the MoE reduction finishes (stock: 0.1 µs). The attention kernel starts 6.1 µs after the reduction instead of 17.0 µs.

The two Engram layers gain the most, 42–49 µs each. Their table lookup, quantization and WKV projection depend only on token hashes, yet the stock path runs them inside the layer. The plugin [computes them on a side stream](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/engram/prefetch.py#L1-L22) at the start of the step, with each GPU taking a quarter of the WKV output columns. The all-reduces stay on the main stream in the same order on every GPU, and the output is bitwise identical to SGLang's inline path. The [dispatch plan](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/plan.py#L26-L28) turns this prefetch on only at six rows; its authors measured it slower at twelve.

**2. Publish the projection output from the GEMM epilogue.** At the end of attention each GPU holds a partial sum of the `wo_b` projection, and the four partials have to be added before the mHC residual update and the next RMSNorm. SGLang already does this in two kernels. The GEMM writes its partial to memory. A fused kernel then [reads it back and pushes it](https://github.com/sgl-project/sglang/blob/7b7015e72e4ffd6641159de0d8f3f466313f4e8b/python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh#L356-L365) to all four GPUs with one store each, waits for the other three, and runs the reduction, mHC and normalization.

The plugin keeps two kernels but does the push inside the GEMM. While the output values are still in registers, the [publishing epilogue](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/wo_b_publish.py#L42-L80) packs eight BF16 columns with warp shuffles and writes them into every GPU's mailbox with one 16-byte NVLink multicast store (`multimem.st`). The host then [launches only the consumer](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/all_reduce_mhc.py#L1251-L1265), as the GEMM's PDL dependent. The consumer polls the mailbox, reduces, mixes and normalizes. Its code is unchanged, so moving the publish does not change its results. Each CTA sends its tile as soon as the tile is done instead of after the whole grid, and nothing reads the partial back just to communicate it. The GEMM still writes its normal output.

![Attention output handoff in SGLang and the NVIDIA plugin for DeepSeek-V4.1-Flash on GB300](/images/blog/sglang-vendor-kernel-runtime/nvidia-output-handoff.svg)

*Kernel boundaries at the attention output for one request (six verify rows). Both paths run two kernels per GPU; the plugin moves the push from the start of the second kernel into the epilogue of the first. Box sizes are not durations. The consumer comes from FlashInfer's Lamport all-reduce, which publishes in a separate kernel; against that design, the fusion also saves one launch and its grid-wide wait.*

The median handoff goes from 10.9 µs to 9.4 µs. The GEMM kernel itself changed as well, so not all of the 1.5 µs comes from the publish. The MoE output uses the same mailbox: a finalize kernel publishes the routed plus shared-expert sum, then the reduction runs. Both paths cover 1–16 rows, which in our tests means one or two requests.

**3. Start the expert GEMMs before routing finishes.** With one request, each of the six rows picks six of 384 experts, 36 token-expert pairs in total. The stock path runs a Triton top-k, a TensorRT-LLM routing kernel (via FlashInfer) that builds the expert-major permutation, the two expert GEMMs and a fused finalize/all-reduce/mHC kernel. The plugin changes the first half:

- A [top-k kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/routed_expert_gemm.py#L143-L145) with one thread per expert reproduces SGLang's router math bit for bit.
- With at most eight rows, no expert can be picked more than eight times, so each one fits in a single eight-row tile. The [route kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cuh#L29-L34) uses that to place tiles with popcounts over a presence bitmap, with no scan and no sort. Its [first instruction](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cu#L9-L13) releases GEMM1, so GEMM1's barrier and tensor-memory setup overlaps routing.
- Each GPU holds 576 intermediate columns per expert, padded to 640 in the shared weight layout. GEMM1 [never loads the all-zero tile](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/routed_expert_gemm.py#L259-L263) and runs 9 of the stock kernel's 10 weight tiles per expert.

Tile height is not a difference: the stock TensorRT-LLM kernels also use eight-row tiles per expert and add tiles for experts chosen by more than eight rows. Across all 800 MoE windows per arm, routing finishes 5.2 µs after it starts (stock: 9.6 µs) and GEMM1 starts at 2.6 µs (stock: 9.3 µs). The median window shrinks from 44.4 to 36.5 µs. GEMM2 and the reduction take about as long as before.

![Measured MoE timeline at one request with and without the NVIDIA plugin](/images/blog/sglang-vendor-kernel-runtime/nvidia-moe-timeline.svg)

*One matched pair from the same GPU, layer and replay, chosen because both windows lie closest to their arm's median. Bars show measured start times and durations on a shared scale, overlap included; a PDL kernel's bar includes the time it spends waiting for its producer. Shared-expert and mHC-coefficient kernels run on side streams.*

The plugin's GEMM1 bar is longer than the stock one. It launches earlier, and its measured duration includes the wait for routing; the window still ends 7.9 µs sooner.

**4. Keep sequential draft sampling in one kernel.** DSpark's Markov head proposes draft tokens one at a time, and each proposal depends on the previous token. The stock draft graph runs six dependent kernels per position (embedding lookup, head GEMM, partial argmax, NVLink exchange, a second partial pass and a finish kernel), 30 for the five positions. The plugin's [Markov kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/markov.cu#L9-L44) handles all five positions in one cooperative launch per GPU. Each warp holds its 16-row slice of the vocabulary-sharded head in registers for the whole launch. GPUs swap 64-bit candidate keys through a symmetric-memory mailbox that doubles as the grid barrier and the cross-GPU argmax. Every wait has a deadline, so a stuck peer produces a fault instead of a hang. The proposal span drops from 85.1 µs to 34.2 µs (median of 20 replays per arm). The draft graph goes from 167 kernels to 97, and the rest of that drop comes from the draft model's own attention, MoE and projection kernels. SGLang's speculative worker and acceptance logic are untouched.

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
