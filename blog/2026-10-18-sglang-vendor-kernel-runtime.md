---
title: "SGLang Low-Latency Mode: Kernels and Runtime"
author: "Authors to be confirmed"
date: "October 18, 2026"
previewImg: /images/blog/sglang-vendor-kernel-runtime/runtime.svg
type: blog
---

*[Intro]*

*[Figure 1: Low-latency inference results]*

## Kernels for Low-Latency Inference

### AMD Low-Latency Kernels

### NVIDIA Low-Latency Kernels

[`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega) is SGLang's low-latency plugin for NVIDIA GPUs. It loads through SGLang's plugin entry point and turns on with `--enable-nvidia-megakernel`. Each supported model is a package with kernels per GPU architecture; the first is DeepSeek-V4.1-Flash with DSpark speculative decoding on GB300. At low concurrency a decode step carries at most a few dozen tokens, so its time goes to small GEMMs, kernel boundaries and tensor-parallel all-reduces, not memory bandwidth. It runs the steps it covers with its own kernels, over the weights and caches SGLang loaded:

- **Communication fused into its neighbors.** The attention output GEMM writes its rows straight into a buffer on every GPU, and one consumer kernel does the reduction, the residual mix and the normalization. The MoE output takes the same path; both build on FlashInfer's low-latency kernels.
- **MoE kernels for a handful of tokens.** A routing kernel and two block-scaled GEMMs process tokens in small tiles grouped by expert and read SGLang's quantized expert weights in place. The router's top-k matches SGLang's bit for bit.
- **Attention that loads ahead.** The attention kernel starts loading the KV cache before the kernel it depends on has finished, wherever the cache writes allow it.
- **The draft step on the GPU.** DSpark's draft proposal runs in one kernel per GPU, and the GPUs agree on each drafted token inside it, with no host-side synchronization.
- **Chained kernels with safeguards.** Kernels stay separate launches chained with programmatic dependent launch (PDL), which on GB300 costs less than synchronizing all blocks of one persistent kernel. At startup each kernel is checked against SGLang's own op and stays off if the check fails, and every cross-GPU wait has a deadline.

Prefill and every step the plugin does not cover run SGLang's own code.

**Results.** Both runs use the same SGLang build and launch on one GB300 node at TP4, including the FlashMLA attention backend and the settings the plugin requires ([listed in the plugin's pull request](https://github.com/sgl-project/sgl-nvidia-mega/pull/1)); only `--enable-nvidia-megakernel` differs. Both fix speculative acceptance at 3.51 tokens per verify step, so the comparison isolates the decode step. On a static 8K-input, 1K-output workload, the plugin raises output speed per user by 38%, 30%, 23% and 11% at one, two, four and eight concurrent requests; time to first token is unchanged. Accuracy is unchanged: GSM8K-200 scores 196 / 200 without the plugin and 197 / 200 with it (8 and 64 concurrent requests), and both pass the 16K and 24–32K long-context sets in full (96 / 96 and 32 / 32).

![Decode speed of DeepSeek-V4.1-Flash on GB300 with the NVIDIA plugin off and on](/images/blog/sglang-vendor-kernel-runtime/nvidia-static-decode.svg)

*Figure [N]. DeepSeek-V4.1-Flash with DSpark on one GB300 node (TP4, FP8 KV cache), static 8K input / 1K output, fixed acceptance of 3.51 tokens per step: tokens/s per user, same SGLang build and launch, plugin off and on.*

## Plugin System: Design and Integration

### Design Goals

### Plugin Interface and Binding

**NVIDIA plugin.** The plugin registers a spec for each model it supports. A spec states its requirements (GPU, parallelism, data types and the server settings it was validated with), and SGLang refuses a launch that misses any of them, listing every reason. The plugin reads SGLang's runtime state only through the binding: the KV pool, the running forward's attention metadata and model-specific tables, passed as data. The few calls and writes that still cross the boundary are listed in the binding file, and a test in the plugin pins them.

### Runtime Dispatch and Fallback

**NVIDIA plugin.** The plugin covers the decode steps it has built kernels for, and a draft model's spec can also supply the speculative draft proposal. Every other step runs the model's own forward. SGLang's scheduler and speculative worker are unchanged and carry no plugin hooks.

### Graph Capture and Replay

**NVIDIA plugin.** Coverage is decided from host state alone, so all ranks agree and a captured CUDA graph replays the decision made at capture. Nothing is compiled or allocated inside a step. After capture, SGLang checks that the cache buffers the plugin reads have not been reallocated.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
