# RMSNorm

A PyTorch RMSNorm reference with naive, shared-memory, no-intermediate, and fused CUDA implementations.

## Test and benchmark

From the repository root, in an environment with CUDA-enabled PyTorch, Ninja,
and the CUDA compiler installed (on Windows, use a shell with MSVC available):

```sh
python tests/test_rmsnorm.py
python benchmarks/benchmark_rmsnorm.py
```

## Project structure

```text
├── src/
│   ├── rmsnorm.py              # PyTorch reference and basic verification
│   ├── rmsnorm_naive.cu        # Naive CUDA RMSNorm implementation
│   ├── rmsnorm_shared.cu       # Shared-memory CUDA RMSNorm implementation
│   ├── copy_kernel.cu          # CUDA tensor-copy kernel
│   └── bindings.cpp            # C++ bindings for the CUDA implementations
├── benchmarks/
│   └── benchmark_rmsnorm.py    # Compare PyTorch and CUDA implementations
├── tests/
│   ├── test_extension.py       # CUDA tensor-copy extension test
│   └── test_rmsnorm.py         # CUDA RMSNorm correctness checks
├── results/
│   ├── pytorch_baseline.csv    # Saved PyTorch baseline timings
│   └── rmsnorm_benchmark.csv   # Saved implementation benchmark timings
└── README.md
```

# PHASES

## Phase 5 — Naive RMSNorm Performance Analysis

- **Design**
  - Three kernels: square → RMS reduction → normalization.
  - Uses an intermediate `squared` tensor.
  - Rough FP32 traffic: ~24 B/element.
  - Arithmetic intensity is very low: ~0.17 FLOP/B.

- **Results**
  - Naive CUDA is faster for very small shapes.
  - Performance degrades relative to PyTorch as input size grows.
  - Estimated effective bandwidth is ~30 GB/s for `128×768` and `128×4096`, but ~170 GB/s for `2048×4096`.

- **Hypothesis**
  - Extra global-memory traffic from writing and rereading `squared` hurts larger workloads.
  - The RMS reduction is under-parallelized: one thread handles one token and loops serially over the hidden dimension.
  - More tokens expose more independent reduction threads, improving GPU utilization and latency hiding.

- **Next Steps**
  - Use one block per token.
  - Parallelize the hidden-dimension reduction across threads.
  - Reduce thread partial sums using shared memory.
  - Then remove the intermediate `squared` tensor to reduce global-memory traffic.

  ## Phase 6–7 — Block-Level Shared-Memory Reduction

- **Design**
  - Changed the reduction from **1 thread → 1 token** to **1 block → 1 token**, with threads cooperatively processing the hidden dimension.
  - Each thread computes a local partial sum in a register, then stores it in shared memory.
  - Perform a tree reduction (`256 → 128 → ... → 1`) with `__syncthreads()` between stages; thread 0 computes the final RMS.
  - Global-memory traffic was otherwise unchanged.

- ### **Results**
- Shared reduction was **worse for small hidden dimensions**: `1×768` was ~2× slower than naive due to the overhead of shared-memory writes, synchronization, and the reduction tree; with `1×4096`, the additional parallelism outweighed this overhead and shared was ~2× faster.
- Shared was most beneficial when the naive kernel had limited token-level parallelism: ~**9× faster** for `128×4096`.
- As token count increased, naive gained more independent work across tokens, narrowing the performance gap: at `2048×4096`, shared was ~**1.8× faster** rather than ~9×.

### **Hypothesis**
- For **small hidden dimensions**, each thread has little useful work, so the overhead of cooperative reduction can exceed the benefit of parallelizing the hidden dimension.
- For **small token counts**, the naive kernel is under-parallelized because it can only exploit parallelism across tokens; parallelizing the hidden dimension greatly reduces the reduction's critical path.
- As **token count increases**, the naive kernel exposes more independent work across tokens and can better utilize the GPU, reducing the relative benefit of hidden-dimension parallelism.
- Global-memory traffic remains unchanged, leaving the intermediate `squared` tensor as a target for the next optimization.

- **Next Step**
  - Remove `squared` and compute `x[i]²` directly during the per-thread reduction, eliminating its global-memory write/read.

## Phase 8 — Remove Intermediate `squared` Tensor

- **Design**
  - Removed the `squared` tensor and compute `x[i]²` directly during the shared-memory reduction.
  - Reduced the theoretical global-memory traffic from **~24 → ~16 B/element**.
  - Eliminated the separate square kernel, reducing the pipeline from **3 → 2 kernels**.

- **Results**
  - No-intermediate was faster across all tested shapes, with speedups of **~1.1–1.9×** over the shared version.
  - Largest improvement was `128×4096`: **193.7 → 101.6 µs (~1.91×)**.
  - `2048×4096`: **545.4 → 330.6 µs (~1.65×)**.

- **Hypothesis**
  - Removing `squared` should improve performance by reducing global-memory traffic and eliminating a kernel launch.
  - A traffic-only model predicts an eventual **24/16 = 1.5×** speedup if both versions achieve the same effective bandwidth.
  - The measured results do not consistently follow this simple model, suggesting that effective memory-system behavior and/or other kernel-level costs differ between implementations.

- **Next Step**
  - Fuse the RMS reduction and normalization into a single kernel, eliminating the `rms` global-memory write/read and another kernel launch.


## Phase 9 — Fuse RMS Reduction and Normalization

- **Design**
  - Fused the RMS reduction and normalization stages into a **single CUDA kernel**.
  - Kept the existing **one-block-per-token** shared-memory reduction.
  - After reduction, thread 0 computes the inverse RMS and stores it in shared memory for use by the entire block.
  - Threads then cooperatively normalize the token and apply `gamma` using coalesced global-memory accesses.
  - Eliminated the intermediate `rms` tensor and reduced the pipeline from **2 → 1 kernel**.

- **Results**
  - Fusion improved performance across all tested shapes, with incremental speedups of **~1.06–3.12×** over the Phase 8 no-intermediate implementation.
  - Largest improvement was `128×768`: **33.31 → 10.66 µs (~3.12×)**.
  - `1×4096`: **22.85 → 13.17 µs (~1.73×)**.
  - `128×4096`: **25.79 → 19.13 µs (~1.35×)**.
  - For the largest workload, `2048×4096`, the improvement narrowed to **331.65 → 312.08 µs (~1.06×)**.
  - Relative to naive CUDA, the fused implementation achieved up to **~21.2× overall speedup** at `128×4096`.

- **Hypothesis**
  - Fusion eliminates the second kernel launch and the global-memory write/read of the per-token RMS value.
  - The eliminated RMS traffic is small relative to the per-element tensor traffic, so the performance gain is unlikely to come primarily from reduced DRAM traffic.
  - The large speedups on smaller and medium-sized workloads suggest that **kernel-launch and inter-kernel overhead represent a significant fraction of execution time** for these shapes.
  - As workload size increases, useful memory and arithmetic work dominates execution time, so eliminating one kernel launch has a smaller relative impact. This is consistent with the much smaller improvement at `2048×4096`.
  - Nsight profiling is needed to verify these hypotheses and determine whether changes in memory-system behavior or other kernel-level effects also contribute.

- **Next Step**
  - Benchmark the fused implementation against optimized PyTorch RMSNorm across representative input shapes, then use Nsight Compute to investigate the hardware-level behavior of the Phase 8 and Phase 9 kernels.

  ## Phase 10 — Benchmark Against Optimized PyTorch

- **Results**
  - The fused CUDA implementation was competitive with, and in the current benchmark run faster than, `torch.nn.RMSNorm` across all tested workloads.
  - The largest relative advantage appeared for very small token counts, where the fused implementation was substantially faster than PyTorch.
  - As token count and total workload size increased, the performance gap narrowed significantly.
  - For the largest workloads, the fused implementation and optimized PyTorch RMSNorm achieved very similar latency.
  - The optimization progression from naive CUDA → shared-memory reduction → removal of the intermediate tensor → kernel fusion produced substantial overall improvement, bringing the custom implementation from well behind PyTorch on larger workloads to roughly comparable performance.

- **Hypothesis**
  - For very small workloads, execution time appears to be dominated by fixed costs rather than arithmetic or memory traffic. The fused kernel's low fixed overhead therefore provides a relatively large advantage.
  - With only one or a few token blocks, the GPU is significantly underutilized, and increasing hidden dimension adds relatively little latency because the additional per-thread work remains small compared with fixed kernel and synchronization costs.
  - As token count increases, the GPU receives substantially more useful work and memory traffic becomes a larger component of total execution time.
  - At large workloads, the performance of the fused implementation converges toward optimized PyTorch, suggesting that both implementations may be approaching a similar hardware bottleneck.
  - These explanations are preliminary; profiling is required before attributing the differences to specific effects such as memory bandwidth, occupancy, register pressure, or kernel overhead.

- **Next Step**
  - Use Nsight Compute to compare the Phase 8 and fused Phase 9 kernels at representative workloads and identify the dominant hardware-level bottlenecks.