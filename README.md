# CUDA RMSNorm Optimization

A profiling-driven CUDA optimization study of RMSNorm implemented as a PyTorch C++/CUDA extension. The project starts from a simple multi-kernel baseline, then progressively improves parallelism and memory behavior through shared-memory reduction, intermediate-tensor elimination, kernel fusion, and a final Nsight-guided register-reuse optimization.

On the largest benchmarked workload (`2048 × 4096`), latency improved from **974.39 µs** for the naive CUDA implementation to **217.65 µs** with register reuse, a **4.48× overall speedup**. The optimized kernel was also **1.48× faster than `torch.nn.RMSNorm`** on this workload.

## RMSNorm

For a token with hidden dimension \(H\), RMSNorm computes

\[
y_i = \frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H} x_j^2 + \epsilon}} \cdot \gamma_i
\]

The operation has low arithmetic intensity: it performs relatively little computation per element while reading the input and scale tensors and writing the output. This makes parallel reduction, kernel-launch overhead, cache behavior, and global-memory traffic important performance considerations.

## Optimization Progression

### 1. Naive CUDA

The initial implementation used three kernels:

```text
square → RMS reduction → normalization
```

- Materialized an intermediate `squared` tensor.
- Used one reduction thread per token, which serially traversed the hidden dimension.
- Required roughly **24 B/element** of FP32 tensor traffic in the simplified traffic model.
- Exposed parallelism primarily across tokens, leaving the reduction under-parallelized when token count was small.

This provided a simple correctness and performance baseline, but left both parallelism and memory traffic as clear optimization targets.

### 2. Block-Level Shared-Memory Reduction

The reduction was changed from **one thread per token** to **one block per token**.

- Threads cooperatively traverse the hidden dimension.
- Each thread accumulates a partial sum in a register.
- Partial sums are reduced through shared memory using a tree reduction (`256 → 128 → ... → 1`).
- `__syncthreads()` is used between reduction stages.

The benefit depends strongly on workload size. In the latest benchmark:

- `1×768`: **29.33 → 33.74 µs**, so synchronization/reduction overhead outweighed the additional parallelism.
- `128×4096`: **298.30 → 39.82 µs**, a **7.49× speedup**.
- `2048×4096`: **974.39 → 546.35 µs**, a **1.78× speedup**.

This showed that exposing hidden-dimension parallelism is particularly valuable when token-level parallelism alone is insufficient.

### 3. Remove the Intermediate `squared` Tensor

The next implementation computes `x[i]²` directly while forming each thread's partial reduction sum.

- Eliminated the `squared` tensor write and reread.
- Removed the standalone square kernel.
- Reduced the pipeline from **three kernels to two**.
- Reduced simplified theoretical FP32 traffic from approximately **24 → 16 B/element**.

The no-intermediate implementation was **1.47–2.56× faster than the shared implementation** across the tested shapes. At `2048×4096`, latency decreased from **546.35 → 333.02 µs** (**1.64×**).

### 4. Fuse Reduction and Normalization

The RMS reduction and normalization stages were then fused into one kernel.

- Retained the one-block-per-token reduction.
- Thread 0 computes the inverse RMS after the shared-memory reduction.
- The inverse RMS is shared with the block.
- Threads immediately normalize the token and apply `gamma`.
- Eliminated the intermediate RMS tensor and the second kernel launch.

Fusion improved performance by **1.07–1.92× over the no-intermediate implementation** across the benchmark suite. At `2048×4096`, the improvement was smaller—**333.02 → 311.23 µs**—suggesting that fixed launch overhead becomes less important as the workload grows.

The fused implementation was faster than `torch.nn.RMSNorm` on all nine tested input shapes in the final benchmark.

### 5. Profiling-Guided Register Reuse

Nsight Compute later showed that the large fused workload was strongly memory-bandwidth bound and still read `x` once during reduction and again during normalization.

The final experiment preserves each thread's input values across the reduction:

- Each thread stores up to 16 input values in a fixed-size local array.
- Fixed loops are unrolled to encourage compiler scalarization into registers.
- The saved values are reused during normalization rather than rereading `x`.
- The simplified theoretical traffic model drops from approximately **16 → 12 B/element**.

This deliberately trades additional register pressure for lower memory traffic. The optimization is highly workload dependent and is analyzed in detail below.

## Benchmark Methodology

All implementations are checked against a PyTorch RMSNorm reference before benchmarking.

The benchmark uses:

- **FP32** inputs and weights.
- Token counts: `1`, `128`, and `2048`.
- Hidden sizes: `768`, `2048`, and `4096`.
- `torch.nn.RMSNorm` as the optimized PyTorch baseline.
- CUDA Events for GPU timing.
- 20 warm-up iterations.
- 20 timing samples, each measuring 100 calls.
- Median per-call latency across the timing samples.
- Module construction and weight initialization outside the timed region.

The results below are derived from the latest benchmark run in `rmsnorm_benchmark1.csv`.

## Performance Results

The `2048`-token workloads best show the optimization progression as the kernels move toward a memory-bandwidth-limited regime.

### Latency at 2048 Tokens

![Latency vs Hidden Size](results/latency.png)

| Hidden size | Naive | Shared | No intermediate | Fused | Register reuse | PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 768  | 187.84 µs | 114.13 µs | 72.84 µs | **45.77 µs** | 52.34 µs | 49.68 µs |
| 2048 | 489.47 µs | 279.87 µs | 171.79 µs | 128.73 µs | **110.67 µs** | 147.18 µs |
| 4096 | 974.39 µs | 546.35 µs | 333.02 µs | 311.23 µs | **217.65 µs** | 322.67 µs |

### Speedup over Naive at 2048 Tokens

![Speedup vs Hidden Size](results/speedup.png)

| Hidden size | Shared | No intermediate | Fused | Register reuse | PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 768  | 1.65× | 2.58× | **4.10×** | 3.59× | 3.78× |
| 2048 | 1.75× | 2.85× | 3.80× | **4.42×** | 3.33× |
| 4096 | 1.78× | 2.93× | 3.13× | **4.48×** | 3.02× |

The final register-reuse optimization is not universally faster. It improves the fused kernel on five of the nine tested shapes, with the largest benefit appearing at `2048×4096`. At smaller or more cache-friendly workloads, the extra register lifetime and instruction overhead can outweigh the reduction in memory traffic.

Two examples illustrate this behavior:

- `2048×4096`: **311.23 → 217.65 µs**, a **1.43× speedup** over fused.
- `128×4096`: **16.16 → 17.98 µs**, an **11.3% latency regression**.

This workload dependence motivated examining the kernels at the hardware level rather than treating any single optimization as universally beneficial.

## Nsight Compute Profiling

Nsight Compute was used to compare the two-kernel no-intermediate implementation and the fused implementation at representative workloads of `1×4096`, `128×4096`, and `2048×4096`.

The analysis focused on kernel duration, SM and memory throughput, cache behavior, device-memory traffic, occupancy, register usage, and waves per SM.

### Small Workload — `1×4096`

The reduction and fused kernels launch only one thread block.

- The fused kernel achieved only about **17% occupancy** and `0.01` waves/SM.
- Compute and memory throughput were both very low relative to peak.
- The primary limitation is therefore **insufficient parallelism and fixed execution overhead**, not compute or memory bandwidth.

Nsight kernel duration was not treated as authoritative for this workload because profiler overhead is significant relative to such a short kernel. CUDA Event measurements are used for end-to-end latency comparisons.

### Medium Workload — `128×4096`

Fusion changed both execution overhead and memory-system behavior.

- Combined no-intermediate kernel duration: approximately **27 µs**.
- Fused kernel duration: approximately **18 µs**.
- Device-memory reads fell from roughly **4.2 MB → 2.2 MB**.
- The fused kernel achieved about **73% occupancy** with only `0.71` waves/SM despite 100% theoretical occupancy.

The lower device traffic indicates that fusing the passes improved temporal locality: data read during the reduction can often still be served by cache during normalization. At this workload, available block-level parallelism is also limited.

### Large Workload — `2048×4096`

At high token count, the GPU has enough blocks to expose substantial parallelism.

For the fused kernel:

- Achieved occupancy: **~95%**
- Waves/SM: `11.38`
- Memory throughput: **~92%**
- Compute throughput: **~23%**
- Device reads: **63.60 MB**
- Device writes: **33.13 MB**

The memory subsystem is operating near its measured peak while compute utilization remains much lower, identifying **memory bandwidth as the dominant bottleneck**.

The larger working set also reduces the effectiveness of cache reuse between the reduction and normalization passes. This explains why fusion gives a much smaller incremental latency improvement at `2048×4096` than at medium workloads.

### Bottleneck Changes with Workload

The profiling results reveal three different regimes:

- **Small workloads:** fixed overhead and insufficient parallelism dominate.
- **Medium workloads:** fusion benefits strongly from cache locality, while block-level parallelism remains limited.
- **Large workloads:** enough parallelism is available to saturate the memory subsystem, making global-memory traffic the primary optimization target.

This profiling result directly motivated the final register-reuse experiment.

## Profiling-Driven Optimization: Register Reuse

### Hypothesis

At `2048×4096`, the fused kernel was already highly occupied and reached roughly 92% memory throughput, but it still read every `x` element twice.

If the first input values could be retained until normalization, the second source read could be eliminated. The expected benefit was reduced memory traffic; the expected cost was increased register pressure and potentially lower occupancy.

### Implementation

Each thread saves the values it processes during the RMS pass:

```text
load x once
    ↓
accumulate x²
    ↓
shared-memory reduction
    ↓
compute inverse RMS
    ↓
reuse saved x values for normalization
```

A fixed `float values[16]` array supports the benchmarked maximum hidden size of 4096 with 256 threads per block. The loops are unrolled to encourage the compiler to scalarize the array into registers.

### Large-Workload Result

Nsight Compute confirms the intended tradeoff at `2048×4096`:

| Metric | Fused | Register reuse |
| --- | ---: | ---: |
| Nsight kernel duration | 312.32 µs | 215.07 µs |
| Device reads | 63.60 MB | 33.58 MB |
| Device writes | 33.13 MB | 32.96 MB |
| Memory throughput | 92.27% | 92.16% |
| Compute throughput | 22.66% | 27.13% |
| Achieved occupancy | 95.02% | 80.75% |
| Registers / thread | 16 | 48 |

Device reads fell by approximately **47%**, while total device traffic fell by approximately **31%**. Despite register usage increasing from **16 → 48 registers/thread** and achieved occupancy falling from roughly **95% → 81%**, the CUDA Event benchmark improved from **311.23 → 217.65 µs**.

Memory throughput remained near **92%**, meaning the optimized kernel was still bandwidth bound, but it had substantially less data to move and therefore completed sooner. The result demonstrates that maximizing occupancy is not itself the objective: lower occupancy can be the correct tradeoff when it removes a more important bottleneck.

### Why It Does Not Help Every Workload

At `128×4096`, device reads changed only slightly—from roughly **2.18 MB → 2.12 MB**—because the fused implementation already obtained substantial reuse from the cache hierarchy.

The register-reuse kernel still paid the cost of increasing register usage from 16 to 48 registers/thread, but removed little expensive device-memory traffic. The benchmark therefore regressed from **16.16 → 17.98 µs**.

The optimization is consequently most valuable when the repeated input access would otherwise propagate far enough through the memory hierarchy to become expensive.

## Key Lessons Learned

- **Optimize the bottleneck, not an individual metric.** The fastest large-workload kernel uses more registers and has lower occupancy because reducing memory traffic matters more.
- **GPU bottlenecks are workload dependent.** The same RMSNorm operation is overhead/parallelism limited at small sizes, cache-sensitive at medium sizes, and bandwidth bound at large sizes.
- **Kernel fusion can improve more than launch overhead.** Keeping dependent passes in one kernel also improves temporal locality and can substantially reduce device-memory traffic.
- **Source-level loads are not equivalent to DRAM traffic.** Cache behavior determines whether rereading a value results in expensive device-memory traffic.
- **Simple theoretical traffic models are useful but incomplete.** They provide optimization direction, while profiling is needed to understand realized cache behavior, occupancy, and throughput.
- **An optimization does not need to win everywhere to be useful.** Register reuse substantially improves the bandwidth-bound case while regressing workloads where cache reuse already makes the second load inexpensive.
- **Benchmarking and profiling answer different questions.** CUDA Events provide stable end-to-end latency measurements, while Nsight Compute explains the hardware behavior behind those measurements.

## Limitations and Future Work

- The register-reuse implementation uses a fixed 256-thread block and stores at most 16 values per thread, so it currently targets hidden sizes up to **4096**.
- The current evaluation covers **FP32** and the nine benchmark shapes formed by three token counts and three hidden sizes.
- A single kernel configuration is used across workloads even though the profiling results show that the optimal resource tradeoff changes with shape.
- Register reuse is intentionally a targeted experiment rather than a universally optimal implementation.

Possible follow-up work includes shape-aware kernel selection, alternative block sizes, warp-shuffle reduction, vectorized memory accesses, and specialized kernels for different hidden dimensions.

## Running the Project

From the repository root, in an environment with CUDA-enabled PyTorch, Ninja, and the CUDA compiler installed (on Windows, use a shell with MSVC available):

```sh
python tests/test_rmsnorm.py
python benchmarks/benchmark_rmsnorm.py
```

## Project Structure

```text
├── src/          # CUDA implementations, PyTorch reference, and C++ bindings
├── benchmarks/   # Benchmark scripts
├── tests/        # Correctness and extension tests
├── results/      # Saved benchmark/profiling results
└── README.md
```
