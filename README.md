# RMSNorm

A PyTorch RMSNorm reference with naive and shared-memory CUDA implementations.

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

- **Results**
  - Little/no benefit across all input sizes.

- **Hypothesis**
  - Parallelizing the reduction addressed the naive kernel's insufficient parallelism, but the modest end-to-end improvement suggests it was not the dominant bottleneck.
  - Excessive global-memory traffic from the intermediate `squared` tensor remains.

- **Next Step**
  - Remove `squared` and compute `x[i]²` directly during the per-thread reduction, eliminating its global-memory write/read.
