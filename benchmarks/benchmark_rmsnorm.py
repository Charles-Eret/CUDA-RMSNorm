import torch
import csv
import os
from pathlib import Path
import statistics

from torch.utils.cpp_extension import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCES = [
    "src/bindings.cpp",
    "src/rmsnorm_naive.cu",
    "src/rmsnorm_shared.cu",
    "src/rmsnorm_no_intermediate.cu",
    "src/rmsnorm_fused.cu",
    "src/rmsnorm_optimizedv1.cu",
]


def get_implementations():
    # Build once, before any timing. Add new CUDA source files above as needed.
    extension = load(
        name="cuda_rmsnorm_extension",
        sources=[str(PROJECT_ROOT / source) for source in CUDA_SOURCES],
        extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else [],
    )

    # Register new versions here. Each callable must accept (x, gamma, eps).
    return {
        "naive": extension.rmsnorm_naive,
        "shared": extension.rmsnorm_shared,
        "no_intermediate": extension.rmsnorm_no_intermediate,
        "fused": extension.rmsnorm_fused,
        "optimizedv1": extension.rmsnorm_optimizedv1,
    }

def rmsnorm_torch(x, gamma, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) # each token gets normalized with its own sum
    return x / rms * gamma

def benchmark_torch_rmsnorm(hidden_size, eps, gamma, x, tokens, expected, results):
    # Create optimized PyTorch RMSNorm for this hidden size
    pytorch_layer = torch.nn.RMSNorm(
        hidden_size,
        eps=eps,
        elementwise_affine=True,
        device="cuda",
        dtype=torch.float32,
    )

    pytorch_layer.weight.copy_(gamma)
    # Benchmark optimized PyTorch separately
    torch.testing.assert_close(
        pytorch_layer(x),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    latency_us = benchmark(lambda: pytorch_layer(x))

    results.append({
        "tokens": tokens,
        "hidden_size": hidden_size,
        "implementation": "pytorch",
        "latency_us": latency_us,
    })
    print(f"{tokens}x{hidden_size} pytorch: {latency_us:.3f} us")

def benchmark(fn, warmup=20, iterations=100, repeats=20):
    # Warm up GPU
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    samples_us = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(iterations):
            fn()

        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)

        avg_us = (elapsed_ms / iterations) * 1000
        samples_us.append(avg_us)

    return statistics.median(samples_us)

def record(results):
    output_path = PROJECT_ROOT / "results" / "rmsnorm_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tokens",
                "hidden_size",
                "implementation",
                "latency_us",
            ],
        )

        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")

@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The RMSNorm benchmark requires a CUDA-capable PyTorch environment.")

    torch.manual_seed(0)
    implementations = get_implementations()
    eps = 1e-6
    results = []

    # (num tokens, hidden size)
    SHAPES = [
        (1, 768),
        (1, 2048),
        (1, 4096),
        (128, 768),
        (128, 2048),
        (128, 4096),
        (2048, 768),
        (2048, 2048),
        (2048, 4096),
    ]

    for tokens, hidden_size in SHAPES:
        x = torch.randn(
            tokens,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        )

        gamma = torch.randn(
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        )

        expected = rmsnorm_torch(x, gamma, eps)
        for name, fn in implementations.items():
            # Validate outside the timed region, using the same inputs for every version.
            torch.testing.assert_close(
                fn(x, gamma, eps), expected, rtol=1e-4, atol=1e-5,
                msg=lambda msg: f"{name} ({tokens}x{hidden_size}): {msg}",
            )
            latency_us = benchmark(lambda: fn(x, gamma, eps))

            results.append({
                "tokens": tokens,
                "hidden_size": hidden_size,
                "implementation": name,
                "latency_us": latency_us,
            })
            print(f"{tokens}x{hidden_size} {name}: {latency_us:.3f} us")

        benchmark_torch_rmsnorm(hidden_size, eps, gamma, x, tokens, expected, results)

    record(results)

if __name__ == "__main__":
    main()
