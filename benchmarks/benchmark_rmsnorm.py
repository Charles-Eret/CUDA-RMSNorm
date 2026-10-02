import torch
import csv
import os
from pathlib import Path

from torch.utils.cpp_extension import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCES = [
    "src/bindings.cpp",
    "src/rmsnorm_naive.cu",
    "src/rmsnorm_shared.cu",
    "src/rmsnorm_no_intermediate.cu",
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
        "pytorch": rmsnorm_torch,
        "naive": extension.rmsnorm_naive,
        "shared": extension.rmsnorm_shared,
        "no_intermediate": extension.rmsnorm_no_intermediate,
    }

def rmsnorm_torch(x, gamma, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) # each token gets normalized with its own sum
    return x / rms * gamma

def benchmark(fn, warmup=20, iterations=100):

    # warm up gpu
    for _ in range(warmup):
        fn()

    # wait for gpu workload to finish
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        fn()

    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)

    avg_ms = elapsed_ms / iterations

    return avg_ms * 1000

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

    implementations = get_implementations()
    eps = 1e-6
    results = []

    SHAPES = [ # (num tokens, hidden size)
        (1, 768),
        (1, 4096),
        (128, 768),
        (128, 4096),
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

    record(results)

if __name__ == "__main__":
    main()
