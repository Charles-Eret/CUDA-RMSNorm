# USAGE:
# ncu --profile-from-start off --set full -o profiles/fused_128x4096 python scripts/profile_rmsnorm.py --implementation fused --tokens 128 --hidden-size 4096
import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUDA_SOURCES = [
    "src/bindings.cpp",
    "src/rmsnorm_naive.cu",
    "src/rmsnorm_shared.cu",
    "src/rmsnorm_no_intermediate.cu",
    "src/rmsnorm_fused.cu",
]


def get_implementations():
    extension = load(
        name="cuda_rmsnorm_extension",
        sources=[str(PROJECT_ROOT / source) for source in CUDA_SOURCES],
        extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"] if os.name == "nt" else [],
    )

    return {
        "naive": extension.rmsnorm_naive,
        "shared": extension.rmsnorm_shared,
        "no_intermediate": extension.rmsnorm_no_intermediate,
        "fused": extension.rmsnorm_fused,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--implementation",
        choices=["naive", "shared", "no_intermediate", "fused"],
        required=True,
    )

    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(0)

    implementations = get_implementations()
    fn = implementations[args.implementation]

    eps = 1e-6

    x = torch.randn(
        args.tokens,
        args.hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    gamma = torch.randn(
        args.hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    # Warm up outside the profiled region.
    for _ in range(args.warmup):
        fn(x, gamma, eps)

    torch.cuda.synchronize()

    print(
        f"Profiling {args.implementation}: "
        f"{args.tokens}x{args.hidden_size}"
    )

    # Only this invocation should be collected by Nsight Compute
    # when ncu is run with --profile-from-start off.
    with torch.cuda.profiler.profile():
        output = fn(x, gamma, eps)
        torch.cuda.synchronize()

    print("Profiled invocation complete.")


if __name__ == "__main__":
    main()