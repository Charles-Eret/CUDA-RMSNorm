import torch

from torch.utils.cpp_extension import load


KERNEL_NAME = "rmsnorm_no_intermediate"

extension = load(
    name=f"cuda_{KERNEL_NAME}_extension",
    sources=[
        "src/bindings.cpp",
        "src/rmsnorm_naive.cu",
        "src/rmsnorm_shared.cu",
        "src/rmsnorm_no_intermediate.cu",
    ],
    extra_cflags=[
        "/Zc:preprocessor",
    ],
    extra_cuda_cflags=[
        "-Xcompiler",
        "/Zc:preprocessor",
    ],
    verbose=True,
)


def rmsnorm_torch(x, gamma, eps=1e-6):
    rms = torch.sqrt(
        torch.mean(x * x, dim=-1, keepdim=True)
        + eps
    )

    return x / rms * gamma

SHAPES = [
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
    )

    gamma = torch.randn(
        hidden_size,
        device="cuda",
    )

    expected = rmsnorm_torch(x, gamma)

    actual = getattr(extension, KERNEL_NAME)(
        x,
        gamma,
        1e-6,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    print(f"{tokens}x{hidden_size}: PASS")
