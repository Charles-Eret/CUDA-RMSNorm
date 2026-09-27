import torch

from torch.utils.cpp_extension import load


extension = load(
    name="cuda_rmsnorm_extension",
    sources=[
        "src/bindings.cpp",
        "src/copy_kernel.cu",
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


x = torch.randn(
    1024,
    device="cuda",
    dtype=torch.float32,
)

y = extension.cuda_copy(x)


print("Input:")
print(x[:10])

print("\nOutput:")
print(y[:10])

print("\nCorrect:", torch.allclose(x, y))