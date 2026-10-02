#include <torch/extension.h>


torch::Tensor rmsnorm_naive(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
);

torch::Tensor rmsnorm_shared(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rmsnorm_naive",
        &rmsnorm_naive,
        "Naive CUDA RMSNorm"
    );
    m.def(
        "rmsnorm_shared",
        &rmsnorm_naive,
        "Shared mem version CUDA RMSNorm"
    );
}