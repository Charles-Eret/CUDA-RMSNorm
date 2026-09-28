#include <torch/extension.h>


// torch::Tensor cuda_copy(torch::Tensor input);

torch::Tensor rmsnorm_naive_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def(
    //     "cuda_copy",
    //     &cuda_copy,
    //     "Copy tensor using CUDA"
    // );

    m.def(
        "rmsnorm_naive",
        &rmsnorm_naive_cuda,
        "Naive CUDA RMSNorm"
    );
}