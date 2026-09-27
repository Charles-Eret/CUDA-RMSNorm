#include <torch/extension.h>


torch::Tensor cuda_copy(torch::Tensor input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cuda_copy",
        &cuda_copy,
        "Copy tensor using CUDA"
    );
}