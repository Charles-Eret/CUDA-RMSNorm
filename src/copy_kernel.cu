#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void copy_kernel(
    const float* input,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i];
    }
}


torch::Tensor cuda_copy(torch::Tensor input) {

    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must be float32"
    );

    input = input.contiguous();

    auto output = torch::empty_like(input);

    int n = input.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    copy_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),    
        output.data_ptr<float>(),
        n
    );

    return output;
}