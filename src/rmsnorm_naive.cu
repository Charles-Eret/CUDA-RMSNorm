#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_kernel(
    const float* x,
    float* squared,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        squared[i] = x[i] * x[i];
    }
}

__global__ void rms_kernel_naive(
    const float* squared,
    float* rms,
    int tokens,
    int hidden_size,
    float eps
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token < tokens) {

        float sum = 0.0f;

        for(int i = 0; i < hidden_size; i++) {
            int index = token * hidden_size + i;
            sum+=squared[index];
        }

        float mean = sum / hidden_size;

        rms[token] = sqrtf(mean + eps);
    }

}

__global__ void normalize_kernel(
    const float* x,
    const float* rms,
    const float* gamma,
    float* output,
    int n,
    int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int token = i / hidden_size;
        int hidden_idx = i % hidden_size;
        output[i] = (x[i] / rms[token]) * gamma[hidden_idx];
    }
}

torch::Tensor rmsnorm_naive_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");

    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "x must be float32"
    );

    TORCH_CHECK(
        gamma.scalar_type() == torch::kFloat32,
        "gamma must be float32"
    );

    x = x.contiguous();
    gamma = gamma.contiguous();

    int tokens = x.size(0);
    int hidden_size = x.size(1);

    TORCH_CHECK(
        gamma.size(0) == hidden_size,
        "gamma size must match hidden size"
    );

    int n = tokens * hidden_size;

    auto squared = torch::empty_like(x);
    auto rms = torch::empty(
        {tokens},
        x.options()
    );
    auto output = torch::empty_like(x);

    int threads = 256;

    int element_blocks =
        (n + threads - 1) / threads;

    square_kernel<<<element_blocks, threads>>>(
        x.data_ptr<float>(),
        squared.data_ptr<float>(),
        n
    );

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );

    int token_blocks =
        (tokens + threads - 1) / threads;

    rms_kernel_naive<<<token_blocks, threads>>>(
        squared.data_ptr<float>(),
        rms.data_ptr<float>(),
        tokens,
        hidden_size,
        eps
    );

    err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );

    normalize_kernel<<<element_blocks, threads>>>(
        x.data_ptr<float>(),
        rms.data_ptr<float>(),
        gamma.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        hidden_size
    );

    err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );

    return output;
}