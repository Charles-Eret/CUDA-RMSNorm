#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rms_kernel_no_intermediate(
    const float* x,
    float* rms,
    int hidden_size,
    float eps
) {
    int token = blockIdx.x;

    __shared__ float shared[256];
    float local_sum = 0.0f;

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) { // use this indexing for better mem coalescing
        int index = token * hidden_size + h;
        local_sum += x[index] * x[index];
    }
    shared[threadIdx.x] = local_sum;

    __syncthreads(); //sync all warps from the same block before continuing

    // reduction loop
    for (int stride = 128; stride > 0; stride /= 2) { // indexing is to avoid bank conflicts
        if (threadIdx.x < stride) { // we can't do an else because all threads of the block need to participate in the block-wide barrier every iteration
            shared[threadIdx.x] += shared[threadIdx.x+stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        rms[token] = sqrtf(shared[0] / hidden_size + eps);
    }

}

__global__ void normalize_kernel_no_intermediate(
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

torch::Tensor rmsnorm_no_intermediate(
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

    auto rms = torch::empty(
        {tokens},
        x.options()
    );
    auto output = torch::empty_like(x);

    int threads = 256;

    int element_blocks = (n + threads - 1) / threads;

    rms_kernel_no_intermediate<<<tokens, threads>>>(
        x.data_ptr<float>(),
        rms.data_ptr<float>(),
        hidden_size,
        eps
    );

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );

    normalize_kernel_no_intermediate<<<element_blocks, threads>>>(
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