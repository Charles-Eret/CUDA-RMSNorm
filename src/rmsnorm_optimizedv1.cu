#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_optimizedv1(
    const float* x,
    int hidden_size,
    float eps,
    const float* gamma,
    float* output
) {
    int token = blockIdx.x;

    __shared__ float shared[256];
    float local_sum = 0.0f;
    float values[16];

    #pragma unroll
    for (int k = 0; k < 16; k++) { // default to max hidden dim of 4096 with this iteration scheme
        int h = threadIdx.x + k * blockDim.x;

        if (h < hidden_size) {
            int index = token * hidden_size + h;

            values[k] = x[index]; // store x in reg for later reuse
            local_sum += values[k] * values[k];
        }
    }
    shared[threadIdx.x] = local_sum;

    __syncthreads(); //sync all warps from the same block before continuing

    // reduction loop
    for (int stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) { // we can't do an else because all threads of the block need to participate in the block-wide barrier every iteration
            shared[threadIdx.x] += shared[threadIdx.x+stride]; // indexing is to avoid bank conflicts
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared[0] = rsqrtf(shared[0] / hidden_size + eps);
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < 16; k++) {
        int h = threadIdx.x + k * blockDim.x;

        if (h < hidden_size) {
            int index = token * hidden_size + h;

            output[index] = (values[k] * shared[0]) * gamma[h];
        }
    }

}

torch::Tensor rmsnorm_optimizedv1(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
) {
    TORCH_CHECK(
        x.size(1) <= 4096,
        "rmsnorm_optimizedv1 currently supports hidden_size <= 4096"
    );

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

    TORCH_CHECK(
        x.device() == gamma.device(),
        "x and gamma must be on the same device"
    );

    x = x.contiguous();
    gamma = gamma.contiguous();

    int tokens = x.size(0);
    int hidden_size = x.size(1);

    TORCH_CHECK(
        hidden_size > 0 && hidden_size <= 4096,
        "rmsnorm_optimizedv1 requires hidden size between 1 and 4096"
    );

    TORCH_CHECK(
        gamma.size(0) == hidden_size,
        "gamma size must match hidden size"
    );

    auto output = torch::empty_like(x);

    int threads = 256;

    kernel_optimizedv1<<<tokens, threads>>>(
        x.data_ptr<float>(),
        hidden_size,
        eps,
        gamma.data_ptr<float>(),
        output.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );

    return output;
}
