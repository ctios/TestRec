#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ops.h"

__global__ void square_plus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val * val + val;
    }
}

torch::Tensor square_plus_cuda(const torch::Tensor& input) {
    auto x = input.contiguous();
    auto output = torch::empty_like(x);
    
    const int64_t n = x.numel();
    const int64_t threads = 1024;
    const int64_t blocks = (n + threads - 1) / threads;
    
    square_plus_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}