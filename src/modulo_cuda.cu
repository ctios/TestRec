#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ops.h"

__global__ void modulo_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int64_t n,
    const int64_t mod) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t val = static_cast<int64_t>(input[idx]);
        output[idx] = val % mod;
    }
}

torch::Tensor modulo_cuda(const torch::Tensor& input, const int64_t mod) {
    auto x = input.contiguous();
    auto output = torch::empty_like(x, torch::kInt64);
    
    const int64_t n = x.numel();
    const int64_t threads = 1024;
    const int64_t blocks = (n + threads - 1) / threads;
    
    modulo_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        n,
        mod
    );
    
    return output;
}