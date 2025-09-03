#include "ops.h"

torch::Tensor modulo_cpu(const torch::Tensor& input, const int64_t mod) {
    auto x = input.contiguous();
    auto output = torch::empty_like(x, torch::kInt64);
    
    auto input_accessor = x.accessor<float, 1>();
    auto output_accessor = output.accessor<int64_t, 1>();
    
    for (int64_t i = 0; i < input.numel(); i++) {
        int64_t val = static_cast<int64_t>(input_accessor[i]);
        output_accessor[i] = val % mod;
    }
    
    return output;
}

#ifdef WITH_CUDA
torch::Tensor modulo_cuda(const torch::Tensor& input, const int64_t mod);
#endif

torch::Tensor modulo(const torch::Tensor& input, const int64_t mod) {
    if (input.is_cuda()) {
#ifdef WITH_CUDA
        return modulo_cuda(input, mod);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        return modulo_cpu(input, mod);
    }
}