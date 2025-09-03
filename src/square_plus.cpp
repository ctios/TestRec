#include "ops.h"

torch::Tensor square_plus_cpu(const torch::Tensor& input) {
    auto x = input.contiguous();
    auto output = torch::empty_like(x);
    
    auto input_accessor = x.accessor<float, 1>();
    auto output_accessor = output.accessor<float, 1>();
    
    for (int64_t i = 0; i < input.numel(); i++) {
        float val = input_accessor[i];
        output_accessor[i] = val * val + val;
    }
    
    return output;
}

#ifdef WITH_CUDA
torch::Tensor square_plus_cuda(const torch::Tensor& input);
#endif

torch::Tensor square_plus(const torch::Tensor& input) {
    if (input.is_cuda()) {
#ifdef WITH_CUDA
        return square_plus_cuda(input);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        return square_plus_cpu(input);
    }
}