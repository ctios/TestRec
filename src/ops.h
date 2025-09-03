#pragma once

#include <torch/torch.h>

// CPU实现
torch::Tensor square_plus_cpu(const torch::Tensor& input);
torch::Tensor modulo_cpu(const torch::Tensor& input, const int64_t mod);

// CUDA实现声明
#ifdef WITH_CUDA
torch::Tensor square_plus_cuda(const torch::Tensor& input);
torch::Tensor modulo_cuda(const torch::Tensor& input, const int64_t mod);
#endif

// 设备无关的包装函数
torch::Tensor square_plus(const torch::Tensor& input);
torch::Tensor modulo(const torch::Tensor& input, const int64_t mod);