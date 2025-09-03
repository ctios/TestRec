#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_plus", &square_plus, "Computes x^2 + x");
    m.def("modulo", &modulo, "Computes x % mod", pybind11::arg("input"), pybind11::arg("mod"));
}