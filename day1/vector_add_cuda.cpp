#include <torch/extension.h>
torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("add_cuda", torch::wrap_pybind_function(add_cuda), "add_cuda");
}