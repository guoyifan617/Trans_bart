#ifndef _NEUTRON_MODULES_CPP_MULANG_EFF_PFFN_FUNC
#define _NEUTRON_MODULES_CPP_MULANG_EFF_PFFN_FUNC

#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>

std::vector<at::Tensor> parallel_positionwise_ff_forward(std::vector<std::map<std::string, torch::Tensor>> tensors, std::map<std::string, bool> bargs, void* act, std::map<std::string, double> dargs, std::vector<int64_t> normalized_shape);

#endif
