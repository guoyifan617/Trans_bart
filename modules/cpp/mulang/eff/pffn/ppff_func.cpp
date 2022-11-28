#include <omp.h>
#include <torch/torch.h>
#include <vector>
#include <map>
#include <string>
#include "../../../base/ffn/pff_func.h"

std::vector<at::Tensor> parallel_positionwise_ff_forward(std::vector<std::map<std::string, torch::Tensor>> tensors, std::map<std::string, bool> bargs, void* act, std::map<std::string, double> dargs, std::vector<int64_t> normalized_shape) {

	int num_inputs = tensors.size();
	std::map<int, at::Tensor> p_rs;

	#pragma omp parallel for
	for(int i = 0; i < num_inputs; i++) {
		p_rs[i] = positionwise_ff_forward(tensors[i], bargs, act, dargs, normalized_shape);
	}

	std::vector<at::Tensor> rs;
	for(int i = 0; i < num_inputs; i++) {
		rs.push_back(p_rs[i]);
	}

	return rs;

}
