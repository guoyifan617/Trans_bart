#include <torch/extension.h>
#include <vector>

// port from modules/cpp/hplstm/lgate.cpp, the only difference is that here fgate is not a sequence.

at::Tensor spreader_forward(torch::Tensor fgate, torch::Tensor igh, torch::Tensor init_cell, int64_t dim, bool inplace=false) {

	torch::Tensor cell;
	if (inplace) {
		cell = igh;
	}
	else {
		cell = igh.clone();
	}
	auto seqlen = cell.size(dim);
	cell.select(dim, 0).addcmul_(init_cell, fgate);
	int64_t i;
	for (i = 1; i < seqlen; i++) {
		cell.select(dim, i).addcmul_(cell.select(dim, i - 1), fgate);
	}

	return cell;
}

std::vector<torch::Tensor> spreader_backward(torch::Tensor grad_cell, torch::Tensor cell, torch::Tensor fgate, torch::Tensor init_cell, int64_t dim) {

	torch::Tensor grad_fgate;
	auto grad_igh = grad_cell.clone();
	auto last_index = grad_cell.size(dim) - 1;
	auto acc_grad_cell = grad_cell.select(dim, last_index);
	auto grad_prev_cell = acc_grad_cell * fgate;
	if (last_index > 0) {
		grad_fgate = acc_grad_cell * cell.select(dim, last_index - 1);
		int64_t i;
		for (i = last_index - 1; i > 0; i--) {
			acc_grad_cell = grad_igh.select(dim, i).add_(grad_prev_cell);
			grad_prev_cell = acc_grad_cell * fgate;
			grad_fgate.addcmul_(acc_grad_cell, cell.select(dim, i - 1));
		}
		acc_grad_cell = grad_igh.select(dim, 0).add_(grad_prev_cell);
		grad_prev_cell = acc_grad_cell * fgate;
		grad_fgate.addcmul_(acc_grad_cell, init_cell);
	}
	else {
		grad_fgate = acc_grad_cell * init_cell;
	}

	return {grad_fgate, grad_igh, grad_prev_cell};
}

std::vector<torch::Tensor> spreader_backward_no_fgate(torch::Tensor grad_cell, torch::Tensor fgate, int64_t dim) {

	auto grad_igh = grad_cell.clone();
	auto last_index = grad_cell.size(dim) - 1;
	auto grad_prev_cell = grad_cell.select(dim, last_index) * fgate;
	int64_t i;
	for (i = last_index - 1; i >= 0; i--) {
		grad_prev_cell = grad_igh.select(dim, i).add_(grad_prev_cell) * fgate;
	}

	return {grad_igh, grad_prev_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &spreader_forward, "Spreader forward");
	m.def("backward", &spreader_backward, "Spreader backward");
	m.def("backward_no_fgate", &spreader_backward_no_fgate, "Spreader backward (no fgate)");
}
