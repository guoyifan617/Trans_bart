#include <torch/extension.h>

at::Tensor movavg_forward(torch::Tensor x, int64_t dim, float beta=0.9, bool inplace=false) {

	torch::Tensor out;
	if (inplace) {
		out = x;
	}
	else {
		out = x.clone();
	}
	float beta1 = beta;
	float beta2 = 1.0 - beta1;
	auto seqlen = x.size(dim);
	int64_t i;
	auto prev_step = out.select(dim, 0).mul_(beta2);
	for (i = 1; i < seqlen; i++) {
		prev_step = out.select(dim, i).mul_(beta2).add_(prev_step, beta1);
	}

	return out;
}

torch::Tensor movavg_backward(torch::Tensor grad_out, int64_t dim, float beta=0.9) {

	float beta1 = beta;
	float beta2 = 1.0 - beta1;
	auto grad_input = grad_out.clone();
	auto last_index = grad_out.size(dim) - 1;
	grad_input.select(dim, last_index).mul_(beta2);
	if (last_index > 0) {
		int64_t i;
		auto grad_prev_out = grad_out.select(dim, last_index) * beta1;
		for (i = last_index - 1; i > 0; i--) {
			auto grad_step = grad_input.select(dim, i).add_(grad_prev_out);// grad_input is initialized as a copy of grad_out, performing the accumulation directly on grad_input is more efficient.
			grad_prev_out = grad_step * beta1;
			grad_step.mul_(beta2);
		}
		grad_input.select(dim, 0).add_(grad_prev_out).mul_(beta2);
	}

	return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &movavg_forward, "MovAvg forward");
	m.def("backward", &movavg_backward, "MovAvg backward");
}
