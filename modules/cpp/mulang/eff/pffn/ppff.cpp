#include <torch/extension.h>
#include "ppff_func.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &parallel_positionwise_ff_forward, "Positionwise FF forward");
}
