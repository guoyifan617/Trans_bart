#encoding: utf-8

from torch.autograd import Function

try:
	import spreader_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	spreader_cpp = load(name="spreader_cpp", sources=["modules/cpp/spreader/spreader.cpp"])

class SpreaderFunction(Function):

	@staticmethod
	def forward(ctx, fgate, igh, init_cell, dim=None, inplace=False):

		cell = spreader_cpp.forward(fgate, igh, init_cell, dim, inplace)
		ctx.save_for_backward(cell, fgate, init_cell)
		ctx.dim = dim

		return cell

	@staticmethod
	def backward(ctx, grad_cell):

		needs_grad_fgate, needs_grad_igh, needs_grad_init_cell = ctx.needs_input_grad[0:3]
		if needs_grad_fgate or needs_grad_igh or needs_grad_init_cell:
			cell, fgate, init_cell = ctx.saved_variables
			if needs_grad_fgate:
				grad_fgate, grad_igh, grad_init_cell = spreader_cpp.backward(grad_cell, cell, fgate, init_cell, ctx.dim)
				return grad_fgate, grad_igh if needs_grad_igh else None, grad_init_cell if needs_grad_init_cell else None, None, None
			else:
				grad_igh, grad_init_cell = spreader_cpp.backward_no_fgate(grad_cell, fgate, ctx.dim)
				return None, grad_igh if needs_grad_igh else None, grad_init_cell if needs_grad_init_cell else None, None, None
		else:
			return None, None, None, None, None

SpreaderFunc = SpreaderFunction.apply
