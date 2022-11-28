#encoding: utf-8

from torch.autograd import Function

try:
	import lgate_nocx_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	lgate_nocx_cpp = load(name="lgate_nocx_cpp", sources=["modules/cpp/hplstm/lgate_nocx.cpp"])

class LGateNocxFunction(Function):

	@staticmethod
	def forward(ctx, fgate, igh, dim=None, inplace=False):

		cell = lgate_nocx_cpp.forward(fgate, igh, dim, inplace)
		ctx.save_for_backward(cell, fgate)
		ctx.dim = dim

		return cell

	@staticmethod
	def backward(ctx, grad_cell):

		needs_grad_fgate, needs_grad_igh = ctx.needs_input_grad[0:2]
		if needs_grad_fgate or needs_grad_igh:
			cell, fgate = ctx.saved_variables
			if needs_grad_fgate:
				grad_fgate, grad_igh = lgate_nocx_cpp.backward(grad_cell, cell, fgate, ctx.dim)
				return grad_fgate if needs_grad_fgate else None, grad_igh if needs_grad_igh else None, None, None
			else:
				grad_igh = lgate_nocx_cpp.backward_no_fgate(grad_cell, fgate, ctx.dim)
				return None, grad_igh if needs_grad_igh else None, None, None
		else:
			return None, None, None, None

LGateNocxFunc = LGateNocxFunction.apply
