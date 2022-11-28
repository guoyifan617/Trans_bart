#encoding: utf-8

import torch
from math import sqrt#, log
from torch import nn
from torch.autograd import Function

from utils.torch.comp import mask_tensor_type, torch_no_grad

from cnfg.ihyp import ieps_default

# almost the same with modules.prune.bern
class DropFunction(Function):

	# output = inputs * bernoulli(1.0 - maskp)
	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, inputs, maskp, inplace=False, non_adversarial=True):

		"""_maskp_check = maskp.gt(0.99)# 1 - eps
		if _maskp_check.int().sum().item() > 0:
			_maskp = maskp.masked_fill_(_maskp_check, 0.99)# 1 - eps
		else:
			_maskp = maskp"""
		mask = maskp.bernoulli().to(mask_tensor_type, non_blocking=True)
		ctx.save_for_backward(inputs, mask)
		ctx.nadv = non_adversarial
		return inputs.masked_fill(mask, 0.0)# inputs.masked_fill_(mask, 0.0) if inplace else

	@staticmethod
	def backward(ctx, grad_outputs):

		if grad_outputs is None:
			return None, None, None, None
		else:
			inputs, mask = ctx.saved_tensors
			_grad_input = grad_outputs.masked_fill(mask, 0.0) if ctx.needs_input_grad[0] else None
			if ctx.needs_input_grad[1]:
				_grad_maskp = grad_outputs * inputs
				if ctx.nadv:
					_grad_maskp = -_grad_maskp
			else:
				_grad_maskp = None
			return _grad_input, _grad_maskp, None, None

class AdvDropout(nn.Module):

	# ignore inplace
	def __init__(self, p, isize, dim, k=sqrt(2.0), inplace=False, eps=ieps_default, **kwargs):

		super(AdvDropout, self).__init__()
		self.p, self.isize, self.dim, self.k, self.eps = float(p), isize, dim, k, eps#, self.inplace
		self.keep_magnitude = (1.0 / (1.0 - self.p))
		self.weight = nn.Parameter(torch.zeros(isize))

	def forward(self, inpute, **kwargs):

		if self.training:
			_isize = [1 for i in range(inpute.dim())]
			_isize[self.dim] = self.isize
			_dw = self.weight.detach()
			_dp = ((self.weight - _dw.mean())/(_dw.std() + self.eps) * self.k).sigmoid()# + 1e-2 eps
			_dp = (_dp * (self.p / _dp.detach().mean().item())).view(_isize).expand_as(inpute)
			out = DropFunction.apply(inpute, _dp, False, False)
			if self.keep_magnitude:
				out = out * self.keep_magnitude

			return out
		else:
			return inpute

	def fix_init(self):

		with torch_no_grad():
			#_init_value = -log(1.0 / self.p - 1.0)
			self.weight.data.zero_()
