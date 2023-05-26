#encoding: utf-8

import torch
from torch.autograd import Function

from modules.base import PositionwiseFF as PositionwiseFFBase

from cnfg.ihyp import *

class FixRegResidue(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	def forward(ctx, res_in, mod_out, reg_w=0.0):

		ctx.save_for_backward(mod_out)
		ctx.reg_w = reg_w

		_bsize = res_in.size(0)
		_real_out = mod_out.narrow(0, 0, _bsize)

		return _real_out + res_in

	@staticmethod
	def backward(ctx, grad_output):

		_bsize = grad_output.size(0)

		mod_out = ctx.saved_tensors[0]
		_reg_w = ctx.reg_w
		_mod_out = mod_out.narrow(0, 0, _bsize)

		_mo_norm, _g_norm = _mod_out.norm(dim=-1, keepdim=True) + 1e-32, grad_output.norm(dim=-1, keepdim=True) + 1e-32

		return grad_output if ctx.needs_input_grad[0] else None, torch.cat((grad_output, _reg_w * (grad_output / (_mo_norm * _g_norm) - ((_mod_out * grad_output).sum(dim=-1, keepdim=True) * _mod_out) / (_mo_norm.pow(3) * _g_norm)),), 0) if ctx.needs_input_grad[1] else None, None

class PositionwiseFF(PositionwiseFFBase):

	# isize: input dimension
	# hsize: hidden dimension

	def __init__(self, *args, **kwargs):

		super(PositionwiseFF, self).__init__(*args, **kwargs)

		self.reg_w = 0.0

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		_r_train = self.training and self.reg_w > 0.0

		out = torch.cat((_out, _out.clone().detach(),), 0) if _r_train else _out

		out = self.net(out)

		_residue_add = _out if self.norm_residual else x

		out = FixRegResidue.apply(_residue_add, out, self.reg_w) if _r_train else (_residue_add + out)

		return out

	def set_reg_w(self, w):

		self.reg_w = w
