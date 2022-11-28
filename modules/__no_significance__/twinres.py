#encoding: utf-8

from torch import nn

from modules.base import PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase

from cnfg.ihyp import *

def get_eff_gate(isize):

	return nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), nn.Sigmoid())

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, *args, **kwargs):

		super(PositionwiseFF, self).__init__(isize, *args, **kwargs)

		self.twine_i = get_eff_gate(isize)
		self.twine_o = get_eff_gate(isize)

	def forward(self, x, **kwargs):

		_out = self.normer(x)

		out = self.net(_out)

		resin = _out if self.norm_residual else x
		out = (self.twine_o(out) * resin).addcmul_(self.twine_i(resin), out)

		return out

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, *args, **kwargs):

		super(ResSelfAttn, self).__init__(isize, *args, **kwargs)

		self.twine_i = get_eff_gate(isize)
		self.twine_o = get_eff_gate(isize)

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		_multi_out = isinstance(outs, tuple)
		out = outs[0] if _multi_out else outs

		if self.drop is not None:
			out = self.drop(out)

		resin = _iQ if self.norm_residual else iQ
		out = (self.twine_o(out) * resin).addcmul_(self.twine_i(resin), out)

		if _multi_out:
			return out, *outs[1:]
		else:
			return out

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, *args, **kwargs):

		super(ResCrossAttn, self).__init__(isize, *args, **kwargs)

		self.twine_i = get_eff_gate(isize)
		self.twine_o = get_eff_gate(isize)

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		_multi_out = isinstance(outs, tuple)
		out = outs[0] if _multi_out else outs

		if self.drop is not None:
			out = self.drop(out)

		resin = _iQ if self.norm_residual else iQ
		out = (self.twine_o(out) * resin).addcmul_(self.twine_i(resin), out)

		if _multi_out:
			return out, *outs[1:]
		else:
			return out
