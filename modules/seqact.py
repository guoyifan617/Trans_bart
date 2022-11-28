#encoding: utf-8

from torch import nn

from modules.base import Dropout, Linear

from cnfg.ihyp import *

class PositionwiseFF(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, bidirectional=True, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(PositionwiseFF, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.trans1, self.trans2 = Linear(isize, _hsize), Linear(_hsize // 2, isize, bias=enable_bias)

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.cumsum_normer = nn.LayerNorm(_hsize // 2, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual, self.bidirectional = norm_residual, bidirectional

	def forward(self, x, mask=None, **kwargs):

		_out = self.normer(x)

		out = self.trans1(_out)
		if self.bidirectional:
			if mask is not None:
				out.masked_fill_(mask, 0.0)
			_cg, _o = out.chunk(2, dim=-1)
			_norm_cumsum, _norm_rcumsum = self.cumsum_normer(_cg.cumsum(dim=1)), self.cumsum_normer(_cg.flip(1).cumsum(dim=1).flip(1))
			_norm_cumsum_gate, _norm_rcumsum_gate = _norm_cumsum.sigmoid(), _norm_rcumsum.sigmoid()
			out = _o * (_norm_cumsum_gate + _norm_rcumsum_gate)# + (1.0 - _norm_cumsum_gate) * _norm_cumsum + (1.0 - _norm_rcumsum_gate) * _norm_rcumsum
		else:
			_cg, _o = out.chunk(2, dim=-1)
			_norm_cumsum = self.cumsum_normer(_cg.cumsum(dim=1))
			_norm_cumsum_gate = _norm_cumsum.sigmoid()
			out = _o * _norm_cumsum_gate# + (1.0 - _norm_cumsum_gate) * _norm_cumsum

		if self.drop is not None:
			out = self.drop(out)

		out = self.trans2(out)

		if self.drop is not None:
			out = self.drop(out)

		out = out + (_out if self.norm_residual else x)

		return out
