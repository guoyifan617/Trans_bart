#encoding: utf-8

import torch
from torch import nn

from modules.base import Custom_Act, Linear, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase

from cnfg.ihyp import *

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, *inputs, custom_act=use_adv_act_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, *inputs, **kwargs)

		self.trans_gf = Linear(isize + isize, isize + isize)
		self.trans_gf_ln = nn.LayerNorm((2, isize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.ff_act = nn.Sequential(Custom_Act() if custom_act else nn.ReLU(inplace=False), Linear(isize, isize, bias=kwargs.get("enable_proj_bias", enable_prev_ln_bias_default)))

	def forward(self, iQ, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]
			_osize = list(_out.size())
			_osize.insert(-1, 2)
			gate, ffn = self.trans_gf_ln(self.trans_gf(torch.cat((_iQ, _out,), dim=-1)).view(_osize)).unbind(-2)
			_out = self.ff_act(ffn).addcmul(_out, gate.sigmoid())

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			_osize = list(outs.size())
			_osize.insert(-1, 2)
			gate, ffn = self.trans_gf_ln(self.trans_gf(torch.cat((_iQ, outs,), dim=-1)).view(_osize)).unbind(-2)
			outs = self.ff_act(ffn).addcmul(outs, gate.sigmoid())

			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, *inputs, custom_act=use_adv_act_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, *inputs, **kwargs)

		self.trans_gf = Linear(isize + isize, isize + isize)
		self.trans_gf_ln = nn.LayerNorm((2, isize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.ff_act = nn.Sequential(Custom_Act() if custom_act else nn.ReLU(inplace=False), Linear(isize, isize, bias=kwargs.get("enable_proj_bias", enable_prev_ln_bias_default)))

	def forward(self, iQ, iK, *inputs, **kwargs):

		_iQ = self.normer(iQ)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			_osize = list(outs.size())
			_osize.insert(-1, 2)
			gate, ffn = self.trans_gf_ln(self.trans_gf(torch.cat((_iQ, outs,), dim=-1)).view(_osize)).unbind(-2)
			outs = self.ff_act(ffn).addcmul(outs, gate.sigmoid())

			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)
