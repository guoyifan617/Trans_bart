#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class ResRNN(nn.Module):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(ResRNN, self).__init__()

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize

		self.net = nn.Sequential(Linear(isize + _osize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize, bias=enable_bias)) if dropout > 0.0 else nn.Sequential(Linear(isize + _osize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))

		self.init_hx = nn.Parameter(torch.zeros(1, _osize))

	# x: (bsize, seql, isize)
	def forward(self, x, states=None, first_step=False, **kwargs):

		if states is None:
			_hx = self.init_hx.expand(x.size(0), -1)
			rs = []
			for xu in x.unbind(1):
				_hx = self.net(torch.cat((xu, _hx,), dim=-1)) + _hx
				rs.append(_hx)

			rs = torch.stack(rs, dim=1)

			return rs
		else:
			_hx = self.init_hx.expand(x.size(0), -1) if first_step else states
			rs = self.net(torch.cat((x, _hx,), dim=-1)) + _hx
			return rs

	def fix_init(self):

		with torch_no_grad():
			self.init_hx.zero_()

class RNN(ResRNN):

	# x: (bsize, seql, isize)
	def forward(self, x, states=None, first_step=False, **kwargs):

		if states is None:
			_hx = self.init_hx.expand(x.size(0), -1)
			rs = []
			for xu in x.unbind(1):
				_hx = self.net(torch.cat((xu, _hx,), dim=-1))
				rs.append(_hx)

			rs = torch.stack(rs, dim=1)

			return rs
		else:
			_hx = self.init_hx.expand(x.size(0), -1) if first_step else states
			rs = self.net(torch.cat((x, _hx,), dim=-1))
			return rs
