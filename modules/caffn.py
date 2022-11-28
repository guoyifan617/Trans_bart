#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, NDWrapper
from modules.group.base import GroupLinear
from utils.base import float2odd

from cnfg.ihyp import *

class BiCFFN(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, num_head=None, kernel_size=5, head_dim=64, custom_act=use_adv_act_default, enable_bias=enable_proj_bias_default, **kwargs):

		super(BiCFFN, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize
		self.num_head = max(2, float2odd(float(isize) / float(head_dim))) if num_head is None else num_head
		self.attn_dim = isize // self.num_head
		ahsize = self.attn_dim * self.num_head

		self.pool = NDWrapper(nn.AvgPool1d(kernel_size, stride=1, padding=0, ceil_mode=False, count_include_pad=True), 3)
		self.ctx_normer = nn.LayerNorm((self.num_head, self.attn_dim,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.ks = kernel_size#, self.k, float(kernel_size)

		self.transi = Linear(isize, ahsize, bias=enable_bias)
		self.transo = Linear(ahsize, isize, bias=enable_bias)

		self.net = nn.Sequential(GroupLinear(ahsize * 3, _hsize, self.num_head, bias=True, trans_input=False, shuffle=False, flatten_output=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), GroupLinear(_hsize, ahsize, self.num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=True), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(GroupLinear(ahsize * 3, _hsize, self.num_head, bias=True, trans_input=False, shuffle=False, flatten_output=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_hsize, ahsize, self.num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=True))

		self.normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.norm_residual = norm_residual

	# mask: (bsize, seql)
	def forward(self, x, mask=None, **kwargs):

		bsize, seql, isize = x.size()
		nheads = self.num_head
		adim = self.attn_dim
		kernel_size = self.ks

		_out = self.normer(x)
		out = self.transi(_out).view(bsize, seql, nheads, adim)
		if mask is not None:
			out.masked_fill_(mask.view(bsize, seql, 1, 1), 0.0)

		_pad = x.new_zeros(bsize, adim, nheads, kernel_size, dtype=out.dtype, device=out.device, requires_grad=False)

		bow_out = self.ctx_normer(self.pool(torch.cat((_pad, out.transpose(1, 3), _pad,), dim=-1)).transpose(1, 3))#.mul_(self.k)
		lbow_out, rbow_out = bow_out.narrow(1, 0, seql), bow_out.narrow(1, kernel_size + 1, seql)

		out = self.transo(self.net(torch.cat((lbow_out, out, rbow_out,), dim=-1)))

		out = out + (_out if self.norm_residual else x)

		return out

class UniCFFN(BiCFFN):

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, num_head=None, kernel_size=5, head_dim=64, custom_act=use_adv_act_default, enable_bias=enable_proj_bias_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		num_head = max(2, float2odd(float(isize) / float(head_dim))) if num_head is None else num_head

		super(UniCFFN, self).__init__(isize, hsize=_hsize, dropout=dropout, norm_residual=norm_residual, num_head=num_head, kernel_size=kernel_size, head_dim=head_dim, custom_act=custom_act, enable_bias=enable_bias)

		ahsize = self.attn_dim * self.num_head
		self.net = nn.Sequential(GroupLinear(ahsize + ahsize, _hsize, self.num_head, bias=True, trans_input=False, shuffle=False, flatten_output=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), GroupLinear(_hsize, ahsize, self.num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=True), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(GroupLinear(ahsize + ahsize, _hsize, self.num_head, bias=True, trans_input=False, shuffle=False, flatten_output=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_hsize, ahsize, self.num_head, bias=enable_bias, trans_input=False, shuffle=False, flatten_output=True))

	def forward(self, x, mask=None, states=None, **kwargs):

		bsize, seql, isize = x.size()
		nheads = self.num_head
		adim = self.attn_dim
		kernel_size = self.ks

		_out = self.normer(x)
		out = self.transi(_out).view(bsize, seql, nheads, adim)
		if mask is not None:
			out.masked_fill_(mask.view(bsize, seql, 1, 1), 0.0)

		if states is None:
			_pad = x.new_zeros(bsize, adim, nheads, kernel_size, dtype=out.dtype, device=out.device, requires_grad=False)
			bow_out = self.ctx_normer(self.pool(torch.cat((_pad, out.narrow(1, 0, seql - 1).transpose(1, 3),), dim=-1)).transpose(1, 3)).narrow(1, 0, seql)#.mul_(self.k)
		else:
			# fine with step-by-step decoding, but do not support sequence force decoding
			if states == "init":
				bow_out = self.ctx_normer(x.new_zeros(nheads, adim, dtype=out.dtype, device=out.device, requires_grad=False)).view(1, 1, nheads, adim).expand(bsize, seql, nheads, adim)
				states_return = torch.cat((x.new_zeros(bsize, adim, nheads, kernel_size - seql, dtype=out.dtype, device=out.device, requires_grad=False), out.transpose(1, 3),), dim=-1)
			else:
				bow_out = self.ctx_normer(self.pool(states).transpose(1, 3)).narrow(1, 0, seql)
				states_return = torch.cat((states.narrow(-1, seql, kernel_size - seql), out.transpose(1, 3),), dim=-1)

		out = self.transo(self.net(torch.cat((bow_out, out,), dim=-1)))

		out = out + (_out if self.norm_residual else x)

		if states is None:
			return out
		else:
			return out, states_return
