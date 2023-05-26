#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Custom_Act, Dropout, Linear
from modules.group.base import GroupLinear
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, num_head=8, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(EncoderLayer, self).__init__()

		_fhsize = isize * 4 if fhsize is None else fhsize

		self.num_head, self.igsize = num_head, isize // num_head
		_proj_size = self.igsize * num_head
		_fgsize = _fhsize // num_head
		_fhsize = _fgsize * num_head

		self.adaptor_csum = Linear(isize, _proj_size * 4, bias=enable_proj_bias)
		self.adaptor_x = Linear(isize, _proj_size, bias=enable_proj_bias)
		self.act = Custom_Act() if custom_act else nn.ReLU(inplace=True)
		self.trans_g = GroupLinear(_proj_size * 3, _proj_size * 3, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False)
		self.ffn = nn.Sequential(GroupLinear(_proj_size * 3, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _fgsize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), GroupLinear(_fhsize, _proj_size, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(GroupLinear(_proj_size * 3, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _fgsize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_fhsize, _proj_size, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False))
		self.trans_o = Linear(_proj_size, isize, bias=enable_proj_bias)

		self.layer_normer_l, self.layer_normer_r = nn.LayerNorm((2, num_head, self.igsize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), nn.LayerNorm((2, num_head, self.igsize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.layer_normer_g = nn.LayerNorm((num_head, 3, self.igsize,), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	# mask: (bsize, seql, 1)
	def forward(self, inputs, mask=None, **kwargs):

		bsize, seql = inputs.size()[:2]
		out_csum, out_x = self.adaptor_csum(inputs), self.adaptor_x(inputs).view(bsize, seql, self.num_head, self.igsize)

		if mask is not None:
			out_csum.masked_fill_(mask, 0.0)

		lc, rc = out_csum.view(bsize, seql, 2, 2, self.num_head, self.igsize).unbind(2)
		# apply act before cumsum?
		(lcg, lci,), (rcg, rci,) = self.layer_normer_l(lc.cumsum(dim=1)).unbind(2), self.layer_normer_r(rc.flip(1).cumsum(dim=1).flip(1)).unbind(2)
		lci, rci = self.act(lci), self.act(rci)

		# bsize, seql, num_head, 3 * igsize
		cell_i = torch.cat((lcg, out_x, rcg,), dim=-1)
		(lcg, ig, rcg,), x = self.layer_normer_g(self.trans_g(cell_i).view(bsize, seql, self.num_head, 3, self.igsize)).sigmoid().unbind(-2), self.ffn(cell_i)

		out = lci * lcg + ig * x + rci * rcg

		return self.trans_o(out.view(bsize, seql, -1)) + inputs

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, disable_pemb=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, disable_pemb=True, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, num_head=num_head)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, num_head=num_head) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		_mask = mask.squeeze(1).unsqueeze(1)
		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
