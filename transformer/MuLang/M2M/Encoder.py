#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout
from modules.elinear import MWLinear
from modules.mulang.base import LayerNorm
from modules.mulang.m2m import PositionwiseFF, SelfAttn
from modules.mulang.o2m import SelfAttn as o2mSelfAttn
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ngroup=None, ntask=None, expand_layer=False, k_rel_pos=use_k_relative_position_encoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, k_rel_pos=k_rel_pos, **kwargs)

		self.ngroup, self.expand_layer = ngroup, expand_layer

		self.layer_normer = LayerNorm(isize if expand_layer else (ngroup, isize,), ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		Attn = o2mSelfAttn if expand_layer else SelfAttn
		self.attn = Attn(isize, _ahsize, isize, ngroup, num_head=num_head, dropout=attn_drop, k_rel_pos=k_rel_pos)
		self.ff = PositionwiseFF(isize, ngroup, hsize=_fhsize, dropout=dropout, act_drop=act_drop, ntask=ntask)

	def forward(self, inputs, attn_w=None, ffn_w=None, taskid=None, mask=None, **kwargs):

		_inputs = self.layer_normer(inputs, taskid=taskid)
		context = self.attn(_inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		_res_add = _inputs if self.norm_residual else inputs
		if self.expand_layer:
			_res_add = _res_add.unsqueeze(-2)
		context = context + _res_add

		if attn_w is not None:
			_osize = list(context.size())
			_osize[-2], _osize[-1] = _osize[-1], _osize[-2]
			context = context.transpose(-1, -2).contiguous().view(context.size(0), -1, self.ngroup).bmm(attn_w).view(_osize).transpose(-1, -2).contiguous()

		context = self.ff(context, weight=ffn_w, taskid=taskid)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, ntask=None, ngroup=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.task_emb = nn.Embedding(ntask, isize, padding_idx=None)
		self.group_weight = nn.Parameter(torch.zeros(ntask, num_layer - 1, 2, ngroup, ngroup))
		self.group_weight_flayer = nn.Parameter(torch.zeros(ntask, ngroup, ngroup))
		self.gw_drop = Dropout(dropout) if dropout > 0.0 else None
		self.transo = MWLinear(isize, isize, ntask, bias=enable_proj_bias_default)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=False)
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=True)] + [_shared_layer for i in range(num_layer - 1)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=(i == 0)) for i in range(num_layer)])

		if norm_output:
			self.out_normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, inputs, taskid=None, mask=None, **kwargs):

		out = self.wemb(inputs) + self.task_emb(taskid).unsqueeze(1)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		_gwf, _gw = self.group_weight_flayer.index_select(0, taskid).softmax(-2), self.group_weight.index_select(0, taskid).softmax(-2)

		if self.drop is not None:
			out = self.drop(out)
			_gwf = self.gw_drop(_gwf)
			_gw = self.gw_drop(_gw)

		_w = [(None, _gwf,)] + [_wu.unbind(1) for _wu in _gw.unbind(1)]
		for net, (_w_attn, _w_ffn,) in zip(self.nets, _w):
			out = net(out, attn_w=_w_attn, ffn_w=_w_ffn, taskid=taskid, mask=mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		return self.transo(out, taskid)

	def load_base(self, base_encoder):

		super(Encoder, self).load_base(base_encoder)

		if hasattr(base_encoder, "task_emb"):
			self.task_emb = base_encoder.task_emb
		if hasattr(base_encoder, "group_weight"):
			self.group_weight = base_encoder.group_weight
		if hasattr(base_encoder, "group_weight_flayer"):
			self.group_weight_flayer = base_encoder.group_weight_flayer

	def fix_init(self):

		super(Encoder, self).fix_init()

		with torch_no_grad():
			self.group_weight.zero_()
			self.group_weight_flayer.zero_()
