#encoding: utf-8

from math import sqrt
from torch import nn

from modules.mulang.eff.base import LayerNorm, MWLinear
from modules.mulang.eff.m2m import PositionwiseFF, SelfAttn
from modules.mulang.eff.o2m import SelfAttn as o2mSelfAttn
from transformer.MuLang.M2M.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ngroup=None, ntask=None, expand_layer=False, k_rel_pos=use_k_relative_position_encoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, ngroup=ngroup, ntask=ntask, expand_layer=expand_layer, k_rel_pos=k_rel_pos, **kwargs)

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
			context = context.transpose(-1, -2).contiguous().view(-1, self.ngroup).mm(attn_w).view(_osize).transpose(-1, -2).contiguous()

		context = self.ff(context, weight=ffn_w, taskid=taskid)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, ntask=None, ngroup=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, ntask=ntask, ngroup=ngroup, share_layer=share_layer, **kwargs)

		self.transo = MWLinear(isize, isize, ntask, bias=enable_proj_bias_default)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=False)
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=True)] + [_shared_layer for i in range(num_layer - 1)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ngroup=ngroup, ntask=ntask, expand_layer=(i == 0)) for i in range(num_layer)])

		if norm_output:
			self.out_normer = LayerNorm(isize, ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, inputs, taskid=None, mask=None, **kwargs):

		out = self.wemb(inputs) + self.task_emb.weight[taskid]
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		_gwf, _gw = self.group_weight_flayer[taskid].softmax(-2), self.group_weight[taskid].softmax(-2)

		if self.drop is not None:
			out = self.drop(out)
			_gwf = self.gw_drop(_gwf)
			_gw = self.gw_drop(_gw)

		_w = [(None, _gwf,)] + [_wu.unbind(0) for _wu in _gw.unbind(0)]
		for net, (_w_attn, _w_ffn,) in zip(self.nets, _w):
			out = net(out, attn_w=_w_attn, ffn_w=_w_ffn, taskid=taskid, mask=mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		return self.transo(out, taskid)
