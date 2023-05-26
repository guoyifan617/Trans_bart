#encoding: utf-8

from torch import nn
from torch.nn import Embedding as EmbeddingBase, Linear as LinearBase, functional as nnFunc

from modules.act import Custom_Act, PruneAct
from modules.base import CrossAttn as CrossAttnBase, Dropout, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class Linear(LinearBase):

	def __init__(self, in_features, out_features, bias=True, prune_ratio=128.0, **kwargs):
		super(Linear, self).__init__(in_features, out_features, bias=bias)

		self.act = PruneAct(prune_ratio)

	def forward(self, input, **kwargs):

		return nnFunc.linear(input, self.act(self.weight), None if self.bias is None else self.act(self.bias))

	def prune_weight(self, thres=0.05):

		with torch_no_grad():
			self.weight.data.copy_(self.act.prune(self.weight.data, thres))
			if self.bias is not None:
				self.bias.data.copy_(self.act.prune(self.bias.data, thres))

class Embedding(EmbeddingBase):

	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False, _weight=None, prune_ratio=128.0, **kwargs):
		super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

		self.act = PruneAct(prune_ratio)

	def forward(self, input, **kwargs):

		return self.act(nnFunc.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse))

	def prune_weight(self, thres=0.05):

		with torch_no_grad():
			self.weight.data.copy_(self.act.prune(self.weight.data, thres))

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias)

		self.net = nn.Sequential(Linear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			self.net.insert(2, Dropout(_act_drop, inplace=inplace_after_Custom_Act))

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.adaptor = Linear(isize, self.hsize * 3, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.query_adaptor = Linear(isize, self.hsize, bias=enable_proj_bias)

		self.kv_adaptor = Linear(isize if k_isize is None else k_isize, self.hsize * 2, bias=enable_proj_bias)

		self.outer = Linear(self.hsize, osize, bias=enable_bias)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)
