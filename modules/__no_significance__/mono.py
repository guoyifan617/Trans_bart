#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.autograd import Function
from torch.nn import functional as nnFunc

from modules.act import Custom_Act
from modules.base import Dropout, GradientReversalLayer, Linear, PositionalEmb as PositionalEmbBase, Scorer

from cnfg.ihyp import *

class DualBiasLinear(nn.Module):

	def __init__(self, in_features, out_features, bias=True, **kwargs):

		super(DualBiasLinear, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(- sqrt(1.0 / in_features), sqrt(1.0 / in_features)))
		if bias:
			self.bias = nn.Parameter(torch.zeros(2, out_features))
		else:
			self.register_parameter("bias", None)

	def forward(self, input, bias_id=0, **kwargs):

		return nnFunc.linear(input, self.weight, self.bias[bias_id])

class PositionalEmb(PositionalEmbBase):

	def forward(self, x, expand=True, sind=None, **kwargs):

		bsize, seql = x.size()

		if sind is None:
			rs = self.w[:seql].unsqueeze(0) if seql <= self.num_pos else torch.cat((self.w, self.get_ext(seql, False)), 0).unsqueeze(0)
		else:
			_seql = seql + sind
			rs = self.w[sind:_seql].unsqueeze(0) if _seql <= self.num_pos else torch.cat((self.w[sind:], self.get_ext(_seql, False)), 0).unsqueeze(0)

		return rs.expand(bsize, seql, self.num_dim) if expand else rs

class FFDiscriminator(nn.Module):

	def __init__(self, isize, num_layer=3, dropout=0.0, custom_act=use_adv_act_default, **kwargs):

		super(FFDiscriminator, self).__init__()

		self.grl = GradientReversalLayer()
		_act = Custom_Act() if custom_act else nn.Sigmoid()
		self.drop_base = Dropout(dropout) if dropout > 0.0 else None
		_drop = Dropout(dropout, inplace=True) if custom_act and (dropout > 0.0) else self.drop_base
		_hsize = isize * 4
		self.nets = nn.ModuleList([nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _hsize), _act, _drop, Linear(_hsize, isize), _drop) for i in range(num_layer)]) if dropout > 0.0 else nn.ModuleList([nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _hsize), _act, Linear(_hsize, isize)) for i in range(num_layer)])
		self.scorer = Scorer(isize, False)

	def forward(self, input, **kwargs):

		out = self.grl(input)
		if self.drop_base is not None:
			out = self.drop_base(out)
		for net in self.nets:
			_out = net(out) + out

		return self.scorer(out)

	def get_clip_para(self):

		_tmpl = []
		for _m in self.modules():
			if isinstance(_m, (Linear, Scorer,)):
				_tmpl.extend(list(_m.parameters()))

		return _tmpl

class GradientBalanceFunction(Function):

	@staticmethod
	def forward(ctx, inputs, mask=None, weight=1.0):

		ctx.weight = weight
		if mask is None:
			ctx.has_mask = False
		else:
			ctx.save_for_backward(mask)
			ctx.has_mask = True

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		if ctx.needs_input_grad[0]:
			_weight = ctx.weight
			if ctx.has_mask:
				_mask = ctx.saved_tensors[0]
				g1m = grad_output1.masked_fill(_mask, 0.0).norm(p=1)
				g2m = grad_output2.masked_fill(_mask, 0.0).norm(p=1)
			else:
				g1m, g2m = grad_output1.norm(p=1), grad_output2.norm(p=1)
			return g2m / g1m * _weight * grad_output1 + grad_output2, None, None
		else:
			return None, None, None

class GradientBalanceVecFunction(Function):

	@staticmethod
	def forward(ctx, inputs, mask=None, weight=1.0, dim=-1):

		ctx.weight = weight
		ctx.dim = dim

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		if ctx.needs_input_grad[0]:
			_weight = ctx.weight
			_dim = ctx.dim
			g1m, g2m = grad_output1.norm(p=2, dim=_dim, keepdim=True), grad_output2.norm(p=2, dim=_dim, keepdim=True)
			g1m.masked_fill_(g1m.eq(0.0), 1.0)
			return g2m / g1m * _weight * grad_output1 + grad_output2, None, None
		else:
			return None, None, None

GradientBalanceFunc = GradientBalanceVecFunction.apply

class GradientBalanceLayer(nn.Module):

	def __init__(self, weight=1.0, **kwargs):

		super(GradientBalanceLayer, self).__init__()

		self.weight = float(weight)

	def forward(self, inputs, mask=None, **kwargs):

		return GradientBalanceFunc(inputs, mask, self.weight)

# cos<a, b> * a
class ScaledGradientGuideFunction(Function):

	@staticmethod
	def forward(ctx, inputs, dim=-1):

		ctx.dim = dim

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		if ctx.needs_input_grad[0]:
			_dim = ctx.dim
			g1m, g2m = grad_output1.norm(p=2, dim=_dim, keepdim=True), grad_output2.norm(p=2, dim=_dim, keepdim=True)
			gm = g1m * g2m
			gm.masked_fill_(gm.eq(0.0), 1.0)
			_weight = (grad_output1 * grad_output2).sum(dim=_dim, keepdim=True) / gm
			_weight.masked_fill_(_weight.lt(0.0), 0.0)
			return _weight * grad_output1, None
		else:
			return None, None

# cos<a, b> * a / |a|
class NormGradientGuideFunction(Function):

	@staticmethod
	def forward(ctx, inputs, dim=-1):

		ctx.dim = dim

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		if ctx.needs_input_grad[0]:
			_dim = ctx.dim
			g1m, g2m = grad_output1.pow(2).sum(dim=_dim, keepdim=True), grad_output2.norm(p=2, dim=_dim, keepdim=True)
			gm = g1m * g2m
			gm.masked_fill_(gm.eq(0.0), 1.0)
			_weight = (grad_output1 * grad_output2).sum(dim=_dim, keepdim=True) / gm
			_weight.masked_fill_(_weight.lt(0.0), 0.0)
			return _weight * grad_output1, None
		else:
			return None, None

# cos<a, b> * |b| * a / |a|
class ProjGradientGuideFunction(Function):

	@staticmethod
	def forward(ctx, inputs, dim=-1):

		ctx.dim = dim

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_output1, grad_output2):

		if ctx.needs_input_grad[0]:
			_dim = ctx.dim
			g1m = grad_output1.pow(2).sum(dim=_dim, keepdim=True)
			g1m.masked_fill_(g1m.eq(0.0), 1.0)
			_weight = (grad_output1 * grad_output2).sum(dim=_dim, keepdim=True) / g1m
			_weight.masked_fill_(_weight.lt(0.0), 0.0)
			return _weight * grad_output1, None
		else:
			return None, None

GradientGuideFunc = ProjGradientGuideFunction.apply
