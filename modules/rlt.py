#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.autograd import Function
from torch.nn import functional as nnFunc

from modules.act import Custom_Act, LGLU, get_act
from modules.base import CrossAttn as CrossAttnBase, Dropout, PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.fmt.parser import parse_none
from utils.math import arcsigmoid
from utils.torch.comp import mask_tensor_type, torch_no_grad

from cnfg.ihyp import *

# portal from modules.advdrop

class MaskFunction(Function):

	# output = inputs * bernoulli(1.0 - p)
	@staticmethod
	def forward(ctx, ip, op, weight, bias):

		mi = ip.bernoulli().to(mask_tensor_type, non_blocking=True)
		mo = op.bernoulli().to(mask_tensor_type, non_blocking=True)
		m = mi | mo
		_w = weight.masked_fill(m, 0.0)
		if bias is None:
			_b = _smo = None
		else:
			_smo = mo.squeeze(-1)
			_b = bias.masked_fill(_smo, 0.0)

		ctx.save_for_backward(ip, op, _smo, weight, bias, m)

		return _w, _b

	@staticmethod
	def backward(ctx, grad_weight, grad_bias):

		if (grad_weight is None) and (grad_bias is None):
			return None, None, None, None
		else:
			ip, op, smo, weight, bias, m = ctx.saved_tensors
			_needs_grad_ip, _needs_grad_op = ctx.needs_input_grad[:2]
			if _needs_grad_ip or _needs_grad_op:
				_grad_m = - grad_weight * weight
				if _needs_grad_ip:
					_grad_ip = (_grad_m * op).sum(1, keepdim=True)
				else:
					_grad_ip = None
				if _needs_grad_op:
					_grad_op = (_grad_m * ip).sum(2, keepdim=True)
					if grad_bias is not None:
						_grad_op -= (grad_bias * bias).unsqueeze(-1)
				else:
					_grad_op = None
			_grad_weight_input = grad_weight.masked_fill(m, 0.0) if ctx.needs_input_grad[2] else None
			_grad_bias_input = grad_bias.masked_fill(smo.squeeze(-1), 0.0) if ctx.needs_input_grad[3] else None
			return _grad_ip, _grad_op, _grad_weight_input, _grad_bias_input

MaskFunc = MaskFunction.apply

class RLTLinear(nn.Module):

	def __init__(self, in_features, out_features, bias=True, ngroup=4, p=None, **kwargs):

		super(RLTLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.ngroup = ngroup
		self.p = max(0.5, (1.0 - 1.0 / ngroup)) if p is None else p
		self.weight = nn.Parameter(torch.Tensor(ngroup, out_features, in_features))
		if bias:
			self.bias = nn.Parameter(torch.zeros(ngroup, out_features))
		else:
			self.register_parameter("bias", None)
		self.drop_i = nn.Parameter(torch.Tensor(ngroup, 1, in_features))
		self.drop_o = nn.Parameter(torch.Tensor(ngroup, out_features, 1))
		self.fix_init()

	def reset_parameters(self):

		self.fix_init()

	def fix_init(self):

		with torch_no_grad():
			_bound = sqrt(1.0 / self.weight.size(1))
			self.weight.data.uniform_(-_bound, _bound)
			_arc_p = arcsigmoid(self.p)
			self.drop_i.data.fill_(_arc_p)
			self.drop_o.data.fill_(_arc_p)
			if self.bias is not None:
				self.bias.data.zero_()

	def forward(self, input, **kwargs):

		_w, _b = self.get_weight_bias()

		return nnFunc.linear(input, _w, _b)

	def get_weight_bias(self, training=None):

		_training = parse_none(training, self.training)
		if _training:
			_w, _b = MaskFunc(self.drop_i.sigmoid(), self.drop_o.sigmoid(), self.weight, self.bias)
			_w = _w.sum(0)
			if _b is not None:
				_b = _b.sum(0)
		else:
			_mo = 1.0 - self.drop_o.sigmoid()#op.gt(0.5)
			_m = (1.0 - self.drop_i.sigmoid()) * _mo#ip.gt(0.5) | _mo
			_w = (self.weight * _m).sum(0)#self.weight.masked_fill(_m, 0.0).sum(0)
			_b = None if self.bias is None else (self.bias * _mo.squeeze(-1)).sum(0)#.masked_fill(_mo.squeeze(-1), 0.0).sum(0)

		return _w, _b

	def extra_repr(self):
		return "num_tickets={}, in_features={}, out_features={}, bias={}".format(self.ngroup, self.in_features, self.out_features, self.bias is not None)

	def to_std(self):

		_rsm = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
		with torch_no_grad():
			_w, _b = self.get_weight_bias(self.drop_i.sigmoid(), self.drop_o.sigmoid(), training=False)
			_rsm.weight.copy_(_w)
			if _b is not None:
				_rsm.bias.copy_(_b)
		return _rsm

# portal from modules.prune.bern
class BernoulliMaskFunction(Function):

	@staticmethod
	def forward(ctx, inputs, maskp, inplace=False):

		_mask = maskp.bernoulli()
		mask = _mask.to(mask_tensor_type, non_blocking=True)
		ctx.save_for_backward(inputs, mask)
		return inputs.masked_fill_(mask, 0.0) if inplace else inputs.masked_fill(mask, 0.0)

	@staticmethod
	def backward(ctx, grad_outputs):

		if grad_outputs is None:
			return None, None, None
		else:
			inputs, mask = ctx.saved_tensors
			_grad_input = grad_outputs.masked_fill(mask, 0.0) if ctx.needs_input_grad[0] else None
			_grad_maskp = -grad_outputs * inputs if ctx.needs_input_grad[1] else None
			return _grad_input, _grad_maskp, None

BernoulliMaskFunc = BernoulliMaskFunction.apply

class BernoulliParameter(nn.Module):

	def __init__(self, tensor_in, p=None, **kwargs):

		super(BernoulliParameter, self).__init__()

		self.data = nn.Parameter(tensor_in)
		self.init_value = None if p is None or p <= 0.0 or p >= 1.0 else arcsigmoid(p)
		self.maskp = nn.Parameter(tensor_in.abs().detach()) if self.init_value is None else nn.Parameter(tensor_in.new_empty(tensor_in.size()).fill_(self.init_value))

	def forward(self):

		return BernoulliMaskFunc(self.data, self.maskp.sigmoid(), False) if self.training else (self.data * (1.0 - self.maskp.sigmoid()))#.masked_fill(self.maskp.sigmoid().ge(0.5), 0.0)

	def fix_init(self):

		if self.maskp is not None:
			with torch_no_grad():
				self.maskp.copy_(self.abs().data) if self.init_value is None else self.maskp.fill_(self.init_value)

class BNLinear(nn.Module):

	def __init__(self, in_features, out_features, bias=True, ngroup=4, p=None, **kwargs):

		super(BNLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.p = max(0.75, 1.0 - 1.0 / (ngroup * ngroup)) if p is None else p
		self.weight = BernoulliParameter(torch.Tensor(ngroup, out_features, in_features), p=self.p)
		if bias:
			self.bias = BernoulliParameter(torch.Tensor(ngroup, out_features), p=self.p)
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):

		self.fix_init()

	def fix_init(self):

		with torch_no_grad():
			_bound = sqrt(1.0 / self.weight.data.size(1))
			self.weight.data.data.uniform_(-_bound, _bound)
			self.weight.fix_init()
			if self.bias is not None:
				self.bias.data.data.zero_()
				self.bias.fix_init()

	def forward(self, input, **kwargs):

		_w, _b = self.get_weight_bias()

		return nnFunc.linear(input, _w, _b)

	def get_weight_bias(self):

		_w = self.weight().sum(0)
		_b = None if self.bias is None else self.bias().sum(0)

		return _w, _b

	def extra_repr(self):
		return "num_tickets={}, in_features={}, out_features={}, bias={}".format(self.ngroup, self.in_features, self.out_features, self.bias is not None)

	def to_std(self):

		_rsm = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
		with torch_no_grad():
			_training = self.training
			if _training:
				self.eval()
			_w, _b = self.get_weight_bias()
			if _training:
				self.train()
			_rsm.weight.copy_(_w)
			if _b is not None:
				_rsm.bias.copy_(_b)
		return _rsm

LTLinear = BNLinear

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, ngroup=4, p=None, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.adaptor = LTLinear(isize, self.hsize * 3, bias=enable_proj_bias, ngroup=ngroup, p=p)
		self.outer = LTLinear(self.hsize, osize, bias=enable_bias, ngroup=ngroup, p=p)

class CrossAttn(CrossAttnBase):

	def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, k_isize=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, ngroup=4, p=None, **kwargs):

		super(CrossAttn, self).__init__(isize, hsize, osize, num_head=num_head, dropout=dropout, k_isize=k_isize, enable_bias=enable_bias, enable_proj_bias=enable_proj_bias, **kwargs)

		self.query_adaptor = LTLinear(isize, self.hsize, bias=enable_proj_bias, ngroup=ngroup, p=p)
		self.kv_adaptor = LTLinear(isize if k_isize is None else k_isize, self.hsize * 2, bias=enable_proj_bias, ngroup=ngroup, p=p)
		self.outer = LTLinear(self.hsize, osize, bias=enable_bias, ngroup=ngroup, p=p)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, ngroup=4, p=None, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias, use_glu=use_glu, **kwargs)

		_ = [LTLinear(isize, _hsize, ngroup=ngroup, p=p)]
		_drop_ind = 2
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), LTLinear(_hsize, isize, bias=enable_bias, ngroup=ngroup, p=p)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(LTLinear(_hsize // 2, isize, bias=enable_bias, ngroup=ngroup, p=p))
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			_.insert(_drop_ind, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, ngroup=4, p=None, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, ngroup=ngroup, p=p, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, ngroup=4, p=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, ngroup=ngroup, p=p, **kwargs)
