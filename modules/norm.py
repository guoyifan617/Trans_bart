#encoding: utf-8

#import torch
from math import sqrt
from numbers import Integral
from torch import nn

#from modules.base import CrossAttn as CrossAttnBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class LayerNorm(nn.LayerNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, bias=True, **kwargs):

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

		if not bias:
			self.register_parameter("bias", None)

	# uncomment the forward function to override the pytorch default implementation in case needed
	"""def forward(self, x):

		_std, _mean = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)# x.std(dim=-1, unbiased=False, keepdim=True), x.mean(dim=-1, keepdim=True)#.detach()
		_xn = (x - _mean) / (_std + self.eps)
		if self.weight is not None:
			if self.bias is not None:
				_xn = self.bias.addcmul(self.weight, _xn)
			else:
				_xn.mul_(self.weight)
		elif self.bias is not None:
			_xn.add_(self.bias)

		return _xn"""

	def reset_parameters(self):

		with torch_no_grad():
			if self.weight is not None:
				self.weight.fill_(1.0)
			if self.bias is not None:
				self.bias.zero_()

	def fix_init(self):

		self.reset_parameters()

	def load_base(self, base_ln):

		with torch_no_grad():
			if self.weight is not None and base_ln.weight is not None:
				self.weight.copy_(base_ln.weight)
				self.weight.requires_grad_(base_ln.weight.requires_grad)
			if self.bias is not None and base_ln.bias is not None:
				self.bias.copy_(base_ln.bias)
				self.bias.requires_grad_(base_ln.bias.requires_grad)

class MagNorm(LayerNorm):

	def forward(self, x, **kwargs):

		_mean = x.mean(dim=-1, keepdim=True)
		_xn = x - _mean
		_xn = _xn / (_xn.norm(p=2,dim=-1,keepdim=True) + self.eps)
		if self.weight is not None:
			if self.bias is not None:
				_xn = self.bias.addcmul(self.weight, _xn)
			else:
				_xn.mul_(self.weight)
		elif self.bias is not None:
			_xn.add_(self.bias)

		return _xn

class SimpNorm(LayerNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, **kwargs):

		super(SimpNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)
		self.bias = None

	def forward(self, x, **kwargs):

		_xn = x.div(x.std(dim=-1, unbiased=False, keepdim=True).add(self.eps))
		if self.weight is not None:
			_xn = _xn.mul_(self.weight)

		return _xn

class RMSNorm(SimpNorm):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, **kwargs):

		super(RMSNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)
		self.k = 1.0 / sqrt(float(normalized_shape if isinstance(normalized_shape, Integral) else normalized_shape[-1]))

	def forward(self, x, **kwargs):

		_xn = x.div(x.norm(p=2, dim=-1, keepdim=True).mul(self.k).add_(self.eps))
		if self.weight is not None:
			_xn = _xn.mul_(self.weight)

		return _xn

class NormMax(nn.Module):

	def __init__(self, dim=-1, k=8.0, eps=ieps_ln_default, **kwargs):

		super(NormMax, self).__init__()

		self.dim, self.eps, self.k = dim, eps, float(k)

	def forward(self, x, mask=None, **kwargs):

		def std(xi, dim=-1, keepdim=False, mask=None, eps=ieps_ln_default):
			if mask is None:
				return xi.std(dim=dim, keepdim=keepdim) if xi.size(dim) > 1 else xi.new_ones(1, dtype=xi.dtype, device=xi.device)
			else:
				_sx, _ne = xi.masked_fill(mask, 0.0).sum(dim=dim, keepdim=True), float(mask.size(dim)) - mask.float().sum(dim=dim, keepdim=True)
				_sx /= _ne
				_xstd = ((xi - _sx).pow(2).masked_fill_(mask, 0.0).sum(dim=dim, keepdim=keepdim) / (_ne - 1.0 + eps)).sqrt()
				return _xstd

		_xn = x / ((std(x, dim=self.dim, keepdim=True, mask=mask, eps=self.eps) + self.eps) / self.k)

		if mask is not None:
			_xn.masked_fill_(mask, -inf_default)

		return _xn.softmax(dim=self.dim)

"""class SelfAttn(SelfAttnBase):

	def __init__(self, *args, **kwargs):

		super(SelfAttn, self).__init__(*args, **kwargs)

		self.normer = NormMax(dim=-1)

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is None:
				seql = nquery
			else:
				seql = nquery + _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		# we do not scale attention scores, since it is done by NormMax
		#scores = scores / sqrt(adim)

		scores = self.normer(scores, mask=None if mask is None else mask.unsqueeze(1))

		if self.drop is not None:
			scores = self.drop(scores)

		out = self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class CrossAttn(CrossAttnBase):

	def __init__(self, *args, **kwargs):

		super(CrossAttn, self).__init__(*args, **kwargs)

		self.normer = NormMax(dim=-1)

	def forward(self, iQ, iK, mask=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		seql = iK.size(1)
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2)
		if (self.real_iK is not None) and self.iK.is_set_to(iK) and self.is_decoding:
			real_iK, real_iV = self.real_iK, self.real_iV
		else:
			real_iK, real_iV = self.kv_adaptor(iK).view(bsize, seql, 2, nheads, adim).unbind(2)
			real_iK, real_iV = real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)
			if self.is_decoding:
				self.iK, self.real_iK, self.real_iV = iK, real_iK, real_iV

		# we do not scale attention scores, since it is done by NormMax
		scores = real_iQ.matmul(real_iK)# / sqrt(adim)

		scores = self.normer(scores, mask=None if mask is None else mask.unsqueeze(1))

		if self.drop is not None:
			scores = self.drop(scores)

		return self.outer(scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize))

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = CrossAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)"""
