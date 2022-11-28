#encoding: utf-8

import torch
from numbers import Integral
from torch import nn
from torch.nn import functional as nnFunc

class LayerNorm(nn.LayerNorm):

	def __init__(self, normalized_shape, ntask=None, eps=1e-5, elementwise_affine=True, **kwargs):

		if isinstance(normalized_shape, Integral):
			normalized_shape = (ntask, normalized_shape,)
		else:
			normalized_shape = tuple([ntask, *normalized_shape])

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, **kwargs)

		self.normalized_shape = self.normalized_shape[1:]

	def forward(self, x, taskid=None, **kwargs):

		if (self.weight is None) and (self.bias is None):
			_xn = nnFunc.layer_norm(x, self.normalized_shape, None, None, self.eps)
		else:
			_std, _mean = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)# x.std(dim=-1, unbiased=False, keepdim=True), x.mean(dim=-1, keepdim=True)
			_xn = (x - _mean) / (_std + self.eps)
			_bsize = [1 for i in range(x.dim() - len(self.normalized_shape))] + list(self.normalized_shape)
			_bsize[0] = x.size(0)
			if self.weight is not None:
				_xn = _xn * self.weight.index_select(0, taskid).view(_bsize)
			if self.bias is not None:
				_xn = _xn.add_(self.bias.index_select(0, taskid).view(_bsize))

		return _xn
