#encoding: utf-8

import torch
from math import ceil
from random import random
from torch import nn

from utils.base import reduce_model_list
from utils.torch.comp import mask_tensor_type

Dropout = nn.Dropout

class TokenDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, keep_magnitude=True, **kwargs):

		super(TokenDropout, self).__init__(p=p, inplace=inplace)
		self.keep_magnitude = (1.0 / (1.0 - self.p)) if keep_magnitude else False
		self.register_buffer("pcache", torch.full((1,), self.p), persistent=False)

	def forward(self, inpute, **kwargs):

		if self.training:
			_ = inpute.dim() - 1
			_p = self.pcache.view([1 for i in range(_)]) if _ > 1 else self.pcache
			mask = _p.expand(inpute.size()[:-1]).bernoulli().to(mask_tensor_type, non_blocking=True).unsqueeze(-1)
			if self.inplace:
				out = inpute.masked_fill_(mask, 0.0)
				if self.keep_magnitude:
					out.mul_(self.keep_magnitude)
			else:
				out = inpute.masked_fill(mask, 0.0)
				if self.keep_magnitude:
					out = out * self.keep_magnitude

			return out
		else:
			return inpute

class PartTokenDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, p1=0.5, keep_magnitude=True, **kwargs):

		super(PartTokenDropout, self).__init__(p=p, inplace=inplace)
		self.p1 = min(float(p1), self.p)
		self.keep_magnitude, self.p2 = (1.0 / (1.0 - self.p)) if keep_magnitude else False, self.p / self.p1
		self.register_buffer("pcache", torch.full((1,), self.p1), persistent=False)

	def forward(self, inpute, **kwargs):

		if self.training:
			_ = inpute.dim() - 1
			_p1 = self.pcache.view([1 for i in range(_)]) if _ > 1 else self.pcache
			_isize = inpute.size()
			mask = _p1.expand(_isize[:-1]).bernoulli().to(mask_tensor_type, non_blocking=True).unsqueeze(-1)
			mask = inpute.new_zeros(_isize, requires_grad=False).masked_fill_(mask, self.p2).bernoulli().to(mask_tensor_type, non_blocking=True)
			if self.inplace:
				out = inpute.masked_fill_(mask, 0.0)
				if self.keep_magnitude:
					out.mul_(self.keep_magnitude)
			else:
				out = inpute.masked_fill(mask, 0.0)
				if self.keep_magnitude:
					out = out * self.keep_magnitude

			return out
		else:
			return inpute

def norm(lin):

	_t = sum(lin)
	return tuple([lu / _t for lu in lin])

def sample(lin):

	_t = random()
	rs, _s = len(lin) - 1, 0.0
	for i, v in enumerate(lin):
		_s += v
		if _s >= _t:
			rs = i
			break

	return rs

class NGramDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, seqdim=1, sample_p=[1.0 / tmpu for tmpu in range(1, 3 + 1)], keep_magnitude=True, **kwargs):

		super(NGramDropout, self).__init__(p=p, inplace=inplace)
		self.seqdim = seqdim
		self.keep_magnitude = (1.0 / (1.0 - self.p)) if keep_magnitude else False
		self.sample_p = norm([float(pu) for pu in sample_p])
		self.max_n = len(sample_p)
		self.register_buffer("pcache", torch.full((1,), self.p), persistent=False)

	def forward(self, inpute, **kwargs):

		if self.training:
			seql = inpute.size(self.seqdim)
			ngram = sample(self.sample_p if seql > self.max_n else norm(self.sample_p[:seql - 1])) + 1
			_msize = list(inpute.size())[:-1]
			_ = len(_msize)
			_p = self.pcache.view([1 for i in range(_)]) if _ > 1 else self.pcache
			if ngram > 1:
				nblock = ceil(float(seql) / float(ngram))
				_msize[self.seqdim] = nblock
				mask = _p.expand(_msize).bernoulli().to(mask_tensor_type, non_blocking=True).repeat([ngram if i == self.seqdim else 1 for i in range(len(_msize))])
				if ngram * nblock != seql:
					mask = mask.narrow(self.seqdim, 0, seql)
				mask = mask.unsqueeze(-1)
			else:
				mask = _p.expand(_msize).bernoulli().to(mask_tensor_type, non_blocking=True).unsqueeze(-1)
			if self.inplace:
				out = inpute.masked_fill_(mask, 0.0)
				if self.keep_magnitude:
					out.mul_(self.keep_magnitude)
			else:
				out = inpute.masked_fill(mask, 0.0)
				if self.keep_magnitude:
					out = out * self.keep_magnitude

			return out
		else:
			return inpute

# Keep the sum of dropout output in the given dim to be 1. Using this (without detaching) does not affect BLEU.
class MagOneDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, dim=-1, eps=1e-6, **kwargs):

		super(MagOneDropout, self).__init__(p=p, inplace=inplace)
		self.dim, self.eps = dim, eps
		self.register_buffer("pcache", torch.full((1,), self.p), persistent=False)

	def forward(self, inpute, **kwargs):

		if self.training:
			_ = inpute.dim()
			_p = self.pcache.view([1 for i in range(_)]) if _ > 1 else self.pcache
			mask = _p.expand_as(inpute).bernoulli().to(mask_tensor_type, non_blocking=True)
			out = inpute.masked_fill_(mask, 0.0) if self.inplace else inpute.masked_fill(mask, 0.0)
			# without detaching seems wrong, but with detaching leads to training issues.
			_sum = out.sum(dim=self.dim, keepdim=True).add_(self.eps)#.detach()

			return out.div_(_sum)
		else:
			return inpute

# problematic, since softmax does not scale linearly, which introduces a gap between training and evaluation
class InfDropout(Dropout):

	def __init__(self, p=0.5, inplace=False, mask_value=-1e32, **kwargs):

		super(InfDropout, self).__init__(p=p, inplace=inplace)
		self.mv = float(mask_value)#, self.keep_magnitude, 1.0 / (1.0 - self.p)
		self.register_buffer("pcache", torch.full((1,), self.p), persistent=False)

	def forward(self, inpute, **kwargs):

		if self.training:
			_ = inpute.dim()
			_p = self.pcache.view([1 for i in range(_)]) if _ > 1 else self.pcache
			mask = _p.expand_as(inpute).bernoulli().to(mask_tensor_type, non_blocking=True)

			return inpute.masked_fill_(mask, self.mv) if self.inplace else inpute.masked_fill(mask, self.mv)#* self.keep_magnitude
		else:
			return inpute

def reduce_model(modin):

	return reduce_model_list(modin, [Dropout, TokenDropout, PartTokenDropout, NGramDropout, MagOneDropout, InfDropout], [lambda m: (m.p, m.inplace,), lambda m: (m.p, m.inplace, m.keep_magnitude,), lambda m: (m.p, m.inplace, m.p1, m.keep_magnitude, m.p2,), lambda m: (m.p, m.inplace, m.seqdim, m.keep_magnitude, m.sample_p, m.max_n,), lambda m: (m.p, m.inplace, m.dim, m.eps,), lambda m: (m.p, m.inplace,)])
