#encoding: utf-8

import torch
from torch import nn

from modules.base import PositionwiseFF as PositionwiseFFBase, ResCrossAttn as ResCrossAttnBase, ResSelfAttn as ResSelfAttnBase
from utils.fmt.parser import parse_none
from utils.torch.ext import randint_t_core

from cnfg.ihyp import *

class GausNoiser(nn.Module):

	def __init__(self, power, inplace=False, **kwargs):

		super(GausNoiser, self).__init__()
		self.power, self.inplace = power, inplace

	# mask: (bsize, seql, 1), otherwise cannot multiply with inpute.size(-1)
	def forward(self, inpute, mask=None, **kwargs):

		if self.training:
			_noise = self.get_noise(inpute.detach(), mask=mask)

			return inpute.add_(_noise) if self.inplace else inpute.add(_noise)

		return inpute

	def get_noise(self, inpute, mask=None):

		if mask is None:
			base_p = inpute.abs().mean() * self.power
		else:
			base_p = inpute.masked_fill(mask, 0.0).norm(p=1) * (self.power / float((mask.numel() - mask.sum().item()) * inpute.size(-1)))

		return torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device).mul_(base_p)

class UniNoiser(GausNoiser):

	def get_noise(self, inpute, mask=None):

		if mask is None:
			base_p = inpute.abs().mean().item() * self.power
		else:
			base_p = inpute.masked_fill(mask, 0.0).norm(p=1).item() / float((mask.numel() - mask.sum().item()) * inpute.size(-1)) * self.power

		return inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-base_p, base_p)

class GausNoiserVec(GausNoiser):

	def __init__(self, power, dim=-1, inplace=False, eps=ieps_noise_default, **kwargs):

		super(GausNoiserVec, self).__init__(power, inplace=inplace)
		self.dim, self.eps = dim, eps

	def get_noise(self, inpute, mask=None):

		_noise = torch.randn(inpute.size(), dtype=inpute.dtype, device=inpute.device)
		base_p = inpute.norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

		return _noise.mul_(base_p)

class UniNoiserVec(GausNoiserVec):

	def get_noise(self, inpute, mask=None):

		_noise = inpute.new_empty(inpute.size(), requires_grad=False).uniform_(-1.0, 1.0)
		base_p = inpute.norm(p=2, dim=self.dim, keepdim=True) / (_noise.norm(p=2, dim=self.dim, keepdim=True) + self.eps) * self.power

		return _noise.mul_(base_p)

# encoder only
class EMixNoiser(nn.Module):

	def __init__(self, power, inplace=False, dim=-1, **kwargs):

		super(EMixNoiser, self).__init__()
		self.power, self.inplace, self.dim = power, inplace, dim#self.alpha	1.0 - power

	# sample_t: (~mask).to(inpute.dtype, non_blocking=True)
	def forward(self, inpute, sample_t=None, **kwargs):

		if self.training:
			_noise = self.get_zero_mean_noise(inpute.detach(), sample_t=sample_t)#get_noise
			#out = inpute.mul_(self.alpha) if self.inplace else inpute.mul(self.alpha)

			return inpute.add_(_noise, alpha=self.power) if self.inplace else inpute.add(_noise, alpha=self.power)#out.add_(_noise, alpha=self.power)

		return inpute

	def get_zero_mean_noise(self, inpute, sample_t=None):

		_noise = self.get_noise(inpute, sample_t=None)

		return _noise.sub_(_noise.mean(dim=self.dim, keepdim=True))

	def get_noise(self, inpute, sample_t=None):

		_tmp = inpute.select(-1, 0)
		_ndraw = _tmp.numel()
		indices = torch.randperm(_ndraw, device=inpute.device) if sample_t is None else sample_t.view(-1).multinomial(_ndraw, replacement=True)

		return inpute.view(-1, inpute.size(-1)).index_select(0, indices).view(inpute.size())

class SEMixNoiser(EMixNoiser):

	# inpute: (bsize, seql, isize)
	def get_noise(self, inpute, sample_t=None):

		bsize, seql, isize = inpute.size()
		indices = (inpute.new_ones(bsize, seql) if sample_t is None else sample_t).multinomial(seql, replacement=True)
		indices = indices + torch.arange(0, bsize * seql, seql, dtype=indices.dtype, device=indices.device).unsqueeze(-1)

		return inpute.view(-1, isize).index_select(0, indices.view(-1)).view(bsize, seql, isize)

class BMixNoiser(EMixNoiser):

	# inpute: (bsize, seql, isize)
	# sample_t: (~mask).to(inpute.dtype, non_blocking=True).transpose(0, 1)
	def get_noise(self, inpute, sample_t=None):

		bsize, seql, isize = inpute.size()
		indices = (inpute.new_ones(seql, bsize) if sample_t is None else sample_t).multinomial(bsize, replacement=True)
		indices = torch.arange(0, bsize * seql, seql, dtype=indices.dtype, device=indices.device).index_select(0, indices.view(-1)).view(seql, bsize).transpose(0, 1).contiguous().add_(torch.arange(seql, dtype=indices.dtype, device=indices.device))#indices.mul_(seql)

		return inpute.view(-1, isize).index_select(0, indices.view(-1)).view(bsize, seql, isize)

# for decoder, with sampling restricted to preceding tokens, but it is unbalanced comparing with BMixNoiser
class DMixNoiser(EMixNoiser):

	def get_noise(self, inpute, sample_t=None):

		bsize, seql, isize = inpute.size()
		indices = (inpute.new_ones(seql, bsize) if sample_t is None else sample_t).multinomial(bsize, replacement=True)
		indices = torch.arange(0, bsize * seql, seql, dtype=indices.dtype, device=indices.device).index_select(0, indices.view(-1)).view(seql, bsize).transpose(0, 1).contiguous().add_(randint_t_core(torch.arange(seql, dtype=inpute.dtype, device=inpute.device).unsqueeze(0).expand(bsize, seql)))

		return inpute.view(-1, isize).index_select(0, indices.view(-1)).view(bsize, seql, isize)

Noiser = UniNoiserVec

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, power=None, custom_noiser=None, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		_noiser = parse_none(custom_noiser, Noiser)
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, iQ, *inputs, noise_mask=None, **kwargs):

		_iQ = self.normer(iQ)

		if self.noiser is not None:
			_iQ = self.noiser(_iQ, noise_mask)

		outs = self.net(_iQ, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class ResCrossAttn(ResCrossAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, power=None, custom_noiser=None, **kwargs):

		super(ResCrossAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		_noiser = parse_none(custom_noiser, Noiser)
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, iQ, iK, *inputs, noise_mask=None, **kwargs):

		_iQ = self.normer(iQ)

		if self.noiser is not None:
			_iQ = self.noiser(_iQ, noise_mask)

		outs = self.net(_iQ, iK, *inputs, **kwargs)

		if isinstance(outs, tuple):
			_out = outs[0]

			if self.drop is not None:
				_out = self.drop(_out)

			return _out + (_iQ if self.norm_residual else iQ), *outs[1:]

		else:
			if self.drop is not None:
				outs = self.drop(outs)

			return outs + (_iQ if self.norm_residual else iQ)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, power=None, custom_noiser=None, **kwargs):

		super(PositionwiseFF, self).__init__(isize, hsize=hsize, dropout=dropout, act_drop=act_drop, **kwargs)

		_noiser = parse_none(custom_noiser, Noiser)
		self.noiser = None if power is None else _noiser(power, inplace=True)

	def forward(self, x, mask=None, **kwargs):

		_out = self.normer(x)
		if self.noiser is not None:
			_out = self.noiser(_out, mask)

		out = self.net(_out)

		out = out + (_out if self.norm_residual else x)

		return out
