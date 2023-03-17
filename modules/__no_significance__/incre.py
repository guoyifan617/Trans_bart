#encoding: utf-8

import torch
from math import exp
from torch import nn

from cnfg.ihyp import *

class SigmoidIncremental:

	# warm_steps: increase from 0.0 to about (0.9866 * target_value) in warm_steps
	# target_value: target value returned after infinity calls to step() function

	def __init__(self, warm_steps, target_value, cur_step=0, **kwargs):

		self.wstep = float(warm_steps) / 5.0
		self.tarv = target_value
		self.cur_step = float(cur_step)

	def step(self):

		self.cur_step += 1.0

		return (2.0 / (1.0 + exp(- self.cur_step / self.wstep)) - 1.0) * self.tarv

class SigmoidITensor(nn.Module):

	# warm_steps: increase from 0.0 to about (0.9866 * target_value) in warm_steps
	# target_value: target value returned after infinity calls to step() function

	def __init__(self, warm_steps, target_value, xseql=cache_len_default, **kwargs):

		super(SigmoidITensor, self).__init__()
		self.wstep = float(warm_steps) / 5.0
		self.tarv = target_value
		self.xseql = xseql
		self.register_buffer("w", ((((torch.arange(1, xseql + 1, dtype=torch.float, requires_grad=False) / self.wstep)).sigmoid() * 2 - 1) * self.tarv).unsqueeze(0).unsqueeze(-1), persistent=False)

	def forward(self, x, expand=True, **kwargs):

		seql = x.size(1)

		out = self.get_ext(seql) if seql > self.xseql else self.w.narrow(1, 0, seql)

		return out.expand_as(x) if expand else out

	def get_ext(self, seql):

		_tmp = ((((torch.arange(self.xseql + 1, seql + 1, dtype=self.w.dtype, device=self.w.device, requires_grad=False) / self.wstep)).sigmoid() * 2.0 - 1.0) * self.tarv).unsqueeze(0).unsqueeze(-1)

		return torch.cat((self.w, _tmp), 1)
