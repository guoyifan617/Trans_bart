#encoding: utf-8

import torch
from torch import nn

from modules.base import Linear

class GateResidual(nn.Module):

	# isize: input dimension

	def __init__(self, isize, **kwargs):

		super(GateResidual, self).__init__()

		self.net = nn.Sequential(Linear(isize * 2, isize), nn.Sigmoid())

	def forward(self, x1, x2, **kwargs):

		gate = self.net(torch.cat((x1, x2,), dim=-1))

		return x1 * gate + x2 * (1.0 - gate)

class SelfGate(nn.Module):

	def __init__(self, isize, **kwargs):

		super(SelfGate, self).__init__()

		self.net = nn.Sequential(Linear(isize * 2, isize * 2), nn.Sigmoid())

	def forward(self, x1, x2, **kwargs):

		_cinput = torch.cat((x1, x2,), dim=-1)
		out = self.net(_cinput) * _cinput
		_isize = list(x1.size())
		_isize.append(2)

		return out.view(_isize).sum(-1)

class RSelfGate(nn.Module):

	def __init__(self, isize, **kwargs):

		super(RSelfGate, self).__init__()

		self.net = nn.Sequential(Linear(isize * 2, isize), nn.Sigmoid())

	def forward(self, x1, x2, **kwargs):

		return self.net(torch.cat((x1, x2,), dim=-1)) * x1
