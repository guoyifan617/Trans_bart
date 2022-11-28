#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, ResidueCombiner

from cnfg.ihyp import *

class GatedCombiner(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, custom_act=use_adv_act_default, **kwargs):

		super(GatedCombiner, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.net = nn.Sequential(Linear(isize * 2, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize), nn.Sigmoid()) if dropout > 0.0 else nn.Sequential(Linear(isize * 2, _hsize), Custom_Act() if custom_act else nn.Sigmoid(), Linear(_hsize, isize), nn.Sigmoid())

	def forward(self, input1, input2, **kwargs):

		gate = self.net(torch.cat((input1, input2,), -1))

		return input1 * gate + (1.0 - gate) * input2

class Node(nn.Module):

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(Node, self).__init__()

		self.comb = ResidueCombiner(isize, 2, None, dropout)
		self.gcomb = GatedCombiner(isize, None, dropout)
		#self.res = Linear(isize, isize, bias=False)

	def forward(self, q, k, select, weight, **kwargs):

		if select == 0:
			return self.comb(q, k) * weight
		elif select == 1:
			return self.gcomb(q, k) * weight
		elif select == 2:
			return (q + k) * weight
		elif select == 3:
			return q.max(k) * weight
		else:
			return q#self.res(q) * weight + q

node_discription = {0: "Feed-Forward Combine", 1:"Gated Combine", 2:"Sum", 3:"Max-over-time Pooling", 4:"Identity"}#Linear

num_node_operation = 5
