#encoding: utf-8

import torch
from torch import nn
from torch.nn import functional as nnFunc

from modules.base import Dropout, Linear
from modules.group.base import GroupLinear
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class Node(nn.Module):

	def __init__(self, num_input, isize, dropout=0.0, **kwargs):

		super(Node, self).__init__()

		self.net = GroupLinear(isize * num_input, isize * num_input, num_input)
		self.act = nn.ReLU(inplace=True)
		# inplace=True is not secure for self.training == False
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.ni = num_input
		self.weight = nn.Parameter(torch.zeros(self.ni))

	def forward(self, *inputs, **kwargs):

		if self.training:
			out = self.act(self.net(torch.cat(inputs, -1)))

			osize = inputs[0].size()
			_osize = list(osize)
			_osize.insert(-1, self.ni)
			out = out.view(_osize).transpose(-1, -2).contiguous().view(-1, self.ni).mv(self.weight.softmax(-1)).view(osize)
		else:
			_si = self.select()
			isize = inputs[0].size(-1)
			out = self.act(nnFunc.linear(inputs[_si], self.net.net.weight[_si], None if self.net.net.bias is None else self.net.net.bias.narrow(0, isize * _si, isize)))

		if self.drop is not None:
			out = self.drop(out)

		return out

	def fix_init(self):

		with torch_no_grad():
			self.weight.zero_()

	def select(self):

		return self.weight.softmax(-1).argmax().item()

	def tell(self):

		_si = self.select()
		return "Node %d -->" % (_si,)

class Cell(nn.Module):

	def __init__(self, num_node, isize, dropout=0.0, act_drop=None, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(Cell, self).__init__()

		_act_drop = parse_none(act_drop, dropout)
		self.nets = nn.ModuleList([Node(i, isize, dropout=_act_drop) for i in range(1, num_node + 1)])
		self.trans = nn.Sequential(Linear(isize * num_node, isize, bias=enable_bias), Dropout(dropout, inplace=True)) if dropout > 0.0 else Linear(isize * num_node, isize, bias=enable_bias)

	def forward(self, x, **kwargs):

		out = [x]
		for net in self.nets:
			out.append(net(*out))

		return self.trans(torch.cat(out[1:], -1))

	def tell(self):

		rs = []
		for i, net in enumerate(self.nets, 1):
			rs.append("%s Node %d" % (net.tell(), i,))

		return "\n".join(rs)
