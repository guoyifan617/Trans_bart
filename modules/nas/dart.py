#encoding: utf-8

import torch
from torch import nn
from torch.nn import functional as nnFunc

from modules.act import GELU, Swish
from modules.base import Dropout, Linear
from modules.sampler import EffSamplerFunc, SamplerFunc
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class Edge(nn.Module):

	def __init__(self, isize, **kwargs):

		super(Edge, self).__init__()

		self.acts = nn.ModuleList([nn.Sigmoid(), nn.Tanh(), nn.ReLU(inplace=True), GELU(), Swish()])
		self.discription = {0: "Zero", 1:"Identity", 2:"Sigmoid", 3:"Tanh", 4: "ReLU", 5:"GeLU", 6:"Swish"}

		self.trans = Linear(isize, isize * len(self.acts))

		self.isize = isize

	def forward(self, x, selnet=None, **kwargs):

		if selnet is None:
			isize = list(x.size())
			isize.append(len(self.acts))
			out = list(self.trans(x).view(isize).unbind(-1))
			# prevent GeLU and Swish from breaking
			out[-1], out[-2] = out[-1].contiguous(), out[-2].contiguous()

			return [x] + [act(outu) for outu, act in zip(out, self.acts)]
		else:
			_sind = selnet * self.isize

			return self.acts[selnet](nnFunc.linear(x, self.trans.weight.narrow(0, _sind, self.isize), self.trans.bias.narrow(0, _sind, self.isize)))

	def num_ops(self):

		return len(self.acts) + 2

class Node(nn.Module):

	def __init__(self, num_input, isize, dropout=0.0, **kwargs):

		super(Node, self).__init__()

		self.nets = nn.ModuleList([Edge(isize) for i in range(num_input)])
		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None
		self.ni, self.nop = num_input, self.nets[0].num_ops()
		self.weight = nn.Parameter(torch.zeros(self.ni * self.nop))

	def forward(self, *inputs, **kwargs):

		if self.training:
			out = []
			for net, x in zip(self.nets, inputs):
				out.extend(net(x))

			osize = out[0].size()
			out = torch.stack(out, -1)
			out = out.view(-1, out.size(-1)).mv(self.weight.softmax(-1).view(self.ni, self.nop).narrow(1, 1, self.nop - 1).contiguous().view(-1)).view(osize)
		else:
			_si, _sop = self.select()
			if _sop == 0:
				_x = inputs[0]
				out = _x.new_zeros(_x.size(), dtype=_x.dtype, device=_x.device)
			elif _sop == 1:
				out = inputs[_si]
			else:
				out = self.nets[_si](inputs[_si], selnet=_sop - 2)

		if self.drop is not None:
			out = self.drop(out)

		return out

	def fix_init(self):

		with torch_no_grad():
			self.weight.zero_()

	def select(self):

		_w = self.weight.softmax(-1).argmax().item()
		si, sop = _w // self.nop, _w % self.nop

		return si, sop

	def tell(self):

		_si, _sop = self.select()
		return "Node %d -- %s ->" % (_si, self.nets[_si].discription[_sop],)

# sampling version of Node
class SNode(Node):

	def forward(self, *inputs, **kwargs):

		if self.training:
			out = []
			for net, x in zip(self.nets, inputs):
				out.extend(net(x))

			osize = out[0].size()
			bsize = osize[0]
			out = torch.stack(out, -1)
			out = out.view(bsize, -1, out.size(-1)).bmm(SamplerFunc(self.weight.softmax(-1), -1, bsize).view(bsize, self.ni, self.nop).narrow(-1, 1, self.nop - 1).contiguous().view(bsize, -1, 1)).view(osize)
		else:
			_si, _sop = self.select()
			if _sop == 0:
				_x = inputs[0]
				out = _x.new_zeros(_x.size(), dtype=_x.dtype, device=_x.device)
			elif _sop == 1:
				out = inputs[_si]
			else:
				out = self.nets[_si](inputs[_si], selnet=_sop - 2)

		if self.drop is not None:
			out = self.drop(out)

		return out

# Efficient implementation of SNode
class ESNode(SNode):

	def __init__(self, num_input, isize, dropout=0.0, **kwargs):

		super(ESNode, self).__init__(num_input, isize, dropout=dropout)

		# remove inplace from dropout which will overide the registration of backward call to torch.autograd.Function
		if dropout > 0.0:
			self.drop = Dropout(dropout)

	def forward(self, *inputs, **kwargs):

		if self.training:
			zero_input = inputs[0].new_zeros(inputs[0].size())
			out = []
			for net, x in zip(self.nets, inputs):
				out.append(zero_input)
				out.extend(net(x))

			bsize = out[0].size(0)
			out = EffSamplerFunc(torch.stack(out, 1), self.weight.softmax(-1), -1, True)
		else:
			_si, _sop = self.select()
			if _sop == 0:
				_x = inputs[0]
				out = _x.new_zeros(_x.size(), dtype=_x.dtype, device=_x.device)
			elif _sop == 1:
				out = inputs[_si]
			else:
				out = self.nets[_si](inputs[_si], selnet=_sop - 2)

		if self.drop is not None:
			out = self.drop(out)

		return out

class Cell(nn.Module):

	def __init__(self, num_node, isize, dropout=0.0, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(Cell, self).__init__()

		self.nets = nn.ModuleList([Node(i, isize, dropout=dropout) for i in range(1, num_node + 1)])
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
