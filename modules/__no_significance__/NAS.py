#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.act import GELU
from modules.base import Dropout, Linear, MultiHeadAttn
from modules.rnncells import LSTMCell4RNMT

from cnfg.ihyp import *

class LNMHAttn(nn.Module):

	def __init__(self, isize, num_head=8, dropout=0.0, **kwargs):

		super(LNMHAttn, self).__init__()

		self.normer = nn.LayerNorm((3, isize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.net = MultiHeadAttn(isize, isize, isize, num_head, dropout)

	def forward(self, iQ, iK, iV, mask=None, **kwargs):

		q, k, v = self.normer(torch.stack((iQ, iK, iV,), -2)).unbind(-2)

		return self.net(q, k, v, mask)

class LNC(nn.Module):

	# isize: input dimension

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(LNC, self).__init__()

		self.net = nn.Sequential(nn.Linear(isize * 3, isize), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Linear(isize * 3, isize)

		self.normer = nn.LayerNorm((3, isize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, q, k, v, mask=None, **kwargs):

		_size = list(q.size())
		_size[-1] = -1

		return self.net(self.normer(torch.stack([q, k, v], dim=-2)).view(_size))

class LNWC(nn.Module):

	# isize: input dimension

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(LNWC, self).__init__()

		self.normer = nn.LayerNorm((3, isize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.w = nn.Parameter(torch.Tensor(3).uniform_(- sqrt(1.0 / 3), sqrt(1.0 / 3)))

	def forward(self, q, k, v, mask=None, **kwargs):

		out = self.normer(torch.stack([q, k, v], dim=-2)).transpose(-1, -2).contiguous().view(-1, 3).mv(self.w).view(q.size())

		return out if self.drop is None else self.drop(out)

class PathDrop(nn.Module):

	def forward(self, input, mask=None, **kwargs):

		_tmp = (input.new_empty(input.size()).uniform_(0.0, 1.0) + input) if self.training else input
		if mask is not None:
			_tmp = _tmp.masked_fill(mask, -inf_default)

		_ind = _tmp.argmax(-1)

		if input.dim() == 1:
			return input[_ind], _ind
		else:
			bsize, isize = input.size()
			_ind_input = torch.arange(0, bsize * isize, isize, dtype=_ind.dtype, device=_ind.device) + _ind
			return input.view(-1).index_select(0, _ind_input), _ind

class GumbleNorm(nn.Module):

	def forward(self, x, tau=1.0, mask=None, **kwargs):

		out = (x - (-x.new_empty(x.size()).uniform_(0.0, 1.0).log2()).log2()) if self.training else x
		if mask is not None:
			out = out.masked_fill(mask, -inf_default)

		return (out / tau).softmax(dim=-1)

class GumbleNormDrop(nn.Module):

	def forward(self, x, tau=1.0, mask=None, **kwargs):

		out = (x - (-x.new_empty(x.size()).uniform_(0.0, 1.0).log2()).log2()) if self.training else x
		if mask is not None:
			out = out.masked_fill(mask, -inf_default)

		out = (out / tau).softmax(dim=-1)

		_ind = out.argmax(-1)

		if x.dim() == 1:
			return out[_ind], _ind
		else:
			bsize, isize = out.size()
			_ind_input = torch.arange(0, bsize * isize, isize, dtype=_ind.dtype, device=_ind.device) + _ind
			return out.view(-1).index_select(0, _ind_input), _ind

class LSTMCtr(nn.Module):

	def __init__(self, ntok, hsize, num_node_type, max_nodes=16, bindemb=True, **kwargs):

		super(LSTMCtr, self).__init__()

		self.emb = nn.Embedding(ntok, hsize)
		self.node_emb = nn.Embedding(max_nodes, hsize)
		self.sos = nn.Parameter(torch.zeros(1, hsize))
		self.sos_node = nn.Parameter(torch.zeros(1, hsize))

		self.net = LSTMCell4RNMT(hsize, hsize)
		self.init_hx = nn.Parameter(torch.zeros(1, hsize))
		self.init_cx = nn.Parameter(torch.zeros(1, hsize))

		self.classifier = nn.Linear(hsize, ntok)
		if bindemb:
			self.classifier.weight = self.emb.weight

		self.nnt = num_node_type

	def forward(self, nstep=1, sel=None, sel_node=None, states=None, select_node=True, **kwargs):

		if states is None:
			iemb = self.sos
			_state = (self.init_hx, self.init_cx,)
		else:
			iemb = self.emb(sel) + (self.sos_node if sel_node is None else self.node_emb(sel_node))
			_state = states

		if nstep > 1:
			hxl = []
			for i in range(nstep):
				hx, cx = self.net(iemb, _state)
				_state = (hx, cx,)
				hxl.append(hx)
			weight_distribution = self.classifier(torch.stack(hxl, -2))
			_tsize = list(hx.size())
			_tsize[-1] = -1
			wd = weight_distribution.narrow(-1, 0, self.nnt).contiguous().view(_tsize) if select_node else weight_distribution.narrow(-1, self.nnt, weight_distribution.size(-1) - self.nnt).contiguous().view(_tsize)
		else:
			hx, cx = self.net(iemb, _state)
			_state = (hx, cx,)
			weight_distribution = self.classifier(hx)
			wd = weight_distribution.narrow(-1, 0, self.nnt) if select_node else weight_distribution.narrow(-1, self.nnt, weight_distribution.size(-1) - self.nnt)

		return wd, _state

class Edge(nn.Module):

	# isize: input dimension

	def __init__(self, isize, dropout=0.0, **kwargs):

		super(Edge, self).__init__()

		_fhsize = isize * 4
		self.nets = nn.ModuleList([nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), nn.Sigmoid(), Dropout(dropout), Linear(_fhsize, isize), nn.Sigmoid()), nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), nn.ReLU(inplace=True), Dropout(dropout), Linear(_fhsize, isize)), nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), GELU(), Dropout(dropout, inplace=True), Linear(_fhsize, isize)), Linear(isize, isize, bias=False)]) if dropout > 0.0 else nn.ModuleList([nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), nn.Sigmoid(), Linear(_fhsize, isize), nn.Sigmoid()), nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), nn.ReLU(inplace=True), Linear(_fhsize, isize)), nn.Sequential(nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, _fhsize), GELU(), Linear(_fhsize, isize)), Linear(isize, isize, bias=False)])

		self.drop = Dropout(dropout) if dropout > 0.0 else None

	def forward(self, x, select, weights, **kwargs):

		out = self.nets[select](x)
		if select > 0:
			return [out * weight + x for weight in weights] if self.drop is None else [self.drop(out) * weight + x for weight in weights]
		else:
			return [out * weight for weight in weights]

class Node(nn.Module):

	# isize: input dimension

	def __init__(self, isize, dropout=0.0, nheads=None, **kwargs):

		super(Node, self).__init__()

		_nhead = (isize // 64) if nheads is None else nheads

		self.nets = nn.ModuleList([LNC(isize, dropout), LNWC(isize, dropout)])

		self.res = Linear(isize, isize, bias=False)

	def forward(self, q, k, v, select, weight, mask=None, **kwargs):

		if select == 2:
			return (q + k + v) * weight
		elif select == 3:
			return torch.stack([q, k, v], dim=-1).max(-1)[0] * weight
		elif select == 4:
			return (q - k) * weight
		elif select == 5:
			return q * k * weight
		elif select > 5:
			return self.res(q) * weight + q
		else:
			out = self.nets[select](q, k, v, mask) * weight
			if select < 1:
				return out + q
			else:
				return out

node_discription = {0: "Linear Combine", 1:"Weighted Combine", 2:"Element-wise Sum", 3:"Max-over-time Pooling", 4:"q - k", 5:"q * k", 6:"Linear Identity"}
edge_discription = {0:"Sigmoid", 1: "ReLU", 2:"GeLU", 3:"Linear Identity"}
