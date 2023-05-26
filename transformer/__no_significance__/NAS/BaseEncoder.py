#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.NAS import Edge, GumbleNormDrop, edge_discription
from modules.base import ResSelfAttn
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad
from utils.train.base import freeze_module, unfreeze_module

from cnfg.ihyp import *

def interp_edge_sel(edge_sel, snod):

	esel = edge_sel.item()
	sel_edge = esel // 5
	sel_ope = esel % 5
	input_node = snod + sel_edge

	return (sel_edge, sel_ope, input_node,)

class EncoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, num_nod=4, max_prev_nodes=4, edge_p=None, norm_output=True, base_cost_rate=2.0, **kwargs):

		super(EncoderLayer, self).__init__()

		_ahsize = parse_none(ahsize, isize)

		num_edge = ((1 + num_nod) * num_nod // 2) if num_nod < (max_prev_nodes + 1) else ((1 + max_prev_nodes) * max_prev_nodes // 2 + max_prev_nodes * (num_nod - max_prev_nodes))
		self.num_nod = num_nod

		self.attn = ResSelfAttn(isize, _ahsize, num_head, dropout=attn_drop, norm_residual=norm_residual)

		self.edges = nn.ModuleList([Edge(isize, dropout) for i in range(num_edge)])
		self.edge_p = nn.Parameter(torch.zeros(num_edge, 5)) if edge_p is None else edge_p

		self.path_normer = GumbleNormDrop()

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

		self.tau = 1.0

		self.max_prev_nodes = max_prev_nodes

		_freq_cost = (isize * 6 + isize * isize * 8) / 1e6
		self.edge_cost = {0:_freq_cost, 1:_freq_cost, 2:_freq_cost, 3: (isize * isize + isize) / 1e6}
		_freq_cost = isize * isize
		self.base_cost = (_freq_cost * 8 + isize * 8) / 1e6 * base_cost_rate

		self.training_arch = False

	def forward(self, inputs, mask=None, edge_mask=None, **kwargs):

		context = self.attn(inputs, mask=mask)

		nodes_output = {-1:context}

		processed_nodes = set()
		lind = 0
		costs = {}
		#_sel_plist = []

		for _cnode in range(self.num_nod):

			_nsrc = min(_cnode + 1, self.max_prev_nodes)
			_snode = _cnode - _nsrc
			rind = lind + _nsrc

			_epq = self.edge_p.narrow(0, lind, _nsrc).contiguous().view(-1)
			_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)

			_edges = self.edges[lind:rind]

			edge_dict = {}
			_ew, _esel = self.path_normer(_epq, self.tau, _edge_mask)
			_sel_edge, _sel_ope, _input_node = interp_edge_sel(_esel, _snode)

			rsl = _edges[_sel_edge](nodes_output[_input_node], _sel_ope, [_ew])
			nodes_output[_cnode] = rsl[0]
			_cost = self.edge_cost.get(_sel_ope, 0.0)
			if _cost > 0.0 and self.training and self.training_arch:
				if _cost in costs:
					costs[_cost] = costs[_cost] + _ew
				else:
					costs[_cost] = _ew
				#_sel_plist.append(_ew.view(-1))
			processed_nodes.add(_input_node)

			lind = rind

		out = []
		for i in (set(range(self.num_nod)) - processed_nodes):
			out.append(nodes_output[i])
		out = out[0] if len(out) == 1 else torch.stack(out, dim=-1).sum(-1)
		if self.out_normer is not None:
			out = self.out_normer(out)

		cost_loss = None
		if self.training and self.training_arch and costs:
			for _cost, _w in costs.items():
				_cost_u = _cost * _w
				cost_loss = _cost_u if cost_loss is None else (cost_loss + _cost_u)
			cost_loss = (cost_loss - self.base_cost).relu()# * torch.cat(_sel_plist, -1).mean()

		if self.training and self.training_arch:
			return out, cost_loss
		else:
			return out

	def get_design(self, edge_mask=None):

		rs = []

		with torch_no_grad():

			lind = 0
			for _cnode in range(self.num_nod):

				rs.append("Node %d:" % (_cnode,))

				_nsrc = min(_cnode + 1, self.max_prev_nodes)
				_snode = _cnode - _nsrc
				rind = lind + _nsrc

				_epq = self.edge_p.narrow(0, lind, _nsrc).contiguous().view(-1)
				_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)

				if _edge_mask is not None:
					_epq = _epq.masked_fill(_edge_mask, -inf_default)
				_esel = _epq.argmax(-1)
				_, sel_ope, input_node = interp_edge_sel(_esel, _snode)
				rs.append("\tnode %d -%s->" % (input_node, edge_discription.get(sel_ope, str(sel_ope)),))

				lind = rind

		return "\n".join(rs)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_nod=4, max_prev_nodes=4, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		num_edge = ((1 + num_nod) * num_nod // 2) if num_nod < (max_prev_nodes + 1) else ((1 + max_prev_nodes) * max_prev_nodes // 2 + max_prev_nodes * (num_nod - max_prev_nodes))
		self.edge_p = nn.Parameter(torch.zeros(num_edge, 5))

		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, False, num_nod, max_prev_nodes, self.edge_p) for i in range(num_layer)])

		self.training_arch = False
		self.train_arch(False)

	def forward(self, inputs, mask=None, **kwargs):

		bsize, seql = inputs.size()
		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		cost_loss = None
		for id, net in enumerate(self.nets):
			if net.training and net.training_arch:
				out, _cost_loss = net(out, mask)
				if _cost_loss is not None:
					cost_loss = _cost_loss if cost_loss is None else (cost_loss + _cost_loss)
			else:
				out = net(out, mask)

		out = out if self.out_normer is None else self.out_normer(out)

		if self.training and self.training_arch:
			return out, cost_loss
		else:
			return out

	def get_design(self, edge_mask=None):

		return self.nets[0].get_design(edge_mask)

	def train_arch(self, mode=True):

		freeze_module(self) if mode else unfreeze_module(self)
		self.edge_p.requires_grad_(mode)
		self.training_arch = mode
		for net in self.nets:
			net.training_arch = mode

	def set_tau(self, value):

		for net in self.nets:
			net.tau = value

	def fix_init(self):

		with torch_no_grad():
			self.edge_p.zero_()
