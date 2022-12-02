#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.AGGNAS import Node, node_discription, num_node_operation
from modules.NAS import GumbleNormDrop
from transformer.Encoder import Encoder as EncoderBase
from utils.torch.comp import torch_no_grad
from utils.train.base import freeze_module, unfreeze_module

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_nod=6, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.nodes = nn.ModuleList([Node(isize, dropout) for i in range(num_nod)])
		self.node_p = nn.Parameter(torch.zeros(num_nod, num_node_operation))
		num_edge = (1 + num_layer * 2 + num_nod) * num_nod // 2
		self.edge_p = nn.Parameter(torch.zeros(2, num_edge))
		self.path_normer = GumbleNormDrop()
		self.node_cost = {0:(isize * isize * 12 + isize * 7) / 1e6, 1:(isize * isize * 12 + isize * 5) / 1e6}
		self.layer_cost = (isize * isize * 12 + isize * 9) / 1e6

		self.tau = 1.0
		self.training_arch = False

	def forward(self, inputs, mask=None, node_mask=None, edge_mask=None, **kwargs):

		bsize, seql = inputs.size()
		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		_num_input_nodes = len(self.nets) + 1
		_node_id = -_num_input_nodes
		nodes_output = {_node_id:out}

		nets = iter(self.nets)
		processed_nodes = set()
		lind = 0
		costs = {}
		_layer_cost_weight = 0.0
		_nw, _nsel = self.path_normer(self.node_p, self.tau, node_mask)
		for _cnode, (node, node_act, act_weight,) in enumerate(zip(self.nodes, _nsel.tolist(), _nw.unbind(0))):

			_nsrc = _cnode + _num_input_nodes
			_snode = _cnode - _nsrc
			rind = lind + _nsrc

			_epq, _epk = self.edge_p.narrow(1, lind, _nsrc).view(2, -1).unbind(0)
			_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)

			_ew, _esel = self.path_normer(_epq, self.tau, _edge_mask)
			_eid = _esel.item() - _num_input_nodes
			if not _eid in processed_nodes:
				processed_nodes.add(_eid)
				if _eid < 0 and _eid > _node_id:
					while _node_id < _eid:
						out = next(nets)(out, mask)
						_node_id += 1
						nodes_output[_node_id] = out
					_layer_cost_weight = _ew
			elif _eid == _node_id:
				_layer_cost_weight = _layer_cost_weight + _ew
			if node_act < num_node_operation - 1:
				_edge_mask_k = torch.arange(_nsrc, dtype=_esel.dtype, device=_esel.device, requires_grad=False).eq(_esel) if _edge_mask is None else (torch.arange(_nsrc, dtype=_esel.dtype, device=_esel.device, requires_grad=False).eq(_esel) + _edge_mask).gt(0)
				_ewk, _eselk = self.path_normer(_epk, self.tau, _edge_mask_k)
				_eidk = _eselk.item() - _num_input_nodes
				if not _eidk in processed_nodes:
					processed_nodes.add(_eidk)
					if _eidk < 0 and _eidk > _node_id:
						while _node_id < _eidk:
							out = next(nets)(out, mask)
							_node_id += 1
							nodes_output[_node_id] = out
						_layer_cost_weight = _ew
				elif _eidk == _node_id:
					_layer_cost_weight = _layer_cost_weight + _ew
				nodes_output[_cnode] = node(_ew * nodes_output[_eid], _ewk * nodes_output[_eidk], node_act, act_weight)
			else:
				nodes_output[_cnode] = node(_ew * nodes_output[_eid], None, node_act, act_weight)
			_cost = self.node_cost.get(node_act, 0.0)
			if _cost > 0.0 and self.training and self.training_arch:
				if _cost in costs:
					costs[_cost] = costs[_cost] + act_weight
				else:
					costs[_cost] = act_weight

			lind = rind

		out = []
		for i in (set(range(len(self.nodes))) - processed_nodes):
			out.append(nodes_output[i])
		out = out[0] if len(out) == 1 else torch.stack(out, dim=-1).sum(-1)

		if self.out_normer is not None:
			out = self.out_normer(out)

		cost_loss = _layer_cost_weight * self.layer_cost
		if self.training and self.training_arch and costs:
			for _cost, _w in costs.items():
				cost_loss = cost_loss + _cost * _w

		if self.training and self.training_arch:
			return out, cost_loss
		else:
			return out

	def get_design(self, node_mask=None, edge_mask=None):

		rs = []
		_num_input_nodes = len(self.nets) + 1

		with torch_no_grad():

			_tmp = self.node_p if node_mask is None else self.node_p.masked_fill(node_mask, -inf_default)
			_nsel = _tmp.argmax(-1)

			lind = 0
			for _cnode, node_act in enumerate(_nsel.tolist()):

				_nsrc = _cnode + _num_input_nodes
				_snode = _cnode - _nsrc
				rind = lind + _nsrc

				_epq, _epk = self.edge_p.narrow(1, lind, _nsrc).view(2, -1).unbind(0)
				_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)
				if _edge_mask is not None:
					_epq = _epq.masked_fill(_edge_mask, -inf_default)
				_esel = _epq.argmax(-1)
				_eid = _esel.item() - _num_input_nodes

				if node_act < num_node_operation - 1:
					_edge_mask_k = torch.arange(_nsrc, dtype=_esel.dtype, device=_esel.device, requires_grad=False).eq(_esel) if _edge_mask is None else (torch.arange(_nsrc, dtype=_esel.dtype, device=_esel.device, requires_grad=False).eq(_esel) + _edge_mask).gt(0)
					_eselk = _epk.masked_fill(_edge_mask_k, -inf_default).argmax(-1)
					_eidk = _eselk.item() - _num_input_nodes
					rs.append("Node %d -%s(node %d, node %d)->:" % (_cnode, node_discription.get(node_act, str(node_act)), _eid, _eidk,))
				else:
					rs.append("Node %d -%s(node %d)->:" % (_cnode, node_discription.get(node_act, str(node_act)), _eid,))

				lind = rind

		return "\n".join(rs)

	def fix_init(self):

		with torch_no_grad():
			self.node_p.zero_()
			self.edge_p.zero_()

	def set_tau(self, value):

		self.tau = value

	def train_arch(self, mode=True):

		freeze_module(self) if mode else unfreeze_module(self)
		self.node_p.requires_grad_(mode)
		self.edge_p.requires_grad_(mode)
		self.training_arch = mode