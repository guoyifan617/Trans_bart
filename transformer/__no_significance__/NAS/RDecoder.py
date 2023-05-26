#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.NAS import Edge, GumbleNormDrop, LSTMCtr, Node, edge_discription, node_discription
from transformer.Decoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none
from utils.torch.comp import mask_tensor_type, torch_no_grad
from utils.train.base import freeze_module, unfreeze_module

from cnfg.ihyp import *

def interp_edge_sel(edge_sel, snod):

	esel = edge_sel.item()
	sel_edge = esel // 5
	sel_ope = esel % 5
	input_node = -2 if sel_edge == 0 else snod + sel_edge
	# sel_act: operation selected in the edge
	sel_act = edge_sel.new_tensor([sel_ope + 8])
	# sel_add_act: input node of the edge, starts from 0 since 0 is the first index of a matrix
	sel_add_act = edge_sel.new_tensor([input_node + 2])

	return sel_act, sel_add_act, input_node, (sel_edge, sel_ope, input_node,)

class DecoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, designer=None, num_nod=8, max_prev_nodes=5, norm_output=True, base_cost_rate=1.01, **kwargs):

		super(DecoderLayer, self).__init__()

		num_edge = ((3 + num_nod) * num_nod // 2) if num_nod < max_prev_nodes else ((2 + max_prev_nodes) * (max_prev_nodes - 1) // 2 + max_prev_nodes * (num_nod - max_prev_nodes + 1))

		self.nodes = nn.ModuleList([Node(isize, dropout, num_head) for i in range(num_nod)])
		self.edges = nn.ModuleList([Edge(isize, dropout) for i in range(num_edge)])

		self.ctr = designer

		self.register_buffer("q_mask_full", torch.as_tensor([1 if i == 0 else 0 for i in range(self.max_prev_nodes)], dtype=mask_tensor_type, device=self.node_p.device).unsqueeze(-1).expand(-1, 5), persistent=False)

		self.path_normer = GumbleNormDrop()

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

		self.tau = 1.0

		self.max_prev_nodes = max_prev_nodes

		_freq_cost = (isize * 3 + isize * isize) / 1e6
		self.edge_cost = {0:_freq_cost, 1:_freq_cost, 2:_freq_cost, 3: (isize * isize + isize) / 1e6}
		_freq_cost = isize * isize
		self.node_cost = {0:(_freq_cost * 4 + isize * 6) / 1e6, 1:(_freq_cost * 3 + isize * 7) / 1e6, 2:(isize * 6 + 3) / 1e6}
		self.base_cost = (_freq_cost * 16 + isize * 11) / 1e6 * base_cost_rate

		self.training_arch(False)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, edge_mask=None, node_mask=None, **kwargs):

		def select_ope(generator, path_normer, sel=None, sel_node=None, rnn_states=None, num_gen=1, select_node=True, tau=1.0, mask=None):

			_weight, _rnn_states = generator(num_gen, sel, sel_node, rnn_states, select_node)

			ope_w, ope_id = path_normer(_weight, tau, mask)

			return ope_w, ope_id, _rnn_states

		if query_unit is None:
			nodes_output = {-2:inpute, -1:inputo}
			res_add_o = inputo

		else:
			nodes_output = {-2:inpute, -1:query_unit}
			res_add_o = query_unit
			if inputo is not None:
				states_in_k, states_in_v = inputo.unbind(-1)
				states_in_k = states_in_k.unbind(-1)
				states_in_v = states_in_v.unbind(-1)
				states_id = 0
			states_return_k = []
			states_return_v = []

		processed_nodes = set()
		lind = 0
		costs = {}
		#_sel_plist = []
		sel_act = None
		sel_add_act = None
		rnn_states = None

		for _cnode, node in enumerate(self.nodes):

			act_weight, sel_act, rnn_states = select_ope(self.ctr, self.path_normer, sel_act, sel_add_act, rnn_states, 1, True, self.tau, None if node_mask is None else node_mask.select(0, _cnode))
			node_act = sel_act.item()

			# number of previous nodes available [-2:_cnode]
			_nsrc = min(_cnode + 2, self.max_prev_nodes)
			_snode = _cnode - _nsrc + 1
			rind = lind + _nsrc

			_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)

			# all edges from previous nodes to current node
			_edges = self.edges[lind:rind]

			edge_dict = {}
			# select edge for q
			# for multi-head attention in decoder, _q_type must equal to inputo
			# _q_type must equal to inputo, do not processing node -2 (inpute)
			_q_mask = self.q_mask_full.narrow(0, 0, _nsrc)
			_q_mask = _q_mask if _edge_mask is None else (_edge_mask + _q_mask).gt(0)

			_ew, _esel, rnn_states = select_ope(self.ctr, self.path_normer, sel_act, None, rnn_states, _nsrc, False, self.tau, _q_mask)
			# number between [0, 5*_cnode)
			sel_act, sel_add_act, _input_node, _dict_k = interp_edge_sel(_esel, _snode)
			edge_dict[_dict_k] = [("q", _ew,)]

			# select edge for k
			if node_act < 7:

				# for node_act in [1, 6], _k_type must equal to _cur_node_type
				if node_act > 0:
					_edge_mask = _q_mask if _edge_mask is None else (_edge_mask + _q_mask).gt(0)

				_ew, _esel, rnn_states = select_ope(self.ctr, self.path_normer, sel_act, sel_add_act, rnn_states, _nsrc, False, self.tau, _edge_mask)
				sel_act, sel_add_act, _input_node, _dict_k = interp_edge_sel(_esel, _snode)
				_k_type = -2 if _input_node == -2 else -1
				if _dict_k in edge_dict:
					edge_dict[_dict_k].append(("k", _ew,))
				else:
					edge_dict[_dict_k] = [("k", _ew,)]
				if node_act < 5:

					# if _k_type == -2, _v_type = -2
					if node_act == 0 and _k_type == -2:
						_ew, _esel, rnn_states = select_ope(self.ctr, self.path_normer, sel_act, sel_add_act, rnn_states, 1, False, self.tau, None if _edge_mask is None else _edge_mask.narrow(-1, 0, 5))
					else:
						_ew, _esel, rnn_states = select_ope(self.ctr, self.path_normer, sel_act, sel_add_act, rnn_states, _nsrc, False, self.tau, _edge_mask)
					sel_act, sel_add_act, _input_node, _dict_k = interp_edge_sel(_esel, _snode)
					if _dict_k in edge_dict:
						edge_dict[_dict_k].append(("v", _ew,))
					else:
						edge_dict[_dict_k] = [("v", _ew,)]
			else:
				_k_type = None

			edge_rs = {}
			for k, v in edge_dict.items():
				_sel_edge, _sel_ope, _input_node = k
				rsk, _w = zip(*v)
				rsl = _edges[_sel_edge](nodes_output[_input_node], _sel_ope, _w)
				_cost = self.edge_cost.get(_sel_ope, 0.0)
				if _cost > 0.0 and self.training and self.training_arch:
					for _wu in _w:
						if _cost in costs:
							costs[_cost] = costs[_cost] + _wu
						else:
							costs[_cost] = _wu
						#_sel_plist.append(_wu.view(-1))
				processed_nodes.add(_input_node)
				for _k, _rs in zip(rsk, rsl):
					edge_rs[_k] = _rs
			if _k_type is None:
				_node_comp_mask = None
				_node_self_attn = False
			elif _k_type == -1:
				_node_comp_mask = tgt_pad_mask
				_node_self_attn = True
			else:
				_node_comp_mask = src_pad_mask
				_node_self_attn = False
			_n_key = edge_rs.get("k", None)
			_n_value = edge_rs.get("v", None)
			if query_unit is not None and node_act == 0 and _node_self_attn:
				if inputo is None:
					_key_p = None
				else:
					_key_p = states_in_k[states_id]
					_value_p = states_in_v[states_id]
					_n_key = torch.cat((_key_p, _n_key,), 1)
					_n_value = torch.cat((_value_p, _n_value,), 1)
					states_id += 1
				states_return_k.append(_n_key)
				states_return_v.append(_n_value)
			nodes_output[_cnode] = node(edge_rs.get("q", None), _n_key, _n_value, node_act, act_weight, _node_comp_mask)
			_cost = self.node_cost.get(node_act, 0.0)
			if _cost > 0.0 and self.training and self.training_arch:
				if _cost in costs:
					costs[_cost] = costs[_cost] + act_weight
				else:
					costs[_cost] = act_weight
				#_sel_plist.append(act_weight.view(-1))

			lind = rind

		outo = []
		for i in (set(range(len(self.nodes))) - processed_nodes):
			outo.append(nodes_output[i])
		outo = outo[0] if len(outo) == 1 else torch.stack(outo, dim=-1).sum(-1)
		if self.out_normer is not None:
			outo = self.out_normer(outo)

		if query_unit is not None:
			states_return = torch.stack((torch.stack(states_in_k, -1), torch.stack(states_in_v, -1),), -1)

		cost_loss = None
		if self.training and self.training_arch and costs:
			for _cost, _w in costs.items():
				_cost_u = _cost * _w
				cost_loss = _cost_u if cost_loss is None else (cost_loss + _cost_u)
			cost_loss = (cost_loss - self.base_cost).relu()# * torch.cat(_sel_plist, -1).mean()

		if query_unit is None:
			if self.training and self.training_arch:
				return outo, cost_loss
			else:
				return outo
		else:
			return outo, states_return

	def get_design(self, node_mask=None, edge_mask=None):

		def select_ope(generator, sel=None, sel_node=None, rnn_states=None, num_gen=1, select_node=True, mask=None):

			_weight, _rnn_states = generator(num_gen, sel, sel_node, rnn_states, select_node)

			if mask is not None:
				_weight = _weight.masked_fill(mask, -inf_default)

			return _weight.argmax(-1), _rnn_states

		rs = []

		with torch_no_grad():

			_tmp = self.node_p if node_mask is None else self.node_p.masked_fill(node_mask, -inf_default)
			_nsel = _tmp.argmax(-1)

			lind = 0
			sel_act = None
			sel_add_act = None
			rnn_states = None
			for _cnode in range(len(self.nodes)):

				sel_act, rnn_states = select_ope(self.ctr, sel_act, sel_add_act, rnn_states, 1, True, None if node_mask is None else node_mask.select(0, _cnode))
				node_act = sel_act.item()
				rs.append("Node %d -%s->:" % (_cnode, node_discription.get(node_act, str(node_act)),))

				_nsrc = min(_cnode + 2, self.max_prev_nodes)
				_snode = _cnode - _nsrc + 1
				rind = lind + _nsrc

				_edge_mask = None if edge_mask is None else edge_mask.narrow(0, lind, _nsrc).view(-1)

				_q_mask = self.q_mask_full.narrow(0, 0, _nsrc)
				_q_mask = _q_mask if _edge_mask is None else (_edge_mask + _q_mask).gt(0)

				_esel, rnn_states = select_ope(self.ctr, sel_act, None, rnn_states, _nsrc, False, _q_mask)
				sel_act, sel_add_act, input_node, _dict_k = interp_edge_sel(_esel, _snode)
				rs.append("\tq: node %d -%s->" % (input_node, edge_discription.get(_dict_k[1], str(_dict_k[1])),))

				if node_act < 7:

					if node_act > 0:
						_edge_mask = _q_mask if _edge_mask is None else (_edge_mask + _q_mask).gt(0)

					_esel, rnn_states = select_ope(self.ctr, sel_act, sel_add_act, rnn_states, _nsrc, False, _edge_mask)
					sel_act, sel_add_act, input_node, _dict_k = interp_edge_sel(_esel, _snode)
					_k_type = -2 if input_node == -2 else -1
					rs.append("\tk: node %d -%s->" % (input_node, edge_discription.get(_dict_k[1], str(_dict_k[1])),))
					if node_act < 5:

						if node_act == 0 and _k_type == -2:
							_esel, rnn_states = select_ope(self.ctr, sel_act, sel_add_act, rnn_states, 1, False, None if _edge_mask is None else _edge_mask.narrow(-1, 0, 5))
						else:
							_esel, rnn_states = select_ope(self.ctr, sel_act, sel_add_act, rnn_states, _nsrc, False, _edge_mask)
						sel_act, sel_add_act, input_node, _dict_k = interp_edge_sel(_esel, _snode)
						rs.append("\tv: node %d -%s->" % (input_node, edge_discription.get(_dict_k[1], str(_dict_k[1])),))
				lind = rind

		return "\n".join(rs)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, num_nod=8, max_prev_nodes=5, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.controller = LSTMCtr(13, 32, 8, 13, True)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, False, self.controller, num_nod, max_prev_nodes) for i in range(num_layer)])

		self.training_arch = False
		self.train_arch(False)

	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		cost_loss = None
		for net in self.nets:
			if net.training and net.training_arch:
				out, _cost_loss = net(inpute, out, src_pad_mask, _mask)
				if _cost_loss is not None:
					cost_loss = _cost_loss if cost_loss is None else (cost_loss + _cost_loss)
			else:
				out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		if self.training and self.training_arch:
			return out, cost_loss
		else:
			return out

	def get_design(self, node_mask=None, edge_mask=None):

		return self.net[0].get_design(node_mask, edge_mask)

	def train_arch(self, mode=True):

		if mode:
			freeze_module(self)
			unfreeze_module(self.controller)
		else:
			unfreeze_module(self)
			freeze_module(self.controller)
		self.training_arch = mode
		for net in self.nets:
			net.training_arch = mode

	def set_tau(self, value):

		for net in self.nets:
			net.tau = value
