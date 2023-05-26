#encoding: utf-8

import torch
from torch import nn

from modules.base import ResidueCombiner
from transformer.AGG.ReentDecoder import Decoder as DecoderBase
from transformer.Decoder import DecoderLayer as DecoderLayerUnit
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayerBase(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_sub=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayerBase, self).__init__()

		self.nets = nn.ModuleList([DecoderLayerUnit(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub, _fhsize)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		outs = []
		if query_unit is None:
			out = inputo

			for net in self.nets:
				out = net(inpute, out, src_pad_mask, tgt_pad_mask)
				outs.append(out)
		else:
			out = query_unit
			states_return = []
			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, None if inputo is None else inputo[_tmp], src_pad_mask, tgt_pad_mask, out)
				outs.append(out)
				states_return.append(_state)

		out = self.combiner(*outs)

		if query_unit is None:
			return out
		else:
			return out, states_return

class R2DecoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_sub=1, num_unit=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(R2DecoderLayer, self).__init__()

		self.nets = nn.ModuleList([DecoderLayerBase(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, num_unit) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub, _fhsize)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		outs = []
		if query_unit is None:
			out = inputo

			for net in self.nets:
				out = net(inpute, out, src_pad_mask, tgt_pad_mask)
				outs.append(out)
		else:
			out = query_unit
			states_return = []
			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, None if inputo is None else inputo.select(-2, _tmp), src_pad_mask, tgt_pad_mask, out)
				outs.append(out)
				states_return.append(_state)

			states_return = torch.stack(states_return, -2)

		out = self.combiner(*outs)

		if query_unit is None:
			return out
		else:
			return out, states_return

class DecoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, num_sub3=1, num_sub=1, num_unit=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__()

		self.nets = nn.ModuleList([R2DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, num_sub, num_unit) for i in range(num_sub3)])

		self.combiner = ResidueCombiner(isize, num_sub3, _fhsize)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		outs = []
		if query_unit is None:
			out = inputo

			for net in self.nets:
				out = net(inpute, out, src_pad_mask, tgt_pad_mask)
				outs.append(out)
		else:
			out = query_unit
			states_return = []
			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, None if inputo is None else inputo.select(-2, _tmp), src_pad_mask, tgt_pad_mask, out)
				outs.append(out)
				states_return.append(_state)

			# states_return: (bsize, seql, num_unit, num_sub, isize)
			states_return = torch.stack(states_return, -2)

		out = self.combiner(*outs)

		if query_unit is None:
			return out
		else:
			return out, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=False, bindemb=False, forbidden_index=None, num_sub3=1, num_sub=1, num_unit=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, num_sub3, num_sub, num_unit) for i in range(num_layer)])
