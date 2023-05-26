#encoding: utf-8

from torch import nn

from modules.base import Dropout, PositionwiseFF
from modules.hplstm.lm.hfn import HPLSTM
from transformer.Decoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		super(DecoderLayer, self).__init__()

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.net = HPLSTM(isize, num_head=num_head, osize=isize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop)

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

	def forward(self, inputo, query_unit=None, **kwargs):

		context, states_return = self.net(query_unit, states=inputo)

		if self.drop is not None:
			context = self.drop(context)

		context = context + query_unit

		context = self.ff(context)

		return context, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=True, **kwargs)

		self.mask = None

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inputo, states=None, **kwargs):

		out = self.wemb(inputo)

		if self.drop is not None:
			out = self.drop(out)

		_states = {} if states is None else states
		states_return = {}
		for i, net in enumerate(self.nets):
			_state = _states.get(i, "init")
			out, _state = net(_state, query_unit=out)
			states_return[i] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		return out, states_return
