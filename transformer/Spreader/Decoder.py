#encoding: utf-8

from torch import nn

from modules.spreader.neural.attn import Spreader
from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.HPLSTM.FNDecoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none
from utils.math import exp_grow as grow_func

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, start=2, end=8, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, **kwargs)

		self.self_attn = Spreader(isize, hsize=_ahsize, start=start, end=end, dropout=attn_drop, norm_residual=norm_residual)
		self.ff = None

	def forward(self, inpute, inputo, src_pad_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo)
		else:
			context, states_return = self.self_attn(query_unit, states=inputo)

		context = self.cross_attn(context, inpute, mask=src_pad_mask)

		if query_unit is None:
			return context
		else:
			return context, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, s_start=2, s_end=8, e_start=4, e_end=16, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=True, **kwargs)

		self.mask = None

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, start=s_start, end=s_end)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, start=_s_start, end=_s_end) for (_s_start, _s_end,) in zip(grow_func(s_start, e_start, num_layer), grow_func(s_end, e_end, num_layer))])
