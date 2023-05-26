#encoding: utf-8

from torch import nn

from modules.base import CrossAttn
from modules.paradoc import GateResidual
from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Doc.Para.Gate.CADecoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(DecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.enc_gr = GateResidual(isize)
		self.cattn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)
		self.cattn_ln = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.gr = GateResidual(isize)
		self.dec_cattn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop, k_isize=isize * 2)
		self.dec_cattn_ln = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.dec_gr = GateResidual(isize)
		self.layer_normer2, self.drop, self.norm_residual = self.cross_attn.normer, self.cross_attn.drop, self.cross_attn.norm_residual
		self.cross_attn = self.cross_attn.net

	def forward(self, inpute, inputo, inputc, dec_context, src_pad_mask=None, tgt_pad_mask=None, context_mask=None, dec_context_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo, mask=tgt_pad_mask)

		else:
			context, states_return = self.self_attn(query_unit, states=inputo)

		_inputs = self.layer_normer2(context)
		_context = self.cross_attn(_inputs, inpute, mask=src_pad_mask)

		if self.drop is not None:
			_inputs = self.drop(_context)

		context = self.enc_gr(_context, (_inputs if self.norm_residual else context))

		_inputs = self.cattn_ln(context)
		_context = self.cattn(_inputs, inputc, mask=context_mask)
		if self.drop is not None:
			_context = self.drop(_context)
		context = self.gr(_context, (_inputs if self.norm_residual else context))

		_inputs = self.dec_cattn_ln(context)
		_context = self.dec_cattn(_inputs, dec_context, mask=dec_context_mask)
		if self.drop is not None:
			_context = self.drop(_context)
		context = self.dec_gr(_context, (_inputs if self.norm_residual else context))

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, nprev_context=2, num_layer_cross=1, drop_tok=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, nprev_context=nprev_context, num_layer_cross=num_layer_cross, drop_tok=drop_tok, **kwargs)

		self.gdec = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer_cross)])
