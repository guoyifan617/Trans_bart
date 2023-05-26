#encoding: utf-8

from torch import nn

from modules.paradoc import RSelfGate
from transformer.Doc.Para.Gate.BaseCAEncoder import EncoderLayer as EncoderLayerBase
from transformer.Doc.Para.Gate.CAEncoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(EncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.gr = RSelfGate(isize)

	def forward(self, inputs, inputc, mask=None, context_mask=None, **kwargs):

		_inputs = self.layer_normer(inputs)
		context = self.attn(_inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		context = context + (_inputs if self.norm_residual else inputs)

		_inputs = self.cattn_ln(context)
		_context = self.cattn(_inputs, inputc, mask=context_mask)
		if self.drop is not None:
			_context = self.drop(_context)
		context = self.gr(context, _context)# (_inputs if self.norm_residual else context)

		context = self.ff(context)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nprev_context=2, num_layer_cross=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, nprev_context=nprev_context, num_layer_cross=num_layer_cross, **kwargs)

		self.context_genc = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer_cross)])
