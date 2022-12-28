#encoding: utf-8

from math import sqrt
from torch import nn

from modules.base import ResidueCombiner
from transformer.AGG.HierEncoder import Encoder as EncoderBase
from transformer.Encoder import EncoderLayer as EncoderLayerUnit

from cnfg.ihyp import *

class EncoderLayerBase(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, num_sub=1, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayerBase, self).__init__()

		self.nets = nn.ModuleList([EncoderLayerUnit(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub, _fhsize)

	def forward(self, inputs, mask=None, **kwargs):

		out = inputs
		outs = []
		for net in self.nets:
			out = net(out, mask)
			outs.append(out)

		return self.combiner(*outs)

class EncoderLayer(nn.Module):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, num_sub=1, num_unit=1, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__()

		self.nets = nn.ModuleList([EncoderLayerBase(isize, _fhsize, dropout, attn_drop, num_head, _ahsize, num_unit) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub, _fhsize)

	def forward(self, inputs, mask=None, **kwargs):

		out = inputs
		outs = []
		for net in self.nets:
			out = net(out, mask)
			outs.append(out)

		return self.combiner(*outs)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=False, num_sub=1, num_unit=1, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize, num_sub, num_unit) for i in range(num_layer)])
		self.combiner = ResidueCombiner(isize, num_layer, _fhsize)

	def forward(self, inputs, mask=None, **kwargs):

		bsize, seql = inputs.size()
		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		outs = []
		for net in self.nets:
			out = net(out, mask)
			outs.append(out)
		out = self.combiner(*outs)

		return out if self.out_normer is None else self.out_normer(out)
