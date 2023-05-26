#encoding: utf-8

from math import sqrt
from torch import nn

from modules.dropout import PartTokenDropout as TokenDropout
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, **kwargs)

		if dropout > 0.0:
			self.drop = TokenDropout(dropout, inplace=True)

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)

		if self.drop is not None:
			out = self.drop(out)

		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def load_base(self, base_encoder):

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		_nets = list(base_encoder.nets)

		self.nets = nn.ModuleList(_nets[:len(self.nets)] + list(self.nets[len(_nets):]))

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer
