#encoding: utf-8

from math import sqrt
from torch import nn

from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, num_emb=2, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.wemb = nn.Embedding(nwd, num_emb * isize, padding_idx=pad_id)
		self.num_emb = num_emb

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		bsize, seql, _ = out.size()
		isize = _ // self.num_emb
		eseql = seql * self.num_emb
		out = out.view(bsize, eseql, isize)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).repeat(1, 1, self.num_emb).view(1, eseql, isize).add(out, alpha=sqrt(isize))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
