#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from transformer.Encoder import Encoder as EncoderBase

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, **kwargs)

		self.lang_emb = nn.Parameter(torch.Tensor(2, isize).uniform_(- sqrt(2.0 / (isize + 2)), sqrt(2.0 / (isize + 2))))

	def forward(self, inputs, mask=None, lang_id=0, **kwargs):

		out = self.wemb(inputs) + self.lang_emb[lang_id]
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
