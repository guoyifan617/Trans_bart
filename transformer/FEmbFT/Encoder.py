#encoding: utf-8

from math import sqrt

from modules.elinear import IPLinear
from transformer.Encoder import Encoder as EncoderBase

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, *inputs, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, *inputs, **kwargs)

		self.emb_p = IPLinear(isize, isize, bias=enable_proj_bias_default)

	def forward(self, inputs, mask=None, **kwargs):

		out = self.emb_p(self.wemb(inputs))
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
