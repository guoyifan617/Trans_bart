#encoding: utf-8

from math import sqrt

from transformer.Encoder import Encoder as EncoderBase

class Encoder(EncoderBase):

	# mask: (2 * bsize, 1, seql) if self.training else (bsize, 1, seql)

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.training:
			out = out.repeat(2, 1, 1)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
