#encoding: utf-8

from math import sqrt

from transformer.BERT.Eff import NMT as NMTBase

class NMT(NMTBase):

	def forward(self, inputs, mask=None, eva_mask=None, emask_p=0.0, **kwargs):

		_mask = inputs.eq(0).unsqueeze(1) if mask is None else mask

		out = self.wemb(inputs)
		out = out * sqrt(out.size(-1))

		if eva_mask is not None:
			out.masked_fill_(eva_mask.unsqueeze(-1), 0.0)
			if emask_p > 0.0:
				out = out * (1.0 / (1.0 - emask_p))

		if self.pemb is not None:
			out = out + self.pemb(inputs, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return self.lsm(self.classifier(out))