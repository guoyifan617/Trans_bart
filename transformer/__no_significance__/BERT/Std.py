#encoding: utf-8

from math import sqrt

from transformer.BERT.Eff import NMT as NMTBase

from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def forward(self, inputs, mask=None, eva_mask=None, emask_p=0.0, **kwargs):

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask

		out = self.wemb(inputs)

		if eva_mask is not None:
			out.masked_fill_(eva_mask.unsqueeze(-1), 0.0)
			if emask_p > 0.0:
				out = out * (1.0 / (1.0 - emask_p))

		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return self.lsm(self.classifier(out))
