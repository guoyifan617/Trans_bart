#encoding: utf-8

from math import sqrt

from transformer.Decoder import Decoder as DecoderBase

class Decoder(DecoderBase):

	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		_pred = self.classifier(out)
		out = self.lsm(_pred)

		if self.training:
			return out, _pred
		else:
			return out
