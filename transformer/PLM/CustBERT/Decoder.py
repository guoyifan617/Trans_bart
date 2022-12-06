#encoding: utf-8

from transformer.PLM.BERT.Decoder import Decoder as DecoderBase

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer=None, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, model_name="bert", **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer=num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, model_name=model_name, **kwargs)

		self.rel_classifier = self.pooler = None

	def forward(self, inpute, mlm_mask=None, word_prediction=False, **kwargs):

		out = self.ff(inpute)
		if mlm_mask is not None:
			out = out[mlm_mask.unsqueeze(-1).expand_as(out)].view(-1, out.size(-1))
		if word_prediction:
			out = self.lsm(self.classifier(out))

		return out
