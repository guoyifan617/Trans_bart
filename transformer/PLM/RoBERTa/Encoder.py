#encoding: utf-8

from transformer.PLM.BERT.Encoder import Encoder as EncoderBase

from cnfg.plm.roberta.base import num_type
from cnfg.plm.roberta.ihyp import *
from cnfg.vocab.plm.roberta import pad_id, pemb_start_ind

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, num_type=num_type, share_layer=False, model_name="roberta", **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, num_type=num_type, share_layer=share_layer, model_name=model_name, **kwargs)

		self.wemb.padding_idx = pad_id

	def forward(self, inputs, token_types=None, mask=None, **kwargs):

		seql = inputs.size(1)
		out = None if self.pemb is None else self.pemb.narrow(0, pemb_start_ind, seql)
		if self.temb is not None:
			_ = self.temb.weight[0] if token_types is None else self.temb(token_types)
			out = _ if out is None else (out + _)
		_ = self.wemb(inputs)
		out = _ if out is None else (out + _)
		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask
		for net in self.nets:
			out = net(out, _mask)

		return out
