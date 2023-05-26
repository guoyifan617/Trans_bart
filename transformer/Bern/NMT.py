#encoding: utf-8

from modules.prune.bern import BernoulliParameter
from transformer.Bern.Decoder import Decoder
from transformer.Bern.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple

from cnfg.ihyp import *

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

	def resetBern(self):

		for _m in self.modules():
			if isinstance(_m, BernoulliParameter):
				_m.reset()

	def bernMaskParameters(self):

		for _m in self.modules():
			if isinstance(_m, BernoulliParameter):
				yield _m.maskp

	def useBernMask(self, value):

		for _m in self.modules():
			if isinstance(_m, BernoulliParameter):
				_m.use_mask(value)

	def train_parameters(self, value):

		for _m in self.modules():
			if isinstance(_m, BernoulliParameter):
				_m.data.requires_grad_(value)
