#encoding: utf-8

from transformer.AGGS.Decoder import Decoder
from transformer.AGGS.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		if self.enc.training and self.enc.training_arch:
			ence, _cost_enc = self.enc(inpute, _mask)
			deco, _cost_dec = self.dec(ence, inputo, _mask)
			if _cost_enc is None:
				_cost = _cost_dec
			else:
				if _cost_dec is None:
					_cost = _cost_enc
				else:
					_cost = _cost_enc + _cost_dec
			return deco, _cost
		else:
			return self.dec(self.enc(inpute, _mask), inputo, _mask)

	def get_design(self, node_mask=None, edge_mask=None):

		return "\n".join(("Encoder AGG:", self.enc.get_design(node_mask, edge_mask), "\nDecoder AGG:", self.dec.get_design(node_mask, edge_mask),))

	def train_arch(self, mode=True):

		self.enc.train_arch(mode)
		self.dec.train_arch(mode)

	def set_tau(self, value):

		self.enc.set_tau(value)
		self.dec.set_tau(value)
