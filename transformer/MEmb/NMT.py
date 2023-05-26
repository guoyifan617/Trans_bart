#encoding: utf-8

from modules.memb import Embedding, Linear
from transformer.Decoder import Decoder
from transformer.MEmb.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output)
		self.num_emb = self.enc.num_emb
		self.isize = isize

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index)
		if global_emb:
			self.dec.wemb = Embedding(tnwd, isize, padding_idx=pad_id)
			self.dec.wemb.weight = emb_w
			if bindDecoderEmb:
				self.dec.classifier = Linear(isize, tnwd)
				self.dec.classifier.weight = self.dec.wemb.weight

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		bsize, seql = inpute.size()
		_mask = _mask.unsqueeze(-1).repeat(1, 1, 1, self.num_emb).view(bsize, 1, seql * self.num_emb)

		return self.dec(self.enc(inpute, _mask), inputo, _mask)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		bsize, seql = inpute.size()
		mask = inpute.eq(pad_id).unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, self.num_emb).view(bsize, 1, seql * self.num_emb)

		_max_len = (seql + max(64, seql // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty)
