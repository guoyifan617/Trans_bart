#encoding: utf-8

from transformer.Mono.Decoder import Decoder
from transformer.Mono.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.mono import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, num_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, num_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index, self.enc.lang_emb)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, lang_id=0, psind=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, _mask, lang_id=lang_id), inputo, _mask, lang_id=lang_id, psind=psind)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, lang_id=0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)

		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask, lang_id=lang_id), mask, beam_size, _max_len, length_penalty, lang_id)
