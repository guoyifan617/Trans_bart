#encoding: utf-8

from transformer.NMT import NMT as NMTBase
from transformer.Spreader.FNDecoder import Decoder
from transformer.Spreader.FNEncoder import Encoder
from utils.fmt.parser import parse_double_value_tuple, parse_none
from utils.spreader import share_spreader_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, s_start=2, s_end=8, e_start=4, e_end=16, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize
		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, s_start=s_start, s_end=s_end, e_start=e_start, e_end=e_end)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, s_start=s_start, s_end=s_end, e_start=e_start, e_end=e_end)

		share_spreader_cache(self)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id) if mask is None else mask

		return self.dec(self.enc(inpute, _mask.unsqueeze(-1)), inputo, _mask.unsqueeze(1))

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		mask = inpute.eq(pad_id)

		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask.unsqueeze(-1)), mask.unsqueeze(1), beam_size, _max_len, length_penalty)
