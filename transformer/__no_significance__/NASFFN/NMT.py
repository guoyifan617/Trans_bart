#encoding: utf-8

from transformer.NASFFN.Decoder import Decoder
from transformer.NASFFN.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.nasffn import is_nas, share_cell
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, share_design=True, global_share=True, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

		if share_design:
			share_cell(self, share_all=False, share_global=global_share)
		self.nas_enc, self.nas_dec = is_nas(self.enc), is_nas(self.dec)
		self.share_design, self.global_share = share_design, global_share

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(*self.enc(inpute, _mask), inputo, _mask)

	def tell(self):

		rs = []
		if self.nas_enc:
			rs.append("NMT:" if self.global_share else "Encoder:")
			if self.share_design:
				rs.append(self.enc.nets[0].ff.net.tell())
			else:
				for net in self.enc.nets:
					rs.append(net.ff.net.tell())
		if (not self.global_share) and self.nas_dec:
			rs.append("Decoder:")
			if self.share_design:
				rs.append(self.dec.nets[0].ff.net.tell())
			else:
				for net in self.dec.nets:
					rs.append(net.ff.net.tell())

		return "\n".join(rs)
