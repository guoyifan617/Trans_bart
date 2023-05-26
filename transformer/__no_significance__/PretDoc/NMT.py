#encoding: utf-8

from torch import nn

from transformer.NMT import NMT as NMTBase
from transformer.PretDoc.Decoder import Decoder
from transformer.PretDoc.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.mono import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, pnwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, num_layer_pret=16, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, pnwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, num_layer_pret)

		emb_w = self.enc.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, inputc, mask=None, context_mask=None, start_sent_id=0, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		_context_mask = inputc.eq(pad_id).unsqueeze(1) if context_mask is None else context_mask
		ence, contexts = self.enc(inpute, inputc, _mask, _context_mask, start_sent_id=start_sent_id)

		return self.dec(ence, inputo, contexts, _mask, None, start_sent_id=start_sent_id)

	def decode(self, inpute, inputc, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)
		context_mask = inputc.eq(pad_id).unsqueeze(1)

		bsize, nsent, seql = inpute.size()
		_max_len = (seql + max(64, seql // 4)) if max_len is None else max_len
		ence, contexts = self.enc(inpute, inputc, mask, context_mask)

		return self.dec.decode(ence, contexts, mask.view(bsize * nsent, 1, seql), None, beam_size, _max_len, length_penalty)

	def load_base(self, base_nmt, base_pret_enc=None):

		self.enc.load_base(base_nmt.enc, base_pret_enc)
		self.dec.load_base(base_nmt.dec)

	def get_loaded_para_models(self):

		return self.enc.get_loaded_para_models() + self.dec.get_loaded_para_models()

	def get_loaded_paras(self):

		return nn.ModuleList(self.get_loaded_para_models()).parameters()
