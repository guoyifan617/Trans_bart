#encoding: utf-8

from modules.kd.ic.p import TopCache as Cache
from transformer.ICKD.Decoder import Decoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, num_topk=None, T=1.0, warm_cache_steps=None, min_gold_p=None, mavg_beta=None, warm_mvavg_steps=None, num_cache_topk=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, **kwargs)

		emb_w = self.enc.wemb.weight if global_emb else None
		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.cache = Cache(tnwd, num_topk, T=T, warm_cache_steps=warm_cache_steps, min_gold_p=min_gold_p, mavg_beta=mavg_beta, warm_mvavg_steps=warm_mvavg_steps, num_cache_topk=(num_topk + num_topk) if num_cache_topk is None else num_cache_topk)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, gold=None, gold_pad_mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		_ = self.enc(inpute, _mask)

		if self.training:
			out, _pred = self.dec(_, inputo, _mask)
			if gold is None:
				return out
			else:
				return out, self.cache(_pred, gold=gold, gold_pad_mask=gold_pad_mask)
		else:
			return self.dec(_, inputo, _mask)
