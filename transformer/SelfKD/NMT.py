#encoding: utf-8

from transformer.NMT import NMT as NMTBase
from transformer.SelfKD.Decoder import Decoder
from transformer.SelfKD.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, kd_layers=None, enable_proj=True, num_topk=64, T=1.0, min_T=None, min_gold_p=None, mix_kd=True, iter_kd=True, remove_gold=False, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		_kd_layers = [] if kd_layers is None else kd_layers
		if _kd_layers and isinstance(_kd_layers[0], (list, tuple)):
			kd_enc_layers, kd_dec_layers = tuple(set(_) for _ in _kd_layers)
		else:
			kd_enc_layers = kd_dec_layers = set(_kd_layers)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, kd_layers=kd_enc_layers, enable_proj=enable_proj)

		emb_w = self.enc.wemb.weight if global_emb else None
		_num_kd_enc, _num_kd_dec = len(kd_enc_layers), len(kd_dec_layers)
		_num_kd_layers = _num_kd_enc + _num_kd_dec
		if iter_kd:
			_num_kd_layers -= (2 if _num_kd_enc > 0 else 1)
		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, kd_layers=kd_dec_layers, enable_proj=enable_proj, num_topk=num_topk, T=T, min_T=min_T, min_gold_p=min_gold_p, num_mix=_num_kd_layers if mix_kd else 0, iter_kd=iter_kd, remove_gold=remove_gold)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, gold=None, gold_pad_mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		if self.training and (gold is not None):
			ence, enc_kd_hiddens = self.enc(inpute, mask=_mask, gold=gold)
			return self.dec(ence, inputo, src_pad_mask=_mask, gold=gold, enc_kd_hiddens=enc_kd_hiddens, gold_pad_mask=gold_pad_mask)
		else:
			return self.dec(self.enc(inpute, mask=_mask), inputo, src_pad_mask=_mask)
