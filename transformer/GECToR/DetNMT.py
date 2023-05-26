#encoding: utf-8

from transformer.GECToR.DetDecoder import Decoder
from transformer.PLM.CustBERT.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple, parse_none
from utils.plm.base import set_ln_ieps
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.plm.custbert import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, model_name="bert", **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)
		enc_model_name, dec_model_name = parse_double_value_tuple(model_name)

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, model_name=model_name, **kwargs)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, model_name=dec_model_name)

		set_ln_ieps(self, ieps_ln_default)
		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, mask=None, mlm_mask=None, word_prediction=True, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, mask=_mask), mlm_mask=mlm_mask, word_prediction=word_prediction)

	def build_task_model(self, *args, **kwargs):

		if hasattr(self.enc, "build_task_model"):
			self.enc.build_task_model(*args, **kwargs)
		if hasattr(self.dec, "build_task_model"):
			self.dec.build_task_model(*args, **kwargs)
