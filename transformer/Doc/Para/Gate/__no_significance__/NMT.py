#encoding: utf-8

from torch import nn

from transformer.Doc.Para.Gate.Decoder import Decoder
from transformer.Doc.Para.Gate.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(nn.Module):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, nprev_context=2, num_layer_context=1, **kwargs):

		super(NMT, self).__init__()

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, nprev_context, num_layer_context)

		emb_w = self.enc.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index, num_layer_context)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		enc_out, enc_context, context, _mask, context_mask = self.enc(inpute, _mask)

		return self.dec(enc_out, inputo, enc_context, context, _mask, context_mask)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)

		bsize, nsent, seql = inpute.size()
		_max_len = (seql + max(64, seql // 4)) if max_len is None else max_len
		enc_out, enc_context, context, mask, context_mask = self.enc(inpute, mask)

		return self.dec.decode(enc_out, enc_context, context, mask, context_mask, beam_size, _max_len, length_penalty)

	def load_base(self, base_nmt):

		self.enc.load_base(base_nmt.enc)
		self.dec.load_base(base_nmt.dec)

	def get_loaded_para_models(self):

		return self.enc.get_loaded_para_models() + self.dec.get_loaded_para_models()

	def get_loaded_paras(self):

		return nn.ModuleList(self.get_loaded_para_models()).parameters()

	def get_loaded_train_paras(self):

		return [tmpu for tmpu in self.get_loaded_paras() if tmpu.requires_grad]
