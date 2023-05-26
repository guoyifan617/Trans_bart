#encoding: utf-8

import torch
from torch import nn

from modules.base import Scorer
from transformer.Encoder import Encoder
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(nn.Module):

	# isize: size of word embedding
	# snwd: number of words for Encoder
	# tnwd: number of words for Decoder
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# global_emb: Sharing the embedding between encoder and decoder, which means you should have a same vocabulary for source and target language
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, clip_value=None, **kwargs):

		super(NMT, self).__init__()

		self.enc1 = Encoder(isize, snwd, num_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		self.enc2 = Encoder(isize, tnwd, num_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		if global_emb:
			self.enc2.wemb.weight = self.enc1.wemb.weight

		self.scorer = Scorer(isize * 2, False)

	# inpute: source sentences from encoder (bsize, seql)
	# inputo: decoded translation (bsize, nquery)
	# mask: user specified mask, otherwise it will be:
	#	inpute.eq(pad_id).unsqueeze(1)

	def forward(self, inpute, inputo, mask1=None, mask2=None, **kwargs):

		_mask1 = inpute.eq(pad_id) if mask1 is None else mask1
		_mask2 = inpute.eq(pad_id) if mask2 is None else mask2

		enc1, enc2 = self.enc1(inpute, _mask1.unsqueeze(1)), self.enc2(inpute, _mask2.unsqueeze(1))

		_mask1, _mask2 = _mask1.unsqueeze(-1), _mask2.unsqueeze(-1)

		out = self.scorer(torch.cat([enc1.masked_fill(_mask1, 0.0).sum(1) / (enc1.size(1) - _mask1.sum(1)).to(enc1, non_blocking=True), enc1.masked_fill(_mask1, -inf_default).max(1)[0], enc2.masked_fill(_mask2, 0.0).sum(1) / (enc2.size(1) - _mask2.sum(1)).to(enc2, non_blocking=True), enc2.masked_fill(_mask2, -inf_default).max(1)[0]], -1))

		return out

	def fix_update(self):

		if self.clip_value is not None:
			with torch_no_grad():
				for para in self.parameters():
					para.data.clamp_(- self.clip_value, self.clip_value)

	def fix_init(self):

		self.fix_update()
