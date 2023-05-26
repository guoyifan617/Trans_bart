#encoding: utf-8

import torch
from torch import nn

from modules.prune.bern import Embedding, LinearBn, PositionwiseFF, ResCrossAttn, ResSelfAttn#, LayerNorm, Linear
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.self_attn = ResSelfAttn(isize, _ahsize, num_head, dropout=attn_drop, norm_residual=self.self_attn.norm_residual)
		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head, dropout=attn_drop, norm_residual=self.cross_attn.norm_residual)

		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=self.ff.norm_residual)

		#self.layer_normer1 = LayerNorm(isize, eps=1e-06)
		#self.layer_normer2 = LayerNorm(isize, eps=1e-06)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=None, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.wemb = Embedding(nwd, isize, padding_idx=pad_id)
		if emb_w is not None:
			self.wemb.weight = emb_w

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.classifier = LinearBn(isize, nwd)
		if bindemb:
			self.classifier.weight = self.wemb.weight

		#self.out_normer = LayerNorm(isize, eps=1e-06) if norm_output else None

	def fix_init(self):

		self.fix_load()
		with torch_no_grad():
			self.wemb.weight.data[pad_id].zero_()
			self.classifier.weight.data[pad_id].zero_()

	def fix_load(self):

		if self.fbl is not None:
			with torch_no_grad():
				self.classifier.bias.data.index_fill_(0, torch.as_tensor(self.fbl, dtype=torch.long, device=self.classifier.bias.data.device), -inf_default)

	def get_sos_emb(self, inpute, bsize=None):

		bsize = inpute.size(0) if bsize is None else bsize

		return self.wemb.weight()[1].view(1, 1, -1).expand(bsize, 1, -1)

	def update_vocab(self, indices):

		_nwd = len(indices)
		_wemb = Embedding(_nwd, self.wemb.weight.data.size(-1), padding_idx=pad_id)
		_classifier = LinearBn(self.classifier.weight.data.size(-1), _nwd)
		with torch_no_grad():
			_wemb.weight.data.copy_(self.wemb.weight.data.index_select(0, indices))
			if self.classifier.weight.data.is_set_to(self.wemb.weight.data):
				_classifier.weight = _wemb.weight
			else:
				_classifier.weight.data.copy_(self.classifier.weight.data.index_select(0, indices))
			_classifier.bias.data.copy_(self.classifier.bias.data.index_select(0, indices))
		self.wemb, self.classifier = _wemb, _classifier
