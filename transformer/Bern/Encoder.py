#encoding: utf-8

from torch import nn

from modules.prune.bern import Embedding, PositionwiseFF, ResSelfAttn#, LayerNorm
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head, dropout=attn_drop, norm_residual=self.attn.norm_residual)

		self.ff = PositionwiseFF(isize, _fhsize, dropout, norm_residual=self.ff.norm_residual)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.wemb = Embedding(nwd, isize, padding_idx=pad_id)

		self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		#self.out_normer = LayerNorm(isize, eps=1e-06) if norm_output else None

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		with torch_no_grad():
			self.wemb.weight.data[pad_id].zero_()

	def update_vocab(self, indices):

		_wemb = Embedding(len(indices), self.wemb.weight.data.size(-1), padding_idx=pad_id)
		with torch_no_grad():
			_wemb.weight.data.copy_(self.wemb.weight.data.index_select(0, indices))
		self.wemb = _wemb
