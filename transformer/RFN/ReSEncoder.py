#encoding: utf-8

#import torch
from math import sqrt
from torch import nn

from modules.rfn import LSTMCell4FFN, ResSelfAttn#, rnncells import LSTMCell4RNMT as
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, enable_sattn_outer=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=self.attn.norm_residual, k_rel_pos=k_rel_pos, enable_outer=enable_sattn_outer)
		self.ff = LSTMCell4FFN(isize, dropout=dropout, act_drop=act_drop)#, osize=isize, hsize=_fhsize

	def forward(self, inputs, cellin, mask=None, **kwargs):

		context = self.attn(inputs, mask=mask)

		out, cellout = self.ff(context, (inputs, cellin))

		return out, cellout

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		#self.init_cx = nn.Parameter(torch.zeros(1, 1, isize))

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		cell = out#self.init_cx.expand_as(out)

		for net in self.nets:
			out, cell = net(out, cell, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		with torch_no_grad():
			self.wemb.weight[pad_id].zero_()
			#self.init_cx.zero_()
