#encoding: utf-8

from torch import nn

from modules.spreader.neural.attn import BiSpreader
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.math import exp_grow as grow_func

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, s_start=2, s_end=8, e_start=4, e_end=16, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, disable_pemb=True, **kwargs)

		if share_layer:
			_shared_layer = BiSpreader(isize, hsize=_ahsize, start=s_start, end=s_end, dropout=attn_drop)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([BiSpreader(isize, hsize=_ahsize, start=_s_start, end=_s_end, dropout=attn_drop) for (_s_start, _s_end,) in zip(grow_func(s_start, e_start, num_layer), grow_func(s_end, e_end, num_layer))])

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)
