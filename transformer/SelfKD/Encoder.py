#encoding: utf-8

from math import sqrt
from torch import nn

from modules.base import Linear
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, kd_layers=None, enable_proj=True, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.kd_layers = set() if kd_layers is None else (kd_layers if isinstance(kd_layers, set) else set(kd_layers))
		self.proj_nets = nn.ModuleList([Linear(isize, isize, bias=enable_proj_bias) for i in range(len(self.kd_layers))]) if enable_proj and self.kd_layers else None

	def forward(self, inputs, mask=None, gold=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		kd_o = []
		for prev_layer_ind, net in enumerate(self.nets):
			if prev_layer_ind in self.kd_layers:
				kd_o.append(out)
			out = net(out, mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if self.training and (gold is not None):
			if (prev_layer_ind + 1) in self.kd_layers:
				kd_o.append(out)
			return out, kd_o if self.proj_nets is None else [_(_o) for _, _o in zip(self.proj_nets, kd_o)]
		else:
			return out
