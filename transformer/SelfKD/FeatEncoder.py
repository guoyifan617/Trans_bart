#encoding: utf-8

import torch
from math import sqrt

from modules.kd.base import GradientAdapterFunc
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.kd.self.feat import get_kd_loss

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, kd_layers=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.kd_layers = set() if kd_layers is None else (kd_layers if isinstance(kd_layers, set) else set(kd_layers))

	def forward(self, inputs, mask=None, gold=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		kd_o = []
		for prev_layer_ind, net in enumerate(self.nets):
			if prev_layer_ind in self.kd_layers:
				out, _ = GradientAdapterFunc(out)
				kd_o.append(_)
			out = net(out, mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if self.training and (gold is not None):
			if (prev_layer_ind + 1) in self.kd_layers:
				kd_o.append(out)
			return out, get_kd_loss(torch.stack(kd_o, dim=0), mask=None if mask is None else mask.squeeze(1)) if len(kd_o) > 1 else out.new_zeros(1)
		else:
			return out
