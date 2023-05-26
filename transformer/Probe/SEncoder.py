#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Linear
from transformer.Encoder import Encoder as EncoderBase
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, **kwargs):

		_fhsize = isize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer if num_layer > 0 else 1, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, **kwargs)

		if num_layer <= 0:
			self.nets = None

		self.classifier = nn.Sequential(Linear(isize, isize, bias=False), Linear(isize, nwd))#nn.Sequential(Linear(isize, _fhsize), nn.Sigmoid(), Linear(_fhsize, isize, bias=False), Linear(isize, nwd))
		self.lsm = nn.LogSoftmax(-1)

		if bindemb:
			list(self.classifier.modules())[-1].weight = self.wemb.weight
		self.bindemb = bindemb

		self.fbl = None if forbidden_index is None else tuple(set(forbidden_index))

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		if self.nets is not None:
			for net in self.nets:
				out = net(out, mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return self.lsm(self.classifier(out))

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		if self.nets is not None:
			_nets = list(base_encoder.nets)
			self.nets = nn.ModuleList(_nets[:len(self.nets)] + list(self.nets[len(_nets):]))

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer

	def fix_init(self):

		self.fix_load()
		if not self.bindemb:
			with torch_no_grad():
				list(self.classifier.modules())[-1].weight.copy_(self.wemb.weight)

	def fix_load(self):

		if self.fbl is not None:
			with torch_no_grad():
				list(self.classifier.modules())[-1].bias.index_fill_(0, torch.as_tensor(self.fbl, dtype=torch.long, device=list(self.classifier.modules())[-1].bias.device), -inf_default)
