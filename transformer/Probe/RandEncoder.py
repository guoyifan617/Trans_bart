#encoding: utf-8

from math import sqrt
from torch import nn

from transformer.Encoder import EncoderLayer
from transformer.Probe.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad
from utils.train.base import freeze_module

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_layer_ana=0, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, num_layer_ana=num_layer_ana, **kwargs)

		self.rwemb = nn.Embedding(nwd, isize, padding_idx=pad_id)
		self.rnets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer_ana)]) if num_layer_ana > 0 else None
		self.rout_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None
		freeze_module(self.rwemb)
		if self.rnets is not None:
			freeze_module(self.rnets)
		if self.rout_normer is not None:
			freeze_module(self.rout_normer)

	def forward(self, inputs, mask=None, no_std_out=False, **kwargs):

		rout = self.rwemb(inputs)
		rout = rout * sqrt(rout.size(-1))
		if self.pemb is not None:
			rout = rout + self.pemb(inputs, expand=False)

		if self.drop is not None:
			rout = self.drop(rout)

		if self.nets is not None:
			for net in self.nets:
				rout = net(rout, mask)
		if self.rout_normer is not None:
			rout = self.rout_normer(rout)
		if no_std_out:
			return rout
		else:
			out = self.wemb(inputs)
			if self.pemb is not None:
				out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

			if self.drop is not None:
				out = self.drop(out)
			for net in self.nets:
				out = net(out, mask)

			if self.out_normer is not None:
				out = self.out_normer(out)

			return out, rout

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		with torch_no_grad():
			self.wemb.weight[pad_id].zero_()
			self.rwemb.weight[pad_id].zero_()
