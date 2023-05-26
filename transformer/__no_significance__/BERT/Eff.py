#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Linear
from modules.dropout import TokenDropout
from transformer.Encoder import Encoder as EncoderBase
from utils.relpos.base import share_rel_pos_cache_enc
from utils.torch.comp import torch_no_grad#, mask_tensor_type

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, ptdrop=0.15, **kwargs):

		super(NMT, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, **kwargs)

		self.classifier = Linear(isize, nwd)
		if bindDecoderEmb:
			self.classifier.weight = self.wemb.weight
		self.lsm = nn.LogSoftmax(-1)
		self.tdrop = TokenDropout(ptdrop)
		self.fbl = None if forbidden_index is None else tuple(set(forbidden_index))
		self.dscale = 1.0 / (1.0 - 1.0 / (num_layer - 2))

		self.nets = share_rel_pos_cache_enc(self.nets)

	def forward(self, inputs, mask=None, eva_mask=None, emask_p=0.0, **kwargs):

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask

		out = self.wemb(inputs)

		if eva_mask is not None:
			out.masked_fill_(eva_mask.unsqueeze(-1), 0.0)
			if emask_p > 0.0:
				out = out * (1.0 / (1.0 - emask_p))

		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		#_depth, _wo_mask, _m_size, _cont_mask = len(self.nets), None, out.size()[:-1], self.training
		#_depth_last = _depth - 1
		_depth = len(self.nets)
		for _ind, net in enumerate(self.nets, 1):
			out = net(out, _mask)
			if _ind < _depth and self.training:
				out = self.tdrop(out)
				"""if _cont_mask:
					if _ind < _depth_last:
						_cur_mask = out.new_full(_m_size, 1.0 / (_depth - _ind)).bernoulli().to(mask_tensor_type, non_blocking=True)
						if _wo_mask is None:
							_wo_mask = ~_cur_mask
						else:
							_cur_mask &= _wo_mask
							_wo_mask |= _cur_mask
						_cont_mask = (_wo_mask.int().sum().item() > 0)
					else:
						_cur_mask = ~_wo_mask
						_cont_mask = False
					if _cur_mask.int().sum().item() > 0:
						out = out.masked_fill(_cur_mask.unsqueeze(-1), 0.0)
				out = out * self.dscale"""

		if self.out_normer is not None:
			out = self.out_normer(out)

		return self.lsm(self.classifier(out))

	def fix_load(self):

		if self.fbl is not None:
			with torch_no_grad():
				self.classifier.bias.index_fill_(0, torch.as_tensor(self.fbl, dtype=torch.long, device=self.classifier.bias.device), -inf_default)

	def fix_init(self):

		if hasattr(self, "fix_load"):
			self.fix_load()
		with torch_no_grad():
			#self.wemb.weight[pad_id].zero_()
			self.classifier.weight[pad_id].zero_()
