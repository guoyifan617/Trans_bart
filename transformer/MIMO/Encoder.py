#encoding: utf-8

import torch
from math import ceil, sqrt
from random import shuffle

from modules.base import Linear
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nmimo=4, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.trans = Linear(isize, isize * nmimo, bias=enable_proj_bias)
		self.nmimo = nmimo

	def forward(self, inputs, mask=None, ensemble_decoding=False, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		if self.training:
			out = self.trans(out).view(list(out.size()) + [self.nmimo])
			rind = []
			_out = []
			_l = list(range(inputs.size(0)))
			_mask = None if mask is None else mask
			for i, _ou in enumerate(out.unbind(-1)):
				if i > 0:
					shuffle(_l)
					_ind = torch.as_tensor(_l, dtype=torch.long, device=inputs.device)
					rind.append(_ind)
					_out.append(_ou.index_select(0, _ind))
					if _mask is not None:
						_mask = _mask & mask.index_select(0, _ind)
				else:
					_out.append(_ou)
			out = torch.stack(_out, -1).sum(-1)
		else:
			rind = None
			if ensemble_decoding:
				out = self.trans(out).view(list(out.size()) + [self.nmimo]).sum(-1)
				_mask = mask
			else:
				bsize, seql, isize = out.size()
				_reduced_bsize = ceil(float(bsize) / float(self.nmimo))
				_er_bsize = _reduced_bsize * self.nmimo
				_npad = _er_bsize - bsize
				_w_trans, _b_trans = self.trans.weight.view(isize, self.nmimo, isize).permute(1, 2, 0).contiguous(), None if self.trans.bias is None else self.trans.bias.view(isize, self.nmimo).transpose(0, 1).contiguous()
				if _npad > 0:
					out = torch.cat((out, out.new_zeros((_npad, seql, isize,)),), dim=0)
					if mask is None:
						_mask = mask
					else:
						_mask = torch.cat((mask, mask.new_ones((_npad, 1, seql)),), dim=0).view(_reduced_bsize, self.nmimo, 1, seql).int().sum(1).eq(self.nmimo)
				else:
					_mask = mask.view(_reduced_bsize, self.nmimo, 1, seql).int().sum(1).eq(self.nmimo)
				out = out.view(_reduced_bsize, self.nmimo, seql, isize)
				out = out.transpose(0, 1).contiguous().view(self.nmimo, _reduced_bsize * seql, isize).bmm(_w_trans).transpose(0, 1).contiguous().view(_reduced_bsize, seql, self.nmimo, isize)
				if _b_trans is not None:
					out = out + _b_trans
				out = out.sum(-2)

		for net in self.nets:
			out = net(out, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if rind is None:
			if ensemble_decoding:
				return out
			else:
				return out, _mask
		else:
			return out, rind, _mask
