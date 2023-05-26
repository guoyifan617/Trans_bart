#encoding: utf-8

from math import sqrt
from torch import nn

from modules.attn.on import Summer
from modules.base import Linear, PositionwiseFF
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.summer = Summer(isize, _ahsize, isize, num_head=num_head, dropout=dropout)
		self.num_anchor, self.num_head, self.attn_dim, self.hsize = self.summer.num_anchor, self.summer.num_head, self.summer.attn_dim, self.summer.hsize

		self.transi = Linear(isize, self.hsize, bias=enable_proj_bias)
		self.transci = Linear(self.num_anchor * self.attn_dim, self.num_head * isize, bias=enable_proj_bias)
		self.transco = Linear(self.num_head * isize, self.num_anchor * self.attn_dim, bias=enable_proj_bias)
		self.transo = Linear(self.hsize, isize, bias=enable_bias)

		if share_layer:
			_shared_layer = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop)
			self.nets = nn.Sequential(*[_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.Sequential(*[PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		out = self.summer(out, mask=mask)

		bsize, num_anchor, isize = out.size()
		num_head, attn_dim = self.num_head, self.attn_dim

		out = self.transci(self.transi(out).view(bsize, num_anchor, num_head, attn_dim).transpose(1, 2).contiguous().view(bsize * num_head, num_anchor * attn_dim)).view(-1, isize)#bsize * num_head * num_head

		out = self.nets(out)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.transo(self.transco(out.view(bsize * num_head, num_head * isize)).view(bsize, num_head, num_anchor, attn_dim).transpose(1, 2).contiguous().view(bsize, num_anchor, isize))

		return out
