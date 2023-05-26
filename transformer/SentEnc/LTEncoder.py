#encoding: utf-8

from math import sqrt
from torch import nn

from modules.attn.on import Summer
from modules.base import Linear, PositionwiseFF as PositionwiseFFBase
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(PositionwiseFFBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, num_head=8, ahsize=None, num_anchor=None, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, hsize=_fhsize, dropout=dropout, enable_bias=enable_bias, **kwargs)

		self.num_head = num_head
		self.attn_dim = _ahsize // num_head
		self.hsize = self.attn_dim * num_head

		self.transi = Linear(isize, self.hsize, bias=enable_proj_bias)
		self.transci = Linear(num_anchor * self.attn_dim, self.num_head * isize, bias=enable_proj_bias)
		self.transco = Linear(self.num_head * isize, num_anchor * self.attn_dim, bias=enable_proj_bias)
		self.transo = Linear(self.hsize, osize, bias=enable_bias)

	# x: (bsize, num_anchor, isize)
	def forward(self, x, **kwargs):

		bsize, num_anchor, isize = x.size()
		num_head, attn_dim = self.num_head, self.attn_dim

		_out = self.normer(x)

		out = self.transci(self.transi(_out).view(bsize, num_anchor, num_head, attn_dim).transpose(1, 2).contiguous().view(bsize * num_head, num_anchor * attn_dim)).view(-1, isize)#bsize * num_head * num_head

		out = self.net(out)

		out = self.transo(self.transco(out.view(bsize * num_head, num_head * isize)).view(bsize, num_head, num_anchor, attn_dim).transpose(1, 2).contiguous().view(bsize, num_anchor, isize))

		out = out + (_out if self.norm_residual else x)

		return out

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.summer = Summer(isize, _ahsize, isize, num_head=num_head, dropout=dropout)
		self.num_anchor = self.summer.num_anchor

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, num_head=num_head, ahsize=_ahsize, num_anchor=self.num_anchor)
			self.nets = nn.Sequential(*[_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.Sequential(*[EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, num_head=num_head, ahsize=_ahsize, num_anchor=self.num_anchor) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		out = self.summer(out, mask=mask)

		out = self.nets(out)

		return out if self.out_normer is None else self.out_normer(out)
