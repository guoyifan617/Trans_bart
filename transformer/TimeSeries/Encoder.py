#encoding: utf-8

from math import sqrt
from torch import nn

from modules.timeseries import PositionwiseFF
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *
from cnfg.timeseries import num_steps

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.ff = PositionwiseFF(isize, num_head=num_head, osize=isize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop)

	# inputs: (bsize, ngroup, seql, isize)
	def forward(self, inputs, mask=None, **kwargs):

		_bsize, _ngroup, _seql, _isize = inputs.size()
		context = self.attn(inputs.view(_bsize * _ngroup, _seql, _isize), mask=mask).view(_bsize, _ngroup, _seql, _isize)

		context = self.ff(context)

		return context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, num_steps=num_steps, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize) for i in range(num_layer)])

		self.net = PositionwiseFF(isize, num_head=num_head, osize=isize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop)
		self.num_steps = num_steps

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		# before or after drop?
		out = self.net(out.unsqueeze(1).expand(-1, self.num_steps, -1, -1))
		_mask = None if mask is None else mask.repeat(1, self.num_steps, 1).view(-1, *mask.size()[1:])

		for net in self.nets:
			out = net(out, _mask)

		out = out.select(1, -1)

		return out if self.out_normer is None else self.out_normer(out)
