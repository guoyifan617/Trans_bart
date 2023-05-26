#encoding: utf-8

from math import sqrt
from torch import nn

from modules.noise import EMixNoiser as Noiser, PositionwiseFF
from transformer.Noise.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, power=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, power=power, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=self.attn.norm_residual, power=power, custom_noiser=Noiser)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, power=power, custom_noiser=Noiser)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, power=0.1, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, power=power, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, power=power)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, power=power) for i in range(num_layer)])

		self.noiser = None if power is None else Noiser(power, inplace=True)

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		if self.training and (mask is not None):
			bsize, seql = inputs.size()
			_noise_mask = (~mask.view(bsize, seql)).to(out.dtype, non_blocking=True)
		else:
			_noise_mask = None
		for net in self.nets:
			out = net(out, mask, _noise_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if self.noiser is not None:
			out = self.noiser(out, _noise_mask)

		return out if self.out_normer is None else self.out_normer(out)

	def set_noise(self, value):

		for _m in self.modules():
			if isinstance(_m, Noiser):
				_m.power = value
