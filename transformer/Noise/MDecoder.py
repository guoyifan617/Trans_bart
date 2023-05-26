#encoding: utf-8

from math import sqrt
from torch import nn

from modules.noise import DMixNoiser as Noiser, PositionwiseFF, ResCrossAttn, ResSelfAttn
from transformer.Noise.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, power=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, power=power, **kwargs)

		self.self_attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, uni_direction_reduction=True, power=power, custom_noiser=Noiser)
		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, power=power, custom_noiser=Noiser)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, power=power, custom_noiser=Noiser)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, share_layer=False, power=0.1, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, power=power, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, power=power)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, power=power) for i in range(num_layer)])
		self.noiser = None if power is None else Noiser(power, inplace=True)

	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)
		_noise_mask = (inputo.ne(0).to(out.dtype, non_blocking=True) if self.training else out.new_ones(inputo.size())).transpose(0, 1).contiguous()

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask, noise_mask=_noise_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if self.noiser is not None:
			out = self.noiser(out, _noise_mask)

		out = self.lsm(self.classifier(out))

		return out

	def set_noise(self, value):

		for _m in self.modules():
			if isinstance(_m, Noiser):
				_m.power = value
