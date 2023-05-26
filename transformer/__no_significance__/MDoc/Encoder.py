#encoding: utf-8

from math import sqrt

from modules.mdoc import PositionalEmb
from transformer.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		self.pemb = PositionalEmb(isize, xseql, 0, 0)
		self.semb = PositionalEmb(isize, 32, 5000, isize, sqrt(2)/16.0)

	# inputs: (bsize, nsent, seql)
	# mask: (bsize, 1, nsent, seql), generated with:
	#	mask = inputs.eq(pad_id).unsqueeze(1)
	# rsentind: the index of sentence which is removed

	def forward(self, inputs, mask=None, rsentind=None, **kwargs):

		out = self.wemb(inputs)
		_pemb_seq = self.pemb(inputs.select(1, 0), expand=False).unsqueeze(1)
		_pemb_sent = self.semb(inputs.select(-1, 0), expand=False, rsentid=rsentind).unsqueeze(2)
		bsize, nsent, seql, osize = out.size()
		out = (_pemb_seq + _pemb_sent + out * sqrt(osize)).view(bsize * nsent, seql, osize)
		_mask = mask.view(bsize * nsent, 1, seql)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		return out.view(bsize, nsent, seql, osize)

"""
		if mask is None:
			out = out.mean(2)#out.max(2)[0]
		else:
			_mask = mask.view(bsize, nsent, seql, 1)
			out.masked_fill(_mask, 0.0)#-inf_default
			_nele = (seql - _mask.sum(2)).to(out, non_blocking=True)
			out = out.sum(2) / _nele#out.max(2)[0]
"""
