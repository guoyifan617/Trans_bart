#encoding: utf-8

from math import ceil, sqrt
from torch import nn

from modules.rfn import LSTMCell4FFN#, rnncells import LSTMCell4RNMT as
from transformer.RFN.Encoder import Encoder as EncoderBase
from utils.base import iternext

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, num_layer_step=6, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, **kwargs)

		self.nlu = num_layer_step
		_nrfn = ceil(float(num_layer) / float(num_layer_step))
		self.mgrs = nn.ModuleList([LSTMCell4FFN(isize, dropout=dropout, act_drop=act_drop) for i in range(_nrfn)]) if _nrfn > 1 else None

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		cell = out
		if self.mgrs is None:
			mgiter = None
		else:
			mgcell = mgout = out
			mgiter = iter(self.mgrs)

		for i, net in enumerate(self.nets, 1):
			out, cell = net(out, cell, mask)
			if (mgiter is not None) and (i % self.nlu == 0):
				_mgm = iternext(mgiter)
				if _mgm is not None:
					mgout, mgcell = _mgm(cell, (mgout, mgcell))#out

		if mgiter is not None:
			_mgm = iternext(mgiter)
			if _mgm is not None:
				mgout, mgcell = _mgm(cell, (mgout, mgcell))#out
			out = mgout

		return out if self.out_normer is None else self.out_normer(out)
