#encoding: utf-8

from math import sqrt
from torch import nn

from modules.mulang.eff.rfn import LSTMCell4FFN
from transformer.MuLang.Eff.Base.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, ntask=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, ntask=ntask, **kwargs)

		self.layer_normer, self.drop = self.attn.normer, self.attn.drop
		self.attn = self.attn.net
		self.ff = LSTMCell4FFN(isize, osize=isize, hsize=_fhsize, dropout=dropout, ntask=ntask)

	def forward(self, inputs, cellin, taskid=None, mask=None, **kwargs):

		_inputs = self.layer_normer(inputs, taskid=taskid)
		context = self.attn(_inputs, mask=mask)

		if self.drop is not None:
			context = self.drop(context)

		return self.ff(context, (inputs, cellin), taskid=taskid)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, ntask=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, ntask=ntask, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ntask=ntask)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize, ntask=ntask) for i in range(num_layer)])

	def forward(self, inputs, taskid=None, mask=None, **kwargs):

		out = self.wemb(inputs) + self.task_emb.weight[taskid]
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		cell = out
		for net in self.nets:
			out, cell = net(out, cell, taskid=taskid, mask=mask)

		if self.out_normer is not None:
			out = self.out_normer(out, taskid=taskid)

		return self.transo(out, taskid)
