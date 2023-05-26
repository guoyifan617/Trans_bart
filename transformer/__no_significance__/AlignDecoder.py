#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.attn.rap import NoGradLinear, ResCrossAttn
from modules.base import Dropout, Linear
from modules.mono import GradientBalanceLayer
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(DecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head, dropout=attn_drop, norm_residual=self.cross_attn.norm_residual)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo, mask=tgt_pad_mask)
		else:
			context, states_return = self.self_attn(query_unit, states=inputo)

		context, _attn = self.cross_attn(context, inpute, mask=src_pad_mask)

		context = self.ff(context)

		if query_unit is None:
			return context, _attn
		else:
			return context, states_return#, _attn

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, enc_t_w=0.1, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.tattn_w = nn.Parameter(torch.Tensor(num_layer * num_head).uniform_(- sqrt(1.0 / (num_layer * num_head)), sqrt(1.0 / (num_layer * num_head))))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None
		self.tenc = Linear(isize, isize, bias=False)
		self.enc_classifier = NoGradLinear(self.classifier.weight, self.classifier.bias)
		self.gbl = GradientBalanceLayer(enc_t_w)

	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		if self.training:
			_inpute, inpute = self.gbl(inpute)

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		attns = []
		for net in self.nets:
			out, _attn = net(inpute, out, src_pad_mask, _mask)
			attns.append(_attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		if self.training:
			# attns: (bsize, num_layer * nheads, nquery, seql) => (bsize, nquery, seql, num_layer * nheads)
			attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1).contiguous()
			_asize = attns.size()

			# inpute: (bsize, seql, isize)
			# attns: (bsize, nquery, seql, num_layer * nheads) => (bsize, nquery, seql)
			# out_enc: (bsize, nquery, isize)
			out_enc = attns.view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(_inpute)

			return torch.stack((out, self.lsm(self.enc_classifier(self.tenc(out_enc))),), dim=1)
		else:
			return out
