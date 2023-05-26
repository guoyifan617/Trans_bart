#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout, Linear
from modules.kd.base import DynT
from transformer.AlignDecoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none
from utils.kd.self.p import get_iter_kd_loss, get_kd_loss
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

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

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, kd_layers=None, enable_proj=True, num_topk=10, T=1.0, min_T=None, min_gold_p=None, num_mix=0, iter_kd=True, remove_gold=False, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=False, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

		self.tattn_w = nn.Parameter(torch.Tensor(num_layer * num_head).uniform_(- sqrt(1.0 / (num_layer * num_head)), sqrt(1.0 / (num_layer * num_head))))
		self.mix_w = nn.Parameter(torch.zeros(num_mix)) if num_mix > 0 else None
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None

		self.kd_layers = set() if kd_layers is None else (kd_layers if isinstance(kd_layers, set) else set(kd_layers))
		self.deep_kd = num_layer in self.kd_layers
		if enable_proj:
			_num_proj = len(self.kd_layers)
			if self.deep_kd:
				_num_proj -= 1
			self.proj_nets = nn.ModuleList([Linear(isize, isize, bias=enable_proj_bias) for i in range(_num_proj)])
		else:
			self.proj_nets = None
		self.num_topk, self.min_gold_p, self.remove_gold = num_topk, min_gold_p, remove_gold
		_T = 1.0 if T is None else T
		self.T = _T if min_T is None else DynT(init_value=_T, min_value=min_T)
		self.kd_loss = get_iter_kd_loss if iter_kd else get_kd_loss

	def forward(self, inpute, inputo, src_pad_mask=None, gold=None, enc_kd_hiddens=None, gold_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)
		kd_o = []
		attns = []
		for prev_layer_ind, net in enumerate(self.nets):
			if prev_layer_ind in self.kd_layers:
				kd_o.append(out)
			out, _attn = net(inpute, out, src_pad_mask, _mask)
			attns.append(_attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		_last_layer_out = out
		_final_predict = self.classifier(out)

		out = self.lsm(_final_predict)

		if self.training and (gold is not None):

			if self.proj_nets is not None:
				kd_o = [_(_o) for _, _o in zip(self.proj_nets, kd_o)]
				#if len(self.nets) in self.kd_layers:
					#kd_o.append(_last_layer_out)

			if enc_kd_hiddens:
				attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1).contiguous()
				_asize = attns.size()
				attns = attns.view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1])
				bsize, seql, isize = enc_kd_hiddens[0].size()
				enc_kd_o = attns.bmm(torch.stack(enc_kd_hiddens, dim=-2).view(bsize, seql, -1)).view(bsize, nquery, -1, isize)
			else:
				enc_kd_o = None

			return out, self.kd_loss(classifier=self.classifier, final_predict=_final_predict, enc_kd_o=enc_kd_o, kd_o=kd_o, gold=gold, gold_pad_mask=gold_pad_mask, num_topk=self.num_topk, T=self.T, min_gold_p=self.min_gold_p, deep_kd=self.deep_kd, mix_p=None if self.mix_w is None else (self.mix_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.mix_w).softmax(dim=0)), remove_gold=self.remove_gold)
		else:
			return out

	def fix_init(self):

		super(Decoder, self).fix_init()
		if self.mix_w is not None:
			_n_mix_i = self.mix_w.size(0)
			with torch_no_grad():
				self.mix_w.uniform_(- sqrt(1.0 / _n_mix_i), sqrt(1.0 / _n_mix_i))
