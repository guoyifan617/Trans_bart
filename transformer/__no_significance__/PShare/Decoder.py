#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.pshare import PositionwiseFF, ResCrossAttn, ResSelfAttn
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, sattnwb1, sattnwb2, cattnwb1, cattnwb2, cattnwb3, ffnwb1, ffnwb2, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, **kwargs)

		self.self_attn = ResSelfAttn(sattnwb1, sattnwb2, _ahsize, num_head, dropout=attn_drop, norm_residual=norm_residual)
		self.cross_attn = ResCrossAttn(cattnwb1, cattnwb2, cattnwb3, _ahsize, num_head, dropout=attn_drop, norm_residual=norm_residual)
		self.ff = PositionwiseFF(ffnwb1, hsize=ffnwb2, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, num_weight=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_ahsize = _ahsize // num_head * num_head

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		_num_weight = max(3, num_layer // 2) if num_weight is None else num_weight

		self.sattnwb1 = nn.Parameter(torch.Tensor(_ahsize * 3, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.sattnwb2 = nn.Parameter(torch.Tensor(isize, _ahsize, _num_weight).uniform_(- 1.0 / sqrt(_ahsize), 1.0 / sqrt(_ahsize)))
		self.cattnwb1 = nn.Parameter(torch.Tensor(_ahsize, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.cattnwb2 = nn.Parameter(torch.Tensor(_ahsize * 2, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.cattnwb3 = nn.Parameter(torch.Tensor(isize, _ahsize, _num_weight).uniform_(- 1.0 / sqrt(_ahsize), 1.0 / sqrt(_ahsize)))
		self.ffnwb1 = nn.Parameter(torch.Tensor(_fhsize, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.ffnwb2 = nn.Parameter(torch.Tensor(isize, _fhsize, _num_weight).uniform_(- 1.0 / sqrt(_fhsize), 1.0 / sqrt(_fhsize)))

		self.nets = nn.ModuleList([DecoderLayer(isize, self.sattnwb1, self.sattnwb2, self.cattnwb1, self.cattnwb2, self.cattnwb3, self.ffnwb1, self.ffnwb2, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])
