#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.pshare import PositionwiseFF, ResSelfAttn
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, attnwb1, attnwb2, ffnwb1, ffnwb2, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(EncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.attn = ResSelfAttn(attnwb1, attnwb2, _ahsize, num_head, dropout=attn_drop)
		self.ff = PositionwiseFF(ffnwb1, hsize=ffnwb2, dropout=dropout, act_drop=act_drop, norm_residual=self.ff.norm_residual)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_weight=None, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_ahsize = _ahsize // num_head * num_head

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, **kwargs)

		_num_weight = max(3, num_layer // 2) if num_weight is None else num_weight

		self.attnwb1 = nn.Parameter(torch.Tensor(_ahsize * 3, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.attnwb2 = nn.Parameter(torch.Tensor(isize, _ahsize, _num_weight).uniform_(- 1.0 / sqrt(_ahsize), 1.0 / sqrt(_ahsize)))
		self.ffnwb1 = nn.Parameter(torch.Tensor(_fhsize, isize, _num_weight).uniform_(- 1.0 / sqrt(isize), 1.0 / sqrt(isize)))
		self.ffnwb2 = nn.Parameter(torch.Tensor(isize, _fhsize, _num_weight).uniform_(- 1.0 / sqrt(_fhsize), 1.0 / sqrt(_fhsize)))

		self.nets = nn.ModuleList([EncoderLayer(isize, self.attnwb1, self.attnwb2, self.ffnwb1, self.ffnwb2, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])
