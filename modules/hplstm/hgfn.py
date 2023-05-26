#encoding: utf-8

import torch
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout
from modules.group.base import GroupLinear
from modules.hplstm.LGate import LGateFunc
from modules.hplstm.base import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase
from utils.base import float2odd
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_osize = parse_none(osize, isize)

		i_hsize = float2odd(float(isize) / num_head) * num_head
		o_hsize = float2odd(float(_osize) / num_head) * num_head
		_fhsize = o_hsize * 4 if fhsize is None else fhsize
		_head_fhsize =float2odd(float(_fhsize) / num_head)
		_fhsize = _head_fhsize * num_head

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=_osize, dropout=dropout, custom_act=custom_act, enable_bias=enable_bias)

		self.act = None
		self.trans_hid = nn.Sequential(GroupLinear(i_hsize + i_hsize, _fhsize, num_head, bias=enable_bias, shuffle=False, trans_input=False, flatten_output=False), nn.LayerNorm((num_head, _head_fhsize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_fhsize, o_hsize * 3, num_head, bias=enable_proj_bias, shuffle=False, trans_input=False, flatten_output=False))
		if dropout > 0.0:
			self.trans_hid.insert(3, Dropout(dropout, inplace=inplace_after_Custom_Act))

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(torch.cat((heads_input.new_zeros(bsize, 1, nheads, adim), heads_input.narrow(1, 0, seql - 1),), dim=1).cumsum(dim=1))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state + heads_input
		igate, fgate, hidden = self.normer_hid(self.trans_hid(torch.cat((heads_input, csum,), dim=-1)).view(bsize, seql, nheads, 3, -1)).unbind(-2)
		fgate = fgate.sigmoid()

		if self.drop is not None:
			hidden = self.drop(hidden)
		igh = igate.sigmoid() * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx, 1, True) if states is None else igh.addcmul_(fgate, self.init_cx if _init_state else states[-1])
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return, cell,)

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		_osize = parse_none(osize, isize)
		_fhsize = float2odd(float(_osize * 4 if fhsize is None else fhsize) / num_head) * num_head

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, dropout=dropout, **kwargs)

		i_hsize = float2odd(float(isize) / num_head) * num_head
		o_hsize = float2odd(float(_osize) / num_head) * num_head

		self.net = MHPLSTMCore(i_hsize, num_head=self.num_head, osize=o_hsize, fhsize=_fhsize, dropout=dropout)

class BiHPLSTM(BiHPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, **kwargs):

		_osize = parse_none(osize, isize)
		_fhsize = float2odd(float(_osize * 4 if fhsize is None else fhsize) / num_head) * num_head

		super(BiHPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, dropout=dropout, **kwargs)

		i_hsize = float2odd(float(isize) / num_head) * num_head
		o_hsize = float2odd(float(_osize) / num_head) * num_head

		self.net = MHPLSTMCore(i_hsize + i_hsize, num_head=self.num_head + self.num_head, osize=o_hsize + o_hsize, fhsize=_fhsize + _fhsize, dropout=dropout)
