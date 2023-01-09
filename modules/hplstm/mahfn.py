#encoding: utf-8

import torch

from modules.hplstm.LGate import LGateFunc
from modules.hplstm.hfn import BiHPLSTM as BiHPLSTMBase, HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase
from utils.base import float2odd
from utils.fmt.parser import parse_none
from utils.torch.c import MovAvgFunc

class MHPLSTMCore(MHPLSTMCoreBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, ma_beta=0.9, **kwargs):

		super(MHPLSTMCore, self).__init__(isize, num_head=num_head, osize=osize, fhsize=fhsize, dropout=dropout, **kwargs)

		self.ma_beta = ma_beta

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(MovAvgFunc(torch.cat((heads_input.new_zeros(bsize, 1, nheads, adim), heads_input.narrow(1, 0, seql - 1),), dim=1), 1, self.ma_beta, True))
		else:
			_init_state = (states == "init")
			if _init_state:
				csum = self.normer_csum(heads_input.new_zeros(1, 1, nheads, adim)).expand(bsize, 1, nheads, adim)
				csum_state_return = heads_input.mul_(1.0 - self.ma_beta)
			else:
				_csum_state = states[0]
				csum = self.normer_csum(_csum_state)
				csum_state_return = _csum_state.mul_(self.ma_beta).add_(heads_input, alpha=1.0 - self.ma_beta)
		gh_input = torch.cat((heads_input, csum,), dim=-1)
		(igate, fgate,), hidden = self.normer_ifg(self.trans_ifg(gh_input).view(bsize, seql, nheads, 2, -1)).sigmoid().unbind(-2), self.trans_hid(gh_input)
		igh = igate * hidden
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
