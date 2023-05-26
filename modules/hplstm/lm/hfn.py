#encoding: utf-8

import torch

from modules.hplstm.LGate import LGateFunc
from modules.hplstm.hfn import HPLSTM as HPLSTMBase, MHPLSTMCore as MHPLSTMCoreBase
from utils.fmt.parser import parse_none

class MHPLSTMCore(MHPLSTMCoreBase):

	def forward(self, heads_input, states=None, head_mask=None, **kwargs):

		bsize, seql, nheads, adim = heads_input.size()
		if states is None:
			csum = self.normer_csum(torch.cat((heads_input.new_zeros(bsize, 1, nheads, adim), heads_input.narrow(1, 0, seql - 1),), dim=1).cumsum(dim=1))
		else:
			_init_state = (states == "init")
			csum = torch.cat((heads_input.new_zeros(bsize, 1, nheads, adim) if _init_state else states[0], heads_input,), dim=1).cumsum(dim=1)
			csum_state_return = csum.narrow(1, seql, 1)
			csum = self.normer_csum(csum.narrow(1, 0, seql))
		gh_input = torch.cat((heads_input, csum,), dim=-1)
		(igate, fgate,), hidden = self.normer_ifg(self.trans_ifg(gh_input).view(bsize, seql, nheads, 2, -1)).sigmoid().unbind(-2), self.trans_hid(gh_input)
		igh = igate * hidden
		if head_mask is not None:
			fgate = fgate.masked_fill(head_mask, 1.0)
			igh.masked_fill_(head_mask, 0.0)

		cell = LGateFunc(fgate, igh, self.init_cx if states is None or _init_state else states[-1], 1, True)
		out = self.trans_og(torch.cat((heads_input, cell), dim=-1)).sigmoid() * cell

		if states is None:
			return out
		else:
			return out, (csum_state_return.detach(), cell.select(1, seql - 1).detach(),)

class HPLSTM(HPLSTMBase):

	def __init__(self, isize, num_head=8, osize=None, fhsize=None, dropout=0.0, act_drop=None, **kwargs):

		_osize = parse_none(osize, isize)
		_fhsize = _osize * 4 if fhsize is None else fhsize

		super(HPLSTM, self).__init__(isize, num_head=num_head, osize=_osize, fhsize=_fhsize, dropout=dropout, **kwargs)

		self.net = MHPLSTMCore(isize, num_head=self.num_head, osize=_osize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop)
