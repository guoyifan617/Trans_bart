#encoding: utf-8

import torch

from modules.mulang.eff.base import LayerNorm
from modules.rfn import LSTMCell4FFN as LSTMCell4FFNBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class LSTMCell4FFN(LSTMCell4FFNBase):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, act_drop=None, ntask=None, **kwargs):

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize

		super(LSTMCell4FFN, self).__init__(isize, osize=_osize, hsize=_hsize, dropout=dropout, act_drop=act_drop, **kwargs)

		self.normer = LayerNorm((3, _osize), ntask=ntask, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

	def forward(self, inpute, state, taskid=None, **kwargs):

		_out, _cell = state

		_icat = torch.cat((inpute, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		(ig, fg, og,), hidden = self.normer(self.trans(_icat).view(osize), taskid=taskid).sigmoid().unbind(-2), self.net(_icat)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell
