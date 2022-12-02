#encoding: utf-8

from math import sqrt
from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, PositionwiseFF as PositionwiseFFBase
from modules.group.base import GroupLinear
from utils.base import float2odd

from cnfg.ihyp import *

#todo: multi-head attention might be more helpful
class AttnLinear(nn.Module):

	def __init__(self, isize, osize, bias=True, hsize=None, num_head=None, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(AttnLinear, self).__init__()

		num_head = isize // 64 if num_head is None else num_head
		if hsize is None:
			_t = (isize + osize) / num_head
			_th = max(float2odd(sqrt(_t * _t + 12 * isize * osize) - _t) / 6.0), 2)
			hsize = _th

		_hhsize = hsize // num_head
		hsize = _hhsize * num_head

		self.vsize = [-1, num_head, _hhsize]
		self.nv = sqrt(_hhsize)

		self.transfer = GroupLinear(isize, hsize, num_head, bias)

		self.adaptor = Linear(_hhsize, _hhsize * 3, bias=enable_bias)

		self.outer = GroupLinear(hsize, osize, num_head, bias)

	def forward(self, inputu, **kwargs):

		isize = list(inputu.size())

		# (merged_bsize, nhead, husize)
		out = self.transfer(inputu).view(self.vsize)

		# (merged_bsize, nhead, husize*3)
		_tsize = list(out.size())
		_tsize.insert(-1, 3)
		q, k, v = self.adaptor(out).view(_tsize).unbind(-2)

		# (merged_bsize, nhead, husize)
		isize[-1] = -1
		out = (q.bmm(k.transpose(1, 2)) / self.nv).softmax(-1).bmm(v).view(isize)

		return self.outer(out)

class PositionwiseFF(PositionwiseFFBase):

	def __init__(self, isize, hsize=None, dropout=0.0, custom_act=use_adv_act_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, custom_act=custom_act, **kwargs)

		self.net = nn.Sequential(AttnLinear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Dropout(dropout, inplace=inplace_after_Custom_Act), Linear(_hsize, isize), Dropout(dropout, inplace=True)) if dropout > 0.0 else nn.Sequential(AttnLinear(isize, _hsize), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize))