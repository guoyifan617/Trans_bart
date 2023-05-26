#encoding: utf-8

from torch import nn

from modules.act import Custom_Act
from modules.base import Dropout, Linear, PositionwiseFF as PositionwiseFFBase
from modules.group.base import GroupLinear
from utils.base import float2odd
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class PositionwiseFF(PositionwiseFFBase):

	# head_dim is only used to infer num_head
	def __init__(self, isize, hsize=None, dropout=0.0, act_drop=None, num_head=None, head_dim=64, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_hsize = isize * 4 if hsize is None else hsize
		_act_drop = parse_none(act_drop, dropout)

		super(PositionwiseFF, self).__init__(isize, hsize=_hsize, dropout=dropout, act_drop=_act_drop, custom_act=custom_act, enable_bias=enable_bias, **kwargs)

		_nhead = max(2, float2odd(float(isize) / float(head_dim))) if num_head is None else num_head
		_head_dim = isize // _nhead
		_isize = _nhead * _head_dim
		_hsize = max(2, float2odd(_hsize - _isize)) * _nhead

		self.net = nn.Sequential(Linear(isize, _isize, bias=enable_proj_bias), GroupLinear(_isize, _hsize, _nhead, bias=True, trans_input=True, shuffle=False, flatten_output=False), Custom_Act() if custom_act else nn.ReLU(inplace=True), GroupLinear(_hsize, _isize, _nhead, bias=True, trans_input=False, shuffle=False, flatten_output=True), Linear(_isize, isize, bias=enable_proj_bias))
		if dropout > 0.0:
			self.net.insert(4, Dropout(dropout, inplace=True))
		if _act_drop > 0.0:
			self.net.insert(3, Dropout(_act_drop, inplace=inplace_after_Custom_Act))
