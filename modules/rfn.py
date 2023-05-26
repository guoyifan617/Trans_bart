#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.act import Custom_Act, LGLU, get_act
from modules.base import Dropout, Linear, PositionwiseFF as PositionwiseFFBase, ResSelfAttn as ResSelfAttnBase, SelfAttn as SelfAttnBase
from modules.rnncells import LSTMCell4RNMT
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DualInputLSTMCell4RNMT(LSTMCell4RNMT):

	def __init__(self, isize, enable_bias=enable_prev_ln_bias_default, **kwargs):

		super(DualInputLSTMCell4RNMT, self).__init__(isize, enable_bias=enable_bias, **kwargs)

		self.trans = Linear(isize + isize + self.osize, self.osize * 4, bias=enable_bias)

	def forward(self, input1, input2, state, **kwargs):

		_out, _cell = state

		osize = list(_out.size())
		osize.insert(-1, 4)

		_comb = self.normer(self.trans(torch.cat((input1, input2, _out,), -1)).view(osize))

		(ig, fg, og,), hidden = _comb.narrow(-2, 0, 3).sigmoid().unbind(-2), self.act(_comb.select(-2, 3))

		if self.drop is not None:
			hidden = self.drop(hidden)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class LSTMCell4FFN(nn.Module):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, act_dropout=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn, **kwargs):

		super(LSTMCell4FFN, self).__init__()

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize
		_act_dropout = parse_none(act_dropout, dropout)

		if (use_glu is not None) and (_hsize % 2 == 1):
			_hsize += 1

		self.trans = Linear(isize + _osize, _osize * 3, bias=enable_bias)
		self.normer = nn.LayerNorm((3, _osize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		_ = [Linear(isize + _osize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)]
		_drop_ind = 3
		if use_glu is None:
			_.extend([Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias)])
		else:
			use_glu = use_glu.lower()
			if use_glu == "glu":
				_.append(nn.GLU())
			else:
				_act = get_act(use_glu, None)
				if _act is not None:
					_.append(_act())
					_drop_ind += 1
				_.append(LGLU())
			_.append(Linear(_hsize // 2, isize, bias=enable_bias))
		if dropout > 0.0:
			_.append(Dropout(dropout, inplace=True))
		if _act_dropout > 0.0:
			_.insert(_drop_ind, Dropout(_act_dropout, inplace=inplace_after_Custom_Act))
		self.net = nn.Sequential(*_)

		self.osize = _osize

	def forward(self, inpute, state, **kwargs):

		_out, _cell = state

		_icat = torch.cat((inpute, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		(ig, fg, og,), hidden = self.normer(self.trans(_icat).view(osize)).sigmoid().unbind(-2), self.net(_icat)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class LSTMCell4AFFN(LSTMCell4FFN):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, act_dropout=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize
		_act_dropout = parse_none(act_dropout, dropout)

		super(LSTMCell4AFFN, self).__init__(isize, osize=_osize, hsize=_hsize, dropout=dropout, act_dropout=_act_dropout, custom_act=custom_act, enable_bias=enable_bias)

		self.net = nn.Sequential(Linear(isize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_dropout > 0.0:
			self.net.insert(3, Dropout(_act_dropout, inplace=inplace_after_Custom_Act))

	def forward(self, inpute, state, **kwargs):

		_out, _cell = state

		_icat = torch.cat((inpute, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		(ig, fg, og,), hidden = self.normer(self.trans(_icat).view(osize)).sigmoid().unbind(-2), self.net(inpute + _out)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class LSTMCell4NAFFN(LSTMCell4AFFN):

	def forward(self, inpute, state, **kwargs):

		_out, _cell = state

		_icat = torch.cat((inpute, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		(ig, fg, og,), hidden = self.normer(self.trans(_icat).view(osize)).sigmoid().unbind(-2), self.net(inpute)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class LSTMCell4StdFFN(LSTMCell4FFN):

	def __init__(self, *args, **kwargs):

		super(LSTMCell4StdFFN, self).__init__(*args, **kwargs)

		self.net = None

	def forward(self, inpute, state, **kwargs):

		_out, _cell = state

		_icat = torch.cat((inpute, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		ig, fg, og = self.normer(self.trans(_icat).view(osize)).sigmoid().unbind(-2)

		_cell = fg * _cell + ig * inpute
		_out = og * _cell

		return _out, _cell

class DualInputLSTMCell4FFN(LSTMCell4FFN):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, act_dropout=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize
		_act_dropout = parse_none(act_dropout, dropout)

		super(DualInputLSTMCell4FFN, self).__init__(isize, osize=_osize, hsize=_hsize, dropout=dropout, act_dropout=_act_dropout, custom_act=custom_act, enable_bias=enable_bias)

		self.trans = Linear(isize + isize + _osize, _osize * 3, bias=enable_bias)
		self.normer = nn.LayerNorm((3, _osize), eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.net = nn.Sequential(Linear(isize + isize + _osize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.net.append(Dropout(dropout, inplace=True))
		if _act_dropout > 0.0:
			self.net.insert(3, Dropout(_act_dropout, inplace=inplace_after_Custom_Act))

	def forward(self, input1, input2, state, **kwargs):

		_out, _cell = state

		_icat = torch.cat((input1, input2, _out), -1)

		osize = list(_out.size())
		osize.insert(-1, 3)

		(ig, fg, og,), hidden = self.normer(self.trans(_icat).view(osize)).sigmoid().unbind(-2), self.net(_icat)

		_cell = fg * _cell + ig * hidden
		_out = og * _cell

		return _out, _cell

class SelfAttn(SelfAttnBase):

	def __init__(self, isize, hsize, osize, enable_outer=True, **kwargs):

		super(SelfAttn, self).__init__(isize, hsize, osize, **kwargs)

		if not enable_outer:
			self.outer = None

	def forward(self, iQ, mask=None, states=None, **kwargs):

		bsize, nquery = iQ.size()[:2]
		nheads = self.num_head
		adim = self.attn_dim

		real_iQ, real_iK, real_iV = self.adaptor(iQ).view(bsize, nquery, 3, nheads, adim).unbind(2)
		real_iQ, real_iK, real_iV = real_iQ.transpose(1, 2), real_iK.permute(0, 2, 3, 1), real_iV.transpose(1, 2)

		if states is not None:
			_h_real_iK, _h_real_iV = states
			if _h_real_iK is None:
				seql = nquery
			else:
				seql = nquery + _h_real_iK.size(-1)
				real_iK, real_iV = torch.cat((_h_real_iK, real_iK,), dim=-1), torch.cat((_h_real_iV, real_iV,), dim=2)

		scores = real_iQ.matmul(real_iK)

		if self.rel_pemb is not None:
			if states is None:
				self.rel_pos_cache = self.get_rel_pos(nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, nquery).permute(1, 2, 0, 3)
			else:
				self.rel_pos_cache = self.get_rel_pos(seql).narrow(0, seql - nquery, nquery).contiguous() if self.ref_rel_posm is None else self.ref_rel_posm.rel_pos_cache
				scores += real_iQ.permute(2, 0, 1, 3).contiguous().view(nquery, bsize * nheads, adim).bmm(self.rel_pemb(self.rel_pos_cache).transpose(1, 2)).view(nquery, bsize, nheads, seql).permute(1, 2, 0, 3)

		scores = scores / sqrt(adim)

		if mask is not None:
			scores.masked_fill_(mask.unsqueeze(1), -inf_default)

		scores = self.normer(scores)

		if self.drop is not None:
			scores = self.drop(scores)

		out = scores.matmul(real_iV).transpose(1, 2).contiguous().view(bsize, nquery, self.hsize)

		if self.outer is not None:
			out = self.outer(out)

		if states is None:
			return out
		else:
			return out, (real_iK, real_iV,)

class ResSelfAttn(ResSelfAttnBase):

	def __init__(self, isize, hsize, num_head=8, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		super(ResSelfAttn, self).__init__(isize, hsize, num_head=num_head, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.net = SelfAttn(isize, hsize, isize, num_head=num_head, dropout=dropout, **kwargs)

class PositionwiseFF(PositionwiseFFBase):

	def forward(self, x, **kwargs):

		return self.net(self.normer(x))

class RNN4FFN(nn.Sequential):

	def __init__(self, isize, osize=None, hsize=None, dropout=0.0, act_dropout=None, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, **kwargs):

		_osize = parse_none(osize, isize)
		_hsize = _osize * 4 if hsize is None else hsize
		_act_dropout = parse_none(act_dropout, dropout)

		super(RNN4FFN, self).__init__(Linear(isize, _hsize, bias=enable_bias), nn.LayerNorm(_hsize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Custom_Act() if custom_act else nn.ReLU(inplace=True), Linear(_hsize, isize, bias=enable_bias))
		if dropout > 0.0:
			self.append(Dropout(dropout, inplace=True))
		if _act_dropout > 0.0:
			self.insert(3, Dropout(_act_dropout, inplace=inplace_after_Custom_Act))
