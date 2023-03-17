#encoding: utf-8

import torch
from torch import nn

from modules.spreader.Spreader import SpreaderFunc
from modules.spreader.SpreaderNocx import SpreaderNocxFunc
from modules.spreader.manual.rnn import Spreader as SpreaderBase
from utils.fmt.parser import parse_none
from utils.init.spreader import build_spread_vector
from utils.torch.comp import flip_mask, torch_no_grad
from utils.torch.ext import arcsigmoid

from cnfg.ihyp import *

class Spreader(SpreaderBase):

	def __init__(self, isize, hsize=None, start=2, end=8, factor=0.5, dropout=0.0, norm_residual=norm_residual_default, **kwargs):

		_hsize = parse_none(hsize, isize)

		super(Spreader, self).__init__(isize, hsize=_hsize, start=start, end=end, factor=factor, dropout=dropout, norm_residual=norm_residual, **kwargs)

		self.decay = nn.Parameter(arcsigmoid(build_spread_vector(start, end, _hsize, f=factor)))
		self.register_buffer("decay_beta", None, persistent=False)
		self.decay_init = tuple(self.decay.tolist())

	def forward(self, x, states=None, **kwargs):

		decay = self.decay.sigmoid()
		_x = self.normer(x)
		out = self.trans(_x).mul_(1.0 - decay)
		bsize, seql, hsize = out.size()

		cx_out = SpreaderNocxFunc(decay, out, 1, False) if (states is None) or (states == "init") else SpreaderFunc(decay, out, states, 1, False)

		out = self.outer(self.normer_csum(cx_out))
		_res_add = _x if self.norm_residual else x
		gate = self.gate(torch.cat((_res_add, out,), dim=-1))

		_res_add = (1.0 - gate).mul(_res_add)
		out = _res_add.addcmul_(gate, out) if self.drop is None else _res_add.add_(self.drop(out * gate))

		if states is None:
			return out
		else:
			return out, cx_out.select(1, -1)

	def fix_init(self):

		with torch_no_grad():
			self.decay.copy_(torch.as_tensor(self.decay_init, dtype=self.decay.dtype, device=self.decay.device))

class BiSpreader(Spreader):

	def forward(self, x, mask=None, pad_reversed_mask=None, **kwargs):

		decay = self.decay.sigmoid()
		bsize, seql = x.size()[:2]
		_x = self.normer(x)
		out = self.trans(_x).mul_(1.0 - decay)
		out = torch.stack((out, out.flip(1),), dim=2)

		_r_mask = pad_reversed_mask if mask is None else torch.stack((mask.new_zeros(bsize, seql, 1), flip_mask(mask, 1),), dim=2)
		if _r_mask is not None:
			out = out.masked_fill(_r_mask, 0.0)

		cx_out = SpreaderNocxFunc(decay, out, 1, True)

		_out_fwd, _out_rvs = cx_out.unbind(2)

		out = self.outer(self.normer_csum(_out_rvs.flip(1).add_(_out_fwd) - out))
		_res_add = _x if self.norm_residual else x
		gate = self.gate(torch.cat((_res_add, out,), dim=-1))

		_res_add = (1.0 - gate).mul(_res_add)

		return _res_add.addcmul_(gate, out) if self.drop is None else _res_add.add_(self.drop(out * gate))
