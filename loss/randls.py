#encoding: utf-8

import torch
from torch.nn.functional import kl_div
from torch.nn.modules.loss import _Loss

from utils.base import eq_indexes

class RandLabelSmoothingLoss(_Loss):

	def __init__(self, nclass, label_smoothing=0.1, ignore_index=-1, reduction="mean", forbidden_index=-1, **kwargs):

		super(RandLabelSmoothingLoss, self).__init__()

		self.reduction = reduction

		fbil = set()
		if isinstance(ignore_index, (list, tuple,)):
			tmp = []
			for _tmp in ignore_index:
				if (_tmp >= 0) and (_tmp not in tmp):
					tmp.append(_tmp)
					if _tmp not in fbil:
						fbil.add(_tmp)
			_nid = len(tmp)
			if _nid > 0:
				self.ignore_index = tuple(tmp) if _nid > 1 else tmp[0]
			else:
				self.ignore_index = ignore_index[0] if len(ignore_index) > 0 else -1
		else:
			self.ignore_index = ignore_index
			if (ignore_index >= 0) and (ignore_index not in fbil):
				fbil.add(ignore_index)

		if isinstance(forbidden_index, (list, tuple,)):
			for fi in forbidden_index:
				if (fi >= 0) and (fi not in fbil):
					fbil.add(fi)
		else:
			if forbidden_index is not None and forbidden_index >= 0:
				fbil.add(forbidden_index)

		smoothing_value = label_smoothing / (nclass - 1 - len(fbil))

		self.register_buffer("fbil", torch.as_tensor(sorted(fbil), dtype=torch.long) if fbil else None, persistent=False)
		weight = torch.full((nclass,), smoothing_value)
		if self.fbil is not None:
			weight.index_fill_(0, self.fbil, 0.0)
		self.register_buffer("weight", weight.unsqueeze(0), persistent=False)
		self.conf = 1.0 - label_smoothing
		self.uniform_upper = 2.0 * smoothing_value

	def forward(self, input, target, mask=None, **kwargs):

		nclass = input.size(-1)
		_input = input.view(-1, nclass) if input.dim() > 2 else input
		_target = target.view(-1, 1)
		if self.training:
			model_prob = _input.new_empty(_input.size(), dtype=_input.dtype, device=_input.device).uniform_(0.0, self.uniform_upper)
			if self.fbil is not None:
				model_prob.index_fill_(1, self.fbil, 0.0)
		else:
			model_prob = self.weight.repeat(_target.size(0), 1)
		model_prob.scatter_(1, _target, self.conf)

		_pad_mask = mask
		if _pad_mask is None:
			if isinstance(self.ignore_index, (list, tuple,)):
				_pad_mask = eq_indexes(_target, self.ignore_index)
			elif self.ignore_index >= 0:
				_pad_mask = _target.eq(self.ignore_index)
		else:
			_pad_mask = _pad_mask.view(-1, 1)
		if _pad_mask is not None:
			model_prob.masked_fill_(_pad_mask, 0.0)

		rs = kl_div(_input, model_prob, reduction=self.reduction)

		return rs.view(input.size()) if self.reduction == "none" and target.dim() > 1 else rs
