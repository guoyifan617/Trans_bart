#encoding: utf-8

import torch
from torch.nn.functional import kl_div

from loss.base import LabelSmoothingLoss as LabelSmoothingLossBase
from utils.base import eq_indexes

class LabelSmoothingLoss(LabelSmoothingLossBase):

	def forward(self, input, target, k=1.0, mask=None, **kwargs):

		if input.size(0) == target.size(0):
			return super(LabelSmoothingLoss, self).forward(input, target)
		else:
			_log_p = input.log()
			_loss_v = super(LabelSmoothingLoss, self).forward(_log_p, target.repeat(2, 1))
			# this training loss leads to NaN, no matter detach or not.
			_rp = torch.cat(tuple(reversed(input.chunk(2, dim=0))), dim=0).detach()
			_pad_mask = mask
			if _pad_mask is None:
				if isinstance(self.ignore_index, (list, tuple,)):
					_pad_mask = eq_indexes(target, self.ignore_index)
				elif self.ignore_index >= 0:
					_pad_mask = target.eq(self.ignore_index)
			if _pad_mask is not None:
				_rp.masked_fill_(_pad_mask.repeat(2, 1).unsqueeze(-1), 0.0)
			_loss_d = kl_div(_log_p, _rp, reduction=self.reduction)
			if k != 1.0:
				_loss_d = _loss_d * k

			return _loss_v + _loss_d
