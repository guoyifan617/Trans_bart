#encoding: utf-8

from torch.nn.functional import kl_div

from loss.base import StdLabelSmoothingLoss as LabelSmoothingLossBase
from utils.base import eq_indexes

class LabelSmoothingLoss(LabelSmoothingLossBase):

	def forward(self, input, target, sample_mask=None, mask=None, **kwargs):

		if sample_mask is None:
			_input, _btarget = input.view(-1, input.size(-1)) if input.dim() > 2 else input, target
			_target = _btarget.view(-1, 1)
			_perform_kl = True
		else:
			_btarget = target.masked_select(sample_mask)
			_target = _btarget.view(-1, 1)
			_perform_kl = (_target.numel() > 0)
			if _perform_kl:
				_input = input.masked_select(sample_mask.unsqueeze(-1)).view(-1, input.size(-1))

		if _perform_kl:
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

			return rs.view(list(_btarget.size()) + [-1]) if self.reduction == "none" and _btarget.dim() > 1 else rs
		else:
			return _target.new_zeros((1,))
