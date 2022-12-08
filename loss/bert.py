#encoding: utf-8

from torch.nn.modules.loss import _Loss

from loss.base import LabelSmoothingLoss as LabelSmoothingLossBase

class LabelSmoothingLoss(_Loss):

	def __init__(self, *args, **kwargs):

		super(LabelSmoothingLoss, self).__init__()
		self.loss = LabelSmoothingLossBase(*args, **kwargs)

	def forward(self, input, target, sample_mask=None, mask=None, **kwargs):

		return self.loss(input, target if sample_mask is None else target[sample_mask], mask=mask)
