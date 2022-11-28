#encoding: utf-8

from torch.nn.modules.loss import _Loss

from utils.torch.ext import cosim

from cnfg.ihyp import ieps_ln_default

def cosim_loss(a, b, mask=None, dim=-1, eps=ieps_ln_default, reduction="mean"):

	loss = cosim(a, b, dim=dim, keepdim=False, eps=eps)
	_is_mean_reduction = reduction == "mean"
	if _is_mean_reduction:
		_num = loss.numel()
	if mask is not None:
		loss.masked_fill_(mask, 0.0)
		if _is_mean_reduction:
			_num -= mask.int().sum().item()
	if reduction != "none":
		loss = loss.sum()
	if _is_mean_reduction:
		loss = loss / float(_num)

	return -loss

class Cosim(_Loss):

	def __init__(self, dim=-1, eps=ieps_ln_default, reduction="mean", **kwargs):

		super(Cosim, self).__init__()
		self.dim, self.eps, self.reduction = dim, eps, reduction

	def forward(self, input, target, mask=None, **kwargs):

		return cosim_loss(input, target, mask=mask, dim=self.dim, eps=self.eps, reduction=self.reduction)
