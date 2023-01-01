#encoding: utf-8

from torch import nn
from torch.nn.functional import kl_div

from modules.kd.base import TLogSoftMax, TSoftMax
from utils.torch.comp import torch_no_grad
from utils.train.base import freeze_module

from cnfg.vocab.base import pad_id

class NMT(nn.Module):

	def __init__(self, teacher=None, student=None, T=None, **kwargs):

		super(NMT, self).__init__()

		self.teacher, self.student, self.T = teacher, student, float(T)

		self.teacher.dec.lsm = TSoftMax(T=T)
		self.student.dec.lsm = TLogSoftMax(T=T)
		freeze_module(self.teacher)
		self.teacher.eval()

	def forward(self, inpute, inputo, mask=None, target_mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		if self.training:
			with torch_no_grad():
				teacher_out = self.teacher(inpute, inputo, mask=_mask)
				if target_mask is not None:
					teacher_out.masked_fill_(target_mask.unsqueeze(-1), 0.0)

		student_out = self.student(inpute, inputo, mask=_mask)
		if self.training:
			student_out, student_out_kd = student_out
			kd_loss = kl_div(student_out_kd, teacher_out, reduction="sum")

			return student_out, kd_loss
		else:
			return student_out

	def train(self, mode=True):

		self.training = mode
		self.student.train(mode=mode)

		return self

	def eval(self):

		self.training = False
		self.student.eval()

		return self
