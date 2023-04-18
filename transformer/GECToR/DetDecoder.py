#encoding: utf-8

from math import sqrt

from modules.base import Linear
from transformer.PLM.BERT.Decoder import Decoder as DecoderBase
from utils.torch.comp import torch_no_grad

from cnfg.vocab.gector.det import vocab_size as num_class

class Decoder(DecoderBase):

	def forward(self, inpute, *args, mlm_mask=None, word_prediction=True, **kwargs):

		out = self.ff(inpute.select(1, 0) if mlm_mask is None else inpute[mlm_mask])
		if word_prediction:
			out = self.lsm(self.classifier(out))

		return out

	def build_task_model(self, *args, num_class=num_class, fix_init=True, **kwargs):

		self.classifier = Linear(self.classifier.weight.size(-1), num_class)
		if fix_init:
			self.fix_task_init()

	def fix_task_init(self):

		with torch_no_grad():
			_ = 1.0 / sqrt(self.classifier.weight.size(-1))
			self.classifier.weight.uniform_(-_, _)
			if self.classifier.bias is not None:
				self.classifier.bias.zero_()
