#encoding: utf-8

from math import sqrt

from transformer.Encoder import Encoder as EncoderBase
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.gector.edit import pad_id, vocab_size as num_edit

class Encoder(EncoderBase):

	def forward(self, inputs, edit=None, mask=None, **kwargs):

		out = self.wemb(inputs)
		if edit is not None:
			out = out + self.edit_emb(edit)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def build_task_model(self, *args, **kwargs):

		self.edit_emb = nn.Embedding(num_edit, self.wemb.size(-1), padding_idx=pad_id)
		self.fix_task_init()

	def fix_task_init(self):

		if hasattr(self, "edit_emb"):
			with torch_no_grad():
				_ = 2.0 / sqrt(sum(self.edit_emb.weight.size()))
				self.edit_emb.weight.uniform_(-_, _)
				self.edit_emb.weight[pad_id].zero_()
