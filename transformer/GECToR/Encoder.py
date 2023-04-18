#encoding: utf-8

from math import sqrt
from torch import nn

from modules.dropout import TokenDropout
from transformer.Encoder import Encoder as EncoderBase
from utils.torch.comp import torch_no_grad

from cnfg.gec.gector import num_type, token_drop
from cnfg.ihyp import *
from cnfg.vocab.gector.edit import pad_id, vocab_size as num_edit

class Encoder(EncoderBase):

	def forward(self, inputs, edit=None, token_types=None, mask=None, **kwargs):

		out = self.wemb(inputs)
		if edit is not None:
			_ = self.edit_emb(edit)
			if self.edit_tdrop is not None:
				_ = self.edit_tdrop(_)
			out = out + _
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.temb is not None:
			out = out + (self.temb.weight[0] if token_types is None else self.temb(token_types))

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def build_task_model(self, *args, token_drop=token_drop, fix_init=True, **kwargs):

		self.edit_emb = nn.Embedding(num_edit, self.wemb.weight.size(-1), padding_idx=pad_id)
		self.temb = None if num_type > 0 else nn.Embedding(num_type, self.wemb.weight.size(-1))
		self.edit_tdrop = TokenDropout(token_drop, inplace=True) if token_drop > 0.0 else None
		if fix_init:
			self.fix_task_init()

	def fix_task_init(self):

		if hasattr(self, "edit_emb"):
			with torch_no_grad():
				_ = 2.0 / sqrt(sum(self.edit_emb.weight.size()))
				self.edit_emb.weight.uniform_(-_, _)
				self.edit_emb.weight[pad_id].zero_()
				if self.temb is not None:
					self.temb.weight.zero_()
