#encoding: utf-8

from torch import nn

from modules.act import Custom_Act, GELU
from modules.base import Linear
from modules.dropout import Dropout
from transformer.PLM.BERT.Decoder import Decoder as DecoderBase

from cnfg.ihyp import enable_ln_parameters, ieps_ln_default, use_adv_act_default
from cnfg.vocab.gector.op import vocab_size as num_op

class Decoder(DecoderBase):

	def forward(self, inpute, mlm_mask=None, word_prediction=False, **kwargs):

		out = self.ff(inpute)
		if mlm_mask is not None:
			out = out[mlm_mask.unsqueeze(-1).expand_as(out)].view(-1, out.size(-1))
		if word_prediction:
			out = self.lsm(self.classifier(out))

		return out

	def build_task_model(self, *args, **kwargs):

		self.edit_emb = nn.Sequential(Linear(isize, isize), Custom_Act() if use_adv_act_default else GELU(), nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters), Linear(isize, num_op))
