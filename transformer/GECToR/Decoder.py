#encoding: utf-8

from math import sqrt

from loss.base import CrossEntropyLoss, LabelSmoothingLoss
from modules.base import Linear
from transformer.PLM.BERT.Decoder import Decoder as DecoderBase
from utils.torch.comp import torch_no_grad

from cnfg.gec.gector import forbidden_indexes, label_smoothing, use_smooth_op_loss
from cnfg.vocab.gector.op import pad_id as op_pad_id, vocab_size as num_op
from cnfg.vocab.plm.custbert import pad_id as mlm_pad_id, vocab_size as mlm_vocab_size

class Decoder(DecoderBase):

	def forward(self, inpute, mlm_mask=None, tgt=None, prediction=False, **kwargs):

		out = self.ff(inpute)
		out_op = self.op_classifier(out)
		out_mlm = None if mlm_mask is None else self.classifier(out[mlm_mask])
		if prediction:
			tag_out = out_op.argmax(-1)
			if mlm_mask is not None:
				tag_out[mlm_mask] = out_mlm.argmax(-1)
		else:
			tag_out = None
		if tgt is None:
			loss = None
		else:
			if self.use_smooth_op_loss > 0.0:
				out_op = self.lsm(out_op)
			loss = (self.op_loss(out_op, tgt) if mlm_mask is None else (self.op_loss(out_op, tgt.masked_fill(mlm_mask, op_pad_id)) + self.mlm_loss(self.lsm(out_mlm), tgt[mlm_mask])))

		return loss, tag_out

	def build_task_model(self, *args, use_smooth_op_loss=use_smooth_op_loss, **kwargs):

		self.use_smooth_op_loss = use_smooth_op_loss
		self.op_classifier = Linear(self.classifier.weight.size(-1), num_op)
		self.op_loss = LabelSmoothingLoss(num_op, use_smooth_op_loss, ignore_index=op_pad_id, reduction="sum", forbidden_index=None) if self.use_smooth_op_loss > 0.0 else CrossEntropyLoss(ignore_index=op_pad_id, reduction="sum")
		self.mlm_loss = LabelSmoothingLoss(mlm_vocab_size, label_smoothing, ignore_index=mlm_pad_id, reduction="sum", forbidden_index=forbidden_indexes)
		self.fix_task_init()

	def fix_task_init(self):

		if hasattr(self, "op_classifier"):
			with torch_no_grad():
				_ = 1.0 / sqrt(self.op_classifier.weight.size(-1))
				self.op_classifier.weight.uniform_(-_, _)
				self.op_classifier.weight[op_pad_id].zero_()
				if self.op_classifier.bias is not None:
					self.op_classifier.bias.zero_()
