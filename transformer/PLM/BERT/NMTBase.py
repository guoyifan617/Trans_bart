#encoding: utf-8

from torch import nn

from modules.base import Linear
from transformer.PLM.NMT import NMT as NMTBase
from utils.fmt.parser import parse_none
from utils.torch.comp import torch_no_grad

class NMT(NMTBase):

	def update_vocab(self, src_indices=None, tgt_indices=None):

		indices = parse_none(src_indices, tgt_indices)
		_nwd = indices.numel()
		_wemb = nn.Embedding(_nwd, self.enc.wemb.weight.size(-1), padding_idx=self.enc.wemb.padding_idx)
		_classifier = Linear(self.dec.classifier.weight.size(-1), _nwd, bias=self.dec.classifier.bias is not None)
		with torch_no_grad():
			_wemb.weight.copy_(self.enc.wemb.weight.index_select(0, indices))
			if self.dec.classifier.weight.is_set_to(self.enc.wemb.weight):
				_classifier.weight = _wemb.weight
			else:
				_classifier.weight.copy_(self.dec.classifier.weight.index_select(0, indices))
			if self.dec.classifier.bias is not None:
				_classifier.bias.copy_(self.dec.classifier.bias.index_select(0, indices))
		self.enc.wemb, self.dec.classifier = _wemb, _classifier
