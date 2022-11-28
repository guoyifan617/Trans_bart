#encoding: utf-8

from torch.nn import Embedding as EmbeddingBase, Linear as LinearBase, functional as nnFunc

class Embedding(EmbeddingBase):

	def forward(self, input, **kwargs):

		return nnFunc.embedding(input, self.weight.narrow(-1, 0, self.embedding_dim) if self.embedding_dim < self.weight.size(-1) else self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

class Linear(LinearBase):

	def forward(self, input, **kwargs):

		return nnFunc.linear(input, self.weight.narrow(-1, 0, self.in_features) if self.in_features < self.weight.size(-1) else self.weight, self.bias)
