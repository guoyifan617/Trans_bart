#encoding: utf-8

import torch
from math import sqrt
from torch import nn
from torch.nn import functional as nnFunc

from utils.fmt.parser import parse_none

# portal from: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py

class AdaptiveEmbedding(nn.Module):

	def __init__(self, n_token, d_embed, cutoffs, d_proj=None, div_val=1, padding_idx=None, sample_softmax=False, **kwargs):

		super(AdaptiveEmbedding, self).__init__()

		_d_proj = parse_none(d_proj, d_embed)
		self.n_token, self.d_embed, self.div_val, self.d_proj = n_token, d_embed, div_val, _d_proj

		self.cutoffs = cutoffs + [n_token]

		self.cutoff_ends = [0] + self.cutoffs

		self.emb_layers = nn.ModuleList()
		self.emb_projs = nn.ParameterList()
		if div_val == 1:
			self.emb_layers.append(nn.Embedding(n_token, d_embed, padding_idx=padding_idx, sparse=sample_softmax))
			if _d_proj != d_embed:
				_init_scale = 1.0 / sqrt(d_embed)
				self.emb_projs.append(nn.Parameter(torch.Tensor(_d_proj, d_embed).uniform_(-_init_scale, _init_scale)))
		else:
			for i in range(len(self.cutoffs)):
				l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
				d_emb_i = d_embed // (div_val ** i)
				self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i, padding_idx=padding_idx))
				_init_scale = 1.0 / sqrt(d_emb_i)
				self.emb_projs.append(nn.Parameter(torch.Tensor(_d_proj, d_emb_i).uniform_(-_init_scale, _init_scale)))

	def forward(self, input, **kwargs):

		if self.div_val == 1:
			embed = self.emb_layers[0](input)
			if self.d_proj != self.d_embed:
				embed = nnFunc.linear(embed, self.emb_projs[0])
		else:
			input_flat = input.view(-1)
			emb_flat = input_flat.new_zeros([input_flat.size(0), self.d_proj])
			for i in range(len(self.cutoffs)):
				l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

				mask_i = (input_flat >= l_idx) & (input_flat < r_idx)
				indices_i = mask_i.nonzero().squeeze()

				if indices_i.numel() > 0:
					input_i = input_flat.index_select(0, indices_i) - l_idx
					emb_i = self.emb_layers[i](input_i)
					emb_i = nnFunc.linear(emb_i, self.emb_projs[i])

					emb_flat.index_copy_(0, indices_i, emb_i)

			embed = emb_flat.view(*input.size(), self.d_proj)

		return embed

# portal from: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/proj_adaptive_softmax.py

class ProjectedAdaptiveLogSoftmax(nn.Module):

	def __init__(self, n_token, d_embed, cutoffs, d_proj=None, div_val=1, ignore_index=-100, reduce=None, reduction="mean", keep_order=True, **kwargs):

		super(ProjectedAdaptiveLogSoftmax, self).__init__()

		_d_proj = parse_none(d_proj, d_embed)
		self.n_token, self.d_embed, self.d_proj, self.div_val, self.ignore_index, self.reduction = n_token, d_embed, _d_proj, div_val, ignore_index, reduction

		self.cutoffs = cutoffs + [n_token]
		self.cutoff_ends = [0] + self.cutoffs

		self.shortlist_size = self.cutoffs[0]
		self.n_clusters = len(self.cutoffs) - 1
		self.head_size = self.shortlist_size + self.n_clusters

		if self.n_clusters > 0:
			self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
			self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

		self.out_layers = nn.ModuleList()
		self.out_projs = nn.ParameterList()

		if div_val == 1:
			_init_scale = 1.0 / sqrt(d_embed)
			for i in range(len(self.cutoffs)):
				if _d_proj != d_embed:
					self.out_projs.append(nn.Parameter(torch.Tensor(_d_proj, d_embed).uniform_(-_init_scale, _init_scale)))
				else:
					self.out_projs.append(None)

			self.out_layers.append(nn.Linear(d_embed, n_token))
		else:
			for i in range(len(self.cutoffs)):
				l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
				d_emb_i = d_embed // (div_val ** i)
				_init_scale = 1.0 / sqrt(d_emb_i)
				self.out_projs.append(nn.Parameter(torch.Tensor(_d_proj, d_emb_i).uniform_(-_init_scale, _init_scale)))

				self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

		self.keep_order = keep_order

	def _compute_logit(self, hidden, weight, bias, proj):

		if proj is None:
			logit = nnFunc.linear(hidden, weight, bias=bias)
		else:
			# for CUDA_MAJOR <= 9 and CUDA_MINOR <= 1
			#proj_hid = nnFunc.linear(hidden, proj.t().contiguous())
			#logit = nnFunc.linear(proj_hid, weight, bias=bias)

			logit = torch.einsum("bd,de,ev->bv", (hidden, proj, weight.t()))
			if bias is not None:
				logit = logit + bias

		return logit

	def forward(self, hidden, target, keep_order=False, **kwargs):

		if self.n_clusters == 0:
			logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
			nll = -logit.log_softmax(dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
		else:
			weights, biases = [], []
			for i in range(len(self.cutoffs)):
				if self.div_val == 1:
					l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
					weight_i = self.out_layers[0].weight[l_idx:r_idx]
					bias_i = self.out_layers[0].bias[l_idx:r_idx]
				else:
					weight_i = self.out_layers[i].weight
					bias_i = self.out_layers[i].bias

				if i == 0:
					weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
					bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

				weights.append(weight_i)
				biases.append(bias_i)

			head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

			head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
			head_logprob = head_logit.log_softmax(dim=1)

			nll = hidden.new_zeros(target.size())

			offset = 0
			cutoff_values = [0] + self.cutoffs
			_ig_mask = target.ne(self.ignore_index) if self.ignore_index >= 0 else None
			for i in range(len(cutoff_values) - 1):
				l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

				mask_i = (target >= l_idx) & (target < r_idx)
				if (_ig_mask is not None) and (self.ignore_index >= l_idx) and (self.ignore_index < r_idx):
					mask_i &= _ig_mask
				indices_i = mask_i.nonzero().squeeze()

				if indices_i.numel() > 0:

					target_i = target.index_select(0, indices_i) - l_idx
					head_logprob_i = head_logprob.index_select(0, indices_i)

					if i == 0:
						logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
					else:
						weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

						hidden_i = hidden.index_select(0, indices_i)

						tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
						tail_logprob_i = tail_logit_i.log_softmax(dim=1)

						logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

					if self.keep_order or keep_order:
						nll.index_copy_(0, indices_i, -logprob_i)
					else:
						nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

					offset += logprob_i.size(0)

		if self.reduction == "sum":
			return nll.sum()
		elif self.reduction == "mean":
			return nll.mean()
		else:
			return nll
