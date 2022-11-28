#encoding: utf-8

import torch

from modules.base import PositionalEmb as PositionalEmbBase

class PositionalEmb(PositionalEmbBase):

	# x: input (bsize, seql/nsent)

	def forward(self, x, expand=True, rsentid=None, **kwargs):#, remove_cls_token=False

		bsize, seql = x.size()

		_seql = seql if rsentid is None else (seql + 1)

		"""if remove_cls_token:
			__seql = _seql - 1
			rs_seq = [self.w[:__seql]] if __seql <= self.num_pos else [self.w, self.get_ext(__seql, False)]
			rs_seq = torch.cat([self.w.new_zeros((1, self.num_dim), dtype=self.w.dtype, device=self.w.device)] + rs_seq, dim=0)
		else:"""
		rs_seq = self.w[:_seql] if _seql <= self.num_pos else torch.cat((self.w, self.get_ext(_seql, False)), 0)

		if rsentid is not None:
			if rsentid == 0:
				rs_seq = rs_seq.narrow(0, 1, seql)
			elif rsentid == seql - 1:
				rs_seq = rs_seq.narrow(0, 0, seql)
			else:
				rs_seq = torch.cat((rs_seq.narrow(0, 0, rsentid), rs_seq.narrow(0, rsentid + 1, seql - rsentid),), dim=0)

		rs_seq = rs_seq.unsqueeze(0)
		if expand:
			rs_seq = rs_seq.expand(bsize, seql, self.num_dim)

		return rs_seq
