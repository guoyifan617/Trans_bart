#encoding: utf-8

import torch.cuda.comm as comm

from parallel.parallelMT import DataParallelMT as DataParallelMTBase
from utils.base import filter_para_grad

class DataParallelMT(DataParallelMTBase):

	def collect_gradients(self):

		if self.ngradev > 1:
			grads = comm.reduce_add_coalesced([[p.data.new_zeros(p.data.size()) if p.grad is None else p.grad for p in filter_para_grad(net.parameters())] for net in self.nets[:self.ngradev]], self.output_device)
			for mp, grad in zip(filter_para_grad(self.module.parameters()), grads):
				mp.grad = grad

	def tell(self):

		return self.module.tell()
