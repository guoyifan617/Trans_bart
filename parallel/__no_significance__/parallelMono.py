#encoding: utf-8

from parallel.base import parallel_apply
from parallel.parallelMTFP import DataParallel as DataParallelMTBase
from utils.fmt.base import clean_list

class DataParallelMT(DataParallelMTBase):

	def forward(self, *inputs, **kwargs):

		if (not self.device_ids) or (len(self.device_ids) == 1):
			return self.module(*inputs, **kwargs) if self.gather_output else [self.module(*inputs, **kwargs)]
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		inputs = clean_list(inputs)
		if len(inputs) == 1:
			return self.module(*inputs[0], **kwargs[0]) if self.gather_output else [self.module(*inputs[0], **kwargs[0])]
		ngpu = len(inputs)
		devices = self.device_ids[:ngpu]
		replicas = self.replicate(self.module, devices) if self.nets is None else self.nets[:ngpu]
		outputs = parallel_apply(replicas, inputs, devices, kwargs)
		if isinstance(outputs[0], tuple):
			outputs = tuple(zip(*outputs))
		return self.gather(outputs, self.output_device) if self.gather_output else outputs

	def train_advers(self, mode=True):

		if self.nets is None:
			self.module.train_advers(mode)
		else:
			for net in self.nets:
				net.train_advers(mode)
