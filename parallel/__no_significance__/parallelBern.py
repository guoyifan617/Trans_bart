#encoding: utf-8

from parallel.parallelMT import DataParallelMT as DataParallelMTBase

class DataParallelMT(DataParallelMTBase):

	def resetBern(self):

		if self.nets is not None:
			for net in self.nets:
				net.resetBern()
		self.module.resetBern()

	def useBernMask(self, value):

		if self.nets is not None:
			for net in self.nets:
				net.useBernMask(value)
		self.module.useBernMask(value)

	def train_parameters(self, value):

		if self.nets is not None:
			for net in self.nets:
				net.train_parameters(value)
		self.module.train_parameters(value)
