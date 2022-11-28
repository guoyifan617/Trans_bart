#encoding: utf-8

from transformer.Doc.Para.Gate.CANMT import NMT as NMTBase

class NMT(NMTBase):

	def get_loaded_paras(self):

		return [self.dec.classifier.weight]
