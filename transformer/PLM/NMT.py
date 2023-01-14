#encoding: utf-8

from transformer.NMT import NMT as NMTBase

class NMT(NMTBase):

	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		enc_model_name, dec_model_name = parse_double_value_tuple(_model_name)
		self.enc.load_plm(plm_parameters, model_name=enc_model_name, **kwargs)
		self.dec.load_plm(plm_parameters, model_name=dec_model_name, **kwargs)
