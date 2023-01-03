#encoding: utf-8

from utils.fmt.base import clean_str, iter_to_str
from utils.fmt.u8 import norm_u8_str, uni_normer
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.char import ldvocab

from cnfg.vocab.plm.custbert import eos_id, init_normal_token_id, init_vocab, sos_id, vocab_size

class Tokenizer:

	def __init__(self, vcbf, norm_u8=True, uni_normer=uni_normer, add_sp_tokens=True, sos_id=sos_id, eos_id=eos_id, split=False, minfreq=False, vsize=vocab_size):

		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.norm_u8, self.uni_normer, self.add_sp_tokens, self.sos_id, self.eos_id, self.split = norm_u8, uni_normer, add_sp_tokens, self.vcb["<sos>"] if sos_id is None else sos_id, self.vcb["<eos>"] if eos_id is None else eos_id, split
		self.rvcb = None

	def __call__(self, x, **kwargs):

		rs = (norm_u8_str(x, uni_normer=self.uni_normer) if self.norm_u8 else x)
		rs = rs.split("\t") if self.split else clean_str(rs)
		rs = [self.vcb[_] for _ in rs if _ in self.vcb]
		if self.add_sp_tokens:
			_ = [self.sos_id]
			_.extend(rs)
			_.append(self.eos_id)
			rs = _

		return rs

	def decode(self, x, **kwargs):

		if self.rvcb is None:
			self.rvcb = reverse_dict(self.vcb)

		return ("\t" if self.split else "").join(self.rvcb[_] for _ in x)

map_line = lambda lin, processor: " ".join(iter_to_str(processor(lin)))
